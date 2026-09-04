#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/detail/validate.hpp"

#include <optional>
#include <algorithm>
#include <bit>
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

namespace lindblad {

// =============================================================================
// StabilizerState — word-packed Gottesman-Knill tableau
// =============================================================================

// ----- inline bit helpers -----

inline bool StabilizerState::get_xz(int row, int col) const {
    return (tab[row * wpr + col / 64] >> (col % 64)) & 1;
}

inline void StabilizerState::set_xz(int row, int col, bool v) {
    uint64_t& w = tab[row * wpr + col / 64];
    const uint64_t mask = 1ULL << (col % 64);
    w = v ? (w | mask) : (w & ~mask);
}

inline void StabilizerState::flip_xz(int row, int col) {
    tab[row * wpr + col / 64] ^= (1ULL << (col % 64));
}

// 64 bits of a row starting at an arbitrary bit offset, spanning the word
// boundary when the offset is not a multiple of 64. The X plane occupies bits
// [0, N) and the Z plane [N, 2N), so the Z plane generally starts mid-word and
// only a spanning read lines the planes up qubit for qubit. The row's trailing
// padding word is what makes the second access in bounds at every offset the
// callers use.
inline uint64_t StabilizerState::bits64(int row, int bitoff) const {
    const uint64_t* r = &tab[static_cast<size_t>(row) * static_cast<size_t>(wpr)];
    const int w = bitoff >> 6;
    const int s = bitoff & 63;
    uint64_t v = r[w] >> s;
    if (s) v |= r[w + 1] << (64 - s);  // a shift by 64 would be undefined
    return v;
}

// Word-level XOR of the X/Z bits of row src into row dest.
// This is the hot path in rowmult: ~64× faster than element-by-element.
inline void StabilizerState::xor_row(int dest, int src) {
    const int bd = dest * wpr, bs = src * wpr;
    for (int w = 0; w < wpr; ++w) tab[bd + w] ^= tab[bs + w];
}

inline void StabilizerState::copy_row(int dest, int src) {
    const int bd = dest * wpr, bs = src * wpr;
    for (int w = 0; w < wpr; ++w) tab[bd + w] = tab[bs + w];
    ph[dest] = ph[src];
}

inline void StabilizerState::zero_row(int row) {
    const int b = row * wpr;
    for (int w = 0; w < wpr; ++w) tab[b + w] = 0;
    ph[row] = 0;
}

// Append a zeroed scratch row (used by deterministic measurement path).
inline void StabilizerState::push_scratch() {
    tab.insert(tab.end(), wpr, 0ULL);
    ph.push_back(0);
    ++num_rows;
}

inline void StabilizerState::pop_scratch() {
    tab.resize(tab.size() - wpr);
    ph.pop_back();
    --num_rows;
}

// ----- constructor -----

// Validates a register width and returns it, so the check can run from the
// FIRST entry of a member initialiser list. Every other initialiser derives a
// buffer size from the width, and a negative one reaches those as an enormous
// unsigned length, so the check has to happen before any of them rather than in
// the constructor body.
static int checked_width(int n, const char* ctx) {
    detail::check_require(n >= 0, ctx,
                          "n_qubits must be >= 0, got " + std::to_string(n));
    return n;
}

StabilizerState::StabilizerState(int n_qubits)
    : n_qubits(checked_width(n_qubits, "StabilizerState")),
      wpr((2 * n_qubits + 63) / 64 + 1),  // + 1: padding for the spanning read
      num_rows(2 * n_qubits),
      tab(static_cast<size_t>(2 * n_qubits) * static_cast<size_t>(wpr), 0ULL),
      ph(2 * n_qubits, 0)
{
    // Reserve space for the scratch row appended during deterministic measurement.
    tab.reserve(static_cast<size_t>(2 * n_qubits + 1) * static_cast<size_t>(wpr));
    ph.reserve(2 * n_qubits + 1);

    // Initial state |0...0⟩:
    // Destabilizers (rows 0..N-1): X_i
    // Stabilizers (rows N..2N-1): Z_i
    for (int i = 0; i < n_qubits; ++i) {
        set_xz(i, i, true);
        set_xz(n_qubits + i, n_qubits + i, true);
    }
}

// ----- rowmult -----
// dest = dest * src in the Pauli group.
//
// The X/Z merge is an XOR of the packed rows. The phase is the accumulated
// power of i from multiplying the two Paulis qubit by qubit, per
// Aaronson-Gottesman 2004 Table 1, with dest as the LEFT factor. Each qubit
// contributes -1, 0 or +1, so the sum is the count of +1 qubits minus the count
// of -1 qubits, and each of those two sets is a boolean function of the four
// bit planes:
//
//   a = x_dest   b = z_dest   c = x_src   d = z_src
//   +1 when  (a & ~b & c & d) | (~a & b & c & ~d) | (a & b & ~c & d)
//   -1 when  (a & ~b & ~c & d) | (~a & b & c & d) | (a & b & c & ~d)
//
// Both are evaluated 64 qubits at a time and counted with popcount, so the
// phase costs O(N/64) word operations rather than O(N) branches. Reading the
// planes needs bits64 because the Z plane starts at bit N, which is generally
// not a word boundary.
void StabilizerState::rowmult(int dest, int src) {
    const int N = n_qubits;
    const int chunks = (N + 63) / 64;

    int n_pos = 0;
    int n_neg = 0;
    for (int w = 0; w < chunks; ++w) {
        const int bit = 64 * w;
        const int valid = (N - bit) >= 64 ? 64 : (N - bit);
        const uint64_t mask =
            (valid == 64) ? ~0ULL : ((1ULL << valid) - 1ULL);

        const uint64_t a = bits64(dest, bit);
        const uint64_t b = bits64(dest, N + bit);
        const uint64_t c = bits64(src,  bit);
        const uint64_t d = bits64(src,  N + bit);

        const uint64_t plus  = (a & ~b &  c &  d)
                             | (~a &  b &  c & ~d)
                             | (a &  b & ~c &  d);
        const uint64_t minus = (a & ~b & ~c &  d)
                             | (~a &  b &  c &  d)
                             | (a &  b &  c & ~d);

        // Masking here rather than on each plane: the qubits past N in the last
        // chunk read whatever the neighbouring plane or the padding holds, and
        // only the count has to exclude them.
        n_pos += std::popcount(plus  & mask);
        n_neg += std::popcount(minus & mask);
    }

    const int cur = ph[dest] ? 2 : 0;
    const int sp  = ph[src]  ? 2 : 0;
    const int np  = ((cur + sp + n_pos - n_neg) % 4 + 4) % 4;
    ph[dest] = (np == 2) ? 1 : 0;

    xor_row(dest, src);
}

// ----- Clifford gate applications -----

void StabilizerState::apply_h(int qubit) {
    detail::check_qubit(qubit, n_qubits, "StabilizerState::apply_h");
    const int N = n_qubits;
    for (int i = 0; i < 2 * N; ++i) {
        const bool x = get_xz(i, qubit);
        const bool z = get_xz(i, N + qubit);
        if (x && z) ph[i] ^= 1;
        set_xz(i, qubit,     z);
        set_xz(i, N + qubit, x);
    }
}

void StabilizerState::apply_s(int qubit) {
    detail::check_qubit(qubit, n_qubits, "StabilizerState::apply_s");
    const int N = n_qubits;
    for (int i = 0; i < 2 * N; ++i) {
        if (get_xz(i, qubit)) {
            ph[i] ^= (uint8_t)get_xz(i, N + qubit);
            flip_xz(i, N + qubit);
        }
    }
}

void StabilizerState::apply_sdg(int qubit) {
    detail::check_qubit(qubit, n_qubits, "StabilizerState::apply_sdg");
    const int N = n_qubits;
    for (int i = 0; i < 2 * N; ++i) {
        if (get_xz(i, qubit)) {
            flip_xz(i, N + qubit);
            ph[i] ^= (uint8_t)get_xz(i, N + qubit);
        }
    }
}

void StabilizerState::apply_cx(int control, int target) {
    detail::check_qubit(control, n_qubits, "StabilizerState::apply_cx");
    detail::check_qubit(target, n_qubits, "StabilizerState::apply_cx");
    detail::check_distinct2(control, target, "StabilizerState::apply_cx");
    const int N = n_qubits;
    for (int i = 0; i < 2 * N; ++i) {
        const bool xc = get_xz(i, control);
        const bool xt = get_xz(i, target);
        const bool zc = get_xz(i, N + control);
        const bool zt = get_xz(i, N + target);
        if (xc && zt && !(xt ^ zc)) ph[i] ^= 1;
        if (xc) flip_xz(i, target);       // X_target ^= X_control
        if (zt) flip_xz(i, N + control);  // Z_control ^= Z_target
    }
}

void StabilizerState::apply_x(int qubit) {
    detail::check_qubit(qubit, n_qubits, "StabilizerState::apply_x");
    const int N = n_qubits;
    for (int i = 0; i < 2 * N; ++i) {
        if (get_xz(i, N + qubit)) ph[i] ^= 1;
    }
}

void StabilizerState::apply_y(int qubit) {
    detail::check_qubit(qubit, n_qubits, "StabilizerState::apply_y");
    const int N = n_qubits;
    for (int i = 0; i < 2 * N; ++i) {
        if (get_xz(i, qubit) ^ get_xz(i, N + qubit)) ph[i] ^= 1;
    }
}

void StabilizerState::apply_z(int qubit) {
    detail::check_qubit(qubit, n_qubits, "StabilizerState::apply_z");
    const int N = n_qubits;
    for (int i = 0; i < 2 * N; ++i) {
        if (get_xz(i, qubit)) ph[i] ^= 1;
    }
}

// ----- composed gates, one sweep each -----

// Row-local forms of the tableau primitives above. Each takes one row's bits
// for the qubits it acts on, plus that row's phase, by reference, so a sequence
// composes entirely in registers. The bodies are the updates apply_h, apply_s,
// apply_sdg, apply_x and apply_cx perform, with the packed-bit access hoisted
// out into the caller's sweep: the sweep is the cost, so a composed gate pays
// for one rather than one per primitive.
namespace {

inline void row_h(unsigned& x, unsigned& z, uint8_t& r) {
    r ^= static_cast<uint8_t>(x & z);
    const unsigned t = x;
    x = z;
    z = t;
}

inline void row_s(unsigned& x, unsigned& z, uint8_t& r) {
    r ^= static_cast<uint8_t>(x & z);
    z ^= x;
}

inline void row_sdg(unsigned& x, unsigned& z, uint8_t& r) {
    if (x) {
        z ^= 1u;
        r ^= static_cast<uint8_t>(z);
    }
}

inline void row_x(unsigned& /*x*/, unsigned& z, uint8_t& r) {
    r ^= static_cast<uint8_t>(z);
}

inline void row_cx(unsigned& xc, unsigned& zc, unsigned& xt, unsigned& zt, uint8_t& r) {
    r ^= static_cast<uint8_t>(xc & zt & (xt ^ zc ^ 1u));
    xt ^= xc;
    zc ^= zt;
}

}  // namespace

// SX = h . s . h collapses to a closed form: the two Hadamards cancel out of
// the bit map, leaving x ^= z, and the three sign terms reduce to z & ~x.
// Action: X -> X, Z -> -Y, Y -> Z.
void StabilizerState::apply_sx(int qubit) {
    detail::check_qubit(qubit, n_qubits, "StabilizerState::apply_sx");
    const int N = n_qubits;
    for (int i = 0; i < 2 * N; ++i) {
        const bool x = get_xz(i, qubit);
        const bool z = get_xz(i, N + qubit);
        if (z && !x) ph[i] ^= 1;
        set_xz(i, qubit, x != z);
    }
}

// SXDG = h . sdg . h, same bit map as SX with the sign term x & z.
// Action: X -> X, Z -> Y, Y -> -Z.
void StabilizerState::apply_sxdg(int qubit) {
    detail::check_qubit(qubit, n_qubits, "StabilizerState::apply_sxdg");
    const int N = n_qubits;
    for (int i = 0; i < 2 * N; ++i) {
        const bool x = get_xz(i, qubit);
        const bool z = get_xz(i, N + qubit);
        if (x && z) ph[i] ^= 1;
        set_xz(i, qubit, x != z);
    }
}

// Loads both qubits' bits and the phase for row i, runs BODY on the locals
// xa/za/xb/zb/r, and stores back. The two-qubit compositions differ only in
// BODY, and writing the load and store once keeps them from drifting apart.
#define LINDBLAD_TABLEAU_SWEEP2(qa, qb, BODY)                        \
    do {                                                             \
        const int N_ = n_qubits;                                     \
        for (int i = 0; i < 2 * N_; ++i) {                           \
            unsigned xa = get_xz(i, (qa)), za = get_xz(i, N_ + (qa));\
            unsigned xb = get_xz(i, (qb)), zb = get_xz(i, N_ + (qb));\
            uint8_t r = ph[i];                                       \
            BODY                                                     \
            set_xz(i, (qa), xa != 0u);                               \
            set_xz(i, N_ + (qa), za != 0u);                          \
            set_xz(i, (qb), xb != 0u);                               \
            set_xz(i, N_ + (qb), zb != 0u);                          \
            ph[i] = r;                                               \
        }                                                            \
    } while (0)

// CZ = h(a) . cx(b,a) . h(a). Symmetric in its operands, as the gate is.
void StabilizerState::apply_cz(int q1, int q2) {
    detail::check_qubit(q1, n_qubits, "StabilizerState::apply_cz");
    detail::check_qubit(q2, n_qubits, "StabilizerState::apply_cz");
    detail::check_distinct2(q1, q2, "StabilizerState::apply_cz");
    LINDBLAD_TABLEAU_SWEEP2(q1, q2,
        row_h(xa, za, r);
        row_cx(xb, zb, xa, za, r);
        row_h(xa, za, r);
    );
}

// SWAP = cx(a,b) . cx(b,a) . cx(a,b).
void StabilizerState::apply_swap(int q1, int q2) {
    detail::check_qubit(q1, n_qubits, "StabilizerState::apply_swap");
    detail::check_qubit(q2, n_qubits, "StabilizerState::apply_swap");
    detail::check_distinct2(q1, q2, "StabilizerState::apply_swap");
    LINDBLAD_TABLEAU_SWEEP2(q1, q2,
        row_cx(xa, za, xb, zb, r);
        row_cx(xb, zb, xa, za, r);
        row_cx(xa, za, xb, zb, r);
    );
}

// CY = sdg(target) . cx(ctrl,target) . s(target).
void StabilizerState::apply_cy(int control, int target) {
    detail::check_qubit(control, n_qubits, "StabilizerState::apply_cy");
    detail::check_qubit(target, n_qubits, "StabilizerState::apply_cy");
    detail::check_distinct2(control, target, "StabilizerState::apply_cy");
    LINDBLAD_TABLEAU_SWEEP2(control, target,
        row_sdg(xb, zb, r);
        row_cx(xa, za, xb, zb, r);
        row_s(xb, zb, r);
    );
}

// ISWAP = cx(a,b) . s(b) . cx(b,a) . cx(a,b).
void StabilizerState::apply_iswap(int q1, int q2) {
    detail::check_qubit(q1, n_qubits, "StabilizerState::apply_iswap");
    detail::check_qubit(q2, n_qubits, "StabilizerState::apply_iswap");
    detail::check_distinct2(q1, q2, "StabilizerState::apply_iswap");
    LINDBLAD_TABLEAU_SWEEP2(q1, q2,
        row_cx(xa, za, xb, zb, r);
        row_s(xb, zb, r);
        row_cx(xb, zb, xa, za, r);
        row_cx(xa, za, xb, zb, r);
    );
}

// ECR = h(b) . s(a) . s(b) . h(b) . cx(a,b) . x(a).
// The transpiler's ECR identity routes through RZX(±π/4), whose middle rotation
// is a T gate, so that sequence cannot run on a tableau even though the product
// is Clifford. This one is built from tableau primitives throughout.
void StabilizerState::apply_ecr(int q1, int q2) {
    detail::check_qubit(q1, n_qubits, "StabilizerState::apply_ecr");
    detail::check_qubit(q2, n_qubits, "StabilizerState::apply_ecr");
    detail::check_distinct2(q1, q2, "StabilizerState::apply_ecr");
    LINDBLAD_TABLEAU_SWEEP2(q1, q2,
        row_h(xb, zb, r);
        row_s(xa, za, r);
        row_s(xb, zb, r);
        row_h(xb, zb, r);
        row_cx(xa, za, xb, zb, r);
        row_x(xa, za, r);
    );
}

#undef LINDBLAD_TABLEAU_SWEEP2

// ----- measurement -----

int StabilizerState::measure(int qubit, bool random, std::mt19937_64& rng) {
    detail::check_qubit(qubit, n_qubits, "StabilizerState::measure");
    const int N = n_qubits;

    // Find a stabilizer (rows N..2N-1) with X bit set on this qubit.
    int p = -1;
    for (int i = N; i < 2 * N; ++i) {
        if (get_xz(i, qubit)) { p = i; break; }
    }

    if (p >= 0) {
        // Random outcome.
        // Eliminate X bit on this qubit from all other rows via rowmult with row p.
        for (int i = 0; i < 2 * N; ++i) {
            if (i != p && get_xz(i, qubit)) rowmult(i, p);
        }

        // Move stabilizer p to destabilizer slot p-N.
        copy_row(p - N, p);

        // Reset stabilizer p to ±Z on this qubit.
        zero_row(p);
        set_xz(p, N + qubit, true);

        int result = 0;
        if (random) {
            std::uniform_int_distribution<int> dist(0, 1);
            result = dist(rng);
        }
        ph[p] = (uint8_t)result;
        return result;
    } else {
        // Deterministic outcome.
        // Use a scratch row to accumulate the product of destabilizers.
        push_scratch();
        const int scratch = num_rows - 1;
        zero_row(scratch);
        set_xz(scratch, N + qubit, true);  // start with Z on this qubit

        for (int i = 0; i < N; ++i) {
            if (get_xz(i, qubit)) rowmult(scratch, i + N);
        }

        const int outcome = ph[scratch] ? 1 : 0;
        pop_scratch();
        return outcome;
    }
}

// ----- expectation_pauli -----

int StabilizerState::expectation_pauli(const std::string& pauli) const {
    const int N = n_qubits;
    if (static_cast<int>(pauli.size()) != N) {
        throw std::invalid_argument("Pauli string length must match n_qubits");
    }

    // Build the target Pauli's X and Z bits.
    // Phase exponent mod 4: 0=+1, 1=+i, 2=-1, 3=-i.
    // Y = iXZ contributes one factor of i per Y qubit.
    std::vector<bool> px(N, false), pz(N, false);
    int p_phase = 0;
    for (int i = 0; i < N; ++i) {
        const char c = pauli[i];
        if      (c == 'X' || c == 'x') { px[i] = true; }
        else if (c == 'Y' || c == 'y') { px[i] = true; pz[i] = true; }
        else if (c == 'Z' || c == 'z') { pz[i] = true; }
    }

    // Check if P commutes with each stabilizer.
    // P anticommutes with g iff symplectic inner product is 1:
    // anti = sum_j (px_j & gz_j) XOR (pz_j & gx_j)  (mod 2)
    for (int s = N; s < 2 * N; ++s) {
        int anti = 0;
        for (int j = 0; j < N; ++j) {
            if (px[j] && get_xz(s, N + j)) anti ^= 1;
            if (pz[j] && get_xz(s, j))     anti ^= 1;
        }
        if (anti & 1) return 0;  // anticommutes → ⟨P⟩ = 0
    }

    // P commutes with all stabilizers.
    // Express P as a product of stabilizers via GF(2) Gaussian elimination.
    // Working copy of stabilizer block (rows N..2N-1).
    const int rows = N;
    const int cols = 2 * N;
    std::vector<std::vector<bool>> mat(rows, std::vector<bool>(cols, false));
    std::vector<int> mat_phase(rows, 0);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) mat[i][j] = get_xz(N + i, j);
        mat_phase[i] = ph[N + i] ? 2 : 0;
    }

    // Target row: [px | pz] with accumulated phase
    std::vector<bool> target(cols, false);
    int target_phase = p_phase;
    for (int j = 0; j < N; ++j) target[j] = px[j];
    for (int j = 0; j < N; ++j) target[N + j] = pz[j];

    // Phase of row1 * row2 per Aaronson-Gottesman Table 1.
    // ph1, ph2, ph_out are exponents mod 4.
    auto pauli_phase_update = [&](const std::vector<bool>& row1, int ph1,
                                   const std::vector<bool>& row2, int ph2,
                                   int& ph_out) {
        int phase_count = 0;
        for (int j = 0; j < N; ++j) {
            const bool x1 = row1[j], z1 = row1[N + j];
            const bool x2 = row2[j], z2 = row2[N + j];
            if      ( x1 &&  z1 &&  x2 && !z2) phase_count--;  // Y*X = -iZ
            else if ( x1 &&  z1 && !x2 &&  z2) phase_count++;  // Y*Z = +iX
            else if ( x1 && !z1 &&  x2 &&  z2) phase_count++;  // X*Y = +iZ
            else if (!x1 &&  z1 &&  x2 &&  z2) phase_count--;  // Z*Y = -iX
            else if ( x1 && !z1 && !x2 &&  z2) phase_count--;  // X*Z = -iY
            else if (!x1 &&  z1 &&  x2 && !z2) phase_count++;  // Z*X = +iY
        }
        ph_out = ((ph1 + ph2 + phase_count) % 4 + 4) % 4;
    };

    int pivot_col = 0;
    for (int col = 0; col < cols && pivot_col < rows; ++col) {
        int pivot = -1;
        for (int r = pivot_col; r < rows; ++r) {
            if (mat[r][col]) { pivot = r; break; }
        }
        if (pivot < 0) continue;

        std::swap(mat[pivot], mat[pivot_col]);
        std::swap(mat_phase[pivot], mat_phase[pivot_col]);

        for (int r = 0; r < rows; ++r) {
            if (r != pivot_col && mat[r][col]) {
                int new_ph;
                pauli_phase_update(mat[r], mat_phase[r], mat[pivot_col], mat_phase[pivot_col], new_ph);
                for (int j = 0; j < cols; ++j) mat[r][j] = mat[r][j] ^ mat[pivot_col][j];
                mat_phase[r] = new_ph;
            }
        }

        if (target[col]) {
            int new_ph;
            pauli_phase_update(target, target_phase, mat[pivot_col], mat_phase[pivot_col], new_ph);
            for (int j = 0; j < cols; ++j) target[j] = target[j] ^ mat[pivot_col][j];
            target_phase = new_ph;
        }

        pivot_col++;
    }

    // If target is all-zero, P is in the stabilizer group; phase gives the eigenvalue.
    for (int j = 0; j < cols; ++j) {
        if (target[j]) return 0;  // not in stabilizer group
    }

    return (target_phase == 0) ? +1 : (target_phase == 2) ? -1 : 0;
}

// ----- entanglement_entropy_bits -----

double StabilizerState::entanglement_entropy_bits(const std::vector<int>& region) const {
    const int N = n_qubits;

    std::vector<bool> in_region(static_cast<std::size_t>(N), false);
    for (const int q : region) {
        if (q < 0 || q >= N) {
            throw std::invalid_argument(
                "StabilizerState::entanglement_entropy_bits: qubit " +
                std::to_string(q) + " is outside a " + std::to_string(N) +
                " qubit register");
        }
        if (in_region[static_cast<std::size_t>(q)]) {
            throw std::invalid_argument(
                "StabilizerState::entanglement_entropy_bits: qubit " +
                std::to_string(q) + " is named twice; a cut has each qubit on "
                "one side of it");
        }
        in_region[static_cast<std::size_t>(q)] = true;
    }
    if (region.empty() || static_cast<int>(region.size()) == N) return 0.0;

    // Rows are the N stabilizer generators restricted to the region's X and Z
    // columns, two bits per region qubit. The entropy is the GF(2) rank of that
    // matrix minus the region size.
    //
    // Word-packed like the rest of this file: the elimination's inner step is a
    // word XOR over the remaining columns rather than a bit at a time, which is
    // the whole reason the tableau is stored this way.
    //
    // No phases and no Pauli products appear here, so this is not the tableau
    // elimination outcome_slab runs and cannot call it: that one multiplies
    // rows in the Pauli group, carrying the Z plane and the sign along, and a
    // rank needs none of that.
    const int width = 2 * static_cast<int>(region.size());
    const int wpr = (width + 63) / 64;

    std::vector<std::uint64_t> rows(static_cast<std::size_t>(N) *
                                        static_cast<std::size_t>(wpr), 0ULL);
    auto row = [&](int r) { return rows.data() + static_cast<std::size_t>(r) * wpr; };
    auto bit_at = [&](const std::uint64_t* v, int c) {
        return ((v[c / 64] >> (c % 64)) & 1ULL) != 0;
    };

    for (int s = 0; s < N; ++s) {
        std::uint64_t* dst = row(s);
        for (std::size_t slot = 0; slot < region.size(); ++slot) {
            const int q = region[slot];
            const int cx = 2 * static_cast<int>(slot);
            const int cz = cx + 1;
            if (get_xz(N + s, q))     dst[cx / 64] |= 1ULL << (cx % 64);
            if (get_xz(N + s, N + q)) dst[cz / 64] |= 1ULL << (cz % 64);
        }
    }

    // Forward elimination only. A rank counts pivots, so rows ABOVE a pivot
    // never need clearing and the back substitution a reduced echelon form
    // would do is pure cost here.
    int rank = 0;
    for (int col = 0; col < width && rank < N; ++col) {
        int pivot = -1;
        for (int r = rank; r < N; ++r) {
            if (bit_at(row(r), col)) { pivot = r; break; }
        }
        if (pivot < 0) continue;

        if (pivot != rank) {
            for (int w = col / 64; w < wpr; ++w) {
                std::swap(row(rank)[w], row(pivot)[w]);
            }
        }

        const std::uint64_t* pivot_row = row(rank);
        for (int r = rank + 1; r < N; ++r) {
            std::uint64_t* target = row(r);
            if (!bit_at(target, col)) continue;
            // Columns before this one are already zero in both rows, so the
            // XOR starts at the pivot's own word.
            for (int w = col / 64; w < wpr; ++w) target[w] ^= pivot_row[w];
        }
        ++rank;
    }

    return static_cast<double>(rank) - static_cast<double>(region.size());
}

// ----- to_statevector -----

// i^k for k in [0, 4), the only complex constants the projector produces.
static inline Complex128 i_power(int k) {
    switch (k & 3) {
        case 0:  return {1.0, 0.0};
        case 1:  return {0.0, 1.0};
        case 2:  return {-1.0, 0.0};
        default: return {0.0, -1.0};
    }
}

Statevector StabilizerState::to_statevector() const {
    const int N = n_qubits;

    // 2^N must be addressable before anything else is worth attempting. The
    // tableau is happy at thousands of qubits; the dense form it is being asked
    // for is not, and a caller learns that here rather than from an allocator.
    if (N < 0 || N > 62) {
        throw std::invalid_argument(
            "StabilizerState::to_statevector: 2^n amplitudes are not "
            "representable for n = " + std::to_string(N));
    }
    const std::size_t dim = std::size_t{1} << N;

    // One basis state with nonzero overlap. The slab's offset is a point of the
    // outcome coset, so it is in the support by construction, which is exactly
    // the condition the projector needs to return something nonzero.
    const OutcomeSlab slab = outcome_slab();
    std::uint64_t start = slab.offset.empty() ? 0ULL : slab.offset[0];
    if (N < 64) start &= (1ULL << N) - 1ULL;

    std::unordered_map<std::uint64_t, Complex128> psi;
    psi.emplace(start, Complex128{1.0, 0.0});

    // psi <- (I + g_s) psi / 2 for each stabilizer generator. The product over
    // all N of them is the projector onto the code space, which is
    // one-dimensional, so what survives is the state itself up to scale.
    for (int s = N; s < 2 * N; ++s) {
        std::unordered_map<std::uint64_t, Complex128> next = psi;  // the I term

        for (const auto& [index, amp] : psi) {
            std::uint64_t out = index;
            int i_pow = ph[s] ? 2 : 0;  // the row's sign, as i^2 = -1

            for (int q = 0; q < N; ++q) {
                const bool xq = get_xz(s, q);
                const bool zq = get_xz(s, N + q);

                // Y = iXZ, so a qubit carrying both bits contributes one i.
                if (xq && zq) ++i_pow;
                // Z reads the ORIGINAL index: it acts before X flips anything.
                if (zq && ((index >> q) & 1ULL)) i_pow += 2;
                if (xq) out ^= (1ULL << q);
            }

            next[out] += amp * i_power(i_pow);
        }

        // Amplitudes here are exact sums of powers of i scaled by powers of
        // one half, so a cancelled term is exactly zero and can be dropped
        // rather than left to grow the map for the remaining generators.
        psi.clear();
        for (const auto& [index, amp] : next) {
            const Complex128 halved = amp * 0.5;
            if (halved.real != 0.0 || halved.imag != 0.0) psi.emplace(index, halved);
        }
    }

    double norm_squared = 0.0;
    for (const auto& [index, amp] : psi) {
        norm_squared += amp.real * amp.real + amp.imag * amp.imag;
    }
    if (!(norm_squared > 0.0) || !is_finite_strict(norm_squared)) {
        throw std::runtime_error(
            "StabilizerState::to_statevector: the stabilizer projector "
            "produced no state, which means the tableau's generators are not "
            "independent");
    }
    const double inv_norm = 1.0 / std::sqrt(norm_squared);

    std::vector<double> real_parts(dim, 0.0);
    std::vector<double> imag_parts(dim, 0.0);
    for (const auto& [index, amp] : psi) {
        real_parts[index] = amp.real * inv_norm;
        imag_parts[index] = amp.imag * inv_norm;
    }

    Statevector sv(N);
    sv.set_amplitudes(real_parts.data(), imag_parts.data(), dim);
    return sv;
}

// ----- outcome slab -----

// A caller who selects the block elimination is told what it costs at ordinary
// sizes. Deduplication belongs to the warning channel, which delivers the first
// occurrence of a message and counts the rest until the next flush, so this
// says it every time and the channel decides what reaches the handler.
static void warn_four_russians_slower() {
    emit_warning(
        "note: Elimination::FourRussians selected for the Clifford outcome "
        "slab. Plain elimination is the default and is faster at ordinary "
        "sizes (measured 0.91x to 0.98x across n = 20 to 160), because it "
        "multiplies only where a pivot bit is set while the block table costs "
        "2^k multiplications regardless. The block route needs n well above "
        "2^k to repay that table; both produce the same outcome distribution.");
}

// dest *= src in the Pauli group, for the elimination's plane-aligned rows.
// Same phase rule as rowmult, but here the X and Z planes are separately word
// aligned, so the four reads are direct loads and no bits past N are ever set,
// which is why no masking is needed before the popcounts.
namespace {

inline void erow_mult(uint64_t* dx, uint64_t* dz, uint8_t& dr,
                      const uint64_t* sx, const uint64_t* sz, uint8_t sr,
                      int xw) {
    int n_pos = 0;
    int n_neg = 0;
    for (int w = 0; w < xw; ++w) {
        const uint64_t a = dx[w], b = dz[w], c = sx[w], d = sz[w];
        const uint64_t plus  = (a & ~b &  c &  d)
                             | (~a &  b &  c & ~d)
                             | (a &  b & ~c &  d);
        const uint64_t minus = (a & ~b & ~c &  d)
                             | (~a &  b &  c &  d)
                             | (a &  b &  c & ~d);
        n_pos += std::popcount(plus);
        n_neg += std::popcount(minus);
    }
    const int np = ((2 * dr + 2 * sr + n_pos - n_neg) % 4 + 4) % 4;
    dr = static_cast<uint8_t>(np == 2 ? 1 : 0);
    for (int w = 0; w < xw; ++w) {
        dx[w] ^= sx[w];
        dz[w] ^= sz[w];
    }
}

}  // namespace

// Extracts the shape of the computational-basis outcome distribution.
//
// A stabilizer group element with an empty X part is ±Z^v, and it acts on a
// basis state |y> as (-1)^(v·y). Such an element therefore does not randomise
// anything: it CONSTRAINS the outcome to v·y = b, where b is its sign bit.
// Elements with a non-empty X part impose no constraint at all. So the support
// is the solution set of a linear system, an affine subspace, and the
// amplitudes across it are equal in magnitude, which makes the distribution
// uniform over it.
//
// Finding the Z-only subgroup is a forward elimination on the X part of the
// stabilizer block: after it, every row below the last pivot has an empty X
// part, and those rows generate exactly that subgroup. The constraint system
// they give is then reduced to echelon form, which yields one particular
// solution and one free direction per non-pivot coordinate.
StabilizerState::OutcomeSlab StabilizerState::outcome_slab(Elimination method) const {
    const int N = n_qubits;
    const int W = words_per_vector();

    OutcomeSlab slab;
    slab.n_qubits = N;
    slab.offset.assign(static_cast<size_t>(W), 0ULL);
    if (N == 0) return slab;

    // The generators are copied out into a local buffer with the X and Z
    // planes SEPARATELY word aligned. In the tableau the Z plane starts at bit
    // N, so lining the planes up per qubit needs a spanning read; here they
    // line up by construction, and the elimination is the one place that cost
    // would be paid over and over.
    const int xw = W;  // both are ceil(N / 64)
    const int stride = 2 * xw;
    std::vector<uint64_t> buf(static_cast<size_t>(N) * static_cast<size_t>(stride), 0ULL);
    std::vector<uint8_t> phs(static_cast<size_t>(N), 0);

    auto rx = [&](int r) { return &buf[static_cast<size_t>(r) * static_cast<size_t>(stride)]; };
    auto rz = [&](int r) { return rx(r) + xw; };
    auto bit_at = [](const uint64_t* v, int i) {
        return ((v[static_cast<size_t>(i / 64)] >> (i % 64)) & 1ULL) != 0;
    };

    // Copied a word at a time rather than a qubit at a time: bit by bit this
    // would be 2*N*N reads for data that is 2*xw words per row. The mask drops
    // what each read picks up past qubit N, which is the neighbouring plane for
    // the X half and the row padding for the Z half.
    std::vector<uint64_t> tail_mask(static_cast<size_t>(xw), ~0ULL);
    for (int w = 0; w < xw; ++w) {
        const int valid = (N - 64 * w) >= 64 ? 64 : (N - 64 * w);
        if (valid < 64) {
            tail_mask[static_cast<size_t>(w)] = (1ULL << valid) - 1ULL;
        }
    }
    for (int r = 0; r < N; ++r) {
        for (int w = 0; w < xw; ++w) {
            const uint64_t m = tail_mask[static_cast<size_t>(w)];
            rx(r)[w] = bits64(N + r, 64 * w) & m;
            rz(r)[w] = bits64(N + r, N + 64 * w) & m;
        }
        phs[static_cast<size_t>(r)] = ph[static_cast<size_t>(N + r)];
    }

    auto swap_erows = [&](int i, int j) {
        if (i == j) return;
        for (int w = 0; w < stride; ++w) std::swap(rx(i)[w], rx(j)[w]);
        std::swap(phs[static_cast<size_t>(i)], phs[static_cast<size_t>(j)]);
    };

    // How the elimination is carried out is the caller's choice, because the
    // two routes suit different sizes. See StabilizerState::Elimination.
    int pivot = 0;
    if (method == Elimination::FourRussians) {
        warn_four_russians_slower();

        // Method of Four Russians: clear a block of kBlock pivot columns from every
        // remaining row with ONE multiplication each, against a precomputed table
        // of the products of every subset of the block's pivots. The table costs
        // 2^kBlock multiplications and saves up to kBlock per remaining row, so it
        // pays whenever the rows below outnumber the table, which is the usual case
        // at these sizes.
        constexpr int kBlock = 6;
        std::vector<uint64_t> table(static_cast<size_t>(1 << kBlock) * static_cast<size_t>(stride), 0ULL);
        std::vector<uint8_t> tphs(static_cast<size_t>(1 << kBlock), 0);

        int col = 0;
        std::vector<int> block_cols;
        while (col < N && pivot < N) {
            const int block_start = pivot;
            block_cols.clear();

            while (col < N && static_cast<int>(block_cols.size()) < kBlock && pivot < N) {
                // A row promoted from below can still carry bits in the block's
                // earlier columns, and clearing those can also clear the column it
                // was picked for. So each candidate is reduced against the pivots
                // already chosen BEFORE it is tested. Candidates passed over stay
                // reduced, which is progress: their mask below comes out zero.
                int found = -1;
                for (int i = pivot; i < N; ++i) {
                    for (size_t j = 0; j < block_cols.size(); ++j) {
                        const int c2 = block_cols[j];
                        if (bit_at(rx(i), c2)) {
                            const int pr = block_start + static_cast<int>(j);
                            erow_mult(rx(i), rz(i), phs[static_cast<size_t>(i)],
                                      rx(pr), rz(pr), phs[static_cast<size_t>(pr)], xw);
                        }
                    }
                    if (bit_at(rx(i), col)) { found = i; break; }
                }
                if (found < 0) { ++col; continue; }
                swap_erows(found, pivot);
                block_cols.push_back(col);
                ++pivot;
                ++col;
            }

            const int k = static_cast<int>(block_cols.size());
            if (k == 0) continue;

            // The mask below is read from a row once and selects a single table
            // entry, which is only sound when each block pivot is the only one
            // carrying its own column. Chosen pivots are upper triangular on the
            // block columns, so back-substitute to clear the rest.
            for (int j = k - 1; j >= 0; --j) {
                for (int i = 0; i < j; ++i) {
                    if (bit_at(rx(block_start + i), block_cols[static_cast<size_t>(j)])) {
                        erow_mult(rx(block_start + i), rz(block_start + i),
                                  phs[static_cast<size_t>(block_start + i)],
                                  rx(block_start + j), rz(block_start + j),
                                  phs[static_cast<size_t>(block_start + j)], xw);
                    }
                }
            }

            const size_t entries = static_cast<size_t>(1) << k;
            std::fill(table.begin(), table.begin() + static_cast<long>(entries * static_cast<size_t>(stride)), 0ULL);
            std::fill(tphs.begin(), tphs.begin() + static_cast<long>(entries), 0);
            for (size_t mask = 1; mask < entries; ++mask) {
                const int j = std::countr_zero(static_cast<unsigned>(mask));
                const size_t prev = mask ^ (static_cast<size_t>(1) << j);
                uint64_t* tx = &table[mask * static_cast<size_t>(stride)];
                const uint64_t* px = &table[prev * static_cast<size_t>(stride)];
                std::copy(px, px + stride, tx);
                tphs[mask] = tphs[prev];
                const int pr = block_start + j;
                erow_mult(tx, tx + xw, tphs[mask],
                          rx(pr), rz(pr), phs[static_cast<size_t>(pr)], xw);
            }

            for (int i = pivot; i < N; ++i) {
                size_t mask = 0;
                for (int j = 0; j < k; ++j) {
                    if (bit_at(rx(i), block_cols[static_cast<size_t>(j)])) {
                        mask |= static_cast<size_t>(1) << j;
                    }
                }
                if (mask) {
                    uint64_t* tx = &table[mask * static_cast<size_t>(stride)];
                    erow_mult(rx(i), rz(i), phs[static_cast<size_t>(i)],
                              tx, tx + xw, tphs[mask], xw);
                }
            }
        }
    } else {
        // Forward elimination on the X part, multiplying only where the bit is
        // actually set. Every row below the last pivot then has an empty X
        // part, and those rows generate the Z-only subgroup.
        for (int col = 0; col < N && pivot < N; ++col) {
            int p = -1;
            for (int i = pivot; i < N; ++i) {
                if (bit_at(rx(i), col)) { p = i; break; }
            }
            if (p < 0) continue;
            swap_erows(p, pivot);
            for (int i = pivot + 1; i < N; ++i) {
                if (bit_at(rx(i), col)) {
                    erow_mult(rx(i), rz(i), phs[static_cast<size_t>(i)],
                              rx(pivot), rz(pivot),
                              phs[static_cast<size_t>(pivot)], xw);
                }
            }
            ++pivot;
        }
    }


    // Rows past the last pivot have an empty X part: the Z-only subgroup. Their
    // Z parts pack identically to an outcome vector, so they are the constraint
    // rows directly.
    const int n_con = N - pivot;
    std::vector<std::vector<uint64_t>> V(
        static_cast<size_t>(n_con), std::vector<uint64_t>(static_cast<size_t>(W), 0ULL));
    std::vector<uint8_t> rhs(static_cast<size_t>(n_con), 0);
    for (int i = 0; i < n_con; ++i) {
        const int row = pivot + i;
        std::copy(rz(row), rz(row) + xw, V[static_cast<size_t>(i)].begin());
        rhs[static_cast<size_t>(i)] = phs[static_cast<size_t>(row)];
    }

    std::vector<int> pivot_col;
    int rank = 0;
    for (int col = 0; col < N && rank < n_con; ++col) {
        const size_t cw = static_cast<size_t>(col / 64);
        const uint64_t cb = 1ULL << (col % 64);
        int p = -1;
        for (int i = rank; i < n_con; ++i) {
            if (V[static_cast<size_t>(i)][cw] & cb) { p = i; break; }
        }
        if (p < 0) continue;
        std::swap(V[static_cast<size_t>(rank)], V[static_cast<size_t>(p)]);
        std::swap(rhs[static_cast<size_t>(rank)], rhs[static_cast<size_t>(p)]);
        for (int i = 0; i < n_con; ++i) {
            if (i == rank) continue;
            if (V[static_cast<size_t>(i)][cw] & cb) {
                for (int w = 0; w < W; ++w) {
                    V[static_cast<size_t>(i)][static_cast<size_t>(w)] ^=
                        V[static_cast<size_t>(rank)][static_cast<size_t>(w)];
                }
                rhs[static_cast<size_t>(i)] ^= rhs[static_cast<size_t>(rank)];
            }
        }
        pivot_col.push_back(col);
        ++rank;
    }

    // Rows past the rank are all-zero on the left. A non-zero right-hand side
    // there would mean 0 = 1, which no state can satisfy, so it is a defect in
    // the tableau rather than an input a caller can produce.
    for (int i = rank; i < n_con; ++i) {
        if (rhs[static_cast<size_t>(i)]) {
            throw std::logic_error(
                "StabilizerState::outcome_slab: stabilizer constraints are "
                "inconsistent; no computational-basis outcome satisfies them");
        }
    }

    // Particular solution: free coordinates zero, each pivot coordinate taking
    // its own row's right-hand side.
    for (size_t i = 0; i < pivot_col.size(); ++i) {
        if (rhs[i]) {
            const int c = pivot_col[i];
            slab.offset[static_cast<size_t>(c / 64)] |= 1ULL << (c % 64);
        }
    }

    // One free direction per non-pivot coordinate: set that coordinate, then
    // set each pivot coordinate the reduced system couples to it.
    std::vector<bool> is_pivot(static_cast<size_t>(N), false);
    for (int c : pivot_col) is_pivot[static_cast<size_t>(c)] = true;
    for (int f = 0; f < N; ++f) {
        if (is_pivot[static_cast<size_t>(f)]) continue;
        std::vector<uint64_t> v(static_cast<size_t>(W), 0ULL);
        v[static_cast<size_t>(f / 64)] |= 1ULL << (f % 64);
        const size_t fw = static_cast<size_t>(f / 64);
        const uint64_t fb = 1ULL << (f % 64);
        for (size_t i = 0; i < pivot_col.size(); ++i) {
            if (V[i][fw] & fb) {
                const int c = pivot_col[i];
                v[static_cast<size_t>(c / 64)] |= 1ULL << (c % 64);
            }
        }
        slab.basis.push_back(std::move(v));
    }
    slab.dim = static_cast<int>(slab.basis.size());
    return slab;
}

// =============================================================================
// StabilizerState::ColumnTableau — bit-sliced gate pass
// =============================================================================

// Word-wide forms of the tableau primitives. Each lane of a word is one row, so
// a single update applies the rule to 64 rows at once, phases included.
namespace {

inline void w_h(uint64_t& x, uint64_t& z, uint64_t& r) {
    r ^= x & z;
    const uint64_t t = x;
    x = z;
    z = t;
}

inline void w_s(uint64_t& x, uint64_t& z, uint64_t& r) {
    r ^= x & z;
    z ^= x;
}

// The row form flips z where x is set and folds the NEW z into the sign, which
// as a word expression reads the OLD z through a complement.
inline void w_sdg(uint64_t& x, uint64_t& z, uint64_t& r) {
    r ^= x & ~z;
    z ^= x;
}

inline void w_x(uint64_t& /*x*/, uint64_t& z, uint64_t& r) {
    r ^= z;
}

inline void w_cx(uint64_t& xc, uint64_t& zc, uint64_t& xt, uint64_t& zt, uint64_t& r) {
    r ^= xc & zt & ~(xt ^ zc);
    xt ^= xc;
    zc ^= zt;
}

// In-place transpose of a 64x64 bit matrix held one row per word, by recursive
// interleaving of off-diagonal blocks: 1152 word operations where a bit-by-bit
// transpose would take 4096.
inline void transpose64(uint64_t a[64]) {
    int j = 32;
    uint64_t m = 0x00000000FFFFFFFFULL;
    while (j) {
        for (int k = 0; k < 64; k = (k + j + 1) & ~j) {
            const uint64_t t = ((a[k] >> j) ^ a[k + j]) & m;
            a[k]     ^= t << j;
            a[k + j] ^= t;
        }
        j >>= 1;
        m ^= (m << j);
    }
}

}  // namespace

StabilizerState::ColumnTableau::ColumnTableau(int n_qubits)
    : nq(checked_width(n_qubits, "StabilizerState::ColumnTableau")),
      rows(2 * n_qubits),
      wpc((2 * n_qubits + 63) / 64),
      ncol(((2 * n_qubits + 63) / 64) * 64),
      planes(static_cast<size_t>(ncol) * static_cast<size_t>(wpc), 0ULL),
      phase(static_cast<size_t>(wpc), 0ULL)
{
    // Same initial state as StabilizerState: destabilizer i is X_i, stabilizer
    // i is Z_i, so column c carries a single set bit at row c.
    for (int i = 0; i < nq; ++i) {
        col(i)[i / 64] |= 1ULL << (i % 64);
        col(nq + i)[(nq + i) / 64] |= 1ULL << ((nq + i) % 64);
    }
}

// Runs BODY once per word of the qubit's two columns, with the row-packed
// phases alongside. Every single-qubit gate is this loop and a different BODY.
// The pointers carry __restrict__ because they never overlap: a qubit's X and Z
// columns are distinct, the two-operand form is guarded by check_distinct2 so
// all four columns differ, and the phases are a separate allocation. Every
// column does live in one buffer, so without the qualifier a compiler must
// assume a store through one could alias a load through another. Measured
// neutral here, within run-to-run noise at n = 20 through 160.
#define LINDBLAD_COLUMN_SWEEP1(q, BODY)                                 \
    do {                                                                \
        uint64_t* __restrict__ xp = col(q);                             \
        uint64_t* __restrict__ zp = col(nq + (q));                      \
        uint64_t* __restrict__ pp = phase.data();                       \
        for (int w = 0; w < wpc; ++w) {                                 \
            uint64_t x = xp[w], z = zp[w];                              \
            uint64_t r = pp[w];                                         \
            BODY                                                        \
            xp[w] = x;                                                  \
            zp[w] = z;                                                  \
            pp[w] = r;                                                  \
        }                                                               \
    } while (0)

#define LINDBLAD_COLUMN_SWEEP2(qa, qb, BODY)                            \
    do {                                                                \
        uint64_t* __restrict__ xap = col(qa);                           \
        uint64_t* __restrict__ zap = col(nq + (qa));                    \
        uint64_t* __restrict__ xbp = col(qb);                           \
        uint64_t* __restrict__ zbp = col(nq + (qb));                    \
        uint64_t* __restrict__ pp  = phase.data();                      \
        for (int w = 0; w < wpc; ++w) {                                 \
            uint64_t xa = xap[w], za = zap[w];                          \
            uint64_t xb = xbp[w], zb = zbp[w];                          \
            uint64_t r = pp[w];                                         \
            BODY                                                        \
            xap[w] = xa;                                                \
            zap[w] = za;                                                \
            xbp[w] = xb;                                                \
            zbp[w] = zb;                                                \
            pp[w] = r;                                                  \
        }                                                               \
    } while (0)

void StabilizerState::ColumnTableau::apply_h(int qubit) {
    detail::check_qubit(qubit, nq, "StabilizerState::ColumnTableau::apply_h");
    LINDBLAD_COLUMN_SWEEP1(qubit, w_h(x, z, r););
}

void StabilizerState::ColumnTableau::apply_s(int qubit) {
    detail::check_qubit(qubit, nq, "StabilizerState::ColumnTableau::apply_s");
    LINDBLAD_COLUMN_SWEEP1(qubit, w_s(x, z, r););
}

void StabilizerState::ColumnTableau::apply_sdg(int qubit) {
    detail::check_qubit(qubit, nq, "StabilizerState::ColumnTableau::apply_sdg");
    LINDBLAD_COLUMN_SWEEP1(qubit, w_sdg(x, z, r););
}

void StabilizerState::ColumnTableau::apply_x(int qubit) {
    detail::check_qubit(qubit, nq, "StabilizerState::ColumnTableau::apply_x");
    LINDBLAD_COLUMN_SWEEP1(qubit, r ^= z;);
}

void StabilizerState::ColumnTableau::apply_y(int qubit) {
    detail::check_qubit(qubit, nq, "StabilizerState::ColumnTableau::apply_y");
    LINDBLAD_COLUMN_SWEEP1(qubit, r ^= x ^ z;);
}

void StabilizerState::ColumnTableau::apply_z(int qubit) {
    detail::check_qubit(qubit, nq, "StabilizerState::ColumnTableau::apply_z");
    LINDBLAD_COLUMN_SWEEP1(qubit, r ^= x;);
}

void StabilizerState::ColumnTableau::apply_sx(int qubit) {
    detail::check_qubit(qubit, nq, "StabilizerState::ColumnTableau::apply_sx");
    LINDBLAD_COLUMN_SWEEP1(qubit, r ^= z & ~x; x ^= z;);
}

void StabilizerState::ColumnTableau::apply_sxdg(int qubit) {
    detail::check_qubit(qubit, nq, "StabilizerState::ColumnTableau::apply_sxdg");
    LINDBLAD_COLUMN_SWEEP1(qubit, r ^= x & z; x ^= z;);
}

void StabilizerState::ColumnTableau::apply_cx(int control, int target) {
    detail::check_qubit(control, nq, "StabilizerState::ColumnTableau::apply_cx");
    detail::check_qubit(target, nq, "StabilizerState::ColumnTableau::apply_cx");
    detail::check_distinct2(control, target, "StabilizerState::ColumnTableau::apply_cx");
    LINDBLAD_COLUMN_SWEEP2(control, target, w_cx(xa, za, xb, zb, r););
}

void StabilizerState::ColumnTableau::apply_cz(int q1, int q2) {
    detail::check_qubit(q1, nq, "StabilizerState::ColumnTableau::apply_cz");
    detail::check_qubit(q2, nq, "StabilizerState::ColumnTableau::apply_cz");
    detail::check_distinct2(q1, q2, "StabilizerState::ColumnTableau::apply_cz");
    LINDBLAD_COLUMN_SWEEP2(q1, q2,
        w_h(xa, za, r);
        w_cx(xb, zb, xa, za, r);
        w_h(xa, za, r);
    );
}

void StabilizerState::ColumnTableau::apply_swap(int q1, int q2) {
    detail::check_qubit(q1, nq, "StabilizerState::ColumnTableau::apply_swap");
    detail::check_qubit(q2, nq, "StabilizerState::ColumnTableau::apply_swap");
    detail::check_distinct2(q1, q2, "StabilizerState::ColumnTableau::apply_swap");
    LINDBLAD_COLUMN_SWEEP2(q1, q2,
        w_cx(xa, za, xb, zb, r);
        w_cx(xb, zb, xa, za, r);
        w_cx(xa, za, xb, zb, r);
    );
}

void StabilizerState::ColumnTableau::apply_cy(int control, int target) {
    detail::check_qubit(control, nq, "StabilizerState::ColumnTableau::apply_cy");
    detail::check_qubit(target, nq, "StabilizerState::ColumnTableau::apply_cy");
    detail::check_distinct2(control, target, "StabilizerState::ColumnTableau::apply_cy");
    LINDBLAD_COLUMN_SWEEP2(control, target,
        w_sdg(xb, zb, r);
        w_cx(xa, za, xb, zb, r);
        w_s(xb, zb, r);
    );
}

void StabilizerState::ColumnTableau::apply_iswap(int q1, int q2) {
    detail::check_qubit(q1, nq, "StabilizerState::ColumnTableau::apply_iswap");
    detail::check_qubit(q2, nq, "StabilizerState::ColumnTableau::apply_iswap");
    detail::check_distinct2(q1, q2, "StabilizerState::ColumnTableau::apply_iswap");
    LINDBLAD_COLUMN_SWEEP2(q1, q2,
        w_cx(xa, za, xb, zb, r);
        w_s(xb, zb, r);
        w_cx(xb, zb, xa, za, r);
        w_cx(xa, za, xb, zb, r);
    );
}

void StabilizerState::ColumnTableau::apply_ecr(int q1, int q2) {
    detail::check_qubit(q1, nq, "StabilizerState::ColumnTableau::apply_ecr");
    detail::check_qubit(q2, nq, "StabilizerState::ColumnTableau::apply_ecr");
    detail::check_distinct2(q1, q2, "StabilizerState::ColumnTableau::apply_ecr");
    LINDBLAD_COLUMN_SWEEP2(q1, q2,
        w_h(xb, zb, r);
        w_s(xa, za, r);
        w_s(xb, zb, r);
        w_h(xb, zb, r);
        w_cx(xa, za, xb, zb, r);
        w_x(xa, za, r);
    );
}

#undef LINDBLAD_COLUMN_SWEEP1
#undef LINDBLAD_COLUMN_SWEEP2

// Converts to the row-major representation everything downstream reads.
// Columns are gathered 64 at a time against 64 rows, transposed as a block, and
// written out as whole words of a row. Columns past 2N exist only to make the
// gather a full block and are zero throughout, so the bits they contribute past
// the end of a row are zero as well.
StabilizerState StabilizerState::ColumnTableau::to_state() const {
    StabilizerState st(nq);
    const int col_blocks = (2 * nq + 63) / 64;

    uint64_t block[64];
    for (int rb = 0; rb < wpc; ++rb) {
        for (int cb = 0; cb < col_blocks; ++cb) {
            for (int t = 0; t < 64; ++t) {
                block[t] = col(64 * cb + t)[rb];
            }
            transpose64(block);
            for (int s = 0; s < 64; ++s) {
                const int row = 64 * rb + s;
                if (row >= rows) break;
                st.tab[static_cast<size_t>(row) * static_cast<size_t>(st.wpr) +
                       static_cast<size_t>(cb)] = block[s];
            }
        }
    }

    for (int i = 0; i < rows; ++i) {
        st.ph[static_cast<size_t>(i)] = static_cast<uint8_t>(
            (phase[static_cast<size_t>(i / 64)] >> (i % 64)) & 1ULL);
    }
    return st;
}

// =============================================================================
// CliffordSimulator
// =============================================================================

// Classifies an angle as a quarter turn: k in {0,1,2,3} for k*(π/2), or -1 when
// the angle is not a multiple of π/2 and the gate is therefore not Clifford.
// Folding a ≈ 2π to 0 catches fmod boundary values such as an input of -1e-12.
//
// is_clifford() and run() both classify through this one function, so the set
// of angles the backend ACCEPTS cannot drift from the set it can EXECUTE. A
// circuit that passes the gate is guaranteed to have a case waiting for it.
static int clifford_quarter_turn(double angle) {
    double a = std::fmod(angle, TWO_PI);
    if (a < 0) a += TWO_PI;
    if (std::abs(a - TWO_PI) < 1e-9) return 0;
    for (int k = 0; k < 4; ++k) {
        if (std::abs(a - static_cast<double>(k) * PI_2) < 1e-9) return k;
    }
    return -1;
}

bool CliffordSimulator::is_clifford(const QuantumCircuit& circuit) {
    using GT = Instruction::GateType;
    for (const auto& inst : circuit.instructions) {
        switch (inst.type) {
            case GT::H: case GT::X: case GT::Y: case GT::Z:
            case GT::S: case GT::SDG:
            case GT::CX: case GT::CZ: case GT::SWAP:
            case GT::MEASURE: case GT::RESET: case GT::BARRIER:
                break;
            // Genuine Clifford gates whose tableau action composes from the
            // primitives StabilizerState exposes. Rejecting them would push a
            // circuit onto the statevector or MPS path with no diagnostic.
            case GT::SX: case GT::SXDG:
            case GT::CY: case GT::ISWAP: case GT::ECR:
                break;
            // Rotations land in the Clifford group only at multiples of π/2.
            case GT::P: case GT::RX: case GT::RY: case GT::RZ:
                if (inst.params.empty()) return false;
                if (clifford_quarter_turn(inst.params[0]) < 0) return false;
                break;
            default:
                return false;
        }
    }
    return true;
}

// True when the pre-measurement stabilizer state is deterministic and every
// MEASURE is terminal: no feedforward, no RESET (both introduce randomness or
// state dependence), and nothing acts on a qubit after it is measured. Under
// this condition the gate pass can run ONCE and each shot samples measurements
// from a copy, instead of re-applying every gate per shot.
static bool clifford_measures_are_terminal(const QuantumCircuit& circuit) {
    std::vector<bool> measured(static_cast<size_t>(circuit.n_qubits), false);
    for (const auto& inst : circuit.instructions) {
        if (inst.type == Instruction::GateType::BARRIER) continue;
        if (inst.condition_clbit >= 0) return false;
        if (inst.type == Instruction::GateType::RESET) return false;
        for (int q : inst.qubits)
            if (q >= 0 && q < circuit.n_qubits && measured[static_cast<size_t>(q)])
                return false;
        if (inst.type == Instruction::GateType::MEASURE)
            measured[static_cast<size_t>(inst.qubits[0])] = true;
    }
    return true;
}

CliffordSimulator::Result CliffordSimulator::run(
    const QuantumCircuit& circuit_in, int shots, uint64_t seed,
    const RunPlan& plan
) {
    using GT = Instruction::GateType;
    ScopedWarningFlush flush_on_exit;
    Result result(circuit_in.n_qubits);

    // Pre-flight: reject any out-of-range operand index up front (this backend
    // surfaces errors by throwing).
    circuit_in.validate_operands();
    // Under Repair::Attempt a repaired copy is executed and the caller's
    // circuit is left exactly as it was handed over; Repair::None binds
    // straight to it and nothing is copied.
    std::optional<QuantumCircuit> repaired_storage =
        circuit_in.validated_physical();
    const QuantumCircuit& circuit =
        repaired_storage ? *repaired_storage : circuit_in;


    const int n_clbits = circuit.n_clbits > 0 ? circuit.n_clbits : circuit.n_qubits;

    std::mt19937_64 rng(seed == 0 ? std::random_device{}() : seed);

    // Angles reaching here have normally passed is_clifford(), which classifies
    // through the same function. A DIRECT run() on a circuit that never went
    // through that gate can still arrive carrying any angle, so one that is not
    // a quarter turn is rejected rather than rounded to the nearest one.
    auto quarter_turn_or_throw = [](const Instruction& in) {
        if (in.params.empty()) {
            throw std::runtime_error(
                "CliffordSimulator: " + in.gate_name() + " carries no angle parameter");
        }
        const int k = clifford_quarter_turn(in.params[0]);
        if (k < 0) {
            throw std::runtime_error(
                "CliffordSimulator: " + in.gate_name() + "(" +
                std::to_string(in.params[0]) +
                ") is not Clifford. Only multiples of π/2 are supported.");
        }
        return k;
    };

    // Apply one non-measurement Clifford gate to `state` (MEASURE/RESET/BARRIER
    // handled by the callers). Shared by both execution paths.
    // Generic over the tableau layout: the bit-sliced ColumnTableau and the
    // row-major StabilizerState expose the same gate names, and the terminal
    // path runs its gate pass on the former while the general path, which
    // interleaves measurement, stays on the latter.
    auto apply_gate = [&](auto& state, const Instruction& inst) {
        switch (inst.type) {
            case GT::H: state.apply_h(inst.qubits[0]); break;
            case GT::S: state.apply_s(inst.qubits[0]); break;
            case GT::SDG: state.apply_sdg(inst.qubits[0]); break;
            case GT::X: state.apply_x(inst.qubits[0]); break;
            case GT::Y: state.apply_y(inst.qubits[0]); break;
            case GT::Z: state.apply_z(inst.qubits[0]); break;
            case GT::SX:   state.apply_sx(inst.qubits[0]); break;
            case GT::SXDG: state.apply_sxdg(inst.qubits[0]); break;
            // P and RZ share a dispatch: rz(π/2) equals s and rz(3π/2) equals
            // sdg up to a global phase, which the stabilizer formalism does not
            // represent, so the two gates act identically on a tableau.
            case GT::P:
            case GT::RZ: {
                switch (quarter_turn_or_throw(inst)) {
                    case 0:  break;                                   // identity
                    case 1:  state.apply_s(inst.qubits[0]);   break;
                    case 2:  state.apply_z(inst.qubits[0]);   break;
                    default: state.apply_sdg(inst.qubits[0]); break;
                }
                break;
            }
            case GT::RX: {
                switch (quarter_turn_or_throw(inst)) {
                    case 0:  break;
                    case 1:  state.apply_sx(inst.qubits[0]);   break;
                    case 2:  state.apply_x(inst.qubits[0]);    break;
                    default: state.apply_sxdg(inst.qubits[0]); break;
                }
                break;
            }
            case GT::RY: {
                // ry(π/2) = h . x and ry(3π/2) = h . z, in circuit order.
                switch (quarter_turn_or_throw(inst)) {
                    case 0:  break;
                    case 1:  state.apply_h(inst.qubits[0]);
                             state.apply_x(inst.qubits[0]); break;
                    case 2:  state.apply_y(inst.qubits[0]); break;
                    default: state.apply_h(inst.qubits[0]);
                             state.apply_z(inst.qubits[0]); break;
                }
                break;
            }
            case GT::CX:    state.apply_cx(inst.qubits[0], inst.qubits[1]); break;
            case GT::CY:    state.apply_cy(inst.qubits[0], inst.qubits[1]); break;
            case GT::CZ:    state.apply_cz(inst.qubits[0], inst.qubits[1]); break;
            case GT::SWAP:  state.apply_swap(inst.qubits[0], inst.qubits[1]); break;
            case GT::ISWAP: state.apply_iswap(inst.qubits[0], inst.qubits[1]); break;
            case GT::ECR:   state.apply_ecr(inst.qubits[0], inst.qubits[1]); break;
            default:
                // Fail loud instead of silently no-op'ing. Reachable only by a
                // DIRECT run() on a non-Clifford circuit; the AUTO dispatch is
                // gated by is_clifford(), so this never fires on that path.
                throw std::invalid_argument(
                    "CliffordSimulator: gate '" + inst.gate_name() +
                    "' is not supported by the tableau backend");
        }
    };

    auto record = [&](const std::vector<int>& clreg) {
        std::string bitstring(n_clbits, '0');
        for (int c = 0; c < n_clbits; ++c)
            if (clreg[c]) bitstring[n_clbits - 1 - c] = '1';
        result.counts[bitstring]++;
    };

    // Anchors resolve against the circuit before any state is touched, so an
    // anchor that cannot fire stops the run here.
    detail::ObservationRunner runner(plan, circuit, StateForm::Stabilizer);
    runner.set_bundle(&result.observations);
    detail::ObservationRunner* watcher = runner.active() ? &runner : nullptr;
    const bool harnessed = !plan.empty();

    if (clifford_measures_are_terminal(circuit)) {
        // The gate pass is deterministic and runs ONCE. Only the sampling that
        // follows it differs from shot to shot, and how it does so is what
        // Options::sampling selects.
        //
        // Gates run bit-sliced, where each one is a few word operations over
        // the columns it touches rather than a strided visit to every row. The
        // single transpose afterwards buys back the row-major layout that
        // measurement, the outcome slab and the returned state all read.
        StabilizerState base(circuit.n_qubits);
        if (harnessed) {
            // Row-major, because the bit-sliced layout is not a state an
            // observer can be handed and a supplied initial state has no
            // column form to be seeded into. One deterministic pass serves
            // every shot, so the observers fire once, which is what describes
            // all of them.
            detail::apply_initial_state(plan, base);
            const StateView view(StateForm::Stabilizer, &base, circuit.n_qubits);
            // Named, not a temporary: begin_shot keeps a pointer to it for the
            // whole shot so observers read the register as it stands.
            const std::vector<int> no_clreg;
            runner.begin_run(circuit.n_qubits, 1);
            runner.begin_shot(0, no_clreg);
            if (watcher) watcher->at_start(view);

            int index = -1;
            for (const auto& inst : circuit.instructions) {
                ++index;
                if (watcher) watcher->before_instruction(index, inst, view);
                detail::FiringGuard fire(watcher, index, inst, view);
                if (inst.type == GT::MEASURE || inst.type == GT::BARRIER) continue;
                apply_gate(base, inst);
            }
            if (watcher) watcher->at_end(view, index);
        } else {
            StabilizerState::ColumnTableau cols(circuit.n_qubits);
            for (const auto& inst : circuit.instructions) {
                if (inst.type == GT::MEASURE || inst.type == GT::BARRIER) continue;
                apply_gate(cols, inst);
            }
            base = cols.to_state();
        }

        // shots == 0 is ONE seeded trajectory, and the only thing it produces
        // is the returned state, so the measurements have to be drawn into it.
        // Terminal Z measurements commute, so the order they are drawn in does
        // not matter. counts stays empty, which is the rest of the contract.
        if (shots == 0) {
            for (const auto& inst : circuit.instructions) {
                if (inst.type != GT::MEASURE) continue;
                (void)base.measure(inst.qubits[0], true, rng);
            }
            runner.end_run();
            result.final_state = std::move(base);
            return result;
        }

        // Auto picks the slab here, which is where it applies.
        if (options.sampling != Options::Sampling::PerShot) {
            // Where each outcome is recorded, read once. Terminal Z
            // measurements commute, so their order does not affect the
            // distribution, only the qubit-to-clbit mapping.
            std::vector<std::pair<int, int>> measured;  // (qubit, clbit)
            for (const auto& inst : circuit.instructions) {
                if (inst.type != GT::MEASURE) continue;
                const int q = inst.qubits[0];
                measured.emplace_back(q, inst.clbits.empty() ? q : inst.clbits[0]);
            }

            const int W = base.words_per_vector();

            // Extracting the slab is an elimination over the whole tableau. A
            // circuit with no measurements has nothing to draw from it, so it
            // must not pay for one: every shot there records the same empty
            // register. The zero-dimensional stand-in keeps the vector sizes
            // the sampling loop expects.
            StabilizerState::OutcomeSlab slab;
            if (measured.empty()) {
                slab.n_qubits = circuit.n_qubits;
                slab.offset.assign(static_cast<size_t>(W), 0ULL);
            } else {
                slab = base.outcome_slab(options.elimination);
            }
            std::vector<uint64_t> y(static_cast<size_t>(W), 0ULL);
            std::vector<int> clreg(static_cast<size_t>(n_clbits), 0);

            // The generator yields 64 bits at a time and they are spent one
            // free direction at a time, so a shot costs one draw per 64
            // dimensions rather than one draw per dimension.
            uint64_t pool = 0;
            int pool_left = 0;

            for (int s = 0; s < shots; ++s) {
                y = slab.offset;
                for (int d = 0; d < slab.dim; ++d) {
                    if (pool_left == 0) { pool = rng(); pool_left = 64; }
                    const bool take = (pool & 1ULL) != 0;
                    pool >>= 1;
                    --pool_left;
                    if (take) {
                        const std::vector<uint64_t>& v = slab.basis[static_cast<size_t>(d)];
                        for (int w = 0; w < W; ++w) {
                            y[static_cast<size_t>(w)] ^= v[static_cast<size_t>(w)];
                        }
                    }
                }
                std::fill(clreg.begin(), clreg.end(), 0);
                for (const auto& qc : measured) {
                    const int q = qc.first, clbit = qc.second;
                    if (clbit < 0 || clbit >= n_clbits) continue;
                    clreg[static_cast<size_t>(clbit)] = static_cast<int>(
                        (y[static_cast<size_t>(q / 64)] >> (q % 64)) & 1ULL);
                }
                record(clreg);
            }
        } else {
            for (int s = 0; s < shots; ++s) {
                StabilizerState state = base;
                std::vector<int> clreg(n_clbits, 0);
                for (const auto& inst : circuit.instructions) {
                    if (inst.type != GT::MEASURE) continue;
                    int q = inst.qubits[0];
                    int clbit = inst.clbits.empty() ? q : inst.clbits[0];
                    int outcome = state.measure(q, true, rng);
                    if (clbit >= 0 && clbit < n_clbits) clreg[clbit] = outcome;
                }
                record(clreg);
            }
        }

        runner.end_run();
        result.final_state = std::move(base);
        return result;
    }

    // General path: mid-circuit measurement / feedforward / reset need a fresh
    // trajectory per shot.

    // The slab describes a terminal measurement of a fixed state, and this
    // route has no fixed state to describe: each trajectory diverges at its
    // first collapse. A caller who asked for it anyway gets the per-shot route
    // and is told so, rather than being left to infer from a timing that the
    // request was dropped. Auto asked for nothing and is not told anything.
    if (options.sampling == Options::Sampling::Slab) {
        emit_warning(
            "note: Sampling::Slab requires terminal measurements, and this "
            "circuit has mid-circuit measurement, feedforward or reset. The "
            "outcome slab describes a terminal measurement of a fixed state, "
            "which such a circuit does not have, so the per-shot route was used "
            "instead. Counts are unaffected.");
    }

    // shots == 0 is ONE seeded trajectory whose outcome lives in the returned
    // state, so the body runs once and records nothing.
    const int trajectories = shots > 0 ? shots : 1;
    runner.begin_run(circuit.n_qubits, trajectories);
    for (int s = 0; s < trajectories; ++s) {
        StabilizerState state(circuit.n_qubits);
        detail::apply_initial_state(plan, state);
        std::vector<int> clreg(n_clbits, 0);

        const StateView view(StateForm::Stabilizer, &state, circuit.n_qubits);
        runner.begin_shot(s, clreg);
        if (watcher) watcher->at_start(view);

        int index = -1;
        for (const auto& inst : circuit.instructions) {
            ++index;
            if (watcher) watcher->before_instruction(index, inst, view);
            detail::FiringGuard fire(watcher, index, inst, view);
            if (inst.condition_clbit >= 0) {
                int cv = (inst.condition_clbit < n_clbits)
                         ? clreg[inst.condition_clbit] : 0;
                if (cv != inst.condition_value) continue;
            }
            if (inst.type == GT::MEASURE) {
                int q = inst.qubits[0];
                int clbit = inst.clbits.empty() ? q : inst.clbits[0];
                int outcome = state.measure(q, true, rng);
                if (clbit >= 0 && clbit < n_clbits) clreg[clbit] = outcome;
            } else if (inst.type == GT::RESET) {
                int q = inst.qubits[0];
                if (state.measure(q, true, rng) == 1) state.apply_x(q);
            } else if (inst.type != GT::BARRIER) {
                apply_gate(state, inst);
            }
        }

        if (watcher) watcher->at_end(view, index);

        if (shots > 0) record(clreg);
        result.final_state = std::move(state);
    }

    runner.end_run();
    return result;
}

} // namespace lindblad
