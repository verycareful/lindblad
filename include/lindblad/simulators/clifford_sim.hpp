#pragma once

#include "lindblad/types.hpp"

#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace lindblad {

class QuantumCircuit;
class DensityMatrix;

// =============================================================================
// StabilizerState — Tableau representation for Clifford circuits
// =============================================================================
//
// Aaronson-Gottesman (2004) tableau with word-packed row storage.
// Rows 0..N-1: destabilizers; rows N..2N-1: stabilizers.
// Columns 0..N-1: X part; columns N..2N-1: Z part.
// Phase bit stored separately per row in ph[] (0=+1, 1=-1).
//
// Each row's X/Z bits are packed into uint64_t words, plus one padding word so
// a 64-bit read starting anywhere inside the data can always touch the next
// word. The Z plane begins at bit N, which is generally mid-word, so lining the
// two planes up per qubit needs a read that spans the boundary.
// rowmult is word-level throughout: the X/Z merge is an XOR, and the phase is
// two popcounts over boolean combinations of the four planes rather than a
// per-qubit branch ladder.

class StabilizerState {
public:
    int n_qubits;

    explicit StabilizerState(int n_qubits);

    // Clifford gates
    void apply_h(int qubit);
    void apply_s(int qubit);
    void apply_sdg(int qubit);
    void apply_cx(int control, int target);
    void apply_x(int qubit);
    void apply_y(int qubit);
    void apply_z(int qubit);

    // Gates whose tableau action is a composition of the primitives above.
    // Each is ONE sweep over the 2N rows: a row's bits are read into locals
    // once, the composed updates run in registers, and the row is written back
    // once. Expressing them as calls to the primitives instead would traverse
    // the tableau once per primitive, and the traversal is what the sweep
    // costs. The compositions are:
    //   sx     = h . s . h            sxdg   = h . sdg . h
    //   cz     = h(a) . cx(b,a) . h(a)
    //   swap   = cx(a,b) . cx(b,a) . cx(a,b)
    //   cy     = sdg(b) . cx(a,b) . s(b)
    //   iswap  = cx(a,b) . s(b) . cx(b,a) . cx(a,b)
    //   ecr    = h(b) . s(a) . s(b) . h(b) . cx(a,b) . x(a)
    // sx and sxdg are small enough to carry closed-form bit and sign rules
    // instead. Global phase is not represented in the stabilizer formalism, so
    // these reproduce each gate up to phase, which is exact for measurement
    // outcomes and Pauli expectations.
    void apply_sx(int qubit);
    void apply_sxdg(int qubit);
    void apply_cz(int q1, int q2);
    void apply_swap(int q1, int q2);
    void apply_cy(int control, int target);
    void apply_iswap(int q1, int q2);
    void apply_ecr(int q1, int q2);

    // Measurement — returns 0 or 1.
    // rng is only consumed when the outcome is random (stabilizer has X-support on qubit).
    int measure(int qubit, bool random, std::mt19937_64& rng);

    // The computational-basis outcome distribution of a stabilizer state is
    // UNIFORM over a coset of a linear subspace of F_2^n, never anything more
    // complicated. `offset` is one point of that coset and `basis` spans the
    // directions along it, so every outcome is `offset` xor some subset-sum of
    // `basis`, each equally likely. Vectors are packed little-endian, qubit q in
    // bit q of word q/64.
    //
    // This is what makes sampling cheap: the shape is extracted once, and a
    // shot is then a draw of `dim` bits and a subset-sum, rather than a fresh
    // measurement pass over the tableau.
    struct OutcomeSlab {
        int n_qubits = 0;
        int dim = 0;                               // free dimension, support is 2^dim
        std::vector<uint64_t> offset;              // one point of the coset
        std::vector<std::vector<uint64_t>> basis;  // dim vectors, each words_per_vector long
    };

    // Words per packed outcome vector.
    int words_per_vector() const { return (n_qubits + 63) / 64; }

    // =========================================================================
    // Elimination — how outcome_slab reduces the stabilizer generators
    // =========================================================================
    // Plain is the DEFAULT. It multiplies a row into another only where the
    // pivot bit is actually set, which happens about half the time, so it costs
    // roughly N^2/4 multiplications.
    //
    // FourRussians clears a block of k pivot columns from each remaining row
    // with ONE multiplication, against a table of the products of every subset
    // of that block's pivots. The table costs 2^k multiplications no matter how
    // many rows are below it to amortise over, so the method pays only once N
    // is well above 2^k. At k = 6 and n = 160 that ratio is about 2.5, and the
    // block route measured slower across n = 20 to 160 (0.91x to 0.98x).
    //
    // It stays selectable because the crossover is a question of size, not of
    // correctness: both produce the same affine subspace, and the block route
    // is the asymptotically better one for a tableau large enough. Selecting it
    // emits a one-time note saying what it costs at ordinary sizes.
    enum class Elimination { Plain, FourRussians };

    // Extracts the outcome distribution. Does not modify the state: the
    // generators are copied into a local, plane-aligned buffer and the
    // elimination runs there.
    OutcomeSlab outcome_slab(Elimination method = Elimination::Plain) const;

    // Expectation of Pauli string (+1, -1, or 0)
    int expectation_pauli(const std::string& pauli) const;

    // =========================================================================
    // ColumnTableau - bit-sliced companion for the gate pass
    // =========================================================================
    //
    // StabilizerState stores a row at a time, so a gate touches one bit in each
    // of 2N rows: 2N strided accesses to do work that is only two columns wide.
    // ColumnTableau holds the same tableau transposed, one machine word per 64
    // rows of a column, which turns the same gate into a few word operations
    // over the two or four columns it acts on. The phases are packed the same
    // way, so the per-row sign update is a word operation too.
    //
    // Nothing else wants this orientation. rowmult, measure and the outcome
    // slab's elimination all work ALONG rows, which here is one bit per column,
    // exactly the access pattern this layout is built to avoid. So a run applies
    // its gates here, converts once with a blocked 64x64 bit transpose, and
    // every later stage reads the row-major state it already understands.
    class ColumnTableau {
    public:
        explicit ColumnTableau(int n_qubits);

        void apply_h(int qubit);
        void apply_s(int qubit);
        void apply_sdg(int qubit);
        void apply_x(int qubit);
        void apply_y(int qubit);
        void apply_z(int qubit);
        void apply_sx(int qubit);
        void apply_sxdg(int qubit);

        void apply_cx(int control, int target);
        void apply_cy(int control, int target);
        void apply_cz(int q1, int q2);
        void apply_swap(int q1, int q2);
        void apply_iswap(int q1, int q2);
        void apply_ecr(int q1, int q2);

        // Blocked bit transpose into the row-major representation.
        StabilizerState to_state() const;

        int n_qubits() const { return nq; }

    private:
        uint64_t* col(int c) {
            return &planes[static_cast<size_t>(c) * static_cast<size_t>(wpc)];
        }
        const uint64_t* col(int c) const {
            return &planes[static_cast<size_t>(c) * static_cast<size_t>(wpc)];
        }

        int nq;    // qubits
        int rows;  // 2 * nq: destabilizers then stabilizers, as in StabilizerState
        int wpc;   // words per column: ceil(rows / 64)
        int ncol;  // columns, rounded up to a multiple of 64 so a block gather
                   // never runs past the buffer
        std::vector<uint64_t> planes;  // ncol columns, wpc words each
        std::vector<uint64_t> phase;   // wpc words, one bit per row
    };

private:
    int              wpr;      // words per row: ceil(2*n_qubits / 64) + 1 padding
    int              num_rows; // normally 2*n_qubits; +1 during scratch measurement
    std::vector<uint64_t> tab; // flat row-major X/Z bits; row i at [i*wpr, (i+1)*wpr)
    std::vector<uint8_t>  ph;  // phase bit per row (0=+1, 1=-1)

    bool get_xz(int row, int col) const;
    uint64_t bits64(int row, int bitoff) const;  // 64 bits from an arbitrary offset
    void set_xz(int row, int col, bool v);
    void flip_xz(int row, int col);
    void xor_row(int dest, int src);  // word-level XOR of X/Z bits
    void copy_row(int dest, int src);
    void zero_row(int row);
    void push_scratch();
    void pop_scratch();

    void rowmult(int dest, int src);
};

// =============================================================================
// CliffordSimulator
// =============================================================================

class CliffordSimulator {
public:
    struct Options {
        // How terminal measurements are turned into shots.
        //
        // Slab reads the outcome distribution's affine subspace off the tableau
        // once, then draws each shot as a subset-sum of its free directions.
        // PerShot replays a measurement pass over a copy of the tableau for
        // every shot.
        //
        // The two sample the SAME distribution. They consume the random stream
        // differently, so a given seed produces different individual bitstrings
        // under each; counts agree in distribution, not shot for shot. PerShot
        // is kept because it is the independent reference the faster path is
        // checked against, and because it is what a caller pinning specific
        // seeded bitstrings depends on.
        //
        // Circuits with mid-circuit measurement, feedforward or reset take the
        // general path regardless: the subspace shape describes a terminal
        // measurement of a fixed state, which is not what those circuits do.
        enum class Sampling { Slab, PerShot };

        Sampling sampling = Sampling::Slab;

        // How the slab's elimination is reduced. See StabilizerState::Elimination:
        // plain is faster at ordinary sizes, the block route is the
        // asymptotically better one and is kept for tableaux large enough to
        // repay its table.
        StabilizerState::Elimination elimination =
            StabilizerState::Elimination::Plain;
    };

    Options options;

    static bool is_clifford(const QuantumCircuit& circuit);

    struct Result {
        StabilizerState final_state;
        std::unordered_map<std::string, int> counts;

        Result(int n) : final_state(n) {}
        Result(Result&&) = default;
        Result& operator=(Result&&) = default;
    };

    Result run(const QuantumCircuit& circuit, int shots = 1024,
               uint64_t seed = 0);
};

} // namespace lindblad
