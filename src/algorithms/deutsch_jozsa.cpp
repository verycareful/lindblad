#include "lindblad/algorithms.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/qudit/qudit_backend.hpp"
#include "lindblad/qudit/qudit_density_matrix.hpp"
#include "lindblad/qudit/qudit_mps.hpp"
#include "lindblad/qudit/qudit_clifford.hpp"
#include "lindblad/qudit/qudit_noise_model.hpp"

#include <string>

namespace lindblad {
namespace algorithms {

QuantumCircuit DeutschJozsa::build_circuit(const QuantumCircuit& oracle, int n) {
    QuantumCircuit qc(n + 1, n);
    qc.x(n);
    for (int i = 0; i <= n; ++i) qc.h(i);
    for (const auto& inst : oracle.instructions) qc.instructions.push_back(inst);
    for (int i = 0; i < n; ++i) qc.h(i);
    for (int i = 0; i < n; ++i) qc.measure(i, i);
    return qc;
}

DeutschJozsa::Result DeutschJozsa::solve(const QuantumCircuit& oracle, int n,
                                          int shots, uint64_t seed) {
    auto qc = build_circuit(oracle, n);
    StatevectorSimulator sim;
    auto res = sim.run(qc, shots, seed);
    // Per-shot execution measures only the n query qubits into the classical register,
    // so bitstrings have length n. Constant oracle → all-zero query bits.
    std::string query_zero(n, '0');
    bool constant = false;
    for (const auto& [bits, cnt] : res.counts) {
        if (bits == query_zero) {
            constant = true;
            break;
        }
    }
    return { constant ? Result::CONSTANT : Result::BALANCED };
}

// =============================================================================
// QuditDeutschJozsa
// =============================================================================

QuditDeutschJozsa::Result QuditDeutschJozsa::solve(
    int n, int d,
    const std::function<int(const std::vector<int>&)>& f,
    uint64_t seed,
    QuditBackend backend,
    const QuditNoiseModel* noise)
{
    if (d < 2)
        throw std::invalid_argument("QuditDeutschJozsa::solve: d must be >= 2");
    if (n < 1)
        throw std::invalid_argument("QuditDeutschJozsa::solve: n must be >= 1");
    if (backend == QuditBackend::CLIFFORD)
        throw std::invalid_argument(
            "QuditDeutschJozsa::solve: CLIFFORD backend does not support a black-box "
            "std::function oracle (no Clifford decomposition exists). Use the "
            "QuditAffineOracle overload for a Clifford-decomposable oracle.");

    const auto Fd  = qudit_gates::qft_matrix(d);
    const auto Fdi = qudit_gates::iqft_matrix(d);
    const auto Xm  = qudit_gates::shift_matrix(d, d - 1);

    auto oracle_fn = [&](const std::vector<int>& x) -> std::vector<int> {
        const int val = f(x);
        if (val < 0 || val >= d)
            throw std::invalid_argument(
                "QuditDeutschJozsa::solve: f returned value out of [0, d)");
        return {val};
    };

    // ── DENSITY_MATRIX path ──────────────────────────────────────────────────
    if (backend == QuditBackend::DENSITY_MATRIX) {
        QuditDensityMatrix dm(n + 1, d);
        dm.apply_1qudit(n, Xm);
        dm.apply_1qudit(n, Fd);
        for (int i = 0; i < n; ++i) dm.apply_1qudit(i, Fd);
        dm.apply_function_oracle(n, 1, oracle_fn);
        for (int i = 0; i < n; ++i) dm.apply_1qudit(i, Fdi);
        if (noise) dm.apply_noise(*noise);
        const auto outcome = dm.measure(seed);
        for (int i = 0; i < n; ++i)
            if (outcome[static_cast<size_t>(i)] != 0)
                return {Verdict::BALANCED, d, n};
        return {Verdict::CONSTANT, d, n};
    }

    // ── MPS path ─────────────────────────────────────────────────────────────
    if (backend == QuditBackend::MPS) {
        QuditMPS mps(n + 1, d);
        mps.apply_1qudit(n, Xm);
        mps.apply_1qudit(n, Fd);
        for (int i = 0; i < n; ++i) mps.apply_1qudit(i, Fd);
        // MPS oracle: f takes flat query index, returns flat output index (addend)
        mps.apply_function_oracle(n, 1,
            [&](int x_flat) -> int {
                auto digits = QuditStatevector::index_to_digits(
                    static_cast<size_t>(x_flat), d, n);
                return f(digits);
            });
        for (int i = 0; i < n; ++i) mps.apply_1qudit(i, Fdi);
        const auto outcome = mps.measure(seed);
        for (int i = 0; i < n; ++i)
            if (outcome[static_cast<size_t>(i)] != 0)
                return {Verdict::BALANCED, d, n};
        return {Verdict::CONSTANT, d, n};
    }

    // ── STATEVECTOR path (default) ───────────────────────────────────────────
    QuditStatevector sv(n + 1, d);
    sv.apply_1qudit(n, Xm);
    sv.apply_1qudit(n, Fd);
    for (int i = 0; i < n; ++i) sv.apply_1qudit(i, Fd);
    sv.apply_function_oracle(n, 1, oracle_fn);
    for (int i = 0; i < n; ++i) sv.apply_1qudit(i, Fdi);
    const auto outcome = sv.measure(seed);
    for (int i = 0; i < n; ++i)
        if (outcome[static_cast<size_t>(i)] != 0)
            return {Verdict::BALANCED, d, n};
    return {Verdict::CONSTANT, d, n};
}

// =============================================================================
// QuditAffineOracle
// =============================================================================

std::vector<int> QuditAffineOracle::eval(const std::vector<int>& x, int d) const {
    const int nin  = num_inputs();
    const int nout = num_outputs();
    if (static_cast<int>(x.size()) != nin)
        throw std::invalid_argument(
            "QuditAffineOracle::eval: x size does not match num_inputs()");
    if (static_cast<int>(b.size()) != nout)
        throw std::invalid_argument(
            "QuditAffineOracle::eval: b size does not match num_outputs()");
    for (int v : x)
        if (v < 0 || v >= d)
            throw std::invalid_argument(
                "QuditAffineOracle::eval: input digit out of [0, d)");

    std::vector<int> out(static_cast<size_t>(nout), 0);
    for (int j = 0; j < nout; ++j) {
        if (static_cast<int>(A[static_cast<size_t>(j)].size()) != nin)
            throw std::invalid_argument(
                "QuditAffineOracle::eval: A row width does not match num_inputs()");
        if (b[static_cast<size_t>(j)] < 0 || b[static_cast<size_t>(j)] >= d)
            throw std::invalid_argument(
                "QuditAffineOracle::eval: constant b out of [0, d)");
        long long acc = b[static_cast<size_t>(j)];
        for (int i = 0; i < nin; ++i) {
            const int aij = A[static_cast<size_t>(j)][static_cast<size_t>(i)];
            if (aij < 0 || aij >= d)
                throw std::invalid_argument(
                    "QuditAffineOracle::eval: coefficient A out of [0, d)");
            acc += static_cast<long long>(aij) * x[static_cast<size_t>(i)];
        }
        long long r = acc % d;
        if (r < 0) r += d;
        out[static_cast<size_t>(j)] = static_cast<int>(r);
    }
    return out;
}

// =============================================================================
// QuditDeutschJozsa — structured (affine) oracle overload
//
// Affine f(x) = a·x + b (mod d) lowers to Clifford generators, so this overload
// supports the CLIFFORD backend (prime d) in addition to SV / DM / MPS. The query
// register collapses deterministically to a (the linear coefficient vector), so
// the verdict is CONSTANT iff a = 0 and BALANCED otherwise.
// =============================================================================

QuditDeutschJozsa::Result QuditDeutschJozsa::solve(
    const QuditAffineOracle& oracle, int d,
    uint64_t seed,
    QuditBackend backend,
    const QuditNoiseModel* noise)
{
    if (d < 2)
        throw std::invalid_argument("QuditDeutschJozsa::solve: d must be >= 2");
    if (oracle.num_outputs() != 1)
        throw std::invalid_argument(
            "QuditDeutschJozsa::solve: affine oracle must have exactly one output row");
    if (oracle.b.size() != 1)
        throw std::invalid_argument(
            "QuditDeutschJozsa::solve: affine oracle b must have exactly one entry");
    const int n = oracle.num_inputs();
    if (n < 1)
        throw std::invalid_argument("QuditDeutschJozsa::solve: n must be >= 1");

    const std::vector<int>& a = oracle.A[0];
    for (int v : a)
        if (v < 0 || v >= d)
            throw std::invalid_argument(
                "QuditDeutschJozsa::solve: affine coefficient out of [0, d)");
    const int b_const = oracle.b[0];
    if (b_const < 0 || b_const >= d)
        throw std::invalid_argument(
            "QuditDeutschJozsa::solve: affine constant out of [0, d)");

    // ── CLIFFORD path — direct stabilizer-tableau simulation of the affine oracle.
    if (backend == QuditBackend::CLIFFORD) {
        if (!QuditCliffordSimulator::is_prime(d))
            throw std::invalid_argument(
                "QuditDeutschJozsa::solve: CLIFFORD backend requires prime d");

        QuditCliffordSimulator c(n + 1, d);
        const int anc = n;                       // single ancilla / output qudit
        c.apply_X(anc, d - 1);                   // ancilla -> |d-1>
        c.apply_H(anc);                          // F_d  -> phase-kickback receiver
        for (int i = 0; i < n; ++i) c.apply_H(i);  // F_d on each query qudit

        // U_f: ancilla += a·x + b  (CADD(k) = CSUM applied k times; constant via X^b)
        if (b_const != 0) c.apply_X(anc, b_const);
        for (int i = 0; i < n; ++i)
            for (int rep = 0; rep < a[static_cast<size_t>(i)]; ++rep)
                c.apply_CSUM(i, anc);

        // F_d† = F_d^3 (H^4 = I in the qudit Clifford group) on each query qudit
        for (int i = 0; i < n; ++i) { c.apply_H(i); c.apply_H(i); c.apply_H(i); }

        // Measure ancilla, then decode each query qudit from an independent
        // snapshot (mirrors QuditBernsteinVazirani's Clifford measurement).
        c.measure_qudit(anc, seed);
        for (int i = 0; i < n; ++i) {
            QuditCliffordSimulator snapshot = c;
            if (snapshot.measure_qudit(i, seed + static_cast<uint64_t>(i) + 1) != 0)
                return {Verdict::BALANCED, d, n};
        }
        return {Verdict::CONSTANT, d, n};
    }

    // ── Non-Clifford backends: materialise the affine map and reuse the
    //    function-oracle overload (which handles SV / DM / MPS).
    auto f = [&oracle, d](const std::vector<int>& x) -> int {
        return oracle.eval(x, d)[0];
    };
    return solve(n, d, f, seed, backend, noise);
}

} // namespace algorithms
} // namespace lindblad
