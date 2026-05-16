#include "lindblad/algorithms.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/qudit/qudit_backend.hpp"
#include "lindblad/qudit/qudit_density_matrix.hpp"
#include "lindblad/qudit/qudit_mps.hpp"
#include "lindblad/qudit/qudit_clifford.hpp"
#include "lindblad/qudit/qudit_noise_model.hpp"

#include <cmath>
#include <map>
#include <random>
#include <vector>

namespace lindblad {
namespace algorithms {

QuantumCircuit Grover::build_circuit(
    const QuantumCircuit& oracle,
    int num_iterations
) {
    int nq = oracle.n_qubits;

    if (num_iterations < 0) {
        num_iterations = static_cast<int>(
            std::round(PI / 4.0 * std::sqrt(static_cast<double>(1 << nq)))
        );
        if (num_iterations < 1) num_iterations = 1;
    }

    QuantumCircuit qc(nq);

    for (int q = 0; q < nq; ++q) qc.h(q);

    for (int iter = 0; iter < num_iterations; ++iter) {
        for (const auto& inst : oracle.instructions)
            qc.instructions.push_back(inst);

        // Diffusion operator: 2|s><s| - I
        for (int q = 0; q < nq; ++q) qc.h(q);
        for (int q = 0; q < nq; ++q) qc.x(q);

        if (nq >= 2) {
            qc.h(nq - 1);
            if (nq == 2) {
                qc.cx(0, 1);
            } else if (nq == 3) {
                qc.ccx(0, 1, 2);
            } else {
                size_t mcu = 1ULL << nq;
                std::vector<Complex128> mcx_mat(mcu * mcu, Complex128(0.0, 0.0));
                for (size_t idx = 0; idx < mcu; ++idx) {
                    if (idx == mcu - 2)      mcx_mat[idx * mcu + (mcu - 1)] = Complex128(1.0, 0.0);
                    else if (idx == mcu - 1) mcx_mat[idx * mcu + (mcu - 2)] = Complex128(1.0, 0.0);
                    else                     mcx_mat[idx * mcu + idx]        = Complex128(1.0, 0.0);
                }
                Instruction mcx_inst;
                mcx_inst.type = Instruction::GateType::UNITARY;
                for (int mq = 0; mq < nq; ++mq) mcx_inst.qubits.push_back(mq);
                mcx_inst.matrix = mcx_mat;
                qc.instructions.push_back(mcx_inst);
            }
            qc.h(nq - 1);
        }

        for (int q = 0; q < nq; ++q) qc.x(q);
        for (int q = 0; q < nq; ++q) qc.h(q);
    }

    return qc;
}

Grover::Result Grover::search(
    const QuantumCircuit& oracle,
    int num_iterations,
    int shots,
    uint64_t seed
) {
    // Resolve auto-iteration count before calling build_circuit so that
    // result.num_iterations reflects what the circuit was actually built with.
    // Note: the default formula assumes a single marked item. For k solutions
    // the optimal count is round(π/4 · √(2^n / k)); pass num_iterations
    // explicitly when k > 1.
    int nq = oracle.n_qubits;
    if (num_iterations < 0) {
        num_iterations = static_cast<int>(
            std::round(PI / 4.0 * std::sqrt(static_cast<double>(1 << nq)))
        );
        if (num_iterations < 1) num_iterations = 1;
    }

    auto circuit = build_circuit(oracle, num_iterations);
    circuit.measure_all();

    StatevectorSimulator sim;
    auto sim_result = sim.run(circuit, shots, seed);

    Result result;
    int max_count = 0;
    for (const auto& [bits, count] : sim_result.counts) {
        if (count > max_count) {
            max_count = count;
            result.solution = bits;
        }
    }
    result.probability = static_cast<double>(max_count) / shots;
    result.num_iterations = num_iterations;

    return result;
}

// =============================================================================
// QuditGrover
// =============================================================================

namespace {
// Exact optimal Grover iteration count: minimises |(2R+1)θ - π/2| where sin(θ)=1/√N.
// Approximation round(π/4·√N) is inaccurate for small N (e.g. N=4→gives 2, optimal=1).
int qudit_grover_auto_iters(int n, int d) {
    const double N = std::pow(static_cast<double>(d), static_cast<double>(n));
    const double theta = std::asin(1.0 / std::sqrt(N));
    return std::max(1, static_cast<int>(std::round(PI / (4.0 * theta) - 0.5)));
}
} // anonymous namespace

QuditGrover::Result QuditGrover::search(
    int n, int d,
    const std::vector<int>& target,
    int num_iterations, int shots, uint64_t seed,
    QuditBackend backend, const QuditNoiseModel* noise)
{
    if (d < 2)
        throw std::invalid_argument("QuditGrover::search: d must be >= 2");
    if (n < 1)
        throw std::invalid_argument("QuditGrover::search: n must be >= 1");
    if (static_cast<int>(target.size()) != n)
        throw std::invalid_argument(
            "QuditGrover::search: target size must equal n");
    for (int v : target)
        if (v < 0 || v >= d)
            throw std::invalid_argument(
                "QuditGrover::search: target digit out of [0, d)");

    return search_with_oracle(n, d,
        [&](const std::vector<int>& x) -> bool { return x == target; },
        num_iterations, shots, seed, backend, noise);
}

QuditGrover::Result QuditGrover::search_with_oracle(
    int n, int d,
    const std::function<bool(const std::vector<int>&)>& is_marked,
    int num_iterations, int shots, uint64_t seed,
    QuditBackend backend, const QuditNoiseModel* noise)
{
    if (d < 2)
        throw std::invalid_argument(
            "QuditGrover::search_with_oracle: d must be >= 2");
    if (n < 1)
        throw std::invalid_argument(
            "QuditGrover::search_with_oracle: n must be >= 1");
    if (shots < 1) shots = 1;

    if (backend == QuditBackend::CLIFFORD)
        throw std::invalid_argument(
            "QuditGrover: CLIFFORD backend is not supported "
            "(Grover's phase oracle applies -1 to marked states, which is non-Clifford)");

    if (num_iterations < 0) num_iterations = qudit_grover_auto_iters(n, d);

    const auto oracle_phase = [&](const std::vector<int>& digits) -> Complex128 {
        return is_marked(digits) ? Complex128(-1.0, 0.0) : Complex128(1.0, 0.0);
    };
    const auto diffusion_phase = [](const std::vector<int>& digits) -> Complex128 {
        for (int x : digits)
            if (x != 0) return Complex128(-1.0, 0.0);
        return Complex128(1.0, 0.0);
    };
    const auto F  = qudit_gates::qft_matrix(d);
    const auto Fd = qudit_gates::iqft_matrix(d);

    std::mt19937_64 rng(seed == 0
        ? static_cast<uint64_t>(std::random_device{}())
        : seed);
    std::map<std::vector<int>, int> counts;

    // ── DENSITY_MATRIX path ──────────────────────────────────────────────────
    if (backend == QuditBackend::DENSITY_MATRIX) {
        for (int shot = 0; shot < shots; ++shot) {
            QuditDensityMatrix dm(n, d);
            for (int q = 0; q < n; ++q) dm.apply_1qudit(q, F);
            for (int iter = 0; iter < num_iterations; ++iter) {
                dm.apply_phase_oracle(oracle_phase);
                for (int q = 0; q < n; ++q) dm.apply_1qudit(q, Fd);
                dm.apply_phase_oracle(diffusion_phase);
                for (int q = 0; q < n; ++q) dm.apply_1qudit(q, F);
            }
            if (noise) dm.apply_noise(*noise);
            counts[dm.measure(rng())]++;
        }
    }
    // ── MPS path ─────────────────────────────────────────────────────────────
    else if (backend == QuditBackend::MPS) {
        for (int shot = 0; shot < shots; ++shot) {
            QuditMPS mps(n, d);
            for (int q = 0; q < n; ++q) mps.apply_1qudit(q, F);
            for (int iter = 0; iter < num_iterations; ++iter) {
                mps.apply_phase_oracle(oracle_phase);
                for (int q = 0; q < n; ++q) mps.apply_1qudit(q, Fd);
                mps.apply_phase_oracle(diffusion_phase);
                for (int q = 0; q < n; ++q) mps.apply_1qudit(q, F);
            }
            counts[mps.measure(rng())]++;
        }
    }
    // ── STATEVECTOR path (default) ───────────────────────────────────────────
    else {
        for (int shot = 0; shot < shots; ++shot) {
            QuditStatevector sv(n, d);
            for (int q = 0; q < n; ++q) sv.apply_1qudit(q, F);
            for (int iter = 0; iter < num_iterations; ++iter) {
                sv.apply_phase_oracle(oracle_phase);
                for (int q = 0; q < n; ++q) sv.apply_1qudit(q, Fd);
                sv.apply_phase_oracle(diffusion_phase);
                for (int q = 0; q < n; ++q) sv.apply_1qudit(q, F);
            }
            counts[sv.measure(rng())]++;
        }
    }

    Result result;
    result.num_iterations = num_iterations;
    result.d = d;
    result.n = n;
    int max_count = 0;
    for (const auto& [outcome, count] : counts) {
        if (count > max_count) {
            max_count = count;
            result.solution = outcome;
        }
    }
    result.probability = static_cast<double>(max_count) / static_cast<double>(shots);
    return result;
}

} // namespace algorithms
} // namespace lindblad
