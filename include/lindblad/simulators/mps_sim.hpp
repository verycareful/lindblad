#pragma once

#include "lindblad/observation.hpp"
#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

namespace lindblad {

class Statevector;
class QuantumCircuit;

// =============================================================================
// MPSTensor — tensor for one qubit site
// =============================================================================

struct MPSTensor {
    // Shape: (bond_left, physical_dim=2, bond_right)
    int bond_left;
    int bond_right;
    // data layout: data[left * 2 * bond_right + phys * bond_right + right]
    std::vector<Complex128> data;

    MPSTensor() : bond_left(1), bond_right(1), data(2, Complex128(0.0, 0.0)) {}
    MPSTensor(int bl, int br) : bond_left(bl), bond_right(br),
        data(bl * 2 * br, Complex128(0.0, 0.0)) {}

    Complex128& operator()(int left, int phys, int right) {
        return data[left * 2 * bond_right + phys * bond_right + right];
    }
    const Complex128& operator()(int left, int phys, int right) const {
        return data[left * 2 * bond_right + phys * bond_right + right];
    }
};

// SVDMethod (BDC default, Jacobi selectable) is declared in types.hpp so the
// qubit and qudit MPS layers share one enum.

// =============================================================================
// MPSState — Matrix Product State
// =============================================================================

class MPSState {
public:
    int n_qubits;
    int max_bond_dim;    // chi — controls accuracy vs memory tradeoff
    // Truncation budget: the maximum FRACTION OF TOTAL WEIGHT (sum of sigma^2)
    // that a bond truncation may discard. Not a magnitude threshold — a bare
    // sigma is never compared against it. See svd_truncate for why: a magnitude
    // threshold asks a question whose answer depends on the scale of the input
    // and on how the target rounded, so the same state could carry a different
    // bond dimension on a different CPU.
    double cutoff;
    // SVD backend: BDC by default, faster as bond dimension grows. Jacobi is
    // selectable and emits a one-time note that it is the slower algorithm.
    // Shared enum lives in types.hpp.
    SVDMethod svd_method = SVDMethod::BDC;
    std::vector<MPSTensor> tensors;

public:
    // cutoff defaults to 1e-16: truncation error is then bounded at the order
    // of the reconstruction error an SVD already carries (~1.1e-16 relative),
    // so nothing the factorisation actually resolved is thrown away.
    MPSState(int n_qubits, int max_bond_dim = 64, double cutoff = 1e-16);

    // Gate application via SVD.
    // validation = policy and tolerance for the unitarity of U. Both matrices
    // are fixed-size, so the check is 8 or 64 complex multiplies.
    void apply_single_qubit_gate(
        const std::array<Complex128, 4>& U, int qubit,
        ValidationOptions validation = {}
    );
    void apply_two_qubit_gate(
        const std::array<Complex128, 16>& U, int q1, int q2,
        ValidationOptions validation = {}
    );

    // Truncation info
    //
    // Total weight (sum of sigma²) discarded across every split so far, each
    // term taken in the scale of the block that split. Two properties bound
    // what the total can answer.
    //
    // It ACCUMULATES, so it grows with the number of splits a run performs, and
    // two runs are comparable only when they perform the same ones.
    //
    // Each term is an ABSOLUTE weight against its two-site block rather than a
    // fraction of it. Gate application does not maintain canonical form, so
    // that block's Frobenius norm is neither the state norm nor fixed across a
    // run: it drifts as singular values are absorbed into the left tensor. A
    // single term is therefore not bounded by 1, and a total well above 1 is
    // ordinary on a deep circuit.
    //
    // The consequence is sharper than a missing unit: the totals do not ORDER
    // bond caps. Once truncation is heavy, a low cap shrinks the blocks it goes
    // on to split and its later terms are small in absolute size, while a high
    // cap keeps larger blocks and reports larger discards while losing less of
    // the state. Deep enough, and the reported total RISES with the bond cap.
    //
    // Read it as a within-run tally of what the splits dropped. To compare one
    // bond cap against another, use the bond profile or a downstream fidelity.
    double truncation_error() const { return total_truncation_error; }
    int current_max_bond_dim() const;

    // ⟨ψ|ψ⟩, by transfer-matrix contraction along the chain. There is no flat
    // amplitude array to sweep here, so this costs O(n·chi³) rather than the
    // O(2^n) a dense state would, and it is the only way to read the norm
    // without materialising the state.
    double norm_sq() const;

    // Rescale tensors[0] so ⟨ψ|ψ⟩ == 1. One site carries the whole factor,
    // which is exact: the norm is multilinear in the tensors, so scaling any
    // single one scales the state. Throws when there is no norm to divide out,
    // a zero or non-finite state, rather than returning it unchanged.
    void normalize();

    // True when the norm is 1 to within atol. A predicate: it answers, it does not
    // repair and it does not throw, and a non-finite state answers false.
    bool is_normalized(double atol = DEFAULT_PHYSICAL_ATOL) const;

    // Judge this state's normalization under a validation policy.
    // Repair::Attempt renormalizes in place; without it Warn reports and
    // leaves the state as it is, Throw raises, and Ignore measures nothing,
    // so opting out costs one branch rather than a full pass. A state with
    // nothing to divide out cannot be rescaled at all, so the response
    // decides that case too rather than the repair request forcing a throw.
    void check_normalized(ValidationOptions validation = {});

    // SVD ladder observability.
    //
    // svd_truncate runs SELECT -> VERIFY -> FALLBACK -> THROW: it distrusts the
    // SVD backend's factorisation and recomputes through the Gram route when
    // verification rejects it. Both outcomes are silent from outside, so these
    // two counters are the only way to tell a state that took the primary path
    // throughout from one that was rescued on every bond.
    //
    // svd_call_count() is the denominator: a bond split calls svd_truncate once,
    // so a bare fallback count means nothing without it. gram_fallback_count()
    // counts only the rescues that SUCCEEDED; a Gram route that also fails
    // verification throws rather than returning.
    //
    // WHICH SPLITS THESE COVER depends on the path the run took. Mid-circuit
    // measurement or feedforward at shots > 0 re-simulates per shot from a
    // fresh chain, so what a caller reads afterwards describes the LAST shot
    // and not the run. Terminal-only measurement, and shots == 0, make one
    // forward pass and the figures cover all of it. The distinction is
    // load-bearing for svd_time_ns() in particular: dividing a last-shot time
    // by a whole run's wall clock understates the share by roughly the shot
    // count.
    std::size_t gram_fallback_count() const { return gram_fallbacks; }
    std::size_t svd_call_count() const { return svd_calls; }

    // Time spent in the bond-split factorisation path, in nanoseconds,
    // accumulated over the same splits svd_call_count() counts. Divide by that
    // count for the mean cost of a split, and by a run's wall time for the
    // share of it that bond splitting accounts for.
    //
    // The interval covers the whole ladder: the factorisation, the verification
    // deciding whether to accept it, and any Gram rescue that verification
    // forced. It stops short of this layer's copy of U, S and V-dagger into its
    // own storage convention, which is marshalling no choice of SVD backend
    // changes.
    //
    // So it is an UPPER BOUND on what a faster SVD kernel could remove rather
    // than an estimate of it. Verification costs roughly one extra rank-slice
    // matrix multiply per split and survives any kernel swap, as does a rescue,
    // so a kernel halving the factorisation alone moves this figure by less
    // than half. A split that threw is not included: the ladder's last rung
    // does not return, and a run that took it has no result to profile.
    //
    // Two steady_clock reads per split are the whole measurement cost, tens of
    // nanoseconds against a factorisation of a bond-dimension-sized block. That
    // is not enough to inflate the reading meaningfully, but it is not nothing,
    // and the smallest splits are where it would show.
    std::uint64_t svd_time_ns() const { return svd_nanos; }

    // Worst factorisation error the VERIFY rung accepted, as a fraction of
    // ||M||_F^2, maximised over splits. A perfect truncated SVD satisfies the
    // Frobenius identity with equality, so this reports the excess over that
    // ideal rather than the raw residual, and a clean run sits at the square
    // of machine epsilon (~1e-32). It is the ladder's own decision variable:
    // how close a run came to being rescued, and how much error the accepted
    // route let through when it was not.
    double max_verify_residual_excess() const { return max_verify_resid_excess; }

    // Measurements and expectation values
    std::vector<double> probabilities_single(int qubit) const;

    // Sequential measurement: sample a full bitstring respecting correlations.
    // Measures qubit 0, conditions on outcome, propagates boundary, repeats.
    // O(N * chi^3) per shot. Modifies internal state (projects measured qubits).
    std::string measure_sequential(std::mt19937_64& rng);

    // Convert to exact statevector (expensive, for small N only)
    Statevector to_statevector() const;

    // The inverse of to_statevector: replace this chain with the factorisation
    // of `sv`, by sequential SVD, keeping this state's qubit count, bond cap and
    // cutoff. Every dense fallback takes this route, a gate with no compact MPS
    // form being applied to the amplitudes and the chain rebuilt from them.
    //
    // The bond cap still applies, so a state needing more bonds than it holds is
    // TRUNCATED rather than refused: that is what running at this cap means.
    //
    // The counters ACCUMULATE rather than reset. truncation_error() describes
    // everything this state has discarded, not merely what the last split
    // discarded, so a chain rebuilt part way through a run still carries what
    // the gates before it cost. Rebuilding into a fresh MPSState is how a caller
    // asks for a clean total.
    //
    // Throws when `sv` does not cover the same number of qubits as this state.
    void rebuild_from_statevector(const Statevector& sv);

private:
    double total_truncation_error = 0.0;
    std::size_t gram_fallbacks = 0;
    std::size_t svd_calls = 0;
    std::uint64_t svd_nanos = 0;
    double max_verify_resid_excess = 0.0;

    // SVD helper
    void svd_truncate(
        const std::vector<Complex128>& matrix,
        int rows, int cols,
        std::vector<Complex128>& U,
        std::vector<double>& S,
        std::vector<Complex128>& Vt,
        int& new_rank
    );

    // Adjacent two-qubit gate application (internal)
    void apply_two_qubit_gate_adjacent(
        const std::array<Complex128, 16>& U, int q1
    );

    // Adjacent SWAP gate (internal)
    void apply_swap_adjacent(int q);
};

// =============================================================================
// MPSSimulator
// =============================================================================

class MPSSimulator {
public:
    // Factorisation every bond split of a run uses, copied onto the chain this
    // simulator builds. Without it the choice is reachable only by driving
    // MPSState directly, since run() constructs its own chain and a chain
    // built inside a call cannot be configured from outside it.
    SVDMethod svd_method = SVDMethod::BDC;

    struct Result {
        MPSState final_state;
        std::unordered_map<std::string, int> counts;
        double simulation_time_seconds = 0.0;

        // Whatever the run's labelled observers collected. Empty unless the
        // RunPlan attached observers carrying labels.
        ObservationBundle observations;

        Result(int n) : final_state(n) {}
        Result(Result&&) = default;
        Result& operator=(Result&&) = default;
    };

    // plan = the harness: where the run starts and what is watched while it
    // runs. An empty plan starts at |0...0> and watches nothing.
    Result run(const QuantumCircuit& circuit, int max_bond_dim = 64,
               int shots = 1024, uint64_t seed = 0, const RunPlan& plan = {});
};

} // namespace lindblad
