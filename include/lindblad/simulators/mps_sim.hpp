// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#pragma once

#include "lindblad/observation.hpp"
#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"
#include "lindblad/detail/fidelity_ledger.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lindblad {

class Statevector;
class QuantumCircuit;

namespace detail {
struct SvdTruncation;
struct MPSDispatch;
struct StateFileAccess;
class RunBudget;

// =============================================================================
// BudgetLink - a chain's tie to the run evolving it
// =============================================================================
// Which run budget this chain's growth is charged against, and where that run
// has got to, while an MPSSimulator run evolves it. The link belongs to the
// object, not to its value: copying or moving a chain never carries it, and
// assigning to one keeps the target's own. So a chain an observer copied, or
// the one a run hands back, never refers to a budget that has gone.
class BudgetLink {
public:
    BudgetLink() = default;
    BudgetLink(const BudgetLink&) noexcept {}
    BudgetLink& operator=(const BudgetLink&) noexcept { return *this; }

    void attach(RunBudget* run_budget) noexcept {
        budget = run_budget;
        shot = -1;
        instruction = -1;
        inst = nullptr;
    }
    void detach() noexcept { attach(nullptr); }

    RunBudget* budget = nullptr;
    int shot = -1;                       // -1 outside a per-shot trajectory
    int instruction = -1;                // -1 before the first instruction
    const Instruction* inst = nullptr;   // the instruction being applied
};
}  // namespace detail

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

// SVDMethod (BDC default, Jacobi selectable), CanonicalForm, UncheckedGates
// and MPS_DEFAULT_CUTOFF are declared in types.hpp so the qubit and qudit MPS
// layers share them.

// =============================================================================
// MPSState — Matrix Product State
// =============================================================================
//
// Canonical form. The chain keeps an OPEN SPAN [lo, hi] of sites, reported by
// open_span(): every site left of lo is left-orthonormal (reshaped as a
// (2 chi_L) x chi_R matrix, A†A = I), every site right of hi is
// right-orthonormal (reshaped as chi_L x (2 chi_R), AA† = I), and the sites
// inside the span carry no guarantee. lo == hi == c is mixed canonical form
// centred on c: the state's whole norm sits in site c, and a two-site block at
// c holds the state's Schmidt coefficients at its bond.
//
// Every operation maintains the span, which is why the tensors are private.
// Reading them is free (tensors()); replacing them goes through set_tensors(),
// which assumes nothing about a chain built by hand. The centre moves by QR
// (rightward) and LQ (leftward) steps, which are exact and never truncate, so
// moving it changes the gauge and never the state. Whether a bond split moves
// it onto its block first is canonical_form's to decide (CanonicalForm in
// types.hpp); measurement, reset and sampling always do.

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
    // Factorisation every bond split asks for first. BDC (autonne's divide and
    // conquer) by default; the alternatives and what each promises are with the
    // enum in types.hpp.
    SVDMethod svd_method = SVDMethod::BDC;
    // Whether a factorisation the VERIFY rung rejects may descend the rescue
    // ladder (autonne Jacobi, then the Gram route), each descent reported
    // through the warning channel. false: the first rejection throws, for a
    // caller who would rather stop than accept a tensor from a kernel they did
    // not name.
    bool svd_rescue = true;
    // The widest register this chain may expand into a dense array: a gate
    // over three or more qubits, MCX, MCP and PERMUTATION (applied to the
    // amplitudes and the chain rebuilt), and to_statevector. 25 qubits under
    // Enforce, 31 under Lift (see QubitLimit in types.hpp).
    QubitLimit qubit_limit = QubitLimit::Enforce;
    // Set only while an MPSSimulator run evolves this chain; see BudgetLink.
    // Every two-site update and dense fallback is checked against the run's
    // memory budget before it allocates.
    detail::BudgetLink budget_link;
    // Which bond splits first move the orthogonality centre onto their block.
    // Always by default; the two policies and what each costs are with the
    // enum in types.hpp.
    CanonicalForm canonical_form = CanonicalForm::Always;
    // Whether a gate no policy measured (Ignore with Repair::None) is measured
    // for this chain's own records anyway. Track by default; the two settings
    // and what each keeps are with the enum in types.hpp.
    UncheckedGates unchecked_gates = UncheckedGates::Track;

public:
    // cutoff defaults to MPS_DEFAULT_CUTOFF (types.hpp), at which a split
    // removes rank deficiency and rounding-level weight but does not compress.
    // The new chain is |0...0>, bond dimension 1 throughout, with its centre on
    // site 0.
    MPSState(int n_qubits, int max_bond_dim = 64,
             double cutoff = MPS_DEFAULT_CUTOFF);

    // The site tensors, left to right, one per qubit. Read-only, so reading
    // them cannot disturb the open span the chain's operations rely on.
    const std::vector<MPSTensor>& tensors() const noexcept { return tensors_; }

    // Replace the chain with `sites`, one tensor per qubit, qubit 0 first.
    //
    // Validated before anything is replaced: the count must equal n_qubits,
    // every bond must be at least 1 with the two outer ones exactly 1,
    // neighbouring bonds must agree, each data array must hold
    // bond_left * 2 * bond_right entries, and every entry must be finite.
    // Throws std::invalid_argument naming the first violation, leaving the
    // state as it was.
    //
    // Nothing is assumed about a chain built by hand, so the open span becomes
    // the whole chain, and the first operation that needs the centre pays the
    // QR steps to find it. The fidelity figures reset to exact: the chain
    // handed in is what later truncation is measured against. Assigning a whole
    // MPSState instead keeps them. The profile counters and truncation_error()
    // are untouched, because they describe splits this object performed.
    void set_tensors(std::vector<MPSTensor> sites);

    // The open span {lo, hi} described above the class. {c, c} is mixed
    // canonical form centred on c. A chain with no qubits reports {0, -1}.
    std::pair<int, int> open_span() const noexcept { return {span_lo, span_hi}; }

    // Move the orthogonality centre to `site`, leaving open_span() equal to
    // {site, site}. A gauge change made of QR and LQ steps: the state, the
    // profile counters and the fidelity figures are unchanged. A bond wider
    // than the rank its neighbouring site can carry is trimmed to that rank
    // on the way, which is lossless.
    void canonicalize(int site);

    // Gate application via SVD.
    // validation = policy and tolerance for the unitarity of U. Both matrices
    // are fixed-size, so the check is 8 or 64 complex multiplies.
    //
    // apply_two_qubit_gate takes U in the project matrix convention
    // (docs/Architecture.md "Conventions", as gates::apply_unitary reads it):
    // bit 0 of the row and column index is the state of q1 and bit 1 the state
    // of q2, so q1 is the least significant. A CX with its control on q1 is
    // therefore nonzero at (0,0), (1,3), (2,2) and (3,1).
    //
    // A single-qubit gate leaves open_span() as it is when U is unitary: a
    // unitary acting on the physical index preserves both orthonormalities. A
    // two-qubit gate on an adjacent pair contracts the pair, applies U and
    // splits the block by truncated SVD, moving the centre onto the pair first
    // when canonical_form calls for it. A non-adjacent pair is brought together
    // by a SWAP chain whose splits follow the same rule, each leaving the
    // centre where the next one starts.
    //
    // Either gate applies U exactly as given, whatever its unitarity. When U
    // is not unitary to DEFAULT_PHYSICAL_ATOL (known from the policy's own
    // measurement, or from one taken for the chain's records as
    // unchecked_gates says), a single-qubit gate on a site outside the open
    // span widens the span over that site, and either gate empties the
    // fidelity figures. Nothing is reported.
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
    // term the absolute weight its split threw away.
    //
    // It ACCUMULATES, so it grows with the number of splits a run performs, and
    // two runs are comparable only when they perform the same ones.
    //
    // A split in canonical gauge discards that weight from the STATE, whose
    // norm its block then carries. On a chain with no collapse, no
    // normalisation and no absorbed profile, the total therefore equals how far
    // norm_sq() has fallen since the chain was built, to rounding. Under
    // CanonicalForm::Auto the splits that run in place contribute weight at
    // their blocks' rounding level instead.
    //
    // It is a weight, not a fidelity. For how close the chain is to the state
    // an untruncated run would hold, read fidelity_estimate() and
    // fidelity_lower_bound().
    double truncation_error() const { return total_truncation_error; }
    int current_max_bond_dim() const;

    // How close the chain is to the state an untruncated run would hold, as a
    // fidelity |⟨exact|chain⟩|² between the two normalised states.
    //
    // fidelity_estimate() is prod_k (1 - eps_k) over each split's discarded
    // fraction, the standard figure, and it is NOT a bound: the true fidelity
    // can lie on either side of it. fidelity_lower_bound() is a floor the true
    // fidelity cannot fall below. The derivation, the example separating the
    // two, and what CanonicalForm::Auto costs them are in
    // detail/fidelity_ledger.hpp.
    //
    // Both read 1 on a new chain and after set_tensors(). Both are EMPTY once a
    // measurement or reset has collapsed the chain (measure_qubit,
    // measure_sequential, or a MEASURE or RESET in a run), or once a gate that
    // is not unitary has been applied, and stay empty: projection renormalises
    // the exact and the truncated state by different factors, and the bound is
    // derived for unitary gates between splits. rebuild_from_statevector's splits count like any other.
    // absorb_profile does not fold them, because they describe these tensors'
    // own history rather than a run's splits.
    std::optional<double> fidelity_estimate() const noexcept {
        return fidelity.estimate();
    }
    std::optional<double> fidelity_lower_bound() const noexcept {
        return fidelity.lower_bound();
    }

    // ⟨ψ|ψ⟩. Sites outside the open span are orthonormal and contract to the
    // identity, so only the span is contracted: at a single-site centre this
    // is that site's Frobenius norm, O(chi²), and with the span open over the
    // whole chain O(n·chi³). There is no flat amplitude array to sweep here,
    // and this is the only way to read the norm without materialising the
    // state.
    double norm_sq() const;

    // Rescale one site so ⟨ψ|ψ⟩ == 1: the first site of the open span, since
    // scaling a site outside it would break that site's orthonormality. One
    // site carrying the whole factor is exact: the norm is multilinear in the
    // tensors, so scaling any single one scales the state. Throws when there
    // is no norm to divide out, a zero or non-finite state, rather than
    // returning it unchanged.
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
    // svd_truncate runs the verified ladder: it distrusts the
    // selected kernel's factorisation and, when verification rejects it,
    // descends to autonne's Jacobi and then to the Gram route. A rescued split
    // is warned about as it happens and is otherwise indistinguishable from a
    // clean one, so these counters are the record of how often it happened.
    //
    // svd_call_count() is the denominator: a bond split calls svd_truncate once,
    // so a bare rescue count means nothing without it. jacobi_rescue_count()
    // and gram_fallback_count() count only the rescues that SUCCEEDED, one or
    // the other per rescued split; a split on which every rung fails throws
    // rather than returning.
    //
    // floor_rejected_weight() is the Gram route's own cost: the sigma weight
    // below its validity floor on every split it rescued (see
    // SvdTruncation::floor_rejected_weight). It is NOT truncation and is kept
    // out of truncation_error(), which would otherwise report a bond that
    // discarded nothing as having lost something. Zero unless some split took
    // the Gram rung.
    //
    // On a chain MPSSimulator::run returns, every figure here covers every
    // split the run performed on every path, the per-shot trajectories
    // included: a run that re-simulates per shot returns the last trajectory's
    // tensors carrying the totals of all of them, and of the shared start an
    // unobserved run of several shots computes once (see absorb_profile).
    std::size_t jacobi_rescue_count() const { return jacobi_rescues; }
    std::size_t gram_fallback_count() const { return gram_fallbacks; }
    double floor_rejected_weight() const { return floor_rejected; }
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

    // Fold another chain's five profile figures into this one: the four tallies
    // add, the worst accepted residual takes the larger of the two. The tensors
    // are untouched, so this is how a chain comes to report splits it did not
    // itself perform. MPSSimulator::run uses it on the per-shot path, where
    // every trajectory runs on a chain of its own and the returned one carries
    // their sum, plus the shared start's once when the run computed one.
    // truncation_error() after absorbing is everything the run
    // discarded, consistent with the accumulate-rather-than-reset contract
    // above, and is not the returned tensors' own history.
    void absorb_profile(const MPSState& other);

    // Measurements and expectation values
    //
    // The marginals ⟨ψ|P_k|ψ⟩ of `qubit` for k = 0, 1, RAW: they sum to
    // norm_sq(), which is 1 only on a normalised state. Reads the chain without
    // moving the centre, contracting only the stretch of the open span on each
    // side of the qubit, so with the centre on the qubit it is a read of that
    // one site.
    std::vector<double> probabilities_single(int qubit) const;

    // Measure `qubit` in the computational basis and collapse the chain onto
    // the outcome, returned as 0 or 1. The centre moves to the qubit first, so
    // its marginals are local and the collapse renormalises that site alone;
    // afterwards the state has unit norm and open_span() is {qubit, qubit}.
    // Draws one uniform from `rng`. The fidelity figures become empty.
    //
    // The outcome is drawn from the qubit's marginals divided by their sum, the
    // chain's own normalised distribution. A chain with no norm, zero or
    // non-finite, is refused with std::runtime_error before the draw, so `rng`
    // is left as it was.
    int measure_qubit(int qubit, std::mt19937_64& rng);

    // Sequential measurement: sample a full bitstring respecting correlations,
    // and collapse the chain onto it. The centre moves to qubit 0, then each
    // qubit in turn is measured at the centre, collapsed, and the centre
    // stepped right by one QR, O(N * chi^3) per call with no environments.
    // Qubit 0 is the rightmost character. Draws one uniform per qubit. The
    // fidelity figures become empty. A chain with no norm is refused, as by
    // measure_qubit, before the first draw.
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
    // the gates before it cost. The fidelity figures accumulate the same way,
    // which is right when `sv` is this chain's own state with a gate applied,
    // as on every dense fallback. Rebuilding into a fresh MPSState is how a
    // caller asks for a clean total and exact figures.
    //
    // The sweep leaves sites 0..n-2 left-orthonormal, so the centre ends on
    // the last site, and each of its splits is taken in canonical gauge.
    //
    // Throws when `sv` does not cover the same number of qubits as this state.
    void rebuild_from_statevector(const Statevector& sv);

private:
    // run()'s instruction dispatcher applies gates it built itself, unitary by
    // construction, through gate_one_site and gate_two_site without measuring
    // them, and a circuit's own matrices under that instruction's policy.
    friend struct detail::MPSDispatch;
    // The failed-run state file writes and rebuilds the chain's storage.
    friend struct detail::StateFileAccess;

    std::vector<MPSTensor> tensors_;
    // The open span; see the class comment. {0, -1} when there are no sites.
    int span_lo = 0;
    int span_hi = 0;
    detail::FidelityLedger fidelity;

    double total_truncation_error = 0.0;
    std::size_t jacobi_rescues = 0;
    std::size_t gram_fallbacks = 0;
    double floor_rejected = 0.0;
    std::size_t svd_calls = 0;
    std::uint64_t svd_nanos = 0;
    double max_verify_resid_excess = 0.0;

    // Which site of a split receives the singular values. Right leaves the
    // left site left-orthonormal and the centre on the right site; Left is the
    // mirror image. A SWAP chain passes the direction its next block lies in,
    // so each split leaves the centre where the next one needs it.
    enum class Absorb { Left, Right };

    // Folds one split's outcome into the profile counters. Every split this
    // class performs reports through here, so no route can count differently.
    void account_split(const detail::SvdTruncation& split, std::uint64_t nanos);

    // SVD helper
    void svd_truncate(
        const std::vector<Complex128>& matrix,
        int rows, int cols,
        std::vector<Complex128>& U,
        std::vector<double>& S,
        std::vector<Complex128>& Vt,
        int& new_rank
    );

    // Whether the split of the block at (q, q+1) moves the centre onto the
    // block first: the CanonicalForm rule, read against the current bonds.
    bool split_needs_centre(int q) const;

    // One step of the centre. shift_right(q): thin QR of site q as a
    // (2 chi_L) x chi_R matrix, site q becomes Q and R multiplies into site
    // q+1. shift_left(q): thin LQ of site q as chi_L x (2 chi_R), site q
    // becomes Q and L multiplies into site q-1. Exact, never truncating, and
    // outside every counter.
    void shift_right(int q);
    void shift_left(int q);

    // Narrow the open span into [a, b] by the fewest steps: raise lo to a,
    // lower hi to b. When one end passes the other, the site that received the
    // last factor is the whole span.
    void focus(int a, int b);

    // Local marginals of the centre site, which must be `site`: the squared
    // norms of its two physical slices.
    std::array<double, 2> centre_marginals(int site) const;

    // Collapse the centre site onto `outcome`, renormalising by the outcome's
    // raw marginal `p_outcome` so the state leaves with unit norm.
    void collapse_centre(int site, int outcome, double p_outcome);

    // The gates after validation, given whether U is unitary to
    // DEFAULT_PHYSICAL_ATOL. gate_one_site applies U to the site and, when it
    // is not unitary, widens the open span over a site outside it and empties
    // the fidelity figures. gate_two_site takes U LSB-first, as the public
    // entry does, and empties the figures when it is not unitary; its splits
    // leave the gauge sound whatever U was.
    void gate_one_site(const std::array<Complex128, 4>& U, int qubit,
                       bool unitary);
    void gate_two_site(const std::array<Complex128, 16>& U, int q1, int q2,
                       bool unitary);

    // apply_two_qubit_gate with U already in the MSB-first order the two-site
    // contraction reads (bit 1 = q1, bit 0 = q2): orders the pair, runs the
    // SWAP chain, applies. The public entry converts to this order once.
    void apply_two_qubit_gate_msb(
        const std::array<Complex128, 16>& U, int q1, int q2
    );

    // Adjacent two-qubit gate application (internal, MSB-first as above)
    void apply_two_qubit_gate_adjacent(
        const std::array<Complex128, 16>& U, int q1,
        Absorb absorb = Absorb::Right
    );

    // Adjacent SWAP gate (internal)
    void apply_swap_adjacent(int q, Absorb absorb);
};

// =============================================================================
// MPSSimulator
// =============================================================================

class MPSSimulator {
public:
    // Factorisation every bond split of a run uses, whether a rejected one may
    // be rescued, and which splits first move the orthogonality centre onto
    // their block, all copied onto every chain this simulator builds. Without
    // them the choices are reachable only by driving MPSState directly, since
    // run() constructs its own chain and a chain built inside a call cannot be
    // configured from outside it. Meaning of each: MPSState::svd_method,
    // MPSState::svd_rescue, MPSState::canonical_form,
    // MPSState::unchecked_gates and MPSState::qubit_limit.
    SVDMethod svd_method = SVDMethod::BDC;
    bool svd_rescue = true;
    CanonicalForm canonical_form = CanonicalForm::Always;
    UncheckedGates unchecked_gates = UncheckedGates::Track;
    QubitLimit qubit_limit = QubitLimit::Enforce;
    // The most memory a run may use, in MiB (2^20 bytes). 0, the default, is
    // automatic: the memory this machine reports available, or
    // FALLBACK_MEMORY_CAP_MB when it gives no coherent reading. NO_MEMORY_CAP
    // means no cap. An MPS run has no fixed footprint to refuse up front; the
    // cap is the budget every two-site update, dense fallback and observer
    // copy is checked against before it allocates.
    uint64_t max_memory_mb = 0;

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
