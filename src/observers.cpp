#include "lindblad/observers.hpp"

#include "lindblad/statevector.hpp"
#include "lindblad/detail/eigen_backend.hpp"
#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <optional>
#include <stdexcept>

namespace lindblad {

namespace {

// Reading past what an observer captured is a caller-side mistake with a
// specific fix (fire fewer times, or index within count()), so it says both.
void check_index(std::size_t k, std::size_t count, const char* who) {
    if (k >= count) {
        throw std::invalid_argument(
            std::string(who) + ": firing " + std::to_string(k) +
            " was not observed; this observer recorded " +
            std::to_string(count) + " of them");
    }
}

// The dense amplitudes of whatever the view holds, subject to the knobs.
// Returns null when the observation is to be omitted.
std::shared_ptr<const Statevector> dense_state(const ObservationContext& ctx,
                                               const std::string& what) {
    auto produced = detail::produce_state(ctx.state, StateForm::Statevector,
                                          ctx.plan.options, what);
    return std::static_pointer_cast<const Statevector>(produced);
}

}  // namespace

// =============================================================================
// BundleWriter
// =============================================================================

void BundleWriter::record(const ObservationContext& ctx, ObservationBundle::Payload payload) {
    points_.push_back(FiringPoint{ctx.shot, ctx.instruction_index, ctx.anchor});

    if (!labelled() || ctx.bundle == nullptr) return;
    bundle_ = ctx.bundle;
    entries_.push_back(Entry{ctx.instruction_index, ctx.shot, std::move(payload)});
}

const FiringPoint& BundleWriter::point(std::size_t k) const {
    if (k >= points_.size()) {
        throw std::invalid_argument(
            "Observer: firing " + std::to_string(k) + " was not observed; this "
            "observer recorded " + std::to_string(points_.size()) + " of them");
    }
    return points_[k];
}

void BundleWriter::end_run() {
    if (bundle_ == nullptr || entries_.empty()) return;

    if (entries_.size() == 1) {
        bundle_->put(label_, std::move(entries_.front().payload));
    } else {
        for (auto& entry : entries_) {
            bundle_->put(label_ + "@" + std::to_string(entry.index) + "#" +
                             std::to_string(entry.shot),
                         std::move(entry.payload));
        }
    }
    entries_.clear();
}

// =============================================================================
// StateObserver
// =============================================================================

void StateObserver::observe(const ObservationContext& ctx) {
    const StateForm wanted = native_ ? ctx.state.form() : form_;

    auto produced = detail::produce_state(ctx.state, wanted, ctx.plan.options,
                                          "StateObserver");
    if (!produced) return;

    states_.emplace_back(wanted, produced);
    record(ctx, ObservationBundle::StatePayload{wanted, produced});
}

const std::shared_ptr<const void>& StateObserver::at(std::size_t k, StateForm wanted) const {
    check_index(k, states_.size(), "StateObserver");
    if (states_[k].first != wanted) {
        throw std::invalid_argument(
            std::string("StateObserver: firing ") + std::to_string(k) +
            " captured a " + to_string(states_[k].first) + ", not a " +
            to_string(wanted));
    }
    return states_[k].second;
}

StateForm StateObserver::form(std::size_t k) const {
    check_index(k, states_.size(), "StateObserver");
    return states_[k].first;
}

const Statevector& StateObserver::statevector(std::size_t k) const {
    return *static_cast<const Statevector*>(at(k, StateForm::Statevector).get());
}

const DensityMatrix& StateObserver::density_matrix(std::size_t k) const {
    return *static_cast<const DensityMatrix*>(at(k, StateForm::DensityMatrix).get());
}

const StabilizerState& StateObserver::stabilizer(std::size_t k) const {
    return *static_cast<const StabilizerState*>(at(k, StateForm::Stabilizer).get());
}

const MPSState& StateObserver::mps(std::size_t k) const {
    return *static_cast<const MPSState*>(at(k, StateForm::MPS).get());
}

// =============================================================================
// ProbabilityObserver
// =============================================================================

void ProbabilityObserver::observe(const ObservationContext& ctx) {
    std::vector<double> probabilities;

    switch (ctx.state.form()) {
        case StateForm::Statevector:
            probabilities = ctx.state.statevector().probabilities();
            break;
        case StateForm::DensityMatrix:
            probabilities = ctx.state.density_matrix().probabilities();
            break;
        case StateForm::Stabilizer:
        case StateForm::MPS: {
            // Neither holds a probability vector, so this is a conversion and
            // is charged as one.
            auto dense = dense_state(ctx, "ProbabilityObserver");
            if (!dense) return;
            probabilities = dense->probabilities();
            break;
        }
    }

    values_.push_back(probabilities);
    record(ctx, std::move(probabilities));
}

const std::vector<double>& ProbabilityObserver::probabilities(std::size_t k) const {
    check_index(k, values_.size(), "ProbabilityObserver");
    return values_[k];
}

std::vector<double> ProbabilityObserver::average() const {
    if (values_.empty()) return {};

    std::vector<double> mean(values_.front().size(), 0.0);
    for (const auto& row : values_) {
        if (row.size() != mean.size()) {
            throw std::invalid_argument(
                "ProbabilityObserver::average: firings have different widths, "
                "so they describe different registers and cannot be averaged");
        }
        for (std::size_t i = 0; i < row.size(); ++i) mean[i] += row[i];
    }
    const double inv = 1.0 / static_cast<double>(values_.size());
    for (double& value : mean) value *= inv;
    return mean;
}

// =============================================================================
// AmplitudeObserver
// =============================================================================

void AmplitudeObserver::observe(const ObservationContext& ctx) {
    std::vector<Complex128> picked;
    picked.reserve(indices_.size());

    if (ctx.state.form() == StateForm::Statevector) {
        const Statevector& sv = ctx.state.statevector();
        for (const std::size_t index : indices_) {
            if (index >= sv.dimension()) {
                throw std::invalid_argument(
                    "AmplitudeObserver: index " + std::to_string(index) +
                    " is outside a " + std::to_string(sv.dimension()) +
                    " amplitude state");
            }
            picked.push_back(sv.amplitude(index));
        }
    } else {
        auto dense = dense_state(ctx, "AmplitudeObserver");
        if (!dense) return;
        for (const std::size_t index : indices_) {
            if (index >= dense->dimension()) {
                throw std::invalid_argument(
                    "AmplitudeObserver: index " + std::to_string(index) +
                    " is outside a " + std::to_string(dense->dimension()) +
                    " amplitude state");
            }
            picked.push_back(dense->amplitude(index));
        }
    }

    values_.push_back(picked);
    record(ctx, std::move(picked));
}

const std::vector<Complex128>& AmplitudeObserver::amplitudes(std::size_t k) const {
    check_index(k, values_.size(), "AmplitudeObserver");
    return values_[k];
}

// =============================================================================
// ExpectationObserver
// =============================================================================

void ExpectationObserver::observe(const ObservationContext& ctx) {
    double value = 0.0;

    switch (ctx.state.form()) {
        case StateForm::Statevector:
            value = observable_.expectation_value(ctx.state.statevector());
            break;

        case StateForm::DensityMatrix:
            value = ctx.state.density_matrix().expectation_value_sparse(observable_);
            break;

        case StateForm::Stabilizer: {
            // Term by term off the tableau. Every term is +1, -1 or 0 there, so
            // the whole observable costs O(terms * n^2) and no amplitude is
            // ever built.
            const StabilizerState& tableau = ctx.state.stabilizer();
            for (const PauliString& term : observable_.terms) {
                if (static_cast<int>(term.pauli.size()) != tableau.n_qubits) {
                    throw std::invalid_argument(
                        "ExpectationObserver: term '" + term.pauli + "' covers " +
                        std::to_string(term.pauli.size()) + " qubits, but the "
                        "state has " + std::to_string(tableau.n_qubits));
                }
                value += term.coeff.real *
                         static_cast<double>(tableau.expectation_pauli(term.pauli));
            }
            break;
        }

        case StateForm::MPS: {
            auto dense = dense_state(ctx, "ExpectationObserver");
            if (!dense) return;
            value = observable_.expectation_value(*dense);
            break;
        }
    }

    values_.push_back(value);
    record(ctx, value);
}

double ExpectationObserver::value(std::size_t k) const {
    check_index(k, values_.size(), "ExpectationObserver");
    return values_[k];
}

double ExpectationObserver::mean() const {
    if (values_.empty()) return 0.0;
    double total = 0.0;
    for (const double value : values_) total += value;
    return total / static_cast<double>(values_.size());
}

double ExpectationObserver::variance() const {
    if (values_.size() < 2) return 0.0;
    const double m = mean();
    double total = 0.0;
    for (const double value : values_) {
        const double delta = value - m;
        total += delta * delta;
    }
    return total / static_cast<double>(values_.size());
}

// =============================================================================
// PurityObserver
// =============================================================================

void PurityObserver::observe(const ObservationContext& ctx) {
    double purity = 1.0;
    if (ctx.state.form() == StateForm::DensityMatrix) {
        purity = ctx.state.density_matrix().purity();
    }
    // The other three hold a pure state by construction. Truncation moves an
    // MPS to a different pure state rather than to a mixed one, so 1 is the
    // answer there too.

    values_.push_back(purity);
    record(ctx, purity);
}

double PurityObserver::value(std::size_t k) const {
    check_index(k, values_.size(), "PurityObserver");
    return values_[k];
}

// =============================================================================
// ClassicalRegisterObserver
// =============================================================================

void ClassicalRegisterObserver::observe(const ObservationContext& ctx) {
    values_.push_back(ctx.clbits);
    record(ctx, ctx.clbits);
}

const std::vector<int>& ClassicalRegisterObserver::clbits(std::size_t k) const {
    check_index(k, values_.size(), "ClassicalRegisterObserver");
    return values_[k];
}

// =============================================================================
// BondDimensionObserver / TruncationObserver
// =============================================================================

void BondDimensionObserver::observe(const ObservationContext& ctx) {
    if (ctx.state.form() != StateForm::MPS) {
        detail::refuse_observation(
            ctx.plan.options,
            std::string("BondDimensionObserver asks for bond dimensions from a "
                        "backend holding a ") + to_string(ctx.state.form()) +
            ", which has no bonds to report.");
        return;
    }

    const MPSState& mps = ctx.state.mps();
    std::vector<int> bonds;
    bonds.reserve(mps.tensors.size());
    for (const auto& tensor : mps.tensors) bonds.push_back(tensor.bond_right);

    values_.push_back(bonds);
    record(ctx, std::move(bonds));
}

const std::vector<int>& BondDimensionObserver::bond_dimensions(std::size_t k) const {
    check_index(k, values_.size(), "BondDimensionObserver");
    return values_[k];
}

void TruncationObserver::observe(const ObservationContext& ctx) {
    if (ctx.state.form() != StateForm::MPS) {
        detail::refuse_observation(
            ctx.plan.options,
            std::string("TruncationObserver asks for discarded weight from a "
                        "backend holding a ") + to_string(ctx.state.form()) +
            ", which discards nothing.");
        return;
    }

    const double discarded = ctx.state.mps().truncation_error();
    values_.push_back(discarded);
    record(ctx, discarded);
}

// =============================================================================
// EntropyObserver
// =============================================================================

namespace {

using Cplx = std::complex<double>;

bool charge_bytes(const StateView& view, std::size_t bytes,
                  const RunPlan::Options& options) {
    return detail::charge_allocation(view, bytes, options, "EntropyObserver");
}

// Entropy in bits from a spectrum that need not be normalised. Weights at or
// below the floor are dropped rather than contributing: p*log(p) tends to zero
// as p does, so a numerical zero must not become a large negative logarithm.
double entropy_bits(std::vector<double> weights, double order) {
    double total = 0.0;
    for (const double w : weights) total += (w > 0.0) ? w : 0.0;
    if (!(total > 0.0)) return 0.0;

    const double floor = 1e-14;
    double entropy = 0.0;

    if (std::abs(order - 1.0) < 1e-12) {
        for (const double w : weights) {
            const double p = w / total;
            if (p > floor) entropy -= p * std::log2(p);
        }
        return entropy;
    }

    double moment = 0.0;
    for (const double w : weights) {
        const double p = w / total;
        if (p > floor) moment += std::pow(p, order);
    }
    if (!(moment > 0.0)) return 0.0;
    return std::log2(moment) / (1.0 - order);
}

// The qubits NOT in region, ascending.
std::vector<int> complement_of(const std::vector<int>& region, int n_qubits) {
    std::vector<bool> named(static_cast<std::size_t>(n_qubits), false);
    for (const int q : region) {
        if (q < 0 || q >= n_qubits) {
            throw std::invalid_argument(
                "EntropyObserver: qubit " + std::to_string(q) +
                " is outside a " + std::to_string(n_qubits) + " qubit register");
        }
        if (named[static_cast<std::size_t>(q)]) {
            throw std::invalid_argument(
                "EntropyObserver: qubit " + std::to_string(q) +
                " is named twice; a cut has each qubit on one side of it");
        }
        named[static_cast<std::size_t>(q)] = true;
    }

    std::vector<int> rest;
    for (int q = 0; q < n_qubits; ++q) {
        if (!named[static_cast<std::size_t>(q)]) rest.push_back(q);
    }
    return rest;
}

// Gather the bits of `index` sitting at the named positions, in list order.
inline std::size_t gather(std::size_t index, const std::vector<int>& positions) {
    std::size_t out = 0;
    for (std::size_t j = 0; j < positions.size(); ++j) {
        if ((index >> positions[j]) & 1u) out |= (std::size_t{1} << j);
    }
    return out;
}

// Eigenvalues of the reduced state of `side`, read off a dense statevector.
// Charged against the guard, since it allocates a copy of the state plus the
// reduced matrix.
std::vector<double> reduced_spectrum_dense(const Statevector& sv,
                                           const std::vector<int>& side,
                                           const std::vector<int>& other,
                                           const StateView& view,
                                           const RunPlan::Options& options,
                                           bool& produced) {
    produced = false;
    const std::size_t dim_side = std::size_t{1} << side.size();
    const std::size_t dim_other = std::size_t{1} << other.size();

    if (!charge_bytes(view, sv.dimension() * sizeof(Cplx) +
                                dim_side * dim_side * sizeof(Cplx),
                      options)) {
        return {};
    }

    // psi laid out as (side, other), so the reduction is a single pass.
    std::vector<Cplx> reshaped(sv.dimension());
    for (std::size_t k = 0; k < sv.dimension(); ++k) {
        const std::size_t s = gather(k, side);
        const std::size_t c = gather(k, other);
        reshaped[s * dim_other + c] = Cplx(sv.real_parts[k], sv.imag_parts[k]);
    }

    std::vector<Cplx> rho(dim_side * dim_side, Cplx(0.0, 0.0));
    for (std::size_t a = 0; a < dim_side; ++a) {
        for (std::size_t b = 0; b <= a; ++b) {
            Cplx sum(0.0, 0.0);
            for (std::size_t c = 0; c < dim_other; ++c) {
                sum += reshaped[a * dim_other + c] *
                       std::conj(reshaped[b * dim_other + c]);
            }
            rho[a * dim_side + b] = sum;
            rho[b * dim_side + a] = std::conj(sum);
        }
    }

    std::vector<double> evals(dim_side, 0.0);
    if (!detail::eigh(rho.data(), static_cast<int>(dim_side),
                      detail::MatrixOrder::RowMajor, evals.data(), nullptr)) {
        throw std::runtime_error(
            "EntropyObserver: the eigensolver failed on the reduced state");
    }
    produced = true;
    return evals;
}

// Scatter bits back out to the positions they were gathered from.
inline std::size_t scatter(std::size_t packed, const std::vector<int>& positions) {
    std::size_t out = 0;
    for (std::size_t j = 0; j < positions.size(); ++j) {
        if ((packed >> j) & 1u) out |= (std::size_t{1} << positions[j]);
    }
    return out;
}

// Reduced spectrum from a density matrix, by partial trace over `other`.
std::optional<std::vector<double>> dm_reduced_spectrum(
    const ObservationContext& ctx, const std::vector<int>& region,
    const std::vector<int>& rest) {
    // Symmetric across the cut only for a PURE global state, so a density
    // matrix must reduce the side that was actually named.
    const std::vector<int>& side = region;
    const std::vector<int>& other = rest;

    const DensityMatrix& dm = ctx.state.density_matrix();
    const std::size_t dim_side = std::size_t{1} << side.size();
    const std::size_t dim_other = std::size_t{1} << other.size();

    if (!charge_bytes(ctx.state, dim_side * dim_side * sizeof(Cplx),
                      ctx.plan.options)) {
        return std::nullopt;
    }

    std::vector<Cplx> rho(dim_side * dim_side, Cplx(0.0, 0.0));
    for (std::size_t a = 0; a < dim_side; ++a) {
        const std::size_t a_bits = scatter(a, side);
        for (std::size_t b = 0; b < dim_side; ++b) {
            const std::size_t b_bits = scatter(b, side);
            Cplx sum(0.0, 0.0);
            for (std::size_t c = 0; c < dim_other; ++c) {
                const std::size_t c_bits = scatter(c, other);
                const Complex128& entry = dm(a_bits | c_bits, b_bits | c_bits);
                sum += Cplx(entry.real, entry.imag);
            }
            rho[a * dim_side + b] = sum;
        }
    }

    std::vector<double> evals(dim_side, 0.0);
    if (!detail::eigh(rho.data(), static_cast<int>(dim_side),
                      detail::MatrixOrder::RowMajor, evals.data(), nullptr)) {
        throw std::runtime_error(
            "EntropyObserver: the eigensolver failed on the reduced state");
    }
    return evals;
}

// Hermitian square root of an n x n Hermitian positive semi-definite matrix,
// row-major in and out. Negative eigenvalues are rounding around zero and are
// clamped: their square root is what does not exist, not the matrix.
std::vector<Cplx> hermitian_sqrt(const std::vector<Cplx>& matrix, int n) {
    std::vector<double> evals(static_cast<std::size_t>(n), 0.0);
    std::vector<Cplx> evecs(static_cast<std::size_t>(n) * n, Cplx(0.0, 0.0));
    if (!detail::eigh(matrix.data(), n, detail::MatrixOrder::RowMajor,
                      evals.data(), evecs.data())) {
        throw std::runtime_error(
            "EntropyObserver: the eigensolver failed on an environment matrix");
    }

    std::vector<Cplx> root(static_cast<std::size_t>(n) * n, Cplx(0.0, 0.0));
    for (int m = 0; m < n; ++m) {
        const double weight = std::sqrt(evals[static_cast<std::size_t>(m)] > 0.0
                                            ? evals[static_cast<std::size_t>(m)]
                                            : 0.0);
        if (weight == 0.0) continue;
        // Eigenvectors come back column-major, column m paired with evals[m].
        const Cplx* column = evecs.data() + static_cast<std::size_t>(m) * n;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                root[static_cast<std::size_t>(i) * n + j] +=
                    column[i] * weight * std::conj(column[j]);
            }
        }
    }
    return root;
}

std::vector<Cplx> matmul(const std::vector<Cplx>& a, const std::vector<Cplx>& b, int n) {
    std::vector<Cplx> out(static_cast<std::size_t>(n) * n, Cplx(0.0, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            const Cplx aik = a[static_cast<std::size_t>(i) * n + k];
            if (aik == Cplx(0.0, 0.0)) continue;
            for (int j = 0; j < n; ++j) {
                out[static_cast<std::size_t>(i) * n + j] +=
                    aik * b[static_cast<std::size_t>(k) * n + j];
            }
        }
    }
    return out;
}

// Reduced spectrum at an MPS bond, WITHOUT assuming canonical form, which gate
// application here does not maintain.
//
// With |psi> = sum_m |L_m>|R_m>, the reduced state's nonzero spectrum is that
// of G_R * conj(G_L), where G_L and G_R are the two environment Gram matrices.
// The conjugate is load bearing: G_L is Hermitian, so the transpose the
// derivation produces is a conjugation, and dropping it returns a plausible
// wrong number rather than an obvious one.
//
// Both Grams are positive semi-definite, so the product's spectrum equals that
// of the Hermitian A^(1/2) G_R A^(1/2) with A = conj(G_L), which is what keeps
// this on the self-adjoint eigensolver.
std::optional<std::vector<double>> mps_bond_spectrum(const MPSState& mps, int cut) {
    std::vector<Cplx> gl{Cplx(1.0, 0.0)};
    int gl_dim = 1;
    for (int q = 0; q < cut; ++q) {
        const MPSTensor& t = mps.tensors[static_cast<std::size_t>(q)];
        const int br = t.bond_right;
        std::vector<Cplx> next(static_cast<std::size_t>(br) * br, Cplx(0.0, 0.0));
        for (int c = 0; c < br; ++c) {
            for (int d = 0; d < br; ++d) {
                Cplx sum(0.0, 0.0);
                for (int a = 0; a < t.bond_left; ++a) {
                    for (int b = 0; b < t.bond_left; ++b) {
                        const Cplx weight = gl[static_cast<std::size_t>(a) * gl_dim + b];
                        if (weight == Cplx(0.0, 0.0)) continue;
                        for (int s = 0; s < 2; ++s) {
                            const Complex128& ta = t(a, s, c);
                            const Complex128& tb = t(b, s, d);
                            sum += weight * Cplx(ta.real, ta.imag) *
                                   std::conj(Cplx(tb.real, tb.imag));
                        }
                    }
                }
                next[static_cast<std::size_t>(c) * br + d] = sum;
            }
        }
        gl = std::move(next);
        gl_dim = br;
    }

    std::vector<Cplx> gr{Cplx(1.0, 0.0)};
    int gr_dim = 1;
    for (int q = static_cast<int>(mps.tensors.size()) - 1; q >= cut; --q) {
        const MPSTensor& t = mps.tensors[static_cast<std::size_t>(q)];
        const int bl = t.bond_left;
        std::vector<Cplx> next(static_cast<std::size_t>(bl) * bl, Cplx(0.0, 0.0));
        for (int a = 0; a < bl; ++a) {
            for (int b = 0; b < bl; ++b) {
                Cplx sum(0.0, 0.0);
                for (int c = 0; c < t.bond_right; ++c) {
                    for (int d = 0; d < t.bond_right; ++d) {
                        const Cplx weight = gr[static_cast<std::size_t>(c) * gr_dim + d];
                        if (weight == Cplx(0.0, 0.0)) continue;
                        for (int s = 0; s < 2; ++s) {
                            const Complex128& ta = t(a, s, c);
                            const Complex128& tb = t(b, s, d);
                            sum += weight * Cplx(ta.real, ta.imag) *
                                   std::conj(Cplx(tb.real, tb.imag));
                        }
                    }
                }
                next[static_cast<std::size_t>(a) * bl + b] = sum;
            }
        }
        gr = std::move(next);
        gr_dim = bl;
    }

    if (gl_dim != gr_dim) {
        throw std::runtime_error(
            "EntropyObserver: the MPS bond dimensions on the two sides of the "
            "cut disagree, which means the chain is malformed");
    }

    std::vector<Cplx> a_matrix(gl.size());
    for (std::size_t i = 0; i < gl.size(); ++i) a_matrix[i] = std::conj(gl[i]);

    const std::vector<Cplx> root = hermitian_sqrt(a_matrix, gl_dim);
    const std::vector<Cplx> product = matmul(matmul(root, gr, gl_dim), root, gl_dim);

    std::vector<double> evals(static_cast<std::size_t>(gl_dim), 0.0);
    if (!detail::eigh(product.data(), gl_dim, detail::MatrixOrder::RowMajor,
                      evals.data(), nullptr)) {
        throw std::runtime_error(
            "EntropyObserver: the eigensolver failed on the MPS bond spectrum");
    }
    return evals;
}

// The bond route applies only to a cut that splits the chain in one place. Any
// other bipartition has no bond to read and falls back to dense amplitudes,
// charged through the knobs like every other conversion.
std::optional<std::vector<double>> mps_bipartition_spectrum(
    const ObservationContext& ctx, const std::vector<int>& region,
    const std::vector<int>& rest) {
    const int n = ctx.state.n_qubits();

    std::vector<int> sorted = region;
    std::sort(sorted.begin(), sorted.end());
    bool is_prefix = true;
    for (std::size_t j = 0; j < sorted.size(); ++j) {
        if (sorted[j] != static_cast<int>(j)) { is_prefix = false; break; }
    }
    bool is_suffix = true;
    for (std::size_t j = 0; j < sorted.size(); ++j) {
        if (sorted[j] != n - static_cast<int>(sorted.size()) + static_cast<int>(j)) {
            is_suffix = false;
            break;
        }
    }

    if (is_prefix || is_suffix) {
        const int cut = is_prefix ? static_cast<int>(sorted.size())
                                  : n - static_cast<int>(sorted.size());
        return mps_bond_spectrum(ctx.state.mps(), cut);
    }

    auto dense = detail::produce_state(ctx.state, StateForm::Statevector,
                                       ctx.plan.options, "EntropyObserver");
    if (!dense) return std::nullopt;

    const Statevector& sv = *static_cast<const Statevector*>(dense.get());
    const bool region_smaller = region.size() <= rest.size();
    bool produced = false;
    auto evals = reduced_spectrum_dense(sv, region_smaller ? region : rest,
                                        region_smaller ? rest : region,
                                        ctx.state, ctx.plan.options, produced);
    if (!produced) return std::nullopt;
    return evals;
}

}  // namespace

EntropyObserver::EntropyObserver(std::vector<int> region, double renyi_order,
                                 std::string label)
    : BundleWriter(std::move(label)), region_(std::move(region)), order_(renyi_order) {
    if (region_.empty()) {
        throw std::invalid_argument(
            "EntropyObserver: a cut needs at least one qubit on each side");
    }
    if (!(renyi_order > 0.0)) {
        throw std::invalid_argument(
            "EntropyObserver: the Renyi order must be positive");
    }
}

void EntropyObserver::observe(const ObservationContext& ctx) {
    const int n = ctx.state.n_qubits();
    const std::vector<int> rest = complement_of(region_, n);
    if (rest.empty()) {
        throw std::invalid_argument(
            "EntropyObserver: the cut names every qubit, so there is no other "
            "side for the state to be entangled with");
    }

    double value = 0.0;
    bool entanglement = true;

    switch (ctx.state.form()) {
        case StateForm::Stabilizer: {
            // Exact, integral, and never touches an amplitude. The spectrum is
            // flat, so the Renyi order does not enter.
            value = ctx.state.stabilizer().entanglement_entropy_bits(region_);
            break;
        }

        case StateForm::MPS: {
            auto spectrum = mps_bipartition_spectrum(ctx, region_, rest);
            if (!spectrum) return;
            value = entropy_bits(*spectrum, order_);
            break;
        }

        case StateForm::Statevector: {
            // Entropy is symmetric across the cut, so reduce the SMALLER side:
            // same answer, and the eigensolve is on the smaller matrix.
            const bool region_smaller = region_.size() <= rest.size();
            const std::vector<int>& side = region_smaller ? region_ : rest;
            const std::vector<int>& other = region_smaller ? rest : region_;
            bool produced = false;
            auto evals = reduced_spectrum_dense(ctx.state.statevector(), side,
                                                other, ctx.state,
                                                ctx.plan.options, produced);
            if (!produced) return;
            value = entropy_bits(std::move(evals), order_);
            break;
        }

        case StateForm::DensityMatrix: {
            auto spectrum = dm_reduced_spectrum(ctx, region_, rest);
            if (!spectrum) return;
            value = entropy_bits(*spectrum, order_);
            // The global state may be mixed, and deciding that costs as much as
            // the state itself, so no entanglement claim is made here.
            entanglement = false;
            break;
        }
    }

    values_.push_back(value);
    entanglement_.push_back(entanglement);
    record(ctx, value);
}

double EntropyObserver::value(std::size_t k) const {
    check_index(k, values_.size(), "EntropyObserver");
    return values_[k];
}

bool EntropyObserver::is_entanglement(std::size_t k) const {
    check_index(k, entanglement_.size(), "EntropyObserver");
    return entanglement_[k];
}

// =============================================================================
// CallbackObserver
// =============================================================================

CallbackObserver::CallbackObserver(Fn fn) : fn_(std::move(fn)) {
    if (!fn_) {
        throw std::invalid_argument("CallbackObserver: callback must be callable");
    }
}

void CallbackObserver::observe(const ObservationContext& ctx) { fn_(ctx); }

}  // namespace lindblad
