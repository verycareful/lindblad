#include "lindblad/observation.hpp"

#include "lindblad/circuit.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/validation.hpp"
#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"

#include <algorithm>
#include <exception>
#include <limits>
#include <stdexcept>

namespace lindblad {

// =============================================================================
// StateForm
// =============================================================================

const char* to_string(StateForm form) {
    switch (form) {
        case StateForm::Statevector:   return "statevector";
        case StateForm::DensityMatrix: return "density matrix";
        case StateForm::Stabilizer:    return "stabilizer";
        case StateForm::MPS:           return "matrix product state";
    }
    return "unknown";
}

namespace {

// Bytes a dense form would occupy, saturating rather than wrapping. A size that
// does not fit in size_t is reported as the largest one, so the guard refuses it
// on the same comparison it uses for everything else instead of needing a
// special case for the overflow.
std::size_t dense_bytes(int n_qubits, bool squared) {
    const int exponent = squared ? 2 * n_qubits : n_qubits;
    const int width = std::numeric_limits<std::size_t>::digits;
    if (exponent < 0 || exponent >= width - 4) return std::numeric_limits<std::size_t>::max();
    return (std::size_t{1} << exponent) * sizeof(Complex128);
}

}  // namespace

// =============================================================================
// StateView - a non-owning window on a backend's live state
// =============================================================================

StateView::StateView(StateForm form, const void* state, int n_qubits)
    : form_(form), state_(state), n_qubits_(n_qubits) {
    if (state == nullptr) {
        throw std::invalid_argument("StateView: no state to view");
    }
}

const Statevector& StateView::statevector() const {
    if (form_ != StateForm::Statevector) {
        throw std::invalid_argument(
            std::string("StateView::statevector: this backend holds a ") +
            to_string(form_) + ", not a statevector. Conversion is a separate "
            "request, so that its cost is visible at the call site.");
    }
    return *static_cast<const Statevector*>(state_);
}

const DensityMatrix& StateView::density_matrix() const {
    if (form_ != StateForm::DensityMatrix) {
        throw std::invalid_argument(
            std::string("StateView::density_matrix: this backend holds a ") +
            to_string(form_) + ", not a density matrix. Conversion is a "
            "separate request, so that its cost is visible at the call site.");
    }
    return *static_cast<const DensityMatrix*>(state_);
}

const StabilizerState& StateView::stabilizer() const {
    if (form_ != StateForm::Stabilizer) {
        throw std::invalid_argument(
            std::string("StateView::stabilizer: this backend holds a ") +
            to_string(form_) + ", not a stabilizer tableau. No conversion into "
            "a tableau exists from any other form.");
    }
    return *static_cast<const StabilizerState*>(state_);
}

const MPSState& StateView::mps() const {
    if (form_ != StateForm::MPS) {
        throw std::invalid_argument(
            std::string("StateView::mps: this backend holds a ") +
            to_string(form_) + ", not a matrix product state. No conversion "
            "into an MPS exists from any other form.");
    }
    return *static_cast<const MPSState*>(state_);
}

std::size_t StateView::state_bytes() const {
    switch (form_) {
        case StateForm::Statevector: {
            const auto& sv = *static_cast<const Statevector*>(state_);
            return sv.dimension() * 2 * sizeof(double);
        }
        case StateForm::DensityMatrix: {
            const auto& dm = *static_cast<const DensityMatrix*>(state_);
            return dm.data.size() * sizeof(Complex128);
        }
        case StateForm::Stabilizer: {
            // 2N rows, each holding 2N X/Z bits plus the padding word the
            // tableau keeps so a 64-bit read starting anywhere is in bounds,
            // plus one phase byte per row.
            const std::size_t rows = 2u * static_cast<std::size_t>(n_qubits_);
            const std::size_t words = (rows + 63u) / 64u + 1u;
            return rows * words * sizeof(std::uint64_t) + rows;
        }
        case StateForm::MPS: {
            const auto& mps = *static_cast<const MPSState*>(state_);
            std::size_t bytes = 0;
            for (const auto& tensor : mps.tensors) {
                bytes += tensor.data.size() * sizeof(Complex128);
            }
            return bytes;
        }
    }
    return 0;
}

bool StateView::convertible_to(StateForm target) const {
    if (target == form_) return true;
    switch (form_) {
        case StateForm::Statevector:
            return target == StateForm::DensityMatrix;
        case StateForm::Stabilizer:
        case StateForm::MPS:
            return target == StateForm::Statevector ||
                   target == StateForm::DensityMatrix;
        case StateForm::DensityMatrix:
            // A mixed state has no statevector, and deciding purity costs a
            // full eigendecomposition to answer "sometimes". Neither a tableau
            // nor an MPS can be recovered from dense data either.
            return false;
    }
    return false;
}

std::size_t StateView::conversion_bytes(StateForm target) const {
    if (target == form_) return state_bytes();
    switch (target) {
        case StateForm::Statevector:   return dense_bytes(n_qubits_, false);
        case StateForm::DensityMatrix: return dense_bytes(n_qubits_, true);
        case StateForm::Stabilizer:
        case StateForm::MPS:           return 0;  // no route, so nothing is allocated
    }
    return 0;
}

// =============================================================================
// Observer
// =============================================================================

void Observer::begin_run(int, int) {}
void Observer::end_run() {}

// =============================================================================
// Anchor
// =============================================================================

Anchor Anchor::at_start() {
    Anchor a;
    a.kind_ = Kind::Start;
    return a;
}

Anchor Anchor::at_end() {
    Anchor a;
    a.kind_ = Kind::End;
    return a;
}

Anchor Anchor::after_instruction(int index) {
    if (index < 0) {
        throw std::invalid_argument(
            "Anchor::after_instruction: index must not be negative");
    }
    Anchor a;
    a.kind_ = Kind::InstructionIndex;
    a.index_ = index;
    return a;
}

Anchor Anchor::after_label(std::string label) {
    if (label.empty()) {
        throw std::invalid_argument(
            "Anchor::after_label: an empty label matches nothing, and an "
            "instruction carrying no label is not an anchor");
    }
    Anchor a;
    a.kind_ = Kind::InstructionLabel;
    a.label_ = std::move(label);
    return a;
}

Anchor Anchor::every_instruction() {
    Anchor a;
    a.kind_ = Kind::EveryInstruction;
    return a;
}

Anchor Anchor::every_layer() {
    Anchor a;
    a.kind_ = Kind::EveryLayer;
    return a;
}

Anchor Anchor::before_each_measurement() {
    Anchor a;
    a.kind_ = Kind::BeforeMeasurement;
    return a;
}

Anchor Anchor::after_each_measurement() {
    Anchor a;
    a.kind_ = Kind::AfterMeasurement;
    return a;
}

Anchor Anchor::where(PredicateFn predicate) {
    if (!predicate) {
        throw std::invalid_argument("Anchor::where: predicate must be callable");
    }
    Anchor a;
    a.kind_ = Kind::Predicate;
    a.predicate_ = std::move(predicate);
    return a;
}

std::string Anchor::name() const {
    switch (kind_) {
        case Kind::Start:             return "at_start";
        case Kind::End:               return "at_end";
        case Kind::InstructionIndex:  return "after_instruction(" + std::to_string(index_) + ")";
        case Kind::InstructionLabel:  return "after_label(" + label_ + ")";
        case Kind::EveryInstruction:  return "every_instruction";
        case Kind::EveryLayer:        return "every_layer";
        case Kind::BeforeMeasurement: return "before_each_measurement";
        case Kind::AfterMeasurement:  return "after_each_measurement";
        case Kind::Predicate:         return "where(predicate)";
    }
    return "anchor";
}

// =============================================================================
// ObservationBundle
// =============================================================================

namespace {

const ObservationBundle::Payload& entry_or_throw(
    const std::unordered_map<std::string, ObservationBundle::Payload>& entries,
    const std::string& label, const char* wanted) {
    const auto it = entries.find(label);
    if (it == entries.end()) {
        throw std::invalid_argument(
            "ObservationBundle: no observation is stored under the label '" +
            label + "', so " + wanted + " cannot be read from it");
    }
    return it->second;
}

template <typename T>
const T& payload_or_throw(const ObservationBundle::Payload& payload,
                          const std::string& label, const char* wanted) {
    if (const T* value = std::get_if<T>(&payload)) return *value;
    throw std::invalid_argument(
        "ObservationBundle: the observation stored under '" + label +
        "' is not " + wanted);
}

}  // namespace

bool ObservationBundle::contains(const std::string& label) const {
    return entries_.find(label) != entries_.end();
}

std::vector<std::string> ObservationBundle::labels() const {
    std::vector<std::string> out;
    out.reserve(entries_.size());
    for (const auto& entry : entries_) out.push_back(entry.first);
    std::sort(out.begin(), out.end());
    return out;
}

double ObservationBundle::number(const std::string& label) const {
    return payload_or_throw<double>(
        entry_or_throw(entries_, label, "a number"), label, "a number");
}

const std::vector<double>& ObservationBundle::reals(const std::string& label) const {
    return payload_or_throw<std::vector<double>>(
        entry_or_throw(entries_, label, "real values"), label, "a list of reals");
}

const std::vector<Complex128>& ObservationBundle::amplitudes(const std::string& label) const {
    return payload_or_throw<std::vector<Complex128>>(
        entry_or_throw(entries_, label, "amplitudes"), label, "a list of amplitudes");
}

const std::vector<int>& ObservationBundle::integers(const std::string& label) const {
    return payload_or_throw<std::vector<int>>(
        entry_or_throw(entries_, label, "integers"), label, "a list of integers");
}

const std::string& ObservationBundle::text(const std::string& label) const {
    return payload_or_throw<std::string>(
        entry_or_throw(entries_, label, "text"), label, "text");
}

StateForm ObservationBundle::form(const std::string& label) const {
    return payload_or_throw<StatePayload>(
        entry_or_throw(entries_, label, "a state"), label, "a state").form;
}

namespace {

const ObservationBundle::StatePayload& state_payload(
    const ObservationBundle::Payload& payload, const std::string& label,
    StateForm wanted) {
    const auto& state = payload_or_throw<ObservationBundle::StatePayload>(
        payload, label, "a state");
    if (state.form != wanted) {
        throw std::invalid_argument(
            "ObservationBundle: the state stored under '" + label + "' is a " +
            to_string(state.form) + ", not a " + to_string(wanted));
    }
    return state;
}

}  // namespace

const Statevector& ObservationBundle::statevector(const std::string& label) const {
    const auto& state = state_payload(
        entry_or_throw(entries_, label, "a statevector"), label, StateForm::Statevector);
    return *static_cast<const Statevector*>(state.state.get());
}

const DensityMatrix& ObservationBundle::density_matrix(const std::string& label) const {
    const auto& state = state_payload(
        entry_or_throw(entries_, label, "a density matrix"), label, StateForm::DensityMatrix);
    return *static_cast<const DensityMatrix*>(state.state.get());
}

const StabilizerState& ObservationBundle::stabilizer(const std::string& label) const {
    const auto& state = state_payload(
        entry_or_throw(entries_, label, "a stabilizer tableau"), label, StateForm::Stabilizer);
    return *static_cast<const StabilizerState*>(state.state.get());
}

const MPSState& ObservationBundle::mps(const std::string& label) const {
    const auto& state = state_payload(
        entry_or_throw(entries_, label, "a matrix product state"), label, StateForm::MPS);
    return *static_cast<const MPSState*>(state.state.get());
}

void ObservationBundle::put(std::string label, Payload payload) {
    if (label.empty()) {
        throw std::invalid_argument("ObservationBundle: an observation needs a label");
    }
    const auto [it, inserted] = entries_.emplace(std::move(label), std::move(payload));
    if (!inserted) {
        throw std::invalid_argument(
            "ObservationBundle: '" + it->first + "' already holds an "
            "observation. Two observations under one label leave one of them "
            "unreachable, and the caller cannot tell which.");
    }
}

// =============================================================================
// InitialState
// =============================================================================

InitialState InitialState::basis(std::uint64_t index) {
    InitialState s;
    s.kind_ = Kind::Basis;
    s.basis_ = index;
    return s;
}

InitialState InitialState::from(std::shared_ptr<const Statevector> sv) {
    if (!sv) throw std::invalid_argument("InitialState::from: no statevector supplied");
    InitialState s;
    s.kind_ = Kind::State;
    s.form_ = StateForm::Statevector;
    s.state_ = std::move(sv);
    return s;
}

InitialState InitialState::from(std::shared_ptr<const DensityMatrix> dm) {
    if (!dm) throw std::invalid_argument("InitialState::from: no density matrix supplied");
    InitialState s;
    s.kind_ = Kind::State;
    s.form_ = StateForm::DensityMatrix;
    s.state_ = std::move(dm);
    return s;
}

InitialState InitialState::from(std::shared_ptr<const StabilizerState> st) {
    if (!st) throw std::invalid_argument("InitialState::from: no stabilizer state supplied");
    InitialState s;
    s.kind_ = Kind::State;
    s.form_ = StateForm::Stabilizer;
    s.state_ = std::move(st);
    return s;
}

InitialState InitialState::from(std::shared_ptr<const MPSState> mps) {
    if (!mps) throw std::invalid_argument("InitialState::from: no MPS supplied");
    InitialState s;
    s.kind_ = Kind::State;
    s.form_ = StateForm::MPS;
    s.state_ = std::move(mps);
    return s;
}

StateForm InitialState::form() const {
    if (kind_ != Kind::State) {
        throw std::invalid_argument(
            "InitialState::form: this initial state is not a supplied state, "
            "so it has no representation to report");
    }
    return form_;
}

// =============================================================================
// ObservationPlan
// =============================================================================

ObservationPlan& ObservationPlan::observe(Anchor anchor, ObserverPtr observer) {
    if (!observer) {
        throw std::invalid_argument(
            "ObservationPlan::observe: no observer supplied for anchor " +
            anchor.name());
    }
    attachments_.push_back(Attachment{std::move(anchor), std::move(observer)});
    return *this;
}

// =============================================================================
// detail - the pieces the backends drive
// =============================================================================

namespace detail {

bool refuse_observation(const RunPlan::Options& options, const std::string& message) {
    switch (options.response) {
        case Response::Throw:
            throw std::invalid_argument(message);
        case Response::Warn:
            emit_warning("note: " + message + " The observation is omitted.");
            return false;
        case Response::Ignore:
            return false;
    }
    return false;
}

std::shared_ptr<const void> produce_state(const StateView& view, StateForm target,
                                          const RunPlan::Options& options,
                                          const std::string& what) {
    const StateForm held = view.form();

    if (target != held) {
        if (!view.convertible_to(target)) {
            refuse_observation(
                options, what + " asks for a " + to_string(target) +
                " from a backend holding a " + to_string(held) +
                ", and no conversion between those exists at all.");
            return nullptr;
        }
        if (options.conversion == Conversion::Never) {
            refuse_observation(
                options, what + " asks for a " + to_string(target) +
                " from a backend holding a " + to_string(held) +
                ", and Conversion::Never is selected.");
            return nullptr;
        }
    }

    if (options.cost == Cost::Guarded) {
        const std::size_t wanted = view.conversion_bytes(target);
        const double live = static_cast<double>(view.state_bytes());
        const double budget = live * options.guard_multiple;
        if (static_cast<double>(wanted) > budget) {
            refuse_observation(
                options, what + " would allocate " + std::to_string(wanted) +
                " bytes against a live state of " +
                std::to_string(view.state_bytes()) +
                " bytes, which is over the guard. Cost::Unlimited allows it.");
            return nullptr;
        }
    } else if (target != held) {
        emit_warning(
            "note: " + what + " is converting a " + to_string(held) + " into a " +
            to_string(target) + ", allocating " +
            std::to_string(view.conversion_bytes(target)) +
            " bytes under Cost::Unlimited.");
    }

    switch (target) {
        case StateForm::Statevector: {
            if (held == StateForm::Statevector) {
                return std::make_shared<const Statevector>(view.statevector().clone());
            }
            if (held == StateForm::Stabilizer) {
                return std::make_shared<const Statevector>(view.stabilizer().to_statevector());
            }
            return std::make_shared<const Statevector>(view.mps().to_statevector());
        }
        case StateForm::DensityMatrix: {
            if (held == StateForm::DensityMatrix) {
                return std::make_shared<const DensityMatrix>(view.density_matrix());
            }
            if (held == StateForm::Statevector) {
                return std::make_shared<const DensityMatrix>(
                    DensityMatrix::from_statevector(view.statevector()));
            }
            const Statevector dense = (held == StateForm::Stabilizer)
                                          ? view.stabilizer().to_statevector()
                                          : view.mps().to_statevector();
            return std::make_shared<const DensityMatrix>(DensityMatrix::from_statevector(dense));
        }
        case StateForm::Stabilizer:
            return std::make_shared<const StabilizerState>(view.stabilizer());
        case StateForm::MPS:
            return std::make_shared<const MPSState>(view.mps());
    }
    return nullptr;
}

bool charge_allocation(const StateView& view, std::size_t bytes,
                       const RunPlan::Options& options, const std::string& what) {
    if (options.cost != Cost::Guarded) return true;

    const double budget = static_cast<double>(view.state_bytes()) * options.guard_multiple;
    if (static_cast<double>(bytes) <= budget) return true;

    return refuse_observation(
        options, what + " would allocate " + std::to_string(bytes) +
        " bytes against a live state of " + std::to_string(view.state_bytes()) +
        " bytes, which is over the guard. Cost::Unlimited allows it.");
}

void apply_initial_state(const RunPlan& plan, Statevector& sv) {
    const InitialState& initial = plan.initial;

    if (initial.is_default()) {
        sv.initialize();
        return;
    }

    if (initial.is_basis()) {
        const std::uint64_t index = initial.basis_index();
        if (index >= sv.dimension()) {
            throw std::invalid_argument(
                "InitialState::basis(" + std::to_string(index) +
                ") is outside a " + std::to_string(sv.dimension()) +
                " amplitude register");
        }
        sv.initialize_basis(static_cast<std::size_t>(index));
        return;
    }

    const StateView view(initial.form(), initial.state(), sv.n_qubits);
    auto produced = produce_state(view, StateForm::Statevector, plan.options,
                                  "InitialState");
    if (!produced) {
        // Warn and Ignore omit an observation, but there is no such thing as
        // omitting the state a run starts from: the alternative to the supplied
        // state is silently simulating a different circuit.
        throw std::invalid_argument(
            "InitialState: a " + std::string(to_string(initial.form())) +
            " cannot be turned into the statevector this backend runs on, and "
            "a run has to start somewhere. Response::Warn and Response::Ignore "
            "govern observations, not the initial state.");
    }

    const Statevector& source = *static_cast<const Statevector*>(produced.get());
    if (source.n_qubits != sv.n_qubits) {
        throw std::invalid_argument(
            "InitialState: the supplied state covers " +
            std::to_string(source.n_qubits) + " qubits, the circuit " +
            std::to_string(sv.n_qubits));
    }
    sv.set_amplitudes(source.real_parts, source.imag_parts, source.dim);
}

void apply_initial_state(const RunPlan& plan, DensityMatrix& dm) {
    const InitialState& initial = plan.initial;

    if (initial.is_default()) {
        dm.initialize();
        return;
    }

    if (initial.is_basis()) {
        const std::uint64_t index = initial.basis_index();
        if (index >= dm.dim) {
            throw std::invalid_argument(
                "InitialState::basis(" + std::to_string(index) +
                ") is outside a " + std::to_string(dm.dim) + " dimensional register");
        }
        std::fill(dm.data.begin(), dm.data.end(), Complex128(0.0, 0.0));
        dm(static_cast<std::size_t>(index), static_cast<std::size_t>(index)) =
            Complex128(1.0, 0.0);
        return;
    }

    const StateView view(initial.form(), initial.state(), dm.n_qubits);
    auto produced = produce_state(view, StateForm::DensityMatrix, plan.options,
                                  "InitialState");
    if (!produced) {
        throw std::invalid_argument(
            "InitialState: a " + std::string(to_string(initial.form())) +
            " cannot be turned into the density matrix this backend runs on, "
            "and a run has to start somewhere.");
    }

    const DensityMatrix& source = *static_cast<const DensityMatrix*>(produced.get());
    if (source.n_qubits != dm.n_qubits) {
        throw std::invalid_argument(
            "InitialState: the supplied state covers " +
            std::to_string(source.n_qubits) + " qubits, the circuit " +
            std::to_string(dm.n_qubits));
    }
    dm.data = source.data;
}

void apply_initial_state(const RunPlan& plan, StabilizerState& state) {
    const InitialState& initial = plan.initial;
    const int n = state.n_qubits;

    if (initial.is_default()) {
        state = StabilizerState(n);
        return;
    }

    if (initial.is_basis()) {
        const std::uint64_t index = initial.basis_index();
        if (n < 64 && index >= (std::uint64_t{1} << n)) {
            throw std::invalid_argument(
                "InitialState::basis(" + std::to_string(index) +
                ") is outside a " + std::to_string(n) + " qubit register");
        }
        state = StabilizerState(n);
        for (int q = 0; q < n && q < 64; ++q) {
            if ((index >> q) & 1ULL) state.apply_x(q);
        }
        return;
    }

    // A tableau describes a stabilizer state, and no other representation can
    // be turned into one: a general statevector is not a stabilizer state at
    // all, so this is impossibility rather than expense.
    if (initial.form() != StateForm::Stabilizer) {
        throw std::invalid_argument(
            "InitialState: a " + std::string(to_string(initial.form())) +
            " cannot be turned into a stabilizer tableau. Only a state that IS "
            "a stabilizer state has one, and recovering it from amplitudes is "
            "not a conversion this library performs.");
    }

    const auto& source = *static_cast<const StabilizerState*>(initial.state());
    if (source.n_qubits != n) {
        throw std::invalid_argument(
            "InitialState: the supplied state covers " +
            std::to_string(source.n_qubits) + " qubits, the circuit " +
            std::to_string(n));
    }
    state = source;
}

// ----- FiringGuard -----

FiringGuard::FiringGuard(ObservationRunner* runner, int index,
                         const Instruction& inst, const StateView& state)
    : runner_(runner), index_(index), inst_(&inst), state_(&state),
      uncaught_(std::uncaught_exceptions()) {}

FiringGuard::~FiringGuard() {
    if (runner_ == nullptr) return;
    if (std::uncaught_exceptions() != uncaught_) return;
    runner_->after_instruction(index_, *inst_, *state_);
}

// ----- ObservationRunner -----

ObservationRunner::ObservationRunner(const RunPlan& plan, const QuantumCircuit& circuit)
    : plan_(plan) {
    const int count = static_cast<int>(circuit.instructions.size());

    for (const auto& attachment : plan.observations.attachments()) {
        switch (attachment.anchor.kind()) {
            case Anchor::Kind::Start:
                start_.push_back(&attachment);
                break;
            case Anchor::Kind::End:
                end_.push_back(&attachment);
                break;
            case Anchor::Kind::EveryInstruction:
                every_.push_back(&attachment);
                break;
            case Anchor::Kind::EveryLayer:
                layer_.push_back(&attachment);
                break;
            case Anchor::Kind::BeforeMeasurement:
                before_measure_.push_back(&attachment);
                break;
            case Anchor::Kind::AfterMeasurement:
                after_measure_.push_back(&attachment);
                break;
            case Anchor::Kind::Predicate:
                predicate_.push_back(&attachment);
                break;

            case Anchor::Kind::InstructionIndex: {
                // Resolution happens here, before any simulation, because an
                // anchor that fires nowhere is indistinguishable from one that
                // fired and found nothing unless the library says so.
                const int index = attachment.anchor.index();
                if (index >= count) {
                    throw std::invalid_argument(
                        "ObservationPlan: " + attachment.anchor.name() +
                        " does not resolve: the circuit has " +
                        std::to_string(count) + " instructions.");
                }
                indexed_[index].push_back(&attachment);
                break;
            }

            case Anchor::Kind::InstructionLabel: {
                const std::string& label = attachment.anchor.label();
                const bool found = std::any_of(
                    circuit.instructions.begin(), circuit.instructions.end(),
                    [&](const Instruction& inst) { return inst.label == label; });
                if (!found) {
                    throw std::invalid_argument(
                        "ObservationPlan: " + attachment.anchor.name() +
                        " does not resolve: no instruction in the circuit "
                        "carries that label. A transpiler pass that removed or "
                        "replaced the instruction removes its label with it.");
                }
                labelled_[label].push_back(&attachment);
                break;
            }
        }
    }

    if (!layer_.empty()) compute_layer_boundaries(circuit);

    active_ = !plan.observations.empty();
}

// Greedy ASAP layering over qubit occupancy: an instruction opens a new layer
// when one of its qubits is already used in the current one. The boundary is
// recorded on the LAST instruction of each layer, which is the point a caller
// asking to see every layer wants to be handed the state at.
void ObservationRunner::compute_layer_boundaries(const QuantumCircuit& circuit) {
    const int count = static_cast<int>(circuit.instructions.size());
    if (count == 0) return;

    std::vector<bool> used(static_cast<std::size_t>(circuit.n_qubits), false);
    for (int i = 0; i < count; ++i) {
        const Instruction& inst = circuit.instructions[i];

        bool collides = false;
        for (const int q : inst.qubits) {
            if (q >= 0 && q < circuit.n_qubits && used[static_cast<std::size_t>(q)]) {
                collides = true;
                break;
            }
        }

        if (collides) {
            layer_end_.insert(i - 1);
            std::fill(used.begin(), used.end(), false);
        }
        for (const int q : inst.qubits) {
            if (q >= 0 && q < circuit.n_qubits) used[static_cast<std::size_t>(q)] = true;
        }
    }
    layer_end_.insert(count - 1);
}

void ObservationRunner::begin_run(int n_qubits, int n_shots) {
    n_shots_ = n_shots;
    for (const auto& attachment : plan_.observations.attachments()) {
        attachment.observer->begin_run(n_qubits, n_shots);
    }
}

void ObservationRunner::end_run() {
    for (const auto& attachment : plan_.observations.attachments()) {
        attachment.observer->end_run();
    }
}

void ObservationRunner::begin_shot(int shot, const std::vector<int>& clbits) {
    shot_ = shot;
    clbits_ = &clbits;
}

void ObservationRunner::fire(const std::vector<const ObservationPlan::Attachment*>& group,
                             const StateView& state, int instruction_index) {
    static const std::vector<int> no_clbits;
    for (const auto* attachment : group) {
        const std::string anchor = attachment->anchor.name();
        const ObservationContext ctx{state,
                                     anchor,
                                     instruction_index,
                                     shot_,
                                     n_shots_,
                                     clbits_ ? *clbits_ : no_clbits,
                                     plan_,
                                     bundle_};
        attachment->observer->observe(ctx);
    }
}

void ObservationRunner::at_start(const StateView& state) {
    if (start_.empty()) return;
    fire(start_, state, -1);
}

void ObservationRunner::at_end(const StateView& state, int last_index) {
    if (end_.empty()) return;
    fire(end_, state, last_index);
}

void ObservationRunner::before_instruction(int index, const Instruction& inst,
                                           const StateView& state) {
    if (before_measure_.empty()) return;
    if (inst.type != Instruction::GateType::MEASURE) return;
    fire(before_measure_, state, index);
}

void ObservationRunner::after_instruction(int index, const Instruction& inst,
                                          const StateView& state) {
    if (!every_.empty()) fire(every_, state, index);

    if (!indexed_.empty()) {
        const auto it = indexed_.find(index);
        if (it != indexed_.end()) fire(it->second, state, index);
    }

    if (!labelled_.empty() && !inst.label.empty()) {
        const auto it = labelled_.find(inst.label);
        if (it != labelled_.end()) fire(it->second, state, index);
    }

    if (!layer_.empty() && layer_end_.count(index) != 0) fire(layer_, state, index);

    if (!after_measure_.empty() && inst.type == Instruction::GateType::MEASURE) {
        fire(after_measure_, state, index);
    }

    for (const auto* attachment : predicate_) {
        if (attachment->anchor.predicate()(inst, index)) {
            static const std::vector<int> no_clbits;
            const std::string anchor = attachment->anchor.name();
            const ObservationContext ctx{state, anchor, index, shot_, n_shots_,
                                         clbits_ ? *clbits_ : no_clbits, plan_,
                                         bundle_};
            attachment->observer->observe(ctx);
        }
    }
}

}  // namespace detail

}  // namespace lindblad
