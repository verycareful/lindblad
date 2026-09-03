#pragma once

#include "lindblad/types.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

// =============================================================================
// observation - reading a simulation, and choosing where it starts
// =============================================================================
// A QuantumCircuit is a simulation of quantum mechanics and holds physics
// only. Observing a state is not physics: nature does not let a state be read
// without disturbance, and the run is identical whether or not anyone looks.
// Injecting a state is not physics either: nothing overwrites a state with one
// handed to it. Both are properties of how the simulation is being run, so
// both live here, in the harness, and neither appears in the circuit.
//
// A RunPlan is that harness. It carries where the run starts, what is watched
// while it runs, and the policy applied when something is asked for that the
// backend cannot hand over. Every simulator takes one as a trailing parameter
// and an empty plan means: start at |0...0>, watch nothing. There is no
// default observation set. A caller who did not ask to look sees nothing.
//
// Include direction is one way and load bearing: the backend headers include
// this one, so this one forward declares their state types and never includes
// them. Accessors that need a complete state type are defined out of line.

namespace lindblad {

class Statevector;
class DensityMatrix;
class StabilizerState;
class MPSState;
class QuantumCircuit;
class ObservationBundle;
struct Instruction;

// -----------------------------------------------------------------------------
// StateForm - which representation a state is held in
// -----------------------------------------------------------------------------
// Each backend holds exactly one of these natively. A caller asking for a
// different one is asking for a conversion, which is what RunPlan::Options
// governs.

enum class StateForm {
    Statevector,    // dense amplitudes, 2^n complex entries
    DensityMatrix,  // dense operator, 4^n complex entries
    Stabilizer,     // stabilizer tableau, O(n^2) bits
    MPS             // matrix product state, bond dimension dependent
};

// Name for messages and bundle reporting.
const char* to_string(StateForm form);

// -----------------------------------------------------------------------------
// The three knobs
// -----------------------------------------------------------------------------
// Three independent questions, deliberately not fused into one enum. Fusing
// them is what makes a policy unsayable: an enumerator that means "translate,
// and throw when translation is impossible" cannot also express "translate,
// and warn when translation is impossible".

// May the library translate into a representation the backend does not hold.
enum class Conversion {
    Convert,  // translate when a route exists
    Never     // hand back only what the backend already holds
};

// Whether an expensive observation has to be asked for explicitly.
enum class Cost {
    Guarded,   // an allocation beyond the guard is refused
    Unlimited  // size never refuses; only impossibility does
};

// What a refusal looks like, whatever caused it.
enum class Response {
    Throw,   // std::invalid_argument, raised at the pre-flight where possible
    Warn,    // report through the warning channel and omit the observation
    Ignore   // omit the observation silently
};

// -----------------------------------------------------------------------------
// StateView - a non-owning window on a backend's live state
// -----------------------------------------------------------------------------
// What an observer is handed at an anchor. Holding a view costs nothing: no
// copy is taken and no conversion is performed until something is asked for.
//
// The typed accessors return the backend's own representation and throw when
// the backend does not hold that form. They never convert, so a caller reading
// through them knows the cost is a pointer dereference. Conversion is the
// separate, explicit set of calls below, and only those consult the options.

class StateView {
public:
    StateView(StateForm form, const void* state, int n_qubits);

    StateForm form() const { return form_; }
    int n_qubits() const { return n_qubits_; }

    // Footprint of the live state in bytes. This is the yardstick the guard
    // measures a requested allocation against, which is why it is derived from
    // the running simulation rather than from a configured constant: it scales
    // with the problem and the machine on its own.
    std::size_t state_bytes() const;

    // The backend's own representation. Throws when form() is not this one.
    const Statevector& statevector() const;
    const DensityMatrix& density_matrix() const;
    const StabilizerState& stabilizer() const;
    const MPSState& mps() const;

    // Whether a conversion route from form() to `target` exists at all. False
    // means no option makes it available: a stabilizer tableau cannot be
    // recovered from an arbitrary statevector, and an MPS cannot be recovered
    // from a dense state at a bond dimension too small to hold it.
    bool convertible_to(StateForm target) const;

    // Bytes the conversion would allocate. Used by the guard, and worth having
    // publicly: a caller can ask before committing to a run.
    std::size_t conversion_bytes(StateForm target) const;

private:
    StateForm form_;
    const void* state_;
    int n_qubits_;
};

// -----------------------------------------------------------------------------
// ObservationContext - what an observer sees when an anchor fires
// -----------------------------------------------------------------------------
// Every member is a view or a scalar. Nothing here is owned by the observer,
// and nothing survives the call, so an observer keeping any of it must copy it.

struct RunPlan;

struct ObservationContext {
    const StateView& state;

    const std::string& anchor;  // label of the anchor that fired
    int instruction_index;      // position reached in the circuit; -1 at start
    int shot;                   // 0-based; 0 for a single trajectory
    int n_shots;                // shots requested; 1 for a single trajectory

    // The classical register as it stands at this point, one entry per clbit.
    const std::vector<int>& clbits;

    // The policy in force, so an observer performing its own conversion
    // answers to the same knobs the library does.
    const RunPlan& plan;

    // The run's label-keyed bundle, or null when the backend keeps none. An
    // observer given a label writes here as well as into its own storage,
    // which is how one observer serves both roads.
    ObservationBundle* bundle;
};

// -----------------------------------------------------------------------------
// Observer - the mechanism
// -----------------------------------------------------------------------------
// Implement this to read anything at all from a running simulation, including
// quantities the library does not ship. The built-in observers in
// observers.hpp are ordinary implementations of this interface with no
// privileged access, which is the property that keeps a user-written observer
// a first-class citizen rather than an escape hatch.
//
// Threading: an observer is invoked from the thread that called run(), in shot
// order and then instruction order, and never concurrently by that run. No
// backend parallelises its shot loop, so an implementation may hold mutable
// state without locking. What is NOT covered is one plan handed to several
// run() calls at once: the observers are then shared across those threads and
// the caller owns the synchronisation, since the plan belongs to the caller.
//
// An observer must not attach observers or mutate the plan from inside a call.

class Observer {
public:
    virtual ~Observer() = default;

    // Called once before the first shot. Sized buffers belong here rather than
    // in the first observe() call, which would otherwise pay for the check on
    // every shot.
    virtual void begin_run(int n_qubits, int n_shots);

    // Called once per firing of every anchor this observer is attached to.
    virtual void observe(const ObservationContext& ctx) = 0;

    // Called once after the last shot. Running aggregates are finalised here.
    virtual void end_run();
};

using ObserverPtr = std::shared_ptr<Observer>;

// -----------------------------------------------------------------------------
// Anchor - a point in the run, named without touching the circuit
// -----------------------------------------------------------------------------
// Anchors are resolved against the circuit actually being run, at the
// pre-flight, before any simulation happens. An anchor that resolves to
// nothing is a failed run that names the anchor, never a silent absence of
// data: an observation that did not happen is indistinguishable from one that
// happened and found nothing, unless the library says so.

class Anchor {
public:
    enum class Kind {
        Start,             // before the first instruction
        End,               // after the last instruction
        InstructionIndex,  // after instruction `index`
        InstructionLabel,  // after every instruction carrying `label`
        EveryInstruction,  // after each instruction; this is trace mode
        EveryLayer,        // at each scheduling layer boundary
        BeforeMeasurement, // before each MEASURE
        AfterMeasurement,  // after each MEASURE
        Predicate          // after each instruction satisfying `predicate`
    };

    using PredicateFn = std::function<bool(const Instruction&, int index)>;

    static Anchor at_start();
    static Anchor at_end();
    static Anchor after_instruction(int index);
    static Anchor after_label(std::string label);
    static Anchor every_instruction();
    static Anchor every_layer();
    static Anchor before_each_measurement();
    static Anchor after_each_measurement();
    static Anchor where(PredicateFn predicate);

    Kind kind() const { return kind_; }
    int index() const { return index_; }
    const std::string& label() const { return label_; }
    const PredicateFn& predicate() const { return predicate_; }

    // How this anchor reports itself in an ObservationContext and in the
    // bundle, e.g. "after_label(qft_done)" or "every_instruction".
    std::string name() const;

private:
    Anchor() = default;

    Kind kind_ = Kind::End;
    int index_ = -1;
    std::string label_;
    PredicateFn predicate_;
};

// -----------------------------------------------------------------------------
// ObservationBundle - label-keyed results, for callers who want data
// -----------------------------------------------------------------------------
// The shape most callers want, and the shape that crosses a language boundary,
// which a C++ observer object does not. Written into by BundleObserver and
// carried on every backend's Result.
//
// Amplitude payloads are Complex128, which is what Statevector::amplitudes()
// hands back, so an amplitude read costs no repacking.

class ObservationBundle {
public:
    struct StatePayload {
        StateForm form;
        std::shared_ptr<const void> state;
    };

    using Payload = std::variant<double,
                                 std::vector<double>,
                                 std::vector<Complex128>,
                                 std::vector<int>,
                                 std::string,
                                 StatePayload>;

    bool contains(const std::string& label) const;
    std::size_t size() const { return entries_.size(); }
    std::vector<std::string> labels() const;  // sorted, so output is stable

    // Typed reads. Each throws std::invalid_argument when the label is absent
    // or holds a different kind. A string-keyed store cannot fail at compile
    // time, so it fails loudly at the point of the mistake instead.
    double number(const std::string& label) const;
    const std::vector<double>& reals(const std::string& label) const;
    const std::vector<Complex128>& amplitudes(const std::string& label) const;
    const std::vector<int>& integers(const std::string& label) const;
    const std::string& text(const std::string& label) const;

    // State payloads. form() says which of the four accessors is valid.
    StateForm form(const std::string& label) const;
    const Statevector& statevector(const std::string& label) const;
    const DensityMatrix& density_matrix(const std::string& label) const;
    const StabilizerState& stabilizer(const std::string& label) const;
    const MPSState& mps(const std::string& label) const;

    // Insertion. A repeated label throws rather than overwriting: two
    // observations under one name means one of them is unreachable, and the
    // caller cannot tell which.
    void put(std::string label, Payload payload);

private:
    std::unordered_map<std::string, Payload> entries_;
};

// -----------------------------------------------------------------------------
// InitialState - where the run starts
// -----------------------------------------------------------------------------
// The whole of the write side. A mid-circuit injection is deliberately absent:
// it is not physics, so it does not belong in a circuit, and every use for it
// is already expressible. Starting elsewhere is this. Resuming from a
// checkpoint is running the rest of the circuit with the checkpoint as its
// initial state. Preparing a state as part of the computation is a preparation
// circuit, which IS physics and stays in the circuit.
//
// A supplied state in a form the backend does not hold answers to the same
// three knobs a read does.

class InitialState {
public:
    InitialState() = default;  // |0...0>

    static InitialState basis(std::uint64_t index);
    static InitialState from(std::shared_ptr<const Statevector> sv);
    static InitialState from(std::shared_ptr<const DensityMatrix> dm);
    static InitialState from(std::shared_ptr<const StabilizerState> st);
    static InitialState from(std::shared_ptr<const MPSState> mps);

    bool is_default() const { return kind_ == Kind::Zero; }
    bool is_basis() const { return kind_ == Kind::Basis; }
    std::uint64_t basis_index() const { return basis_; }

    // Valid only when neither is_default() nor is_basis().
    StateForm form() const;
    const void* state() const { return state_.get(); }

private:
    enum class Kind { Zero, Basis, State };

    Kind kind_ = Kind::Zero;
    std::uint64_t basis_ = 0;
    StateForm form_ = StateForm::Statevector;
    std::shared_ptr<const void> state_;
};

// -----------------------------------------------------------------------------
// ObservationPlan - anchors and the observers attached to them
// -----------------------------------------------------------------------------

class ObservationPlan {
public:
    struct Attachment {
        Anchor anchor;
        ObserverPtr observer;
    };

    // Ownership is shared and only shared. A plan holding a reference to a
    // stack object outlives it as easily as not, and the resulting crash
    // surfaces far from its cause.
    ObservationPlan& observe(Anchor anchor, ObserverPtr observer);

    bool empty() const { return attachments_.empty(); }
    const std::vector<Attachment>& attachments() const { return attachments_; }

private:
    std::vector<Attachment> attachments_;
};

// -----------------------------------------------------------------------------
// RunPlan - the harness a simulator runs the circuit inside
// -----------------------------------------------------------------------------
// One trailing parameter on every simulator's run(). Default constructed it
// starts at |0...0> and watches nothing, which is what a caller who passes
// nothing gets, and it is the zero-overhead path: empty() short circuits every
// anchor check in the shot loop.

struct RunPlan {
    struct Options {
        // Gate fusion rewrites instructions into blocks, which renumbers the
        // positions and drops the labels anchors are named by. Suppress turns
        // fusion off for an observed run, so every anchor keeps meaning what it
        // said, at the cost of the throughput fusion would have bought.
        //
        // Keep asks for fusion anyway, and then anchors resolve against the
        // FUSED circuit, because that is what actually executes. A position
        // anchor therefore indexes blocks rather than the caller's
        // instructions, and a label fusion absorbed no longer resolves at all,
        // which fails the run rather than silently never firing.
        enum class Fusion { Suppress, Keep };

        Conversion conversion = Conversion::Convert;
        Cost cost = Cost::Guarded;
        Response response = Response::Throw;
        Fusion fusion = Fusion::Suppress;

        // Guarded refuses an allocation exceeding this multiple of the live
        // state's own footprint. One means: handing back a copy of what the
        // backend already holds is fine, and anything that costs more than the
        // simulation itself has to be asked for. The criterion is a ratio
        // rather than a byte count so that it needs no tuning and no knowledge
        // of the machine.
        double guard_multiple = 1.0;
    };

    Options options;
    InitialState initial;
    ObservationPlan observations;

    bool empty() const { return initial.is_default() && observations.empty(); }
};

// -----------------------------------------------------------------------------
// detail - what the backends drive
// -----------------------------------------------------------------------------
// Not part of the caller-facing surface. A backend constructs one runner per
// run(), which resolves every anchor against the circuit before simulating and
// then fires whatever matches at each point.

namespace detail {

// Deliver a refusal the way the response knob says. Returns false always, so a
// caller can `return refuse_observation(...)` and read as declining.
bool refuse_observation(const RunPlan::Options& options, const std::string& message);

// Produce `target` from what the view holds, judged by the knobs. Returns null
// when the observation is to be omitted, and throws under Response::Throw.
// `what` names the requester so the message says whose request was refused.
std::shared_ptr<const void> produce_state(const StateView& view, StateForm target,
                                          const RunPlan::Options& options,
                                          const std::string& what);

// Charge an allocation an observer is about to make against the guard, the
// same comparison produce_state applies to a conversion. Returns false when the
// observation is to be omitted, having already delivered the refusal.
bool charge_allocation(const StateView& view, std::size_t bytes,
                       const RunPlan::Options& options, const std::string& what);

// Seed a backend's state from the plan. The default plan initialises to
// |0...0>, which is what a caller who passed no plan gets, so a backend calls
// this unconditionally in place of its own initialise.
//
// A supplied state answers to the conversion knob, but never to the response
// knob: Warn and Ignore omit an OBSERVATION, and there is no such thing as
// omitting the state a run starts from. The alternative to the state the caller
// asked for is silently simulating a different circuit, so this throws.
void apply_initial_state(const RunPlan& plan, Statevector& sv);
void apply_initial_state(const RunPlan& plan, DensityMatrix& dm);
void apply_initial_state(const RunPlan& plan, StabilizerState& state);
void apply_initial_state(const RunPlan& plan, MPSState& mps);

// Fires an instruction's anchors when the enclosing scope ends, however it
// ends. Backends whose instruction loop skips work with `continue` (a barrier,
// a condition that does not hold) would otherwise fire on some iterations and
// not others, and an anchor that silently does not fire is indistinguishable
// from one that fired and found nothing.
//
// Nothing is fired while an exception is propagating: the run is already
// failing, the state is not one anybody asked to see, and a throw from a
// destructor during unwinding terminates the process.
class ObservationRunner;

class FiringGuard {
public:
    FiringGuard(ObservationRunner* runner, int index, const Instruction& inst,
                const StateView& state);
    ~FiringGuard();

    FiringGuard(const FiringGuard&) = delete;
    FiringGuard& operator=(const FiringGuard&) = delete;

private:
    ObservationRunner* runner_;
    int index_;
    const Instruction* inst_;
    const StateView* state_;
    int uncaught_;
};

class ObservationRunner {
public:
    // Resolves the plan against this circuit. An anchor that cannot fire is a
    // failed run raised here, not a silent absence of data later.
    ObservationRunner(const RunPlan& plan, const QuantumCircuit& circuit);

    // False when nothing is attached, which is the check that keeps an
    // unobserved run free of every per-instruction test below.
    bool active() const { return active_; }

    void begin_run(int n_qubits, int n_shots);
    void end_run();

    // The bundle observers with a label write into. Set once, before the run,
    // to the one living on the backend's Result.
    void set_bundle(ObservationBundle* bundle) { bundle_ = bundle; }

    // `clbits` must stay alive for the shot; the runner holds it by pointer so
    // an observer reads the register as it stands rather than a copy.
    void begin_shot(int shot, const std::vector<int>& clbits);

    void at_start(const StateView& state);
    void at_end(const StateView& state, int last_index);
    void before_instruction(int index, const Instruction& inst, const StateView& state);
    void after_instruction(int index, const Instruction& inst, const StateView& state);

private:
    using Group = std::vector<const ObservationPlan::Attachment*>;

    void compute_layer_boundaries(const QuantumCircuit& circuit);
    void fire(const Group& group, const StateView& state, int instruction_index);

    const RunPlan& plan_;
    bool active_ = false;

    int shot_ = 0;
    int n_shots_ = 1;
    const std::vector<int>* clbits_ = nullptr;
    ObservationBundle* bundle_ = nullptr;

    Group start_;
    Group end_;
    Group every_;
    Group layer_;
    Group before_measure_;
    Group after_measure_;
    Group predicate_;

    std::unordered_map<int, Group> indexed_;
    std::unordered_map<std::string, Group> labelled_;
    std::unordered_set<int> layer_end_;
};

}  // namespace detail

}  // namespace lindblad
