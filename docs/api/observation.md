# Observation and the run harness

## What this API is for

A `QuantumCircuit` describes physics and holds nothing else. Watching a
simulation is not physics: the run is identical whether or not anyone looks.
Choosing the state a run starts from is not physics either, since nothing in
nature overwrites a state with one handed to it.

Both are properties of how a simulation is being run, so both live in a
`RunPlan`, which every simulator accepts as a trailing argument. The circuit
stays a pure description of the computation, and the same circuit can be run
watched or unwatched without being rebuilt.

A `RunPlan` answers three questions:

- where the run starts, through `InitialState`
- what is watched while it runs, through `ObservationPlan`
- what happens when something is asked for that the backend cannot hand over,
  through `RunPlan::Options`

There is no default observation set. An empty plan starts at the all-zero state
and watches nothing, which is what a caller who passes no plan gets.

## Header to include

```cpp
#include "lindblad/observation.hpp"   // the harness: RunPlan and its parts
#include "lindblad/observers.hpp"     // the built-in observers
```

The backend headers include the first of these already, so a file that includes
a simulator has `RunPlan` available. Include `observers.hpp` when attaching
built-in observers, and `observation.hpp` directly when writing your own.

## Namespace

```cpp
namespace lindblad;
```

Internal pieces the backends drive live in `lindblad::detail` and are not part
of the caller-facing surface.

## Class definitions

```cpp
struct RunPlan {
    struct Options { /* see below */ };

    Options          options;
    InitialState     initial;        // default: the all-zero state
    ObservationPlan  observations;   // default: empty, watches nothing

    bool empty() const;              // true when neither is set
};
```

```cpp
class ObservationPlan {
public:
    ObservationPlan& observe(Anchor anchor, ObserverPtr observer);
    bool empty() const;
};

using ObserverPtr = std::shared_ptr<Observer>;
```

Ownership is shared and only shared. A plan that held a reference to a stack
object would outlive it as easily as not, and the resulting crash surfaces a
long way from its cause.

```cpp
class Observer {
public:
    virtual ~Observer() = default;
    virtual void begin_run(int n_qubits, int n_shots);
    virtual void observe(const ObservationContext& ctx) = 0;
    virtual void end_run();
};
```

The built-in observers are ordinary implementations of this interface with no
privileged access. An observer you write sits beside them rather than beneath
them.

```cpp
struct ObservationContext {
    const StateView&        state;
    const std::string&      anchor;             // which anchor fired
    int                     instruction_index;  // -1 at the start
    int                     shot;               // 0-based
    int                     n_shots;
    const std::vector<int>& clbits;             // the register as it stands
    const RunPlan&          plan;
    ObservationBundle*      bundle;             // may be null
};
```

Nothing in the context is owned by the observer and nothing survives the call,
so an observer keeping any of it must copy it.

## `RunPlan::Options`

Four independent knobs. They are separate rather than fused into one enum
because fusing them makes policies unsayable: an enumerator meaning "convert,
and throw when conversion is impossible" cannot also express "convert, and warn
when conversion is impossible".

| Field | Values | Default | Governs |
|---|---|---|---|
| `conversion` | `Convert`, `Never` | `Convert` | whether the library may translate into a representation the backend does not hold |
| `cost` | `Guarded`, `Unlimited` | `Guarded` | whether an expensive observation must be asked for explicitly |
| `response` | `Throw`, `Warn`, `Ignore` | `Throw` | what a refusal looks like, whatever caused it |
| `fusion` | `Suppress`, `Keep` | `Suppress` | whether a watched run keeps gate fusion |
| `guard_multiple` | `double` | `1.0` | how much `Guarded` allows, as a multiple of the live state |

### How a refusal happens

An observation fails to produce for exactly three reasons, and all three are
delivered by `response`:

- the backend does not hold that data and `Never` is selected
- no conversion exists at all
- the conversion exists but exceeds the guard

`Warn` and `Ignore` both omit the observation and differ only in whether the
warning channel hears about it. Under `Throw` the run stops.

### What the guard measures

`Guarded` refuses an allocation exceeding `guard_multiple` times the footprint
of the live state the backend already holds. The yardstick is the running
simulation rather than a configured byte count, so it scales with the problem
and the machine without tuning.

That catches the cases worth catching: a tableau expanded to `2^n` amplitudes
against a tableau costing `O(n^2)`, a statevector squared into a `4^n` density
matrix, or a full-state copy taken once per shot. A single copy of the state
passes, which is correct: copying what the backend already holds is not
expensive relative to holding it.

`Unlimited` means exactly that. Size never refuses, only impossibility does, and
a conversion taken under it reports its cost through the warning channel.

### What is actually impossible

No option makes these available:

- a stabilizer tableau from any other representation, since a general state is
  not a stabilizer state at all
- a statevector from a density matrix, since a mixed state has none
- a matrix product state as an OBSERVATION from another representation, because
  a factorisation needs a bond cap and an observation has none to use. The
  initial-state path does factorise a supplied statevector, using that run's own
  bond cap, which is the difference between the two directions

### Fusion

Gate fusion rewrites instructions into blocks, which renumbers positions and
drops the labels anchors are named by. `Suppress` therefore turns fusion off for
a watched run: every anchor keeps meaning what it said, and the run gives up the
throughput fusion would have bought.

`Keep` asks for fusion anyway. Anchors then resolve against the fused circuit,
because that is what executes: a position anchor indexes blocks rather than your
instructions, and a label that fusion absorbed no longer resolves, which fails
the run rather than silently never firing.

## `InitialState`

```cpp
InitialState();                                              // the all-zero state
static InitialState basis(std::uint64_t index);              // one basis state
static InitialState from(std::shared_ptr<const Statevector>);
static InitialState from(std::shared_ptr<const DensityMatrix>);
static InitialState from(std::shared_ptr<const StabilizerState>);
static InitialState from(std::shared_ptr<const MPSState>);
```

A supplied state is converted to the backend's own representation when it
arrives in another, subject to `conversion` and the guard.

It never answers to `response`. `Warn` and `Ignore` omit an observation, and
there is no coherent way to omit the state a run starts from: the alternative to
the state you asked for is silently simulating a different circuit. A supplied
state that cannot be produced throws under every setting.

A state handed to the MPS backend is factorised at that run's bond cap, so one
needing more bond dimension than the cap allows is truncated rather than
refused. That is what running it at that cap means, and the discarded weight is
what `truncation_error()` reports.

## Anchors

An anchor names a point in the run without the circuit carrying anything.

```cpp
Anchor::at_start();                  // before the first instruction
Anchor::at_end();                    // after the last
Anchor::after_instruction(int k);    // after instruction k
Anchor::after_label(std::string s);  // after every instruction carrying label s
Anchor::every_instruction();         // after each one: trace mode
Anchor::every_layer();               // at each layer boundary
Anchor::before_each_measurement();
Anchor::after_each_measurement();
Anchor::where(PredicateFn pred);     // after each instruction satisfying pred
```

Layer boundaries are computed by greedy layering over qubit occupancy: an
instruction opens a new layer when one of its qubits is already used in the
current one, and the boundary falls on the last instruction of each layer.

### Resolution

Anchors are resolved against the circuit being run, before any state is touched.
An anchor that cannot fire fails the run and names itself: an index past the end
of the circuit, or a label no instruction carries, including one a transpiler
pass removed along with the instruction that held it.

This is deliberate and is not governed by `response`. A plan that does not match
its circuit is a mistake in the calling code, not a capability the backend
lacks, and an anchor that silently never fires is indistinguishable from one
that fired and found nothing.

### When anchors fire

An instruction that the backend skips still fires its anchors. A barrier, or a
conditioned gate whose condition does not hold, is a point the run reached, and
a caller watching every instruction wants the state there whether or not it
changed.

On execution paths where one evolution serves every shot, the observers fire
once, because that single evolution describes all of the shots. This applies to
terminal-measurement circuits, where nothing before the measurements is
stochastic. On per-shot trajectory paths, where each shot diverges at its first
collapse, the observers fire once per shot.

## Observers

Every built-in observer keeps its own results, so the typed road is to construct
one, attach it, and read it afterwards. Giving one a label additionally writes
its results into the run's `ObservationBundle`.

| Observer | Reads | Notes |
|---|---|---|
| `StateObserver` | the state itself | native form by default; naming a form asks for a conversion |
| `ProbabilityObserver` | the full outcome distribution | `average()` across firings |
| `AmplitudeObserver` | named amplitudes | cost is the number of indices, not `2^n` |
| `ExpectationObserver` | expectation of a `SparsePauliOp` | `mean()` and `variance()` across firings |
| `PurityObserver` | `Tr(rho^2)` | exactly 1 on the three pure backends |
| `EntropyObserver` | entropy across a cut, in bits | see below |
| `ClassicalRegisterObserver` | the classical bits mid-run | what a feedforward circuit decided |
| `BondDimensionObserver` | MPS bond dimensions | refuses on other backends |
| `TruncationObserver` | MPS discarded weight | refuses on other backends |
| `CallbackObserver` | whatever you compute | for a one-off; anything durable is better as its own class |

Every observer records where each firing happened:

```cpp
struct FiringPoint {
    int         shot;
    int         instruction;   // -1 at the start
    std::string anchor;
};
```

`count()` gives the number of firings and `point(k)` gives the coordinates of
the k-th, which is the same k the observer's own results are indexed by. An
observer can be attached to several anchors, so a bare sequence of readings
would otherwise be ambiguous.

### Entropy

`EntropyObserver(region, renyi_order = 1.0, label = {})` reports the von Neumann
entropy of the reduced state on `region`, in bits. `renyi_order` selects a
Renyi entropy instead, where 1 is von Neumann and 2 is the negative log of the
reduced purity.

On the statevector, MPS and Clifford backends the global state is pure, so this
IS the entanglement entropy across the cut. On the density matrix it is the
entropy of the reduced state, which for a mixed global state mixes classical
correlation with entanglement and is a different quantity. `is_entanglement(k)`
reports which of the two was handed back, rather than leaving it to be inferred
from the backend.

The method differs per backend by more than a constant:

| Backend | Method | Cost |
|---|---|---|
| Clifford | GF(2) rank of the generators restricted to the region | `O(n * |A|^2)` bit operations, no amplitudes |
| MPS | spectrum of the two environment Gram matrices | `O(n * chi^3)` |
| statevector | reduced matrix of the smaller side, then an eigensolve | proportional to `2^n` times the smaller side |
| density matrix | partial trace, then an eigensolve | proportional to the region and its complement |

The Clifford answer is exact and lands on an integer, because a stabilizer
state's reduced state is maximally mixed on its support: the spectrum is flat,
so every Renyi order agrees with the von Neumann value.

The MPS route reads the bond directly only when the cut splits the chain in one
place, meaning the region is a prefix or a suffix. Any other bipartition has no
single bond to read and falls back to dense amplitudes, charged through the
knobs like every other conversion.

## `ObservationBundle`

The label-keyed results, carried on every backend's `Result` as
`result.observations`. It is what an observer with a label writes into, and the
shape that survives being copied, stored, or handed across a language boundary.

```cpp
bool contains(const std::string& label) const;
std::vector<std::string> labels() const;         // sorted

double                        number(const std::string& label) const;
const std::vector<double>&    reals(const std::string& label) const;
const std::vector<Complex128>& amplitudes(const std::string& label) const;
const std::vector<int>&       integers(const std::string& label) const;
const std::string&            text(const std::string& label) const;

StateForm                 form(const std::string& label) const;
const Statevector&        statevector(const std::string& label) const;
const DensityMatrix&      density_matrix(const std::string& label) const;
const StabilizerState&    stabilizer(const std::string& label) const;
const MPSState&           mps(const std::string& label) const;
```

An observer that fired exactly once writes under its plain label. One that fired
repeatedly writes each firing under `label@<instruction>#<shot>`, so no firing
overwrites another.

## Exceptions and preconditions

- An anchor that does not resolve throws `std::invalid_argument`, whatever
  `response` says.
- `Anchor::after_label` rejects an empty label, and `Anchor::where` rejects an
  empty predicate.
- A supplied initial state that cannot be produced throws, whatever `response`
  says.
- Reading a bundle label that is absent, or holds a different kind than the
  accessor asks for, throws `std::invalid_argument`.
- Writing a label the bundle already holds throws: two observations under one
  name leave one unreachable and the caller cannot tell which.
- `StateView`'s typed accessors throw when the backend holds a different
  representation. They never convert, so reading through one always costs a
  pointer dereference and a conversion is always explicit.
- `EntropyObserver` rejects a region naming a qubit twice, a qubit outside the
  register, or every qubit in it, since a cut needs a side to be entangled with.
- Reading a firing index at or beyond `count()` throws.

Observers are invoked from the thread that called `run()`, in shot order and
then instruction order, and never concurrently by that run, so an
implementation may hold mutable state without locking. One plan handed to
several concurrent `run()` calls shares its observers across those threads, and
the synchronisation is then the caller's, since the plan is the caller's.

## Example usage

Watching a state partway through, and an observable at the end:

```cpp
#include "lindblad/observation.hpp"
#include "lindblad/observers.hpp"
#include "lindblad/simulators/statevector_sim.hpp"

using namespace lindblad;

QuantumCircuit qc(3);
qc.h(0);
qc.cx(0, 1);
qc.barrier();          // label it so an anchor can name the point
qc.instructions.back().label = "after_entangling";
qc.cx(1, 2);

auto mid    = std::make_shared<StateObserver>();
auto energy = std::make_shared<ExpectationObserver>(hamiltonian);

RunPlan plan;
plan.observations.observe(Anchor::after_label("after_entangling"), mid);
plan.observations.observe(Anchor::at_end(), energy);

StatevectorSimulator sim;
auto result = sim.run(qc, 1024, 42, plan);

const Statevector& psi = mid->statevector();
double value = energy->mean();
```

The same run through the bundle instead, by giving the observers labels:

```cpp
auto mid = std::make_shared<StateObserver>("midpoint");
plan.observations.observe(Anchor::after_label("after_entangling"), mid);

auto result = sim.run(qc, 1024, 42, plan);
const Statevector& psi = result.observations.statevector("midpoint");
```

Starting somewhere other than the all-zero state:

```cpp
RunPlan plan;
plan.initial = InitialState::basis(5);   // |101> on three qubits
```

Entanglement across a cut, on every step of the circuit:

```cpp
auto entropy = std::make_shared<EntropyObserver>(std::vector<int>{0, 1});
plan.observations.observe(Anchor::every_instruction(), entropy);

auto result = sim.run(qc, 0, 42, plan);
for (std::size_t k = 0; k < entropy->count(); ++k) {
    const FiringPoint& where = entropy->point(k);
    // where.instruction, where.shot, entropy->value(k)
}
```

Asking a Clifford run for amplitudes, which is a conversion and is guarded:

```cpp
RunPlan plan;
plan.options.cost = Cost::Unlimited;   // a tableau expands to 2^n amplitudes
auto dense = std::make_shared<StateObserver>(StateForm::Statevector);
plan.observations.observe(Anchor::at_end(), dense);
```

## Related pages

- [Simulators](simulators.md) for the backends that accept a `RunPlan`
- [Statevector](statevector.md), [Operators](operators.md) for the types
  observers hand back
- [Validation](validation.md) for the separate policy governing physical
  validity of caller-supplied operators
