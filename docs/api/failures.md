# Failures

## What the API is for

Every `run()` of `StatevectorSimulator`, `DensityMatrixSimulator`,
`MPSSimulator`, `CliffordSimulator` and `LocalBackend` either returns a result
that is an answer or throws. A run that fails never returns a `Result`: on every
result a run does return, `success` is `true` and `error_message` is empty.

Two rules decide what a failure costs:

- Everything the circuit, the options and the run plan decide on their own is
  checked before the first gate. A failure there throws before any state is
  touched, so nothing has been computed and nothing is lost.
- A failure after the first gate throws too, and what the run had computed is
  kept: in this thread's slot, where `take_failed_run()` hands it back, and by
  default in a folder on disk that the exception names, which
  `load_failed_run()` reads back.

This page covers the exception types, every failure a run can meet, every knob
that governs one, and the failed-run record.

## Header to include

```cpp
#include "lindblad/errors.hpp"      // Error, the four exception types, FailurePoint, RunPhase
#include "lindblad/failed_run.hpp"  // FailedRun, take_failed_run, load_failed_run
```

## Namespace

Everything on this page is in `lindblad`.

## Types

### FailurePoint

```cpp
struct FailurePoint {
    int shot = -1;            // -1: not inside a shot
    int instruction = -1;     // -1: not at an instruction
    std::string gate;         // empty when instruction == -1
    std::vector<int> qubits;  // the instruction's operands
};
```

Where a failure happened, when it belongs to one instruction or one shot. A
refusal before the run names no shot; a failure while sampling, after the
instructions have run, names no instruction.

### The exception types

```cpp
class Error {
public:
    const std::string& entry_point() const noexcept;
    const std::optional<FailurePoint>& where() const noexcept;
    const std::optional<std::filesystem::path>& saved_to() const noexcept;
};

class InvalidArgument : public std::invalid_argument, public Error;
class OutOfRange      : public std::out_of_range,      public Error;
class RuntimeFailure  : public std::runtime_error,     public Error;
class InternalError   : public std::logic_error,       public Error;
```

| Type | Also a | Means |
|---|---|---|
| `InvalidArgument` | `std::invalid_argument` | a caller mistake or an exceeded limit, known before the first gate, or a refusal the `response` knob turned into a throw |
| `OutOfRange` | `std::out_of_range` | an index outside a register or a range |
| `RuntimeFailure` | `std::runtime_error` | a failure met while running that makes the answer wrong |
| `InternalError` | `std::logic_error` | a defect in Lindblad; the message asks for a report |

- Each type is also its standard counterpart, so an existing
  `catch (const std::invalid_argument&)` keeps matching.
  `catch (const lindblad::Error&)` catches everything Lindblad raises itself.
- `Error` is not itself derived from `std::exception`. Each concrete type
  already is, once, through its standard base, and a second copy would make
  `catch (const std::exception&)` ambiguous, so it would stop matching.
- Your own exceptions (from an observer you wrote) and `std::bad_alloc` are
  never wrapped: they reach you as the same object, of the same type.
- A message starts with the entry point you called
  (`StatevectorSimulator::run: ...`). When one instruction is at fault it ends
  with `(instruction 4: cx on qubits 0, 1)`, and with `at shot 7` when the
  failure happened inside a shot.
- An `InternalError` message ends by asking you to report it on the issue
  tracker with the message.
- When the run's partial results were saved, the message ends with
  `Partial results saved to <folder>.` and `saved_to()` returns the folder.

### RunPhase

```cpp
enum class RunPhase { BeforeFirstGate, MidRun };
```

Whether stopping now would lose any work. `BeforeFirstGate` holds until the
first instruction of the first shot has executed; everything after is `MidRun`,
including an anchor at the start of any later shot. `Response::Auto` decides by
it (see [Failures that leave the answer intact](#failures-that-leave-the-answer-intact)).
Observers read it from `ObservationContext::phase`.

### FailedRun

```cpp
struct FailedRun {
    using State = std::variant<std::monostate, Statevector, DensityMatrix,
                               MPSState, StabilizerState>;

    std::string library_version;
    std::string entry_point;          // e.g. "StatevectorSimulator::run"
    std::string backend;              // statevector | density_matrix | mps | clifford
    int n_qubits = 0;
    int shots_requested = 0;
    int shots_completed = 0;          // shots whose counts are in `counts`
    std::uint64_t seed = 0;           // the seed actually used, also when 0 was passed
    std::optional<FailurePoint> where;
    std::string exception_type;       // the dynamic type, demangled where possible
    std::string message;              // what(), as thrown
    std::vector<std::pair<std::string, std::string>> options;  // name, JSON value

    std::unordered_map<std::string, int> counts;   // finished shots only
    ObservationBundle observations;   // every observer flushed as the run failed
    State state;                      // moved out of the run, never copied
    std::optional<QuantumCircuit> circuit;         // the circuit, as passed
    std::optional<NoiseModel> noise_model;         // density-matrix runs

    std::optional<std::filesystem::path> saved_to;
    std::string save_note;            // what was not saved and why; empty when all was
};
```

- `state` is the state the run was evolving when it failed: the working buffer
  of a statevector run, the density matrix, the MPS chain of the shot in
  progress, or the tableau. A Clifford run that fails inside its bit-sliced gate
  pass keeps no state, since that layout is not a `StabilizerState`.
- `options` holds every field of the backend's options and of
  `RunPlan::Options`, each as a JSON value (`"max_memory_mb"`, `"4096"`;
  `"plan.response"`, `"\"Auto\""`).
- `where` is what the exception named, else the shot and instruction the run
  had reached.

## Constructors

Lindblad constructs the exception types itself; a caller catches them. Each
concrete type takes the full message, the entry point and, when there is one,
where the failure happened:

```cpp
InvalidArgument(const std::string& message, std::string entry_point,
                std::optional<FailurePoint> where = {});
OutOfRange(const std::string& message, std::string entry_point,
           std::optional<FailurePoint> where = {});
RuntimeFailure(const std::string& message, std::string entry_point,
               std::optional<FailurePoint> where = {});
InternalError(const std::string& message, std::string entry_point,
              std::optional<FailurePoint> where = {});
```

`Error` has no public constructor. `FailedRun` is a plain aggregate, filled by
a run that fails or by `load_failed_run()`.

## Knobs and their defaults

| Knob | Set on | Default |
|---|---|---|
| `response` | `RunPlan::Options` | `Response::Auto` |
| `ValidationOptions` | each instruction or call | `Validation::Throw` |
| `svd_rescue` | `MPSSimulator`, `MPSState` | `true` |
| `qubit_limit` | `StatevectorSimulator::Options`, `MPSSimulator`, `LocalBackend::Config` | `QubitLimit::Enforce` |
| `max_memory_mb` | the statevector, density-matrix and Clifford `Options`, `MPSSimulator`, `LocalBackend::Config` | `0` (automatic) |
| `save_failed_runs` | `RunPlan::Options` | `SaveFailedRuns::Save` |
| `failed_run_dir` | `RunPlan::Options` | empty (the default folder) |

### QubitLimit

```cpp
enum class QubitLimit { Enforce, Lift };
```

- `Enforce` keeps the default ceilings: a statevector of 30 qubits (the
  `Statevector` constructor, `StatevectorSimulator`, and
  `LocalBackend::max_qubits()`), and an MPS dense fallback of 25 qubits (a gate
  over three or more qubits, `MCX` with more than two controls, `MCP`,
  `PERMUTATION`, and `MPSState::to_statevector`).
- `Lift` raises the statevector ceiling to 59, the widest state whose byte
  count, $16 \cdot 2^{59}$, still fits 64 bits, and the MPS dense fallback to
  31. The MPS ceiling is lower because the fallback rebuilds the chain by
  factorising a $2 \times 2^{n-1}$ block, and the factorisation takes its
  dimensions as `int`.
- Nothing raises either ceiling further, and lifting never skips the memory
  check.
- `Statevector(n, QubitLimit)` and `StabilizerState::to_statevector(QubitLimit)`
  take the limit as an argument, defaulting to `Enforce`.

A register over the limit is refused before the first gate, and the message says
which setting lifts it.

### The memory cap

`max_memory_mb` is the most memory a run may use, in MiB ($2^{20}$ bytes):

- a positive value is that many MiB;
- `NO_MEMORY_CAP` means no cap (a value only a caller passes);
- `0`, the default, is automatic: the memory the machine reports available,
  read at most once a second, or `FALLBACK_MEMORY_CAP_MB` (4096 MiB) when the
  machine gives no coherent reading. On Linux that reading is `MemAvailable`,
  and it counts only when it parses as a number, is not zero, converts to bytes
  without overflow, and is no larger than `MemTotal`.

Before the first gate, a run whose fixed buffers exceed the cap is refused with
`InvalidArgument`, naming what it needs, the cap, and where the cap came from:

- the statevector simulator counts two states, $2 \cdot 16 \cdot 2^n$ bytes;
- the density-matrix simulator counts one matrix, $16 \cdot 4^n$ bytes (a prefix
  snapshot is judged on its own, and taken only when it fits);
- the Clifford simulator counts three tableau-sized buffers;
- the MPS simulator has no fixed footprint to refuse up front.

While the run goes on, a run budget checks every further allocation before it is
made: an MPS chain's two-site updates, its dense fallbacks and dense sampling,
every conversion an observer asks for, and the copies a `StateObserver` keeps to
the end of the run. Going over raises `RuntimeFailure` naming the allocation,
its size, the peak it would reach and the cap. The budget counts what Lindblad
allocates, not the workspace a factorisation kernel below it uses.

### Why an out-of-memory kill cannot be caught

On Linux an allocation can succeed and the kernel's out-of-memory killer can end
the process later, with `SIGKILL` and no exception; `systemd-oomd` can end a
whole group of processes under memory pressure. No program can catch either.
The checks above exist so a run never gets there: its fixed footprint is refused
before the first gate, and its growth is refused before it is allocated. What
stays out of reach is another process allocating during the run, since a reading
of available memory is not a reservation.

## Failed runs: take_failed_run and load_failed_run

### What is kept

A run that fails after its first instruction has executed leaves a `FailedRun`.
It holds the counts of the shots that finished, every observation the observers
had made (each observer's buffered firings are flushed as the run fails; one
whose `end_run` throws is named in `save_note`), the state being evolved, the
circuit, the noise model of a density-matrix run, the seed actually used, every
option, and where the run was.

### The slot

```cpp
std::optional<FailedRun> take_failed_run();
```

- One slot per thread. A newer failure on the thread replaces the record in it,
  and starting a run does not clear it.
- `take_failed_run()` moves the record out and leaves the slot empty.
- `Estimator::run_batch` runs its indices on worker threads; the lowest-indexed
  failure's record is moved into the calling thread's slot with its exception.
- After a successful save, the record in the slot keeps the scalar fields,
  `where`, `save_note` and `saved_to`. The parts written to disk (counts,
  observations, circuit, noise model, state) are released from memory, and
  `load_failed_run(saved_to)` reads them back. A part that could not be written
  stays in memory, and `save_note` says why.
- With `SaveFailedRuns::DoNotSave` the whole record stays in memory, and the next
  failure on the thread replaces it. That is the one case in which something is
  lost, and only because saving was switched off.

### The folder

Unless `RunPlan::Options::save_failed_runs` is `DoNotSave`, a failed run is saved
into a new folder under the first of:

1. `RunPlan::Options::failed_run_dir`, when set;
2. `$XDG_STATE_HOME/lindblad/failed-runs`, when `XDG_STATE_HOME` is an absolute
   path;
3. `$HOME/.local/state/lindblad/failed-runs`;
4. on Windows, `%LOCALAPPDATA%\lindblad\failed-runs`.

The temporary directory is never used, since on many systems it is memory.

The folder is named `<YYYYmmdd-HHMMSS>-<pid>-<thread>-<counter>`, so two
failures in the same second never share one. It is written as
`<name>.partial` and renamed when complete, so a folder without the suffix was
written whole; a save that cannot complete removes its partial folder. On POSIX
systems the folder is readable by its owner only (0700, files 0600).

| File | Contents |
|---|---|
| `manifest.json` | every scalar field, `where`, the options, `save_note`, the state's form, and each file's size and CRC-32C |
| `counts.json` | `{"shots_completed": N, "counts": {"0101": 12, ...}}`, keys in the counts convention (clbit 0 rightmost) |
| `circuit.json` | `QuantumCircuit::to_json()` |
| `noise_model.json` | `NoiseModel::to_json()`, for a density-matrix run |
| `observations.json` | every observation; numbers at 17 significant digits, NaN and infinities as the strings `"NaN"`, `"Infinity"`, `"-Infinity"` |
| `observations/<index>-<label>.bin` | one state file per observed state |
| `state.bin` | the state being evolved |

A state file is little-endian: an 8-byte magic `LBSTATE1`, the format version,
the form (0 statevector, 1 density matrix, 2 MPS, 3 stabilizer), the qubit
count, then the state's own storage, streamed from where the state holds it
without a copy. An MPS file also carries the chain's settings, its open span,
its fidelity figures and its counters, so a loaded chain is the chain that
failed.

Before the state, and before any observed states, the free space is checked. A
state that does not fit with 64 MiB to spare is not written; everything else is,
and `save_note` says `The state was not saved: it needs X MiB, Y MiB free; it is
in memory, take it with lindblad::take_failed_run().`

### Saved runs are never deleted

Lindblad never deletes a saved run. The folders accumulate until you delete
them, and each can hold a full state: 16 GiB for a 30-qubit statevector run. A
program that fails runs repeatedly, left unwatched, can fill the disk. Watch the
folder named above, point `failed_run_dir` at a folder you manage, or set
`save_failed_runs = SaveFailedRuns::DoNotSave` where keeping the record in
memory is enough.

### Loading a saved run

```cpp
FailedRun load_failed_run(const std::filesystem::path& folder);
```

Reads a saved folder back in full. Every file the manifest lists is checked for
its size and its CRC-32C before any of them is read, and a file that is missing,
resized, fails its checksum, is malformed, or is named outside the folder is
refused with `InvalidArgument` naming the file. The returned record's
`saved_to` is `folder`. A loaded state can seed a new run through
`RunPlan::initial` (`InitialState::from`).

## Exceptions: the failures a run can meet

Every failure a `run()` of the four simulators or `LocalBackend` can meet, grouped
by what it costs.

### Failures before the first gate

Always thrown, with no knob, before any state is touched. They leave no failed-run
record.

- A circuit with no qubits.
- A register over the qubit limit.
- An option out of range: `fusion_max_qubit` outside [2, 6], a negative
  `fusion_threshold`, a negative thread cap, a bond cap below 1.
- An operand or classical-bit index outside the register (`OutOfRange`).
- An unbound parameterised gate: a `PARAM_*` gate, or symbolic expressions
  `bind_parameters()` has not resolved.
- A gate naming the wrong number of qubits for its type, a qubit named twice,
  or fewer parameters than its type reads.
- A parameter that is NaN or infinite.
- A `UNITARY` whose matrix does not have $(2^k)^2$ entries for its $k$
  operands, and a `PERMUTATION` whose map is not a bijection of $[0, 2^k)$.
- A gate the backend cannot apply: on MPS, a gate its dense fallback would need
  at a width over the chain's limit; on Clifford, a gate with no tableau form or
  a rotation that is not a multiple of $\pi/2$.
- A noise channel whose width does not match the gate it is attached to.
- A fixed memory footprint over the cap.
- A starting state that cannot be produced.
- A run plan that does not match the circuit: an anchor that resolves to
  nothing, an absent label, two observers claiming one label.
- An observer asking for something that does not exist: an amplitude index
  outside the register, a malformed entropy region, an observable of the wrong
  width, with no terms, or not Hermitian.

Each is an `InvalidArgument` (an `OutOfRange` for an index). Physical checks on
what you hand in (a matrix that is not unitary, a Kraus set that is not trace
preserving, a state that is not normalised) are governed by
`ValidationOptions` instead, whose default also throws before the first gate;
see [Validation](validation.md).

### Failures during the run that make the answer wrong

Always thrown, with no knob. The failed-run record keeps what was computed.

- A state with no norm, zero or non-finite, met at a collapse or a sample:
  `RuntimeFailure`, before anything is drawn.
- An MPS split that fails every rung of the SVD ladder (with `svd_rescue` off,
  the first rejection): `std::runtime_error`.
- The run budget exceeded by growth during the run: `RuntimeFailure`.
- `std::bad_alloc` from the allocator: propagates as itself.
- An exception from your own observer: propagates as itself.
- A final state that is zero or not finite, which only a matrix let through by
  `Validation::Warn` or `Validation::Ignore`, or an MPS starting chain with no
  norm, can produce: `RuntimeFailure`, rather than returning it as an answer.
- A consistency check inside Lindblad that fails: `InternalError`.

### Failures that leave the answer intact

These have a knob, because leaving one observation out does not make the rest of
the run wrong.

| Failure | Knob | Default |
|---|---|---|
| An observation this backend cannot produce, or one that needs a conversion under `Conversion::Never` | `RunPlan::Options::response` | `Auto` |
| An observation over the memory guard (`Cost::Guarded`, `guard_multiple`) | `response` | `Auto` |
| The entropy observer's eigensolver failing on a reduced state or a bond spectrum | `response` | `Auto` |
| An MPS split rescued by the Jacobi or Gram rung | `svd_rescue` | `true`: warn and continue |

`Response::Auto` throws `InvalidArgument` when the failure is decided before any
instruction of any shot has run (every pre-flight decision, and an anchor at the
start of the first shot), because stopping then loses nothing. After that it
warns and leaves the observation out, because stopping would throw away every
shot and instruction already paid for. `Throw`, `Warn` and `Ignore` apply as
named in both phases. A warning goes through the warning channel, and a message
repeated during a run is collapsed into one line with a count at its end. See
[Observation and the run harness](observation.md).

## Example usage

```cpp
#include "lindblad/circuit.hpp"
#include "lindblad/errors.hpp"
#include "lindblad/failed_run.hpp"
#include "lindblad/simulators/statevector_sim.hpp"

#include <iostream>

using namespace lindblad;

int main() {
    QuantumCircuit qc(3, 3);
    qc.h(0).cx(0, 1).measure(1, 0).x(2).cx(1, 2).measure_all();

    StatevectorSimulator sim;
    try {
        auto result = sim.run(qc, 1000, 42);
        // Returned, so result.counts is an answer.
        std::cout << result.counts.size() << " outcomes\n";
    } catch (const InvalidArgument& e) {
        // Refused before the first gate: change what the message names.
        std::cerr << e.what() << "\n";
    } catch (const RuntimeFailure& e) {
        std::cerr << e.what() << "\n";
        if (e.saved_to()) {
            FailedRun saved = load_failed_run(*e.saved_to());
            std::cerr << saved.shots_completed << " of " << saved.shots_requested
                      << " shots finished\n";
        } else if (auto record = take_failed_run()) {
            std::cerr << "not saved: " << record->save_note << "\n";
        }
    }
}
```

## Related pages

- [Observation and the run harness](observation.md): the `response` knob,
  observers and the run plan
- [Simulators](simulators.md): each backend's options and execution strategy
- [LocalBackend](backends.md)
- [Validation](validation.md): `ValidationOptions` for physical checks
- [Statevector](statevector.md): the `Statevector` constructor's qubit limit
- [Noise](noise.md): `NoiseModel::to_json` and `from_json`
- [Estimator](estimator.md): `run_batch` and failed runs
- [Hardware info](hw-info.md): the available-memory reading behind the
  automatic cap
