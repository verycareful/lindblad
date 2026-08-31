# Physical-Validity Validation

Some operands are the right size and perfectly memory-safe, yet not physically
meaningful: a supplied gate matrix that is not unitary, a Kraus set that does
not preserve trace. The math runs and returns a well-defined number, and that
number is wrong in a way nothing downstream notices.

Lindblad checks those properties by default, and lets the caller decide what
happens when one fails.

This is a separate class from index bounds and operand structure. Those two are
checked unconditionally at every primitive, because violating them is undefined
behaviour rather than a wrong answer. See `docs/api/gates.md` for that contract.

## Header

```cpp
#include "lindblad/validation.hpp"
```

`lindblad/gates.hpp` and `lindblad/circuit.hpp` both include it, so most
translation units have it in scope without naming the file.

## Namespace

`lindblad`

## Validation

An enum class naming what to do when a physical property does not hold.

- `Throw`: reject the call with `std::invalid_argument`. The default.
- `Warn`: report through the warning handler, then proceed.
- `Fix`: repair and continue where a repair is defined, otherwise throw.
- `Ignore`: skip the check entirely. The only policy that costs nothing.

`Fix` repairs normalization: a state or density matrix handed over out of
normalization is divided by what it actually sums to, and the call continues
with a correct object.

`Fix` repairs unitarity by unitary polar projection. The nearest unitary to a
square matrix, in the Frobenius sense, is the unitary polar factor: with
`M = W S V-dagger` its thin SVD, that factor is `W V-dagger`. Replacing the
singular values with ones discards exactly the non-unitary part and keeps every
direction, so a matrix that drifted through accumulated rounding is pulled back
to the unitary closest to what the caller meant.

The repair is verified rather than trusted. Its whole postcondition is that the
output is unitary, and the same residual that rejected the input measures that,
so the result is re-measured against the caller's `atol` and a projection
landing outside it raises `std::invalid_argument` rather than returning. A
projection that silently failed to converge would hand back an operand as
unphysical as the one it replaced, under a policy that promised a correction.

`QuantumCircuit::unitary` applies the repair to the matrix it stores, so the
projection runs once when the instruction is built rather than once per shot,
and every later use of that instruction sees the repaired operand. The caller's
own matrix is not modified.

For trace preservation, which has no cheap canonical repair, `Fix` throws
`std::invalid_argument` naming the property it could not repair. Saying so is
preferred to a `Fix` that quietly returns an unphysical result under a policy
that promised to correct it.

`Fix` also throws on the two states that cannot be repaired at all. A zero state
offers no direction to normalize toward, and a non-finite one has a NaN norm, so
dividing by either manufactures garbage rather than recovering a state.

## ValidationOptions

A trivially copyable aggregate carrying the two knobs.

- `Validation policy`: defaults to `Throw`.
- `double atol`: absolute tolerance on the residual. Defaults to
  `DEFAULT_PHYSICAL_ATOL`, which is `1e-12`.

`DEFAULT_PHYSICAL_ATOL` is the one number every physical-validity check in the
library judges against, and predicates over the same properties default to it
too (`Operator::is_unitary`, `KrausChannel::is_valid`, `DensityMatrix::is_valid`,
and `is_normalized` on every state class). Asking "is this valid?" and letting a
policy judge it therefore agree by construction rather than by coincidence.

There is no relative tolerance anywhere in this library. `atol` is absolute, so
one number has to mean the same thing at every register size, which is a
constraint on how residuals are measured rather than on the number itself.

Construct it inline at a call site:

```cpp
apply_unitary(sv, {0, 1}, U);                            // Throw at 1e-12
apply_unitary(sv, {0, 1}, U, {Validation::Warn});        // Warn at 1e-12
apply_unitary(sv, {0, 1}, U, {Validation::Throw, 1e-10});
apply_unitary(sv, {0, 1}, U, {Validation::Ignore});      // no check at all
```

## Choosing a tolerance

A correct gate over irrational amplitudes is never bit-exactly unitary, so the
check has to carry a tolerance at all. `1e-12` is loose enough to accept a
deeply composed or fused matrix, which holds unitarity to around `1e-13`, and
tight enough to reject anything that drifted for a reason.

`Operator::is_unitary`, `Operator::is_hermitian` and `KrausChannel::is_valid`
default to the same `1e-12`, so asking whether an operand is acceptable gives
the same verdict as handing it to a primitive. A looser default on the query
would let a matrix pass `is_unitary()` and then be rejected by the kernel it was
checked for.

The tolerance is absolute and applies to the worst entry of the residual. Note
that this is stricter than the convention common elsewhere in the ecosystem,
where a relative term is usually combined with an absolute one and the effective
allowance on a unit-magnitude entry is correspondingly wider.

The failure message reports the measured residual, so a caller who needs a
different tolerance can read the number rather than guess it:

```text
apply_unitary: matrix is not unitary (max |U†U - I| = 3.7e-09, atol = 1e-12)
```

That number is reproducible: the residual arithmetic is compiled under IEEE
semantics rather than the project-wide fast-math flags, so the same operand
measures the same on every target and compiler. The residuals evaluate
`U†U - I` and its channel equivalents, which are near-cancellations, and a
permissive floating-point model perturbs those enough to move a verdict for an
operand sitting near `atol`. The policy dispatch around them is ordinary code
and is not quarantined; only the measurement is.

## What is checked

Unitarity of a caller-supplied matrix:

- `gates::apply_unitary`
- `QuantumCircuit::unitary`
- `DensityMatrix::apply_gate`
- `MPSState::apply_single_qubit_gate`, `MPSState::apply_two_qubit_gate`
- `QuditStatevector::apply_1qudit`, `apply_2qudit`, `apply_kqudit`
- `QuditMPS::apply_1qudit`, `apply_2qudit_adjacent`, `apply_2qudit`
- `QuditDensityMatrix::apply_1qudit`, `apply_2qudit`

Trace preservation of a caller-supplied channel:

- `DensityMatrix::apply_kraus`, `Σ K†K = I`
- `DensityMatrix::apply_channel_superop`, in the superoperator form of the same
  condition: `Σ_ro S[(ro,ro),(ri,ci)] = δ_ri,ci`
- `QuditDensityMatrix::apply_kraus_1qudit`, `apply_kraus_2qudit`

A channel with no operators is NOT on this list, deliberately. It is rejected
before any residual is measured, as a malformed argument rather than as invalid
physics, so the rejection is unconditional and `Validation::Ignore` does not
suppress it. Sum over no operators is the zero matrix, which would fuse to an
all-zero superoperator and leave a state whose trace is zero. `Ignore` means the
caller accepts an operand whose physics is off by more than `atol`; it cannot
also mean they accept losing the state.

Normalization of a caller-supplied state:

- `Statevector::set_amplitudes`, both overloads: `⟨ψ|ψ⟩ = 1`

Plus `check_normalized(ValidationOptions)` on every state class, which a caller
invokes on a state they already hold rather than one they are handing over:
`Statevector`, `DensityMatrix`, `MPSState`, `QuditStatevector`,
`QuditDensityMatrix` and `QuditMPS`. The density-matrix classes measure
`Tr(ρ) = 1` instead, which is the same idea over a different object.

Each class also carries `is_normalized(atol)`, a plain predicate that answers
without repairing and without throwing. A non-finite state answers false.

The Clifford backend has no arbitrary-matrix entry point, so nothing there
carries a policy.

### Why normalization is checked at hand-over and not per gate

A supplied matrix is unitary or it is not, and it never changes afterwards. A
state's norm drifts as gates are applied, because floating-point arithmetic
rounds. So a check that fired after every gate would eventually reject states
that nothing is wrong with, and would pay a full `2^n` sweep each time to do it.

Checking where a caller hands a whole state over has neither problem: the state
has not drifted yet, it is a claim rather than a computation, and the cost is
paid once.

Every reference simulator makes the same call. Qiskit validates in
`StatePreparation` and raises unless `normalize=True`; Cirq raises from
`validate_normalized_state_vector`; PennyLane raises from `qml.StatePrep` unless
`normalize=True`. None of the three checks per gate.

The library's own outputs are exempt for the same reason: the final-state copy
inside the statevector simulator passes `Ignore`, because those amplitudes carry
whatever rounding the circuit accumulated and are not a caller's claim about
anything.

## What it costs

For a k-operand matrix of side `N = 2^k`, the residual is `N^3/2` complex
multiplies against a kernel that sweeps `2^n · 2^k` amplitudes, so the check is
`4^k / 2^n` of the work it guards. At twenty qubits with a two-qubit gate that
is fifteen parts per million. It approaches the kernel's own cost only for wide
gates on tiny registers.

`Ignore` returns before measuring anything, so opting out costs one predictable
branch.

## Circuits carry their policy

`Instruction` holds a `ValidationOptions` describing its matrix.
`QuantumCircuit::unitary` applies it at ingress and stores it, so the same
matrix is judged by the same policy wherever it is later executed.

```cpp
QuantumCircuit qc(2);
qc.unitary(U, {0, 1}, "my_gate", {Validation::Warn, 1e-10});
```

`QuantumCircuit::validate_physical()` is the pre-flight every backend `run()`
performs. It is the one place every matrix in a circuit is seen exactly once,
whatever route it arrived by, which matters because a matrix can enter a
circuit without passing through `unitary()`: the QASM parser builds
instructions directly. It runs before gate fusion, so a matrix is judged while
it is still the caller's rather than after it has been multiplied into a block.

Because the pre-flight has already judged every matrix, execution does not
judge them again. Every backend follows the same rule: below its `run()`
pre-flight, each kernel call applies the matrix under `Ignore`, whether that is
the statevector trajectory, the terminal-measurement pass, the fusion builder,
the density-matrix gate loop, or the MPS dispatcher. Re-checking would measure
the same unchanged matrix once per shot.

The rule is about the route, not the backend. A primitive reached DIRECTLY,
with no `run()` above it, has had no pre-flight, so it applies the caller's
policy as given: `StatevectorSimulator::apply_instruction`,
`DensityMatrix::apply_gate`, `MPSState::apply_single_qubit_gate` and
`apply_two_qubit_gate` each take a `ValidationOptions` for that reason, and
default it to `Throw`. `run()` passes `Ignore` into those same entry points.

Instructions the library synthesises rather than receives carry `Ignore`. A
fused block, a controlled-U built by repeated squaring in phase estimation, and
a decomposition product are the library's own arithmetic, not a caller's
declaration, and their distance from exact unitarity is accumulated rounding.

Two consequences of the policy living on the instruction are worth knowing:

- `compose`, the routing and layout passes, and `inverse` preserve it, because
  they copy instructions.
- QASM has no syntax for a policy, so an export and re-import resets to the
  `Throw` default. The reset makes a circuit louder rather than quieter.

## The warning channel

Every warning the library emits goes through one handler, so a caller can
capture, redirect, or silence all of them in one place. The default writes to
stderr.

- `set_warning_handler(WarningHandler)`: install a handler. An empty handler
  restores the stderr default.
- `emit_warning(const std::string&)`: report a warning.
- `flush_warnings()`: emit pending repeat counts and clear them.
- `ScopedWarningFlush`: flushes when the scope ends, on every exit path
  including an exception. Each backend `run()` holds one.

`Warn` is reachable from inside a shots loop, where the same violation recurs
once per gate per shot on a matrix that has not changed. The first occurrence
of a message reaches the handler immediately and later identical ones are
counted, so a ten-thousand-shot run reports one line and a count rather than
ten thousand lines:

```text
lindblad: apply_unitary: matrix is not unitary (max |U†U - I| = 3.7e-09, atol = 1e-12)
lindblad: apply_unitary: matrix is not unitary (max |U†U - I| = 3.7e-09, atol = 1e-12) [repeated 9999 more times]
```

Above sixty-four distinct messages, counting stops and every occurrence is
emitted: a workload generating unbounded distinct warnings is one where the
counts are not the interesting part.

The handler is invoked with the warning lock held, which is what keeps a sink
from being entered by two OpenMP threads at once. It must therefore not call
back into `set_warning_handler`, `emit_warning`, or `flush_warnings`.
