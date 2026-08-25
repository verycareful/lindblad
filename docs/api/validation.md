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

`Fix` currently has no repair defined for any property, so it throws
`std::invalid_argument` naming the property it could not repair. Trace
preservation has no cheap canonical repair and is expected to keep throwing;
saying so is preferred to a `Fix` that quietly returns an unphysical result
under a policy that promised to correct it.

## ValidationOptions

A trivially copyable aggregate carrying the two knobs.

- `Validation policy`: defaults to `Throw`.
- `double atol`: absolute tolerance on the worst entry of the residual.
  Defaults to `1e-12`.

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
tight enough to reject anything that drifted for a reason. It is deliberately
stricter than the `1e-8` that `Operator::is_unitary` and
`KrausChannel::is_valid` default to: those answer a question at the tolerance
common elsewhere in the ecosystem, while `atol` enforces a contract.

The failure message reports the measured residual, so a caller who needs a
different tolerance can read the number rather than guess it:

```text
apply_unitary: matrix is not unitary (max |U†U - I| = 3.7e-09, atol = 1e-12)
```

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

The Clifford backend has no arbitrary-matrix entry point, so nothing there
carries a policy.

Full-state normalization is deliberately not on this list. It is a full `2^n`
sweep per gate and it drifts under ordinary unitary evolution, so it stays
available through explicit checkers a caller invokes on purpose.

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
judge them again: the per-shot trajectory, the terminal-measurement pass, and
the fusion builder all apply instructions under `Ignore`. Re-checking would
measure the same unchanged matrix once per shot.

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
