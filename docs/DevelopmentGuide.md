# Development Guide

## Branching and Change Scope

Recommended workflow:

1. Create a feature branch from the main branch.
2. Keep changes focused on a single subsystem or user-visible capability.
3. Update tests and docs in the same branch.

## Coding Conventions

- Language standard: C++23
- Keep public declarations in `include/lindblad/`
- Keep implementation in matching `src/` location
- Favor descriptive names over abbreviations in public APIs
- Ensure exceptions and error strings are clear at subsystem boundaries

## Raising Errors and Warnings

Every refusal and failure goes through `include/lindblad/detail/report.hpp`, and
every exception Lindblad raises is one of the four types in
`include/lindblad/errors.hpp` (see [docs/api/failures.md](api/failures.md)).
Classify a site by the first question that answers yes:

1. Can it only fail if Lindblad itself is wrong? Call `detail::raise_internal()`,
   which raises `InternalError` and appends the request to report it.
2. Is it a caller mistake (the call, the circuit, the options, the plan) or an
   exceeded limit (qubits, memory)? Call `detail::raise<InvalidArgument>()`, or
   `detail::raise<OutOfRange>()` for an index outside its range.
3. Is it a physical check on something the caller handed in (unitarity, trace
   preservation, normalisation)? It belongs to `ValidationOptions`.
4. Is it met while running, and does it make the answer wrong? Call
   `detail::raise<RuntimeFailure>()` with a `FailurePoint`.
5. Does it leave the answer intact, an optional observation left out? Call
   `detail::respond()` with the knob that governs it. Adding a knob where none
   exists is a design decision, not a fix.
6. Is it only information? `emit_warning("note: ...")`.

Rules that go with it:

- Write `detail::raise<E>(...)` qualified outside `namespace lindblad::detail`:
  `<csignal>` declares a global `raise(int)`.
- A message starts with the public entry point the caller called
  (`StatevectorSimulator::run`), says what is wrong and what to change, and
  gives both sides of every comparison. Pass a `FailurePoint` for an
  instruction rather than formatting its index by hand. Describe the fact in
  the present tense.
- Anything the circuit, options or plan decide on their own is checked before
  the first gate. A check that fires only when execution reaches a gate belongs
  in the pass before it.
- Never add a knob to soften a failure that makes the answer wrong.
- Inside a `run()`, never catch to convert: the failure path has to see the
  exception to keep and save the partial run. Your caller's exceptions and
  `std::bad_alloc` are never wrapped.
- Each error type is also its standard counterpart, so retyping a
  `std::runtime_error` as `InvalidArgument` breaks every
  `EXPECT_THROW(..., std::runtime_error)` that reaches it; find those tests
  before changing a type.

## Performance-Sensitive Code

For code in `src/gates/`, `src/statevector.cpp`, and simulator kernels:

- Avoid unnecessary allocations in inner loops
- Prefer precomputed constants in repeated operations
- Keep branch behavior predictable in hot loops
- Validate correctness first, then optimize with measurable benchmarks

## Test Strategy

Current unit tests live in `tests/` and are built into `lindblad_tests`.

When adding a new feature:

- Add positive-path tests
- Add at least one edge case test
- Add regression tests for fixed bugs

Run before submitting:

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Benchmark Strategy

Use `benchmarks/` when introducing performance-sensitive changes.

- Benchmark old and new behavior for representative sizes
- Record command line and machine details for reproducibility
- Avoid claiming performance improvements without measured results

## Documentation Policy

- Update `README.md` for setup or behavior changes visible to users
- Update subsystem docs in `docs/` for architectural or API changes
- Keep descriptions concrete and aligned with current implementation

## Dependency Management

Dependencies are declared through CMake FetchContent at top-level `CMakeLists.txt`.

When adding a dependency:

- Justify why existing dependencies are insufficient
- Pin to a known release tag
- Keep optional dependencies behind CMake options where feasible

## Release Readiness Checklist

- Project config builds successfully on target platforms
- Unit tests pass
- Critical benchmarks run successfully
- Public docs and license metadata are current
