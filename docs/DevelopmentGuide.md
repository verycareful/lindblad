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