# lindblad Documentation Master Guide

This file is the working blueprint for documenting lindblad in a way that is readable for new contributors and reusable across sessions.

## Purpose

The documentation goal is to explain both:

- how to use the public API
- how the implementation is organized and where behavior lives

The project is intentionally deep, so every major public algorithm and every public method group should have documentation close to the code that defines it.

## Documentation Hierarchy

Use this order of detail:

1. `README.md` — primary entry point, project overview, build/run guide, and links to all other docs
2. `docs/Architecture.md` — subsystem map and runtime flow
3. `docs/APIOverview.md` — public headers and class-level reference index
4. `docs/algorithms/*.md` — one page per algorithm family or public algorithm
5. `docs/api/*.md` — optional deep dives for public classes, options, results, and helper methods
6. `docs/BuildAndTest.md` and `docs/DevelopmentGuide.md` — contributor workflow and validation

## Required Structure for Every Algorithm Page

Every file under `docs/algorithms/` should include these sections in this order:

- Purpose
- Theory summary
- Required inputs
- How to invoke the algorithm
- Header include instructions
- Simulator or primitive dependencies
- Example code
- Return values and outputs
- Exceptions or failure modes
- Common pitfalls
- Testing notes
- Related source files

For low-level helpers, include only the sections that matter, but always document:

- what it does
- where it is declared
- what it returns
- what it expects as input
- what can go wrong

## Required Structure for Method and Class Pages

For any page documenting a public class, method, `Options`, or `Result` type, include:

- what the API is for
- the header to include
- the namespace
- required fields or arguments
- default values if they matter
- return type or result shape
- exceptions, assertions, or preconditions
- example usage
- related algorithm page

## Documentation Style Rules

- Prefer direct, practical language.
- Assume the reader is capable but unfamiliar with the codebase.
- Explain the exact function call or circuit build step when documenting an algorithm.
- Show concrete include lines and constructor/setup steps.
- Mention how inputs are represented and in what order bits/qubits are interpreted.
- Call out when outputs are counts, bitstrings, costs, states, or structured result objects.
- Explain whether the simulator path is exact, sampled, noisy, or optimized.
- Do not hide edge cases in footnotes.
- It is acceptable for docs to be long if that prevents ambiguity.

## Algorithm Documentation Map

Create one page per public algorithm family. Variants of the same algorithm should stay together on the same page unless they differ enough to justify a separate architecture or usage model.

- `docs/algorithms/vqe.md`
- `docs/algorithms/qaoa.md`
- `docs/algorithms/maqaoa.md`
- `docs/algorithms/qpe.md`
- `docs/algorithms/grover.md`
- `docs/algorithms/deutsch-jozsa.md`
- `docs/algorithms/bernstein-vazirani.md`  # includes recursive, probabilistic, and distributed BV variants
- `docs/algorithms/simon.md`
- `docs/algorithms/ising.md`
- `docs/algorithms/dispatch.md`
- `docs/algorithms/qft.md`  # exact QFT, IQFT, AQFT (Kitaev approximation), and semi-classical (Griffiths-Niu) QFT with feedforward

MAQAOA is the main exception: it deserves its own dedicated page because it has substantially more modes, configuration surface, and usage patterns than QAOA.

If new algorithm families are added later, create a new page instead of burying details in a generic overview.

## Reusable Workflow for New Documentation

When adding a new doc:

1. Find the public header first.
2. Read the implementation and any tests that prove the intended behavior.
3. Write the page closest to the public API.
4. Add concrete usage examples.
5. Cross-link the README, API overview, and any related algorithm pages.
6. Update `changes_version.md` with one line per modified file.
7. Remove or rewrite older docs only if they are superseded by the new pages.

## Reusable Workflow for Updating Existing Documentation

When improving an old page:

1. Confirm whether the page is still the right home for the content.
2. If not, move the content into the appropriate algorithm or API page.
3. Preserve the useful parts, but remove duplication and stale summaries.
4. Add links to the canonical page rather than repeating long explanations.
5. Keep examples aligned with the current public API and tests.

## Reuse After Context Cleanup

If a new chat starts or context is lost, read this file first, then use it to rebuild the doc plan.

The recommended recovery order is:

1. Read this file
2. Read `README.md`
3. Read `docs/APIOverview.md`
4. Read the relevant public header in `include/lindblad/`
5. Read the corresponding tests in `tests/`
6. Write or update the matching doc page

## Current Priority

The initial algorithm documentation pass (BV family, QAOA, MAQAOA, QPE, Grover, Deutsch-Jozsa, Simon, QFT including semi-classical variant) is complete. Remaining priorities:

1. Write test suite for R.1.5.1: feedforward infrastructure + semi-classical QFT correctness.
2. Ensure all algorithm pages meet the full required-section checklist above.
3. Keep API deep-dive pages (`docs/api/*.md`) in sync with header changes.
4. Expand `docs/Architecture.md` and `docs/APIOverview.md` as new subsystems are added.
5. Add usage examples for noise model workflows and transpiler passes.

## Notes for Future Sessions

This guide should be treated as the canonical documentation plan until the project structure changes.
