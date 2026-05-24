# Contributing to Lindblad

Thank you for your interest in improving Lindblad. All of the following contribution types are welcome:

- **Bug reports** — unexpected behavior, incorrect simulation results, crashes
- **Code patches and pull requests** — correctness fixes, performance improvements, new features
- **Documentation improvements** — fixes, clarifications, or additions to `docs/`
- **Performance benchmarks** — new benchmark cases or profiling results
- **Anything else** — if you are unsure, open an issue and ask

## Licensing of Contributions

By submitting any contribution you grant the author (Sricharan Suresh) a perpetual, worldwide, irrevocable, non-exclusive, royalty-free license to use, reproduce, modify, adapt, publish, distribute, sublicense, and commercialize the contribution under any license terms the author chooses. **You retain copyright ownership of your contribution.** Full terms are in §6.3–6.4 of [LICENSE](LICENSE).

## Pull Request Process

**Bug fixes** can be submitted as a pull request directly — no prior discussion required.

**New features or non-trivial design changes** require an issue or discussion first. This avoids wasted effort on PRs that conflict with the project roadmap or design constraints. Open an issue, describe what you want to add and why, and wait for a go-ahead before writing code.

For all PRs:

1. Fork the repository and create a branch from `main`.
2. Make your changes.
3. Ensure the test suite passes (see [docs/BuildAndTest.md](docs/BuildAndTest.md)).
4. Submit the PR with a clear description of what changed and why.

The author retains sole discretion over whether to accept, modify, or reject any contribution (LICENSE §6.5).

## AI-Assisted Contributions

AI tools (Copilot, Claude, GPT, etc.) may be used to help draft code, write tests, or prepare reviews and PR descriptions. However:

> **Lindblad is ~40,000+ lines of C++23 with non-trivial invariants** (qubit ordering conventions, SIMD memory layout, Lindblad master-equation semantics, transpiler correctness). AI models hallucinate frequently on codebases of this size and complexity.

**Every AI-assisted contribution must be fully verified by the human contributor before submission.** This means:

- Read every line the AI produced. Do not submit output you have not personally reviewed.
- Run the relevant tests and confirm they pass.
- Verify that any new test added by the AI actually exercises the claim it makes — AI-generated tests frequently test the wrong thing or produce vacuous assertions.
- For bug reports drafted with AI help: reproduce the issue yourself before filing. Do not file hallucinated bugs.
- For AI-written PR descriptions or review comments: confirm every claim against the actual source code.

PRs that contain verifiable AI hallucinations (wrong function names, wrong invariants, fabricated test failures) will be closed without further review.

## Code Style

Lindblad follows the conventions described in the source and summarised here:

- **Qubit ordering**: LSB at qubit 0 throughout — `amp[K]` directly encodes the integer K where qubit i contributes 2^i. See [docs/Architecture.md](docs/Architecture.md) for the full convention and worked examples. Any algorithm or test that interprets qubit states as integers must verify the convention end-to-end with a non-symmetric value.
- **Section headers**: `// ===...===` border, class name and brief description on the next line.
- **Function comments**: multi-line `//` blocks above the function; parameters documented as `param_name = description`.
- **Inline comments**: one space after the declaration, e.g. `int dim; // 2^n_qubits`.
- **No aspirational comments**: explain *why* (hidden constraints, invariants, workarounds), never *what* — identifiers already do that.
- **Math**: use Unicode directly (⟨⟩, †, µ, →); LaTeX-style prose for formulas.
- **No unnecessary abstractions**: three similar lines is better than a premature helper. Do not add error handling for scenarios that cannot happen.
- Match the style of the surrounding code. If in doubt, look at a nearby file in the same subsystem.

## Bug Reports

A good bug report includes:

- The version of Lindblad (`lindblad --version` or the `LINDBLAD_VERSION_LABEL` macro).
- A minimal reproducer — the smallest circuit or call sequence that triggers the bug.
- Expected vs. actual output, including any relevant statevector amplitudes or measurement counts.
- Build environment: OS, compiler version, CMake version.

If you are filing a report based on a comparison with Qiskit or another reference simulator, include the reference output too.

## Recognition

Contributors may be acknowledged by name in release notes, the CHANGELOG, and the [CONTRIBUTORS](CONTRIBUTORS) file at the author's discretion (LICENSE §6.6). This is voluntary and does not constitute compensation or transfer of any rights.
