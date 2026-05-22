# Visualisation

Header: [`include/lindblad/visualisation.hpp`](../../include/lindblad/visualisation.hpp) (re-exports the types from [`include/lindblad/circuit.hpp`](../../include/lindblad/circuit.hpp))

Namespace: `lindblad`

The visualisation subsystem turns a `QuantumCircuit` into a circuit diagram in
one of four backends. All four share a single layered layout pass: gates are
packed ASAP into columns, then a backend-agnostic `CircuitDocument` is fed to
the chosen renderer.

## DrawMode

```cpp
enum class DrawMode { ASCII, SVG, LATEX, HTML };
```

- `ASCII` : monospaced UTF-8 text grid; falls back to portable ASCII when
  `DrawOptions::ascii_safe` is set
- `SVG` : self-contained SVG (inline `<style>`, no external CSS or fonts);
  every gate is wrapped in a `<g class="lb-glyph">` carrying `data-gate`,
  `data-col`, `data-qubits` attributes for downstream consumers
- `LATEX` : Quantikz environment only (no `\documentclass` shell)
- `HTML` : standalone HTML page embedding the SVG plus hover styling

## ParamFormat

```cpp
enum class ParamFormat { Pretty, Raw };
```

- `Pretty` : snaps recognised rational multiples of pi (pi/8, pi/6, pi/4,
  pi/3, pi/2, 2pi/3, 3pi/4, 5pi/6, pi, 3pi/2, 2pi, 3pi, 4pi) within a
  tolerance of `1e-6` and renders them as `pi/2`, `-3pi/4`, etc. Misses fall
  through to `%.4f`
- `Raw` : always `%.4f` decimal

## DrawOptions

```cpp
struct DrawOptions {
    int  fold_width     = 120;
    bool show_clbits    = false;
    bool show_params    = true;
    bool ascii_safe     = false;
    ParamFormat param_format = ParamFormat::Pretty;
    int  cell_width_px  = 48;
    int  cell_height_px = 48;
    bool include_legend = false;
};
```

Fields:

- `fold_width` : ASCII wrap column. `0` disables folding entirely
- `show_clbits` : when `false`, the bundled classical-bit wire is hidden.
  Measurements still render on the qubit wire but emit no drop line.
  Conditional gates carry the `if c[k]=v` tag inline
- `show_params` : when `false`, gate labels omit parameters (`RX` instead of
  `RX(pi/2)`)
- `ascii_safe` : ASCII-mode palette swap. Replaces UTF-8 box-drawing glyphs
  with portable ASCII (`-|+*X[]`)
- `param_format` : per-call ParamFormat selector
- `cell_width_px`, `cell_height_px` : SVG and HTML grid pitch in CSS pixels
- `include_legend` : LaTeX and HTML emit a short gate legend after the
  diagram

## QuantumCircuit::draw

```cpp
std::string draw(DrawMode mode = DrawMode::ASCII,
                 const DrawOptions& opts = {}) const;
```

Builds a `CircuitDocument` via the visualiser's layout pass, then dispatches
to the chosen renderer. Output is always a single `std::string`; callers
write it to a file or to stdout.

`to_ascii()` is retained as a thin wrapper around `draw(DrawMode::ASCII, {})`
for backwards compatibility. New code should call `draw()` directly.

## QuantumCircuit::draw_to_file

```cpp
void draw_to_file(const std::string& path,
                  DrawMode mode = DrawMode::ASCII,
                  const DrawOptions& opts = {}) const;
```

Convenience wrapper that opens `path` in binary mode and streams
`draw(mode, opts)` into it. Throws `std::runtime_error` with the offending
path embedded in the message when the open fails. Saves the
`#include <fstream>` + `std::ofstream` dance at every call site:

```cpp
qc.draw_to_file("bell.svg", DrawMode::SVG);
```

Missing parent directories are NOT created automatically. The caller is
responsible for `mkdir -p` if needed.

## CLI : lindblad_draw

Built when `LINDBLAD_BUILD_APPS=ON` (the default for top-level builds).
Lives at `apps/lindblad_draw.cpp`; produced as `lindblad_draw` in the
build's executable directory.

```text
lindblad_draw [options] <circuit.qasm>
lindblad_draw [options] --stdin
lindblad_draw [options] --demo <name>
```

Examples:

```bash
lindblad_draw bell.qasm
lindblad_draw --mode svg bell.qasm > bell.svg
lindblad_draw --mode latex --demo ghz
cat bell.qasm | lindblad_draw --stdin
lindblad_draw --demo bell --output bell.txt
```

Backend selection:

- `--mode ascii|svg|latex|html` (default: ascii)

Output destination:

- `--output <path>` writes via `draw_to_file()` instead of stdout

Visualisation options (mirror `DrawOptions` fields):

- `--ascii-safe` ASCII portable palette
- `--show-clbits` render the bundled c-wire
- `--no-show-params` strip parameters from gate labels
- `--param-format pretty|raw` numeric formatting
- `--fold N` ASCII fold width (0 disables)
- `--cell-px N` SVG/HTML cell pitch
- `--legend` LaTeX/HTML legend block

Input sources:

- positional `<circuit.qasm>` reads from a file
- `--stdin` reads QASM from standard input
- `--demo <name>` picks a built-in circuit; `--list-demos` prints the list
- `--qasm3` parses the input as QASM 3 (default: QASM 2)

The CLI is a convenience front-end for one-shot rendering. For any
performance-sensitive or scripted workflow, call `QuantumCircuit::draw()` /
`draw_to_file()` from C++ directly instead of spawning the binary.

## Python bindings

The Python bindings expose `DrawMode`, `ParamFormat`, `DrawOptions`,
`QuantumCircuit.draw()`, and `QuantumCircuit.draw_to_file()` so the
visualiser is reachable from `import lindblad as lb`. The bindings stay in
sync with the C++ API as a convenience, but direct C++ is the recommended
path for any performance-sensitive or batch workflow: the Python wrapper
unavoidably crosses the binding boundary and serialises through the GIL.

## Examples

```cpp
#include "lindblad/circuit.hpp"

using namespace lindblad;

int main() {
    QuantumCircuit qc(2, 2, "bell");
    qc.h(0).cx(0, 1).measure(0, 0).measure(1, 1);

    // Terminal-friendly text grid
    std::cout << qc.draw() << "\n";

    // Show the bundled c-wire and use the raw decimal param format
    DrawOptions opts;
    opts.show_clbits  = true;
    opts.param_format = ParamFormat::Raw;
    std::cout << qc.draw(DrawMode::ASCII, opts) << "\n";

    // Embed in a Markdown / HTML doc
    std::string svg = qc.draw(DrawMode::SVG);

    // Drop into an arXiv preprint
    std::string latex = qc.draw(DrawMode::LATEX);
    // -> \begin{quantikz}\n\lstick{$q_{0}$} & \gate{H} & \ctrl{1} & \meter{} \\
    //    \lstick{$q_{1}$} & \qw      & \targ{}  & \meter{}\n\end{quantikz}

    // Save a share-ready interactive page
    std::ofstream("bell.html") << qc.draw(DrawMode::HTML);
}
```

## Implementation

Internal types live in namespace `lindblad::viz` (not part of the public API):

- `CircuitDocument` : ordered layers plus row labels and captured options
- `Layer` : every glyph sharing the same column
- `Glyph` : one logical gate plus its parts, optional strut, conditional info
- `GlyphPart` : closed `std::variant` of seven visual primitives: `BoxPart`,
  `CtrlBulletPart`, `XorTargetPart`, `SwapXPart`, `MeasurePart`, `ResetPart`,
  `BarrierPart`

Three gate-symbol tiers feed `build_glyph()`:

- Tier 1 (`gate_symbols.cpp`) : data table of single-qubit box gates. To
  restyle `H`, edit one line.
- Tier 2 (`composite_catalogue.cpp`) : declarative slot/role table for the
  controlled and multi-bullet gates (`CX CY CZ ... CCX CCZ CSWAP RCCX`) plus
  the two-qubit interaction gates (`RXX RYY RZZ RZX ECR`) that render as a
  single tall labelled box.
- Tier 3 (`gate_builders.cpp`) : hand-written builders for `BARRIER`,
  `MEASURE`, `RESET`, `UNITARY`.

The ASCII renderer paints onto a char grid built from per-codepoint cells.
The SVG renderer emits semantic `<g class="lb-glyph">` containers so the HTML
backend can attach hover behaviour via CSS only. The LaTeX renderer emits a
Quantikz matrix one row per qubit plus an optional `\setwiretype{c}` row for
the c-wire.

## Known Limitations

- **Non-contiguous multi-qubit gates** (e.g. `UNITARY` on qubits `{0, 3}`):
  the layout reserves rows 0 through 3 to prevent any later gate from
  slipping under the tall box. This inflates the document's column count
  relative to `QuantumCircuit::depth()` on circuits with sparse multi-qubit
  unitaries
- **Long parameter labels** (e.g. `U(0.7854, 1.5708, 0.0)`) render as a
  single-line box rather than a three-row box. Multi-line boxes are a
  candidate refinement for a future patch
- **Folding** for very wide circuits is not yet enabled; the
  `DrawOptions::fold_width` field is reserved for that addition
- **LaTeX barriers** emit `\barrier[\dashed]{N}` only on the topmost barrier
  row of a glyph; multi-row barriers in Quantikz are not directly modelled
- **Matplotlib (mpl) mode** is deliberately omitted from `DrawMode`. The
  matplotlib `Figure` object can only be constructed from Python, so MPL
  support will be a future Python bindings deliverable on top of the SVG
  renderer
