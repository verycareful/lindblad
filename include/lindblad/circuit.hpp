#pragma once

#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace lindblad {

// =============================================================================
// ParamExpr — symbolic parameter expression tree used by the QASM 3 parser
// =============================================================================
// QASM 3 allows gate arguments to reference named parameters (`input float θ;`)
// and to combine them with literals via `+`, `-`, `*`, `/`. Rather than forcing
// every angle to a `double` at parse time, we keep an explicit expression tree
// so the circuit can be re-bound later (VQE / parameter sweeps) without a full
// re-parse.
//
// Kind = Literal   → `literal` holds the concrete value
// Kind = Name      → `name` holds the parameter identifier
// Kind = BinaryOp  → `op` is one of '+', '-', '*', '/' and `lhs`, `rhs` are subtrees
//
// `eval(bindings)` resolves the tree against a name→value map and throws if a
// referenced name is missing.

struct ParamExpr {
    enum class Kind { Literal, Name, BinaryOp };
    Kind kind = Kind::Literal;
    double literal = 0.0;
    std::string name;
    char op = '+';
    std::unique_ptr<ParamExpr> lhs;
    std::unique_ptr<ParamExpr> rhs;

    ParamExpr() = default;
    ParamExpr(const ParamExpr& other);
    ParamExpr& operator=(const ParamExpr& other);
    ParamExpr(ParamExpr&&) noexcept = default;
    ParamExpr& operator=(ParamExpr&&) noexcept = default;

    static ParamExpr make_literal(double v);
    static ParamExpr make_name(std::string n);
    static ParamExpr make_binary(char op_char, ParamExpr l, ParamExpr r);

    // Evaluate the tree against a name → value binding map. Throws
    // std::runtime_error if a referenced name has no binding.
    double eval(const std::unordered_map<std::string, double>& bindings) const;
};

// =============================================================================
// CowMatrix — copy-on-write, immutable gate matrix
// =============================================================================
// A gate matrix is logically constant once built, but Instruction is copied a
// lot (per estimator evaluation, per transpiler pass via to_circuit/from_circuit,
// compose/inverse/control). Storing the matrix behind a shared_ptr makes those
// copies share one buffer instead of deep-copying 2^k x 2^k complex data.
// There is no in-place mutation: assigning a new vector rebinds to a fresh
// shared buffer. Read access mirrors the std::vector API and implicitly
// converts to `const std::vector<Complex128>&`, so existing call sites are
// unchanged.
class CowMatrix {
    std::shared_ptr<const std::vector<Complex128>> data_;

    static const std::vector<Complex128>& empty_vec() {
        static const std::vector<Complex128> e;
        return e;
    }

public:
    CowMatrix() = default;
    CowMatrix(std::vector<Complex128> v)
        : data_(std::make_shared<const std::vector<Complex128>>(std::move(v))) {}
    CowMatrix(std::initializer_list<Complex128> v)
        : data_(std::make_shared<const std::vector<Complex128>>(v)) {}

    CowMatrix& operator=(std::vector<Complex128> v) {
        data_ = std::make_shared<const std::vector<Complex128>>(std::move(v));
        return *this;
    }
    CowMatrix& operator=(std::initializer_list<Complex128> v) {
        data_ = std::make_shared<const std::vector<Complex128>>(v);
        return *this;
    }

    bool empty() const noexcept { return !data_ || data_->empty(); }
    size_t size() const noexcept { return data_ ? data_->size() : 0; }
    const Complex128& operator[](size_t i) const { return (*data_)[i]; }
    const Complex128* data() const noexcept { return data_ ? data_->data() : nullptr; }
    std::vector<Complex128>::const_iterator begin() const {
        return (data_ ? *data_ : empty_vec()).begin();
    }
    std::vector<Complex128>::const_iterator end() const {
        return (data_ ? *data_ : empty_vec()).end();
    }

    // Implicit view as a const vector so functions taking
    // `const std::vector<Complex128>&` accept an Instruction matrix directly.
    operator const std::vector<Complex128>&() const {
        return data_ ? *data_ : empty_vec();
    }

    bool operator==(const CowMatrix& o) const {
        return static_cast<const std::vector<Complex128>&>(*this) ==
               static_cast<const std::vector<Complex128>&>(o);
    }
    bool operator!=(const CowMatrix& o) const { return !(*this == o); }
};

// =============================================================================
// Gate instruction — what gets stored in the circuit
// =============================================================================

struct Instruction {
    enum class GateType {
        // Single qubit
        H, X, Y, Z, S, SDG, T, TDG, SX, SXDG,
        RX, RY, RZ, P, U, U1, U2, U3,
        // Two qubit
        CX, CY, CZ, CH, SWAP, ISWAP,
        CRX, CRY, CRZ, CP, CU, ECR, RZX, RXX, RYY, RZZ,
        // Three qubit
        CCX, CCZ, CSWAP, RCCX,
        // Special
        MEASURE, RESET, BARRIER,
        // Custom unitary
        UNITARY,
        // Multi-controlled (arbitrary control count) and basis permutation.
        // MCX: qubits = [controls..., target]; flips target when all controls
        //      are |1>. MCP: qubits are symmetric controls; applies phase
        //      params[0] when all are |1>. PERMUTATION: applies the basis-index
        //      map `permutation` (size 2^k) to the k target qubits as an
        //      amplitude gather — the structured-oracle building block (Shor,
        //      Grover diffusion) that avoids a dense 2^k x 2^k matrix.
        MCX, MCP, PERMUTATION,
        // Parameterised (symbolic)
        PARAM_RX, PARAM_RY, PARAM_RZ, PARAM_P, PARAM_U
    };

    GateType type;
    std::vector<int> qubits;     // target qubits (indices)
    std::vector<int> clbits;     // classical bits (for measure)
    std::vector<double> params;  // numeric parameters (theta, phi, lambda)

    // For symbolic/parameterised circuits
    std::vector<std::string> param_names;

    // For custom unitaries — copy-on-write: shared, immutable
    // buffer so Instruction copies do not deep-copy 2^k × 2^k complex data.
    CowMatrix matrix;

    // Physical-validity policy for `matrix`, set where the matrix entered the
    // circuit and carried with it from there. run()'s pre-flight is what reads
    // it: a matrix can reach a circuit without passing through unitary() (the
    // QASM parser builds instructions directly), so the pre-flight is the one
    // place every matrix is guaranteed to be seen once.
    //
    // Instructions the library synthesises rather than receives (a fused
    // block, a KAK product) carry Ignore: their matrix is the library's own
    // arithmetic, not a caller's declaration.
    ValidationOptions validation;

    // For PERMUTATION: basis-index map of size 2^k over the k target qubits.
    // permutation[x] = image of sub-state x (LSB = qubits[0]). Empty otherwise.
    std::vector<int> permutation;

    // Symbolic parameter expressions — populated by the QASM 3 parser when a
    // gate angle uses a named parameter. Empty for purely-numeric instructions
    // (numeric callers continue to use `params` exclusively, no regression).
    // When `param_exprs` is non-empty the corresponding `params` slot is left
    // empty until `QuantumCircuit::bind_parameters()` resolves the tree.
    std::vector<ParamExpr> param_exprs;

    // Metadata
    std::string label;
    int condition_clbit = -1;     // classical conditioning (-1 = none)
    int condition_value = 0;

    // Scheduling metadata (set by ASAP/ALAP passes; -1 = unscheduled)
    int schedule_time = -1;

    // Utility: gate name as string
    std::string gate_name() const;

    // Number of qubits this gate acts on
    int num_qubits() const { return static_cast<int>(qubits.size()); }

    // Is this a parameterised (symbolic) gate?
    bool is_parameterised() const;

    // Is this a measurement or reset?
    bool is_classical() const;
};

// =============================================================================
// DrawMode : output backend selector for QuantumCircuit::draw()
// =============================================================================
// One enum value per supported renderer. The default (ASCII) is suitable for
// terminal output; SVG / LATEX / HTML target embedding in docs, papers, or
// browsers respectively. The visualisation subsystem dispatches on this value
// after building a single backend-agnostic CircuitDocument.

enum class DrawMode {
    ASCII,   // Plain text grid, monospaced
    SVG,     // Self-contained SVG with semantic classes and data attributes
    LATEX,   // Quantikz environment (no document shell)
    HTML     // Standalone HTML page embedding the SVG with hover styling
};

// =============================================================================
// ParamFormat : how numeric gate parameters are rendered inside labels
// =============================================================================
// Pretty mode snaps recognised rational multiples of pi to symbolic strings
// (e.g. pi/2) before falling back to a 4-decimal fixed format. Raw mode always
// emits the 4-decimal form regardless of value.

enum class ParamFormat {
    Pretty,  // pi-snap recognised multiples, then %.4f decimal
    Raw      // Always %.4f decimal
};

// =============================================================================
// DrawOptions : per-call visualisation configuration
// =============================================================================
// Holds every knob that influences layout or rendering. Defaults are tuned for
// terminal ASCII output. SVG/HTML coordinate sizes are in CSS pixels and feed
// the SVG viewBox. fold_width = 0 disables ASCII folding entirely.

struct DrawOptions {
    int fold_width = 120;                            // ASCII wrap column; 0 disables folding
    bool show_clbits = false;                        // when false, the bundled c-wire is hidden
    bool show_params = true;                         // when false, gate labels omit parameters
    bool ascii_safe = false;                         // swap UTF-8 box-drawing for portable ASCII
    ParamFormat param_format = ParamFormat::Pretty;  // numeric parameter formatting strategy
    int cell_width_px = 48;                          // SVG/HTML horizontal grid pitch (px)
    int cell_height_px = 48;                         // SVG/HTML vertical grid pitch (px)
    bool include_legend = false;                     // LaTeX/HTML emit a small gate legend
};

// =============================================================================
// QasmExportOptions : per-call QASM export configuration
// =============================================================================
// R.1.18.0. OpenQASM 2.0 has no gate-modifier syntax, so MCX / MCP /
// PERMUTATION have no faithful representation there; by default to_qasm2()
// refuses them loudly. Setting decompose_unrepresentable lowers the three ops
// to standard gates at export time via the shared exact decompositions
// (MCX / MCP to X, H, P, CP, CX, CCX; PERMUTATION to SWAPs for wire
// relabelings, transposition networks for general basis maps). The circuit
// object is never modified; only the emitted text is decomposed. General
// PERMUTATION maps produce one multi-controlled network per displaced basis
// state, so the output can be large — that cost is inherent to the format.

// Whether a UNITARY instruction may be lowered to standard gates at export.
//
// FormatDefault is not indecision, it is the statement that each format should
// do what its own capabilities allow:
//
//   QASM 2 REFUSES. There is no literal-matrix syntax, so any UNITARY has to be
//   restructured into gates the caller never wrote, and the single-qubit
//   corrections land in U(theta, phi, lambda), which spans SU(2) rather than
//   U(2). Any operand carrying a global phase therefore loses it, and QASM 2
//   has nowhere to put it. Restructuring a circuit and discarding information
//   are both things to ask about rather than assume.
//
//   QASM 3 LOWERS. `gphase` is first-class there, so the same decomposition is
//   exact: nothing is discarded and there is nothing to consent to.
//
// Always and Never override that per call, in the same direction for both
// formats, so a value read at a call site means one thing wherever it is used.
enum class UnitaryLowering {
    FormatDefault,  // QASM 2 refuses; QASM 3 lowers
    Always,         // lower in both formats
    Never           // refuse in both formats
};

// A 2-qubit UNITARY lowers through the KAK decomposition into { U, RXX, RYY,
// RZZ }; a 1-qubit one through an Euler decomposition into U. Three or more
// qubits ALWAYS refuse, in every format and under every setting here, because
// no exact decomposition for them exists in this project. That refusal is a
// statement about the operand, not about these options, and its message says
// so rather than implying a flag would help.
// Whether a classical condition on an instruction is written into the exported
// text. FormatDefault differs per format for the same reason UnitaryLowering's
// does: one format can say it without loss and the other cannot.
//
//   QASM 3 EMITS. `if (c[k] == v) { ... }` names one bit and takes a block, so
//   any instruction's text goes inside it and nothing is lost. Its own parser
//   reads that form, so the round trip is exact.
//
//   QASM 2 REFUSES. Its `if` compares a WHOLE classical register to an integer,
//   so a single-bit condition has an exact spelling there only when the
//   register IS that one bit. Wider registers can only drop it, and discarding
//   control flow is something to ask about rather than assume.
//
// Under Always, a QASM 2 export takes the exact spelling where the register is
// one bit wide, and otherwise emits the instruction unconditioned, warns once
// through the diagnostic sink, and records the dropped condition in the text as
// a `// dropped condition: c[k] == v` comment.
enum class ConditionExport {
    FormatDefault,  // QASM 2 refuses; QASM 3 emits
    Always,         // emit in both formats
    Never           // refuse in both formats
};

struct QasmExportOptions {
    bool decompose_unrepresentable = false; // lower MCX/MCP/PERMUTATION at export

    UnitaryLowering unitary_lowering = UnitaryLowering::FormatDefault;

    ConditionExport condition_export = ConditionExport::FormatDefault;

    // QASM 2 only, and meaningful only where lowering actually happens. A
    // lowering that drops a global phase throws unless this is set; with it
    // set the export proceeds, warns through the diagnostic sink once per
    // lowered instruction, and records the dropped angle in the emitted text
    // as a `// global phase: <alpha>` comment. There is no QASM 3 counterpart
    // because no QASM 3 lowering is lossy.
    bool accept_global_phase_loss = false;
};

// =============================================================================
// QuantumCircuit
// =============================================================================

class QuantumCircuit {
public:
    int n_qubits;
    int n_clbits;
    std::string name;
    std::vector<Instruction> instructions;

    // Parameter registry (for parameterised circuits)
    std::unordered_map<std::string, double> parameter_bindings;
    std::vector<std::string> parameter_names;  // ordered

public:
    // Constructors
    QuantumCircuit();
    explicit QuantumCircuit(int n_qubits, int n_clbits = 0);
    QuantumCircuit(int n_qubits, int n_clbits, const std::string& name);

    // =========================================================================
    // Gate construction API — fluent interface (returns *this)
    // =========================================================================

    // Single-qubit gates
    QuantumCircuit& h(int qubit);
    QuantumCircuit& x(int qubit);
    QuantumCircuit& y(int qubit);
    QuantumCircuit& z(int qubit);
    QuantumCircuit& s(int qubit);
    QuantumCircuit& sdg(int qubit);
    QuantumCircuit& t(int qubit);
    QuantumCircuit& tdg(int qubit);
    QuantumCircuit& sx(int qubit);
    QuantumCircuit& sxdg(int qubit);
    QuantumCircuit& rx(double theta, int qubit);
    QuantumCircuit& ry(double theta, int qubit);
    QuantumCircuit& rz(double theta, int qubit);
    QuantumCircuit& p(double lambda, int qubit);
    QuantumCircuit& u(double theta, double phi, double lambda, int qubit);
    QuantumCircuit& u1(double lambda, int qubit);
    QuantumCircuit& u2(double phi, double lambda, int qubit);
    QuantumCircuit& u3(double theta, double phi, double lambda, int qubit);

    // Two-qubit gates
    QuantumCircuit& cx(int control, int target);
    QuantumCircuit& cy(int control, int target);
    QuantumCircuit& cz(int control, int target);
    QuantumCircuit& ch(int control, int target);
    QuantumCircuit& swap(int q1, int q2);
    QuantumCircuit& iswap(int q1, int q2);
    QuantumCircuit& crx(double theta, int control, int target);
    QuantumCircuit& cry(double theta, int control, int target);
    QuantumCircuit& crz(double theta, int control, int target);
    QuantumCircuit& cp(double lambda, int control, int target);
    QuantumCircuit& cu(double theta, double phi, double lambda, double gamma,
                       int control, int target);
    QuantumCircuit& ecr(int q1, int q2);
    QuantumCircuit& rzx(double theta, int q1, int q2);
    QuantumCircuit& rxx(double theta, int q1, int q2);
    QuantumCircuit& ryy(double theta, int q1, int q2);
    QuantumCircuit& rzz(double theta, int q1, int q2);

    // Three-qubit gates
    QuantumCircuit& ccx(int c1, int c2, int target);
    QuantumCircuit& ccz(int c1, int c2, int target);
    QuantumCircuit& cswap(int ctrl, int q1, int q2);
    QuantumCircuit& rccx(int c1, int c2, int target);

    // Custom unitary
    // validation = policy and tolerance for the unitarity of `matrix`. It is
    // applied here, at ingress, and stored on the instruction so run() applies
    // the same policy to the same matrix.
    QuantumCircuit& unitary(const std::vector<Complex128>& matrix,
                            const std::vector<int>& qubits,
                            const std::string& label = "",
                            ValidationOptions validation = {});

    // Multi-controlled X: flip `target` when every control qubit is |1>.
    // Any number of controls (0 controls == plain X). Applied natively by the
    // statevector and density-matrix backends (no dense 2^n matrix). The
    // transpiler does NOT yet lower it: BasisTranslator throws on MCX under a
    // non-empty basis_gates, and SABRE routes it only if all wire pairs are
    // already adjacent (CX/CCX ladder decomposition is planned).
    QuantumCircuit& mcx(const std::vector<int>& controls, int target);

    // Multi-controlled phase: multiply by exp(i*lambda) when every listed
    // qubit is |1> (the qubits are symmetric). 1 qubit == P, 2 == CP.
    QuantumCircuit& mcp(double lambda, const std::vector<int>& qubits);

    // Basis permutation on `qubits`: sub-state x -> perm[x] (perm has size
    // 2^qubits.size(), LSB = qubits[0]). Unitary iff perm is a bijection;
    // the builder validates that. Used for reversible-classical oracles
    // (Shor modular multiplication, Grover diffusion) without a dense matrix.
    QuantumCircuit& permute(const std::vector<int>& perm,
                            const std::vector<int>& qubits,
                            const std::string& label = "");

    // Special operations
    QuantumCircuit& measure(int qubit, int clbit);
    QuantumCircuit& measure_all();
    QuantumCircuit& barrier(std::vector<int> qubits = {});
    QuantumCircuit& reset(int qubit);

    // Classically-conditioned gates (feedforward)
    // p_if: apply P(angle) to qubit only if clreg[clbit] == clval (default: ==1)
    QuantumCircuit& p_if(double angle, int qubit, int clbit, int clval = 1);
    // add_if: general conditional gate — applies any gate type when clreg[clbit] == clval
    QuantumCircuit& add_if(int clbit, int clval, Instruction::GateType type,
                           const std::vector<int>& qubits,
                           const std::vector<double>& params = {});

    // Parameterised gate versions (symbolic)
    QuantumCircuit& rx(const std::string& param_name, int qubit);
    QuantumCircuit& ry(const std::string& param_name, int qubit);
    QuantumCircuit& rz(const std::string& param_name, int qubit);

    // =========================================================================
    // Parameter binding
    // =========================================================================

    QuantumCircuit assign_parameters(
        const std::unordered_map<std::string, double>& bindings
    ) const;

    // Bind symbolic parameter expressions emitted by `from_qasm3()`.
    // For every instruction whose `param_exprs` is non-empty, evaluate each
    // expression against `bindings`, populate `params` with the resulting
    // values, and clear `param_exprs`. Instructions with purely-numeric
    // parameters are left untouched. Throws std::runtime_error if any
    // referenced name has no binding.
    void bind_parameters(
        const std::unordered_map<std::string, double>& bindings
    );

    // =========================================================================
    // Circuit operations
    // =========================================================================

    QuantumCircuit compose(const QuantumCircuit& other,
                           const std::vector<int>& qubits = {}) const;
    QuantumCircuit inverse() const;
    QuantumCircuit repeat(int n) const;
    QuantumCircuit control(int num_ctrl_qubits = 1) const;

    // =========================================================================
    // Analysis
    // =========================================================================

    int depth() const;
    int size() const;
    std::unordered_map<std::string, int> count_ops() const;
    int num_parameters() const;

    // =========================================================================
    // Export / Import
    // =========================================================================

    // QASM 2.0: MCX/MCP/PERMUTATION throw by default (no faithful QASM 2
    // encoding exists); opts.decompose_unrepresentable lowers them at export.
    std::string to_qasm2(const QasmExportOptions& opts = {}) const;
    // QASM 3.0: MCX/MCP emit natively via the `ctrl @` modifier; PERMUTATION
    // is always lowered at export (SWAPs for wire relabelings, exact
    // transposition networks otherwise) — QASM 3 has no permutation primitive.
    std::string to_qasm3(const QasmExportOptions& opts = {}) const;
    static QuantumCircuit from_qasm2(const std::string& qasm);
    static QuantumCircuit from_qasm3(const std::string& qasm);

    // JSON serialization (zero-dependency, hand-rolled)
    std::string to_json() const;
    static QuantumCircuit from_json(const std::string& json);

    // =========================================================================
    // Visualisation
    // =========================================================================

    // Layered, parameter-aware circuit renderer. Builds a single backend-
    // agnostic CircuitDocument via ASAP packing and dispatches to one of four
    // renderers selected by `mode`. The optional `opts` controls folding,
    // c-wire visibility, parameter formatting, and SVG/HTML cell sizes; see
    // DrawOptions for per-field semantics. Returns the rendered string.
    // mode = output backend (ASCII / SVG / LATEX / HTML)
    // opts = per-call visualisation configuration (defaults are terminal-friendly)
    std::string draw(DrawMode mode = DrawMode::ASCII,
                     const DrawOptions& opts = {}) const;

    // R.1.10.3 : convenience wrapper that writes the chosen renderer's
    // output directly to a file at `path`. Equivalent to opening an
    // std::ofstream and streaming draw(mode, opts) into it, but raises
    // std::runtime_error on a failed open instead of silently dropping
    // the output, and keeps rendering and file I/O at a single call site.
    //
    // Also exposed through the Python bindings as
    //   qc.draw_to_file("bell.svg", DrawMode.SVG)
    // for ergonomics. Direct C++ remains the recommended path for any
    // performance-sensitive workflow: the Python wrapper unavoidably
    // crosses the binding boundary and serialises through the GIL.
    //
    // path = filesystem destination; missing parent directories are NOT
    //        created (caller is responsible)
    // mode = output backend (ASCII / SVG / LATEX / HTML)
    // opts = per-call visualisation configuration
    void draw_to_file(const std::string& path,
                      DrawMode mode = DrawMode::ASCII,
                      const DrawOptions& opts = {}) const;

    // Compatibility wrapper for draw(DrawMode::ASCII), kept so existing
    // callers continue to compile; scheduled for removal once tests, bindings
    // and docs migrate.
    std::string to_ascii() const;

    // Pre-flight validation for backend run(): checks that every instruction's
    // qubit and classical-bit indices are in range. The per-gate builders
    // already validate at construction time, but instructions can also enter a
    // circuit via compose() index remapping, control(), the QASM parsers, and
    // transpiler passes; this sweep guarantees no out-of-range index reaches a
    // kernel regardless of ingress. Throws std::out_of_range on the first bad
    // index (backends run it inside run()'s try, so it surfaces through Result).
    void validate_operands() const;

    // Pre-flight validation for backend run(): checks that every instruction
    // carrying a caller-supplied matrix is unitary, under that instruction's
    // own ValidationOptions. This is the one place every matrix in a circuit
    // is seen exactly once, whatever route it arrived by, and it runs before
    // gate fusion so a matrix is checked while it is still the caller's rather
    // than after it has been multiplied into a block. Instructions carrying
    // Validation::Ignore, and matrices whose size does not match their operand
    // count (a structural error, reported where sizes are checked), are
    // skipped. Throws std::invalid_argument under Validation::Throw.
    void validate_physical() const;

    // The repairing form of the pre-flight, for a backend that is about to
    // execute. Behaves exactly as validate_physical() on every policy but Fix:
    // Throw rejects, Warn reports, Ignore is skipped. Under Fix it performs the
    // repair the policy asks for.
    //
    // It returns a repaired COPY rather than mutating, and only when a repair
    // actually ran, so:
    //
    //   - the caller's circuit is never modified, matching the guarantee
    //     QuantumCircuit::unitary already makes about the caller's own matrix;
    //   - run() keeps taking a const circuit, so no backend signature moves;
    //   - a circuit needing no repair, which is nearly all of them, allocates
    //     nothing at all.
    //
    // A backend consumes it the way it already consumes gate fusion: keep an
    // optional owned circuit, point at whichever one is live, and execute
    // through the pointer.
    //
    // The repair belongs here rather than at the kernels below because a
    // backend reads each instruction's policy once, in this pass, and then
    // applies every gate under Validation::Ignore so an unchanged matrix is not
    // re-measured per gate per shot. A repair at a kernel would never be reached
    // on this path.
    std::optional<QuantumCircuit> validated_physical() const;

private:
    void validate_qubit(int qubit) const;
    void validate_clbit(int clbit) const;
    void add_param_name(const std::string& name);
};

} // namespace lindblad
