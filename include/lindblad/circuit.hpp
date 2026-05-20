#pragma once

#include "lindblad/types.hpp"

#include <memory>
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
        // Parameterised (symbolic)
        PARAM_RX, PARAM_RY, PARAM_RZ, PARAM_P, PARAM_U
    };

    GateType type;
    std::vector<int> qubits;     // target qubits (indices)
    std::vector<int> clbits;     // classical bits (for measure)
    std::vector<double> params;  // numeric parameters (theta, phi, lambda)

    // For symbolic/parameterised circuits
    std::vector<std::string> param_names;

    // For custom unitaries
    std::vector<Complex128> matrix;  // 2^k × 2^k unitary

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
    QuantumCircuit& unitary(const std::vector<Complex128>& matrix,
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

    std::string to_qasm2() const;
    std::string to_qasm3() const;
    static QuantumCircuit from_qasm2(const std::string& qasm);
    static QuantumCircuit from_qasm3(const std::string& qasm);

    // JSON serialization (zero-dependency, hand-rolled)
    std::string to_json() const;
    static QuantumCircuit from_json(const std::string& json);

    // =========================================================================
    // Visualisation
    // =========================================================================

    std::string to_ascii() const;

private:
    void validate_qubit(int qubit) const;
    void validate_clbit(int clbit) const;
    void add_param_name(const std::string& name);
};

} // namespace lindblad
