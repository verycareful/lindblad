#include "lindblad/circuit.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/detail/validate_physical.hpp"

#include "visualisation/document.hpp"
#include "visualisation/render_ascii.hpp"
#include "visualisation/render_svg.hpp"
#include "visualisation/render_latex.hpp"
#include "visualisation/render_html.hpp"

#include "transpiler/high_level_decompose.hpp"
#include "transpiler/two_qubit_decompose.hpp"
#include "lindblad/validation.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iomanip>

namespace lindblad {

// =============================================================================
// ParamExpr — deep copy + evaluation
// =============================================================================
// The copy constructor needs to deep-clone any owned subtrees so two
// instructions never share the same `unique_ptr<ParamExpr>` payload.

ParamExpr::ParamExpr(const ParamExpr& other)
    : kind(other.kind), literal(other.literal), name(other.name), op(other.op),
      lhs(other.lhs ? std::make_unique<ParamExpr>(*other.lhs) : nullptr),
      rhs(other.rhs ? std::make_unique<ParamExpr>(*other.rhs) : nullptr) {}

ParamExpr& ParamExpr::operator=(const ParamExpr& other) {
    if (this == &other) return *this;
    kind = other.kind;
    literal = other.literal;
    name = other.name;
    op = other.op;
    lhs = other.lhs ? std::make_unique<ParamExpr>(*other.lhs) : nullptr;
    rhs = other.rhs ? std::make_unique<ParamExpr>(*other.rhs) : nullptr;
    return *this;
}

ParamExpr ParamExpr::make_literal(double v) {
    ParamExpr e;
    e.kind = Kind::Literal;
    e.literal = v;
    return e;
}

ParamExpr ParamExpr::make_name(std::string n) {
    ParamExpr e;
    e.kind = Kind::Name;
    e.name = std::move(n);
    return e;
}

ParamExpr ParamExpr::make_binary(char op_char, ParamExpr l, ParamExpr r) {
    ParamExpr e;
    e.kind = Kind::BinaryOp;
    e.op = op_char;
    e.lhs = std::make_unique<ParamExpr>(std::move(l));
    e.rhs = std::make_unique<ParamExpr>(std::move(r));
    return e;
}

double ParamExpr::eval(const std::unordered_map<std::string, double>& bindings) const {
    switch (kind) {
        case Kind::Literal: return literal;
        case Kind::Name: {
            auto it = bindings.find(name);
            if (it == bindings.end()) {
                throw std::runtime_error(
                    "ParamExpr::eval: no binding for parameter '" + name + "'");
            }
            return it->second;
        }
        case Kind::BinaryOp: {
            const double l = lhs->eval(bindings);
            const double r = rhs->eval(bindings);
            switch (op) {
                case '+': return l + r;
                case '-': return l - r;
                case '*': return l * r;
                case '/':
                    if (r == 0.0) {
                        throw std::runtime_error(
                            "ParamExpr::eval: division by zero");
                    }
                    return l / r;
                default:
                    throw std::runtime_error(
                        std::string("ParamExpr::eval: unknown operator '") + op + "'");
            }
        }
    }
    return 0.0;
}

// =============================================================================
// Instruction utilities
// =============================================================================

std::string Instruction::gate_name() const {
    switch (type) {
        case GateType::H:     return "h";
        case GateType::X:     return "x";
        case GateType::Y:     return "y";
        case GateType::Z:     return "z";
        case GateType::S:     return "s";
        case GateType::SDG:   return "sdg";
        case GateType::T:     return "t";
        case GateType::TDG:   return "tdg";
        case GateType::SX:    return "sx";
        case GateType::SXDG:  return "sxdg";
        case GateType::RX:    return "rx";
        case GateType::RY:    return "ry";
        case GateType::RZ:    return "rz";
        case GateType::P:     return "p";
        case GateType::U:     return "u";
        case GateType::U1:    return "u1";
        case GateType::U2:    return "u2";
        case GateType::U3:    return "u3";
        case GateType::CX:    return "cx";
        case GateType::CY:    return "cy";
        case GateType::CZ:    return "cz";
        case GateType::CH:    return "ch";
        case GateType::SWAP:  return "swap";
        case GateType::ISWAP: return "iswap";
        case GateType::CRX:   return "crx";
        case GateType::CRY:   return "cry";
        case GateType::CRZ:   return "crz";
        case GateType::CP:    return "cp";
        case GateType::CU:    return "cu";
        case GateType::ECR:   return "ecr";
        case GateType::RZX:   return "rzx";
        case GateType::RXX:   return "rxx";
        case GateType::RYY:   return "ryy";
        case GateType::RZZ:   return "rzz";
        case GateType::CCX:   return "ccx";
        case GateType::CCZ:   return "ccz";
        case GateType::CSWAP: return "cswap";
        case GateType::RCCX:  return "rccx";
        case GateType::MEASURE: return "measure";
        case GateType::RESET:  return "reset";
        case GateType::BARRIER: return "barrier";
        case GateType::UNITARY: return label.empty() ? "unitary" : label;
        case GateType::MCX:     return "mcx";
        case GateType::MCP:     return "mcp";
        case GateType::PERMUTATION: return label.empty() ? "permutation" : label;
        case GateType::PARAM_RX: return "rx";
        case GateType::PARAM_RY: return "ry";
        case GateType::PARAM_RZ: return "rz";
        case GateType::PARAM_P:  return "p";
        case GateType::PARAM_U:  return "u";
    }
    return "unknown";
}

bool Instruction::is_parameterised() const {
    return type == GateType::PARAM_RX ||
           type == GateType::PARAM_RY ||
           type == GateType::PARAM_RZ ||
           type == GateType::PARAM_P  ||
           type == GateType::PARAM_U;
}

bool Instruction::is_classical() const {
    return type == GateType::MEASURE || type == GateType::RESET;
}

// =============================================================================
// QuantumCircuit constructors
// =============================================================================

QuantumCircuit::QuantumCircuit()
    : n_qubits(0), n_clbits(0) {}

QuantumCircuit::QuantumCircuit(int n_qubits, int n_clbits)
    : n_qubits(n_qubits), n_clbits(n_clbits) {
    if (n_qubits < 0) throw std::invalid_argument("n_qubits must be >= 0");
    if (n_clbits < 0) throw std::invalid_argument("n_clbits must be >= 0");
}

QuantumCircuit::QuantumCircuit(int n_qubits, int n_clbits, const std::string& name)
    : n_qubits(n_qubits), n_clbits(n_clbits), name(name) {
    if (n_qubits < 0) throw std::invalid_argument("n_qubits must be >= 0");
    if (n_clbits < 0) throw std::invalid_argument("n_clbits must be >= 0");
}

// =============================================================================
// Validation
// =============================================================================

void QuantumCircuit::validate_qubit(int qubit) const {
    if (qubit < 0 || qubit >= n_qubits)
        throw std::out_of_range("Qubit index " + std::to_string(qubit) +
                                " out of range [0, " + std::to_string(n_qubits) + ")");
}

void QuantumCircuit::validate_clbit(int clbit) const {
    if (clbit < 0 || clbit >= n_clbits)
        throw std::out_of_range("Classical bit index " + std::to_string(clbit) +
                                " out of range [0, " + std::to_string(n_clbits) + ")");
}

void QuantumCircuit::validate_operands() const {
    for (const auto& inst : instructions) {
        for (int q : inst.qubits) validate_qubit(q);
        for (int c : inst.clbits) validate_clbit(c);
        if (inst.condition_clbit >= 0) validate_clbit(inst.condition_clbit);
    }
}

void QuantumCircuit::validate_physical() const {
    for (const auto& inst : instructions) {
        if (inst.type != Instruction::GateType::UNITARY) continue;
        if (inst.validation.policy == Validation::Ignore) continue;

        const std::vector<Complex128>& m = inst.matrix;
        const std::size_t rows = std::size_t(1) << inst.qubits.size();
        if (m.size() != rows * rows) continue;

        detail::check_unitary(m, rows, inst.validation,
                              inst.gate_name().c_str());
    }
}

void QuantumCircuit::add_param_name(const std::string& pname) {
    for (const auto& existing : parameter_names) {
        if (existing == pname) return;
    }
    parameter_names.push_back(pname);
}

// =============================================================================
// Single-qubit gates
// =============================================================================

#define LINDBLAD_SINGLE_GATE(name_func, gate_type) \
    QuantumCircuit& QuantumCircuit::name_func(int qubit) { \
        validate_qubit(qubit); \
        Instruction inst; \
        inst.type = Instruction::GateType::gate_type; \
        inst.qubits = {qubit}; \
        instructions.push_back(std::move(inst)); \
        return *this; \
    }

LINDBLAD_SINGLE_GATE(h, H)
LINDBLAD_SINGLE_GATE(x, X)
LINDBLAD_SINGLE_GATE(y, Y)
LINDBLAD_SINGLE_GATE(z, Z)
LINDBLAD_SINGLE_GATE(s, S)
LINDBLAD_SINGLE_GATE(sdg, SDG)
LINDBLAD_SINGLE_GATE(t, T)
LINDBLAD_SINGLE_GATE(tdg, TDG)
LINDBLAD_SINGLE_GATE(sx, SX)
LINDBLAD_SINGLE_GATE(sxdg, SXDG)

#undef LINDBLAD_SINGLE_GATE

#define LINDBLAD_PARAM1_GATE(name_func, gate_type) \
    QuantumCircuit& QuantumCircuit::name_func(double param, int qubit) { \
        validate_qubit(qubit); \
        Instruction inst; \
        inst.type = Instruction::GateType::gate_type; \
        inst.qubits = {qubit}; \
        inst.params = {param}; \
        instructions.push_back(std::move(inst)); \
        return *this; \
    }

LINDBLAD_PARAM1_GATE(rx, RX)
LINDBLAD_PARAM1_GATE(ry, RY)
LINDBLAD_PARAM1_GATE(rz, RZ)
LINDBLAD_PARAM1_GATE(p, P)

#undef LINDBLAD_PARAM1_GATE

QuantumCircuit& QuantumCircuit::u(double theta, double phi, double lambda, int qubit) {
    validate_qubit(qubit);
    Instruction inst;
    inst.type = Instruction::GateType::U;
    inst.qubits = {qubit};
    inst.params = {theta, phi, lambda};
    instructions.push_back(std::move(inst));
    return *this;
}

QuantumCircuit& QuantumCircuit::u1(double lambda, int qubit) {
    validate_qubit(qubit);
    Instruction inst;
    inst.type = Instruction::GateType::U1;
    inst.qubits = {qubit};
    inst.params = {lambda};
    instructions.push_back(std::move(inst));
    return *this;
}

QuantumCircuit& QuantumCircuit::u2(double phi, double lambda, int qubit) {
    validate_qubit(qubit);
    Instruction inst;
    inst.type = Instruction::GateType::U2;
    inst.qubits = {qubit};
    inst.params = {phi, lambda};
    instructions.push_back(std::move(inst));
    return *this;
}

QuantumCircuit& QuantumCircuit::u3(double theta, double phi, double lambda, int qubit) {
    validate_qubit(qubit);
    Instruction inst;
    inst.type = Instruction::GateType::U3;
    inst.qubits = {qubit};
    inst.params = {theta, phi, lambda};
    instructions.push_back(std::move(inst));
    return *this;
}

// =============================================================================
// Two-qubit gates
// =============================================================================

#define LINDBLAD_TWO_GATE(name_func, gate_type) \
    QuantumCircuit& QuantumCircuit::name_func(int q1, int q2) { \
        validate_qubit(q1); \
        validate_qubit(q2); \
        if (q1 == q2) [[unlikely]] \
            throw std::invalid_argument(#name_func ": qubits must be distinct (got q1=q2=" + std::to_string(q1) + ")"); \
        Instruction inst; \
        inst.type = Instruction::GateType::gate_type; \
        inst.qubits = {q1, q2}; \
        instructions.push_back(std::move(inst)); \
        return *this; \
    }

LINDBLAD_TWO_GATE(cx, CX)
LINDBLAD_TWO_GATE(cy, CY)
LINDBLAD_TWO_GATE(cz, CZ)
LINDBLAD_TWO_GATE(ch, CH)
LINDBLAD_TWO_GATE(swap, SWAP)
LINDBLAD_TWO_GATE(iswap, ISWAP)
LINDBLAD_TWO_GATE(ecr, ECR)

#undef LINDBLAD_TWO_GATE

#define LINDBLAD_CTRL_PARAM1_GATE(name_func, gate_type) \
    QuantumCircuit& QuantumCircuit::name_func(double param, int control, int target) { \
        validate_qubit(control); \
        validate_qubit(target); \
        if (control == target) [[unlikely]] \
            throw std::invalid_argument(#name_func ": control and target must be distinct (got " + std::to_string(control) + ")"); \
        Instruction inst; \
        inst.type = Instruction::GateType::gate_type; \
        inst.qubits = {control, target}; \
        inst.params = {param}; \
        instructions.push_back(std::move(inst)); \
        return *this; \
    }

LINDBLAD_CTRL_PARAM1_GATE(crx, CRX)
LINDBLAD_CTRL_PARAM1_GATE(cry, CRY)
LINDBLAD_CTRL_PARAM1_GATE(crz, CRZ)
LINDBLAD_CTRL_PARAM1_GATE(cp, CP)

#undef LINDBLAD_CTRL_PARAM1_GATE

QuantumCircuit& QuantumCircuit::cu(double theta, double phi, double lambda, double gamma,
                                    int control, int target) {
    validate_qubit(control);
    validate_qubit(target);
    if (control == target) [[unlikely]]
        throw std::invalid_argument("cu: control and target must be distinct (got " + std::to_string(control) + ")");
    Instruction inst;
    inst.type = Instruction::GateType::CU;
    inst.qubits = {control, target};
    inst.params = {theta, phi, lambda, gamma};
    instructions.push_back(std::move(inst));
    return *this;
}

#define LINDBLAD_ISING_GATE(name_func, gate_type) \
    QuantumCircuit& QuantumCircuit::name_func(double theta, int q1, int q2) { \
        validate_qubit(q1); \
        validate_qubit(q2); \
        if (q1 == q2) [[unlikely]] \
            throw std::invalid_argument(#name_func ": qubits must be distinct (got q1=q2=" + std::to_string(q1) + ")"); \
        Instruction inst; \
        inst.type = Instruction::GateType::gate_type; \
        inst.qubits = {q1, q2}; \
        inst.params = {theta}; \
        instructions.push_back(std::move(inst)); \
        return *this; \
    }

LINDBLAD_ISING_GATE(rzx, RZX)
LINDBLAD_ISING_GATE(rxx, RXX)
LINDBLAD_ISING_GATE(ryy, RYY)
LINDBLAD_ISING_GATE(rzz, RZZ)

#undef LINDBLAD_ISING_GATE

// =============================================================================
// Three-qubit gates
// =============================================================================

static void validate_distinct_3q(int a, int b, int c, const char* gate) {
    if (a == b || a == c || b == c) [[unlikely]]
        throw std::invalid_argument(
            std::string(gate) + ": all three qubits must be distinct (got " +
            std::to_string(a) + "," + std::to_string(b) + "," + std::to_string(c) + ")");
}

QuantumCircuit& QuantumCircuit::ccx(int c1, int c2, int target) {
    validate_qubit(c1); validate_qubit(c2); validate_qubit(target);
    validate_distinct_3q(c1, c2, target, "ccx");
    Instruction inst;
    inst.type = Instruction::GateType::CCX;
    inst.qubits = {c1, c2, target};
    instructions.push_back(std::move(inst));
    return *this;
}

QuantumCircuit& QuantumCircuit::ccz(int c1, int c2, int target) {
    validate_qubit(c1); validate_qubit(c2); validate_qubit(target);
    validate_distinct_3q(c1, c2, target, "ccz");
    Instruction inst;
    inst.type = Instruction::GateType::CCZ;
    inst.qubits = {c1, c2, target};
    instructions.push_back(std::move(inst));
    return *this;
}

QuantumCircuit& QuantumCircuit::cswap(int ctrl, int q1, int q2) {
    validate_qubit(ctrl); validate_qubit(q1); validate_qubit(q2);
    validate_distinct_3q(ctrl, q1, q2, "cswap");
    Instruction inst;
    inst.type = Instruction::GateType::CSWAP;
    inst.qubits = {ctrl, q1, q2};
    instructions.push_back(std::move(inst));
    return *this;
}

QuantumCircuit& QuantumCircuit::rccx(int c1, int c2, int target) {
    validate_qubit(c1); validate_qubit(c2); validate_qubit(target);
    validate_distinct_3q(c1, c2, target, "rccx");
    Instruction inst;
    inst.type = Instruction::GateType::RCCX;
    inst.qubits = {c1, c2, target};
    instructions.push_back(std::move(inst));
    return *this;
}

// =============================================================================
// Custom unitary
// =============================================================================

QuantumCircuit& QuantumCircuit::unitary(const std::vector<Complex128>& matrix,
                                         const std::vector<int>& qubits,
                                         const std::string& label,
                                         ValidationOptions validation) {
    for (int q : qubits) validate_qubit(q);

    // A wrong-size matrix is a structural error, reported where the size is
    // checked. Measuring unitarity on it would read past the operand, so the
    // physical check only runs on an operand whose shape already holds.
    const std::size_t rows = std::size_t(1) << qubits.size();
    if (matrix.size() == rows * rows)
        detail::check_unitary(matrix, rows, validation, "unitary");

    Instruction inst;
    inst.type = Instruction::GateType::UNITARY;
    inst.qubits = qubits;
    inst.matrix = matrix;
    inst.label = label;
    inst.validation = validation;
    instructions.push_back(std::move(inst));
    return *this;
}

QuantumCircuit& QuantumCircuit::mcx(const std::vector<int>& controls, int target) {
    validate_qubit(target);
    for (int c : controls) {
        validate_qubit(c);
        if (c == target)
            throw std::invalid_argument("mcx: control qubit equals target");
    }
    Instruction inst;
    inst.type = Instruction::GateType::MCX;
    inst.qubits = controls;          // controls first, ...
    inst.qubits.push_back(target);   // ... target last
    instructions.push_back(std::move(inst));
    return *this;
}

QuantumCircuit& QuantumCircuit::mcp(double lambda, const std::vector<int>& qubits) {
    if (qubits.empty())
        throw std::invalid_argument("mcp: needs at least one qubit");
    for (int q : qubits) validate_qubit(q);
    Instruction inst;
    inst.type = Instruction::GateType::MCP;
    inst.qubits = qubits;
    inst.params = {lambda};
    instructions.push_back(std::move(inst));
    return *this;
}

QuantumCircuit& QuantumCircuit::permute(const std::vector<int>& perm,
                                        const std::vector<int>& qubits,
                                        const std::string& label) {
    for (int q : qubits) validate_qubit(q);
    const size_t k = qubits.size();
    const size_t sub_dim = size_t(1) << k;
    if (perm.size() != sub_dim)
        throw std::invalid_argument(
            "permute: permutation size must be 2^qubits.size()");
    // Validate bijection over [0, 2^k) — a non-bijective map is non-unitary.
    std::vector<char> seen(sub_dim, 0);
    for (int v : perm) {
        if (v < 0 || static_cast<size_t>(v) >= sub_dim)
            throw std::invalid_argument("permute: image index out of range");
        if (seen[static_cast<size_t>(v)])
            throw std::invalid_argument("permute: map is not a bijection");
        seen[static_cast<size_t>(v)] = 1;
    }
    Instruction inst;
    inst.type = Instruction::GateType::PERMUTATION;
    inst.qubits = qubits;
    inst.permutation = perm;
    inst.label = label;
    instructions.push_back(std::move(inst));
    return *this;
}

// =============================================================================
// Special operations
// =============================================================================

QuantumCircuit& QuantumCircuit::measure(int qubit, int clbit) {
    validate_qubit(qubit);
    validate_clbit(clbit);
    Instruction inst;
    inst.type = Instruction::GateType::MEASURE;
    inst.qubits = {qubit};
    inst.clbits = {clbit};
    instructions.push_back(std::move(inst));
    return *this;
}

QuantumCircuit& QuantumCircuit::measure_all() {
    if (n_clbits < n_qubits) {
        n_clbits = n_qubits;
    }
    for (int i = 0; i < n_qubits; ++i) {
        measure(i, i);
    }
    return *this;
}

QuantumCircuit& QuantumCircuit::barrier(std::vector<int> qubits) {
    if (qubits.empty()) {
        qubits.resize(n_qubits);
        for (int i = 0; i < n_qubits; ++i) qubits[i] = i;
    }
    for (int q : qubits) validate_qubit(q);
    Instruction inst;
    inst.type = Instruction::GateType::BARRIER;
    inst.qubits = qubits;
    instructions.push_back(std::move(inst));
    return *this;
}

QuantumCircuit& QuantumCircuit::reset(int qubit) {
    validate_qubit(qubit);
    Instruction inst;
    inst.type = Instruction::GateType::RESET;
    inst.qubits = {qubit};
    instructions.push_back(std::move(inst));
    return *this;
}

// =============================================================================
// Classically-conditioned gates (feedforward)
// =============================================================================

QuantumCircuit& QuantumCircuit::p_if(double angle, int qubit, int clbit, int clval) {
    validate_qubit(qubit);
    validate_clbit(clbit);
    Instruction inst;
    inst.type = Instruction::GateType::P;
    inst.qubits = {qubit};
    inst.params = {angle};
    inst.condition_clbit = clbit;
    inst.condition_value = clval;
    instructions.push_back(std::move(inst));
    return *this;
}

QuantumCircuit& QuantumCircuit::add_if(int clbit, int clval, Instruction::GateType type,
                                        const std::vector<int>& qubits,
                                        const std::vector<double>& params) {
    validate_clbit(clbit);
    for (int q : qubits) validate_qubit(q);
    Instruction inst;
    inst.type = type;
    inst.qubits = qubits;
    inst.params = params;
    inst.condition_clbit = clbit;
    inst.condition_value = clval;
    instructions.push_back(std::move(inst));
    return *this;
}

// =============================================================================
// Parameterised gates
// =============================================================================

QuantumCircuit& QuantumCircuit::rx(const std::string& param_name, int qubit) {
    validate_qubit(qubit);
    add_param_name(param_name);
    Instruction inst;
    inst.type = Instruction::GateType::PARAM_RX;
    inst.qubits = {qubit};
    inst.param_names = {param_name};
    instructions.push_back(std::move(inst));
    return *this;
}

QuantumCircuit& QuantumCircuit::ry(const std::string& param_name, int qubit) {
    validate_qubit(qubit);
    add_param_name(param_name);
    Instruction inst;
    inst.type = Instruction::GateType::PARAM_RY;
    inst.qubits = {qubit};
    inst.param_names = {param_name};
    instructions.push_back(std::move(inst));
    return *this;
}

QuantumCircuit& QuantumCircuit::rz(const std::string& param_name, int qubit) {
    validate_qubit(qubit);
    add_param_name(param_name);
    Instruction inst;
    inst.type = Instruction::GateType::PARAM_RZ;
    inst.qubits = {qubit};
    inst.param_names = {param_name};
    instructions.push_back(std::move(inst));
    return *this;
}

// =============================================================================
// Parameter binding
// =============================================================================

QuantumCircuit QuantumCircuit::assign_parameters(
    const std::unordered_map<std::string, double>& bindings
) const {
    QuantumCircuit result = *this;
    result.parameter_bindings.insert(bindings.begin(), bindings.end());

    // Replace parameterised instructions with concrete ones
    for (auto& inst : result.instructions) {
        if (!inst.is_parameterised()) continue;

        bool all_bound = true;
        std::vector<double> resolved_params;
        for (const auto& pname : inst.param_names) {
            auto it = result.parameter_bindings.find(pname);
            if (it != result.parameter_bindings.end()) {
                resolved_params.push_back(it->second);
            } else {
                all_bound = false;
                break;
            }
        }

        if (all_bound) {
            // Convert to concrete gate
            switch (inst.type) {
                case Instruction::GateType::PARAM_RX: inst.type = Instruction::GateType::RX; break;
                case Instruction::GateType::PARAM_RY: inst.type = Instruction::GateType::RY; break;
                case Instruction::GateType::PARAM_RZ: inst.type = Instruction::GateType::RZ; break;
                case Instruction::GateType::PARAM_P:  inst.type = Instruction::GateType::P;  break;
                case Instruction::GateType::PARAM_U:  inst.type = Instruction::GateType::U;  break;
                default: break;
            }
            inst.params = resolved_params;
            inst.param_names.clear();
        }
    }

    // Remove resolved parameters from the names list
    std::vector<std::string> remaining;
    for (const auto& pname : result.parameter_names) {
        if (bindings.find(pname) == bindings.end()) {
            remaining.push_back(pname);
        }
    }
    result.parameter_names = remaining;

    return result;
}

// =============================================================================
// bind_parameters — resolve ParamExpr trees emitted by from_qasm3()
// =============================================================================
// Walks every instruction. If `param_exprs` is non-empty we evaluate each
// expression against `bindings` and write the result into `params`. The
// expression vector is then cleared so the instruction is indistinguishable
// from one constructed with literal angles. Numeric instructions are skipped
// entirely (no copy, no allocation).
//
// We also merge the supplied bindings into `parameter_bindings` so subsequent
// queries (`assign_parameters`, `num_parameters`) see the same view.

void QuantumCircuit::bind_parameters(
    const std::unordered_map<std::string, double>& bindings
) {
    parameter_bindings.insert(bindings.begin(), bindings.end());

    for (auto& inst : instructions) {
        if (inst.param_exprs.empty()) continue;

        inst.params.clear();
        inst.params.reserve(inst.param_exprs.size());
        for (const auto& expr : inst.param_exprs) {
            inst.params.push_back(expr.eval(parameter_bindings));
        }
        inst.param_exprs.clear();
    }
}

// =============================================================================
// Circuit operations
// =============================================================================

QuantumCircuit QuantumCircuit::compose(const QuantumCircuit& other,
                                        const std::vector<int>& qubits) const {
    QuantumCircuit result = *this;

    if (qubits.empty()) {
        // Direct composition — same qubit mapping
        if (other.n_qubits > result.n_qubits) {
            result.n_qubits = other.n_qubits;
        }
        if (other.n_clbits > result.n_clbits) {
            result.n_clbits = other.n_clbits;
        }
        for (auto inst : other.instructions) {
            result.instructions.push_back(std::move(inst));
        }
    } else {
        // Map other's qubits to specified positions
        if (static_cast<int>(qubits.size()) != other.n_qubits) {
            throw std::invalid_argument("Qubit mapping size mismatch");
        }
        if (other.n_clbits > result.n_clbits) {
            result.n_clbits = other.n_clbits;
        }
        for (auto inst : other.instructions) {
            for (auto& q : inst.qubits) {
                q = qubits[q];
            }
            result.instructions.push_back(std::move(inst));
        }
    }

    // Merge parameters
    for (const auto& pname : other.parameter_names) {
        result.add_param_name(pname);
    }

    return result;
}

QuantumCircuit QuantumCircuit::inverse() const {
    QuantumCircuit result(n_qubits, n_clbits);
    result.name = name + "_inv";

    // Reverse the instruction order
    for (auto it = instructions.rbegin(); it != instructions.rend(); ++it) {
        Instruction inv_inst = *it;

        // Skip non-invertible operations
        if (inv_inst.type == Instruction::GateType::MEASURE ||
            inv_inst.type == Instruction::GateType::RESET ||
            inv_inst.type == Instruction::GateType::BARRIER) {
            if (inv_inst.type == Instruction::GateType::BARRIER) {
                result.instructions.push_back(std::move(inv_inst));
            }
            continue;
        }

        // Apply appropriate inverse transformations
        switch (inv_inst.type) {
            // Self-inverse gates
            case Instruction::GateType::H:
            case Instruction::GateType::X:
            case Instruction::GateType::Y:
            case Instruction::GateType::Z:
            case Instruction::GateType::CX:
            case Instruction::GateType::CY:
            case Instruction::GateType::CZ:
            case Instruction::GateType::CH:
            case Instruction::GateType::SWAP:
            case Instruction::GateType::CCX:
            case Instruction::GateType::CCZ:
            case Instruction::GateType::CSWAP:
            case Instruction::GateType::ECR:
            case Instruction::GateType::MCX:  // multi-controlled X is self-inverse
                break;

            // S <-> SDG
            case Instruction::GateType::S:
                inv_inst.type = Instruction::GateType::SDG;
                break;
            case Instruction::GateType::SDG:
                inv_inst.type = Instruction::GateType::S;
                break;

            // T <-> TDG
            case Instruction::GateType::T:
                inv_inst.type = Instruction::GateType::TDG;
                break;
            case Instruction::GateType::TDG:
                inv_inst.type = Instruction::GateType::T;
                break;

            // SX <-> SXDG
            case Instruction::GateType::SX:
                inv_inst.type = Instruction::GateType::SXDG;
                break;
            case Instruction::GateType::SXDG:
                inv_inst.type = Instruction::GateType::SX;
                break;

            // Negate parameters for rotation gates
            case Instruction::GateType::RX:
            case Instruction::GateType::RY:
            case Instruction::GateType::RZ:
            case Instruction::GateType::CRX:
            case Instruction::GateType::CRY:
            case Instruction::GateType::CRZ:
            case Instruction::GateType::RXX:
            case Instruction::GateType::RYY:
            case Instruction::GateType::RZZ:
            case Instruction::GateType::RZX:
                for (auto& param : inv_inst.params) {
                    param = -param;
                }
                break;

            // Phase gates: negate
            case Instruction::GateType::P:
            case Instruction::GateType::U1:
            case Instruction::GateType::CP:
            case Instruction::GateType::MCP:  // multi-controlled phase: negate lambda
                for (auto& param : inv_inst.params) {
                    param = -param;
                }
                break;

            // Permutation inverse: invert the index map (perm[x]=y -> inv[y]=x).
            case Instruction::GateType::PERMUTATION: {
                std::vector<int> inv(inv_inst.permutation.size());
                for (size_t x = 0; x < inv_inst.permutation.size(); ++x)
                    inv[static_cast<size_t>(inv_inst.permutation[x])] =
                        static_cast<int>(x);
                inv_inst.permutation = std::move(inv);
                break;
            }

            // U(theta, phi, lambda) → U(-theta, -lambda, -phi)
            case Instruction::GateType::U:
            case Instruction::GateType::U3: {
                double theta = inv_inst.params[0];
                double phi = inv_inst.params[1];
                double lam = inv_inst.params[2];
                inv_inst.params = {-theta, -lam, -phi};
                break;
            }

            // U2(phi, lambda) → U2(-lambda - pi, -phi + pi)
            case Instruction::GateType::U2: {
                double phi = inv_inst.params[0];
                double lam = inv_inst.params[1];
                inv_inst.params = {-lam - PI, -phi + PI};
                break;
            }

            // CU(theta, phi, lambda, gamma) → CU(-theta, -lambda, -phi, -gamma)
            case Instruction::GateType::CU: {
                double theta = inv_inst.params[0];
                double phi   = inv_inst.params[1];
                double lam   = inv_inst.params[2];
                double gamma = inv_inst.params[3];
                inv_inst.params = {-theta, -lam, -phi, -gamma};
                break;
            }

            // ISWAP† is the conjugate-transpose (negate the i phase)
            // iSWAP = diag(1,0,0,1) + i*offdiag → iSWAP† = diag(1,0,0,1) - i*offdiag
            // Decompose as S†(0) S†(1) SWAP CZ which is always available
            // For now, emit as UNITARY with the conjugate-transposed matrix
            case Instruction::GateType::ISWAP: {
                inv_inst.type = Instruction::GateType::UNITARY;
                inv_inst.matrix = {
                    Complex128(1,0), Complex128(0,0), Complex128(0,0), Complex128(0,0),
                    Complex128(0,0), Complex128(0,0), Complex128(0,-1), Complex128(0,0),
                    Complex128(0,0), Complex128(0,-1), Complex128(0,0), Complex128(0,0),
                    Complex128(0,0), Complex128(0,0), Complex128(0,0), Complex128(1,0)
                };
                break;
            }

            case Instruction::GateType::UNITARY: {
                size_t dim = static_cast<size_t>(
                    std::round(std::sqrt(static_cast<double>(inv_inst.matrix.size()))));
                std::vector<Complex128> conj_t(dim * dim);
                for (size_t r = 0; r < dim; ++r)
                    for (size_t c = 0; c < dim; ++c)
                        conj_t[r * dim + c] = Complex128(
                             inv_inst.matrix[c * dim + r].real,
                            -inv_inst.matrix[c * dim + r].imag);
                inv_inst.matrix = std::move(conj_t);
                break;
            }

            // Symbolic gates cannot be inverted without resolving parameters; skip.
            case Instruction::GateType::PARAM_RX:
            case Instruction::GateType::PARAM_RY:
            case Instruction::GateType::PARAM_RZ:
            case Instruction::GateType::PARAM_P:
            case Instruction::GateType::PARAM_U:
                continue;

            default:
                throw std::runtime_error(
                    "inverse() not implemented for gate: " + inv_inst.gate_name()
                );
        }

        result.instructions.push_back(std::move(inv_inst));
    }

    return result;
}

QuantumCircuit QuantumCircuit::repeat(int n) const {
    QuantumCircuit result(n_qubits, n_clbits);
    for (int i = 0; i < n; ++i) {
        for (const auto& inst : instructions) {
            result.instructions.push_back(inst);
        }
    }
    return result;
}

// =============================================================================
// control() — build a controlled version of the circuit
// =============================================================================
// Adds num_ctrl_qubits control qubits (indices [0, num_ctrl_qubits)).
// Original circuit qubits are shifted up by num_ctrl_qubits.
// Each gate is replaced by its controlled counterpart.

QuantumCircuit QuantumCircuit::control(int num_ctrl_qubits) const {
    if (num_ctrl_qubits < 1)
        throw std::invalid_argument("control: num_ctrl_qubits must be >= 1");

    int total_qubits = n_qubits + num_ctrl_qubits;
    QuantumCircuit result(total_qubits, n_clbits);
    result.name = name + "_ctrl";

    using GT = Instruction::GateType;

    for (const auto& inst : instructions) {
        // Shift all original qubit indices up
        std::vector<int> shifted_qubits;
        for (int q : inst.qubits) shifted_qubits.push_back(q + num_ctrl_qubits);

        // Skip non-gate operations (measure, reset, barrier pass through shifted)
        if (inst.type == GT::BARRIER) {
            Instruction out = inst;
            out.qubits = shifted_qubits;
            result.instructions.push_back(std::move(out));
            continue;
        }
        if (inst.type == GT::MEASURE || inst.type == GT::RESET) {
            Instruction out = inst;
            out.qubits = shifted_qubits;
            result.instructions.push_back(std::move(out));
            continue;
        }

        // Generic controlled gate. Controls occupy ci.qubits[0..nc) and are
        // therefore the LOW bits of the matrix index under the project
        // convention (apply_unitary: index bit i = qubits[i]); the original
        // gate's qubits follow as the HIGH bits. The U block lives on the
        // index slice whose control bits are ALL 1 (interleaved layout, the
        // same structure as the hand-built controlled-U in shor.cpp/qpe.cpp):
        //   M[(2^nc - 1) | (r << nc), (2^nc - 1) | (c << nc)] = U[r][c]
        // and identity on every other diagonal entry. Gates without a known
        // matrix cannot be controlled silently: parameterised placeholders
        // throw instead of degrading to identity (no silent failures).
        auto emit_generic_ctrl = [&]() {
            if (inst.is_parameterised())
                throw std::runtime_error(
                    "control(): cannot control an unresolved parameterised gate '" +
                    inst.gate_name() + "'; call assign_parameters() first");

            const int gate_qubits = static_cast<int>(inst.qubits.size());
            const int total_gate_qubits = gate_qubits + num_ctrl_qubits;
            const size_t gate_dim = 1ULL << gate_qubits;
            const size_t total_dim = 1ULL << total_gate_qubits;

            std::vector<Complex128> gate_matrix;
            if (inst.type == GT::UNITARY) {
                if (inst.matrix.size() != gate_dim * gate_dim)
                    throw std::runtime_error(
                        "control(): UNITARY instruction matrix size mismatch");
                gate_matrix = inst.matrix;
            } else {
                // Extract the named gate's matrix (LSB convention: index bit i
                // = local qubit i) by simulating the lone instruction on a
                // standalone register with operand order preserved.
                QuantumCircuit local(gate_qubits);
                Instruction local_inst = inst;
                local_inst.condition_clbit = -1;  // raw gate action only
                local_inst.qubits.clear();
                for (int i = 0; i < gate_qubits; ++i)
                    local_inst.qubits.push_back(i);
                local.instructions.push_back(std::move(local_inst));
                gate_matrix = Operator::from_circuit(local).data;
            }

            std::vector<Complex128> ctrl_matrix(total_dim * total_dim, Complex128(0, 0));
            const size_t ctrl_mask = (1ULL << num_ctrl_qubits) - 1;
            for (size_t s = 0; s < total_dim; ++s)
                if ((s & ctrl_mask) != ctrl_mask)
                    ctrl_matrix[s * total_dim + s] = Complex128(1, 0);
            for (size_t r = 0; r < gate_dim; ++r)
                for (size_t c = 0; c < gate_dim; ++c)
                    ctrl_matrix[(ctrl_mask | (r << num_ctrl_qubits)) * total_dim +
                                (ctrl_mask | (c << num_ctrl_qubits))] =
                        gate_matrix[r * gate_dim + c];

            Instruction ci;
            ci.type = GT::UNITARY;
            ci.matrix = std::move(ctrl_matrix);
            for (int k = 0; k < num_ctrl_qubits; ++k) ci.qubits.push_back(k);
            for (int q : shifted_qubits) ci.qubits.push_back(q);
            ci.label = "c_" + inst.gate_name();
            ci.condition_clbit = inst.condition_clbit;
            ci.condition_value = inst.condition_value;
            // The policy governs the matrix, and the controlled matrix is
            // unitary exactly when the block it was built from is: the control
            // structure contributes identity on every unselected slice, which
            // changes no residual. A caller who opted out of the check on the
            // source matrix has therefore opted out of the same check here, and
            // dropping the field would make a legal circuit unrunnable by the
            // act of controlling it.
            ci.validation = inst.validation;
            result.instructions.push_back(std::move(ci));
        };

        if (num_ctrl_qubits == 1) {
            int ctrl = 0;
            if (inst.qubits.size() == 1) {
                int tgt = shifted_qubits[0];
                Instruction ci;
                ci.qubits = {ctrl, tgt};
                ci.params = inst.params;
                ci.condition_clbit = inst.condition_clbit;
                ci.condition_value = inst.condition_value;
                ci.validation = inst.validation;
                bool handled = true;
                switch (inst.type) {
                    case GT::X:   ci.type = GT::CX;  break;
                    case GT::Y:   ci.type = GT::CY;  break;
                    case GT::Z:   ci.type = GT::CZ;  break;
                    case GT::H:   ci.type = GT::CH;  break;
                    case GT::RX:  ci.type = GT::CRX; break;
                    case GT::RY:  ci.type = GT::CRY; break;
                    case GT::RZ:  ci.type = GT::CRZ; break;
                    case GT::P: case GT::U1:
                        ci.type = GT::CP; break;
                    case GT::S:   ci.type = GT::CP; ci.params = {PI_2};  break;
                    case GT::SDG: ci.type = GT::CP; ci.params = {-PI_2}; break;
                    case GT::T:   ci.type = GT::CP; ci.params = {PI_4};  break;
                    case GT::TDG: ci.type = GT::CP; ci.params = {-PI_4}; break;
                    case GT::U: case GT::U3:
                        ci.type = GT::CU;
                        ci.params = {inst.params[0], inst.params[1], inst.params[2], 0.0};
                        break;
                    default:
                        handled = false; break;
                }
                if (handled) {
                    result.instructions.push_back(std::move(ci));
                } else {
                    emit_generic_ctrl();
                }
                continue;
            }

            if (inst.qubits.size() == 2) {
                int q0 = shifted_qubits[0], q1 = shifted_qubits[1];
                switch (inst.type) {
                    case GT::CX: {
                        Instruction ci; ci.type = GT::CCX; ci.qubits = {ctrl, q0, q1};
                        ci.condition_clbit = inst.condition_clbit;
                        ci.condition_value = inst.condition_value;
                        ci.validation = inst.validation;
                        result.instructions.push_back(std::move(ci));
                        continue;
                    }
                    case GT::SWAP: {
                        Instruction ci; ci.type = GT::CSWAP; ci.qubits = {ctrl, q0, q1};
                        ci.condition_clbit = inst.condition_clbit;
                        ci.condition_value = inst.condition_value;
                        ci.validation = inst.validation;
                        result.instructions.push_back(std::move(ci));
                        continue;
                    }
                    default:
                        emit_generic_ctrl();
                        continue;
                }
            }
        }

        // num_ctrl_qubits > 1, or size >= 3 with num_ctrl == 1
        emit_generic_ctrl();
    }

    return result;
}

// =============================================================================
// Analysis
// =============================================================================

int QuantumCircuit::depth() const {
    if (instructions.empty()) return 0;

    // Track the depth at each qubit wire
    std::vector<int> qubit_depth(n_qubits, 0);

    for (const auto& inst : instructions) {
        if (inst.type == Instruction::GateType::BARRIER) continue;

        // Find the max depth among all qubits this instruction touches
        int max_d = 0;
        for (int q : inst.qubits) {
            max_d = std::max(max_d, qubit_depth[q]);
        }

        // Increment and assign to all touched qubits
        for (int q : inst.qubits) {
            qubit_depth[q] = max_d + 1;
        }
    }

    return *std::max_element(qubit_depth.begin(), qubit_depth.end());
}

int QuantumCircuit::size() const {
    int count = 0;
    for (const auto& inst : instructions) {
        if (inst.type != Instruction::GateType::BARRIER) {
            count++;
        }
    }
    return count;
}

std::unordered_map<std::string, int> QuantumCircuit::count_ops() const {
    std::unordered_map<std::string, int> counts;
    for (const auto& inst : instructions) {
        counts[inst.gate_name()]++;
    }
    return counts;
}

int QuantumCircuit::num_parameters() const {
    return static_cast<int>(parameter_names.size());
}

// =============================================================================
// QASM Export (basic QASM 2.0)
// =============================================================================

namespace {

// Does a UNITARY get lowered by this exporter, under these options?
//
// `format_lowers_by_default` carries the one thing the option cannot know: what
// the target format can represent. QASM 2 passes false (no literal-matrix
// syntax, and its lowering discards any global phase), QASM 3 passes true
// (`gphase` makes the same lowering exact). Always and Never override both in
// the same direction, so the enum value means one thing wherever it is read.
bool lower_unitary_here(const QasmExportOptions& opts,
                        bool format_lowers_by_default) {
    switch (opts.unitary_lowering) {
        case UnitaryLowering::Always: return true;
        case UnitaryLowering::Never:  return false;
        case UnitaryLowering::FormatDefault: break;
    }
    return format_lowers_by_default;
}

// Whether this export writes a classical condition into its text.
bool emit_conditions_here(const QasmExportOptions& opts,
                          bool format_emits_by_default) {
    switch (opts.condition_export) {
        case ConditionExport::Always: return true;
        case ConditionExport::Never:  return false;
        case ConditionExport::FormatDefault: break;
    }
    return format_emits_by_default;
}

// A QASM 2 `if` takes one quantum operation, not a block, so an instruction
// that lowered into several gates gets one `if` per emitted statement. That is
// equivalent to conditioning the group: lowering produces quantum gates only,
// none of which writes a classical bit, so the condition cannot change between
// the first statement and the last.
//
// Two line kinds are left alone. A comment carries no operation, and
// `if (...) // text` is not a statement. A `barrier` is a directive rather
// than a qop, which the OpenQASM 2.0 grammar does not admit after an `if`.
std::string qasm2_condition_prefix(const std::string& body, int value) {
    const std::string guard = "if (c == " + std::to_string(value) + ") ";
    std::ostringstream out;
    std::size_t pos = 0;
    while (pos < body.size()) {
        const std::size_t eol = body.find('\n', pos);
        const std::size_t end = (eol == std::string::npos) ? body.size() : eol + 1;
        const std::string line = body.substr(pos, end - pos);
        const std::size_t first = line.find_first_not_of(" \t");
        const bool statement =
            first != std::string::npos &&
            line.compare(first, 2, "//") != 0 &&
            line.compare(first, 7, "barrier") != 0;
        if (statement) out << line.substr(0, first) << guard << line.substr(first);
        else           out << line;
        pos = end;
    }
    return out.str();
}

// QASM 3 conditions a block, so the instruction's text goes inside one whatever
// it lowered into.
std::string qasm3_condition_block(const std::string& body, int clbit, int value) {
    std::ostringstream out;
    out << "if (c[" << clbit << "] == " << value << ") {\n";
    std::size_t pos = 0;
    while (pos < body.size()) {
        const std::size_t eol = body.find('\n', pos);
        const std::size_t end = (eol == std::string::npos) ? body.size() : eol + 1;
        const std::string line = body.substr(pos, end - pos);
        if (line.find_first_not_of(" \t\n") != std::string::npos) out << "  ";
        out << line;
        pos = end;
    }
    if (!body.empty() && body.back() != '\n') out << "\n";
    out << "}\n";
    return out.str();
}

// One instruction's text is built in `body` and moved into `out` when this goes
// out of scope. The flush being a destructor is what lets every emission branch
// end an instruction with `continue`, as they all did before conditions were
// written at all. clbit < 0 means the text is emitted as it stands.
struct EmittedInstruction {
    std::ostringstream& out;
    std::ostringstream& body;
    bool qasm3;
    int clbit;
    int value;

    ~EmittedInstruction() {
        const std::string text = body.str();
        if (clbit < 0) { out << text; return; }
        out << (qasm3 ? qasm3_condition_block(text, clbit, value)
                      : qasm2_condition_prefix(text, value));
    }
};

// Above two qubits no exact decomposition exists anywhere in this project, so
// no export option can make the operand representable. Both exporters say so
// in the same words, and neither names a flag, because naming one would send
// the caller to try something that cannot work.
[[noreturn]] void throw_unitary_too_wide(const char* fn, std::size_t width,
                                         const std::string& gname) {
    throw std::runtime_error(
        std::string(fn) + ": the " + std::to_string(width) + "-qubit unitary '" +
        gname + "' cannot be exported: no exact lowering above two qubits "
        "exists. Use to_json() for a lossless round trip, or decompose it into "
        "standard gates before exporting");
}

}  // namespace

std::string QuantumCircuit::to_qasm2(const QasmExportOptions& opts) const {
    std::ostringstream out;
    out << "OPENQASM 2.0;\n";
    out << "include \"qelib1.inc\";\n";
    out << "qreg q[" << n_qubits << "];\n";
    if (n_clbits > 0) {
        out << "creg c[" << n_clbits << "];\n";
    }

    // R.1.18.0: opt-in export-time lowering of MCX/MCP/PERMUTATION. The
    // lowered alphabet ({X, H, P, CP, CX, CCX, SWAP}) is fully covered by the
    // generic emission paths below, so pre-expanding the instruction list is
    // all that is needed. The circuit object itself is never modified.
    std::vector<Instruction> expanded;
    const std::vector<Instruction>* insts = &instructions;
    if (opts.decompose_unrepresentable) {
        bool any = false;
        for (const auto& inst : instructions) {
            if (hld::is_high_level(inst)) { any = true; break; }
        }
        if (any) {
            for (const auto& inst : instructions) {
                if (hld::is_high_level(inst)) {
                    std::vector<Instruction> low = hld::lower_fully(inst);
                    expanded.insert(expanded.end(),
                                    std::make_move_iterator(low.begin()),
                                    std::make_move_iterator(low.end()));
                } else {
                    expanded.push_back(inst);
                }
            }
            insts = &expanded;
        }
    }

    for (const auto& inst : *insts) {
        // Built in its own buffer so a classical condition can wrap the
        // finished text; `emitted` flushes it into `out` on scope exit.
        std::ostringstream oss;
        int cond_clbit = -1;
        if (inst.condition_clbit >= 0) {
            if (!emit_conditions_here(opts, /*format_emits_by_default=*/false)) {
                throw std::runtime_error(
                    "to_qasm2: the instruction '" + inst.gate_name() + "' carries "
                    "a classical condition. OpenQASM 2.0's `if` compares a whole "
                    "classical register rather than one bit, so a single-bit "
                    "condition has an exact spelling here only when the register "
                    "IS that one bit. Set QasmExportOptions::condition_export = "
                    "ConditionExport::Always to export anyway, which takes the "
                    "exact spelling where it exists and otherwise drops the "
                    "condition and records it in a comment. to_qasm3() and "
                    "to_json() both carry it exactly");
            }
            if (n_clbits == 1) {
                cond_clbit = inst.condition_clbit;
            } else {
                emit_warning(
                    "to_qasm2: dropped the classical condition c[" +
                    std::to_string(inst.condition_clbit) + "] == " +
                    std::to_string(inst.condition_value) + " on '" +
                    inst.gate_name() + "', because OpenQASM 2.0 conditions a "
                    "whole classical register and this circuit declares " +
                    std::to_string(n_clbits) + " classical bits");
                oss << "// dropped condition: c[" << inst.condition_clbit
                    << "] == " << inst.condition_value
                    << " (OpenQASM 2.0 conditions a whole classical register)"
                    << "\n";
            }
        }
        EmittedInstruction emitted{out, oss, /*qasm3=*/false, cond_clbit,
                                   inst.condition_value};

        std::string gname = inst.gate_name();

        if (inst.type == Instruction::GateType::BARRIER) {
            oss << "barrier";
            for (size_t i = 0; i < inst.qubits.size(); ++i) {
                if (i > 0) oss << ",";
                oss << " q[" << inst.qubits[i] << "]";
            }
            oss << ";\n";
            continue;
        }

        if (inst.type == Instruction::GateType::MEASURE) {
            oss << "measure q[" << inst.qubits[0] << "] -> c[" << inst.clbits[0] << "];\n";
            continue;
        }

        if (inst.type == Instruction::GateType::RESET) {
            oss << "reset q[" << inst.qubits[0] << "];\n";
            continue;
        }

        // MCX/MCP/PERMUTATION have no faithful qelib1 representation (QASM 2
        // has no gate modifiers and no permutation encoding); refuse rather
        // than emit silently-wrong QASM. Reachable only on the default path:
        // decompose_unrepresentable pre-expanded these above.
        if (inst.type == Instruction::GateType::MCX ||
            inst.type == Instruction::GateType::MCP ||
            inst.type == Instruction::GateType::PERMUTATION) {
            throw std::runtime_error(
                "to_qasm2: " + gname + " is not representable in OpenQASM 2.0; "
                "set QasmExportOptions::decompose_unrepresentable to lower it "
                "at export, or use to_qasm3()");
        }

        // UNITARY, at every width. OpenQASM 2.0 has no literal-matrix syntax,
        // so the operand has no spelling of its own and must be either lowered
        // into standard gates or refused.
        //
        // TWO SEPARATE CONSENTS are involved, and they are not the same thing.
        // Lowering RESTRUCTURES the circuit into gates the caller never wrote.
        // Lowering also cannot carry a global phase, because the single-qubit
        // corrections land in U(theta, phi, lambda), which spans SU(2) rather
        // than U(2), and QASM 2 has nowhere to put the remainder. The first is
        // governed by unitary_lowering, the second by accept_global_phase_loss.
        // Keeping them apart lets a caller permit restructuring while still
        // refusing to lose information, which is a real position: an operand
        // already in SU(2) lowers exactly and no loss arises at all.
        if (inst.type == Instruction::GateType::UNITARY) {
            const std::size_t width = inst.qubits.size();
            if (width >= 3) throw_unitary_too_wide("to_qasm2", width, gname);

            if (!lower_unitary_here(opts, /*format_lowers_by_default=*/false)) {
                throw std::runtime_error(
                    "to_qasm2: the " + std::to_string(width) + "-qubit unitary '" +
                    gname + "' is not representable in OpenQASM 2.0 (no "
                    "literal-matrix syntax). Set QasmExportOptions::"
                    "unitary_lowering = UnitaryLowering::Always to lower it into "
                    "standard gates at export; that lowering cannot carry a "
                    "global phase, so set accept_global_phase_loss as well if the "
                    "operand has one. to_json() round-trips losslessly instead");
            }

            std::vector<Instruction> lowered;
            double alpha = 0.0;

            if (width == 1) {
                const auto& m = inst.matrix;
                double a_mag = std::sqrt(m[0].real*m[0].real + m[0].imag*m[0].imag);
                double b_mag = std::sqrt(m[1].real*m[1].real + m[1].imag*m[1].imag);
                double theta = 2.0 * std::atan2(b_mag, a_mag);
                double phi = 0.0, lambda = 0.0;
                if (a_mag > 1e-9) {
                    alpha = std::atan2(m[0].imag, m[0].real);
                    if (b_mag > 1e-9) {
                        lambda = std::atan2(-m[1].imag, -m[1].real) - alpha;
                        phi    = std::atan2( m[2].imag,  m[2].real) - alpha;
                    } else {
                        // theta ~ 0: entire phase lives in d
                        lambda = std::atan2(m[3].imag, m[3].real) - alpha;
                    }
                } else {
                    // theta ~ pi: read phases directly from b and c; alpha
                    // undefined, pick a canonical SU(2) representative.
                    lambda = std::atan2(-m[1].imag, -m[1].real);
                    phi    = std::atan2( m[2].imag,  m[2].real);
                }
                Instruction u;
                u.type = Instruction::GateType::U;
                u.qubits = inst.qubits;
                u.params = {theta, phi, lambda};
                u.condition_clbit = inst.condition_clbit;
                u.condition_value = inst.condition_value;
                u.validation = inst.validation;
                lowered.push_back(std::move(u));
            } else {
                const auto low = tqd::lower_2q_unitary(inst);
                if (!low) {
                    throw std::runtime_error(
                        "to_qasm2: the 2-qubit unitary '" + gname + "' could not "
                        "be decomposed: the decomposition did not reproduce the "
                        "operand. Use to_json() for a lossless round trip");
                }
                lowered = low->instructions;
                alpha = low->global_phase;
            }

            if (std::abs(alpha) > 1e-9) {
                if (!opts.accept_global_phase_loss) {
                    throw std::runtime_error(
                        "to_qasm2: lowering the " + std::to_string(width) +
                        "-qubit unitary '" + gname + "' would drop a global phase "
                        "of " + std::to_string(alpha) + " radians, which OpenQASM "
                        "2.0 cannot represent. Set QasmExportOptions::"
                        "accept_global_phase_loss to proceed and record the loss "
                        "in a comment, or use to_qasm3(), which carries it exactly "
                        "via gphase");
                }
                // One warning per UNITARY OPERAND whose phase is dropped,
                // not one per gate the lowering emits: the loss is a fact about
                // the operand, and counting per emitted gate would tie the
                // warning count to how many gates the decomposition happens to
                // produce. It goes through the shared sink so a caller can
                // capture or silence it, and the comment below puts the same
                // number in the text so the loss survives in the file itself.
                //
                // Delivery is a separate matter: the warning channel
                // deduplicates on message text, so several operands dropping an
                // identical phase are delivered once and counted.
                emit_warning(
                    "to_qasm2: dropped a global phase of " +
                    std::to_string(alpha) + " radians lowering the " +
                    std::to_string(width) + "-qubit unitary '" + gname + "'");
                oss << "// global phase: " << std::setprecision(15) << alpha
                    << " (dropped: OpenQASM 2.0 cannot represent global phase; "
                    << "use to_qasm3() for a lossless round trip)\n";
            }

            for (const auto& g : lowered) {
                oss << g.gate_name();
                if (!g.params.empty()) {
                    oss << "(";
                    for (size_t i = 0; i < g.params.size(); ++i) {
                        if (i > 0) oss << ",";
                        oss << std::setprecision(15) << g.params[i];
                    }
                    oss << ")";
                }
                for (size_t i = 0; i < g.qubits.size(); ++i) {
                    if (i > 0) oss << ",";
                    oss << " q[" << g.qubits[i] << "]";
                }
                oss << ";\n";
            }
            continue;
        }

        // Symbolic PARAM_* gates: emit with the symbolic parameter name as argument.
        // QASM 2.0 parsers accept these in gate-body scope; the expression is
        // preserved in the output string even if full re-parse requires binding.
        if (inst.type == Instruction::GateType::PARAM_U) {
            oss << "u(";
            for (size_t i = 0; i < 3; ++i) {
                if (i) oss << ",";
                oss << (i < inst.param_names.size() ? inst.param_names[i] : "0");
            }
            oss << ") q[" << inst.qubits[0] << "];\n";
            continue;
        }
        if (inst.type == Instruction::GateType::PARAM_RX ||
            inst.type == Instruction::GateType::PARAM_RY ||
            inst.type == Instruction::GateType::PARAM_RZ ||
            inst.type == Instruction::GateType::PARAM_P) {
            const std::string& pname = inst.param_names.empty() ? "0" : inst.param_names[0];
            oss << gname << "(" << pname << ") q[" << inst.qubits[0] << "];\n";
            continue;
        }

        oss << gname;
        if (!inst.params.empty()) {
            oss << "(";
            for (size_t i = 0; i < inst.params.size(); ++i) {
                if (i > 0) oss << ",";
                oss << std::setprecision(15) << inst.params[i];
            }
            oss << ")";
        }
        for (size_t i = 0; i < inst.qubits.size(); ++i) {
            if (i > 0) oss << ",";
            oss << " q[" << inst.qubits[i] << "]";
        }
        oss << ";\n";
    }

    return out.str();
}

std::string QuantumCircuit::to_qasm3(const QasmExportOptions& opts) const {
    std::ostringstream out;
    out << "OPENQASM 3.0;\n";
    out << "include \"stdgates.inc\";\n";
    out << "qubit[" << n_qubits << "] q;\n";
    if (n_clbits > 0) {
        out << "bit[" << n_clbits << "] c;\n";
    }

    // R.1.18.0: PERMUTATION has no QASM 3 primitive, so it is ALWAYS lowered
    // at export — to SWAPs when the basis map is a pure wire relabeling, to an
    // exact transposition synthesis otherwise. The one-level lowering keeps
    // MCX nodes, which the loop below emits compactly via `ctrl @`. MCX / MCP
    // themselves are representable and are NOT pre-expanded.
    std::vector<Instruction> expanded;
    const std::vector<Instruction>* insts = &instructions;
    {
        bool any_perm = false;
        for (const auto& inst : instructions) {
            if (inst.type == Instruction::GateType::PERMUTATION) { any_perm = true; break; }
        }
        if (any_perm) {
            for (const auto& inst : instructions) {
                if (inst.type == Instruction::GateType::PERMUTATION) {
                    std::vector<Instruction> low =
                        hld::lower_permutation(inst.permutation, inst.qubits);
                    if (inst.condition_clbit >= 0) {
                        for (Instruction& g : low) {
                            g.condition_clbit = inst.condition_clbit;
                            g.condition_value = inst.condition_value;
                        }
                    }
                    expanded.insert(expanded.end(),
                                    std::make_move_iterator(low.begin()),
                                    std::make_move_iterator(low.end()));
                } else {
                    expanded.push_back(inst);
                }
            }
            insts = &expanded;
        }
    }

    for (const auto& inst : *insts) {
        // As in to_qasm2: one buffer per instruction, wrapped on scope exit.
        std::ostringstream oss;
        int cond_clbit = -1;
        if (inst.condition_clbit >= 0) {
            if (!emit_conditions_here(opts, /*format_emits_by_default=*/true)) {
                throw std::runtime_error(
                    "to_qasm3: the instruction '" + inst.gate_name() + "' carries "
                    "a classical condition and QasmExportOptions::condition_export "
                    "is Never. QASM 3 writes it exactly as an if block, so clear "
                    "that setting to export it");
            }
            cond_clbit = inst.condition_clbit;
        }
        EmittedInstruction emitted{out, oss, /*qasm3=*/true, cond_clbit,
                                   inst.condition_value};

        std::string gname = inst.gate_name();

        if (inst.type == Instruction::GateType::BARRIER) {
            oss << "barrier";
            for (size_t i = 0; i < inst.qubits.size(); ++i) {
                if (i > 0) oss << ",";
                oss << " q[" << inst.qubits[i] << "]";
            }
            oss << ";\n";
            continue;
        }

        if (inst.type == Instruction::GateType::MEASURE) {
            oss << "c[" << inst.clbits[0] << "] = measure q[" << inst.qubits[0] << "];\n";
            continue;
        }

        if (inst.type == Instruction::GateType::RESET) {
            oss << "reset q[" << inst.qubits[0] << "];\n";
            continue;
        }

        // MCX: `ctrl(k) @ x` with the controls listed first, matching the
        // instruction's [controls..., target] operand order. Round-trip: the
        // parser's ctrl-stack fast path restores MCX for k ≥ 3; k ≤ 2
        // canonicalises to the named cx / ccx forms (identical unitaries).
        // A control-free mcx degenerates to a plain x.
        if (inst.type == Instruction::GateType::MCX) {
            const size_t k = inst.qubits.size() - 1;
            if (k == 0) {
                oss << "x q[" << inst.qubits[0] << "];\n";
                continue;
            }
            oss << "ctrl(" << k << ") @ x";
            for (size_t i = 0; i < inst.qubits.size(); ++i) {
                oss << (i == 0 ? " " : ", ") << "q[" << inst.qubits[i] << "]";
            }
            oss << ";\n";
            continue;
        }

        // MCP: the phase lands on the all-ones subspace, so any operand can
        // serve as the modifier's base target; the last one is chosen. A
        // single-qubit MCP degenerates to a plain p(λ).
        if (inst.type == Instruction::GateType::MCP) {
            const double lambda = inst.params.empty() ? 0.0 : inst.params[0];
            const size_t m = inst.qubits.size();
            if (m == 1) {
                oss << "p(" << std::setprecision(15) << lambda << ") q["
                    << inst.qubits[0] << "];\n";
                continue;
            }
            oss << "ctrl(" << (m - 1) << ") @ p("
                << std::setprecision(15) << lambda << ")";
            for (size_t i = 0; i < m; ++i) {
                oss << (i == 0 ? " " : ", ") << "q[" << inst.qubits[i] << "]";
            }
            oss << ";\n";
            continue;
        }

        // A UNITARY wider than one qubit. QASM 3 has no literal-matrix syntax
        // either, so it must be lowered or refused; falling through to the
        // generic tail would write a call to a gate the file never defines and
        // drop the matrix entirely (issue #79).
        if (inst.type == Instruction::GateType::UNITARY && inst.qubits.size() >= 2) {
            const std::size_t width = inst.qubits.size();
            if (width >= 3) throw_unitary_too_wide("to_qasm3", width, gname);

            if (!lower_unitary_here(opts, /*format_lowers_by_default=*/true)) {
                throw std::runtime_error(
                    "to_qasm3: the 2-qubit unitary '" + gname + "' was not "
                    "lowered because QasmExportOptions::unitary_lowering is "
                    "Never. QASM 3 has no literal-matrix syntax, so the operand "
                    "has no other spelling; clear that setting to lower it "
                    "exactly, or use to_json()");
            }

            const auto lowered = tqd::lower_2q_unitary(inst);
            if (!lowered) {
                throw std::runtime_error(
                    "to_qasm3: the 2-qubit unitary '" + gname + "' could not be "
                    "decomposed: the decomposition did not reproduce the "
                    "operand. Use to_json() for a lossless round trip");
            }

            // `gphase` is what makes this exact rather than merely equivalent:
            // the emitted single-qubit corrections span SU(2), so the operand's
            // global phase has to be restored explicitly. QASM 2 has nowhere to
            // put this, which is the whole reason the two formats differ here.
            if (std::abs(lowered->global_phase) > 1e-9) {
                oss << "gphase(" << std::setprecision(15)
                    << lowered->global_phase << ");\n";
            }
            for (const auto& g : lowered->instructions) {
                oss << g.gate_name();
                if (!g.params.empty()) {
                    oss << "(";
                    for (size_t i = 0; i < g.params.size(); ++i) {
                        if (i > 0) oss << ", ";
                        oss << std::setprecision(15) << g.params[i];
                    }
                    oss << ")";
                }
                for (size_t i = 0; i < g.qubits.size(); ++i) {
                    if (i > 0) oss << ",";
                    oss << " q[" << g.qubits[i] << "]";
                }
                oss << ";\n";
            }
            continue;
        }

        // 1-qubit UNITARY: Euler-angle decomposition + lossless global phase
        // via gphase. Unlike QASM 2.0, QASM 3 has a first-class `gphase` that
        // commutes with everything and is promoted to a relative phase under
        // `ctrl @` modifiers, so the round-trip is exact.
        if (inst.type == Instruction::GateType::UNITARY && inst.qubits.size() == 1) {
            const auto& m = inst.matrix;
            double a_mag = std::sqrt(m[0].real*m[0].real + m[0].imag*m[0].imag);
            double b_mag = std::sqrt(m[1].real*m[1].real + m[1].imag*m[1].imag);
            double theta = 2.0 * std::atan2(b_mag, a_mag);
            double phi = 0.0, lambda = 0.0, alpha = 0.0;
            if (a_mag > 1e-9) {
                alpha = std::atan2(m[0].imag, m[0].real);
                if (b_mag > 1e-9) {
                    lambda = std::atan2(-m[1].imag, -m[1].real) - alpha;
                    phi    = std::atan2( m[2].imag,  m[2].real) - alpha;
                } else {
                    lambda = std::atan2(m[3].imag, m[3].real) - alpha;
                }
            } else {
                lambda = std::atan2(-m[1].imag, -m[1].real);
                phi    = std::atan2( m[2].imag,  m[2].real);
            }
            if (std::abs(alpha) > 1e-9) {
                oss << "gphase(" << std::setprecision(15) << alpha << ");\n";
            }
            oss << "u(" << std::setprecision(15) << theta
                << ", " << phi << ", " << lambda
                << ") q[" << inst.qubits[0] << "];\n";
            continue;
        }

        oss << gname;
        if (!inst.params.empty()) {
            oss << "(";
            for (size_t i = 0; i < inst.params.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << std::setprecision(15) << inst.params[i];
            }
            oss << ")";
        }
        for (size_t i = 0; i < inst.qubits.size(); ++i) {
            if (i > 0) oss << ",";
            oss << " q[" << inst.qubits[i] << "]";
        }
        oss << ";\n";
    }

    return out.str();
}

// Forward declarations of the bridge functions in qasm{2,3}_parser.cpp
QuantumCircuit qasm2_parse_impl(const std::string& qasm);
QuantumCircuit qasm3_parse_impl(const std::string& qasm);

QuantumCircuit QuantumCircuit::from_qasm2(const std::string& qasm) {
    return qasm2_parse_impl(qasm);
}

QuantumCircuit QuantumCircuit::from_qasm3(const std::string& qasm) {
    return qasm3_parse_impl(qasm);
}

// =============================================================================
// JSON serialization — zero-dependency
// =============================================================================

// Helper: escape a string for JSON output
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    out += '"';
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\t': out += "\\t";  break;
            case '\r': out += "\\r";  break;
            default:   out += c;      break;
        }
    }
    out += '"';
    return out;
}

// GateType enum -> string (reuse gate_name() for OP types, explicit for specials)
static std::string gate_type_to_str(Instruction::GateType t) {
    // Create a temp instruction to leverage gate_name()
    Instruction tmp;
    tmp.type = t;
    return tmp.gate_name();
}

// String -> GateType reverse lookup
static Instruction::GateType str_to_gate_type(const std::string& s) {
    using GT = Instruction::GateType;
    static const std::unordered_map<std::string, GT> map = {
        {"h", GT::H}, {"x", GT::X}, {"y", GT::Y}, {"z", GT::Z},
        {"s", GT::S}, {"sdg", GT::SDG}, {"t", GT::T}, {"tdg", GT::TDG},
        {"sx", GT::SX}, {"sxdg", GT::SXDG},
        {"rx", GT::RX}, {"ry", GT::RY}, {"rz", GT::RZ}, {"p", GT::P},
        {"u", GT::U}, {"u1", GT::U1}, {"u2", GT::U2}, {"u3", GT::U3},
        {"cx", GT::CX}, {"cy", GT::CY}, {"cz", GT::CZ}, {"ch", GT::CH},
        {"swap", GT::SWAP}, {"iswap", GT::ISWAP},
        {"crx", GT::CRX}, {"cry", GT::CRY}, {"crz", GT::CRZ}, {"cp", GT::CP},
        {"cu", GT::CU}, {"ecr", GT::ECR}, {"rzx", GT::RZX},
        {"rxx", GT::RXX}, {"ryy", GT::RYY}, {"rzz", GT::RZZ},
        {"ccx", GT::CCX}, {"ccz", GT::CCZ}, {"cswap", GT::CSWAP}, {"rccx", GT::RCCX},
        {"measure", GT::MEASURE}, {"reset", GT::RESET}, {"barrier", GT::BARRIER},
        {"unitary", GT::UNITARY},
        {"mcx", GT::MCX}, {"mcp", GT::MCP}, {"permutation", GT::PERMUTATION},
        {"param_rx", GT::PARAM_RX}, {"param_ry", GT::PARAM_RY},
        {"param_rz", GT::PARAM_RZ}, {"param_p", GT::PARAM_P}, {"param_u", GT::PARAM_U}
    };
    auto it = map.find(s);
    if (it != map.end()) return it->second;
    return GT::UNITARY;  // fallback for custom labels
}

namespace {

constexpr ValidationOptions kDefaultValidation{};

// The policy is written as a name rather than as the enumerator's integer, so
// a stored circuit cannot change meaning if the enumeration is ever reordered.
const char* validation_policy_to_str(Validation p) {
    switch (p) {
        case Validation::Throw:  return "throw";
        case Validation::Warn:   return "warn";
        case Validation::Fix:    return "fix";
        case Validation::Ignore: return "ignore";
    }
    return "throw";
}

Validation validation_policy_from_str(const std::string& name) {
    if (name == "throw")  return Validation::Throw;
    if (name == "warn")   return Validation::Warn;
    if (name == "fix")    return Validation::Fix;
    if (name == "ignore") return Validation::Ignore;
    // Guessing here would pick a policy the file did not ask for, and the
    // wrong guess towards Ignore turns a rejected operand into a wrong answer.
    throw std::runtime_error(
        "QuantumCircuit::from_json: unknown validation policy '" + name + "'");
}

} // namespace

std::string QuantumCircuit::to_json() const {
    // Unbound symbolic parameter expressions (produced by from_qasm3) have no
    // JSON representation; silently dropping them would corrupt the
    // round-trip, so refuse loudly instead.
    for (const auto& inst : instructions) {
        if (!inst.param_exprs.empty()) {
            throw std::runtime_error(
                "QuantumCircuit::to_json: instruction '" + inst.gate_name() +
                "' carries unbound symbolic parameter expressions; call "
                "bind_parameters() before serialising");
        }
    }

    std::ostringstream o;
    o << std::setprecision(17);  // full double precision

    o << "{";
    o << "\"version\":\"1.0\",";
    o << "\"name\":" << json_escape(name) << ",";
    o << "\"n_qubits\":" << n_qubits << ",";
    o << "\"n_clbits\":" << n_clbits << ",";

    // Parameters
    o << "\"parameter_names\":[";
    for (size_t i = 0; i < parameter_names.size(); ++i) {
        if (i > 0) o << ",";
        o << json_escape(parameter_names[i]);
    }
    o << "],";

    // Instructions
    o << "\"instructions\":[";
    for (size_t i = 0; i < instructions.size(); ++i) {
        if (i > 0) o << ",";
        const auto& inst = instructions[i];
        o << "{";
        o << "\"gate\":" << json_escape(gate_type_to_str(inst.type)) << ",";

        // Qubits
        o << "\"qubits\":[";
        for (size_t j = 0; j < inst.qubits.size(); ++j) {
            if (j > 0) o << ",";
            o << inst.qubits[j];
        }
        o << "],";

        // Classical bits
        o << "\"clbits\":[";
        for (size_t j = 0; j < inst.clbits.size(); ++j) {
            if (j > 0) o << ",";
            o << inst.clbits[j];
        }
        o << "],";

        // Params
        o << "\"params\":[";
        for (size_t j = 0; j < inst.params.size(); ++j) {
            if (j > 0) o << ",";
            o << inst.params[j];
        }
        o << "]";

        // Param names (symbolic)
        if (!inst.param_names.empty()) {
            o << ",\"param_names\":[";
            for (size_t j = 0; j < inst.param_names.size(); ++j) {
                if (j > 0) o << ",";
                o << json_escape(inst.param_names[j]);
            }
            o << "]";
        }

        // Label
        if (!inst.label.empty()) {
            o << ",\"label\":" << json_escape(inst.label);
        }

        // Permutation basis-index map (R.1.18.0): serialised natively so JSON
        // is the lossless round-trip format for PERMUTATION (QASM 3 lowers it
        // to gates at export instead).
        if (inst.type == Instruction::GateType::PERMUTATION && !inst.permutation.empty()) {
            o << ",\"permutation\":[";
            for (size_t j = 0; j < inst.permutation.size(); ++j) {
                if (j > 0) o << ",";
                o << inst.permutation[j];
            }
            o << "]";
        }

        // Custom unitary matrix
        if (inst.type == Instruction::GateType::UNITARY && !inst.matrix.empty()) {
            o << ",\"matrix\":[";
            for (size_t j = 0; j < inst.matrix.size(); ++j) {
                if (j > 0) o << ",";
                o << "[" << inst.matrix[j].real << "," << inst.matrix[j].imag << "]";
            }
            o << "]";
        }

        // Physical-validity policy. The matrix is written entry by entry
        // above, so the policy governing it has to travel with it or a circuit
        // that ran before being written throws when read back.
        //
        // Emitted only where it differs from the default, which keeps a file
        // written by any other route loading unchanged. An absent field
        // therefore means Throw, and that direction is the safe one: a policy
        // that fails to survive a round trip produces a loud rejection rather
        // than a silent opt-out.
        if (inst.validation.policy != kDefaultValidation.policy ||
            inst.validation.atol != kDefaultValidation.atol) {
            o << ",\"validation\":{\"policy\":"
              << json_escape(validation_policy_to_str(inst.validation.policy))
              << ",\"atol\":" << inst.validation.atol << "}";
        }

        // Conditioning
        if (inst.condition_clbit >= 0) {
            o << ",\"condition_clbit\":" << inst.condition_clbit;
            o << ",\"condition_value\":" << inst.condition_value;
        }

        o << "}";
    }
    o << "]";

    o << "}";
    return o.str();
}

// =============================================================================
// JSON deserialization — minimal hand-rolled parser
// =============================================================================

// Simple JSON token reader
namespace {

struct JsonReader {
    const std::string& s;
    size_t pos = 0;

    void skip_ws() {
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r'))
            ++pos;
    }

    char peek() { skip_ws(); return (pos < s.size()) ? s[pos] : '\0'; }
    char next() { skip_ws(); return (pos < s.size()) ? s[pos++] : '\0'; }

    void expect(char c) {
        char got = next();
        if (got != c)
            throw std::runtime_error(std::string("JSON parse error: expected '") + c + "', got '" + got + "'");
    }

    std::string read_string() {
        expect('"');
        std::string out;
        while (pos < s.size() && s[pos] != '"') {
            if (s[pos] == '\\' && pos + 1 < s.size()) {
                ++pos;
                switch (s[pos]) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    default: out += s[pos]; break;
                }
            } else {
                out += s[pos];
            }
            ++pos;
        }
        if (pos < s.size()) ++pos;  // skip closing '"'
        return out;
    }

    double read_number() {
        skip_ws();
        size_t start = pos;
        if (pos < s.size() && s[pos] == '-') ++pos;
        while (pos < s.size() && (std::isdigit(s[pos]) || s[pos] == '.' || s[pos] == 'e' || s[pos] == 'E' || s[pos] == '+' || s[pos] == '-')) {
            if ((s[pos] == '+' || s[pos] == '-') && pos > start + 1 && s[pos-1] != 'e' && s[pos-1] != 'E') break;
            ++pos;
        }
        return std::stod(s.substr(start, pos - start));
    }

    int read_int() { return static_cast<int>(read_number()); }

    // Skip a JSON value we don't care about
    void skip_value() {
        skip_ws();
        if (s[pos] == '"') { read_string(); return; }
        if (s[pos] == '[') {
            ++pos;
            if (peek() != ']') {
                skip_value();
                while (peek() == ',') { ++pos; skip_value(); }
            }
            expect(']');
            return;
        }
        if (s[pos] == '{') {
            ++pos;
            if (peek() != '}') {
                read_string(); expect(':'); skip_value();
                while (peek() == ',') { ++pos; read_string(); expect(':'); skip_value(); }
            }
            expect('}');
            return;
        }
        // number / true / false / null
        while (pos < s.size() && s[pos] != ',' && s[pos] != '}' && s[pos] != ']')
            ++pos;
    }
};

} // anonymous namespace

QuantumCircuit QuantumCircuit::from_json(const std::string& json) {
    JsonReader r{json};
    r.expect('{');

    int nq = 0, nc = 0;
    std::string circ_name;
    std::vector<std::string> param_names_list;
    std::vector<Instruction> insts;

    while (r.peek() != '}') {
        if (r.peek() == ',') r.next();
        std::string key = r.read_string();
        r.expect(':');

        if (key == "n_qubits") {
            nq = r.read_int();
        } else if (key == "n_clbits") {
            nc = r.read_int();
        } else if (key == "name") {
            circ_name = r.read_string();
        } else if (key == "parameter_names") {
            r.expect('[');
            while (r.peek() != ']') {
                if (r.peek() == ',') r.next();
                param_names_list.push_back(r.read_string());
            }
            r.expect(']');
        } else if (key == "instructions") {
            r.expect('[');
            while (r.peek() != ']') {
                if (r.peek() == ',') r.next();
                r.expect('{');

                Instruction inst;
                while (r.peek() != '}') {
                    if (r.peek() == ',') r.next();
                    std::string ikey = r.read_string();
                    r.expect(':');

                    if (ikey == "gate") {
                        inst.type = str_to_gate_type(r.read_string());
                    } else if (ikey == "qubits") {
                        r.expect('[');
                        while (r.peek() != ']') {
                            if (r.peek() == ',') r.next();
                            inst.qubits.push_back(r.read_int());
                        }
                        r.expect(']');
                    } else if (ikey == "clbits") {
                        r.expect('[');
                        while (r.peek() != ']') {
                            if (r.peek() == ',') r.next();
                            inst.clbits.push_back(r.read_int());
                        }
                        r.expect(']');
                    } else if (ikey == "params") {
                        r.expect('[');
                        while (r.peek() != ']') {
                            if (r.peek() == ',') r.next();
                            inst.params.push_back(r.read_number());
                        }
                        r.expect(']');
                    } else if (ikey == "param_names") {
                        r.expect('[');
                        while (r.peek() != ']') {
                            if (r.peek() == ',') r.next();
                            inst.param_names.push_back(r.read_string());
                        }
                        r.expect(']');
                    } else if (ikey == "label") {
                        inst.label = r.read_string();
                    } else if (ikey == "permutation") {
                        r.expect('[');
                        while (r.peek() != ']') {
                            if (r.peek() == ',') r.next();
                            inst.permutation.push_back(r.read_int());
                        }
                        r.expect(']');
                    } else if (ikey == "matrix") {
                        // Accumulate into a local vector, then assign once:
                        // Instruction::matrix is copy-on-write (immutable), so
                        // it cannot be push_back-ed into in place.
                        std::vector<Complex128> mat;
                        r.expect('[');
                        while (r.peek() != ']') {
                            if (r.peek() == ',') r.next();
                            r.expect('[');
                            double re = r.read_number();
                            r.expect(',');
                            double im = r.read_number();
                            r.expect(']');
                            mat.push_back(Complex128(re, im));
                        }
                        r.expect(']');
                        inst.matrix = std::move(mat);
                    } else if (ikey == "validation") {
                        r.expect('{');
                        while (r.peek() != '}') {
                            if (r.peek() == ',') r.next();
                            const std::string vkey = r.read_string();
                            r.expect(':');
                            if (vkey == "policy") {
                                inst.validation.policy =
                                    validation_policy_from_str(r.read_string());
                            } else if (vkey == "atol") {
                                inst.validation.atol = r.read_number();
                            } else {
                                r.skip_value();
                            }
                        }
                        r.expect('}');
                    } else if (ikey == "condition_clbit") {
                        inst.condition_clbit = r.read_int();
                    } else if (ikey == "condition_value") {
                        inst.condition_value = r.read_int();
                    } else {
                        r.skip_value();
                    }
                }
                r.expect('}');
                insts.push_back(std::move(inst));
            }
            r.expect(']');
        } else {
            r.skip_value();
        }
    }
    r.expect('}');

    QuantumCircuit qc(nq, nc, circ_name);
    qc.parameter_names = std::move(param_names_list);
    qc.instructions = std::move(insts);
    return qc;
}

// =============================================================================
// Visualisation : QuantumCircuit::draw()
// =============================================================================
// Dispatches a CircuitDocument (produced by lindblad::viz::build_document) to
// one of four renderers based on DrawMode. The renderers live in
// src/visualisation/render_*.cpp and share the CircuitDocument data model.
//
// to_ascii() is a thin compatibility wrapper that calls draw(ASCII) and is
// scheduled for removal once external callers migrate.

std::string QuantumCircuit::draw(DrawMode mode, const DrawOptions& opts) const {
    const auto doc = lindblad::viz::build_document(*this, opts);
    switch (mode) {
        case DrawMode::ASCII: return lindblad::viz::render_ascii(doc, opts);
        case DrawMode::SVG:   return lindblad::viz::render_svg  (doc, opts);
        case DrawMode::LATEX: return lindblad::viz::render_latex(doc, opts);
        case DrawMode::HTML:  return lindblad::viz::render_html (doc, opts);
    }
    return lindblad::viz::render_ascii(doc, opts); // unreachable; silences warnings
}

std::string QuantumCircuit::to_ascii() const {
    return draw(DrawMode::ASCII, {});
}

// =============================================================================
// QuantumCircuit::draw_to_file : R.1.10.3 file-output convenience
// =============================================================================
// Renders via draw(mode, opts) and writes the resulting string verbatim to
// `path` in binary mode (so LF line endings survive on Windows / WSL hosts).
// A failed open surfaces as std::runtime_error with the path in the message
// rather than the silent truncation that std::ofstream would otherwise
// produce when the destination directory is missing or the path is
// unwritable.

void QuantumCircuit::draw_to_file(const std::string& path,
                                  DrawMode mode,
                                  const DrawOptions& opts) const {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error(
            "QuantumCircuit::draw_to_file: could not open '" + path +
            "' for writing");
    }
    out << draw(mode, opts);
}

} // namespace lindblad
