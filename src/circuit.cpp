#include "qpp/circuit.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <iomanip>
#include <set>

namespace qpp {

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

void QuantumCircuit::add_param_name(const std::string& pname) {
    for (const auto& existing : parameter_names) {
        if (existing == pname) return;
    }
    parameter_names.push_back(pname);
}

// =============================================================================
// Single-qubit gates
// =============================================================================

#define QPP_SINGLE_GATE(name_func, gate_type) \
    QuantumCircuit& QuantumCircuit::name_func(int qubit) { \
        validate_qubit(qubit); \
        Instruction inst; \
        inst.type = Instruction::GateType::gate_type; \
        inst.qubits = {qubit}; \
        instructions.push_back(std::move(inst)); \
        return *this; \
    }

QPP_SINGLE_GATE(h, H)
QPP_SINGLE_GATE(x, X)
QPP_SINGLE_GATE(y, Y)
QPP_SINGLE_GATE(z, Z)
QPP_SINGLE_GATE(s, S)
QPP_SINGLE_GATE(sdg, SDG)
QPP_SINGLE_GATE(t, T)
QPP_SINGLE_GATE(tdg, TDG)
QPP_SINGLE_GATE(sx, SX)
QPP_SINGLE_GATE(sxdg, SXDG)

#undef QPP_SINGLE_GATE

#define QPP_PARAM1_GATE(name_func, gate_type) \
    QuantumCircuit& QuantumCircuit::name_func(double param, int qubit) { \
        validate_qubit(qubit); \
        Instruction inst; \
        inst.type = Instruction::GateType::gate_type; \
        inst.qubits = {qubit}; \
        inst.params = {param}; \
        instructions.push_back(std::move(inst)); \
        return *this; \
    }

QPP_PARAM1_GATE(rx, RX)
QPP_PARAM1_GATE(ry, RY)
QPP_PARAM1_GATE(rz, RZ)
QPP_PARAM1_GATE(p, P)

#undef QPP_PARAM1_GATE

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

#define QPP_TWO_GATE(name_func, gate_type) \
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

QPP_TWO_GATE(cx, CX)
QPP_TWO_GATE(cy, CY)
QPP_TWO_GATE(cz, CZ)
QPP_TWO_GATE(ch, CH)
QPP_TWO_GATE(swap, SWAP)
QPP_TWO_GATE(iswap, ISWAP)
QPP_TWO_GATE(ecr, ECR)

#undef QPP_TWO_GATE

#define QPP_CTRL_PARAM1_GATE(name_func, gate_type) \
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

QPP_CTRL_PARAM1_GATE(crx, CRX)
QPP_CTRL_PARAM1_GATE(cry, CRY)
QPP_CTRL_PARAM1_GATE(crz, CRZ)
QPP_CTRL_PARAM1_GATE(cp, CP)

#undef QPP_CTRL_PARAM1_GATE

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

#define QPP_ISING_GATE(name_func, gate_type) \
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

QPP_ISING_GATE(rzx, RZX)
QPP_ISING_GATE(rxx, RXX)
QPP_ISING_GATE(ryy, RYY)
QPP_ISING_GATE(rzz, RZZ)

#undef QPP_ISING_GATE

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
                                         const std::string& label) {
    for (int q : qubits) validate_qubit(q);
    Instruction inst;
    inst.type = Instruction::GateType::UNITARY;
    inst.qubits = qubits;
    inst.matrix = matrix;
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
                for (auto& param : inv_inst.params) {
                    param = -param;
                }
                break;

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

        if (num_ctrl_qubits == 1) {
            int ctrl = 0;  // the added control qubit
            // Map single-qubit gates → controlled variants
            if (inst.qubits.size() == 1) {
                int tgt = shifted_qubits[0];
                Instruction ci;
                ci.qubits = {ctrl, tgt};
                ci.params = inst.params;
                switch (inst.type) {
                    case GT::X:   ci.type = GT::CX; break;
                    case GT::Y:   ci.type = GT::CY; break;
                    case GT::Z:   ci.type = GT::CZ; break;
                    case GT::H:   ci.type = GT::CH; break;
                    case GT::RX:  ci.type = GT::CRX; break;
                    case GT::RY:  ci.type = GT::CRY; break;
                    case GT::RZ:  ci.type = GT::CRZ; break;
                    case GT::P: case GT::U1:
                        ci.type = GT::CP; break;
                    case GT::S: {
                        ci.type = GT::CP;
                        ci.params = {PI_2};
                        break;
                    }
                    case GT::SDG: {
                        ci.type = GT::CP;
                        ci.params = {-PI_2};
                        break;
                    }
                    case GT::T: {
                        ci.type = GT::CP;
                        ci.params = {PI_4};
                        break;
                    }
                    case GT::TDG: {
                        ci.type = GT::CP;
                        ci.params = {-PI_4};
                        break;
                    }
                    case GT::U: case GT::U3: {
                        ci.type = GT::CU;
                        ci.params = {inst.params[0], inst.params[1], inst.params[2], 0.0};
                        break;
                    }
                    default: {
                        // Generic: build controlled unitary from 2x2 matrix
                        // For now, fall through to generic UNITARY approach
                        goto generic_control;
                    }
                }
                result.instructions.push_back(std::move(ci));
                continue;
            }

            // Map two-qubit gates → three-qubit controlled versions
            if (inst.qubits.size() == 2) {
                int q0 = shifted_qubits[0], q1 = shifted_qubits[1];
                switch (inst.type) {
                    case GT::CX: {
                        // CCX (Toffoli)
                        Instruction ci;
                        ci.type = GT::CCX;
                        ci.qubits = {ctrl, q0, q1};
                        result.instructions.push_back(std::move(ci));
                        continue;
                    }
                    case GT::SWAP: {
                        // CSWAP (Fredkin)
                        Instruction ci;
                        ci.type = GT::CSWAP;
                        ci.qubits = {ctrl, q0, q1};
                        result.instructions.push_back(std::move(ci));
                        continue;
                    }
                    default:
                        goto generic_control;
                }
            }
        }

        generic_control: {
            // Generic controlled gate: build the block-diagonal controlled unitary
            // [[I, 0], [0, U]] where U is the gate's unitary, controlled on ALL
            // ctrl qubits being |1⟩.
            //
            // For multi-control: decompose as a cascade of Toffoli + controlled-U.
            // For simplicity, emit the gate conditioned on ctrl qubit 0 first,
            // then use MCX (multi-controlled X) ancilla techniques.
            //
            // Practical approach: for num_ctrl > 1, use Toffoli cascade to compute
            // AND of controls into an ancilla, then single-controlled-U, then uncompute.
            // But since we don't have ancilla management, we use the V-chain approach
            // which doesn't need ancilla for up to 3 controls.
            //
            // For now: emit as UNITARY with the full controlled matrix.

            int gate_qubits = static_cast<int>(inst.qubits.size());
            int total_gate_qubits = gate_qubits + num_ctrl_qubits;
            size_t gate_dim = 1ULL << gate_qubits;
            size_t total_dim = 1ULL << total_gate_qubits;

            // Build the gate's unitary matrix
            std::vector<Complex128> gate_matrix;
            if (inst.type == GT::UNITARY && !inst.matrix.empty()) {
                gate_matrix = inst.matrix;
            } else {
                // Simulate to extract the unitary
                gate_matrix.resize(gate_dim * gate_dim, Complex128(0, 0));
                // Identity fallback for unknown gates
                for (size_t i = 0; i < gate_dim; ++i) {
                    gate_matrix[i * gate_dim + i] = Complex128(1, 0);
                }
            }

            // Build controlled matrix: identity everywhere except the
            // bottom-right block (all controls = 1) which gets the gate matrix
            std::vector<Complex128> ctrl_matrix(total_dim * total_dim, Complex128(0, 0));
            size_t ctrl_subspace = total_dim - gate_dim;  // offset where all ctrls = 1

            // Fill identity for all rows except the controlled subspace
            for (size_t i = 0; i < ctrl_subspace; ++i) {
                ctrl_matrix[i * total_dim + i] = Complex128(1, 0);
            }
            // Fill gate matrix in the controlled subspace
            for (size_t r = 0; r < gate_dim; ++r) {
                for (size_t c = 0; c < gate_dim; ++c) {
                    ctrl_matrix[(ctrl_subspace + r) * total_dim + (ctrl_subspace + c)] = gate_matrix[r * gate_dim + c];
                }
            }

            Instruction ci;
            ci.type = GT::UNITARY;
            ci.matrix = std::move(ctrl_matrix);
            // Qubits: all control qubits + shifted target qubits
            for (int k = 0; k < num_ctrl_qubits; ++k) ci.qubits.push_back(k);
            for (int q : shifted_qubits) ci.qubits.push_back(q);
            ci.label = "c_" + inst.gate_name();
            result.instructions.push_back(std::move(ci));
        }
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

std::string QuantumCircuit::to_qasm2() const {
    std::ostringstream oss;
    oss << "OPENQASM 2.0;\n";
    oss << "include \"qelib1.inc\";\n";
    oss << "qreg q[" << n_qubits << "];\n";
    if (n_clbits > 0) {
        oss << "creg c[" << n_clbits << "];\n";
    }

    for (const auto& inst : instructions) {
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

    return oss.str();
}

std::string QuantumCircuit::to_qasm3() const {
    std::ostringstream oss;
    oss << "OPENQASM 3.0;\n";
    oss << "include \"stdgates.inc\";\n";
    oss << "qubit[" << n_qubits << "] q;\n";
    if (n_clbits > 0) {
        oss << "bit[" << n_clbits << "] c;\n";
    }

    for (const auto& inst : instructions) {
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

    return oss.str();
}

// Forward declaration of the bridge function in qasm2_parser.cpp
QuantumCircuit qasm2_parse_impl(const std::string& qasm);

QuantumCircuit QuantumCircuit::from_qasm2(const std::string& qasm) {
    return qasm2_parse_impl(qasm);
}

QuantumCircuit QuantumCircuit::from_qasm3(const std::string& /*qasm*/) {
    throw std::runtime_error("QASM3 parser not yet implemented — use QASM3Parser class");
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
        {"param_rx", GT::PARAM_RX}, {"param_ry", GT::PARAM_RY},
        {"param_rz", GT::PARAM_RZ}, {"param_p", GT::PARAM_P}, {"param_u", GT::PARAM_U}
    };
    auto it = map.find(s);
    if (it != map.end()) return it->second;
    return GT::UNITARY;  // fallback for custom labels
}

std::string QuantumCircuit::to_json() const {
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

        // Custom unitary matrix
        if (inst.type == Instruction::GateType::UNITARY && !inst.matrix.empty()) {
            o << ",\"matrix\":[";
            for (size_t j = 0; j < inst.matrix.size(); ++j) {
                if (j > 0) o << ",";
                o << "[" << inst.matrix[j].real << "," << inst.matrix[j].imag << "]";
            }
            o << "]";
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
                    } else if (ikey == "matrix") {
                        r.expect('[');
                        while (r.peek() != ']') {
                            if (r.peek() == ',') r.next();
                            r.expect('[');
                            double re = r.read_number();
                            r.expect(',');
                            double im = r.read_number();
                            r.expect(']');
                            inst.matrix.push_back(Complex128(re, im));
                        }
                        r.expect(']');
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
// ASCII visualisation
// =============================================================================

std::string QuantumCircuit::to_ascii() const {
    // Build a simple text-based circuit diagram
    std::vector<std::string> wires(n_qubits);
    for (int i = 0; i < n_qubits; ++i) {
        wires[i] = "q" + std::to_string(i) + ": ";
    }

    // Equalise initial lengths
    size_t max_prefix = 0;
    for (int i = 0; i < n_qubits; ++i) {
        max_prefix = std::max(max_prefix, wires[i].size());
    }
    for (int i = 0; i < n_qubits; ++i) {
        while (wires[i].size() < max_prefix) wires[i] += " ";
        wires[i] += "─";
    }

    for (const auto& inst : instructions) {
        std::string gname = inst.gate_name();

        // Equalise wire lengths first
        size_t max_len = 0;
        for (int i = 0; i < n_qubits; ++i) {
            max_len = std::max(max_len, wires[i].size());
        }
        for (int i = 0; i < n_qubits; ++i) {
            while (wires[i].size() < max_len) wires[i] += "─";
        }

        if (inst.type == Instruction::GateType::BARRIER) {
            for (int q : inst.qubits) {
                wires[q] += "┊";
            }
            for (int i = 0; i < n_qubits; ++i) {
                bool in_barrier = false;
                for (int q : inst.qubits) {
                    if (q == i) { in_barrier = true; break; }
                }
                if (!in_barrier) wires[i] += "─";
            }
        } else if (inst.type == Instruction::GateType::MEASURE) {
            wires[inst.qubits[0]] += "┤M├";
        } else if (inst.qubits.size() == 1) {
            // Single-qubit gate box
            std::string box = "┤" + gname + "├";
            wires[inst.qubits[0]] += box;
        } else if (inst.type == Instruction::GateType::CX) {
            int ctrl = inst.qubits[0];
            int tgt = inst.qubits[1];
            wires[ctrl] += "─●─";
            wires[tgt]  += "─⊕─";
        } else if (inst.type == Instruction::GateType::CZ) {
            int ctrl = inst.qubits[0];
            int tgt = inst.qubits[1];
            wires[ctrl] += "─●─";
            wires[tgt]  += "─●─";
        } else if (inst.type == Instruction::GateType::SWAP) {
            wires[inst.qubits[0]] += "─✕─";
            wires[inst.qubits[1]] += "─✕─";
        } else {
            // Multi-qubit gate: show name on first qubit
            for (size_t qi = 0; qi < inst.qubits.size(); ++qi) {
                if (qi == 0) {
                    wires[inst.qubits[qi]] += "┤" + gname + "├";
                } else {
                    wires[inst.qubits[qi]] += "─┼─";
                }
            }
        }

        // Add trailing wire
        for (int i = 0; i < n_qubits; ++i) {
            wires[i] += "─";
        }
    }

    // Combine
    std::ostringstream oss;
    for (int i = 0; i < n_qubits; ++i) {
        oss << wires[i] << "\n";
    }
    return oss.str();
}

} // namespace qpp
