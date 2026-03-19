// qasm3_parser.cpp — QASM 3.0 parser (stub)
// Full QASM 3.0 parsing requires a more complex grammar
// with classical control flow, timing, etc.

#include "qpp/circuit.hpp"

namespace qpp {

class QASM3Parser {
public:
    static QuantumCircuit parse(const std::string& /*qasm*/) {
        throw std::runtime_error(
            "QASM 3.0 parser not yet implemented. "
            "Please use QASM 2.0 or construct circuits programmatically."
        );
    }
};

} // namespace qpp
