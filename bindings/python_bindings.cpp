#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

#include "qpp/types.hpp"
#include "qpp/statevector.hpp"
#include "qpp/circuit.hpp"
#include "qpp/gates.hpp"
#include "qpp/operators.hpp"
#include "qpp/noise.hpp"
#include "qpp/simulators/statevector_sim.hpp"
#include "qpp/primitives.hpp"
#include "qpp/algorithms.hpp"
#include "qpp/transpiler.hpp"
#include "qpp/backends/local_backend.hpp"

namespace py = pybind11;

PYBIND11_MODULE(qpp_python, m) {
    m.doc() = "q++ — High-performance quantum computing framework";

    // =========================================================================
    // Complex128
    // =========================================================================
    py::class_<qpp::Complex128>(m, "Complex128")
        .def(py::init<double, double>(), py::arg("real") = 0.0, py::arg("imag") = 0.0)
        .def_readwrite("real", &qpp::Complex128::real)
        .def_readwrite("imag", &qpp::Complex128::imag)
        .def("norm_sq", &qpp::Complex128::norm_sq)
        .def("conj", &qpp::Complex128::conj)
        .def("__repr__", [](const qpp::Complex128& c) {
            return "(" + std::to_string(c.real) + "+" + std::to_string(c.imag) + "j)";
        });

    // =========================================================================
    // Statevector
    // =========================================================================
    py::class_<qpp::Statevector>(m, "Statevector")
        .def(py::init<int>())
        .def("initialize_basis", &qpp::Statevector::initialize_basis)
        .def("amplitude", &qpp::Statevector::amplitude)
        .def("probability", &qpp::Statevector::probability)
        .def("norm", &qpp::Statevector::norm)
        .def("normalize", &qpp::Statevector::normalize)
        .def("measure_once", &qpp::Statevector::measure_once)
        .def("sample_counts", &qpp::Statevector::sample_counts)
        .def("to_string", &qpp::Statevector::to_string)
        .def("to_numpy", [](const qpp::Statevector& sv) -> py::array_t<std::complex<double>> {
            py::array_t<std::complex<double>> arr({static_cast<py::ssize_t>(sv.dim)});
            auto buf = arr.mutable_unchecked<1>();
            for (size_t i = 0; i < sv.dim; ++i) {
                buf(i) = std::complex<double>(sv.real_parts[i], sv.imag_parts[i]);
            }
            return arr;
        }, "Return amplitude array as a complex128 numpy array")
        .def("probabilities", &qpp::Statevector::probabilities)
        .def_readonly("n_qubits", &qpp::Statevector::n_qubits)
        .def_readonly("dim", &qpp::Statevector::dim);

    // =========================================================================
    // QuantumCircuit
    // =========================================================================
    py::class_<qpp::QuantumCircuit>(m, "QuantumCircuit")
        .def(py::init<int, int>(), py::arg("n_qubits"), py::arg("n_clbits") = 0)
        .def(py::init<int, int, const std::string&>())
        // Single qubit gates
        .def("h", &qpp::QuantumCircuit::h)
        .def("x", &qpp::QuantumCircuit::x)
        .def("y", &qpp::QuantumCircuit::y)
        .def("z", &qpp::QuantumCircuit::z)
        .def("s", &qpp::QuantumCircuit::s)
        .def("sdg", &qpp::QuantumCircuit::sdg)
        .def("t", &qpp::QuantumCircuit::t)
        .def("tdg", &qpp::QuantumCircuit::tdg)
        .def("sx", &qpp::QuantumCircuit::sx)
        .def("sxdg", &qpp::QuantumCircuit::sxdg)
        .def("rx", py::overload_cast<double, int>(&qpp::QuantumCircuit::rx))
        .def("ry", py::overload_cast<double, int>(&qpp::QuantumCircuit::ry))
        .def("rz", py::overload_cast<double, int>(&qpp::QuantumCircuit::rz))
        .def("p", &qpp::QuantumCircuit::p)
        .def("u", &qpp::QuantumCircuit::u)
        // Two qubit gates
        .def("cx", &qpp::QuantumCircuit::cx)
        .def("cy", &qpp::QuantumCircuit::cy)
        .def("cz", &qpp::QuantumCircuit::cz)
        .def("ch", &qpp::QuantumCircuit::ch)
        .def("swap", &qpp::QuantumCircuit::swap)
        .def("crx", &qpp::QuantumCircuit::crx)
        .def("cry", &qpp::QuantumCircuit::cry)
        .def("crz", &qpp::QuantumCircuit::crz)
        .def("cp", &qpp::QuantumCircuit::cp)
        .def("rxx", &qpp::QuantumCircuit::rxx)
        .def("ryy", &qpp::QuantumCircuit::ryy)
        .def("rzz", &qpp::QuantumCircuit::rzz)
        // Three qubit gates
        .def("ccx", &qpp::QuantumCircuit::ccx)
        .def("cswap", &qpp::QuantumCircuit::cswap)
        // Operations
        .def("measure", &qpp::QuantumCircuit::measure)
        .def("measure_all", &qpp::QuantumCircuit::measure_all)
        .def("barrier", &qpp::QuantumCircuit::barrier, py::arg("qubits") = std::vector<int>{})
        .def("reset", &qpp::QuantumCircuit::reset)
        // Analysis
        .def("depth", &qpp::QuantumCircuit::depth)
        .def("size", &qpp::QuantumCircuit::size)
        .def("count_ops", &qpp::QuantumCircuit::count_ops)
        .def("num_parameters", &qpp::QuantumCircuit::num_parameters)
        // Composition
        .def("compose", &qpp::QuantumCircuit::compose)
        .def("inverse", &qpp::QuantumCircuit::inverse)
        .def("repeat", &qpp::QuantumCircuit::repeat)
        // Export
        .def("to_qasm2", &qpp::QuantumCircuit::to_qasm2)
        .def("to_qasm3", &qpp::QuantumCircuit::to_qasm3)
        .def("to_json", &qpp::QuantumCircuit::to_json)
        .def_static("from_json", &qpp::QuantumCircuit::from_json)
        .def("to_ascii", &qpp::QuantumCircuit::to_ascii)
        .def_readwrite("n_qubits", &qpp::QuantumCircuit::n_qubits)
        .def_readwrite("n_clbits", &qpp::QuantumCircuit::n_clbits)
        .def_readwrite("name", &qpp::QuantumCircuit::name);

    // =========================================================================
    // StatevectorSimulator
    // =========================================================================
    py::class_<qpp::StatevectorSimulator>(m, "StatevectorSimulator")
        .def(py::init<>())
        .def("run", &qpp::StatevectorSimulator::run,
             py::arg("circuit"), py::arg("shots") = 0, py::arg("seed") = 0);

    py::class_<qpp::StatevectorSimulator::Result>(m, "StatevectorResult")
        .def_readonly("counts", &qpp::StatevectorSimulator::Result::counts)
        .def_readonly("final_state", &qpp::StatevectorSimulator::Result::final_state)
        .def_readonly("simulation_time_seconds", &qpp::StatevectorSimulator::Result::simulation_time_seconds)
        .def_readonly("success", &qpp::StatevectorSimulator::Result::success)
        .def_readonly("error_message", &qpp::StatevectorSimulator::Result::error_message);

    // =========================================================================
    // PauliString
    // =========================================================================
    py::class_<qpp::PauliString>(m, "PauliString")
        .def(py::init<>())
        .def(py::init<const std::string&, qpp::Complex128>(),
             py::arg("pauli"), py::arg("coeff") = qpp::Complex128(1.0, 0.0))
        .def_readwrite("pauli", &qpp::PauliString::pauli)
        .def_readwrite("coeff", &qpp::PauliString::coeff)
        .def("n_qubits", &qpp::PauliString::n_qubits)
        .def("__repr__", [](const qpp::PauliString& ps) {
            return "PauliString('" + ps.pauli + "', " +
                   std::to_string(ps.coeff.real) + "+" +
                   std::to_string(ps.coeff.imag) + "j)";
        });

    // =========================================================================
    // SparsePauliOp
    // =========================================================================
    py::class_<qpp::SparsePauliOp>(m, "SparsePauliOp")
        .def(py::init<>())
        .def(py::init<const std::vector<qpp::PauliString>&>(), py::arg("terms"))
        .def_static("from_list", &qpp::SparsePauliOp::from_list)
        .def("to_matrix", &qpp::SparsePauliOp::to_matrix)
        .def("expectation_value", &qpp::SparsePauliOp::expectation_value)
        .def("simplify", &qpp::SparsePauliOp::simplify)
        .def("n_qubits", &qpp::SparsePauliOp::n_qubits)
        .def_readwrite("terms", &qpp::SparsePauliOp::terms);

    // =========================================================================
    // Estimator and Sampler
    // =========================================================================
    py::class_<qpp::Estimator>(m, "Estimator")
        .def(py::init<>())
        .def("run_single", &qpp::Estimator::run_single)
        .def("run_batch", &qpp::Estimator::run_batch);

    py::class_<qpp::Sampler>(m, "Sampler")
        .def(py::init<>())
        .def("run_single", &qpp::Sampler::run_single);

    // =========================================================================
    // Transpile
    // =========================================================================
    m.def("transpile", &qpp::transpile,
          py::arg("circuit"),
          py::arg("coupling_map") = qpp::CouplingMap(),
          py::arg("basis_gates") = std::vector<std::string>{},
          py::arg("optimization_level") = 1);

    // =========================================================================
    // LocalBackend
    // =========================================================================
    py::class_<qpp::backends::LocalBackend>(m, "LocalBackend")
        .def(py::init<>())
        .def("run", &qpp::backends::LocalBackend::run,
             py::arg("circuit"), py::arg("shots") = 1024, py::arg("seed") = 0)
        .def("name", &qpp::backends::LocalBackend::name);

    py::class_<qpp::backends::BackendResult>(m, "BackendResult")
        .def_readonly("counts", &qpp::backends::BackendResult::counts)
        .def_readonly("success", &qpp::backends::BackendResult::success)
        .def_readonly("simulation_time_seconds", &qpp::backends::BackendResult::simulation_time_seconds);
}
