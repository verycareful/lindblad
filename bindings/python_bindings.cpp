#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

#include "lindblad/types.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/gates.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/primitives.hpp"
#include "lindblad/algorithms.hpp"
#include "lindblad/transpiler.hpp"
#include "lindblad/backends/local_backend.hpp"

namespace py = pybind11;

PYBIND11_MODULE(lindblad_python, m) {
    m.doc() = "lindblad — High-performance quantum computing framework";

    // =========================================================================
    // Complex128
    // =========================================================================
    py::class_<lindblad::Complex128>(m, "Complex128")
        .def(py::init<double, double>(), py::arg("real") = 0.0, py::arg("imag") = 0.0)
        .def_readwrite("real", &lindblad::Complex128::real)
        .def_readwrite("imag", &lindblad::Complex128::imag)
        .def("norm_sq", &lindblad::Complex128::norm_sq)
        .def("conj", &lindblad::Complex128::conj)
        .def("__repr__", [](const lindblad::Complex128& c) {
            return "(" + std::to_string(c.real) + "+" + std::to_string(c.imag) + "j)";
        });

    // =========================================================================
    // Statevector
    // =========================================================================
    py::class_<lindblad::Statevector>(m, "Statevector")
        .def(py::init<int>())
        .def("initialize_basis", &lindblad::Statevector::initialize_basis)
        .def("amplitude", &lindblad::Statevector::amplitude)
        .def("probability", &lindblad::Statevector::probability)
        .def("norm", &lindblad::Statevector::norm)
        .def("normalize", &lindblad::Statevector::normalize)
        .def("measure_once", &lindblad::Statevector::measure_once)
        .def("sample_counts", &lindblad::Statevector::sample_counts)
        .def("to_string", &lindblad::Statevector::to_string)
        .def("to_numpy", [](const lindblad::Statevector& sv) -> py::array_t<std::complex<double>> {
            py::array_t<std::complex<double>> arr({static_cast<py::ssize_t>(sv.dim)});
            auto buf = arr.mutable_unchecked<1>();
            for (size_t i = 0; i < sv.dim; ++i) {
                buf(i) = std::complex<double>(sv.real_parts[i], sv.imag_parts[i]);
            }
            return arr;
        }, "Return amplitude array as a complex128 numpy array")
        .def("probabilities", &lindblad::Statevector::probabilities)
        .def_readonly("n_qubits", &lindblad::Statevector::n_qubits)
        .def_readonly("dim", &lindblad::Statevector::dim);

    // =========================================================================
    // Visualisation : DrawMode, ParamFormat, DrawOptions
    // =========================================================================
    py::enum_<lindblad::DrawMode>(m, "DrawMode")
        .value("ASCII", lindblad::DrawMode::ASCII)
        .value("SVG",   lindblad::DrawMode::SVG)
        .value("LATEX", lindblad::DrawMode::LATEX)
        .value("HTML",  lindblad::DrawMode::HTML);

    py::enum_<lindblad::ParamFormat>(m, "ParamFormat")
        .value("Pretty", lindblad::ParamFormat::Pretty)
        .value("Raw",    lindblad::ParamFormat::Raw);

    py::class_<lindblad::DrawOptions>(m, "DrawOptions")
        .def(py::init<>())
        .def_readwrite("fold_width",     &lindblad::DrawOptions::fold_width)
        .def_readwrite("show_clbits",    &lindblad::DrawOptions::show_clbits)
        .def_readwrite("show_params",    &lindblad::DrawOptions::show_params)
        .def_readwrite("ascii_safe",     &lindblad::DrawOptions::ascii_safe)
        .def_readwrite("param_format",   &lindblad::DrawOptions::param_format)
        .def_readwrite("cell_width_px",  &lindblad::DrawOptions::cell_width_px)
        .def_readwrite("cell_height_px", &lindblad::DrawOptions::cell_height_px)
        .def_readwrite("include_legend", &lindblad::DrawOptions::include_legend);

    // =========================================================================
    // QuantumCircuit
    // =========================================================================
    py::class_<lindblad::QuantumCircuit>(m, "QuantumCircuit")
        .def(py::init<int, int>(), py::arg("n_qubits"), py::arg("n_clbits") = 0)
        .def(py::init<int, int, const std::string&>())
        // Single qubit gates
        .def("h", &lindblad::QuantumCircuit::h)
        .def("x", &lindblad::QuantumCircuit::x)
        .def("y", &lindblad::QuantumCircuit::y)
        .def("z", &lindblad::QuantumCircuit::z)
        .def("s", &lindblad::QuantumCircuit::s)
        .def("sdg", &lindblad::QuantumCircuit::sdg)
        .def("t", &lindblad::QuantumCircuit::t)
        .def("tdg", &lindblad::QuantumCircuit::tdg)
        .def("sx", &lindblad::QuantumCircuit::sx)
        .def("sxdg", &lindblad::QuantumCircuit::sxdg)
        .def("rx", py::overload_cast<double, int>(&lindblad::QuantumCircuit::rx))
        .def("ry", py::overload_cast<double, int>(&lindblad::QuantumCircuit::ry))
        .def("rz", py::overload_cast<double, int>(&lindblad::QuantumCircuit::rz))
        .def("p", &lindblad::QuantumCircuit::p)
        .def("u", &lindblad::QuantumCircuit::u)
        // Two qubit gates
        .def("cx", &lindblad::QuantumCircuit::cx)
        .def("cy", &lindblad::QuantumCircuit::cy)
        .def("cz", &lindblad::QuantumCircuit::cz)
        .def("ch", &lindblad::QuantumCircuit::ch)
        .def("swap", &lindblad::QuantumCircuit::swap)
        .def("crx", &lindblad::QuantumCircuit::crx)
        .def("cry", &lindblad::QuantumCircuit::cry)
        .def("crz", &lindblad::QuantumCircuit::crz)
        .def("cp", &lindblad::QuantumCircuit::cp)
        .def("rxx", &lindblad::QuantumCircuit::rxx)
        .def("ryy", &lindblad::QuantumCircuit::ryy)
        .def("rzz", &lindblad::QuantumCircuit::rzz)
        // Three qubit gates
        .def("ccx", &lindblad::QuantumCircuit::ccx)
        .def("cswap", &lindblad::QuantumCircuit::cswap)
        // Operations
        .def("measure", &lindblad::QuantumCircuit::measure)
        .def("measure_all", &lindblad::QuantumCircuit::measure_all)
        .def("barrier", &lindblad::QuantumCircuit::barrier, py::arg("qubits") = std::vector<int>{})
        .def("reset", &lindblad::QuantumCircuit::reset)
        // Analysis
        .def("depth", &lindblad::QuantumCircuit::depth)
        .def("size", &lindblad::QuantumCircuit::size)
        .def("count_ops", &lindblad::QuantumCircuit::count_ops)
        .def("num_parameters", &lindblad::QuantumCircuit::num_parameters)
        // Composition
        .def("compose", &lindblad::QuantumCircuit::compose)
        .def("inverse", &lindblad::QuantumCircuit::inverse)
        .def("repeat", &lindblad::QuantumCircuit::repeat)
        .def("control", &lindblad::QuantumCircuit::control, py::arg("num_ctrl_qubits") = 1)
        // Export
        .def("to_qasm2", &lindblad::QuantumCircuit::to_qasm2)
        .def("to_qasm3", &lindblad::QuantumCircuit::to_qasm3)
        .def("to_json", &lindblad::QuantumCircuit::to_json)
        .def_static("from_json", &lindblad::QuantumCircuit::from_json)
        .def("to_ascii", &lindblad::QuantumCircuit::to_ascii)
        .def("draw", &lindblad::QuantumCircuit::draw,
             py::arg("mode") = lindblad::DrawMode::ASCII,
             py::arg("opts") = lindblad::DrawOptions{})
        .def_readwrite("n_qubits", &lindblad::QuantumCircuit::n_qubits)
        .def_readwrite("n_clbits", &lindblad::QuantumCircuit::n_clbits)
        .def_readwrite("name", &lindblad::QuantumCircuit::name);

    // =========================================================================
    // StatevectorSimulator
    // =========================================================================
    py::class_<lindblad::StatevectorSimulator>(m, "StatevectorSimulator")
        .def(py::init<>())
        .def("run", &lindblad::StatevectorSimulator::run,
             py::arg("circuit"), py::arg("shots") = 0, py::arg("seed") = 0);

    py::class_<lindblad::StatevectorSimulator::Result>(m, "StatevectorResult")
        .def_readonly("counts", &lindblad::StatevectorSimulator::Result::counts)
        .def_readonly("final_state", &lindblad::StatevectorSimulator::Result::final_state)
        .def_readonly("simulation_time_seconds", &lindblad::StatevectorSimulator::Result::simulation_time_seconds)
        .def_readonly("success", &lindblad::StatevectorSimulator::Result::success)
        .def_readonly("error_message", &lindblad::StatevectorSimulator::Result::error_message);

    // =========================================================================
    // PauliString
    // =========================================================================
    py::class_<lindblad::PauliString>(m, "PauliString")
        .def(py::init<>())
        .def(py::init<const std::string&, lindblad::Complex128>(),
             py::arg("pauli"), py::arg("coeff") = lindblad::Complex128(1.0, 0.0))
        .def_readwrite("pauli", &lindblad::PauliString::pauli)
        .def_readwrite("coeff", &lindblad::PauliString::coeff)
        .def("n_qubits", &lindblad::PauliString::n_qubits)
        .def("__repr__", [](const lindblad::PauliString& ps) {
            return "PauliString('" + ps.pauli + "', " +
                   std::to_string(ps.coeff.real) + "+" +
                   std::to_string(ps.coeff.imag) + "j)";
        });

    // =========================================================================
    // SparsePauliOp
    // =========================================================================
    py::class_<lindblad::SparsePauliOp>(m, "SparsePauliOp")
        .def(py::init<>())
        .def(py::init<const std::vector<lindblad::PauliString>&>(), py::arg("terms"))
        .def_static("from_list", &lindblad::SparsePauliOp::from_list)
        .def("to_matrix", &lindblad::SparsePauliOp::to_matrix)
        .def("expectation_value", &lindblad::SparsePauliOp::expectation_value)
        .def("simplify", &lindblad::SparsePauliOp::simplify)
        .def("n_qubits", &lindblad::SparsePauliOp::n_qubits)
        .def_readwrite("terms", &lindblad::SparsePauliOp::terms);

    // =========================================================================
    // Estimator and Sampler
    // =========================================================================
    py::class_<lindblad::Estimator>(m, "Estimator")
        .def(py::init<>())
        .def("run_single", &lindblad::Estimator::run_single)
        .def("run_batch", &lindblad::Estimator::run_batch);

    py::class_<lindblad::Sampler>(m, "Sampler")
        .def(py::init<>())
        .def("run_single", &lindblad::Sampler::run_single);

    // =========================================================================
    // Transpile
    // =========================================================================
    m.def("transpile", &lindblad::transpile,
          py::arg("circuit"),
          py::arg("coupling_map") = lindblad::CouplingMap(),
          py::arg("basis_gates") = std::vector<std::string>{},
          py::arg("optimization_level") = 1);

    // =========================================================================
    // LocalBackend
    // =========================================================================
    py::class_<lindblad::backends::LocalBackend>(m, "LocalBackend")
        .def(py::init<>())
        .def("run", &lindblad::backends::LocalBackend::run,
             py::arg("circuit"), py::arg("shots") = 1024, py::arg("seed") = 0)
        .def("name", &lindblad::backends::LocalBackend::name);

    py::class_<lindblad::backends::BackendResult>(m, "BackendResult")
        .def_readonly("counts", &lindblad::backends::BackendResult::counts)
        .def_readonly("success", &lindblad::backends::BackendResult::success)
        .def_readonly("simulation_time_seconds", &lindblad::backends::BackendResult::simulation_time_seconds);
}
