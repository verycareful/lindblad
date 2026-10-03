// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.31.1 - everything the circuit alone decides is refused before the first
// gate, on every backend.
//
// 1.1.31.0 put one pass in front of every run: indices first, over the whole
// circuit, then each instruction's parameters, operands and matrix. Before it,
// a NaN angle reached the state, a wrong-sized matrix was caught by whichever
// kernel met it first, and a malformed instruction was read out of bounds.
// Each refusal is pinned here as exactly what the caller receives: the type,
// the run's name, the whole message and the position, on all four backends,
// with no failed-run record, since nothing has been computed.
//
// Two tests are KNOWN RED until 1.1.31.2: a condition bit below -1, built by
// hand or read from JSON, runs the gate unconditioned instead of being
// refused.

#include <gtest/gtest.h>

#include "lindblad/backends/local_backend.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/errors.hpp"
#include "lindblad/failed_run.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/types.hpp"
#include "v11311_helpers.hpp"

#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <typeinfo>
#include <string>
#include <vector>

using namespace lindblad;
using GT = Instruction::GateType;

namespace {

struct Backend {
    const char* entry_point;
    std::function<void(const QuantumCircuit&)> run;
};

// Every backend, run directly and through LocalBackend set to it.
std::vector<Backend> every_backend() {
    using backends::LocalBackend;
    const auto local = [](LocalBackend::SimType type) {
        return [type](const QuantumCircuit& qc) {
            LocalBackend::Config cfg;
            cfg.simulator = type;
            (void)LocalBackend(cfg).run(qc, 4, 1);
        };
    };
    return {
        {"StatevectorSimulator::run", [](const QuantumCircuit& qc) { (void)StatevectorSimulator().run(qc, 4, 1); }},
        {"DensityMatrixSimulator::run",
         [](const QuantumCircuit& qc) { (void)DensityMatrixSimulator().run(qc, NoiseModel{}, 4, 1); }},
        {"MPSSimulator::run", [](const QuantumCircuit& qc) { (void)MPSSimulator().run(qc, 8, 4, 1); }},
        {"CliffordSimulator::run", [](const QuantumCircuit& qc) { (void)CliffordSimulator().run(qc, 4, 1); }},
        {"StatevectorSimulator::run", local(LocalBackend::SimType::STATEVECTOR)},
        {"DensityMatrixSimulator::run", local(LocalBackend::SimType::DENSITY_MATRIX)},
        {"MPSSimulator::run", local(LocalBackend::SimType::MPS)},
        {"CliffordSimulator::run", local(LocalBackend::SimType::CLIFFORD)},
    };
}

// `run` refuses `qc` before the first gate with exactly E, whose message is
// the run's name, `what`, and the position of instruction `index`.
template <class E>
void expect_refused(const Backend& b, const QuantumCircuit& qc, std::size_t index,
                    const std::string& what) {
    const Instruction& inst = qc.instructions[index];
    std::string position = " (instruction " + std::to_string(index) + ": " + inst.gate_name();
    if (!inst.qubits.empty()) {
        position += inst.qubits.size() == 1 ? " on qubit " : " on qubits ";
        for (std::size_t i = 0; i < inst.qubits.size(); ++i) {
            if (i > 0) position += ", ";
            position += std::to_string(inst.qubits[i]);
        }
    }
    position += ")";
    const std::uint64_t stores = detail::failed_run_stores();
    const auto e = v11311::thrown<E>([&] { b.run(qc); });
    EXPECT_EQ(detail::failed_run_stores(), stores) << "a refusal left a failed-run record";
    ASSERT_TRUE(e.has_value());
    EXPECT_EQ(e->entry_point(), b.entry_point);
    EXPECT_EQ(std::string(e->what()), std::string(b.entry_point) + ": " + what + position);
    v11311::expect_point(e->where(), -1, static_cast<int>(index), inst.gate_name(), inst.qubits);
}

Instruction raw(GT type, std::vector<int> qubits) {
    Instruction inst;
    inst.type = type;
    inst.qubits = std::move(qubits);
    return inst;
}

}  // namespace

// =============================================================================
// A parameter that is not a number (F04)
// =============================================================================

TEST(V11311Preflight, ANonFiniteParameterIsRefusedOnEveryBackend) {
    const double values[] = {std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity(),
                             -std::numeric_limits<double>::infinity()};
    for (const double v : values) {
        // The refused gate sits after a valid one, so the position is not the
        // first instruction by accident. Every backend refuses before it can
        // classify the gate (an RX of a non-finite angle is no quarter turn).
        QuantumCircuit qc(2, 2);
        qc.h(0);
        qc.rx(v, 1);
        qc.measure_all();
        for (const Backend& b : every_backend()) {
            SCOPED_TRACE(std::string(b.entry_point) + " " + std::to_string(v));
            expect_refused<InvalidArgument>(
                b, qc, 1, "parameter 0 is " + std::to_string(v) + "; every parameter must be finite");
        }
    }
}

TEST(V11311Preflight, ANonFiniteParameterInAnyPositionIsNamedByIndex) {
    QuantumCircuit qc(1);
    qc.u(0.1, 0.2, std::numeric_limits<double>::infinity(), 0);
    expect_refused<InvalidArgument>(every_backend()[0], qc, 0,
                                    "parameter 2 is " +
                                        std::to_string(std::numeric_limits<double>::infinity()) +
                                        "; every parameter must be finite");
}

// =============================================================================
// A matrix of the wrong size (F05), and the other shapes of an instruction
// =============================================================================

TEST(V11311Preflight, AWrongSizedUnitaryIsRefusedOnEveryBackend) {
    QuantumCircuit qc(2);
    qc.h(0);
    Instruction u = raw(GT::UNITARY, {0, 1});
    u.matrix = std::vector<Complex128>(4, Complex128(1.0, 0.0));
    qc.instructions.push_back(u);
    for (const Backend& b : every_backend()) {
        SCOPED_TRACE(b.entry_point);
        expect_refused<InvalidArgument>(
            b, qc, 1, "the UNITARY acts on 2 qubits, so its matrix must have 16 entries; it has 4");
    }
}

TEST(V11311Preflight, AMeasureOrResetOnTwoQubitsIsRefused) {
    // Every backend reads one operand of each; a second was silently ignored.
    QuantumCircuit measure(2, 2);
    Instruction m = raw(GT::MEASURE, {0, 1});
    m.clbits = {0, 1};
    measure.instructions.push_back(m);
    QuantumCircuit reset(2);
    reset.instructions.push_back(raw(GT::RESET, {0, 1}));
    for (const Backend& b : every_backend()) {
        SCOPED_TRACE(b.entry_point);
        expect_refused<InvalidArgument>(b, measure, 0, "the gate acts on 1 qubit; it names 2");
        expect_refused<InvalidArgument>(b, reset, 0, "the gate acts on 1 qubit; it names 2");
    }
}

TEST(V11311Preflight, AGateNamingOneQubitTwiceIsRefused) {
    QuantumCircuit qc(2);
    qc.instructions.push_back(raw(GT::CX, {1, 1}));
    for (const Backend& b : every_backend()) {
        SCOPED_TRACE(b.entry_point);
        expect_refused<InvalidArgument>(b, qc, 0,
                                        "qubit 1 is named twice; a gate's operands must be distinct");
    }
}

TEST(V11311Preflight, APermutationThatIsNotABijectionIsRefused) {
    // The map is checked before any backend's own check, so the tableau, which
    // supports no PERMUTATION at all, reports the map too.
    QuantumCircuit qc(2);
    Instruction p = raw(GT::PERMUTATION, {0, 1});
    const auto with_map = [&](std::vector<int> map) {
        QuantumCircuit c = qc;
        p.permutation = std::move(map);
        c.instructions.push_back(p);
        return c;
    };
    for (const Backend& b : every_backend()) {
        SCOPED_TRACE(b.entry_point);
        expect_refused<InvalidArgument>(b, with_map({0, 1, 1, 3}), 0,
                                        "the PERMUTATION's map sends two states to 1, so it is not a bijection");
        expect_refused<InvalidArgument>(b, with_map({0, 1, 2}), 0,
                                        "the PERMUTATION acts on 2 qubits, so its map must have 4 entries; it has 3");
        expect_refused<InvalidArgument>(b, with_map({0, 1, 2, 4}), 0,
                                        "the PERMUTATION's map sends a state to 4, outside [0, 4)");
    }
}

// =============================================================================
// Indices (F02), checked over the whole circuit first
// =============================================================================

TEST(V11311Preflight, AnIndexOutsideItsRegisterIsReportedBeforeAnyOtherFault) {
    // Instruction 0 is an unbound parameter and instruction 3 names a qubit the
    // register does not have. The index is what is reported: a register
    // mistake is the one a caller fixes first.
    QuantumCircuit qc(3, 1);
    Instruction unbound = raw(GT::PARAM_RX, {0});
    unbound.param_names = {"theta"};
    qc.instructions.push_back(unbound);
    qc.h(1);
    qc.h(2);
    qc.instructions.push_back(raw(GT::X, {7}));
    for (const Backend& b : every_backend()) {
        SCOPED_TRACE(b.entry_point);
        expect_refused<OutOfRange>(b, qc, 3, "qubit index 7 out of range [0, 3)");
    }
}

TEST(V11311Preflight, AClassicalOrConditionBitOutsideItsRegisterIsRefused) {
    QuantumCircuit measure(1, 2);
    Instruction m = raw(GT::MEASURE, {0});
    m.clbits = {3};
    measure.instructions.push_back(m);

    QuantumCircuit conditioned(1, 1);
    conditioned.x(0);
    conditioned.instructions.back().condition_clbit = 2;
    conditioned.instructions.back().condition_value = 1;
    for (const Backend& b : every_backend()) {
        SCOPED_TRACE(b.entry_point);
        expect_refused<OutOfRange>(b, measure, 0, "classical bit index 3 out of range [0, 2)");
        expect_refused<OutOfRange>(b, conditioned, 0,
                                   "the condition's classical bit index 2 out of range [0, 1)");
    }
}

// KNOWN RED until 1.1.31.2. condition_clbit = -1 means "no condition", and
// every backend tests >= 0, so any other negative value runs the gate
// unconditioned: here the X fires and every shot reads 1, where a condition on
// a clbit holding 0 would have skipped it. A condition bit below -1 is outside
// the register like one above it, and is refused the same way. The fix gives a
// condition its own flag; this hand-built pin is rewritten with it.
TEST(V11311Preflight, AConditionBitBelowMinusOneIsRefused) {
    QuantumCircuit qc(1, 1);
    qc.x(0);
    qc.instructions.back().condition_clbit = -7;
    qc.instructions.back().condition_value = 1;
    qc.measure(0, 0);
    for (const Backend& b : every_backend()) {
        SCOPED_TRACE(b.entry_point);
        const std::uint64_t stores = detail::failed_run_stores();
        const auto e = v11311::thrown<OutOfRange>([&] { b.run(qc); });
        EXPECT_EQ(detail::failed_run_stores(), stores);
        if (!e) continue;
        EXPECT_EQ(e->entry_point(), b.entry_point);
        v11311::expect_message(*e, {"-7"});
        v11311::expect_point(e->where(), -1, 0, "x", {0});
    }
}

// KNOWN RED until 1.1.31.2. The same condition read from JSON is accepted as
// it stands, so the circuit a file describes runs a gate the file conditions.
// Whichever layer the fix refuses it in, the file must never become a circuit
// that runs the gate unconditioned: either from_json refuses it as exactly
// InvalidArgument naming the condition, or every backend refuses what it
// returns as a condition bit outside the register.
TEST(V11311Preflight, FromJsonRefusesAConditionBitBelowMinusOne) {
    QuantumCircuit qc(1, 1);
    qc.add_if(0, 1, GT::X, {0});
    qc.measure(0, 0);
    std::string json = qc.to_json();
    const std::string tag = "\"condition_clbit\":0";
    const std::size_t at = json.find(tag);
    ASSERT_NE(at, std::string::npos) << json;
    json.replace(at, tag.size(), "\"condition_clbit\":-7");

    std::optional<QuantumCircuit> parsed;
    try {
        parsed = QuantumCircuit::from_json(json);
    } catch (const InvalidArgument& e) {
        EXPECT_EQ(typeid(e), typeid(InvalidArgument)) << e.what();
        v11311::expect_message(e, {"condition", "-7"});
        return;
    } catch (const std::exception& e) {
        FAIL() << "from_json refused with " << v11311::type_name(typeid(e)) << ": " << e.what();
    }
    ASSERT_TRUE(parsed.has_value());
    for (const Backend& b : every_backend()) {
        SCOPED_TRACE(b.entry_point);
        const auto e = v11311::thrown<OutOfRange>([&] { b.run(*parsed); });
        if (e) v11311::expect_message(*e, {"-7"});
    }
}

// =============================================================================
// Configuration the caller hands in
// =============================================================================

TEST(V11311Preflight, ALocalBackendSimTypeOutsideTheEnumIsACallerMistake) {
    backends::LocalBackend::Config cfg;
    cfg.simulator = static_cast<backends::LocalBackend::SimType>(17);
    backends::LocalBackend backend(cfg);
    QuantumCircuit qc(1, 1);
    qc.h(0).measure(0, 0);
    const auto e = v11311::thrown<InvalidArgument>([&] { (void)backend.run(qc, 4, 1); });
    ASSERT_TRUE(e.has_value());
    EXPECT_EQ(e->entry_point(), "LocalBackend::run");
    EXPECT_EQ(std::string(e->what()),
              "LocalBackend::run: Config::simulator holds 17, which is not one of the "
              "LocalBackend::SimType values");
}

// =============================================================================
// eval_expectation answers to the same limits as run (M19)
// =============================================================================

TEST(V11311Preflight, EvalExpectationRefusesARegisterOverTheLimitByName) {
    const int n = ENFORCED_MAX_QUBITS + 1;
    QuantumCircuit qc(n);
    qc.h(0);
    std::string label(static_cast<std::size_t>(n), 'I');
    label[0] = 'Z';
    const SparsePauliOp z(std::vector<PauliString>{PauliString(label)});
    const auto e = v11311::thrown<InvalidArgument>([&] { (void)StatevectorSimulator().eval_expectation(qc, z); });
    ASSERT_TRUE(e.has_value());
    EXPECT_EQ(e->entry_point(), "StatevectorSimulator::eval_expectation");
    EXPECT_EQ(std::string(e->what()),
              "StatevectorSimulator::eval_expectation: the circuit has " + std::to_string(n) +
                  " qubits, over the statevector limit of " + std::to_string(ENFORCED_MAX_QUBITS) +
                  "; Options::qubit_limit = QubitLimit::Lift raises it to " +
                  std::to_string(LIFTED_MAX_QUBITS));
}

TEST(V11311Preflight, EvalExpectationRefusesAStateOverTheMemoryCap) {
    // One state is all it holds: 2^17 amplitudes are 2 MiB, over a 1 MiB cap
    // and within a 2 MiB one, where the expectation is computed.
    const int n = 17;
    QuantumCircuit qc(n);
    qc.x(0);
    std::string label(static_cast<std::size_t>(n), 'I');
    label[0] = 'Z';
    const SparsePauliOp z(std::vector<PauliString>{PauliString(label)});
    const std::uint64_t need = (std::uint64_t{1} << n) * sizeof(Complex128);

    StatevectorSimulator tight;
    tight.options.max_memory_mb = 1;
    const auto e = v11311::thrown<InvalidArgument>([&] { (void)tight.eval_expectation(qc, z); });
    ASSERT_TRUE(e.has_value());
    EXPECT_EQ(e->entry_point(), "StatevectorSimulator::eval_expectation");
    EXPECT_EQ(std::string(e->what()),
              "StatevectorSimulator::eval_expectation: the run needs " + std::to_string(need) +
                  " bytes (" + std::to_string(need >> 20) +
                  " MiB) for its state buffers, over the cap of " +
                  std::to_string(std::uint64_t{1} << 20) + " bytes (1 MiB) from max_memory_mb = 1");

    StatevectorSimulator enough;
    enough.options.max_memory_mb = 2;
    double value = 0.0;
    ASSERT_NO_THROW(value = enough.eval_expectation(qc, z));
    EXPECT_EQ(value, -1.0) << "Z on qubit 0 of |...01>";
}
