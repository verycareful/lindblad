// R.1.19.1 test wave — Clifford tableau fail-loud validation.
//
// R.1.19.0 added bounds checks to every StabilizerState gate and to measure()
// (a bad qubit indexed a tableau column directly: out-of-bounds access, UB),
// distinctness enforcement to apply_cx, and turned CliffordSimulator::run()'s
// gate dispatch `default: break;` (a silent no-op for any gate the tableau
// does not implement) into a throw. That silent default was reachable only by
// a DIRECT run() on a non-Clifford circuit (the AUTO backend dispatch is gated
// by is_clifford()), so a direct caller previously got a wrong answer with no
// signal.
//
// This backend surfaces errors by throwing (it has no Result error channel),
// so run() propagates both the pre-flight bounds throw and the unsupported-gate
// throw to the caller.

#include <gtest/gtest.h>

#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"

#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace lindblad;

namespace {

constexpr int kN = 3;
const std::vector<int> kBadIndices = {-1, kN, kN + 1, 64};

using SqGate = std::function<void(StabilizerState&, int)>;
std::vector<std::pair<const char*, SqGate>> single_qubit_gates() {
    return {
        {"apply_h",   [](StabilizerState& s, int q) { s.apply_h(q); }},
        {"apply_s",   [](StabilizerState& s, int q) { s.apply_s(q); }},
        {"apply_sdg", [](StabilizerState& s, int q) { s.apply_sdg(q); }},
        {"apply_x",   [](StabilizerState& s, int q) { s.apply_x(q); }},
        {"apply_y",   [](StabilizerState& s, int q) { s.apply_y(q); }},
        {"apply_z",   [](StabilizerState& s, int q) { s.apply_z(q); }},
    };
}

} // namespace

// =============================================================================
// StabilizerState single-qubit gates
// =============================================================================

TEST(R1191CliffordValidation, SingleQubitOutOfRangeThrows) {
    for (const auto& [name, fn] : single_qubit_gates()) {
        for (int bad : kBadIndices) {
            SCOPED_TRACE(std::string(name) + " q=" + std::to_string(bad));
            StabilizerState st(kN);
            EXPECT_THROW(fn(st, bad), std::out_of_range);
        }
    }
}

TEST(R1191CliffordValidation, SingleQubitValidDoesNotThrow) {
    for (const auto& [name, fn] : single_qubit_gates()) {
        SCOPED_TRACE(name);
        StabilizerState st(kN);
        EXPECT_NO_THROW(fn(st, 0));
        EXPECT_NO_THROW(fn(st, kN - 1));
    }
}

// =============================================================================
// StabilizerState::apply_cx
// =============================================================================

TEST(R1191CliffordValidation, CxOutOfRangeThrows) {
    for (int bad : kBadIndices) {
        SCOPED_TRACE("bad=" + std::to_string(bad));
        { StabilizerState st(kN); EXPECT_THROW(st.apply_cx(bad, 1), std::out_of_range); }
        { StabilizerState st(kN); EXPECT_THROW(st.apply_cx(0, bad), std::out_of_range); }
    }
}

TEST(R1191CliffordValidation, CxNonDistinctThrows) {
    StabilizerState st(kN);
    EXPECT_THROW(st.apply_cx(1, 1), std::invalid_argument);
}

TEST(R1191CliffordValidation, CxValidDoesNotThrow) {
    StabilizerState st(kN);
    EXPECT_NO_THROW(st.apply_cx(0, 1));
    EXPECT_NO_THROW(st.apply_cx(2, 0));
}

// =============================================================================
// StabilizerState::measure
// =============================================================================

TEST(R1191CliffordValidation, MeasureOutOfRangeThrows) {
    std::mt19937_64 rng(123);
    for (int bad : kBadIndices) {
        SCOPED_TRACE("bad=" + std::to_string(bad));
        StabilizerState st(kN);
        EXPECT_THROW(st.measure(bad, true, rng), std::out_of_range);
    }
}

TEST(R1191CliffordValidation, MeasureValidDoesNotThrow) {
    std::mt19937_64 rng(123);
    StabilizerState st(kN);
    for (int q = 0; q < kN; ++q) {
        SCOPED_TRACE("q=" + std::to_string(q));
        EXPECT_NO_THROW(st.measure(q, true, rng));
    }
}

// =============================================================================
// CliffordSimulator::run — unsupported gate is loud, not a silent no-op
// =============================================================================

TEST(R1191CliffordValidation, RunUnsupportedGateThrows) {
    // T is not a Clifford gate; a direct run() reaches the dispatch default,
    // which used to `break` (silently skip the gate) and now throws.
    QuantumCircuit qc(1);
    qc.t(0);
    CliffordSimulator sim;
    EXPECT_THROW(sim.run(qc, /*shots=*/16), std::invalid_argument);
}

TEST(R1191CliffordValidation, RunUnsupportedCliffordGateThrows) {
    // RZZ at π/2 is genuinely Clifford (it equals cx(a,b) · s(b) · cx(a,b)) but
    // the tableau dispatch carries no case for it, so a direct run() must throw
    // rather than silently drop the gate. is_clifford() rejects it as well,
    // which is what keeps the AUTO route from ever reaching this throw.
    QuantumCircuit qc(2);
    qc.rzz(PI_2, 0, 1);
    EXPECT_FALSE(CliffordSimulator::is_clifford(qc));
    CliffordSimulator sim;
    EXPECT_THROW(sim.run(qc, /*shots=*/16), std::invalid_argument);
}

TEST(R1191CliffordValidation, RunCliffordCircuitStillSucceeds) {
    // Positive control: a genuine Clifford circuit runs without throwing.
    QuantumCircuit qc(2);
    qc.h(0);
    qc.cx(0, 1);
    CliffordSimulator sim;
    EXPECT_NO_THROW(sim.run(qc, /*shots=*/16));
}

// =============================================================================
// Message format
// =============================================================================

TEST(R1191CliffordValidation, MessageFormatMatchesCircuitLayer) {
    StabilizerState st(kN);
    try {
        st.apply_h(88);
        FAIL() << "expected std::out_of_range";
    } catch (const std::out_of_range& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("StabilizerState::apply_h"), std::string::npos);
        EXPECT_NE(msg.find("88"), std::string::npos);
        EXPECT_NE(msg.find("out of range [0, 3)"), std::string::npos);
    }
}
