// R.1.22.1 test wave - is a physical-validity policy OBEYED where the matrix
// is used?
//
// This is the missing half of the R.1.21.1 wave. tests/test_r1211_propagation.cpp
// covers whether the policy SURVIVES BEING CARRIED: plain copy, compose,
// repeat, inverse, control() and the DAG and JSON round trips. Every test there
// asks whether the setting arrives. None asks whether anything reads it.
//
// The distinction is not academic. The policy arrives perfectly at every
// consumer, and three of four consumers then apply the matrix without
// consulting it, so a circuit the library accepted becomes one that cannot run.
// R1211Propagation could not see that, and one detail explains why: its
// ControlOfAnOptedOutMatrixStaysRunnable asserts
// QuantumCircuit::validate_physical(), the run() pre-flight, which happens to
// be the single consumer implemented correctly.
//
// So the shape of every test below is the same and deliberately end to end:
// build a circuit the library ACCEPTS because the caller opted out, then ask
// each backend to run it. Anything that throws has substituted its own policy
// for the caller's.
//
// The operand is a deliberately non-unitary matrix. Under Validation::Ignore
// that is a legitimate thing to put in a circuit, which is exactly why losing
// the Ignore matters: the circuit stops running rather than starting to produce
// wrong numbers.

#include <gtest/gtest.h>

#include "r1211_policy_probe.hpp"

#include "lindblad/circuit.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/transpiler.hpp"
#include "lindblad/validation.hpp"

#include <string>
#include <vector>

using namespace lindblad;

namespace {

// Identity with one diagonal entry stretched. The unitarity residual is
// |1.5^2 - 1| = 1.25, which no tolerance in the project would accept, so a
// backend that checks at all will be seen to check.
std::vector<Complex128> stretched_identity(int k) {
    const size_t rows = size_t(1) << k;
    std::vector<Complex128> m(rows * rows, Complex128(0.0, 0.0));
    for (size_t i = 0; i < rows; ++i) m[i * rows + i] = Complex128(1.0, 0.0);
    m[0] = Complex128(1.5, 0.0);
    return m;
}

std::vector<int> first_k(int k) {
    std::vector<int> t(static_cast<size_t>(k));
    for (int i = 0; i < k; ++i) t[static_cast<size_t>(i)] = i;
    return t;
}

// A circuit holding one k-qubit UNITARY the caller has opted out of checking.
// Building it is itself part of the contract: under Ignore or Warn the
// constructor must accept the operand, so a throw here would be a different
// bug from the ones below.
QuantumCircuit opted_out(int n, int k, Validation policy) {
    QuantumCircuit qc(n);
    qc.unitary(stretched_identity(k), first_k(k), "opted_out", {policy});
    return qc;
}

// Four adjacent 2-qubit UNITARY operands, which is what ConsolidateBlocks needs
// before it will attempt a decomposition at all: a block of one is returned by
// the early-out without the decomposition being reached.
QuantumCircuit opted_out_block(Validation policy) {
    QuantumCircuit qc(2);
    const auto m = stretched_identity(2);
    for (int i = 0; i < 4; ++i) qc.unitary(m, {0, 1}, "opted_out", {policy});
    return qc;
}

constexpr int kWidths[] = {1, 2, 3};

// The backends do not agree on how a rejected circuit surfaces, and asserting
// the wrong one produces a test that cannot fail.
//
// StatevectorSimulator::run and DensityMatrixSimulator::run wrap their bodies in
// a try/catch and report through Result::success, so EXPECT_NO_THROW on either
// is vacuous: it holds whatever the backend decided. MPSSimulator::run has no
// such catch and surfaces by throwing, so there is no Result flag to read.
//
// Hence two helpers rather than one. Each states the SAME contract, "this
// backend accepted the circuit", in the only terms its backend offers.
::testing::AssertionResult accepted(bool success, const std::string& message) {
    if (success) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure()
           << "the backend rejected a circuit the library accepted: " << message;
}

}  // namespace

// =============================================================================
// The pre-flight, which is the consumer that gets this right
// =============================================================================

TEST(R1221PolicyAtPointOfUse, ThePreflightHonoursTheOptOut) {
    for (int k : kWidths) {
        SCOPED_TRACE("k = " + std::to_string(k));
        EXPECT_NO_THROW(opted_out(4, k, Validation::Ignore).validate_physical())
            << "validate_physical() skips an instruction under Ignore, so the "
               "circuit is legal and every backend below is being handed a "
               "circuit the library accepts";
    }
}

// =============================================================================
// Per backend, end to end
// =============================================================================

TEST(R1221PolicyAtPointOfUse, StatevectorRunsAnOptedOutUnitary) {
    for (int k : kWidths) {
        SCOPED_TRACE("k = " + std::to_string(k));
        StatevectorSimulator sim;
        QuantumCircuit qc = opted_out(4, k, Validation::Ignore);
        const auto r = sim.run(qc, /*shots=*/0, /*seed=*/0);
        EXPECT_TRUE(accepted(r.success, r.error_message))
            << "this backend forwards inst.validation to the kernel and is the "
               "reference the others are measured against";
    }
}

TEST(R1221PolicyAtPointOfUse, MpsRunsAnOptedOutUnitary) {
    for (int k : kWidths) {
        SCOPED_TRACE("k = " + std::to_string(k));
        MPSSimulator sim;
        QuantumCircuit qc = opted_out(4, k, Validation::Ignore);
        EXPECT_NO_THROW(sim.run(qc, /*max_bond_dim=*/64, /*shots=*/0, /*seed=*/0))
            << "the 1- and 2-qubit paths pass Validation::Ignore to the kernel "
               "explicitly; the 3-qubit path reaches the full-statevector "
               "fallback, which applies the matrix without the caller's policy";
    }
}

TEST(R1221PolicyAtPointOfUse, DensityMatrixRunsAnOptedOutUnitary) {
    const NoiseModel noise;
    for (int k : kWidths) {
        SCOPED_TRACE("k = " + std::to_string(k));
        DensityMatrixSimulator sim;
        QuantumCircuit qc = opted_out(4, k, Validation::Ignore);
        const auto r = sim.run(qc, noise, /*shots=*/0, /*seed=*/0);
        EXPECT_TRUE(accepted(r.success, r.error_message))
            << "this backend pre-flights under the caller's policy and then "
               "applies the matrix under a default one, at every width";
    }
}

TEST(R1221PolicyAtPointOfUse, TranspileDoesNotRejectAnOptedOutUnitary) {
    QuantumCircuit qc = opted_out_block(Validation::Ignore);
    EXPECT_NO_THROW(transpile(qc, CouplingMap(), {}, /*optimization_level=*/3))
        << "an optimisation pass must decline what it cannot factor, not "
           "refuse to run. A transpiler that throws changes whether a program "
           "runs, which is not its job";
}

// =============================================================================
// Warn, which must report and proceed rather than reject
// =============================================================================

// Warn is the policy where a silent escalation is hardest to notice: the caller
// asked to be told and to continue, and being told by an exception is not that.
TEST(R1221PolicyAtPointOfUse, WarnIsNotEscalatedToThrow) {
    for (int k : kWidths) {
        SCOPED_TRACE("k = " + std::to_string(k));
        r1211::WarningProbe probe;

        StatevectorSimulator sv;
        QuantumCircuit qc_sv = opted_out(4, k, Validation::Warn);
        const auto r = sv.run(qc_sv, /*shots=*/0, /*seed=*/0);
        EXPECT_TRUE(accepted(r.success, r.error_message))
            << "Warn must not become Throw";

        MPSSimulator mps;
        QuantumCircuit qc_mps = opted_out(4, k, Validation::Warn);
        EXPECT_NO_THROW(mps.run(qc_mps, /*max_bond_dim=*/64, /*shots=*/0, /*seed=*/0));

        EXPECT_GT(probe.count(), 0u)
            << "Warn must report through the warning channel, so silence here "
               "would mean the check never ran at all";
    }
}
