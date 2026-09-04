// R.1.12.1 total-coverage suite, Batch 2: lindblad/operators.hpp QuantumInfo
// metrics. Plan: docs (R.1.12.1 coverage plan), section "Batch 2: engines".
//
// Pins fidelity/entropy/entanglement/concurrence against known closed-form
// values and checks the DensityMatrix and Statevector partial_trace paths
// agree on an asymmetric state. Test-only release content.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace lindblad;

namespace {

constexpr double kTol = 1e-9;

// Build a state from raw amplitudes. The default policy is the library's own,
// so a caller passing a vector that is not normalized is refused: every caller
// below hands over an already-normalized vector and is held to that. The one
// test that deliberately starts off-normalization passes Fix and says so.
Statevector ket(const std::vector<Complex128>& amps,
                ValidationOptions validation = {}) {
    Statevector sv(static_cast<int>(std::log2(amps.size())));
    sv.set_amplitudes(amps, validation);
    return sv;
}

Statevector bell() {
    return ket({Complex128(INV_SQRT2, 0), Complex128(0, 0),
                Complex128(0, 0), Complex128(INV_SQRT2, 0)});
}

}  // namespace

// =============================================================================
// state_fidelity
// =============================================================================

TEST(R1121QuantumInfo, StateFidelityPureKnownValues) {
    auto zero = ket({Complex128(1, 0), Complex128(0, 0)});
    auto one = ket({Complex128(0, 0), Complex128(1, 0)});
    auto plus = ket({Complex128(INV_SQRT2, 0), Complex128(INV_SQRT2, 0)});

    EXPECT_NEAR(QuantumInfo::state_fidelity(zero, zero), 1.0, kTol);
    EXPECT_NEAR(QuantumInfo::state_fidelity(zero, one), 0.0, kTol);
    EXPECT_NEAR(QuantumInfo::state_fidelity(zero, plus), 0.5, kTol);
}

TEST(R1121QuantumInfo, StateFidelityDensityMatrixPath) {
    auto z = ket({Complex128(1, 0), Complex128(0, 0)});
    auto p = ket({Complex128(INV_SQRT2, 0), Complex128(INV_SQRT2, 0)});
    auto rz = DensityMatrix::from_statevector(z);
    auto rp = DensityMatrix::from_statevector(p);
    EXPECT_NEAR(QuantumInfo::state_fidelity(rz, rz), 1.0, kTol);
    EXPECT_NEAR(QuantumInfo::state_fidelity(rz, rp), 0.5, kTol);
}

// =============================================================================
// process_fidelity / average_gate_fidelity
// =============================================================================

TEST(R1121QuantumInfo, ProcessAndAverageGateFidelity) {
    Operator I({Complex128(1, 0), Complex128(0, 0),
                Complex128(0, 0), Complex128(1, 0)}, 1);
    Operator Z({Complex128(1, 0), Complex128(0, 0),
                Complex128(0, 0), Complex128(-1, 0)}, 1);

    // F_proc is squared: |Tr(U†V)|²/d².
    EXPECT_NEAR(QuantumInfo::process_fidelity(I, I), 1.0, kTol);
    EXPECT_NEAR(QuantumInfo::process_fidelity(Z, Z), 1.0, kTol);
    EXPECT_NEAR(QuantumInfo::process_fidelity(I, Z), 0.0, kTol);  // Tr(Z)=0

    // F_avg = (d*F_proc + 1)/(d + 1), d = 2.
    EXPECT_NEAR(QuantumInfo::average_gate_fidelity(I, I), 1.0, kTol);
    EXPECT_NEAR(QuantumInfo::average_gate_fidelity(I, Z), 1.0 / 3.0, kTol);
}

TEST(R1121QuantumInfo, ProcessAndAverageGateFidelityRzFamily) {
    // F_proc(RZ(a), RZ(b)) = cos^2((a-b)/2); F_avg = (2*F_proc + 1)/3 at d=2.
    auto rz_op = [](double theta) {
        QuantumCircuit qc(1);
        qc.rz(theta, 0);
        return Operator::from_circuit(qc);
    };
    for (double a : {0.0, 0.5, 1.3, PI / 2, PI}) {
        for (double b : {0.0, 0.7, -1.1, PI}) {
            SCOPED_TRACE("a=" + std::to_string(a) + " b=" + std::to_string(b));
            double c = std::cos((a - b) / 2.0);
            double fproc = c * c;
            EXPECT_NEAR(QuantumInfo::process_fidelity(rz_op(a), rz_op(b)), fproc, 1e-7);
            EXPECT_NEAR(QuantumInfo::average_gate_fidelity(rz_op(a), rz_op(b)),
                        (2.0 * fproc + 1.0) / 3.0, 1e-7);
        }
    }
}

// =============================================================================
// entropy
// =============================================================================

TEST(R1121QuantumInfo, EntropyNonUniformBaseConversion) {
    // S_2(rho) = S_e(rho) / ln 2 for any rho; check on a non-uniform spectrum.
    DensityMatrix rho(1);
    rho(0, 0) = Complex128(0.25, 0);
    rho(1, 1) = Complex128(0.75, 0);
    rho(0, 1) = Complex128(0, 0);
    rho(1, 0) = Complex128(0, 0);
    const double s2 = -0.25 * std::log2(0.25) - 0.75 * std::log2(0.75);
    EXPECT_NEAR(QuantumInfo::entropy(rho, 2.0), s2, kTol);
    EXPECT_NEAR(QuantumInfo::entropy(rho, std::exp(1.0)), s2 * std::log(2.0), kTol);
}

TEST(R1121QuantumInfo, EntropyPureIsZeroMixedIsLog) {
    auto pure = DensityMatrix::from_statevector(
        ket({Complex128(1, 0), Complex128(0, 0)}));
    EXPECT_NEAR(QuantumInfo::entropy(pure), 0.0, kTol);

    DensityMatrix mm(1);
    mm(0, 0) = Complex128(0.5, 0);
    mm(1, 1) = Complex128(0.5, 0);
    EXPECT_NEAR(QuantumInfo::entropy(mm, 2.0), 1.0, kTol);          // 1 bit
    EXPECT_NEAR(QuantumInfo::entropy(mm, std::exp(1.0)), std::log(2.0), kTol);
}

// =============================================================================
// entanglement_entropy / concurrence
// =============================================================================

TEST(R1121QuantumInfo, EntanglementEntropyProductVsBell) {
    auto product = ket({Complex128(1, 0), Complex128(0, 0),
                        Complex128(0, 0), Complex128(0, 0)});  // |00>
    EXPECT_NEAR(QuantumInfo::entanglement_entropy(product, {0}), 0.0, kTol);
    EXPECT_NEAR(QuantumInfo::entanglement_entropy(bell(), {0}), 1.0, kTol);
}

TEST(R1121QuantumInfo, ConcurrenceBellIsOneProductIsZero) {
    EXPECT_NEAR(QuantumInfo::concurrence(DensityMatrix::from_statevector(bell())),
                1.0, 1e-7);
    auto product = ket({Complex128(1, 0), Complex128(0, 0),
                        Complex128(0, 0), Complex128(0, 0)});
    EXPECT_NEAR(QuantumInfo::concurrence(DensityMatrix::from_statevector(product)),
                0.0, 1e-7);
}

TEST(R1121QuantumInfo, ConcurrenceWernerStateClosedForm) {
    // Werner state rho = p|Phi+><Phi+| + (1-p) I/4 has concurrence
    // C = max(0, (3p-1)/2) (local-unitary invariant; same for any Bell state).
    auto werner = [](double p) {
        DensityMatrix rho(2);
        for (size_t i = 0; i < 4; ++i)
            for (size_t j = 0; j < 4; ++j) rho(i, j) = Complex128(0, 0);
        const double iden = (1.0 - p) / 4.0;
        rho(0, 0) = Complex128(0.5 * p + iden, 0);
        rho(3, 3) = Complex128(0.5 * p + iden, 0);
        rho(0, 3) = Complex128(0.5 * p, 0);
        rho(3, 0) = Complex128(0.5 * p, 0);
        rho(1, 1) = Complex128(iden, 0);
        rho(2, 2) = Complex128(iden, 0);
        return rho;
    };
    for (double p : {1.0, 0.7, 1.0 / 3.0, 0.2}) {
        SCOPED_TRACE("p = " + std::to_string(p));
        double expected = std::max(0.0, (3.0 * p - 1.0) / 2.0);
        EXPECT_NEAR(QuantumInfo::concurrence(werner(p)), expected, 1e-6);
    }
}

// =============================================================================
// partial_trace — DM path and SV path agree (asymmetric state)
// =============================================================================

TEST(R1121QuantumInfo, PartialTraceNonContiguousUnsortedAndTraceAll) {
    // Product state: q0 = |0>, q1 = |+>, q2 = 0.6|0> + 0.8|1>.
    // amp[q0 + 2 q1 + 4 q2] = a[q0] b[q1] c[q2].
    const double s = INV_SQRT2;
    std::vector<Complex128> amps(8, Complex128(0, 0));
    amps[0] = Complex128(0.6 * s, 0);  // q1=0,q2=0
    amps[2] = Complex128(0.6 * s, 0);  // q1=1,q2=0
    amps[4] = Complex128(0.8 * s, 0);  // q1=0,q2=1
    amps[6] = Complex128(0.8 * s, 0);  // q1=1,q2=1
    auto sv = ket(amps);

    // Keep qubit 1 (trace out {0,2}); unsorted list must give the same result.
    auto r1 = QuantumInfo::partial_trace(sv, {0, 2});
    auto r1u = QuantumInfo::partial_trace(sv, {2, 0});
    ASSERT_EQ(r1.dim, 2u);
    EXPECT_NEAR(r1(0, 0).real, 0.5, kTol);
    EXPECT_NEAR(r1(0, 1).real, 0.5, kTol);  // |+><+|
    EXPECT_NEAR(r1(1, 0).real, 0.5, kTol);
    EXPECT_NEAR(r1(1, 1).real, 0.5, kTol);
    for (size_t i = 0; i < 4; ++i)
        EXPECT_NEAR(r1.data[i].real, r1u.data[i].real, kTol) << "unsorted trace-out " << i;

    // Keep qubit 2 (trace out {0,1}): rho2 = [[0.36,0.48],[0.48,0.64]].
    auto r2 = QuantumInfo::partial_trace(sv, {0, 1});
    EXPECT_NEAR(r2(0, 0).real, 0.36, kTol);
    EXPECT_NEAR(r2(0, 1).real, 0.48, kTol);
    EXPECT_NEAR(r2(1, 1).real, 0.64, kTol);

    // Keep qubits {1,2} (trace out {0}): a pure product -> purity 1, trace 1.
    auto r12 = QuantumInfo::partial_trace(sv, {0});
    ASSERT_EQ(r12.dim, 4u);
    EXPECT_NEAR(r12.trace(), 1.0, kTol);
    EXPECT_NEAR(r12.purity(), 1.0, 1e-9) << "product substate stays pure";

    // Trace out everything -> 1x1 matrix with trace 1.
    auto rall = QuantumInfo::partial_trace(sv, {0, 1, 2});
    ASSERT_EQ(rall.dim, 1u);
    EXPECT_NEAR(rall(0, 0).real, 1.0, kTol);
}

TEST(R1121QuantumInfo, PartialTraceDensityAndStatevectorPathsAgree) {
    // An asymmetric, non-maximally-entangled 2-qubit state.
    auto sv = ket({Complex128(0.5, 0.1), Complex128(-0.2, 0.4),
                   Complex128(0.3, -0.3), Complex128(0.5, 0.2)},
                  {Validation::Throw, DEFAULT_PHYSICAL_ATOL, Repair::Attempt});
    auto rho = DensityMatrix::from_statevector(sv);

    auto from_sv = QuantumInfo::partial_trace(sv, {1});
    auto from_rho = QuantumInfo::partial_trace(rho, {1});
    ASSERT_EQ(from_sv.dim, from_rho.dim);
    for (size_t i = 0; i < from_sv.data.size(); ++i) {
        EXPECT_NEAR(from_sv.data[i].real, from_rho.data[i].real, kTol) << "re " << i;
        EXPECT_NEAR(from_sv.data[i].imag, from_rho.data[i].imag, kTol) << "im " << i;
    }
    // Reduced state keeps unit trace.
    EXPECT_NEAR(from_sv.trace(), 1.0, kTol);
}

// =============================================================================
// pauli_expectation_values
// =============================================================================

TEST(R1121QuantumInfo, PauliExpectationValuesOnComputationalBasis) {
    auto zero = ket({Complex128(1, 0), Complex128(0, 0)});
    auto ev = QuantumInfo::pauli_expectation_values(zero, {"Z", "X", "Y"});
    ASSERT_EQ(ev.size(), 3u);
    EXPECT_NEAR(ev[0], 1.0, kTol);  // <Z> = +1 on |0>
    EXPECT_NEAR(ev[1], 0.0, kTol);  // <X> = 0
    EXPECT_NEAR(ev[2], 0.0, kTol);  // <Y> = 0

    auto plus = ket({Complex128(INV_SQRT2, 0), Complex128(INV_SQRT2, 0)});
    auto evp = QuantumInfo::pauli_expectation_values(plus, {"X", "Z"});
    EXPECT_NEAR(evp[0], 1.0, kTol);  // <X> = +1 on |+>
    EXPECT_NEAR(evp[1], 0.0, kTol);  // <Z> = 0
}
