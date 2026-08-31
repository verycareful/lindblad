// R.1.13.1 test patch — MPS SVD backend toggle + contraction + sampling.
// Covers audit F-23 (SVDMethod: Jacobi default, BDC opt-in with a loud broken
// warning), F-6 (zero-copy Eigen-GEMM two-site contraction), and F-3/F-4
// (read-only hoisted sampling). Correctness is checked against the exact
// statevector backend; the BDC warning message is captured from std::cerr.
//
// NOTE: this file is the ONLY place in the test binary that selects
// SVDMethod::BDC on the qubit MPS layer. warn_bdc_broken_once() latches a
// static flag belonging to that layer alone, so the first BDC selection in the
// qubit MPS wins and every later one is silent. The qudit layer carries its own
// separate latch, which is what lets its twin test still see its own warning.
// Do not add BDC selection to other qubit-MPS tests.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/types.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"

#include <array>
#include <iostream>
#include <sstream>
#include <string>

using namespace lindblad;

namespace {

// Short local alias, library-sourced value (the literal was one ULP low).
constexpr double kInv2 = INV_SQRT2;

std::array<Complex128, 4> hadamard() {
    return {Complex128(kInv2, 0), Complex128(kInv2, 0),
            Complex128(kInv2, 0), Complex128(-kInv2, 0)};
}

// A CNOT as a 4x4 (any valid two-qubit unitary triggers svd_truncate).
std::array<Complex128, 16> cnot() {
    std::array<Complex128, 16> U{};
    // basis order per apply_two_qubit_gate (LSB = first operand): identity on
    // |00>,|01>, swap |10><->|11>. Exact orientation is irrelevant to the SVD
    // path being exercised — only that it is a genuine 2-qubit unitary.
    U[0]  = Complex128(1, 0);
    U[5]  = Complex128(1, 0);
    U[11] = Complex128(1, 0);
    U[14] = Complex128(1, 0);
    return U;
}

std::vector<Complex128> run_sv(const QuantumCircuit& qc) {
    StatevectorSimulator sim;
    return sim.run(qc, 0, 0).final_state.amplitudes();
}

std::vector<Complex128> run_mps(const QuantumCircuit& qc, int chi = 64) {
    MPSSimulator sim;
    return sim.run(qc, chi, 0, 0).final_state.to_statevector().amplitudes();
}

void expect_amps_close(const std::vector<Complex128>& a,
                       const std::vector<Complex128>& b, double tol = 1e-9) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i].real, b[i].real, tol) << "re @ " << i;
        EXPECT_NEAR(a[i].imag, b[i].imag, tol) << "im @ " << i;
    }
}

} // namespace

// The default backend, pinned. Truncation values move with it, so a default
// that drifts silently changes every state the MPS engine produces; the qudit
// layer carries its own copy of this assertion.
TEST(R1131Mps, DefaultSvdMethodIsBdc) {
    MPSState mps(3);
    EXPECT_EQ(mps.svd_method, SVDMethod::BDC)
        << "the qubit MPS SVD default moved. Truncation values shift with the "
           "backend, so this is a behaviour change and not a preference.";
}

// F-6: the two-site GEMM contraction (adjacent and SWAP-chained non-adjacent)
// must reproduce the exact statevector for an entangling circuit.
TEST(R1131Mps, TwoQubitContractionMatchesStatevector) {
    QuantumCircuit qc(4);
    qc.h(0).cx(0, 1).cx(1, 2);
    qc.cx(0, 3);                 // non-adjacent -> SWAP chain
    qc.rzz(0.6, 1, 3).ry(0.4, 2).cx(2, 3);
    expect_amps_close(run_mps(qc), run_sv(qc));
}

// F-3/F-4: read-only hoisted sampling reproduces the correct distribution.
TEST(R1131Mps, SamplingReproducesGhzDistribution) {
    QuantumCircuit qc(3, 3);
    qc.h(0).cx(0, 1).cx(1, 2).measure_all();

    MPSSimulator sim;
    auto res = sim.run(qc, 64, /*shots=*/4000, /*seed=*/42);

    int total = 0;
    for (const auto& [k, v] : res.counts) {
        EXPECT_TRUE(k == "000" || k == "111")
            << "GHZ must only yield 000 or 111, got " << k;
        total += v;
    }
    EXPECT_EQ(total, 4000);
    EXPECT_GT(res.counts["000"], 1200);   // both ~2000; comfortably > 1200
    EXPECT_GT(res.counts["111"], 1200);
}

// Selecting the non-default backend emits a one-time note saying what it
// costs. A caller who chose Jacobi for accuracy reasons is opting out of the
// faster algorithm without being told, which is the thing the note prevents.
//
// The latch fires once per layer, and this is the only qubit-MPS Jacobi
// selection in the binary, so this test is the only one that can ever observe
// the qubit text.
TEST(R1131Mps, JacobiSelectionEmitsSlowerNote) {
    std::ostringstream capture;
    std::streambuf* old = std::cerr.rdbuf(capture.rdbuf());

    {
        MPSState mps(2);
        mps.svd_method = SVDMethod::Jacobi;
        mps.apply_single_qubit_gate(hadamard(), 0);
        mps.apply_two_qubit_gate(cnot(), 0, 1);   // forces svd_truncate -> note
    }

    std::cerr.rdbuf(old);
    const std::string out = capture.str();
    EXPECT_NE(out.find("Jacobi"), std::string::npos)
        << "the note was not emitted; got: [" << out << "]";
    EXPECT_NE(out.find("BDC is the default"), std::string::npos)
        << "the note must name what the caller is opting out of; got: [" << out
        << "]";
    EXPECT_NE(out.find("qubit MPS"), std::string::npos)
        << "the note must name its layer, since the qudit layer latches "
           "separately and a reader seeing one needs to know which; got: ["
        << out << "]";
    EXPECT_EQ(out.find("BROKEN"), std::string::npos)
        << "nothing here is broken: both backends are accepted by the verify "
           "rung on the first attempt, and the note is about speed; got: ["
        << out << "]";

    // The message must point at nothing the reader cannot reach. A note that
    // cites a path absent from a published clone sends the reader nowhere, and
    // the latch means no other test can catch it.
    for (const char* unreachable : {"docs/plans", "docs/superpowers",
                                    "Audit docs", ".md"}) {
        EXPECT_EQ(out.find(unreachable), std::string::npos)
            << "the warning cites '" << unreachable
            << "', which is absent from a published clone; got: [" << out << "]";
    }
}
