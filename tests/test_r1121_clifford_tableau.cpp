// R.1.12.1 total-coverage suite, Batch 2: lindblad/simulators/clifford_sim.hpp
// (StabilizerState, CliffordSimulator). Plan section "Batch 2: engines".
//
// Pins Clifford gate conjugation through expectation_pauli (with asymmetric
// LSB-first strings), deterministic measurement, is_clifford acceptance, and
// CliffordSimulator agreement with the statevector simulator on measured
// circuits. Test-only release content.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"

#include <cmath>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using namespace lindblad;

// =============================================================================
// expectation_pauli — single-qubit conjugation
// =============================================================================

TEST(R1121Clifford, GroundStateStabilizers) {
    StabilizerState s(1);
    EXPECT_EQ(s.expectation_pauli("Z"), 1);   // |0> is +1 eigenstate of Z
    EXPECT_EQ(s.expectation_pauli("X"), 0);   // X indeterminate
}

TEST(R1121Clifford, HadamardSwapsXandZ) {
    StabilizerState s(1);
    s.apply_h(0);
    EXPECT_EQ(s.expectation_pauli("X"), 1);   // |+> is +1 eigenstate of X
    EXPECT_EQ(s.expectation_pauli("Z"), 0);
}

TEST(R1121Clifford, SAfterHGivesYEigenstate) {
    StabilizerState s(1);
    s.apply_h(0);
    s.apply_s(0);            // |+> -> |+i>
    EXPECT_EQ(s.expectation_pauli("Y"), 1);
    EXPECT_EQ(s.expectation_pauli("X"), 0);
}

TEST(R1121Clifford, XFlipsZSign) {
    StabilizerState s(1);
    s.apply_x(0);
    EXPECT_EQ(s.expectation_pauli("Z"), -1);  // |1>
}

// Full single-qubit conjugation table: for each gate G and each prepared
// eigenstate, expectation_pauli on (X, Y, Z) must equal the analytic
// expectation of G applied to that state (sign in {-1, 0, +1}).
namespace {
enum class Prep { Zero, Plus, PlusI };   // |0> (+Z), |+> (+X), |+i> (+Y)
enum class G1 { H, S, SDG, X, Y, Z };

void prepare(StabilizerState& s, Prep p) {
    switch (p) {
        case Prep::Zero: break;
        case Prep::Plus: s.apply_h(0); break;
        case Prep::PlusI: s.apply_h(0); s.apply_s(0); break;
    }
}
void apply_g1(StabilizerState& s, G1 g) {
    switch (g) {
        case G1::H: s.apply_h(0); break;
        case G1::S: s.apply_s(0); break;
        case G1::SDG: s.apply_sdg(0); break;
        case G1::X: s.apply_x(0); break;
        case G1::Y: s.apply_y(0); break;
        case G1::Z: s.apply_z(0); break;
    }
}
}  // namespace

TEST(R1121Clifford, SingleQubitConjugationTable) {
    struct Cell { Prep prep; G1 gate; int ex, ey, ez; };
    const Cell table[] = {
        // input |0> (+Z)
        {Prep::Zero, G1::H,   1, 0, 0}, {Prep::Zero, G1::S,   0, 0, 1},
        {Prep::Zero, G1::SDG, 0, 0, 1}, {Prep::Zero, G1::X,   0, 0,-1},
        {Prep::Zero, G1::Y,   0, 0,-1}, {Prep::Zero, G1::Z,   0, 0, 1},
        // input |+> (+X)
        {Prep::Plus, G1::H,   0, 0, 1}, {Prep::Plus, G1::S,   0, 1, 0},
        {Prep::Plus, G1::SDG, 0,-1, 0}, {Prep::Plus, G1::X,   1, 0, 0},
        {Prep::Plus, G1::Y,  -1, 0, 0}, {Prep::Plus, G1::Z,  -1, 0, 0},
        // input |+i> (+Y)
        {Prep::PlusI, G1::H,  0,-1, 0}, {Prep::PlusI, G1::S,  -1, 0, 0},
        {Prep::PlusI, G1::SDG, 1, 0, 0}, {Prep::PlusI, G1::X,  0,-1, 0},
        {Prep::PlusI, G1::Y,  0, 1, 0}, {Prep::PlusI, G1::Z,  0,-1, 0},
    };
    for (const Cell& c : table) {
        StabilizerState s(1);
        prepare(s, c.prep);
        apply_g1(s, c.gate);
        SCOPED_TRACE("prep " + std::to_string(int(c.prep)) +
                     " gate " + std::to_string(int(c.gate)));
        EXPECT_EQ(s.expectation_pauli("X"), c.ex);
        EXPECT_EQ(s.expectation_pauli("Y"), c.ey);
        EXPECT_EQ(s.expectation_pauli("Z"), c.ez);
    }
}

// =============================================================================
// expectation_pauli — asymmetric LSB convention on 2 qubits
// =============================================================================

TEST(R1121Clifford, ExpectationPauliIsLsbFirst) {
    StabilizerState s(2);
    s.apply_x(0);  // q0 = 1, q1 = 0
    // "ZI" = Z on qubit 0, I on qubit 1.
    EXPECT_EQ(s.expectation_pauli("ZI"), -1) << "Z on flipped qubit 0";
    EXPECT_EQ(s.expectation_pauli("IZ"), 1) << "Z on untouched qubit 1";
}

TEST(R1121Clifford, CxCreatesZZCorrelation) {
    StabilizerState s(2);
    s.apply_h(0);
    s.apply_cx(0, 1);  // Bell state
    EXPECT_EQ(s.expectation_pauli("ZZ"), 1) << "Bell state is +1 eigenstate of ZZ";
    EXPECT_EQ(s.expectation_pauli("XX"), 1) << "and of XX";
}

// =============================================================================
// Deterministic measurement
// =============================================================================

TEST(R1121Clifford, DeterministicMeasurementOutcomes) {
    std::mt19937_64 rng(123);
    StabilizerState s0(1);
    EXPECT_EQ(s0.measure(0, true, rng), 0);  // |0> -> 0 deterministically

    StabilizerState s1(1);
    s1.apply_x(0);
    EXPECT_EQ(s1.measure(0, true, rng), 1);  // |1> -> 1 deterministically
}

// =============================================================================
// is_clifford acceptance
// =============================================================================

TEST(R1121Clifford, IsCliffordAcceptsCliffordRejectsNonClifford) {
    QuantumCircuit cliff(2);
    cliff.h(0).s(1).cx(0, 1).x(0).y(1).z(0).sdg(1);
    EXPECT_TRUE(CliffordSimulator::is_clifford(cliff));

    QuantumCircuit with_t(1);
    with_t.t(0);
    EXPECT_FALSE(CliffordSimulator::is_clifford(with_t));

    QuantumCircuit with_rot(1);
    with_rot.rx(0.7, 0);
    EXPECT_FALSE(CliffordSimulator::is_clifford(with_rot));
}

TEST(R1121Clifford, IsCliffordPhaseGateAngleGrid) {
    // P(angle) is Clifford iff angle is a multiple of pi/2 (incl. the 2*pi
    // boundary and small negative inputs that fmod to ~0).
    const double pi = PI;
    for (double a : {0.0, pi / 2, pi, 3 * pi / 2, 2 * pi, -1e-12}) {
        QuantumCircuit qc(1);
        qc.p(a, 0);
        EXPECT_TRUE(CliffordSimulator::is_clifford(qc))
            << "P(" << a << ") should be Clifford";
    }
    for (double a : {pi / 4, pi / 3, 0.7, pi / 8}) {
        QuantumCircuit qc(1);
        qc.p(a, 0);
        EXPECT_FALSE(CliffordSimulator::is_clifford(qc))
            << "P(" << a << ") is NOT Clifford";
    }
}

TEST(R1121Clifford, MeasurementCollapseIsRepeatable) {
    // |+> measured once is random; once collapsed, Z is definite and a repeat
    // measure returns the SAME outcome.
    std::mt19937_64 rng(2024);
    StabilizerState s(1);
    s.apply_h(0);  // |+>: Z indeterminate
    EXPECT_EQ(s.expectation_pauli("Z"), 0);
    int first = s.measure(0, true, rng);
    EXPECT_TRUE(first == 0 || first == 1);
    EXPECT_EQ(s.expectation_pauli("Z"), first == 0 ? 1 : -1)
        << "post-measure Z is definite and matches the outcome";
    EXPECT_EQ(s.measure(0, true, rng), first) << "re-measure repeats the outcome";
}

// =============================================================================
// CliffordSimulator vs statevector — measured circuits
// =============================================================================

namespace {
// Total-variation distance between two empirical counts maps.
double clifford_tv(const std::unordered_map<std::string, int>& a,
                   const std::unordered_map<std::string, int>& b) {
    int ta = 0, tb = 0;
    for (auto& [k, n] : a) ta += n;
    for (auto& [k, n] : b) tb += n;
    if (ta == 0 || tb == 0) return 1.0;
    double tv = 0.0;
    std::vector<std::string> keys;
    for (auto& [k, n] : a) keys.push_back(k);
    for (auto& [k, n] : b)
        if (a.find(k) == a.end()) keys.push_back(k);
    for (const auto& k : keys) {
        double pa = a.count(k) ? double(a.at(k)) / ta : 0.0;
        double pb = b.count(k) ? double(b.at(k)) / tb : 0.0;
        tv += std::abs(pa - pb);
    }
    return 0.5 * tv;
}
}  // namespace

TEST(R1121Clifford, CzAndSwapDecompositionsMatchStatevector) {
    // Random Clifford circuits over {h,s,sdg,x,y,z,cx,cz,swap}; the stabilizer
    // simulator (which decomposes cz/swap) must agree with the statevector
    // simulator distribution. cz and swap appear in every circuit.
    std::mt19937_64 gen(99);
    for (int trial = 0; trial < 6; ++trial) {
        const int n = 3;
        QuantumCircuit qc(n, n);
        qc.cz(0, 1);
        qc.swap(1, 2);
        std::uniform_int_distribution<int> pick(0, 6);
        std::uniform_int_distribution<int> qd(0, n - 1);
        for (int g = 0; g < 12; ++g) {
            int q = qd(gen), q2 = (q + 1) % n;
            switch (pick(gen)) {
                case 0: qc.h(q); break;
                case 1: qc.s(q); break;
                case 2: qc.sdg(q); break;
                case 3: qc.x(q); break;
                case 4: qc.cx(q, q2); break;
                case 5: qc.cz(q, q2); break;
                case 6: qc.swap(q, q2); break;
            }
        }
        qc.measure_all();
        CliffordSimulator csim;
        StatevectorSimulator ssim;
        auto cc = csim.run(qc, 6000, 31 + trial).counts;
        auto sc = ssim.run(qc, 6000, 53 + trial).counts;
        SCOPED_TRACE("trial " + std::to_string(trial));
        EXPECT_LT(clifford_tv(cc, sc), 0.07)
            << "Clifford and statevector distributions must agree";
    }
}

TEST(R1121Clifford, DeterministicCircuitGivesSingleOutcome) {
    QuantumCircuit qc(2, 2);
    qc.x(0).cx(0, 1).measure_all();  // |00> -> |11>
    CliffordSimulator sim;
    auto res = sim.run(qc, 256, 7);
    ASSERT_EQ(res.counts.size(), 1u);
    EXPECT_EQ(res.counts.begin()->first, "11");
    EXPECT_EQ(res.counts.begin()->second, 256);
}

TEST(R1121Clifford, BellOnlyCorrelatedOutcomesMatchStatevector) {
    QuantumCircuit qc(2, 2);
    qc.h(0).cx(0, 1).measure_all();

    CliffordSimulator csim;
    auto cres = csim.run(qc, 4000, 11);
    // Only "00" and "11" may appear; "01"/"10" are forbidden by the stabilizers.
    for (const auto& [bits, n] : cres.counts) {
        EXPECT_TRUE(bits == "00" || bits == "11") << "unexpected key " << bits;
    }
    int total = 0;
    for (const auto& [bits, n] : cres.counts) total += n;
    EXPECT_EQ(total, 4000);
    EXPECT_GT(cres.counts.count("00") ? cres.counts.at("00") : 0, 1500);
    EXPECT_GT(cres.counts.count("11") ? cres.counts.at("11") : 0, 1500);

    // Statevector run agrees on the support set.
    StatevectorSimulator ssim;
    auto sres = ssim.run(qc, 4000, 11);
    for (const auto& [bits, n] : sres.counts)
        EXPECT_TRUE(bits == "00" || bits == "11") << "SV unexpected key " << bits;
}

TEST(R1121Clifford, GhzThreeQubitOnlyAllZeroOrAllOne) {
    QuantumCircuit qc(3, 3);
    qc.h(0).cx(0, 1).cx(1, 2).measure_all();
    CliffordSimulator sim;
    auto res = sim.run(qc, 2000, 5);
    for (const auto& [bits, n] : res.counts)
        EXPECT_TRUE(bits == "000" || bits == "111") << "unexpected key " << bits;
}
