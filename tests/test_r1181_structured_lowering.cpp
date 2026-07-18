// R.1.18.1 test suite — HighLevelDecompose lowering of MCX / MCP / PERMUTATION.
// Covers the R.1.18.0 feature (#52): the exact gate-level lowering of the
// structured ops, the stage-0 pass, its preset composition rule, routed
// end-to-end behaviour, condition propagation, and the mandatory
// non-symmetric integer-convention check on the permutation index map.
//
// Reference strategy (same as the R.1.13.1 suite): every lowered realization
// is applied to an all-distinct NON-symmetric amplitude vector and compared
// against an independent brute-force reference, so a convention bug (LSB
// ordering, control mask, permutation direction) cannot hide behind a
// symmetric input. Lowered output is additionally alphabet-checked so a
// silently-unlowered op cannot pass by executing natively.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/gates.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/transpiler.hpp"

#include "../src/transpiler/high_level_decompose.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;
using GT = Instruction::GateType;

namespace {

constexpr double kTol = 1e-12;

// All-distinct, non-symmetric amplitude vector (not normalised: the lowered
// sequences are exact unitaries, so relative comparison is what matters and
// distinct values expose any index or phase mistake).
std::vector<Complex128> distinct_state(int nq) {
    const size_t dim = size_t(1) << nq;
    std::vector<Complex128> v(dim);
    for (size_t k = 0; k < dim; ++k)
        v[k] = Complex128(0.1 * static_cast<double>(k + 1),
                          -0.05 * static_cast<double>(k + 2));
    return v;
}

std::vector<Complex128> ref_mcx(const std::vector<Complex128>& in,
                                const std::vector<int>& controls, int target) {
    size_t cmask = 0;
    for (int c : controls) cmask |= (size_t(1) << c);
    const size_t tmask = size_t(1) << target;
    std::vector<Complex128> out(in.size());
    for (size_t k = 0; k < in.size(); ++k)
        out[k] = ((k & cmask) == cmask) ? in[k ^ tmask] : in[k];
    return out;
}

std::vector<Complex128> ref_mcp(const std::vector<Complex128>& in,
                                const std::vector<int>& qubits, double lambda) {
    size_t mask = 0;
    for (int q : qubits) mask |= (size_t(1) << q);
    const Complex128 ph = Complex128::exp_i(lambda);
    std::vector<Complex128> out = in;
    for (size_t k = 0; k < in.size(); ++k)
        if ((k & mask) == mask) out[k] = in[k] * ph;
    return out;
}

std::vector<Complex128> ref_perm(const std::vector<Complex128>& in,
                                 const std::vector<int>& qubits,
                                 const std::vector<int>& perm) {
    const int k = static_cast<int>(qubits.size());
    std::vector<Complex128> out(in.size(), Complex128(0.0, 0.0));
    for (size_t idx = 0; idx < in.size(); ++idx) {
        size_t sub = 0;
        for (int i = 0; i < k; ++i)
            if (idx & (size_t(1) << qubits[i])) sub |= (size_t(1) << i);
        const size_t nsub = static_cast<size_t>(perm[sub]);
        size_t nidx = idx;
        for (int i = 0; i < k; ++i) {
            const size_t bit = size_t(1) << qubits[i];
            if (nsub & (size_t(1) << i)) nidx |= bit; else nidx &= ~bit;
        }
        out[nidx] = in[idx];
    }
    return out;
}

void expect_amps_close(const std::vector<Complex128>& a,
                       const std::vector<Complex128>& b, double tol = kTol) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i].real, b[i].real, tol) << "re mismatch at " << i;
        EXPECT_NEAR(a[i].imag, b[i].imag, tol) << "im mismatch at " << i;
    }
}

// Run the stage-0 pass alone (public surface: PassManager + the pass class)
// and return the lowered instruction list.
std::vector<Instruction> lower_via_pass(const QuantumCircuit& qc) {
    PassManager pm;
    pm.append(std::make_unique<HighLevelDecompose>());
    TranspilationContext ctx;
    auto dag = DAGCircuit::from_circuit(qc);
    return pm.run(dag, ctx).to_circuit().instructions;
}

// The fully-lowered base alphabet the pass promises. SWAP appears only on the
// permutation relabel path; MEASURE/RESET/BARRIER pass through untouched.
bool in_lowered_alphabet(GT t) {
    switch (t) {
        case GT::X: case GT::H: case GT::P: case GT::CP:
        case GT::CX: case GT::CCX: case GT::SWAP:
        case GT::MEASURE: case GT::RESET: case GT::BARRIER:
            return true;
        default:
            return false;
    }
}

void expect_lowered_alphabet(const std::vector<Instruction>& insts) {
    for (const auto& inst : insts) {
        EXPECT_TRUE(in_lowered_alphabet(inst.type))
            << "unexpected gate '" << inst.gate_name() << "' in lowered output";
    }
}

// Apply an instruction list to a seeded statevector via the simulator's
// public single-instruction entry point.
std::vector<Complex128> apply_all(int nq,
                                  const std::vector<Complex128>& in,
                                  const std::vector<Instruction>& insts) {
    StatevectorSimulator sim;
    Statevector sv(nq);
    sv.set_amplitudes(in);
    for (const auto& inst : insts) sim.apply_instruction(sv, inst);
    return sv.amplitudes();
}

} // namespace

// =============================================================================
// MCX lowering vs brute-force reference
// =============================================================================

TEST(R1181Lowering, McxLoweredMatchesReferenceAllWidths) {
    // k = 0..6 controls, target on the top wire. k >= 3 exercises the
    // H-conjugated phase recursion and the borrowed-wire halving.
    for (int k = 0; k <= 6; ++k) {
        SCOPED_TRACE("controls k=" + std::to_string(k));
        const int nq = k + 1;
        std::vector<int> controls;
        for (int c = 0; c < k; ++c) controls.push_back(c);
        const int target = k;

        QuantumCircuit qc(nq);
        qc.mcx(controls, target);
        auto lowered = lower_via_pass(qc);
        expect_lowered_alphabet(lowered);

        auto in = distinct_state(nq);
        expect_amps_close(apply_all(nq, in, lowered),
                          ref_mcx(in, controls, target));
    }
}

TEST(R1181Lowering, McxScrambledWiresMatchesReference) {
    // Non-contiguous, non-monotonic wires on a wider register: any hidden
    // assumption that controls are low/adjacent shows up here.
    const int nq = 6;
    const std::vector<int> controls{5, 0, 3};
    const int target = 2;
    QuantumCircuit qc(nq);
    qc.mcx(controls, target);
    auto lowered = lower_via_pass(qc);
    expect_lowered_alphabet(lowered);

    auto in = distinct_state(nq);
    expect_amps_close(apply_all(nq, in, lowered),
                      ref_mcx(in, controls, target));
}

// =============================================================================
// MCP lowering vs brute-force reference
// =============================================================================

TEST(R1181Lowering, McpLoweredMatchesReferenceAllWidths) {
    const double lambda = 0.9123;  // irrational-ish: no accidental phase snap
    for (int m = 1; m <= 6; ++m) {
        SCOPED_TRACE("qubits m=" + std::to_string(m));
        const int nq = m;
        std::vector<int> qubits;
        for (int q = 0; q < m; ++q) qubits.push_back(q);

        QuantumCircuit qc(nq);
        qc.mcp(lambda, qubits);
        auto lowered = lower_via_pass(qc);
        expect_lowered_alphabet(lowered);

        auto in = distinct_state(nq);
        expect_amps_close(apply_all(nq, in, lowered),
                          ref_mcp(in, qubits, lambda));
    }
}

TEST(R1181Lowering, McpNegativeAngleScrambledWires) {
    const int nq = 5;
    const std::vector<int> qubits{4, 1, 3};
    const double lambda = -1.234567;
    QuantumCircuit qc(nq);
    qc.mcp(lambda, qubits);
    auto lowered = lower_via_pass(qc);
    expect_lowered_alphabet(lowered);

    auto in = distinct_state(nq);
    expect_amps_close(apply_all(nq, in, lowered),
                      ref_mcp(in, qubits, lambda));
}

// =============================================================================
// PERMUTATION lowering: relabel subclass, general maps, Shor-style map
// =============================================================================

TEST(R1181Lowering, PermutationRelabelLowersToSwapsOnly) {
    // sigma = cyclic wire rotation (bit j -> bit (j+1) mod 3):
    // perm[x] = bit-rotate(x). A pure relabel must produce ONLY SWAPs,
    // at most k-1 of them.
    const int k = 3;
    std::vector<int> perm(1 << k);
    for (int x = 0; x < (1 << k); ++x) {
        int img = 0;
        for (int j = 0; j < k; ++j)
            if (x & (1 << j)) img |= (1 << ((j + 1) % k));
        perm[x] = img;
    }
    const std::vector<int> qubits{0, 2, 3};  // non-contiguous placement
    const int nq = 4;
    QuantumCircuit qc(nq);
    qc.permute(perm, qubits);
    auto lowered = lower_via_pass(qc);

    ASSERT_LE(lowered.size(), size_t(k - 1));
    for (const auto& inst : lowered) EXPECT_EQ(inst.type, GT::SWAP);

    auto in = distinct_state(nq);
    expect_amps_close(apply_all(nq, in, lowered), ref_perm(in, qubits, perm));
}

TEST(R1181Lowering, PermutationGeneralMapMatchesReference) {
    // {2,0,3,1} is NOT a relabel (image of 1 is 0, not one-hot): forces the
    // transposition-synthesis path. Placed on non-adjacent wires.
    const std::vector<int> perm{2, 0, 3, 1};
    const std::vector<int> qubits{1, 3};
    const int nq = 4;
    QuantumCircuit qc(nq);
    qc.permute(perm, qubits);
    auto lowered = lower_via_pass(qc);
    expect_lowered_alphabet(lowered);

    auto in = distinct_state(nq);
    expect_amps_close(apply_all(nq, in, lowered), ref_perm(in, qubits, perm));
}

TEST(R1181Lowering, PermutationFullCycleIncrementMatchesReference) {
    // x -> (x+1) mod 8: one 8-cycle, no fixed points, maximally non-local in
    // bit space. Exercises long transposition chains.
    const int k = 3;
    std::vector<int> perm(1 << k);
    for (int x = 0; x < (1 << k); ++x) perm[x] = (x + 1) % (1 << k);
    const std::vector<int> qubits{0, 1, 2};
    const int nq = 3;
    QuantumCircuit qc(nq);
    qc.permute(perm, qubits);
    auto lowered = lower_via_pass(qc);
    expect_lowered_alphabet(lowered);

    auto in = distinct_state(nq);
    expect_amps_close(apply_all(nq, in, lowered), ref_perm(in, qubits, perm));
}

TEST(R1181Lowering, PermutationShorStyleModularMultiply) {
    // The flagship producer of general maps: x -> (a*x) mod 2^k with odd a
    // (a bijection since gcd(a, 2^k) = 1). a = 3, k = 4.
    const int k = 4;
    const int dim = 1 << k;
    std::vector<int> perm(dim);
    for (int x = 0; x < dim; ++x) perm[x] = (3 * x) % dim;
    const std::vector<int> qubits{0, 1, 2, 3};
    QuantumCircuit qc(k);
    qc.permute(perm, qubits);
    auto lowered = lower_via_pass(qc);
    expect_lowered_alphabet(lowered);

    auto in = distinct_state(k);
    expect_amps_close(apply_all(k, in, lowered), ref_perm(in, qubits, perm));
}

// =============================================================================
// Non-symmetric integer convention, end to end (mandatory per project rules)
// =============================================================================

TEST(R1181Lowering, PermutationConventionEndToEndNonSymmetric) {
    // Prepare K = 5 (q0 = 1, q2 = 1), apply the increment map through the
    // FULL preset pipeline (constrained map + basis), measure_all. LSB
    // convention: amp index K = 5 -> perm gives K = 6 -> counts key is the
    // big-endian string of 6 over 4 clbits: 0110 (q0 rightmost). A symmetric
    // value would mask a direction or ordering bug; 5 -> 6 cannot.
    const int nq = 4;
    const int dim = 1 << nq;
    std::vector<int> perm(dim);
    for (int x = 0; x < dim; ++x) perm[x] = (x + 1) % dim;

    QuantumCircuit qc(nq, nq);
    qc.x(0).x(2);                       // |0101> = K = 5
    qc.permute(perm, {0, 1, 2, 3});     // K -> 6 = |0110>
    qc.measure_all();

    for (int level = 0; level <= 3; ++level) {
        SCOPED_TRACE("level " + std::to_string(level));
        auto out = transpile(qc, CouplingMap::linear(nq), {"cx", "u3"}, level);
        StatevectorSimulator sim;
        auto res = sim.run(out, 128, 7);
        ASSERT_FALSE(res.counts.empty());
        for (const auto& [bits, count] : res.counts) {
            (void)count;
            EXPECT_EQ(bits, "0110") << "level " << level;
        }
    }
}

// =============================================================================
// Preset composition behaviour (the native-preservation half of the rule)
// =============================================================================

TEST(R1181Lowering, UnconstrainedNoBasisKeepsOpsNative) {
    // With no coupling constraint and no basis, the preset pipelines must NOT
    // lower: the backends run the structured ops natively and decomposition
    // would only pessimize. All three ops must survive as themselves.
    QuantumCircuit qc(4);
    qc.mcx({0, 1, 2}, 3);
    qc.mcp(0.5, {0, 1, 2});
    qc.permute({1, 0, 3, 2}, {0, 1});

    for (int level = 0; level <= 3; ++level) {
        SCOPED_TRACE("level " + std::to_string(level));
        auto out = transpile(qc, CouplingMap(), {}, level);
        int mcx = 0, mcp = 0, permn = 0;
        for (const auto& inst : out.instructions) {
            mcx  += (inst.type == GT::MCX) ? 1 : 0;
            mcp  += (inst.type == GT::MCP) ? 1 : 0;
            permn += (inst.type == GT::PERMUTATION) ? 1 : 0;
        }
        EXPECT_EQ(mcx, 1);
        EXPECT_EQ(mcp, 1);
        EXPECT_EQ(permn, 1);
    }
}

TEST(R1181Lowering, PassIsNoOpWithoutHighLevelOps) {
    // The pass returns the input DAG unchanged when no high-level op is
    // present, pinning the scan-first cheap path. Baseline is the DAG
    // round-trip of the same circuit, NOT the original instruction vector:
    // to_circuit() emits in topological order, which may legally reorder
    // independent gates (e.g. the t(2) against the h/cx chain), and that
    // reordering belongs to the round-trip, not to this pass.
    QuantumCircuit qc(3);
    qc.h(0).cx(0, 1).t(2).ccx(0, 1, 2).swap(0, 2);
    const auto baseline =
        DAGCircuit::from_circuit(qc).to_circuit().instructions;
    auto lowered = lower_via_pass(qc);
    ASSERT_EQ(lowered.size(), baseline.size());
    for (size_t i = 0; i < lowered.size(); ++i) {
        EXPECT_EQ(lowered[i].type, baseline[i].type) << "index " << i;
        EXPECT_EQ(lowered[i].qubits, baseline[i].qubits) << "index " << i;
    }
}

// =============================================================================
// Routed end-to-end + feedforward condition propagation
// =============================================================================

TEST(R1181Lowering, RoutedEndToEndMcxOnLinearMap) {
    // All-ones controls force the flip: |1110> -> mcx -> |1111>. Through
    // routing + optimisation + translation the sampled key must be exactly
    // 1111 at every level (measure remapping compensates the routing
    // permutation).
    QuantumCircuit qc(4, 4);
    qc.x(0).x(1).x(2);
    qc.mcx({0, 1, 2}, 3);
    qc.measure_all();

    for (int level = 0; level <= 3; ++level) {
        SCOPED_TRACE("level " + std::to_string(level));
        auto out = transpile(qc, CouplingMap::linear(4), {"cx", "u3"}, level);
        for (const auto& inst : out.instructions) {
            EXPECT_NE(inst.type, GT::MCX) << "mcx must be lowered";
        }
        StatevectorSimulator sim;
        auto res = sim.run(out, 128, 11);
        ASSERT_FALSE(res.counts.empty());
        for (const auto& [bits, count] : res.counts) {
            (void)count;
            EXPECT_EQ(bits, "1111") << "level " << level;
        }
    }
}

TEST(R1181Lowering, ConditionPropagatesThroughLowering) {
    // Structural half: a conditioned MCX lowers to a block where EVERY gate
    // carries the condition (the pass may not drop or partially apply it).
    QuantumCircuit qc(4, 1);
    qc.h(0);
    qc.measure(0, 0);
    qc.add_if(0, 1, GT::MCX, {1, 2, 3});
    auto lowered = lower_via_pass(qc);

    int conditioned_gates = 0;
    for (const auto& inst : lowered) {
        if (inst.type == GT::MEASURE || inst.type == GT::H) continue;
        EXPECT_EQ(inst.condition_clbit, 0)
            << "lowered gate '" << inst.gate_name() << "' lost the condition";
        EXPECT_EQ(inst.condition_value, 1);
        ++conditioned_gates;
    }
    EXPECT_GE(conditioned_gates, 1);
}

TEST(R1181Lowering, ConditionedMcxBehavesEndToEnd) {
    // Behavioural half: x(0) forces clbit 0 to 1, the conditioned MCX on
    // all-ones controls then fires deterministically: key 1111 plus the
    // trigger bit. Through the full pipeline at every level.
    QuantumCircuit qc(4, 4);
    qc.x(0);
    qc.measure(0, 0);
    qc.x(1).x(2);
    qc.add_if(0, 1, GT::MCX, {0, 1, 2, 3});  // controls 0,1,2 -> target 3
    qc.measure_all();

    for (int level = 0; level <= 3; ++level) {
        SCOPED_TRACE("level " + std::to_string(level));
        auto out = transpile(qc, CouplingMap::linear(4), {"cx", "u3"}, level);
        StatevectorSimulator sim;
        auto res = sim.run(out, 64, 13);
        ASSERT_FALSE(res.counts.empty());
        for (const auto& [bits, count] : res.counts) {
            (void)count;
            EXPECT_EQ(bits, "1111") << "level " << level;
        }
    }
}

// =============================================================================
// Internal seam: relabel detection + validation (../src include precedent:
// the visualiser suites include composite_catalogue.hpp the same way)
// =============================================================================

TEST(R1181Lowering, AsQubitRelabelDetectsRotation) {
    // The 3-wire rotation from PermutationRelabelLowersToSwapsOnly: sigma
    // must be exactly {1, 2, 0} (bit j of the input moves to bit j+1 mod 3).
    const int k = 3;
    std::vector<int> perm(1 << k);
    for (int x = 0; x < (1 << k); ++x) {
        int img = 0;
        for (int j = 0; j < k; ++j)
            if (x & (1 << j)) img |= (1 << ((j + 1) % k));
        perm[x] = img;
    }
    auto sigma = hld::as_qubit_relabel(perm);
    ASSERT_TRUE(sigma.has_value());
    EXPECT_EQ(*sigma, (std::vector<int>{1, 2, 0}));
}

TEST(R1181Lowering, AsQubitRelabelRejectsGeneralMaps) {
    // Increment map: perm[0] = 1 != 0 disqualifies immediately.
    EXPECT_FALSE(hld::as_qubit_relabel({1, 2, 3, 0}).has_value());
    // One-hot images but inconsistent on a composite state:
    // 1->1, 2->2, but 3->0 breaks the linearity a relabel demands.
    EXPECT_FALSE(hld::as_qubit_relabel({0, 1, 2, 0}).has_value());
    // XOR-style map: 3 -> 3 fine, but image of 1 is 3 (not one-hot).
    EXPECT_FALSE(hld::as_qubit_relabel({0, 3, 2, 1}).has_value());
}

TEST(R1181Lowering, LowerPermutationValidatesBijection) {
    // from_json can construct unvalidated instructions, so the lowering
    // itself must fail loud on garbage maps.
    EXPECT_THROW(hld::lower_permutation({0, 0, 1, 2}, {0, 1}),
                 std::invalid_argument);
    EXPECT_THROW(hld::lower_permutation({0, 1, 2, 4}, {0, 1}),
                 std::invalid_argument);
    EXPECT_THROW(hld::lower_permutation({0, 1, 2}, {0, 1}),
                 std::invalid_argument);
}
