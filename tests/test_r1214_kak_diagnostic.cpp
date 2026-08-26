// R.1.22.0 diagnostic: does ConsolidateBlocks actually consolidate, and if not,
// at which stage does it stop?
//
// Issue #78 reports that the KAK decomposition inside ConsolidateBlocks fails
// on degenerate Weyl coordinates, which every common two-qubit gate has. The
// pass carries a verification net, so a failed decomposition is discarded and
// the ORIGINAL instructions are kept: the output stays correct and the
// optimisation silently does not happen.
//
// That is why the existing coverage cannot see it.
// R1121Passes.ConsolidateBlocksFullTwoQubitSetPreservesSemantics asserts
// semantic equivalence, and the fallback IS equivalent, so it passes either
// way. An equivalence assertion can never detect a fallback.
//
// A block can fail to be consolidated at FOUR distinct stages, and the emitted
// circuit looks identical in every case: unchanged input. Each section below
// isolates one stage, so a failure names the stage rather than the symptom.
//
//   1. BLOCK FORMATION. A block is a run of ADJACENT two-qubit gates on one
//      pair; a single-qubit gate on either wire ends the run. An entangler
//      sandwiched in local gates forms a block of ONE and is returned by the
//      block_count == 1 early-out without the decomposition being called.
//   2. THE COUNT GUARD. A block is kept only when it lowers the two-qubit
//      count. The decomposition emits up to three interaction rotations, so
//      a block must hold at least FOUR two-qubit gates before consolidation
//      is unambiguously worth keeping.
//   3. THE DECOMPOSITION. The verification net rebuilds the decomposition and
//      compares it against the block, discarding it on mismatch.
//   4. CANONICALITY. A decomposition can be correct without being minimal: a
//      Weyl point outside the canonical chamber costs an extra rotation.
//
// The blocks are bare runs of entanglers with no surrounding local gates, so
// the whole output must be KAK alphabet and no case analysis is needed about
// what else was in the circuit.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/dag.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/transpiler.hpp"

#include <string>
#include <vector>

using namespace lindblad;

namespace {

QuantumCircuit run_consolidate(const QuantumCircuit& qc) {
    auto dag = DAGCircuit::from_circuit(qc);
    TranspilationContext ctx{};
    return ConsolidateBlocks().run(dag, ctx).to_circuit();
}

const char* type_name(Instruction::GateType t) {
    switch (t) {
        case Instruction::GateType::U:       return "U";
        case Instruction::GateType::RXX:     return "RXX";
        case Instruction::GateType::RYY:     return "RYY";
        case Instruction::GateType::RZZ:     return "RZZ";
        case Instruction::GateType::CX:      return "CX";
        case Instruction::GateType::CY:      return "CY";
        case Instruction::GateType::CZ:      return "CZ";
        case Instruction::GateType::CH:      return "CH";
        case Instruction::GateType::SWAP:    return "SWAP";
        case Instruction::GateType::ISWAP:   return "ISWAP";
        case Instruction::GateType::ECR:     return "ECR";
        case Instruction::GateType::CP:      return "CP";
        case Instruction::GateType::CRX:     return "CRX";
        case Instruction::GateType::CRY:     return "CRY";
        case Instruction::GateType::CRZ:     return "CRZ";
        case Instruction::GateType::CU:      return "CU";
        case Instruction::GateType::RZX:     return "RZX";
        case Instruction::GateType::UNITARY: return "UNITARY";
        case Instruction::GateType::H:       return "H";
        case Instruction::GateType::RX:      return "RX";
        case Instruction::GateType::RY:      return "RY";
        case Instruction::GateType::RZ:      return "RZ";
        default:                             return "other";
    }
}

std::string render(const QuantumCircuit& qc) {
    std::string s;
    for (const auto& inst : qc.instructions) {
        if (!s.empty()) s += " ";
        s += type_name(inst.type);
        s += "(";
        for (size_t i = 0; i < inst.qubits.size(); ++i) {
            if (i) s += ",";
            s += std::to_string(inst.qubits[i]);
        }
        s += ")";
    }
    return s.empty() ? "<empty>" : s;
}

// The KAK output alphabet: local corrections as U, interaction as RXX/RYY/RZZ.
bool in_kak_alphabet(const QuantumCircuit& qc) {
    for (const auto& inst : qc.instructions) {
        switch (inst.type) {
            case Instruction::GateType::U:
            case Instruction::GateType::RXX:
            case Instruction::GateType::RYY:
            case Instruction::GateType::RZZ:
                break;
            default:
                return false;
        }
    }
    return true;
}

int count_2q(const QuantumCircuit& qc) {
    int n = 0;
    for (const auto& inst : qc.instructions)
        if (inst.qubits.size() == 2) ++n;
    return n;
}

struct Case {
    const char* name;
    void (*apply)(QuantumCircuit&);
};

const Case kEntanglers[] = {
    {"cx",    [](QuantumCircuit& c) { c.cx(0, 1); }},
    {"cy",    [](QuantumCircuit& c) { c.cy(0, 1); }},
    {"cz",    [](QuantumCircuit& c) { c.cz(0, 1); }},
    {"ch",    [](QuantumCircuit& c) { c.ch(0, 1); }},
    {"swap",  [](QuantumCircuit& c) { c.swap(0, 1); }},
    {"iswap", [](QuantumCircuit& c) { c.iswap(0, 1); }},
    {"ecr",   [](QuantumCircuit& c) { c.ecr(0, 1); }},
    {"cp",    [](QuantumCircuit& c) { c.cp(0.6, 0, 1); }},
    {"crx",   [](QuantumCircuit& c) { c.crx(0.7, 0, 1); }},
    {"cry",   [](QuantumCircuit& c) { c.cry(-0.4, 0, 1); }},
    {"crz",   [](QuantumCircuit& c) { c.crz(1.1, 0, 1); }},
    {"rxx",   [](QuantumCircuit& c) { c.rxx(0.5, 0, 1); }},
    {"ryy",   [](QuantumCircuit& c) { c.ryy(-0.3, 0, 1); }},
    {"rzz",   [](QuantumCircuit& c) { c.rzz(0.9, 0, 1); }},
    {"rzx",   [](QuantumCircuit& c) { c.rzx(0.8, 0, 1); }},
    {"cu",    [](QuantumCircuit& c) { c.cu(0.4, 0.5, -0.6, 0.2, 0, 1); }},
};

QuantumCircuit n_adjacent(void (*apply)(QuantumCircuit&), int n) {
    QuantumCircuit qc(2);
    for (int i = 0; i < n; ++i) apply(qc);
    return qc;
}

// A block of four is the shortest that the count guard cannot reject on
// grounds of size, since the decomposition emits at most three rotations.
constexpr int kBlockLen = 4;

}  // namespace

// =============================================================================
// Stage 1 - block formation, which decides whether the decomposition runs
// =============================================================================

// A local gate on either wire terminates the run, so this holds two blocks of
// one rather than one block of two, and the block_count == 1 early-out returns
// before the decomposition is called. Asserted so that a change to block
// collection cannot quietly invalidate every case below it.
TEST(R1214KakDiagnostic, AnInterleavedLocalGateEndsTheBlock) {
    QuantumCircuit in(2);
    in.cz(0, 1);
    in.rx(0.4, 0);
    in.cz(0, 1);

    const QuantumCircuit out = run_consolidate(in);
    EXPECT_EQ(out.instructions.size(), in.instructions.size())
        << "expected the circuit to pass through untouched.\n"
        << "  out: " << render(out);
    EXPECT_EQ(count_2q(out), 2)
        << "both CZ gates should survive as separate one-gate blocks.\n"
        << "  out: " << render(out);
}

// =============================================================================
// Stage 2 - the count guard, and where its threshold actually falls
// =============================================================================

// Scans block length so a failure says WHICH lengths consolidate rather than
// only that some do not. A decomposition emitting at most three rotations must
// be kept from length four upward; below that, declining is correct.
TEST(R1214KakDiagnostic, ConsolidationBeginsAtBlockLengthFour) {
    for (const Case& cs : kEntanglers) {
        SCOPED_TRACE(cs.name);
        std::string table;
        for (int n = 2; n <= 6; ++n) {
            const QuantumCircuit out = run_consolidate(n_adjacent(cs.apply, n));
            table += "  n=" + std::to_string(n) + " -> " + render(out) + "\n";
        }
        for (int n = kBlockLen; n <= 6; ++n) {
            const QuantumCircuit out = run_consolidate(n_adjacent(cs.apply, n));
            EXPECT_TRUE(in_kak_alphabet(out))
                << cs.name << ": length " << n << " was handed back "
                << "unconsolidated.\n" << table;
        }
    }
}

// =============================================================================
// Stage 3 - the decomposition itself
// =============================================================================

TEST(R1214KakDiagnostic, EveryTwoQubitEntanglerBlockIsConsolidated) {
    for (const Case& cs : kEntanglers) {
        SCOPED_TRACE(cs.name);
        const QuantumCircuit in = n_adjacent(cs.apply, kBlockLen);
        ASSERT_EQ(count_2q(in), kBlockLen);

        const QuantumCircuit out = run_consolidate(in);
        EXPECT_TRUE(in_kak_alphabet(out))
            << cs.name << ": block was handed back unconsolidated.\n"
            << "  in : " << render(in) << "\n"
            << "  out: " << render(out);
        EXPECT_LE(count_2q(out), 3)
            << cs.name << ": KAK emits at most three interaction rotations.\n"
            << "  out: " << render(out);
    }
}

// A raw 2-qubit UNITARY carries no gate structure for the pass to lean on, so
// it isolates the decomposition from the gate-to-matrix conversion.
TEST(R1214KakDiagnostic, AGenericUnitaryBlockIsConsolidated) {
    QuantumCircuit src(2);
    src.h(0).cx(0, 1).rz(0.7, 1).t(0);
    const auto u4 = Operator::from_circuit(src).data;

    QuantumCircuit in(2);
    for (int i = 0; i < kBlockLen; ++i) in.unitary(u4, {0, 1});

    const QuantumCircuit out = run_consolidate(in);
    EXPECT_TRUE(in_kak_alphabet(out))
        << "generic UNITARY block was handed back unconsolidated.\n"
        << "  out: " << render(out);
}

// =============================================================================
// Stage 4 - is the correct decomposition also the MINIMAL one?
// =============================================================================

// The rotation count is the number of nonzero Weyl coordinates, a property of
// the operand rather than of any implementation. CX, CZ and SWAP are each their
// own inverse, so a four-fold block is the identity: a local operator, needing
// no interaction term at all. Anything emitted here is pure waste.
TEST(R1214KakDiagnostic, SelfInverseBlocksEmitNoInteraction) {
    const Case kSelfInverse[] = {
        {"cx",    [](QuantumCircuit& c) { c.cx(0, 1); }},
        {"cz",    [](QuantumCircuit& c) { c.cz(0, 1); }},
        {"swap",  [](QuantumCircuit& c) { c.swap(0, 1); }},
    };
    for (const Case& cs : kSelfInverse) {
        SCOPED_TRACE(cs.name);
        const QuantumCircuit out = run_consolidate(n_adjacent(cs.apply, 4));
        EXPECT_EQ(count_2q(out), 0)
            << cs.name << "^4 is the identity, so no interaction term is "
            << "needed.\n  out: " << render(out);
    }
}

// Five copies of a self-inverse gate compose back to one copy of it. CX and CZ
// sit at the Weyl point (pi/4, 0, 0): one nonzero coordinate, so exactly one
// interaction rotation is both necessary and sufficient.
TEST(R1214KakDiagnostic, SingleCoordinateOperandsEmitExactlyOneRotation) {
    const Case kOneRotation[] = {
        {"cx", [](QuantumCircuit& c) { c.cx(0, 1); }},
        {"cz", [](QuantumCircuit& c) { c.cz(0, 1); }},
    };
    for (const Case& cs : kOneRotation) {
        SCOPED_TRACE(cs.name);
        const QuantumCircuit out = run_consolidate(n_adjacent(cs.apply, 5));
        EXPECT_EQ(count_2q(out), 1)
            << cs.name << "^5 is " << cs.name << ", at Weyl point "
            << "(pi/4, 0, 0).\n  out: " << render(out);
    }
}

// iSWAP sits at (pi/4, pi/4, 0): two nonzero coordinates, so two rotations
// suffice. A decomposition can be correct and still land on an equivalent Weyl
// point outside the canonical chamber, which costs a third rotation. This test
// separates "correct" from "minimal", and it is the only one here that a
// correct-but-uncanonical implementation fails.
TEST(R1214KakDiagnostic, ISwapEmitsTwoRotationsNotThree) {
    QuantumCircuit in(2);
    for (int i = 0; i < 5; ++i) in.iswap(0, 1);

    const QuantumCircuit out = run_consolidate(in);
    EXPECT_EQ(count_2q(out), 2)
        << "iSWAP^5 is iSWAP, at Weyl point (pi/4, pi/4, 0). Three rotations "
        << "here means the chosen Weyl point is equivalent but not canonical.\n"
        << "  out: " << render(out);
}

// =============================================================================
// #79 - to_qasm3 and a multi-qubit UNITARY
// =============================================================================
//
// In the same release because the two are linked: an exact QASM export of a
// 2-qubit UNITARY needs a working two-qubit decomposition, which is what #78
// supplies. to_qasm3 currently special-cases UNITARY only at one qubit and
// lets anything wider fall through to the generic tail, which writes a call to
// a gate the file never defines and drops the matrix entirely.
//
// One thing this settles that was never checked: what from_qasm3 does with
// that text. The failure message reports both the emitted QASM and the
// parser's behaviour, so the answer arrives whichever way it goes.

TEST(R1214Qasm3Unitary, TwoQubitUnitaryRoundTripsThroughQasm3) {
    QuantumCircuit src(2);
    src.h(0).cx(0, 1).rz(0.7, 1).t(0);
    const auto u4 = Operator::from_circuit(src).data;

    QuantumCircuit in(2);
    in.unitary(u4, {0, 1});

    std::string text;
    ASSERT_NO_THROW(text = in.to_qasm3())
        << "to_qasm3 is expected to either lower the operand or refuse it "
        << "loudly, not to fail unexpectedly";

    bool threw = false;
    std::string what;
    QuantumCircuit back(2);
    try {
        back = QuantumCircuit::from_qasm3(text);
    } catch (const std::exception& e) {
        threw = true;
        what = e.what();
    }

    ASSERT_FALSE(threw)
        << "from_qasm3 rejected what to_qasm3 produced.\n"
        << "  parser said: " << what << "\n"
        << "  emitted QASM 3:\n" << text;

    const auto ma = Operator::from_circuit(in).data;
    const auto mb = Operator::from_circuit(back).data;
    ASSERT_EQ(ma.size(), mb.size()) << "emitted QASM 3:\n" << text;

    Complex128 phase(1, 0);
    for (size_t i = 0; i < ma.size(); ++i) {
        if (mb[i].norm_sq() > 1e-12 && ma[i].norm_sq() > 1e-12) {
            phase = ma[i] * Complex128(mb[i].real, -mb[i].imag) *
                    (1.0 / mb[i].norm_sq());
            break;
        }
    }
    for (size_t i = 0; i < ma.size(); ++i) {
        const Complex128 bp = mb[i] * phase;
        EXPECT_NEAR(ma[i].real, bp.real, 1e-7)
            << "re @ " << i << ", emitted QASM 3:\n" << text;
        EXPECT_NEAR(ma[i].imag, bp.imag, 1e-7)
            << "im @ " << i << ", emitted QASM 3:\n" << text;
    }
}

// =============================================================================
// The property the existing coverage DOES assert, kept as the safety statement
// =============================================================================

// Whatever the decomposition does, the verification net keeps this true. It is
// here so that a fix to the structural tests above cannot trade correctness for
// consolidation without the trade showing up.
TEST(R1214KakDiagnostic, ConsolidationNeverChangesTheOperator) {
    for (const Case& cs : kEntanglers) {
        SCOPED_TRACE(cs.name);
        const QuantumCircuit in = n_adjacent(cs.apply, kBlockLen);
        const QuantumCircuit out = run_consolidate(in);

        const auto ma = Operator::from_circuit(in).data;
        const auto mb = Operator::from_circuit(out).data;
        ASSERT_EQ(ma.size(), mb.size());

        Complex128 phase(1, 0);
        for (size_t i = 0; i < ma.size(); ++i) {
            if (mb[i].norm_sq() > 1e-12 && ma[i].norm_sq() > 1e-12) {
                phase = ma[i] * Complex128(mb[i].real, -mb[i].imag) *
                        (1.0 / mb[i].norm_sq());
                break;
            }
        }
        for (size_t i = 0; i < ma.size(); ++i) {
            const Complex128 bp = mb[i] * phase;
            EXPECT_NEAR(ma[i].real, bp.real, 1e-7) << "re @ " << i;
            EXPECT_NEAR(ma[i].imag, bp.imag, 1e-7) << "im @ " << i;
        }
    }
}
