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

#include <Eigen/Dense>

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/dag.hpp"
#include "lindblad/gates.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/transpiler.hpp"
#include "lindblad/validation.hpp"

#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <utility>
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

// Two circuits are the same operator up to a global phase. The phase is read
// off the first entry where both operators are far enough from zero to divide
// by, then applied to every entry.
//
// Returning an AssertionResult rather than running EXPECTs inside a helper is
// what keeps a sweep readable: a failing operand reports once, naming the entry
// that disagreed, instead of sixteen times.
::testing::AssertionResult same_operator_up_to_phase(const QuantumCircuit& a,
                                                     const QuantumCircuit& b,
                                                     double tol = 1e-7) {
    const auto ma = Operator::from_circuit(a).data;
    const auto mb = Operator::from_circuit(b).data;
    if (ma.size() != mb.size())
        return ::testing::AssertionFailure()
               << "operator size " << ma.size() << " vs " << mb.size();

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
        if (std::abs(ma[i].real - bp.real) > tol ||
            std::abs(ma[i].imag - bp.imag) > tol)
            return ::testing::AssertionFailure()
                   << "entry " << i << ": expected (" << ma[i].real << ", "
                   << ma[i].imag << "), got (" << bp.real << ", " << bp.imag
                   << ") after dividing out the global phase";
    }
    return ::testing::AssertionSuccess();
}

// Gate types, operands AND parameters, at full double precision. render() drops
// the parameters, which is what a determinism check most needs to see: a tie in
// the candidate ranking broken by iteration order moves the angles while
// leaving the gate sequence identical.
std::string fingerprint(const QuantumCircuit& qc) {
    std::ostringstream os;
    os << std::setprecision(17);
    for (const auto& inst : qc.instructions) {
        os << type_name(inst.type) << "(";
        for (size_t i = 0; i < inst.qubits.size(); ++i) {
            if (i) os << ",";
            os << inst.qubits[i];
        }
        os << ";";
        for (size_t i = 0; i < inst.params.size(); ++i) {
            if (i) os << ",";
            os << inst.params[i];
        }
        os << ") ";
    }
    return os.str();
}

// Instruction::matrix is row-major: element (r, c) sits at r * rows + c. Read
// off unitarity_deviation, which forms (U†U)_ij as Σ_m conj(U[m*rows+i]) *
// U[m*rows+j].
std::vector<Complex128> from_eigen4(const Eigen::Matrix4cd& m) {
    std::vector<Complex128> v(16);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            v[r * 4 + c] = Complex128(m(r, c).real(), m(r, c).imag());
    return v;
}

// std::uniform_real_distribution and std::normal_distribution are
// implementation-defined: the same engine and the same seed give different
// draws on libstdc++ and libc++. These sweeps run on four legs and exist partly
// to catch differences BETWEEN legs, so the draws are generated here instead,
// leaving std::mt19937_64 (whose output sequence IS exactly specified) as the
// only source of variation. The operands are then bit-identical everywhere, and
// a leg that disagrees disagrees about the decomposition rather than the input.

// 2^-53, the spacing that turns the top 53 bits of a 64-bit draw into [0, 1).
constexpr double kUnitScale = std::numeric_limits<double>::epsilon() / 2.0;

double next_unit(std::mt19937_64& rng) {
    return static_cast<double>(rng() >> 11) * kUnitScale;
}

double next_in(std::mt19937_64& rng, double lo, double hi) {
    return lo + (hi - lo) * next_unit(rng);
}

// Box-Muller. u1 is taken from (0, 1] so the logarithm stays finite; the sine
// half of the pair is discarded, which costs draws and buys simplicity.
double next_gauss(std::mt19937_64& rng) {
    const double u1 = 1.0 - next_unit(rng);
    const double u2 = next_unit(rng);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(TWO_PI * u2);
}

// Haar on U(4), by QR of a complex Ginibre matrix.
//
// The phase fix on R's diagonal is load-bearing, not tidiness: QR is unique
// only up to a diagonal unitary, and implementations pick a convention that
// leaves Q biased. Dividing out arg(r_ii) restores Haar.
Eigen::Matrix4cd haar_u4(std::mt19937_64& rng) {
    Eigen::Matrix4cd z;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            // Named locals, never two draws as arguments to one call:
            // argument evaluation order is unspecified, so an inline pair would
            // consume the stream in an order the compiler chooses.
            const double re = next_gauss(rng);
            const double im = next_gauss(rng);
            z(r, c) = std::complex<double>(re, im);
        }
    }

    Eigen::HouseholderQR<Eigen::Matrix4cd> qr(z);
    Eigen::Matrix4cd q = qr.householderQ();
    const Eigen::Matrix4cd r = qr.matrixQR();
    for (int i = 0; i < 4; ++i) {
        const std::complex<double> d = r(i, i);
        const double mag = std::abs(d);
        if (mag > 0.0) q.col(i) *= d / mag;
    }
    return q;
}

// A local factor drawn from SU(2). theta comes through acos rather than
// uniformly, because a uniform Euler angle piles the axis up at the poles and
// would leave the local corrections barely varying across a sweep.
void append_random_su2(QuantumCircuit& qc, int qubit, std::mt19937_64& rng) {
    // Named locals for the same reason as in haar_u4: unspecified argument
    // evaluation order would otherwise make the draw order compiler-dependent.
    const double theta = 2.0 * std::acos(std::sqrt(next_unit(rng)));
    const double phi = next_in(rng, -PI, PI);
    const double lambda = next_in(rng, -PI, PI);
    qc.u(theta, phi, lambda, qubit);
}

// Uniform in the Weyl coordinates, dressed either side with SU(2) locals. This
// is deliberately NOT Haar: Haar measure concentrates away from the chamber
// boundary, which is exactly where the branch selection is most delicate, so a
// Haar sweep under-samples the region the decomposition finds hardest.
//
// rxx(t) carries the coordinate kx = t/2 (the emitted angle is 2*kx), so
// drawing t from [-PI, PI] sweeps kx across the full [-PI_2, PI_2].
QuantumCircuit weyl_uniform_circuit(std::mt19937_64& rng) {
    QuantumCircuit qc(2);
    append_random_su2(qc, 0, rng);
    append_random_su2(qc, 1, rng);
    const double tx = next_in(rng, -PI, PI);
    const double ty = next_in(rng, -PI, PI);
    const double tz = next_in(rng, -PI, PI);
    qc.rxx(tx, 0, 1).ryy(ty, 0, 1).rzz(tz, 0, 1);
    append_random_su2(qc, 0, rng);
    append_random_su2(qc, 1, rng);
    return qc;
}

// A block of n DIFFERENT operands. n_adjacent repeats one gate, so its block
// composes to operand^n; a sweep built that way would test the fourth power of
// its sample rather than the sample.
QuantumCircuit block_of(const std::vector<std::vector<Complex128>>& mats) {
    QuantumCircuit qc(2);
    for (const auto& m : mats) qc.unitary(m, {0, 1});
    return qc;
}

// A block of n copies of one operand, where the operand is given as a circuit.
// Used to place a block's Weyl point exactly: repeating the target itself would
// compose to target^n and land somewhere else entirely, so the caller builds
// the operand from angles divided by n.
QuantumCircuit repeated_unitary(const QuantumCircuit& operand, int n) {
    const auto m = Operator::from_circuit(operand).data;
    QuantumCircuit qc(2);
    for (int i = 0; i < n; ++i) qc.unitary(m, {0, 1});
    return qc;
}

// The parameter of every interaction rotation, in emitted order.
std::vector<double> interaction_angles(const QuantumCircuit& qc) {
    std::vector<double> out;
    for (const auto& inst : qc.instructions) {
        switch (inst.type) {
            case Instruction::GateType::RXX:
            case Instruction::GateType::RYY:
            case Instruction::GateType::RZZ:
                out.push_back(inst.params.empty() ? 0.0 : inst.params[0]);
                break;
            default:
                break;
        }
    }
    return out;
}

// Fixed so a failure is reproducible from the seed alone.
constexpr std::uint64_t kSeed = 0x5EED1214ULL;
constexpr int kSweep = 200;

using BlockMaker = std::function<QuantumCircuit(std::mt19937_64&)>;

// Runs a sweep and returns "" on success, otherwise the failure count and the
// FIRST failing trial. One message per trial would let a systemic break bury
// its own first example under hundreds of lines, and the count is what
// separates a systemic break from an edge case.
std::string sweep_report(std::uint64_t seed, int trials, const BlockMaker& make) {
    std::mt19937_64 rng(seed);
    int failures = 0;
    std::string first;
    for (int t = 0; t < trials; ++t) {
        const QuantumCircuit in = make(rng);
        const QuantumCircuit out = run_consolidate(in);

        std::string why;
        if (!in_kak_alphabet(out)) {
            why = "handed back unconsolidated: " + render(out);
        } else if (count_2q(out) > 3) {
            why = "emitted more than three interaction rotations: " + render(out);
        } else {
            const auto eq = same_operator_up_to_phase(in, out);
            if (!eq) why = std::string("operator changed: ") + eq.message();
        }

        if (!why.empty()) {
            ++failures;
            if (first.empty()) first = "trial " + std::to_string(t) + ", " + why;
        }
    }
    if (failures == 0) return "";
    return std::to_string(failures) + " of " + std::to_string(trials) +
           " operands failed; first was " + first;
}

// A block of kBlockLen independent Haar operands.
QuantumCircuit haar_block(std::mt19937_64& rng) {
    std::vector<std::vector<Complex128>> mats;
    for (int i = 0; i < kBlockLen; ++i) mats.push_back(from_eigen4(haar_u4(rng)));
    return block_of(mats);
}

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
// An exact QASM export of a 2-qubit UNITARY needs a working two-qubit
// decomposition, which is why this sits beside the tests above. to_qasm3
// lowers the operand through KAK into u / rxx / ryy / rzz and carries any
// residual global phase on a gphase, so the emitted text is exact and
// from_qasm3 can read it back.
//
// The failure message reports both the emitted QASM and the parser's response,
// so a break names which half of the round trip gave way.

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

    EXPECT_TRUE(same_operator_up_to_phase(in, back))
        << "emitted QASM 3:\n" << text;
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

        EXPECT_TRUE(same_operator_up_to_phase(in, out))
            << "  in : " << render(in) << "\n"
            << "  out: " << render(out);
    }
}

// =============================================================================
// The count guard, which decides whether a valid decomposition is WORTH keeping
// =============================================================================

// docs/api/transpiler.md states that this pass reduces two-qubit gate count.
// The guard is what makes that true rather than aspirational, so the property
// is asserted directly, over every entangler at every block length the pass can
// see. A decomposition that is correct but longer must be declined.
TEST(R1214KakDiagnostic, TheCountGuardNeverIncreasesTwoQubitCount) {
    for (const Case& cs : kEntanglers) {
        SCOPED_TRACE(cs.name);
        for (int n = 1; n <= 8; ++n) {
            const QuantumCircuit in = n_adjacent(cs.apply, n);
            const QuantumCircuit out = run_consolidate(in);
            EXPECT_LE(count_2q(out), count_2q(in))
                << cs.name << " at block length " << n
                << ": the pass emitted MORE two-qubit gates than it consumed.\n"
                << "  in : " << render(in) << "\n"
                << "  out: " << render(out);
        }
    }
}

// A block of one skips the decomposition entirely via the block_count == 1
// early-out. Nothing is lost by that, since a replacement for a single
// two-qubit gate can never hold fewer, but the early-out is load-bearing for
// every length scan above it and is pinned separately.
TEST(R1214KakDiagnostic, AOneGateBlockIsReturnedUntouched) {
    for (const Case& cs : kEntanglers) {
        SCOPED_TRACE(cs.name);
        const QuantumCircuit in = n_adjacent(cs.apply, 1);
        const QuantumCircuit out = run_consolidate(in);
        EXPECT_EQ(out.instructions.size(), in.instructions.size())
            << "  out: " << render(out);
        EXPECT_EQ(count_2q(out), 1) << "  out: " << render(out);
    }
}

// The guard breaks a tie on two-qubit count with total instruction count. A
// generic operand decomposes to three interaction rotations, so a block of
// THREE generic operands ties at three, and the decomposition then loses on
// total length: three rotations plus up to four local corrections against three
// instructions. Declining is the correct verdict and is asserted here because a
// guard that only compared two-qubit counts would accept it.
TEST(R1214KakDiagnostic, ATieOnTwoQubitCountIsDeclined) {
    QuantumCircuit src(2);
    src.h(0).cx(0, 1).rz(0.7, 1).t(0);
    const auto u4 = Operator::from_circuit(src).data;

    QuantumCircuit in(2);
    for (int i = 0; i < 3; ++i) in.unitary(u4, {0, 1});
    ASSERT_EQ(count_2q(in), 3);

    const QuantumCircuit out = run_consolidate(in);
    EXPECT_FALSE(in_kak_alphabet(out))
        << "a block of three generic operands ties at three interaction "
           "rotations and is longer overall, so it must be declined.\n"
        << "  out: " << render(out);
    EXPECT_EQ(count_2q(out), 3) << "  out: " << render(out);
}

// =============================================================================
// End to end, through the public entry point
// =============================================================================

// Everything above drives ConsolidateBlocks directly. This drives transpile()
// at the level where the pass actually runs, over a battery of shapes, and
// asserts the two things a caller is entitled to: the circuit still means the
// same thing, and the two-qubit count did not go up. The second half is the
// contract docs/api/transpiler.md states and nothing previously checked.
TEST(R1214KakDiagnostic, TranspileLevelThreePreservesSemanticsAndNeverAddsTwoQubitGates) {
    std::vector<QuantumCircuit> battery;

    // Bare entangler runs, one per gate, at a length the pass will act on.
    for (const Case& cs : kEntanglers) battery.push_back(n_adjacent(cs.apply, kBlockLen));

    // Interleaved locals, so blocks end and restart.
    {
        QuantumCircuit qc(2);
        for (int i = 0; i < 3; ++i) { qc.cx(0, 1); qc.rx(0.3 * (i + 1), 0); qc.cz(0, 1); }
        battery.push_back(qc);
    }
    // Three wires, so the pass has to pick pairs.
    {
        QuantumCircuit qc(3);
        qc.h(0).cx(0, 1).cx(1, 2).cx(0, 1).cx(1, 2).rz(0.4, 2).cx(0, 1);
        battery.push_back(qc);
    }
    // Raw operands with no gate structure to lean on.
    {
        QuantumCircuit src(2);
        src.h(0).cx(0, 1).rz(0.7, 1).t(0);
        const auto u4 = Operator::from_circuit(src).data;
        QuantumCircuit qc(2);
        for (int i = 0; i < kBlockLen; ++i) qc.unitary(u4, {0, 1});
        battery.push_back(qc);
    }
    // Nothing for the pass to do at all.
    {
        QuantumCircuit qc(2);
        qc.h(0).rz(0.2, 1).sx(0);
        battery.push_back(qc);
    }

    for (size_t i = 0; i < battery.size(); ++i) {
        SCOPED_TRACE("circuit " + std::to_string(i));
        const QuantumCircuit& in = battery[i];
        const QuantumCircuit out = transpile(in, CouplingMap(), {}, 3);

        EXPECT_TRUE(same_operator_up_to_phase(in, out))
            << "  in : " << render(in) << "\n  out: " << render(out);
        EXPECT_LE(count_2q(out), count_2q(in))
            << "level 3 increased the two-qubit count.\n"
            << "  in : " << render(in) << "\n  out: " << render(out);
    }
}

// =============================================================================
// Exact emitted parameters, not merely how many
// =============================================================================

// Counting rotations says the decomposition found the right SHAPE. It says
// nothing about the numbers, and the R.1.21.3 defect was a number that moved
// with -march. This release put a great deal of new floating-point work in a
// hot path, so the angles are pinned too.
//
// The emitted parameter for a Weyl coordinate k is 2k, so every operand below,
// sitting at a coordinate of PI_4 on each of its non-zero axes, must emit
// exactly 2 * PI_4 on each. That value is DERIVED from constants.hpp. An angle
// read off a run and pasted back in would pin whatever this build happened to
// produce, which is the opposite of a test.
//
// Each gate here is its own inverse, so an ODD power composes back to the gate
// itself and the block's Weyl point is the gate's own. Five rather than three,
// because the count guard declines a tie: at three the decomposition emits
// three rotations against a block of three and loses on total instruction
// count.
//
// The SIGN is deliberately not pinned. A coordinate of -PI_4 describes the same
// operator with the difference absorbed into the local factors, so requiring a
// particular sign would constrain a free choice rather than a contract.
TEST(R1214KakDiagnostic, EmittedInteractionAnglesAreExact) {
    // Loose enough for a numerical decomposition, tight enough that a value
    // which is not 2 * PI_4 to twelve places fails.
    constexpr double kAngleTol = 1e-12;
    const double expected = 2.0 * PI_4;

    struct Expect {
        const char* name;
        void (*apply)(QuantumCircuit&);
        size_t rotations;
        const char* point;
    };
    const Expect cases[] = {
        {"cx",    [](QuantumCircuit& c) { c.cx(0, 1); },    1, "(pi/4, 0, 0)"},
        {"cz",    [](QuantumCircuit& c) { c.cz(0, 1); },    1, "(pi/4, 0, 0)"},
        {"iswap", [](QuantumCircuit& c) { c.iswap(0, 1); }, 2, "(pi/4, pi/4, 0)"},
        {"swap",  [](QuantumCircuit& c) { c.swap(0, 1); },  3, "(pi/4, pi/4, pi/4)"},
    };

    for (const Expect& cs : cases) {
        SCOPED_TRACE(cs.name);
        const QuantumCircuit in = n_adjacent(cs.apply, 5);
        const QuantumCircuit out = run_consolidate(in);

        ASSERT_TRUE(in_kak_alphabet(out))
            << cs.name << "^5 was handed back unconsolidated.\n"
            << "  out: " << render(out);

        const std::vector<double> angles = interaction_angles(out);
        ASSERT_EQ(angles.size(), cs.rotations)
            << cs.name << "^5 is " << cs.name << ", at Weyl point " << cs.point
            << ".\n  out: " << render(out);

        for (size_t i = 0; i < angles.size(); ++i) {
            EXPECT_NEAR(std::abs(angles[i]), expected, kAngleTol)
                << cs.name << ": rotation " << i << " carries " << angles[i]
                << ", expected magnitude 2 * PI_4 for a coordinate of pi/4.\n"
                << "  out: " << render(out);
        }
    }
}

// =============================================================================
// Randomised sweeps - two distributions, because they cover different ground
// =============================================================================

// Haar is the distributional statement: a generic operand decomposes. Every
// block is kBlockLen INDEPENDENT operands rather than one repeated, so the
// thing decomposed is the sample rather than its fourth power.
TEST(R1214KakDiagnostic, HaarOperandsAlwaysConsolidate) {
    EXPECT_EQ(sweep_report(kSeed, kSweep, haar_block), "")
        << "seed 0x5EED1214, block length " << kBlockLen;
}

// Uniform in the Weyl coordinates, which Haar is not. Haar measure concentrates
// away from the chamber boundary, so it under-samples exactly the region where
// the square-root branch selection and the chamber folding are most delicate.
TEST(R1214KakDiagnostic, WeylUniformOperandsAlwaysConsolidate) {
    EXPECT_EQ(sweep_report(kSeed, kSweep, [](std::mt19937_64& rng) {
        std::vector<std::vector<Complex128>> mats;
        for (int i = 0; i < kBlockLen; ++i)
            mats.push_back(Operator::from_circuit(weyl_uniform_circuit(rng)).data);
        return block_of(mats);
    }), "") << "seed 0x5EED1214, block length " << kBlockLen;
}

// =============================================================================
// The chamber boundary, where the branch selection is most likely to flip
// =============================================================================

// XX, YY and ZZ pairwise commute, so a block of n copies of
// rxx(tx/n).ryy(ty/n).rzz(tz/n) composes exactly to rxx(tx).ryy(ty).rzz(tz).
// That is what places the BLOCK on a chosen Weyl point: repeating the target
// itself would compose to target^n and land somewhere else.
TEST(R1214KakDiagnostic, ChamberBoundaryOperandsConsolidate) {
    // Comfortably above the decomposition's clustering tolerance, so an operand
    // meant to sit just off the boundary is genuinely off it rather than
    // numerically indistinguishable from it.
    constexpr double kNudge = 1e-6;

    struct Point {
        const char* name;
        double tx, ty, tz;  // block-level angles; the coordinate is t/2
    };
    const Point points[] = {
        {"kx exactly on the boundary",  PI_2,          0.0,           0.0},
        {"kx just inside",              PI_2 - kNudge, 0.0,           0.0},
        {"kx just outside, folds back", PI_2 + kNudge, 0.0,           0.0},
        {"the iSWAP point",             PI_2,          PI_2,          0.0},
        {"the SWAP point",              PI_2,          PI_2,          PI_2},
        {"kx on it, ky just inside",    PI_2,          PI_2 - kNudge, 0.0},
        {"all three just outside",      PI_2 + kNudge, PI_2 + kNudge, PI_2 + kNudge},
    };

    for (const Point& p : points) {
        SCOPED_TRACE(p.name);
        const double n = static_cast<double>(kBlockLen);

        QuantumCircuit operand(2);
        if (p.tx != 0.0) operand.rxx(p.tx / n, 0, 1);
        if (p.ty != 0.0) operand.ryy(p.ty / n, 0, 1);
        if (p.tz != 0.0) operand.rzz(p.tz / n, 0, 1);

        const QuantumCircuit in = repeated_unitary(operand, kBlockLen);
        const QuantumCircuit out = run_consolidate(in);

        EXPECT_TRUE(in_kak_alphabet(out))
            << "handed back unconsolidated.\n  out: " << render(out);
        EXPECT_LE(count_2q(out), 3) << "  out: " << render(out);
        EXPECT_TRUE(same_operator_up_to_phase(in, out))
            << "  out: " << render(out);
    }
}

// =============================================================================
// Operands the SU(4) normalisation has to bring in
// =============================================================================

// e^{i*alpha}*U is unitary but has det scaled by e^{4*i*alpha}, so it is the
// operand the SU(4) normalisation exists for. Without that step the magic-basis
// Gram matrix is not the one the branch selection assumes.
TEST(R1214KakDiagnostic, OperandsOutsideSU4AreNormalisedAndConsolidated) {
    QuantumCircuit src(2);
    src.h(0).cx(0, 1).rz(0.7, 1).t(0);
    auto u4 = Operator::from_circuit(src).data;

    const double alpha = PI / 5.0;
    const Complex128 scale(std::cos(alpha), std::sin(alpha));
    for (auto& e : u4) e = e * scale;

    QuantumCircuit in(2);
    for (int i = 0; i < kBlockLen; ++i) in.unitary(u4, {0, 1});

    const QuantumCircuit out = run_consolidate(in);
    EXPECT_TRUE(in_kak_alphabet(out))
        << "handed back unconsolidated.\n  out: " << render(out);
    EXPECT_TRUE(same_operator_up_to_phase(in, out))
        << "  out: " << render(out);
}

// =============================================================================
// Properties of the pass itself, which no operand-by-operand test can state
// =============================================================================

// The branch selection ranks candidates and takes the best. A tie broken by
// iteration order would make the emitted angles depend on the standard library
// implementation, which is the kind of difference that shows up as a
// cross-platform CI failure and nowhere else.
TEST(R1214KakDiagnostic, DecompositionIsDeterministic) {
    std::mt19937_64 rng(kSeed);
    for (int trial = 0; trial < 32; ++trial) {
        SCOPED_TRACE("trial " + std::to_string(trial));
        const QuantumCircuit in = haar_block(rng);
        EXPECT_EQ(fingerprint(run_consolidate(in)),
                  fingerprint(run_consolidate(in)))
            << "the same operand decomposed twice gave different output";
    }
}

// Running the pass on its own output must be a no-op. The three emitted
// rotations are adjacent, so they form a block of three that the count guard
// has to decline; if it ever accepts one, repeated transpilation would drift.
TEST(R1214KakDiagnostic, ConsolidationIsIdempotent) {
    std::mt19937_64 rng(kSeed);
    for (int trial = 0; trial < 32; ++trial) {
        SCOPED_TRACE("trial " + std::to_string(trial));
        const QuantumCircuit once = run_consolidate(haar_block(rng));
        const QuantumCircuit twice = run_consolidate(once);
        EXPECT_EQ(fingerprint(once), fingerprint(twice))
            << "a second pass changed the output.\n"
            << "  once : " << render(once) << "\n"
            << "  twice: " << render(twice);
    }
}

// The verification net is what made the R.1.22.0 defect invisible: it discards
// a bad factorisation and keeps the original instructions. That makes it the
// one component whose failure mode is silence, so it is asserted directly.
//
// unitary() measures unitarity and would reject this operand at construction,
// so it only reaches the pass with the check opted out. The net is a separate
// numerical rebuild-and-compare and does not consult ValidationOptions.
TEST(R1214KakDiagnostic, ACorruptedOperandIsHandedBackUnchanged) {
    QuantumCircuit src(2);
    src.h(0).cx(0, 1).rz(0.7, 1).t(0);
    auto bad = Operator::from_circuit(src).data;
    bad[0] = bad[0] * 1.5;

    QuantumCircuit in(2);
    const ValidationOptions unchecked{Validation::Ignore};
    for (int i = 0; i < kBlockLen; ++i) in.unitary(bad, {0, 1}, "", unchecked);

    QuantumCircuit out(2);
    ASSERT_NO_THROW(out = run_consolidate(in))
        << "a non-unitary operand must be declined, not propagated as an "
           "exception out of an optimisation pass";

    EXPECT_EQ(count_2q(out), kBlockLen)
        << "the block should have been handed back untouched.\n"
        << "  out: " << render(out);
    EXPECT_FALSE(in_kak_alphabet(out))
        << "a factorisation of a non-unitary operand was accepted.\n"
        << "  out: " << render(out);
}

// The pass recovers a UNITARY's 4x4 by pushing four basis states through
// gates::apply_unitary, when the instruction already holds those sixteen
// numbers in the same row-major layout. Reading them directly would be both
// cheaper and free of the unitarity check that the simulation drags in.
//
// That equivalence is an assumption until something states it, so this states
// it. The non-unitary case is the one that matters: a direct read has to
// reproduce the simulated result even for a matrix no check would accept, since
// that is precisely the operand the substitution exists to handle.
//
// Applying a matrix to |col> selects column col, so the comparison is exact
// rather than approximate and needs no tolerance of its own.
TEST(R1214KakDiagnostic, BasisColumnSimulationEqualsTheStoredMatrix) {
    QuantumCircuit src(2);
    src.h(0).cx(0, 1).rz(0.7, 1).t(0);
    const auto good = Operator::from_circuit(src).data;

    auto broken = good;
    broken[0] = broken[0] * 1.5;

    const std::pair<const char*, const std::vector<Complex128>*> cases[] = {
        {"a unitary operand", &good},
        {"a non-unitary operand", &broken},
    };

    const ValidationOptions unchecked{Validation::Ignore};
    for (const auto& c : cases) {
        SCOPED_TRACE(c.first);
        const std::vector<Complex128>& m = *c.second;
        ASSERT_EQ(m.size(), 16u);

        for (int col = 0; col < 4; ++col) {
            Statevector basis(2);
            basis.initialize_basis(static_cast<size_t>(col));
            gates::apply_unitary(basis, {0, 1}, m, unchecked);

            const auto amps = basis.amplitudes();
            ASSERT_EQ(amps.size(), 4u);
            for (int row = 0; row < 4; ++row) {
                const Complex128& stored = m[row * 4 + col];
                EXPECT_DOUBLE_EQ(amps[row].real, stored.real)
                    << "real part at (" << row << ", " << col << ")";
                EXPECT_DOUBLE_EQ(amps[row].imag, stored.imag)
                    << "imaginary part at (" << row << ", " << col << ")";
            }
        }
    }
}
