// 1.1.27.1 test wave - the repair knob through a circuit round trip (#110).
//
// An Instruction carries the policy governing its matrix, and the matrix is
// written entry by entry into the circuit's JSON, so the policy has to travel
// with it or a circuit that ran before being written throws when it is read
// back. Splitting one enum into two knobs therefore splits what has to be
// serialised, and a knob that is written but not read, or read but not written,
// is a silent policy change across a save and load.
//
// The stored form names both knobs rather than writing their enumerator
// integers, so reordering either enumeration cannot change what an existing
// file means.
//
// One stored name carries both: "fix" was the fused spelling of repair followed
// by throw, and a circuit written under it keeps that meaning on the way back
// in. It is accepted on read and never produced on write, so there is one way
// to write each policy and two ways to read the one that used to have a name of
// its own.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"
#include "r1211_policy_probe.hpp"

#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;

namespace {

constexpr double kAtol = DEFAULT_PHYSICAL_ATOL;

std::vector<Complex128> hadamard() {
    const double h = INV_SQRT2;
    return {Complex128(h, 0.0), Complex128(h, 0.0),
            Complex128(h, 0.0), Complex128(-h, 0.0)};
}

// A one-instruction circuit carrying exactly the policy under test.
QuantumCircuit circuit_with(const ValidationOptions& v) {
    QuantumCircuit qc(1);
    qc.unitary(hadamard(), {0}, "u", v);
    return qc;
}

ValidationOptions round_trip(const ValidationOptions& v) {
    const QuantumCircuit back = QuantumCircuit::from_json(circuit_with(v).to_json());
    EXPECT_EQ(back.instructions.size(), 1u);
    return back.instructions.empty() ? ValidationOptions{}
                                     : back.instructions[0].validation;
}

}  // namespace

// =============================================================================
// Every policy survives the trip
// =============================================================================

TEST(V11271ValidationSerialisation, EveryPolicySurvivesARoundTrip) {
    // All six, because a knob that is dropped on the way out is invisible to a
    // test that only ever writes the default.
    for (const auto& v : r1211::kAllPolicies) {
        const ValidationOptions back = round_trip(v);
        EXPECT_EQ(back.policy, v.policy) << r1211::policy_name(v);
        EXPECT_EQ(back.repair, v.repair) << r1211::policy_name(v);
        EXPECT_EQ(back.atol, v.atol) << r1211::policy_name(v);
    }
}

TEST(V11271ValidationSerialisation, ANonDefaultToleranceSurvivesAlongsideTheKnobs) {
    // atol sits between the two knobs in the aggregate and in the stored form,
    // so a reader that mismatched the fields would show up here rather than in
    // the loop above, where every tolerance is the default.
    const ValidationOptions v{Validation::Warn, 1e-9, Repair::Attempt};
    const ValidationOptions back = round_trip(v);
    EXPECT_EQ(back.policy, Validation::Warn);
    EXPECT_EQ(back.atol, 1e-9);
    EXPECT_EQ(back.repair, Repair::Attempt);
}

TEST(V11271ValidationSerialisation, ARepairOnlyDifferenceIsStillWritten) {
    // The block is emitted only where it differs from the default, which keeps
    // a file written by any other route loading unchanged. An instruction that
    // differs ONLY in the repair knob is the case a comparison written before
    // the split would miss: its policy and tolerance are both default.
    const ValidationOptions v{Validation::Throw, kAtol, Repair::Attempt};
    const std::string json = circuit_with(v).to_json();
    EXPECT_NE(json.find("\"validation\""), std::string::npos)
        << "an instruction differing only in its repair knob wrote no policy, "
           "so the knob is lost on the way out";
    EXPECT_EQ(round_trip(v).repair, Repair::Attempt);
}

TEST(V11271ValidationSerialisation, TheDefaultWritesNothingAndReadsBackDefault) {
    // The other half of the same rule. A default-policy instruction carries no
    // block at all, and an absent block means the default rather than a
    // zero-initialised one.
    const QuantumCircuit qc = circuit_with(ValidationOptions{});
    const std::string json = qc.to_json();
    EXPECT_EQ(json.find("\"validation\""), std::string::npos)
        << "the default policy was written out, which changes every stored "
           "circuit for no gain";

    const QuantumCircuit back = QuantumCircuit::from_json(json);
    ASSERT_EQ(back.instructions.size(), 1u);
    EXPECT_EQ(back.instructions[0].validation.policy, Validation::Throw);
    EXPECT_EQ(back.instructions[0].validation.repair, Repair::None);
    EXPECT_EQ(back.instructions[0].validation.atol, DEFAULT_PHYSICAL_ATOL);
}

// =============================================================================
// Names rather than integers
// =============================================================================

TEST(V11271ValidationSerialisation, BothKnobsAreStoredAsNames) {
    // Writing the enumerator's integer would make a stored circuit's meaning
    // depend on the declaration order of an enum, so a reordering that is
    // invisible in source would silently change what every existing file says.
    const std::string json =
        circuit_with({Validation::Warn, kAtol, Repair::Attempt}).to_json();
    EXPECT_NE(json.find("\"warn\""), std::string::npos)
        << "the response is not stored by name. Got: " << json;
    EXPECT_NE(json.find("\"attempt\""), std::string::npos)
        << "the repair knob is not stored by name. Got: " << json;
}

TEST(V11271ValidationSerialisation, TheFusedNameSetsBothKnobs) {
    // "fix" named repair followed by throw, and a circuit stored under it has
    // to keep meaning that. Reading it as the response alone would drop the
    // repair; reading it as the repair alone would drop the response.
    const std::string json =
        circuit_with({Validation::Warn, kAtol, Repair::Attempt}).to_json();

    std::string fused = json;
    const std::size_t at = fused.find("\"warn\"");
    ASSERT_NE(at, std::string::npos);
    fused.replace(at, std::string("\"warn\"").size(), "\"fix\"");
    // Remove the repair key so the fused name is the only thing saying it.
    const std::size_t rk = fused.find(",\"repair\":\"attempt\"");
    ASSERT_NE(rk, std::string::npos);
    fused.erase(rk, std::string(",\"repair\":\"attempt\"").size());

    const QuantumCircuit back = QuantumCircuit::from_json(fused);
    ASSERT_EQ(back.instructions.size(), 1u);
    EXPECT_EQ(back.instructions[0].validation.policy, Validation::Throw)
        << "the fused name did not carry its response";
    EXPECT_EQ(back.instructions[0].validation.repair, Repair::Attempt)
        << "the fused name did not carry its repair";
}

TEST(V11271ValidationSerialisation, TheFusedNameIsNeverWritten) {
    // Accepted on read, never produced. Emitting it would leave two spellings
    // of one policy in the stored form, which is the ambiguity the split
    // removed from the API.
    const std::string json =
        circuit_with({Validation::Throw, kAtol, Repair::Attempt}).to_json();
    EXPECT_EQ(json.find("\"fix\""), std::string::npos)
        << "the fused spelling was written out. Got: " << json;
}

TEST(V11271ValidationSerialisation, AFileWithNoRepairKeyReadsAsNoRepair) {
    // A circuit written before the knob existed names a policy and a tolerance
    // and nothing else. Absent has to mean None, which is also the default, so
    // such a file loads with the behaviour it was written under.
    const std::string json =
        "{\"version\":\"1.0\",\"name\":\"c\",\"n_qubits\":1,\"n_clbits\":0,"
        "\"instructions\":[{\"name\":\"u\",\"qubits\":[0],\"params\":[],"
        "\"matrix\":[[0.7071067811865476,0],[0.7071067811865476,0],"
        "[0.7071067811865476,0],[-0.7071067811865476,0]],"
        "\"validation\":{\"policy\":\"warn\",\"atol\":1e-09}}]}";

    const QuantumCircuit back = QuantumCircuit::from_json(json);
    ASSERT_EQ(back.instructions.size(), 1u);
    EXPECT_EQ(back.instructions[0].validation.policy, Validation::Warn);
    EXPECT_EQ(back.instructions[0].validation.atol, 1e-9);
    EXPECT_EQ(back.instructions[0].validation.repair, Repair::None);
}

// =============================================================================
// A name that means nothing is refused
// =============================================================================

TEST(V11271ValidationSerialisation, AnUnknownRepairNameIsRefused) {
    // Guessing would pick a knob setting the file did not ask for, and the
    // wrong guess towards None turns a repaired operand into an unrepaired one
    // with no indication that anything changed.
    std::string json =
        circuit_with({Validation::Throw, kAtol, Repair::Attempt}).to_json();
    const std::size_t at = json.find("\"attempt\"");
    ASSERT_NE(at, std::string::npos);
    json.replace(at, std::string("\"attempt\"").size(), "\"maybe\"");

    EXPECT_THROW((void)QuantumCircuit::from_json(json), std::runtime_error);
}

TEST(V11271ValidationSerialisation, AnUnknownPolicyNameIsStillRefused) {
    // The rule the repair knob was written to match, asserted alongside it so
    // that the two cannot drift apart.
    std::string json = circuit_with({Validation::Warn, kAtol}).to_json();
    const std::size_t at = json.find("\"warn\"");
    ASSERT_NE(at, std::string::npos);
    json.replace(at, std::string("\"warn\"").size(), "\"perhaps\"");

    EXPECT_THROW((void)QuantumCircuit::from_json(json), std::runtime_error);
}

// =============================================================================
// The policy travels with the instruction
// =============================================================================

TEST(V11271ValidationSerialisation, BothKnobsSurviveACompose) {
    // compose copies instructions, so it carries the policy with them. The
    // repair knob has to ride along with the response rather than being reset
    // to the default, which would quietly disable a repair the caller asked
    // for.
    const ValidationOptions v{Validation::Warn, 1e-9, Repair::Attempt};
    QuantumCircuit inner(1);
    inner.unitary(hadamard(), {0}, "u", v);

    // compose returns a new circuit rather than mutating the receiver.
    const QuantumCircuit outer = QuantumCircuit(2).compose(inner, {1});

    ASSERT_EQ(outer.instructions.size(), 1u);
    EXPECT_EQ(outer.instructions[0].validation.policy, Validation::Warn);
    EXPECT_EQ(outer.instructions[0].validation.atol, 1e-9);
    EXPECT_EQ(outer.instructions[0].validation.repair, Repair::Attempt);
}

TEST(V11271ValidationSerialisation, BothKnobsSurviveACircuitCopy) {
    const ValidationOptions v{Validation::Ignore, 1e-8, Repair::Attempt};
    const QuantumCircuit original = circuit_with(v);
    const QuantumCircuit copy = original;

    ASSERT_EQ(copy.instructions.size(), 1u);
    EXPECT_EQ(copy.instructions[0].validation.policy, Validation::Ignore);
    EXPECT_EQ(copy.instructions[0].validation.atol, 1e-8);
    EXPECT_EQ(copy.instructions[0].validation.repair, Repair::Attempt);
}
