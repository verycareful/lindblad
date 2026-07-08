// R.1.13.1 test patch — CowMatrix (audit F-18, BREAKING).
// The copy-on-write, immutable gate matrix behind Instruction::matrix. Verifies
// the value semantics (read API, equality, implicit const-vector view) AND the
// performance-critical invariant that copies SHARE one buffer instead of
// deep-copying 2^k x 2^k complex data.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/types.hpp"

#include <vector>

using namespace lindblad;

namespace {
std::vector<Complex128> sample4() {
    return {Complex128(1, 0), Complex128(0, 0),
            Complex128(0, 0), Complex128(0, 1)};
}
} // namespace

TEST(R1131CowMatrix, DefaultIsEmpty) {
    CowMatrix m;
    EXPECT_TRUE(m.empty());
    EXPECT_EQ(m.size(), 0u);
    EXPECT_EQ(m.data(), nullptr);
    EXPECT_EQ(m.begin(), m.end());
}

TEST(R1131CowMatrix, ReadApiMirrorsVector) {
    CowMatrix m(sample4());
    EXPECT_FALSE(m.empty());
    ASSERT_EQ(m.size(), 4u);
    EXPECT_EQ(m[0], Complex128(1, 0));
    EXPECT_EQ(m[3], Complex128(0, 1));
    EXPECT_NE(m.data(), nullptr);

    size_t count = 0;
    for (const auto& c : m) { (void)c; ++count; }
    EXPECT_EQ(count, 4u);
}

TEST(R1131CowMatrix, ImplicitConstVectorView) {
    CowMatrix m(sample4());
    const std::vector<Complex128>& v = m;   // implicit conversion
    ASSERT_EQ(v.size(), 4u);
    EXPECT_EQ(v[3], Complex128(0, 1));
    // The view must alias the shared buffer, not a copy.
    EXPECT_EQ(v.data(), m.data());
}

TEST(R1131CowMatrix, CopySharesBuffer) {
    CowMatrix a(sample4());
    CowMatrix b = a;                 // copy
    EXPECT_EQ(a.data(), b.data());   // same underlying buffer (COW, no deep copy)
    EXPECT_EQ(a, b);
}

TEST(R1131CowMatrix, AssignRebindsWithoutAffectingOtherHandles) {
    CowMatrix a(sample4());
    CowMatrix b = a;
    const Complex128* a_buf = a.data();

    b = std::vector<Complex128>{Complex128(2, 0), Complex128(3, 0)};
    EXPECT_NE(a.data(), b.data());   // b rebound to a fresh buffer
    EXPECT_EQ(a.data(), a_buf);      // a untouched
    ASSERT_EQ(a.size(), 4u);
    EXPECT_EQ(a[0], Complex128(1, 0));
    ASSERT_EQ(b.size(), 2u);
    EXPECT_EQ(b[1], Complex128(3, 0));
}

TEST(R1131CowMatrix, EqualityIsByValueNotIdentity) {
    CowMatrix a(sample4());
    CowMatrix b(sample4());          // equal contents, distinct buffers
    EXPECT_NE(a.data(), b.data());
    EXPECT_EQ(a, b);

    CowMatrix c(std::vector<Complex128>{Complex128(9, 9)});
    EXPECT_NE(a, c);
    EXPECT_TRUE(a != c);
}

TEST(R1131CowMatrix, InitializerListConstructionAndAssignment) {
    CowMatrix m{Complex128(1, 0), Complex128(0, 1)};
    ASSERT_EQ(m.size(), 2u);
    EXPECT_EQ(m[1], Complex128(0, 1));
    m = {Complex128(5, 0)};
    ASSERT_EQ(m.size(), 1u);
    EXPECT_EQ(m[0], Complex128(5, 0));
}

// The headline F-18 win: copying an Instruction (hence a whole QuantumCircuit)
// does NOT deep-copy the gate matrix — both circuits point at one buffer.
TEST(R1131CowMatrix, CircuitCopyDoesNotDeepCopyMatrix) {
    QuantumCircuit qc(1);
    qc.unitary(sample4(), {0}, "u");
    ASSERT_EQ(qc.instructions.size(), 1u);

    QuantumCircuit copy = qc;        // full circuit copy
    ASSERT_EQ(copy.instructions.size(), 1u);
    EXPECT_EQ(qc.instructions[0].matrix.data(),
              copy.instructions[0].matrix.data())
        << "circuit copy deep-copied the gate matrix (F-18 regression)";
    EXPECT_EQ(qc.instructions[0].matrix, copy.instructions[0].matrix);
}
