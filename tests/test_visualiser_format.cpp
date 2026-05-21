// =============================================================================
// tests/test_visualiser_format.cpp : R.1.10.1 FormatParamsTest
// =============================================================================
// Direct tests for the numeric and symbolic parameter formatters declared in
// src/visualisation/format_params.hpp. The pi-snap table is exercised entry
// by entry (positive and negative); the ParamExpr recursion is exercised
// for every node kind plus precedence-aware parenthesisation.

#include "lindblad/circuit.hpp"
#include "lindblad/visualisation.hpp"

#include "../src/visualisation/format_params.hpp"
#include "../src/visualisation/gate_symbols.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <string>

using namespace lindblad;
using namespace lindblad::viz;

namespace {

// UTF-8 pi (U+03C0) byte sequence; the formatter writes pi as 0xCF 0x80.
const std::string kPi = "\xcf\x80";
// UTF-8 middle dot (U+00B7) used for multiplication.
const std::string kMidDot = "\xc2\xb7";

constexpr double kPiD = 3.141592653589793238462643383279502884;

DrawOptions default_opts() { return DrawOptions{}; }

} // namespace

// =============================================================================
// format_param(double) : pi-snap table -- positive magnitudes
// =============================================================================

TEST(FormatParamsTest, PiSnapsToPi) {
    EXPECT_EQ(format_param(kPiD, ParamFormat::Pretty), kPi);
}

TEST(FormatParamsTest, TwoPiSnapsToTwoPi) {
    EXPECT_EQ(format_param(2.0 * kPiD, ParamFormat::Pretty), "2" + kPi);
}

TEST(FormatParamsTest, ThreePiSnaps) {
    EXPECT_EQ(format_param(3.0 * kPiD, ParamFormat::Pretty), "3" + kPi);
}

TEST(FormatParamsTest, FourPiSnaps) {
    EXPECT_EQ(format_param(4.0 * kPiD, ParamFormat::Pretty), "4" + kPi);
}

TEST(FormatParamsTest, ThreeHalvesPiSnaps) {
    EXPECT_EQ(format_param(1.5 * kPiD, ParamFormat::Pretty), "3" + kPi + "/2");
}

TEST(FormatParamsTest, PiOverTwoSnaps) {
    EXPECT_EQ(format_param(kPiD / 2.0, ParamFormat::Pretty), kPi + "/2");
}

TEST(FormatParamsTest, TwoPiOverThreeSnaps) {
    EXPECT_EQ(format_param(2.0 * kPiD / 3.0, ParamFormat::Pretty), "2" + kPi + "/3");
}

TEST(FormatParamsTest, PiOverThreeSnaps) {
    EXPECT_EQ(format_param(kPiD / 3.0, ParamFormat::Pretty), kPi + "/3");
}

TEST(FormatParamsTest, ThreePiOverFourSnaps) {
    EXPECT_EQ(format_param(3.0 * kPiD / 4.0, ParamFormat::Pretty), "3" + kPi + "/4");
}

TEST(FormatParamsTest, PiOverFourSnaps) {
    EXPECT_EQ(format_param(kPiD / 4.0, ParamFormat::Pretty), kPi + "/4");
}

TEST(FormatParamsTest, FivePiOverSixSnaps) {
    EXPECT_EQ(format_param(5.0 * kPiD / 6.0, ParamFormat::Pretty), "5" + kPi + "/6");
}

TEST(FormatParamsTest, PiOverSixSnaps) {
    EXPECT_EQ(format_param(kPiD / 6.0, ParamFormat::Pretty), kPi + "/6");
}

TEST(FormatParamsTest, PiOverEightSnaps) {
    EXPECT_EQ(format_param(kPiD / 8.0, ParamFormat::Pretty), kPi + "/8");
}

// =============================================================================
// format_param(double) : negative pi-snaps
// =============================================================================

TEST(FormatParamsTest, NegativePiSnaps) {
    EXPECT_EQ(format_param(-kPiD, ParamFormat::Pretty), "-" + kPi);
}

TEST(FormatParamsTest, NegativePiOverTwoSnaps) {
    EXPECT_EQ(format_param(-kPiD / 2.0, ParamFormat::Pretty), "-" + kPi + "/2");
}

TEST(FormatParamsTest, NegativeThreePiOverFourSnaps) {
    EXPECT_EQ(format_param(-3.0 * kPiD / 4.0, ParamFormat::Pretty),
              "-3" + kPi + "/4");
}

TEST(FormatParamsTest, NegativePiOverEightSnaps) {
    EXPECT_EQ(format_param(-kPiD / 8.0, ParamFormat::Pretty), "-" + kPi + "/8");
}

// =============================================================================
// format_param(double) : zero, miss, raw mode
// =============================================================================

TEST(FormatParamsTest, ZeroFormatsAsBareZero) {
    EXPECT_EQ(format_param(0.0, ParamFormat::Pretty), "0");
}

TEST(FormatParamsTest, NegativeZeroFormatsAsBareZero) {
    EXPECT_EQ(format_param(-0.0, ParamFormat::Pretty), "0");
}

TEST(FormatParamsTest, MissesFallBackToFourDecimal) {
    EXPECT_EQ(format_param(1.0, ParamFormat::Pretty), "1.0000");
}

TEST(FormatParamsTest, ArbitraryDecimalFormatsWithFourDecimals) {
    EXPECT_EQ(format_param(0.31415, ParamFormat::Pretty), "0.3142");
}

TEST(FormatParamsTest, RawModeNeverPiSnaps) {
    // Even an exact pi must format as 3.1416 in Raw mode.
    EXPECT_EQ(format_param(kPiD, ParamFormat::Raw), "3.1416");
}

TEST(FormatParamsTest, RawModeFourDecimalForArbitraryValue) {
    EXPECT_EQ(format_param(0.123456, ParamFormat::Raw), "0.1235");
}

TEST(FormatParamsTest, RawModeZeroFormatsAsZeroPointZero) {
    EXPECT_EQ(format_param(0.0, ParamFormat::Raw), "0.0000");
}

// =============================================================================
// format_param(ParamExpr) : node kinds
// =============================================================================

TEST(FormatParamsTest, ParamExprLiteralFormatsViaDoubleOverload) {
    auto e = ParamExpr::make_literal(kPiD / 2.0);
    EXPECT_EQ(format_param(e, ParamFormat::Pretty), kPi + "/2");
}

TEST(FormatParamsTest, ParamExprNamePassesThrough) {
    auto e = ParamExpr::make_name("theta");
    EXPECT_EQ(format_param(e, ParamFormat::Pretty), "theta");
}

TEST(FormatParamsTest, ParamExprAddRendersWithPlus) {
    auto e = ParamExpr::make_binary('+',
        ParamExpr::make_name("theta"),
        ParamExpr::make_literal(kPiD / 4.0));
    EXPECT_EQ(format_param(e, ParamFormat::Pretty),
              "theta+" + kPi + "/4");
}

TEST(FormatParamsTest, ParamExprSubRendersWithMinus) {
    auto e = ParamExpr::make_binary('-',
        ParamExpr::make_name("alpha"),
        ParamExpr::make_name("beta"));
    EXPECT_EQ(format_param(e, ParamFormat::Pretty), "alpha-beta");
}

TEST(FormatParamsTest, ParamExprMulRendersWithMiddleDot) {
    auto e = ParamExpr::make_binary('*',
        ParamExpr::make_literal(2.0),
        ParamExpr::make_name("theta"));
    EXPECT_EQ(format_param(e, ParamFormat::Pretty),
              "2.0000" + kMidDot + "theta");
}

TEST(FormatParamsTest, ParamExprDivRendersWithSlash) {
    auto e = ParamExpr::make_binary('/',
        ParamExpr::make_name("theta"),
        ParamExpr::make_literal(2.0));
    EXPECT_EQ(format_param(e, ParamFormat::Pretty), "theta/2.0000");
}

// =============================================================================
// ParamExpr precedence parens
// =============================================================================

TEST(FormatParamsTest, LowerPrecedenceChildOfMulGetsParens) {
    // (a+b)*c -- the inner '+' has lower precedence than '*' so it must
    // be parenthesised.
    auto e = ParamExpr::make_binary('*',
        ParamExpr::make_binary('+',
            ParamExpr::make_name("a"), ParamExpr::make_name("b")),
        ParamExpr::make_name("c"));
    EXPECT_EQ(format_param(e, ParamFormat::Pretty),
              "(a+b)" + kMidDot + "c");
}

TEST(FormatParamsTest, EqualPrecedenceChildSkipsParens) {
    // a*b*c -- the inner '*' has equal precedence as the outer '*' so no
    // parens are added.
    auto e = ParamExpr::make_binary('*',
        ParamExpr::make_binary('*',
            ParamExpr::make_name("a"), ParamExpr::make_name("b")),
        ParamExpr::make_name("c"));
    EXPECT_EQ(format_param(e, ParamFormat::Pretty),
              "a" + kMidDot + "b" + kMidDot + "c");
}

TEST(FormatParamsTest, HigherPrecedenceChildOfAddSkipsParens) {
    // a*b+c -- the inner '*' is HIGHER precedence than '+', so no parens.
    auto e = ParamExpr::make_binary('+',
        ParamExpr::make_binary('*',
            ParamExpr::make_name("a"), ParamExpr::make_name("b")),
        ParamExpr::make_name("c"));
    EXPECT_EQ(format_param(e, ParamFormat::Pretty),
              "a" + kMidDot + "b+c");
}

TEST(FormatParamsTest, RightChildOfMulAlsoGetsParensForLowerPrec) {
    // a*(b+c) -- the right '+' has lower precedence than the outer '*'.
    auto e = ParamExpr::make_binary('*',
        ParamExpr::make_name("a"),
        ParamExpr::make_binary('+',
            ParamExpr::make_name("b"), ParamExpr::make_name("c")));
    EXPECT_EQ(format_param(e, ParamFormat::Pretty),
              "a" + kMidDot + "(b+c)");
}

// =============================================================================
// format_gate_label
// =============================================================================

TEST(FormatParamsTest, LabelWithoutParamsHidesParens) {
    // Hadamard's catalogue entry has show_params = false.
    Instruction inst;
    inst.type   = Instruction::GateType::H;
    inst.qubits = {0};
    auto label = format_gate_label(inst, symbol_catalogue().at(Instruction::GateType::H),
                                    default_opts());
    EXPECT_EQ(label, "H");
}

TEST(FormatParamsTest, LabelWithSingleNumericParam) {
    Instruction inst;
    inst.type   = Instruction::GateType::RX;
    inst.qubits = {0};
    inst.params = {kPiD / 2.0};
    auto label = format_gate_label(inst, symbol_catalogue().at(Instruction::GateType::RX),
                                    default_opts());
    EXPECT_EQ(label, "RX(" + kPi + "/2)");
}

TEST(FormatParamsTest, LabelWithThreeNumericParams) {
    Instruction inst;
    inst.type   = Instruction::GateType::U;
    inst.qubits = {0};
    inst.params = {kPiD / 2.0, kPiD / 4.0, 0.0};
    auto label = format_gate_label(inst, symbol_catalogue().at(Instruction::GateType::U),
                                    default_opts());
    EXPECT_EQ(label, "U(" + kPi + "/2, " + kPi + "/4, 0)");
}

TEST(FormatParamsTest, OptShowParamsFalseStripsParens) {
    Instruction inst;
    inst.type   = Instruction::GateType::RX;
    inst.qubits = {0};
    inst.params = {kPiD / 2.0};
    DrawOptions opts;
    opts.show_params = false;
    auto label = format_gate_label(inst, symbol_catalogue().at(Instruction::GateType::RX), opts);
    EXPECT_EQ(label, "RX");
}

TEST(FormatParamsTest, SymbolicParamExprPreferredOverNumeric) {
    // When both param_exprs and params are populated, param_exprs wins.
    Instruction inst;
    inst.type   = Instruction::GateType::RX;
    inst.qubits = {0};
    inst.params = {1.5708};
    inst.param_exprs.push_back(ParamExpr::make_name("theta"));
    auto label = format_gate_label(inst, symbol_catalogue().at(Instruction::GateType::RX),
                                    default_opts());
    EXPECT_EQ(label, "RX(theta)");
}

TEST(FormatParamsTest, RawParamFormatPropagatesIntoLabel) {
    Instruction inst;
    inst.type   = Instruction::GateType::RX;
    inst.qubits = {0};
    inst.params = {kPiD / 2.0};
    DrawOptions opts;
    opts.param_format = ParamFormat::Raw;
    auto label = format_gate_label(inst, symbol_catalogue().at(Instruction::GateType::RX), opts);
    EXPECT_EQ(label, "RX(1.5708)");
}

TEST(FormatParamsTest, NoParamsProducesBareLabelEvenWhenShowParamsTrue) {
    Instruction inst;
    inst.type   = Instruction::GateType::RX;
    inst.qubits = {0};
    // No params and no exprs: the label should not include "()".
    auto label = format_gate_label(inst, symbol_catalogue().at(Instruction::GateType::RX),
                                    default_opts());
    EXPECT_EQ(label, "RX");
}
