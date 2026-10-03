// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.31.2 - every number the library writes or reads as text, in any locale.
//
// A process can switch its global C++ locale, and a named one sets the C
// locale too. 1.1.31.1 pinned the circuit's and the noise model's exporters
// against a decimal-comma locale (test_v11311_json.cpp). The same defect sat
// in every other place a number becomes text: the four drawings, a
// statevector's text, the failed-run folder name and the Estimator's transpile
// key followed the C++ locale, the drawings' parameter labels followed the C
// locale through printf, and the QASM readers read through std::stod, which
// follows the C locale as well. Every one now goes through the writers and the
// reader in lindblad/detail/text.hpp, and these tests hold each to the text it
// produces in the classic locale.
//
// The C locale cannot be set without a decimal-comma locale installed on the
// host. Those tests skip on a host that has none, and say so.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/detail/text.hpp"
#include "lindblad/errors.hpp"
#include "lindblad/failed_run.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"
#include "v11311_helpers.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <typeinfo>
#include <vector>

using namespace lindblad;
using GT = Instruction::GateType;
using v11311::ScopedCommaCLocale;
using v11311::ScopedCommaLocale;

namespace {

bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }

// What printf writes for `v` under `format` in the C locale this suite runs
// in, which is "C" unless a test sets it. The writers' expectations are
// derived from it rather than typed out.
std::string c_formatted(const char* format, double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, format, v);
    return buf;
}

// Whether `T` can be written into a TextBuilder.
template <class T>
concept Writable = requires(detail::TextBuilder& b, T v) { b << v; };

// A circuit whose drawings carry numbers in every renderer: a fractional
// angle, a two-qubit gate, a conditioned gate and a measurement.
QuantumCircuit drawn() {
    QuantumCircuit qc(3, 3);
    qc.h(0).rx(0.3, 1).cx(0, 2).measure(2, 2);
    qc.add_if(2, 1, GT::X, {1});
    return qc;
}

// 45 px cells put every gate centre on a half pixel, so the SVG carries
// fractional coordinates and a decimal comma would show in them.
DrawOptions drawing_options() {
    DrawOptions opts;
    opts.show_clbits = true;
    opts.include_legend = true;
    opts.cell_width_px = 45;
    opts.cell_height_px = 45;
    return opts;
}

constexpr DrawMode kEveryMode[] = {DrawMode::ASCII, DrawMode::SVG, DrawMode::LATEX, DrawMode::HTML};

}  // namespace

// =============================================================================
// The writers and the builder
// =============================================================================

TEST(V11312Locale, TheWritersIgnoreTheGlobalLocale) {
    const double fraction = 1234567.25;
    const std::string g17 = c_formatted("%.17g", fraction);
    const std::string g15 = c_formatted("%.15g", fraction);
    const std::string f4 = c_formatted("%.4f", fraction);
    const ScopedCommaLocale comma;
    EXPECT_EQ(detail::integer_text(1234567), std::to_string(1234567));
    EXPECT_EQ(detail::integer_text(-1234), std::to_string(-1234));
    EXPECT_EQ(detail::integer_text(std::numeric_limits<std::int64_t>::min()),
              std::to_string(std::numeric_limits<std::int64_t>::min()));
    EXPECT_EQ(detail::integer_text(std::numeric_limits<std::uint64_t>::max()),
              std::to_string(std::numeric_limits<std::uint64_t>::max()));
    EXPECT_EQ(detail::integer_text(0xBEEFu, 16), "beef");
    EXPECT_EQ(detail::double_text(fraction, std::numeric_limits<double>::max_digits10), g17);
    EXPECT_EQ(detail::double_text(fraction, std::numeric_limits<double>::digits10), g15);
    EXPECT_EQ(detail::fixed_text(fraction, 4), f4);
}

TEST(V11312Locale, ATextBuilderTakesTextAndRefusesEveryNumber) {
    // A number reaches exporter text only through the writers: writing one
    // directly does not compile.
    static_assert(Writable<const char*>);
    static_assert(Writable<std::string>);
    static_assert(Writable<std::string_view>);
    static_assert(Writable<char>);
    static_assert(!Writable<int>);
    static_assert(!Writable<unsigned>);
    static_assert(!Writable<std::size_t>);
    static_assert(!Writable<std::int64_t>);
    static_assert(!Writable<signed char>);
    static_assert(!Writable<unsigned char>);
    static_assert(!Writable<bool>);
    static_assert(!Writable<float>);
    static_assert(!Writable<double>);
    detail::TextBuilder b;
    b << "q[" << detail::integer_text(3) << ']';
    EXPECT_EQ(b.str(), "q[3]");
    EXPECT_EQ(std::move(b).str(), "q[3]");
}

// =============================================================================
// The reader keeps std::stod's contract
// =============================================================================

TEST(V11312Locale, TheReaderReadsWhatStodReads) {
    // The suite's C locale is "C", where std::stod is the classic reader the
    // QASM parsers were written against.
    for (const std::string text : {"0.5", "  0.5", "\t7", "+0.5", "-0.5", "+.5", ".5", "5.",
                                   "2 * pi", "1e", "1e5x", "-0", "12abc", "6.02214076e23"}) {
        SCOPED_TRACE(text);
        std::size_t want_used = 0;
        const double want = std::stod(text, &want_used);
        std::size_t used = 0;
        const double got = detail::parse_double(text, &used);
        EXPECT_TRUE(same_bits(got, want)) << got << " vs " << want;
        EXPECT_EQ(used, want_used);
        EXPECT_TRUE(same_bits(detail::parse_double(text), want)) << "without `used`";
    }
    for (const std::string text : {"", " ", "abc", "+", "+-1", "-+1", "--1", "+ 1", ".", "pi"}) {
        SCOPED_TRACE(text);
        EXPECT_THROW((void)std::stod(text), std::invalid_argument) << "the table is wrong";
        try {
            (void)detail::parse_double(text);
            ADD_FAILURE() << "accepted";
        } catch (const std::exception& e) {
            EXPECT_EQ(typeid(e), typeid(std::invalid_argument)) << e.what();
            EXPECT_EQ(std::string(e.what()), "parse_double: no number at the start of '" + text + "'");
        }
    }
    for (const std::string text : {"1e999", "-1e999", "1e-400"}) {
        SCOPED_TRACE(text);
        EXPECT_THROW((void)std::stod(text), std::out_of_range) << "the table is wrong";
        try {
            (void)detail::parse_double(text);
            ADD_FAILURE() << "accepted";
        } catch (const std::exception& e) {
            EXPECT_EQ(typeid(e), typeid(std::out_of_range)) << e.what();
            EXPECT_EQ(std::string(e.what()), "parse_double: '" + text + "' is outside the range of a double");
        }
    }
}

TEST(V11312Locale, TheReaderIgnoresTheGlobalLocale) {
    const ScopedCommaLocale comma;
    std::size_t used = 0;
    EXPECT_EQ(detail::parse_double("1234.5", &used), 1234.5);
    EXPECT_EQ(used, std::string_view("1234.5").size());
}

// =============================================================================
// QASM, under the C++ locale
// =============================================================================

TEST(V11312Locale, AWideCircuitsQasmIsTheSameUnderGroupedThousands) {
    // 1235 qubits: every index of four digits is grouped, and the register
    // sizes too.
    QuantumCircuit plain(1235, 1235);
    plain.rx(0.5, 1234).cx(1000, 1234).barrier({1000, 1234}).reset(1001).measure(1234, 1234);
    QuantumCircuit conditioned = plain;
    conditioned.add_if(1234, 1, GT::X, {1001});
    const std::string classic2 = plain.to_qasm2();
    const std::string classic3 = conditioned.to_qasm3();
    ASSERT_NE(classic2.find("q[1234]"), std::string::npos) << classic2;
    ASSERT_NE(classic3.find("c[1234] == 1"), std::string::npos) << classic3;
    const ScopedCommaLocale comma;
    EXPECT_EQ(plain.to_qasm2(), classic2);
    EXPECT_EQ(conditioned.to_qasm3(), classic3);
    EXPECT_EQ(QuantumCircuit::from_qasm2(classic2).to_qasm2(), classic2);
    EXPECT_EQ(QuantumCircuit::from_qasm3(classic3).to_qasm3(), classic3);
}

// =============================================================================
// The drawings, a statevector's text, the folder name
// =============================================================================

TEST(V11312Locale, EveryDrawingIsTheSameUnderADecimalComma) {
    const QuantumCircuit qc = drawn();
    const DrawOptions opts = drawing_options();
    std::vector<std::string> classic;
    for (const DrawMode mode : kEveryMode) classic.push_back(qc.draw(mode, opts));
    ASSERT_NE(classic[1].find(".5\""), std::string::npos) << "no fractional coordinate to test";
    const ScopedCommaLocale comma;
    for (std::size_t m = 0; m < classic.size(); ++m) {
        SCOPED_TRACE(m);
        EXPECT_EQ(qc.draw(kEveryMode[m], opts), classic[m]);
    }
}

TEST(V11312Locale, AWideDrawingIsTheSameUnderGroupedThousands) {
    // 1001 rows: the LaTeX row labels and control offsets, the HTML caption's
    // qubit count and the SVG's data-qubits all reach four digits.
    QuantumCircuit qc(1001, 1);
    qc.h(1000).cx(0, 1000).measure(1000, 0);
    const DrawOptions opts = drawing_options();
    std::vector<std::string> classic;
    for (const DrawMode mode : kEveryMode) classic.push_back(qc.draw(mode, opts));
    ASSERT_NE(classic[2].find("q_{1000}"), std::string::npos);
    ASSERT_NE(classic[3].find("1001 qubits"), std::string::npos);
    const ScopedCommaLocale comma;
    for (std::size_t m = 0; m < classic.size(); ++m) {
        SCOPED_TRACE(m);
        EXPECT_EQ(qc.draw(kEveryMode[m], opts), classic[m]);
    }
}

TEST(V11312Locale, AStatevectorsTextIsTheSameUnderADecimalComma) {
    // Eleven qubits: dim = 2048, and the count of entries not shown is grouped.
    Statevector sv(11);
    std::vector<Complex128> amps(std::size_t{1} << 11, Complex128(0.0, 0.0));
    amps[0] = Complex128(INV_SQRT2, 0.0);
    amps[1] = Complex128(0.0, -INV_SQRT2);
    sv.set_amplitudes(amps);
    const std::string classic = sv.to_string(4);
    ASSERT_NE(classic.find(c_formatted("%.4f", INV_SQRT2)), std::string::npos) << classic;
    const ScopedCommaLocale comma;
    EXPECT_EQ(sv.to_string(4), classic);
}

TEST(V11312Locale, AFailedRunsFolderNameHasNoSeparatorsUnderGroupedThousands) {
    // <YYYYmmdd-HHMMSS>-<pid>-<thread>-<counter>: a process id is usually over
    // a thousand, so a grouped one would carry a separator.
    const v11311::TempDir dir("locale-name");
    QuantumCircuit qc(1, 1);
    qc.unitary(std::vector<Complex128>(4, Complex128(0.0, 0.0)), {0}, "zero", {Validation::Ignore});
    qc.measure(0, 0);
    RunPlan plan;
    plan.options.failed_run_dir = dir.path();
    std::optional<FailedRun> record;
    {
        const ScopedCommaLocale comma;
        (void)take_failed_run();
        EXPECT_THROW((void)StatevectorSimulator().run(qc, 3, 1, plan), RuntimeFailure);
        record = take_failed_run();
    }
    ASSERT_TRUE(record.has_value());
    ASSERT_TRUE(record->saved_to.has_value()) << record->save_note;
    const std::string name = record->saved_to->filename().string();
    EXPECT_TRUE(std::regex_match(name, std::regex("[0-9]{8}-[0-9]{6}-[0-9]+-[0-9a-f]+-[0-9]+"))) << name;
}

// =============================================================================
// Under a decimal-comma C locale
// =============================================================================

TEST(V11312Locale, LabelsAndTheQasmReadersIgnoreTheCLocale) {
    QuantumCircuit qc(1);
    qc.rx(0.3, 0);
    DrawOptions raw;
    raw.param_format = ParamFormat::Raw;
    const std::string classic_drawing = qc.draw(DrawMode::ASCII, raw);
    ASSERT_NE(classic_drawing.find(c_formatted("%.4f", 0.3)), std::string::npos) << classic_drawing;
    const std::string qasm2 = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\nrx(0.25) q[0];\n";
    const std::string qasm3 = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[1] q;\nrx(0.25) q[0];\n";

    const ScopedCommaCLocale comma;
    if (!comma.ok()) GTEST_SKIP() << "no decimal-comma C locale is installed on this host";
    SCOPED_TRACE(comma.name());
    EXPECT_EQ(qc.draw(DrawMode::ASCII, raw), classic_drawing);
    const QuantumCircuit from2 = QuantumCircuit::from_qasm2(qasm2);
    ASSERT_EQ(from2.instructions.size(), 1u);
    EXPECT_EQ(from2.instructions[0].params, std::vector<double>{0.25});
    const QuantumCircuit from3 = QuantumCircuit::from_qasm3(qasm3);
    ASSERT_EQ(from3.instructions.size(), 1u);
    EXPECT_EQ(from3.instructions[0].params, std::vector<double>{0.25});
    EXPECT_EQ(detail::parse_double("0.25"), 0.25);
}

// =============================================================================
// A circuit's JSON carries a non-finite parameter
// =============================================================================

TEST(V11312Locale, ACircuitsNonFiniteNumbersSurviveItsJson) {
    // The circuit's JSON writes every double through json_number, which spells
    // the values JSON has no number for as strings, and reads them back.
    QuantumCircuit qc(1);
    qc.rx(0.5, 0).ry(0.5, 0).rz(0.5, 0);
    qc.instructions[0].params[0] = quiet_nan_strict();
    qc.instructions[1].params[0] = std::numeric_limits<double>::infinity();
    qc.instructions[2].params[0] = -std::numeric_limits<double>::infinity();
    const std::string json = qc.to_json();
    EXPECT_NE(json.find("\"NaN\""), std::string::npos) << json;
    EXPECT_NE(json.find("\"Infinity\""), std::string::npos) << json;
    EXPECT_NE(json.find("\"-Infinity\""), std::string::npos) << json;
    const QuantumCircuit back = QuantumCircuit::from_json(json);
    ASSERT_EQ(back.instructions.size(), 3u);
    EXPECT_FALSE(is_finite_strict(back.instructions[0].params[0]));
    EXPECT_TRUE(same_bits(back.instructions[1].params[0], std::numeric_limits<double>::infinity()));
    EXPECT_TRUE(same_bits(back.instructions[2].params[0], -std::numeric_limits<double>::infinity()));
    EXPECT_EQ(back.to_json(), json);
}
