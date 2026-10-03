// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.31.1 - numbers in text, in any locale, and the noise model's JSON.
//
// 1.1.31.0 shared one JSON reader and writer between the circuit, the noise
// model and the failed-run folder, wrote doubles with 17 significant digits in
// the classic locale, read them back with std::from_chars, and refused an
// integer field holding a fraction or a value an int cannot hold (M7). It
// also gave NoiseModel a lossless to_json and from_json, which nothing tested.
//
// A process can switch its global C++ locale (a desktop application set to a
// German or French locale does), and every stream that does not choose its
// own then formats a number with a decimal comma and grouped thousands. The
// tests install such a locale with a facet of their own, so they need no
// locale installed on the host.
//
// One defect these tests found shipped red in 1.1.31.1 and was fixed in
// 1.1.31.2: the circuit's to_json and QASM exporters, and the noise model's
// integer fields, wrote through streams that follow the global locale, so
// under a decimal-comma locale rx(0.5) was written as rx(0,5) and qubit 1234 as
// 1.234. test_v11312_locale.cpp carries the same check to every other writer
// and reader of numbers.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/detail/json.hpp"
#include "lindblad/errors.hpp"
#include "lindblad/failed_run.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "v11311_helpers.hpp"

#include <clocale>
#include <cstdint>
#include <cstring>
#include <limits>
#include <locale>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lindblad;

namespace {

using v11311::ScopedCommaLocale;

double parse(const std::string& token) { return detail::JsonReader::parse_number(token); }

int read_int(const std::string& text) {
    detail::JsonReader r{text};
    return r.read_int();
}

bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }

}  // namespace

// =============================================================================
// The shared reader and writer (M7)
// =============================================================================

TEST(V11311Json, AnIntFieldRefusesAFractionAndAValueOutOfRange) {
    EXPECT_EQ(read_int("2147483647"), std::numeric_limits<int>::max());
    EXPECT_EQ(read_int("-2147483648"), std::numeric_limits<int>::min());
    EXPECT_EQ(read_int("-0"), 0);
    const auto refused = [](const std::string& text, const std::string& what) {
        const auto e = v11311::thrown<std::runtime_error>([&] { (void)read_int(text); });
        ASSERT_TRUE(e.has_value());
        EXPECT_EQ(std::string(e->what()), "JSON parse error: " + what);
    };
    refused("1.5", std::to_string(1.5) + " is not a whole number");
    refused("2147483648", std::to_string(2147483648.0) + " is outside the range of an int");
    refused("-2147483649", std::to_string(-2147483649.0) + " is outside the range of an int");
    refused("1e300", std::to_string(1e300) + " is outside the range of an int");
}

TEST(V11311Json, ATokenThatIsNotWhollyANumberIsRefused) {
    for (const std::string token : {"abc", "1.5.2", "1e", "--1", "", "1e400"}) {
        SCOPED_TRACE(token);
        const auto e = v11311::thrown<std::runtime_error>([&] { (void)parse(token); });
        ASSERT_TRUE(e.has_value());
        EXPECT_EQ(std::string(e->what()), "JSON parse error: '" + token + "' is not a number");
    }
}

TEST(V11311Json, EveryDoubleSurvivesTheTextExactly) {
    const double values[] = {0.1, 1.0 / 3.0, -0.0, std::numeric_limits<double>::denorm_min(),
                             std::numeric_limits<double>::min(), std::numeric_limits<double>::max(),
                             -std::numeric_limits<double>::max(), 6.02214076e23, PI};
    for (const double v : values) {
        SCOPED_TRACE(v);
        const std::string text = detail::json_number(v);
        detail::JsonReader r{text};
        EXPECT_TRUE(same_bits(r.read_double(), v)) << text;
    }
    // JSON has no NaN or infinity; they are written as strings and read back.
    EXPECT_EQ(detail::json_number(std::numeric_limits<double>::infinity()), "\"Infinity\"");
    EXPECT_EQ(detail::json_number(-std::numeric_limits<double>::infinity()), "\"-Infinity\"");
    EXPECT_EQ(detail::json_number(quiet_nan_strict()), "\"NaN\"");
    for (const std::string text : {"\"Infinity\"", "\"-Infinity\"", "\"NaN\""}) {
        detail::JsonReader r{text};
        const double v = r.read_double();
        EXPECT_EQ(is_finite_strict(v), false) << text;
    }
    detail::JsonReader bad{std::string("\"Inf\"")};
    EXPECT_THROW((void)bad.read_double(), std::runtime_error);
}

TEST(V11311Json, TheReaderAndWriterIgnoreTheGlobalLocale) {
    const ScopedCommaLocale comma;
    EXPECT_EQ(detail::json_number(0.5), "0.5");
    EXPECT_EQ(detail::json_number(1234567.25), "1234567.25");
    EXPECT_EQ(parse("0.5"), 0.5);
    EXPECT_EQ(parse("1234567.25"), 1234567.25);
}

TEST(V11311Json, TheReaderIgnoresTheCLocaleToo) {
    // The C locale is a separate setting. A host with no decimal-comma locale
    // installed cannot set one, and the test says so rather than passing.
    const char* previous = std::setlocale(LC_NUMERIC, nullptr);
    const std::string saved = previous != nullptr ? previous : "C";
    const char* candidates[] = {"de_DE.UTF-8", "de_DE.utf8", "fr_FR.UTF-8", "fr_FR.utf8"};
    const char* chosen = nullptr;
    for (const char* name : candidates) {
        if (std::setlocale(LC_NUMERIC, name) != nullptr) {
            chosen = name;
            break;
        }
    }
    if (chosen == nullptr) {
        std::setlocale(LC_NUMERIC, saved.c_str());
        GTEST_SKIP() << "no decimal-comma C locale is installed on this host";
    }
    EXPECT_EQ(parse("0.5"), 0.5) << chosen;
    EXPECT_EQ(detail::json_number(0.25), "0.25") << chosen;
    std::setlocale(LC_NUMERIC, saved.c_str());
}

// =============================================================================
// NoiseModel to_json / from_json
// =============================================================================

namespace {

// A model using every part the format carries: channels of one and two
// qubits, errors for all qubits and for chosen ones, before and after the
// gate, and readout errors.
NoiseModel full_model() {
    NoiseModel m;
    m.add_all_qubit_quantum_error(NoiseChannels::depolarizing(0.013), "h");
    m.add_all_qubit_quantum_error(NoiseChannels::amplitude_damping(0.07), "x", /*after_gate=*/false);
    m.add_quantum_error(NoiseChannels::depolarizing(0.021, 2), "cx", {0, 1});
    m.add_quantum_error(NoiseChannels::thermal_relaxation(50.0, 70.0, 0.3), "sx", {2});
    m.add_quantum_error(NoiseChannels::pauli(0.01, 0.02, 0.03), "rz", {1}, false);
    m.add_quantum_error(NoiseChannels::coherent_unitary(0.1, 0.2, 0.3), "rx", {0});
    m.add_quantum_error(NoiseChannels::reset(0.04, 0.05), "s", {2});
    m.add_readout_error(ReadoutError{0.011, 0.023}, 0);
    m.add_readout_error(ReadoutError{1.0 / 3.0, 0.0}, 2);
    return m;
}

QuantumCircuit noisy_circuit() {
    QuantumCircuit qc(3, 3);
    qc.h(0).x(1).cx(0, 1).sx(2).rz(0.4, 1).rx(0.7, 0).s(2).h(2);
    qc.measure_all();
    return qc;
}

}  // namespace

TEST(V11311NoiseModelJson, AModelRoundTripsExactly) {
    const NoiseModel original = full_model();
    const std::string text = original.to_json();
    const NoiseModel back = NoiseModel::from_json(text);
    EXPECT_EQ(back.to_json(), text) << "a second trip changed the text";

    // Every operator entry, bit for bit, and every readout figure.
    ASSERT_EQ(back.basis_gate_errors.size(), original.basis_gate_errors.size());
    for (const auto& [gate, errors] : original.basis_gate_errors) {
        SCOPED_TRACE(gate);
        ASSERT_EQ(back.basis_gate_errors.count(gate), 1u);
        const auto& got = back.basis_gate_errors.at(gate);
        ASSERT_EQ(got.size(), errors.size());
        for (std::size_t i = 0; i < errors.size(); ++i) {
            EXPECT_EQ(got[i].qubits, errors[i].qubits);
            EXPECT_EQ(got[i].after_gate, errors[i].after_gate);
            EXPECT_EQ(got[i].channel.n_qubits, errors[i].channel.n_qubits);
            ASSERT_EQ(got[i].channel.operators.size(), errors[i].channel.operators.size());
            for (std::size_t k = 0; k < errors[i].channel.operators.size(); ++k) {
                const auto& a = errors[i].channel.operators[k];
                const auto& b = got[i].channel.operators[k];
                ASSERT_EQ(a.size(), b.size());
                EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(Complex128)), 0) << k;
            }
        }
    }
    ASSERT_EQ(back.readout_errors.size(), original.readout_errors.size());
    for (const auto& [q, err] : original.readout_errors) {
        EXPECT_TRUE(same_bits(back.readout_errors.at(q).prob_meas_0_prep_1, err.prob_meas_0_prep_1));
        EXPECT_TRUE(same_bits(back.readout_errors.at(q).prob_meas_1_prep_0, err.prob_meas_1_prep_0));
    }
    EXPECT_EQ(back.noisy_gates, original.noisy_gates);
    EXPECT_EQ(back.is_ideal(), original.is_ideal());
}

TEST(V11311NoiseModelJson, ALoadedModelSimulatesExactlyAsTheOriginal) {
    const NoiseModel original = full_model();
    const NoiseModel back = NoiseModel::from_json(original.to_json());
    QuantumCircuit gates = noisy_circuit();
    gates.instructions.resize(gates.instructions.size() - 3);  // the gates alone
    const auto a = DensityMatrixSimulator().run(gates, original, 0, 9);
    const auto b = DensityMatrixSimulator().run(gates, back, 0, 9);
    ASSERT_EQ(a.final_state.data.size(), b.final_state.data.size());
    EXPECT_EQ(std::memcmp(a.final_state.data.data(), b.final_state.data.data(),
                          a.final_state.data.size() * sizeof(Complex128)),
              0);
    const auto ca = DensityMatrixSimulator().run(noisy_circuit(), original, 400, 9);
    const auto cb = DensityMatrixSimulator().run(noisy_circuit(), back, 400, 9);
    EXPECT_EQ(ca.counts, cb.counts);
}

TEST(V11311NoiseModelJson, AnEmptyModelRoundTripsAndStaysIdeal) {
    const NoiseModel empty;
    const NoiseModel back = NoiseModel::from_json(empty.to_json());
    EXPECT_TRUE(back.is_ideal());
    EXPECT_EQ(back.to_json(), empty.to_json());
}

TEST(V11311NoiseModelJson, AMalformedDocumentIsRefused) {
    const std::string good = full_model().to_json();
    const auto refused = [](const std::string& text, const std::string& what) {
        const auto e = v11311::thrown<InvalidArgument>([&] { (void)NoiseModel::from_json(text); });
        ASSERT_TRUE(e.has_value());
        EXPECT_EQ(e->entry_point(), "NoiseModel::from_json");
        EXPECT_EQ(std::string(e->what()), "NoiseModel::from_json: " + what);
    };
    std::string other = good;
    v11311::replace_value(other, "format", "\"lindblad.circuit\"");
    refused(other, "the document's format is \"lindblad.circuit\", not \"lindblad.noise_model\"");

    std::string later = good;
    v11311::replace_value(later, "version", "2");
    refused(later, "the document is version 2; this build reads version 1");

    std::string unversioned = good;
    unversioned.erase(unversioned.find(",\"version\":1"), std::strlen(",\"version\":1"));
    refused(unversioned, "the document is version none; this build reads version 1");

    // A channel's width outside what an operator's length can address, and
    // one whose operators do not match it.
    const int widest = (std::numeric_limits<std::size_t>::digits - 1) / 2;
    std::string wide = good;
    wide.replace(wide.find("\"n_qubits\":1"), std::strlen("\"n_qubits\":1"),
                 "\"n_qubits\":" + std::to_string(widest + 1));
    refused(wide, "a channel's n_qubits must be in [1, " + std::to_string(widest) + "], got " +
                      std::to_string(widest + 1));
    std::string zero = good;
    zero.replace(zero.find("\"n_qubits\":1"), std::strlen("\"n_qubits\":1"), "\"n_qubits\":0");
    refused(zero, "a channel's n_qubits must be in [1, " + std::to_string(widest) + "], got 0");
    std::string mismatch = good;
    mismatch.replace(mismatch.find("\"n_qubits\":1"), std::strlen("\"n_qubits\":1"), "\"n_qubits\":2");
    refused(mismatch, "operator 0 of a 2-qubit channel has 4 entries; it must have 16");
}

// =============================================================================
// The exporters under a decimal-comma locale
// =============================================================================

TEST(V11311JsonLocale, ACircuitWrittenUnderADecimalCommaReadsBackTheSame) {
    QuantumCircuit qc(1, 1);
    qc.rx(0.5, 0).measure(0, 0);
    const std::string classic = qc.to_json();
    std::string json;
    {
        const ScopedCommaLocale comma;
        json = qc.to_json();
    }
    EXPECT_EQ(json, classic) << "the text depends on the process's locale";
    const QuantumCircuit back = QuantumCircuit::from_json(json);
    ASSERT_FALSE(back.instructions.empty());
    EXPECT_EQ(back.instructions[0].params, std::vector<double>{0.5});
}

TEST(V11311JsonLocale, QasmExportedUnderADecimalCommaIsTheSameProgram) {
    QuantumCircuit qc(1, 1);
    qc.rx(0.5, 0).measure(0, 0);
    const std::string classic2 = qc.to_qasm2();
    const std::string classic3 = qc.to_qasm3();
    const ScopedCommaLocale comma;
    EXPECT_EQ(qc.to_qasm2(), classic2);
    EXPECT_EQ(qc.to_qasm3(), classic3);
}

TEST(V11311JsonLocale, AWideCircuitWrittenUnderGroupedThousandsReadsBackTheSame) {
    // 1235 qubits: every count and index of four digits is grouped.
    QuantumCircuit qc(1235, 1);
    qc.x(1234).measure(1234, 0);
    const std::string classic = qc.to_json();
    std::string json;
    {
        const ScopedCommaLocale comma;
        json = qc.to_json();
    }
    EXPECT_EQ(json, classic);
}

TEST(V11311JsonLocale, ANoiseModelWrittenUnderGroupedThousandsReadsBackTheSame) {
    NoiseModel m;
    m.add_quantum_error(NoiseChannels::depolarizing(0.01), "x", {1234});
    m.add_readout_error(ReadoutError{0.02, 0.03}, 1000);
    const std::string classic = m.to_json();
    std::string json;
    {
        const ScopedCommaLocale comma;
        json = m.to_json();
    }
    EXPECT_EQ(json, classic);
    // 1000 written as "1.000" would read back as qubit 1, silently.
    const NoiseModel back = NoiseModel::from_json(json);
    EXPECT_EQ(back.readout_errors.count(1000), 1u);
    EXPECT_EQ(back.readout_errors.count(1), 0u);
}

TEST(V11311JsonLocale, AFailedRunSavedUnderADecimalCommaLoadsBackTheSame) {
    // The folder's own numbers are written in the classic locale; its
    // circuit.json comes from the circuit's to_json.
    const v11311::TempDir dir("locale");
    QuantumCircuit qc(2, 2);
    qc.unitary(std::vector<Complex128>(4, Complex128(0.0, 0.0)), {0}, "zero", {Validation::Ignore});
    qc.rx(0.5, 1).measure(1, 1).x(1);
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
    const FailedRun loaded = load_failed_run(*record->saved_to);
    ASSERT_TRUE(loaded.circuit.has_value());
    EXPECT_EQ(loaded.circuit->to_json(), qc.to_json());
}
