// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#include "lindblad/noise.hpp"
#include "lindblad/detail/json.hpp"
#include "lindblad/detail/report.hpp"
#include "lindblad/detail/text.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lindblad {

// =============================================================================
// ReadoutError
// =============================================================================

std::array<std::array<double, 2>, 2> ReadoutError::assignment_matrix() const {
    return {{
        {{1.0 - prob_meas_1_prep_0, prob_meas_0_prep_1}},
        {{prob_meas_1_prep_0, 1.0 - prob_meas_0_prep_1}}
    }};
}

// =============================================================================
// NoiseModel
// =============================================================================

void NoiseModel::add_quantum_error(
    const KrausChannel& error,
    const std::string& gate_name,
    const std::vector<int>& qubits,
    bool after_gate
) {
    // Naming the qubits fixes the width here, so the mismatch is caught at the
    // line that is wrong rather than at the run. An empty list means "whichever
    // qubits the gate acts on", whose width is not known until an instruction
    // is in hand, so that case is checked where the channel is resolved.
    if (!qubits.empty() &&
        error.n_qubits != static_cast<int>(qubits.size())) {
        throw std::invalid_argument(
            "NoiseModel::add_quantum_error: the channel acts on " +
            std::to_string(error.n_qubits) + " qubit(s) but is attached to " +
            std::to_string(qubits.size()) + " on gate '" + gate_name +
            "'. A channel and the qubits it is applied to have to agree.");
    }

    GateError ge;
    ge.channel = error;
    ge.qubits = qubits;
    ge.after_gate = after_gate;
    basis_gate_errors[gate_name].push_back(ge);

    if (std::find(noisy_gates.begin(), noisy_gates.end(), gate_name) == noisy_gates.end()) {
        noisy_gates.push_back(gate_name);
    }
}

void NoiseModel::add_readout_error(const ReadoutError& error, int qubit) {
    readout_errors[qubit] = error;
}

void NoiseModel::add_all_qubit_quantum_error(
    const KrausChannel& error,
    const std::string& gate_name,
    bool after_gate
) {
    add_quantum_error(error, gate_name, {}, after_gate);
}

std::vector<NoiseModel::GateError> NoiseModel::errors_for_gate(
    const std::string& gate_name,
    const std::vector<int>& qubits
) const {
    std::vector<GateError> result;

    auto it = basis_gate_errors.find(gate_name);
    if (it == basis_gate_errors.end()) return result;

    for (const auto& ge : it->second) {
        if (ge.qubits.empty()) {
            // Applies to all qubits
            result.push_back(ge);
        } else if (ge.qubits == qubits) {
            result.push_back(ge);
        }
    }

    return result;
}

const std::vector<NoiseModel::GateError>* NoiseModel::errors_for_gate_ref(
    const std::string& gate_name
) const {
    auto it = basis_gate_errors.find(gate_name);
    return it == basis_gate_errors.end() ? nullptr : &it->second;
}

bool NoiseModel::is_ideal() const {
    return basis_gate_errors.empty() && readout_errors.empty();
}

NoiseModel NoiseModel::from_t1_t2(
    const std::vector<double>& t1,
    const std::vector<double>& t2,
    const std::unordered_map<std::string, double>& gate_times,
    const std::unordered_map<std::string, std::vector<int>>& gate_qubits
) {
    int n = static_cast<int>(t1.size());
    if (static_cast<int>(t2.size()) != n) {
        throw std::invalid_argument("t1 and t2 must have the same length");
    }
    for (int q = 0; q < n; ++q) {
        if (t1[q] <= 0.0 || t2[q] <= 0.0) {
            throw std::invalid_argument("T1 and T2 must be positive");
        }
        if (t2[q] > 2.0 * t1[q]) {
            throw std::invalid_argument("T2 must be <= 2*T1 for qubit " + std::to_string(q));
        }
    }

    NoiseModel model;

    for (const auto& [gate_name, gate_time] : gate_times) {
        // Determine which qubits to apply this gate's noise to
        std::vector<int> qubits_for_gate;
        auto it = gate_qubits.find(gate_name);
        if (it != gate_qubits.end() && !it->second.empty()) {
            qubits_for_gate = it->second;
        } else {
            // Apply to all qubits
            qubits_for_gate.resize(n);
            for (int q = 0; q < n; ++q) qubits_for_gate[q] = q;
        }

        for (int q : qubits_for_gate) {
            if (q < 0 || q >= n) continue;
            auto channel = NoiseChannels::thermal_relaxation(t1[q], t2[q], gate_time);
            model.add_quantum_error(channel, gate_name, {q});
        }
    }

    return model;
}

// =============================================================================
// NoiseModel::to_json / from_json - a lossless round trip
// =============================================================================
// Format, version 1:
//
//   {"format":"lindblad.noise_model","version":1,
//    "basis_gate_errors":[{"gate":"cx","errors":[{"qubits":[0,1],
//        "after_gate":true,"channel":{"n_qubits":2,
//        "operators":[[[re,im],[re,im],...],...]}}]}],
//    "readout_errors":[{"qubit":0,"prob_meas_0_prep_1":0.01,
//        "prob_meas_1_prep_0":0.02}],
//    "noisy_gates":["cx"]}
//
// An operator is its (2^n)^2 entries in storage order, each [real, imag].

namespace {

constexpr const char* NOISE_MODEL_FORMAT = "lindblad.noise_model";
constexpr int NOISE_MODEL_VERSION = 1;

// The widest channel whose operator length, (2^n)^2, a size_t holds: 2n has to
// stay below its width.
constexpr int MAX_CHANNEL_QUBITS = (std::numeric_limits<std::size_t>::digits - 1) / 2;

void write_channel(detail::TextBuilder& o, const KrausChannel& channel) {
    o << "{\"n_qubits\":" << detail::integer_text(channel.n_qubits) << ",\"operators\":[";
    for (std::size_t k = 0; k < channel.operators.size(); ++k) {
        if (k > 0) o << ',';
        o << '[';
        const std::vector<Complex128>& op = channel.operators[k];
        for (std::size_t e = 0; e < op.size(); ++e) {
            if (e > 0) o << ',';
            o << '[' << detail::json_number(op[e].real) << ','
              << detail::json_number(op[e].imag) << ']';
        }
        o << ']';
    }
    o << "]}";
}

KrausChannel read_channel(detail::JsonReader& r) {
    KrausChannel channel;
    bool have_width = false;
    r.expect('{');
    while (r.peek() != '}') {
        if (r.peek() == ',') r.next();
        const std::string key = r.read_string();
        r.expect(':');
        if (key == "n_qubits") {
            channel.n_qubits = r.read_int();
            have_width = true;
        } else if (key == "operators") {
            r.expect('[');
            while (r.peek() != ']') {
                if (r.peek() == ',') r.next();
                std::vector<Complex128> op;
                r.expect('[');
                while (r.peek() != ']') {
                    if (r.peek() == ',') r.next();
                    r.expect('[');
                    const double re = r.read_double();
                    r.expect(',');
                    const double im = r.read_double();
                    r.expect(']');
                    op.emplace_back(re, im);
                }
                r.expect(']');
                channel.operators.push_back(std::move(op));
            }
            r.expect(']');
        } else {
            r.skip_value();
        }
    }
    r.expect('}');

    if (!have_width || channel.n_qubits < 1 || channel.n_qubits > MAX_CHANNEL_QUBITS) {
        detail::raise<InvalidArgument>("NoiseModel::from_json",
            "a channel's n_qubits must be in [1, " + std::to_string(MAX_CHANNEL_QUBITS) +
                "], got " + (have_width ? std::to_string(channel.n_qubits) : "none"));
    }
    const std::size_t side = std::size_t{1} << channel.n_qubits;
    for (std::size_t k = 0; k < channel.operators.size(); ++k) {
        if (channel.operators[k].size() != side * side) {
            detail::raise<InvalidArgument>("NoiseModel::from_json",
                "operator " + std::to_string(k) + " of a " +
                    std::to_string(channel.n_qubits) + "-qubit channel has " +
                    std::to_string(channel.operators[k].size()) + " entries; it must have " +
                    std::to_string(side * side));
        }
    }
    return channel;
}

std::vector<int> read_int_array(detail::JsonReader& r) {
    std::vector<int> out;
    r.expect('[');
    while (r.peek() != ']') {
        if (r.peek() == ',') r.next();
        out.push_back(r.read_int());
    }
    r.expect(']');
    return out;
}

}  // namespace

std::string NoiseModel::to_json() const {
    detail::TextBuilder o;
    o << "{\"format\":" << detail::json_escape(NOISE_MODEL_FORMAT)
      << ",\"version\":" << detail::integer_text(NOISE_MODEL_VERSION) << ",\"basis_gate_errors\":[";

    std::vector<std::string> gates;
    gates.reserve(basis_gate_errors.size());
    for (const auto& entry : basis_gate_errors) gates.push_back(entry.first);
    std::sort(gates.begin(), gates.end());
    for (std::size_t g = 0; g < gates.size(); ++g) {
        if (g > 0) o << ',';
        o << "{\"gate\":" << detail::json_escape(gates[g]) << ",\"errors\":[";
        const std::vector<GateError>& errors = basis_gate_errors.at(gates[g]);
        for (std::size_t e = 0; e < errors.size(); ++e) {
            if (e > 0) o << ',';
            o << "{\"qubits\":[";
            for (std::size_t q = 0; q < errors[e].qubits.size(); ++q) {
                if (q > 0) o << ',';
                o << detail::integer_text(errors[e].qubits[q]);
            }
            o << "],\"after_gate\":" << (errors[e].after_gate ? "true" : "false")
              << ",\"channel\":";
            write_channel(o, errors[e].channel);
            o << '}';
        }
        o << "]}";
    }

    o << "],\"readout_errors\":[";
    std::vector<int> qubits;
    qubits.reserve(readout_errors.size());
    for (const auto& entry : readout_errors) qubits.push_back(entry.first);
    std::sort(qubits.begin(), qubits.end());
    for (std::size_t i = 0; i < qubits.size(); ++i) {
        if (i > 0) o << ',';
        const ReadoutError& err = readout_errors.at(qubits[i]);
        o << "{\"qubit\":" << detail::integer_text(qubits[i])
          << ",\"prob_meas_0_prep_1\":" << detail::json_number(err.prob_meas_0_prep_1)
          << ",\"prob_meas_1_prep_0\":" << detail::json_number(err.prob_meas_1_prep_0) << '}';
    }

    o << "],\"noisy_gates\":[";
    for (std::size_t i = 0; i < noisy_gates.size(); ++i) {
        if (i > 0) o << ',';
        o << detail::json_escape(noisy_gates[i]);
    }
    o << "]}";
    return std::move(o).str();
}

NoiseModel NoiseModel::from_json(const std::string& json) {
    detail::JsonReader r{json};
    NoiseModel model;
    std::string format;
    int version = 0;
    bool have_version = false;

    r.expect('{');
    while (r.peek() != '}') {
        if (r.peek() == ',') r.next();
        const std::string key = r.read_string();
        r.expect(':');
        if (key == "format") {
            format = r.read_string();
        } else if (key == "version") {
            version = r.read_int();
            have_version = true;
        } else if (key == "basis_gate_errors") {
            r.expect('[');
            while (r.peek() != ']') {
                if (r.peek() == ',') r.next();
                std::string gate;
                std::vector<GateError> errors;
                r.expect('{');
                while (r.peek() != '}') {
                    if (r.peek() == ',') r.next();
                    const std::string gkey = r.read_string();
                    r.expect(':');
                    if (gkey == "gate") {
                        gate = r.read_string();
                    } else if (gkey == "errors") {
                        r.expect('[');
                        while (r.peek() != ']') {
                            if (r.peek() == ',') r.next();
                            GateError ge;
                            r.expect('{');
                            while (r.peek() != '}') {
                                if (r.peek() == ',') r.next();
                                const std::string ekey = r.read_string();
                                r.expect(':');
                                if (ekey == "qubits") {
                                    ge.qubits = read_int_array(r);
                                } else if (ekey == "after_gate") {
                                    ge.after_gate = r.read_bool();
                                } else if (ekey == "channel") {
                                    ge.channel = read_channel(r);
                                } else {
                                    r.skip_value();
                                }
                            }
                            r.expect('}');
                            errors.push_back(std::move(ge));
                        }
                        r.expect(']');
                    } else {
                        r.skip_value();
                    }
                }
                r.expect('}');
                model.basis_gate_errors[gate] = std::move(errors);
            }
            r.expect(']');
        } else if (key == "readout_errors") {
            r.expect('[');
            while (r.peek() != ']') {
                if (r.peek() == ',') r.next();
                int qubit = 0;
                ReadoutError err{0.0, 0.0};
                r.expect('{');
                while (r.peek() != '}') {
                    if (r.peek() == ',') r.next();
                    const std::string rkey = r.read_string();
                    r.expect(':');
                    if (rkey == "qubit") {
                        qubit = r.read_int();
                    } else if (rkey == "prob_meas_0_prep_1") {
                        err.prob_meas_0_prep_1 = r.read_double();
                    } else if (rkey == "prob_meas_1_prep_0") {
                        err.prob_meas_1_prep_0 = r.read_double();
                    } else {
                        r.skip_value();
                    }
                }
                r.expect('}');
                model.readout_errors[qubit] = err;
            }
            r.expect(']');
        } else if (key == "noisy_gates") {
            r.expect('[');
            while (r.peek() != ']') {
                if (r.peek() == ',') r.next();
                model.noisy_gates.push_back(r.read_string());
            }
            r.expect(']');
        } else {
            r.skip_value();
        }
    }
    r.expect('}');

    if (format != NOISE_MODEL_FORMAT) {
        detail::raise<InvalidArgument>("NoiseModel::from_json",
            "the document's format is \"" + format + "\", not \"" +
                std::string(NOISE_MODEL_FORMAT) + "\"");
    }
    if (!have_version || version != NOISE_MODEL_VERSION) {
        detail::raise<InvalidArgument>("NoiseModel::from_json",
            "the document is version " + (have_version ? std::to_string(version) : "none") +
                "; this build reads version " + std::to_string(NOISE_MODEL_VERSION));
    }
    return model;
}

} // namespace lindblad
