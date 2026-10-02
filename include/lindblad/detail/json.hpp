// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#pragma once

#include "lindblad/types.hpp"

#include <cctype>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace lindblad {
namespace detail {

// =============================================================================
// json - the zero-dependency JSON pieces every serializer shares
// =============================================================================
// QuantumCircuit::to_json / from_json, NoiseModel::to_json / from_json and the
// failed-run files all read and write through these, so a string, a double
// and a malformed document are handled the same way everywhere.

// A string as a JSON string literal, quotes included.
inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    out += '"';
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\t': out += "\\t";  break;
            case '\r': out += "\\r";  break;
            default:   out += c;      break;
        }
    }
    out += '"';
    return out;
}

// A double as JSON, exactly: 17 significant digits in the classic locale, so
// reading it back yields the same double. JSON has no NaN or infinity, so
// those are written as the strings "NaN", "Infinity" and "-Infinity", which
// JsonReader::read_double reads back. The bit pattern decides which, since
// the build's fast-math flags make no promise about comparisons with NaN.
inline std::string json_number(double v) {
    if (!is_finite_strict(v)) {
        std::uint64_t bits;
        std::memcpy(&bits, &v, sizeof bits);
        if ((bits & 0x000FFFFFFFFFFFFFULL) != 0) return "\"NaN\"";
        return (bits >> 63) != 0 ? "\"-Infinity\"" : "\"Infinity\"";
    }
    std::ostringstream o;
    o.imbue(std::locale::classic());
    o << std::setprecision(17) << v;
    return o.str();
}

// =============================================================================
// JsonReader - a minimal hand-rolled reader over one document
// =============================================================================
// A cursor over `s`. Every malformed token throws std::runtime_error ("JSON
// parse error: ..."); a malformed number throws what std::stod throws.
struct JsonReader {
    const std::string& s;
    size_t pos = 0;

    void skip_ws() {
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r'))
            ++pos;
    }

    char peek() { skip_ws(); return (pos < s.size()) ? s[pos] : '\0'; }
    char next() { skip_ws(); return (pos < s.size()) ? s[pos++] : '\0'; }

    void expect(char c) {
        char got = next();
        if (got != c)
            throw std::runtime_error(std::string("JSON parse error: expected '") + c + "', got '" + got + "'");
    }

    std::string read_string() {
        expect('"');
        std::string out;
        while (pos < s.size() && s[pos] != '"') {
            if (s[pos] == '\\' && pos + 1 < s.size()) {
                ++pos;
                switch (s[pos]) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    default: out += s[pos]; break;
                }
            } else {
                out += s[pos];
            }
            ++pos;
        }
        if (pos < s.size()) ++pos;  // skip closing '"'
        return out;
    }

    double read_number() {
        skip_ws();
        size_t start = pos;
        if (pos < s.size() && s[pos] == '-') ++pos;
        while (pos < s.size() && (std::isdigit(s[pos]) || s[pos] == '.' || s[pos] == 'e' || s[pos] == 'E' || s[pos] == '+' || s[pos] == '-')) {
            if ((s[pos] == '+' || s[pos] == '-') && pos > start + 1 && s[pos-1] != 'e' && s[pos-1] != 'E') break;
            ++pos;
        }
        return parse_number(s.substr(start, pos - start));
    }

    // A number in the C locale whatever the process's, so 0.5 reads as 0.5
    // under a locale whose decimal separator is a comma, and a subnormal reads
    // as itself, as json_number writes both. A token that is not wholly a
    // number, or one outside double's range, is refused.
    static double parse_number(const std::string& token) {
        double value = 0.0;
#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
        const char* first = token.data();
        const char* last = first + token.size();
        const auto [ptr, ec] = std::from_chars(first, last, value);
        if (ec == std::errc() && ptr == last) return value;
#else
        std::istringstream in(token);
        in.imbue(std::locale::classic());
        in >> value;
        if (!in.fail() && in.peek() == std::char_traits<char>::eof()) return value;
#endif
        throw std::runtime_error("JSON parse error: '" + token + "' is not a number");
    }

    // A whole number an int can hold. A fraction is refused rather than
    // truncated, since a field that holds a count or an index has no meaning
    // for 1.5, and converting a double outside int's range is undefined (NaN
    // fails both range comparisons).
    int read_int() {
        const double d = read_number();
        if (!(d >= static_cast<double>(std::numeric_limits<int>::min()) &&
              d <= static_cast<double>(std::numeric_limits<int>::max()))) {
            throw std::runtime_error("JSON parse error: " + std::to_string(d) +
                                     " is outside the range of an int");
        }
        const int value = static_cast<int>(d);
        if (static_cast<double>(value) != d) {
            throw std::runtime_error("JSON parse error: " + std::to_string(d) +
                                     " is not a whole number");
        }
        return value;
    }

    // A double written by json_number: a number, or one of the strings NaN,
    // Infinity and -Infinity.
    double read_double() {
        if (peek() != '"') return read_number();
        const std::string word = read_string();
        if (word == "NaN") return quiet_nan_strict();
        if (word == "Infinity") return std::numeric_limits<double>::infinity();
        if (word == "-Infinity") return -std::numeric_limits<double>::infinity();
        throw std::runtime_error("JSON parse error: expected a number, got \"" + word + "\"");
    }

    // A JSON literal true or false.
    bool read_bool() {
        skip_ws();
        if (s.compare(pos, 4, "true") == 0) { pos += 4; return true; }
        if (s.compare(pos, 5, "false") == 0) { pos += 5; return false; }
        throw std::runtime_error("JSON parse error: expected true or false");
    }

    // Skip a JSON value we don't care about
    void skip_value() {
        skip_ws();
        if (s[pos] == '"') { read_string(); return; }
        if (s[pos] == '[') {
            ++pos;
            if (peek() != ']') {
                skip_value();
                while (peek() == ',') { ++pos; skip_value(); }
            }
            expect(']');
            return;
        }
        if (s[pos] == '{') {
            ++pos;
            if (peek() != '}') {
                read_string(); expect(':'); skip_value();
                while (peek() == ',') { ++pos; read_string(); expect(':'); skip_value(); }
            }
            expect('}');
            return;
        }
        // number / true / false / null
        while (pos < s.size() && s[pos] != ',' && s[pos] != '}' && s[pos] != ']')
            ++pos;
    }
};

}  // namespace detail
}  // namespace lindblad
