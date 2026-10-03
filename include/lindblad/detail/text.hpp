// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#pragma once

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <clocale>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <ios>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>

namespace lindblad {
namespace detail {

// =============================================================================
// text - numbers in text that no locale can change
// =============================================================================
// Every number the library writes into text (JSON, OpenQASM, SVG, LaTeX, a
// drawing's labels, a folder name, a cache key) goes through these writers. A
// C++ stream follows the process's global locale unless told otherwise, and a
// desktop application set to a German or French locale makes every such
// stream write 0.5 as "0,5" and 1234 as "1.234": a QASM file that no parser
// accepts, or a JSON qubit index that reads back as qubit 1. printf and
// std::to_string follow the C locale the same way for a double, and a named
// global C++ locale sets that too.
//
// TextBuilder below makes the rule mechanical: it has no way to write a
// number at all, so one cannot reach the text except through these.

// An integer in decimal (or `base`), as digits with a leading minus when
// negative. std::to_chars is locale-independent by specification.
template <std::integral Int>
    requires(!std::same_as<Int, bool> && !std::same_as<Int, char>)
std::string integer_text(Int v, int base = 10) {
    // digits in base 2 plus a sign is the longest any base can need
    char buf[std::numeric_limits<Int>::digits + 2];
    const auto [end, ec] = std::to_chars(buf, buf + sizeof buf, v, base);
    (void)ec;  // the buffer holds every value of Int in every base
    return std::string(buf, end);
}

// A double as `%.<significant_digits>g` prints it in the C locale: the
// shortest of fixed and scientific notation, trailing zeros dropped, and
// "nan", "inf" or "-inf" for the non-finite values. The stream is imbued with
// the classic locale, so the decimal point is '.' and nothing is grouped.
inline std::string double_text(double v, int significant_digits) {
    std::ostringstream o;
    o.imbue(std::locale::classic());
    o << std::setprecision(significant_digits) << v;
    return o.str();
}

// A double as `%.<decimals>f` prints it in the C locale: fixed notation with
// exactly `decimals` digits after the point, and "nan", "inf" or "-inf" for
// the non-finite values.
inline std::string fixed_text(double v, int decimals) {
    std::ostringstream o;
    o.imbue(std::locale::classic());
    o << std::fixed << std::setprecision(decimals) << v;
    return o.str();
}

// A double read from the front of `text` the way std::stod reads one in the
// C locale, whatever the process's: leading white space skipped, an optional
// '+' or '-', then the longest number that follows, so "0.5" is one half under
// a decimal-comma locale and "2 * pi" reads as 2. `*used`, when given,
// receives the number of characters consumed, white space included. Throws
// std::invalid_argument when no number starts the text and std::out_of_range
// when it lies outside a double's range (1e999, or 1e-400 below the smallest
// subnormal), as std::stod does, so a caller that tells the two apart keeps
// working.
//
// Two forms read differently from std::stod. A subnormal value reads as
// itself, where std::stod may call it out of range. And where the standard
// library has a floating-point std::from_chars, a hexadecimal form ("0x10")
// reads as its leading 0. OpenQASM writes neither.
inline double parse_double(std::string_view text, std::size_t* used = nullptr) {
    const auto is_space = [](char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r';
    };
    const auto shown = [&] { return std::string(text); };
    double value = 0.0;
#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
    std::size_t number = 0;
    while (number < text.size() && is_space(text[number])) ++number;
    // from_chars takes a leading '-' and no '+'; std::stod takes either, once.
    if (number < text.size() && text[number] == '+' &&
        !(number + 1 < text.size() && (text[number + 1] == '+' || text[number + 1] == '-'))) {
        ++number;
    }
    const auto [stop, ec] =
        std::from_chars(text.data() + number, text.data() + text.size(), value);
    if (ec == std::errc::invalid_argument) {
        throw std::invalid_argument("parse_double: no number at the start of '" + shown() + "'");
    }
    if (ec == std::errc::result_out_of_range) {
        throw std::out_of_range("parse_double: '" + shown() + "' is outside the range of a double");
    }
    const std::size_t end = static_cast<std::size_t>(stop - text.data());
#else
    // std::strtod reads the C locale's decimal point. Hand it the text with
    // every '.' spelled that way, cut at the first character that already
    // was one (where a classic reader stops anyway), and it reads exactly what
    // it would in the C locale.
    (void)is_space;
    std::string local(text);
    const char* const point = std::localeconv()->decimal_point;
    if (point != nullptr && point[0] != '\0' && point[0] != '.' && point[1] == '\0') {
        const std::size_t cut = local.find(point[0]);
        if (cut != std::string::npos) local.resize(cut);
        std::replace(local.begin(), local.end(), '.', point[0]);
    }
    char* stop = nullptr;
    errno = 0;
    value = std::strtod(local.c_str(), &stop);
    if (stop == local.c_str()) {
        throw std::invalid_argument("parse_double: no number at the start of '" + shown() + "'");
    }
    // ERANGE also marks a subnormal result, which reads as itself here as it
    // does through from_chars; only an infinity or a flush to zero is out of
    // range. The exponent bits decide infinity, since the build's fast-math
    // flags let a comparison with one fold away.
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof bits);
    const bool infinite = ((bits >> 52) & 0x7FFu) == 0x7FFu;
    if (errno == ERANGE && (infinite || value == 0.0)) {
        throw std::out_of_range("parse_double: '" + shown() + "' is outside the range of a double");
    }
    const std::size_t end = static_cast<std::size_t>(stop - local.c_str());
#endif
    if (used != nullptr) *used = end;
    return value;
}

// =============================================================================
// TextBuilder - an output buffer that refuses numbers
// =============================================================================
// Accepts text and single characters. Writing any arithmetic value is a
// compile error, so every number in the output is spelled by integer_text or
// double_text, and a site that forgets cannot build.
class TextBuilder {
public:
    TextBuilder& operator<<(std::string_view s) {
        text_.append(s);
        return *this;
    }
    TextBuilder& operator<<(char c) {
        text_ += c;
        return *this;
    }
    template <class T>
        requires(std::is_arithmetic_v<T> && !std::same_as<T, char>)
    TextBuilder& operator<<(T) = delete;

    const std::string& str() const& { return text_; }
    std::string str() && { return std::move(text_); }
    bool empty() const { return text_.empty(); }

private:
    std::string text_;
};

}  // namespace detail
}  // namespace lindblad
