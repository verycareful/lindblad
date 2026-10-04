// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.31.3 - a failed run names its exception type the same on every
// platform.
//
// On the MSVC ABI, type_info::name() is readable but carries the class key:
// the Windows CI leg recorded FailedRun::exception_type as
// "class lindblad::InvalidArgument" where Linux records
// "lindblad::InvalidArgument", and nine tests failed on it. The saved folder
// records the field and loads on any platform, so it carries one spelling.
// detail::without_class_keys removes the keys; it is compiled everywhere, so
// these tests run on every platform, not only where the MSVC ABI produces
// such names.

#include <gtest/gtest.h>

#include "lindblad/detail/failure_collector.hpp"

#include <string>
#include <utility>
#include <vector>

using namespace lindblad;

TEST(V11313TypeNames, TheMsvcClassKeysAreRemoved) {
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"class lindblad::InvalidArgument", "lindblad::InvalidArgument"},
        {"class std::runtime_error", "std::runtime_error"},
        {"struct CallerError", "CallerError"},
        {"enum lindblad::Response", "lindblad::Response"},
        {"union U", "U"},
        {"class std::basic_string<char,struct std::char_traits<char>,class std::allocator<char> >",
         "std::basic_string<char,std::char_traits<char>,std::allocator<char> >"},
        {"struct std::pair<class A,struct B>", "std::pair<A,B>"},
    };
    for (const auto& [msvc, expected] : cases) {
        SCOPED_TRACE(msvc);
        EXPECT_EQ(detail::without_class_keys(msvc), expected);
    }
}

TEST(V11313TypeNames, ANameWithoutAClassKeyIsLeftAsItIs) {
    // The Itanium ABI's demangled names, and identifiers that merely contain a
    // key word, pass through unchanged.
    for (const std::string name : {"lindblad::RuntimeFailure", "std::runtime_error",
                                   "subclass", "my_struct::unionfind", "classy::enum_like",
                                   "std::vector<int, std::allocator<int> >", ""}) {
        SCOPED_TRACE(name);
        EXPECT_EQ(detail::without_class_keys(name), name);
    }
}
