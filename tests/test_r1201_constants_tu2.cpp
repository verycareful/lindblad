// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// R.1.20.1 — second translation unit for the constants suite.
//
// Exists solely so test_r1201_constants.cpp can assert that `inline constexpr`
// in constants.hpp yields ONE shared object rather than one per including
// translation unit. That property is not observable from inside a single TU: a
// non-inline namespace-scope `constexpr` has internal linkage and would give
// this file its own PI at its own address, and every value comparison would
// still pass. Only the addresses differ, so only a second TU can see it.
//
// Deliberately minimal — no gtest, no assertions here. It hands back addresses
// and values and nothing else; the assertions live with the rest of the suite.

#include "lindblad/constants.hpp"

namespace r1201_tu2 {

const double* pi_address()        { return &lindblad::PI; }
const double* inv_sqrt2_address() { return &lindblad::INV_SQRT2; }

double pi_value()        { return lindblad::PI; }
double inv_sqrt2_value() { return lindblad::INV_SQRT2; }

} // namespace r1201_tu2
