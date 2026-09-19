// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#pragma once

namespace lindblad {

enum class QuditBackend {
    STATEVECTOR,     // QuditStatevector (default; exact, any d)
    DENSITY_MATRIX,  // QuditDensityMatrix (mixed states + Kraus/Lindblad noise, any d)
    MPS,             // QuditMPS (low-entanglement tensor network, any d)
    CLIFFORD,        // QuditCliffordSimulator (prime d only; Clifford circuits only)
};

} // namespace lindblad
