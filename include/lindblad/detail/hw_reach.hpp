// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#pragma once

#include <vector>

namespace lindblad {
namespace detail {

// The logical processors an OpenMP team started by the calling thread may be
// placed on, indexed by processor number: the OpenMP place list under
// OMP_PROC_BIND, else the calling thread's affinity mask. False, with
// `allowed` empty, where nothing narrows the team (no restriction is read, or
// the platform is not Linux). hw::threads_per_llc() counts these per L3; the
// threading model keys its cache by them, since a measurement made on one set
// of processors says nothing about another. Internal: not part of the public
// hw_info surface.
bool team_processors(std::vector<bool>& allowed);

}  // namespace detail
}  // namespace lindblad
