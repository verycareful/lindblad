// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#pragma once

#include <cstddef>

namespace lindblad {
namespace hw {

// =============================================================================
// hw_info — best-effort runtime hardware property detection
// =============================================================================
// Small, dependency-free queries for hardware properties that drive runtime
// performance-tuning decisions (e.g. the statevector gate-fusion engagement
// point scales with the last-level cache). Detection is best-effort by
// design: every query has an explicit "unknown" return value so the caller
// chooses its own documented fallback -- nothing here guesses silently.

// Size in bytes of ONE last-level (L3) cache instance, or 0 if unknown.
//
// Per-instance is deliberate. On multi-CCD parts (e.g. Zen 4: two CCDs with
// 32 MiB of L3 each) a thread's working set effectively lives in its own
// CCD's slice -- cross-CCD L3 traffic costs DRAM-like latency -- so the
// per-instance size, not the package total, is the right yardstick for
// "does this working set still fit in cache". It is also what the OS
// reports for cpu 0, which keeps detection trivial and uniform.
//
// Detected once, cached thread-safely; subsequent calls are a load.
std::size_t llc_bytes();

}  // namespace hw
}  // namespace lindblad
