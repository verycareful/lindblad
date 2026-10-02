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

// Memory the operating system reports it could hand out now without
// swapping, in bytes, or 0 if unknown: Linux's MemAvailable (/proc/meminfo),
// Windows' available physical memory, macOS's free and inactive pages. Read
// afresh on every call, since it changes as the process and the machine
// allocate; a caller deciding whether a buffer fits asks at the moment it
// would allocate.
//
// On Linux only a coherent reading counts: MemAvailable parsed as a number,
// non-zero, convertible to bytes without overflow, and no larger than
// MemTotal. Anything else is 0.
//
// On Linux the figure also answers to the memory limits of the cgroups the
// process runs in (a container, a systemd slice, a batch job), which the
// kernel enforces by ending the process: it is the smaller of MemAvailable
// and the room left under the tightest limit from the process's own cgroup
// up to the root, where the room is the limit less usage, not counting
// inactive file cache. A cgroup at its limit reads as 1 byte, not 0, since
// nothing more fits there.
std::size_t available_memory_bytes();

// The same figure, read at most once a second and shared between calls in
// between. The memory cap every run answers to is taken from this: a run
// asks once, before it starts, and reading /proc/meminfo on every call would
// cost more than a small run does. The cap is a snapshot either way, since
// another process can allocate the moment after any reading.
std::size_t recent_available_memory_bytes();

}  // namespace hw
}  // namespace lindblad
