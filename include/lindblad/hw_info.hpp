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
#include <vector>

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

// How many last-level (L3) cache instances the machine has, or 0 if unknown:
// distinct caches, not cores that share one. Two chiplets with an L3 each read
// 2. The product with llc_bytes() is the cache a team of threads spread over
// the whole machine can hold in total, which is what bounds the number of
// private working sets such a team keeps resident. Counted over every logical
// processor the operating system lists, so a process confined to part of the
// machine sees the machine's figure.
//
// Detected once, cached thread-safely; subsequent calls are a load.
std::size_t llc_instances();

// The logical processors of each last-level (L3) cache instance: one entry
// per instance, each a sorted list of processor numbers, the entries ordered
// by their lowest processor. On a Zen 4 part with two chiplets of six cores
// and SMT this is {0-5, 12-17} and {6-11, 18-23}. Empty if unknown.
//
// Processor numbers are the operating system's own: Linux's cpuN numbers (the
// numbers sched_getaffinity and OMP_PLACES use), Windows' group * 64 + bit,
// and on macOS, which reports no cache topology, 0 .. N-1 in one domain when
// an L3 is reported at all.
//
// Only a coherent reading counts. Every online processor must name its own L3
// and appear in it, and the instances must partition the processors the
// operating system lists; anything else (a processor without an L3 entry, two
// lists that overlap, a list that does not parse) makes the whole answer
// empty rather than a partial one.
//
// Counted over every logical processor the operating system lists, so a
// process confined to part of the machine sees the machine's figure; see
// threads_per_llc() for the confined one. Detected once, cached thread-safely;
// subsequent calls are a load.
const std::vector<std::vector<std::size_t>>& llc_domains();

// How many physical cores the machine has, or 0 if unknown: logical
// processors that share a core counted once. Counted over every logical
// processor the operating system lists, like llc_instances(). Linux reads
// the cores' sibling lists (core_cpus_list, else thread_siblings_list),
// Windows its RelationProcessorCore records, macOS hw.physicalcpu.
//
// Detected once, cached thread-safely; subsequent calls are a load.
std::size_t physical_cores();

// How many logical processors share each physical core (SMT width: 1 without
// SMT, 2 with it), or 0 if unknown. 0 also when the cores are not all the same
// width (a hybrid part with SMT on only some cores), since no single figure
// is true of that machine. Wherever it is non-zero, physical_cores() *
// smt_width() is the number of logical processors the operating system lists.
//
// Detected once, cached thread-safely; subsequent calls are a load.
std::size_t smt_width();

// How many logical processors one L3 instance offers the calling thread's
// OpenMP team, or 0 if llc_domains() is unknown: the largest number of
// processors the team may be placed on that sit under one L3. On Linux the
// team's reach is narrowed two ways. Under OpenMP binding (OMP_PROC_BIND or
// OMP_PLACES) it is the runtime's place list, since the runtime binds the
// calling thread itself to one place and its own affinity mask then says
// nothing about the team. Without binding it is the calling thread's affinity
// mask (taskset, a cpuset, a batch job's binding). A process pinned to six of
// the twelve processors of one chiplet reads 6, not 12. Where neither can be
// read, and on Windows and macOS, no restriction is assumed and the figure is
// the size of the largest instance. Logical processors, not cores: divide by
// smt_width() for cores when the team may use both siblings. Read afresh on
// every call, since the affinity and the places can change.
std::size_t threads_per_llc();

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

// How many threads a parallel region the calling thread starts now may use:
// OpenMP's current limit for that thread, which OMP_NUM_THREADS, the
// caller's own omp_set_num_threads and a simulator's max_parallel_threads
// (detail::ScopedThreadCap) all set. 1 in a build without OpenMP. Read afresh
// on every call, since a cap applies only for the length of the call that
// set it.
std::size_t available_threads();

// The operating system's memory page size in bytes, or 0 if unknown
// (sysconf(_SC_PAGESIZE) on Linux and macOS, GetSystemInfo on Windows).
// Detected once, cached thread-safely.
std::size_t page_bytes();

}  // namespace hw
}  // namespace lindblad
