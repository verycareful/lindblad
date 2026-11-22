# Hardware Info

Best-effort runtime detection of hardware properties that change how the library
tunes itself: the last-level cache size, which the statevector simulator asks
to decide whether to fuse gates; the memory the operating system reports
available, which sets the memory cap a run answers to when `max_memory_mb` is 0
and decides whether a per-shot run keeps its prefix snapshot; the threads a
parallel region may use, which sizes every statevector pass's team; the
processor topology (which logical processors share an L3, how many physical
cores there are, how many processors share a core), which says how many
threads one cache instance can feed; and the memory page size, which a
statevector's amplitude arrays are aligned to.

The whole module is built on one rule: **nothing here guesses silently.** Every
query has an explicit "unknown" return, and the caller picks its own documented
fallback. A detector that invented a plausible number would hand the caller a
tuning decision made on fiction, and the caller would have no way to tell.

## Header

```cpp
#include "lindblad/hw_info.hpp"
```

## Namespace

`lindblad::hw`

The nested namespace is deliberate: these are machine properties, not quantum
ones, and they do not belong in the same name space as gates and states.

## Functions

### `llc_bytes`

```cpp
std::size_t llc_bytes();
```

Size in bytes of **one** last-level (L3) cache instance. Returns `0` when the
size cannot be determined.

Per-instance, not the package total, and that distinction is the point of the
function. On a multi-CCD part such as Zen 4, two chiplets carry 32 MiB of L3
each. A thread's working set effectively lives in its own chiplet's slice,
because crossing to the other one costs DRAM-like latency. So the question "does
this working set still fit in cache" is answered by one instance, never by the
sum. It is also what the operating system reports for cpu 0, which keeps
detection uniform across platforms.

Detected once and cached thread-safely through a function-local static, so every
call after the first is a load.

### `llc_instances`

```cpp
std::size_t llc_instances();
```

How many last-level (L3) cache instances the machine has: distinct caches, not
the cores that share one. Two chiplets with an L3 each read `2`. Returns `0`
when the count cannot be determined.

The product with `llc_bytes()` is the cache that threads spread over the whole
machine hold in total, which bounds how many private working sets such a team
keeps resident; one instance alone (`llc_bytes()`) remains the yardstick for
whether a single working set fits.

- **Linux**: the distinct `shared_cpu_list` values of the level-3 caches that the
  logical processors report under `/sys/devices/system/cpu`.
- **Windows**: the number of level-3 `RelationCache` records.
- **macOS**: `1` where the system reports an L3 size at all, else `0`.

The count covers every logical processor the operating system lists, so a
process confined to part of the machine sees the machine's figure.

Detected once and cached thread-safely, like `llc_bytes`.

### `llc_domains`

```cpp
const std::vector<std::vector<std::size_t>>& llc_domains();
```

The logical processors of each last-level (L3) cache instance: one entry per
instance, each a sorted list of processor numbers, ordered by the lowest
processor in each. On a Zen 4 part with two chiplets of six cores and SMT this
is `{0-5, 12-17}` and `{6-11, 18-23}`. Empty when unknown.

Processor numbers are the operating system's own: Linux's `cpuN` numbers (the
ones `sched_getaffinity` and `OMP_PLACES` use), Windows' `group * 64 + bit`,
and on macOS, which publishes no cache topology, `0` to `N-1` in a single
domain where an L3 is reported at all.

Only a coherent reading counts. Every online processor must name its own L3 and
appear in it, and the instances must partition the processors the operating
system lists. A processor without an L3 entry, two lists that overlap, or a
list that does not parse makes the whole answer empty rather than a partial
one.

- **Linux**: the `shared_cpu_list` of each online processor's level-3 cache
  index under `/sys/devices/system/cpu`. An offline processor is skipped.
- **Windows**: the level-3 `RelationCache` records of
  `GetLogicalProcessorInformationEx`.
- **macOS**: one domain of `hw.logicalcpu` processors where `hw.l3cachesize`
  is reported, else empty.

The domains cover every logical processor the operating system lists, so a
process confined to part of the machine still sees the machine's domains; use
`threads_per_llc` for the confined figure. Detected once and cached
thread-safely, like `llc_bytes`.

### `physical_cores`

```cpp
std::size_t physical_cores();
```

How many physical cores the machine has: logical processors that share a core
counted once. Returns `0` when unknown. On the part above this is `12` for `24`
logical processors.

- **Linux**: the distinct sibling lists (`topology/core_cpus_list`, else
  `topology/thread_siblings_list`) of the online processors, under the same
  coherence rule as `llc_domains`.
- **Windows**: the number of `RelationProcessorCore` records.
- **macOS**: `hw.physicalcpu`.

Counted over every logical processor the operating system lists. Detected once
and cached thread-safely.

### `smt_width`

```cpp
std::size_t smt_width();
```

How many logical processors share each physical core: `1` without SMT, `2` with
it. Returns `0` when unknown, and also when the cores are not all the same width
(a hybrid part with SMT on only some of them), since no single figure is true of
that machine. Wherever it is non-zero, `physical_cores() * smt_width()` is the
number of logical processors the operating system lists. Detected once and
cached thread-safely.

### `threads_per_llc`

```cpp
std::size_t threads_per_llc();
```

How many logical processors one L3 instance offers the calling thread: the
largest number of processors the calling thread may run on that sit under one
L3. Returns `0` when `llc_domains` is unknown.

On Linux the thread's affinity mask is honoured, whether it comes from
`taskset`, a cpuset or a batch job's binding. A process pinned to six of the
twelve processors of one chiplet reads `6`, not `12`. Where the mask cannot be
read, and on Windows and macOS, no restriction is assumed and the figure is the
size of the largest instance.

These are logical processors, not cores. When the process may use both
siblings of every core, divide by `smt_width` for the cores. Read afresh on
every call, since the affinity can change.

### `available_memory_bytes`

```cpp
std::size_t available_memory_bytes();
```

Memory the operating system reports it could hand out now without swapping, in
bytes, or `0` when unknown. Read afresh on every call, since it changes as the
process and the machine allocate.

- **Linux**: `MemAvailable` from `/proc/meminfo`, the kernel's own estimate,
  which counts page cache it can drop. Only a coherent reading counts: the field
  must parse as a number, be non-zero, convert to bytes without overflow, and be
  no larger than `MemTotal`, read in the same pass. Anything else is `0`.
  When the process runs under a cgroup memory limit (a container, a systemd
  slice, a batch scheduler's job), the figure is the smaller of `MemAvailable`
  and the room under the tightest limit from the process's own cgroup up to the
  root: that limit less the cgroup's usage, not counting inactive file cache.
  The kernel ends a process that goes over its cgroup's limit without an
  exception, so the machine's figure alone would overstate what a run can use.
  The limits are read at `/sys/fs/cgroup` (cgroup version 2) or
  `/sys/fs/cgroup/memory` (the version 1 memory controller), where they are
  normally mounted. A cgroup at its limit reads as 1 byte rather than `0`,
  since that is an answer: nothing more fits.
- **Windows**: the available physical memory from `GlobalMemoryStatusEx`.
- **macOS**: free pages plus inactive ones, which the kernel reclaims before it
  swaps.

### `recent_available_memory_bytes`

```cpp
std::size_t recent_available_memory_bytes();
```

The same figure, read at most once a second and shared between the calls in
between. Every simulator run takes its automatic memory cap from this: a run asks
once, before it starts, and reading `/proc/meminfo` on every call would cost
more than a small run does. The cap is a snapshot either way, since another
process can allocate the moment after any reading (see
[The memory cap](failures.md#the-memory-cap)).

### `available_threads`

```cpp
std::size_t available_threads();
```

How many threads a parallel region the calling thread starts now may use:
OpenMP's current limit for that thread, which `OMP_NUM_THREADS`, the caller's
own `omp_set_num_threads` and a simulator's `max_parallel_threads` all set.
`1` in a build without OpenMP. Read afresh on every call, since a simulator's
cap applies only for the length of the call that set it. The statevector
passes size their teams from it (see `min_amplitudes_per_thread` in
[Simulators](simulators.md)).

### `page_bytes`

```cpp
std::size_t page_bytes();
```

The operating system's memory page size in bytes, or `0` if unknown:
`sysconf(_SC_PAGESIZE)` on Linux and macOS, `GetSystemInfo` on Windows.
Detected once and cached. A `Statevector` aligns each amplitude array of at
least one page to it, so the real and imaginary arrays start at the same offset
within a page: a kernel that loads both and then stores both would otherwise
have its next loads held back behind its own stores, which the CPU matches on
the low address bits. When the size is unknown the arrays keep cache-line
alignment.

## Detection and its limits

Three platform paths, each asking the operating system rather than the CPU:

- **Windows**: enumerates `RelationCache` records from
  `GetLogicalProcessorInformation` and takes the first L3 entry. The topology
  queries use `GetLogicalProcessorInformationEx`.
- **macOS**: `sysctlbyname("hw.l3cachesize")`. The topology queries use
  `hw.logicalcpu` and `hw.physicalcpu`.
- **Linux, including WSL**: `sysconf(_SC_LEVEL3_CACHE_SIZE)`, which glibc
  answers from sysfs for cpu 0. The topology queries read the per-processor
  `cache` and `topology` directories of sysfs.

Returning `0` (or an empty `llc_domains`) is a normal outcome, not an error. Containers frequently expose no
cache topology, and unusual platforms may report nothing. A caller must handle
it, which is why the return is documented rather than hidden behind a default.

## Exceptions

None. No function here throws or reports failure through an error code. Failure
is the `0` return, or the empty list from `llc_domains`.

## Example

The library's own use, reduced to its essentials. The statevector simulator
turns the cache size into the register width at which gate fusion starts paying:

```cpp
#include "lindblad/hw_info.hpp"

#include <algorithm>
#include <cstddef>

// 16 bytes per amplitude: separate double arrays for real and imaginary parts.
constexpr std::size_t kBytesPerAmp = 16;
constexpr std::size_t kFallback = std::size_t(32) << 20;  // 32 MiB
constexpr std::size_t kMin      = std::size_t(1)  << 20;  // 1 MiB
constexpr std::size_t kMax      = std::size_t(1)  << 30;  // 1 GiB

int first_bandwidth_bound_width() {
    std::size_t llc = lindblad::hw::llc_bytes();
    if (llc == 0) llc = kFallback;               // the caller's own fallback
    llc = std::clamp(llc, kMin, kMax);           // guard against garbage

    int n = 1;
    while ((kBytesPerAmp << n) <= llc) ++n;      // first n where the state exceeds one instance
    return n;
}
```

Two things in that example are worth copying rather than skipping.

The **fallback belongs to the caller**, not to `llc_bytes`. The simulator picks
32 MiB because that is a common L3 size and because erring high makes fusion
engage later, and a missed fusion win costs far less than fusing a state that
was still cache-resident.

The **clamp** is not defensive padding. A detector reading an unexpected sysfs
value could return something absurd, and clamping keeps the derived threshold
inside a range where the formula still means something.

## Why the cache size drives fusion

A cache-resident statevector is compute-bound. The specialised per-gate kernels
already win there, and collapsing several gates into one dense block would add
arithmetic rather than remove it. Once the state no longer fits, simulation turns
bandwidth-bound and the number of sweeps over memory is what costs, so fusing
gates into fewer passes pays.

Measured on a part with 32 MiB of L3: fusing a 32 MiB state ran 3.3 times
slower, fusing a 64 MiB state ran 2.7 times faster, and smaller cache-resident
states regressed by as much as 15 times when fused. The boundary is therefore
"strictly larger than one cache instance", and it is derived from the machine
rather than hardcoded.

## Related pages

- `docs/api/simulators.md` for `StatevectorSimulator::Options`, whose
  `fusion_enable`, `fusion_threshold` and `fusion_max_qubit` fields override the
  automatic engagement point described above.
- `docs/Architecture.md` for where fusion sits in the execution pipeline.
- `docs/api/failures.md` for the memory cap the available-memory reading feeds.
