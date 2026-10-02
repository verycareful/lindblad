# Hardware Info

Best-effort runtime detection of hardware properties that change how the library
tunes itself. There are two: the last-level cache size, which the statevector
simulator asks to decide whether to fuse gates, and the memory the operating
system reports available, which sets the memory cap a run answers to when
`max_memory_mb` is 0 and decides whether a per-shot run keeps its prefix
snapshot.

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

## Detection and its limits

Three platform paths, each asking the operating system rather than the CPU:

- **Windows**: enumerates `RelationCache` records from
  `GetLogicalProcessorInformation` and takes the first L3 entry.
- **macOS**: `sysctlbyname("hw.l3cachesize")`.
- **Linux, including WSL**: `sysconf(_SC_LEVEL3_CACHE_SIZE)`, which glibc
  answers from sysfs for cpu 0.

Returning `0` is a normal outcome, not an error. Containers frequently expose no
cache topology, and unusual platforms may report nothing. A caller must handle
it, which is why the return is documented rather than hidden behind a default.

## Exceptions

None. No function here throws or reports failure through an error code. Failure
is the `0` return.

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
