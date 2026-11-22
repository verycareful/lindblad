// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#include "lindblad/hw_info.hpp"
#include "lindblad/detail/hw_reach.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <utility>
#include <string>
#include <system_error>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <unistd.h>
#else
#include <unistd.h>
#endif

#if defined(__linux__)
#include <sched.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace lindblad {
namespace hw {

namespace {

// =============================================================================
// LLC detection — one platform branch per OS, 0 on any failure
// =============================================================================

#if !defined(_WIN32) && !defined(__APPLE__)
// Parse a Linux sysfs cache-size string ("32768K", "32M", "1G"; plain bytes
// when unsuffixed). Returns 0 on anything unparseable.
std::size_t parse_sysfs_size(const std::string& s) {
    if (s.empty()) return 0;
    char* end = nullptr;
    const unsigned long long v = std::strtoull(s.c_str(), &end, 10);
    if (end == s.c_str() || v == 0) return 0;
    switch (*end) {
        case 'K': case 'k': return static_cast<std::size_t>(v) << 10;
        case 'M': case 'm': return static_cast<std::size_t>(v) << 20;
        case 'G': case 'g': return static_cast<std::size_t>(v) << 30;
        case '\0': case '\n': return static_cast<std::size_t>(v);
        default: return 0;
    }
}
#endif

std::size_t detect_llc_bytes() {
#if defined(_WIN32)
    // Enumerate RelationCache records; the first L3 entry carries the
    // per-instance size (uniform across instances on mainstream parts).
    DWORD len = 0;
    GetLogicalProcessorInformation(nullptr, &len);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || len == 0) return 0;
    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buf(
        len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) + 1);
    if (!GetLogicalProcessorInformation(buf.data(), &len)) return 0;
    const std::size_t n = len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
    for (std::size_t i = 0; i < n; ++i) {
        if (buf[i].Relationship == RelationCache && buf[i].Cache.Level == 3) {
            return static_cast<std::size_t>(buf[i].Cache.Size);
        }
    }
    return 0;
#elif defined(__APPLE__)
    // Intel Macs report hw.l3cachesize; Apple Silicon has no exposed L3
    // (the system-level cache is not published), so this correctly returns
    // "unknown" there and the caller's fallback applies.
    std::size_t v = 0;
    std::size_t sz = sizeof(v);
    if (sysctlbyname("hw.l3cachesize", &v, &sz, nullptr, 0) == 0 && v > 0) {
        return v;
    }
    return 0;
#else
    // Linux (including WSL2). glibc's sysconf reads sysfs for cpu 0, which
    // is exactly the per-instance figure documented in the header.
#ifdef _SC_LEVEL3_CACHE_SIZE
    {
        const long v = ::sysconf(_SC_LEVEL3_CACHE_SIZE);
        if (v > 0) return static_cast<std::size_t>(v);
    }
#endif
    // sysfs fallback (musl and friends): cpu0's level-3 index entry.
    for (int idx = 0; idx < 8; ++idx) {
        const std::string base =
            "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(idx);
        std::ifstream lvl(base + "/level");
        int level = 0;
        if (!(lvl >> level) || level != 3) continue;
        std::ifstream szf(base + "/size");
        std::string s;
        if (szf >> s) return parse_sysfs_size(s);
    }
    return 0;
#endif
}

std::size_t detect_llc_instances() {
#if defined(_WIN32)
    // One RelationCache record per cache instance.
    DWORD len = 0;
    GetLogicalProcessorInformation(nullptr, &len);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || len == 0) return 0;
    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buf(
        len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) + 1);
    if (!GetLogicalProcessorInformation(buf.data(), &len)) return 0;
    const std::size_t n = len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
    std::size_t count = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (buf[i].Relationship == RelationCache && buf[i].Cache.Level == 3) ++count;
    }
    return count;
#elif defined(__APPLE__)
    // One shared L3 where the system reports one at all.
    return detect_llc_bytes() > 0 ? 1 : 0;
#else
    // Linux: every logical processor lists the processors that share each of
    // its caches, so the L3 instances are the distinct lists.
    namespace fs = std::filesystem;
    std::set<std::string> lists;
    std::error_code ec;
    for (fs::directory_iterator it("/sys/devices/system/cpu", ec), end; !ec && it != end;
         it.increment(ec)) {
        const std::string name = it->path().filename().string();
        if (name.size() < 4 || name.compare(0, 3, "cpu") != 0 ||
            name.find_first_not_of("0123456789", 3) != std::string::npos)
            continue;
        for (int idx = 0; idx < 8; ++idx) {
            const std::string base = it->path().string() + "/cache/index" + std::to_string(idx);
            std::ifstream lvl(base + "/level");
            int level = 0;
            if (!(lvl >> level) || level != 3) continue;
            std::ifstream shared(base + "/shared_cpu_list");
            std::string list;
            if (shared >> list) lists.insert(list);
            break;
        }
    }
    return lists.size();
#endif
}

// =============================================================================
// Topology detection - L3 domains, physical cores, SMT width. One platform
// branch per OS; an empty or 0 result on any failure or incoherent reading.
// =============================================================================

// The physical cores of the machine and how wide each is.
struct CoreShape {
    std::size_t cores = 0;
    std::size_t smt = 0;
};

#if !defined(_WIN32) && !defined(__APPLE__)

// Processor numbers beyond this are not a reading of any machine; a list
// that names one is malformed, and the bound keeps a corrupt range such as
// "0-4294967295" from allocating.
constexpr unsigned long long kMaxCpuNumber = 1ULL << 16;

// A whole word as a count, in no locale, or false.
bool parse_count(const std::string& word, unsigned long long& out) {
    const char* first = word.data();
    const char* last = first + word.size();
    const auto [ptr, ec] = std::from_chars(first, last, out);
    return ec == std::errc() && ptr == last;
}

// The first line of a sysfs file, or false when it cannot be read.
bool read_first_line(const std::string& path, std::string& line) {
    std::ifstream in(path);
    return static_cast<bool>(std::getline(in, line));
}

// A Linux CPU list ("0-5,12-17", "0,12", "7") as sorted processor numbers.
// False for anything that is not exactly that: an empty list, an empty item,
// a reversed range, a number past kMaxCpuNumber, a processor named twice.
bool parse_cpu_list(const std::string& text, std::vector<std::size_t>& out) {
    out.clear();
    std::string trimmed = text;
    while (!trimmed.empty() && (trimmed.back() == '\n' || trimmed.back() == '\r' ||
                                trimmed.back() == ' '))
        trimmed.pop_back();
    if (trimmed.empty()) return false;
    std::size_t pos = 0;
    while (pos <= trimmed.size()) {
        const std::size_t comma = std::min(trimmed.find(',', pos), trimmed.size());
        const std::string item = trimmed.substr(pos, comma - pos);
        const std::size_t dash = item.find('-');
        unsigned long long lo = 0;
        unsigned long long hi = 0;
        if (dash == std::string::npos) {
            if (!parse_count(item, lo)) return false;
            hi = lo;
        } else if (!parse_count(item.substr(0, dash), lo) ||
                   !parse_count(item.substr(dash + 1), hi)) {
            return false;
        }
        if (lo > hi || hi >= kMaxCpuNumber) return false;
        for (unsigned long long c = lo; c <= hi; ++c) out.push_back(static_cast<std::size_t>(c));
        pos = comma + 1;
    }
    std::sort(out.begin(), out.end());
    return std::adjacent_find(out.begin(), out.end()) == out.end();
}

// The sysfs directory of every online logical processor, by processor number.
// An offline processor keeps its cpuN directory but has no topology or cache
// entries, and says so in its `online` file (cpu0 has none: it is always up).
std::vector<std::pair<std::size_t, std::string>> online_cpu_dirs() {
    namespace fs = std::filesystem;
    std::vector<std::pair<std::size_t, std::string>> cpus;
    std::error_code ec;
    for (fs::directory_iterator it("/sys/devices/system/cpu", ec), end; !ec && it != end;
         it.increment(ec)) {
        const std::string name = it->path().filename().string();
        unsigned long long id = 0;
        if (name.size() < 4 || name.compare(0, 3, "cpu") != 0 ||
            !parse_count(name.substr(3), id))
            continue;
        std::string online;
        if (read_first_line(it->path().string() + "/online", online) && online == "0") continue;
        cpus.emplace_back(static_cast<std::size_t>(id), it->path().string());
    }
    std::sort(cpus.begin(), cpus.end());
    return cpus;
}

// The groups of processors that `list_of(dir)` names for each online
// processor, as sorted lists ordered by lowest processor. Empty unless every
// processor names a list that parses and contains itself and the lists
// partition exactly the online processors: a partial or overlapping answer
// is not a topology.
template <typename ListOf>
std::vector<std::vector<std::size_t>> linux_processor_groups(ListOf list_of) {
    const auto cpus = online_cpu_dirs();
    if (cpus.empty()) return {};
    std::set<std::vector<std::size_t>> groups;
    for (const auto& [id, dir] : cpus) {
        std::string text;
        std::vector<std::size_t> members;
        if (!list_of(dir, text) || !parse_cpu_list(text, members)) return {};
        if (!std::binary_search(members.begin(), members.end(), id)) return {};
        groups.insert(std::move(members));
    }
    std::set<std::size_t> seen;
    for (const auto& group : groups) {
        for (const std::size_t c : group) {
            if (!seen.insert(c).second) return {};
        }
    }
    if (seen.size() != cpus.size()) return {};
    for (const auto& [id, dir] : cpus) {
        if (!seen.count(id)) return {};
    }
    return {groups.begin(), groups.end()};
}

// Width shared by every group, or 0 when they differ.
std::size_t uniform_width(const std::vector<std::vector<std::size_t>>& groups) {
    if (groups.empty()) return 0;
    for (const auto& g : groups) {
        if (g.size() != groups.front().size()) return 0;
    }
    return groups.front().size();
}

#endif

std::vector<std::vector<std::size_t>> detect_llc_domains() {
#if defined(_WIN32)
    // One RelationCache record per cache instance; its mask names the
    // processors of its processor group, numbered group * 64 + bit.
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationCache, nullptr, &len);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || len == 0) return {};
    std::vector<char> buf(len);
    if (!GetLogicalProcessorInformationEx(
            RelationCache, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()),
            &len))
        return {};
    constexpr std::size_t bits = sizeof(KAFFINITY) * 8;
    std::set<std::vector<std::size_t>> groups;
    for (std::size_t off = 0; off < len;) {
        const auto* rec = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off);
        if (rec->Size == 0) return {};
        off += rec->Size;
        if (rec->Relationship != RelationCache || rec->Cache.Level != 3) continue;
        std::vector<std::size_t> members;
        for (std::size_t b = 0; b < bits; ++b) {
            if ((rec->Cache.GroupMask.Mask >> b) & 1u)
                members.push_back(static_cast<std::size_t>(rec->Cache.GroupMask.Group) * bits + b);
        }
        if (members.empty()) return {};
        groups.insert(std::move(members));
    }
    std::set<std::size_t> seen;
    for (const auto& g : groups) {
        for (const std::size_t c : g) {
            if (!seen.insert(c).second) return {};
        }
    }
    return {groups.begin(), groups.end()};
#elif defined(__APPLE__)
    // No cache topology is published: one domain of every logical processor
    // where an L3 is reported at all (llc_bytes() is 0 on Apple Silicon, so
    // the answer is empty there).
    int logical = 0;
    std::size_t sz = sizeof(logical);
    if (detect_llc_bytes() == 0 || sysctlbyname("hw.logicalcpu", &logical, &sz, nullptr, 0) != 0 ||
        logical <= 0)
        return {};
    std::vector<std::size_t> all(static_cast<std::size_t>(logical));
    for (std::size_t i = 0; i < all.size(); ++i) all[i] = i;
    return {all};
#else
    // Linux: the processors that share each online processor's L3 are the
    // list its level-3 cache index names.
    return linux_processor_groups([](const std::string& dir, std::string& text) {
        for (int idx = 0; idx < 8; ++idx) {
            const std::string base = dir + "/cache/index" + std::to_string(idx);
            std::string level;
            if (!read_first_line(base + "/level", level) || level != "3") continue;
            return read_first_line(base + "/shared_cpu_list", text);
        }
        return false;
    });
#endif
}

CoreShape detect_core_shape() {
#if defined(_WIN32)
    // One RelationProcessorCore record per physical core; its mask names the
    // core's logical processors.
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || len == 0) return {};
    std::vector<char> buf(len);
    if (!GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()), &len))
        return {};
    std::size_t cores = 0;
    std::size_t width = 0;
    bool uniform = true;
    for (std::size_t off = 0; off < len;) {
        const auto* rec = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off);
        if (rec->Size == 0) return {};
        off += rec->Size;
        if (rec->Relationship != RelationProcessorCore) continue;
        std::size_t here = 0;
        for (WORD g = 0; g < rec->Processor.GroupCount; ++g) {
            KAFFINITY m = rec->Processor.GroupMask[g].Mask;
            for (; m != 0; m &= m - 1) ++here;
        }
        if (here == 0) return {};
        if (cores > 0 && here != width) uniform = false;
        width = here;
        ++cores;
    }
    if (cores == 0) return {};
    return {cores, uniform ? width : 0};
#elif defined(__APPLE__)
    int physical = 0;
    int logical = 0;
    std::size_t sz = sizeof(physical);
    if (sysctlbyname("hw.physicalcpu", &physical, &sz, nullptr, 0) != 0 || physical <= 0) return {};
    sz = sizeof(logical);
    if (sysctlbyname("hw.logicalcpu", &logical, &sz, nullptr, 0) != 0 || logical < physical) {
        return {static_cast<std::size_t>(physical), 0};
    }
    // Every core has the same width only if the logical count divides evenly.
    const bool even = logical % physical == 0;
    return {static_cast<std::size_t>(physical), even ? static_cast<std::size_t>(logical / physical) : 0};
#else
    // Linux: the processors of one core list each other as siblings.
    // core_cpus_list is the current name, thread_siblings_list the older one.
    const auto cores = linux_processor_groups([](const std::string& dir, std::string& text) {
        return read_first_line(dir + "/topology/core_cpus_list", text) ||
               read_first_line(dir + "/topology/thread_siblings_list", text);
    });
    return {cores.size(), uniform_width(cores)};
#endif
}

#if defined(__linux__)
// The processors an OpenMP team started by the calling thread may be placed
// on, by processor number, or false when nothing narrows them but the
// operating system's own scheduling. Under OMP_PROC_BIND the runtime binds
// the primary thread to one place, so the calling thread's own affinity mask
// is a single processor and says nothing about the team; the place list is
// what the team spreads over. Without binding, the team runs wherever the
// calling thread's affinity mask (taskset, a cpuset, a batch job's binding)
// lets it, and the mask is read instead.
bool allowed_cpus(std::vector<bool>& allowed) {
    allowed.clear();
#ifdef _OPENMP
    if (omp_get_proc_bind() != omp_proc_bind_false) {
        const int places = omp_get_num_places();
        for (int p = 0; p < places; ++p) {
            std::vector<int> ids(static_cast<std::size_t>(std::max(0, omp_get_place_num_procs(p))));
            if (ids.empty()) continue;
            omp_get_place_proc_ids(p, ids.data());
            for (const int id : ids) {
                if (id < 0) continue;
                if (static_cast<std::size_t>(id) >= allowed.size()) allowed.resize(id + 1, false);
                allowed[static_cast<std::size_t>(id)] = true;
            }
        }
        if (!allowed.empty()) return true;
    }
#endif
    // The kernel rejects a mask smaller than its own processor count, which
    // can exceed the highest online number, so start generous and grow.
    for (std::size_t slots = 1024; slots <= kMaxCpuNumber; slots *= 2) {
        cpu_set_t* raw = CPU_ALLOC(slots);
        if (raw == nullptr) return false;
        const std::unique_ptr<cpu_set_t, void (*)(cpu_set_t*)> set(raw, [](cpu_set_t* p) { CPU_FREE(p); });
        const std::size_t bytes = CPU_ALLOC_SIZE(slots);
        CPU_ZERO_S(bytes, set.get());
        if (::sched_getaffinity(0, bytes, set.get()) != 0) continue;
        allowed.assign(slots, false);
        for (std::size_t c = 0; c < slots; ++c) allowed[c] = CPU_ISSET_S(c, bytes, set.get());
        return true;
    }
    return false;
}
#endif

// =============================================================================
// Available memory - one platform branch per OS, 0 on any failure
// =============================================================================

#if !defined(_WIN32) && !defined(__APPLE__)

// The first word of a cgroup interface file as a count. False for a missing
// file, for "max" (no limit set at that level) and for anything else that is
// not a number.
bool read_cgroup_count(const std::string& path, unsigned long long& out) {
    std::ifstream in(path);
    std::string word;
    return static_cast<bool>(in >> word) && parse_count(word, out);
}

// The value of `key` in a cgroup memory.stat file.
bool read_cgroup_stat(const std::string& path, const char* key, unsigned long long& out) {
    std::ifstream in(path);
    std::string name;
    std::string value;
    while (in >> name >> value) {
        if (name == key) return parse_count(value, out);
    }
    return false;
}

// The memory this process can still use under the limits of the cgroups it
// runs in, or false when no limit applies or none can be read. A container, a
// systemd slice or a batch scheduler's job caps memory below what
// /proc/meminfo reports for the whole machine, and the kernel ends a process
// that goes over it with no exception, so a cap read from the machine alone
// would let a run walk into exactly that.
//
// A limit set on a parent binds every cgroup under it, so each level from the
// process's own cgroup up to the root is read and the smallest room wins. The
// room at a level is its limit less its working set: usage less the inactive
// file cache, which the kernel reclaims before it kills anything (the figure
// container runtimes report). Read at the standard mount points: the cgroup v1
// memory controller at /sys/fs/cgroup/memory when this process is in one,
// otherwise cgroup v2 at /sys/fs/cgroup.
bool cgroup_room_bytes(unsigned long long& room) {
    std::ifstream self("/proc/self/cgroup");
    std::string line;
    std::string v1_path;
    std::string v2_path;
    // Each line is hierarchy-id:controllers:path. The v2 line has id 0 and no
    // controllers; a v1 line lists its controllers separated by commas.
    while (std::getline(self, line)) {
        const std::size_t first = line.find(':');
        if (first == std::string::npos) continue;
        const std::size_t second = line.find(':', first + 1);
        if (second == std::string::npos) continue;
        const std::string id = line.substr(0, first);
        const std::string controllers = line.substr(first + 1, second - first - 1);
        const std::string path = line.substr(second + 1);
        if (id == "0" && controllers.empty()) {
            v2_path = path;
            continue;
        }
        std::size_t start = 0;
        while (start <= controllers.size()) {
            const std::size_t end = std::min(controllers.find(',', start), controllers.size());
            if (controllers.compare(start, end - start, "memory") == 0) v1_path = path;
            start = end + 1;
        }
    }

    const bool v1 = !v1_path.empty();
    const std::string root = v1 ? "/sys/fs/cgroup/memory" : "/sys/fs/cgroup";
    const char* limit_file = v1 ? "/memory.limit_in_bytes" : "/memory.max";
    const char* usage_file = v1 ? "/memory.usage_in_bytes" : "/memory.current";
    const char* inactive_key = v1 ? "total_inactive_file" : "inactive_file";
    std::string path = v1 ? v1_path : v2_path;
    if (path.empty() || path.front() != '/') {
        // No cgroup line for this hierarchy, or a path outside this view of
        // it: the mount point itself is the nearest level there is.
        path = "/";
    }

    bool limited = false;
    unsigned long long smallest = 0;
    for (;;) {
        const std::string dir = root + (path == "/" ? std::string() : path);
        unsigned long long limit = 0;
        unsigned long long usage = 0;
        if (read_cgroup_count(dir + limit_file, limit) &&
            read_cgroup_count(dir + usage_file, usage)) {
            unsigned long long inactive = 0;
            if (!read_cgroup_stat(dir + "/memory.stat", inactive_key, inactive)) inactive = 0;
            const unsigned long long working = usage > inactive ? usage - inactive : 0;
            const unsigned long long here = limit > working ? limit - working : 0;
            if (!limited || here < smallest) smallest = here;
            limited = true;
        }
        if (path == "/") break;
        const std::size_t slash = path.find_last_of('/');
        path = slash == 0 ? std::string("/") : path.substr(0, slash);
    }
    if (limited) room = smallest;
    return limited;
}

#endif

std::size_t detect_available_memory_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) return 0;
    return static_cast<std::size_t>(status.ullAvailPhys);
#elif defined(__APPLE__)
    // Free pages plus inactive ones, which the kernel reclaims before it
    // swaps: the nearest macOS figure to Linux's MemAvailable.
    vm_statistics64_data_t stats;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          reinterpret_cast<host_info64_t>(&stats),
                          &count) != KERN_SUCCESS)
        return 0;
    const long page = ::sysconf(_SC_PAGESIZE);
    if (page <= 0) return 0;
    return static_cast<std::size_t>(stats.free_count + stats.inactive_count) *
           static_cast<std::size_t>(page);
#else
    // MemAvailable is the kernel's own estimate of what can be allocated
    // without swapping, page cache it can drop included; MemFree alone would
    // understate it on any machine that has been running a while.
    //
    // Only a coherent reading is an answer. A field that does not parse as a
    // number stops the read. Zero is no answer, a kB figure whose byte count
    // overflows is not a reading of any machine, and available memory above
    // the machine's MemTotal, read in the same pass, is not a reading of this
    // one. Each of those returns 0, so the caller falls back.
    std::ifstream meminfo("/proc/meminfo");
    std::string key;
    unsigned long long value = 0;
    std::string unit;
    unsigned long long total_kb = 0;
    unsigned long long avail_kb = 0;
    bool have_total = false;
    bool have_avail = false;
    while (meminfo >> key >> value) {
        std::getline(meminfo, unit);
        if (key == "MemTotal:") {
            total_kb = value;
            have_total = true;
        } else if (key == "MemAvailable:") {
            avail_kb = value;
            have_avail = true;
        }
        if (have_total && have_avail) break;
    }
    constexpr unsigned long long max_kb = std::numeric_limits<std::size_t>::max() >> 10;
    const bool coherent = have_avail && avail_kb != 0 && avail_kb <= max_kb &&
                          !(have_total && avail_kb > total_kb);
    const std::size_t machine = coherent ? static_cast<std::size_t>(avail_kb) << 10 : 0;

    unsigned long long room = 0;
    if (!cgroup_room_bytes(room)) return machine;
    // A cgroup limit applies, so the answer is the smaller of the machine's
    // figure and the room under the limit. A cgroup at its limit has no room,
    // which is an answer (nothing more fits) rather than the absence of one,
    // so it reads as 1 byte, never as the 0 that sends callers to a fallback.
    const std::size_t limit_room = static_cast<std::size_t>(
        std::min<unsigned long long>(room, std::numeric_limits<std::size_t>::max()));
    const std::size_t answer = machine == 0 ? limit_room : std::min(machine, limit_room);
    return answer == 0 ? 1 : answer;
#endif
}

}  // namespace

std::size_t llc_bytes() {
    static const std::size_t cached = detect_llc_bytes();  // magic static: thread-safe
    return cached;
}

std::size_t llc_instances() {
    static const std::size_t cached = detect_llc_instances();  // magic static: thread-safe
    return cached;
}

}  // namespace hw

namespace detail {
bool team_processors(std::vector<bool>& allowed) {
#if defined(__linux__)
    return hw::allowed_cpus(allowed);
#else
    allowed.clear();
    return false;
#endif
}
}  // namespace detail

namespace hw {

const std::vector<std::vector<std::size_t>>& llc_domains() {
    static const std::vector<std::vector<std::size_t>> cached = detect_llc_domains();  // magic static: thread-safe
    return cached;
}

namespace {
const CoreShape& core_shape() {
    static const CoreShape cached = detect_core_shape();  // magic static: thread-safe
    return cached;
}
}  // namespace

std::size_t physical_cores() { return core_shape().cores; }

std::size_t smt_width() { return core_shape().smt; }

std::size_t threads_per_llc() {
    const auto& domains = llc_domains();
#if defined(__linux__)
    std::vector<bool> allowed;
    const bool restricted = allowed_cpus(allowed);
#endif
    std::size_t largest = 0;
    for (const auto& domain : domains) {
        std::size_t n = domain.size();
#if defined(__linux__)
        if (restricted) {
            n = 0;
            for (const std::size_t c : domain) n += c < allowed.size() && allowed[c] ? 1 : 0;
        }
#endif
        largest = std::max(largest, n);
    }
    return largest;
}

std::size_t available_memory_bytes() { return detect_available_memory_bytes(); }

std::size_t recent_available_memory_bytes() {
    // One reading serves every call inside the window. The stamp is published
    // after the reading (release, then acquire on the read side), so a caller
    // that sees a fresh stamp also sees the reading it belongs to. Two callers
    // refreshing at once both read the machine, which costs a second read and
    // nothing else.
    constexpr std::int64_t max_age_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(1)).count();
    static std::atomic<std::int64_t> read_at{-1};
    static std::atomic<std::size_t> reading{0};
    const std::int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 std::chrono::steady_clock::now().time_since_epoch())
                                 .count();
    const std::int64_t stamp = read_at.load(std::memory_order_acquire);
    if (stamp >= 0 && now - stamp < max_age_ns) {
        return reading.load(std::memory_order_relaxed);
    }
    const std::size_t fresh = detect_available_memory_bytes();
    reading.store(fresh, std::memory_order_relaxed);
    read_at.store(now, std::memory_order_release);
    return fresh;
}

std::size_t available_threads() {
#ifdef _OPENMP
    return static_cast<std::size_t>(std::max(1, omp_get_max_threads()));
#else
    return 1;
#endif
}

std::size_t page_bytes() {
    static const std::size_t cached = [] {  // magic static: thread-safe
#if defined(_WIN32)
        SYSTEM_INFO info;
        GetSystemInfo(&info);
        return static_cast<std::size_t>(info.dwPageSize);
#else
        const long page = ::sysconf(_SC_PAGESIZE);
        return page > 0 ? static_cast<std::size_t>(page) : std::size_t{0};
#endif
    }();
    return cached;
}

}  // namespace hw
}  // namespace lindblad
