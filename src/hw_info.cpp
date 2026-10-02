// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#include "lindblad/hw_info.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
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

// =============================================================================
// Available memory - one platform branch per OS, 0 on any failure
// =============================================================================

#if !defined(_WIN32) && !defined(__APPLE__)

// A whole word as a count, in no locale, or false.
bool parse_count(const std::string& word, unsigned long long& out) {
    const char* first = word.data();
    const char* last = first + word.size();
    const auto [ptr, ec] = std::from_chars(first, last, out);
    return ec == std::errc() && ptr == last;
}

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

}  // namespace hw
}  // namespace lindblad
