// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#include "lindblad/hw_info.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
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
    if (!have_avail || avail_kb == 0 || avail_kb > max_kb) return 0;
    if (have_total && avail_kb > total_kb) return 0;
    return static_cast<std::size_t>(avail_kb) << 10;
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
