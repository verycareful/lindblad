#include "lindblad/hw_info.hpp"

#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
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

}  // namespace

std::size_t llc_bytes() {
    static const std::size_t cached = detect_llc_bytes();  // magic static: thread-safe
    return cached;
}

}  // namespace hw
}  // namespace lindblad
