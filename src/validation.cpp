#include "lindblad/validation.hpp"

#include <cstddef>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

// =============================================================================
// Warning channel - one handler, deduplicated
// =============================================================================
// State is a handler plus a count per distinct message, both under one mutex.
// The handler is invoked while that mutex is held, which is what guarantees a
// user-supplied sink is never entered from two threads at once: warnings
// originate inside OpenMP parallel regions, so without it every caller would
// have to write their own lock. The cost of that guarantee is that a handler
// must not call back into this API; doing so deadlocks on a non-recursive
// mutex, and it is documented on the declarations.

namespace lindblad {

namespace {

// Above this many distinct messages, deduplication stops and every occurrence
// is emitted. A workload generating unbounded distinct warnings is one where
// the counts are not the interesting part, and the map must not grow without
// limit to track them.
constexpr std::size_t MAX_TRACKED_MESSAGES = 64;

std::mutex& warning_mutex() {
    static std::mutex m;
    return m;
}

WarningHandler& warning_handler() {
    static WarningHandler h;
    return h;
}

// message -> how many times it has been seen since the last flush, counting
// the occurrence that was emitted.
std::unordered_map<std::string, std::size_t>& warning_counts() {
    static std::unordered_map<std::string, std::size_t> counts;
    return counts;
}

// Caller holds the mutex.
//
// The default sink writes to std::cerr rather than C stderr. The two share a
// file descriptor but not a buffer, so a caller who swaps std::cerr's rdbuf to
// capture library output would see nothing from a fprintf. Redirecting
// std::cerr is the idiomatic way to capture C++ diagnostics, and it has to keep
// working.
void deliver_locked(const std::string& message) {
    const WarningHandler& h = warning_handler();
    if (h) {
        h(message);
    } else {
        std::cerr << "lindblad: " << message << '\n';
    }
}

// Caller holds the mutex.
void flush_locked() {
    auto& counts = warning_counts();
    for (const auto& entry : counts) {
        if (entry.second > 1) {
            deliver_locked(entry.first + " [repeated " +
                           std::to_string(entry.second - 1) + " more times]");
        }
    }
    counts.clear();
}

} // namespace

void set_warning_handler(WarningHandler handler) {
    std::lock_guard<std::mutex> lock(warning_mutex());
    // Flush first, so a pending count reaches the handler that saw the
    // occurrence it counts rather than the one replacing it.
    flush_locked();
    warning_handler() = std::move(handler);
}

void emit_warning(const std::string& message) {
    std::lock_guard<std::mutex> lock(warning_mutex());
    auto& counts = warning_counts();

    auto it = counts.find(message);
    if (it != counts.end()) {
        ++it->second;
        return;
    }

    if (counts.size() < MAX_TRACKED_MESSAGES) counts.emplace(message, 1);
    deliver_locked(message);
}

void flush_warnings() {
    std::lock_guard<std::mutex> lock(warning_mutex());
    flush_locked();
}

} // namespace lindblad
