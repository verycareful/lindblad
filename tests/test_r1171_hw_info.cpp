// =============================================================================
// R.1.17.1 test suite — hw::llc_bytes() (R1171HwInfo)
// =============================================================================
// R.1.17.0 added lindblad::hw::llc_bytes() (include/lindblad/hw_info.hpp): the
// size in bytes of ONE last-level cache instance, 0 when the platform cannot
// report it, detected once and cached. It is the input to the statevector
// fusion AUTO engagement point (first n whose 16·2^n-byte statevector exceeds
// one LLC instance, clamped so the effective LLC is at least 1 MiB).
//
// These tests are cross-platform: they hardcode NO cache size. They pin the
// documented contract (determinism, the "0 or a plausible size" envelope) and
// the arithmetic the fusion path derives from the value, and they PRINT the
// detected value so CI logs record per-machine reality (house precedent: the
// visualiser/frontier suites print verdicts). Every assertion holds whether the
// machine has 1 MiB or 512 MiB of L3, or reports nothing at all.

#include <gtest/gtest.h>

#include "lindblad/hw_info.hpp"

#include <algorithm>
#include <cstddef>
#include <iostream>

using namespace lindblad;

namespace {

// Envelope for a "plausible" per-instance L3 size. Real parts range from a few
// hundred KiB (embedded / small cores) to a few hundred MiB (server CCX); the
// bounds are deliberately loose so no real hardware trips them, while still
// catching a wildly-wrong detection (e.g. a byte/KiB unit slip, or a negative
// value reinterpreted as a huge size_t).
constexpr std::size_t kPlausibleMin = std::size_t(256) << 10;  // 256 KiB
constexpr std::size_t kPlausibleMax = std::size_t(4) << 30;    // 4 GiB

// Mirror of the fusion auto-threshold arithmetic in statevector_sim.cpp: the
// effective LLC is clamped to [1 MiB, 1 GiB], then the engagement point is the
// first n whose statevector (16·2^n bytes, SoA double re+im per amplitude)
// STRICTLY exceeds it. Computed on the TEST side so the clamp contract is
// pinned without exposing the internal helper.
constexpr std::size_t kBytesPerAmp = 16;
constexpr std::size_t kLlcClampMin = std::size_t(1) << 20;  // 1 MiB
constexpr std::size_t kLlcClampMax = std::size_t(1) << 30;  // 1 GiB

int implied_auto_threshold(std::size_t llc) {
    const std::size_t clamped = std::clamp(llc, kLlcClampMin, kLlcClampMax);
    int n = 1;
    while ((kBytesPerAmp << n) <= clamped) ++n;  // first n: 16·2^n > clamped
    return n;
}

}  // namespace

// Determinism / caching: the value is detected once and cached, so repeated
// calls must agree exactly (no re-probe drift, no per-call recomputation).
TEST(R1171HwInfo, DeterministicAcrossCalls) {
    const std::size_t a = hw::llc_bytes();
    const std::size_t b = hw::llc_bytes();
    const std::size_t c = hw::llc_bytes();
    EXPECT_EQ(a, b);
    EXPECT_EQ(b, c);
}

// Sanity envelope: the documented "unknown" sentinel is 0; any non-zero value
// must be a plausible per-instance L3 size. Print the detected value so the CI
// log records what THIS machine reported.
TEST(R1171HwInfo, PlausibleEnvelopeAndPrint) {
    const std::size_t llc = hw::llc_bytes();
    std::cout << "[ R1171HwInfo ] llc_bytes() detected: " << llc << " bytes";
    if (llc == 0) {
        std::cout << " (unknown — documented valid return)";
    } else {
        std::cout << " (~" << (llc >> 20) << " MiB)";
    }
    std::cout << std::endl;

    if (llc != 0) {
        EXPECT_GE(llc, kPlausibleMin);
        EXPECT_LE(llc, kPlausibleMax);
    } else {
        SUCCEED() << "llc_bytes() returned 0 (unknown); fusion falls back to 32 MiB";
    }
}

// Consistency with the fusion engagement contract: whatever the machine
// reports, the implied auto threshold never drops below 17 qubits once the
// (clamped) LLC is at least 1 MiB — the guarantee small-n tests rely on to
// stay on the unfused path by default on ANY hardware. Also verify the clamp
// floor directly: an implausibly tiny detection value still yields ≥ 17.
TEST(R1171HwInfo, ImpliedAutoThresholdRespectsClampFloor) {
    const std::size_t llc = hw::llc_bytes();

    // The clamp floor pins the minimum: 16·2^n > 1 MiB ⇒ 2^(n+4) > 2^20 ⇒ n ≥ 17.
    EXPECT_GE(implied_auto_threshold(kLlcClampMin), 17);
    // Even a garbage sub-floor detection (e.g. 4 KiB) is clamped up to 1 MiB.
    EXPECT_GE(implied_auto_threshold(std::size_t(4) << 10), 17);
    // The real detected value (when known) also honours the floor. When
    // unknown (0), the fallback is 32 MiB, whose implied threshold is ≥ 17 too.
    const std::size_t effective = (llc == 0) ? (std::size_t(32) << 20) : llc;
    EXPECT_GE(implied_auto_threshold(effective), 17);

    std::cout << "[ R1171HwInfo ] implied auto fusion threshold: "
              << implied_auto_threshold(effective) << " qubits" << std::endl;
}
