// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#pragma once

#include "lindblad/constants.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <new>

#if defined(_MSC_VER)
#  include <intrin.h>
#endif

namespace lindblad {

// =============================================================================
// Cross-platform macros for compiler intrinsics
// =============================================================================

// Population count for 64-bit integers — used in Pauli expectation values
#if defined(_MSC_VER)
#  define LINDBLAD_POPCOUNT64(x) static_cast<int>(__popcnt64(x))
#else
#  define LINDBLAD_POPCOUNT64(x) __builtin_popcountll(x)
#endif

// Note on OpenMP pragmas:
// MSVC uses OpenMP 2.0 (legacy) and silently ignores unrecognised pragma clauses
// like the aligned() clause on #pragma omp simd (OpenMP 4.0+).
// We use conditional compilation to remove the pragma on MSVC; GCC/Clang use it.

// =============================================================================
// is_finite_strict / quiet_nan_strict - non-finite detection by bit pattern
// =============================================================================
// Finiteness read straight off the IEEE-754 exponent field (finite iff those
// eleven bits are not all ones), and a quiet NaN built from the matching
// pattern. One integer compare each, no <cmath> call, and no dependence on how
// the standard library spells the classification.
//
// Both are correct only while the build omits -ffinite-math-only. Under that
// flag clang attaches nofpclass(nan inf) to floating-point parameters and
// return values, so a NaN crossing ANY function boundary is poison, and poison
// reads back as whatever occupies the register, inconsistently between uses.
// The caller forms that poison before the call, so the body below cannot
// recover it however it is written. The CMake flag list therefore pairs
// -ffast-math with -fno-finite-math-only.
//
// Use is_finite_strict for every guard whose job is to DETECT a non-finite
// value, and quiet_nan_strict for every marker whose job is to BE detectably
// non-finite. Do not use +/-infinity as a comparison seed (best-so-far, lower
// bound); an explicit bool flag is correct there, since a seed exists precisely
// to lose its first comparison.
inline bool is_finite_strict(double x) noexcept {
    std::uint64_t bits;
    std::memcpy(&bits, &x, sizeof bits);
    return (bits & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
}

// Built from the bit pattern rather than from
// std::numeric_limits<double>::quiet_NaN(), so the marker does not depend on
// the library constant being materialised.
inline double quiet_nan_strict() noexcept {
    constexpr std::uint64_t kQuietNaNBits = 0x7FF8000000000000ULL;
    double x;
    std::memcpy(&x, &kQuietNaNBits, sizeof x);
    return x;
}

// =============================================================================
// is_normalizable - is there a norm here that can be divided out?
// =============================================================================
// Normalising a state divides every amplitude by its norm (a density matrix, by
// its trace). That is meaningful only while the norm is a real positive number
// carrying information, and two states have no repair at all:
//
//   - the zero state: there is no direction to normalize toward, and the
//     division is by zero;
//   - a non-finite state: the norm is NaN, and dividing by it spreads the NaN
//     across every amplitude rather than removing it.
//
// The floor is machine epsilon rather than a chosen decimal. Epsilon is the
// spacing of doubles at 1.0, so a norm at or below it is not distinguishable
// from zero at the scale a normalized state occupies; scaling such a state up
// to unit norm would manufacture a direction out of rounding error rather than
// recover one. Callers pass the norm, not its square.
inline bool is_normalizable(double norm) noexcept {
    return is_finite_strict(norm) &&
           norm > std::numeric_limits<double>::epsilon();
}

// =============================================================================
// Complex128 — SIMD-friendly complex number with explicit memory layout
// =============================================================================
// Do NOT use std::complex<double> for the hot path.
// It generates suboptimal SIMD code due to strict aliasing rules.

struct alignas(16) Complex128 {
    double real = 0.0;
    double imag = 0.0;

    constexpr Complex128() noexcept = default;
    constexpr Complex128(double r, double i) noexcept : real(r), imag(i) {}
    constexpr explicit Complex128(double r) noexcept : real(r), imag(0.0) {}

    // Arithmetic
    inline constexpr Complex128 operator+(const Complex128& other) const noexcept {
        return {real + other.real, imag + other.imag};
    }

    inline constexpr Complex128 operator-(const Complex128& other) const noexcept {
        return {real - other.real, imag - other.imag};
    }

    inline constexpr Complex128 operator*(const Complex128& other) const noexcept {
        return {
            real * other.real - imag * other.imag,
            real * other.imag + imag * other.real
        };
    }

    inline constexpr Complex128 operator*(double scalar) const noexcept {
        return {real * scalar, imag * scalar};
    }

    inline constexpr Complex128 operator/(double scalar) const noexcept {
        return {real / scalar, imag / scalar};
    }

    inline constexpr Complex128& operator+=(const Complex128& other) noexcept {
        real += other.real;
        imag += other.imag;
        return *this;
    }

    inline constexpr Complex128& operator-=(const Complex128& other) noexcept {
        real -= other.real;
        imag -= other.imag;
        return *this;
    }

    inline constexpr Complex128& operator*=(const Complex128& other) noexcept {
        double r = real * other.real - imag * other.imag;
        double i = real * other.imag + imag * other.real;
        real = r;
        imag = i;
        return *this;
    }

    inline constexpr Complex128& operator*=(double scalar) noexcept {
        real *= scalar;
        imag *= scalar;
        return *this;
    }

    inline constexpr Complex128 operator-() const noexcept {
        return {-real, -imag};
    }

    inline constexpr bool operator==(const Complex128& other) const noexcept {
        return real == other.real && imag == other.imag;
    }

    inline constexpr bool operator!=(const Complex128& other) const noexcept {
        return !(*this == other);
    }

    // Conjugate
    inline constexpr Complex128 conj() const noexcept {
        return {real, -imag};
    }

    // Magnitude squared
    inline constexpr double norm_sq() const noexcept {
        return real * real + imag * imag;
    }

    // Magnitude
    inline double norm() const noexcept {
        return std::sqrt(norm_sq());
    }

    // Phase angle
    inline double arg() const noexcept {
        return std::atan2(imag, real);
    }

    // Polar form: r * exp(i*theta)
    static inline Complex128 polar(double r, double theta) noexcept {
        return {r * std::cos(theta), r * std::sin(theta)};
    }

    // exp(i*theta)
    static inline Complex128 exp_i(double theta) noexcept {
        return {std::cos(theta), std::sin(theta)};
    }
};

// Free function scalar * Complex128
inline constexpr Complex128 operator*(double scalar, const Complex128& c) noexcept {
    return {scalar * c.real, scalar * c.imag};
}

// =============================================================================
// Aligned memory allocation — cross-platform wrappers
// =============================================================================
// 64-byte alignment required for AVX-512

constexpr size_t CACHE_LINE_SIZE = 64;

// alignment = a power of two, at least CACHE_LINE_SIZE
inline double* aligned_alloc_doubles(size_t count, size_t alignment = CACHE_LINE_SIZE) {
    void* ptr = nullptr;
    size_t bytes = count * sizeof(double);
    // Ensure allocation is a multiple of alignment
    if (bytes % alignment != 0) {
        bytes = ((bytes / alignment) + 1) * alignment;
    }
#if defined(_MSC_VER)
    ptr = _aligned_malloc(bytes, alignment);
#else
    ptr = std::aligned_alloc(alignment, bytes);
#endif
    if (!ptr) throw std::bad_alloc();
    return static_cast<double*>(ptr);
}

inline void aligned_free(void* ptr) noexcept {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

// =============================================================================
// SVDMethod - which factorisation a bond split asks for
// =============================================================================
// The values name ALGORITHMS, prefixed by provider only where two providers
// offer the same algorithm. Unprefixed names are autonne's kernels
// (shek014/autonne, fetched by the build); the Eigen-prefixed names are
// Eigen's, kept selectable so a caller can hold the two providers against
// each other on their own workload.
//
// BDC is the DEFAULT on both the qubit and qudit MPS. autonne's
// divide-and-conquer runs Householder bidiagonalisation and Gu-Eisenstat's
// secular-equation merge with deflation, is O(n^3) with a constant the spectrum
// barely moves, and returns orthonormal factors by construction. Its accuracy
// promise is absolute, |s_i - s_i(true)| <= 64 * max(rows, cols) * eps * s_max,
// which is what a bond split truncating on weight needs; measured on a 128x128
// decaying spectrum it is 2.7x faster than Jacobi and no faster on a flat one.
//
// Jacobi is autonne's one-sided cyclic Jacobi. Slower as the block grows and
// the spectrum decays, but RELATIVELY accurate on every singular value down to
// a column floor of 2^-500, so it resolves the tail of a graded spectrum that
// an absolute bound treats as noise. It is also the ladder's first rescue when
// the primary factorisation fails verification (see detail/svd_truncate.hpp),
// because it reaches the same answer by an independent road.
//
// EigenJacobi and EigenBDC are Eigen's JacobiSVD and BDCSVD, run through the
// strict-FP translation unit in src/eigen_backend.cpp. Every kernel is
// available in every build, so the public API does not change shape with the
// build configuration. Selecting either Jacobi emits a one-time note per MPS
// layer that it is the slower algorithm, and selecting EigenBDC emits one
// saying it is in force, since it is not the default either.
enum class SVDMethod { Jacobi, BDC, EigenJacobi, EigenBDC };

// The enumerator's name, for diagnostics.
constexpr const char* to_string(SVDMethod m) noexcept {
    switch (m) {
        case SVDMethod::Jacobi:      return "Jacobi";
        case SVDMethod::BDC:         return "BDC";
        case SVDMethod::EigenJacobi: return "EigenJacobi";
        case SVDMethod::EigenBDC:    return "EigenBDC";
    }
    return "SVDMethod(?)";
}

// =============================================================================
// SvdRejection - what a bond split does with a factorisation it rejects
// =============================================================================
// Every factorisation a bond split receives is verified before it is used (see
// detail/svd_truncate.hpp): its kept factors must be finite and must rebuild
// the block to within the truncation it reports plus the backward error a
// stable SVD is allowed. A kernel can also decline a block and return nothing.
//
// Fix, the DEFAULT, repairs a rejection by descending the ladder: autonne's
// Jacobi (skipped when Jacobi was the kernel that failed), then the Gram route
// when svd_accept_gram allows it, each candidate verified the same way. A
// split that no permitted rung repairs throws. A repaired split is as
// trustworthy as one the kernel got right, so the answer is intact.
//
// Throw ends the run with RuntimeFailure at the first rejection, for a caller
// who would rather stop than take a tensor from a kernel they did not name.
//
// Ignore uses a rejected factorisation as it is. The state built from it is
// UNVERIFIED and can be wrong with nothing further to say so; the fidelity
// figures are withdrawn, since they would describe a factorisation that did
// not happen. A kernel that returned nothing still leaves nothing to use, so
// that case descends the ladder as under Fix, and whatever the first rung to
// produce a factorisation gives is used, verified or not. A non-finite entry
// carried into the chain is still refused where every state's is: at a
// collapse or a sample, and in the final state.
//
// Throw and Ignore are not the default, and each emits a one-time note per MPS
// layer saying so.
enum class SvdRejection { Fix, Throw, Ignore };

constexpr const char* to_string(SvdRejection r) noexcept {
    switch (r) {
        case SvdRejection::Fix:    return "Fix";
        case SvdRejection::Throw:  return "Throw";
        case SvdRejection::Ignore: return "Ignore";
    }
    return "SvdRejection(?)";
}

// =============================================================================
// SvdReport - how a repaired or ignored rejection is reported
// =============================================================================
// Read only after SvdRejection::Fix repaired a factorisation or Ignore used one;
// Throw ends the run before there is anything to report. Warn, the DEFAULT,
// emits one warning per rung taken and per factorisation used unverified,
// naming the layer, the block shape, the kernel and what verification found.
// Silent emits none. Either way every repair and every unverified use is
// counted on the chain (jacobi_rescue_count(), gram_fallback_count(),
// ignored_rejection_count()). Silent is not the default and emits a one-time
// note per MPS layer saying so.
enum class SvdReport { Warn, Silent };

constexpr const char* to_string(SvdReport r) noexcept {
    switch (r) {
        case SvdReport::Warn:   return "Warn";
        case SvdReport::Silent: return "Silent";
    }
    return "SvdReport(?)";
}

// Default weight cutoff of both MPS layers: a bond split may discard at most
// this fraction of its block's weight (Σ sigma²). It sits at the relative
// accuracy to which a double-precision sum of weights is known, so at the
// default a split removes exact rank deficiency and weight its block's own
// total cannot distinguish from rounding, and does not compress. A caller who
// wants compression raises the cutoff, and CanonicalForm::Auto reads any value
// above this one as that request.
inline constexpr double MPS_DEFAULT_CUTOFF = 1e-16;

// =============================================================================
// CanonicalForm - when an MPS restores canonical gauge before a bond split
// =============================================================================
// A bond split truncates on the singular values of a two-site block. Those are
// the state's Schmidt coefficients only in mixed canonical form centred on the
// block: every site to its left left-orthonormal, every site to its right
// right-orthonormal. In any other gauge the block's spectrum is weighted by the
// environments on either side, so a cap or a cutoff applied to it keeps and
// drops the wrong directions, and the weight it reports discarding is not a
// fraction of the state.
//
// Both MPS layers track which sites are orthonormal (MPSState::open_span) and
// move the orthogonality centre with QR and LQ steps, which never truncate.
// The policy decides which splits pay for that move.
//
//   Always = every split. The default.
//   Auto   = only the splits that can discard real weight: those where the
//            bond cap can bind, because the block's rank bound
//            min(d chi_L, d chi_R) exceeds max_bond_dim, and every split once
//            the cutoff is above MPS_DEFAULT_CUTOFF. Any other split runs where
//            the chain stands and skips the QR steps. Its truncation is still
//            sound, since it drops only exact rank deficiency (the same in
//            every gauge) and weight below the resolution of its block's
//            total, but it keeps too much: a direction that is rounding noise
//            in the state can carry real weight in a block weighted by an
//            uncentred environment, and survives the cutoff there.
//
// That surplus is why Always is the default. The bonds Auto keeps grow wider
// than the state needs, and the extra rank costs more in every later split
// than the QR steps it saved. On a 16-qubit brickwork at an exact cap, Auto
// ends with a middle bond of 256 where Always ends with 143 at the same
// fidelity, and runs 1.9x slower; where the cap binds, the two agree to within
// a few percent. Auto stays selectable for a workload that measures otherwise.
//
// Measurement, reset and sampling move the centre under both policies: each
// reads one site's marginals, which is a local read only at the centre.
enum class CanonicalForm { Auto, Always };

// The enumerator's name, for diagnostics.
constexpr const char* to_string(CanonicalForm f) noexcept {
    switch (f) {
        case CanonicalForm::Auto:   return "Auto";
        case CanonicalForm::Always: return "Always";
    }
    return "CanonicalForm(?)";
}

// =============================================================================
// UncheckedGates - what an MPS does with a gate no policy measured
// =============================================================================
// Both MPS layers keep two records that let them answer cheaply, and both rest
// on every gate being unitary:
//
//   - which sites are orthonormal (MPSState::open_span). A read of the norm, a
//     marginal, a measurement or a sample touches only the centre because every
//     other site is known orthonormal; a non-unitary matrix on such a site
//     breaks that, and every later read returns a number that is not the
//     chain's.
//   - the fidelity ledger (detail::FidelityLedger). Its bound is derived for
//     unitary gates between splits; after a non-unitary one its figures no
//     longer mean anything.
//
// So after every gate the chain needs to know whether the matrix it applied is
// unitary to DEFAULT_PHYSICAL_ATOL. When it is not, a single-site gate on a site
// outside the open span widens the span over that site, so the next move of
// the centre sweeps it, and any gate invalidates the ledger, as a collapse
// does. The matrix is still applied exactly as given and nothing is reported:
// this is the chain keeping its own records, not a judgement of the operand.
//
// Under Throw, Warn or Repair::Attempt the policy measures the matrix anyway,
// so the answer is there whatever this setting says. The one policy that
// measures nothing, Ignore with Repair::None, leaves it to this setting:
//
//   Track         = measure the matrix for the records. The default: the
//                   caller asked for the operand not to be judged, not for the
//                   chain's later reads to be wrong.
//   AssumeUnitary = take the matrix to be unitary and measure nothing. The
//                   cheapest route, for a caller who knows their matrices are
//                   unitary; if one is not, the chain's reads and fidelity
//                   figures come out wrong and the caller has accepted that.
//
// A collapse is not a gate and this setting does not reach it: the projection
// onto an outcome is the library's own non-unitary step, and it invalidates the
// ledger under both. Gates the library builds itself are unitary by
// construction and are never measured.
enum class UncheckedGates { Track, AssumeUnitary };

// The enumerator's name, for diagnostics.
constexpr const char* to_string(UncheckedGates g) noexcept {
    switch (g) {
        case UncheckedGates::Track:         return "Track";
        case UncheckedGates::AssumeUnitary: return "AssumeUnitary";
    }
    return "UncheckedGates(?)";
}

// =============================================================================
// PrefixReuse - whether a dense per-shot run keeps a copy of its shared start
// =============================================================================
// A run that samples shot by shot (a mid-circuit measurement, feedforward, or
// a RESET that can change the state) repeats, in every shot, everything before
// its first MEASURE, RESET or conditioned instruction. That stretch draws
// nothing, so it can run once into a snapshot every shot copies, with the
// random stream, and so the seeded counts, exactly those of a rerun. What the
// snapshot costs is one more full state held for the run: 16 * 2^n bytes for
// the statevector simulator, 16 * 4^n for the density-matrix simulator. The
// MPS and Clifford backends take their snapshot whenever no observer is
// attached, since a chain or a tableau is small beside the gates it saves.
//
//   Hardware = take it when the memory the operating system reports available
//              at that moment (hw::available_memory_bytes) is at least twice
//              the snapshot, so the machine keeps as much free as the copy
//              takes, and the run still fits its memory cap (max_memory_mb,
//              or the automatic cap when that is 0). No reading means no
//              snapshot. The default.
//   Manual   = take it when the run, snapshot included, fits its memory cap.
//   Off      = never; every shot reruns its prefix.
//
// An observed run never takes one, whatever this says: anchors inside the
// prefix fire once per shot with that shot's index.
enum class PrefixReuse { Hardware, Manual, Off };

// The enumerator's name, for diagnostics.
constexpr const char* to_string(PrefixReuse r) noexcept {
    switch (r) {
        case PrefixReuse::Hardware: return "Hardware";
        case PrefixReuse::Manual:   return "Manual";
        case PrefixReuse::Off:      return "Off";
    }
    return "PrefixReuse(?)";
}

// =============================================================================
// ShotMemory - how a run's shot threads share its memory cap
// =============================================================================
// A run that walks its shots one by one runs them across a team of threads,
// each holding its own working state (and its own scratch) beside what the run
// already holds. max_parallel_shots chooses the team (0 = automatic, 1 =
// serial, k >= 2 = at most k); this chooses how the team meets the run's memory
// cap (max_memory_mb, or the automatic cap when that is 0).
//
//   Shrink        = the team shrinks to as many working states as fit beside
//                   what the run holds, never below one. A run that fits
//                   serially is never refused because of its threads. The
//                   default.
//   FixedShare    = the team is the one asked for, and the cap left after what
//                   the run holds is split evenly across it: every thread is
//                   held to its own share, and a thread whose state outgrows
//                   its share fails the run. For a backend whose state has a
//                   fixed size, a share smaller than one state is refused
//                   before the first shot.
//   SerialIfShort = the whole team asked for fits, or the run is serial.
//   Exact         = exactly the team asked for (max_parallel_shots must then
//                   be set), refused before the first shot when its states do
//                   not fit.
//
// The counts a seeded run produces do not depend on the team, so this changes
// only how fast a run goes and how much memory it holds.
enum class ShotMemory { Shrink, FixedShare, SerialIfShort, Exact };

// The enumerator's name, for diagnostics.
constexpr const char* to_string(ShotMemory m) noexcept {
    switch (m) {
        case ShotMemory::Shrink:        return "Shrink";
        case ShotMemory::FixedShare:    return "FixedShare";
        case ShotMemory::SerialIfShort: return "SerialIfShort";
        case ShotMemory::Exact:         return "Exact";
    }
    return "ShotMemory(?)";
}

// =============================================================================
// ShotCounting - how a run sampled shot by shot keeps its counts
// =============================================================================
// Every mode gives the same counts; only the bookkeeping, and so the speed,
// differs. Shots are counted per chunk (the chunks below) and the chunks'
// counts are merged in chunk order under one lock, so on a team the merge is
// serial work as long as the outcomes are many. Two costs trade off:
//
//   - an integer key (the register packed into 64 bits) is cheap to count
//     and to merge, but its string is built when the run hands its counts
//     back, on one thread, once per distinct outcome;
//   - a string key costs a string per shot, built on the shot's own thread,
//     and merges by moving the string's entry, so the serial merge builds
//     nothing.
//
// Integer keys win when outcomes repeat (few strings to build at the end);
// string keys win when nearly every outcome is distinct (the end would build
// one per shot, serially).
//
//   Auto           = chosen per chunk by each shot thread from its last chunk:
//                    when more than shot_counting_distinct_fraction of that
//                    chunk's shots were distinct outcomes, its counts go under
//                    their strings at once (on that thread) and the next chunk
//                    is counted under strings; otherwise under integers. A
//                    thread's first chunk is counted under integers. The
//                    default.
//   Packed         = integer keys throughout; the strings are built at the
//                    end.
//   PackedPerChunk = integer keys within a chunk; each shot thread puts its
//                    chunk under strings when the chunk ends, so the merge
//                    builds none. A run on one thread builds them at the end.
//   Strings        = a string key for every shot.
//
// A register wider than 64 clbits is counted under strings in every mode.
enum class ShotCounting { Auto, Packed, PackedPerChunk, Strings };

// The enumerator's name, for diagnostics.
constexpr const char* to_string(ShotCounting m) noexcept {
    switch (m) {
        case ShotCounting::Auto:           return "Auto";
        case ShotCounting::Packed:         return "Packed";
        case ShotCounting::PackedPerChunk: return "PackedPerChunk";
        case ShotCounting::Strings:        return "Strings";
    }
    return "ShotCounting(?)";
}

// ShotCounting::Auto counts a thread's next chunk under strings when more than
// this fraction of its last chunk's shots were distinct outcomes. Options::
// shot_counting_distinct_fraction = 0 selects it; a caller's value lies in
// (0, 1]. At 1 a chunk can never exceed it, so Auto counts as Packed.
//
// The value is measured, on a Ryzen 9 7900X (24 threads, pinned, teams of 1, 6,
// 12 and 24) over 14 runs: statevector and density-matrix rounds, density-matrix
// terminal sampling, and Clifford Slab, PerShot and mid-circuit runs, with 16 to
// 10^6 distinct outcomes. Auto is within 5% of the fastest of the four modes on
// 54 of the 56 run and team pairs.
//
// Two limits are known. On a team, a chunk of SHOT_CHUNK_TARGET shots is judged
// by its own distinct outcomes, which overstates how distinct the whole run is:
// a statevector run of 10^5 shots with 2413 distinct outcomes looks about 90%
// distinct to each chunk, so it is counted under strings, which costs 20% on a
// 24-thread team of a 10 ms run (9.9 ms against 11.8 ms) and 1% or less on
// teams of 6 and 12. Setting ShotCounting::Packed removes it. And on a
// 24-thread Clifford Slab run of 10^5 shots at 50 qubits, PackedPerChunk is 13%
// faster than Auto (12.7 ms against 14.4 ms), both well ahead of the Packed
// merge.
inline constexpr double SHOT_COUNTING_DISTINCT_FRACTION = 0.5;

// =============================================================================
// Shot chunks - the unit a shot thread takes, and its random stream
// =============================================================================
// The shots of a run are cut into chunks of
//
//   chunk = ceil(shots / SHOT_CHUNK_TARGET)
//
// consecutive shots, so a run has at most SHOT_CHUNK_TARGET chunks. Each chunk
// draws from its own random stream (ShotRng in shot_rng.hpp), started from the
// run's seed and the chunk's index, and a thread always runs a whole chunk, so
// a seeded run gives the same counts on any number of threads: the chunks and
// their streams depend on the shot count only.
//
// The value is measured. When a run's shots run on more than one thread, every
// chunk's counts are merged in chunk order under one lock, which cost 0.5 to
// 0.7 us of wall time per chunk on a 24-thread machine whatever the chunk's
// size, so the chunk count is a fixed cost of a parallel run: about 0.35 ms at
// 512. 512 chunks still balance a team well: 21 or 22 chunks per thread at 24
// threads, and whole numbers of chunks per thread up to 128 threads. A run on
// one thread keeps one counts map across its chunks and pays nothing per
// chunk.
inline constexpr int SHOT_CHUNK_TARGET = 512;

// =============================================================================
// Automatic shot teams - when a run's shots go across threads
// =============================================================================
// With max_parallel_shots = 0, a run's shots run across threads only when at
// least SHOT_TEAM_MIN_CACHED_STATES of its per-thread working states fit in one
// last-level-cache instance (hw::llc_bytes); the team is then sized by the
// run's work, as a gate kernel's is. Below that, each shot thread streams its
// own state from memory with its gates on one thread, and the run is faster
// with its shots on one thread and every gate threaded.
//
// The value is measured: on a 32 MiB-L3 part with 24 threads, statevector runs
// sampled shot by shot (a mid-circuit measurement, and three rounds of
// measuring every qubit) ran 10 to 18 times faster across shot threads up to
// 16 qubits and 1.3 to 1.7 times faster at 18 qubits (8 states per cache), but
// 0.4 to 0.9 times as fast at 19 (4 states) and 0.2 to 0.4 at 20 (2), whatever
// the team. At 17 and 18 qubits teams larger than the cache holds still ran
// faster than ones it holds, so the cache decides whether the shots go across
// threads, not how many threads they take.
inline constexpr std::size_t SHOT_TEAM_MIN_CACHED_STATES = 8;

// =============================================================================
// QubitLimit - how wide a register a dense representation may be
// =============================================================================
// Enforce keeps the default ceilings, sized so a dense state fits a
// workstation. Lift raises the statevector ceilings to LIFTED_MAX_QUBITS, the
// widest statevector whose byte count (16 * 2^n) fits 64 bits, and the MPS
// dense fallback to LIFTED_MPS_DENSE_MAX_QUBITS; nothing lifts either.
// Lifting never bypasses the memory check.
//
// The MPS ceiling is lower because the fallback rebuilds the chain by
// factorising a 2 x 2^(n-1) block, and the factorisation takes its dimensions
// as int: 2^30 columns (n = 31) is the widest block it can address.
enum class QubitLimit { Enforce, Lift };

inline constexpr int ENFORCED_MAX_QUBITS = 30;
inline constexpr int ENFORCED_MPS_DENSE_MAX_QUBITS = 25;
inline constexpr int LIFTED_MAX_QUBITS = 59;
inline constexpr int LIFTED_MPS_DENSE_MAX_QUBITS = 31;

// The widest statevector a limit allows, and the widest register the MPS
// backend may expand into a dense array (a gate over three or more qubits, a
// statevector read).
inline constexpr int max_statevector_qubits(QubitLimit limit) {
    return limit == QubitLimit::Lift ? LIFTED_MAX_QUBITS : ENFORCED_MAX_QUBITS;
}
inline constexpr int max_mps_dense_qubits(QubitLimit limit) {
    return limit == QubitLimit::Lift ? LIFTED_MPS_DENSE_MAX_QUBITS
                                     : ENFORCED_MPS_DENSE_MAX_QUBITS;
}

// max_memory_mb values. 0 means automatic (the machine's available memory,
// else FALLBACK_MEMORY_CAP_MB). NO_MEMORY_CAP is only ever passed by a
// caller, to say there is no cap.
inline constexpr std::uint64_t NO_MEMORY_CAP = std::numeric_limits<std::uint64_t>::max();
inline constexpr std::uint64_t FALLBACK_MEMORY_CAP_MB = 4096;

// =============================================================================
// Statevector parallel grain - how much work a thread needs before it helps
// =============================================================================
// A statevector gate kernel runs in parallel only when every thread it uses
// gets at least this many amplitudes of the state, and it uses
// min(available threads, 2^n / grain) threads; one thread's worth runs
// serially, without starting a parallel region. Starting a region costs a
// fixed time that grows with its thread count, so the rule spends threads
// only where each one's share of the sweep outweighs it, and uses fewer
// threads where a full team would cost more to start than it saves.
//
// The value is measured: the cheapest corpus kernel (a controlled phase,
// which touches a quarter of the amplitudes with two multiplies each) gained
// from threads from about 1365 amplitudes per thread, at both 12 and 24
// threads; 2048 is that rounded up to a power of two. Every other kernel does
// more work per amplitude and crosses earlier.
//
// StatevectorSimulator::Options::min_amplitudes_per_thread overrides it for a
// run (0 keeps this default).
inline constexpr std::size_t DEFAULT_MIN_AMPLITUDES_PER_THREAD = 2048;

// =============================================================================
// PhaseTimes - where one run() spent its wall-clock time
// =============================================================================
// Filled only when the caller asks (a backend's Options::record_phase_times);
// a run that is not asked reads no clock for it. The five phases tile the run
// from entry to the point its result is complete, with no gaps and no overlap,
// so total() is that whole span. Seconds, from std::chrono::steady_clock.
//
//   setup      = everything before the circuit's first instruction can run:
//                validation, preflight, the memory budget, a repaired copy,
//                the working state and its initial value, RNG seeding.
//   fusion     = the gate-fusion pre-pass. Exactly 0 when it does not engage.
//   evolution  = applying instructions to the state: the single forward pass,
//                every shot's trajectory (prefix snapshot, restores, collapses
//                included), or the single seeded trajectory; and the
//                observation harness that rides on them.
//   sampling   = turning states into counts: the sampler's block sums, draws,
//                bitstrings and the counts map. A run sampled shot by shot on
//                one thread charges each shot's bitstring and count here, so
//                asking costs two clock reads per shot. When its shots run on
//                several threads, the whole shot loop is charged to evolution
//                and none of it here: one clock cannot tile several threads'
//                time.
//   final_copy = the checks on the finished state and handing it to the
//                result (the working buffer is moved, not copied).
//
// A backend's own simulation_time_seconds is a different span (it starts after
// option validation and stops before final_copy), so the two are not
// interchangeable.
struct PhaseTimes {
    double setup = 0.0;
    double fusion = 0.0;
    double evolution = 0.0;
    double sampling = 0.0;
    double final_copy = 0.0;

    double total() const noexcept {
        return setup + fusion + evolution + sampling + final_copy;
    }
};

// Mathematical constants (PI, INV_SQRT2, ...) live in constants.hpp, included
// at the top of this header, so every one of those names is visible to anything
// including types.hpp.

} // namespace lindblad
