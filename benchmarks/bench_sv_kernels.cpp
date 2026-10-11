// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// =============================================================================
// bench_sv_kernels.cpp - statevector gate kernels by target qubit and threads
// =============================================================================
// The cross-engine suite times whole circuits, which mix gates on every qubit,
// so a kernel that is fast on low qubits and starved on high ones shows up only
// as an average. A kernel's cost depends on WHICH qubits it touches: the
// stride between the amplitudes it pairs is 2^q, which decides how the work
// splits across threads and how much of a SIMD vector one pair fills. This
// binary times each kernel the comparison corpus uses at every target position,
// on all threads and on one, at three register sizes:
//
//   n = 18  state 4 MiB, cache-resident on one core
//   n = 22  state 64 MiB, the size of both L3 instances of a two-CCD part
//   n = 26  state 1 GiB, DRAM-resident
//
// Every case is checked before it is timed. A kernel that does nothing keeps
// the norm exactly, so a norm check cannot see it; instead the gate is applied
// once and kCheckSamples amplitudes are compared with an independent reference
// that recomputes them from the gate's own matrix by bit arithmetic. A case
// that disagrees fails with an error instead of reporting a time. After the
// timed loop the norm is checked against a bound that grows with the number of
// applications, which catches a kernel that corrupts the state slowly.
//
// A check nobody has seen fail is not known to work, so every run also carries
// selftest cases: deliberately wrong kernels (one that does nothing, one on the
// wrong qubit, one with its angles swapped) that the reference check MUST
// reject. A selftest the check accepts is an error.
//
// Names, for --benchmark_filter:
//
//   kernel/<gate>/n<N>/q/<q>/<threads>            one-qubit gates: q = 0..n-1
//   kernel/<gate>/n<N>/hi/<q>/<threads>           two-qubit gates on (0, q),
//                                                 three-qubit on (0, 1, q)
//   kernel/<gate>/n<N>/lo/<q>/<threads>           two-qubit gates on (q, n-1),
//                                                 three-qubit on (q, n-2, n-1)
//   kernel/unitary_k<k>/n<N>/block/<p>/<threads>  a dense k-qubit block on
//                                                 targets p .. p+k-1
//   kernel/unitary_k<k>/n<N>/spread/0/<threads>   targets spread over 0 .. n-1
//   roofline/<sweep|triad>/n<N>/<threads>         this machine's ceilings
//   selftest/<fault>                              the check, shown to bite
//
// <threads> is "all" (the OpenMP default when the binary starts) or "one".
// The hi sweeps vary the highest operand, which decides how a kernel splits
// across threads; the lo sweeps vary the lowest, which decides how much of a
// vector a pair fills. The fused unitary is the kernel gate fusion emits.
//
// The roofline cases bound what any kernel can reach on this machine at each
// size. "sweep" is an in-place pass over a state's two planes doing a phase
// rotation, the cheapest full-state kernel there is; "triad" is the STREAM
// triad a = b + s*c over three arrays of 2^n doubles, for comparison with
// published figures. Both report bytes moved per second.
//
// The full set is long (tens of minutes). Filter to what a change touches, e.g.
//   --benchmark_filter='kernel/cx/n22/hi/'
//
// This is an internal micro-benchmark and deliberately not named
// bench_compare_*: there is no Qiskit half to pair it against.

#include <benchmark/benchmark.h>

#include "lindblad/constants.hpp"
#include "lindblad/gates.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/types.hpp"
#include "lindblad/validation.hpp"

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace lindblad;

namespace {

using cplx = std::complex<double>;

constexpr int kSizes[] = {18, 22, 26};
constexpr int kFusedWidths[] = {2, 3, 4, 5};
constexpr int kCheckSamples = 64;
constexpr double kEps = std::numeric_limits<double>::epsilon();

// Seeds for the deterministic inputs. Any fixed values serve; these only have
// to differ from each other so the state and the dense blocks are unrelated.
constexpr std::uint64_t kStateSeed = 0x6c696e64626c6164ULL;
constexpr std::uint64_t kMatrixSeed = 0x6b65726e656c7321ULL;
constexpr std::uint64_t kSampleSeed = 0x73616d706c657321ULL;

// Gate angles: irrational multiples of PI, so no case reduces to a Clifford or
// an identity that a kernel could get right by accident.
const double kRzTheta = PI / 5.0;
const double kU3Theta = PI / 3.0;
const double kU3Phi = PI / 7.0;
const double kU3Lambda = PI / 11.0;
const double kCpLambda = PI / 9.0;

// The OpenMP default when the binary started, which is what "all" means.
int g_all_threads = 1;

// =============================================================================
// Deterministic inputs
// =============================================================================

// SplitMix64: one 64-bit value per input, independent of evaluation order, so
// the state is the same however many threads build it.
std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// A value in [-1, 1) from the top 53 bits.
double unit_signed(std::uint64_t bits) {
    return static_cast<double>(bits >> 11) * 0x1.0p-52 - 1.0;
}

// Every amplitude nonzero and unrelated to its neighbours, then normalised: a
// kernel that mixes the wrong pair cannot land on the right value by symmetry.
void fill_random_state(Statevector& sv) {
    double* re = sv.real_parts;
    double* im = sv.imag_parts;
    const std::size_t dim = sv.dim;
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < dim; ++i) {
        re[i] = unit_signed(splitmix64(kStateSeed ^ (2 * i)));
        im[i] = unit_signed(splitmix64(kStateSeed ^ (2 * i + 1)));
    }
    sv.normalize();
}

// A Haar-like k-qubit unitary: Gram-Schmidt over random complex columns.
// Row-major, 2^k x 2^k, as gates::apply_unitary takes it.
std::vector<cplx> random_unitary(int k, std::uint64_t seed) {
    const std::size_t d = std::size_t{1} << k;
    std::vector<cplx> col_major(d * d);
    for (std::size_t e = 0; e < d * d; ++e)
        col_major[e] = cplx(unit_signed(splitmix64(seed ^ (2 * e))),
                            unit_signed(splitmix64(seed ^ (2 * e + 1))));
    for (std::size_t c = 0; c < d; ++c) {
        cplx* v = &col_major[c * d];
        for (std::size_t p = 0; p < c; ++p) {
            const cplx* u = &col_major[p * d];
            cplx dot = 0.0;
            for (std::size_t r = 0; r < d; ++r) dot += std::conj(u[r]) * v[r];
            for (std::size_t r = 0; r < d; ++r) v[r] -= dot * u[r];
        }
        double norm = 0.0;
        for (std::size_t r = 0; r < d; ++r) norm += std::norm(v[r]);
        norm = std::sqrt(norm);
        for (std::size_t r = 0; r < d; ++r) v[r] /= norm;
    }
    std::vector<cplx> row_major(d * d);
    for (std::size_t r = 0; r < d; ++r)
        for (std::size_t c = 0; c < d; ++c) row_major[r * d + c] = col_major[c * d + r];
    return row_major;
}

// =============================================================================
// One timed case: a kernel, the qubits it acts on, and its reference matrix
// =============================================================================
// matrix = row-major, 2^k x 2^k, with qubits[0] the least significant bit of
// the row and column index (the library's multi-qubit matrix convention).

struct KernelCase {
    std::vector<int> qubits;
    std::vector<cplx> matrix;
    std::function<void(Statevector&)> apply;
};

// The 2^k basis indices sharing i's background bits, ordered by the packed
// value of the target bits (bit j of the position = bit qubits[j] of the index).
std::vector<std::size_t> block_indices(std::size_t i, const std::vector<int>& qubits) {
    std::size_t base = i;
    for (int q : qubits) base &= ~(std::size_t{1} << q);
    const std::size_t d = std::size_t{1} << qubits.size();
    std::vector<std::size_t> out(d);
    for (std::size_t c = 0; c < d; ++c) {
        std::size_t idx = base;
        for (std::size_t j = 0; j < qubits.size(); ++j)
            if ((c >> j) & 1) idx |= std::size_t{1} << qubits[j];
        out[c] = idx;
    }
    return out;
}

std::size_t packed_row(std::size_t i, const std::vector<int>& qubits) {
    std::size_t r = 0;
    for (std::size_t j = 0; j < qubits.size(); ++j)
        if ((i >> qubits[j]) & 1) r |= std::size_t{1} << j;
    return r;
}

// Applies the case once and compares kCheckSamples amplitudes with
// sum_c M[row][c] * old[c]. Each side rounds: 2^k - 1 additions in its own
// order, one product per term, and a matrix entry that may itself be one
// rounding off (the kernel computes its trigonometry separately). So each is
// within (2^k + 2) eps of the exact value relative to the summed magnitudes,
// and the two within twice that. why = the first disagreement, for the error
// message.
bool apply_and_check(const KernelCase& kc, Statevector& sv, std::string& why) {
    const std::size_t d = std::size_t{1} << kc.qubits.size();
    struct Sample {
        std::size_t index;
        std::size_t row;
        std::vector<cplx> old;
    };
    std::vector<Sample> samples;
    samples.reserve(kCheckSamples);
    for (int s = 0; s < kCheckSamples; ++s) {
        const std::size_t i = splitmix64(kSampleSeed + static_cast<std::uint64_t>(s)) % sv.dim;
        Sample smp{i, packed_row(i, kc.qubits), {}};
        for (std::size_t idx : block_indices(i, kc.qubits))
            smp.old.emplace_back(sv.real_parts[idx], sv.imag_parts[idx]);
        samples.push_back(std::move(smp));
    }

    kc.apply(sv);

    for (const Sample& smp : samples) {
        cplx want = 0.0;
        double scale = 0.0;
        for (std::size_t c = 0; c < d; ++c) {
            const cplx term = kc.matrix[smp.row * d + c] * smp.old[c];
            want += term;
            scale += std::abs(term);
        }
        const cplx got(sv.real_parts[smp.index], sv.imag_parts[smp.index]);
        const double bound = 2.0 * static_cast<double>(d + 2) * kEps * scale;
        if (std::abs(got - want) > bound) {
            char buf[256];
            std::snprintf(buf, sizeof buf,
                          "amplitude %zu is (%.17g, %.17g), the reference gives "
                          "(%.17g, %.17g), off by %.3g against a bound of %.3g",
                          smp.index, got.real(), got.imag(), want.real(), want.imag(),
                          std::abs(got - want), bound);
            why = buf;
            return false;
        }
    }
    return true;
}

// Restores the caller's OpenMP thread count when the case ends.
class ThreadScope {
public:
    explicit ThreadScope(int threads) {
#ifdef _OPENMP
        saved_ = omp_get_max_threads();
        omp_set_num_threads(threads);
#else
        (void)threads;
#endif
    }
    ~ThreadScope() {
#ifdef _OPENMP
        omp_set_num_threads(saved_);
#endif
    }
    ThreadScope(const ThreadScope&) = delete;
    ThreadScope& operator=(const ThreadScope&) = delete;

private:
    int saved_ = 1;
};

void run_case(benchmark::State& state, int n, int threads, const KernelCase& kc) {
    ThreadScope scope(threads);
    Statevector sv(n);
    fill_random_state(sv);

    std::string why;
    if (!apply_and_check(kc, sv, why)) {
        state.SkipWithError(("reference check failed: " + why).c_str());
        return;
    }

    for (auto _ : state) {
        kc.apply(sv);
        benchmark::ClobberMemory();
    }

    // One application leaves an error vector of norm at most
    // (2^k + 2) sqrt(2^k) eps (each block of 2^k amplitudes rounds as in
    // apply_and_check, and the blocks are orthogonal), so the drift grows with
    // the number of applications. norm() is itself a sum over 2^n terms and can
    // be 2^n eps off, which dominates at large n but still catches a kernel
    // that scales or corrupts the state.
    const double applications = static_cast<double>(state.iterations()) + 1.0;
    const double d = static_cast<double>(std::size_t{1} << kc.qubits.size());
    const double norm_bound = applications * (d + 2.0) * std::sqrt(d) * kEps +
                              static_cast<double>(sv.dim) * kEps;
    if (std::abs(sv.norm() - 1.0) > norm_bound) {
        state.SkipWithError("the norm drifted beyond the rounding bound during timing");
        return;
    }

    const double bytes = 16.0 * static_cast<double>(sv.dim);
    state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) *
                            static_cast<std::int64_t>(bytes));
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) *
                            static_cast<std::int64_t>(sv.dim));
    state.counters["threads"] = threads;
}

// =============================================================================
// Reference matrices, in the library's conventions
// =============================================================================

std::vector<cplx> mat_h() {
    return {INV_SQRT2, INV_SQRT2, INV_SQRT2, -INV_SQRT2};
}
std::vector<cplx> mat_x() { return {0.0, 1.0, 1.0, 0.0}; }
std::vector<cplx> mat_rz(double t) {
    return {std::polar(1.0, -t / 2.0), 0.0, 0.0, std::polar(1.0, t / 2.0)};
}
std::vector<cplx> mat_u3(double t, double phi, double lam) {
    return {std::cos(t / 2.0), -std::polar(1.0, lam) * std::sin(t / 2.0),
            std::polar(1.0, phi) * std::sin(t / 2.0),
            std::polar(1.0, phi + lam) * std::cos(t / 2.0)};
}
// Two-qubit matrices on [first, second] with first the LSB: index = b0 + 2 b1.
std::vector<cplx> mat_cx() {  // control = qubits[0], target = qubits[1]
    std::vector<cplx> m(16, 0.0);
    m[0 * 4 + 0] = 1.0; m[1 * 4 + 3] = 1.0; m[2 * 4 + 2] = 1.0; m[3 * 4 + 1] = 1.0;
    return m;
}
std::vector<cplx> mat_cp(double lam) {
    std::vector<cplx> m(16, 0.0);
    m[0] = 1.0; m[5] = 1.0; m[10] = 1.0; m[15] = std::polar(1.0, lam);
    return m;
}
std::vector<cplx> mat_swap() {
    std::vector<cplx> m(16, 0.0);
    m[0 * 4 + 0] = 1.0; m[1 * 4 + 2] = 1.0; m[2 * 4 + 1] = 1.0; m[3 * 4 + 3] = 1.0;
    return m;
}
// [c1, c2, target]: flips bit 2 where bits 0 and 1 are set (3 <-> 7).
std::vector<cplx> mat_ccx() {
    std::vector<cplx> m(64, 0.0);
    for (int r = 0; r < 8; ++r) {
        const int c = (r == 3) ? 7 : (r == 7) ? 3 : r;
        m[r * 8 + c] = 1.0;
    }
    return m;
}

std::vector<Complex128> to_library(const std::vector<cplx>& m) {
    std::vector<Complex128> out;
    out.reserve(m.size());
    for (const cplx& z : m) out.emplace_back(z.real(), z.imag());
    return out;
}

// =============================================================================
// Registration
// =============================================================================

struct ThreadMode {
    const char* name;
    int threads;
};

std::vector<ThreadMode> thread_modes() {
    return {{"all", g_all_threads}, {"one", 1}};
}

void add(const std::string& name, int n, int threads, KernelCase kc) {
    benchmark::RegisterBenchmark(
        name.c_str(),
        [n, threads, kc = std::move(kc)](benchmark::State& st) { run_case(st, n, threads, kc); })
        ->Unit(benchmark::kMillisecond)
        ->UseRealTime();
}

std::string prefix(const char* gate, int n, const char* pattern, int pos, const char* mode) {
    return std::string("kernel/") + gate + "/n" + std::to_string(n) + "/" + pattern + "/" +
           std::to_string(pos) + "/" + mode;
}

void register_one_qubit(int n, const ThreadMode& tm) {
    for (int q = 0; q < n; ++q) {
        add(prefix("h", n, "q", q, tm.name), n, tm.threads,
            {{q}, mat_h(), [q](Statevector& s) { gates::apply_h(s, q); }});
        add(prefix("x", n, "q", q, tm.name), n, tm.threads,
            {{q}, mat_x(), [q](Statevector& s) { gates::apply_x(s, q); }});
        add(prefix("rz", n, "q", q, tm.name), n, tm.threads,
            {{q}, mat_rz(kRzTheta), [q](Statevector& s) { gates::apply_rz(s, q, kRzTheta); }});
        add(prefix("u3", n, "q", q, tm.name), n, tm.threads,
            {{q}, mat_u3(kU3Theta, kU3Phi, kU3Lambda),
             [q](Statevector& s) { gates::apply_u3(s, q, kU3Theta, kU3Phi, kU3Lambda); }});
    }
}

void register_two_qubit(int n, const ThreadMode& tm) {
    auto both = [&](const char* pattern, int pos, int a, int b) {
        add(prefix("cx", n, pattern, pos, tm.name), n, tm.threads,
            {{a, b}, mat_cx(), [a, b](Statevector& s) { gates::apply_cx(s, a, b); }});
        add(prefix("cp", n, pattern, pos, tm.name), n, tm.threads,
            {{a, b}, mat_cp(kCpLambda), [a, b](Statevector& s) { gates::apply_cp(s, a, b, kCpLambda); }});
        add(prefix("swap", n, pattern, pos, tm.name), n, tm.threads,
            {{a, b}, mat_swap(), [a, b](Statevector& s) { gates::apply_swap(s, a, b); }});
    };
    for (int q = 1; q < n; ++q) both("hi", q, 0, q);
    for (int q = 0; q < n - 1; ++q) both("lo", q, q, n - 1);
}

void register_three_qubit(int n, const ThreadMode& tm) {
    auto ccx = [&](const char* pattern, int pos, int c1, int c2, int t) {
        add(prefix("ccx", n, pattern, pos, tm.name), n, tm.threads,
            {{c1, c2, t}, mat_ccx(), [c1, c2, t](Statevector& s) { gates::apply_ccx(s, c1, c2, t); }});
    };
    for (int q = 2; q < n; ++q) ccx("hi", q, 0, 1, q);
    for (int q = 0; q < n - 2; ++q) ccx("lo", q, q, n - 2, n - 1);
}

// The fused path applies its blocks with validation off, as this does.
void register_fused(int n, const ThreadMode& tm) {
    for (int k : kFusedWidths) {
        const std::vector<cplx> m = random_unitary(k, kMatrixSeed + static_cast<std::uint64_t>(k));
        const std::vector<Complex128> lib = to_library(m);
        const std::string gate = "unitary_k" + std::to_string(k);
        auto block = [&](const char* pattern, int pos, std::vector<int> targets) {
            add(prefix(gate.c_str(), n, pattern, pos, tm.name), n, tm.threads,
                {targets, m, [targets, lib](Statevector& s) {
                     gates::apply_unitary(s, targets, lib, {Validation::Ignore, 0.0});
                 }});
        };
        const int positions[] = {0, 1, 2, 3, (n - k) / 2, n - k};
        for (int p : positions) {
            std::vector<int> t(static_cast<std::size_t>(k));
            for (int j = 0; j < k; ++j) t[static_cast<std::size_t>(j)] = p + j;
            block("block", p, t);
        }
        std::vector<int> spread(static_cast<std::size_t>(k));
        for (int j = 0; j < k; ++j)
            spread[static_cast<std::size_t>(j)] = static_cast<int>(
                std::lround(static_cast<double>(j) * (n - 1) / (k - 1)));
        block("spread", 0, spread);
    }
}

// =============================================================================
// Self-tests: wrong kernels the reference check must reject
// =============================================================================

constexpr int kSelfTestQubits = 18;
constexpr int kSelfTestOperand = kSelfTestQubits / 2;  // away from both ends

void run_selftest(benchmark::State& state, const KernelCase& wrong) {
    Statevector sv(kSelfTestQubits);
    fill_random_state(sv);
    std::string why;
    const bool accepted = apply_and_check(wrong, sv, why);
    for (auto _ : state) {
    }
    if (accepted) {
        state.SkipWithError("the reference check accepted a deliberately wrong kernel");
        return;
    }
    state.SetLabel("rejected as it must be");
}

void register_selftests() {
    constexpr int a = kSelfTestOperand;
    const std::vector<KernelCase> faults = {
        // does nothing at all: the shape a narrowed loop bound produces
        {{a}, mat_h(), [](Statevector&) {}},
        // CX on the right control and the wrong target
        {{0, a}, mat_cx(), [](Statevector& s) { gates::apply_cx(s, 0, a + 1); }},
        // U3 with phi and lambda exchanged
        {{a}, mat_u3(kU3Theta, kU3Phi, kU3Lambda),
         [](Statevector& s) { gates::apply_u3(s, a, kU3Theta, kU3Lambda, kU3Phi); }},
    };
    const char* names[] = {"selftest/noop", "selftest/wrong_target", "selftest/swapped_angles"};
    for (std::size_t i = 0; i < faults.size(); ++i) {
        KernelCase kc = faults[i];
        benchmark::RegisterBenchmark(
            names[i], [kc](benchmark::State& st) { run_selftest(st, kc); })
            ->Iterations(1);
    }
}

// =============================================================================
// Roofline references
// =============================================================================

void roofline_sweep(benchmark::State& state, int n, int threads) {
    ThreadScope scope(threads);
    Statevector sv(n);
    fill_random_state(sv);
    const double c = std::cos(kRzTheta);
    const double s = std::sin(kRzTheta);
    double* __restrict__ re = sv.real_parts;
    double* __restrict__ im = sv.imag_parts;
    const std::size_t dim = sv.dim;
    for (auto _ : state) {
        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < dim; ++i) {
            const double r = re[i];
            const double m = im[i];
            re[i] = c * r - s * m;
            im[i] = s * r + c * m;
        }
        benchmark::ClobberMemory();
    }
    // Read and write both planes.
    state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) *
                            static_cast<std::int64_t>(32 * dim));
    state.counters["threads"] = threads;
}

void roofline_triad(benchmark::State& state, int n, int threads) {
    ThreadScope scope(threads);
    const std::size_t len = std::size_t{1} << n;
    double* a = aligned_alloc_doubles(len);
    double* b = aligned_alloc_doubles(len);
    double* c = aligned_alloc_doubles(len);
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < len; ++i) {
        a[i] = 0.0;
        b[i] = unit_signed(splitmix64(kStateSeed ^ i));
        c[i] = unit_signed(splitmix64(kMatrixSeed ^ i));
    }
    const double scalar = std::sin(kCpLambda);
    for (auto _ : state) {
        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < len; ++i) a[i] = b[i] + scalar * c[i];
        benchmark::ClobberMemory();
    }
    // STREAM's convention: two reads and one write, no write-allocate.
    state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) *
                            static_cast<std::int64_t>(24 * len));
    state.counters["threads"] = threads;
    aligned_free(a);
    aligned_free(b);
    aligned_free(c);
}

void register_roofline(int n, const ThreadMode& tm) {
    const int threads = tm.threads;
    benchmark::RegisterBenchmark(
        ("roofline/sweep/n" + std::to_string(n) + "/" + tm.name).c_str(),
        [n, threads](benchmark::State& st) { roofline_sweep(st, n, threads); })
        ->Unit(benchmark::kMillisecond)
        ->UseRealTime();
    benchmark::RegisterBenchmark(
        ("roofline/triad/n" + std::to_string(n) + "/" + tm.name).c_str(),
        [n, threads](benchmark::State& st) { roofline_triad(st, n, threads); })
        ->Unit(benchmark::kMillisecond)
        ->UseRealTime();
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _OPENMP
    g_all_threads = omp_get_max_threads();
#endif
    register_selftests();
    for (int n : kSizes) {
        for (const ThreadMode& tm : thread_modes()) {
            register_roofline(n, tm);
            register_one_qubit(n, tm);
            register_two_qubit(n, tm);
            register_three_qubit(n, tm);
            register_fused(n, tm);
        }
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
