// R.1.13.1 test patch — qudit layer parallelism + MPS sampler.
// Covers audit F-17 (OpenMP apply_1qudit / apply_2qudit / apply_kqudit with
// thread-local scratch) and F-5 (QuditMPS::measure sequential environment
// sampler). Kernel outputs are checked against an independent serial gather at
// a dimension above the OpenMP threshold (dim >= 2^12), so a race or a wrong
// stride shows up as a mismatch.

#include <gtest/gtest.h>

#include "lindblad/types.hpp"
#include "lindblad/qudit/qudit_statevector.hpp"
#include "lindblad/qudit/qudit_mps.hpp"

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace lindblad;

namespace {

using QSV = QuditStatevector;

std::vector<Complex128> distinct_amps(size_t dim) {
    std::vector<Complex128> v(dim);
    for (size_t k = 0; k < dim; ++k)
        v[k] = Complex128(0.2 + 0.001 * static_cast<double>(k),
                          -0.1 + 0.0007 * static_cast<double>(k));
    return v;
}

std::vector<Complex128> distinct_matrix(size_t rows_cols) {
    std::vector<Complex128> U(rows_cols * rows_cols);
    for (size_t r = 0; r < rows_cols; ++r)
        for (size_t c = 0; c < rows_cols; ++c)
            U[r * rows_cols + c] =
                Complex128(1.0 + 0.01 * static_cast<double>(r * rows_cols + c),
                           0.3 - 0.005 * static_cast<double>(c));
    return U;
}

std::vector<Complex128> ref_1qudit(const std::vector<Complex128>& in, int d,
                                   int n, int q, const std::vector<Complex128>& U) {
    std::vector<Complex128> out(in.size());
    for (size_t idx = 0; idx < in.size(); ++idx) {
        auto dg = QSV::index_to_digits(idx, d, n);
        const int old = dg[static_cast<size_t>(q)];
        Complex128 s(0, 0);
        for (int col = 0; col < d; ++col) {
            dg[static_cast<size_t>(q)] = col;
            s += U[static_cast<size_t>(old * d + col)] * in[QSV::digits_to_index(dg, d)];
        }
        out[idx] = s;
    }
    return out;
}

std::vector<Complex128> ref_2qudit(const std::vector<Complex128>& in, int d,
                                   int n, int q0, int q1,
                                   const std::vector<Complex128>& U) {
    const int d2 = d * d;
    std::vector<Complex128> out(in.size());
    for (size_t idx = 0; idx < in.size(); ++idx) {
        auto dg = QSV::index_to_digits(idx, d, n);
        const int a0 = dg[static_cast<size_t>(q0)];
        const int a1 = dg[static_cast<size_t>(q1)];
        const int r = a1 * d + a0;               // row: new_q1*d + new_q0
        Complex128 s(0, 0);
        for (int c = 0; c < d2; ++c) {
            dg[static_cast<size_t>(q0)] = c % d;
            dg[static_cast<size_t>(q1)] = c / d;
            s += U[static_cast<size_t>(r * d2 + c)] * in[QSV::digits_to_index(dg, d)];
        }
        out[idx] = s;
    }
    return out;
}

std::vector<Complex128> ref_kqudit(const std::vector<Complex128>& in, int d,
                                   int n, const std::vector<int>& qudits,
                                   const std::vector<Complex128>& U) {
    const int k = static_cast<int>(qudits.size());
    const size_t dk = QSV::ipow(static_cast<size_t>(d), k);
    std::vector<Complex128> out(in.size());
    for (size_t idx = 0; idx < in.size(); ++idx) {
        auto dg = QSV::index_to_digits(idx, d, n);
        size_t r = 0;
        for (int i = 0; i < k; ++i)
            r += static_cast<size_t>(dg[static_cast<size_t>(qudits[i])]) *
                 QSV::ipow(static_cast<size_t>(d), i);
        Complex128 s(0, 0);
        for (size_t c = 0; c < dk; ++c) {
            for (int i = 0; i < k; ++i)
                dg[static_cast<size_t>(qudits[i])] = static_cast<int>(
                    (c / QSV::ipow(static_cast<size_t>(d), i)) % static_cast<size_t>(d));
            s += U[r * dk + c] * in[QSV::digits_to_index(dg, d)];
        }
        out[idx] = s;
    }
    return out;
}

void expect_amps_close(const std::vector<Complex128>& a,
                       const std::vector<Complex128>& b, double tol = 1e-9) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i].real, b[i].real, tol) << "re @ " << i;
        EXPECT_NEAR(a[i].imag, b[i].imag, tol) << "im @ " << i;
    }
}

} // namespace

// F-17: parallel apply_1qudit matches the serial reference above threshold.
TEST(R1131Qudit, Apply1quditParallelMatchesSerial) {
    const int d = 3, n = 8;                 // dim = 6561 > 4096 (OMP engaged)
    const size_t dim = QSV::ipow(d, n);
    const auto in = distinct_amps(dim);
    const auto U = distinct_matrix(static_cast<size_t>(d));

    QSV sv(n, d);
    sv.amplitudes = in;
    sv.apply_1qudit(3, U);
    expect_amps_close(sv.amplitudes, ref_1qudit(in, d, n, 3, U));
}

// F-17: parallel apply_2qudit (arbitrary, non-adjacent pair) vs serial.
TEST(R1131Qudit, Apply2quditParallelMatchesSerial) {
    const int d = 3, n = 8;
    const size_t dim = QSV::ipow(d, n);
    const auto in = distinct_amps(dim);
    const auto U = distinct_matrix(static_cast<size_t>(d * d));

    QSV sv(n, d);
    sv.amplitudes = in;
    sv.apply_2qudit(1, 5, U);
    expect_amps_close(sv.amplitudes, ref_2qudit(in, d, n, 1, 5, U));
}

// F-17: parallel apply_kqudit (k=3) vs serial.
TEST(R1131Qudit, ApplyKquditParallelMatchesSerial) {
    const int d = 3, n = 8;
    const size_t dim = QSV::ipow(d, n);
    const auto in = distinct_amps(dim);
    const std::vector<int> qudits{0, 3, 6};
    const auto U = distinct_matrix(QSV::ipow(static_cast<size_t>(d), 3));  // 27x27

    QSV sv(n, d);
    sv.amplitudes = in;
    sv.apply_kqudit(qudits, U);
    expect_amps_close(sv.amplitudes, ref_kqudit(in, d, n, qudits, U));
}

// F-5: QuditMPS::measure sequential sampler reproduces the |amplitude|^2
// distribution of the exact dense state it was built from.
TEST(R1131Qudit, MpsMeasureMatchesDenseDistribution) {
    const int d = 2, n = 3;
    const double inv2 = INV_SQRT2;  // short local alias, library-sourced value
    const std::vector<Complex128> H2{Complex128(inv2, 0), Complex128(inv2, 0),
                                     Complex128(inv2, 0), Complex128(-inv2, 0)};
    // CX (d=2), row = new_q1*2 + new_q0: identity on q0=0 block, flip q1 on q0=1.
    std::vector<Complex128> CX(16, Complex128(0, 0));
    CX[0 * 4 + 0] = Complex128(1, 0);   // |q1=0,q0=0> -> itself
    CX[2 * 4 + 1] = Complex128(1, 0);   // |q1=0,q0=1> -> |q1=1,q0=1>
    CX[1 * 4 + 2] = Complex128(1, 0);   // |q1=1,q0=0> -> |q1=0,q0=0>? build any unitary
    CX[3 * 4 + 3] = Complex128(1, 0);
    // (exact orientation is irrelevant: we build the MPS from the resulting
    //  dense state and compare the sampler to that same state.)

    QSV sv(n, d);
    sv.apply_1qudit(0, H2);
    sv.apply_1qudit(1, H2);
    sv.apply_2qudit(0, 1, CX);
    sv.apply_1qudit(2, {Complex128(0.6, 0), Complex128(0.8, 0),
                        Complex128(0.8, 0), Complex128(-0.6, 0)});
    sv.normalize();

    // Reference probabilities from the dense state.
    std::vector<double> pref(sv.dim, 0.0);
    for (size_t k = 0; k < sv.dim; ++k) pref[k] = sv.amplitudes[k].norm_sq();

    QuditMPS mps(sv);
    const int shots = 20000;
    std::vector<int> hist(sv.dim, 0);
    for (int s = 0; s < shots; ++s) {
        auto digits = mps.measure(/*seed=*/static_cast<uint64_t>(1000 + s));
        ASSERT_EQ(digits.size(), static_cast<size_t>(n));
        hist[QSV::digits_to_index(digits, d)]++;
    }

    for (size_t k = 0; k < sv.dim; ++k) {
        const double f = static_cast<double>(hist[k]) / shots;
        EXPECT_NEAR(f, pref[k], 0.03) << "outcome " << k;
    }
}

// F-23 (qudit layer): selecting SVDMethod::BDC on the qudit MPS emits the loud
// broken-BDCSVD warning. This is the ONLY qudit-MPS BDC selection in the test
// binary, so warn_bdc_broken_once_qudit()'s process-global latch fires here.
TEST(R1131Qudit, MpsBdcSelectionWarns) {
    std::ostringstream capture;
    std::streambuf* old = std::cerr.rdbuf(capture.rdbuf());

    {
        QuditMPS mps(2, 2);
        mps.svd_method = SVDMethod::BDC;
        std::vector<Complex128> U(16, Complex128(0, 0));   // 4x4 identity (d^2)
        for (int i = 0; i < 4; ++i) U[static_cast<size_t>(i * 4 + i)] = Complex128(1, 0);
        mps.apply_2qudit_adjacent(0, U);                    // -> qmps_svd -> warn
    }

    std::cerr.rdbuf(old);
    const std::string out = capture.str();
    EXPECT_NE(out.find("BDC"), std::string::npos)
        << "qudit BDC warning not emitted; got: [" << out << "]";
    EXPECT_NE(out.find("BROKEN"), std::string::npos);
    EXPECT_NE(out.find("qudit"), std::string::npos);
}
