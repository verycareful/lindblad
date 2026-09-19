// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.28.1 test wave - the theta harvest writes what the SVD receives.
//
// LINDBLAD_MPS_THETA_HARVEST=ON arms a dump inside the MPS bond split: one
// representative two-site block per distinct shape, written on first sight as
// autonne-format hexfloat, plus a shape histogram at exit. Its purpose is to
// hand a kernel author the blocks this consumer actually forms, and a
// corrupted corpus would send them benchmarking against matrices that exist
// nowhere. So the properties are exactness (the file holds the bits the
// simulation held), fidelity (the block is the one the SVD was about to
// receive, in the layout it receives it), and the first-sight rule (an
// interrupted run leaves a complete file for every shape it reached).
//
// Every test skips unless the macro is defined, so a default build reports a
// skipped suite rather than a missing one. Two facts shape the design:
//
//   - The output directory is relative to the working directory at the time
//     of the offer. Each test changes into a fresh temporary directory and
//     restores the original on exit, failure included.
//
//   - A shape is written on FIRST sight per process. A qubit split is always
//     (2 bl) x (2 br), so odd shapes are unreachable from the simulator, and
//     the direct tests use those. The library-block test hand-builds a chain
//     with a prime outer bond so its 2 x 26 block cannot come from any cap in
//     use; if the file is nonetheless absent after the offer, an earlier test
//     in this process produced that shape first, and the test says so rather
//     than failing against a file it did not write.
//
// The histogram is written by a static destructor and cannot be read from
// inside the process that writes it. tests/tools/test_theta_harvest_histogram.py
// runs this suite in a fresh process and checks it, and its expected counts are
// derived from the offers the tests below make: 1x3, 1x5, 3x5, 3x7 and 2x26
// once each, 5x7 twice, and the 2x2 and 4x2 blocks of a three-qubit GHZ once
// each. Changing an offer here changes that file.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/constants.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/types.hpp"

#ifdef LINDBLAD_MPS_THETA_HARVEST
#include "lindblad/detail/theta_harvest.hpp"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace lindblad;
namespace fs = std::filesystem;

namespace {

#ifdef LINDBLAD_MPS_THETA_HARVEST
constexpr bool kHarvestOn = true;
#else
constexpr bool kHarvestOn = false;
#endif

#define SKIP_UNLESS_HARVEST()                                                    \
    do {                                                                         \
        if (!kHarvestOn) {                                                       \
            GTEST_SKIP() << "configure with -DLINDBLAD_MPS_THETA_HARVEST=ON to run"; \
        }                                                                        \
    } while (0)

#ifdef LINDBLAD_MPS_THETA_HARVEST

using lindblad::detail::theta_harvest_offer;

constexpr double kEps = std::numeric_limits<double>::epsilon();
constexpr double kSlack = 64.0;
constexpr std::uint64_t kSeed = 42;

// The directory the writer uses, relative to the working directory, and the
// file name it derives from a shape. Both restate the writer's own contract
// so a change to either surfaces here.
const char* const kHarvestDir = "theta_harvest";

std::string block_file(int rows, int cols) {
    return std::string(kHarvestDir) + "/theta_" + std::to_string(rows) + "x" +
           std::to_string(cols) + ".hexfloat";
}

// Change into a fresh temporary directory for the scope, restore on exit.
// Timestamp plus counter keeps rapid successive tests apart without a
// process id, which would need a platform header.
class ScopedTempCwd {
public:
    ScopedTempCwd() : original_(fs::current_path()) {
        static int counter = 0;
        ++counter;
        std::ostringstream name;
        name << "lindblad_harvest_"
             << std::chrono::steady_clock::now().time_since_epoch().count()
             << "_" << counter;
        dir_ = fs::temp_directory_path() / name.str();
        fs::create_directories(dir_);
        fs::current_path(dir_);
    }
    ~ScopedTempCwd() {
        std::error_code ec;
        fs::current_path(original_, ec);
        fs::remove_all(dir_, ec);
    }
    const fs::path& dir() const { return dir_; }

private:
    fs::path original_;
    fs::path dir_;
};

// One parsed hexfloat file: the header line and the values in file order.
struct Parsed {
    std::string header;
    std::vector<Complex128> values;
    std::size_t lines = 0;
};

Parsed parse_block(const std::string& path) {
    Parsed p;
    std::ifstream in(path, std::ios::binary);
    if (!in) return p;
    std::getline(in, p.header);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        ++p.lines;
        std::istringstream ls(line);
        std::string re, im;
        ls >> re >> im;
        // strtod reads the %a form back exactly, which is the property the
        // hexfloat format exists for.
        p.values.push_back(Complex128(std::strtod(re.c_str(), nullptr),
                                      std::strtod(im.c_str(), nullptr)));
    }
    return p;
}

bool same_bits(double a, double b) {
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

// Values chosen to stress the rendering: a non-dyadic rational, negative
// zero, the smallest subnormal, the largest finite, the smallest normal, the
// value with every mantissa bit set, and epsilon. A decimal rendering loses
// at least one of these.
std::vector<double> stress_values() {
    return {1.0 / 3.0,
            -0.0,
            std::numeric_limits<double>::denorm_min(),
            std::numeric_limits<double>::max(),
            std::numeric_limits<double>::min(),
            std::nextafter(1.0, 0.0),
            kEps,
            -PI,
            SQRT2,
            0.0};
}

std::vector<Complex128> stress_block(int rows, int cols) {
    const auto pool = stress_values();
    std::vector<Complex128> block(static_cast<std::size_t>(rows) * cols);
    for (std::size_t i = 0; i < block.size(); ++i) {
        block[i] = Complex128(pool[i % pool.size()], -pool[(i * 3 + 1) % pool.size()]);
    }
    return block;
}

using Mat2 = std::array<Complex128, 4>;
using Mat4 = std::array<Complex128, 16>;

// kron(A, B) with A on the HIGH bit, the index convention the adjacent kernel
// uses for (q1, q2): entry (i*2+j, k*2+l) = A(i,k) B(j,l).
Mat4 kron(const Mat2& A, const Mat2& B) {
    Mat4 out{};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k)
                for (int l = 0; l < 2; ++l)
                    out[static_cast<std::size_t>((i * 2 + j) * 4 + (k * 2 + l))] =
                        A[static_cast<std::size_t>(i * 2 + k)] * B[static_cast<std::size_t>(j * 2 + l)];
    return out;
}

Mat4 matmul(const Mat4& A, const Mat4& B) {
    Mat4 out{};
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) {
            Complex128 sum(0.0, 0.0);
            for (int k = 0; k < 4; ++k)
                sum += A[static_cast<std::size_t>(r * 4 + k)] * B[static_cast<std::size_t>(k * 4 + c)];
            out[static_cast<std::size_t>(r * 4 + c)] = sum;
        }
    return out;
}

// A generic SU(4) with no symmetry under transposition and none under
// swapping the two qubits' roles: (H x S) CX (T x Ry(pi/3)), with CX
// controlled on the high bit. Its lack of symmetry is asserted where it is
// used, so the layout check cannot be silently blind.
Mat4 asymmetric_gate() {
    const double c = std::cos(PI / 6.0), s = std::sin(PI / 6.0);  // Ry(pi/3)
    const Mat2 H = {Complex128(INV_SQRT2, 0.0), Complex128(INV_SQRT2, 0.0),
                    Complex128(INV_SQRT2, 0.0), Complex128(-INV_SQRT2, 0.0)};
    const Mat2 S = {Complex128(1.0, 0.0), Complex128(0.0, 0.0),
                    Complex128(0.0, 0.0), Complex128(0.0, 1.0)};
    const Mat2 T = {Complex128(1.0, 0.0), Complex128(0.0, 0.0),
                    Complex128(0.0, 0.0), Complex128(std::cos(PI_4), std::sin(PI_4))};
    const Mat2 Ry = {Complex128(c, 0.0), Complex128(-s, 0.0),
                     Complex128(s, 0.0), Complex128(c, 0.0)};
    Mat4 cx{};
    cx[0 * 4 + 0] = Complex128(1.0, 0.0);
    cx[1 * 4 + 1] = Complex128(1.0, 0.0);
    cx[2 * 4 + 3] = Complex128(1.0, 0.0);
    cx[3 * 4 + 2] = Complex128(1.0, 0.0);
    return matmul(kron(H, S), matmul(cx, kron(T, Ry)));
}

double max_abs_diff(const Mat4& A, const Mat4& B) {
    double worst = 0.0;
    for (std::size_t i = 0; i < 16; ++i) {
        const double dr = A[i].real - B[i].real, di = A[i].imag - B[i].imag;
        worst = std::max(worst, std::sqrt(dr * dr + di * di));
    }
    return worst;
}

Mat4 transpose(const Mat4& A) {
    Mat4 out{};
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            out[static_cast<std::size_t>(c * 4 + r)] = A[static_cast<std::size_t>(r * 4 + c)];
    return out;
}

// Both qubits' roles exchanged: (i*2+j, k*2+l) -> (j*2+i, l*2+k).
Mat4 swap_roles(const Mat4& A) {
    Mat4 out{};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k)
                for (int l = 0; l < 2; ++l)
                    out[static_cast<std::size_t>((j * 2 + i) * 4 + (l * 2 + k))] =
                        A[static_cast<std::size_t>((i * 2 + j) * 4 + (k * 2 + l))];
    return out;
}

#endif  // LINDBLAD_MPS_THETA_HARVEST

}  // namespace

TEST(V11281ThetaHarvest, RoundTripIsBitExact) {
    SKIP_UNLESS_HARVEST();
#ifdef LINDBLAD_MPS_THETA_HARVEST
    ScopedTempCwd cwd;
    const int rows = 3, cols = 5;
    const auto block = stress_block(rows, cols);
    theta_harvest_offer(block, rows, cols);

    const Parsed p = parse_block(block_file(rows, cols));
    ASSERT_FALSE(p.header.empty()) << "no file written for a shape never seen before";
    ASSERT_EQ(p.values.size(), block.size());
    for (std::size_t i = 0; i < block.size(); ++i) {
        EXPECT_TRUE(same_bits(p.values[i].real, block[i].real))
            << "entry " << i << " real part did not round-trip: wrote "
            << block[i].real << ", read " << p.values[i].real;
        EXPECT_TRUE(same_bits(p.values[i].imag, block[i].imag))
            << "entry " << i << " imaginary part did not round-trip: wrote "
            << block[i].imag << ", read " << p.values[i].imag;
    }
#endif
}

TEST(V11281ThetaHarvest, HeaderNamesTheShapeAndTheOrder) {
    SKIP_UNLESS_HARVEST();
#ifdef LINDBLAD_MPS_THETA_HARVEST
    ScopedTempCwd cwd;
    const int rows = 3, cols = 7;
    const auto block = stress_block(rows, cols);
    theta_harvest_offer(block, rows, cols);

    const Parsed p = parse_block(block_file(rows, cols));
    // The exact line autonne's reader expects: format tag, version, kind,
    // name, rows, cols, order. A drift in any token is a corpus nobody can
    // load.
    EXPECT_EQ(p.header, "autonne-hexfloat 1 matrix theta_3x7 3 7 rowmajor");
    EXPECT_EQ(p.lines, static_cast<std::size_t>(rows) * cols)
        << "one line per element, no more and no fewer";
#endif
}

TEST(V11281ThetaHarvest, EntriesAreInRowMajorOrder) {
    SKIP_UNLESS_HARVEST();
#ifdef LINDBLAD_MPS_THETA_HARVEST
    // The block is offered row-major and must be written in that order, so
    // element (r, c) is line r*cols + c. Distinct dyadic values per position
    // make any permutation visible.
    ScopedTempCwd cwd;
    const int rows = 1, cols = 3;
    std::vector<Complex128> block(static_cast<std::size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            block[static_cast<std::size_t>(r) * cols + c] =
                Complex128(1.0 + r, 0.5 + c);
        }
    }
    theta_harvest_offer(block, rows, cols);
    const Parsed p = parse_block(block_file(rows, cols));
    ASSERT_EQ(p.values.size(), block.size());
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const auto& z = p.values[static_cast<std::size_t>(r) * cols + c];
            EXPECT_EQ(z.real, 1.0 + r) << "(" << r << "," << c << ")";
            EXPECT_EQ(z.imag, 0.5 + c) << "(" << r << "," << c << ")";
        }
    }
    // The directory did not exist before the offer; the writer creates it.
    EXPECT_TRUE(fs::is_directory(cwd.dir() / kHarvestDir));
#endif
}

TEST(V11281ThetaHarvest, FirstSightWins) {
    SKIP_UNLESS_HARVEST();
#ifdef LINDBLAD_MPS_THETA_HARVEST
    // The first block of a shape is the one kept, so an interrupted run still
    // leaves a complete file for every shape it reached. Later blocks of the
    // shape advance the count only.
    ScopedTempCwd cwd;
    const int rows = 5, cols = 7;
    const auto first = stress_block(rows, cols);
    std::vector<Complex128> second(first.size());
    for (std::size_t i = 0; i < second.size(); ++i) {
        second[i] = Complex128(static_cast<double>(i), -static_cast<double>(i));
    }
    theta_harvest_offer(first, rows, cols);
    theta_harvest_offer(second, rows, cols);

    const Parsed p = parse_block(block_file(rows, cols));
    ASSERT_EQ(p.values.size(), first.size());
    for (std::size_t i = 0; i < first.size(); ++i) {
        EXPECT_TRUE(same_bits(p.values[i].real, first[i].real))
            << "entry " << i << ": the second offer overwrote the first";
        EXPECT_TRUE(same_bits(p.values[i].imag, first[i].imag))
            << "entry " << i << ": the second offer overwrote the first";
    }
#endif
}

TEST(V11281ThetaHarvest, NonFiniteValuesRoundTrip) {
    SKIP_UNLESS_HARVEST();
#ifdef LINDBLAD_MPS_THETA_HARVEST
    // The simulator never forms a non-finite block (validation rejects the
    // gate first), but the writer must not corrupt one: %a renders inf and
    // nan by name and strtod reads both back. A NaN's payload is not
    // specified through that path, so only its NaN-ness is asserted.
    ScopedTempCwd cwd;
    const int rows = 1, cols = 5;
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<Complex128> block = {
        Complex128(inf, 0.0), Complex128(-inf, 1.0), Complex128(nan, -inf),
        Complex128(0.0, nan), Complex128(-0.0, inf)};
    theta_harvest_offer(block, rows, cols);

    const Parsed p = parse_block(block_file(rows, cols));
    ASSERT_EQ(p.values.size(), block.size());
    EXPECT_TRUE(same_bits(p.values[0].real, inf));
    EXPECT_TRUE(same_bits(p.values[1].real, -inf));
    EXPECT_TRUE(std::isnan(p.values[2].real));
    EXPECT_TRUE(same_bits(p.values[2].imag, -inf));
    EXPECT_TRUE(std::isnan(p.values[3].imag));
    EXPECT_TRUE(same_bits(p.values[4].real, -0.0));
    EXPECT_TRUE(same_bits(p.values[4].imag, inf));
#endif
}

TEST(V11281ThetaHarvest, LibraryBlockIsTheOneTheSvdReceives) {
    SKIP_UNLESS_HARVEST();
#ifdef LINDBLAD_MPS_THETA_HARVEST
    // A three-site chain with a prime outer bond, so the block formed at the
    // (0, 1) split is 2 x 26 and no cap in use can produce that shape. The
    // expected block is computed here from the documented layout: contract
    // the two site tensors into theta[l*2 + p1, p2*br + r], then apply the
    // gate with U indexed (po1*2 + po2, pi1*2 + pi2), po1 belonging to the
    // left site. A harvest that wrote the pre-gate block, the transpose, or a
    // different index order fails against this.
    ScopedTempCwd cwd;
    const int bond = 13;
    const int cap = 4;  // the split's rank is at most 2, so the cap never binds
    MPSState chain(3, cap);
    chain.tensors[0] = MPSTensor(1, bond);
    chain.tensors[1] = MPSTensor(bond, bond);
    chain.tensors[2] = MPSTensor(bond, 1);
    std::mt19937_64 rng(kSeed);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    for (auto& t : chain.tensors) {
        for (auto& z : t.data) z = Complex128(unit(rng), unit(rng));
    }
    // The gate must have no symmetry the layout check could hide behind: a
    // transposed contraction applies U^T, and a swap of the two sites' roles
    // applies the role-swapped U. Both are asserted distinct from U here, so
    // the comparison below is known to be able to fail for either mistake.
    const Mat4 U = asymmetric_gate();
    const double rounding = kSlack * 4.0 * kEps;
    ASSERT_GT(max_abs_diff(U, transpose(U)), rounding) << "U is symmetric";
    ASSERT_GT(max_abs_diff(U, swap_roles(U)), rounding) << "U is symmetric under a role swap";

    const int bl = chain.tensors[0].bond_left;   // 1
    const int bm = chain.tensors[0].bond_right;  // 13
    const int br = chain.tensors[1].bond_right;  // 13
    const int rows = bl * 2, cols = 2 * br;
    const auto& T1 = chain.tensors[0];
    const auto& T2 = chain.tensors[1];

    // theta[l*2+p1, p2*br+r] = sum_m T1(l,p1,m) T2(m,p2,r)
    std::vector<Complex128> theta(static_cast<std::size_t>(rows) * cols, Complex128(0.0, 0.0));
    for (int l = 0; l < bl; ++l)
        for (int p1 = 0; p1 < 2; ++p1)
            for (int p2 = 0; p2 < 2; ++p2)
                for (int r = 0; r < br; ++r) {
                    Complex128 sum(0.0, 0.0);
                    for (int m = 0; m < bm; ++m) sum += T1(l, p1, m) * T2(m, p2, r);
                    theta[static_cast<std::size_t>(l * 2 + p1) * cols + (p2 * br + r)] = sum;
                }
    // theta_new[l*2+po1, po2*br+r] = sum_{pi1,pi2} U[po1*2+po2, pi1*2+pi2] theta[l*2+pi1, pi2*br+r]
    std::vector<Complex128> expected(theta.size(), Complex128(0.0, 0.0));
    for (int l = 0; l < bl; ++l)
        for (int po1 = 0; po1 < 2; ++po1)
            for (int po2 = 0; po2 < 2; ++po2)
                for (int r = 0; r < br; ++r) {
                    Complex128 sum(0.0, 0.0);
                    for (int pi1 = 0; pi1 < 2; ++pi1)
                        for (int pi2 = 0; pi2 < 2; ++pi2)
                            sum += U[static_cast<std::size_t>((po1 * 2 + po2) * 4 + (pi1 * 2 + pi2))] *
                                   theta[static_cast<std::size_t>(l * 2 + pi1) * cols + (pi2 * br + r)];
                    expected[static_cast<std::size_t>(l * 2 + po1) * cols + (po2 * br + r)] = sum;
                }
    double scale_sq = 0.0;
    for (const auto& z : expected) scale_sq += z.real * z.real + z.imag * z.imag;
    const double scale = std::sqrt(scale_sq);

    chain.apply_two_qubit_gate(U, 0, 1);
    ASSERT_EQ(chain.svd_call_count(), 1u);

    const std::string file = block_file(rows, cols);
    if (!fs::exists(file)) {
        GTEST_SKIP() << "a " << rows << "x" << cols << " block was offered earlier "
                        "in this process, so the harvest kept that one; run this "
                        "suite alone to check the library block";
    }
    const Parsed p = parse_block(file);
    ASSERT_EQ(p.header, "autonne-hexfloat 1 matrix theta_2x26 2 26 rowmajor");
    ASSERT_EQ(p.values.size(), expected.size());
    // The library contracts the site tensors through a GEMM whose summation
    // order is its own, so the comparison is to rounding on a block of inner
    // dimension bm rather than to the bit.
    const double tol = kSlack * static_cast<double>(bm) * kEps * scale;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(p.values[i].real, expected[i].real, tol) << "entry " << i;
        EXPECT_NEAR(p.values[i].imag, expected[i].imag, tol) << "entry " << i;
    }
#endif
}

TEST(V11281ThetaHarvest, SimulatorRunOffersEverySplit) {
    SKIP_UNLESS_HARVEST();
#ifdef LINDBLAD_MPS_THETA_HARVEST
    // Through the main entry point, so the offer site inside the adjacent
    // kernel is reached by a real run. A three-qubit GHZ forms a 2x2 block at
    // the first cx and a 4x2 block at the second. Whether those two files
    // appear here depends on what ran earlier in this process, so the files
    // are asserted by the fresh-process histogram test rather than here; this
    // test pins that the run performs the two splits the histogram expects.
    ScopedTempCwd cwd;
    QuantumCircuit qc(3);
    qc.h(0).cx(0, 1).cx(1, 2);
    MPSSimulator sim;
    const auto r = sim.run(qc, 4, 0, kSeed);
    EXPECT_EQ(r.final_state.svd_call_count(), 2u);
    for (const auto& shape : {std::pair<int, int>{2, 2}, std::pair<int, int>{4, 2}}) {
        const std::string file = block_file(shape.first, shape.second);
        if (fs::exists(file)) {
            const Parsed p = parse_block(file);
            EXPECT_EQ(p.lines, static_cast<std::size_t>(shape.first) * shape.second)
                << file << " is malformed";
        }
    }
#endif
}
