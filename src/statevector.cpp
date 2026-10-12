// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#include "lindblad/statevector.hpp"
#include "lindblad/detail/born_draw.hpp"
#include "lindblad/detail/memory_budget.hpp"
#include "lindblad/detail/sv_runs.hpp"
#include "lindblad/detail/sv_sampler.hpp"
#include "lindblad/detail/text.hpp"
#include "lindblad/detail/validate_physical.hpp"
#include "lindblad/hw_info.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <cstring>
#include <random>
#include <stdexcept>
#include <numeric>

namespace lindblad {

// =============================================================================
// Construction / Destruction
// =============================================================================

namespace {
// Validate the qubit count BEFORE it reaches `1ULL << n` in the initializer
// list. Computing the shift with a negative or oversized n is undefined
// behaviour (shift-width overflow), so the guard must run first — a body check
// after the init list is too late. Called from the n_qubits member initializer,
// which precedes `dim` in declaration order, so a bad n throws before the shift.
inline int validated_n_qubits(int n, QubitLimit limit) {
    const int ceiling = max_statevector_qubits(limit);
    if (n < 1 || n > ceiling) {
        std::string msg = "n_qubits must be in [1, " + std::to_string(ceiling) +
                          "], got " + std::to_string(n);
        if (limit == QubitLimit::Enforce && n > ceiling && n <= LIFTED_MAX_QUBITS)
            msg += "; QubitLimit::Lift raises the ceiling to " +
                   std::to_string(LIFTED_MAX_QUBITS);
        detail::raise<InvalidArgument>("Statevector", msg);
    }
    return n;
}

// Alignment of each amplitude array: the page size when the array fills at
// least a page, so the two arrays start at the same offset within a page.
// A kernel loads re[i] and im[i], then stores them; if the arrays sat 64 bytes
// apart modulo 4 KiB (consecutive heap blocks do), the next iteration's load
// of im would share its low 12 address bits with the store to re just issued,
// and the CPU, which matches loads against pending stores on those bits,
// would hold the load back: measured 13% on a Grover circuit at 2^21
// amplitudes. A smaller array keeps cache-line alignment.
std::size_t state_alignment(std::size_t count) {
    const std::size_t page = hw::page_bytes();
    const bool usable = page >= CACHE_LINE_SIZE && (page & (page - 1)) == 0;
    return usable && count * sizeof(double) >= page ? page : CACHE_LINE_SIZE;
}
} // namespace

Statevector::Statevector(int n_qubits, QubitLimit limit)
    : n_qubits(validated_n_qubits(n_qubits, limit))
    , dim(1ULL << this->n_qubits)   // this->n_qubits is already validated
    , real_parts(nullptr)
    , imag_parts(nullptr)
{
    const std::size_t alignment = state_alignment(dim);
    real_parts = aligned_alloc_doubles(dim, alignment);
    // A constructor that throws runs no destructor, so the first buffer is
    // freed here when the second cannot be had.
    try {
        imag_parts = aligned_alloc_doubles(dim, alignment);
    } catch (...) {
        aligned_free(real_parts);
        throw;
    }

    initialize();
}

Statevector::~Statevector() {
    aligned_free(real_parts);
    aligned_free(imag_parts);
}

// =============================================================================
// Move semantics
// =============================================================================

Statevector::Statevector(Statevector&& other) noexcept
    : n_qubits(other.n_qubits)
    , dim(other.dim)
    , real_parts(other.real_parts)
    , imag_parts(other.imag_parts)
{
    other.real_parts = nullptr;
    other.imag_parts = nullptr;
    other.n_qubits = 0;
    other.dim = 0;
}

Statevector& Statevector::operator=(Statevector&& other) noexcept {
    if (this != &other) {
        aligned_free(real_parts);
        aligned_free(imag_parts);

        n_qubits = other.n_qubits;
        dim = other.dim;
        real_parts = other.real_parts;
        imag_parts = other.imag_parts;

        other.real_parts = nullptr;
        other.imag_parts = nullptr;
        other.n_qubits = 0;
        other.dim = 0;
    }
    return *this;
}

// =============================================================================
// Initialisation
// =============================================================================

void Statevector::initialize() {
    // Parallel zero-fill causes first-touch to distribute pages across NUMA
    // nodes on multi-socket hardware; free on UMA.
    double* re = real_parts;
    double* im = imag_parts;
    detail::for_each_share(dim, detail::streaming_work(dim), [re, im](size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) {
            re[i] = 0.0;
            im[i] = 0.0;
        }
    });
    real_parts[0] = 1.0;  // |0...0⟩
}

void Statevector::initialize_basis(size_t k) {
    if (k >= dim) {
        throw std::out_of_range("Basis state index out of range");
    }
    double* re = real_parts;
    double* im = imag_parts;
    detail::for_each_share(dim, detail::streaming_work(dim), [re, im](size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) {
            re[i] = 0.0;
            im[i] = 0.0;
        }
    });
    real_parts[k] = 1.0;
}

// Both overloads judge the policy against the CALLER'S buffer before writing
// anything, so a hand-over this refuses leaves the object holding whatever it
// held before rather than the amplitudes just rejected. That includes a repair
// that cannot run: a buffer with no norm to divide out is settled before the
// copy, so the response arrives with the object still intact. Under
// Repair::Attempt on a buffer that CAN be rescaled the copy happens first and
// normalize() runs on the result, which is the same state either way and
// avoids scaling the caller's memory.
void Statevector::set_amplitudes(const double* real, const double* imag,
                                 size_t count, ValidationOptions validation) {
    if (count != dim) {
        throw std::invalid_argument("Amplitude count must match dimension");
    }
    bool repair = false;
    if (!detail::measurement_unused(validation)) {
        const char* ctx = "Statevector::set_amplitudes";
        const double ns = detail::state_norm_sq(real, imag, count);
        repair = detail::check_normalized(ns, validation, ctx) &&
                 detail::normalization_repairable(ns, validation, ctx,
                                                  detail::STATE_NORMALIZATION);
    }
    std::memcpy(real_parts, real, dim * sizeof(double));
    std::memcpy(imag_parts, imag, dim * sizeof(double));
    if (repair) normalize();
}

void Statevector::set_amplitudes(const std::vector<Complex128>& amplitudes,
                                 ValidationOptions validation) {
    if (amplitudes.size() != dim) {
        throw std::invalid_argument("Amplitude count must match dimension");
    }
    bool repair = false;
    if (!detail::measurement_unused(validation)) {
        const char* ctx = "Statevector::set_amplitudes";
        const double ns =
            detail::state_norm_sq(amplitudes.data(), amplitudes.size());
        repair = detail::check_normalized(ns, validation, ctx) &&
                 detail::normalization_repairable(ns, validation, ctx,
                                                  detail::STATE_NORMALIZATION);
    }
    for (size_t i = 0; i < dim; ++i) {
        real_parts[i] = amplitudes[i].real;
        imag_parts[i] = amplitudes[i].imag;
    }
    if (repair) normalize();
}

// =============================================================================
// Amplitude access
// =============================================================================

Complex128 Statevector::amplitude(size_t index) const {
    if (index >= dim) {
        throw std::out_of_range("Amplitude index out of range");
    }
    return {real_parts[index], imag_parts[index]};
}

std::vector<Complex128> Statevector::amplitudes() const {
    std::vector<Complex128> result(dim);
    for (size_t i = 0; i < dim; ++i) {
        result[i] = {real_parts[i], imag_parts[i]};
    }
    return result;
}

// =============================================================================
// Probabilities
// =============================================================================

double Statevector::probability(size_t index) const {
    if (index >= dim) {
        throw std::out_of_range("Probability index out of range");
    }
    return real_parts[index] * real_parts[index] +
           imag_parts[index] * imag_parts[index];
}

std::vector<double> Statevector::probabilities() const {
    // Serial on purpose. The vector's constructor zero-fills the whole result
    // on the calling thread, so the result sits in that core's cache; a team
    // writing it would pull every line across to its own cores, and a team
    // spanning two L3 domains does that at a third of one thread's speed
    // (measured at 2^16 to 2^20 amplitudes), the next call's zero-fill pulling
    // the lines back again.
    std::vector<double> probs(dim);
    for (size_t i = 0; i < dim; ++i) {
        probs[i] = real_parts[i] * real_parts[i] + imag_parts[i] * imag_parts[i];
    }
    return probs;
}

// =============================================================================
// Norm
// =============================================================================

// Each thread sums its share and the shares are added in thread order
// (detail::reduce_shares), so the last bits depend on the team size and on
// nothing else.
double Statevector::norm_sq() const {
    const double* re = real_parts;
    const double* im = imag_parts;
    return detail::reduce_shares(
        dim, detail::streaming_work(dim), 0.0,
        [re, im](size_t b, size_t e) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t i = b; i < e; ++i) sum += re[i] * re[i] + im[i] * im[i];
            return sum;
        },
        [](double acc, double part) { return acc + part; });
}

double Statevector::norm() const {
    return std::sqrt(norm_sq());
}

void Statevector::normalize() {
    const double n = norm();
    // Refuses rather than returning quietly. A caller who asked for
    // normalization and got an unnormalized state back has been told nothing,
    // and the two states this rejects (zero, and non-finite) are precisely the
    // ones where dividing by the norm produces garbage rather than a state.
    if (!is_normalizable(n)) {
        detail::raise<RuntimeFailure>("Statevector::normalize",
            "no norm to divide out; the state is zero or non-finite");
    }
    const double inv_n = 1.0 / n;
    double* re = real_parts;
    double* im = imag_parts;
    detail::for_each_share(dim, detail::streaming_work(dim), [re, im, inv_n](size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) {
            re[i] *= inv_n;
            im[i] *= inv_n;
        }
    });
}

// Measured through the quarantined sum rather than through norm_sq() above:
// this value is compared against a tolerance, and norm_sq() splits the sum
// across threads, so its last bits depend on the thread count.
// A verdict that changes with how many cores happened to be free is not one.
bool Statevector::is_normalized(double atol) const {
    return std::abs(detail::state_norm_sq(real_parts, imag_parts, dim) - 1.0) <=
           atol;
}

void Statevector::check_normalized(ValidationOptions validation) {
    // Returns before measuring when nothing would consume the residual. The
    // sum is a full 2^n sweep, and opting out of a check is meant to cost a
    // branch rather than a pass.
    if (detail::measurement_unused(validation)) return;
    const char* ctx = "Statevector::check_normalized";
    const double ns = detail::state_norm_sq(real_parts, imag_parts, dim);
    if (detail::check_normalized(ns, validation, ctx) &&
        detail::normalization_repairable(ns, validation, ctx,
                                         detail::STATE_NORMALIZATION)) {
        normalize();
    }
}

// =============================================================================
// Inner product
// =============================================================================

Complex128 Statevector::inner_product(const Statevector& other) const {
    if (dim != other.dim) {
        throw std::invalid_argument("Dimension mismatch for inner product");
    }

    // ⟨this|other⟩ = sum_i conj(this_i) * other_i, summed share by share and
    // added in thread order (detail::reduce_shares).
    struct Sum {
        double re;
        double im;
    };
    const double* ar = real_parts;
    const double* ai = imag_parts;
    const double* br = other.real_parts;
    const double* bi = other.imag_parts;
    const Sum s = detail::reduce_shares(
        dim, detail::streaming_work(dim), Sum{0.0, 0.0},
        [ar, ai, br, bi](size_t b, size_t e) {
            double re = 0.0;
            double im = 0.0;
            #pragma omp simd reduction(+:re,im)
            for (size_t i = b; i < e; ++i) {
                re += ar[i] * br[i] + ai[i] * bi[i];
                im += ar[i] * bi[i] - ai[i] * br[i];
            }
            return Sum{re, im};
        },
        [](Sum acc, Sum part) { return Sum{acc.re + part.re, acc.im + part.im}; });
    return {s.re, s.im};
}

// =============================================================================
// Measurement sampling
// =============================================================================

// Both samplers draw from |a_i|^2 / total, the state's own normalised
// distribution (detail::born_draw_*), and refuse a state with no norm before
// drawing.

std::string Statevector::measure_once(uint64_t seed) const {
    double total = 0.0;
    for (size_t i = 0; i < dim; ++i)
        total += real_parts[i] * real_parts[i] + imag_parts[i] * imag_parts[i];
    detail::require_norm_to_sample(std::sqrt(total), "Statevector::measure_once");

    std::mt19937_64 rng(seed == 0 ? std::random_device{}() : seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    const size_t outcome = detail::born_draw_linear(
        dim, total, dist(rng), [this](size_t i) {
            return real_parts[i] * real_parts[i] + imag_parts[i] * imag_parts[i];
        });

    // Convert to bitstring (MSB first)
    std::string bits(n_qubits, '0');
    for (int b = 0; b < n_qubits; ++b) {
        if ((outcome >> (n_qubits - 1 - b)) & 1) {
            bits[b] = '1';
        }
    }
    return bits;
}

std::unordered_map<std::string, int> Statevector::sample_counts(
    int shots, uint64_t seed
) const {
    std::unordered_map<std::string, int> counts;
    const size_t draws = shots > 0 ? static_cast<size_t>(shots) : 0;

    const detail::StateSampler sampler(real_parts, imag_parts, dim, draws);
    detail::require_norm_to_sample(std::sqrt(sampler.total()), "Statevector::sample_counts");

    std::mt19937_64 rng(seed == 0 ? std::random_device{}() : seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    // One bitstring per distinct outcome, qubit 0 rightmost.
    detail::sample_outcomes(sampler, draws, [&] { return dist(rng); },
                            [&](std::uint64_t outcome, int count) {
                                std::string bits(n_qubits, '0');
                                for (int b = 0; b < n_qubits; ++b) {
                                    if ((outcome >> (n_qubits - 1 - b)) & 1) {
                                        bits[b] = '1';
                                    }
                                }
                                counts[bits] += count;
                            });

    return counts;
}

// =============================================================================
// Clone
// =============================================================================

Statevector Statevector::clone() const {
    Statevector copy(n_qubits);
    std::memcpy(copy.real_parts, real_parts, dim * sizeof(double));
    std::memcpy(copy.imag_parts, imag_parts, dim * sizeof(double));
    return copy;
}

// =============================================================================
// String representation
// =============================================================================

std::string Statevector::to_string(int precision) const {
    detail::TextBuilder oss;
    oss << "Statevector(" << detail::integer_text(n_qubits) << " qubits, dim="
        << detail::integer_text(dim) << "):\n";

    size_t max_show = std::min(dim, static_cast<size_t>(32));

    for (size_t i = 0; i < max_show; ++i) {
        double r = real_parts[i];
        double im = imag_parts[i];
        double prob = r * r + im * im;

        if (prob > 1e-12) {
            // Format as binary string
            std::string bits(n_qubits, '0');
            for (int b = 0; b < n_qubits; ++b) {
                if ((i >> (n_qubits - 1 - b)) & 1) bits[b] = '1';
            }
            oss << "  |" << bits << "⟩: "
                << detail::fixed_text(r, precision) << (im >= 0 ? "+" : "")
                << detail::fixed_text(im, precision) << "i"
                << "  (p=" << detail::fixed_text(prob, precision) << ")\n";
        }
    }

    if (dim > max_show) {
        oss << "  ... (" << detail::integer_text(dim - max_show) << " more entries)\n";
    }

    return std::move(oss).str();
}

namespace detail {

// =============================================================================
// StateSampler (sv_sampler.hpp)
// =============================================================================

StateSampler::StateSampler(const double* re, const double* im, std::size_t dim,
                           std::size_t shots)
    : re_(re), im_(im), dim_(dim), block_(block_for(dim, shots)), nb_(dim / block_) {
    prefix_.reset(new double[nb_]);

    // Block sums across a team, written where each thread will leave them;
    // then the running sum on this thread.
    double* prefix = prefix_.get();
    const std::size_t block = block_;
    for_each_share(nb_, streaming_work(dim), [re, im, prefix, block](std::size_t b, std::size_t e) {
        for (std::size_t k = b; k < e; ++k) {
            const std::size_t base = k * block;
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (std::size_t i = base; i < base + block; ++i) sum += re[i] * re[i] + im[i] * im[i];
            prefix[k] = sum;
        }
    });
    double acc = 0.0;
    for (std::size_t k = 0; k < nb_; ++k) {
        acc += prefix[k];
        prefix[k] = acc;
    }
    total_ = acc;

    // The last amplitude of positive weight: in the last block whose running
    // sum grows, the last amplitude of positive weight.
    std::size_t k = nb_ - 1;
    while (k > 0 && prefix[k] <= prefix[k - 1]) --k;
    for (std::size_t i = (k + 1) * block; i-- > k * block;) {
        if (re[i] * re[i] + im[i] * im[i] > 0.0) {
            last_positive_ = i;
            break;
        }
    }
}

std::size_t StateSampler::block_for(std::size_t dim, std::size_t shots) noexcept {
    const std::size_t per_shot = dim / (shots > 0 ? shots : 1);
    return std::clamp<std::size_t>(std::bit_floor(std::max<std::size_t>(per_shot, 1)), 1,
                                   std::min(kMaxBlock, std::max<std::size_t>(dim, 1)));
}

bool StateSampler::dense_tally(std::size_t dim, std::size_t shots) noexcept {
    return dim / kDenseTallyPerShot <= shots;
}

std::uint64_t StateSampler::scratch_bytes(std::size_t dim, std::size_t shots) noexcept {
    const std::uint64_t blocks = dim / block_for(dim, shots);
    const std::uint64_t chunk = std::min<std::uint64_t>(shots, kSampleChunk);
    std::uint64_t bytes = saturating_mul(blocks, sizeof(double));
    bytes = saturating_add(bytes, saturating_mul(chunk, sizeof(double) + sizeof(std::size_t)));
    if (shots > 0 && dense_tally(dim, shots))
        bytes = saturating_add(bytes, saturating_mul(dim, sizeof(int)));
    return bytes;
}

std::size_t StateSampler::draw(double u) const noexcept {
    const double target = u * total_;
    const double* prefix = prefix_.get();
    const std::size_t k =
        static_cast<std::size_t>(std::upper_bound(prefix, prefix + nb_, target) - prefix);
    if (k == nb_) return last_positive_;
    double running = k > 0 ? prefix[k - 1] : 0.0;
    std::size_t last = k * block_;
    for (std::size_t i = k * block_; i < (k + 1) * block_; ++i) {
        const double w = re_[i] * re_[i] + im_[i] * im_[i];
        if (w <= 0.0) continue;
        running += w;
        last = i;
        if (target < running) return i;
    }
    return last;
}

void StateSampler::draw_all(const double* u, std::size_t* out, std::size_t n) const {
    // A draw walks up to block_ amplitudes, so a chunk is worth n * block_
    // amplitudes of streaming work.
    const std::size_t work =
        block_ > std::numeric_limits<std::size_t>::max() / std::max<std::size_t>(n, 1)
            ? std::numeric_limits<std::size_t>::max()
            : n * block_;
    for_each_share(n, streaming_work(work), [this, u, out](std::size_t b, std::size_t e) {
        for (std::size_t i = b; i < e; ++i) out[i] = draw(u[i]);
    });
}

bool state_is_finite_and_nonzero(const Statevector& sv) noexcept {
    const std::size_t dim = sv.dim;
    const double* re = sv.real_parts;
    const double* im = sv.imag_parts;
    struct Scan {
        double norm_sq;
        std::size_t non_finite;
    };
    const Scan s = reduce_shares(
        dim, streaming_work(dim), Scan{0.0, 0},
        [re, im](std::size_t b, std::size_t e) {
            Scan part{0.0, 0};
            for (std::size_t i = b; i < e; ++i) {
                if (!is_finite_strict(re[i]) || !is_finite_strict(im[i])) ++part.non_finite;
                part.norm_sq += re[i] * re[i] + im[i] * im[i];
            }
            return part;
        },
        [](Scan acc, Scan part) {
            return Scan{acc.norm_sq + part.norm_sq, acc.non_finite + part.non_finite};
        });
    return s.non_finite == 0 && s.norm_sq > 0.0;
}

}  // namespace detail

} // namespace lindblad
