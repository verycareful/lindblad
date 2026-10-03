// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#include "lindblad/failed_run.hpp"
#include "lindblad/detail/failure_collector.hpp"
#include "lindblad/detail/fidelity_ledger.hpp"
#include "lindblad/detail/json.hpp"
#include "lindblad/detail/report.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#if defined(__SSE4_2__)
#include <nmmintrin.h>
#endif

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

// =============================================================================
// failed_run_io - saving a failed run to a folder, and reading it back
// =============================================================================
// A saved run is a folder:
//
//   manifest.json            every scalar field, the options, where the run
//                            failed, save_note, and each file's size and CRC-32C
//   counts.json              the finished shots' counts
//   circuit.json             QuantumCircuit::to_json
//   noise_model.json         NoiseModel::to_json (density-matrix runs)
//   observations.json        every observation; states point at their own file
//   observations/<i>-<label>.bin
//   state.bin                the state the run was evolving
//
// It is written as <name>.partial and renamed when complete, so a folder whose
// name has no suffix was written whole. The manifest is written last and is
// the one file without a checksum of its own: it holds everyone else's.
//
// State files are little-endian:
//
//   "LBSTATE1"  u32 version (1)  u32 form  u32 n_qubits  u32 0
//   form 0, statevector     2^n doubles of real parts, then 2^n of imaginary
//   form 1, density matrix  u64 entries, then the entries as held (row-major)
//   form 2, MPS             the chain's settings, span, fidelity figures and
//                           counters, then u32 sites and per site i32 bond_left,
//                           i32 bond_right, u64 entries, the entries
//   form 3, stabilizer      u32 words per row, u32 rows (2n), u64 words, the
//                           X/Z words, u64 rows, the phase bytes
//
// Every buffer is streamed straight from where the state holds it, without a
// copy, which matters when the run failed for lack of memory.

namespace lindblad {
namespace detail {

// The state file is little-endian on every host. A big-endian host reverses
// each word on the way out and on the way in, so a folder saved on one host
// loads on any other; a host that is neither has no defined word order.
static_assert(std::endian::native == std::endian::little ||
                  std::endian::native == std::endian::big,
              "the failed-run state file needs a little- or big-endian host");
constexpr bool HOST_IS_LITTLE_ENDIAN = std::endian::native == std::endian::little;

namespace {

// Reverses each `word`-byte word of the `n` bytes at `p`, in place. Called only
// on a big-endian host, so a little-endian build compiles it and leaves it out.
[[maybe_unused]] void reverse_words(unsigned char* p, std::size_t n, std::size_t word) noexcept {
    for (std::size_t i = 0; i + word <= n; i += word) std::reverse(p + i, p + i + word);
}

}  // namespace

// =============================================================================
// crc32c
// =============================================================================

std::uint32_t crc32c(std::uint32_t crc, const void* data, std::size_t n) noexcept {
    const auto* p = static_cast<const unsigned char*>(data);
    crc = ~crc;
#if defined(__SSE4_2__)
    while (n >= 8) {
        std::uint64_t word;
        std::memcpy(&word, p, 8);
        crc = static_cast<std::uint32_t>(_mm_crc32_u64(crc, word));
        p += 8;
        n -= 8;
    }
    while (n--) crc = _mm_crc32_u8(crc, *p++);
#else
    // 0x82F63B78 is the reflected Castagnoli polynomial, the constant that
    // defines CRC-32C.
    static const auto table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1u) ? (c >> 1) ^ 0x82F63B78u : (c >> 1);
            t[i] = c;
        }
        return t;
    }();
    while (n--) crc = table[(crc ^ *p++) & 0xFFu] ^ (crc >> 8);
#endif
    return ~crc;
}

namespace {

namespace fs = std::filesystem;

constexpr const char* FAILED_RUN_FORMAT = "lindblad.failed_run";
constexpr int FAILED_RUN_VERSION = 1;
// The manifest's own size and CRC-32C. Every other file is checked against the
// manifest; this file is what the manifest is checked against.
constexpr const char* MANIFEST_CHECK_FILE = "manifest.crc32c";
constexpr char STATE_MAGIC[8] = {'L', 'B', 'S', 'T', 'A', 'T', 'E', '1'};
constexpr std::uint32_t STATE_VERSION = 1;
constexpr std::size_t HEADER_BYTES = 8 + 4 * sizeof(std::uint32_t);
// Large buffers are written and read in pieces of this size, so no single
// stream call has to hold a whole state.
constexpr std::size_t CHUNK_BYTES = std::size_t{64} << 20;
// Room kept free on the disk beside a state file.
constexpr std::uint64_t FREE_SPACE_MARGIN = std::uint64_t{64} << 20;

enum class FormCode : std::uint32_t { Statevector = 0, DensityMatrix = 1, MPS = 2, Stabilizer = 3 };

const char* form_word(FormCode form) {
    switch (form) {
        case FormCode::Statevector:   return "statevector";
        case FormCode::DensityMatrix: return "density_matrix";
        case FormCode::MPS:           return "mps";
        case FormCode::Stabilizer:    return "stabilizer";
    }
    return "?";
}

std::string mib(std::uint64_t bytes) { return std::to_string(bytes >> 20) + " MiB"; }

}  // namespace

// =============================================================================
// FailedRunFile / FailedRunReader - one file out, one file in
// =============================================================================

// A file written in one pass, keeping its CRC-32C and byte count as it goes.
// Any write that does not complete throws std::runtime_error naming the file
// as `name`, its path inside the folder as the manifest lists it
// ("observations/0-dm.bin"), so the save note points at one file.
class FailedRunFile {
public:
    FailedRunFile(fs::path path, std::string name)
        : path_(std::move(path)), name_(std::move(name)),
          out_(path_, std::ios::binary | std::ios::trunc) {
        if (!out_) throw std::runtime_error("cannot create " + name_);
    }

    void write(const void* data, std::size_t n) {
        const auto* p = static_cast<const char*>(data);
        while (n > 0) {
            const std::size_t piece = std::min(n, CHUNK_BYTES);
            out_.write(p, static_cast<std::streamsize>(piece));
            if (!out_) throw std::runtime_error("writing " + name_ + " failed");
            crc_ = crc32c(crc_, p, piece);
            bytes_ += piece;
            p += piece;
            n -= piece;
        }
    }
    void write_text(const std::string& text) { write(text.data(), text.size()); }
    // `n` bytes of `word`-byte numbers (8 for a double, a std::uint64_t or each
    // half of a Complex128), written little-endian whatever the host.
    void write_words(const void* data, std::size_t n, std::size_t word) {
        if constexpr (HOST_IS_LITTLE_ENDIAN) {
            write(data, n);
        } else {
            const auto* p = static_cast<const unsigned char*>(data);
            std::vector<unsigned char> chunk;
            while (n > 0) {
                const std::size_t piece = std::min(n, CHUNK_BYTES);
                chunk.assign(p, p + piece);
                reverse_words(chunk.data(), piece, word);
                write(chunk.data(), piece);
                p += piece;
                n -= piece;
            }
        }
    }
    template <class T>
    void write_value(T value) {
        static_assert(std::is_trivially_copyable_v<T>);
        write_words(&value, sizeof value, sizeof value);
    }

    // Closes the file and leaves it readable and writable by its owner only.
    void close() {
        out_.close();
        if (!out_) throw std::runtime_error("closing " + name_ + " failed");
#if !defined(_WIN32)
        std::error_code ec;
        fs::permissions(path_, fs::perms::owner_read | fs::perms::owner_write,
                        fs::perm_options::replace, ec);
#endif
    }

    std::uint64_t bytes() const noexcept { return bytes_; }
    std::uint32_t crc() const noexcept { return crc_; }

private:
    fs::path path_;
    std::string name_;
    std::ofstream out_;
    std::uint64_t bytes_ = 0;
    std::uint32_t crc_ = 0;
};

// A file read front to back, knowing how much is left, so a length read from
// the file is checked against what the file holds before anything is sized
// from it. A short file throws std::runtime_error.
class FailedRunReader {
public:
    explicit FailedRunReader(const fs::path& path) : in_(path, std::ios::binary) {
        if (!in_) throw std::runtime_error("cannot be opened");
        std::error_code ec;
        remaining_ = fs::file_size(path, ec);
        if (ec) throw std::runtime_error("has no readable size");
    }

    void read(void* data, std::uint64_t n) {
        if (n > remaining_) throw std::runtime_error("ends before its contents do");
        auto* p = static_cast<char*>(data);
        while (n > 0) {
            const std::size_t piece = static_cast<std::size_t>(std::min<std::uint64_t>(n, CHUNK_BYTES));
            in_.read(p, static_cast<std::streamsize>(piece));
            if (!in_) throw std::runtime_error("ends before its contents do");
            remaining_ -= piece;
            p += piece;
            n -= piece;
        }
    }
    // `n` bytes of `word`-byte numbers stored little-endian, in the host's
    // order once read.
    void read_words(void* data, std::uint64_t n, std::size_t word) {
        read(data, n);
        if constexpr (!HOST_IS_LITTLE_ENDIAN) {
            reverse_words(static_cast<unsigned char*>(data), static_cast<std::size_t>(n), word);
        }
    }
    template <class T>
    T read_value() {
        static_assert(std::is_trivially_copyable_v<T>);
        T value;
        read_words(&value, sizeof value, sizeof value);
        return value;
    }

    std::uint64_t remaining() const noexcept { return remaining_; }

private:
    std::ifstream in_;
    std::uint64_t remaining_ = 0;
};

namespace {

void write_header(FailedRunFile& f, FormCode form, int n_qubits) {
    f.write(STATE_MAGIC, sizeof STATE_MAGIC);
    f.write_value<std::uint32_t>(STATE_VERSION);
    f.write_value<std::uint32_t>(static_cast<std::uint32_t>(form));
    f.write_value<std::uint32_t>(static_cast<std::uint32_t>(n_qubits));
    f.write_value<std::uint32_t>(0);
}

// A value read from a file that must be one of `count` enumerators.
std::uint32_t read_enum(FailedRunReader& r, std::uint32_t count, const char* what) {
    const std::uint32_t v = r.read_value<std::uint32_t>();
    if (v >= count) {
        throw std::runtime_error(std::string("holds ") + what + " " + std::to_string(v) +
                                 ", which is not one this build knows");
    }
    return v;
}

}  // namespace

// =============================================================================
// StateFileAccess - the storage of the states with private members
// =============================================================================

struct StateFileAccess {
    // What write_mps writes before the sites, field by field in its order: the
    // bond cap, the cutoff, seven settings, the span, the two fidelity figures
    // and their flag, the truncation total, the three rescue counts, the floor,
    // the call and time counters, the residual excess, and the site count.
    static constexpr std::uint64_t MPS_SETTINGS_BYTES =
        sizeof(std::int32_t) + sizeof(double) + 7 * sizeof(std::uint32_t) +
        2 * sizeof(std::int32_t) + 2 * sizeof(double) + sizeof(std::uint32_t) +
        sizeof(double) + 3 * sizeof(std::uint64_t) + sizeof(double) +
        2 * sizeof(std::uint64_t) + sizeof(double) + sizeof(std::uint32_t);
    // What write_mps writes per site before its entries: two bonds and a count.
    static constexpr std::uint64_t MPS_SITE_HEADER_BYTES =
        2 * sizeof(std::int32_t) + sizeof(std::uint64_t);

    static std::uint64_t mps_bytes(const MPSState& c) {
        std::uint64_t bytes = HEADER_BYTES + MPS_SETTINGS_BYTES;
        for (const MPSTensor& t : c.tensors_) {
            bytes += MPS_SITE_HEADER_BYTES +
                     static_cast<std::uint64_t>(t.data.size()) * sizeof(Complex128);
        }
        return bytes;
    }

    static void write_mps(FailedRunFile& f, const MPSState& c) {
        write_header(f, FormCode::MPS, c.n_qubits);
        f.write_value<std::int32_t>(c.max_bond_dim);
        f.write_value<double>(c.cutoff);
        f.write_value<std::uint32_t>(static_cast<std::uint32_t>(c.svd_method));
        f.write_value<std::uint32_t>(static_cast<std::uint32_t>(c.svd_rejection));
        f.write_value<std::uint32_t>(c.svd_accept_gram ? 1u : 0u);
        f.write_value<std::uint32_t>(static_cast<std::uint32_t>(c.svd_report));
        f.write_value<std::uint32_t>(static_cast<std::uint32_t>(c.canonical_form));
        f.write_value<std::uint32_t>(static_cast<std::uint32_t>(c.unchecked_gates));
        f.write_value<std::uint32_t>(static_cast<std::uint32_t>(c.qubit_limit));
        f.write_value<std::int32_t>(c.span_lo);
        f.write_value<std::int32_t>(c.span_hi);
        f.write_value<double>(c.fidelity.retained_);
        f.write_value<double>(c.fidelity.distance_);
        f.write_value<std::uint32_t>(c.fidelity.valid_ ? 1u : 0u);
        f.write_value<double>(c.total_truncation_error);
        f.write_value<std::uint64_t>(c.jacobi_rescues);
        f.write_value<std::uint64_t>(c.gram_fallbacks);
        f.write_value<std::uint64_t>(c.ignored_rejections);
        f.write_value<double>(c.floor_rejected);
        f.write_value<std::uint64_t>(c.svd_calls);
        f.write_value<std::uint64_t>(c.svd_nanos);
        f.write_value<double>(c.max_verify_resid_excess);
        f.write_value<std::uint32_t>(static_cast<std::uint32_t>(c.tensors_.size()));
        for (const MPSTensor& t : c.tensors_) {
            f.write_value<std::int32_t>(t.bond_left);
            f.write_value<std::int32_t>(t.bond_right);
            f.write_value<std::uint64_t>(t.data.size());
            f.write_words(t.data.data(), t.data.size() * sizeof(Complex128), sizeof(double));
        }
    }

    static MPSState read_mps(FailedRunReader& r, int n_qubits) {
        // Every site takes at least its two bonds, its entry count and two
        // entries in the file, so the file's own size bounds the site count
        // before a chain of that length is built. Without this, a header
        // claiming 2^31 qubits would allocate a chain far larger than the file.
        constexpr std::uint64_t min_site_bytes = MPS_SITE_HEADER_BYTES + 2 * sizeof(Complex128);
        if (static_cast<std::uint64_t>(n_qubits) * min_site_bytes > r.remaining()) {
            throw std::runtime_error("is too short to hold a chain of " +
                                     std::to_string(n_qubits) + " sites");
        }
        const std::int32_t max_bond_dim = r.read_value<std::int32_t>();
        const double cutoff = r.read_value<double>();
        if (max_bond_dim < 1) throw std::runtime_error("holds a bond cap below 1");
        MPSState c(n_qubits, max_bond_dim, cutoff);
        c.svd_method = static_cast<SVDMethod>(read_enum(r, 4, "an SVD method"));
        c.svd_rejection = static_cast<SvdRejection>(read_enum(r, 3, "an SVD rejection policy"));
        c.svd_accept_gram = read_enum(r, 2, "a Gram-route flag") == 1;
        c.svd_report = static_cast<SvdReport>(read_enum(r, 2, "an SVD report policy"));
        c.canonical_form = static_cast<CanonicalForm>(read_enum(r, 2, "a canonical form"));
        c.unchecked_gates = static_cast<UncheckedGates>(read_enum(r, 2, "an unchecked-gates policy"));
        c.qubit_limit = static_cast<QubitLimit>(read_enum(r, 2, "a qubit limit"));
        const std::int32_t span_lo = r.read_value<std::int32_t>();
        const std::int32_t span_hi = r.read_value<std::int32_t>();
        c.fidelity.retained_ = r.read_value<double>();
        c.fidelity.distance_ = r.read_value<double>();
        c.fidelity.valid_ = read_enum(r, 2, "a fidelity flag") == 1;
        c.total_truncation_error = r.read_value<double>();
        c.jacobi_rescues = static_cast<std::size_t>(r.read_value<std::uint64_t>());
        c.gram_fallbacks = static_cast<std::size_t>(r.read_value<std::uint64_t>());
        c.ignored_rejections = static_cast<std::size_t>(r.read_value<std::uint64_t>());
        c.floor_rejected = r.read_value<double>();
        c.svd_calls = static_cast<std::size_t>(r.read_value<std::uint64_t>());
        c.svd_nanos = r.read_value<std::uint64_t>();
        c.max_verify_resid_excess = r.read_value<double>();

        const std::uint32_t sites = r.read_value<std::uint32_t>();
        if (sites != static_cast<std::uint32_t>(n_qubits)) {
            throw std::runtime_error("holds " + std::to_string(sites) + " sites for " +
                                     std::to_string(n_qubits) + " qubits");
        }
        std::vector<MPSTensor> tensors;
        tensors.reserve(sites);
        for (std::uint32_t i = 0; i < sites; ++i) {
            const std::int32_t bl = r.read_value<std::int32_t>();
            const std::int32_t br = r.read_value<std::int32_t>();
            const std::uint64_t entries = r.read_value<std::uint64_t>();
            // MPSTensor sizes itself as bl * 2 * br in int arithmetic, so the
            // product is checked to fit an int before it is formed there.
            const std::uint64_t expected = 2 * static_cast<std::uint64_t>(bl > 0 ? bl : 0) *
                                           static_cast<std::uint64_t>(br > 0 ? br : 0);
            if (bl < 1 || br < 1 || entries != expected ||
                expected > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
                entries * sizeof(Complex128) > r.remaining()) {
                throw std::runtime_error("site " + std::to_string(i) + " is malformed");
            }
            MPSTensor t(bl, br);
            r.read_words(t.data.data(), entries * sizeof(Complex128), sizeof(double));
            tensors.push_back(std::move(t));
        }
        for (std::uint32_t i = 0; i + 1 < sites; ++i) {
            if (tensors[i].bond_right != tensors[i + 1].bond_left) {
                throw std::runtime_error("sites " + std::to_string(i) + " and " +
                                         std::to_string(i + 1) + " disagree on their bond");
            }
        }
        if (sites > 0 && (tensors.front().bond_left != 1 || tensors.back().bond_right != 1)) {
            throw std::runtime_error("the chain's outer bonds are not 1");
        }
        const bool span_ok = sites == 0 ? (span_lo == 0 && span_hi == -1)
                                        : (span_lo >= 0 && span_lo <= span_hi &&
                                           span_hi < static_cast<std::int32_t>(sites));
        if (!span_ok) throw std::runtime_error("holds an open span outside the chain");
        c.tensors_ = std::move(tensors);
        c.span_lo = span_lo;
        c.span_hi = span_hi;
        return c;
    }

    static std::uint64_t stabilizer_bytes(const StabilizerState& s) {
        const std::uint64_t rows = 2 * static_cast<std::uint64_t>(s.n_qubits);
        return HEADER_BYTES + 24 + rows * static_cast<std::uint64_t>(s.wpr) * 8 + rows;
    }

    // Writes the 2n rows of generators; a scratch row a measurement appends
    // and removes is not part of the state.
    static void write_stabilizer(FailedRunFile& f, const StabilizerState& s) {
        const std::uint64_t rows = 2 * static_cast<std::uint64_t>(s.n_qubits);
        const std::uint64_t words = rows * static_cast<std::uint64_t>(s.wpr);
        write_header(f, FormCode::Stabilizer, s.n_qubits);
        f.write_value<std::uint32_t>(static_cast<std::uint32_t>(s.wpr));
        f.write_value<std::uint32_t>(static_cast<std::uint32_t>(rows));
        f.write_value<std::uint64_t>(words);
        f.write_words(s.tab.data(), static_cast<std::size_t>(words) * sizeof(std::uint64_t),
                      sizeof(std::uint64_t));
        f.write_value<std::uint64_t>(rows);
        f.write(s.ph.data(), static_cast<std::size_t>(rows));
    }

    static StabilizerState read_stabilizer(FailedRunReader& r, int n_qubits) {
        const std::uint32_t wpr = r.read_value<std::uint32_t>();
        const std::uint32_t rows = r.read_value<std::uint32_t>();
        const std::uint64_t words = r.read_value<std::uint64_t>();
        const std::uint64_t want_rows = 2 * static_cast<std::uint64_t>(n_qubits);
        const std::uint64_t want_wpr = (want_rows + 63) / 64 + 1;
        if (rows != want_rows || wpr != want_wpr || words != want_rows * want_wpr ||
            words * 8 > r.remaining()) {
            throw std::runtime_error("is not the tableau of " + std::to_string(n_qubits) +
                                     " qubits");
        }
        StabilizerState s(n_qubits);
        r.read_words(s.tab.data(), words * sizeof(std::uint64_t), sizeof(std::uint64_t));
        const std::uint64_t phases = r.read_value<std::uint64_t>();
        if (phases != want_rows) throw std::runtime_error("holds the wrong number of phases");
        r.read(s.ph.data(), phases);
        return s;
    }
};

namespace {

// =============================================================================
// State files
// =============================================================================

std::uint64_t state_file_bytes(const FailedRun::State& state) {
    return std::visit(
        [](const auto& s) -> std::uint64_t {
            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                return 0;
            } else if constexpr (std::is_same_v<T, Statevector>) {
                return HEADER_BYTES + 2 * static_cast<std::uint64_t>(s.dim) * sizeof(double);
            } else if constexpr (std::is_same_v<T, DensityMatrix>) {
                return HEADER_BYTES + 8 +
                       static_cast<std::uint64_t>(s.data.size()) * sizeof(Complex128);
            } else if constexpr (std::is_same_v<T, MPSState>) {
                return StateFileAccess::mps_bytes(s);
            } else {
                return StateFileAccess::stabilizer_bytes(s);
            }
        },
        state);
}

void write_statevector(FailedRunFile& f, const Statevector& sv) {
    write_header(f, FormCode::Statevector, sv.n_qubits);
    f.write_words(sv.real_parts, sv.dim * sizeof(double), sizeof(double));
    f.write_words(sv.imag_parts, sv.dim * sizeof(double), sizeof(double));
}

void write_density_matrix(FailedRunFile& f, const DensityMatrix& dm) {
    write_header(f, FormCode::DensityMatrix, dm.n_qubits);
    f.write_value<std::uint64_t>(dm.data.size());
    f.write_words(dm.data.data(), dm.data.size() * sizeof(Complex128), sizeof(double));
}

void write_state(FailedRunFile& f, const FailedRun::State& state) {
    std::visit(
        [&f](const auto& s) {
            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, Statevector>) {
                write_statevector(f, s);
            } else if constexpr (std::is_same_v<T, DensityMatrix>) {
                write_density_matrix(f, s);
            } else if constexpr (std::is_same_v<T, MPSState>) {
                StateFileAccess::write_mps(f, s);
            } else if constexpr (std::is_same_v<T, StabilizerState>) {
                StateFileAccess::write_stabilizer(f, s);
            }
        },
        state);
}

// Reads a state file back in full. Every length is checked against what the
// file holds before anything is allocated from it.
FailedRun::State read_state(const fs::path& path) {
    FailedRunReader r(path);
    char magic[sizeof STATE_MAGIC];
    r.read(magic, sizeof magic);
    if (std::memcmp(magic, STATE_MAGIC, sizeof magic) != 0) {
        throw std::runtime_error("is not a Lindblad state file");
    }
    const std::uint32_t version = r.read_value<std::uint32_t>();
    if (version != STATE_VERSION) {
        throw std::runtime_error("is state file version " + std::to_string(version) +
                                 "; this build reads version " + std::to_string(STATE_VERSION));
    }
    const std::uint32_t form = read_enum(r, 4, "a state form");
    const std::uint32_t n_qubits = r.read_value<std::uint32_t>();
    (void)r.read_value<std::uint32_t>();
    const int n = static_cast<int>(n_qubits);

    switch (static_cast<FormCode>(form)) {
        case FormCode::Statevector: {
            if (n_qubits < 1 || n_qubits > static_cast<std::uint32_t>(LIFTED_MAX_QUBITS) ||
                r.remaining() != (std::uint64_t{2} << n_qubits) * sizeof(double)) {
                throw std::runtime_error("does not hold a statevector of " +
                                         std::to_string(n_qubits) + " qubits");
            }
            Statevector sv(n, QubitLimit::Lift);
            r.read_words(sv.real_parts, sv.dim * sizeof(double), sizeof(double));
            r.read_words(sv.imag_parts, sv.dim * sizeof(double), sizeof(double));
            return FailedRun::State(std::move(sv));
        }
        case FormCode::DensityMatrix: {
            const std::uint64_t entries = r.read_value<std::uint64_t>();
            // 4^n entries of sizeof(Complex128) bytes must fit a 64-bit byte
            // count: 2n plus the entry's own power of two stays below 64.
            constexpr std::uint32_t max_dm_qubits = static_cast<std::uint32_t>(
                (std::numeric_limits<std::uint64_t>::digits - 1 -
                 (std::bit_width(sizeof(Complex128)) - 1)) / 2);
            if (n_qubits < 1 || n_qubits > max_dm_qubits ||
                entries != (std::uint64_t{1} << (2 * n_qubits)) ||
                r.remaining() != entries * sizeof(Complex128)) {
                throw std::runtime_error("does not hold a density matrix of " +
                                         std::to_string(n_qubits) + " qubits");
            }
            DensityMatrix dm(n);
            r.read_words(dm.data.data(), entries * sizeof(Complex128), sizeof(double));
            return FailedRun::State(std::move(dm));
        }
        case FormCode::MPS: {
            if (n_qubits > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("holds an impossible qubit count");
            }
            return FailedRun::State(StateFileAccess::read_mps(r, n));
        }
        case FormCode::Stabilizer: {
            // 2n rows of ceil(2n / 64) + 1 words; the file's own size bounds n.
            if (n_qubits > (1u << 20)) throw std::runtime_error("holds an impossible qubit count");
            return FailedRun::State(StateFileAccess::read_stabilizer(r, n));
        }
    }
    throw std::runtime_error("holds an unknown state form");
}

FormCode form_code(const FailedRun::State& state) {
    switch (state.index()) {
        case 1:  return FormCode::Statevector;
        case 2:  return FormCode::DensityMatrix;
        case 3:  return FormCode::MPS;
        default: return FormCode::Stabilizer;
    }
}

// =============================================================================
// Observation payloads
// =============================================================================

// Bytes a state payload's file needs, read off the payload without copying it.
std::uint64_t payload_state_bytes(const ObservationBundle::StatePayload& p) {
    switch (p.form) {
        case StateForm::Statevector: {
            const auto* sv = static_cast<const Statevector*>(p.state.get());
            return HEADER_BYTES + 2 * static_cast<std::uint64_t>(sv->dim) * sizeof(double);
        }
        case StateForm::DensityMatrix: {
            const auto* dm = static_cast<const DensityMatrix*>(p.state.get());
            return HEADER_BYTES + 8 + static_cast<std::uint64_t>(dm->data.size()) * sizeof(Complex128);
        }
        case StateForm::MPS:
            return StateFileAccess::mps_bytes(*static_cast<const MPSState*>(p.state.get()));
        case StateForm::Stabilizer:
            return StateFileAccess::stabilizer_bytes(
                *static_cast<const StabilizerState*>(p.state.get()));
    }
    return 0;
}

void write_payload_state(FailedRunFile& f, const ObservationBundle::StatePayload& p) {
    switch (p.form) {
        case StateForm::Statevector:
            write_statevector(f, *static_cast<const Statevector*>(p.state.get()));
            return;
        case StateForm::DensityMatrix:
            write_density_matrix(f, *static_cast<const DensityMatrix*>(p.state.get()));
            return;
        case StateForm::MPS:
            StateFileAccess::write_mps(f, *static_cast<const MPSState*>(p.state.get()));
            return;
        case StateForm::Stabilizer:
            StateFileAccess::write_stabilizer(f, *static_cast<const StabilizerState*>(p.state.get()));
            return;
    }
}

ObservationBundle::StatePayload state_payload(FailedRun::State&& state) {
    switch (state.index()) {
        case 1:
            return {StateForm::Statevector,
                    std::make_shared<const Statevector>(std::move(std::get<Statevector>(state)))};
        case 2:
            return {StateForm::DensityMatrix,
                    std::make_shared<const DensityMatrix>(std::move(std::get<DensityMatrix>(state)))};
        case 3:
            return {StateForm::MPS,
                    std::make_shared<const MPSState>(std::move(std::get<MPSState>(state)))};
        case 4:
            return {StateForm::Stabilizer,
                    std::make_shared<const StabilizerState>(std::move(std::get<StabilizerState>(state)))};
        default:
            throw std::runtime_error("names a state file that holds no state");
    }
}

// The most characters of a label a state file's name keeps. A file name is
// limited to 255 bytes on every filesystem the library targets, and a label
// has no limit; 64 leaves the index and the extension room under any of them
// while still telling one file from another at a glance.
constexpr std::size_t LABEL_FILE_CHARS = 64;

// A label as a file name: its first LABEL_FILE_CHARS characters, with letters,
// digits, '.', '_' and '-' kept and anything else replaced by '_'. The entry's
// position is prefixed, so two labels that sanitise or shorten alike still name
// two files. observations.json records each file beside its full label, so
// nothing is lost when the name is shortened.
std::string label_file(std::size_t index, const std::string& label) {
    std::string name = std::to_string(index) + "-";
    const std::size_t kept = std::min(label.size(), LABEL_FILE_CHARS);
    for (std::size_t i = 0; i < kept; ++i) {
        const char c = label[i];
        const bool keep = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                          (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
        name += keep ? c : '_';
    }
    return "observations/" + name + ".bin";
}

std::string real_array(const std::vector<double>& values) {
    std::string out = "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out += ',';
        out += json_number(values[i]);
    }
    return out + "]";
}

std::vector<double> read_real_array(JsonReader& r) {
    std::vector<double> out;
    r.expect('[');
    while (r.peek() != ']') {
        if (r.peek() == ',') r.next();
        out.push_back(r.read_double());
    }
    r.expect(']');
    return out;
}

std::vector<int> read_int_array(JsonReader& r) {
    std::vector<int> out;
    r.expect('[');
    while (r.peek() != ']') {
        if (r.peek() == ',') r.next();
        out.push_back(r.read_int());
    }
    r.expect(']');
    return out;
}

// The raw JSON text of the value at the reader's position, which is how an
// option's value is kept: as written.
std::string read_raw_value(JsonReader& r) {
    r.skip_ws();
    const std::size_t start = r.pos;
    r.skip_value();
    std::string raw = r.s.substr(start, r.pos - start);
    while (!raw.empty() && (raw.back() == ' ' || raw.back() == '\n' || raw.back() == '\t' ||
                            raw.back() == '\r')) {
        raw.pop_back();
    }
    return raw;
}

// =============================================================================
// The folder
// =============================================================================

struct SavedFile {
    std::string name;
    std::uint64_t bytes;
    std::uint32_t crc;
};

// Writes `name` in `folder` through `body`, and records it. A file that cannot
// be written whole is removed, so a disk that fills part way through leaves no
// unlisted fragment behind in the folder.
template <class Body>
void write_file(const fs::path& folder, const std::string& name, std::vector<SavedFile>& files,
                Body&& body) {
    const fs::path path = folder / fs::path(name);
    try {
        FailedRunFile f(path, name);
        body(f);
        f.close();
        files.push_back(SavedFile{name, f.bytes(), f.crc()});
    } catch (...) {
        std::error_code ec;
        fs::remove(path, ec);
        throw;
    }
}

// Where saved runs go when the caller names no folder. Never the temporary
// directory, which on many systems is memory.
fs::path default_failed_run_dir() {
#if defined(_WIN32)
    if (const char* local = std::getenv("LOCALAPPDATA"); local != nullptr && *local != '\0') {
        return fs::path(local) / "lindblad" / "failed-runs";
    }
    return {};
#else
    // XDG_STATE_HOME counts only when absolute, as the XDG specification says.
    if (const char* xdg = std::getenv("XDG_STATE_HOME"); xdg != nullptr && *xdg != '\0') {
        const fs::path state(xdg);
        if (state.is_absolute()) return state / "lindblad" / "failed-runs";
    }
    if (const char* home = std::getenv("HOME"); home != nullptr && *home != '\0') {
        return fs::path(home) / ".local" / "state" / "lindblad" / "failed-runs";
    }
    return {};
#endif
}

// <YYYYmmdd-HHMMSS>-<pid>-<thread>-<counter>: two failures in the same second
// differ by thread or by the process-wide counter.
std::string failed_run_name() {
    static std::atomic<unsigned> counter{0};
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &now);
    const long pid = static_cast<long>(_getpid());
#else
    localtime_r(&now, &tm);
    const long pid = static_cast<long>(::getpid());
#endif
    char stamp[32];
    std::strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", &tm);
    const std::size_t thread = std::hash<std::thread::id>{}(std::this_thread::get_id());
    return std::string(stamp) + '-' + integer_text(pid) + '-' +
           integer_text(thread & 0xFFFFFFFFu, 16) + '-' + integer_text(counter.fetch_add(1));
}

std::string point_json(const std::optional<FailurePoint>& where) {
    if (!where) return "null";
    std::string out = "{\"shot\":" + std::to_string(where->shot) +
                      ",\"instruction\":" + std::to_string(where->instruction) +
                      ",\"gate\":" + json_escape(where->gate) + ",\"qubits\":[";
    for (std::size_t i = 0; i < where->qubits.size(); ++i) {
        if (i > 0) out += ',';
        out += std::to_string(where->qubits[i]);
    }
    return out + "]}";
}

void add_note(FailedRun& r, const std::string& text) {
    if (!r.save_note.empty()) r.save_note += ' ';
    r.save_note += text;
}

std::uint64_t free_bytes(const fs::path& folder) {
    std::error_code ec;
    const fs::space_info info = fs::space(folder, ec);
    return ec ? 0 : static_cast<std::uint64_t>(info.available);
}

}  // namespace

// =============================================================================
// save_failed_run
// =============================================================================

void save_failed_run(FailedRun& r, const std::filesystem::path& dir) noexcept {
    fs::path partial;
    try {
        const fs::path root = dir.empty() ? default_failed_run_dir() : dir;
        if (root.empty()) {
            add_note(r, "Nothing was saved: there is no folder to save into; set "
                        "RunPlan::Options::failed_run_dir, or HOME or XDG_STATE_HOME.");
            return;
        }
        std::error_code ec;
        fs::create_directories(root, ec);
        if (ec) {
            add_note(r, "Nothing was saved: " + root.string() + " cannot be created (" +
                            ec.message() + ").");
            return;
        }

        fs::path final_path;
        for (int attempt = 0; attempt < 16 && partial.empty(); ++attempt) {
            const std::string name = failed_run_name();
            const fs::path candidate = root / (name + ".partial");
            if (fs::create_directory(candidate, ec) && !ec) {
                partial = candidate;
                final_path = root / name;
            } else if (ec) {
                add_note(r, "Nothing was saved: a folder cannot be created in " + root.string() +
                                " (" + ec.message() + ").");
                return;
            }
        }
        if (partial.empty()) {
            add_note(r, "Nothing was saved: no unused folder name was found in " +
                            root.string() + ".");
            return;
        }
#if !defined(_WIN32)
        fs::permissions(partial, fs::perms::owner_all, fs::perm_options::replace, ec);
#endif

        std::vector<SavedFile> files;
        bool wrote_counts = false, wrote_circuit = false, wrote_noise = false;
        bool wrote_observations = false, wrote_state = false;

        // Each part on its own: one that cannot be written stays in memory and
        // is named in the note, and the others are still saved.
        try {
            std::vector<std::pair<std::string, int>> sorted(r.counts.begin(), r.counts.end());
            std::sort(sorted.begin(), sorted.end());
            std::string text = "{\"shots_completed\":" + std::to_string(r.shots_completed) +
                               ",\"counts\":{";
            for (std::size_t i = 0; i < sorted.size(); ++i) {
                if (i > 0) text += ',';
                text += json_escape(sorted[i].first) + ":" + std::to_string(sorted[i].second);
            }
            text += "}}";
            write_file(partial, "counts.json", files, [&](FailedRunFile& f) { f.write_text(text); });
            wrote_counts = true;
        } catch (const std::exception& e) {
            add_note(r, std::string("The counts were not saved: ") + e.what() + ".");
        }

        if (r.circuit) {
            try {
                const std::string text = r.circuit->to_json();
                write_file(partial, "circuit.json", files, [&](FailedRunFile& f) { f.write_text(text); });
                wrote_circuit = true;
            } catch (const std::exception& e) {
                add_note(r, std::string("The circuit was not saved: ") + e.what() + ".");
            }
        }

        if (r.noise_model) {
            try {
                const std::string text = r.noise_model->to_json();
                write_file(partial, "noise_model.json", files,
                           [&](FailedRunFile& f) { f.write_text(text); });
                wrote_noise = true;
            } catch (const std::exception& e) {
                add_note(r, std::string("The noise model was not saved: ") + e.what() + ".");
            }
        }

        const std::size_t files_before_observations = files.size();
        try {
            const std::vector<std::string> labels = r.observations.labels();
            std::uint64_t state_bytes = 0;
            for (const std::string& label : labels) {
                const auto& payload = r.observations.payload(label);
                if (const auto* sp = std::get_if<ObservationBundle::StatePayload>(&payload)) {
                    state_bytes += payload_state_bytes(*sp);
                }
            }
            const std::uint64_t available = free_bytes(partial);
            if (state_bytes > 0 && available < state_bytes + FREE_SPACE_MARGIN) {
                add_note(r, "The observations were not saved: their states need " +
                                mib(state_bytes) + ", " + mib(available) +
                                " free; they are in memory, take them with "
                                "lindblad::take_failed_run().");
            } else {
                if (state_bytes > 0) fs::create_directory(partial / "observations");
                std::string text = "{\"entries\":[";
                for (std::size_t i = 0; i < labels.size(); ++i) {
                    if (i > 0) text += ',';
                    const std::string& label = labels[i];
                    text += "{\"label\":" + json_escape(label) + ",";
                    const auto& payload = r.observations.payload(label);
                    if (const auto* v = std::get_if<double>(&payload)) {
                        text += "\"kind\":\"number\",\"value\":" + json_number(*v);
                    } else if (const auto* v = std::get_if<std::vector<double>>(&payload)) {
                        text += "\"kind\":\"reals\",\"value\":" + real_array(*v);
                    } else if (const auto* v = std::get_if<std::vector<Complex128>>(&payload)) {
                        text += "\"kind\":\"amplitudes\",\"value\":[";
                        for (std::size_t k = 0; k < v->size(); ++k) {
                            if (k > 0) text += ',';
                            text += "[" + json_number((*v)[k].real) + "," +
                                    json_number((*v)[k].imag) + "]";
                        }
                        text += "]";
                    } else if (const auto* v = std::get_if<std::vector<int>>(&payload)) {
                        text += "\"kind\":\"integers\",\"value\":[";
                        for (std::size_t k = 0; k < v->size(); ++k) {
                            if (k > 0) text += ',';
                            text += std::to_string((*v)[k]);
                        }
                        text += "]";
                    } else if (const auto* v = std::get_if<std::string>(&payload)) {
                        text += "\"kind\":\"text\",\"value\":" + json_escape(*v);
                    } else if (const auto* v = std::get_if<ObservationBundle::StatePayload>(&payload)) {
                        const std::string file = label_file(i, label);
                        write_file(partial, file, files,
                                   [&](FailedRunFile& f) { write_payload_state(f, *v); });
                        text += "\"kind\":\"state\",\"file\":" + json_escape(file);
                    }
                    text += "}";
                }
                text += "]}";
                write_file(partial, "observations.json", files,
                           [&](FailedRunFile& f) { f.write_text(text); });
                wrote_observations = true;
            }
        } catch (const std::exception& e) {
            // Whole or not at all: state files already written for earlier
            // entries go with the rest, and so does their folder.
            for (std::size_t k = files_before_observations; k < files.size(); ++k) {
                std::error_code ec;
                fs::remove(partial / fs::path(files[k].name), ec);
            }
            files.resize(files_before_observations);
            std::error_code ec;
            fs::remove(partial / "observations", ec);
            add_note(r, std::string("The observations were not saved: ") + e.what() + ".");
        }

        if (!std::holds_alternative<std::monostate>(r.state)) {
            try {
                const std::uint64_t need = state_file_bytes(r.state);
                const std::uint64_t available = free_bytes(partial);
                if (available < need + FREE_SPACE_MARGIN) {
                    add_note(r, "The state was not saved: it needs " + mib(need) + ", " +
                                    mib(available) +
                                    " free; it is in memory, take it with "
                                    "lindblad::take_failed_run().");
                } else {
                    write_file(partial, "state.bin", files,
                               [&](FailedRunFile& f) { write_state(f, r.state); });
                    wrote_state = true;
                }
            } catch (const std::exception& e) {
                add_note(r, std::string("The state was not saved: ") + e.what() +
                                "; it is in memory, take it with lindblad::take_failed_run().");
            }
        }

        // The manifest, last: it names every file above with its checksum.
        std::string m = "{\"format\":" + json_escape(FAILED_RUN_FORMAT) +
                        ",\"version\":" + std::to_string(FAILED_RUN_VERSION) +
                        ",\"library_version\":" + json_escape(r.library_version) +
                        ",\"entry_point\":" + json_escape(r.entry_point) +
                        ",\"backend\":" + json_escape(r.backend) +
                        ",\"n_qubits\":" + std::to_string(r.n_qubits) +
                        ",\"shots_requested\":" + std::to_string(r.shots_requested) +
                        ",\"shots_completed\":" + std::to_string(r.shots_completed) +
                        // A 64-bit seed is written as a string: a JSON number
                        // is a double, which holds only 53 bits exactly.
                        ",\"seed\":" + json_escape(std::to_string(r.seed)) +
                        ",\"state_form\":" +
                        (std::holds_alternative<std::monostate>(r.state)
                             ? std::string("null")
                             : json_escape(form_word(form_code(r.state)))) +
                        ",\"where\":" + point_json(r.where) +
                        ",\"exception_type\":" + json_escape(r.exception_type) +
                        ",\"message\":" + json_escape(r.message) + ",\"options\":{";
        for (std::size_t i = 0; i < r.options.size(); ++i) {
            if (i > 0) m += ',';
            m += json_escape(r.options[i].first) + ":" + r.options[i].second;
        }
        m += "},\"save_note\":" + json_escape(r.save_note) + ",\"files\":[";
        for (std::size_t i = 0; i < files.size(); ++i) {
            if (i > 0) m += ',';
            m += "{\"name\":" + json_escape(files[i].name) +
                 ",\"bytes\":" + std::to_string(files[i].bytes) +
                 ",\"crc32c\":" + std::to_string(files[i].crc) + "}";
        }
        m += "]}";
        std::vector<SavedFile> manifest_only;
        write_file(partial, "manifest.json", manifest_only,
                   [&](FailedRunFile& f) { f.write_text(m); });
        // An edit to the manifest (a file entry removed, a qubit count
        // changed) is refused on load like an edit to any other file.
        const std::string manifest_check =
            "{\"bytes\":" + std::to_string(manifest_only.front().bytes) +
            ",\"crc32c\":" + std::to_string(manifest_only.front().crc) + "}";
        write_file(partial, MANIFEST_CHECK_FILE, manifest_only,
                   [&](FailedRunFile& f) { f.write_text(manifest_check); });

        fs::rename(partial, final_path);
        partial.clear();
        r.saved_to = final_path;

        // What is on disk leaves memory; load_failed_run(saved_to) reads it
        // back. What could not be written stays.
        if (wrote_counts) r.counts.clear();
        if (wrote_observations) r.observations = ObservationBundle();
        if (wrote_circuit) r.circuit.reset();
        if (wrote_noise) r.noise_model.reset();
        if (wrote_state) r.state = std::monostate{};
    } catch (const std::exception& e) {
        try {
            add_note(r, std::string("Nothing was saved: ") + e.what() + ".");
        } catch (...) {
        }
    } catch (...) {
        try {
            add_note(r, "Nothing was saved.");
        } catch (...) {
        }
    }
    // A folder that was never completed is removed, so no half-written run
    // is left to be mistaken for one. The error_code overload still allocates
    // as it walks the folder, and this function never throws.
    if (!partial.empty()) {
        try {
            std::error_code ec;
            fs::remove_all(partial, ec);
        } catch (...) {
        }
    }
}

}  // namespace detail

// =============================================================================
// load_failed_run
// =============================================================================

namespace {

namespace fs = std::filesystem;

constexpr const char* LOAD = "load_failed_run";

// A manifest's file name must stay inside the folder: relative, no "..", and
// only the characters the writer uses.
bool safe_name(const std::string& name) {
    if (name.empty() || name.front() == '/' || name.find("..") != std::string::npos) return false;
    for (const char c : name) {
        const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                        (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-' || c == '/';
        if (!ok) return false;
    }
    return true;
}

std::string read_text(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot be opened");
    std::ostringstream o;
    o << in.rdbuf();
    return o.str();
}

// The file's CRC-32C and size, streamed.
std::pair<std::uint32_t, std::uint64_t> file_checksum(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot be opened");
    std::vector<char> buffer(std::size_t{1} << 20);
    std::uint32_t crc = 0;
    std::uint64_t bytes = 0;
    while (in) {
        in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize got = in.gcount();
        if (got <= 0) break;
        crc = detail::crc32c(crc, buffer.data(), static_cast<std::size_t>(got));
        bytes += static_cast<std::uint64_t>(got);
    }
    return {crc, bytes};
}

// Runs `parse` on one file, turning whatever it throws into InvalidArgument
// naming the file.
template <class Parse>
void parse_file(const std::string& name, Parse&& parse) {
    try {
        parse();
    } catch (const std::exception& e) {
        detail::raise<InvalidArgument>(LOAD, name + " " + e.what());
    }
}

// The seed the manifest records. It is written as a string of decimal digits,
// since a JSON number holds only 53 bits, and read back as exactly that: no
// sign, no white space, no prefix, nothing after the digits, and no value a
// 64-bit seed cannot hold.
std::uint64_t read_seed(const std::string& text) {
    const bool digits_only =
        !text.empty() &&
        std::all_of(text.begin(), text.end(), [](char c) { return c >= '0' && c <= '9'; });
    if (!digits_only) {
        throw std::runtime_error("records the seed \"" + text + "\", which is not a string of digits");
    }
    std::uint64_t seed = 0;
    const char* const end = text.data() + text.size();
    const auto [stop, ec] = std::from_chars(text.data(), end, seed);
    if (ec != std::errc() || stop != end) {
        throw std::runtime_error("records the seed \"" + text +
                                 "\", which is more than a 64-bit seed holds");
    }
    return seed;
}

// A byte count or checksum the manifest records: a whole number below
// 2^digits. It arrives as a double, and converting one outside the target's
// range is undefined, so the range is checked first.
std::uint64_t read_unsigned(detail::JsonReader& j, int digits, const char* what) {
    const double d = j.read_number();
    if (!(d >= 0.0) || d >= std::ldexp(1.0, digits) || d != std::floor(d)) {
        throw std::runtime_error("records " + std::to_string(d) + " as " + what);
    }
    return static_cast<std::uint64_t>(d);
}

// The qubit count of a state read from a state file.
int state_qubits(const FailedRun::State& state) {
    return std::visit(
        [](const auto& s) -> int {
            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                return -1;
            } else {
                return s.n_qubits;
            }
        },
        state);
}

// The state form a backend's run evolves, as the manifest names it.
std::string form_of_backend(const std::string& backend) {
    return backend == "clifford" ? std::string(detail::form_word(detail::FormCode::Stabilizer))
                                 : backend;
}

}  // namespace

FailedRun load_failed_run(const std::filesystem::path& folder) {
    FailedRun r;
    std::string manifest;
    try {
        manifest = read_text(folder / "manifest.json");
    } catch (const std::exception& e) {
        detail::raise<InvalidArgument>(LOAD, "manifest.json in " + folder.string() + " " + e.what());
    }

    // The manifest is checked against its own size and CRC-32C before anything
    // in it is believed.
    std::string check;
    try {
        check = read_text(folder / detail::MANIFEST_CHECK_FILE);
    } catch (const std::exception& e) {
        detail::raise<InvalidArgument>(LOAD, std::string(detail::MANIFEST_CHECK_FILE) + " in " +
                                                 folder.string() + " " + e.what());
    }
    std::uint64_t check_bytes = 0;
    std::uint64_t check_crc = 0;
    parse_file(detail::MANIFEST_CHECK_FILE, [&] {
        detail::JsonReader j{check};
        j.expect('{');
        while (j.peek() != '}') {
            if (j.peek() == ',') j.next();
            const std::string key = j.read_string();
            j.expect(':');
            if (key == "bytes") {
                check_bytes = read_unsigned(j, std::numeric_limits<std::uint64_t>::digits,
                                            "a byte count");
            } else if (key == "crc32c") {
                check_crc = read_unsigned(j, std::numeric_limits<std::uint32_t>::digits,
                                          "a CRC-32C");
            } else {
                j.skip_value();
            }
        }
        j.expect('}');
    });
    const std::uint32_t manifest_crc = detail::crc32c(0, manifest.data(), manifest.size());
    if (manifest.size() != check_bytes) {
        detail::raise<InvalidArgument>(LOAD, "manifest.json holds " +
                                                 std::to_string(manifest.size()) +
                                                 " bytes; " + detail::MANIFEST_CHECK_FILE +
                                                 " recorded " + std::to_string(check_bytes));
    }
    if (manifest_crc != check_crc) {
        detail::raise<InvalidArgument>(LOAD, "manifest.json fails its checksum: CRC-32C " +
                                                 std::to_string(manifest_crc) + ", " +
                                                 detail::MANIFEST_CHECK_FILE + " recorded " +
                                                 std::to_string(check_crc));
    }

    struct Listed {
        std::string name;
        std::uint64_t bytes = 0;
        std::uint32_t crc = 0;
    };
    std::vector<Listed> files;
    std::string format;
    int version = 0;
    std::string state_form;

    parse_file("manifest.json", [&] {
        detail::JsonReader j{manifest};
        j.expect('{');
        while (j.peek() != '}') {
            if (j.peek() == ',') j.next();
            const std::string key = j.read_string();
            j.expect(':');
            if (key == "format") {
                format = j.read_string();
            } else if (key == "version") {
                version = j.read_int();
            } else if (key == "library_version") {
                r.library_version = j.read_string();
            } else if (key == "entry_point") {
                r.entry_point = j.read_string();
            } else if (key == "backend") {
                r.backend = j.read_string();
            } else if (key == "n_qubits") {
                r.n_qubits = j.read_int();
            } else if (key == "shots_requested") {
                r.shots_requested = j.read_int();
            } else if (key == "shots_completed") {
                r.shots_completed = j.read_int();
            } else if (key == "seed") {
                r.seed = read_seed(j.read_string());
            } else if (key == "state_form") {
                if (j.peek() == '"') state_form = j.read_string();
                else j.skip_value();
            } else if (key == "where") {
                if (j.peek() != '{') {
                    j.skip_value();
                    continue;
                }
                FailurePoint point;
                j.expect('{');
                while (j.peek() != '}') {
                    if (j.peek() == ',') j.next();
                    const std::string wkey = j.read_string();
                    j.expect(':');
                    if (wkey == "shot") point.shot = j.read_int();
                    else if (wkey == "instruction") point.instruction = j.read_int();
                    else if (wkey == "gate") point.gate = j.read_string();
                    else if (wkey == "qubits") point.qubits = detail::read_int_array(j);
                    else j.skip_value();
                }
                j.expect('}');
                r.where = point;
            } else if (key == "exception_type") {
                r.exception_type = j.read_string();
            } else if (key == "message") {
                r.message = j.read_string();
            } else if (key == "options") {
                j.expect('{');
                while (j.peek() != '}') {
                    if (j.peek() == ',') j.next();
                    std::string name = j.read_string();
                    j.expect(':');
                    r.options.emplace_back(std::move(name), detail::read_raw_value(j));
                }
                j.expect('}');
            } else if (key == "save_note") {
                r.save_note = j.read_string();
            } else if (key == "files") {
                j.expect('[');
                while (j.peek() != ']') {
                    if (j.peek() == ',') j.next();
                    Listed entry;
                    j.expect('{');
                    while (j.peek() != '}') {
                        if (j.peek() == ',') j.next();
                        const std::string fkey = j.read_string();
                        j.expect(':');
                        if (fkey == "name") {
                            entry.name = j.read_string();
                        } else if (fkey == "bytes") {
                            entry.bytes = read_unsigned(
                                j, std::numeric_limits<std::uint64_t>::digits, "a byte count");
                        } else if (fkey == "crc32c") {
                            entry.crc = static_cast<std::uint32_t>(read_unsigned(
                                j, std::numeric_limits<std::uint32_t>::digits, "a CRC-32C"));
                        } else {
                            j.skip_value();
                        }
                    }
                    j.expect('}');
                    files.push_back(std::move(entry));
                }
                j.expect(']');
            } else {
                j.skip_value();
            }
        }
        j.expect('}');
        if (format != detail::FAILED_RUN_FORMAT) {
            throw std::runtime_error("is not a failed-run manifest (format \"" + format + "\")");
        }
        if (version != detail::FAILED_RUN_VERSION) {
            throw std::runtime_error("is version " + std::to_string(version) +
                                     "; this build reads version " +
                                     std::to_string(detail::FAILED_RUN_VERSION));
        }
    });

    // Every listed file is checked before any is read: present, the size the
    // manifest says, and the same checksum.
    for (const Listed& entry : files) {
        if (!safe_name(entry.name)) {
            detail::raise<InvalidArgument>(LOAD, "manifest.json lists \"" + entry.name +
                                                     "\", which is not a file in the folder");
        }
        const fs::path path = folder / fs::path(entry.name);
        std::error_code ec;
        if (!fs::is_regular_file(path, ec)) {
            detail::raise<InvalidArgument>(LOAD, entry.name + " is missing from " + folder.string());
        }
        std::pair<std::uint32_t, std::uint64_t> seen{};
        try {
            seen = file_checksum(path);
        } catch (const std::exception& e) {
            detail::raise<InvalidArgument>(LOAD, entry.name + " " + e.what());
        }
        if (seen.second != entry.bytes) {
            detail::raise<InvalidArgument>(LOAD, entry.name + " holds " + std::to_string(seen.second) +
                                                     " bytes; the manifest recorded " +
                                                     std::to_string(entry.bytes));
        }
        if (seen.first != entry.crc) {
            detail::raise<InvalidArgument>(LOAD, entry.name + " fails its checksum: CRC-32C " +
                                                     std::to_string(seen.first) +
                                                     ", the manifest recorded " +
                                                     std::to_string(entry.crc));
        }
    }

    const auto listed = [&](const std::string& name) {
        return std::any_of(files.begin(), files.end(),
                           [&](const Listed& entry) { return entry.name == name; });
    };

    if (listed("counts.json")) {
        const std::string text = read_text(folder / "counts.json");
        parse_file("counts.json", [&] {
            detail::JsonReader j{text};
            j.expect('{');
            while (j.peek() != '}') {
                if (j.peek() == ',') j.next();
                const std::string key = j.read_string();
                j.expect(':');
                if (key == "counts") {
                    j.expect('{');
                    while (j.peek() != '}') {
                        if (j.peek() == ',') j.next();
                        std::string bits = j.read_string();
                        j.expect(':');
                        r.counts[std::move(bits)] = j.read_int();
                    }
                    j.expect('}');
                } else {
                    j.skip_value();
                }
            }
            j.expect('}');
        });
    }

    if (listed("circuit.json")) {
        const std::string text = read_text(folder / "circuit.json");
        parse_file("circuit.json", [&] {
            r.circuit = QuantumCircuit::from_json(text);
            if (r.circuit->n_qubits != r.n_qubits) {
                throw std::runtime_error("holds a circuit of " +
                                         std::to_string(r.circuit->n_qubits) +
                                         " qubits; the manifest records " +
                                         std::to_string(r.n_qubits));
            }
        });
    }

    if (listed("noise_model.json")) {
        const std::string text = read_text(folder / "noise_model.json");
        parse_file("noise_model.json", [&] { r.noise_model = NoiseModel::from_json(text); });
    }

    if (listed("observations.json")) {
        const std::string text = read_text(folder / "observations.json");
        parse_file("observations.json", [&] {
            detail::JsonReader j{text};
            j.expect('{');
            while (j.peek() != '}') {
                if (j.peek() == ',') j.next();
                const std::string key = j.read_string();
                j.expect(':');
                if (key != "entries") {
                    j.skip_value();
                    continue;
                }
                j.expect('[');
                while (j.peek() != ']') {
                    if (j.peek() == ',') j.next();
                    std::string label, kind, file;
                    std::optional<ObservationBundle::Payload> payload;
                    j.expect('{');
                    while (j.peek() != '}') {
                        if (j.peek() == ',') j.next();
                        const std::string ekey = j.read_string();
                        j.expect(':');
                        if (ekey == "label") {
                            label = j.read_string();
                        } else if (ekey == "kind") {
                            kind = j.read_string();
                        } else if (ekey == "file") {
                            file = j.read_string();
                        } else if (ekey == "value") {
                            if (kind == "number") {
                                payload = j.read_double();
                            } else if (kind == "reals") {
                                payload = detail::read_real_array(j);
                            } else if (kind == "amplitudes") {
                                std::vector<Complex128> amps;
                                j.expect('[');
                                while (j.peek() != ']') {
                                    if (j.peek() == ',') j.next();
                                    j.expect('[');
                                    const double re = j.read_double();
                                    j.expect(',');
                                    const double im = j.read_double();
                                    j.expect(']');
                                    amps.emplace_back(re, im);
                                }
                                j.expect(']');
                                payload = std::move(amps);
                            } else if (kind == "integers") {
                                payload = detail::read_int_array(j);
                            } else if (kind == "text") {
                                payload = j.read_string();
                            } else {
                                throw std::runtime_error("holds an entry of unknown kind \"" +
                                                         kind + "\"");
                            }
                        } else {
                            j.skip_value();
                        }
                    }
                    j.expect('}');
                    if (kind == "state") {
                        if (!listed(file)) {
                            throw std::runtime_error("names " + file +
                                                     ", which the manifest does not list");
                        }
                        try {
                            payload = detail::state_payload(detail::read_state(folder / fs::path(file)));
                        } catch (const std::exception& e) {
                            throw std::runtime_error("names " + file + ", which " + e.what());
                        }
                    }
                    if (!payload) throw std::runtime_error("holds an entry with no value");
                    r.observations.put(std::move(label), std::move(*payload));
                }
                j.expect(']');
            }
            j.expect('}');
        });
    }

    if (listed("state.bin")) {
        // The state file, the manifest and the backend that ran must agree on
        // what the state is, so an edited manifest cannot hand back a state of
        // another width or form under this run's name.
        parse_file("state.bin", [&] {
            r.state = detail::read_state(folder / "state.bin");
            const std::string form = detail::form_word(detail::form_code(r.state));
            if (form != state_form || form != form_of_backend(r.backend)) {
                throw std::runtime_error("holds a " + form + " state; the manifest records a " +
                                         state_form + " state from the " + r.backend +
                                         " backend");
            }
            if (state_qubits(r.state) != r.n_qubits) {
                throw std::runtime_error("holds a state of " +
                                         std::to_string(state_qubits(r.state)) +
                                         " qubits; the manifest records " +
                                         std::to_string(r.n_qubits));
            }
        });
    }

    r.saved_to = folder;
    return r;
}

}  // namespace lindblad
