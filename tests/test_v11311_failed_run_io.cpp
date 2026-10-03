// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.31.1 - a saved run's folder, under every kind of trouble.
//
// A failed run's folder outlives the process that wrote it, so the loader is
// handed whatever the folder holds by then: a file edited, truncated, deleted
// or crafted. Every file carries its size and CRC-32C in the manifest, the
// manifest carries its own in manifest.crc32c, and every length read from a
// file is checked against what the file holds before anything is sized from
// it. Each refusal is pinned here: exactly InvalidArgument from
// load_failed_run, naming the file, and nothing allocated for a crafted size.
//
// The edits a person makes with a text editor are made the same way here, and
// re-signed where the test is aimed past the checksums at the check behind
// them (v11311_helpers.hpp: edit_manifest, relist).
//
// A save that cannot write everything is driven with RLIMIT_FSIZE, which fails
// a write part way exactly as a disk that fills does, with no seam in the
// library. Then the threads, and the batch primitive that moves a record from
// a worker thread to the caller's.
//
// Five defects these tests found shipped red in 1.1.31.1 and were fixed in
// 1.1.31.2: a state observation whose label is longer than a file name may be
// lost every observation on disk; the manifest's seed was read with a parser
// that wraps "-1" and ignores trailing text; the message after a partial save
// agreed its verb and pronoun with the number of parts, so a single plural
// part read "The observations was not saved; take it"; the note for a part
// that could not be written named it without its folder; and a batch whose
// reported failure left no record discarded the record the caller already
// held. The V11312 tests are 1.1.31.2's own, beside the pins they extend.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/detail/failure_collector.hpp"
#include "lindblad/errors.hpp"
#include "lindblad/failed_run.hpp"
#include "lindblad/observation.hpp"
#include "lindblad/observers.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/primitives.hpp"
#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "v11311_helpers.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <latch>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace lindblad;
namespace fs = std::filesystem;

namespace {

// A normless state met at the MEASURE of shot 0: a mid-run RuntimeFailure on
// a run walked shot by shot, which the run saves.
QuantumCircuit normless_trajectory(int n) {
    QuantumCircuit qc(n, n);
    qc.unitary(std::vector<Complex128>(4, Complex128(0.0, 0.0)), {0}, "zero", {Validation::Ignore});
    qc.measure(1, 1).x(1);
    return qc;
}

// Fails a statevector run of `qc` into `dir` and returns the record, with the
// exception it threw.
std::pair<FailedRun, std::optional<RuntimeFailure>> fail_into(const fs::path& dir, const QuantumCircuit& qc,
                                                              const RunPlan& base = {}) {
    RunPlan plan = base;
    plan.options.failed_run_dir = dir;
    (void)take_failed_run();
    auto e = v11311::thrown<RuntimeFailure>([&] { (void)StatevectorSimulator().run(qc, 3, 1, plan); });
    auto record = take_failed_run();
    EXPECT_TRUE(record.has_value());
    return {record ? std::move(*record) : FailedRun{}, std::move(e)};
}

// A saved statevector run's folder, and a saved one from each other backend
// for the state-file checks of their own forms.
fs::path saved_folder(const fs::path& dir) {
    auto [record, e] = fail_into(dir, normless_trajectory(2));
    EXPECT_TRUE(record.saved_to.has_value()) << record.save_note;
    return record.saved_to.value_or(fs::path{});
}

// load_failed_run refuses `folder` with exactly InvalidArgument whose message
// is "load_failed_run: " + `what`.
void expect_refused(const fs::path& folder, const std::string& what) {
    const auto e = v11311::thrown<InvalidArgument>([&] { (void)load_failed_run(folder); });
    ASSERT_TRUE(e.has_value());
    EXPECT_EQ(e->entry_point(), "load_failed_run");
    EXPECT_EQ(std::string(e->what()), "load_failed_run: " + what);
}

// A state file as the writer lays it out: the header, then `payload`.
std::string state_file(std::uint32_t version, std::uint32_t form, std::uint32_t n_qubits,
                       const std::string& payload) {
    std::string out = "LBSTATE1";
    for (const std::uint32_t word : {version, form, n_qubits, std::uint32_t{0}}) {
        char bytes[4];
        std::memcpy(bytes, &word, 4);  // the suite runs on little-endian hosts
        out.append(bytes, 4);
    }
    return out + payload;
}

std::string u64(std::uint64_t v) {
    char bytes[8];
    std::memcpy(bytes, &v, 8);
    return std::string(bytes, 8);
}

// Replaces state.bin and records its new size and checksum, so only its
// contents are left to be judged.
void craft_state(const fs::path& folder, const std::string& bytes) {
    v11311::write_file(folder / "state.bin", bytes);
    v11311::relist(folder, "state.bin");
}

}  // namespace

// =============================================================================
// Edits the checksums catch
// =============================================================================

TEST(V11311FailedRunIo, AByteFlippedInTheStateIsRefusedNamingTheFile) {
    const v11311::TempDir dir("flip");
    const fs::path folder = saved_folder(dir.path());
    std::string bytes = v11311::read_file(folder / "state.bin");
    const std::uint32_t recorded = v11311::crc_of(bytes);
    bytes[bytes.size() / 2] ^= 0x01;
    v11311::write_file(folder / "state.bin", bytes);
    expect_refused(folder, "state.bin fails its checksum: CRC-32C " + std::to_string(v11311::crc_of(bytes)) +
                               ", the manifest recorded " + std::to_string(recorded));
}

TEST(V11311FailedRunIo, ATruncatedOrMissingFileIsRefusedNamingIt) {
    const v11311::TempDir dir("truncate");
    const fs::path folder = saved_folder(dir.path());
    const std::string bytes = v11311::read_file(folder / "counts.json");
    v11311::write_file(folder / "counts.json", bytes.substr(0, bytes.size() - 3));
    expect_refused(folder, "counts.json holds " + std::to_string(bytes.size() - 3) +
                               " bytes; the manifest recorded " + std::to_string(bytes.size()));
    fs::remove(folder / "counts.json");
    expect_refused(folder, "counts.json is missing from " + folder.string());
}

TEST(V11311FailedRunIo, AnEditedManifestIsRefusedByItsOwnChecksum) {
    const v11311::TempDir dir("manifest");
    const fs::path folder = saved_folder(dir.path());
    const std::string original = v11311::read_file(folder / "manifest.json");
    const std::uint32_t recorded = v11311::crc_of(original);

    // A qubit count changed in place: the same length, another checksum.
    std::string edited = original;
    v11311::replace_value(edited, "n_qubits", "3");
    v11311::write_file(folder / "manifest.json", edited);
    expect_refused(folder, "manifest.json fails its checksum: CRC-32C " +
                               std::to_string(v11311::crc_of(edited)) +
                               ", manifest.crc32c recorded " + std::to_string(recorded));

    // A file's entry removed: another length.
    edited = original;
    const std::size_t at = edited.find("{\"name\":\"counts.json\"");
    ASSERT_NE(at, std::string::npos);
    edited.erase(at, edited.find('}', at) + 2 - at);
    v11311::write_file(folder / "manifest.json", edited);
    expect_refused(folder, "manifest.json holds " + std::to_string(edited.size()) +
                               " bytes; manifest.crc32c recorded " + std::to_string(original.size()));
}

TEST(V11311FailedRunIo, AMissingManifestOrManifestChecksumIsRefused) {
    const v11311::TempDir dir("missing");
    const fs::path folder = saved_folder(dir.path());
    fs::remove(folder / "manifest.crc32c");
    expect_refused(folder, "manifest.crc32c in " + folder.string() + " cannot be opened");
    fs::remove(folder / "manifest.json");
    expect_refused(folder, "manifest.json in " + folder.string() + " cannot be opened");
    expect_refused(dir.path() / "no-such-folder",
                   "manifest.json in " + (dir.path() / "no-such-folder").string() + " cannot be opened");
}

// =============================================================================
// Edits re-signed: the checks behind the checksums
// =============================================================================

TEST(V11311FailedRunIo, TheManifestAndTheFilesMustAgreeOnWhatTheRunWas) {
    const v11311::TempDir dir("agree");
    {
        const fs::path folder = saved_folder(dir.path());
        v11311::edit_manifest(folder, [](std::string& m) { v11311::replace_value(m, "n_qubits", "3"); });
        expect_refused(folder, "circuit.json holds a circuit of 2 qubits; the manifest records 3");
    }
    {
        // With the circuit's entry gone, the state is what disagrees.
        const fs::path folder = saved_folder(dir.path());
        v11311::edit_manifest(folder, [](std::string& m) {
            v11311::replace_value(m, "n_qubits", "3");
            const std::size_t at = m.find("{\"name\":\"circuit.json\"");
            m.erase(at, m.find('}', at) + 2 - at);
        });
        expect_refused(folder, "state.bin holds a state of 2 qubits; the manifest records 3");
    }
    {
        const fs::path folder = saved_folder(dir.path());
        v11311::edit_manifest(folder, [](std::string& m) { v11311::replace_value(m, "backend", "\"density_matrix\""); });
        expect_refused(folder, "state.bin holds a statevector state; the manifest records a statevector "
                               "state from the density_matrix backend");
    }
    {
        const fs::path folder = saved_folder(dir.path());
        v11311::edit_manifest(folder, [](std::string& m) { v11311::replace_value(m, "state_form", "\"mps\""); });
        expect_refused(folder, "state.bin holds a statevector state; the manifest records a mps state "
                               "from the statevector backend");
    }
}

TEST(V11311FailedRunIo, AManifestOfAnotherFormatOrVersionIsRefused) {
    const v11311::TempDir dir("format");
    {
        const fs::path folder = saved_folder(dir.path());
        v11311::edit_manifest(folder, [](std::string& m) { v11311::replace_value(m, "format", "\"other\""); });
        expect_refused(folder, "manifest.json is not a failed-run manifest (format \"other\")");
    }
    {
        const fs::path folder = saved_folder(dir.path());
        v11311::edit_manifest(folder, [](std::string& m) { v11311::replace_value(m, "version", "2"); });
        expect_refused(folder, "manifest.json is version 2; this build reads version 1");
    }
}

TEST(V11311FailedRunIo, ANumberThatIsNotACountIsRefused) {
    // A byte count or a CRC must be a whole number in its type's range; an
    // int field must be whole. Converting anything else would be undefined.
    const v11311::TempDir dir("numbers");
    const auto with_counts_entry = [&](const std::string& key, const std::string& raw) {
        const fs::path folder = saved_folder(dir.path());
        v11311::edit_manifest(folder, [&](std::string& m) {
            const std::size_t at = m.find("{\"name\":\"counts.json\"");
            std::string entry = m.substr(at, m.find('}', at) + 1 - at);
            const std::string original = entry;
            v11311::replace_value(entry, key, raw);
            m.replace(at, original.size(), entry);
        });
        return folder;
    };
    expect_refused(with_counts_entry("bytes", "-1"),
                   "manifest.json records " + std::to_string(-1.0) + " as a byte count");
    expect_refused(with_counts_entry("bytes", "12.5"),
                   "manifest.json records " + std::to_string(12.5) + " as a byte count");
    expect_refused(with_counts_entry("bytes", "1e30"),
                   "manifest.json records " + std::to_string(1e30) + " as a byte count");
    expect_refused(with_counts_entry("crc32c", "4294967296"),
                   "manifest.json records " + std::to_string(4294967296.0) + " as a CRC-32C");

    const fs::path folder = saved_folder(dir.path());
    v11311::edit_manifest(folder, [](std::string& m) { v11311::replace_value(m, "shots_completed", "1.5"); });
    expect_refused(folder, "manifest.json JSON parse error: " + std::to_string(1.5) + " is not a whole number");
}

TEST(V11311FailedRunIo, AFileNameThatLeavesTheFolderIsRefused) {
    const v11311::TempDir dir("escape");
    for (const std::string name : {"../escape.json", "/etc/passwd", "sub/../../x", "bad name.json"}) {
        SCOPED_TRACE(name);
        const fs::path folder = saved_folder(dir.path());
        v11311::edit_manifest(folder, [&](std::string& m) {
            const std::string tag = "\"name\":\"counts.json\"";
            m.replace(m.find(tag), tag.size(), "\"name\":\"" + name + "\"");
        });
        expect_refused(folder, "manifest.json lists \"" + name + "\", which is not a file in the folder");
    }
}

TEST(V11311FailedRunIo, AnObservationNamingAnUnlistedFileIsRefused) {
    const v11311::TempDir dir("unlisted");
    RunPlan plan;
    plan.observations.observe(Anchor::after_instruction(0), std::make_shared<StateObserver>("s"));
    auto [record, e] = fail_into(dir.path(), normless_trajectory(2), plan);
    ASSERT_TRUE(record.saved_to.has_value()) << record.save_note;
    const fs::path folder = *record.saved_to;
    std::string observations = v11311::read_file(folder / "observations.json");
    const std::string listed = "observations/0-s.bin";
    ASSERT_NE(observations.find(listed), std::string::npos) << observations;
    observations.replace(observations.find(listed), listed.size(), "observations/9-x.bin");
    v11311::write_file(folder / "observations.json", observations);
    v11311::relist(folder, "observations.json");
    expect_refused(folder, "observations.json names observations/9-x.bin, which the manifest does not list");
}

// =============================================================================
// Crafted state files: refused before anything is sized from them
// =============================================================================

TEST(V11311FailedRunIo, AStateFileOfAnotherKindIsRefused) {
    const v11311::TempDir dir("kind");
    const auto refused = [&](const std::string& bytes, const std::string& what) {
        const fs::path folder = saved_folder(dir.path());
        craft_state(folder, bytes);
        expect_refused(folder, "state.bin " + what);
    };
    std::string wrong_magic = state_file(1, 0, 2, std::string(64, '\0'));
    wrong_magic[0] = 'X';
    refused(wrong_magic, "is not a Lindblad state file");
    refused(state_file(2, 0, 2, std::string(64, '\0')), "is state file version 2; this build reads version 1");
    refused(state_file(1, 7, 2, std::string(64, '\0')),
            "holds a state form 7, which is not one this build knows");
    refused("LBSTATE1", "ends before its contents do");
}

TEST(V11311FailedRunIo, AClaimedWidthTheFileCannotHoldIsRefusedWithoutAllocating) {
    // Each header claims a register far wider than its few bytes could hold.
    // Building any of these states would ask for gigabytes to exabytes; the
    // file's own size refuses them first, so this test finishes at once.
    const v11311::TempDir dir("crafted");
    const auto refused = [&](const std::string& bytes, const std::string& what) {
        const fs::path folder = saved_folder(dir.path());
        craft_state(folder, bytes);
        expect_refused(folder, "state.bin " + what);
    };
    const std::string small(256, '\0');
    // Statevector: 2^n doubles twice, which the payload must match exactly.
    refused(state_file(1, 0, 3, std::string(2 * 4 * 8, '\0')), "does not hold a statevector of 3 qubits");
    refused(state_file(1, 0, 59, small), "does not hold a statevector of 59 qubits");
    refused(state_file(1, 0, 60, small), "does not hold a statevector of 60 qubits");
    // Density matrix: the entry count must be 4^n and present.
    refused(state_file(1, 1, 40, u64(std::uint64_t{1} << 40) + small),
            "does not hold a density matrix of 40 qubits");
    refused(state_file(1, 1, 2, u64(16) + std::string(15 * 16, '\0')),
            "does not hold a density matrix of 2 qubits");
    // MPS: a chain of 2^31 - 1 sites in 256 bytes, and a count past int.
    refused(state_file(1, 2, (1u << 31) - 1, small), "is too short to hold a chain of 2147483647 sites");
    refused(state_file(1, 2, 1u << 31, small), "holds an impossible qubit count");
    // Stabilizer: past the bound, and a tableau the payload does not hold.
    refused(state_file(1, 3, (1u << 20) + 1, small), "holds an impossible qubit count");
    refused(state_file(1, 3, 1000, small), "is not the tableau of 1000 qubits");
}

// =============================================================================
// A save that cannot write everything
// =============================================================================

#if !defined(_WIN32)
// What is missing is listed after "Not saved:", with no verb or pronoun to
// agree with the number of parts.
TEST(V11311FailedRunIo, AStateTooLargeToWriteStaysInMemoryAndTheMessageSaysSo) {
    // 2^12 amplitudes make a 64 KiB state file; every other part is a few
    // KiB. A 16 KiB file limit fails the state's write part way, as a full
    // disk does: the part is removed, the rest is saved, the state stays in
    // the record, and the exception names what is missing (I2).
    const v11311::TempDir dir("partial");
    std::pair<FailedRun, std::optional<RuntimeFailure>> outcome;
    {
        v11311::ScopedFileSizeLimit limit(16 * 1024);
        ASSERT_TRUE(limit.ok());
        outcome = fail_into(dir.path(), normless_trajectory(12));
    }
    auto& [record, e] = outcome;
    ASSERT_TRUE(e.has_value());
    ASSERT_TRUE(record.saved_to.has_value()) << record.save_note;
    EXPECT_EQ(std::string(e->what()),
              "StatevectorSimulator::run: no norm to sample from; the state is zero or non-finite "
              "(instruction 1: measure on qubit 1) at shot 0. Partial results saved to " +
                  record.saved_to->string() +
                  ". Not saved: the state. Take from memory with lindblad::take_failed_run().");
    EXPECT_EQ(record.save_note,
              "The state was not saved: writing state.bin failed; it is in memory, take it with "
              "lindblad::take_failed_run().");
    ASSERT_EQ(record.state.index(), 1u) << "the state that could not be written was released";
    EXPECT_EQ(std::get<Statevector>(record.state).n_qubits, 12);
    EXPECT_FALSE(fs::exists(*record.saved_to / "state.bin")) << "a part written part way was left";

    // The folder holds the rest and loads without the state.
    const FailedRun loaded = load_failed_run(*record.saved_to);
    EXPECT_EQ(loaded.state.index(), 0u);
    EXPECT_EQ(loaded.save_note, record.save_note);
    EXPECT_TRUE(loaded.circuit.has_value());
}

// The note names the file by its path in the folder, as the manifest lists
// it, and the exception lists what is missing with no verb or pronoun to
// agree.
TEST(V11311FailedRunIo, ObservationsThatCannotAllBeWrittenAreSavedNoneAtAll) {
    // A density-matrix capture of six qubits is a 64 KiB file; the run's own
    // statevector is 1 KiB. Under a 16 KiB limit the capture's write fails,
    // and the observations are saved whole or not at all: no stray file, no
    // empty folder, nothing listed (M13).
    const v11311::TempDir dir("observations");
    RunPlan plan;
    plan.options.cost = Cost::Unlimited;
    plan.observations.observe(Anchor::after_instruction(0),
                              std::make_shared<StateObserver>(StateForm::DensityMatrix, "dm"));
    std::pair<FailedRun, std::optional<RuntimeFailure>> outcome;
    {
        v11311::WarningCapture quiet;
        v11311::ScopedFileSizeLimit limit(16 * 1024);
        ASSERT_TRUE(limit.ok());
        outcome = fail_into(dir.path(), normless_trajectory(6), plan);
    }
    auto& [record, e] = outcome;
    ASSERT_TRUE(e.has_value());
    ASSERT_TRUE(record.saved_to.has_value()) << record.save_note;
    EXPECT_EQ(record.save_note, "The observations were not saved: writing observations/0-dm.bin failed.");
    EXPECT_FALSE(fs::exists(*record.saved_to / "observations"));
    EXPECT_FALSE(fs::exists(*record.saved_to / "observations.json"));
    EXPECT_TRUE(fs::exists(*record.saved_to / "state.bin"));
    ASSERT_TRUE(record.observations.contains("dm")) << "the observations were released unsaved";
    EXPECT_EQ(record.observations.form("dm"), StateForm::DensityMatrix);
    const std::string what = e->what();
    const std::string tail =
        ". Not saved: the observations. Take from memory with lindblad::take_failed_run().";
    ASSERT_GE(what.size(), tail.size());
    EXPECT_EQ(what.substr(what.size() - tail.size()), tail) << what;
    EXPECT_EQ(load_failed_run(*record.saved_to).observations.size(), 0u);
}

// Two parts missing are listed in the order the record keeps them, separated
// by a comma.
TEST(V11311FailedRunIo, TwoPartsThatCannotBeWrittenAreBothListed) {
    const v11311::TempDir dir("two-parts");
    RunPlan plan;
    plan.observations.observe(Anchor::after_instruction(0), std::make_shared<StateObserver>("s"));
    std::pair<FailedRun, std::optional<RuntimeFailure>> outcome;
    {
        v11311::ScopedFileSizeLimit limit(16 * 1024);
        ASSERT_TRUE(limit.ok());
        outcome = fail_into(dir.path(), normless_trajectory(12), plan);
    }
    auto& [record, e] = outcome;
    ASSERT_TRUE(e.has_value());
    ASSERT_TRUE(record.saved_to.has_value()) << record.save_note;
    EXPECT_NE(record.state.index(), 0u);
    EXPECT_TRUE(record.observations.contains("s"));
    const std::string what = e->what();
    const std::string tail =
        ". Not saved: the state, the observations. Take from memory with lindblad::take_failed_run().";
    ASSERT_GE(what.size(), tail.size());
    EXPECT_EQ(what.substr(what.size() - tail.size()), tail) << what;
}
#endif

// =============================================================================
// A long label, and the seed
// =============================================================================

// A state observation is saved as observations/<index>-<label>.bin with at
// most 64 sanitised characters of the label, so a label longer than a file
// name may be (255 bytes on most filesystems) still names a writable file.
// Observations are saved whole or not at all, so one unwritable file would
// lose every observation of the run from the folder.
TEST(V11311FailedRunIo, AStateObservationWithALongLabelIsSavedAndLoadsBack) {
    const v11311::TempDir dir("long-label");
    const std::string label(300, 'a');
    RunPlan plan;
    plan.observations.observe(Anchor::after_instruction(0), std::make_shared<StateObserver>(label));
    auto [record, e] = fail_into(dir.path(), normless_trajectory(2), plan);
    ASSERT_TRUE(record.saved_to.has_value()) << record.save_note;
    EXPECT_EQ(record.save_note, "");
    EXPECT_EQ(record.observations.size(), 0u) << "the observations were not saved";
    const FailedRun loaded = load_failed_run(*record.saved_to);
    ASSERT_TRUE(loaded.observations.contains(label));
    EXPECT_EQ(loaded.observations.form(label), StateForm::Statevector);
}

// Two labels that agree in their first 64 characters shorten to the same text,
// and the index in front keeps their files apart.
TEST(V11312FailedRunIo, TwoLongLabelsAlikeInTheirFirst64CharactersBothLoadBack) {
    const v11311::TempDir dir("long-labels");
    const std::string first = std::string(80, 'b') + "-one";
    const std::string second = std::string(80, 'b') + "-two";
    RunPlan plan;
    plan.observations.observe(Anchor::after_instruction(0), std::make_shared<StateObserver>(first));
    plan.observations.observe(Anchor::after_instruction(0), std::make_shared<StateObserver>(second));
    auto [record, e] = fail_into(dir.path(), normless_trajectory(2), plan);
    ASSERT_TRUE(record.saved_to.has_value()) << record.save_note;
    EXPECT_EQ(record.save_note, "");
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(*record.saved_to / "observations")) {
        files.push_back(entry.path().filename().string());
    }
    std::sort(files.begin(), files.end());
    EXPECT_EQ(files, (std::vector<std::string>{"0-" + std::string(64, 'b') + ".bin",
                                               "1-" + std::string(64, 'b') + ".bin"}));
    const FailedRun loaded = load_failed_run(*record.saved_to);
    EXPECT_TRUE(loaded.observations.contains(first));
    EXPECT_TRUE(loaded.observations.contains(second));
}

// The manifest's seed is written as a string of digits, since a JSON number
// holds only 53 bits, and read back as exactly that. A sign ("-1" would wrap to
// 2^64 - 1 under a lenient reader), white space, a prefix or trailing text is
// refused like every other malformed number in the manifest.
TEST(V11311FailedRunIo, ASeedThatIsNotAStringOfDigitsIsRefused) {
    const v11311::TempDir dir("seed");
    for (const std::string seed : {"-1", " 12", "12abc", "0x10", "+5"}) {
        SCOPED_TRACE(seed);
        const fs::path folder = saved_folder(dir.path());
        v11311::edit_manifest(folder, [&](std::string& m) { v11311::replace_value(m, "seed", "\"" + seed + "\""); });
        const auto e = v11311::thrown<InvalidArgument>([&] { (void)load_failed_run(folder); });
        ASSERT_TRUE(e.has_value());
        EXPECT_EQ(std::string(e->what()), "load_failed_run: manifest.json records the seed \"" + seed +
                                              "\", which is not a string of digits");
    }
}

TEST(V11311FailedRunIo, ASeedOutOfRangeIsRefusedNamingTheManifest) {
    const v11311::TempDir dir("seed-range");
    const fs::path folder = saved_folder(dir.path());
    v11311::edit_manifest(folder, [](std::string& m) {
        v11311::replace_value(m, "seed", "\"18446744073709551616\"");  // 2^64
    });
    const auto e = v11311::thrown<InvalidArgument>([&] { (void)load_failed_run(folder); });
    ASSERT_TRUE(e.has_value());
    EXPECT_EQ(std::string(e->what()),
              "load_failed_run: manifest.json records the seed \"18446744073709551616\", which is "
              "more than a 64-bit seed holds");
}

TEST(V11312FailedRunIo, TheLargestSeedLoadsBack) {
    const v11311::TempDir dir("seed-max");
    const fs::path folder = saved_folder(dir.path());
    const std::uint64_t largest = std::numeric_limits<std::uint64_t>::max();
    v11311::edit_manifest(folder, [&](std::string& m) {
        v11311::replace_value(m, "seed", "\"" + std::to_string(largest) + "\"");
    });
    EXPECT_EQ(load_failed_run(folder).seed, largest);
}

// =============================================================================
// Threads
// =============================================================================

TEST(V11311FailedRunIo, ThreadsFailingAtOnceEachSaveOneCompleteFolder) {
    const v11311::TempDir dir("threads");
    constexpr int kThreads = 4;
    std::latch start(kThreads);
    std::vector<std::optional<FailedRun>> records(kThreads);
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            RunPlan plan;
            plan.options.failed_run_dir = dir.path();
            start.arrive_and_wait();
            try {
                (void)StatevectorSimulator().run(normless_trajectory(2), 3, 100 + t, plan);
            } catch (const RuntimeFailure&) {
            }
            records[static_cast<std::size_t>(t)] = take_failed_run();
        });
    }
    for (auto& th : threads) th.join();

    std::vector<fs::path> folders;
    for (int t = 0; t < kThreads; ++t) {
        SCOPED_TRACE(t);
        ASSERT_TRUE(records[static_cast<std::size_t>(t)].has_value()) << "a thread's record is in its own slot";
        const FailedRun& r = *records[static_cast<std::size_t>(t)];
        ASSERT_TRUE(r.saved_to.has_value()) << r.save_note;
        EXPECT_EQ(r.seed, static_cast<std::uint64_t>(100 + t));
        folders.push_back(*r.saved_to);
        const FailedRun loaded = load_failed_run(*r.saved_to);
        EXPECT_EQ(loaded.seed, r.seed) << "a folder holds another thread's run";
    }
    std::sort(folders.begin(), folders.end());
    EXPECT_EQ(std::adjacent_find(folders.begin(), folders.end()), folders.end()) << "two runs shared a folder";
    std::size_t entries = 0;
    for (const auto& entry : fs::directory_iterator(dir.path())) {
        ++entries;
        EXPECT_EQ(entry.path().extension(), "") << "an incomplete folder was left: " << entry.path();
    }
    EXPECT_EQ(entries, static_cast<std::size_t>(kThreads));
}

// =============================================================================
// Estimator::run_batch carries the record to the caller's thread
// =============================================================================

namespace {

// Every index of a batch fails mid-run: a sampled estimate of a state with no
// norm. Each index's circuit carries its own angle, which is how a record is
// traced to the index it came from.
QuantumCircuit normless_ansatz() {
    QuantumCircuit qc(1);
    qc.unitary(std::vector<Complex128>(4, Complex128(0.0, 0.0)), {0}, "zero", {Validation::Ignore});
    qc.rx("theta", 0);
    return qc;
}

// Sampled estimates (shots > 0) go through a statevector run, which is what
// keeps a record; no transpilation, so the circuit is run as written.
void make_sampling(Estimator& est) {
    est.options.shots = 16;
    est.options.seed = 3;
    est.options.optimization_level = 0;
}

double angle_of(const FailedRun& r) {
    for (const Instruction& inst : r.circuit->instructions)
        if (inst.type == Instruction::GateType::RX) return inst.params[0];
    return std::numeric_limits<double>::quiet_NaN();
}

}  // namespace

TEST(V11311FailedRunIo, ABatchHandsTheLowestFailingIndexsRecordToTheCaller) {
#ifdef _OPENMP
    const int previous = omp_get_max_threads();
    omp_set_num_threads(4);
#endif
    std::vector<std::vector<double>> params;
    for (int i = 0; i < 8; ++i) params.push_back({0.1 * (i + 1)});
    Estimator est;
    make_sampling(est);
    const SparsePauliOp z(std::vector<PauliString>{PauliString("Z")});
    (void)take_failed_run();
    const auto e = v11311::thrown<RuntimeFailure>([&] { (void)est.run_batch(normless_ansatz(), z, params); });
#ifdef _OPENMP
    omp_set_num_threads(previous);
#endif
    ASSERT_TRUE(e.has_value());
    auto record = take_failed_run();
    ASSERT_TRUE(record.has_value()) << "the reported failure's record stayed on its worker";
    ASSERT_TRUE(record->saved_to.has_value()) << record->save_note;
    EXPECT_EQ(record->saved_to, e->saved_to()) << "the record is not the reported failure's";
    const FailedRun loaded = load_failed_run(*record->saved_to);
    ASSERT_TRUE(loaded.circuit.has_value());
    EXPECT_EQ(angle_of(loaded), params[0][0]) << "the record came from another index";
}

TEST(V11311FailedRunIo, ABatchWhoseFailureLeftNoRecordAttachesNone) {
    // The caller's slot holds an older record. Every index of the batch is
    // refused before its first gate, so none leaves a record, and the caller
    // finds the older one still there rather than a stale one from a worker
    // (I4).
    const v11311::TempDir dir("batch-old");
    {
        RunPlan plan;
        plan.options.failed_run_dir = dir.path();
        plan.options.save_failed_runs = RunPlan::Options::SaveFailedRuns::DoNotSave;
        (void)take_failed_run();
        EXPECT_THROW((void)StatevectorSimulator().run(normless_trajectory(2), 3, 4242, plan), RuntimeFailure);
    }
    std::vector<std::vector<double>> params(8, {std::numeric_limits<double>::quiet_NaN()});
    Estimator est;
    make_sampling(est);
    const SparsePauliOp z(std::vector<PauliString>{PauliString("Z")});
    QuantumCircuit ansatz(1);
    ansatz.rx("theta", 0);
    const auto e = v11311::thrown<InvalidArgument>([&] { (void)est.run_batch(ansatz, z, params); });
    ASSERT_TRUE(e.has_value());
    v11311::expect_message(*e, {"parameter 0 is " + std::to_string(params[0][0])});
    const auto record = take_failed_run();
    ASSERT_TRUE(record.has_value()) << "the caller's own record was taken";
    EXPECT_EQ(record->seed, 4242u) << "a record from the batch replaced the caller's";
}

// As above, but only index 0 is refused before its first gate; the later
// indices fail mid-run and each leaves a record. On the caller's own thread
// such a record would replace the caller's (the newest wins) and then be
// discarded, since only index 0 is reported. The batch puts only the reported
// failure's record into the caller's slot, and leaves the slot as it was
// otherwise, on one thread or several.
TEST(V11311FailedRunIo, ABatchLeavesTheCallersRecordWhenTheReportedFailureLeftNone) {
#ifdef _OPENMP
    const int previous = omp_get_max_threads();
#endif
    for (const int threads : {1, 4}) {
        SCOPED_TRACE(std::to_string(threads) + " threads");
#ifdef _OPENMP
        omp_set_num_threads(threads);
#endif
        const v11311::TempDir dir("batch-mixed");
        {
            RunPlan plan;
            plan.options.failed_run_dir = dir.path();
            plan.options.save_failed_runs = RunPlan::Options::SaveFailedRuns::DoNotSave;
            (void)take_failed_run();
            EXPECT_THROW((void)StatevectorSimulator().run(normless_trajectory(2), 3, 4242, plan),
                         RuntimeFailure);
        }
        std::vector<std::vector<double>> params{{std::numeric_limits<double>::quiet_NaN()}};
        for (int i = 1; i < 8; ++i) params.push_back({0.1 * i});
        Estimator est;
        make_sampling(est);
        const SparsePauliOp z(std::vector<PauliString>{PauliString("Z")});
        const auto e = v11311::thrown<InvalidArgument>([&] { (void)est.run_batch(normless_ansatz(), z, params); });
        ASSERT_TRUE(e.has_value());
        const auto record = take_failed_run();
        if (!record) {
            ADD_FAILURE() << "the caller's own record was discarded";
            continue;
        }
        EXPECT_EQ(record->seed, 4242u) << "a record from the batch replaced the caller's";
    }
#ifdef _OPENMP
    omp_set_num_threads(previous);
#endif
}

// A batch that does not fail leaves the caller's record where it was, and
// putting it back is not a new failure: the count of records stored on the
// caller's thread is unchanged.
TEST(V11312FailedRunIo, ABatchThatSucceedsLeavesTheCallersRecordAndTheStoreCount) {
    const v11311::TempDir dir("batch-ok");
    {
        RunPlan plan;
        plan.options.failed_run_dir = dir.path();
        plan.options.save_failed_runs = RunPlan::Options::SaveFailedRuns::DoNotSave;
        (void)take_failed_run();
        EXPECT_THROW((void)StatevectorSimulator().run(normless_trajectory(2), 3, 4242, plan), RuntimeFailure);
    }
    const std::uint64_t stores = detail::failed_run_stores();
    QuantumCircuit ansatz(1);
    ansatz.rx("theta", 0);
    Estimator est;
    make_sampling(est);
    const SparsePauliOp z(std::vector<PauliString>{PauliString("Z")});
    const std::vector<std::vector<double>> params{{0.1}, {0.2}, {0.3}, {0.4}};
    EXPECT_EQ(est.run_batch(ansatz, z, params).size(), params.size());
    EXPECT_EQ(detail::failed_run_stores(), stores);
    const auto record = take_failed_run();
    ASSERT_TRUE(record.has_value()) << "the caller's own record was taken";
    EXPECT_EQ(record->seed, 4242u);
}
