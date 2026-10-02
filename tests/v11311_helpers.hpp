// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

#pragma once

// Helpers shared by the 1.1.31.1 suites, which pin how a run fails: the typed
// errors, the failed-run record, and the folder a failure saves.
//
// thrown<E> is how every one of them asserts a failure. It accepts exactly E,
// compared by typeid, and fails the test for anything else: a derived type, a
// base type, a foreign exception or no exception at all. An EXPECT_THROW on a
// std base would accept every Lindblad type deriving from it, which is the
// looseness these suites exist to rule out, so none of them uses one for a
// Lindblad failure.
//
// The folder helpers rewrite a saved run the way a person editing it by hand
// would, and then re-sign what they changed, so a test can walk the loader
// past its checksums to the check it is aiming at.

#include <gtest/gtest.h>

#include "lindblad/detail/failure_collector.hpp"
#include "lindblad/errors.hpp"
#include "lindblad/validation.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <typeinfo>
#include <utility>
#include <vector>

#if defined(__GNUG__)
#include <cxxabi.h>
#endif

#if !defined(_WIN32)
#include <csignal>
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace v11311 {

namespace fs = std::filesystem;

// =============================================================================
// The suite's state folder
// =============================================================================
// Defined in v11311_failed_run_env.cpp. Every run the suite saves without
// naming a folder lands under it, and it is removed when the suite ends.
fs::path suite_state_root();

// Where a failed run with no failed_run_dir is saved while the suite runs.
inline fs::path default_failed_run_folder() {
    return suite_state_root() / "state" / "lindblad" / "failed-runs";
}

// =============================================================================
// Exceptions
// =============================================================================

inline std::string type_name(const std::type_info& type) {
#if defined(__GNUG__)
    int status = 0;
    char* name = abi::__cxa_demangle(type.name(), nullptr, nullptr, &status);
    if (status == 0 && name != nullptr) {
        std::string out(name);
        std::free(name);
        return out;
    }
    std::free(name);
#endif
    return type.name();
}

inline bool contains(std::string_view haystack, std::string_view needle) {
    return haystack.find(needle) != std::string_view::npos;
}

// Runs `fn`, which must throw exactly E, and returns what it threw. Anything
// else fails the calling test and returns nothing.
template <class E, class Fn>
std::optional<E> thrown(Fn&& fn) {
    try {
        std::forward<Fn>(fn)();
    } catch (const E& e) {
        if (typeid(e) != typeid(E)) {
            ADD_FAILURE() << "expected exactly " << type_name(typeid(E)) << ", got "
                          << type_name(typeid(e)) << ": " << e.what();
            return std::nullopt;
        }
        return e;
    } catch (const std::exception& e) {
        ADD_FAILURE() << "expected " << type_name(typeid(E)) << ", got "
                      << type_name(typeid(e)) << ": " << e.what();
        return std::nullopt;
    } catch (...) {
        ADD_FAILURE() << "expected " << type_name(typeid(E))
                      << ", got an exception not derived from std::exception";
        return std::nullopt;
    }
    ADD_FAILURE() << "expected " << type_name(typeid(E)) << ", but nothing was thrown";
    return std::nullopt;
}

// A Lindblad error's message must contain every one of `needles`.
inline void expect_message(const std::exception& e,
                           std::initializer_list<std::string_view> needles) {
    const std::string what = e.what();
    for (std::string_view needle : needles) {
        EXPECT_TRUE(contains(what, needle))
            << "expected [" << needle << "] in the message [" << what << "]";
    }
}

// A FailurePoint with exactly these coordinates.
inline void expect_point(const std::optional<lindblad::FailurePoint>& where, int shot,
                         int instruction, const std::string& gate,
                         const std::vector<int>& qubits) {
    ASSERT_TRUE(where.has_value()) << "the error names no position";
    EXPECT_EQ(where->shot, shot);
    EXPECT_EQ(where->instruction, instruction);
    EXPECT_EQ(where->gate, gate);
    EXPECT_EQ(where->qubits, qubits);
}

// =============================================================================
// The process environment
// =============================================================================

// Sets (or, with nullopt, removes) one environment variable for a scope and
// puts the previous value back afterwards.
class ScopedEnv {
public:
    ScopedEnv(std::string name, const std::optional<std::string>& value) : name_(std::move(name)) {
        if (const char* old = std::getenv(name_.c_str()); old != nullptr) old_ = std::string(old);
        apply(value);
    }
    ~ScopedEnv() { apply(old_); }
    ScopedEnv(const ScopedEnv&) = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;

private:
    void apply(const std::optional<std::string>& value) {
#if defined(_WIN32)
        _putenv_s(name_.c_str(), value ? value->c_str() : "");
#else
        if (value) {
            ::setenv(name_.c_str(), value->c_str(), 1);
        } else {
            ::unsetenv(name_.c_str());
        }
#endif
    }

    std::string name_;
    std::optional<std::string> old_;
};

// A fresh, empty folder under the suite's state root, removed with the scope.
class TempDir {
public:
    explicit TempDir(const std::string& tag) {
        static int counter = 0;
        path_ = suite_state_root() / "tmp" / (tag + "-" + std::to_string(counter++));
        fs::remove_all(path_);
        fs::create_directories(path_);
    }
    ~TempDir() {
        std::error_code ec;
        fs::permissions(path_, fs::perms::owner_all, fs::perm_options::add, ec);
        fs::remove_all(path_, ec);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    const fs::path& path() const { return path_; }

private:
    fs::path path_;
};

// =============================================================================
// The warning channel
// =============================================================================

// Everything the warning channel delivers while it lives. The handler is put
// back on every exit, an exception included, so a throwing run inside the
// scope cannot leave the channel pointing at a destroyed capture.
class WarningCapture {
public:
    WarningCapture() {
        lindblad::flush_warnings();
        lindblad::set_warning_handler([this](const std::string& m) { messages_.push_back(m); });
    }
    ~WarningCapture() {
        lindblad::flush_warnings();
        lindblad::set_warning_handler(nullptr);
    }
    WarningCapture(const WarningCapture&) = delete;
    WarningCapture& operator=(const WarningCapture&) = delete;

    // Every delivery so far, repeat tallies included, after a flush.
    const std::vector<std::string>& messages() {
        lindblad::flush_warnings();
        return messages_;
    }

    // How many deliveries contain `needle`.
    std::size_t count(std::string_view needle) {
        std::size_t n = 0;
        for (const std::string& m : messages()) n += contains(m, needle) ? 1 : 0;
        return n;
    }

private:
    std::vector<std::string> messages_;
};

// =============================================================================
// Files
// =============================================================================

inline std::string read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

inline void write_file(const fs::path& path, const std::string& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

inline std::uint32_t crc_of(const std::string& bytes) {
    return lindblad::detail::crc32c(0, bytes.data(), bytes.size());
}

// Rewrites manifest.crc32c to match manifest.json as it now stands.
inline void resign_manifest(const fs::path& folder) {
    const std::string manifest = read_file(folder / "manifest.json");
    write_file(folder / "manifest.crc32c",
               "{\"bytes\":" + std::to_string(manifest.size()) +
                   ",\"crc32c\":" + std::to_string(crc_of(manifest)) + "}");
}

// Edits manifest.json through `edit` and re-signs it, so the loader believes
// the edit and goes on to whatever it changed.
inline void edit_manifest(const fs::path& folder, const std::function<void(std::string&)>& edit) {
    std::string manifest = read_file(folder / "manifest.json");
    edit(manifest);
    write_file(folder / "manifest.json", manifest);
    resign_manifest(folder);
}

// Replaces the first `"key":<value>` in `json`, the value running to the next
// ',' or '}' outside a string, with `raw`.
inline void replace_value(std::string& json, const std::string& key, const std::string& raw) {
    const std::string tag = "\"" + key + "\":";
    const std::size_t at = json.find(tag);
    ASSERT_NE(at, std::string::npos) << "no " << tag << " in " << json;
    std::size_t end = at + tag.size();
    bool in_string = false;
    for (; end < json.size(); ++end) {
        const char c = json[end];
        if (c == '"' && json[end - 1] != '\\') in_string = !in_string;
        if (!in_string && (c == ',' || c == '}')) break;
    }
    json.replace(at + tag.size(), end - (at + tag.size()), raw);
}

// After the bytes of a listed file have been changed, records its new size
// and CRC-32C in the manifest and re-signs the manifest, so only the file's
// own contents are left to be judged.
inline void relist(const fs::path& folder, const std::string& name) {
    const std::string bytes = read_file(folder / name);
    edit_manifest(folder, [&](std::string& m) {
        const std::string tag = "{\"name\":\"" + name + "\",";
        const std::size_t at = m.find(tag);
        ASSERT_NE(at, std::string::npos) << name << " is not listed in " << m;
        const std::size_t close = m.find('}', at);
        m.replace(at, close + 1 - at,
                  tag + "\"bytes\":" + std::to_string(bytes.size()) +
                      ",\"crc32c\":" + std::to_string(crc_of(bytes)) + "}");
    });
}

// Every path under `folder`, relative to it, sorted.
inline std::vector<std::string> listing(const fs::path& folder) {
    std::vector<std::string> out;
    for (const auto& entry : fs::recursive_directory_iterator(folder)) {
        out.push_back(fs::relative(entry.path(), folder).generic_string());
    }
    std::sort(out.begin(), out.end());
    return out;
}

// =============================================================================
// A file size limit, for a write that fails part way without any seam
// =============================================================================
// POSIX only. RLIMIT_FSIZE makes a write past `bytes` fail with EFBIG, as a
// disk that fills part way through a file does, and SIGXFSZ (which would end
// the process) is ignored for the scope. Both are process-wide, so nothing
// else in the test may write a large regular file while it lives; the test
// runner's own output goes to a terminal or a pipe and is not limited.
#if !defined(_WIN32)
class ScopedFileSizeLimit {
public:
    explicit ScopedFileSizeLimit(rlim_t bytes) {
        ok_ = ::getrlimit(RLIMIT_FSIZE, &old_) == 0;
        if (!ok_) return;
        old_handler_ = std::signal(SIGXFSZ, SIG_IGN);
        rlimit limited = old_;
        limited.rlim_cur = bytes;
        ok_ = ::setrlimit(RLIMIT_FSIZE, &limited) == 0;
    }
    ~ScopedFileSizeLimit() {
        ::setrlimit(RLIMIT_FSIZE, &old_);
        std::signal(SIGXFSZ, old_handler_);
    }
    ScopedFileSizeLimit(const ScopedFileSizeLimit&) = delete;
    ScopedFileSizeLimit& operator=(const ScopedFileSizeLimit&) = delete;

    bool ok() const { return ok_; }

private:
    rlimit old_{};
    void (*old_handler_)(int) = SIG_DFL;
    bool ok_ = false;
};
#endif

}  // namespace v11311
