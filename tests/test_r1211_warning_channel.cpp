// R.1.21.1 test wave - the library's warning channel.
//
// Warn is one of the four Class C policies, and it is the only one whose
// observable effect is a side channel rather than a return value or an
// exception. Everything about that channel is global mutable state (one
// handler, one count table), so every test here runs under a fixture that
// resets both, and the tests that exercise the default sink manage it
// explicitly rather than inheriting an installed handler.
//
// Two behaviours are pinned here that have no other witness anywhere in the
// suite: the default sink writes through std::cerr rather than C stderr, and
// the handler is invoked under the channel lock so an OpenMP-parallel emit
// cannot enter a user sink from two threads at once.

#include <gtest/gtest.h>

#include "lindblad/validation.hpp"

#include <atomic>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace lindblad;

namespace {

// The cap on distinct tracked messages, from src/validation.cpp. Beyond it
// deduplication stops rather than letting the count table grow without bound.
constexpr std::size_t MAX_TRACKED_MESSAGES = 64;

// Redirects std::cerr for the lifetime of the object. This is the mechanism a
// caller uses to capture library diagnostics, and the reason the default sink
// must not write through C stdio: the two share a file descriptor but not a
// buffer, so a printf would bypass this entirely.
class CerrCapture {
public:
    CerrCapture() : previous_(std::cerr.rdbuf(buffer_.rdbuf())) {}
    ~CerrCapture() { std::cerr.rdbuf(previous_); }
    CerrCapture(const CerrCapture&) = delete;
    CerrCapture& operator=(const CerrCapture&) = delete;

    std::string text() const { return buffer_.str(); }

private:
    std::ostringstream buffer_;
    std::streambuf* previous_;
};

// Returns the channel to a known state: no handler installed, no pending
// counts. Called on both sides of every test so neither the order tests run in
// nor a warning emitted by an unrelated suite can change an outcome here.
void reset_channel() {
    set_warning_handler(nullptr);
    flush_warnings();
}

std::size_t occurrences(const std::vector<std::string>& lines,
                        const std::string& needle) {
    std::size_t n = 0;
    for (const auto& line : lines)
        if (line.find(needle) != std::string::npos) ++n;
    return n;
}

} // namespace

// =============================================================================
// The default sink
// =============================================================================

TEST(R1211WarningSink, DefaultSinkWritesThroughStdCerr) {
    // REGRESSION PIN. A sink written with std::fprintf(stderr, ...) shares a
    // file descriptor with std::cerr but not a buffer, so a caller who swaps
    // std::cerr's streambuf to capture library output receives nothing.
    // Redirecting std::cerr is the idiomatic way to capture C++ diagnostics and
    // two suites in this tree already depend on it working.
    reset_channel();
    std::string captured;
    {
        CerrCapture capture;
        emit_warning("default sink probe");
        captured = capture.text();
    }
    reset_channel();

    EXPECT_NE(captured.find("default sink probe"), std::string::npos)
        << "the default sink did not reach a redirected std::cerr; got: ["
        << captured << "]";
}

TEST(R1211WarningSink, DefaultSinkPrefixesTheLibraryName) {
    reset_channel();
    std::string captured;
    {
        CerrCapture capture;
        emit_warning("prefixed probe");
        captured = capture.text();
    }
    reset_channel();

    EXPECT_NE(captured.find("lindblad: prefixed probe"), std::string::npos)
        << "an unattributed line in a user's stderr is not actionable; got: ["
        << captured << "]";
}

TEST(R1211WarningSink, DefaultSinkTerminatesEachMessageWithANewline) {
    reset_channel();
    std::string captured;
    {
        CerrCapture capture;
        emit_warning("first line");
        emit_warning("second line");
        captured = capture.text();
    }
    reset_channel();

    EXPECT_EQ(captured, "lindblad: first line\nlindblad: second line\n");
}

TEST(R1211WarningSink, InstallingAHandlerDivertsTheDefaultSink) {
    reset_channel();
    std::vector<std::string> lines;
    std::string captured;
    {
        CerrCapture capture;
        set_warning_handler([&lines](const std::string& m) { lines.push_back(m); });
        emit_warning("diverted");
        captured = capture.text();
    }
    reset_channel();

    EXPECT_EQ(captured, "")
        << "a handler is a redirection, not a duplication; got: [" << captured
        << "]";
    ASSERT_EQ(lines.size(), 1u);
    EXPECT_EQ(lines[0], "diverted")
        << "the handler receives the bare message; the 'lindblad: ' prefix "
           "belongs to the default sink's formatting, not to the message";
}

TEST(R1211WarningSink, EmptyHandlerRestoresTheDefaultSink) {
    reset_channel();
    std::vector<std::string> lines;
    set_warning_handler([&lines](const std::string& m) { lines.push_back(m); });
    set_warning_handler(nullptr);

    std::string captured;
    {
        CerrCapture capture;
        emit_warning("back to cerr");
        captured = capture.text();
    }
    reset_channel();

    EXPECT_TRUE(lines.empty())
        << "the removed handler must not still be receiving messages";
    EXPECT_NE(captured.find("back to cerr"), std::string::npos);
}

// =============================================================================
// Routing, deduplication and flushing
// =============================================================================

class R1211WarningChannel : public ::testing::Test {
protected:
    void SetUp() override {
        reset_channel();
        lines.clear();
        set_warning_handler([this](const std::string& m) { lines.push_back(m); });
    }

    void TearDown() override { reset_channel(); }

    std::vector<std::string> lines;
};

TEST_F(R1211WarningChannel, HandlerReceivesTheMessageVerbatim) {
    emit_warning("exact text, with punctuation: 1e-12 (atol)");
    ASSERT_EQ(lines.size(), 1u);
    EXPECT_EQ(lines[0], "exact text, with punctuation: 1e-12 (atol)");
}

TEST_F(R1211WarningChannel, DistinctMessagesAreEachDelivered) {
    emit_warning("alpha");
    emit_warning("beta");
    emit_warning("gamma");
    ASSERT_EQ(lines.size(), 3u);
    EXPECT_EQ(lines[0], "alpha");
    EXPECT_EQ(lines[1], "beta");
    EXPECT_EQ(lines[2], "gamma");
}

TEST_F(R1211WarningChannel, IdenticalMessagesAreDeliveredOnlyOnce) {
    // Warn is reachable from inside a shots loop, where one unchanged matrix
    // produces the same message once per gate per shot. Delivering all of them
    // would bury every other diagnostic.
    for (int i = 0; i < 500; ++i) emit_warning("repeating");
    EXPECT_EQ(lines.size(), 1u)
        << "500 identical warnings must collapse to one delivery";
}

TEST_F(R1211WarningChannel, FlushReportsTheSuppressedCount) {
    for (int i = 0; i < 5; ++i) emit_warning("counted");
    ASSERT_EQ(lines.size(), 1u);

    flush_warnings();
    ASSERT_EQ(lines.size(), 2u) << "the repeat summary was not emitted";
    EXPECT_NE(lines[1].find("counted"), std::string::npos)
        << "the summary must identify which message it counts; got: " << lines[1];
    EXPECT_NE(lines[1].find("repeated 4 more times"), std::string::npos)
        << "five occurrences, one already delivered, so four remain; got: "
        << lines[1];
}

TEST_F(R1211WarningChannel, SingleOccurrenceProducesNoSummary) {
    emit_warning("solitary");
    flush_warnings();
    EXPECT_EQ(lines.size(), 1u)
        << "a message seen once needs no repeat line; a '[repeated 0 more "
           "times]' would be noise";
}

TEST_F(R1211WarningChannel, FlushIsIdempotent) {
    for (int i = 0; i < 3; ++i) emit_warning("once only");
    flush_warnings();
    const std::size_t after_first = lines.size();
    flush_warnings();
    flush_warnings();
    EXPECT_EQ(lines.size(), after_first)
        << "flushing clears the counts, so a second flush has nothing to say";
}

TEST_F(R1211WarningChannel, FlushResetsDeduplication) {
    emit_warning("cyclic");
    emit_warning("cyclic");
    ASSERT_EQ(occurrences(lines, "cyclic"), 1u);
    flush_warnings();
    lines.clear();

    emit_warning("cyclic");
    EXPECT_EQ(lines.size(), 1u)
        << "after a flush the message is unseen again and must be delivered, "
           "otherwise a warning could be silenced for the rest of the process";
}

TEST_F(R1211WarningChannel, FlushSummarisesEveryPendingMessage) {
    emit_warning("first");
    emit_warning("first");
    emit_warning("second");
    emit_warning("second");
    emit_warning("second");
    lines.clear();

    flush_warnings();
    ASSERT_EQ(lines.size(), 2u);
    EXPECT_EQ(occurrences(lines, "first"), 1u);
    EXPECT_EQ(occurrences(lines, "second"), 1u);
    EXPECT_EQ(occurrences(lines, "repeated 1 more times"), 1u);
    EXPECT_EQ(occurrences(lines, "repeated 2 more times"), 1u);
}

TEST_F(R1211WarningChannel, EmptyMessageIsStillDelivered) {
    emit_warning("");
    ASSERT_EQ(lines.size(), 1u)
        << "an empty message is a caller error, not a reason to drop output";
    EXPECT_EQ(lines[0], "");
}

TEST_F(R1211WarningChannel, LongMessageSurvivesIntact) {
    const std::string long_message(8192, 'x');
    emit_warning(long_message);
    ASSERT_EQ(lines.size(), 1u);
    EXPECT_EQ(lines[0], long_message)
        << "the channel must not truncate; a residual and a tolerance at the "
           "end of a message are the actionable part";
}

// =============================================================================
// Handler replacement
// =============================================================================

TEST(R1211WarningHandover, ReplacementFlushesToTheOutgoingHandler) {
    // A repeat count belongs to the sink that saw the occurrence it counts.
    // Delivering it to the replacement would attribute work to a handler that
    // was not installed when it happened.
    reset_channel();
    std::vector<std::string> outgoing;
    std::vector<std::string> incoming;

    set_warning_handler([&outgoing](const std::string& m) { outgoing.push_back(m); });
    for (int i = 0; i < 4; ++i) emit_warning("handover");
    ASSERT_EQ(outgoing.size(), 1u);

    set_warning_handler([&incoming](const std::string& m) { incoming.push_back(m); });
    reset_channel();

    ASSERT_EQ(outgoing.size(), 2u)
        << "the pending count did not reach the handler being replaced";
    EXPECT_NE(outgoing[1].find("repeated 3 more times"), std::string::npos);
    EXPECT_TRUE(incoming.empty())
        << "the incoming handler never saw the first occurrence and must not "
           "receive its repeat count";
}

TEST(R1211WarningHandover, CountsDoNotSurviveIntoTheNewHandler) {
    reset_channel();
    std::vector<std::string> first;
    std::vector<std::string> second;

    set_warning_handler([&first](const std::string& m) { first.push_back(m); });
    emit_warning("shared text");

    set_warning_handler([&second](const std::string& m) { second.push_back(m); });
    emit_warning("shared text");
    reset_channel();

    EXPECT_EQ(second.size(), 1u)
        << "the replacement flushed the table, so the message is unseen from "
           "the new handler's point of view and must be delivered to it";
}

// =============================================================================
// Saturation
// =============================================================================

TEST_F(R1211WarningChannel, DeduplicationHoldsUpToTheTrackingCap) {
    for (std::size_t i = 0; i < MAX_TRACKED_MESSAGES; ++i)
        emit_warning("tracked " + std::to_string(i));
    ASSERT_EQ(lines.size(), MAX_TRACKED_MESSAGES);

    lines.clear();
    for (std::size_t i = 0; i < MAX_TRACKED_MESSAGES; ++i)
        emit_warning("tracked " + std::to_string(i));
    EXPECT_TRUE(lines.empty())
        << "every one of these is already tracked and must be counted rather "
           "than delivered";
}

TEST_F(R1211WarningChannel, MessagesBeyondTheCapAreDeliveredEveryTime) {
    // A workload producing unbounded distinct warnings is one where the counts
    // are not the interesting part, and the table must not grow to track them.
    // The trade is that those messages stop being deduplicated.
    for (std::size_t i = 0; i < MAX_TRACKED_MESSAGES; ++i)
        emit_warning("tracked " + std::to_string(i));
    lines.clear();

    for (int i = 0; i < 5; ++i) emit_warning("untracked overflow");
    EXPECT_EQ(lines.size(), 5u)
        << "an untracked message cannot be recognised as a repeat, so it is "
           "delivered rather than silently dropped";
}

TEST_F(R1211WarningChannel, SaturationDoesNotDisturbAlreadyTrackedMessages) {
    for (std::size_t i = 0; i < MAX_TRACKED_MESSAGES; ++i)
        emit_warning("tracked " + std::to_string(i));
    for (int i = 0; i < 20; ++i)
        emit_warning("overflow " + std::to_string(i));
    lines.clear();

    emit_warning("tracked 0");
    EXPECT_TRUE(lines.empty())
        << "a message already in the table keeps its deduplication regardless "
           "of how many untracked ones arrived after it";
}

TEST_F(R1211WarningChannel, FlushAfterSaturationClearsTheTable) {
    for (std::size_t i = 0; i < MAX_TRACKED_MESSAGES + 20; ++i)
        emit_warning("saturating " + std::to_string(i));
    flush_warnings();
    lines.clear();

    emit_warning("saturating 0");
    EXPECT_EQ(lines.size(), 1u)
        << "the table was cleared, so room exists again and the message is "
           "delivered as a first occurrence";
}

// =============================================================================
// ScopedWarningFlush
// =============================================================================

TEST_F(R1211WarningChannel, ScopedFlushEmitsPendingCountsOnScopeExit) {
    {
        ScopedWarningFlush guard;
        for (int i = 0; i < 3; ++i) emit_warning("scoped");
        EXPECT_EQ(lines.size(), 1u) << "still inside the scope";
    }
    ASSERT_EQ(lines.size(), 2u);
    EXPECT_NE(lines[1].find("repeated 2 more times"), std::string::npos);
}

TEST_F(R1211WarningChannel, ScopedFlushEmitsPendingCountsWhenUnwinding) {
    // Each backend run() holds one of these. A run that throws part way must
    // still attribute its repeat counts to itself rather than leaving them to
    // accumulate into whichever run comes next.
    try {
        ScopedWarningFlush guard;
        for (int i = 0; i < 3; ++i) emit_warning("unwound");
        throw std::runtime_error("simulated backend failure");
    } catch (const std::runtime_error&) {
    }
    ASSERT_EQ(lines.size(), 2u)
        << "the repeat count was lost when the scope exited by exception";
    EXPECT_NE(lines[1].find("repeated 2 more times"), std::string::npos);
}

TEST_F(R1211WarningChannel, NestedScopedFlushesAreHarmless) {
    {
        ScopedWarningFlush outer;
        emit_warning("nested");
        {
            ScopedWarningFlush inner;
            emit_warning("nested");
        }
        EXPECT_EQ(lines.size(), 2u)
            << "the inner scope flushed a pending count of one repeat";
        emit_warning("nested");
    }
    EXPECT_EQ(lines.size(), 3u)
        << "the outer flush found a single fresh occurrence pending, which "
           "needs no summary line";
}

TEST(R1211WarningScopeGuard, IsNeitherCopyableNorAssignable) {
    // Copying the guard would flush twice, splitting one run's repeat count
    // across two summaries.
    EXPECT_FALSE(std::is_copy_constructible<ScopedWarningFlush>::value);
    EXPECT_FALSE(std::is_copy_assignable<ScopedWarningFlush>::value);
}

// =============================================================================
// Concurrency
// =============================================================================
// The handler is invoked with the channel lock held, which is what keeps a
// user sink from being entered by two OpenMP threads at once. Warnings
// originate inside parallel regions, so without that guarantee every caller
// would have to write their own lock.

#ifdef _OPENMP

TEST(R1211WarningConcurrency, HandlerIsNeverEnteredConcurrently) {
    reset_channel();

    std::atomic<int> depth{0};
    std::atomic<int> max_depth{0};
    std::atomic<int> deliveries{0};
    std::atomic<int> spin_sink{0};
    int observed_threads = 1;

    set_warning_handler([&](const std::string&) {
        const int now = depth.fetch_add(1) + 1;
        int previous = max_depth.load();
        while (now > previous && !max_depth.compare_exchange_weak(previous, now)) {
        }
        // Widen the window a concurrent entry would have to land in. This makes
        // a missing lock far more likely to be observed; it is not what makes
        // the assertion valid, which is why nothing here depends on timing.
        for (int spin = 0; spin < 256; ++spin) spin_sink.fetch_add(1);
        deliveries.fetch_add(1);
        depth.fetch_sub(1);
    });

    // The first MAX_TRACKED_MESSAGES distinct messages fill the count table.
    // Past that every unrecognised message is delivered rather than counted,
    // which is what produces enough handler entries to be worth racing.
    #pragma omp parallel
    {
        #pragma omp single
        observed_threads = omp_get_num_threads();

        #pragma omp for schedule(static, 1)
        for (int i = 0; i < 8192; ++i)
            emit_warning("concurrent " + std::to_string(i));
    }

    const int entries = deliveries.load();
    const int worst = max_depth.load();
    reset_channel();

    if (observed_threads < 2) {
        GTEST_SKIP() << "OpenMP runtime granted only 1 thread; cannot exercise "
                        "concurrent entry into the warning handler.";
    }
    EXPECT_GT(entries, 0) << "no warning reached the handler at all";
    EXPECT_EQ(worst, 1)
        << "the handler was entered by " << worst
        << " threads at once; it is invoked under the channel lock precisely so "
           "a user sink never has to be thread safe";
}

TEST(R1211WarningConcurrency, ConcurrentIdenticalMessagesDeliverExactlyOnce) {
    // find-then-insert is only atomic because it happens under the lock.
    // Without it two threads can both miss the lookup and both deliver, which
    // would show up here as more than one delivery.
    reset_channel();

    std::atomic<int> deliveries{0};
    int observed_threads = 1;
    set_warning_handler([&](const std::string&) { deliveries.fetch_add(1); });

    #pragma omp parallel
    {
        #pragma omp single
        observed_threads = omp_get_num_threads();

        #pragma omp for schedule(static, 1)
        for (int i = 0; i < 8192; ++i)
            emit_warning("one message from every thread");
    }

    const int delivered = deliveries.load();
    reset_channel();

    if (observed_threads < 2) {
        GTEST_SKIP() << "OpenMP runtime granted only 1 thread; cannot exercise "
                        "the find-then-insert race.";
    }
    EXPECT_EQ(delivered, 1)
        << "8192 identical concurrent warnings produced " << delivered
        << " deliveries; the lookup and the insert must be one critical section";
}
#endif
