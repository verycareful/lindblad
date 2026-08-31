#pragma once

// R.1.21.1 test wave - the shared Class C policy contract.
//
// Nineteen entry points take a ValidationOptions, and every one of them owes
// the same promises on three policies: Throw rejects, Warn reports and
// proceeds, Ignore does not look. Stating that contract once here and applying
// it per entry point keeps the per-backend files to the part that actually
// differs, which is how each primitive is reached and what a violating operand
// looks like for it.
//
// Fix is the policy that splits, so it has two helpers rather than one. Where
// the property has a repair, Fix performs it and returns; where it has none,
// Fix says so and throws. Unitarity is repaired by polar projection and
// normalization by division, so those entry points take
// expect_repairs_invalid. Trace preservation has no cheap canonical repair, so
// its entry points take expect_rejects_invalid. Choosing the wrong one is not a
// style question: it asserts the opposite thing about the same policy.
//
// The helpers take a callable rather than an operand so the caller can build a
// fresh state per invocation. That matters: each policy is exercised by a
// separate call, and under Warn or Ignore the call succeeds and leaves the
// state unphysical, so reusing one state would make later policies act on
// input the earlier ones corrupted.

#include <gtest/gtest.h>

#include "lindblad/validation.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace r1211 {

// Captures warnings for the duration of a scope, leaving the channel clean on
// both entry and exit. Deduplication is keyed on message text and survives
// until a flush, so without the reset on construction a probe could miss a
// warning that an earlier probe in the same test had already seen.
class WarningProbe {
public:
    WarningProbe() {
        lindblad::set_warning_handler(nullptr);
        lindblad::flush_warnings();
        lindblad::set_warning_handler(
            [this](const std::string& m) { lines_.push_back(m); });
    }

    ~WarningProbe() {
        lindblad::set_warning_handler(nullptr);
        lindblad::flush_warnings();
    }

    WarningProbe(const WarningProbe&) = delete;
    WarningProbe& operator=(const WarningProbe&) = delete;

    std::size_t count() const { return lines_.size(); }
    const std::vector<std::string>& lines() const { return lines_; }

    bool any_contains(const std::string& needle) const {
        for (const auto& line : lines_)
            if (line.find(needle) != std::string::npos) return true;
        return false;
    }

private:
    std::vector<std::string> lines_;
};

// Result of one guarded call.
struct CallOutcome {
    bool threw = false;
    std::string message;
};

template <typename Apply>
CallOutcome call_under(Apply& apply_with, lindblad::Validation policy,
                       double atol = 1e-12) {
    CallOutcome out;
    try {
        apply_with(lindblad::ValidationOptions{policy, atol});
    } catch (const std::invalid_argument& e) {
        out.threw = true;
        out.message = e.what();
    }
    return out;
}

// The full contract against an operand that violates its physical property.
// `label` names the entry point so a failure identifies which of the nineteen
// broke rather than only which policy.
template <typename Apply>
void expect_rejects_invalid(const char* label, Apply apply_with) {
    {
        const auto out = call_under(apply_with, lindblad::Validation::Throw);
        EXPECT_TRUE(out.threw)
            << label << ": Throw accepted a physically invalid operand";
        EXPECT_FALSE(out.message.empty())
            << label << ": Throw gave an empty diagnostic";
    }
    {
        WarningProbe probe;
        const auto out = call_under(apply_with, lindblad::Validation::Warn);
        EXPECT_FALSE(out.threw)
            << label << ": Warn threw; a policy that throws is Throw. Message: "
            << out.message;
        EXPECT_GE(probe.count(), 1u)
            << label << ": Warn proceeded without reporting anything, which is "
                        "indistinguishable from Ignore";
    }
    {
        const auto out = call_under(apply_with, lindblad::Validation::Fix);
        EXPECT_TRUE(out.threw)
            << label << ": Fix returned without repairing anything; a silent "
                        "Fix returns an unphysical result under a policy that "
                        "promised a correction";
        EXPECT_NE(out.message.find("no repair defined"), std::string::npos)
            << label << ": Fix must say a repair is missing. Got: "
            << out.message;
    }
    {
        WarningProbe probe;
        const auto out = call_under(apply_with, lindblad::Validation::Ignore);
        EXPECT_FALSE(out.threw)
            << label << ": Ignore checked anyway. Message: " << out.message;
        EXPECT_EQ(probe.count(), 0u)
            << label << ": Ignore reported something; it is the one policy "
                        "that costs nothing and says nothing";
    }
}

// The same contract for an entry point whose property HAS a repair. Three of
// the four policies behave identically to the case above and are asserted the
// same way; Fix is the one that differs, and it differs completely. A repair
// exists, the caller opted into it, so the call must carry it out and return
// rather than report that no repair is defined.
//
// What "repaired" means cannot be stated here, because it differs per entry
// point: a stored instruction matrix, a statevector, a density matrix. This
// asserts the POLICY contract, and each call site asserts the effect where it
// can actually see it.
template <typename Apply>
void expect_repairs_invalid(const char* label, Apply apply_with) {
    {
        const auto out = call_under(apply_with, lindblad::Validation::Throw);
        EXPECT_TRUE(out.threw)
            << label << ": Throw accepted a physically invalid operand";
        EXPECT_FALSE(out.message.empty())
            << label << ": Throw gave an empty diagnostic";
    }
    {
        WarningProbe probe;
        const auto out = call_under(apply_with, lindblad::Validation::Warn);
        EXPECT_FALSE(out.threw)
            << label << ": Warn threw; a policy that throws is Throw. Message: "
            << out.message;
        EXPECT_GE(probe.count(), 1u)
            << label << ": Warn proceeded without reporting anything, which is "
                        "indistinguishable from Ignore";
    }
    {
        const auto out = call_under(apply_with, lindblad::Validation::Fix);
        EXPECT_FALSE(out.threw)
            << label
            << ": Fix declined to repair a property that has a repair and "
               "ships one, so the caller was refused a correction it asked for "
               "and the library can perform. Message: "
            << out.message;
        EXPECT_EQ(out.message.find("no repair defined"), std::string::npos)
            << label
            << ": the diagnostic claims no repair exists for a property whose "
               "repair is implemented and reachable. Got: "
            << out.message;
    }
    {
        WarningProbe probe;
        const auto out = call_under(apply_with, lindblad::Validation::Ignore);
        EXPECT_FALSE(out.threw)
            << label << ": Ignore checked anyway. Message: " << out.message;
        EXPECT_EQ(probe.count(), 0u)
            << label << ": Ignore reported something; it is the one policy "
                        "that costs nothing and says nothing";
    }
}

// A physically valid operand must pass under every policy, silently. Without
// this half, an entry point that rejected everything would satisfy the tests
// above.
template <typename Apply>
void expect_accepts_valid(const char* label, Apply apply_with) {
    const lindblad::Validation policies[]{
        lindblad::Validation::Throw, lindblad::Validation::Warn,
        lindblad::Validation::Fix, lindblad::Validation::Ignore};
    for (auto policy : policies) {
        WarningProbe probe;
        const auto out = call_under(apply_with, policy);
        EXPECT_FALSE(out.threw)
            << label << ": a valid operand was rejected under policy "
            << static_cast<int>(policy) << ". Message: " << out.message;
        EXPECT_EQ(probe.count(), 0u)
            << label << ": a valid operand produced a warning under policy "
            << static_cast<int>(policy);
    }
}

// A tolerance wide enough to cover the defect must accept it under Throw, and
// one tighter than the operand's own rounding must reject it. Together these
// establish that atol is consulted rather than a fixed constant being used.
template <typename Apply>
void expect_tolerance_is_honoured(const char* label, Apply apply_with,
                                  double deviation) {
    {
        const auto out =
            call_under(apply_with, lindblad::Validation::Throw, deviation * 10.0);
        EXPECT_FALSE(out.threw)
            << label << ": a tolerance ten times the actual deviation still "
                        "rejected. Message: " << out.message;
    }
    {
        const auto out =
            call_under(apply_with, lindblad::Validation::Throw, deviation / 10.0);
        EXPECT_TRUE(out.threw)
            << label << ": a tolerance a tenth of the actual deviation "
                        "accepted, so atol is not being consulted";
    }
}

} // namespace r1211
