#pragma once

// R.1.21.1 test wave - the shared Class C policy contract.
//
// Nineteen entry points take a ValidationOptions, and every one of them owes
// the same promises on the three responses: Throw rejects, Warn reports and
// proceeds, Ignore does not look. Stating that contract once here and applying
// it per entry point keeps the per-backend files to the part that actually
// differs, which is how each primitive is reached and what a violating operand
// looks like for it.
//
// Repair::Attempt is the knob that splits the contract, so it has two helpers
// rather than one. Where the property has a repair, the call performs it and
// returns whatever the response says; where it has none, asking for one throws
// under every response, because that is a mistake in the calling code rather
// than a property of the operand. Unitarity is repaired by polar projection and
// normalization by division, so those entry points take
// expect_repairs_invalid. Trace preservation has no cheap canonical repair, so
// its entry points take expect_rejects_invalid. Choosing the wrong one is not a
// style question: it asserts the opposite thing about the same operand.
//
// Both helpers walk all six combinations rather than the repairing one alone.
// The response knob is independent of the repair knob by construction, and a
// contract that only ever asked for a repair under Throw could not tell an
// implementation that honoured that independence from one that ignored the
// response entirely.
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

// `repair` defaults to None so a call naming only a response reads as the
// response contract it is asserting, and the repairing cases name the knob they
// are exercising.
template <typename Apply>
CallOutcome call_under(Apply& apply_with, lindblad::Validation policy,
                       double atol = 1e-12,
                       lindblad::Repair repair = lindblad::Repair::None) {
    CallOutcome out;
    try {
        apply_with(lindblad::ValidationOptions{policy, atol, repair});
    } catch (const std::invalid_argument& e) {
        out.threw = true;
        out.message = e.what();
    }
    return out;
}

// The three responses, for the loops that must cover all of them.
inline const lindblad::Validation kResponses[]{
    lindblad::Validation::Throw, lindblad::Validation::Warn,
    lindblad::Validation::Ignore};

// Every reachable policy, for a test whose claim is about all of them at once
// rather than about one. Six, because the two knobs are independent: three
// responses, each with and without a repair.
inline const lindblad::ValidationOptions kAllPolicies[]{
    {lindblad::Validation::Throw},
    {lindblad::Validation::Warn},
    {lindblad::Validation::Ignore},
    {lindblad::Validation::Throw, lindblad::DEFAULT_PHYSICAL_ATOL,
     lindblad::Repair::Attempt},
    {lindblad::Validation::Warn, lindblad::DEFAULT_PHYSICAL_ATOL,
     lindblad::Repair::Attempt},
    {lindblad::Validation::Ignore, lindblad::DEFAULT_PHYSICAL_ATOL,
     lindblad::Repair::Attempt},
};

inline const char* response_name(lindblad::Validation v) {
    switch (v) {
        case lindblad::Validation::Throw:  return "Throw";
        case lindblad::Validation::Warn:   return "Warn";
        case lindblad::Validation::Ignore: return "Ignore";
    }
    return "?";
}

// Names both knobs, so a failure inside a six-way loop says which of the six
// broke rather than printing an enumerator's integer.
inline std::string policy_name(const lindblad::ValidationOptions& v) {
    return std::string(response_name(v.policy)) +
           (v.repair == lindblad::Repair::Attempt ? " with Repair::Attempt"
                                                  : " with Repair::None");
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
        WarningProbe probe;
        const auto out = call_under(apply_with, lindblad::Validation::Ignore);
        EXPECT_FALSE(out.threw)
            << label << ": Ignore checked anyway. Message: " << out.message;
        EXPECT_EQ(probe.count(), 0u)
            << label << ": Ignore reported something; it is the one policy "
                        "that costs nothing and says nothing";
    }
    // Asking for a repair this property does not define is a mistake in the
    // calling code, so it is refused under every response. A response that
    // absorbed it would return an unphysical operand to a caller who asked for
    // a correction the library cannot make, which is the one outcome the repair
    // knob exists to rule out.
    for (auto response : kResponses) {
        const auto out = call_under(apply_with, response, 1e-12,
                                    lindblad::Repair::Attempt);
        EXPECT_TRUE(out.threw)
            << label << ": Repair::Attempt returned under "
            << response_name(response)
            << " for a property with no repair, so an unphysical operand was "
               "accepted under a request to correct it";
        EXPECT_NE(out.message.find("no repair defined"), std::string::npos)
            << label << ": Repair::Attempt under " << response_name(response)
            << " must say a repair is missing. Got: " << out.message;
    }
}

// The same contract for an entry point whose property HAS a repair. Three of
// the four policies behave identically to the case above and are asserted the
// same way; Repair::Attempt is what differs, and it differs completely. A
// repair exists and the caller opted into it, so the call must carry it out and
// return rather than report that no repair is defined, whatever the response
// says. The response governs an operand that is still invalid afterwards, and a
// repaired one is not.
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
        WarningProbe probe;
        const auto out = call_under(apply_with, lindblad::Validation::Ignore);
        EXPECT_FALSE(out.threw)
            << label << ": Ignore checked anyway. Message: " << out.message;
        EXPECT_EQ(probe.count(), 0u)
            << label << ": Ignore reported something; it is the one policy "
                        "that costs nothing and says nothing";
    }
    // The repair runs under every response, because the response decides only
    // what happens to an operand that is STILL invalid. An implementation that
    // repaired under Throw alone would pass the old four-policy contract and
    // fail here, which is the point of walking the row.
    for (auto response : kResponses) {
        const auto out = call_under(apply_with, response, 1e-12,
                                    lindblad::Repair::Attempt);
        EXPECT_FALSE(out.threw)
            << label << ": Repair::Attempt under " << response_name(response)
            << " declined to repair a property whose repair is implemented and "
               "reachable, so the caller was refused a correction it asked for "
               "and the library can perform. Message: "
            << out.message;
        EXPECT_EQ(out.message.find("no repair defined"), std::string::npos)
            << label << ": the diagnostic under " << response_name(response)
            << " claims no repair exists for a property that has one. Got: "
            << out.message;
    }
}

// A physically valid operand must pass under every policy, silently. Without
// this half, an entry point that rejected everything would satisfy the tests
// above.
// Every combination is walked, including the repairing ones: an operand inside
// tolerance owes nothing to either knob, so a repair must not run and nothing
// must be said. An implementation that repaired unconditionally rather than on
// a measured violation would pass every test above and fail here.
template <typename Apply>
void expect_accepts_valid(const char* label, Apply apply_with) {
    const lindblad::Repair repairs[]{lindblad::Repair::None,
                                     lindblad::Repair::Attempt};
    for (auto repair : repairs) {
        for (auto response : kResponses) {
            WarningProbe probe;
            const auto out = call_under(apply_with, response, 1e-12, repair);
            EXPECT_FALSE(out.threw)
                << label << ": a valid operand was rejected under "
                << response_name(response)
                << (repair == lindblad::Repair::Attempt ? " with Repair::Attempt"
                                                        : " with Repair::None")
                << ". Message: " << out.message;
            EXPECT_EQ(probe.count(), 0u)
                << label << ": a valid operand produced a warning under "
                << response_name(response)
                << (repair == lindblad::Repair::Attempt ? " with Repair::Attempt"
                                                        : " with Repair::None");
        }
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
