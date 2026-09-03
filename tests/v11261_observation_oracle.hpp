#pragma once

// Helpers shared by the 1.1.26.1 observation suites.
//
// RecordingObserver is the instrument every anchor claim is measured with, and
// it is deliberately a plain Observer implementation living in test code. The
// release claims a caller-written observer has no less access than a built-in
// one, and a suite that measured firing through a built-in observer would
// never exercise that claim. Every coordinate an ObservationContext carries is
// copied out, including the thread the call arrived on, because the interface
// promises the calling thread and nothing else in the tree asserts it.
//
// capture_warnings is defined here rather than taken from the Clifford oracle
// that also has one: that header pulls two backends and a set of stabilizer
// oracles into every translation unit including it, and none of this wave
// needs them.
//
// The circuits live here because their anchor expectations are hand computed
// against their exact shape. A suite rebuilding one locally could drift from
// the arithmetic written out in the comments below.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/observation.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/validation.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace v11261 {

using lindblad::Anchor;
using lindblad::Instruction;
using lindblad::ObservationContext;
using lindblad::Observer;
using lindblad::QuantumCircuit;
using lindblad::RunPlan;
using lindblad::StateForm;

// =============================================================================
// The warning channel
// =============================================================================

// Everything the channel delivered while `fn` ran. The channel deduplicates,
// so a message emitted many times arrives once followed by a repeat tally,
// which is why the two are separated below rather than counted together.
inline std::vector<std::string> capture_warnings(const std::function<void()>& fn) {
    lindblad::flush_warnings();
    std::vector<std::string> captured;
    lindblad::set_warning_handler(
        [&captured](const std::string& m) { captured.push_back(m); });
    fn();
    lindblad::set_warning_handler(nullptr);
    return captured;
}

// The entries that are first deliveries rather than repeat tallies.
inline std::vector<std::string> first_deliveries(const std::vector<std::string>& msgs) {
    std::vector<std::string> out;
    for (const std::string& m : msgs) {
        if (m.find("[repeated ") == std::string::npos) out.push_back(m);
    }
    return out;
}

inline bool any_contains(const std::vector<std::string>& msgs, const std::string& needle) {
    for (const std::string& m : msgs) {
        if (m.find(needle) != std::string::npos) return true;
    }
    return false;
}

// =============================================================================
// RecordingObserver - one firing in, one row out
// =============================================================================

struct Firing {
    int shot = -1;
    int instruction = -2;  // -1 is a real value (at_start), so the unset one is not
    std::string anchor;
    StateForm form = StateForm::Statevector;
    int n_qubits = -1;
    int n_shots = -1;
    std::vector<int> clbits;
    std::thread::id thread;
};

class RecordingObserver : public Observer {
public:
    void begin_run(int n_qubits, int n_shots) override {
        ++begins_;
        begin_qubits_ = n_qubits;
        begin_shots_ = n_shots;
    }

    void observe(const ObservationContext& ctx) override {
        Firing f;
        f.shot = ctx.shot;
        f.instruction = ctx.instruction_index;
        f.anchor = ctx.anchor;
        f.form = ctx.state.form();
        f.n_qubits = ctx.state.n_qubits();
        f.n_shots = ctx.n_shots;
        f.clbits = ctx.clbits;
        f.thread = std::this_thread::get_id();
        firings_.push_back(std::move(f));
    }

    void end_run() override { ++ends_; }

    const std::vector<Firing>& firings() const { return firings_; }
    std::size_t count() const { return firings_.size(); }

    int begins() const { return begins_; }
    int ends() const { return ends_; }
    int begin_qubits() const { return begin_qubits_; }
    int begin_shots() const { return begin_shots_; }

    // Each firing's instruction index, in arrival order. Comparing whole
    // sequences rather than probing one entry is what pins ORDER as well as
    // membership, and order is half of what an anchor promises.
    std::vector<int> indices() const {
        std::vector<int> out;
        out.reserve(firings_.size());
        for (const Firing& f : firings_) out.push_back(f.instruction);
        return out;
    }

    std::vector<int> shots() const {
        std::vector<int> out;
        out.reserve(firings_.size());
        for (const Firing& f : firings_) out.push_back(f.shot);
        return out;
    }

    std::vector<std::string> anchors() const {
        std::vector<std::string> out;
        out.reserve(firings_.size());
        for (const Firing& f : firings_) out.push_back(f.anchor);
        return out;
    }

private:
    std::vector<Firing> firings_;
    int begins_ = 0;
    int ends_ = 0;
    int begin_qubits_ = -1;
    int begin_shots_ = -1;
};

using RecorderPtr = std::shared_ptr<RecordingObserver>;

inline RecorderPtr recorder() { return std::make_shared<RecordingObserver>(); }

// =============================================================================
// Circuits whose anchor expectations are worked out by hand
// =============================================================================

// Four qubits, laid out so the layering is not the obvious one. The greedy
// walk closes a layer when an instruction collides with a qubit already used
// inside it, and clears the WHOLE occupancy set at that point rather than
// carrying the untouched qubits forward:
//
//   index 0  h(0)      no collision       used {0}
//   index 1  h(1)      no collision       used {0,1}
//   index 2  cx(0,1)   collides on 0   -> layer ends at 1, then used {0,1}
//   index 3  cx(2,3)   no collision       used {0,1,2,3}
//   index 4  x(0)      collides on 0   -> layer ends at 3, then used {0}
//   index 5  cx(1,2)   no collision       used {0,1,2}
//   end of circuit                     -> layer ends at 5
//
// Qubit 3 goes idle from index 4 onward, and index 3 shares a layer with two
// instructions touching neither of its qubits, so a layering computed any
// other way lands somewhere other than {1, 3, 5}.
inline QuantumCircuit layered_circuit() {
    QuantumCircuit qc(4);
    qc.h(0);
    qc.h(1);
    qc.cx(0, 1);
    qc.cx(2, 3);
    qc.x(0);
    qc.cx(1, 2);
    return qc;
}

inline std::vector<int> layered_circuit_layer_ends() { return {1, 3, 5}; }

// Mid-circuit measurement plus feedforward, which is what puts a backend on
// its per-shot trajectory route: the X on qubit 1 runs or does not run
// depending on an outcome drawn that shot.
inline QuantumCircuit feedforward_circuit() {
    QuantumCircuit qc(2, 2);
    qc.h(0);
    qc.measure(0, 0);
    qc.add_if(0, 1, Instruction::GateType::X, {1});
    qc.measure(1, 1);
    return qc;
}

// Instructions carry their label as a public field, so labelling happens after
// the builder call rather than through it.
inline QuantumCircuit& label_instruction(QuantumCircuit& qc, int index, std::string label) {
    qc.instructions[static_cast<std::size_t>(index)].label = std::move(label);
    return qc;
}

// =============================================================================
// Running one
// =============================================================================

inline lindblad::StatevectorSimulator::Result run_sv(const QuantumCircuit& qc,
                                                     const RunPlan& plan,
                                                     int shots = 0,
                                                     std::uint64_t seed = 20261) {
    lindblad::StatevectorSimulator sim;
    return sim.run(qc, shots, seed, plan);
}

// A plan carrying one observer on one anchor, which is most of what the anchor
// suites need.
inline RunPlan plan_with(Anchor anchor, lindblad::ObserverPtr observer) {
    RunPlan plan;
    plan.observations.observe(std::move(anchor), std::move(observer));
    return plan;
}

// =============================================================================
// How a run reports a failure, which is a property of the backend
// =============================================================================
// The two styles are not interchangeable and the choice is not the failure's:
// StatevectorSimulator and DensityMatrixSimulator carry an error channel on
// their Result and catch inside run() to report through it, while MPSSimulator
// and CliffordSimulator have no such field and surface the same failure by
// throwing. A suite asserting that a run failed has to ask the backend it ran
// on, so both questions are spelled here once instead of at every call site.
//
// Each returns the message, so a caller can go on to assert what the failure
// named without repeating the run.

inline std::string sv_run_failure(const QuantumCircuit& qc, const RunPlan& plan,
                                  int shots = 0) {
    lindblad::StatevectorSimulator sim;
    auto result = sim.run(qc, shots, 20261, plan);
    EXPECT_FALSE(result.success) << "the run was expected to fail";
    return result.error_message;
}

template <typename Fn>
inline std::string throwing_run_failure(Fn&& fn) {
    try {
        fn();
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    ADD_FAILURE() << "the run was expected to throw std::invalid_argument";
    return {};
}

}  // namespace v11261
