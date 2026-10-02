// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

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
// capture_warnings puts the warning handler back on every exit, a throwing
// run included, so a refusal inside it cannot leave the channel pointing at a
// capture that no longer exists.
//
// The circuits live here because their anchor expectations are hand computed
// against their exact shape. A suite rebuilding one locally could drift from
// the arithmetic written out in the comments below.

#include <gtest/gtest.h>

#include "lindblad/circuit.hpp"
#include "lindblad/observation.hpp"
#include "lindblad/noise.hpp"
#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "v11311_helpers.hpp"
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
    v11311::WarningCapture capture;
    fn();
    return capture.messages();
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

// Four qubits, laid out so the layering is not the obvious one. Layers are the
// scheduler's (asap_schedule_times): a gate starts at the first cycle every
// operand wire is free.
//
//   index 0  h(0)      cycle 0
//   index 1  h(1)      cycle 0
//   index 2  cx(0,1)   cycle 1
//   index 3  cx(2,3)   cycle 0   qubits 2 and 3 are free from the start
//   index 4  x(0)      cycle 2
//   index 5  cx(1,2)   cycle 2
//
// A boundary is an index after which every instruction of cycles 0..T has run
// and nothing later has. Layer 0 has none: cx(2,3) belongs to it but runs after
// cx(0,1) from layer 1, so no executed prefix is exactly layer 0. Layer 1 ends
// at index 3, where cycles 0 and 1 are both complete, and layer 2 at the end.
// Reading the circuit in order and closing a layer on each qubit collision
// gives {1, 3, 5} instead, which is why this circuit separates the two.
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

inline std::vector<int> layered_circuit_layer_ends() { return {3, 5}; }

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
// How a run refuses
// =============================================================================
// Every backend's run() throws when it refuses, with the run's own name as the
// entry point and as the start of the message. A refusal the plan or the
// circuit decides comes before the first gate, so it leaves no failed-run
// record behind: nothing had been computed. run_refusal asserts all of that,
// and the exact type, and returns the message, so a caller can go on to assert
// what the refusal named without repeating the run.

template <class E = lindblad::InvalidArgument, class Run>
inline std::string run_refusal(const std::string& entry_point, Run&& run) {
    const std::uint64_t stores = lindblad::detail::failed_run_stores();
    const auto e = v11311::thrown<E>(std::forward<Run>(run));
    EXPECT_EQ(lindblad::detail::failed_run_stores(), stores)
        << "a refusal before the first gate left a failed-run record";
    if (!e) return {};
    EXPECT_EQ(e->entry_point(), entry_point);
    const std::string message = e->what();
    EXPECT_EQ(message.rfind(entry_point + ": ", 0), 0u) << message;
    return message;
}

inline std::string sv_run_failure(const QuantumCircuit& qc, const RunPlan& plan,
                                  int shots = 0) {
    lindblad::StatevectorSimulator sim;
    return run_refusal("StatevectorSimulator::run", [&] { sim.run(qc, shots, 20261, plan); });
}

inline std::string dm_run_failure(const QuantumCircuit& qc, const RunPlan& plan,
                                  int shots = 16) {
    lindblad::DensityMatrixSimulator sim;
    const lindblad::NoiseModel noise;
    return run_refusal("DensityMatrixSimulator::run",
                       [&] { sim.run(qc, noise, shots, 20261, plan); });
}

inline std::string clifford_run_failure(const QuantumCircuit& qc, const RunPlan& plan,
                                        int shots = 16) {
    lindblad::CliffordSimulator sim;
    return run_refusal("CliffordSimulator::run", [&] { sim.run(qc, shots, 20261, plan); });
}

inline std::string mps_run_failure(const QuantumCircuit& qc, const RunPlan& plan,
                                   int shots = 16) {
    lindblad::MPSSimulator sim;
    return run_refusal("MPSSimulator::run", [&] { sim.run(qc, 8, shots, 20261, plan); });
}

}  // namespace v11261
