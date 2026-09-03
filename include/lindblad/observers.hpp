#pragma once

#include "lindblad/observation.hpp"
#include "lindblad/operators.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// =============================================================================
// observers - the built-in catalogue
// =============================================================================
// Ordinary implementations of Observer with no privileged access to anything.
// That is the property worth preserving: an observer written by a caller sits
// beside these rather than beneath them, and adding a quantity to this file is
// never the only way to obtain one.
//
// Every observer here keeps its results, so the typed road is to construct one,
// attach it, and read it afterwards. Giving one a label additionally writes its
// results into the run's ObservationBundle, which is the road that crosses a
// language boundary. One class serves both.

namespace lindblad {

// -----------------------------------------------------------------------------
// FiringPoint - where in the run an observation was taken
// -----------------------------------------------------------------------------
// Recorded for every firing of every observer below, because the k-th reading
// is not self describing: an observer attached to two anchors, or watched over
// many shots, produces a sequence whose entries otherwise cannot be told apart.

struct FiringPoint {
    int shot;            // 0-based; 0 for a single trajectory
    int instruction;     // index reached in the circuit; -1 at the start
    std::string anchor;  // which anchor fired, as Anchor::name() spells it
};

// -----------------------------------------------------------------------------
// BundleWriter - the labelled half, shared by every observer below
// -----------------------------------------------------------------------------
// Firings are buffered and written at end_run, because the key depends on how
// many there were: an observer that fired once writes under its plain label,
// and one that fired repeatedly writes under "label@index#shot" so that no
// firing overwrites another. A label that collides with one already in the
// bundle is refused by the bundle itself.

class BundleWriter : public Observer {
public:
    explicit BundleWriter(std::string label = {}) : label_(std::move(label)) {}

    const std::string& label() const override { return label_; }
    bool labelled() const { return !label_.empty(); }

    // How many times this observer fired, and where each firing happened. Every
    // observer's own results are in the same order, so results[k] belongs to
    // point(k).
    std::size_t count() const { return points_.size(); }
    const FiringPoint& point(std::size_t k) const;
    const std::vector<FiringPoint>& points() const { return points_; }

    void end_run() override;

protected:
    // Record one firing: always its point, and its payload too when a label was
    // given. Called once per firing that produced a reading, so an observer
    // that declined to produce one does not record a point for it either.
    void record(const ObservationContext& ctx, ObservationBundle::Payload payload);

private:
    struct Entry {
        int index;
        int shot;
        ObservationBundle::Payload payload;
    };

    std::string label_;
    ObservationBundle* bundle_ = nullptr;
    std::vector<Entry> entries_;
    std::vector<FiringPoint> points_;
};

// -----------------------------------------------------------------------------
// StateObserver - the state itself
// -----------------------------------------------------------------------------
// Default is the backend's own representation, which never converts and never
// refuses. Naming a form asks for that one, which answers to the knobs.

class StateObserver : public BundleWriter {
public:
    explicit StateObserver(std::string label = {})
        : BundleWriter(std::move(label)) {}
    StateObserver(StateForm form, std::string label = {})
        : BundleWriter(std::move(label)), native_(false), form_(form) {}

    bool preflight(const PreflightContext& ctx) override;
    void observe(const ObservationContext& ctx) override;

    StateForm form(std::size_t k = 0) const;

    const Statevector& statevector(std::size_t k = 0) const;
    const DensityMatrix& density_matrix(std::size_t k = 0) const;
    const StabilizerState& stabilizer(std::size_t k = 0) const;
    const MPSState& mps(std::size_t k = 0) const;

private:
    const std::shared_ptr<const void>& at(std::size_t k, StateForm wanted) const;

    bool native_ = true;
    StateForm form_ = StateForm::Statevector;
    std::vector<std::pair<StateForm, std::shared_ptr<const void>>> states_;
};

// -----------------------------------------------------------------------------
// ProbabilityObserver - the full outcome distribution
// -----------------------------------------------------------------------------

class ProbabilityObserver : public BundleWriter {
public:
    explicit ProbabilityObserver(std::string label = {})
        : BundleWriter(std::move(label)) {}

    bool preflight(const PreflightContext& ctx) override;
    void observe(const ObservationContext& ctx) override;

    const std::vector<double>& probabilities(std::size_t k = 0) const;

    // Entrywise mean over every firing. Empty when nothing was observed.
    std::vector<double> average() const;

private:
    std::vector<std::vector<double>> values_;
};

// -----------------------------------------------------------------------------
// AmplitudeObserver - named amplitudes, without materialising the rest
// -----------------------------------------------------------------------------
// The cheap way to look at a state: cost is the number of indices asked for,
// not 2^n, on any backend that holds amplitudes natively.

class AmplitudeObserver : public BundleWriter {
public:
    explicit AmplitudeObserver(std::vector<std::size_t> indices, std::string label = {})
        : BundleWriter(std::move(label)), indices_(std::move(indices)) {}

    bool preflight(const PreflightContext& ctx) override;
    void observe(const ObservationContext& ctx) override;

    const std::vector<std::size_t>& indices() const { return indices_; }
    const std::vector<Complex128>& amplitudes(std::size_t k = 0) const;

private:
    std::vector<std::size_t> indices_;
    std::vector<std::vector<Complex128>> values_;
};

// -----------------------------------------------------------------------------
// ExpectationObserver - <O> for a sparse Pauli observable
// -----------------------------------------------------------------------------
// A reduction, so it streams over the live state on every backend. On the
// tableau it costs O(terms * n^2) with no conversion at all, which is the case
// that would otherwise expand a compact state into 2^n amplitudes to compute
// one number.

class ExpectationObserver : public BundleWriter {
public:
    explicit ExpectationObserver(SparsePauliOp observable, std::string label = {})
        : BundleWriter(std::move(label)), observable_(std::move(observable)) {}

    void observe(const ObservationContext& ctx) override;

    const std::vector<double>& values() const { return values_; }
    double value(std::size_t k = 0) const;

    // Mean and variance across firings. With one firing the variance is zero,
    // which is a statement about the sample and not about the state.
    double mean() const;
    double variance() const;

private:
    SparsePauliOp observable_;
    std::vector<double> values_;
};

// -----------------------------------------------------------------------------
// PurityObserver - Tr(rho^2)
// -----------------------------------------------------------------------------
// Exactly 1 on the three backends that hold a pure state by construction,
// which is an answer rather than a refusal.

class PurityObserver : public BundleWriter {
public:
    explicit PurityObserver(std::string label = {}) : BundleWriter(std::move(label)) {}

    void observe(const ObservationContext& ctx) override;

    const std::vector<double>& values() const { return values_; }
    double value(std::size_t k = 0) const;

private:
    std::vector<double> values_;
};

// -----------------------------------------------------------------------------
// ClassicalRegisterObserver - the classical bits, mid-run
// -----------------------------------------------------------------------------
// What a feedforward circuit decided, at the point it decided it.

class ClassicalRegisterObserver : public BundleWriter {
public:
    explicit ClassicalRegisterObserver(std::string label = {})
        : BundleWriter(std::move(label)) {}

    void observe(const ObservationContext& ctx) override;

    const std::vector<int>& clbits(std::size_t k = 0) const;

private:
    std::vector<std::vector<int>> values_;
};

// -----------------------------------------------------------------------------
// BondDimensionObserver / TruncationObserver - MPS structure as it evolves
// -----------------------------------------------------------------------------
// Both refuse on any other backend through the response knob, since neither
// quantity exists there rather than being expensive to obtain.

class BondDimensionObserver : public BundleWriter {
public:
    explicit BondDimensionObserver(std::string label = {})
        : BundleWriter(std::move(label)) {}

    bool preflight(const PreflightContext& ctx) override;
    void observe(const ObservationContext& ctx) override;

    const std::vector<int>& bond_dimensions(std::size_t k = 0) const;

private:
    std::vector<std::vector<int>> values_;
};

class TruncationObserver : public BundleWriter {
public:
    explicit TruncationObserver(std::string label = {})
        : BundleWriter(std::move(label)) {}

    bool preflight(const PreflightContext& ctx) override;
    void observe(const ObservationContext& ctx) override;

    const std::vector<double>& values() const { return values_; }

private:
    std::vector<double> values_;
};

// -----------------------------------------------------------------------------
// EntropyObserver - how entangled the two sides of a cut are
// -----------------------------------------------------------------------------
// Reports the von Neumann entropy of the reduced state on `region`, in BITS.
// On the statevector, MPS and Clifford backends the global state is pure, so
// that IS the entanglement entropy across the cut. On the density matrix it is
// the entropy of the reduced state, which for a mixed global state mixes
// classical correlation with entanglement and is a different quantity: the
// observer reports which of the two it handed back rather than leaving the
// caller to infer it from the backend.
//
// `renyi_order` selects a Renyi entropy; 1 is von Neumann, 2 is -log2 of the
// reduced purity. Every order agrees on the Clifford backend, whose reduced
// spectra are flat.
//
// Cost per backend, since it varies by more than a constant:
//   Clifford     O(n * |A|^2) bit operations, no amplitudes, exact integers
//   MPS          O(n * chi^3), from the two environment Gram matrices
//   statevector  the reduced matrix of the SMALLER side, then an eigensolve
//   density      partial trace to the smaller side, then an eigensolve

class EntropyObserver : public BundleWriter {
public:
    explicit EntropyObserver(std::vector<int> region, double renyi_order = 1.0,
                             std::string label = {});

    bool preflight(const PreflightContext& ctx) override;
    void observe(const ObservationContext& ctx) override;

    const std::vector<double>& values() const { return values_; }
    double value(std::size_t k = 0) const;

    // False when the global state was mixed, so the figure is the reduced
    // state's entropy rather than an entanglement entropy.
    bool is_entanglement(std::size_t k = 0) const;

private:
    std::vector<int> region_;
    double order_ = 1.0;
    std::vector<double> values_;
    std::vector<bool> entanglement_;
};

// -----------------------------------------------------------------------------
// CallbackObserver - the shortest path to something bespoke
// -----------------------------------------------------------------------------
// For a quantity wanted once, where a class would be ceremony. Anything
// durable is better as its own Observer, which can hold state and name itself
// in a refusal message.

class CallbackObserver : public Observer {
public:
    using Fn = std::function<void(const ObservationContext&)>;

    explicit CallbackObserver(Fn fn);

    void observe(const ObservationContext& ctx) override;

private:
    Fn fn_;
};

}  // namespace lindblad
