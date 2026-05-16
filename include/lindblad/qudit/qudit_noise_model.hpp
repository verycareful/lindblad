#pragma once
#include "lindblad/types.hpp"
#include <map>
#include <vector>

namespace lindblad {

// =============================================================================
// QuditLindbladOp — a single Lindblad jump operator with associated rate
// =============================================================================

struct QuditLindbladOp {
    std::vector<Complex128> L;  // d×d row-major Lindblad operator
    double rate;                // γ ≥ 0
};

// =============================================================================
// QuditKrausChannel — a completely positive trace-preserving map via Kraus ops
// =============================================================================

struct QuditKrausChannel {
    std::vector<std::vector<Complex128>> ops;  // list of d×d Kraus operators
};

// =============================================================================
// QuditQuditNoise — combined noise specification per qudit
// =============================================================================

struct QuditQuditNoise {
    QuditKrausChannel kraus;
    std::vector<QuditLindbladOp> lindblad;
};

// =============================================================================
// QuditNoiseModel — per-qudit noise specifications with factory methods
// =============================================================================

class QuditNoiseModel {
public:
    std::map<int, QuditQuditNoise> per_qudit;  // qudit index → noise spec

    // -------------------------------------------------------------------------
    // Factory methods — build common channels
    // -------------------------------------------------------------------------

    // Generalised depolarising channel for a d-level system.
    // d² Kraus operators: K_0 = sqrt(1-p)*I; K_{a,b} = sqrt(p/(d²-1))*X^a*Z^b
    // for (a,b) ≠ (0,0), with X = shift_matrix(d,1) and Z = clock gate.
    static QuditKrausChannel depolarizing_channel(int d, double p);

    // Amplitude damping for a d-level system.
    // Models decay from level j to j-1. Returns d Kraus operators.
    // gamma must satisfy (d-1)*gamma ≤ 1.
    static QuditKrausChannel amplitude_damping_channel(int d, double gamma);

    // Phase damping (dephasing) for a d-level system.
    // K_0 = sqrt(1-p)*I; K_j = sqrt(p)*|j><j| for j=0..d-1. Returns 1+d ops.
    static QuditKrausChannel phase_damping_channel(int d, double p);

    // Single Lindblad operator for amplitude damping:
    // L = sqrt(gamma) * lower-shift (L[j-1,j] = sqrt(j) for j=1..d-1), rate=1.
    static QuditLindbladOp amplitude_damping_lindblad(int d, double gamma);

    // d-1 Lindblad operators for dephasing (clock gate powers scaled by sqrt(gamma)).
    // L_j[k,k] = sqrt(gamma)*omega^{j*k} for j=1..d-1, rate=1 each.
    static std::vector<QuditLindbladOp> dephasing_lindblad(int d, double gamma);

    // -------------------------------------------------------------------------
    // Convenience: add a channel / ops to the model for a specific qudit
    // -------------------------------------------------------------------------

    void add_depolarizing(int q, int d, double p);
    void add_amplitude_damping(int q, int d, double gamma);
    void add_phase_damping(int q, int d, double p);
    void add_lindblad_op(int q, std::vector<Complex128> L, double rate);
};

} // namespace lindblad
