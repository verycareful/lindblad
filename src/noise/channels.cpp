#include "lindblad/noise.hpp"

#include <cmath>
#include <stdexcept>

namespace lindblad {
namespace NoiseChannels {

// =============================================================================
// Depolarizing channel
// With probability p, applies a random Pauli error
// =============================================================================

KrausChannel depolarizing(double p, int n_qubits) {
    // General n-qubit Pauli twirl:
    //   E(rho) = (1-p) rho + p/(4^n - 1) * sum_{P != I} P rho P
    // K_0 = sqrt(1-p) I, K_P = sqrt(p/(4^n-1)) P for each non-identity Pauli.
    //
    // The hard cap protects against the 4^n operator count: n = 6 already
    // means 4095 Kraus matrices of dimension 64. Larger registers should
    // model depolarisation with per-qubit channels instead.
    if (n_qubits < 1 || n_qubits > 6)
        throw std::invalid_argument(
            "depolarizing: n_qubits must be in [1, 6] (got " +
            std::to_string(n_qubits) + "); compose per-qubit channels for "
            "wider registers");
    if (p < 0.0 || p > 1.0)
        throw std::invalid_argument("depolarizing: p must be in [0, 1]");

    KrausChannel ch;
    ch.n_qubits = n_qubits;

    const size_t dim = 1ULL << n_qubits;
    const size_t n_paulis = 1ULL << (2 * n_qubits);  // 4^n
    const double w_id  = std::sqrt(1.0 - p);
    const double w_err = std::sqrt(p / static_cast<double>(n_paulis - 1));

    // i^k lookup for the Y = iXZ phase factor.
    static const Complex128 I_POW[4] = {
        Complex128(1.0, 0.0), Complex128(0.0, 1.0),
        Complex128(-1.0, 0.0), Complex128(0.0, -1.0)
    };

    // Pauli index m encodes two bits per qubit: bit (2q) = X component,
    // bit (2q+1) = Z component for qubit q (both set = Y). Each Pauli has
    // exactly one non-zero entry per column j, at row j ^ x_mask, with phase
    // i^{#Y} * (-1)^{popcount(j & z_mask)}. Qubit q maps to index bit q
    // (project LSB convention); the channel as a whole is permutation
    // symmetric, so this choice is also convention-safe for apply_kraus.
    for (size_t m = 0; m < n_paulis; ++m) {
        const double w = (m == 0) ? w_id : w_err;
        if (w == 0.0) continue;  // p == 0 (skip error terms) or p == 1 (skip K0)

        uint64_t x_mask = 0, z_mask = 0;
        int y_count = 0;
        for (int q = 0; q < n_qubits; ++q) {
            const bool xb = (m >> (2 * q)) & 1ULL;
            const bool zb = (m >> (2 * q + 1)) & 1ULL;
            if (xb) x_mask |= (1ULL << q);
            if (zb) z_mask |= (1ULL << q);
            if (xb && zb) ++y_count;
        }
        const Complex128 yphase = I_POW[y_count & 3] * w;

        std::vector<Complex128> K(dim * dim, Complex128(0.0, 0.0));
        for (size_t j = 0; j < dim; ++j) {
            const size_t row = j ^ x_mask;
            Complex128 v = yphase;
            if (LINDBLAD_POPCOUNT64(j & z_mask) & 1) v = -v;
            K[row * dim + j] = v;
        }
        ch.operators.push_back(std::move(K));
    }

    return ch;
}

// =============================================================================
// Amplitude damping: T1 relaxation
// =============================================================================

KrausChannel amplitude_damping(double gamma) {
    KrausChannel ch;
    ch.n_qubits = 1;

    // K0 = [[1, 0], [0, sqrt(1-gamma)]]
    ch.operators.push_back({
        Complex128(1.0, 0.0),                    Complex128(0.0, 0.0),
        Complex128(0.0, 0.0),                    Complex128(std::sqrt(1.0 - gamma), 0.0)
    });

    // K1 = [[0, sqrt(gamma)], [0, 0]]
    ch.operators.push_back({
        Complex128(0.0, 0.0),                    Complex128(std::sqrt(gamma), 0.0),
        Complex128(0.0, 0.0),                    Complex128(0.0, 0.0)
    });

    return ch;
}

// =============================================================================
// Phase damping
// =============================================================================

KrausChannel phase_damping(double lambda) {
    KrausChannel ch;
    ch.n_qubits = 1;

    // K0 = [[1, 0], [0, sqrt(1-lambda)]]
    ch.operators.push_back({
        Complex128(1.0, 0.0),                      Complex128(0.0, 0.0),
        Complex128(0.0, 0.0),                      Complex128(std::sqrt(1.0 - lambda), 0.0)
    });

    // K1 = [[0, 0], [0, sqrt(lambda)]]
    ch.operators.push_back({
        Complex128(0.0, 0.0),                      Complex128(0.0, 0.0),
        Complex128(0.0, 0.0),                      Complex128(std::sqrt(lambda), 0.0)
    });

    return ch;
}

// =============================================================================
// Thermal relaxation
// =============================================================================

KrausChannel thermal_relaxation(
    double T1, double T2, double gate_time,
    double excited_state_population
) {
    if (T2 > 2.0 * T1) {
        throw std::invalid_argument("T2 must be <= 2*T1 for physical noise");
    }
    if (gate_time < 0.0 || T1 <= 0.0 || T2 <= 0.0) {
        throw std::invalid_argument("T1, T2, gate_time must be positive");
    }

    // Correct Kraus operators for T1/T2 thermal relaxation (generalized)
    // Following Nielsen & Chuang + Qiskit Aer approach.
    //
    // p1  = excited_state_population (thermal equilibrium population of |1⟩)
    // p0  = 1 - p1
    // e1  = exp(-gate_time / T1)  (amplitude decay factor)
    // e2  = exp(-gate_time / T2)  (decoherence factor)
    //
    // The 4 Kraus operators for non-zero excited state population:
    // K0 = [[sqrt(1-p1)*sqrt(e1*(1+e1)/... }], ...]]  ← computed from the generator
    //
    // Standard parametrisation (trace-preserving set):
    // Let:
    //   a = exp(-gate_time / T1)  (excited state decay)
    //   b = exp(-gate_time / T2)  (total dephasing)
    //   e2_factor = exp(-(1/T2 - 1/(2*T1)) * gate_time)  (pure dephasing factor)
    //
    // For p1 = 0 (ground state at T=0):
    // K0 = [[1, 0], [0, sqrt(a)]]
    // K1 = [[0, sqrt(1-a)], [0, 0]]
    // K2 = [[0, 0], [0, sqrt((1-a)*(1-e2_factor))]]  -- if T2 < 2T1
    //
    // For p1 > 0 (finite temperature), we use the generalized amplitude damping model:

    double e1 = std::exp(-gate_time / T1);          // |1> decay amplitude squared
    double p1 = std::max(0.0, std::min(1.0, excited_state_population));
    double p0 = 1.0 - p1;

    // Fractional excess dephasing: rate = 1/T2 - 1/(2*T1).
    //
    // The Kraus operators below carry sqrt(e1 * e_phi) on the off-diagonal,
    // so the channel's coherence factor is sqrt(e1) * sqrt(e_phi)
    //   = exp(-t/(2*T1)) * sqrt(e_phi).
    // For the defining transverse decay exp(-t/T2) this requires
    //   e_phi = exp(-2 * t * (1/T2 - 1/(2*T1))).
    // The factor 2 is load-bearing: without it the channel dephases at half
    // the requested rate for every T2 < 2*T1, and only the T2 = 2*T1 boundary
    // comes out correct.
    double pure_dephasing_rate = std::max(0.0, 1.0 / T2 - 0.5 / T1);
    double e_phi = std::exp(-2.0 * gate_time * pure_dephasing_rate);

    KrausChannel ch;
    ch.n_qubits = 1;

    // Generalized amplitude damping + dephasing
    // The 4-operator set (valid for any T1, T2 with T2 >= T1/2 and p1 in [0,1]):
    //
    // sqrt factors
    double sqrt_e1   = std::sqrt(e1);
    double sqrt_1me1 = std::sqrt(1.0 - e1);
    double sqrt_e_phi= std::sqrt(e_phi);

    // K0 = sqrt(p0) * [[1, 0], [0, sqrt(e1) * sqrt(e_phi)]]
    // K1 = sqrt(p0) * [[0, sqrt(1-e1)], [0, 0]]
    // K2 = sqrt(p1) * [[sqrt(e1) * sqrt(e_phi), 0], [0, 1]]
    // K3 = sqrt(p1) * [[0, 0], [sqrt(1-e1), 0]]
    // K4 (if T2 < 2*T1): dephasing-only operator for both polarisations
    //    K4 = sqrt(p0) * [[0, 0], [0, sqrt(e1*(1-e_phi))]]
    //    K5 = sqrt(p1) * [[sqrt(e1*(1-e_phi)), 0], [0, 0]]

    double sqrt_p0 = std::sqrt(p0);
    double sqrt_p1 = std::sqrt(p1);
    double off_diag = sqrt_e1 * sqrt_e_phi;

    // K0: decay of |1> + partial dephasing (staying in |0>)
    ch.operators.push_back({
        Complex128(sqrt_p0, 0.0),       Complex128(0.0, 0.0),
        Complex128(0.0, 0.0),           Complex128(sqrt_p0 * off_diag, 0.0)
    });

    // K1: emission from |1> to |0> (T1 decay)
    ch.operators.push_back({
        Complex128(0.0, 0.0),           Complex128(sqrt_p0 * sqrt_1me1, 0.0),
        Complex128(0.0, 0.0),           Complex128(0.0, 0.0)
    });

    // K2: excitation from |0> to |1> (thermal absorption, only if p1 > 0)
    ch.operators.push_back({
        Complex128(sqrt_p1 * off_diag, 0.0),  Complex128(0.0, 0.0),
        Complex128(0.0, 0.0),                  Complex128(sqrt_p1, 0.0)
    });

    // K3: stimulated emission / absorption
    ch.operators.push_back({
        Complex128(0.0, 0.0),           Complex128(0.0, 0.0),
        Complex128(sqrt_p1 * sqrt_1me1, 0.0), Complex128(0.0, 0.0)
    });

    // K4, K5: pure dephasing contribution (non-zero when T2 < 2*T1)
    if (e_phi < 1.0 - 1e-15) {
        double sqrt_1me_phi_e1 = std::sqrt(std::max(0.0, e1 * (1.0 - e_phi)));
        // K4 (ground state, phase decoherence)
        ch.operators.push_back({
            Complex128(0.0, 0.0),           Complex128(0.0, 0.0),
            Complex128(0.0, 0.0),           Complex128(sqrt_p0 * sqrt_1me_phi_e1, 0.0)
        });
        // K5 (excited state, phase decoherence)
        ch.operators.push_back({
            Complex128(sqrt_p1 * sqrt_1me_phi_e1, 0.0), Complex128(0.0, 0.0),
            Complex128(0.0, 0.0),                        Complex128(0.0, 0.0)
        });
    }

    return ch;
}

// =============================================================================
// Pauli channel
// =============================================================================

KrausChannel pauli(double px, double py, double pz) {
    double pi_val = 1.0 - px - py - pz;
    if (pi_val < -1e-10) {
        throw std::invalid_argument("Pauli probabilities must sum to <= 1");
    }

    KrausChannel ch;
    ch.n_qubits = 1;

    ch.operators.push_back({
        Complex128(std::sqrt(std::max(0.0, pi_val)), 0.0), Complex128(0.0, 0.0),
        Complex128(0.0, 0.0),                               Complex128(std::sqrt(std::max(0.0, pi_val)), 0.0)
    });
    ch.operators.push_back({
        Complex128(0.0, 0.0),              Complex128(std::sqrt(px), 0.0),
        Complex128(std::sqrt(px), 0.0),    Complex128(0.0, 0.0)
    });
    ch.operators.push_back({
        Complex128(0.0, 0.0),               Complex128(0.0, -std::sqrt(py)),
        Complex128(0.0, std::sqrt(py)),     Complex128(0.0, 0.0)
    });
    ch.operators.push_back({
        Complex128(std::sqrt(pz), 0.0),    Complex128(0.0, 0.0),
        Complex128(0.0, 0.0),              Complex128(-std::sqrt(pz), 0.0)
    });

    return ch;
}

// =============================================================================
// Bit flip
// =============================================================================

KrausChannel bit_flip(double p) {
    return pauli(p, 0.0, 0.0);
}

// =============================================================================
// Phase flip
// =============================================================================

KrausChannel phase_flip(double p) {
    return pauli(0.0, 0.0, p);
}

// =============================================================================
// Bit-phase flip
// =============================================================================

KrausChannel bit_phase_flip(double p) {
    return pauli(0.0, p, 0.0);
}

// =============================================================================
// Reset error
// =============================================================================

KrausChannel reset(double p0, double p1) {
    KrausChannel ch;
    ch.n_qubits = 1;

    double pi_val = 1.0 - p0 - p1;

    // K0 = sqrt(1-p0-p1) * I
    ch.operators.push_back({
        Complex128(std::sqrt(std::max(0.0, pi_val)), 0.0), Complex128(0.0, 0.0),
        Complex128(0.0, 0.0),                               Complex128(std::sqrt(std::max(0.0, pi_val)), 0.0)
    });

    // K1 = sqrt(p0) * |0><0|
    ch.operators.push_back({
        Complex128(std::sqrt(p0), 0.0),  Complex128(0.0, 0.0),
        Complex128(0.0, 0.0),            Complex128(0.0, 0.0)
    });

    // K2 = sqrt(p0) * |0><1|
    ch.operators.push_back({
        Complex128(0.0, 0.0),            Complex128(std::sqrt(p0), 0.0),
        Complex128(0.0, 0.0),            Complex128(0.0, 0.0)
    });

    // K3 = sqrt(p1) * |1><0|
    ch.operators.push_back({
        Complex128(0.0, 0.0),            Complex128(0.0, 0.0),
        Complex128(std::sqrt(p1), 0.0),  Complex128(0.0, 0.0)
    });

    // K4 = sqrt(p1) * |1><1|
    ch.operators.push_back({
        Complex128(0.0, 0.0),            Complex128(0.0, 0.0),
        Complex128(0.0, 0.0),            Complex128(std::sqrt(p1), 0.0)
    });

    return ch;
}

// =============================================================================
// Coherent unitary error
// =============================================================================

KrausChannel coherent_unitary(double theta, double phi, double lambda) {
    KrausChannel ch;
    ch.n_qubits = 1;

    double cos_half = std::cos(theta / 2.0);
    double sin_half = std::sin(theta / 2.0);

    ch.operators.push_back({
        Complex128(cos_half, 0.0),
        Complex128(-sin_half * std::cos(lambda), -sin_half * std::sin(lambda)),
        Complex128(sin_half * std::cos(phi), sin_half * std::sin(phi)),
        Complex128(cos_half * std::cos(phi + lambda), cos_half * std::sin(phi + lambda))
    });

    return ch;
}

} // namespace NoiseChannels
} // namespace lindblad
