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
    KrausChannel ch;
    ch.n_qubits = n_qubits;

    if (n_qubits == 1) {
        double sqrt_1mp = std::sqrt(1.0 - p);
        double sqrt_p3 = std::sqrt(p / 3.0);

        // K0 = sqrt(1-p) * I
        ch.operators.push_back({
            Complex128(sqrt_1mp, 0.0), Complex128(0.0, 0.0),
            Complex128(0.0, 0.0),      Complex128(sqrt_1mp, 0.0)
        });
        // K1 = sqrt(p/3) * X
        ch.operators.push_back({
            Complex128(0.0, 0.0),      Complex128(sqrt_p3, 0.0),
            Complex128(sqrt_p3, 0.0),  Complex128(0.0, 0.0)
        });
        // K2 = sqrt(p/3) * Y
        ch.operators.push_back({
            Complex128(0.0, 0.0),       Complex128(0.0, -sqrt_p3),
            Complex128(0.0, sqrt_p3),   Complex128(0.0, 0.0)
        });
        // K3 = sqrt(p/3) * Z
        ch.operators.push_back({
            Complex128(sqrt_p3, 0.0),   Complex128(0.0, 0.0),
            Complex128(0.0, 0.0),       Complex128(-sqrt_p3, 0.0)
        });
    } else if (n_qubits == 2) {
        // 2-qubit depolarizing: 16 Pauli terms
        size_t dim = 4;
        double sqrt_1mp = std::sqrt(1.0 - p);
        double sqrt_p15 = std::sqrt(p / 15.0);

        // K0 = sqrt(1-p) * I⊗I
        std::vector<Complex128> I4(dim * dim, Complex128(0.0, 0.0));
        for (size_t i = 0; i < dim; ++i) I4[i * dim + i] = Complex128(sqrt_1mp, 0.0);
        ch.operators.push_back(I4);

        // All 15 non-identity 2-qubit Paulis
        // P ∈ {I,X,Y,Z} ⊗ {I,X,Y,Z} minus I⊗I
        std::vector<std::vector<Complex128>> paulis_1q = {
            // I
            {Complex128(1,0), Complex128(0,0), Complex128(0,0), Complex128(1,0)},
            // X
            {Complex128(0,0), Complex128(1,0), Complex128(1,0), Complex128(0,0)},
            // Y
            {Complex128(0,0), Complex128(0,-1), Complex128(0,1), Complex128(0,0)},
            // Z
            {Complex128(1,0), Complex128(0,0), Complex128(0,0), Complex128(-1,0)}
        };

        for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
                if (a == 0 && b == 0) continue;  // skip I⊗I

                // Tensor product: (Pa ⊗ Pb) scaled by sqrt(p/15)
                std::vector<Complex128> K(dim * dim, Complex128(0.0, 0.0));
                for (size_t i = 0; i < 2; ++i) {
                    for (size_t j = 0; j < 2; ++j) {
                        for (size_t k = 0; k < 2; ++k) {
                            for (size_t l = 0; l < 2; ++l) {
                                K[(i*2+k) * dim + (j*2+l)] =
                                    paulis_1q[a][i*2+j] * paulis_1q[b][k*2+l] * sqrt_p15;
                            }
                        }
                    }
                }
                ch.operators.push_back(K);
            }
        }
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

    // Fractional excess dephasing: rate = 1/T2 - 1/(2*T1)
    double pure_dephasing_rate = std::max(0.0, 1.0 / T2 - 0.5 / T1);
    double e_phi = std::exp(-gate_time * pure_dephasing_rate);  // pure dephasing factor

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
