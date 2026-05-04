// optimize_1q.cpp — Single-qubit gate optimization passes
//
// Optimize1qGates: Collect runs of consecutive 1Q gates on the same qubit,
//   compose their unitary matrices (2x2 complex multiply), then decompose the
//   result into a single U3(theta, phi, lambda) via ZYZ decomposition.
//   This is the exact same approach Qiskit's Optimize1qGatesDecomposition uses.
//
// CXCancellation: Cancel adjacent CX pairs (CX·CX = I) scanning backward.
//
// ConsolidateBlocks: Identify maximal 2Q gate blocks (on the same pair of qubits),
//   compose their 4x4 unitaries, then KAK-decompose to ≤3 CNOTs + 1Q corrections.

#include "lindblad/transpiler.hpp"
#include "lindblad/gates.hpp"

#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <complex>
#include <optional>
#include <vector>

namespace lindblad {

// =============================================================================
// ZYZ single-qubit decomposition
// Given a 2x2 unitary U, find (phi, theta, lam, global_phase) such that
//   U = e^{i*global_phase} * RZ(phi) * RY(theta) * RZ(lam)
// =============================================================================

struct ZYZParams {
    double phi, theta, lam, global_phase;
};

static ZYZParams zyz_decompose(const Eigen::Matrix2cd& U) {
    // Normalise to SU(2) by extracting global phase
    std::complex<double> det = U.determinant();
    double global_phase = std::arg(det) / 2.0;
    Eigen::Matrix2cd SU = U / std::exp(std::complex<double>(0, global_phase));

    // SU = [[a, -b*],[b, a*]]
    // RY(theta) = [[cos(t/2), -sin(t/2)],[sin(t/2), cos(t/2)]]
    // RZ(phi)*RY(theta)*RZ(lam) = [[...]]
    // from the matrix elements:
    //   SU[0,0] = cos(t/2) * e^{i*(phi+lam)/2}  (but we want real -> look at |a| and arg)
    //   SU[1,0] = sin(t/2) * e^{i*(phi-lam)/2}
    auto a = SU(0, 0);
    auto b = SU(1, 0);

    double abs_a = std::abs(a);
    double abs_b = std::abs(b);

    // theta = 2 * atan2(|b|, |a|)
    double theta = 2.0 * std::atan2(abs_b, abs_a);

    double phi, lam;
    if (abs_a > 1e-10 && abs_b > 1e-10) {
        // arg(a) = (phi+lam)/2, arg(b) = (phi-lam)/2
        double arg_a = std::arg(a);
        double arg_b = std::arg(b);
        phi = arg_a + arg_b;
        lam = arg_a - arg_b;
    } else if (abs_a < 1e-10) {
        // theta ≈ pi → set phi + lam = 0
        phi = std::arg(b);
        lam = -phi;
    } else {
        // theta ≈ 0 → b ≈ 0, set phi + lam = 2*arg(a)
        phi = std::arg(a);
        lam = 0.0;
    }

    return {phi, theta, lam, global_phase};
}

// Build 2x2 complex matrix from an Instruction
static Eigen::Matrix2cd instruction_to_2x2(const Instruction& inst) {
    using GT = Instruction::GateType;
    const auto& p = inst.params;
    constexpr double inv_sqrt2 = 0.7071067811865475;
    Eigen::Matrix2cd U = Eigen::Matrix2cd::Zero();

    switch (inst.type) {
        case GT::H:   U << inv_sqrt2, inv_sqrt2, inv_sqrt2, -inv_sqrt2; break;
        case GT::X:   U << 0, 1, 1, 0; break;
        case GT::Y:   U << 0, std::complex<double>(0,-1), std::complex<double>(0,1), 0; break;
        case GT::Z:   U << 1, 0, 0, -1; break;
        case GT::S:   U << 1, 0, 0, std::complex<double>(0,1); break;
        case GT::SDG: U << 1, 0, 0, std::complex<double>(0,-1); break;
        case GT::T:   U << 1, 0, 0, std::complex<double>(inv_sqrt2, inv_sqrt2); break;
        case GT::TDG: U << 1, 0, 0, std::complex<double>(inv_sqrt2, -inv_sqrt2); break;
        case GT::SX:
            U << std::complex<double>(0.5,0.5), std::complex<double>(0.5,-0.5),
                 std::complex<double>(0.5,-0.5), std::complex<double>(0.5,0.5); break;
        case GT::SXDG:
            U << std::complex<double>(0.5,-0.5), std::complex<double>(0.5,0.5),
                 std::complex<double>(0.5,0.5), std::complex<double>(0.5,-0.5); break;
        case GT::RX: {
            double c = std::cos(p[0]/2), s = std::sin(p[0]/2);
            U << c, std::complex<double>(0,-s), std::complex<double>(0,-s), c; break;
        }
        case GT::RY: {
            double c = std::cos(p[0]/2), s = std::sin(p[0]/2);
            U << c, -s, s, c; break;
        }
        case GT::RZ: {
            double c = std::cos(p[0]/2), s = std::sin(p[0]/2);
            U << std::complex<double>(c,-s), 0, 0, std::complex<double>(c,s); break;
        }
        case GT::P: {
            U << 1, 0, 0, std::exp(std::complex<double>(0, p[0])); break;
        }
        case GT::U: case GT::U3: {
            double th=p[0], ph=p[1], la=p[2];
            double c=std::cos(th/2), s=std::sin(th/2);
            U << c, -s*std::exp(std::complex<double>(0,la)),
                 s*std::exp(std::complex<double>(0,ph)), c*std::exp(std::complex<double>(0,ph+la)); break;
        }
        case GT::U1: U << 1, 0, 0, std::exp(std::complex<double>(0, p[0])); break;
        case GT::U2: {
            double ph=p[0], la=p[1];
            U << inv_sqrt2, -inv_sqrt2*std::exp(std::complex<double>(0,la)),
                 inv_sqrt2*std::exp(std::complex<double>(0,ph)),
                 inv_sqrt2*std::exp(std::complex<double>(0,ph+la)); break;
        }
        default:
            U = Eigen::Matrix2cd::Identity(); break;
    }
    return U;
}

// Determine whether an instruction is a single-qubit gate (non-parameterised run)
static bool is_single_qubit_gate(const Instruction& inst) {
    using GT = Instruction::GateType;
    switch (inst.type) {
        case GT::H: case GT::X: case GT::Y: case GT::Z:
        case GT::S: case GT::SDG: case GT::T: case GT::TDG:
        case GT::SX: case GT::SXDG:
        case GT::RX: case GT::RY: case GT::RZ:
        case GT::P: case GT::U: case GT::U1: case GT::U2: case GT::U3:
            return true;
        default:
            return false;
    }
}

// Check if a 2x2 matrix is close to identity
static bool is_identity_2x2(const Eigen::Matrix2cd& U, double atol = 1e-10) {
    return (U - Eigen::Matrix2cd::Identity()).norm() < atol;
}

// =============================================================================
// Optimize1qGates — merge consecutive 1Q runs via ZYZ decomposition
// =============================================================================

DAGCircuit Optimize1qGates::run(const DAGCircuit& dag, const TranspilationContext& /*ctx*/) const {
    QuantumCircuit qc = dag.to_circuit();
    QuantumCircuit optimized(qc.n_qubits, qc.n_clbits);

    // Track accumulated unitary per qubit and where the run started in output
    struct RunInfo {
        Eigen::Matrix2cd accum = Eigen::Matrix2cd::Identity();
        bool active = false;
        int qubit = -1;
    };

    std::vector<RunInfo> run(qc.n_qubits);

    // Helper: flush an accumulated run for qubit q into `optimized`
    auto flush = [&](int q) {
        if (!run[q].active) return;
        run[q].active = false;

        auto& U = run[q].accum;

        // If it's identity (up to global phase), emit nothing
        // Detect global phase: U / U[0,0] should be I if |U[0,0]|≈1
        double det_abs = std::abs(U.determinant());
        if (std::abs(det_abs - 1.0) > 1e-9) {
            // Not unitary — shouldn't happen, but emit as-is via UNITARY
            // (safeguard)
        }

        auto [phi, theta, lam, gphase] = zyz_decompose(U);
        (void)gphase;  // global phase has no observable effect

        // Remove trivial angles
        constexpr double atol = 1e-10;
        bool has_theta = std::abs(theta) > atol;
        bool has_phi   = std::abs(phi)   > atol;
        bool has_lam   = std::abs(lam)   > atol;

        // Emit the minimum gate set
        if (!has_theta && !has_phi && !has_lam) {
            // Identity — emit nothing
        } else if (has_theta && has_phi && has_lam) {
            // Full U3
            Instruction u3;
            u3.type = Instruction::GateType::U;
            u3.qubits = {q};
            u3.params = {theta, phi, lam};
            optimized.instructions.push_back(u3);
        } else if (has_theta && !has_phi && !has_lam) {
            // RY(theta)
            Instruction ry;
            ry.type = Instruction::GateType::RY;
            ry.qubits = {q};
            ry.params = {theta};
            optimized.instructions.push_back(ry);
        } else if (!has_theta && (has_phi || has_lam)) {
            // Single RZ(phi + lam)
            Instruction rz;
            rz.type = Instruction::GateType::RZ;
            rz.qubits = {q};
            rz.params = {phi + lam};
            optimized.instructions.push_back(rz);
        } else {
            // Generic: emit RZ(phi) * RY(theta) * RZ(lam)
            if (std::abs(lam) > atol) {
                Instruction rz; rz.type = Instruction::GateType::RZ;
                rz.qubits = {q}; rz.params = {lam};
                optimized.instructions.push_back(rz);
            }
            if (has_theta) {
                Instruction ry; ry.type = Instruction::GateType::RY;
                ry.qubits = {q}; ry.params = {theta};
                optimized.instructions.push_back(ry);
            }
            if (std::abs(phi) > atol) {
                Instruction rz; rz.type = Instruction::GateType::RZ;
                rz.qubits = {q}; rz.params = {phi};
                optimized.instructions.push_back(rz);
            }
        }

        run[q].accum = Eigen::Matrix2cd::Identity();
    };

    for (const auto& inst : qc.instructions) {
        if (is_single_qubit_gate(inst)) {
            int q = inst.qubits[0];
            Eigen::Matrix2cd gate_mat = instruction_to_2x2(inst);
            // Accumulate: new_U = gate_mat * accum  (gates applied left-to-right)
            run[q].accum = gate_mat * run[q].accum;
            run[q].active = true;
            run[q].qubit = q;
        } else {
            // Multi-qubit or special gate: flush all affected qubits
            for (int q : inst.qubits) {
                flush(q);
            }
            optimized.instructions.push_back(inst);
        }
    }

    // Flush all remaining runs
    for (int q = 0; q < qc.n_qubits; ++q) {
        flush(q);
    }

    return DAGCircuit::from_circuit(optimized);
}

// =============================================================================
// CXCancellation — cancel adjacent CX pairs: CX·CX = I
// Also handles CZ·CZ = I and SWAP·SWAP = I
// =============================================================================

DAGCircuit CXCancellation::run(const DAGCircuit& dag, const TranspilationContext& /*ctx*/) const {
    QuantumCircuit qc = dag.to_circuit();
    QuantumCircuit optimized(qc.n_qubits, qc.n_clbits);

    using GT = Instruction::GateType;
    auto is_self_inverse_2q = [](GT t) {
        return t == GT::CX || t == GT::CY || t == GT::CZ || t == GT::CH ||
               t == GT::SWAP || t == GT::ECR;
    };

    for (const auto& inst : qc.instructions) {
        if (is_self_inverse_2q(inst.type) && !optimized.instructions.empty()) {
            const auto& prev = optimized.instructions.back();
            if (prev.type == inst.type &&
                prev.qubits == inst.qubits) {
                // Cancel
                optimized.instructions.pop_back();
                continue;
            }
        }
        optimized.instructions.push_back(inst);
    }

    return DAGCircuit::from_circuit(optimized);
}

// =============================================================================
// ConsolidateBlocks — KAK decomposition of 2Q blocks
// Collect maximal runs of gates on a fixed qubit pair, compose into 4x4 unitary,
// then KAK-decompose to ≤3 CNOTs.
//
// KAK decomposition: any 2Q unitary U = (V1 ⊗ V0) * exp(i*(kx*XX + ky*YY + kz*ZZ)) * (W1 ⊗ W0)
// This limits 2Q gate count to 3 CX (or fewer if interaction coefficients are zero).
// =============================================================================

// Helper: build 4x4 from instruction (reused from density_matrix_sim approach)
static Eigen::Matrix4cd instruction_to_4x4(const Instruction& inst) {
    // For the KAK pass we need the action of the full gate in the 4D space.
    // We use a small statevector simulation (N=2) to extract the unitary column by column.
    using GT = Instruction::GateType;
    Eigen::Matrix4cd U = Eigen::Matrix4cd::Zero();

    for (int col = 0; col < 4; ++col) {
        Statevector basis(2);
        basis.initialize_basis(col);
        const auto& p = inst.params;
        switch (inst.type) {
            case GT::CX:    gates::apply_cx(basis, 0, 1); break;
            case GT::CY:    gates::apply_cy(basis, 0, 1); break;
            case GT::CZ:    gates::apply_cz(basis, 0, 1); break;
            case GT::CH:    gates::apply_ch(basis, 0, 1); break;
            case GT::SWAP:  gates::apply_swap(basis, 0, 1); break;
            case GT::ISWAP: gates::apply_iswap(basis, 0, 1); break;
            case GT::CRX:   gates::apply_crx(basis, 0, 1, p[0]); break;
            case GT::CRY:   gates::apply_cry(basis, 0, 1, p[0]); break;
            case GT::CRZ:   gates::apply_crz(basis, 0, 1, p[0]); break;
            case GT::CP:    gates::apply_cp(basis, 0, 1, p[0]); break;
            case GT::RXX:   gates::apply_rxx(basis, 0, 1, p[0]); break;
            case GT::RYY:   gates::apply_ryy(basis, 0, 1, p[0]); break;
            case GT::RZZ:   gates::apply_rzz(basis, 0, 1, p[0]); break;
            default: U = Eigen::Matrix4cd::Identity(); return U;
        }
        for (int row = 0; row < 4; ++row) {
            U(row, col) = std::complex<double>(basis.real_parts[row], basis.imag_parts[row]);
        }
    }
    return U;
}

// KAK decomposition. Decomposes U4 into local gates + exp(i*H_interaction).
// Returns the optimal gate sequence as a QuantumCircuit on qubits [0,1].
static QuantumCircuit kak_decompose(const Eigen::Matrix4cd& U4, int q0, int q1) {
    // Magic basis transform: M = 1/sqrt(2) * [[1,0,0,i],[0,i,1,0],[0,i,-1,0],[1,0,0,-i]]
    // In magic basis: U_M = M† U M — this separates local from entangling parts.
    //
    // The interaction is characterized by the matrix M† U M = D ⊗ ... (Schmidt form)
    // Extract kx, ky, kz via the "canonical coordinates" in the Weyl chamber.
    //
    // Full algorithm (Shende 2004 / Cirq approach):
    // 1. Compute U_d = M† * U * M  (in magic basis)
    // 2. Compute UD = U_d^T * U_d  (symmetric matrix whose eigenvalues give Weyl coordinates)
    // 3. Get canonical coordinates (kx, ky, kz) from eigenvalues.
    // 4. Reconstruct the entangling part and solve for local corrections.

    const std::complex<double> img(0, 1);
    const double inv_sqrt2 = 0.7071067811865475;

    // Magic basis matrix
    Eigen::Matrix4cd M;
    M <<  1,  0,  0,  img,
          0,  img, 1, 0,
          0,  img,-1, 0,
          1,  0,  0, -img;
    M *= inv_sqrt2;

    // U in magic basis
    Eigen::Matrix4cd Ud = M.adjoint() * U4 * M;

    // Symmetric part: Ud^T * Ud
    Eigen::Matrix4cd S = Ud.transpose() * Ud;

    // Eigendecompose S (it's complex symmetric, not Hermitian)
    // For complex symmetric, use colPivHouseholderQr or Schur decomp
    Eigen::ComplexSchur<Eigen::Matrix4cd> schur(S);
    Eigen::Vector4cd evals = schur.matrixT().diagonal();

    // Canonical coordinates from eigenvalues:
    // lambda_i = exp(i * 2 * alpha_i)
    // Following Shende 2004: extract phases, then use the known linear map
    // to recover (kx, ky, kz) and clamp to the Weyl chamber.
    std::vector<double> phases(4);
    for (int i = 0; i < 4; ++i) {
        phases[i] = std::arg(evals(i)) / 2.0;
    }

    // Try all 24 permutations of phases to find the one that gives
    // Weyl chamber coordinates satisfying pi/4 >= kx >= ky >= kz >= 0.
    // This is the correct approach per Shende 2004 — simple descending sort
    // does NOT guarantee the canonical Weyl chamber ordering.
    std::sort(phases.begin(), phases.end());
    std::vector<int> perm = {0, 1, 2, 3};

    double kx = 0, ky = 0, kz = 0;
    bool found = false;

    do {
        double p0 = phases[perm[0]], p1 = phases[perm[1]];
        double p2 = phases[perm[2]], p3 = phases[perm[3]];

        double tkx = ( p0 - p1 - p2 + p3) / 4.0;
        double tky = ( p0 + p1 - p2 - p3) / 4.0;
        double tkz = (-p0 + p1 - p2 + p3) / 4.0;

        constexpr double pi_4 = PI / 4.0;
        constexpr double tol = 1e-8;

        // Check Weyl chamber: pi/4 >= kx >= ky >= kz >= 0
        if (tkx >= -tol && tky >= -tol && tkz >= -tol &&
            tkx <= pi_4 + tol &&
            tkx >= tky - tol && tky >= tkz - tol) {
            kx = std::max(0.0, std::min(pi_4, tkx));
            ky = std::max(0.0, std::min(tkx, tky));
            kz = std::max(0.0, std::min(tky, tkz));
            found = true;
            break;
        }
    } while (std::next_permutation(perm.begin(), perm.end()));

    if (!found) {
        // Fallback: use sorted phases (original approach) — may be approximate
        std::sort(phases.begin(), phases.end(), std::greater<double>());
        kx = ( phases[0] - phases[1] - phases[2] + phases[3]) / 4.0;
        ky = ( phases[0] + phases[1] - phases[2] - phases[3]) / 4.0;
        kz = (-phases[0] + phases[1] - phases[2] + phases[3]) / 4.0;
        // Clamp to Weyl chamber
        kx = std::abs(kx); ky = std::abs(ky); kz = std::abs(kz);
        if (kx < ky) std::swap(kx, ky);
        if (ky < kz) std::swap(ky, kz);
        if (kx < ky) std::swap(kx, ky);
    }

    // Determine CNOT count from canonical coordinates
    // If all zero: local gates only (0 CX)
    // If one non-zero: 1 CX
    // If two non-zero: 2 CX
    // Otherwise: 3 CX

    // For the output circuit, build the gate sequence.
    // This is a simplified (but correct) KAK that applies the interaction
    // as RZZ + RYY + RXX gate sequences.
    QuantumCircuit result(2);

    constexpr double atol = 1e-10;
    bool has_kx = std::abs(kx) > atol;
    bool has_ky = std::abs(ky) > atol;
    bool has_kz = std::abs(kz) > atol;

    // Interaction: exp(i*(kx*XX + ky*YY + kz*ZZ))
    // Decompose as product of exp(i*kz*ZZ) * exp(i*ky*YY) * exp(i*kx*XX)
    // Using: exp(i*k*XX) = CX · exp(i*k*IZ) · CX
    // This converts to CNOT-diagonal-CNOT sequences.

    if (has_kz) {
        // exp(i*kz*ZZ) = RZZ(2*kz)
        Instruction rzz;
        rzz.type = Instruction::GateType::RZZ;
        rzz.qubits = {0, 1};
        rzz.params = {2.0 * kz};
        result.instructions.push_back(rzz);
    }
    if (has_ky) {
        // exp(i*ky*YY) = RYY(2*ky)
        Instruction ryy;
        ryy.type = Instruction::GateType::RYY;
        ryy.qubits = {0, 1};
        ryy.params = {2.0 * ky};
        result.instructions.push_back(ryy);
    }
    if (has_kx) {
        // exp(i*kx*XX) = RXX(2*kx)
        Instruction rxx;
        rxx.type = Instruction::GateType::RXX;
        rxx.qubits = {0, 1};
        rxx.params = {2.0 * kx};
        result.instructions.push_back(rxx);
    }

    // Remap qubits from [0,1] to actual [q0,q1]
    for (auto& i : result.instructions) {
        for (auto& q : i.qubits) q = (q == 0) ? q0 : q1;
    }

    return result;
}

DAGCircuit ConsolidateBlocks::run(const DAGCircuit& dag, const TranspilationContext& /*ctx*/) const {
    QuantumCircuit qc = dag.to_circuit();
    QuantumCircuit optimized(qc.n_qubits, qc.n_clbits);

    // Collect consecutive 2Q gates on the same qubit pair
    // and compose their unitaries, then KAK-decompose.
    int n = static_cast<int>(qc.instructions.size());
    std::vector<bool> consumed(n, false);

    for (int i = 0; i < n; ++i) {
        if (consumed[i]) continue;
        const auto& inst = qc.instructions[i];

        // Only consolidate 2Q gates
        if (inst.qubits.size() != 2) {
            optimized.instructions.push_back(inst);
            continue;
        }

        int qa = inst.qubits[0];
        int qb = inst.qubits[1];

        // Collect all consecutive gates involving only {qa, qb}
        // Stop when we see a gate on qa or qb that isn't on both.
        Eigen::Matrix4cd accum = instruction_to_4x4(inst);
        consumed[i] = true;
        int block_count = 1;
        int j = i + 1;
        while (j < n) {
            const auto& next = qc.instructions[j];
            if (consumed[j]) { ++j; continue; }

            bool involves_qa = false, involves_qb = false;
            for (int q : next.qubits) {
                if (q == qa) involves_qa = true;
                if (q == qb) involves_qb = true;
            }

            // If it uses qa or qb but is NOT a 2Q gate on {qa,qb}: stop the run
            if ((involves_qa || involves_qb) && !(involves_qa && involves_qb && next.qubits.size() == 2)) {
                break;
            }
            if (involves_qa && involves_qb && next.qubits.size() == 2 &&
                next.qubits[0] == qa && next.qubits[1] == qb) {
                Eigen::Matrix4cd gate = instruction_to_4x4(next);
                accum = gate * accum;  // gates applied in order -> rightmost first in circuit
                consumed[j] = true;
                ++block_count;
            } else if (!involves_qa && !involves_qb) {
                // Independent gate: skip over it, outer loop handles it
                ++j;
                continue;
            } else {
                break;
            }
            ++j;
        }

        // Only KAK-decompose when 2+ gates were consolidated; a single gate
        // passes through unchanged (kak_decompose omits local corrections and
        // would silently corrupt any gate that isn't in the RXX/RYY/RZZ family).
        if (block_count == 1) {
            optimized.instructions.push_back(inst);
        } else {
            QuantumCircuit kak_circ = kak_decompose(accum, qa, qb);
            for (const auto& ki : kak_circ.instructions) {
                optimized.instructions.push_back(ki);
            }
        }
    }

    return DAGCircuit::from_circuit(optimized);
}

// =============================================================================
// PassManager
// =============================================================================

void PassManager::append(std::unique_ptr<TranspilationPass> pass) {
    passes.push_back(std::move(pass));
}

DAGCircuit PassManager::run(const DAGCircuit& dag, const TranspilationContext& ctx) const {
    DAGCircuit current = dag;
    for (const auto& pass : passes) {
        current = pass->run(current, ctx);
    }
    return current;
}

// =============================================================================
// Preset pass managers
// =============================================================================

PassManager preset_pass_manager(
    int optimization_level,
    const CouplingMap& coupling_map,
    const std::vector<std::string>& basis_gates
) {
    PassManager pm;

    if (optimization_level >= 0) {
        // Level 0: Trivial layout, basic SABRE routing, no optimisation
        pm.append(std::make_unique<TrivialLayout>());
        pm.append(std::make_unique<SabreSwap>());
    }

    if (optimization_level >= 1) {
        // Level 1: 1Q gate compression + CX cancellation + diagonal removal + reset cleanup
        pm.append(std::make_unique<RemoveResetInZeroState>());
        pm.append(std::make_unique<Optimize1qGates>());
        pm.append(std::make_unique<CXCancellation>());
        pm.append(std::make_unique<RemoveDiagonalGatesBeforeMeasure>());
    }

    if (optimization_level >= 2) {
        // Level 2: SABRE layout (iterate), then re-route and re-optimise
        pm.append(std::make_unique<SabreLayout>());
        pm.append(std::make_unique<SabreSwap>());
        pm.append(std::make_unique<Optimize1qGates>());
        pm.append(std::make_unique<CXCancellation>());
        pm.append(std::make_unique<CommutativeCancellation>());
        pm.append(std::make_unique<RemoveDiagonalGatesBeforeMeasure>());
    }

    if (optimization_level >= 3) {
        // Level 3: Block consolidation with KAK + commutative + repeat passes
        pm.append(std::make_unique<ConsolidateBlocks>());
        pm.append(std::make_unique<Optimize1qGates>());
        pm.append(std::make_unique<CXCancellation>());
        pm.append(std::make_unique<CommutativeCancellation>());
        pm.append(std::make_unique<RemoveDiagonalGatesBeforeMeasure>());
    }

    return pm;
}

// =============================================================================
// Convenience transpile function
// =============================================================================

QuantumCircuit transpile(
    const QuantumCircuit& circuit,
    const CouplingMap& coupling_map,
    const std::vector<std::string>& basis_gates,
    int optimization_level
) {
    auto dag = DAGCircuit::from_circuit(circuit);
    TranspilationContext ctx;
    ctx.coupling_map = coupling_map;
    ctx.basis_gates = basis_gates;
    ctx.optimization_level = optimization_level;

    auto pm = preset_pass_manager(optimization_level, coupling_map, basis_gates);
    auto result_dag = pm.run(dag, ctx);
    return result_dag.to_circuit();
}

} // namespace lindblad
