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
//
// PassManager, preset_pass_manager and transpile() live in
// preset_pass_manager.cpp: pipeline composition is not a 1-qubit-optimisation
// concern (SRP).

#include "lindblad/transpiler.hpp"
#include "lindblad/gates.hpp"
#include "lindblad/simulators/statevector_sim.hpp"

#include "../two_qubit_decompose.hpp"

#include <Eigen/Dense>

#include <algorithm>
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

    // With RZ(x) = diag(e^{-ix/2}, e^{ix/2}):
    //   RZ(phi)*RY(theta)*RZ(lam) =
    //     [[ cos(t/2)*e^{-i(phi+lam)/2}, -sin(t/2)*e^{-i(phi-lam)/2} ],
    //      [ sin(t/2)*e^{+i(phi-lam)/2},  cos(t/2)*e^{+i(phi+lam)/2} ]]
    // so from the matrix elements:
    //   arg(SU[0,0]) = -(phi+lam)/2      arg(SU[1,0]) = +(phi-lam)/2
    // The signs are load-bearing: arg(a) = +(phi+lam)/2 instead yields a
    // merged gate with the correct theta but wrong phi/lambda, so
    // transpilation silently changes circuit unitaries (issue #31).
    auto a = SU(0, 0);
    auto b = SU(1, 0);

    double abs_a = std::abs(a);
    double abs_b = std::abs(b);

    // theta = 2 * atan2(|b|, |a|)
    double theta = 2.0 * std::atan2(abs_b, abs_a);

    double phi, lam;
    if (abs_a > 1e-10 && abs_b > 1e-10) {
        // phi+lam = -2*arg(a), phi-lam = +2*arg(b)
        double arg_a = std::arg(a);
        double arg_b = std::arg(b);
        phi = arg_b - arg_a;
        lam = -(arg_a + arg_b);
    } else if (abs_a < 1e-10) {
        // theta ≈ pi: only phi-lam = 2*arg(b) is defined; set phi + lam = 0
        phi = std::arg(b);
        lam = -phi;
    } else {
        // theta ≈ 0: only phi+lam = -2*arg(a) is defined; set lam = 0
        phi = -2.0 * std::arg(a);
        lam = 0.0;
    }

    return {phi, theta, lam, global_phase};
}

// Build 2x2 complex matrix from an Instruction
static Eigen::Matrix2cd instruction_to_2x2(const Instruction& inst) {
    using GT = Instruction::GateType;
    const auto& p = inst.params;
    constexpr double inv_sqrt2 = INV_SQRT2;
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
        // Classically-conditioned gates depend on runtime state: they must
        // never be merged into an unconditional run (the merged rotation
        // would silently drop the condition). They break the run and pass
        // through untouched, like multi-qubit gates.
        if (is_single_qubit_gate(inst) && inst.condition_clbit < 0) {
            int q = inst.qubits[0];
            Eigen::Matrix2cd gate_mat = instruction_to_2x2(inst);
            // Accumulate: new_U = gate_mat * accum  (gates applied left-to-right)
            run[q].accum = gate_mat * run[q].accum;
            run[q].active = true;
            run[q].qubit = q;
        } else {
            // Multi-qubit, special, or conditioned gate: flush affected qubits
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
            // Cancellation also requires IDENTICAL classical conditioning:
            // a conditional CX must never cancel an unconditional one (they
            // fire under different runtime states). Equal-condition pairs
            // (including both unconditioned, clbit == -1) cancel validly.
            if (prev.type == inst.type &&
                prev.qubits == inst.qubits &&
                prev.condition_clbit == inst.condition_clbit &&
                prev.condition_value == inst.condition_value) {
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

// Helper: build 4x4 from instruction via basis-column statevector simulation.
// Returns nullopt for anything that cannot (or must not) be consolidated:
// classically-conditioned gates and unknown gate types. The previous identity
// fallback silently DELETED unsupported gates from consolidated blocks.
static std::optional<Eigen::Matrix4cd> instruction_to_4x4(const Instruction& inst) {
    using GT = Instruction::GateType;
    if (inst.condition_clbit >= 0) return std::nullopt;
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
            case GT::CU:    gates::apply_cu(basis, 0, 1, p[0], p[1], p[2], p[3]); break;
            case GT::ECR:   gates::apply_ecr(basis, 0, 1); break;
            case GT::RZX:   gates::apply_rzx(basis, 0, 1, p[0]); break;
            case GT::RXX:   gates::apply_rxx(basis, 0, 1, p[0]); break;
            case GT::RYY:   gates::apply_ryy(basis, 0, 1, p[0]); break;
            case GT::RZZ:   gates::apply_rzz(basis, 0, 1, p[0]); break;
            case GT::UNITARY:
                if (inst.matrix.size() != 16) return std::nullopt;
                gates::apply_unitary(basis, {0, 1}, inst.matrix);
                break;
            default: return std::nullopt;
        }
        for (int row = 0; row < 4; ++row) {
            U(row, col) = std::complex<double>(basis.real_parts[row], basis.imag_parts[row]);
        }
    }
    return U;
}

// 4x4 matrix of a 2-qubit circuit on qubits {0,1} (basis-column simulation).
static Eigen::Matrix4cd circuit_to_4x4(const QuantumCircuit& qc2) {
    Eigen::Matrix4cd U = Eigen::Matrix4cd::Zero();
    StatevectorSimulator sim;
    for (int col = 0; col < 4; ++col) {
        Statevector basis(2);
        basis.initialize_basis(col);
        // A basis-column extraction, not an execution: the operands here came
        // from the circuit under optimisation, which is judged where it enters
        // and again by run()'s pre-flight.
        for (const auto& ki : qc2.instructions)
            sim.apply_instruction(basis, ki, {Validation::Ignore});
        for (int row = 0; row < 4; ++row)
            U(row, col) = std::complex<double>(basis.real_parts[row],
                                               basis.imag_parts[row]);
    }
    return U;
}

// Equality up to a global phase, anchored on B's largest-magnitude entry.
static bool matrices_equal_up_to_phase(const Eigen::Matrix4cd& A,
                                       const Eigen::Matrix4cd& B,
                                       double tol = 1e-6) {
    int bi = 0, bj = 0;
    double best = 0.0;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            if (std::abs(B(i, j)) > best) {
                best = std::abs(B(i, j));
                bi = i;
                bj = j;
            }
    if (best < 1e-12) return A.norm() < tol;
    const std::complex<double> phase = A(bi, bj) / B(bi, bj);
    if (std::abs(std::abs(phase) - 1.0) > tol) return false;
    return (A - phase * B).norm() < tol;
}

// Tensor-product split: W (4x4) = W1 ⊗ W0  (W1 = qubit-1/MSB, W0 = qubit-0/LSB)
// W is promised to be a tensor product (up to numerical error).
static void tensor_factor(const Eigen::Matrix4cd& W,
                           Eigen::Matrix2cd& W0, Eigen::Matrix2cd& W1) {
    double norms[4] = {
        W.block<2,2>(0,0).norm(), W.block<2,2>(0,2).norm(),
        W.block<2,2>(2,0).norm(), W.block<2,2>(2,2).norm()
    };
    int best = (int)(std::max_element(norms, norms + 4) - norms);
    int ri = (best >= 2) ? 2 : 0, ci = (best % 2) ? 2 : 0;

    Eigen::JacobiSVD<Eigen::Matrix2cd> svd0(W.block<2,2>(ri, ci),
                                             Eigen::ComputeFullU | Eigen::ComputeFullV);
    W0 = svd0.matrixU() * svd0.matrixV().adjoint();

    Eigen::Matrix2cd W0inv = W0.adjoint();
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            W1(i, j) = (W0inv * W.block<2,2>(2 * i, 2 * j)).trace() / 2.0;

    Eigen::JacobiSVD<Eigen::Matrix2cd> svd1(W1, Eigen::ComputeFullU | Eigen::ComputeFullV);
    W1 = svd1.matrixU() * svd1.matrixV().adjoint();
}

// Emit a single-qubit U3 gate, or nothing if the matrix is identity.
static void emit_1q(QuantumCircuit& circ, const Eigen::Matrix2cd& U2, int q) {
    if (is_identity_2x2(U2, 1e-10)) return;
    auto [phi, theta, lam, gphase] = zyz_decompose(U2);
    (void)gphase;
    constexpr double atol = 1e-10;
    if (std::abs(theta) < atol && std::abs(phi) < atol && std::abs(lam) < atol) return;
    Instruction u3;
    u3.type = Instruction::GateType::U;
    u3.qubits = {q};
    u3.params = {theta, phi, lam};
    circ.instructions.push_back(u3);
}

// Takagi factorisation of a complex symmetric UNITARY S: a real orthogonal O
// with O^T S O diagonal.
//
// Write S = A + iB with A, B real symmetric. S is unitary here (S = Ud^T Ud
// with Ud unitary, and S* S = Ud* Ud^T = (Ud Ud*)* = I), and expanding
// S* S = I gives A B = B A. Commuting real symmetric matrices share a real
// orthonormal eigenbasis, so ONE real orthogonal matrix diagonalises S:
// diagonalise A, then inside each degenerate eigenvalue cluster of A
// diagonalise B restricted to that cluster. Where both are degenerate on a
// cluster, S restricted to it is a multiple of the identity and any
// orthonormal basis of the cluster serves, so two levels suffice at 4x4.
//
// O^T O = I holds by construction. That is the point: it removes the
// distinct-eigenvalue precondition that reading Takagi vectors off a Schur
// decomposition carries, and every common two-qubit gate violates that
// precondition (CZ and CNOT, iSWAP, SWAP and the identity all repeat a Weyl
// coordinate; only a generic unitary has three distinct ones).
static Eigen::Matrix4d takagi_real_orthogonal(const Eigen::Matrix4cd& S) {
    const Eigen::Matrix4d A = S.real();
    const Eigen::Matrix4d B = S.imag();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> es(A);
    Eigen::Matrix4d O = es.eigenvectors();
    const Eigen::Vector4d w = es.eigenvalues();  // ascending

    // S unitary puts the eigenvalues of A in [-1, 1], so an absolute cluster
    // tolerance is meaningful without scaling by the matrix norm.
    constexpr double kClusterTol = 1e-9;

    int start = 0;
    for (int k = 1; k <= 4; ++k) {
        if (k == 4 || std::abs(w(k) - w(start)) > kClusterTol) {
            const int dim = k - start;
            if (dim > 1) {
                const Eigen::MatrixXd basis = O.block(0, start, 4, dim);
                Eigen::MatrixXd blk = basis.transpose() * B * basis;
                blk = 0.5 * (blk + blk.transpose());  // shed asymmetric round-off
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es2(blk);
                O.block(0, start, 4, dim) = basis * es2.eigenvectors();
            }
            start = k;
        }
    }
    return O;
}

// Weyl chamber reduction. Many (kx, ky, kz) describe the same operator up to
// local gates, and they do not all cost the same: a point with a zero
// coordinate needs one fewer interaction rotation, so reducing into the
// canonical chamber saves a two-qubit gate rather than merely tidying numbers.
// iSWAP is the case that shows it, canonical at (pi/4, pi/4, 0) for two
// rotations while the square-root search reaches it at (pi/2, 3pi/4, 3pi/4),
// needing three.
//
// Three generators, each absorbed entirely into the local factors:
//   SHIFT   exp(-i(pi/2)·P⊗P) = -i(P⊗P), so moving one coordinate by pi/2
//           costs a local P on each wire, plus a global phase nobody observes.
//   NEGATE  conjugating by P⊗I flips the two coordinates whose Pauli
//           anticommutes with P and leaves the third alone.
//   SWAP    conjugating by a local Clifford transposes two coordinates.
static void canonicalise_weyl(std::array<double, 3>& k,
                              Eigen::Matrix2cd& W0, Eigen::Matrix2cd& W1,
                              Eigen::Matrix2cd& V0, Eigen::Matrix2cd& V1) {
    const std::complex<double> img(0.0, 1.0);

    std::array<Eigen::Matrix2cd, 3> P;
    P[0] <<   0,    1,     1,   0;   // X, paired with the kx coordinate
    P[1] <<   0, -img,   img,   0;   // Y, with ky
    P[2] <<   1,    0,     0,  -1;   // Z, with kz

    Eigen::Matrix2cd Sg;
    Sg << 1, 0, 0, img;
    Eigen::Matrix2cd Hg;
    Hg << 1, 1, 1, -1;
    Hg *= INV_SQRT2;

    const auto shift = [&](int i) {
        k[i] -= PI_2;
        V0 = V0 * P[i];
        V1 = V1 * P[i];
    };
    const auto negate_other_two = [&](int i) {
        for (int j = 0; j < 3; ++j)
            if (j != i) k[j] = -k[j];
        V1 = V1 * P[i];
        W1 = P[i] * W1;
    };
    const auto swap_coords = [&](int i, int j) {
        const int a = std::min(i, j), b = std::max(i, j);
        const Eigen::Matrix2cd C =
            (a == 0 && b == 1) ? Sg : (a == 0 && b == 2) ? Hg : (Sg * Hg * Sg);
        std::swap(k[i], k[j]);
        V0 = V0 * C.adjoint();
        V1 = V1 * C.adjoint();
        W0 = C * W0;
        W1 = C * W1;
    };

    // Fold each coordinate into (-pi/4, pi/4]. Four shifts return the local
    // factors to where they started while subtracting 2pi, and a 2pi change in
    // a coordinate is physically nothing (P⊗P has eigenvalues ±1, so
    // exp(-2·pi·i·P⊗P) = I). Only n mod 4 shifts need applying; the rest comes
    // off the number without touching a matrix.
    for (int i = 0; i < 3; ++i) {
        const int n = static_cast<int>(std::floor((k[i] + PI_4) / PI_2));
        const int reps = ((n % 4) + 4) % 4;
        for (int r = 0; r < reps; ++r) shift(i);
        k[i] -= static_cast<double>(n - reps) * PI_2;
    }

    // Order by descending magnitude, then drive the two leading coordinates
    // non-negative. Negations arrive in pairs, so the trailing coordinate is
    // the one allowed to stay negative.
    constexpr double kOrderTol = 1e-12;
    for (int pass = 0; pass < 3; ++pass)
        for (int i = 0; i < 2; ++i)
            if (std::abs(k[i]) < std::abs(k[i + 1]) - kOrderTol)
                swap_coords(i, i + 1);

    if (k[0] < 0.0 && k[1] < 0.0)       negate_other_two(2);
    else if (k[0] < 0.0 && k[2] < 0.0)  negate_other_two(1);
    else if (k[1] < 0.0 && k[2] < 0.0)  negate_other_two(0);
    else if (k[0] < 0.0)                negate_other_two(1);
    else if (k[1] < 0.0)                negate_other_two(0);
}

// KAK decomposition. Decomposes U4 into local gates + exp(i*H_interaction).
// U = (V1⊗V0) · exp(i*(kx·XX + ky·YY + kz·ZZ)) · (W1⊗W0)
// Returns the full gate sequence as a QuantumCircuit on qubits [0,1].
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
    constexpr double inv_sqrt2 = INV_SQRT2;

    // Magic basis matrix
    Eigen::Matrix4cd M;
    M <<  1,  0,  0,  img,
          0,  img, 1, 0,
          0,  img,-1, 0,
          1,  0,  0, -img;
    M *= inv_sqrt2;

    // The magic-basis correspondence SU(2)⊗SU(2) <-> SO(4) requires U in
    // SU(4); a product of gate unitaries is only in U(4). The phase removed
    // here is global and unobservable, and the caller compares up to exactly
    // that phase.
    const std::complex<double> det4 = U4.determinant();
    const Eigen::Matrix4cd Usu = U4 / std::pow(det4, 0.25);

    // U in magic basis
    Eigen::Matrix4cd Ud = M.adjoint() * Usu * M;

    // Ud is real orthogonal exactly when U4 is local, so the decomposition is
    // a hunt for two real orthogonal factors, Ud = O_V · D · O_W with D diagonal
    // unitary. Squaring isolates O_W:
    //     S := Ud^T Ud = O_W^T D² O_W
    // so a Takagi factorisation S = O Λ O^T with O real orthogonal delivers
    // O_W = O^T and D² = Λ.
    Eigen::Matrix4cd S = Ud.transpose() * Ud;

    const Eigen::Matrix4d O = takagi_real_orthogonal(S);
    const Eigen::Matrix4cd Ocx = O.cast<std::complex<double>>();
    const Eigen::Vector4cd lambda = (Ocx.transpose() * S * Ocx).diagonal();

    Eigen::Vector4d base;
    for (int j = 0; j < 4; ++j) base(j) = std::arg(lambda(j)) / 2.0;

    // D is a SQUARE ROOT of Λ, so each diagonal entry carries an independent
    // sign, and O's columns carry an arbitrary order. Only one family of
    // (order, sign) choices puts D in the canonical magic-basis positions,
    //     d0 = -kx + ky - kz        d2 =  kx + ky + kz
    //     d1 = -kx - ky + kz        d3 =  kx - ky - kz
    // which invert to kx = (d2+d3)/2, ky = (d0+d2)/2, kz = (d1+d2)/2. Only in
    // that arrangement is M D M† the interaction that the emitted
    // RXX/RYY/RZZ reproduce.
    //
    // Substituting the inverse back into the forward map leaves a residual of
    // +-(sum d)/2 on every entry, so the whole consistency requirement
    // collapses to one scalar condition: sum(d) = 0 (mod 4pi). A permutation
    // cannot change a sum, so it constrains only HOW MANY entries are
    // sign-flipped, never which ones. This is strictly stronger than
    // det(D) = +1 (sum(d) = 0 mod 2pi); requiring only the weaker condition
    // strands every operand whose phase sum is pi, iSWAP among them.
    constexpr double kFourPi = 2.0 * TWO_PI;
    // Slack on a whole-turn count, not on a physical quantity: the sum is a
    // few additions of values already accurate to a handful of ulps.
    constexpr double kTurnTol = 1e-9;
    const double sum_base = base.sum();

    std::array<int, 16> valid_masks{};
    int n_masks = 0;
    for (int mask = 0; mask < 16; ++mask) {
        int flips = 0;
        for (int b = 0; b < 4; ++b) flips += (mask >> b) & 1;
        const double turns = (sum_base + PI * flips) / kFourPi;
        if (std::abs(turns - std::round(turns)) < kTurnTol) valid_masks[n_masks++] = mask;
    }

    // Every surviving candidate is a CORRECT decomposition; they differ only in
    // how many interaction rotations they cost. Rank by that, because reducing
    // gate count is the purpose of the pass and a coordinate outside the Weyl
    // chamber can need three rotations where two would do.
    constexpr double kAtol = 1e-10;
    std::array<int, 4> perm = {0, 1, 2, 3};
    std::array<int, 4> best_perm = {0, 1, 2, 3};
    Eigen::Vector4d best_d = Eigen::Vector4d::Zero();
    double kx = 0.0, ky = 0.0, kz = 0.0;
    int best_nonzero = 4;
    double best_magnitude = 0.0;
    bool found = false;

    do {
        for (int mi = 0; mi < n_masks; ++mi) {
            const int mask = valid_masks[mi];
            Eigen::Vector4d d;
            for (int j = 0; j < 4; ++j)
                d(j) = base(perm[j]) + (((mask >> j) & 1) ? PI : 0.0);

            const double tkx = (d(2) + d(3)) / 2.0;
            const double tky = (d(0) + d(2)) / 2.0;
            const double tkz = (d(1) + d(2)) / 2.0;

            int nonzero = 0;
            double magnitude = 0.0;
            for (const double v : {tkx, tky, tkz}) {
                if (std::abs(v) > kAtol) ++nonzero;
                magnitude += std::abs(v);
            }

            if (!found || nonzero < best_nonzero ||
                (nonzero == best_nonzero && magnitude < best_magnitude - kAtol)) {
                found = true;
                best_nonzero = nonzero;
                best_magnitude = magnitude;
                best_perm = perm;
                best_d = d;
                kx = tkx;
                ky = tky;
                kz = tkz;
            }
        }
    } while (std::next_permutation(perm.begin(), perm.end()));

    // Leaving the operand undecomposed beats guessing at it: run()'s
    // verification net then keeps the original block.
    if (!found) return QuantumCircuit(2);

    Eigen::Matrix4d Ow = Eigen::Matrix4d::Zero();
    for (int j = 0; j < 4; ++j) Ow.col(j) = O.col(best_perm[j]);

    // Both local factors must be SPECIAL orthogonal, since SU(2)⊗SU(2)
    // corresponds to SO(4) rather than O(4). det(O_V)*det(O_W) is pinned at
    // det(Ud)/det(D) = 1, so the two determinants always share a sign and one
    // column flip corrects both at once. The flip is free: it leaves
    // S = O Λ O^T intact.
    if (Ow.determinant() < 0.0) Ow.col(0) = -Ow.col(0);

    Eigen::Vector4cd ad_inv_diag;
    for (int j = 0; j < 4; ++j)
        ad_inv_diag(j) = std::exp(std::complex<double>(0.0, -best_d(j)));

    const Eigen::Matrix4cd Owc = Ow.cast<std::complex<double>>();

    // O_V = Ud · O · D⁻¹ needs no realness check. G := Ud O satisfies
    // G^T G = O^T S O = Λ = D², so O_V^T O_V = I, and O_V is unitary too,
    // being a product of unitaries. Complex-orthogonal AND unitary forces
    // O_V^T = O_V†, hence O_V = conj(O_V): real.
    const Eigen::Matrix4cd L_M = Owc.transpose();                      // O_W
    const Eigen::Matrix4cd K_M = Ud * Owc * ad_inv_diag.asDiagonal();  // O_V

    // Convert to physical basis
    Eigen::Matrix4cd W_phys = M * L_M * M.adjoint();  // W1 ⊗ W0 (pre-rotation)
    Eigen::Matrix4cd V_phys = M * K_M * M.adjoint();  // V1 ⊗ V0 (post-rotation)

    Eigen::Matrix2cd W0, W1, V0, V1;
    tensor_factor(W_phys, W0, W1);
    tensor_factor(V_phys, V0, V1);

    // The rotation-count ranking above picks the cheapest point the
    // square-root search can reach, which is not always the cheapest point
    // that exists. Reducing into the Weyl chamber closes that gap, moving the
    // cost into local factors that are emitted either way.
    std::array<double, 3> weyl = {kx, ky, kz};
    canonicalise_weyl(weyl, W0, W1, V0, V1);
    kx = weyl[0];
    ky = weyl[1];
    kz = weyl[2];

    // Build result circuit: (W1⊗W0) → interaction → (V1⊗V0)
    QuantumCircuit result(2);
    constexpr double atol = 1e-10;

    emit_1q(result, W0, 0);
    emit_1q(result, W1, 1);

    if (std::abs(kz) > atol) {
        Instruction rzz;
        rzz.type = Instruction::GateType::RZZ;
        rzz.qubits = {0, 1};
        rzz.params = {2.0 * kz};
        result.instructions.push_back(rzz);
    }
    if (std::abs(ky) > atol) {
        Instruction ryy;
        ryy.type = Instruction::GateType::RYY;
        ryy.qubits = {0, 1};
        ryy.params = {2.0 * ky};
        result.instructions.push_back(ryy);
    }
    if (std::abs(kx) > atol) {
        Instruction rxx;
        rxx.type = Instruction::GateType::RXX;
        rxx.qubits = {0, 1};
        rxx.params = {2.0 * kx};
        result.instructions.push_back(rxx);
    }

    emit_1q(result, V0, 0);
    emit_1q(result, V1, 1);

    // Remap qubits from [0,1] to actual [q0,q1]
    for (auto& inst : result.instructions)
        for (auto& q : inst.qubits) q = (q == 0) ? q0 : q1;

    return result;
}

// =============================================================================
// tqd::lower_2q_unitary - the ConsolidateBlocks numerics, reached by the QASM
// exporters through src/transpiler/two_qubit_decompose.hpp
// =============================================================================

namespace tqd {

std::optional<Lowered> lower_2q_unitary(const Instruction& inst) {
    if (inst.qubits.size() != 2) return std::nullopt;

    const auto u4 = instruction_to_4x4(inst);
    if (!u4) return std::nullopt;

    // Both the 4x4 above and the circuit below live on qubits [0, 1]; the
    // operand order of `inst` is applied at the end, so the two never disagree
    // about which wire is which.
    const QuantumCircuit kak = kak_decompose(*u4, 0, 1);
    const Eigen::Matrix4cd rebuilt = circuit_to_4x4(kak);
    if (!matrices_equal_up_to_phase(rebuilt, *u4)) return std::nullopt;

    // Recover the phase the emitted sequence dropped. The comparison is taken
    // at the largest-magnitude entry, which is the best-conditioned place to
    // divide; U(theta, phi, lambda) spans SU(2), so a U(2) operand always
    // leaves exactly one global phase behind and never more.
    int bi = 0, bj = 0;
    double best = 0.0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (std::abs(rebuilt(i, j)) > best) {
                best = std::abs(rebuilt(i, j));
                bi = i;
                bj = j;
            }
        }
    }

    Lowered out;
    out.global_phase =
        (best > 1e-12) ? std::arg((*u4)(bi, bj) / rebuilt(bi, bj)) : 0.0;
    out.instructions = kak.instructions;
    for (auto& emitted : out.instructions) {
        for (auto& q : emitted.qubits) q = (q == 0) ? inst.qubits[0] : inst.qubits[1];
        emitted.condition_clbit = inst.condition_clbit;
        emitted.condition_value = inst.condition_value;
        emitted.validation = inst.validation;
    }
    return out;
}

}  // namespace tqd

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

        // Collect all consecutive gates involving only {qa, qb}.
        // Stop when we see a gate on qa or qb that isn't on both, or a gate
        // we cannot represent as a 4x4 (conditioned / unknown): those must
        // never be absorbed into the block, since an identity fallback would
        // silently delete them.
        auto accum_opt = instruction_to_4x4(inst);
        if (!accum_opt) {
            optimized.instructions.push_back(inst);
            continue;
        }
        Eigen::Matrix4cd accum = *accum_opt;
        std::vector<Instruction> block_insts = {inst};
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
                auto gate_opt = instruction_to_4x4(next);
                if (!gate_opt) break;  // unsupported: leave it for the outer loop
                accum = (*gate_opt) * accum;  // gates applied in order -> rightmost first in circuit
                block_insts.push_back(next);
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

        // A lone gate skips the decomposition entirely. Nothing is lost by
        // that: its block holds one two-qubit gate, so the count guard below
        // could only accept a replacement with fewer instructions overall, and
        // a decomposition emitting an interaction rotation plus local
        // corrections never has fewer. Skipping is the same verdict, reached
        // without doing the work.
        if (block_count == 1) {
            optimized.instructions.push_back(inst);
        } else {
            QuantumCircuit kak_circ = kak_decompose(accum, qa, qb);

            // Two independent gates decide whether the decomposition is used,
            // and both are load-bearing. The verification net answers "is this
            // decomposition VALID": rebuild its 4x4 and require equality with
            // the block up to global phase. The count guard below answers "is
            // it WORTH keeping". Folding the two together would let a
            // numerical failure hide behind a size comparison.
            QuantumCircuit local(2);
            for (auto ki : kak_circ.instructions) {
                for (auto& q : ki.qubits) q = (q == qa) ? 0 : 1;
                local.instructions.push_back(std::move(ki));
            }
            const bool exact =
                matrices_equal_up_to_phase(circuit_to_4x4(local), accum);

            // Two-qubit gates are the metric. They dominate cost in every
            // backend, and they are the only count that is trustworthy at this
            // point: the cleanup sweep that follows this pass at level 3
            // merges and cancels single-qubit gates, so a total-gate
            // comparison taken here judges a circuit that no longer exists by
            // the time it runs. The two-qubit count is stable across that
            // sweep, since CXCancellation matches only CX pairs and this pass
            // emits RXX/RYY/RZZ. Total count therefore breaks ties only, where
            // it cannot invert the decision.
            const auto count_2q = [](const std::vector<Instruction>& v) {
                int n = 0;
                for (const auto& in : v)
                    if (in.qubits.size() == 2) ++n;
                return n;
            };
            const int kak_2q = count_2q(kak_circ.instructions);
            const int block_2q = count_2q(block_insts);
            const bool cheaper =
                kak_2q < block_2q ||
                (kak_2q == block_2q &&
                 kak_circ.instructions.size() <= block_insts.size());

            if (exact && cheaper) {
                for (const auto& ki : kak_circ.instructions) {
                    optimized.instructions.push_back(ki);
                }
            } else {
                for (const auto& bi : block_insts) {
                    optimized.instructions.push_back(bi);
                }
            }
        }
    }

    return DAGCircuit::from_circuit(optimized);
}

} // namespace lindblad
