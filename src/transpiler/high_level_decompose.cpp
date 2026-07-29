// =============================================================================
// high_level_decompose.cpp — exact lowering of MCX / MCP / PERMUTATION
// =============================================================================
// See high_level_decompose.hpp for the module contract. Everything here is an
// EXACT algebraic identity; the .1 test wave verifies each construction by
// unitary / statevector equivalence against the native backend paths.
//
// Complexity (ancilla-free, self-contained on the instruction's own qubits):
//   MCP(m qubits)  — the λ/2 recursion runs m-2 levels; each level emits two
//                    inner MCXs of shrinking width, lowered by borrowed-wire
//                    halving (quadratic per MCX) → O(m³) base gates worst case.
//   MCX(k ctrls)   — H-conjugated MCP(π) → O(k³) worst case.
//   PERMUTATION    — relabel subclass: ≤ k-1 SWAPs; general map: one
//                    pattern-MCX network per displaced basis state.
// The cubic worst case is the price of using NO wires beyond the operands; a
// borrowed-wire v-chain (Barenco Lemma 7.2/7.3, linear per MCX) is the known
// upgrade if lowering cost ever matters. It does not affect exactness.
//
// Qubit-ordering convention (CLAUDE.md / docs/Architecture.md): LSB is
// qubits[0] everywhere. perm[x] indexes sub-states with bit j of x carried by
// wire qubits[j]; MCX instruction qubits are [controls..., target].

#include "high_level_decompose.hpp"

#include "lindblad/transpiler.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace lindblad::hld {

namespace {

using GT = Instruction::GateType;

// ---- tiny instruction factories --------------------------------------------
// All lowering paths funnel through these so the emitted shapes stay uniform.

Instruction make_gate(GT t, std::vector<int> qubits, std::vector<double> params = {}) {
    Instruction inst;
    inst.type   = t;
    inst.qubits = std::move(qubits);
    inst.params = std::move(params);
    return inst;
}

void emit_x  (std::vector<Instruction>& out, int q)               { out.push_back(make_gate(GT::X,  {q})); }
void emit_h  (std::vector<Instruction>& out, int q)               { out.push_back(make_gate(GT::H,  {q})); }
void emit_p  (std::vector<Instruction>& out, double l, int q)     { out.push_back(make_gate(GT::P,  {q},    {l})); }
void emit_cp (std::vector<Instruction>& out, double l, int a, int b) { out.push_back(make_gate(GT::CP, {a, b}, {l})); }
void emit_cx (std::vector<Instruction>& out, int c, int t)        { out.push_back(make_gate(GT::CX, {c, t})); }
void emit_ccx(std::vector<Instruction>& out, int a, int b, int t) { out.push_back(make_gate(GT::CCX,{a, b, t})); }
void emit_swap(std::vector<Instruction>& out, int a, int b)       { out.push_back(make_gate(GT::SWAP,{a, b})); }

void validate_distinct(const std::vector<int>& qubits, const char* who) {
    std::vector<int> sorted = qubits;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        throw std::invalid_argument(std::string(who) + ": duplicate qubit operand");
    }
}

// ---- MCX via borrowed-wire halving -----------------------------------------
// mcx_with_borrow(C → t, borrow d): exact C^kX using ONE borrowed wire d in an
// ARBITRARY (dirty) state. k ≤ 2 is direct. k ≥ 3 splits C into halves C1, C2
// and emits, in circuit order,
//
//     B A B A   with  B = C^{|C1|}X(C1 → d),  A = C^{|C2|+1}X(C2 ∪ {d} → t).
//
// Net flip of t = AND(C2)·(d ⊕ AND(C1))  XOR  AND(C2)·d = AND(C1)·AND(C2)
// = AND(C), for ANY initial d, and d is restored by the second B. The
// sub-calls always have a borrow available: B does not involve t (borrow t),
// A does not involve C1 (borrow C1's first wire). Gate count is O(k²).

void mcx_with_borrow(std::vector<Instruction>& out,
                     const std::vector<int>& controls, int target, int borrow) {
    const size_t k = controls.size();
    if (k == 0) { emit_x(out, target); return; }
    if (k == 1) { emit_cx(out, controls[0], target); return; }
    if (k == 2) { emit_ccx(out, controls[0], controls[1], target); return; }

    const size_t h = (k + 1) / 2; // ⌈k/2⌉
    const std::vector<int> c1(controls.begin(), controls.begin() + static_cast<std::ptrdiff_t>(h));
    std::vector<int> c2(controls.begin() + static_cast<std::ptrdiff_t>(h), controls.end());

    std::vector<int> c2d = c2;
    c2d.push_back(borrow);

    // B: C1 → borrow, borrowing t (t is uninvolved in B).
    // A: C2 ∪ {borrow} → t, borrowing c1[0] (C1 is uninvolved in A).
    mcx_with_borrow(out, c1,  borrow, target);
    mcx_with_borrow(out, c2d, target, c1[0]);
    mcx_with_borrow(out, c1,  borrow, target);
    mcx_with_borrow(out, c2d, target, c1[0]);
}

// ---- MCP via the λ/2 recursion ---------------------------------------------
// mcp_rec applies phase λ exactly on the all-ones subspace of `qubits`:
//
//   m == 1 : P(λ)          m == 2 : CP(λ)
//   m ≥ 3  : with a = q1..q_{m-2}, b = q_{m-1}, t = q_m,
//            CP(λ/2)(b,t) · MCX(a→b) · CP(-λ/2)(b,t) · MCX(a→b)
//            · mcp_rec(λ/2, a ∪ {t})
//
// Phase bookkeeping on a basis state (all operators here map basis states to
// phased basis states, so this argument is a complete unitary proof): the two
// CP(±λ/2) terms contribute λ/2·t·AND(a) with sign +1 when b = 1 and -1 when
// b = 0; the recursion contributes λ/2·AND(a)·t by induction. Total: λ on
// all-ones, 0 elsewhere. The inner MCXs borrow t, which they never touch.

void mcp_rec(std::vector<Instruction>& out,
             double lambda, const std::vector<int>& qubits) {
    const size_t m = qubits.size();
    if (m == 1) { emit_p(out, lambda, qubits[0]); return; }
    if (m == 2) { emit_cp(out, lambda, qubits[0], qubits[1]); return; }

    const std::vector<int> a(qubits.begin(), qubits.end() - 2);
    const int b = qubits[m - 2];
    const int t = qubits[m - 1];
    const double half = lambda / 2.0;

    emit_cp(out, half, b, t);
    mcx_with_borrow(out, a, b, t);
    emit_cp(out, -half, b, t);
    mcx_with_borrow(out, a, b, t);

    std::vector<int> rest = a;
    rest.push_back(t);
    mcp_rec(out, half, rest);
}

// ---- PERMUTATION helpers ---------------------------------------------------

// Emit the transposition of basis states `sa` ↔ `sb` (sa ≠ sb) over wires
// `qubits`. Construction: conjugate by a CX fan (control = pivot wire, the
// lowest differing bit) so the two states differ ONLY at the pivot, then apply
// an MCX targeting the pivot with controls pattern-matched (X-conjugated
// anti-controls where the shared pattern has a 0). The fan is a commuting
// product of self-inverse CXs, so applying it on both sides is an exact
// conjugation and every other basis state is untouched.
void emit_transposition(std::vector<Instruction>& out,
                        int sa, int sb, const std::vector<int>& qubits) {
    const int k = static_cast<int>(qubits.size());
    const int d = sa ^ sb;
    int pivot = 0;
    while (((d >> pivot) & 1) == 0) { ++pivot; }

    // Pattern state: the one with a 0 at the pivot bit (unchanged by the fan).
    const int base = (((sa >> pivot) & 1) == 0) ? sa : sb;

    // Fan in: fold every other differing bit onto the pivot's control.
    std::vector<std::pair<int,int>> fan; // (control wire, target wire)
    for (int i = 0; i < k; ++i) {
        if (i != pivot && ((d >> i) & 1)) {
            fan.emplace_back(qubits[pivot], qubits[i]);
        }
    }
    for (const auto& [c, t] : fan) { emit_cx(out, c, t); }

    // Pattern-controlled MCX on the pivot. k == 1 degenerates to a plain X
    // (the only 1-qubit transposition is |0⟩ ↔ |1⟩).
    std::vector<int> anti;     // wires needing X-conjugation (pattern bit 0)
    std::vector<int> controls; // all non-pivot wires
    for (int i = 0; i < k; ++i) {
        if (i == pivot) { continue; }
        controls.push_back(qubits[i]);
        if (((base >> i) & 1) == 0) { anti.push_back(qubits[i]); }
    }
    for (int q : anti) { emit_x(out, q); }
    if (controls.empty()) {
        emit_x(out, qubits[pivot]);
    } else {
        std::vector<int> mq = controls;
        mq.push_back(qubits[pivot]);
        out.push_back(make_gate(GT::MCX, std::move(mq)));
    }
    for (int q : anti) { emit_x(out, q); }

    // Fan out (same set: commuting, self-inverse).
    for (const auto& [c, t] : fan) { emit_cx(out, c, t); }
}

} // anonymous namespace

// =============================================================================
// Public entry points
// =============================================================================

bool is_high_level(const Instruction& inst) {
    return inst.type == GT::MCX || inst.type == GT::MCP ||
           inst.type == GT::PERMUTATION;
}

std::vector<Instruction> lower_mcx(const std::vector<int>& controls, int target) {
    std::vector<int> all = controls;
    all.push_back(target);
    validate_distinct(all, "lower_mcx");

    std::vector<Instruction> out;
    const size_t k = controls.size();
    if (k == 0) { emit_x(out, target); return out; }
    if (k == 1) { emit_cx(out, controls[0], target); return out; }
    if (k == 2) { emit_ccx(out, controls[0], controls[1], target); return out; }

    // C^kX = H(t) · MCP(π on C ∪ {t}) · H(t): exactly the phase-π (Z-type)
    // multi-control conjugated into the X basis on the target.
    emit_h(out, target);
    mcp_rec(out, PI, all);
    emit_h(out, target);
    return out;
}

std::vector<Instruction> lower_mcp(double lambda, const std::vector<int>& qubits) {
    if (qubits.empty()) {
        throw std::invalid_argument("lower_mcp: needs at least one qubit");
    }
    validate_distinct(qubits, "lower_mcp");
    std::vector<Instruction> out;
    mcp_rec(out, lambda, qubits);
    return out;
}

std::vector<Instruction> lower_ccx(int a, int b, int target) {
    validate_distinct({a, b, target}, "lower_ccx");
    // The standard T-ladder (Nielsen & Chuang §4.3, T = P(π/4)): six CXs,
    // seven T/T†, two Hs. The identity is EXACT as a matrix equality — no
    // global-phase discrepancy — so it composes safely under conditions.
    constexpr double kT = PI / 4.0;
    std::vector<Instruction> out;
    emit_h (out, target);
    emit_cx(out, b, target);
    emit_p (out, -kT, target);
    emit_cx(out, a, target);
    emit_p (out,  kT, target);
    emit_cx(out, b, target);
    emit_p (out, -kT, target);
    emit_cx(out, a, target);
    emit_p (out,  kT, b);
    emit_p (out,  kT, target);
    emit_h (out, target);
    emit_cx(out, a, b);
    emit_p (out,  kT, a);
    emit_p (out, -kT, b);
    emit_cx(out, a, b);
    return out;
}

std::optional<std::vector<int>> as_qubit_relabel(const std::vector<int>& perm) {
    const size_t dim = perm.size();
    // dim = 2^k with k ≥ 0; k = 0 (dim 1) is the empty relabel.
    size_t k = 0;
    while ((size_t(1) << k) < dim) { ++k; }
    if ((size_t(1) << k) != dim) { return std::nullopt; }

    if (dim == 0 || perm[0] != 0) { return std::nullopt; }

    // Images of the one-hot states fix σ; every image must itself be one-hot.
    std::vector<int> sigma(k, -1);
    std::vector<char> used(k, 0);
    for (size_t j = 0; j < k; ++j) {
        const int img = perm[size_t(1) << j];
        if (img <= 0 || (img & (img - 1)) != 0) { return std::nullopt; }
        size_t s = 0;
        while ((1 << s) != img) { ++s; }
        if (used[s]) { return std::nullopt; }
        sigma[j] = static_cast<int>(s);
        used[s] = 1;
    }

    // σ must explain EVERY entry, not just the one-hots.
    for (size_t x = 0; x < dim; ++x) {
        int img = 0;
        for (size_t j = 0; j < k; ++j) {
            if ((x >> j) & 1) { img |= (1 << sigma[j]); }
        }
        if (perm[x] != img) { return std::nullopt; }
    }
    return sigma;
}

std::vector<Instruction> lower_permutation(const std::vector<int>& perm,
                                           const std::vector<int>& qubits) {
    const size_t k = qubits.size();
    const size_t dim = size_t(1) << k;
    validate_distinct(qubits, "lower_permutation");
    if (perm.size() != dim) {
        throw std::invalid_argument(
            "lower_permutation: permutation size must be 2^qubits.size()");
    }
    { // Bijection check: from_json can build unvalidated instructions.
        std::vector<char> seen(dim, 0);
        for (int v : perm) {
            if (v < 0 || static_cast<size_t>(v) >= dim || seen[static_cast<size_t>(v)]) {
                throw std::invalid_argument(
                    "lower_permutation: map is not a bijection over [0, 2^k)");
            }
            seen[static_cast<size_t>(v)] = 1;
        }
    }

    std::vector<Instruction> out;

    // Cheap subclass: pure wire relabeling → selection-network of ≤ k-1 SWAPs.
    if (auto sigma_opt = as_qubit_relabel(perm)) {
        const std::vector<int>& sigma = *sigma_opt;
        // content[p] = original bit currently sitting on wire position p;
        // loc[j] = current position of original bit j. Walk positions
        // ascending and place the bit that must END at p (σ[j] = p there).
        std::vector<int> content(k), loc(k), sigma_inv(k);
        for (size_t j = 0; j < k; ++j) {
            content[j] = static_cast<int>(j);
            loc[j]     = static_cast<int>(j);
            sigma_inv[static_cast<size_t>(sigma[j])] = static_cast<int>(j);
        }
        for (size_t p = 0; p < k; ++p) {
            const int j = sigma_inv[p];        // bit that must end at position p
            const int cp = loc[static_cast<size_t>(j)];
            if (cp == static_cast<int>(p)) { continue; }
            emit_swap(out, qubits[p], qubits[static_cast<size_t>(cp)]);
            const int j2 = content[p];
            content[p] = j;   loc[static_cast<size_t>(j)]  = static_cast<int>(p);
            content[static_cast<size_t>(cp)] = j2;  loc[static_cast<size_t>(j2)] = cp;
        }
        return out;
    }

    // General basis map: cycle decomposition into transpositions. A cycle
    // (c0 c1 … cr) with perm[c_i] = c_{i+1} equals the transposition product
    // (c0 c1)(c0 c2)…(c0 cr) applied FIRST-to-LAST in circuit order.
    std::vector<char> visited(dim, 0);
    for (size_t s = 0; s < dim; ++s) {
        if (visited[s] || perm[s] == static_cast<int>(s)) { visited[s] = 1; continue; }
        std::vector<int> cycle;
        size_t cur = s;
        while (!visited[cur]) {
            visited[cur] = 1;
            cycle.push_back(static_cast<int>(cur));
            cur = static_cast<size_t>(perm[cur]);
        }
        for (size_t i = 1; i < cycle.size(); ++i) {
            emit_transposition(out, cycle[0], cycle[i], qubits);
        }
    }
    return out;
}

std::vector<Instruction> lower_fully(const Instruction& inst) {
    std::vector<Instruction> lowered;
    switch (inst.type) {
        case GT::MCX: {
            std::vector<int> controls(inst.qubits.begin(), inst.qubits.end() - 1);
            lowered = lower_mcx(controls, inst.qubits.back());
            break;
        }
        case GT::MCP: {
            const double lambda = inst.params.empty() ? 0.0 : inst.params[0];
            lowered = lower_mcp(lambda, inst.qubits);
            break;
        }
        case GT::PERMUTATION: {
            // One-level lowering may keep MCX nodes (compact QASM 3); flatten
            // them here so the fully-lowered alphabet holds.
            std::vector<Instruction> mixed =
                lower_permutation(inst.permutation, inst.qubits);
            for (const Instruction& g : mixed) {
                if (g.type == GT::MCX) {
                    std::vector<int> controls(g.qubits.begin(), g.qubits.end() - 1);
                    std::vector<Instruction> flat =
                        lower_mcx(controls, g.qubits.back());
                    lowered.insert(lowered.end(), flat.begin(), flat.end());
                } else {
                    lowered.push_back(g);
                }
            }
            break;
        }
        default:
            return { inst };
    }

    // A conditioned high-level op lowers to a uniformly conditioned block.
    if (inst.condition_clbit >= 0) {
        for (Instruction& g : lowered) {
            g.condition_clbit = inst.condition_clbit;
            g.condition_value = inst.condition_value;
        }
    }
    return lowered;
}

} // namespace lindblad::hld

namespace lindblad {

// =============================================================================
// HighLevelDecompose — pre-routing stage-0 pass
// =============================================================================
// Contract in include/lindblad/transpiler.hpp. The scan-first structure keeps
// the common case (no high-level ops) allocation-free beyond the round-trip.

DAGCircuit HighLevelDecompose::run(
    const DAGCircuit& dag, const TranspilationContext& ctx
) const {
    using GateType = Instruction::GateType;
    QuantumCircuit qc = dag.to_circuit();

    // Routability floor (R.1.18.2, issue #67). SABRE executes a 3-qubit gate
    // only when all three wire pairs are simultaneously adjacent, and a
    // triangle-free target (path, grid, heavy-hex — every realistic device
    // topology) can never provide that. Under a constrained coupling map this
    // pass therefore guarantees an output with NO gate wider than two qubits:
    // every CCX — from the lowering above OR written by the user — is
    // flattened into the exact 6-CX T-ladder, which routes on any connected
    // map. Unconstrained targets keep CCX: with no routing consumer the
    // 3-qubit form is strictly better (a ccx-bearing basis keeps it native,
    // and flattening would force CX into a stream that never needed it).
    const bool floor_at_2q = ctx.coupling_map.n_physical_qubits > 0;

    bool any = false;
    for (const Instruction& inst : qc.instructions) {
        if (hld::is_high_level(inst) ||
            (floor_at_2q && inst.type == GateType::CCX)) {
            any = true;
            break;
        }
    }
    if (!any) { return dag; }

    std::vector<Instruction> out;
    out.reserve(qc.instructions.size());
    auto push = [&out, floor_at_2q](Instruction&& inst) {
        if (!(floor_at_2q && inst.type == GateType::CCX)) {
            out.push_back(std::move(inst));
            return;
        }
        std::vector<Instruction> ladder =
            hld::lower_ccx(inst.qubits[0], inst.qubits[1], inst.qubits[2]);
        for (Instruction& g : ladder) {
            g.condition_clbit = inst.condition_clbit; // uniform block, exactly
            g.condition_value = inst.condition_value; // like lower_fully
            out.push_back(std::move(g));
        }
    };
    for (const Instruction& inst : qc.instructions) {
        if (!hld::is_high_level(inst)) {
            push(Instruction(inst));
            continue;
        }
        std::vector<Instruction> lowered = hld::lower_fully(inst);
        for (Instruction& g : lowered) { push(std::move(g)); }
    }
    qc.instructions = std::move(out);
    return DAGCircuit::from_circuit(qc);
}

} // namespace lindblad
