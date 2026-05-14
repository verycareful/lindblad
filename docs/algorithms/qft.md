# QFT (Quantum Fourier Transform)

This page documents `lindblad::algorithms::QFT`, a standalone primitive for building and executing Quantum Fourier Transforms.

## Purpose

The Quantum Fourier Transform is the quantum analogue of the discrete Fourier transform. It is a critical subroutine in phase estimation (QPE), Shor's algorithm, and quantum signal processing.

In Lindblad, `QFT` is provided as a standalone class capable of generating exact QFT circuits, approximate QFTs (AQFT), and inverse QFTs (IQFT). It can be composed as a subroutine into larger algorithms or executed directly.

## Theory Summary

The standard $n$-qubit exact QFT is constructed from Hadamard (`H`) and controlled-phase (`CP`) gates.

1. For each qubit $j$ from $0$ to $n-1$:
   - Apply $H(j)$
   - For each qubit $k > j$:
     - Apply $CP(\pi / 2^{k-j})$ controlled by $k$ targeting $j$
2. Optional: apply $\lfloor n/2 \rfloor$ SWAP gates to reverse the final bit ordering.

```text
       ┌───┐ ┌──────────┐ ┌──────────┐
q_0: ──┤ H ├─┤ CP(π/2)  ├─┤ CP(π/4)  ├─ ... ─ × ─
       └───┘ └────┬─────┘ └────┬─────┘         │
q_1: ─────────────■────────────┼───── ... ─ × ─ ┤
                               │               │
q_2: ──────────────────────────■───── ... ─ × ──
```

For each qubit $j$, the control dots (■) sit on higher-index qubits $k > j$, and the CP gate box sits on qubit $j$. Final SWAPs reverse the wire order: `SWAP(0, n-1)`, `SWAP(1, n-2)`, etc.

### Complexity

| Variant | Gate count | Circuit depth | Clifford-simulable? |
|---|---|---|---|
| Exact QFT | $O(n^2)$ | $O(n)$ | No (for $n \ge 3$) |
| AQFT (degree $m$) | $O(n \cdot m)$ | $O(m)$ | Yes (if $m=1$) |
| Semi-classical (iterative) QFT | $O(n^2)$ single-qubit + feedforward | $O(n)$ sequential | Yes (for $n \le 2$) |

## Required Inputs

| Parameter | Type | Description |
|---|---|---|
| `n` | `int` | Number of qubits. Must be ≥ 1. |
| `opts.do_swaps` | `bool` | Append bit-reversal SWAPs (default: `true`). |
| `opts.approximation_degree` | `int` | Kitaev approximation degree `m` (default: `0` = exact). |
| `opts.inverse` | `bool` | Build IQFT instead of QFT (default: `false`). |

For `QFT::run()` and `QFT::apply()`:

| Parameter | Type | Description |
|---|---|---|
| `input_state` | `const QuantumCircuit&` | Circuit that prepares the input state from `\|0…0⟩`. Its `n_qubits` determines the register size. |
| `backend` | `backends::LocalBackend&` | (Optional) Explicit simulator backend. Omit to use the default Statevector backend. |
| `shots` | `int` | Number of measurement shots. `0` = no measurement (state inspection); `> 0` = sample the output distribution. |
| `seed` | `uint64_t` | RNG seed for reproducible sampling. |

## Variants Overview

### Exact QFT
The standard transform. Gate count scales as $n(n-1)/2$ CP gates plus $n$ H gates. Memory-heavy on physical devices but necessary for exact results.

### Inverse QFT (IQFT)
The adjoint of the QFT. Constructed by reversing the exact QFT gate sequence and negating all CP angles. This is the variant used at the end of Quantum Phase Estimation.

### Approximate QFT (AQFT)
*Kitaev approximation* / *Coppersmith approximation*.
Because CP gates with very small angles $\theta < \pi / 2^m$ contribute very little to the final state fidelity but consume significant circuit resources, AQFT drops them entirely.

- $m = 0$: exact QFT
- $m = 1$: retains only $CP(\pi/2)$ (the CS gate). This variant uses exclusively Clifford gates.
- $m = \log(n)$: standard hardware-efficient cutoff

### Semi-Classical QFT (Griffiths-Niu)

The Griffiths-Niu semi-classical QFT replaces all quantum CP gates with measurement feedforward. Each qubit is measured mid-circuit, and the result classically controls subsequent $P$ rotations on the next qubit, replacing the quantum control wire.

**Algorithm (forward QFT):**

For $j = n-1$ down to $j = 0$:
1. For each already-measured qubit $k > j$:
   - If $c[k] = 1$: apply $P(\pi / 2^{k-j})$ to qubit $j$
2. Apply $H(j)$
3. Measure qubit $j$ → classical bit $c[j]$

The output classical bits $c[n-1] \ldots c[0]$ hold the QFT Fourier coefficients directly, with $c[n-1]$ as the most significant bit.

**Key properties:**
- Uses only single-qubit gates ($H$ and $P$) — no two-qubit gates at all
- Requires mid-circuit measurement and classically-controlled gates (feedforward)
- The quantum state is fully collapsed at the end; output is always a bitstring, not a quantum state
- Clifford-compatible for $n \le 2$ (the only angles used are $\pi/2$ and smaller — but $\pi/2 = S$ is Clifford, while $\pi/4$ and below are not)
- The framework implements feedforward via `p_if()` instructions honouring `condition_clbit`/`condition_value` in all four simulators

## How to Invoke — Semi-Classical QFT

Include the header:

```cpp
#include "lindblad/algorithms.hpp"
```

### Running the iterative (semi-classical) QFT

```cpp
// Prepare an input state: |1⟩ on 3 qubits
lindblad::QuantumCircuit input(3);
input.x(0);

// Run semi-classical forward QFT with default Statevector backend
auto result = lindblad::algorithms::QFT::run_iterative(input, /*shots=*/1024);

// result.backend_result.counts: histogram of 3-bit measurement outcomes
for (const auto& [bits, count] : result.backend_result.counts) {
    std::cout << bits << ": " << count << "\n";
}
```

### Using an explicit backend

```cpp
lindblad::backends::LocalBackend::Config cfg;
cfg.simulator = lindblad::backends::LocalBackend::SimType::DENSITY_MATRIX;
lindblad::backends::LocalBackend dm_backend(cfg);

auto result = lindblad::algorithms::QFT::run_iterative(input, dm_backend, 512);
```

### Building the iterative circuit directly (for composition)

```cpp
// 4-qubit iterative QFT circuit (4 qubits + 4 classical bits)
auto iter_qft = lindblad::algorithms::QFT::build_iterative_circuit(4);

// 4-qubit iterative inverse QFT circuit
auto iter_iqft = lindblad::algorithms::QFT::build_iterative_inverse_circuit(4);
```

## Clifford-Simulability Note

The Clifford simulator can only execute circuits built from $\{H, S, CX, X, Y, Z\}$.

The QFT requires $CP(\theta)$ where $\theta = \pi / 2^k$.
- $k=1 \implies \theta = \pi/2$: This is the `CS` (controlled-S) gate, which is **Clifford**.
- $k \ge 2 \implies \theta \le \pi/4$: These are `CT` (controlled-T) gates and beyond, which are **non-Clifford**.

Therefore:
- Exact QFT on 1 or 2 qubits is Clifford.
- Exact QFT on $\ge 3$ qubits is **not Clifford-simulable**.
- AQFT with $m=1$ is Clifford-simulable for all $n$.

> [!WARNING]
> Attempting to run `QFT::run()` on an exact QFT ($n \ge 3$) using `SimType::CLIFFORD` will result in a runtime error. Use Statevector, DM, or MPS instead.
>
> For `run_iterative()`, the Clifford backend is only valid for $n \le 2$ (the only angle used is $\pi/2 = S$, which is Clifford). For $n \ge 3$, use any other simulator.

## How to Invoke

Include the header:

```cpp
#include "lindblad/algorithms.hpp"
```

### Composing as a Subroutine

If you want to use the QFT as part of a larger circuit (e.g. QPE):

```cpp
QuantumCircuit qc(4);
// ... prepare some state ...

// Append an exact IQFT to the circuit
qc = QFT::apply(qc, { /*do_swaps=*/true, /*approx_degree=*/0, /*inverse=*/true });

// Or use the convenience wrapper:
auto iqft_subcircuit = QFT::build_inverse_circuit(4);
// compose them...
```

### Direct Execution

To run a QFT directly and inspect the resulting statevector:

```cpp
QuantumCircuit initial_state(3);
initial_state.x(0);  // |100> input

// Run with default Statevector simulator
auto result = QFT::run(initial_state);

// result.backend_result contains the unmeasured exact final state (if shots=0)
```

### Approximate QFT

```cpp
// Build a 10-qubit AQFT dropping angles smaller than π/8 (m=3)
auto aqft_circuit = QFT::build_approximate_circuit(10, 3);
```

## Return Values and Outputs

`QFT::run()` returns a `QFT::Result`:

```cpp
struct Result {
    backends::BackendResult backend_result;  // counts, timing, success flag
    int n_qubits;
    bool clifford_compatible;
};
```

- `backend_result.counts`: Measurement outcome histogram. Only populated when `shots > 0`.
- `backend_result.simulation_time_seconds`: Wall time for the simulation step.
- `backend_result.success`: Whether the backend run completed without error.
- `n_qubits`: The number of qubits in the input circuit.
- `clifford_compatible`: Check this before dispatching to a Clifford backend.

`QFT::build_circuit()`, `build_inverse_circuit()`, `build_approximate_circuit()`, and `apply()` all return a `QuantumCircuit` with no classical bits (no measurements attached).

## Exceptions and Failure Modes

| Condition | Exception | Thrown by |
|---|---|---|
| `n <= 0` | `std::invalid_argument` | `build_circuit`, `build_inverse_circuit`, `build_approximate_circuit` (propagated) |
| `approximation_degree < 0` | `std::invalid_argument` | `build_approximate_circuit` |
| Exact QFT ($n \ge 3$) on `SimType::CLIFFORD` backend | Runtime backend error | `backend.run()` inside `QFT::run()` |

The `QFT::apply()` and `QFT::run()` methods propagate any exception thrown by `build_circuit()`.

## Common Pitfalls

- **Bit reversal (do_swaps)**: The standard QFT algorithm naturally outputs qubits in reversed order (MSB vs LSB). The framework appends SWAP gates by default to fix this. If you are handling wire-reversal manually (e.g. as a subroutine caller), set `Options::do_swaps = false`.
- **Backend Selection**: Don't use the Clifford backend for $n \ge 3$ exact QFT. The `QFT::Result` struct contains a `clifford_compatible` boolean you can check before dispatching to custom backends.
- **shots = 0 vs shots > 0**: With `shots = 0`, no `measure_all()` is appended; the backend operates on the state without sampling. This is the correct mode for statevector inspection and for composing QFT as a subroutine in a larger measurement workflow.

## Testing Notes

QFT correctness is verified by:
- Applying QFT followed immediately by IQFT to a known input state and checking that the output matches the original input (round-trip identity test).
- Checking the output of a 2-qubit QFT against the analytically known DFT matrix.
- Verifying that `clifford_compatible` is set correctly for the n=1, n=2, n=3 exact cases and for AQFT with m=1.
- Verifying that `run_iterative()` on a computational basis state produces output bitstrings whose histogram is consistent with the analytically known QFT output probabilities.
- Checking that `build_iterative_circuit()` and `build_iterative_inverse_circuit()` compose correctly as approximate round-trips (the measurement basis distributions should match between the two routes).

Tests live in `tests/test_qft.cpp` (or within the algorithms test suite). The `clifford_compatible` flag can be unit-tested by constructing circuits and reading the result field without running a backend.

## Related Source Files

- [docs/api/qft.md](../api/qft.md)
- [include/lindblad/algorithms.hpp](../../include/lindblad/algorithms.hpp)
- [src/algorithms/qft.cpp](../../src/algorithms/qft.cpp)
