# QFT API Deep Dive

This page documents the public `lindblad::algorithms::QFT` API in detail.

## Header and Namespace

- Header: `include/lindblad/algorithms.hpp`
- Namespace: `lindblad::algorithms`

## Class Overview

`QFT` is a standalone class containing static methods for generating Quantum Fourier Transform circuits and executing them on local backends.

It separates circuit construction from execution, allowing the QFT/IQFT/AQFT subcircuits to be used as subroutines in larger algorithms (like `QPE`).

## `QFT::Options`

Configuration structure for QFT circuit generation.

```cpp
struct Options {
    bool do_swaps;
    int approximation_degree;
    bool inverse;

    Options(bool swaps = true, int approx_deg = 0, bool inv = false);
};
```

- `do_swaps`: If true, appends `⌊n/2⌋` `SWAP` gates at the end of the transform to reverse the bit order (MSB vs LSB correction).
- `approximation_degree`: The Kitaev approximation degree `m`. If `m=0`, exact QFT. If `m>0`, all `CP(θ)` gates with `|θ| < π/2^m` are dropped.
- `inverse`: If true, builds the Inverse QFT (IQFT). Reverses the topological order of all gates and negates all CP angles.

## `QFT::Result`

Returned by `QFT::run()`.

```cpp
struct Result {
    backends::BackendResult backend_result;
    int n_qubits;
    bool clifford_compatible;
};
```

- `backend_result`: Contains the standard simulator output (`counts`, `simulation_time_seconds`, `success`, etc.). See [LocalBackend API](backends.md).
- `n_qubits`: Number of qubits in the input circuit.
- `clifford_compatible`: True if the generated QFT circuit contains only Clifford gates. This is true only if `n ≤ 2` (exact) or `approximation_degree == 1` (AQFT). If false, running on the `CLIFFORD` backend will throw an error.

## Circuit Builders

These methods construct and return a `QuantumCircuit` containing the QFT instruction sequence. They do not simulate anything.

### `build_circuit`

```cpp
static QuantumCircuit build_circuit(int n, const Options& opts = Options{});
```

Builds the base QFT or IQFT on `n` qubits according to `opts`.

**Throws:** `std::invalid_argument` if `n <= 0`.

### `build_inverse_circuit`

```cpp
static QuantumCircuit build_inverse_circuit(int n, bool do_swaps = true);
```

Convenience wrapper. Equivalent to `build_circuit(n, {do_swaps, 0, true})`. Used internally by `QPE`.

**Throws:** `std::invalid_argument` if `n <= 0` (propagated from `build_circuit`).

### `build_approximate_circuit`

```cpp
static QuantumCircuit build_approximate_circuit(int n, int m);
```

Convenience wrapper for AQFT. Equivalent to `build_circuit(n, {true, m, false})`.

**Throws:**
- `std::invalid_argument` if `m < 0`.
- `std::invalid_argument` if `n <= 0` (propagated from `build_circuit`).

### `apply`

```cpp
static QuantumCircuit apply(const QuantumCircuit& qc, const Options& opts = Options{});
```

Takes an existing circuit `qc`, copies it, generates a QFT subcircuit matching `qc.n_qubits`, and appends the QFT instructions to the end. Returns the composed circuit.

## Execution

### `run` (Backend Injection)

```cpp
static Result run(
    const QuantumCircuit& input_state,
    backends::LocalBackend& backend,
    const Options& opts = Options{},
    int shots = 0,
    uint64_t seed = 0
);
```

Applies the QFT to `input_state` and executes it on the provided `backend`.
- If `shots == 0`, no measurements are added. Use this with `STATEVECTOR` or `DENSITY_MATRIX` backends to inspect the exact output amplitudes.
- If `shots > 0`, `measure_all()` is appended to the circuit and the backend samples the distribution.

**Preconditions:** The backend simulator must be compatible with the circuit. Running an exact QFT with `n ≥ 3` on `SimType::CLIFFORD` will produce a runtime error; check `Result::clifford_compatible` beforehand.

### `run` (Statevector Default)

```cpp
static Result run(
    const QuantumCircuit& input_state,
    const Options& opts = Options{},
    int shots = 0,
    uint64_t seed = 0
);
```

Convenience overload that internally constructs a `LocalBackend` configured to use `SimType::STATEVECTOR` and delegates to the primary `run` method.

## Semi-Classical (Iterative) QFT — Griffiths & Niu 1996

These methods implement the feedforward variant of the QFT. Every instruction in the returned circuit has `n_clbits == n`. The framework honours `condition_clbit`/`condition_value` on each `P` gate in all four simulators; `shots` must be `> 0` since feedforward requires per-shot execution.

### `build_iterative_circuit`

```cpp
static QuantumCircuit build_iterative_circuit(int n);
```

Builds the semi-classical forward QFT circuit. The circuit has `n` qubits and `n` classical bits.

Processing order: qubit `n-1` first (MSB of the QFT output), qubit `0` last (LSB).

For each qubit `j` (from `n-1` down to `0`):
1. For each previously measured qubit `k > j`: `p_if(π / 2^{k-j}, j, k, 1)` — apply `P` to qubit `j` if `c[k] == 1`.
2. `H(j)`
3. `measure(j, j)` — collapse to classical bit `c[j]`

The output bitstring `c[n-1]...c[0]` holds the QFT Fourier coefficients.

**Throws:** `std::invalid_argument` if `n <= 0`.

### `build_iterative_inverse_circuit`

```cpp
static QuantumCircuit build_iterative_inverse_circuit(int n);
```

Builds the semi-classical inverse QFT circuit. Processing order: qubit `0` first, qubit `n-1` last.

For each qubit `j` (from `0` to `n-1`):
1. For each previously measured qubit `k < j`: `p_if(-π / 2^{j-k}, j, k, 1)`.
2. `H(j)`
3. `measure(j, j)`

**Throws:** `std::invalid_argument` if `n <= 0`.

### `run_iterative` (Backend Injection)

```cpp
static Result run_iterative(
    const QuantumCircuit& input_state,
    backends::LocalBackend& backend,
    int shots = 1024,
    uint64_t seed = 0
);
```

Composes `input_state` with the iterative QFT circuit (n qubits + n classical bits), then executes on `backend`.

- `shots` must be `> 0`; feedforward requires per-shot simulation.
- All four simulator backends support feedforward: Statevector, DensityMatrix, MPS, and Clifford (for `n ≤ 2` only).
- `Result::clifford_compatible` is always `false` (the iterative circuit uses non-Clifford angles for `n ≥ 3`).

**Throws:**
- `std::invalid_argument` if `shots <= 0`.
- Propagates any exception from `build_iterative_circuit` or the backend.

### `run_iterative` (Statevector Default)

```cpp
static Result run_iterative(
    const QuantumCircuit& input_state,
    int shots = 1024,
    uint64_t seed = 0
);
```

Convenience overload. Constructs a `LocalBackend` with `SimType::STATEVECTOR` and delegates to the primary `run_iterative` method.

**Example:**

```cpp
// Prepare |1⟩⊗3
lindblad::QuantumCircuit input(3);
input.x(0).x(1).x(2);

// Run semi-classical forward QFT
auto result = lindblad::algorithms::QFT::run_iterative(input, 2048);
// result.backend_result.counts holds the 3-bit measurement histogram.

// Use an explicit DensityMatrix backend (to test noise effects)
lindblad::backends::LocalBackend::Config cfg;
cfg.simulator = lindblad::backends::LocalBackend::SimType::DENSITY_MATRIX;
lindblad::backends::LocalBackend dm_backend(cfg);

auto dm_result = lindblad::algorithms::QFT::run_iterative(input, dm_backend, 1024);
```

## Related Pages

- [docs/algorithms/qft.md](../algorithms/qft.md)
- [docs/api/backends.md](backends.md)
