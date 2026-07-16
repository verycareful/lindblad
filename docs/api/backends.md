# LocalBackend API Deep Dive

This page documents the public `lindblad::backends::LocalBackend` API for executing quantum circuits on local simulators.

## Header and Namespace

- Header: `include/lindblad/backends/local_backend.hpp`
- Namespace: `lindblad::backends`

## Overview

`LocalBackend` provides a unified, high-level interface to all local quantum simulators (Statevector, DensityMatrix, Clifford, MPS). It automatically selects the optimal simulator based on circuit properties and configuration, simplifying execution workflows.

## `BackendResult`

Unified result type returned by `run()` and `run_batch()`.

Fields:

- `counts`: bitstring → count histogram (executed shots)
- `simulation_time_seconds`: wall-clock execution time
- `success`: true if execution completed without error
- `error_message`: populated if `success == false`
- `backend_name`: name of simulator used (`"lindblad_local_simulator"`)
- `shots`: number of samples collected

## `SimType` Enum

Simulator backend selection:

```cpp
enum class SimType {
    STATEVECTOR,    // Pure-state O(2^n) simulation
    DENSITY_MATRIX, // Mixed-state O(4^n) with Kraus noise
    CLIFFORD,       // Stabilizer tableau O(n^2) for Clifford circuits
    MPS,            // Approximate O(n*chi^2) for large n
    AUTO            // Auto-select (default)
};
```

**AUTO selection heuristic** (implementation-dependent):

- If `noise_model.channels` is non-empty → `DENSITY_MATRIX`
- Else if circuit uses only Clifford gates (H, S, CNOT, X, Y, Z, M) → `CLIFFORD`
- Else if circuit depth > threshold and n_qubits > threshold → `MPS`
- Else → `STATEVECTOR` (default)

## `Config` Structure

Configuration parameters for `LocalBackend`.

Fields and defaults:

- `simulator = SimType::AUTO`: which backend to use
- `max_parallel_threads = 0`: OpenMP thread limit (0 = auto-detect based on hardware)
- `max_memory_mb = 0`: memory budget in megabytes (0 = no limit)
- `mps_bond_dim = 64`: bond dimension for MPS simulator

## Construction

### Default Constructor

```cpp
LocalBackend();
```

Creates a backend with default `Config` (AUTO simulator selection).

### Explicit Configuration

```cpp
explicit LocalBackend(const Config& cfg);
```

Stores the provided `Config`.

## Methods

### `run()`

Execute a single circuit.

Signature:

```cpp
BackendResult run(
    const QuantumCircuit& circuit,
    int shots = 1024,
    uint64_t seed = 0
);
```

Parameters:

- `circuit`: the quantum circuit to execute
- `shots`: number of measurement samples (default 1024)
- `seed`: RNG seed for reproducibility (default 0)

Returns:

- `BackendResult` with measurement counts and execution metadata

Behavior:

1. Select simulator based on `config.simulator`
2. Apply `noise_model` if configured
3. Execute circuit for `shots` samples
4. Return counts histogram and timing

### `run_batch()`

Execute multiple circuits (for parallel/batch execution patterns).

Signature:

```cpp
std::vector<BackendResult> run_batch(
    const std::vector<QuantumCircuit>& circuits,
    int shots = 1024,
    uint64_t seed = 0
);
```

Parameters:

- `circuits`: vector of quantum circuits
- `shots`: samples per circuit (default 1024)
- `seed`: base RNG seed (incremented per circuit)

Returns:

- Vector of `BackendResult`, one per circuit

Behavior:

- Distributes circuits across threads if `max_parallel_threads > 1`
- Each circuit uses independent RNG stream (seed + circuit index)
- Returns results in same order as input

## Properties

### `name()`

```cpp
std::string name() const;
```

Returns: `"lindblad_local_simulator"`

### `version()`

```cpp
std::string version() const;
```

Returns the build version string derived from the CMake `LINDBLAD_VERSION_LABEL` compile definition (e.g. `"R.1.15.1"`). The value tracks `LINDBLAD_VERSION_LABEL` in `CMakeLists.txt` and cannot drift from the project version.

### `max_qubits()`

```cpp
int max_qubits() const;
```

Returns: `30` (maximum recommended qubits for accurate simulation)

Note: For larger systems or approximate solutions, use `SimType::MPS` with bond dimension tuning.

## Public Members

### `config`

```cpp
Config config;
```

Mutable configuration accessible after construction. Changes take effect on next `run()` or `run_batch()` call.

### `noise_model`

```cpp
NoiseModel noise_model;
```

Mutable noise model. When non-empty (with gate/readout channels), automatically switches to `DENSITY_MATRIX` if `SimType::AUTO` is configured.

## Usage Examples

### Basic Statevector Execution

```cpp
lindblad::backends::LocalBackend backend;
backend.config.simulator = lindblad::backends::LocalBackend::SimType::STATEVECTOR;

lindblad::QuantumCircuit qc(2);
qc.h(0).cx(0, 1);

auto result = backend.run(qc, 1024);
for (const auto& [bitstring, count] : result.counts) {
    std::cout << bitstring << ": " << count << "\n";
}
```

### Noisy Simulation with Density Matrix

```cpp
lindblad::backends::LocalBackend backend;
backend.noise_model.add_quantum_error(
    lindblad::NoiseChannels::amplitude_damping(0.01), "h");
backend.noise_model.add_quantum_error(
    lindblad::NoiseChannels::amplitude_damping(0.01), "cx");

lindblad::QuantumCircuit qc(3);
qc.h(0).h(1).h(2);
qc.cx(0, 1).cx(1, 2);

// AUTO will select DENSITY_MATRIX due to noise_model
auto result = backend.run(qc, 512);
```

### Large System with MPS

```cpp
lindblad::backends::LocalBackend::Config cfg;
cfg.simulator = lindblad::backends::LocalBackend::SimType::MPS;
cfg.mps_bond_dim = 128;

lindblad::backends::LocalBackend backend(cfg);

lindblad::QuantumCircuit qc(20);  // Large circuit
// ... build circuit ...

auto result = backend.run(qc, 100);  // Fewer shots for MPS
std::cout << "Simulation time: " << result.simulation_time_seconds << "s\n";
```

### Batch Execution

```cpp
lindblad::backends::LocalBackend backend;

std::vector<lindblad::QuantumCircuit> circuits;
for (int i = 0; i < 10; i++) {
    lindblad::QuantumCircuit qc(5);
    qc.h(0).cx(0, 1).cx(1, 2);
    circuits.push_back(qc);
}

auto results = backend.run_batch(circuits, 512);
for (size_t i = 0; i < results.size(); ++i) {
    std::cout << "Circuit " << i << " success: " 
              << results[i].success << "\n";
}
```

## Workflow Patterns

### 1. Ideal Statevector Inspection (Reference Implementation)

```cpp
backend.config.simulator = SimType::STATEVECTOR;
auto result = backend.run(circuit, 0);  // shots=0 → final state inspection
```

### 2. Noisy NISQ Simulation

```cpp
backend.noise_model = NoiseModel::realistic_ibm_5q();
backend.config.simulator = SimType::AUTO;  // Will use DENSITY_MATRIX
auto result = backend.run(circuit, 2048);
```

### 3. Approximate Large-Scale

```cpp
backend.config.simulator = SimType::MPS;
backend.config.mps_bond_dim = 64;
auto result = backend.run(circuit_20q, 100);
```

### 4. Algorithm Iteration

```cpp
for (int layer = 0; layer < num_layers; ++layer) {
    circuit_dynamic.add_layer(...);
    auto result = backend.run(circuit_dynamic, 1024);
    // Analyze, adjust parameters
}
```

## Performance Considerations

**Simulator Selection Guide**:

| Scenario | Recommended SimType | Reason |
|----------|-------------------|--------|
| n ≤ 20, ideal | STATEVECTOR | Exact, O(2^n) scalable |
| n ≤ 20, noisy | DENSITY_MATRIX | Exact mixed-state, Kraus integration |
| n ≤ 10, Clifford only | CLIFFORD | Polynomial O(n^2), fast |
| n > 20 | MPS | Approximate, O(n*chi^2) with chi=64-256 |

**Thread Scaling**:

- `max_parallel_threads = 0` (default): system CPU count
- `max_parallel_threads = 1`: single-threaded (reproducible, lower overhead)
- `max_parallel_threads = N`: limit to N threads

**Memory Requirements**:

- Statevector: ~16 · 2^n MB (complex128)
- DensityMatrix: ~64 · 2^n MB (complex128)
- Clifford: ~1 · n^2 MB (integer tableau)
- MPS: ~256 · n · chi^2 MB (chi=bond dimension)

## Compatibility

**Requires**:

- [Circuit API](circuit.md) — quantum circuit construction
- [Noise API](noise.md) — noise model integration
- [Simulators API](simulators.md) — underlying backends

**Used by**:

- [Estimator API](estimator.md) — can use LocalBackend for execution
- [Sampler API](sampler.md) — can delegate to LocalBackend
- Algorithm classes (QAOA, VQE, etc.) — for circuit execution

## See Also

- [Simulators API](simulators.md) — Detailed documentation of individual backends (Statevector, DensityMatrix, Clifford, MPS)
- [Circuit API](circuit.md) — Quantum circuit construction and manipulation
- [Noise API](noise.md) — Kraus operators, noise channels, thermal models
- [Sampler API](sampler.md) — Bitstring sampling primitive (uses LocalBackend internally)
- [Estimator API](estimator.md) — Expectation value computation (uses LocalBackend internally)
                                                                                                                                                                                                                                                                                                                                              