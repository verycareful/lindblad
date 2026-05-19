# Shor API Deep Dive

This page documents the public `lindblad::algorithms::Shor` API in detail.

## Header and Namespace

- Header: `include/lindblad/algorithms.hpp`
- Namespace: `lindblad::algorithms`

## Class Overview

`Shor` factors a composite integer N into two non-trivial factors via quantum order finding (Shor 1994). The implementation combines classical pre-screening with a QPE-based period-finding circuit using controlled modular-exponentiation unitaries.

## Options

```cpp
struct Options {
    int  n_eval_qubits = 0;   // 0 = auto: 2*⌈log₂N⌉ + 1
    int  max_attempts  = 10;
    uint64_t seed      = 0;
    backends::LocalBackend::SimType simulator =
        backends::LocalBackend::SimType::STATEVECTOR;
};
```

- `n_eval_qubits` — number of evaluation qubits in the QPE circuit. More qubits = higher phase precision = better chance of recovering the correct period. Auto mode uses 2·⌈log₂N⌉ + 1, which gives at least 2n+1 bits of precision for an n-bit number
- `max_attempts` — maximum number of random bases to try before returning failure
- `seed` — RNG seed for base selection and circuit execution. 0 = non-deterministic
- `simulator` — backend type. STATEVECTOR (default), DENSITY_MATRIX, and MPS are supported. CLIFFORD is not supported (modular exponentiation gates are non-Clifford)

## Result

```cpp
struct Result {
    uint64_t    factor;     // non-trivial factor p (0 on failure)
    uint64_t    cofactor;   // N / p (0 on failure)
    bool        success;
    int         attempts;
    std::string method;     // "trivial_gcd" | "perfect_power" | "quantum"
};
```

- `method = "trivial_gcd"` — N was even, or gcd(a, N) was non-trivial for a random or trial base
- `method = "perfect_power"` — N = b^e for some integers b ≥ 2, e ≥ 2
- `method = "quantum"` — period found via QPE circuit and continued-fraction recovery

## `factorize`

```cpp
Result factorize(uint64_t N) const;
```

Main entry point. Classical pre-screening runs first (O(1)); quantum path runs only when classical methods fail.

Execution order:
1. Reject N < 4 (`std::invalid_argument`)
2. Reject prime N (`std::invalid_argument`, deterministic Miller-Rabin)
3. Even N → `{2, N/2, true, 0, "trivial_gcd"}`
4. Perfect power N = b^e → `{b, N/b, true, 0, "perfect_power"}`
5. Trial GCD with {3, 5, 7, 11, 13}
6. Quantum: loop up to `max_attempts` random bases, each running `find_order` → GCD extraction

## `build_period_finding_circuit`

```cpp
static QuantumCircuit build_period_finding_circuit(
    uint64_t a, uint64_t N, int n_eval, int n_target
);
```

Builds the QPE circuit for order finding of base `a` modulo `N`.

- Total qubits: `n_eval + n_target`
- Eval register (qubits 0..n_eval-1): H-initialised, then IQFT
- Target register (qubits n_eval..n_eval+n_target-1): initialised to |1⟩
- Controlled-U^(2^k) gates use the LSB convention: control qubit k maps to bit 0 of the unitary matrix index
- No measurements appended — caller adds `measure_all()` as needed

## `find_order`

```cpp
static uint64_t find_order(
    uint64_t a, uint64_t N, int n_eval,
    backends::LocalBackend& backend,
    uint64_t seed = 0
);
```

Builds the period-finding circuit, runs it on `backend` with 128 shots, and recovers the multiplicative order r = ord_N(a) via continued-fraction expansion of the most-frequent measured phase.

Returns r on success, 0 on failure (zero measurement, no valid convergent, or a^r ≢ 1 mod N for all candidates).

## Example

```cpp
#include "lindblad/algorithms.hpp"

using namespace lindblad;
using namespace lindblad::algorithms;

int main() {
    Shor::Options opts;
    opts.seed = 42;
    Shor shor(opts);

    auto r = shor.factorize(15);
    // r.factor * r.cofactor == 15
    // r.success == true
}
```

## Notes and Preconditions

- N must be composite and ≥ 4
- The controlled-unitary matrices are dense permutation matrices of size 2^(n_target+1) × 2^(n_target+1). Memory usage is O(4^n_target) per gate
- Practical for N ≤ ~100 on desktop hardware
- The IQFT stage uses `QFT::build_inverse_circuit(n_eval, true)`

## Related Pages

- [docs/algorithms/shor.md](../algorithms/shor.md)
- [docs/api/qpe.md](qpe.md)
- [docs/APIOverview.md](../APIOverview.md)
