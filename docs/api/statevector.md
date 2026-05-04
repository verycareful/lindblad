# Statevector API Deep Dive

This page documents the public `lindblad::Statevector` class.

## Header and Namespace

- Header: `include/lindblad/statevector.hpp`
- Namespace: `lindblad`

## Class Overview

`Statevector` stores a quantum state in a structure-of-arrays layout with
separate aligned real and imaginary buffers. Qubit index `q` corresponds to
bit position `q` in the amplitude index (qubit 0 is the least significant bit).

The class is move-only; copy construction and copy assignment are disabled.

## Construction and Initialization

```cpp
explicit Statevector(int n_qubits);
```

Behavior:

- Accepts `n_qubits` in [0, 30]
- Throws `std::invalid_argument` outside that range
- Allocates aligned buffers and initializes to |0...0>

Initialization helpers:

- `initialize()` resets to |0...0>
- `initialize_basis(k)` sets |k> and throws if `k >= dim`

## Amplitude and Probability Access

- `amplitude(index)` returns the complex amplitude at `index`
- `amplitudes()` returns a full vector copy of amplitudes
- `probability(index)` and `probabilities()` compute |amp|^2

No normalization check is enforced; callers are responsible for maintaining
normalized states.

## Norm and Normalization

- `norm_sq()` and `norm()` compute the squared norm and norm
- `normalize()` scales to unit norm and throws if norm < 1e-15

## Inner Product

```cpp
Complex128 inner_product(const Statevector& other) const;
```

- Throws `std::invalid_argument` if dimensions differ
- Computes sum_i conj(this_i) * other_i

## Measurement Sampling

```cpp
std::string measure_once(uint64_t seed = 0) const;
std::unordered_map<std::string, int> sample_counts(int shots, uint64_t seed = 0) const;
```

Behavior:

- `seed == 0` uses `std::random_device` to seed the RNG
- `measure_once` does a linear scan of cumulative probability
- `sample_counts` precomputes cumulative probabilities and uses `lower_bound`
- Bitstrings are returned MSB-first (leftmost char is the most significant bit)

## Cloning and Debug Output

- `clone()` returns a deep copy of the statevector
- `to_string(precision)` prints up to 32 non-zero basis states
  with a ket-style label and probability annotation

## Example

```cpp
#include "lindblad/statevector.hpp"
#include "lindblad/gates.hpp"

using namespace lindblad;

int main() {
    Statevector sv(2);
    gates::apply_h(sv, 0);
    auto counts = sv.sample_counts(1000, 42);
    return counts.empty() ? 1 : 0;
}
```

## Related Pages

- [docs/APIOverview.md](../APIOverview.md)
- [docs/api/simulators.md](simulators.md)
