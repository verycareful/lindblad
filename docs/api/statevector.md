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

- Accepts `n_qubits` in **[1, 30]**
- Throws `std::invalid_argument` outside that range
- Allocates aligned buffers and initializes to |0...0>

Initialization helpers:

- `initialize()` resets to |0...0>
- `initialize_basis(k)` sets |k> and throws if `k >= dim`

## Amplitude and Probability Access

- `amplitude(index)` returns the complex amplitude at `index`
- `amplitudes()` returns a full vector copy of amplitudes
- `probability(index)` and `probabilities()` compute |amp|^2

## Setting Amplitudes

```cpp
void set_amplitudes(const double* real, const double* imag, size_t count,
                    ValidationOptions validation = {});
void set_amplitudes(const std::vector<Complex128>& amplitudes,
                    ValidationOptions validation = {});
```

This is the one point at which a caller hands a whole state over, so it is where
normalization is judged. The default policy is `Throw`, matching every other
physical-validity check.

- `Throw` (default): an unnormalized hand-over raises `std::invalid_argument`
  naming the residual and the tolerance
- `Fix`: the amplitudes are accepted and renormalized
- `Warn`: reported through the warning handler, then accepted unchanged
- `Ignore`: no check, at the cost of one branch

```cpp
sv.set_amplitudes(amps);                            // must already be normalized
sv.set_amplitudes(amps, {Validation::Fix});         // normalize on the way in
sv.set_amplitudes(amps, {Validation::Ignore});      // deliberately unnormalized
```

`Ignore` is the right choice when the amplitudes are not meant to be a physical
state, for instance when probing index arithmetic with a deliberately arbitrary
vector.

The policy is judged against the caller's buffer before anything is written, so
a hand-over that is refused leaves the object holding whatever it held before,
not the amplitudes that were just rejected.

## Norm and Normalization

- `norm_sq()` and `norm()` compute the squared norm and norm. Both use an
  OpenMP reduction, so their last bits depend on the thread count. They are for
  computation, not for comparing against a tolerance.
- `normalize()` scales to unit norm. It throws `std::runtime_error` when there
  is no norm to divide out, which is a zero or non-finite state. It does not
  return an unnormalized state quietly, because a caller who asked for
  normalization and received none has been told nothing.
- `is_normalized(atol)` is a predicate: it answers, and neither repairs nor
  throws. A non-finite state answers false. Defaults to `DEFAULT_PHYSICAL_ATOL`.
- `check_normalized(validation)` applies a policy to the state as it stands,
  with the same four behaviours as `set_amplitudes` above.

`is_normalized` and `check_normalized` measure through a summation that does not
depend on thread count or vector width, which is why they do not simply call
`norm_sq()`. A verdict that moved with the number of free cores would not be a
verdict.

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
