# Dispatch Helpers

This page documents `qpp::SoftDispatchResult` and the dispatch-oriented helpers used with QAOA and MAQAOA outputs.

## Purpose

The dispatch helpers convert sampled bitstring distributions into a more useful fractional or integer decision structure.

They are intended for optimization problems where the sampled distribution represents generator commitment, resource allocation, or similar binary decisions.

## Theory Summary

A sampler returns a histogram over bitstrings.

`SoftDispatchResult` turns that histogram into:

- a soft assignment vector representing fractional commitment
- a best bitstring from the highest-probability sampled state
- optional rounded and greedy dispatch solutions

This makes it easier to move from quantum samples to a practical dispatch schedule.

## Required Inputs

- A sampler count map as `std::unordered_map<std::string, int>`
- The total number of shots
- Optional generator capacities and demand when using greedy dispatch

## How to Invoke

Include the header:

```cpp
#include "qpp/dispatch.hpp"
```

Construct the helper from raw counts:

```cpp
using namespace qpp;

std::unordered_map<std::string, int> counts = {
    {"101", 40},
    {"111", 12},
    {"001", 8}
};

SoftDispatchResult result(counts);
result.compute();

std::string thresholded = result.threshold_round(0.5);
```

If you have generator capacities and demand, use greedy dispatch:

```cpp
std::vector<double> capacities = {5.0, 4.0, 3.0};
double demand = 7.0;
auto selected = result.greedy_dispatch(capacities, demand);
```

## Header Include Instructions

Use:

```cpp
#include "qpp/dispatch.hpp"
```

## Simulator and Primitive Dependencies

These helpers operate on existing sampler outputs rather than directly on a simulator.

They are commonly used after:

- QAOA
- MAQAOA
- other binary optimization routines

## Public API Details

### `SoftDispatchResult`

Fields:

- `counts`: raw sampled counts
- `total_shots`: total number of shots
- `soft_assignment`: fractional assignment vector computed from the counts
- `best_bitstring`: highest-probability measured bitstring
- `best_probability`: probability of the best bitstring

Methods:

- `compute()` fills the derived fields from the raw counts
- `threshold_round(double threshold = 0.5)` returns a rounded MSB-first bitstring
- `greedy_dispatch(const std::vector<double>& generator_capacities, double demand)` returns selected generator indices
- `expected_cost(const std::function<double(const std::string&)>& cost_fn)` computes expected cost under the sampled distribution
- `top_k(int k)` returns the `k` most probable bitstrings and probabilities

## Example Code

```cpp
#include "qpp/dispatch.hpp"

using namespace qpp;

int main() {
    std::unordered_map<std::string, int> counts = {
        {"110", 30},
        {"100", 20}
    };

    SoftDispatchResult result(counts);
    result.compute();

    std::cout << "best: " << result.best_bitstring << '\n';
    std::cout << "rounded: " << result.threshold_round() << '\n';
}
```

## Return Values and Outputs

- `compute()` populates the soft assignment and best bitstring fields
- `threshold_round()` returns a binary string
- `greedy_dispatch()` returns selected generator indices
- `expected_cost()` returns a scalar objective estimate
- `top_k()` returns the most likely sampled states and their probabilities

## Exceptions and Failure Modes

Common issues include:

- empty counts maps
- count strings whose length does not match the dispatch problem size
- capacity vectors that do not match the number of binary variables
- demand values that cannot be met by the available capacities

## Common Pitfalls

- The helper works on sampled distributions, not on the underlying Hamiltonian directly.
- `threshold_round()` and `greedy_dispatch()` solve different decision problems.
- The bitstring convention remains MSB-first.

## Testing Notes

Dispatch-related behavior is exercised indirectly through the microgrid and MAQAOA tests.

## Related Source Files

- [docs/api/dispatch.md](../api/dispatch.md)
- [include/qpp/dispatch.hpp](../../include/qpp/dispatch.hpp)
- [tests/test_maqaoa_microgrid.cpp](../../tests/test_maqaoa_microgrid.cpp)
- [tests/test_maqaoa_20qubit.cpp](../../tests/test_maqaoa_20qubit.cpp)
