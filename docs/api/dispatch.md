# SoftDispatchResult API Deep Dive

This page documents the public `lindblad::SoftDispatchResult` API.

## Header and Namespace

- Header: `include/lindblad/dispatch.hpp`
- Namespace: `lindblad`

## Type Overview

`SoftDispatchResult` summarizes a sampler count distribution into:

- a fractional soft assignment per bit
- a best bitstring with its probability
- optional rounded or greedy dispatch outputs

The constructor computes `total_shots` and calls `compute()` immediately.

## Fields

- `counts`: raw sampled counts (bitstring -> count); keys follow the project convention (qubit 0 is the rightmost character)
- `total_shots`: sum of all counts
- `soft_assignment`: fractional assignment indexed by QUBIT/GENERATOR: `soft_assignment[i]` belongs to qubit i (it reads key position n-1-i). Frozen in R.1.12; it previously indexed raw string positions, i.e. the reversed generator order
- `best_bitstring`: most probable counts key (qubit 0 rightmost)
- `best_probability`: probability of `best_bitstring`
- `threshold_round()` returns an INDEX-ORDER string (`result[i]` = qubit/generator i), not a counts key; reverse it if a key is needed

## Construction

```cpp
explicit SoftDispatchResult(const std::unordered_map<std::string, int>& counts_in);
```

Behavior:

- sums counts into `total_shots`
- calls `compute()` to fill derived fields

## Methods

### `compute()`

- infers bitstring length from the first key in `counts`
- populates `soft_assignment`, `best_bitstring`, and `best_probability`
- returns early if `counts` is empty or `total_shots == 0`

### `threshold_round(double threshold = 0.5)`

- returns an MSB-first bitstring
- uses `>= threshold` to mark a bit as `1`

### `greedy_dispatch(const std::vector<double>& generator_capacities, double demand)`

- orders indices by descending `soft_assignment`
- selects until `supplied >= demand`
- returns selected indices in the chosen order
- throws `std::invalid_argument` if capacity size does not match assignment size

### `expected_cost(const std::function<double(const std::string&)>& cost_fn)`

- returns `0.0` if `total_shots == 0`
- otherwise computes the expected value under the sampled distribution

### `top_k(int k)`

- returns the `k` most probable bitstrings and their probabilities
- assumes `total_shots > 0` to avoid division by zero

## Exceptions and Preconditions

- `greedy_dispatch` throws `std::invalid_argument` if the capacity vector length does not match the bitstring length
- `top_k` expects `total_shots > 0` (non-empty counts)

## Example

```cpp
#include "lindblad/dispatch.hpp"

using namespace lindblad;

int main() {
    std::unordered_map<std::string, int> counts = {
        {"110", 30},
        {"100", 20}
    };

    SoftDispatchResult result(counts);
    std::string rounded = result.threshold_round(0.5);
    auto top = result.top_k(2);
    (void)rounded;
    (void)top;
    return 0;
}
```

## Related Pages

- [docs/algorithms/dispatch.md](../algorithms/dispatch.md)
- [docs/APIOverview.md](../APIOverview.md)
