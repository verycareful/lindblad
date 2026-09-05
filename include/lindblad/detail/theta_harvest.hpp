#pragma once

// =============================================================================
// theta_harvest - capture real two-site blocks on their way into the SVD
// =============================================================================
//
// A bond split factorises the two-site block formed by contracting a gate into
// its neighbouring tensors. Benchmarking an SVD kernel against matrices that
// are not those blocks measures the wrong thing, and the shapes and spectra a
// simulation actually produces are not obvious from the outside, so this
// records them at the one place they exist.
//
// Compiled in ONLY when LINDBLAD_MPS_THETA_HARVEST is defined, which
// -DLINDBLAD_MPS_THETA_HARVEST=ON sets. In a normal build this header declares
// nothing and src/detail/theta_harvest.cpp is not in the target, so the cost
// and the API surface are both absent rather than merely unused.
//
// What is kept, and why not everything: at a bond cap of 64 a block is 128 by
// 128 complex, roughly 650 KB as hexfloat text, and a saturating run performs
// hundreds of splits. Keeping all of them is tens of megabytes of near
// duplicates. One representative per distinct shape carries the geometry, and
// the histogram carries how often each shape occurred, which together are what
// a kernel benchmark needs.
//
// Output is the format autonne reads, written independently rather than
// through autonne's own header: this has to work in a build that does not link
// autonne at all.
//
//   autonne-hexfloat 1 matrix <name> <rows> <cols> rowmajor
//   <re> <im>        one element per line, %a tokens, row-major order
//
// Not thread safe, matching the counters it sits beside: bond splits along a
// chain are strictly ordered, since each writes the tensors the next contracts.

#ifdef LINDBLAD_MPS_THETA_HARVEST

#include "lindblad/types.hpp"

#include <vector>

namespace lindblad {
namespace detail {

// Offer one two-site block to the harvest. The first block of a given shape is
// written out; later ones of that shape only advance its count. `data` is
// row-major, rows*cols elements, and is the block as the SVD will receive it.
void theta_harvest_offer(const std::vector<Complex128>& data, int rows, int cols);

} // namespace detail
} // namespace lindblad

#endif // LINDBLAD_MPS_THETA_HARVEST
