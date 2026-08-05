// THE PRODUCER'S UNION ENTMAX SOLVE -- host declarations.
//
// **WHY THIS EXISTS AS CUDA AT ALL.** The Triton lowering of this solve was measured at
// 255 registers WITH SPILLS at 16.67% occupancy until hand-restructured, and even after
// the restructure (115 reg, 33.33%) it retained an uncontrollable `tl.sort` scratch
// layout with 75% bank-conflict-excessive wavefronts, and it is a JIT against a
// closed-world AOT policy that forbids exactly that. See
// `rola-scratch/workflows/20260801-forward/producer-p2b.md` and `producer-p3.md`.
//
// **THE STRUCTURAL IDEA: ONE ROW PER (LOGICAL) WARP.** Every collective in the entmax
// solve -- the sort, both prefix scans, every reduction -- runs ALONG ONE TOKEN'S ROW.
// Triton had to express them as block-scope collectives over a 2D (ROW_TILE, BLOCK_COLS)
// tile, which is what (a) welded register demand to the CTA tile, so `num_warps` was not
// a lever, and (b) produced the bank-conflicted shared scratch. Assigning one row to one
// warp makes every collective warp-scope and shuffle-based: register demand becomes
// O(items-per-lane) and decouples from the warp count, the solve needs ZERO shared
// memory and ZERO block barriers, and the sort's conflicted scratch is not tuned but
// ELIMINATED. CUB's warp primitives are the library half of the spec's
// "hand-roll semantics, never primitives" rule.
//
// **THE SUPPORT WORD LAYOUT IS FROZEN** (§4.1's per-leaf write bitmap, nine downstream
// consumers). One CTA therefore still owns exactly the 32 tokens of one support word, so
// the `1 << (token % 32)` packing needs no global atomics and no zero-init pass.
#pragma once

#include <cstdint>

#include <torch/types.h>

namespace rola {
namespace entmax {

// One launch = one union routing LEVEL = one width class = one alpha, exactly as today
// (`_launch_union_fwd` is called inside the per-level loop, and Triton already
// specialized a separate binary per `next_power_of_2(width)`). Launch count is
// unchanged by this migration.
//
// `read_values`/`write_values` accept BOTH route-value layouts through one stride
// 5-tuple `(stream, token, col, tile, lane)`: a rank-3 dense level has tile=lane=0, a
// rank-4 tile-major arena slice has token=0. The single address expression
// `stream*s + tok*t + (tok>>5)*tile + col*c + (tok&31)*lane` is correct for both, which
// is why layout is NOT a codegen axis here (Triton needed a `tl.where(TILE_MAJOR, ...)`).
void union_forward(const torch::Tensor& read_logits, const torch::Tensor& write_logits,
                   torch::Tensor& midpoint_values, torch::Tensor& support_words,
                   torch::Tensor& read_values, torch::Tensor& write_values, double alpha);

void union_backward(const torch::Tensor& read_logits, const torch::Tensor& write_logits,
                    const torch::Tensor& midpoint_values, const torch::Tensor& support_words,
                    const torch::Tensor& read_values, const torch::Tensor& write_values,
                    const torch::Tensor& d_read, const torch::Tensor& d_write,
                    torch::Tensor& d_read_logits, torch::Tensor& d_write_logits, double alpha);

}  // namespace entmax
}  // namespace rola
