# `entmax/solve_row.cuh` — the per-row exact solve components

These are the arithmetic of `factor.cu` and `entmax.cu` with the **operand fetch lifted
out**. A solve here takes a *logit source* — any object with
`float operator()(int col) const`, evaluated by the lane that owns `col` — instead of a
global base pointer plus a stride tuple. Two sources exist:

* `TileLogits` — one row of the CTA's fp32 logit tile in shared memory, which is what the
  fused producer hands it;
* `PlaneLogits` — a strided global plane, addressed exactly as `factor.cu` addresses it.

Nothing else changes. The phases, the sentinel padding, the packed-key reduction that
picks the greatest valid `k` and carries its `tau`, the bf16 support boundary — all of it
is the same arithmetic, which is the property that lets the fused kernel and the
standalone kernels be compared on support sets rather than on tolerances.

## Three components

| function | what it solves |
|---|---|
| `solve_entmax_row<LW, IPT, ALPHA15>` | the exact `alpha in {1.5, 2}` threshold-after-sort solve; returns the lane's support mask and leaves amplitudes in `p` |
| `solve_softmax_row<LW, IPT>` | the full-support level: shifted exponentials over one warp sum |
| `solve_union_row<LW, IPT, ALPHA15>` | one entmax solve on the midpoint determines a shared support, then the sigmoid split sends `read` and the log-domain renormalization sends `write` |

`solve_union_row` composes `solve_entmax_row` rather than restating it — its phases B/C/D
*are* the entmax solve, fed a midpoint instead of a logit, and that identity was already
the design of record in `entmax.md`. It is expressed here for the first time as a call.

## No candidate mask

The producer's launch plan never passes one (`_Plan.add` issues
`entmax_factor_forward(logits, None, ...)` on every arm), so `valid_count == width` and
the empty-row branch of `factor.cu` cannot be taken. The masked surface —
`production_factor`, the standalone entmax op — keeps its own kernel and its own
empty-row autograd contract.

## Register discipline

Every collective is warp-scope and shuffle-based, so register demand stays
`O(items-per-lane)` and shared memory is needed only for the merge sort's scratch. The
`static_assert`s on each `TempStorage` size are what turn a future CUB that stops
resolving to a pure-shfl specialization into a compile error rather than a race.
