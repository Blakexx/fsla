# `csrc/rola/src/entmax/factor.{cuh,cu}` — the producer's independent (per-side) solves

Hand CUDA over CUB warp primitives: the four kernels that turn one side's routing
LOGITS into route VALUES, for the two activations a level can carry on its own —
the exact α=1.5/2.0 threshold solve (with §4.1's packed per-leaf support bitmap)
and softmax. `factor.cuh` is the host declaration set, `factor.cu` the
implementation and its dispatch.

Design of record: this document. The union split's own solve, and the warp-row
vocabulary this file shares with it, are [`entmax.md`](entmax.md).

Gates: `tests/integration/test_production_levels.py` (the fp64 oracle, forward AND
VJP, across `shared_solve` / `mixed_per_level` / `dense_only`),
`tests/unit/test_entmax_production.py` (the masked `production_factor` surface, the
support word's bit-31 and tail-word contracts, the tile-major arena),
`tests/integration/test_execution_identity_gates.py` (which routing family reaches
which kernel).

Symbols are `entmax.md`'s: a *stream* is one (batch, head)-like routing stream, a
*token* one row of the solve, `width` the level's branch width with `col` indexing
it, `LW` the logical warp width, `IPT` the items per lane (`W_PAD = LW * IPT`),
`tau` the threshold, *support* the set of columns above it.

## <a id="provenance"></a>1. Where these came from, and what that constrains

These four were Triton (`rola/routing/entmax/production.py`'s
`_production_fwd_kernel`, `_production_bwd_kernel` and their softmax pair) until
P73 S5. The port is why `rola/` now imports no Triton, why `pyproject.toml` no
longer depends on it, and why `build.md`'s "not `fla`, not `triton` — the kernels
are CUDA" is true rather than aspirational. Decisively, it is what lets P65 train
through this path: a Triton backward in the training path is the exact thing the
CUDA-engine-first ruling forbids.

**The port is NOT judged against Triton.** The two lowerings associate the prefix
scan differently — Triton scans the padded row with `tl.cumsum`, this one
accumulates serially within a lane and then runs one `WarpScan` across lanes — so
the fp32 sums are not bitwise equal and a kernel-vs-kernel gate would be measuring
the wrong thing (the matching-the-naive antipattern). The governing gate is the
fp64 reference, and it asserts the SUPPORT SET identical, not merely close.

## <a id="the-solve"></a>2. The forward: the union solve's phases B/C/D, fed one tensor

`factor_forward_kernel` is `union_forward_kernel` with the union split removed. Its
phases are that kernel's, unchanged and documented there
([forward-phases](entmax.md#forward-phases)): row max/min, the finite row-relative
sentinel that pads without putting an infinity in the scan, `WarpMergeSort`
descending, the `(Σz, Σz²)` `WarpScan`, the packed-key `Max` reduction that selects
the greatest valid `k` and carries its `tau` in the same 64-bit word, and the
amplitude `((α−1)(z − tau))_+` — squared at α=1.5.

What is NOT there: phase A's midpoint fold and phase E's write normalization. What
IS there and is not in the union kernel: the candidate mask.

### <a id="mask"></a>2.1 The mask, and why it is a per-ROW quantity

`production_factor` is a public surface and takes an optional boolean candidate
mask, laid out like the logits. A masked element is not a candidate: it leaves the
support by construction rather than by a correction applied afterwards.

The consequence that shapes the kernel is that the member count is then a
reduction, not `width`:

* the row max and min are taken over the CANDIDATES, and a fully-masked row has
  neither — `row_max` is defined to 0 there and the sentinel to −1, so the sort and
  the scan still run on finite values;
* the greatest-valid-`k` search runs over `rank < valid_count`, not `rank < width`,
  which is what keeps a masked column from being admitted by a prefix that never
  included it;
* an empty row produces `chosen = −1`, hence `tau = 0`, hence no support and no
  amplitude — the autograd contract
  `test_production_factor_empty_rows_preserve_autograd_contract` pins.

The mask pointer is NULL on every plan-level path (`_launch_factor_into` and the
batched entry), and the count reduction is skipped there — one uniform branch on a
pointer, not a codegen axis, because a mask arm would double the instantiation
matrix to buy a prologue.

### <a id="support"></a>2.2 The support word

Identical to the union solve's, including the two things that are easy to get
wrong: support is the set of POST-CAST bf16 nonzeros and never the pre-quantization
solve ([bf16-support-boundary](entmax.md#bf16-support-boundary)) — which holds even
when this level's values are STORED in fp32, because the word describes what the
consumer will read — and the 32-token word is packed by the epilogue butterfly over
the lanes at or above `LW`
([epilogue-butterfly](entmax.md#epilogue-butterfly)), one CTA owning exactly one
word ([frozen-support-word](entmax.md#frozen-support-word)).

## <a id="backward"></a>3. The backward: a rank-1 correction, support as an INPUT

The α-entmax VJP is `dz = s ⊙ (dp − ⟨s, dp⟩ / Σs)` with `s = p^(2−α)` on support
and 0 off it — `1` at α=2, `√p` at α=1.5. Softmax's is the same shape with `s = p`
and the denominator identically 1, which is why `softmax_backward_kernel` carries no
support word and no `α`.

Two facts make this a five-line kernel rather than a Jacobian:

* **Support is an INPUT**, unpacked from the same frozen word the forward emitted,
  so the two directions cannot disagree about the support set by construction. Off
  support every term is an exact zero by multiplication, which is what
  `assert_off_support_zero` asserts at `rtol=0` — the FORM of the VJP, not a
  magnitude.
* **The two sums are one reduction.** `(Σ s·dp, Σ s)` travel as a `Pair` through a
  single `WarpReduce`, exactly as the union backward's does.

It RE-READS `values` and the cotangents in the second pass rather than staging them
in `IPT`-sized registers. That is the union backward's MEASURED trade
([backward](entmax.md#backward)) and not a stylistic preference: at `IPT = 8` the
staged form is what puts that kernel over its 64-register bound. Re-reading keeps
this family at zero spill across all 96 instantiations, which `tools/ratify.py`
enforces.

`accumulate` selects `atomicAdd` over a store. It is a parameter and not a fact
about the tensor because a TIED level runs two solves into ONE gradient buffer, and
only the second of them may accumulate — a property of the walk, not of the memory.

## <a id="batched-levels"></a>4. One launch across levels

`producer.py` has always claimed "one packed projection and one batched solve", and
until P73 S5 only the first half was true: the solve was a Python `for` loop over
`routing.levels`, one launch and three allocations per level.

The launch shape here makes the claim true. Each entry takes BASE tensors plus
per-level element offsets; `blockIdx.z` indexes a `LevelTable` that travels in the
kernel's parameter space (448 B at `MAX_BATCHED_LEVELS = 16`, against a 4 KB
floor), so batching costs no device table, no H2D copy and no indirection.

**The grouping rule is exactness, not convenience.** Levels may share a launch only
when they share the width CLASS, the α, the operator and the output dtype — that
is, only when they select the same template arm. The class is load-bearing: `LW`
and `IPT` set how the sort permutes and how the prefix scan associates, so running
a width-3 level in a width-64 arm would change the fp32 rounding of `tau` and could
move a support boundary. Widths inside a launch may differ (3 and 4 are one class);
their padded widths may not. What batching changes is the launch SHAPE; the
arithmetic each row performs is the arithmetic the per-level loop performed.

The host `build_table` refuses a mixed-class table rather than padding to the
largest, and the caller (`production.py`) is what groups the plan walk into tables.
