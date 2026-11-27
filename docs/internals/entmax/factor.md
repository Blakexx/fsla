# `csrc/fsla/src/entmax/factor.{cuh,cu}` — the producer's independent (per-side) solves

Hand CUDA over CUB warp primitives: the four kernels that turn one side's routing
LOGITS into route VALUES, for the two activations a level can carry on its own —
the exact α=1.5/2.0 threshold solve (with §4.1's packed per-leaf support bitmap)
and softmax. `factor.cuh` is the host declaration set, `factor.cu` the
implementation and its dispatch.

Design of record: this document. The warp-row vocabulary and the threshold these kernels speak are [`entmax.md`](entmax.md).

Gates: `tests/integration/test_production_levels.py` (the fp64 oracle, forward AND
VJP, across `tied` / `mixed_per_level` / `dense_only` / `grouped`),
`tests/unit/test_entmax_production.py` (the packed-projection read, the dense levels' complete
write, the batched solve against the per-level loop, the plan's launch count).

Symbols are `entmax.md`'s: a *stream* is one (batch, head)-like routing stream, a
*token* one row of the solve, `width` the level's branch width with `col` indexing
it, `LW` the logical warp width, `IPT` the items per lane (`W_PAD = LW * IPT`),
`tau` the threshold, *support* the set of columns above it.

## <a id="logit-dtype"></a>
## The logit plane's dtype is a template axis

The kernels keep a `LogitT` template parameter, and the contract holds bf16 alone (`hardware/kernels/<kernel>/parts/call.json`: the library sends the solves
nothing else, `fsla/ops/produce.py`'s `projection_dtype`; a plane of another dtype is refused naming the fact), so every arm reads a bf16 plane. Every
kernel in this family widens a logit to fp32 **in register** before it does any arithmetic.

**Why bf16 is the shipped one.** The router GEMM runs bf16 operands on the tensor cores,
where cuBLAS's output is bf16 — there is no bf16-operand / fp32-output GEMM to ask for.
Measured on GA102 at the pinned cell (`B=2, T=2048, hidden=512, H=8, C=257`): fp32 SIMT
0.4516 ms against bf16 tensor core 0.1638 ms, **2.76x**, and the class ratio on a clean
8192³ GEMM is 69.9 against 23.9 TFLOP/s. Widening the plane back to fp32 before the
solve would cost ~0.055 ms and one launch, which is why the widening is a register
operation inside these kernels instead of a pass over memory.

**What it does NOT relax.** The support-exactness bar binds the SOLVE, and the solve
still runs in fp32 on whatever it is handed. What the bf16 plane changes is the LOGITS,
and that change is gated at the seam: the reference solve is fed the production logits
and the support sets must agree bitwise
(`tests/integration/test_production_levels.py::test_the_solve_seam_is_exact_on_the_production_logits`).
The support difference against a wholly-fp64 projection is purpose-2 accounting and is
reported, not gated.

**The gradient rides the same plane.** `d_logits` is `LogitT` too — autograd requires a
cotangent in the input's dtype, and a fp32 gradient plane feeding a bf16 GEMM backward
would be a widening with nothing on the other end of it.

## <a id="provenance"></a>1. Why these are CUDA, and how they are judged

These four kernels are CUDA so that `fsla/` imports no Triton and the training path
holds no Triton backward, which the CUDA-engine-first ruling forbids.

**The solve is judged against the fp64 reference, never against another lowering.** A
different lowering associates the prefix scan differently (a padded-row `tl.cumsum`
against this one's serial per-lane sums and one `WarpScan` across lanes), so the fp32
sums are not bitwise equal and a kernel-vs-kernel gate would measure the wrong thing (the
matching-the-naive antipattern). The governing gate asserts the SUPPORT SET identical,
not merely close.

## <a id="the-solve"></a>2. The forward: the threshold solve of one logit tensor

`factor_forward_kernel`'s
phases B and C are the header function `threshold<LW, IPT, ALPHA15>`, called on
the logit row ([`entmax.md#threshold`](entmax.md#threshold)): row max/min,
<a id="sentinel"></a>the finite row-relative sentinel that pads without putting an infinity in the
scan, the descending `RowSort`, the `(Σz, Σz²)` `WarpScan`, and <a id="packedkey"></a>the
`packed_key` `Max` reduction that selects the greatest valid `k` and carries its `tau` in
the same 64-bit word. Phase D
([forward-phases](entmax.md#forward-phases)) is the amplitude `((α−1)(z − tau))_+`,
squared at α=1.5.

<a id="sort-bytes"></a><a id="stream32"></a><a id="stream32-2"></a><a id="stream32-3"></a><a id="stream32-4"></a>
<a id="nstreams"></a><a id="nstreams-2"></a><a id="nstreams-3"></a><a id="nstreams-4"></a>The
shared-memory slot, its alias
with the support staging, and the stream split are the header's
([`entmax.md#smem-slot-alias`](entmax.md#smem-slot-alias),
[`entmax.md#stream-split`](entmax.md#stream-split)).

<a id="near-line-200"></a>A level that stores two sides from one solve (`values_second`: a read entry `tied()` to the write side) hands
read and write the SAME amplitudes, never a second solve; `values_second` absent means the level stores one side.

<a id="read-heads"></a>**The read side is `G` heads a write head** (`fsla/routing/entmax/production.py`). The logits are one plane, a stream a
write head (`b * H_w + w`), and every solve reads its own column span of it (`fsla.routing.projection.RouterLayout`). The read plane is
`[B * H_w * G, T, ·]`, query head `w * G + g` at row `(b * H_w + w) * G + g`, so the public `[B, T, H, ·]` levels are a permute of it; read head
`g`'s solves write the strided view of its rows, whose streams are the logits'. A level's write solve writes the first tied head's rows as its
`values_second`; a further tied head's rows are copied from the write side's after the launches. Each solve writes its own support words: the
write side's plane at `[B * H_w, ·, ·]` and the read side's at `[B * H, ·, ·]`, a read head's columns its view of the read plane's.

### <a id="candidates"></a>2.1 The candidates are the row's width

<a id="factorforward"></a>Every column below the level's `width` is a candidate, and the member count passed to
`threshold` is `width`: the entries take no candidate mask, so the kernel runs no per-row count
reduction. The row max and min, the sort and the greatest-valid-`k` search all run over the
`width` members, and a padded lane (`col >= width`) is a non-member.

### <a id="support"></a>2.2 The support word

<a id="near-line-204"></a><a id="near-line-212"></a>The two things that are
easy to get
wrong: support is the set of POST-CAST bf16 nonzeros and never the pre-quantization
solve ([bf16-support-boundary](entmax.md#bf16-support-boundary)) — the word describes what the
consumer will read — and the 32-token word is packed by the epilogue butterfly over
the lanes at or above `LW`
([epilogue-butterfly](entmax.md#epilogue-butterfly)), one CTA owning exactly one
word ([frozen-support-word](entmax.md#frozen-support-word)).

## <a id="backward"></a>3. The backward: a rank-1 correction, support as an INPUT

<a id="launch-bounds"></a>The α-entmax VJP is `dz = s ⊙ (dp − ⟨s, dp⟩ / Σs)` with `s = p^(2−α)` on support
and 0 off it — `1` at α=2, `√p` at α=1.5. Softmax's is the same shape with `s = p`
and the denominator identically 1, which is why `softmax_backward_kernel` carries no
support word and no `α`.

Two facts make this a five-line kernel rather than a Jacobian:

* **Support is an INPUT**, unpacked from the same frozen word the forward emitted,
  so the two directions cannot disagree about the support set by construction. Off
  support every term is an exact zero by multiplication — the FORM of the VJP, not a
  magnitude.
* **The two sums are two butterflies.** `Σ s·dp` and `Σ s` each go through `warp_sum`:
  `__shfl_xor_sync` over the lane bits below `LW` with the whole warp shuffling, so every
  lane of a logical warp ends with the sum, with no mask, no branch and no broadcast after.
  A reduction that branches inside the trip loop is a convergence barrier there.

<a id="factorbackward"></a>Both backward entries STORE into `d_logits`; a TIED level's second cotangent (`d_values_second`)
is summed into `dp` inside the one launch, so no launch adds into a gradient another wrote. Where more than one read head ties a level, the
further heads' cotangents are summed into the second one before the launch (in the stored dtype); at one tied head the launch is the tie's
alone.

<a id="backward-stages"></a>**The operands are STAGED a token ahead and read from memory once.** A trip issues the next token's loads before it reduces and stores its own, so a load's latency passes under the reduction and not in front of it; the reduction's terms stay in registers for the stores, and the entmax backward reads its support words once a block (a word holds the block's 32 tokens). MEASURED on an H100 at 65,536 tokens, 8 streams, two levels: the backward takes 0.39 to 0.47 of the time of a form that read each load in the trip that issued it and every value twice (widths 48, 64 and 256; alpha 1.5, alpha 2 and softmax; tied and untied). Three things the form rests on:

* **A staged load is kept RAW** (`BackwardReads::issue`). The conversion from bf16 is itself a read, and a warp stops at the first instruction that reads a load still in flight.
* **No load is guarded.** An item with no entry -- a column past the level's width, a token past the call's end -- reads the block's first entry (`fv`, `fd`, `f2`) and a select zeroes it. A guarded load is a branch on a lane's own value in front of the warp's reduction; with none, every thread reaches the reduction, and only the stores are guarded. The square root is taken of every item and selected after for the same reason.
* **A row's offsets are walked, never multiplied.** Inside a block a token's offset is the first token's plus `local * (token + lane)` strides (the block's tokens share one tile), so a tensor keeps one row offset stepped a trip and an item adds a column step. A product a column is the same in every trip, and the toolkit keeps each such product in a register for the whole kernel.

The eight-item arms take the backward's register class (`backward_min_blocks(IPT)`, [`entmax.md`](entmax.md#residency)): they hold a token's raw operands and its terms together. An arm of one or two trips is unrolled whole (`UNROLL`). A block of more logical warps than its word has tokens (a logical warp of two lanes) idles whole warps, which return at once. What the build's rules over loops find is declared in `arms.bzl` at the exact count: nothing in the softmax backward; in the entmax backward no load read in its trip, and 2 to 34 convergence instructions an arm whose cause is OPEN -- the counts fit the slow paths of the trip's division and of each item's square root (4 an item where alpha is 1.5; the softmax backward, with the same guarded stores and neither, holds none), and no listing was read to settle it.

TWO FORMS FAILED, recorded so they are not rebuilt. The trips unrolled whole, a straight line of eight: the toolkit computed every trip's offsets ahead and held them, the eight-item arms spilled at 64 and at 80 registers a thread, and a one-item arm held 59 live. Two tokens held ahead at fixed registers: where a level is tied a token is three raw arrays, 48 of an eight-item arm's 80 registers, and it spilled.

## <a id="batched-levels"></a>4. One launch across levels

The producer runs one packed projection and one batched solve, not a loop of one launch
and three allocations a level. Each entry takes BASE tensors plus
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

The matcher refuses a mixed-class launch rather than padding to the largest
(`entmax_class_refusal`, [`../common/plan/entmax_launch.md`](../common/plan/entmax_launch.md); the host `build_table` raises it), and the caller (`production.py`) is what groups the plan walk into tables;
its class and its level bound are the matcher's, which it reads from the extension.

## <a id="stored-form"></a>5. The stored form is what the solve WRITES

The solve's levels have one consumer, the producer's torch path
(`fsla.ops.produce`'s reference, through `production_routing_levels`), which folds
them and quantizes them into the bundle's rows (`RouteFactors.from_levels`); it reads
`bf16`. So `OutT` is `__nv_bfloat16` on that path and the solve writes the stored
representation directly; there is no fp32 plane and no narrowing pass between them. The
entmax and softmax arms are both instantiated for it
(`factor.cu::check_values`), and the value dtype is read off
the output tensor rather than templated at the host entry, so the choice is one
allocation in `fsla/routing/entmax/production.py`.

**Why writing bf16 is not the same as rounding to bf16.** The solve's amplitudes come
from a closed form whose zeros are structural: an entry outside the support is assigned
exactly `0`, and an entry inside it is `((alpha-1)(z-tau))^2` clamped at zero, which is
nonzero whenever the support test passed at fp32. Writing that value narrowed to bf16
preserves both directions of the support test -- `0` stays `0`, and a positive fp32
value below bf16's smallest subnormal cannot arise, because the smallest amplitude the
solve can produce and still call supported is bounded away from it by the threshold
arithmetic. That identity is the self-masking theorem's storage precondition
(`docs/architecture/self-masking-theorem.md`) and it is gated without tolerance by
`tests/integration/test_production_levels.py`'s oracle rows, which compare the stored levels' support sets exactly against the fp64 oracle's over four stress
fixtures x four routing shapes x two alphas. The stored form is bf16 alone: the contract holds it, and `production.py` names it once, where it allocates the planes.
