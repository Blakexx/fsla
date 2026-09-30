# `csrc/rola/src/decode/decode_fold.cuh` — the decode step's operand pass

Every per-token operand a decode step consumes, read from the internal format (docs/api.md
section 2.1), normalized where the producer did not already leave the level on the simplex,
and staged in the SHARED MEMORY of the CTA that is about to consume it.

Symbols: `B` batch, `H` heads, `BH = B * H`, `bh = b * H + h`; `D` levels; `b_l` level
`l`'s width; `d_v` the value width. The operands are the read and write coefficient rows
(`[B, 1, H, b_l]` each, u8 or f16 unit-max digits in the producer's own layout) with their
fp32 row maxima (`[B, 1, H]` each), the write gain (`[B, 1, H]`) and the token's value
digits (`[B, 1, H, d_v]` s8, the u8 low byte beside it at two digits, and the fp32 token
scale).

---

## <a id="why-a-header"></a>1. Why the fold is not a launch

The operands are TINY — `2 * sum_l b_l + d_v + 1` values per batch-head, kilobytes at any
shipped topology. The fold was never about moving bytes faster; it is about not being a
launch.

A decode step is launch-bound. Measured on an RTX 3080 Ti at `B = 2`, widths `(16, 16)`,
`d_v = 64`, paged backing: the torch operand chain was eight launches and 15.8 µs of a 39.1
µs device chain, one CUDA kernel replaced them at 29.9 µs, and folding that kernel into the
step took the step to 24.6 µs on 2 launches. Each step of that is the same argument — the
work was always the consuming CTA's own business, and every boundary drawn around it cost
more than the work.

So the fold is a HEADER. The step folds its own operands, the growth probe folds its own
write side, and between them there is no launch, no plane and no ordering constraint.

## <a id="the-identity"></a>2. The fold is an address, not a copy pass

For a CONTIGUOUS operand the fold is not even that: `[B, 1, H, W]` contiguous has element
`(b, 0, h, w)` at `b*H*W + h*W + w`, and `[BH, W]` with `bh = b*H + h` has it at
`(b*H + h)*W + w` — the same offset. So the "layout copies" the torch path issued were
never transpositions; they were `.float()` casts that had to allocate a destination.

A routing level is not generally contiguous, though: the producer hands out VIEWS of one
packed plane, and `.contiguous()` on the way in would be exactly the launch this design
exists to delete. So each operand arrives as a `Span` — its base and its `(b, h, w)`
strides — and the fold is the address the kernel computes, `bh -> (b, h)` by the consumer's
own convention.

What the entry refuses is `T != 1`: the fold is defined on one token and a second one would
be silently dropped.

<a id="raw"></a>A digit's RAW value rounds nothing — a u8 digit is its integer and an f16 one widens
exactly to fp32 — and a level that carries its own mass takes its amplitude as
`raw * (scale / full_scale)`, one fp32 multiply by a per-row factor (`full_scale` is 255 for
u8 and 1 for f16). `tests/integration/test_decode_fold.py` cell 1 states that through the
step's own output, on strided views as well as contiguous ones.

## <a id="the-dtype-arms"></a>3. The dtype arms are runtime, and the step's template axes are not

The fold reads three INDEPENDENT storage families: the coefficient rows (u8 or f16, ruled
2026-09-29: u8 serves depth <= 2, f16 any depth), the write gain (`float` or `bfloat16`) and
the value digits (s8, with a u8 low byte or without). They are independent because the
routing bundle and the value projection are different producers.

Making them template axes would multiply the step kernel's instantiation matrix for a few
hundred bytes of gather in a prologue. So `DecodeParams` carries flags (`rows_f16`,
`g_bf16`, and the low byte's presence) and each gather site picks a fully typed loop:

```
if (p.rows_f16) fold_digits<D, __half>(...); else fold_digits<D, uint8_t>(...);
```

The branch is on the DECLARED coefficient type, not on the depth, and it is taken once per
SIDE, not once per element; both arms are complete — a dispatch, not a fast path with a
fallback. A dtype outside the format is a REFUSAL (the bundle's own, then the host entry's),
never a silent widening.

<a id="fold-value"></a>The value is folded once a step into the CTA's resident row: the
digits' integer `q` (`hi` at one digit, `hi * 256 + lo` at two) times the token's
`scale / 127` or `scale / 32767`. `load_value` issues the loads before the digit pass and
`value_of` does the arithmetic after it, so the channel's loads ride the same round trip.

Contrast `DECAY` and `D` (`decode.md#instantiation-matrix`), which ARE template axes: each
picks a different loop shape or a different live-register count in the walk's innermost
path, where a runtime branch would cost a register or an instruction on every iteration
rather than a few hundred bytes once per step.

## <a id="declared-order"></a>4. The normalizing sum, and its DECLARED order

A read level the producer did not already leave on the simplex still carries its own mass,
and the decode arm is defined on the simplex, so that level is divided by its row sum
(`RouteFactors.read_needs_normalization` is the predicate, stated once and read by both the
step's `normalize` argument and `RouteFactors.read_simplex`). The row's scale cancels, so the
normalized amplitude is `raw / sum(raw)`: one divide per digit, no scale read. The write side
is never normalized, whatever the read side's flags say.

<!-- ABSENT: reduce_kernel -->
A sum has an ORDER, and floating-point addition is not associative. The torch reduction
this replaced had whatever order that release's `reduce_kernel` chose — an order no gate can
hold anything to, because it is not written down anywhere. This one is:

> **The declared order.** Lane `j` of the level's warp accumulates
> `row[j], row[j+32], row[j+64], …` into an fp32 register in ASCENDING index order.
> The 32 lane partials are then folded by five `__shfl_down_sync` steps with offsets
> `16, 8, 4, 2, 1`, each adding lane `j + off` into lane `j`. Lane 0 holds the sum.
> Lanes past the row's width contribute an exact `0.0f`.

That is a total order over the additions, reproducible on any device. For u8 digits every
partial is an integer below `2^24`, so the sum is EXACT and the order moves nothing; for f16
digits the order is observable, and it is the declared one.

<a id="the-gate"></a>**How it is gated now that nothing publishes it.** The fold has no
output a test can read — its destination is shared memory the walk consumes and discards.
So `tests/integration/test_decode_fold.py` gates it through the STEP, on a fixture where
the step's whole arithmetic collapses onto the divisor: `D = 1`, a read level that is
nonzero everywhere, an empty write side, and a state whose rows are zero except at one leaf
`s0`. Then `y[c] = a * e[c] / (a * m + eps)` with `a = raw[s0] / row_sum(raw)` — one
multiply per column and one divide, so `y` is a function of the divisor's last bits and of
nothing else. The reference computes exactly that from a torch reimplementation of the
declared order.

Two cells are the declaration's teeth: one shows the declared order and a plain ascending
one DISAGREE on an f16 row where summation order is observable, and one puts that row through
the step and shows `y` follows the declared order rather than the ascending one. So the
gate is answering a question about order and not merely about arithmetic in general.

`rola.ops.decode.fold_levels` is the torch fold those references are built on, kept for
that purpose and called from nowhere on the shipped path — the same arrangement as
`_write_atom_bitmap` and for the same reason: a fold checked against its own output proves
nothing.

## <a id="no-allocation"></a>5. Nothing here allocates, and nothing is left behind

The fold's destinations are shared memory, so there is nothing per-step to allocate, retain
or address-pin. That is what makes it capturable by construction: the phase reads no device
value on the host, and a captured graph replays it with no buffer identity to preserve.

`dials` is the one operand the fold does not touch, and the boundary is principled: it is a
per-HEAD constant of the decay arm rather than a per-token operand, and its fp32 cast
carries a refusal (`rola.ops.decay.leaf_rate_dials`) that the host has to raise.

## Comments moved from source

Verbatim `//:` prose blocks from `csrc/rola/src/decode/decode_fold.cuh`, in source order, one heading per declaration. Source keeps only a short decl block pointing here.

<a id="fold-digits"></a>
### `fold_digits`

THE DIGIT PASS: every digit of both sides in one pass, a warp a mask word. A lane loads its
digit and its row's fp32 scale together (one round trip), the warp's ballot of the
exact-nonzero test is the digit mask's word ([`decode_lattice.md#the-digit-mask`](decode_lattice.md#the-digit-mask)),
and the lane stages `raw * (scale / full_scale)` -- or, for a read level the fold
normalizes, the raw digit itself, until its sum is known. Slots past a level's width are
not loaded and ballot a zero.

<a id="row-sum"></a>
### `sum_staged_rows`

THE DECLARED SUM of a normalized level's raw digits, read back from their staging. Warp
`first + l` owns level `l`; lane `j` accumulates `raw[j], raw[j + 32], ...` in ascending index
order into an fp32 register, then the five `__shfl_down_sync` steps fold lane `j`
into lane `j - off`. Lanes past the row's width contribute an exact `0.0f`.
docs/internals/decode/decode_fold.md#declared-order

<a id="normalize-rows"></a>
### `normalize_rows`

A normalized level's amplitudes, `raw / sum`, written in place over its staging on the
far side of the barrier that published the sums.

## More comments moved from source

Verbatim trailing/banner `//` comments from `csrc/rola/src/decode/decode_fold.cuh` condensed at their call site into a short pointer.

<a id="note-l3"></a>
### near line 3

THE DECODE FOLD -- every per-token operand, read from the internal format, normalized and
staged in SHARED MEMORY by the CTA that is about to consume it.

Three facts a reader must not miss:
  1. THE FOLD IS A GATHER, NOT A COPY PASS. Its inputs are whatever VIEW the producer
     handed out -- a level is ordinarily a slice of one packed plane -- so the fold
     indexes by stride and the `[B, 1, H, W] -> [B * H, W]` fold is the address it
     computes, never a materialized intermediate.
  2. THE SUM'S ORDER IS DECLARED, not incidental: lane-strided ascending partials
     folded by a 5-step `__shfl_down_sync` tree. A reduction whose order is a
     property of whatever the library did that release is a reduction no gate can
     hold to bit-identity.
  3. IT IS A HEADER, NOT A KERNEL. The step folds its own operands and the growth
     probe folds its own write side; between them there is no launch and no plane.

The declared order, the dtype arms and the cast argument:
docs/internals/decode/decode_fold.md
