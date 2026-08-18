# `csrc/rola/src/decode/decode_fold.cu` — the decode step's operand pass

Every per-token operand a decode step consumes, widened, normalized where the producer
did not already leave the level on the simplex, and published into the sequence's own
scratch — in ONE launch, before the step's three kernels run.

Symbols: `B` batch, `H` heads, `BH = B * H`, `bh = b * H + h`; `D` levels; `b_l` level
`l`'s width; `d_v` the value width. The operands are the read and write amplitude rows
(`[BH, b_l]` fp32 each), the write gain (`[BH]` fp32) and the token's value
(`[BH, d_v]` fp32).

---

## <a id="why-a-kernel"></a>1. Why the fold is a kernel at all

The operands are TINY — `2 * sum_l b_l + d_v + 1` floats per batch-head, kilobytes at
any shipped topology. The fold is not there to move bytes faster; it is there to be ONE
launch instead of eight.

A decode step is launch-bound. Measured on an RTX 3080 Ti at `B = 2`, widths `(16, 16)`,
`d_v = 64`, paged backing, the captured step ran three RoLA kernels and eight torch
operand launches; the eight were 15.8 µs of a 39.1 µs device chain — 40 % of the step
spent preparing 512 floats. The eight were the per-level normalizing sum (`D`), its
divide (`D`), and the widening cast on both sides (`2D`); each one launches, reads a
handful of values, writes a handful, and returns.

Nothing about that work is hard. All of it is the same CTA's business.

## <a id="not-in-a-kernel-we-have"></a>2. Why it is its own kernel and not part of one

The obvious economy would be to fold inside `decode_expand_kernel`, which already runs
one CTA per batch-head and already reads exactly these rows. It does not work, and the
reason is the WALK ORDER rather than the arithmetic:

- The **growth probe** consumes the folded WRITE side, and it runs BEFORE the expansion
  — that is the whole of its design (`decode_expand.md` §7): the host learns whether the
  step's write set is resident without deriving that set itself. A fold living inside the
  expansion would leave the probe with raw operands.
- Folding inside the PROBE instead moves the problem: the probe exists only under the
  paged mask, and the dense backing runs no probe at all
  (`docs/internals/engine/decode_dag.md`). The dense step would then need its own fold,
  which is the per-mask duplication the engine's one-walk rule refuses.
- A growth step is the same walk twice around an admission. Folding once, ahead of both,
  is also what makes the second walk free of operand work.

So the fold is a node of its own, ahead of the probe, on both masks.

## <a id="the-identity"></a>3. The fold is an address, not a copy pass

For a CONTIGUOUS operand the fold is not even that: `[B, 1, H, W]` contiguous has element
`(b, 0, h, w)` at `b*H*W + h*W + w`, and `[BH, W]` with `bh = b*H + h` has it at
`(b*H + h)*W + w` — the same offset. So the "layout copies" the torch path issued were
never transpositions; they were `.float()` casts that had to allocate a destination.

A routing level is not generally contiguous, though: the producer hands out VIEWS of one
packed plane, and `.contiguous()` on the way in would be exactly the launch this kernel
exists to delete. So each operand arrives as a `Span` — its base and its `(b, h, w)`
strides — and the fold is the address the kernel computes, `bh -> (b, h)` by the
consumer's own convention.

What the entry refuses is `T != 1`: the fold is defined on one token and a second one
would be silently dropped.

The widening rounds nothing — bf16 and fp32 both embed exactly in fp32 — so every
cast-only output is bit-identical to the torch fold it replaced, and
`tests/integration/test_decode_fold.py` cell 1 states that as `torch.equal`, on strided
views as well as contiguous ones.

## <a id="declared-order"></a>4. The normalizing sum, and its DECLARED order

A read level the producer did not already leave on the simplex still carries its own
mass, and the decode arm is defined on the simplex, so that level is divided by its row
sum (`RouteFactors.read_needs_normalization` is the predicate, stated once and read by
both this fold and `RouteFactors.read_simplex`).

A sum has an ORDER, and floating-point addition is not associative. The torch reduction
this replaced had whatever order that release's `reduce_kernel` chose — an order no gate
can hold anything to, because it is not written down anywhere. This one is:

> **The declared order.** Lane `j` of the level's warp accumulates
> `row[j], row[j+32], row[j+64], …` into an fp32 register in ASCENDING index order.
> The 32 lane partials are then folded by five `__shfl_down_sync` steps with offsets
> `16, 8, 4, 2, 1`, each adding lane `j + off` into lane `j`. Lane 0 holds the sum.
> Lanes past the row's width contribute an exact `0.0f`.

That is a total order over the additions, reproducible on any device, and
`tests/integration/test_decode_fold.py` reimplements exactly it in torch and holds the
kernel to `torch.equal` against it. Cell 4 of that file is the declaration's teeth: on a
row where summation order is observable, the declared order and a plain ascending one
DISAGREE — so the gate is answering a question about order and not merely about
arithmetic.

<a id="the-gate"></a>**The gate's independence.** `rola.ops.decode.fold_levels` is the
torch fold the gates compare against, kept for that purpose and called from nowhere on
the shipped path — the same arrangement as `_write_atom_bitmap` and for the same reason:
a fold checked against its own output proves nothing.

**What did NOT stay bit-identical.** `y` is not guaranteed bit-identical to the
pre-fusion step, because the divisor's summation order changed. It is identical whenever
the addends are bf16 — an fp32 accumulator absorbs bf16 addends exactly, so every order
agrees, which covers the producer's own output. Where it can differ (fp32 read levels
carrying their own mass) it differs by the divisor's last bits, and the step stays inside
`tests/oracle/test_decode_vs_oracle.py`'s tolerance against the fp64 oracle. The
dense-versus-paged `torch.equal` is untouched by any of this: both backings consume the
same folded operands.

## <a id="the-shape"></a>5. The launch

One CTA per batch-head, `kDecodeThreads` threads, no dynamic shared memory. Warp `l < D`
owns level `l`'s normalizing sum (`D <= 4 <= kDecodeWarps`, so one barrier separates the
sums from every publication); the whole CTA then writes both sides' rows, the value and
the gain. Warps past `D` idle through the first phase — the alternative is a second
partition of a kernel whose entire body is a few hundred bytes of traffic.

The instantiation matrix is the storage type of the routing levels, of the gain and of
the value, each in `{float, bfloat16}` — eight arms. The three are independent because
the routing bundle and the value projection are different producers: a bundle carrying
bf16 levels beside an fp32 gain is the ordinary case, not an oddity. A dtype outside that
set is a REFUSAL at the entry, never a host-side widening.

## <a id="no-allocation"></a>6. Nothing here allocates

Every destination is a `DecodeScratch` field, allocated once per sequence and memoized
above any captured region (`decode.md` §capture). That is what makes the fold capturable
by construction: the node reads no device value on the host, and the buffers it writes
are the same addresses on every replay. The torch chain it replaced allocated a fresh
tensor per operand per step.

`dials` is the one operand the fold does not touch, and the boundary is principled: it is
a per-HEAD constant of the decay arm rather than a per-token operand, and its fp32 cast
carries a refusal (`rola.ops.decay.leaf_rate_dials`) that the host has to raise.
