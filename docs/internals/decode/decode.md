# `csrc/rola/src/decode/decode.cuh` + `csrc/rola/src/decode/decode.cu` — the `T = 1` decode path

This doc covers **two** source files, because they are one component: the header
carries the parameter block and the objects both decode translation units share,
and the `.cu` carries the GEMV kernel, its dispatch and its host entry. The
mirror rule is one doc per source basename, so `decode.cuh` and `decode.cu` land
here together. **Part A** is the header; **Part B** is the translation unit.

`decode_expand.cu` — the support expansion that runs first — has its own doc,
[`decode_expand.md`](decode_expand.md). The host-facing declaration split is
[`decode_api.md`](decode_api.md).

Symbols used throughout: `N` is the leaf count; `D` the number of routing levels
(`1 <= D <= 4`); `b_l` level `l`'s width and `S_l = prod(b_{l..D-1})` its span, so
`S_0 = N`; `BH = B * H` the flattened batch-head axis; `d_v` the value width and
`cols = d_v + 1` (the mass column is unconditional; raw normalization is dead —
there is no `GLOBAL` axis anywhere in this repository, decode included);
`n_words = ceil(N/32)`; `M` the number of touched rows (`popcount(bits_r |
bits_w)`); `n_split` the split-K degree; `BT`/`BC` are the tiled consumer's token
and owner tiles, which appear here only to be ruled out.

**The tiled consumer, in this document, is a RETIRED arm.** It was the prefill
kernel this path was designed against, and P67 D2 deleted it whole — sources,
mirror docs and all (`DELETIONS.md`; revival `git show c7eaeb9:<path>`). The
contrasts drawn against it below are kept deliberately and read in the past
tense: decode's shape is what it is BECAUSE of what it declined to inherit, and a
design record that erases the thing declined leaves the decision unexplained.
Where a section says a constant is restated rather than included, the header it
was not included from is one of the deleted ones, and the restatement is now
simply this file's own constant. The surviving prefill arm is the CHUNK consumer
([`chunk/chunk.md`](chunk/chunk.md)) — a different kernel, named as such wherever
it is meant.

---

# Part A — the header (`decode.cuh`)

## <a id="no-stage-2"></a>1. `T = 1` has no stage 2

The decode design ruled that at `T = 1` **every** stage-2 decision is
degenerate:

| stage-2 decision | why it is degenerate at `T = 1` |
|---|---|
| `BT` | one token, so there is no tiling |
| `P` | nothing to split |
| packing | `K` is one token's support, and the masks subsume it |
| statistics | a population of one |

So the decode path bypasses selection ARCHITECTURALLY — not by taking a cheap
default — and its config freezes at the prefill→decode boundary. What is left per
step is data plane only, and its shape is the one §6b ratified: a
**token-stationary, state-streaming gather-GEMV**.

## <a id="kronecker-support"></a>2. The structure, and the one fact it rests on

At `T = 1` a leaf `s` is in the token's read support iff EVERY level's digit
`d_l(s)` has a nonzero read amplitude, so

```
supp(R) = prod_l supp_l(R)                      (a Cartesian product)
```

and the support's BITMAP is therefore a Kronecker product of the per-level
indicator vectors. That makes the exact support an `N`-bit object that EXPANDS BY
BROADCAST-COPY:

| `N` | exact support bitmap | state stream |
|---|---|---|
| 65,536 | 8 KiB | 130 MiB |
| 1,048,576 | 128 KiB | 2 GiB |

**Materializing the truth is cheaper than bounding it.** That is why there is no
candidate list, no over-enumeration and no per-row membership test anywhere on
this path: `rola_decode_expand` builds `bits_r` and `bits_w` exactly, and
`rola_decode_gemv` `__ffs`-walks them, touching only rows that ARE in the
support. What a bounded candidate enumeration would cost instead, and the objects
this path therefore does not carry, are in
[`decode_expand.md`](decode_expand.md#why-a-bitmap).

## <a id="bc-absent"></a>3. `BC` is absent, and that is the point (spec I1)

Owner grouping is a property of how the TILED consumer amortizes a `[BC][cols]`
block across many tokens; with one token there is nothing to amortize. Nothing in
`decode.cuh`, `decode_expand.cu` or `decode.cu` is an expression in `BC`:

- the support is leaf-granular;
- the work partition is over 32-leaf bitmap WORDS — a machine quantity, not a
  plan quantity;
- the DENSE state address is `(bh * N + leaf) * cols + c`, which is leaf-major
  already: owner `o`'s leaf `i` sits at row `o * BC + i = leaf` in the tiled
  kernel's own layout;
- the PAGED one is keyed to the 16-leaf atom ([§4](#paged-address)), which mentions
  no `BC` either — that re-key is what buys it.

`BC` is a launch-shape fact of that kernel and never a layout fact.

## <a id="paged-address"></a>4. Paged and dense are ONE walk

<a id="the-atom"></a>The page granule is the **atom** — `kAtomLeaves = 16`
consecutive leaves in canonical leaf order, spec §4.5's
`page_states := MMA_K_QUANTUM`. It is geometry-owned and fixed across every config
forever, which is exactly why the paged address is `BC`-free and why this path can
carry it without acquiring a `BC` ([§3](#bc-absent)). `kAtomLeaves` is mirrored in
`decode.cuh` rather than included, for the same reason `kMaxLevels` is
([§5](#max-levels)): the decode TUs share no header with the tiled consumer.

The walk resolves an atom's first row as an ELEMENT offset from `p.state`, and the
two backings differ in that offset and in nothing else:

| backing | `page_tbl` | atom base | slot |
|---|---|---|---|
| dense | `nullptr` | `bh * N + (atom << 4)` | the atom itself |
| paged | `[BH, N/16]` int32 | `slot << 4` | `page_tbl[bh * atoms_per_bh + atom]` |

so `row = state + atom_base + (leaf & 15) * cols` under both, and the dense form
recombines to `bh * N + leaf` — today's address, bit for bit. Writing the dense arm
through the atom rather than through the leaf costs nothing and buys the property
that matters: a topology whose `N` is NOT a whole number of atoms (`(3, 5, 7)`,
`(7, 9)`, `(33,)` all appear in the gates) still addresses correctly densely, and it
simply has no paged keying at all — `AtomKeying` refuses to build one.

**The resolve is once per touched ATOM, not per row.** It is a value-keyed cache
(`if ((leaf >> 4) != cached_atom) reload`), so it costs at most two `int32` loads per
32-leaf bitmap word and — the part worth stating — it is correct whatever order the
walk reaches leaves in. The canonical order that makes 16 consecutive rows share a
slot is what makes it CHEAP; nothing about its correctness depends on it.

The table row is pinned as a plain origin and everything per atom is a displacement
from it, which is the addressing discipline the chunk consumer's SASS gate forced and
this path adopts. The widening multiply that forms `atom_base` is paid once per atom
where the previous code paid one per ROW, so the change strictly reduces it.

<a id="absent-atom"></a>**An absent atom (`slot < 0`) reads as zeros and is not
written.** That is exact rather than defensive: a step's plan admits its write set
EXACTLY (the growth probe's published condensation, [§4b](#growth)), so an atom with no
slot is outside
it, and a never-deposited leaf holds zeros in the dense backing too. The
self-normalizing readout then contributes exactly nothing through it — the
mechanical form of the dead-leaf argument, gated by
`tests/tier2_geometry/test_decode_paging.py`.

## <a id="growth"></a>4b. Growth — the step's own new work

A step may deposit into an atom that is not resident. At `T = 1` the write support is
a Cartesian product ([§2](#kronecker-support)), so the step's write-ATOM set is the
per-level exact-nonzero indicators' product ORed 16-deep — and the probe below computes
exactly that, on the device, as a by-product of the question it was launched to answer.

**MOST STEPS HAVE NO GROWTH, AND THE STEP FINDS THAT OUT FIRST.** The exact admission is
dear against the launch it precedes — a `plan_exact` and a cross-stream event against a
0.09 to 0.10 ms kernel (RTX 3090, medians of 15 x 50) — and a sequence whose routing has
settled writes atoms that are already resident. So the order per step is a QUESTION and
then, only if the answer is yes, `rola_op`'s order at `L = 1`:

```
_growth_probe(routes.write)        ->  one kernel + one [BH] int32 read: anything new?
                                       and, either way, the step's write-ATOM set
  no  -> _decode_step              ->  the walk, against the residency it already has
  yes -> state._decode_entry       ->  plan_exact(workspace.atom_bits, fresh=False)
         arena.wait()              ->  the ONE event the launch waits on
         _decode_step              ->  the walk, resolving through the grown table
```

**THE SET IS DERIVED ONCE, AND ON THE DEVICE.** The probe has to condense the write
expansion to atoms in order to test it against the table, so publishing that condensation
costs one byte per atom and removes the host derivation from a growth step entirely.
Measured (RTX 3090, medians of 15 x 50, the same three cells): a growth step is
0.615–0.626 ms against 0.756–0.862 ms for the same step planning from a host-derived
bitmap, i.e. −17% to −27%; a steady step is 0.222–0.242 ms, of which the probe is
0.118–0.156. So the trigger is no longer a trade at all — it makes steady steps ~3x
cheaper AND growth steps cheaper than the preamble-every-step order that preceded it.

`rola.ops.decode._write_atom_bitmap` survives as the ENUMERATED REFERENCE for that
condensation — the same set derived from the Cartesian-product definition in torch, with
the comparison in `tests/tier2_geometry/test_decode_growth_trigger.py` as its only
caller. It is not a fallback and not a second path a caller may select; it is what makes
the gate a check rather than the kernel agreeing with itself.

The pool is reserved once, so admission never moves a base pointer and there is nothing for
the kernel to discover. `plan_exact`'s single size read is the only host synchronization a
growth step adds beyond the probe's own.

<a id="graph-capture"></a>**NAMED FUTURE WORK: the step is now HOST-WAIT-BOUND, and the
next move is structural.** Of a 0.22-0.24 ms steady step, 0.12-0.16 ms is the host waiting
for a 0.007 ms kernel to become visible: the probe's kernel enqueues in 0.0071 ms and a
bare `torch.cuda.synchronize()` costs 0.0039, but launch-to-host-visible-completion on this
WSL2 host is ~0.05-0.08 ms. Blocking is already the cheapest of the five mechanisms
measured (0.093 ms against 0.099 for `.any().item()`, 0.111 for a stream sync, 0.139 for
`.cpu()`, 0.152 for an event), so there is nothing left to tune -- the floor IS the host
round trip. Two structural exits, neither scheduled: capture the whole step in a CUDA graph
(the workspace is already allocation-free and `ctr` is reset by the step's own last CTA
precisely so a step is capturable), or consume the probe's verdict ON THE DEVICE so no host
round trip happens at all -- which would mean a device-side admission, the successor
`PageArena._admit_pages` already names in its own docstring. Both are post-P67.

<a id="growth-probe"></a>**THE PROBE, and why it is exact.** `rola_decode_growth_probe` is
a kernel BESIDE the expansion, in `decode_expand.cu`, and it calls that TU's own
`expand_side` over the same `amp_write` pointers the step itself expands. It condenses the
result 16-deep — a bitmap word is exactly two atoms, because the atom is 16 leaves and leaf
order is bit order — PUBLISHES that condensation as `[BH, N/16]` bool, ANDs it against
`page_table >= 0`, and writes one `[BH]` int32. So "the probe and the admission agree about
the write support" is a property of the CALL GRAPH rather than of an audit — the admission
plans from the very bytes the verdict was read off; what
`tests/tier2_geometry/test_decode_growth_trigger.py` can still catch is a condensation or
a residency test that drifted, and it sweeps both directions of the agreement.

Every CTA writes its own entry unconditionally, so the buffer needs no zero-fill and the
reduction needs no atomic. The host read is ONE BLOCKING pinned device-to-host copy into a
workspace-owned buffer: measured against the same copy issued asynchronously and awaited on
an event, 0.093 ms against 0.152 (medians of 15 x 100). The host must wait for the kernel
either way, and the event path adds a record and a query on top of the wait the copy
already performs.

**Why not the prefill arm's stats pass.** `chunk_stats` reduces over a
`[BH, T, sum_l width_l]` routing stream; at a population of one that launch would be pure
overhead, and it would be a THIRD derivation of a set the step's own expansion already
holds. The reference derivation `_write_atom_bitmap` keeps for the gate costs `D - 1`
broadcast ANDs over a `[BH, N]` bool — four times the page table's own bytes, transient,
measured at 0.130–0.236 ms, which is what the published condensation removes from a growth
step. Deriving at atom grain directly, without the leaf expansion, is not available in
general: an atom is 16 CONSECUTIVE leaves and the mixed radix does not align a level
boundary with it.

## <a id="max-levels"></a>5. `kMaxLevels` — restated, not included

Spec §1 gives `1 <= D <= 4`. `decode.cuh` states that bound itself rather than
inheriting it from a prefill header: it was restated rather than included from the
tiled consumer's `consumer.cuh` because the decode TUs shared no other object with
that kernel and a one-way include would have implied they did — and the same holds
of the chunk consumer today. The SPEC is the shared authority here; a header is
not.

## <a id="level-width-capacity"></a>6. The level-width capacity is a GATE, not a convenience

`rola/routing/topology.py`'s `MAX_BRANCH_WIDTH = 256` is the widest level the
PRODUCER will emit. A decode path whose capacity were narrower would accept a
topology the producer can build and then read past the end of a staged amplitude
row. So the relation `kDecodeMaxLevelWidth >= kProducerMaxBranchWidth` is
enforced in two halves:

- the **compile-time** half is the `static_assert` in `decode.cuh`;
- the **runtime** half is `rola_decode_capacity()`, bound into the extension so
  `tests/unit/test_decode_capacity.py` can assert the two constants agree.

A C++ assertion cannot read a Python constant. The pair is what makes this
checkable rather than merely written down — and see
[§33](#producer-width-mirror) for why the mirror is exposed separately from the
capacity.

## <a id="decode-threads"></a>7. `kDecodeThreads = 256`

Both decode kernels launch 256 threads. The number is not free-floating; three
things pin it:

1. at `d_v = 64` it is exactly two columns per lane, with 128 B coalesced row
   loads;
2. it matched the tiled consumer's `kConsumerThreads`, so the prefill and decode
   arms of that era agreed on a block size;
3. 6 CTAs of 256 fill sm_86's 1536-thread limit exactly.

## <a id="coarse-group"></a>8. `kCoarseGroupWords = 32` — the coarse popcount index

Bitmap words per COARSE GROUP. `pc_coarse` holds one exclusive-prefix entry per
group plus a total, so a CTA resolves its row range with one binary search over
`ceil(n_words / 32)` entries and one 32-word scan, instead of a pass over the
bitmap.

The same array is consulted DURING the walk to skip wholly-empty groups
(`pc_coarse[g + 1] == pc_coarse[g]`), which is what keeps the bitmap read from
dominating a high-batch, high-sparsity step. See [§24](#coarse-consult).

## <a id="params-amps"></a>9. `DecodeParams` — the amplitude streams

`amp_read[l]` / `amp_write[l]` are `[BH, b_l]` CONTIGUOUS: the `T = 1` degeneracy
of the tiled consumer's `[BH, n_tiles, BT, pitch]` streams with `n_tiles = BT = 1`.
There is no pitch, because the decode op folds `[B, 1, H, b_l] -> [BH, b_l]` and
**the fold IS the contiguity**.

`level_amp_offset[l]` is where level `l`'s staged amplitude row starts inside the
flat `sum_l b_l` SMEM arrays. It is host-computed and carried rather than
recomputed, because the kernel needs it on the innermost path.

The remaining per-level integers are `level_width[l] = b_l`,
`level_stride[l] = prod(b_{l+1..D-1})`, `level_span[l] = S_l` (with `S_0 = N`) and
`level_row_offset[l]`, an offset into the flattened per-head dial array.

## <a id="detached-clock"></a>10. `amp_clock` — the detached clock, and why there is no sweep

`amp_clock[l]` is the DETACHED clock stream (`build_clock_stream`'s stored write
amplitude), and is `nullptr` under `decay = None`.

Its `stored != 0 => live != 0` guarantee is what makes §2.4's no-sweep claim
EXACT: a leaf outside `supp(W)` has a zero factor at some level, so its clock is
`0`, so `keep^0 == 1` exactly and its stored state is unchanged. Decay therefore
fires only on written rows — and written rows are streamed anyway. **There is no
entry sweep and no exit sweep in this kernel.**

## <a id="state-in-place"></a>11. `state` is read and written in place

`state` is the dense `[BH, N, cols]` plane or the arena's `[slots, 16, cols]` one
([§4](#paged-address)), with `cols = d_v + 1` (the mass column is
unconditional — raw is dead), and it is read AND
written in place: a row's read contribution and its write-back are performed by
the SAME lane in program order. The read-then-write ordering the tiled kernel
spends a barrier on per tile costs one register here. The semantics of that order
are [§27](#update-then-read).

`y` is `[BH, d_v]` — the FUSED ratio readout, `num / (den + eps)`.

## <a id="bitmap-fields"></a>12. The bitmap fields, and the union that is never stored

`bits_r` and `bits_w` are `[BH, n_words]` each — [§2](#kronecker-support)'s exact
support. `scratch` is the expansion's ping-pong partner and is
`[BH, 2 * n_words]` — the host carve hands it the last two of the flat
allocation's four `BH * n_words` blocks, and `decode_expand.cu` indexes it as
`p.scratch + bh * 2 * nw`. (The field's own trailing comment in `decode.cuh` says
`[BH, n_words]`; that comment is stale, and the extent above is the one the code
computes.) Why the pair is needed at all:
[`decode_expand.md`](decode_expand.md#ping-pong). `pc_coarse` is
`[BH, n_groups + 1]`, an exclusive prefix whose entry `n_groups` is `M`.

`bits_r | bits_w` is formed word-by-word in a REGISTER at walk time and never
stored: it is one instruction and one third of the storage a materialized third
bitmap would cost.

## <a id="splitk-workspace"></a>13. The split-K workspace

`ws` is `[BH, n_split, cols]` and `ctr` is `[BH]`. The workspace is SLOT-ORDERED
and summed by the last-arriving CTA, which is what makes decode's `y`
RUN-REPRODUCIBLE. Why that diverges from the tiled consumer on purpose is
[§16](#determinism).

## <a id="extract-bits"></a>14. `extract_bits` — the funnel shift, written once

Extract `count` bits of a packed bitmap starting at BIT index `start`, returned in
the low `count` bits (`count <= 32`). This is the shift-and-merge the Kronecker
copy is built from, and it is written once because it is the one place a funnel
shift can be off by one.

<a id="shift-32-ub"></a>The `(start & 31) == 0` fork is **not** an aligned fast path bolted onto a general
one — it is the C++ shift operator's undefined behaviour at a shift width of 32,
avoided at the one site that can reach it (`lo >> 0 | hi << 32`). Both arms
compute the same function. The standing *"no branch to dodge edge cases"* rule is
satisfied because there is ONE strategy here — gather 32 bits from at most two
source words — and the fork is inside it.

`drop_low_set_bits(word, k)` drops the LOWEST `k` set bits (`k <= 32`):
`word & (word - 1)` clears the lowest set bit, so `k` applications drop exactly
the `k` rows that belong to the CTA before this one. `keep_low_set_bits(word, k)`
is its mirror, for the CTA's last word. Both are exact by construction rather than
by an index comparison, which is what lets the row partition be defined in ROW
counts while the work is handed out in WORDS. The order in which they are applied
is load-bearing: [§23](#boundary-cut-order).

---

# Part B — the translation unit (`decode.cu`)

## <a id="gemv-shape"></a>15. Why it is a GEMV, and where the reductions are not

§6b's ratified shape, literally: the token's routing weights and value are
RESIDENT (SMEM/registers), the STATE streams past, and the grid partitions the
support.

What makes it a GEMV rather than a GEMM is that there is one token, so the
contraction index is the LEAF and the output index is the COLUMN. That is why
there is no MMA here **and no warp reduction either**: lanes are mapped to OUTPUT
COLUMNS, so each lane accumulates its own partials in registers for the whole row
stream and the dot never crosses a lane.

The only reductions in the file are the 8-way cross-warp fold and the
`n_split`-way cross-CTA combine, both over a small FIXED index set. Both are
fixed-order sums rather than tree primitives because each is `cols` INDEPENDENT
reductions over 8 (or `n_split`) values — a shape no block-reduce primitive has.

## <a id="determinism"></a>16. Determinism is a shipped property here, and it diverges from the tiled consumer on purpose

The tiled consumer emits each token's output as a split-K partial through
`atomicAdd` into `num`/`den`, so its `y` is not reproducible against its own
binary wherever a `bh` holds more than one owner. **Measured: 384 of 976 fixture
cases differ between two runs of one binary.**

Prefill's `y` feeds a loss or an eval aggregate, and that is tolerable. Decode's
`y` feeds an argmax or a sample that picks the NEXT TOKEN, so a last-bit
difference in `den` flips a near-tie logit and the divergence compounds
autoregressively for the rest of the generation.

At `T = 1` the partials are `n_split * cols` floats per `bh`, so the fixed-order
combine costs one workspace slot per CTA, one `__threadfence()` and one
`atomicAdd` on an INT counter — an atomic that defines the SCHEDULE, not the
arithmetic. See [§30](#slot-order).

The STATE is unconditionally reproducible for a stronger reason: the row
partition is a PARTITION, so every written row has exactly one owning CTA, warp
and lane, and no atomic touches `state` at all.

## <a id="no-barrier"></a>17. No barrier per anything

Two `__syncthreads` per CTA per step (three when `n_split > 1`), against the
tiled consumer's six per candidate tile. The tiled kernel spends one of its six
ordering the tile's readout against the tile's fold; at `T = 1` the two are a
single lane's two statements on one row in program order, and the ordering is
UPDATE-THEN-READ because §2's survival is inclusive at `j = t`
([§27](#update-then-read)).

## <a id="decode-max-exponent"></a>18. `kDecodeMaxExponent` — the same convention, structurally inert here

`expf(88) = 1.65e38 < FLT_MAX`. THE SAME VALUE AND THE SAME CONVENTION as the
tiled consumer's `intra_panel.cuh` carried in `kPanelMaxExponent` — restated here
rather than included, because that header was that kernel's panel algebra and
decode shares none of it. The header went with the arm (`DELETIONS.md`); the
convention is the spec's rule 3, and this file states it for itself.

On this path the clamp is **structurally inert**, and that is worth saying rather
than leaving a reader to wonder: the exponent is `clock * log1pf(-rate)` with
`clock >= 0` and `rate in (0, 1)`, so `log1pf(-rate) < 0` and the exponent is
non-positive. `expf` can underflow to zero (the correct limit) but cannot
overflow.

It is written anyway so that ONE convention governs every survival exponential in
the repository, and so a future decay form that CAN go positive finds the clamp
already in place. `tests/unit/test_decode_capacity.py` pins the two constants
equal.

## <a id="locate-row"></a>19. `locate_row` — two steps, neither a scan of the bitmap

Locate the bitmap word holding global set-bit index `row`, and how many set bits
of that word precede it. A binary search over the coarse index lands in a 32-word
group, then at most 32 popcounts walk to the word. This is what the coarse index
is FOR, and it is the same array the walk consults to skip empty groups
([§24](#coarse-consult)).

## <a id="template-axes"></a>20. Why there is no `GLOBAL` axis, and what `NC` is

Raw normalization is dead everywhere in this repository (revival = git;
[`DELETIONS.md`](DELETIONS.md)): the mass column and the `num/den` divide are
UNCONDITIONAL, so `cols = DV + 1` at every instantiation and there is nothing left
for a `GLOBAL` template axis to select between. `decode_gemv_kernel<DV, D,
DECAY>` takes exactly three template parameters.

`NC = DV / 32` is columns per lane. The row load is `NC` fully coalesced 128 B
transactions and the lane's accumulators are its own output columns — the whole
reason no reduction is needed on the dot ([§15](#gemv-shape)).

## <a id="smem-ledger"></a>21. The SMEM ledger, and the fold's `+3` pitch

| block | extent | what it holds |
|---|---|---|
| `sa_r`, `sa_w` | `[total_rows]` fp32 | the per-level amplitude rows, UNCOMPACTED and indexed by DIGIT |
| `sa_c`, `sd` | `[total_rows]` fp32 | clock amplitudes and this head's dial row; `DECAY` only |
| `v_tok` | `[DV]` fp32 | the token's value, resident |
| `fold` | `[warps][cols + 3]` fp32 | the cross-warp partials |

The amplitude rows are indexed by DIGIT and are not compacted: **the bitmap
carries membership, so the digit IS the index**, and no compaction, id list or
flag packing exists on this path.

<a id="fold-bank-padding"></a>The `+3` pads the fold's row stride to `cols + 3` so the eight warps' stores land
on distinct banks. The fold reads DOWN the warp axis, so an unpadded stride
sharing a factor with 32 would collide. `gcd(cols + 3, 32)` is 4 at `cols = 65`
and 1 at `cols = 32` and at `cols = 64 + 3`.

## <a id="row-scales"></a>22. `sr` and `sw` — the two per-token row scales

At `T = 1` there is no owner rectangle, so the per-LEVEL factors the tiled kernel
hoists into `sr`/`sw` for the fixed levels `l < j` stay in the per-LEAF product
instead. The split between `R[0,s]` and `sr` is an owner-rectangle artefact, and
the product is the same product either way. What remains here is exactly the
per-TOKEN part: the side gains. The read gain is fixed at `1` (`global` fixes
`g_read := 1`; raw is dead, so there is no arm where it is anything else).

## <a id="row-range-to-words"></a>23. The CTA's row range, resolved into a word range

The partition is defined in ROWS — popcount-balanced by construction,
`row_lo = seg * M / n_split` — and handed out in WORDS. The two boundary words are
cut by dropping / keeping set bits ([§14](#extract-bits)), which is exact rather
than an index comparison. An empty range is expressed as `w_start = 1,
w_last = 0`, so the walk below runs zero iterations without a separate arm.

<a id="boundary-cut-order"></a>`keep_low_set_bits` is applied BEFORE `drop_low_set_bits`, so that when a CTA's
whole row range lies inside ONE word the two together leave exactly bits
`[drop, keep)`. Reversing them silently truncates that case.

## <a id="coarse-consult"></a>24. The coarse-index consult at walk time

A group whose prefix does not advance holds no touched row, so its 32 words need
not be read at all. It is consulted once per GROUP rather than once per word — the
two `pc` loads would otherwise cost as much as the bitmap words they save — which
is what keeps the bitmap read from dominating a high-batch, high-sparsity step.
The boundary words `w_start` and `w_last` are non-empty by construction, so a skip
can never cross them.

The union `bits_r[w] | bits_w[w]` is formed in a REGISTER and never stored: one
instruction, one third of the storage a materialized third bitmap would cost.

## <a id="divisions-per-word"></a>25. The `D` integer divisions are paid once per WORD

NVIDIA hardware has no integer divide (~20 instructions each), so a per-row
mixed-radix decomposition would put ~80 instructions on a row whose payload is
`cols * 4` bytes — inside the sm_86 budget but **not** the sm_80 one, i.e. it
would make the A100 dense arm instruction-bound. Amortized over a word it is ~2.5
per row on the dense arm.

## <a id="aligned-subranges"></a>26. The word is walked in aligned sub-ranges

Each sub-range lies inside ONE inner block, so the innermost digit is
`cur + (bit - q)` with no division, and the outer digits are constant across it.
The trip count is `ceil(32 / b_{D-1})`: **1** for every topology whose innermost
width reaches a word (`widths = (256, 256)`) and **2** at the flagship `(8, 16)`.
ONE uniform loop that degenerates — not a fast path with a fallback.

Bits at or beyond `N` are never set: the expansion's destination-word loop stops
at `span_out`, which is `S_0 = N` at level 0, leaving the tail zero. So the walk
needs no leaf bound test and never fabricates a row.

## <a id="update-then-read"></a>27. Update first, then read the updated row — and that order IS the semantics

Not an optimization. §2's survival is INCLUSIVE at `j = t`: the token's own
deposit is read by the token itself with survival exactly `1`.

The tiled consumer expresses that as TWO terms, because at `BT > 1` the deposit
and the readout live in different GEMMs: the readout against the decayed entry
state, plus the causal intra Gram's self term `G[0,0] * Vw`. At `T = 1` those two
are

```
a_r * keep^c * M_entry  +  a_r * a_w * sw * v   ==   a_r * S_exit
```

so **the whole intra Gram collapses into reading the row after it is written**.
One pass over the row, one register, and the causal structure is exact rather than
reconstructed.

A row in `supp(R) \ supp(W)` has `keep == 1` and no deposit, so
`S_exit == M_entry` and the same expression is its readout too — one path, no arm.

Both off-by-ones — decaying after the deposit, and reading before it — are `O(1)`
errors that G4/G6 discriminate at the fp32 tolerance.

## <a id="decay-at-read"></a>28. Decay at read, via the detached clock, with no sweep and no exponent split

At `T = 1` the tiled kernel's `GLA_MASS_DECAY_SPEC.md` §12 rule-1 half-mass shift
is exactly the IDENTITY. With `P_ref = P[0]/2`:

```
readout:  fr * s0                 collapses to   keep^c
fold:     s0 * (s0 M + W fw Vw)   collapses to   keep^c M + W Vw
```

So on this path there is no clock prefix buffer, no `p_ref`, no `s0`, no division
by a cumulative survival, and no `fw` underflow class — **ONE `expf` where the
tiled kernel evaluates three.**

`log1pf` IS the composition (never a rounded linear-domain `1 - rate`) and there
is no epsilon floor: both of §12 rule 2's prohibitions, inherited verbatim.

The mass column is the `v := 1` column of the identical recurrence, unconditional
on every arm — one lane-0 statement, never a second path.

## <a id="fold-and-sr"></a>29. The 8-way fold, and where `sr` is applied

The cross-warp fold runs in FIXED warp order. `sr` is applied THERE, before the
split-K store, exactly as the tiled kernel emits `sr * y` into `num` and
`sr * den_acc` into `den`. That makes the combine below a plain sum, and keeps the
two consumers' dividend and divisor the same objects.

## <a id="slot-order"></a>30. The deterministic split-K combine

SLOT ORDER, always. This is the whole determinism argument: `n_split` fp32 addends
summed in a FIXED order give one answer per binary, where an `atomicAdd` race
gives one answer per arrival order.

`__ldcg` bypasses L1 so the reader sees the other CTAs' stores that the
`__threadfence` above made visible.

`n_split == 1` takes an early return before any of this — not a special case, the
same expression with zero combine steps.

The counter is reset to `0` by the last CTA, so no `memset` launch stands between
two decode steps and the counter is safe to capture in a CUDA graph.

## <a id="instantiation-matrix"></a>31. The instantiation matrix

`DV(2) x D(4) x DECAY(2) = 16`. Note what is **not** an axis: no `BT` (one
token), no `BC` (no owner blocking), no `MmaPrec`/precision split (there is no
MMA, and the tf32x3/bf16 split died with the `PREC` axis everywhere else —
`DELETIONS.md`), no `GLOBAL` (raw normalization is dead; the mass column is
unconditional — [§20](#template-axes)). Every axis is a structural plan fact with
few legal values, which is the standing *"branch on structure, not thresholds"*
rule. `d_v` is restricted to `{32, 64}` by a `TORCH_CHECK`, and `D` outside `[1,
4]` is refused in the dispatch's `default`.

## <a id="flat-bits"></a>32. The `bits` workspace is ONE flat allocation

Carved into `bits_r | bits_w | scratch` blocks of `[BH, n_words]`, `[BH, n_words]`
and `[BH, 2 * n_words]`. Flat rather than shaped, because a `[BH, 4 * n_words]`
shape would imply a per-`bh` interleaving the carve does not use — **and a shape
that lies is worse than no shape.**

## <a id="producer-width-mirror"></a>33. Why the producer-width mirror is bound separately

`rola_decode_producer_width_mirror()` exposes `kProducerMaxBranchWidth` separately
from the capacity because the two are different claims:

- `capacity >= producer` is the SAFETY property, and the `static_assert`
  ([§6](#level-width-capacity)) covers it;
- `mirror == the real Python constant` is what stops that assertion from being
  satisfied by a STALE mirror after the producer's limit moves.

Only a test that can see both languages can check the second, which is why this is
bound at all.

## <a id="empty-not-zeros"></a>34. `y` is `at::empty`, not `at::zeros`

Every element is written by the `n_split == 1` CTA or by the last-arriving CTA of
the combine, so a zero-fill would be a launch that the launch-latency-bound
small-batch regime pays for nothing — and an unwritten element showing as garbage
rather than as a plausible zero is what a gate wants.

## <a id="smem-refusal"></a>35. The staged-amplitude SMEM budget is a declared capacity

The host entry queries `cudaDevAttrMaxSharedMemoryPerBlock` and `TORCH_CHECK`s
`decode_gemv_smem_bytes(total_rows, d_v, cols, decay)` against it, naming
`sum_l width_l` in the message. It is **refused loudly rather than taken as a
slower path** — there is no spill-to-global arm for the staged rows.

The expansion's arm selection is not this check; it is structural and lives in
[`decode_expand.md`](decode_expand.md#smem-arm).
