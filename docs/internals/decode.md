# `csrc/rola/src/decode.cuh` + `csrc/rola/src/decode.cu` — the `T = 1` decode path

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
`cols = d_v + is_global`; `n_words = ceil(N/32)`; `M` the number of touched rows
(`popcount(bits_r | bits_w)`); `n_split` the split-K degree; `BT`/`BC` are the
tiled consumer's token and owner tiles, which appear here only to be ruled out.

---

# Part A — the header (`decode.cuh`)

## <a id="no-stage-2"></a>1. `T = 1` has no stage 2

`THUNDER_RUN_SPEC.md` §6b ruled that at `T = 1` **every** stage-2 decision is
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
- the unpaged state address is `(bh * N + leaf) * cols + c`, which is leaf-major
  already: owner `o`'s leaf `i` sits at row `o * BC + i = leaf` in the tiled
  kernel's own layout.

`BC` is a launch-shape fact of that kernel and never a layout fact.

## <a id="unpaged-only"></a>4. Unpaged only, deliberately

The paged address under §4.5's atom re-key is
`((slot << 4 | (leaf & 15)) * cols + c)` and is likewise `BC`-free, so nothing in
this path's `BC`-freedom depends on which granule the arena keys. It is simply
not implemented here: decode REFUSES a page table rather than computing an
address it does not own. That refusal is a hard `TORCH_CHECK` in
`rola/ops/decode.py` and in the extension, never a silent fallback.

## <a id="max-levels"></a>5. `kMaxLevels` — restated, not included

Spec §1 gives `1 <= D <= 4`. That is the same bound the consumer's `kMaxLevels`
states, and it is restated in `decode.cuh` rather than included from
`consumer.cuh` because the decode TUs share no other object with the tiled
consumer, and a one-way include would imply they do.

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
2. it matches the tiled consumer's `kConsumerThreads`, so the two agree on a
   block size;
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

`state` is `[BH, N, cols]` with `cols = d_v + is_global`, and it is read AND
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

## <a id="determinism"></a>16. Determinism is a shipped property here — and, since P10, everywhere

Decode shipped the deterministic split-K combine FIRST, because its `y` feeds an
argmax or a sample that picks the NEXT TOKEN: a last-bit difference in `den`
flips a near-tie logit and the divergence compounds autoregressively. (The tiled
consumer of that era emitted through `atomicAdd` and measured 384 of 976 fixture
cases differing between two runs of one binary.)

P10 brought the tiled consumer to the same standard by the same principle —
single-writer partials, one fixed fold order
([`consumer.md#emit-workspace`](consumer.md#emit-workspace)) — so slot-order
determinism is now an engine-wide property, not a decode divergence.

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

`expf(88) = 1.65e38 < FLT_MAX`. THE SAME VALUE AND THE SAME CONVENTION as
`intra_panel.cuh`'s `kPanelMaxExponent`
([`intra_panel.md`](intra_panel.md#rule-3)), restated rather than included
because that header is the tiled consumer's panel algebra and decode shares none
of it.

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

## <a id="template-axes"></a>20. `GLOBAL` and `NC` — why `GLOBAL` is a template axis

`GLOBAL` changes the STATE'S SHAPE (the `+1` mass column), so it is a template
axis for the same reason it is one in the tiled consumer: a runtime test would
make `cols` a runtime value in every state address, and would keep the mass
accumulator live on an arm that has no mass.

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
per-TOKEN part: the side gains. Under `GLOBAL` the read gain is fixed at `1`.

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

The mass column is the `v := 1` column of the identical recurrence, so `GLOBAL`
adds one lane-0 statement rather than a second path.

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

`DV(2) x D(4) x DECAY(2) x GLOBAL(2) = 32`. Note what is **not** an axis: no `BT`
(one token), no `BC` (no owner blocking), no `MmaPrec` (there is no MMA). Every
axis is a structural plan fact
with few legal values, which is the standing *"branch on structure, not
thresholds"* rule. `d_v` is restricted to `{32, 64}` by a `TORCH_CHECK`, and `D`
outside `[1, 4]` is refused in the dispatch's `default`.

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
