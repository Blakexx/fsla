# `csrc/rola/src/decode/decode.cuh` + `csrc/rola/src/decode/decode.cu` — the `T = 1` decode path

This doc covers **two** source files, because they are one component: the header
carries the parameter block and the objects both decode translation units share,
and the `.cu` carries the GEMV kernel, its dispatch and its host entry. The
mirror rule is one doc per source basename, so `decode.cuh` and `decode.cu` land
here together. **Part A** is the header; **Part B** is the translation unit.

`decode_expand.cuh` — the support expansion the prologue runs — has its own doc,
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
this path: the step serves `supp(R)` and `supp(W)` exactly and `__ffs`-walks
them, touching only rows that ARE in the support. It serves them rather than
storing them — level 0 is a broadcast, not data
([`decode_expand.md`](decode_expand.md#the-view)) — which is a statement about
where the truth lives, not about how exact it is. What a bounded candidate enumeration would cost instead, and the objects
this path therefore does not carry, are in
[`decode_expand.md`](decode_expand.md#why-a-bitmap).

## <a id="bc-absent"></a>3. `BC` is absent, and that is the point (spec I1)

Owner grouping is a property of how the TILED consumer amortizes a `[BC][cols]`
block across many tokens; with one token there is nothing to amortize. Nothing in
`decode.cuh`, `decode_expand.cuh` or `decode.cu` is an expression in `BC`:

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
EXACTLY (the step's own published condensation, [§4b](#atom-bits)), and a batch-head
whose write set it cannot cover does not walk at all ([§4c](#per-bh-verdict)), so an atom
with no slot is outside the write set, and a never-deposited leaf holds zeros in the dense backing too. The
self-normalizing readout then contributes exactly nothing through it — the
mechanical form of the dead-leaf argument, gated by
`tests/integration/test_decode_paging.py`.

## <a id="growth"></a>4b. Growth — the step's own new work, and its own question

A step may deposit into an atom that is not resident. At `T = 1` the write support is
a Cartesian product ([§2](#kronecker-support)), so the step's write-ATOM set is the
per-level exact-nonzero indicators' product ORed 16-deep.

<a id="absorbed-verdict"></a>**THE STEP ASKS THE QUESTION ITSELF, IN ITS OWN PROLOGUE.**
The exact admission is dear against the launch it precedes — a `plan_exact` and a
cross-stream event against a 0.09 to 0.10 ms kernel (RTX 3090, medians of 15 x 50) — and a
sequence whose routing has settled writes atoms that are already resident. What makes the
question nearly free is that the step has already built the object it is asked about: the
write map is the one the walk consumes, served over the inner block the same
[`expand_inner`](decode_expand.md#the-recursion) built, so the residency test is a pass
over the WRITE SUPPORT's candidate atoms ([§4b](#candidate-atoms)) and an indirection
each, and no second launch, no second fold and no second expansion exist at all. There is nothing left for a probe kernel to be.

So the order per step is ONE kernel, and then, only where the answer was yes and no pool
covered it, `rola_op`'s order at `L = 1`:

```
decode_step        ->  fold, expand, THE VERDICT, walk, and the write-ATOM set,
                       per batch-head, in one launch
growth_host.copy_  ->  the ONE 4-byte read, after the launch that produced it
  no  -> done                      ->  the walk ran, against the residency it had
  yes -> state._decode_entry_arena ->  plan_exact(scratch.atom_bits, fresh=False)
         arena.wait()              ->  the ONE event the launch waits on
         _decode_step              ->  the same step again, and the batch-heads that
                                       already ran do not run twice ([§4c](#done-flags))
```

<a id="the-table-test"></a>**THE TABLE TEST.** A bitmap WORD is exactly TWO atoms, because
the atom is 16 leaves and leaf order is the bitmap's bit order, so an atom's write
membership is a HALF-WORD TEST and no loop over leaves exists on this path. The count of
atoms that are written and unmapped is a block reduction over those tests. Every CTA of a
batch-head computes it, independently and identically: the two objects it reads are the
CTA's own expansion and a page table that no launch mutates until every CTA of that
batch-head has arrived ([§4c](#the-last-arriver)). Recomputing beat publishing when it was
measured — the split CTAs' recompute of the fold and the expansion is what removed the
launch — and the same argument governs here.

<a id="candidate-atoms"></a>**THE TEST RUNS OVER THE WRITE SUPPORT, NOT THE ATOM SPACE.**
An outer digit with no write amplitude has a wholly dead `S_1`-leaf range, so only the
digits in `lw` ([§12](#the-factorized-index)) can contribute. The enumeration is the
rectangle `[n_live x atoms_per_digit(S_1)]`: entry `k` contributes the atoms its digit's
leaf range touches, and an atom straddling two live digits is taken by the LOWER one
(`a > a_hi(k-1)`, exact because `a_hi` is monotone along an ascending digit list). That
makes the rectangle a PARTITION mapping monotonically onto ascending atom ids — which is
not a convenience: the pool claim's canonical ranks ([§4e](#the-claim)) are defined in
ascending atom id and this enumeration is where that order comes from. The same
enumeration serves the claim and the condensation.

<a id="atom-bits"></a>**THE CONDENSATION IS PUBLISHED, and it is the only thing that is.**
The test already has the write-atom set in hand, so writing it out costs one byte per atom
and removes the host derivation from a growth step entirely: `plan_exact` plans from
`scratch.atom_bits` and derives no support. One CTA per batch-head (`seg == 0`) publishes
it, UNCONDITIONALLY — a dead atom's `false` is as much of the set as a live one's `true`,
and publishing only the growing batch-heads would leave the buffer carrying an older step's
answer for the others, which is the very buffer the admission then plans from. It is two
passes, a clear over the row and a set over the candidate rectangle, because "everything
outside the write support is false" is a statement about the whole row while the support
is not. The clear is the one remaining term in a steady step that scales with `N`, and it
is one byte per atom on one CTA per batch-head.

The leaf-granular map is published by NOTHING. Both side maps live and die inside the CTA
that built them ([§12](#bitmap-fields)); the step's one leaf-level output is this
16-deep condensation of one of them.

`rola.ops.decode._write_atom_bitmap` survives as the ENUMERATED REFERENCE for that
condensation — the same set derived from the Cartesian-product definition in torch, with
the comparison in `tests/integration/test_decode_growth_trigger.py` as its only caller. It
is not a fallback and not a second path a caller may select; it is what makes the gate a
check rather than the kernel agreeing with itself ([§39](#second-derivation)).

## <a id="the-gate"></a>4c. The verdict is PER BATCH-HEAD, and the done flags are what make a replay idempotent

<a id="per-bh-verdict"></a>**WHY A VERDICT AT ALL.** A deposit into an ABSENT atom is
DROPPED ([§4](#absent-atom)) — `if (in_w && atom_present)`. So a host that read the growth
verdict LAZILY (every `K` steps, a pinned poll) would run steps whose writes are silently
lost. Laziness is not an available design at any `K > 1`; the only sound shapes are
per-step synchronization, or a step that DECLINES TO RUN where its writes are not yet
addressable. This is the second, and it is what turns the verdict from a host decision into
a device one — which is in turn what lets the step be enqueued before any host has seen it,
and captured in a graph ([§4d](#capture)).

A NEEDY batch-head — one whose write set it can neither address nor admit — runs nothing:
no walk, no `y`, no state, no counter beyond its arrival. Its verdict lands in `growth[bh]`
and its write-atom set in `atom_bits`, which is exactly what the host admission needs and
all of it.

The skip is PER BATCH-HEAD and not all-or-nothing, and the thing that used to make that
unsafe is the done flags below. A batch that shares one verdict wastes every resident
sequence's step on the rarest one's growth; a batch that skips per sequence must answer
what a replay does to the sequences that already deposited.

<a id="done-flags"></a>**THE DONE FLAGS.** `DecodeScratch.done` is `[BH]` int32, zeroed
ONCE at build. A batch-head that advances sets its flag. On the caller's replay a done
batch-head **walks and reads but does not deposit** — and reproduces its `y` BIT FOR BIT
while doing so, because [update-then-read](#update-then-read) means the rows it would
deposit into already hold the deposit and the same expression is their readout. So `y` is
complete for the whole batch after the replay without `y` ever becoming a buffer that
survives between steps, and the contract the caller sees is:

> gated step -> `growth_pending` -> `admit_growth` -> replay
> advances every sequence EXACTLY ONCE.

`tests/integration/test_decode_growth_trigger.py` states it as `torch.equal` on `y` AND on
the gathered plane against the admit-everything-first order, over a MIXED batch — one
batch-head needy, the rest not — which is the cell that can tell the per-batch-head
contract from the global skip it replaced.

<a id="the-last-arriver"></a>**EVERY CROSS-LAUNCH MUTATION HAPPENS AT ONE POINT, AND THAT
IS THE WHOLE RACE ARGUMENT.** The done flag, the page table's fold ([§4e](#the-claim)) and
the pool cursor are all read by the PROLOGUE of every CTA of a batch-head, and all written
by that batch-head's LAST-ARRIVING CTA — elected from `ctr[bh]`, the split-K counter that
already existed, which the needy path increments too so that the election is the same
statement in both branches. A CTA arrives only after it has finished reading, so no CTA of
a launch can observe a value its own launch wrote. Without that, a done flag set early
would make a sibling CTA skip its own share of the SAME step, and two CTAs' claims would
disagree about which atoms are free.

<a id="growth-reduction"></a>**THE REDUCTION, AND THE RESET.** The last of those per-batch-head
writers to arrive at `growth_ctr` ORs `growth` into the `[1]` `growth_any` the host reads.
Nothing needy is the same statement as every batch-head done, so that one test also decides
whether the done flags are zeroed for the next step: the flags reset themselves exactly
when the round is over, no `memset` launch stands between two decode steps, and the whole
thing is safe to capture. `growth_ctr` resets itself the same way `ctr` does.

## <a id="the-claim"></a>4e. The slack pool — a growth step that admits itself

Where the state reserves slack ([`paging.md#the-slack-pool`](../paging/paging.md#the-slack-pool)),
a batch-head whose write set is not resident does not have to go to the host at all: it
takes slots from its own pool region and advances. `pool_cap = 0` — the shipped default —
is exact commitment, and the paragraphs below describe machinery that does not run.

**THE ASSIGNMENT IS REPLICATED, NOT COMMUNICATED.** Every CTA of the batch-head computes
the same mapping: the touched-unmapped atoms in ASCENDING ATOM ID take
`pool_slots[bh][cursor + rank]`, with `rank` from a block scan over that batch-head's
atoms. No atomic claims a slot, nothing spins, and nothing is broadcast — which is what
keeps the step one launch and keeps it capturable. It is ALL-OR-NOTHING: a demand larger
than the batch-head's remaining pool is a needy batch-head and the host path exactly
([§4c](#per-bh-verdict)), because a partial claim would deposit part of a step and leave
the replay to deposit the rest twice.

**IT WRITES A SHADOW ROW, NOT THE TABLE, AND THAT IS FORCED.** The claim's predicate is
`page_tbl[a] < 0`. A prologue write into the table would change a sibling CTA's predicate,
which would change its ranks, and two CTAs would then assign the same slot to different
atoms — the ordering ambiguity that makes replicated assignment unsound. So the claim
writes `pool_map[bh]`: a WHOLE copy of that batch-head's table row with the claim merged
in, every atom present, claimed or resident or absent. The walk then addresses through
THAT ROW INSTEAD of the table, by one pointer chosen once in the prologue, and the walk's
own body knows nothing about pools — no extra test, no extra load, no register held across
it. The batch-head's last-arriving CTA folds the row's new entries into the real table and
advances the cursor ([§4c](#the-last-arriver)), so by the next step the claim is ordinary
residency and the shadow row is dead.

**AN ADMITTED ATOM IS AN ADMITTED ATOM.** Pool slots are ordinary arena slots, committed
and ZEROED by the host before they are named, so an atom the device admitted is
indistinguishable from one `plan_exact` admitted: same bytes, same table format, no
migration and no copy anywhere. That is what keeps the dense-vs-paged `torch.equal` gate a
statement about ADDRESSING, and it is asserted directly — a pooled arena and a
host-admitted one reach the same state and the same `y`.

## <a id="capture"></a>4d. CUDA-graph capture — `decode_step_gated`

With no host read inside it the step is capturable, and `rola.engine.dags.decode_dag.decode_step_gated`
is that entry: the same kernel in the same order as `decode_forward` minus the verdict
read. The CALLER owns the verdict, at whatever synchronization it already has (a decode
loop syncs once per token to sample):

```
y, state = decode_step_gated(...)   # or graph.replay()
if growth_pending(state):           # one 4-byte read, at the caller's own sync
    admit_growth(state)             # plan_exact from the step's own condensation
    y, state = decode_step_gated(...)  # or graph.replay(); now it runs
```

**A REPLAY IS SELF-CORRECTING**, which is what makes one captured graph enough: the verdict
is re-derived inside it, so after the admission it is 0 and the same graph performs the
step it had skipped — for the batch-heads that had skipped it, and for no others
([§4c](#done-flags)). Nothing in the loop reallocates — the workspace is built once, every
counter and flag is reset by the last CTA that reads it, and the arena's base address and
page table are objects an admission writes IN PLACE (the reservation never moves), so a
graph captured over them survives growth. The eager `decode_forward` remains THE base
path; capture is an optimization the caller selects, not a fallback either way.

Measured before the absorption (RTX 3080 Ti, medians of 15 x 50, one steady paged step,
`d_v = 64`, `BH = 6`), a loop of steps with one synchronize at the end, against the
admit-preamble-every-step order that preceded the device-side verdict:

| cell | steady, S4b order | steady, verdict order | steady, CAPTURED |
|---|---|---|---|
| 16x16 BC64    | 0.2468 | 0.2046 | **0.0367** |
| 8x8x8x8 BC128 | 0.3254 | 0.2220 | **0.0573** |
| 64x64 BC256   | 0.2811 | 0.2127 | **0.0553** |

The host time the caller's thread spends per steady step falls from 0.259–0.318 ms to
0.008–0.010 ms under capture: the host-wait floor S4b measured (launch-to-visible ~0.05–0.08
ms on this WSL2 host) is not tuned away, it is left with nothing to sit in front of.

Host-side ADMISSION is what the slack pool removes for a growth step that fits it
([§4e](#the-claim)); a step that overruns the pool, or a state with no pool at all,
surfaces to the host exactly as it always did.

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

Bitmap words per COARSE GROUP, and it is exactly a warp. `pc_coarse` holds one
exclusive-prefix entry per group plus a total, so a CTA resolves its row range with one
binary search over `ceil(n_words / 32)` entries and one WARP-WIDE popcount of the group it
lands in ([§19](#locate-row)), instead of a pass over the bitmap.

The same array is consulted DURING the walk to skip wholly-empty groups
(`pc_coarse[g + 1] == pc_coarse[g]`), which is what keeps the bitmap read from
dominating a high-batch, high-sparsity step. See [§24](#coarse-consult).

## <a id="params-amps"></a>9. `DecodeParams` — the operands, as addresses

`read[l]` / `write[l]` / `g_write` / `v` are `Span`s: a base pointer and the `(b, h, w)`
strides of the CALLER'S OWN `[B, 1, H, *]` view. There is no `[BH, b_l]` plane between the
producer and the walk — the step folds these into shared memory itself
([`decode_fold.md`](decode_fold.md)), so the `[B, 1, H, W] -> [BH, W]` fold is an ADDRESS
the kernel computes rather than a pass that ran.

`normalize[l]` says whether read level `l` still carries its own mass, so the fold divides
it by its row sum. `levels_bf16`, `g_bf16` and `v_bf16` name each operand family's storage
type; the fold branches on them once per side, never per element
([`decode_fold.md#the-dtype-arms`](decode_fold.md#the-dtype-arms)).

`level_amp_offset[l]` is where level `l`'s staged amplitude row starts inside the flat
`sum_l b_l` SMEM arrays. It is host-computed and carried rather than recomputed, because
the kernel needs it on the innermost path.

The remaining per-level integers are `level_width[l] = b_l`,
`level_stride[l] = prod(b_{l+1..D-1})`, `level_span[l] = S_l` (with `S_0 = N`) and
`level_row_offset[l]`, an offset into the flattened per-head dial array.

## <a id="detached-clock"></a>10. The detached clock, and why there is no sweep

The clock is the STORED WRITE ALLOCATION ITSELF — `c[t, s] = prod_l p_write[t, d_l(s)]`,
carrying no side gain, because `g_write` scales the deposit and not the clock. It is
therefore not a stream of its own: the step reads the same staged write rows the deposit
reads, and a second fold of one fact would be a second derivation of it.

Its `stored != 0 => live != 0` guarantee is what makes §2.4's no-sweep claim EXACT: a leaf
outside `supp(W)` has a zero factor at some level, so its clock is `0`, so `keep^0 == 1`
exactly and its stored state is unchanged. Decay therefore fires only on written rows —
and written rows are streamed anyway. **There is no entry sweep and no exit sweep in this
kernel.**

## <a id="state-in-place"></a>11. `state` is read and written in place

`state` is the dense `[BH, N, cols]` plane or the arena's `[slots, 16, cols]` one
([§4](#paged-address)), with `cols = d_v + 1` (the mass column is
unconditional — raw is dead), and it is read AND
written in place: a row's read contribution and its write-back are performed by
the SAME lane in program order. The read-then-write ordering the tiled kernel
spends a barrier on per tile costs one register here. The semantics of that order
are [§27](#update-then-read).

`y` is `[BH, d_v]` — the FUSED ratio readout, `num / (den + eps)`.

## <a id="bitmap-fields"></a>12. The bitmaps never leave the CTA that builds them, and level 0 is never built at all

[§2](#kronecker-support)'s exact support is served by two `SideView`s over one INNER BLOCK
of `4 * inner_words` words of SHARED MEMORY, built by the CTA that is about to walk it
([`decode_expand.md#the-view`](decode_expand.md#the-view)). NOTHING PUBLISHES ANYTHING:
the arm that spills the block to `work` exists only for an INNER span too wide for the
budget ([`decode_expand.md#smem-arm`](decode_expand.md#smem-arm)), and the step's one
leaf-level output is the write map's 16-deep condensation ([§4b](#atom-bits)).

<a id="the-factorized-index"></a>**THE INDEX IS COUNTED, NOT SCANNED.** With level 0 a
broadcast rather than data, the row count is a function of two small objects:

| block | extent | what it holds |
|---|---|---|
| `cum` | `[b_0 + 1]` i32 | union rows in outer digits strictly before `d`; `cum[b_0] = M` |
| `lw` | `[b_0]` i32 | the outer digits the WRITE side reaches, ascending |

A digit's CLASS is which sides carry amplitude there — `0` is dead on both — and every
leaf in that digit's range takes the inner block's bits for that class. So the rows in
digit `d` are one class popcount of the inner block, `cum` is one block scan over
`b_0 <= kDecodeMaxLevelWidth == kDecodeThreads` (hence exactly one pass), and
`rows_before(X)` is `cum[X / S_1]` plus the partial digit's own words. `lw` is the second
scan of the same pass and is what the atom enumeration walks
([§4b](#candidate-atoms)).

The class popcounts are a SERIAL loop over the inner block rather than a prefix table:
the callers are one thread per outer digit and one per coarse group, so the block covers
`inner_words` of work per thread in parallel, which is never more than the bitmap-wide
popcount this index replaces.

The coarse index `pc` is `n_groups + 1` words of the same shared memory, an exclusive
prefix whose entry `n_groups` is `M`.

<a id="coarse-index"></a>`pc[g]` counts the set bits of `bits_r | bits_w` in groups
strictly before `g` — the same VALUES a bitmap-wide popcount produces, at one
`rows_before` each, which is what leaves the row partition, `locate_row` and the walk's
group skip untouched by the factorization. The count is taken per CLASS rather than per
side because the two sides' popcounts do not sum to `popcount(r | w)`.

`bits_r | bits_w` is otherwise formed word-by-word in a REGISTER at walk time and never
stored: it is one instruction and one third of the storage a materialized third bitmap
would cost.

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
group, then ONE LANE PER GROUP WORD popcounts the group at once, a warp prefix places the
row and a ballot selects the word. This is what the coarse index is FOR, and it is the
same array the walk consults to skip empty groups ([§24](#coarse-consult)).

**ONE WARP, NOT ONE THREAD, AND THAT IS A MEASURED CHOICE.** The coarse group is exactly a
warp wide, so the whole group resolves in one parallel popcount. Walking those 32 words
serially on `tid == 0` while the block waited was, under the sampling profiler, the step's
single largest barrier stall — 32 % of all samples at the load immediately after that
barrier, 100 % of them attributed to it. The two boundaries (`row_lo` and `row_hi - 1`)
then take one warp each, side by side.

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
| `sa_r`, `sa_w` | `[total_rows]` fp32 | both sides' amplitude rows, UNCOMPACTED and indexed by DIGIT |
| `sd` | `[total_rows]` fp32 | this head's dial row; `DECAY` only |
| `v_tok` | `[DV]` fp32 | the token's value, resident |
| `fold` | `[warps][cols + 3]` fp32 | the cross-warp partials |
| `pc` | `[n_groups + 1]` i32 | the coarse popcount index |
| `cum` | `[b_0 + 1]` i32 | union rows before each outer digit; `cum[b_0] = M` |
| `lw` | `[b_0]` i32 | the write side's live outer digits, ascending |
| `bits` | `[4 * inner_words]` u32 | both sides' inner blocks and the expansion's ping-pong; `IN_SMEM` only |

Every one of them is FILLED BY THIS CTA, from the raw operands and from nothing else: the
step allocates no plane, reads no `[BH, b_l]` intermediate and publishes no bitmap. That is
the whole of [§21b](#per-cta-prologue).

The amplitude rows are indexed by DIGIT and are not compacted: **the bitmap
carries membership, so the digit IS the index**, and no compaction, id list or
flag packing exists on this path.

<a id="fold-bank-padding"></a>The `+3` pads the fold's row stride to `cols + 3` so the eight warps' stores land
on distinct banks. The fold reads DOWN the warp axis, so an unpadded stride
sharing a factor with 32 would collide. `gcd(cols + 3, 32)` is 4 at `cols = 65`
and 1 at `cols = 32` and at `cols = 64 + 3`.

## <a id="per-cta-prologue"></a>21b. The prologue is per-CTA, and that is the design

The step is one kernel with three phases — fold, expand, walk — and the grid is
`(n_split, BH)`, so the first two run once per SPLIT-K CTA rather than once per batch-head.
Every CTA re-derives the same amplitude rows and the same support bitmap for its `bh`, then
walks only its own row range.

That redundancy is the price of two properties, and it was measured before it was taken:

- **The support map never reaches DRAM.** A shared per-`bh` map would have to be written by
  someone and read by everyone, which is a global round trip on the walk's hottest index —
  or a launch boundary to order the write against the reads, which is the launch this
  design exists to delete.
- **The split partition stays.** The alternative — one CTA per batch-head, `n_split = 1` —
  costs no recompute at all and was priced directly: at widths `(64, 64)` the GEMV goes
  16.2 → 63.5 µs at `B = 2` and 67.3 → 131.5 µs at `B = 32` (RTX 3080 Ti, dense, captured
  replay). The walk is row-starved without the split, and a topology's `N` is exactly what
  decides that, so there is no regime where collapsing the grid is the better shape.

The recompute itself is an address walk over `2 * sum_l b_l` floats, `ceil(S_1 / 32)`
words of inner block and `b_0` digit classes — bounded by the LEVEL WIDTHS, not by the leaf
space and not by the state — which is why paying it `n_split` times still lands ahead of
paying one extra launch once.

**IT USED TO BE BOUNDED BY THE LEAF SPACE, AND THAT WAS THE FLOOR.** With level 0
materialized the recompute was `ceil(N/32)` words per side per CTA, and `n_split` grows
with `N` (`min(ceil(N/256), ...)`), so the step's fixed cost was quadratic in the leaf
space: 22.5 µs at `N = 256` against 101.4 µs at `N = 65,536` for a step whose touched-row
count had barely moved. Serving level 0 as a broadcast instead
([`decode_expand.md#the-cost`](decode_expand.md#the-cost)) takes that cell to 36.9 µs, and
takes the shared-memory footprint from 39.5 KiB to 9.0 KiB — which is the difference
between 2 CTAs/SM and the 5 the kernel declares ([§30b](#declared-residency)).

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

## <a id="declared-residency"></a>30b. The residency is DECLARED, and it is the whole
perf story of the fusion

`decode_step_kernel` carries
`__launch_bounds__(kDecodeThreads, decode_step_blocks_per_sm(DECAY, IN_SMEM))`
— 5 CTAs/SM on the shipped arm, one fewer for decay and one fewer again for the
global-scratch arm — and that second argument is not decoration. It is a
REGISTER CAP the assembler must meet, and the census is what proves it met it with zero
spills.

**Why it has to be declared.** The fusion moved a prologue into a kernel whose loop was
already the whole point, and the assembler priced the new live values the way it prices
anything: it took more registers. Left alone it took 57 on the shipped `(d_v=64, D=2,
no decay)` arm against the pre-fusion GEMV's 48 — which is 4 CTAs/SM instead of 5, a fifth
of the residency, on a walk that is LATENCY-BOUND over state rows. Measured: at widths
`(64, 64)`, `B = 32`, dense, that alone cost +10.8 % of the whole step and turned a
launch-count win into a wall-clock loss. With the budget declared it is 48 registers again,
no spills, and the same cell lands within noise of the pre-fusion wall while running one
kernel instead of three.

The lesson is the standing register policy read in the other direction: registers are worth
spending when they buy fewer instructions or a tensor-core op in the hot loop, and this
kernel's extra ones bought neither — they were prologue state the allocator kept alive
across a loop that never reads it. A budget is how that gets said in code rather than
hoped for.

The DECAY arm gets one fewer CTA because it carries a second per-leaf product (`o_rate`)
and a survival exponential through the same loop; it does not fit the tighter cap without
spilling, and the arm is a TEMPLATE AXIS, so the budget is a compile-time function of it
rather than a threshold on a measurement.

The GLOBAL-SCRATCH arm (`!IN_SMEM`) gets one fewer again, for the same kind of reason and
on the same kind of axis: its inner block lives in DEVICE memory, so a 64-bit base
survives into the walk and every bitmap word is a global load rather than a shared
instruction. It is a different memory regime, not a bigger version of the same one, and at
48 registers it spilled while every shared-memory arm did not. No shipped topology selects
it ([`decode_expand.md#smem-arm`](decode_expand.md#smem-arm)), so the CTA it gives up costs
nothing that is measured; what it buys is a census with zero spills everywhere.

**The claim's SIZE comes out of the fold that applies it** — an atom the shadow row maps
and the table does not IS a claimed atom — rather than being carried across the walk or
re-derived from the write-support enumeration afterwards. Both alternatives were built and
both spilled: carrying it holds an int through the loop, and re-deriving it holds the
enumeration's three operands (the inner block, the live-digit list, its length) instead.
Counting it where the fold already touches every atom holds nothing at all.

**What the fusion also had to buy back, in barriers.** The prologue is
`fold_row_sums` → one barrier → both sides' rows, the value and the dials → one barrier →
one expansion over BOTH sides sharing `D - 1` barriers and every integer division. Written
as the obvious two sequential per-side folds and two sequential per-side expansions it was
6 barriers at `D = 2` instead of 4, and the same cell paid ~3 % for it.

## <a id="instantiation-matrix"></a>31. The instantiation matrix

`DV(2) x D(4) x DECAY(2) x IN_SMEM(2) = 32`. Note what is **not** an axis: no `BT` (one
token), no `BC` (no owner blocking), no `MmaPrec`/precision split (there is no MMA, and the
tf32x3/bf16 split died with the `PREC` axis everywhere else — `DELETIONS.md`), no `GLOBAL`
(raw normalization is dead; the mass column is unconditional — [§20](#template-axes)), and
no OPERAND STORAGE TYPE. The last is the interesting one: the fold reads three independent
dtype families, so making them template axes would multiply the matrix by eight for a few
hundred bytes of gather. They are runtime flags selecting a typed loop instead
([`decode_fold.md#the-dtype-arms`](decode_fold.md#the-dtype-arms)). `IN_SMEM` earns its
place where they do not: it decides the ADDRESS SPACE the whole walk indexes, which a
runtime pointer would make generic. Every axis is a structural plan fact with few legal
values, which is the standing *"branch on structure, not thresholds"* rule. `d_v` is
restricted to `{32, 64}` by a `TORCH_CHECK`, and `D` outside `[1, 4]` is refused in the
dispatch's `default`.

## <a id="flat-bits"></a>32. `work` — the expansion's scratch, and usually empty

`work` is `[BH * n_split * 4 * inner_words]` int32 when the inner block does not fit in
shared memory, and ZERO-LENGTH when it does — which is every shipped topology. Its extent is one
authority's answer (`rola_decode_work_words`, which asks the kernel's own sizing function
against the device's real budget) rather than a host-side mirror of the same arithmetic,
because a mirror is the thing that drifts.

It is indexed PER CTA at `(bh * n_split + seg)`, not per batch-head: every split-K CTA
expands its own copy ([§21b](#per-cta-prologue)), so the slices are private by construction
and no two CTAs ever write the same word.

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
`decode_step_smem_bytes(...)` against it, naming `sum_l width_l` in the message. It is
**refused loudly rather than taken as a slower path** — there is no spill-to-global arm for
the staged amplitude rows.

The BITMAP's arm selection is not this check; it is structural, it comes first, and the
refusal above applies to what is left after it
([`decode_expand.md`](decode_expand.md#smem-arm)).

---

# Part C — the host side (`rola/ops/decode.py`, `rola/engine/dags/decode_dag.py`)

The step's ORDER, the two memo disciplines and the capturability declaration are the
DAG's page ([`engine/decode_dag.md`](../engine/decode_dag.md)); what follows is what the
KERNELS impose on the host.

## <a id="decay-domain"></a>36. Decode's decay domain is strictly larger, as a consequence

The tiled consumer refused a routing whose per-leaf reference exponent exceeded
`PANEL_MAX_EXPONENT`, because its rule-1 lowering exponentiated `keep^{+P}` and
`keep^{-P}` separately and their product is wrong — not merely rounded — once one
saturates while the other underflows. At `T = 1` that rule-1 half-mass shift is EXACTLY
the identity: the readout's `fr * s0` collapses to `keep^c`, and the fold's
`s0 * (s0 M + W fw Vw)` to `keep^c M + W Vw`. So decode forms ONE exponential, never a
quotient of two, and has no admissibility precondition to check — which is why it does
not import one.

## <a id="forward-only"></a>37. Forward only, and the guard is defence in depth

The decode kernels have no backward. The layer's decode envelope check already
refuses training mode and gradient reachability before the decode fork is reached, so
the module's own check exists for a direct caller.

## <a id="verdict-read"></a>38. The verdict's read is ONE BLOCKING pinned copy, and it cannot be anything else

The step publishes `growth_any` at the end of its own launch, so the eager path's read is
behind it BY DATA DEPENDENCE and there is nothing left to enqueue it in front of: the step
is the only launch either way. It is one `[1]` int32 into a workspace-owned PINNED buffer,
copied blocking. That was already the measured choice for a read standing in front of a
kernel (0.093 ms against 0.152 for an asynchronous copy plus an event, RTX 3090, medians of
15 × 100 — the host must wait for the kernel either way, and the event adds a record and a
query on top of the wait the copy already performs), and with the ordering forced there is
no second shape to compare.

Under capture the read leaves the step entirely — the caller performs it, at its own
synchronization ([§4d](#capture)) — which is where the host's participation in a steady
step actually goes.

## <a id="second-derivation"></a>39. `_write_atom_bitmap` is the gate's independent second derivation

The shipped path derives the write-atom set on the device (the step publishes it), so
`_write_atom_bitmap` has no production caller and is not meant to acquire one. It
survives deliberately as a SECOND derivation of an object the kernel computes, written
from the definition rather than from the kernel's structure, so the equality
`tests/integration/test_decode_growth_trigger.py` asserts is a real check and not the
kernel agreeing with itself. The decode paging and VMM gates plan from it for the same
reason: a fixture using the kernel's own answer could not catch the kernel.

The derivation, from the definition: at `T = 1` a leaf is written iff every level's write
digit is nonzero, so the write support is the Cartesian product of the per-level
exact-nonzero indicators and the atom set is that product ORed 16-deep. It is the same
object `rola.engine.facts.planes.atom_bits` produces for the chunk arm and the same one the step
condenses, reached by neither's route.
