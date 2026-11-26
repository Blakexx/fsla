# `csrc/fsla/src/decode/decode.cuh` + `csrc/fsla/src/decode/decode.cu` — the `T = 1` decode path

<a id="note-l3"></a>This doc covers **two** source files, because they are one component: the header
carries the parameter block and the objects both decode translation units share,
and the `.cu` carries the GEMV kernel, its dispatch and its host entry. The
mirror rule is one doc per source basename, so `decode.cuh` and `decode.cu` land
here together. **Part A** is the header; **Part B** is the translation unit; **Part C** is what the
kernels impose on the host; **Part D** is the int8 step's page, lanes, prologue and work queue.

`decode_lattice.cuh` — the `(k, m)` box lattice's factor tables the prologue
builds and the walk resolves a unit from — has its own doc,
[`decode_lattice.md`](decode_lattice.md). The host-facing declaration split is
[`decode_api.md`](decode_api.md).

Symbols used throughout: `N` is the leaf count; `D` the number of routing levels
(`1 <= D <= 4`); `b_l` level `l`'s width; `BH = B * H` the flattened batch-head
axis; `d_v` the value width and `cols = d_v + 1` (the mass column is
unconditional; raw normalization is dead — there is no `GLOBAL` axis anywhere in
this repository, decode included); `ctas` the step's resident grid ([the work
queue](#queue)); `k`, `m`,
`s_l`, `g_l`, `BC` the `(k, m)` lattice's own symbols
([`decode_lattice.md`](decode_lattice.md)).

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
step is data plane only, and its shape is a **token-stationary, state-streaming
gather-GEMV**.

## <a id="kronecker-support"></a>2. The structure, and the one fact it rests on

At `T = 1` a leaf `s` is in the token's read support iff EVERY level's digit
`d_l(s)` has a nonzero read amplitude, so

```
supp(R) = prod_l supp_l(R)                      (a Cartesian product)
```

a Cartesian product at EVERY granularity the `(k, m)` box names: an owner
coordinate, a fixed run offset, an innermost digit. That is the one fact the
walk rests on, and it prunes at each granularity in turn rather than at one:

- an owner coordinate dead on a side kills every unit under it — it never
  enters that side's owner list at all
  ([`decode_lattice.md#the-owner-lists`](decode_lattice.md#the-owner-lists));
- a UNIT whose fixed levels have an absent digit on a side is dead on that side
  WHOLE — one register test kills every leaf it would have held
  (`Unit.live_r`/`.live_w`, [`decode_lattice.md#the-unit`](decode_lattice.md#the-unit));
- what survives that is walked over the SET BITS of the innermost level's
  `s_{D-1} <= kDecodeMaxSpan`-wide slice — a fixed, capped cost per live unit,
  not a cost that grows with `N`.

**The leaf order IS the lattice, so there is no map over the leaf space to
consult, exact or bounded.** The step's factor tables — `sum_l ceil(B_l/32)`
mask bits and at most `sum_l g_l` owner-list entries per side
([`decode_lattice.md`](decode_lattice.md)) — are the whole apparatus the walk
reaches a leaf through; nothing here is a term in `N`, and nothing here is
materialized and then swept the way a leaf-granular bitmap would be. The walk
composes a leaf from three small factors instead of testing membership in a
big one.

## <a id="bc-absent"></a>3. `BC` is the lattice's owner, never a launch shape or an address term

With one token there is nothing to amortize across a block of leaves, so the decode descriptor
states no launch-shape `BC` (decode's schema names none, `hardware/kernels/decode/schema.json`; `tests/integration/test_decode_bc_blindness.py`). The `(k, m)`
lattice's `BC = prod_l s_l` is the leaf count one OWNER holds, one unit of the partition the
walk enumerates ([`decode_lattice.md`](decode_lattice.md)): a LAYOUT fact (`λ / BC = O` by
construction). No address is computed from it: the state's address is keyed to the
16-leaf ATOM ([§4](#dense-address)), so `BC` is the walk's unit boundary and nothing more.

## <a id="dense-address"></a>4. The state is one dense plane

<a id="the-atom"></a><a id="katomshift"></a>The page granule is the **atom** — `kAtomLeaves = 16`
consecutive leaves in LATTICE leaf order, the state page's size
(`MMA_K_QUANTUM` in Python). Under the mixed-radix intra-box order an atom is 16
consecutive `local` values at a multiple of 16, so it is an ALIGNED DIGIT RECTANGLE for
every admissible span vector — the levels whose fields lie above bit 4 are fixed, the
level straddling it keeps an aligned contiguous sub-run, and the levels below are free
(`lattice_atom_rectangle`, [`../common/plan/lattice.md`](../common/plan/lattice.md)).
`kAtomLeaves` is geometry-owned and fixed across every config forever, which is exactly why
the address goes through the atom rather than through `BC` ([§3](#bc-absent)).

The state is ONE plane, `[BH, N / 16, page bytes]`, atom-major, every page resident. The walk
resolves an atom's page as a BYTE offset from the head's plane,
`plane + atom * Page::kBytes` with `plane = state + bh * atoms_per_bh * Page::kBytes`, and a
leaf's row is `leaf & 15` inside the page's planes
([`common/state_page.md#geometry`](../common/state_page.md#geometry)). The address is plain
arithmetic at each use: no table stands between an atom and its page and the walk caches
nothing per atom, so it is correct whatever order it reaches leaves in. `N` is a whole number of
atoms by the admission law ([`docs/internals/state.md`](../state.md)).

A leaf never deposited into holds zeros, and the self-normalizing readout then contributes
exactly nothing through it — the mechanical form of the dead-leaf argument.

## <a id="capture"></a>4b. CUDA-graph capture — `decode_step_gated`

The step has no host read inside it, so it is capturable, and `fsla.decode_step_gated`
(`fsla/engine/decode_step.py`) is the decode entry. It takes its inputs only -- `decode_step_gated(routes, values, masks, state, decay=None)` -- and builds its
descriptor itself, by a memoized lookup (`fsla.ops.produce._decode_descriptor`) that after the call's first reads no device and allocates nothing.

Nothing in the loop reallocates: the workspace is built once and every counter is reset by the
last CTA that reads it, so a graph captured over a step replays for the sequence's life. The
layer's one-token `forward` runs the step eagerly; `FSLA._step` -- the producer's launch, the
step and the output projection -- is the same step whole, and a caller that captures it replays
one graph a token (`tests/integration/test_decode_graph_step.py`); capture is an optimization
the caller selects.

## <a id="phase-ledger"></a>4c. The phase ledger

`DecodePhase` books every warp's cycles into four phases -- FOLD (phase A of
[the prologue](#prologue)), FACTOR (phases B and C), WALK (the head's positions and their count,
the claims and the leaves) and COMBINE (the
fold, the head's sums and its finish, and the choice of the next head) -- with the shared
`PhaseClock` (`common/phase_clock.md`), summed over every head the CTA visits; every warp
publishes its row at the CTA's exit. The ledger is bound by `fsla_decode_ledger_bind` as
`[ctas][warps][kDecodePhases]` int64 (`warps` the plan's, `DecodePlan<B>::kWarps` at the card's build), a row per CTA; unbound, nothing is written and the
clock's cost is its reads.

## <a id="max-levels"></a><a id="kmaxlevels-2"></a><a id="kmaxlevels"></a>5. `kMaxLevels` — read from the planner's launch header

`1 <= D <= 4`. The bound has ONE definition, `kDecodeMaxLevels` in the planner's launch header
(`csrc/fsla/src/common/plan/decode_launch.h`, [`../common/plan/decode_launch.md`](../common/plan/decode_launch.md)), and
`decode.cuh`'s `kMaxLevels` reads it: the launch function's per-level widths and the kernel's per-level arrays are
the same size by construction. The contract's width facts (`width_bits_1` .. `width_bits_4`) name the same levels, which
`tests/unit/test_planner.py` ties to the constant. The page granule and an owner's shape come from
the box (`csrc/fsla/src/common/plan/lattice.h`, where the lattice's constants and its law live),
so the two families write and read ONE plane in ONE order; what decode adds is a RUNTIME carrier,
the lattice's fields as scalars under the lattice's own names (§9).

<a id="kproducermaxbranchwidth"></a>
## <a id="level-width-capacity"></a>6. The level-width capacity is a GATE, not a convenience

The CEILING below is the producer's; the FLOOR is the contract's: a level's width is a call fact of the descriptor (`width_bits_k`, its log2), whose range runs from
`fsla._state.MIN_LEVEL_WIDTH` (the one admission law's floor, the same one prefill's seam runs) to this capacity (the depth parts of `hardware/kernels/decode/parts`), so a width
outside it is refused when the descriptor is made, naming the fact: the entry takes its widths from the launch function and keeps no capacity check of its own.

`fsla/routing/topology.py`'s `MAX_BRANCH_WIDTH = 256` is the widest level the
PRODUCER will emit. A decode path whose capacity were narrower would accept a
topology the producer can build and then read past the end of a staged amplitude
row. So the relation `kDecodeMaxLevelWidth >= kProducerMaxBranchWidth` is
enforced in two halves:

- The **compile-time** half is the `static_assert` in `decode.cuh`;
- The **runtime** half is `fsla_decode_capacity()`, bound into the extension so
  `tests/unit/test_decode_capacity.py` can assert the two constants agree.

A C++ assertion cannot read a Python constant. The pair is what makes this
checkable rather than merely written down — and see
[§33](#producer-width-mirror) for why the mirror is exposed separately from the
capacity.

## <a id="decode-threads"></a><a id="kdecodethreads"></a>7. The plan's threads — the width IS the arm

The step launches ONE CTA an SM of the threads its card's registers hold at `kDecodeRegsPerThread` (128) each (`decode_threads`, the one function the launch and the compiled plan share) --
`DecodePlan<B>::kThreads`, 512 (16 warps) on every tabulated card's 64K registers -- and one wave of them: the host takes the declared residency off the kernel's own launch bound and sizes
the grid from it ([§21b](#the-split-policy)). The registers are the whole-atom walk's: a warp
holds a quarter of an atom's leaves a lane group in flight ([`#whole-atoms`](#whole-atoms)),
which the 64-register cap of 1024 threads could not hold without spilling; at 512 no arm
spills, and a visit's prologue is issued by 16 warps
([`#prologue`](#prologue)).

<a id="decodeparams"></a><a id="decodeparams-note-l152"></a><a id="filllevels"></a><a id="near-line-98"></a>
## <a id="params-amps"></a>9. `DecodeParams` — the operands, as addresses

`read[l]` / `write[l]` (the coefficient rows), `read_scale[l]` / `write_scale[l]` (their fp32
maxima; a producer's write gain is a factor of write level 0's), `v` (the bf16 values) and `v_scale` are `Span`s: a
base pointer
and the `(b, h, w)` strides of the CALLER'S OWN `[B, 1, H, *]` view. There is no `[BH, b_l]` plane between the
producer and the walk — the step folds these into shared memory itself
([`decode_fold.md`](decode_fold.md)), so the `[B, 1, H, W] -> [BH, W]` fold is an ADDRESS
the kernel computes rather than a pass that ran.

No level is normalized, on either side: the fold reads each as stored ([`docs/CONSTITUTION.md`
R23](../../CONSTITUTION.md#r23)): u8 digits at every level, a zero an exact zero.

`level_amp_offset[l]` is where level `l`'s staged amplitude row starts inside the flat
`sum_l b_l` SMEM arrays. It is host-computed and carried rather than recomputed, because
the kernel needs it on the innermost path.

The remaining per-level integers are `level_width[l] = b_l` and
`level_row_offset[l]`, an offset into the flattened per-head dial array (both the launch function's, `plan::DecodeArgs`: the widths from the descriptor's `width_bits_k`, the offsets their prefix sums). Everything a
LEAF-order walk would need beyond that — a level's stride or span in the flattened leaf
space — has no counterpart here: the walk never addresses a leaf as a flat integer, only
as the `(owner, fixed run offsets, innermost digit)` decomposition the `(k, m)` box's fields carry
([`decode_lattice.md`](decode_lattice.md), and `lat_*` fields on this same struct).

<a id="decodeparams-note-l174"></a><a id="gsuf"></a><a id="near-line-129"></a><a id="near-line-132"></a>The box's
fields are runtime scalars under the lattice's own names, the launch layout's (`DecodeLayout`): `lat_s[l] = k * m_l` an owner's span, `lat_g[l] = B_l /
lat_s[l]` its owner-grid extent, `lat_gsuf` the grid's suffix products, and `lat_run_shift[l] = log2 prod_{l' > l}
s_l'` the run offset's place in the owner-local index, re-based on the innermost run for the unit index. Every span is
a power of two, so every suffix product is a SHIFT and no device division exists on this path. `mask_off[l]` /
`mask_words[l]` / `mask_total` lay out the digit masks, one plane a side. `fill_levels` fills the per-level geometry
every decode entry validates the same way and reports `total_rows`.

## <a id="detached-clock"></a><a id="log1pf"></a>10. The detached clock, and why there is no sweep

The clock is each leaf's SHARE of the token's write — `c[t, s] = prod_l k_l[d_l(s)] / sum_d k_l[d]`, the write
amplitude times `1 / prod_l sum_l` ([`decode_fold.md#row-sum`](decode_fold.md#row-sum)) — scale-free, so a magnitude a
level carries (a producer's write gain among them) scales the deposit and not the clock. It is therefore not a stream
of its own: the step reads the same staged write rows the deposit reads, and a second fold of one fact would be a
second derivation of it.

Its `stored != 0 => live != 0` guarantee is what makes §2.4's no-sweep claim EXACT: a leaf
outside `supp(W)` has a zero factor at some level, so its clock is `0`, so `keep^0 == 1`
exactly and its stored state is unchanged. Decay therefore fires only on written rows —
and written rows are streamed anyway. **There is no entry sweep and no exit sweep in this
kernel.**

## <a id="state-in-place"></a><a id="state"></a>11. `state` is read and written in place

`state` is the dense `[BH, N / 16, page bytes]` uint8, one
int8-format page an atom ([the int8 page](#int8-state), [§4](#dense-address)), and it is read AND
written in place: a row's read contribution and its write-back are performed by the SAME lane
group in program order, so the read-then-write ordering costs a register and no barrier. The semantics of that order
are [§27](#update-then-read).

`y` is `[BH, d_v]` — the FUSED ratio readout, `num / (den + eps)`.

## <a id="factor-tables"></a>12. The factor tables never leave the CTA that builds them

[§2](#kronecker-support)'s exact support is served by the `(k, m)` lattice's factor tables
— the digit masks and the two owner-coordinate lists
([`decode_lattice.md`](decode_lattice.md)) — built in SHARED MEMORY by the CTA that is
about to walk them. NOTHING PUBLISHES ANYTHING: the step's leaf-level output is the
state's own deposits, and the tables themselves die with
the CTA.

The walk is COUNTED, NOT SCANNED, in the same sense a leaf-space bitmap walk never was:
`resolve_unit` decomposes a unit index by shifts into an owner rank and the fixed run
offsets directly ([`decode_lattice.md#the-unit`](decode_lattice.md#the-unit)), so there is no
prefix-sum structure over the leaf or row space to consult. What it reads is the owner-list's
own realized count `n_o`
per (list, level) — a single small array [`build_lists`](decode_lattice.md#the-owner-lists)
produces with the lists, consulted directly at unit-resolve time rather than through a
second index over it.

## <a id="queue-buffers"></a>13. The queue's buffers

`yacc` is `[BH, cols]` fp32, a head's readout sums, and `queue` is `[3 BH + 2]` int32, the work
queue's counters. Both are allocated once per sequence and reset by the kernel that last reads
them ([the work queue](#queue)).

## <a id="extract-bits"></a><a id="extract-bits-2"></a>14. `extract_bits` — the funnel shift, written once

Extract `count` bits of a packed bitmap starting at BIT index `start`, returned in
the low `count` bits (`count <= 32`). This is the shift-and-merge every lattice
slice ([`decode_lattice.md#the-digit-mask`](decode_lattice.md#the-digit-mask)'s
`level_slice`) is read through — an owner's span at any level, one extract — and
it is written once because it is the one place a funnel shift can be off by one.
A SINGLE DIGIT never straddles a word, so the unit's per-level membership tests
go through `mask_bit` (one load, one shift) instead.

<a id="shift-32-ub"></a>The `(start & 31) == 0` fork is **not** an aligned fast path bolted onto a general
one — it is the C++ shift operator's undefined behaviour at a shift width of 32,
avoided at the one site that can reach it (`lo >> 0 | hi << 32`). Both arms
compute the same function. The standing *"no branch to dodge edge cases"* rule is
satisfied because there is ONE strategy here — gather 32 bits from at most two
source words — and the fork is inside it.

---

# Part B — the translation unit (`decode.cu`)

## <a id="gemv-shape"></a>15. Why it is a GEMV, and where the reductions are not

§6b's ratified shape, literally: the token's routing weights and value are
RESIDENT (SMEM/registers), the STATE streams past, and the grid partitions the
lattice's UNIT RANK SPACE ([`decode_lattice.md#the-unit`](decode_lattice.md#the-unit)).

What makes it a GEMV rather than a GEMM is that there is one token, so the
contraction index is the LEAF and the output index is the COLUMN. That is why
there is no MMA here: a lane holds eight channels of a leaf
([`#leaf-lanes`](#leaf-lanes)) and accumulates their partials in registers for the
whole walk, so the dot never leaves the lane.

The reductions are after the walk and over small index sets: the lane groups'
butterfly, the CTA's fold over its warps, and the head's sums, which every visit
adds into ([the finish](#queue-finish)). The first two are fixed-order; the third
is atomic, in arrival order ([§16](#determinism)).

## <a id="determinism"></a>16. `y` is reproducible to its sums' order, and the state bit for bit

RULED (Blake, 2026-09-29): "I don't care about determinism." The work queue adds each visit's
partial into the head's sums with `atomicAdd`, in ARRIVAL order ([the finish](#queue-finish)),
so `y` moves by the reassociation of a few adds a head, each addend an fp32 sum over a visit's
leaves. The gates compare `y` under that bound (`tests/oracle/fixtures.py`, `reordered_close`:
`rtol = 2^-14`, `atol = 2^-16 * max |y|`).

The STATE stays byte-reproducible, for a structural reason: the write term's positions
PARTITION its leaves ([the positions](#queue-positions)), so every written leaf has exactly one
depositing lane group, and its new bytes are a function of its old bytes and the token's
operands alone. No atomic touches `state`, and the gates compare it bit for bit.

## <a id="no-barrier"></a>17. No barrier per LEAF, and that is the axis that matters

A visit costs a FIXED number of barriers: the prologue's three ([`#prologue`](#prologue)),
the position count's one, the fold's one,
and two around the choice of the next head. None is per position and none is per leaf; the
count never grows with `N` or with the support.

**THE WALK NEVER NEEDED ONE, AND IT DECLARES SO.** The walk sits between `//: @barrier-free` and
`//: @barrier-free-end`, so a CTA barrier in it fails the standards lint. A lane group's read contribution and
its write-back are one statement in program order, and the ordering is UPDATE-THEN-READ because §2's survival is
inclusive at `j = t` ([§27](#update-then-read)); a warp's claims are its own atomics.

<a id="kdecodemaxexponent"></a>
## <a id="decode-max-exponent"></a>18. `kDecodeMaxExponent` — the same convention, structurally inert here

`expf(88) = 1.65e38 < FLT_MAX`: the one clamp convention every survival exponential in the
repository follows, stated here for decode itself.

On this path the clamp is **structurally inert**, and that is worth saying rather
than leaving a reader to wonder: the exponent is `clock * log1pf(-rate)` with
`clock >= 0` and `rate in (0, 1)`, so `log1pf(-rate) < 0` and the exponent is
non-positive. `expf` can underflow to zero (the correct limit) but cannot
overflow.

It is written anyway so that ONE convention governs every survival exponential in
the repository, and so a future decay form that CAN go positive finds the clamp
already in place. `tests/unit/test_decode_capacity.py` pins the two constants
equal.

## <a id="template-axes"></a><a id="cols"></a>20. Why there is no `GLOBAL` axis

Raw normalization is dead everywhere in this repository (revival = git;
[`DELETIONS.md`](../DELETIONS.md)): the mass column and the `num/den` divide are
UNCONDITIONAL, so `cols = DV + 1` at every instantiation and there is nothing left
for a `GLOBAL` template axis to select between. `decode_step_kernel<DV, D,
DECAY>` takes exactly three template parameters.

## <a id="smem-ledger"></a><a id="decode-step-kernel-note-l95"></a>21. The SMEM ledger, and the fold's `+3` pitch

| block | extent | what it holds |
|---|---|---|
| `sa_r`, `sa_w` | `[total_rows]` fp32 | both sides' amplitude rows, UNCOMPACTED and indexed by DIGIT |
| `sd` | `[total_rows]` fp32 | this head's dial row; `DECAY` only |
| `v_tok` | `[DV]` fp32 | the token's value, resident |
| `fold` | `[warps][cols + 3]` fp32 | a warp's partials, its mass and its position count ([the finish](#queue-finish)) |
| `mask_r`, `mask_w` | `[mask_total]` u32 | one bit per digit, per side ([`decode_lattice.md#the-digit-mask`](decode_lattice.md#the-digit-mask)) |
| `olist` | `[4][olist_total]` i32 | the four KINDS' live owners per level, compacted: the i-th live owner, one indexed load ([`decode_lattice.md#the-owner-lists`](decode_lattice.md#the-owner-lists)) |
| `dlist` | `[dlist_total]` i32 | the WRITE side's live digits, OUTER levels only — term 0's own grain ([`decode_lattice.md#the-write-grain`](decode_lattice.md#the-write-grain)) |

Every one of them is FILLED BY THIS CTA, from the raw operands and from nothing else: the
step allocates no plane, reads no `[BH, b_l]` intermediate and publishes no bitmap. That is
the whole of [§21b](#per-cta-prologue).

The amplitude rows are indexed by DIGIT and are not compacted: **the digit mask
carries membership, so the digit IS the index**, and no compaction, id list or
flag packing exists on this path.

<a id="fold-bank-padding"></a>The row is `cols + 3` floats: the `DV` partials, the mass at `DV`, the warp's
position count at `DV + 1`, and the rest padding the stride off a multiple of 32. Warp 0
reads DOWN the warp axis, sixteen rows per column, so an unpadded stride sharing a factor
with 32 would collide; `gcd(cols + 3, 32)` is 4 at `DV = 64` and at `DV = 32`.

## <a id="per-cta-prologue"></a><a id="near-line-169"></a>21b. The prologue is per visit, and that is the design

Every visit a CTA pays to a batch-head folds and factors that head itself
([`#prologue`](#prologue)), then walks the positions it claims. The tables live and die in its
shared memory; nothing is published but the head's position count ([the positions](#queue-positions)).

That redundancy is the price of the factor tables never reaching DRAM. A shared per-head table
would have to be written by someone and read by everyone -- a global round trip on the walk's
hottest index, or a launch boundary to order the write against the reads, which is the launch
this design exists to delete.

**Nothing in the prologue's fixed cost is a term in `N`.** Both digit masks are
`sum_l ceil(B_l/32)` words per side; the owner and digit lists are one warp a job over arrays
at most `lat_g[l] <= kDecodeMaxLevelWidth` wide ([`decode_lattice.md`](decode_lattice.md)) --
bounded by the LEVEL WIDTHS and the OWNER-GRID EXTENTS, never by the leaf space and never by
the state. The leaf-space bitmap this design replaced (`DELETIONS.md`) did carry a term in `N`.

<a id="the-split-policy"></a>**THE GRID IS ONE WAVE OF THE DECLARED RESIDENCY**, work
permitting: the step sizes it itself (`ctas` -1) from the descriptor's `SMs * residency` (`plan::decode_grid`), the residency the launch bound
itself ([§30](#declared-residency)), bounded above by
`BH` times the unit supply, so a small topology does not launch CTAs with nothing to claim. The UNIT SUPPLY is
`ceil(unit_ceiling / kDecodeUnitsPerCta)`, at least one: `unit_ceiling = N / s_{D-1}` counts a walk unit a run of
the innermost level's span -- exact where the atom rectangle is one level, an over-count where it spans more -- and
`kDecodeUnitsPerCta` = 16 (`common/plan/decode_launch.h`) is the smallest share worth a CTA's prologue (its fold and factor tables), so a small
topology is not split past the point the replicated prologue spends the parallelism. A
second wave would pay a head's prologue for work the resident CTAs could claim. Both bounds are
static facts, the same every step and baked into a captured launch; a gate forces a count to prove the result does
not depend on it.

**THE HOST'S CEILING IS WHAT THE DEVICE SPENDS.** The count a step actually enumerates is a
function of ITS OWN SUPPORT and exists only on the device, once the factor tables do
([`decode_lattice.md#the-terms`](decode_lattice.md#the-terms)), and it varies from head to
head. So nothing is sized per head: a head with more live positions takes more claims, and a
CTA that runs out of its heads steals one ([the next head](#queue-next)).

## <a id="row-scales"></a><a id="sr"></a>22. No per-token row scale

At `T = 1` there is no owner rectangle to hoist per-level factors over, so they stay in
the per-LEAF product: each level's scale rides its staged amplitudes (`raw * scale / 255`),
and there is no per-token factor beside them. A producer's write gain is a factor of write
level 0's scale ([`docs/CONSTITUTION.md` R23](../../CONSTITUTION.md#r23)), and a read side has
no gain of its own: the ratio readout cancels any per-token read factor but in the floor.

<a id="t"></a>
## <a id="aligned-subranges"></a>26. The unit's leaves are walked in ascending order, over the innermost run's set bits

<a id="the-term-loop"></a>**A POSITION NAMES ONE TERM**, of the `D + 1` products whose
disjoint union is the set the step touches
([`decode_lattice.md#the-terms`](decode_lattice.md#the-terms)), and each term's index
space is laid out as positions on its own ([the positions](#queue-positions)). That is a LATENCY decision, not
bookkeeping: inside a term every level's KIND is invariant, so the owner-list
bases, the realized counts, the padded field widths and the membership predicates all
hoist out of the unit loop and the per-unit dependent chain is the one the
product-of-unions walk had. The alternative — laying the terms end to end in ONE index
space and naming each unit's term by a scan over `D + 1` offsets — was built and measured
at **+2.3 ns of dependent latency per unit**, which on the scattered cell cost more than
the units the decomposition saves. The loop is not unrolled over `D`, so one copy of the
walk body serves every term.

Term 0 IS the write product, and it is the ONE term whose rank space is not
`live owners x the full run radix`: under the split-level declaration its outer support is
sparse INSIDE the owners it touches, so that space enumerated 2048 units for 256 live ones
at the flagship's scattered cell — and the dead ones, interleaved one-in-eight, idled seven
of every CTA's eight warps. It enumerates LIVE DIGITS instead
([`decode_lattice.md#the-write-grain`](decode_lattice.md#the-write-grain)).
<a id="the-unit-partition"></a><a id="units"></a>The queue hands out each term's UNIT INDEX SPACE directly -- a
position is a unit of a whole-atom write term, or `kGroups` consecutive units of any other
term, one a lane group ([the positions](#queue-positions)) -- so there is no separate
row-range-to-word resolution step: a unit IS its lane group's share, by construction, and
`resolve_unit` ([`decode_lattice.md#the-unit`](decode_lattice.md#the-unit)) is what turns a
unit index into its `(owner, r_0 .. r_{D-2})` pair before any leaf inside it is touched.
Every touched leaf therefore has exactly one owning lane group -- the positions PARTITION the
units, so no atomic ever touches `state` ([§16](#determinism)).

**A WARP TAKES A WHOLE UNIT, NOT A WHOLE OWNER, AND THAT IS A MEASURED CHOICE.** A unit is
already the DRAM granule: under the mixed-radix order its leaves are the innermost level's
contiguous run, which at the flagship spans is one 16-leaf ATOM, so owner-contiguity
across units buys nothing on top of it — the sectors are whole either way. Partitioning at
OWNER grain instead caps the walk's parallelism at the LIVE OWNER COUNT and hands each
warp the entire `BC`-leaf box to walk serially; measured on the wide sparse arm that cost
roughly a factor of two, and at `N = 256` (two live owners against eight warps) rather
more. Positions are claimed in ascending order, so a head's warps together sweep the plane
in one contiguous stretch.

Within one resolved unit the walk visits the SET BITS of THAT TERM'S slice — `cw` for the
write product, `cr` for a read-remainder term, already conditioned by `resolve_unit`
([`decode_lattice.md#the-unit`](decode_lattice.md#the-unit)) — ascending, by `__ffs` — ascending bit index is ascending LEAF order, because the unit's
leaves ARE the innermost level's run in order. A dead innermost digit costs one `__ffs`
step and no address at all, and the two sides' membership comes off the already-resolved
`cr`/`cw` bitfields with no division anywhere. That ascending order keeps a
unit's leaves on one page ([§4](#dense-address)): at the flagship spans the whole unit is ONE atom.

The trip count per unit is `popcount(cr | cw) <= s_{D-1} <= kDecodeMaxSpan`
([`decode_lattice.md#capacities`](decode_lattice.md#capacities)) — a small,
compile-time-boundable cost per LIVE unit, never a cost that varies with `N`.

EVERY LEVEL BUT THE INNERMOST LEAVES THIS LOOP. A unit fixes their digits, so their
amplitude product (and the decay rate's) is formed ONCE in `resolve_unit` and a slot costs
ONE innermost amplitude rather than `D` of them.

A leaf outside the lattice's own partition of `N` is never visited: `owners * BC == N`
exactly (`decode_lattice.md#the-refusals`), so the walk needs no leaf bound test and never
fabricates a row.

<a id="decode-step-kernel-note-l532"></a>
## <a id="update-then-read"></a>27. Update first, then read the updated row — and that order IS the semantics

Not an optimization. §2's survival is INCLUSIVE at `j = t`: the token's own
deposit is read by the token itself with survival exactly `1`.

A multi-token kernel expresses that as TWO terms, because the deposit and the readout
live in different products: the readout against the decayed entry state, plus the
causal intra Gram's self term `G[0,0] * Vw`. At `T = 1` those two are

```
a_r * keep^c * M_entry  +  a_r * a_w * v   ==   a_r * S_exit
```

so **the whole intra Gram collapses into reading the row after it is written**.
One pass over the row, one register, and the causal structure is exact rather than
reconstructed.

A row in `supp(R) \ supp(W)` has `keep == 1` and no deposit, so
`S_exit == M_entry` and the same expression is its readout too — one path, no arm.

Both off-by-ones — decaying after the deposit, and reading before it — are `O(1)`
errors the oracle tier discriminates at the fp32 tolerance.

<a id="decode-step-kernel-note-l541"></a>
## <a id="decay-at-read"></a>28. Decay at read, via the detached clock, with no sweep and no exponent split

At `T = 1` the half-mass shift a multi-token tile applies to keep its exponents bounded is
exactly the IDENTITY. With `P_ref = P[0]/2`:

```
readout:  fr * s0                 collapses to   keep^c
fold:     s0 * (s0 M + W fw Vw)   collapses to   keep^c M + W Vw
```

So on this path there is no clock prefix buffer, no `p_ref`, no `s0`, no division
by a cumulative survival, and no `fw` underflow class — **ONE `expf`, never a quotient
of two.**

`log1pf` IS the composition (never a rounded linear-domain `1 - rate`) and there
is no epsilon floor.

The mass column is the `v := 1` column of the identical recurrence, unconditional
on every arm — one lane-0 statement, never a second path.

## <a id="the-fold"></a>29. The CTA's fold

After its last claim a visit folds its warps' partials in shared memory and adds them into the
head's sums once: one set of atomics a visit, never one a warp
([the finish](#queue-finish)). There is no read-side row scale left to apply: the fold stages
each digit's amplitude at its row's scale ([`decode_fold.md`](decode_fold.md)), and the ratio
readout leaves a read side no gain of its own ([§22](#row-scales)).

## <a id="declared-residency"></a><a id="kdecoderesidentthreads"></a>30. The residency is DECLARED

`decode_step_kernel` carries `__launch_bounds__(Dp::kThreads, Dp::kBlocksPerSm)` (`Dp` its plan at its build), and
the second argument is not decoration: it is a REGISTER CAP the assembler must meet, and the census is what proves it
met it without spilling. `kBlocksPerSm` is 1, so every arm runs at 128 registers a thread --
few warps and full registers, the ideal regime -- and none spills (`tools/sass_gate.py`). The
residency has no template axes: at one CTA an SM there is no CTA of residency for DECAY or the
deepest arm to trade for live state.

## <a id="instantiation-matrix"></a><a id="torchcheck"></a><a id="near-line-51"></a>31. The instantiation matrix

`DV(2) x D(4) x DECAY(2) = 16`. Note what is **not** an axis: no `BT` (one token), no `BC`
(no owner-blocked amortization — [§3](#bc-absent)), no precision axis (there is no MMA),
no `GLOBAL` (raw normalization is dead; the mass column is unconditional
— [§20](#template-axes)), no OPERAND STORAGE TYPE (the fold reads each level's row type as a
runtime dense bit selecting a typed load instead — a template axis there would multiply the
matrix for a few hundred bytes of gather,
[`decode_fold.md#the-dtype-arms`](decode_fold.md#the-dtype-arms)), and no ADDRESS-SPACE arm: the factor tables are
shared memory unconditionally, at
every shipped topology, so there is no second arm for a template parameter to select
between. `d_v` is restricted to `{32, 64}` by a `STD_TORCH_CHECK`, and `D` outside `[1, 4]` is
refused by name. The kernels live in a NAMED namespace (`fsla::decode::detail`): nvcc mangles an
anonymous-namespace kernel with a hash of the build path, so a record made in one worktree
would name a kernel no other checkout's build produces.

<a id="fsla-decode-producer-width-mirror"></a>
## <a id="producer-width-mirror"></a>33. Why the producer-width mirror is bound separately

`fsla_decode_producer_width_mirror()` exposes `kProducerMaxBranchWidth` separately
from the capacity because the two are different claims:

- `capacity >= producer` is the SAFETY property, and the `static_assert`
  ([§6](#level-width-capacity)) covers it;
- `mirror == the real Python constant` is what stops that assertion from being
  satisfied by a STALE mirror after the producer's limit moves.

Only a test that can see both languages can check the second, which is why this is
bound at all.

## <a id="empty-not-zeros"></a><a id="y"></a>34. `y` is `at::empty`, not `at::zeros`

Every element is written by its head's finisher ([the finish](#queue-finish)),
so a zero-fill would be a launch that the launch-latency-bound
small-batch regime pays for nothing — and an unwritten element showing as garbage
rather than as a plausible zero is what a gate wants.

## <a id="smem-refusal"></a>35. The staged-amplitude SMEM budget is a declared capacity

<a id="smem"></a>The step's shared memory is the kernel's own layout (`DecodeSmem`, a `SmemLayout`: [`../common/smem_layout.md`](../common/smem_layout.md)), compiled from the card's budget: each control field, the scan's storage, the clock's rows, the token value and a fold row a warp are fixed tiles, each its own shared object, and what the budget leaves is one tile (`StepWords`) that holds the call's staged rows and tables, addressed at run time. The launch reserves the layout up to the call's last word (`reserved_to`), not the whole tile, under a ceiling raised once a device to the layout's bytes ([`../common/smem_layout.md#reserve`](../common/smem_layout.md#reserve)). The launch function states what a call needs of that tile (`decode_step_words`, `DecodeLayout::step_words`), and the host entry's launch refuses a step that needs more than the tile holds on the card
(`decode_smem_refusal`), naming `sum_l width_l` in the message. That
function's total is the staged amplitude rows, the dial row and BOTH factor-table blocks (the digit masks, the owner
lists compacted from them and the write side's digit lists —
`2 * mask_total + 4 * olist_total + dlist_total` words,
[`decode_lattice.md`](decode_lattice.md)):
everything of a step whose size the call decides. It is **refused loudly
rather than taken as a slower path** — there is no spill-to-global arm for any of it, and
there is no second address space for the factor tables to fall back to either.

---

# Part C — the host side (`fsla/ops/decode.py`, `fsla/engine/decode_step.py`)

The step's ORDER, the scratch's holding discipline and the capturability statement are the
step's page ([`engine/decode_step.md`](../engine/decode_step.md)); what follows is what the
KERNELS impose on the host.

## <a id="decay-domain"></a>36. Decode's decay domain has no admissibility precondition

A lowering that exponentiates `keep^{+P}` and `keep^{-P}` separately is wrong — not merely
rounded — once one saturates while the other underflows, which bounds the routings it can
take. At `T = 1` that half-mass shift is the identity (§28): decode forms ONE exponential,
never a quotient of two, so it has no admissibility precondition to check.

## <a id="forward-only"></a>37. Forward only, and the guard is defence in depth

The decode kernels have no backward. The layer's decode envelope check already
refuses training mode and gradient reachability before the decode fork is reached, so
the module's own check exists for a direct caller.

## <a id="elsewhere"></a>40. Declarations whose reasoning lives on another page

Each declaration below is documented by the page that owns its mechanism.

- <a id="decode-stamp-kernel-note-l63"></a>`decode_stamp_kernel`'s footprint fields:
  [`decode_api.md#build-stamp`](decode_api.md#build-stamp)
- <a id="decodestampkernel"></a>`decode_stamp_kernel`: [`decode_api.md#build-stamp`](decode_api.md#build-stamp)
- <a id="fsladecodebuildstamp"></a>`fsla_decode_build_stamp`: [`decode_api.md#build-stamp`](decode_api.md#build-stamp)
- <a id="decode-step-kernel-note-l137"></a>the packed term table, one word a (term, level):
  [`decode_lattice.md#the-terms`](decode_lattice.md#the-terms)
- <a id="units-2"></a>the term's row of the packed table, hoisted once a term:
  [`decode_lattice.md#the-terms`](decode_lattice.md#the-terms)
- <a id="u"></a>the two resolvers, one walk: [`decode_lattice.md#the-write-grain`](decode_lattice.md#the-write-grain)
- <a id="near-line-150"></a>the write side's live-digit list:
  [`decode_lattice.md#the-write-grain`](decode_lattice.md#the-write-grain)
- <a id="decode-step-kernel-note-l474"></a>the enumerated run is the term's:
  [`decode_lattice.md#the-unit`](decode_lattice.md#the-unit)
- <a id="decodeparams-note-l190"></a>the walk's unit: [`decode_lattice.md#the-unit`](decode_lattice.md#the-unit)
- <a id="decodeparams-note-l221"></a>the compacted owner lists:
  [`decode_lattice.md#the-owner-lists`](decode_lattice.md#the-owner-lists)
- <a id="kunitblockwarps"></a>`kUnitBlockWarps`, the rotation's block:
  [`decode_lattice.md#krotlevel`](decode_lattice.md#krotlevel)
- <a id="kdecodemaxspan"></a>`kDecodeMaxSpan`, the lattice's one capacity:
  [`decode_lattice.md#capacities`](decode_lattice.md#capacities)
- <a id="fill-lattice"></a>`fill_lattice`: [`decode_lattice.md#the-refusals`](decode_lattice.md#the-refusals)
- <a id="floor-inner"></a>the innermost-first capacity (`lattice_box`):
  [`decode_lattice.md#the-refusals`](decode_lattice.md#the-refusals)
- <a id="decode-arm-0"></a>the declared arm list: [`decode_api.md#arm-subset`](decode_api.md#arm-subset)
- <a id="fsladecodearms"></a>`fsla_decode_arms`: [`decode_api.md#arm-subset`](decode_api.md#arm-subset)
- <a id="launch"></a>the one launch site: [`../common/arm_switch.md`](../common/arm_switch.md)
- <a id="rowsf16"></a>the rows' storage type: [`decode_fold.md#the-dtype-arms`](decode_fold.md#the-dtype-arms)
- <a id="spanof"></a>`span_of`, the fold as an address: [`decode_fold.md#the-identity`](decode_fold.md#the-identity)
- <a id="span"></a>`Span`: [`decode_fold.md#the-identity`](decode_fold.md#the-identity)

# Part D — the int8 step: the page, the lanes, the prologue, the work queue

<a id="int8-state"></a>
## The int8 state page, and the unit that reads one

THE STATE'S PAGE IS THE INT8 STATE FORMAT -- `[16 x DV hi s16][16 x DV lo u8][16 x DV/32 exponent s8][16 mass fp32]`,
`common/state_page.md` -- and the plane count is the leaf's own fact for this step:

* a leaf this step WRITES is read in full, `(hi << 8 | lo) * 2^e`, updated in fp32, and stored: a lane holds
  eight consecutive channels ([`#leaf-lanes`](#leaf-lanes)), so a GROUP is four lanes, and its exponent is their
  max of `|v|` (eight maxima in the lane, then two shuffles); each value is rounded once to its group's s24 quantum, and
  the group's first lane stores the exponent and the row's first lane the fp32 mass beside it. The readout then reads the STORED value, so the next step reads exactly what this
  one wrote. The plane choice keys on `in_w`, the step's own write set;
* a leaf this step only READS pulls the **hi plane alone**, `hi * 2^(e + 8)` -- the s16 operand every readout
  takes, a DECLARED statement of the arithmetic rather than an approximation of it -- and the fp32 mass. The
  gates mirror the declared store in fp64 (`tests/oracle/fixtures.py`, `state_rounded`) and allow its quantum.

The address is a BYTE offset: `atom * Blocks<DV>::kBytes` into the head's plane, the
atom-major page sequence with every page resident ([§4](#dense-address)).

PREFILL AND DECODE SHARE ONE ADMISSION LAW (Blake, 2026-08-29): the descriptor. `N`
comes from the level widths and must be a whole number of pages, which a lawful routing
makes it by construction; a width outside the contract (so any ragged `N`) is REFUSED when the descriptor is made, naming the fact, and again by the entry, which runs the contract's check on the descriptor's words it is handed; it is never padded, and
decode carries no shape, tolerance or topology prefill does not. The dense plane is
`[BH, N / 16, page_bytes]` — the same page sequence with every page resident.

<a id="leaf-lanes"></a>
## The leaf's lanes: a warp walks `32 / (DV / 8)` units at once

A leaf row is `DV / 8` lanes of eight channels each (`LeafLanes<DV>`): one 16-byte load of
its hi plane, one 8-byte load of its lo plane, its group's exponent byte and (the row's first
lane) the mass. A warp is therefore `kGroups = 32 / (DV / 8)` lane groups -- four at `DV = 64`
-- and each group resolves and walks its OWN unit, so a warp holds `kGroups` leaves' loads in
flight where it held one, and a unit's resolve and a leaf's address, loop control and
amplitude reads are paid once for `kGroups` leaves. The channels are unpacked by a byte
permute (`state_page.cuh`'s `mantissa8`: the permute's sign replication extends the s24) and
the stored row is packed back the same way (`pack_hi8`, `pack_lo8`).

Why: the one-leaf-a-warp walk measured (sm_86 (ampere branch, unreproducible here), N 64K, d_v 64, H 16 at k ~27, ncu): DRAM 25%,
0.88 eligible warps a scheduler, 66% of the stall samples on the leaf's loads, and ~144 warp
instructions a leaf -- latency- and issue-bound at once, each warp holding one leaf's
loads in flight and paying the whole leaf's overhead per 2-channel slice.

The warp's partial is summed across its groups by a fixed butterfly of `__shfl_xor_sync`
(offsets `DV / 8, 2 DV / 8, ... 16`) after the walk, so every lane of a channel slice ends
holding the warp's sum and the first group deposits it into the CTA's fold
([the finish](#queue-finish)).

<a id="prologue"></a>
## The prologue: three phases, three barriers

Every visit derives its head's tables itself before it walks
([`#per-cta-prologue`](#per-cta-prologue)), in three phases:

- **Phase A -- every operand's load at once.** A value channel's digits and scale
  (the arithmetic deferred past the digit pass, so its load does not queue ahead of the
  digits'), the decay dials, and the digit pass: a warp a mask word, each lane loading its
  digit and its row's scale together, staging `raw * scale / full_scale` where the supplied mask's bit is on and storing the mask's word
  ([`decode_fold.md#fold-digits`](decode_fold.md#fold-digits)).
- **Phase B -- one warp a job.** Warp `l` the level's owner lists, warp `D + l` its write
  digit list ([`decode_lattice.md#build-lists`](decode_lattice.md#build-lists)), and at a
  `DECAY` step warp `2 D - 1 + l` write level `l`'s row sum ([`decode_fold.md#row-sum`](decode_fold.md#row-sum)).
- **Phase C.** The decay's share scale and the term layout, whose barrier ends it.

The visit then lays out the head's positions and publishes their count
([the positions](#queue-positions)).

The digit pass takes the thread's index through `denseref::ops::opaque`, so a slot's level and
offsets are formed a head and not held in registers across every head's walk.

The prologue's floor is the per-warp instruction count: phase A issues ~250
instructions in each of 32 warps, and an empty barrier costs ~400 cycles across 32 warps
(both measured) -- so the next cut is fewer, fuller warps, not a shorter chain.

<a id="whole-atoms"></a>
## The write term by whole atoms

When the innermost level's WRITE side is not listed (every digit of it is walked, the producer's choice for a declaration whose activation has no exact
zeros; the launch function's `write_dense`, derived from the descriptor's innermost `write_listed_k`, which the producer's rule `fsla.ops.produce._decode_descriptor` states from `Topology.activation_properties`) and the innermost
run covers an atom, every write-term unit is a whole atom, and the step walks it as one: a warp
takes a unit, and its `kGroups` lane groups take `16 / kGroups` of the atom's leaves each
(leaf `k * kGroups + grp`). Every leaf's loads are issued before any leaf's math, and for a
fixed `k` the warp's 32 lanes read one contiguous run of the page -- the hi plane in 512-byte
pieces at `DV = 64` -- so the page is read whole, the pattern the microbenchmarks measured at
715-820 GB/s against 446-508 for a leaf's four scattered pieces. Every other term, and a write
term whose innermost level is sparse, keeps a leaf a lane group
([`#leaf-lanes`](#leaf-lanes)); both forms run one statement of the leaf's arithmetic
([`decode_leaf.md`](decode_leaf.md)).

<a id="atom-loads"></a>THE LOADS' PLACE IS STATED, NOT LEFT TO THE ASSEMBLER. An atom's update is bound by memory latency: a warp walks its atoms one
after another, and each waits one round trip for its leaves. The four leaves' sixteen loads are therefore issued together, ahead of the first
leaf's arithmetic, and each leaf's loads sit under `resident` (`ops::launch_known(p.atoms_per_bh)`, [warp.md](../common/atoms/warp.md#launch-known)), a predicate only the launch knows. Without it the
assembler issues some leaves' loads after the first leaf's shuffles, which wait on the first leaf's data, and the atom pays two round trips in
a row.

THE RECORD (2026-11-26; this kernel at sm_90a, toolkit 13.5, the library built four times with only the loads' condition changed; the loop
read is the whole-atom walk of the arm at two levels, 64 channels, no decay; the count is of loads issued after their trip has waited for
another load, over the sixteen arms; the time is a step of 65,536 leaves a head at 64 heads with decay, on an H100 against the commit before
this form, interleaved on one box):

| the loads' condition | loads behind the first shuffle, of 16 | loads behind another's wait, an arm | a step |
|---|---|---|---|
| none (`true`) | 5 | 2 to 14 | 1.60 ms |
| `ops::opaque(1) != 0` | 8 | 2 to 14 | not timed |
| a `ops::once(true)` loop | 5 | 3 to 14 | not timed |
| `ops::launch_known(p.atoms_per_bh)` | 0 | 1 | 0.77 ms |

The commit before this form, whose loads sat under the page table's test, measured 0.77 ms in the same sessions. What moves the loads in
this kernel and not in a small loop of the same shape is not known.

ONLY THE WRITE SUPPORT IS THE TERM'S. A dense declaration does not make every stored digit
nonzero -- a softmax digit can quantize to a zero u8 -- and a leaf outside the write support
belongs to a read term, which reads it. So the atom walk reads and deposits a leaf only under
its write bit (`Unit::cw`), and stores nothing else; reading it here as well double-counted it
in `y` (caught by the oracle's sparse-write cells).

The decision is the declaration's, not the data's: branching on how many of an atom's leaves a
token writes would be a fuzzy threshold (KERNEL_STANDARDS R9). A topology that declares a sparse
innermost write side walks its write term a leaf a lane group, correctly, at the pieces' rate.

MEASURED (sm_86 (ampere branch, unreproducible here), N 64K at (256, 256), d_v 64, B 1, 16 warps of <= 128 registers, no spill): H 16
at k ~27, 159 -> 116 us -- the write term moves its 43.8 MB of pages at ~650
GB/s against the page pattern's ~820, the read terms their ~19 MB of pieces at ~475 against the
pieces' ~480; H 4 at k ~5, 23 -> 16 us. The read terms are at their pattern's ceiling: the next
lever there is the page's layout, not the walk.

<a id="coefficient-rows"></a>
## The coefficient rows, as the bundle holds them

The step reads the bundle's rows as they are: u8 unit-max digits at every level, each row beside its fp32 max; the
value in bf16 against its
largest channel, and the fp32 token scale. The host entry refuses any other type -- the bundle
refuses the same at construction, so a refusal here
names a direct caller of the ABI. How the fold turns each into a staged amplitude:
[`decode_fold.md#raw`](decode_fold.md#raw).

<a id="queue"></a>
## The work queue

A head's work is known only on the device, after its own prologue, and routing makes it
uneven: at the probe cells one head's write term is a few units and another's hundreds. So the
grid is not cut per head. `ctas` CTAs -- one wave of the declared residency
([§21b](#the-split-policy)) -- stay resident and VISIT heads: a CTA starts on head
`blockIdx.x % BH`, takes an unstarted head from a cursor when it finishes one, and once none is
left steals the head with the most work outstanding ([the next head](#queue-next)). A visit
runs the prologue for its head, walks the POSITIONS its warps claim, and adds its partial into
the head's sums.

The queue is `[3 BH + 2]` int32 (`DecodeScratch.queue`): per head the positions CLAIMED, the
positions FINISHED and the position COUNT (`-1` until published), then the
unstarted-head CURSOR and the EXIT count. The sums are `yacc`, `[BH, cols]` fp32. Both are
allocated once per sequence: a head's finisher zeroes its row of sums, and the last CTA to exit
resets every word, so no `memset` launch stands between two steps and the step stays
capturable ([§4b](#capture)).

Nothing waits on another CTA: a visit's claims, its adds and its choice of next head are atomics
on words. The one-wave grid is for cost, not for progress.

<a id="queue-positions"></a><a id="near-line-122"></a>**THE POSITIONS.** A head's positions are its terms' shares, laid
end to end, term by term ([`decode_lattice.md#the-terms`](decode_lattice.md#the-terms)): a
whole-atom write term ([`#whole-atoms`](#whole-atoms)) gives one position a unit, since a warp
walks the unit whole, and every other term `ceil(units / kGroups)`, since a warp's lane groups
take `kGroups` consecutive units ([`#leaf-lanes`](#leaf-lanes)). Thread 0 writes the terms'
first positions to `s_first` and the count past the last, then publishes the head's count to
`s_blocks` and to `q_blocks(bh)` for the stealers (every visitor derives the same one), and one
barrier makes it visible to the CTA. A head whose support is empty still
has one position, which walks nothing, so its finish runs and writes its `y` as
`0 / (0 + eps)`.

<a id="queue-claim"></a>**THE CLAIM.** Lane 0 of a warp claims a position with one `atomicAdd`
on the head's claimed word, and the warp walks it; a claim at or past the count ends the warp's
walk of the head. The NEXT claim is issued before the current position is walked, so its round
trip hides behind the walk. Each warp therefore overshoots the count by one claim a visit,
which is harmless: the word is only ever compared with the count, and reset at exit.

<a id="queue-finish"></a>**THE FINISH.** After its last claim each warp sums its lane groups'
partials by the butterfly ([`#leaf-lanes`](#leaf-lanes)) and writes a row of `fold`: its
partials, its mass and the positions it walked ([§21](#fold-bank-padding)). After one barrier,
warp 0 sums the rows a column a lane and adds the CTA's partial into the head's sums, one
`atomicAdd` a column a visit; then every lane fences its adds and lane 0 adds the visit's
positions to the head's finished word. The visit that brings that word to the count is the
FINISHER: it fences, reads the sums through `__ldcg` (past L1, which the other SMs' adds never
reached), writes `y = sums / (den + eps)`, and zeroes the sums for the next step. A visit that walked nothing adds nothing and counts nothing.

ONE SET OF ATOMICS A VISIT, NOT A WARP. Each warp adding its own partial measured a 12.9K-cycle
combine at `H = 4`: sixteen warps' adds to the same 65 addresses serialize at L2. The fold costs
one barrier and a sixteen-row sum.

<a id="queue-next"></a>**THE NEXT HEAD.** After the finish and a barrier, warp 0 picks the CTA's
next head: an unstarted one from the cursor, while the grid is smaller than `BH`; otherwise the
head with the most positions left (`count - claimed`, read through `__ldcg`, a warp-wide argmax
per 32 heads, the lower index on a tie) among those whose count is published. It steals that
head only if at least the plan's `kWarps` positions are left: a steal pays a whole prologue, so it
takes a head only where every warp gets a position. With nothing to take, the CTA exits, and
the last one out resets the queue.

MEASURED (sm_86 (ampere branch, unreproducible here), N 64K at (256, 256), `d_v = 64`, B 1), a split-K grid (every head cut the same
number of ways) against the queue:

| cell | split-K grid | work queue |
|---|---|---|
| k ~27, H 16 | 116 us | 105.4 us |
| k ~5, H 16 | 31 us | 30.2 us |
| k ~5, H 4 | 16.3 us | 18.2 us |

The cells gain where heads are uneven; the thin `H = 4` cell loses 1.9 us, because a visit
costs more than a split CTA did: a single visit's prologue-to-walk span is ~5.3K cycles against
~3.8K (the positions, the position count's barrier, and the fold's and the next head's barriers), and a
sparse head pays a claim's round trip for every few leaves.
