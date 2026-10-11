<!-- ABSENT: K31_R0_ABI_RIPPLE_SPEC cum intra_panel kPanelMaxExponent GLA_MASS_DECAY_SPEC MmaPrec PREC PANEL_MAX_EXPONENT -->
# `csrc/rola/src/decode/decode.cuh` + `csrc/rola/src/decode/decode.cu` — the `T = 1` decode path

This doc covers **two** source files, because they are one component: the header
carries the parameter block and the objects both decode translation units share,
and the `.cu` carries the GEMV kernel, its dispatch and its host entry. The
mirror rule is one doc per source basename, so `decode.cuh` and `decode.cu` land
here together. **Part A** is the header; **Part B** is the translation unit.

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
([`decode_lattice.md`](decode_lattice.md), spec `K31_R0_ABI_RIPPLE_SPEC.md`
§2.1–2.5); `BT` is the tiled consumer's token tile, which appears here only to
be ruled out.

**The tiled consumer, in this document, is a RETIRED arm.** It was the prefill
kernel this path was designed against; P67 D2 deleted its shipped shape whole
and K31 R2 deleted what remained of it entirely, extracting what decode and
paging still need into `csrc/rola/src/facts/` (`DELETIONS.md`; revival by tag
`baseline/pre-k31` or the deletion commits' parents). The contrasts drawn
against it below are kept deliberately and read in the past tense: decode's
shape is what it is BECAUSE of what it declined to inherit, and a design record
that erases the thing declined leaves the decision unexplained. Where a section
says a constant is restated rather than included, the header it was not
included from is one of the deleted ones, and the restatement is now simply
this file's own constant. There is no surviving prefill consumer to name a
current doc for; the prefill arm is deleted pending K31 the rebuild.

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

## <a id="bc-absent"></a>3. `BC` names TWO different things, and only one of them is decode's

Owner-blocked AMORTIZATION is a property of how the TILED consumer shares a
`[BC][cols]` block's cost across many tokens; with one token there is nothing
to amortize, and `DecodeGeometry`'s own docstring is explicit that `BC` MUST
NOT become one of its fields — that arm never existed here
([`engine/plan.py`](../../../rola/engine/plan.py)).

The `(k, m)` lattice's `BC = prod_l s_l` is a different quantity that DOES
exist on this path: the leaf count one OWNER holds, i.e. one unit of the grid
partition the walk enumerates ([`decode_lattice.md`](decode_lattice.md)).
Nothing about it is a launch shape or an amortization decision — it is a
LAYOUT fact, the same one the tiled consumer's `[BC][cols]` block address used
before it (spec §2.6's survivor table: `λ / BC = O`, by construction rather
than by a checked property). The DENSE state address is `(bh * N + leaf) *
cols + c`, leaf-major already, and the PAGED one is keyed to the 16-leaf ATOM
([§4](#paged-address)) rather than to the owner — so even where `BC` names a
real quantity here, no address in either backing is computed FROM it directly;
it is the walk's unit boundary, not a term either address expression carries.

## <a id="paged-address"></a>4. Paged and dense are ONE walk

<a id="the-atom"></a>The page granule is the **atom** — `kAtomLeaves = 16`
consecutive leaves in LATTICE leaf order, spec §4.5's
`page_states := MMA_K_QUANTUM`. Under the mixed-radix intra-box order an atom is 16
consecutive `local` values at a multiple of 16, so it is an ALIGNED DIGIT RECTANGLE for
every admissible span vector — the levels whose fields lie above bit 4 are fixed, the
level straddling it keeps an aligned contiguous sub-run, and the levels below are free
(`rola/ops/lattice.py`). One condition survives, `16 | BC` (an atom lies inside one
owner), checked at [`fill_lattice`](decode_lattice.md#the-refusals) where a page table is
present; it is what lets the candidate-atom test ([§4b](#candidate-atoms)) read a unit's
membership out of ONE aligned window of its innermost write slice. `kAtomLeaves` is geometry-owned and
fixed across every config forever, which is exactly why the paged address addresses
through the atom rather than through `BC` ([§3](#bc-absent)); it is mirrored in
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
(`if ((leaf >> 4) != cached_atom) reload`), so it costs at most one `int32` load per
16-leaf atom, and — the part worth stating — it is correct whatever order the walk
reaches leaves in. The lattice's own ASCENDING `C` order ([§26](#aligned-subranges)) is
what makes 16 consecutive leaves share a slot and makes the cache CHEAP; nothing about
its correctness depends on that order.

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
the exact admission is dear against the launch it precedes, and a sequence whose routing
has settled writes atoms that are already resident. What makes the question nearly free is
that the step has already built the objects it is asked about: the write-side factor tables
([`decode_lattice.md`](decode_lattice.md)) are the ones the walk itself resolves units from,
so the residency test is a pass over the WRITE list's candidate atoms
([§4b](#candidate-atoms)) and a page-table indirection each, and no second launch, no second
fold and no second factor pass exist at all. There is nothing left for a probe kernel to be.

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

<a id="the-table-test"></a>**THE TABLE TEST.** The count of atoms that are written and unmapped
is a block reduction (`cub::BlockScan::ExclusiveSum`) over one `page_row[a] < 0` test per
candidate atom. The batch-head's OWNER computes it, once, from its own factor tables and
the page table, and publishes the outcome to every other visit
([the admission](#queue-admission)).

<a id="candidate-atoms"></a>**THE CANDIDATE ATOMS ARE ENUMERATED FROM THE SAME LATTICE
PRODUCT THE WALK USES, RESTRICTED TO THE WRITE LIST.** A unit's leaves are a CONTIGUOUS
ALIGNED RUN ([`decode_lattice.md#the-unit`](decode_lattice.md#the-unit)), so an atom and a
unit nest one way or the other and the index says which: the low `atoms_per_unit_shift`
bits of `i` pick an atom INSIDE a unit, and `lat_units_per_atom_shift` consecutive units
make ONE atom. Exactly one of the two shifts is ever non-zero, and both are PAGED-ONLY
quantities the host derives once (`decode_lattice.md#the-refusals`).

`candidate_atom(i)` therefore resolves the write-side unit (or the `1 <<
lat_units_per_atom_shift` consecutive ones) through `resolve_unit` over the CANDIDATE TERM
SLOT `D + 1` — the write product in the `live owners x run radix` space, which is the only
one monotone in atom id and is exactly the enumeration this claim was ratified on
([`decode_lattice.md#the-candidate-slot`](decode_lattice.md#the-candidate-slot)); it shares
the walk's owner lists and needs no list plane of its own — and an atom is live where a resolved unit is `valid && live_w` and the
ALIGNED WINDOW of its innermost write slice that the atom covers is non-zero. At the
flagship spans `(8, 16)` a unit IS an atom, both shifts are zero, and the test is "the
unit's 16-bit write slice is non-zero".

The whole enumeration is
`n_candidates = (units[D + 1] >> lat_units_per_atom_shift) << atoms_per_unit_shift` —
a PARTITION mapping monotonically onto ascending atom ids by construction, because unit
rank and the atom index inside a unit are BOTH ascending in the lattice's own leaf order.
That
ordering is not a convenience: the pool claim's canonical ranks ([§4e](#the-claim)) are
defined in ascending atom id and this enumeration is where that order comes from. The same
enumeration serves the table test, the claim and the condensation below.

<a id="atom-bits"></a>**THE CONDENSATION IS PUBLISHED, and it is the only thing that is.**
the test already has the write-atom set in hand, so writing it out costs one byte per atom
and removes the host derivation from a growth step entirely: `plan_exact` plans from
`scratch.atom_bits` and derives no support. The batch-head's owner publishes it,
UNCONDITIONALLY — a dead atom's `false` is as much of the set as a live one's `true`,
and publishing only the growing batch-heads would leave the buffer carrying an older step's
answer for the others, which is the very buffer the admission then plans from. It is two
passes, a clear over the row and a set over the candidate rectangle, because "everything
outside the write support is false" is a statement about the whole row while the support
is not. The clear is the one remaining term in a steady step that scales with `N`, and it
is one byte per atom on the batch-head's owner.

The leaf-granular map is published by NOTHING. The factor tables live and die inside the
CTA that built them ([§12](#factor-tables)); the step's one leaf-level output is this
16-deep condensation of the write side's.

`rola.ops.decode._write_atom_bitmap` survives as the ENUMERATED REFERENCE for that
condensation — the same set derived from the Cartesian-product definition in torch,
wired into the engine DAG as `write_atom_bitmap`'s `reference_only=True` node
(`rola/engine/dags/decode_dag.py`) and exercised directly by the paging/growth
integration tests (`tests/integration/test_decode_growth_trigger.py`,
`test_decode_graph_step.py`, `test_decode_paging.py`, `test_page_arena_vmm.py`). It
is not a fallback and not a second path a caller may select — `reference_only` is what
`validate_dag` refuses to let an executed recipe select — it is what makes the gate a
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
its owner publishes a position count of zero, so no warp claims, and finishes it with its
growth flag set ([the finish](#queue-finish)) — no walk, no `y`, no state. Its verdict lands
in `growth[bh]` and its write-atom set in `atom_bits`, which is exactly what the host
admission needs and all of it.

The skip is PER BATCH-HEAD and not all-or-nothing, and the thing that used to make that
unsafe is the done flags below. A batch that shares one verdict wastes every resident
sequence's step on the rarest one's growth; a batch that skips per sequence must answer
what a replay does to the sequences that already deposited.

<a id="done-flags"></a>**THE DONE FLAGS.** `DecodeScratch.done` is `[BH]` int32, zeroed
ONCE at build. A batch-head that advances sets its flag. On the caller's replay a done
batch-head **walks and reads but does not deposit** — and reproduces its `y` while doing
so, up to the order its sums are added in ([§16](#determinism)), because
[update-then-read](#update-then-read) means the rows it would deposit into already hold the
deposit and the same expression is their readout. So `y` is
complete for the whole batch after the replay without `y` ever becoming a buffer that
survives between steps, and the contract the caller sees is:

> gated step -> `growth_pending` -> `admit_growth` -> replay
> advances every sequence EXACTLY ONCE.

`tests/integration/test_decode_growth_trigger.py` states it as `torch.equal` on the
gathered plane, and the reordering bound on `y`, against the admit-everything-first order, over a MIXED batch — one
batch-head needy, the rest not — which is the cell that can tell the per-batch-head
contract from the global skip it replaced.

<a id="one-writer"></a>**EVERY CROSS-LAUNCH MUTATION HAS ONE WRITER, AND THAT IS THE WHOLE RACE
ARGUMENT.** The page table, the pool cursor and the done flag are read by every visit to a
batch-head. The table and the cursor are written by the head's OWNER inside its admission,
before it publishes, and no other visit reads either until it has seen that publication
([the admission](#queue-admission)). The done flag is written by the head's FINISHER, which
runs only once every position has been claimed and walked ([the finish](#queue-finish)): a
visit that reads the flag set has no position left to take. Without that, a done flag set
early would make a sibling visit skip its own share of the SAME step.

<a id="growth-reduction"></a>**THE REDUCTION, AND THE RESET.** The last of those per-batch-head
writers to arrive at `growth_ctr` ORs `growth` into the `[1]` `growth_any` the host reads.
Nothing needy is the same statement as every batch-head done, so that one test also decides
whether the done flags are zeroed for the next step: the flags reset themselves exactly
when the round is over, no `memset` launch stands between two decode steps, and the whole
thing is safe to capture. `growth_ctr` resets itself the same way the queue's words do
([the work queue](#queue)).

## <a id="the-claim"></a>4e. The slack pool — a growth step that admits itself

Where the state reserves slack ([`paging.md#the-slack-pool`](../paging/paging.md#the-slack-pool)),
a batch-head whose write set is not resident does not have to go to the host at all: it
takes slots from its own pool region and advances. `pool_cap = 0` — the shipped default —
is exact commitment, and the paragraphs below describe machinery that does not run.

<a id="queue-claim-pool"></a>**THE ASSIGNMENT IS THE OWNER'S.** The batch-head's owner
([the admission](#queue-admission)) maps its touched-unmapped atoms, in ASCENDING ATOM ID, to
`pool_slots[bh][cursor + rank]`, with `rank` from a block scan over the candidate atoms; it
writes the slots into the page table and advances the cursor by the demand, all before it
publishes the head's count. It is ALL-OR-NOTHING: a demand larger than the batch-head's
remaining pool is a needy batch-head and the host path exactly ([§4c](#per-bh-verdict)),
because a partial claim would deposit part of a step and leave the replay to deposit the rest
twice.

**IT WRITES THE TABLE ITSELF.** No other visit reads the head's table row before the owner's
publication, so the claim is ordinary residency the moment the walk can see it, and the walk
addresses through the one table it always does: the walk's body knows nothing about pools.
The split-K grid replicated the claim in every CTA of the batch-head, where a table write
would have moved a sibling's predicate; it wrote a shadow row the walk read beside the table,
and its last CTA folded that row in. One owner deleted all three (`DELETIONS.md`).

**AN ADMITTED ATOM IS AN ADMITTED ATOM.** Pool slots are ordinary arena slots, committed
and ZEROED by the host before they are named, so an atom the device admitted is
indistinguishable from one `plan_exact` admitted: same bytes, same table format, no
migration and no copy anywhere. That is what keeps the dense-vs-paged `torch.equal` gate a
statement about ADDRESSING, and it is asserted directly — a pooled arena and a
host-admitted one reach the same state and the same `y`.

## <a id="capture"></a>4d. CUDA-graph capture — `decode_step_gated`

With no host read inside it the step is capturable, and `rola.engine.dags.decode_dag.decode_step_gated`
is the decode entry. The CALLER owns the verdict, at whatever synchronization it already has
(a decode loop syncs once per token to sample):

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
graph captured over them survives growth. The layer's one-token `forward` runs this loop
eagerly; `RoLA._step` -- the producer's launch, the gated step and the output projection --
is the same step whole, and a caller that captures it replays one graph a token
(`tests/integration/test_decode_graph_step.py`); capture is an optimization the caller
selects, not a fallback either way.

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

## <a id="phase-ledger"></a>4f. The phase ledger

`DecodePhase` books every warp's cycles into five phases -- FOLD (phase A of
[the prologue](#prologue)), FACTOR (phases B and C), ADMIT (the head's positions and its
admission, through the admission's barrier), WALK (the claims and the leaves) and COMBINE (the
fold, the head's sums and its finish, and the choice of the next head) -- with the shared
`PhaseClock` (`common/phase_clock.md`), summed over every head the CTA visits; every warp
publishes its row at the CTA's exit. The ledger is bound by `rola_decode_ledger_bind` as
`[ctas][kDecodeWarps][kDecodePhases]` int64, a row per CTA; unbound, nothing is written and the
clock's cost is its reads.

## <a id="max-levels"></a>5. `kMaxLevels` — restated, not included

Spec §1 gives `1 <= D <= 4`. `decode.cuh` states that bound itself rather than
inheriting it from a prefill header: it was restated rather than included from the
tiled consumer's `consumer.cuh` because the decode TUs shared no other object with
that kernel and a one-way include would have implied they did. There is no prefill
consumer left to include it from at all today (`DELETIONS.md`), which makes the same
choice load-bearing for a different reason now: the SPEC is the shared authority here,
a header is not, and a header cannot be even where one exists.

## <a id="level-width-capacity"></a>6. The level-width capacity is a GATE, not a convenience

The CEILING below is the producer's; the FLOOR is `rola.ops.decode._admit` (K53), which
refuses a `B_l` below the one admission law's floor through `rola._state.StateFormat`,
the same check prefill's seam runs, before a call reaches this capacity check at all.

`rola/routing/topology.py`'s `MAX_BRANCH_WIDTH = 256` is the widest level the
PRODUCER will emit. A decode path whose capacity were narrower would accept a
topology the producer can build and then read past the end of a staged amplitude
row. So the relation `kDecodeMaxLevelWidth >= kProducerMaxBranchWidth` is
enforced in two halves:

- The **compile-time** half is the `static_assert` in `decode.cuh`;
- The **runtime** half is `rola_decode_capacity()`, bound into the extension so
  `tests/unit/test_decode_capacity.py` can assert the two constants agree.

A C++ assertion cannot read a Python constant. The pair is what makes this
checkable rather than merely written down — and see
[§33](#producer-width-mirror) for why the mirror is exposed separately from the
capacity.

## <a id="decode-threads"></a>7. `kDecodeThreads` — the width IS the arm

The step launches ONE CTA of 512 threads an SM -- 16 warps of up to 128 registers -- and one
wave of them: the host takes the declared residency off the kernel's own launch bound and sizes
the grid from it ([§21b](#the-split-policy)). The registers are the whole-atom walk's: a warp
holds a quarter of an atom's leaves a lane group in flight ([`#whole-atoms`](#whole-atoms)),
which the 64-register cap of 1024 threads could not hold without spilling; at 512 no arm
spills, and a visit's prologue is issued by 16 warps where it was 32
([`#prologue`](#prologue)).

## <a id="params-amps"></a>9. `DecodeParams` — the operands, as addresses

`read[l]` / `write[l]` (the coefficient rows), `read_scale[l]` / `write_scale[l]` (their fp32
maxima), `g_write`, `v` (the f16 values) and `v_scale` are `Span`s: a base pointer
and the `(b, h, w)` strides of the CALLER'S OWN `[B, 1, H, *]` view. There is no `[BH, b_l]` plane between the
producer and the walk — the step folds these into shared memory itself
([`decode_fold.md`](decode_fold.md)), so the `[B, 1, H, W] -> [BH, W]` fold is an ADDRESS
the kernel computes rather than a pass that ran.

`normalize[l]` says whether read level `l` still carries its own mass, so the fold divides
it by its row sum. `floor_read` and `floor_write` name each side's dense levels, a bit a level:
their zero digits, in a live row, read as the floor ([`../facts/liveness_contract.md#floor`](../facts/liveness_contract.md#floor)).
`g_bf16` names the gain's storage type; the fold branches on it once a step, never per element ([`decode_fold.md#the-dtype-arms`](decode_fold.md#the-dtype-arms)).

`level_amp_offset[l]` is where level `l`'s staged amplitude row starts inside the flat
`sum_l b_l` SMEM arrays. It is host-computed and carried rather than recomputed, because
the kernel needs it on the innermost path.

The remaining per-level integers are `level_width[l] = b_l` and
`level_row_offset[l]`, an offset into the flattened per-head dial array. Everything a
LEAF-order walk would need beyond that — a level's stride or span in the flattened leaf
space — has no counterpart here: the walk never addresses a leaf as a flat integer, only
as the `(owner, fixed run offsets, innermost digit)` decomposition the `(k, m)` box's fields carry
([`decode_lattice.md`](decode_lattice.md), and `lat_*` fields on this same struct).

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

## <a id="factor-tables"></a>12. The factor tables never leave the CTA that builds them

[§2](#kronecker-support)'s exact support is served by the `(k, m)` lattice's factor tables
— the digit masks and the two owner-coordinate lists
([`decode_lattice.md`](decode_lattice.md)) — built in SHARED MEMORY by the CTA that is
about to walk them. NOTHING PUBLISHES ANYTHING: the step's one leaf-level output is the
write map's 16-deep condensation ([§4b](#atom-bits)), and the tables themselves die with
the CTA.

The walk is COUNTED, NOT SCANNED, in the same sense a leaf-space bitmap walk never was:
`resolve_unit` decomposes a unit index by shifts into an owner rank and the fixed run
offsets directly ([`decode_lattice.md#the-unit`](decode_lattice.md#the-unit)), so there is no
prefix-sum structure over the leaf or row space left to consult here at all — the row
count the tiled consumer's `cum`/coarse-index apparatus existed to answer is not a
question this design asks. What replaces it is the owner-list's own realized count `n_o`
per (list, level) — a single small array [`build_lists`](decode_lattice.md#the-owner-lists)
produces with the lists, consulted directly at unit-resolve time rather than through a
second index over it.

## <a id="queue-buffers"></a>13. The queue's buffers

`yacc` is `[BH, cols]` fp32, a head's readout sums, and `queue` is `[4 BH + 2]` int32, the work
queue's counters. Both are allocated once per sequence and reset by the kernel that last reads
them ([the work queue](#queue)).

## <a id="extract-bits"></a>14. `extract_bits` — the funnel shift, written once

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

RULED (Blake, 2026-09-29): "I don't care about determinism." The split-K grid combined `y`
in slot order, so it was byte-reproducible at a fixed split; the work queue adds each visit's
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
the admission's one (a paged owner's claim and condensation add their own), the fold's one,
and two around the choice of the next head. None is per position and none is per leaf; the
count never grew with `N` or with a leaf-space bitmap's word count, which is the axis the
tiled consumer's per-candidate-tile barriers scaled on.

**THE WALK NEVER NEEDED ONE, AND IT DECLARES SO.** The walk sits between `//: @barrier-free` and
`//: @barrier-free-end`, so a CTA barrier in it fails the standards lint. A lane group's read contribution and
its write-back are one statement in program order, and the ordering is UPDATE-THEN-READ because §2's survival is
inclusive at `j = t` ([§27](#update-then-read)); a warp's claims are its own atomics.

## <a id="decode-max-exponent"></a>18. `kDecodeMaxExponent` — the same convention, structurally inert here

`expf(88) = 1.65e38 < FLT_MAX`. THE SAME VALUE AND THE SAME CONVENTION as the
tiled consumer's intra_panel.cuh carried in kPanelMaxExponent — restated here
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

## <a id="template-axes"></a>20. Why there is no `GLOBAL` axis

Raw normalization is dead everywhere in this repository (revival = git;
[`DELETIONS.md`](../DELETIONS.md)): the mass column and the `num/den` divide are
UNCONDITIONAL, so `cols = DV + 1` at every instantiation and there is nothing left
for a `GLOBAL` template axis to select between. `decode_step_kernel<DV, D,
DECAY>` takes exactly three template parameters.

## <a id="smem-ledger"></a>21. The SMEM ledger, and the fold's `+3` pitch

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

## <a id="per-cta-prologue"></a>21b. The prologue is per visit, and that is the design

Every visit a CTA pays to a batch-head folds and factors that head itself
([`#prologue`](#prologue)), then walks the positions it claims. The tables live and die in its
shared memory; nothing is published but the admission's outcome.

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
permitting: `rola.ops.decode.derive_ctas` takes `SMs * residency`, with the residency read back
off the launch bound itself ([`decode_api.md` §2](decode_api.md#residency)), bounded above by
`BH` times the unit supply, so a small topology does not launch CTAs with nothing to claim. A
second wave would pay a head's prologue for work the resident CTAs could claim. Both bounds are
static facts, frozen for the sequence and baked into a captured launch.

**THE HOST'S CEILING IS WHAT THE DEVICE SPENDS.** The count a step actually enumerates is a
function of ITS OWN SUPPORT and exists only on the device, once the factor tables do
([`decode_lattice.md#the-terms`](decode_lattice.md#the-terms)), and it varies from head to
head. So nothing is sized per head: a head with more live positions takes more claims, and a
CTA that runs out of its heads steals one ([the next head](#queue-next)). The split-K grid
this replaced cut every head the same number of ways at the dense ceiling and retired the
segments its support left empty; the queue has no segment to retire (`DELETIONS.md`).

## <a id="row-scales"></a>22. `sr` and `sw` — the two per-token row scales

At `T = 1` there is no owner rectangle, so the per-LEVEL factors the tiled kernel
hoists into `sr`/`sw` for the fixed levels `l < j` stay in the per-LEAF product
instead. The split between `R[0,s]` and `sr` is an owner-rectangle artefact, and
the product is the same product either way. What remains here is exactly the
per-TOKEN part: the side gains. There is no read-side gain at all: the ratio
readout fixes it to 1 identically, in every arm.

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
([`decode_lattice.md#the-write-grain`](decode_lattice.md#the-write-grain)). The
CANDIDATE-ATOM enumeration ([§4d](#candidate-atoms)) does NOT follow it there: a flat digit
rank is not monotone in atom id, and the pool claim's canonical ranks are defined in that
order, so a term slot `D + 1` holds the write product in the old space and the candidates
read that. Two enumerations of one set, each in the order its consumer needs.

<a id="the-unit-partition"></a>The queue hands out each term's UNIT INDEX SPACE directly -- a
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
`cr`/`cw` bitfields with no division anywhere. That ascending order is what makes
`cached_atom` — the paged base translation's one piece of reuse ([§4](#paged-address)) —
correct and cheap at once: at the flagship spans the whole unit is ONE atom, so the
translation is resolved once per unit.

The trip count per unit is `popcount(cr | cw) <= s_{D-1} <= kDecodeMaxSpan`
([`decode_lattice.md#capacities`](decode_lattice.md#capacities)) — a small,
compile-time-boundable cost per LIVE unit, never a cost that varies with `N`.

EVERY LEVEL BUT THE INNERMOST LEAVES THIS LOOP. A unit fixes their digits, so their
amplitude product (and the decay rate's) is formed ONCE in `resolve_unit` and a slot costs
ONE innermost amplitude rather than `D` of them.

A leaf outside the lattice's own partition of `N` is never visited: `owners * BC == N`
exactly (`decode_lattice.md#the-refusals`), so the walk needs no leaf bound test and never
fabricates a row.

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

## <a id="the-fold"></a>29. The CTA's fold

After its last claim a visit folds its warps' partials in shared memory and adds them into the
head's sums once: one set of atomics a visit, never one a warp
([the finish](#queue-finish)). There is no read-side row scale left to apply: the fold stages
normalized amplitudes ([`decode_fold.md`](decode_fold.md)), and the ratio readout fixes the
read-side gain to 1 ([§22](#row-scales)).

## <a id="declared-residency"></a>30. The residency is DECLARED

`decode_step_kernel` carries `__launch_bounds__(kDecodeThreads, kDecodeBlocksPerSm)`, and the
second argument is not decoration: it is a REGISTER CAP the assembler must meet, and the census
is what proves it met it without spilling. `kDecodeBlocksPerSm` is
`kDecodeResidentThreads / kDecodeThreads = 1`, so every arm runs at 128 registers a thread --
few warps and full registers, the ideal regime -- and none spills (`tools/sass_gate.py`). It has
no template axes: DECAY and the deepest arm used to buy their extra live state by giving up a
CTA of residency, and at one CTA an SM there is none to give up.

## <a id="instantiation-matrix"></a>31. The instantiation matrix

`DV(2) x D(4) x DECAY(2) = 16`. Note what is **not** an axis: no `BT` (one token), no `BC`
(no owner-blocked amortization — [§3](#bc-absent)), no `MmaPrec`/precision split (there is
no MMA, and the tf32x3/bf16 split died with the `PREC` axis everywhere else —
`DELETIONS.md`), no `GLOBAL` (raw normalization is dead; the mass column is unconditional
— [§20](#template-axes)), no OPERAND STORAGE TYPE (the fold reads the rows' type, the gain's
and the value's digit count as runtime flags selecting a typed loop instead — a template axis
there would multiply the matrix for a few hundred bytes of gather,
[`decode_fold.md#the-dtype-arms`](decode_fold.md#the-dtype-arms)), and — the axis this rung
removed — no ADDRESS-SPACE arm: the factor tables are shared memory unconditionally, at
every shipped topology, so there is no second arm for a template parameter to select
between. `d_v` is restricted to `{32, 64}` by a `STD_TORCH_CHECK`, and `D` outside `[1, 4]` is
refused in the dispatch's `default`.

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

Every element is written by its head's finisher ([the finish](#queue-finish)),
so a zero-fill would be a launch that the launch-latency-bound
small-batch regime pays for nothing — and an unwritten element showing as garbage
rather than as a plausible zero is what a gate wants.

## <a id="smem-refusal"></a>35. The staged-amplitude SMEM budget is a declared capacity

The host entry queries `cudaDevAttrMaxSharedMemoryPerBlock` and `STD_TORCH_CHECK`s
`decode_step_smem_bytes(...)` against it, naming `sum_l width_l` in the message. That
function's total is the staged amplitude rows, the dial row, the token value, the
cross-warp fold buffer and BOTH factor-table blocks (the digit masks, the owner liveness
bitmasks and the two lists they compact into —
`2 * mask_total + 4 * omask_total + 4 * olist_total + dlist_total` words,
[`decode_lattice.md`](decode_lattice.md)):
everything a step's prologue puts in shared memory, all of it. It is **refused loudly
rather than taken as a slower path** — there is no spill-to-global arm for any of it, and
there is no second address space for the factor tables to fall back to either.

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
object `rola.engine.facts.liveness.page_bits` folds from the facts pass's words and the same
one the step condenses, reached by neither's route.

## Comments moved from source

Verbatim `//:` prose blocks from `csrc/rola/src/decode/decode.cu`, in source order, one heading per declaration. Source keeps only a short decl block pointing here.

<a id="kdecodemaxexponent"></a>
### `kDecodeMaxExponent`

`expf(88) = 1.65e38 < FLT_MAX`. THE SAME VALUE AND THE SAME CONVENTION as
the tiled consumer's intra_panel.cuh kPanelMaxExponent, restated because decode shared none of
that header's panel algebra and now outlives it. On this path the clamp is STRUCTURALLY INERT -- the exponent
cannot go positive -- and is written anyway so ONE convention governs every survival
exponential in the repository: docs/internals/decode/decode.md#decode-max-exponent

<a id="decode-stamp-kernel-note-l63"></a>
### in `decode_stamp_kernel`, near line 63

THE ENUMERATION'S SHAPE AND THE LAUNCH'S WIDTH ARE PART OF THE FOOTPRINT, alongside
the two struct sizes and the span ceiling. The kind and term counts are what a walk's
unit index MEANS, and the warp count is HOW MANY CTAs derive the step -- and a binary
that disagrees about either is a different step however its structs measure. Both
lessons were paid for: the union-of-products enumeration moved every walk in the
extension without moving one byte of either struct, and the fat-CTA launch shape moved
the derivation count from 256 to 80 without moving the enumeration. IT IS THE MOST
SIGNIFICANT FIELD of the same positional scheme, so the three older fields still
divide out unchanged and the inversion's plausibility bound still holds
(`kKinds * kMaxTerms * kDecodeWarps` is 192 at eight warps, 768 at thirty-two).
THE PARTITION'S GRANULARITY RIDES THE SAME FIELD, additively, because a binary that
cuts the unit space at a different block width walks a different step however its
width and its structs measure -- the round-4 lesson applied to the second of the two
knobs the launch shape now carries:
docs/internals/decode/decode_api.md#build-stamp

<a id="decode-step-kernel-note-l137"></a>
### in `decode_step_kernel`, near line 137

THE TERM TABLE, ONE PACKED WORD PER (term, level): the field's position, its width and
the level's realized count in one read where three arrays used to be read separately.
`D + 2` rows -- the walk's `D + 1` terms and the admission's candidate slot:
docs/internals/decode/decode_lattice.md#the-terms

<a id="sr"></a>
### `sr`

The two per-token row scales. At `T = 1` there is no owner rectangle, so what
remains here is exactly the per-TOKEN part -- the side gains; the per-LEVEL factors
the tiled kernel hoists stay in the per-LEAF product.
docs/internals/decode/decode.md#row-scales

<a id="win-bits"></a>
### `win_bits`

THE CANDIDATE ATOMS, enumerated from the same lattice product the walk uses, and a
PARTITION of the write side's atoms with no straddle case at all -- which is what the
pool claim's canonical ranks need. A unit's leaves are a CONTIGUOUS ALIGNED run, so an
atom and a unit nest one way or the other and the index says which: the low
`atoms_per_unit_shift` bits pick an atom INSIDE a unit, and `lat_units_per_atom_shift`
consecutive units make ONE atom. Exactly one of the two is non-zero.
THE WINDOW IS THE UNIT'S RUN, CLIPPED TO AN ATOM. At the flagship spans `(8, 16)` a
unit IS an atom: `n_units` and the atom index are both one, and membership is the
unit's write slice being non-zero.
IT NEEDS NO SECOND ENUMERATION AT ALL, because TERM 0 IS THE WRITE PRODUCT: the walk's
own first term is the write side, in its own order, with `live_w`/`cw` the
conditioning that term carries. docs/internals/decode/decode.md#candidate-atoms

<a id="paged"></a>
### `paged`

Three outcomes PER BATCH-HEAD: advance, no-op, or re-read without depositing:
docs/internals/decode/decode.md#absorbed-verdict

<a id="a"></a>
### `a`

THE TABLE TEST: how many atoms this batch-head's write side reaches that the page
table does not map, counted by the head's owner and published with its admission:
docs/internals/decode/decode.md#the-table-test

<a id="decode-step-kernel-note-l355"></a>
### in `decode_step_kernel`, near line 355

THE CONDENSATION, published UNCONDITIONALLY by the head's owner: a dead
atom's `false` is as much of the set as a live one's `true`, and publishing only
the growing batch-heads would leave the buffer carrying an older step's answer for
the others -- which is the set a host admission then plans from.

<a id="decode-step-kernel-note-l397"></a>
### in `decode_step_kernel`, near line 397

THE PAGED BASE TRANSLATION, resolved ONCE PER TOUCHED ATOM. The table row is
pinned here (a plain origin, no per-atom widening multiply); the cache is keyed on
the atom VALUE, so it is correct whatever order the walk reaches leaves in.
`atom_base` is the atom's page as a BYTE offset from `p.state`: dense is
`(bh * atoms_per_bh + atom) * kBytes`, paged is `slot * kBytes`.
docs/internals/decode/decode.md#paged-address

<a id="units"></a>
### `units`

THE UNIT IS THE (owner, outer-run) PAIR and the queue hands out ITS OWN index
space -- not the owner's -- so every touched leaf has exactly one owning lane group
and the state needs no atomic:
docs/internals/decode/decode.md#the-unit-partition
A WARP TAKES A WHOLE UNIT, and a unit is the DRAM granule itself: under the
mixed-radix order a unit's leaves are the innermost level's contiguous run, which at
the flagship spans is one 16-leaf ATOM. Owner-contiguity across units buys nothing
on top of that -- the sectors are already whole -- while partitioning at owner grain
would cap the walk's parallelism at the LIVE OWNER COUNT and give each warp the
whole `BC`-leaf box to walk serially.
EVERY LEVEL BUT THE INNERMOST LEAVES THE SLOT LOOP AND THAT IS THE UNIT'S POINT:
a unit fixes their digits, so their amplitude product is formed ONCE, in
`resolve_unit`, and every slot below costs ONE innermost amplitude rather than `D`.
A POSITION NAMES ONE TERM, and that is a LATENCY decision: every level's kind is
invariant inside a term, so the list bases, the counts, the field widths and the
membership predicates are one packed word a level and the per-unit dependent chain is the
union walk's.
The alternative -- one index space, term named per unit -- was BUILT AND MEASURED at
`+2.3 ns` per unit, which cost the scattered cell more than the decomposition saves.
docs/internals/decode/decode.md#the-term-loop

<a id="units-2"></a>
### `units`

THE TERM'S ROW OF THE PACKED TABLE, hoisted once per term: every level's kind,
field width, field position and realized count is loop-invariant inside the term,
so the resolver below reads ONE word per level instead of three:
docs/internals/decode/decode_lattice.md#the-terms

<a id="u"></a>
### `u`

TWO RESOLVERS, ONE WALK. The write product is enumerated from live DIGITS and
every other term from live owners crossed with the run radix; both hand back the
identical `Unit<D>`, and the selector is the term loop's own invariant. Each lane group
resolves its own unit, so one resolve's instructions serve `kGroups` units.
docs/internals/decode/decode_lattice.md#the-write-grain

<a id="decode-step-kernel-note-l474"></a>
### in `decode_step_kernel`, near line 474

THE ENUMERATED RUN IS THIS TERM'S. Term 0 walks the write run; every other term
is disjoint from `supp(W)` by construction and walks the read remainder it names.
A UNION `br | bw` WOULD DOUBLE-COUNT: a write unit whose innermost read digits
leave `W` reaches leaves that the innermost split term also enumerates.
`valid` needs no test here -- `resolve_unit` has already zeroed the slices of a
unit its term does not realize. docs/internals/decode/decode_lattice.md#the-unit

<a id="t"></a>
### `t`

THE SLOT LOOP IS THE INNERMOST LEVEL'S RUN, in ASCENDING lattice order -- the
unit's own contiguous run -- iterated over the TERM's set bits, one leaf a lane group
an iteration, so a dead innermost digit costs one `__ffs` step and no address at all.
The warp iterates while ANY group holds a bit; a group whose run is spent rides the
remaining iterations with its loads and stores predicated off and a zero amplitude. Ascending is what
makes `cached_atom` the walk's only reuse and enough of it:
docs/internals/decode/decode.md#aligned-subranges
NO SPECULATIVE UNROLLING: the trip count is a runtime popcount, and an unrolled
body duplicates every live value of the walk's tightest scope.

<a id="decode-step-kernel-note-l516"></a>
### in `decode_step_kernel`, near line 516

AN ABSENT ATOM READS AS ZEROS AND IS NOT WRITTEN. Exact, not defensive: the
plan admits the step's write set exactly, so an atom with no slot is outside
it, and a never-deposited leaf holds zeros in the dense backing too. The
self-normalizing readout then contributes exactly nothing through it.
docs/internals/decode/decode.md#absent-atom

<a id="decode-step-kernel-note-l532"></a>
### in `decode_step_kernel`, near line 532

UPDATE FIRST, THEN READ THE UPDATED ROW -- AND THAT ORDER IS THE SEMANTICS,
not an optimization: survival is INCLUSIVE at `j = t`, so the whole causal
intra Gram collapses into reading the row after it is written. A row in
`supp(R) \ supp(W)` has `keep == 1` and no deposit, so the same expression is
its readout too -- one path, no arm.
A DONE BATCH-HEAD TAKES THE READ AND NOT THE DEPOSIT, and reproduces its `y`
while doing so: docs/internals/decode/decode.md#done-flags
HAZARD update-then-read -- docs/internals/decode/decode.md#update-then-read

<a id="decode-step-kernel-note-l541"></a>
### in `decode_step_kernel`, near line 541

DECAY AT READ, VIA THE DETACHED CLOCK, WITH NO SWEEP AND NO EXPONENT
SPLIT: at `T = 1` the tiled kernel's rule-1 half-mass shift is exactly the
identity, so ONE `expf` replaces its three. `log1pf` IS the composition
(never a rounded `1 - rate`) and there is no epsilon floor -- both of that
rule's rule 2's prohibitions: docs/internals/decode/decode.md#decay-at-read

<a id="decode-arm-0"></a>
### `DECODE_ARM_0`

THE DECLARED ARM LIST, AS DATA. The decode family's instantiation matrix is the
product `d_v(2) x D(4) x decay(2)`, and it is written out one row per arm so a
BUILD can name a subset of it by index -- `ROLA_DECODE_ARMS=5` compiles arm 5 alone
and nothing else. That is an ITERATION knob and only that: the default below is
every declared arm, so a plain build and every gate build are the closed-world one,
and a subset binary is refused a shape it does not carry rather than mis-launching
it. docs/build.md, docs/internals/decode/decode_api.md#arm-subset
THE LAUNCH IS STILL WRITTEN ONCE. `dispatch_switch.cuh` refuses the MACRO LADDER --
a `#define` per axis whose BODY restates the `<<<...>>>` -- and this is not one: the
macro carries the arm's three CONSTANTS and its comparison, the launch itself is the
single generic lambda below, and no `<<<...>>>` is spelled twice.

<a id="fill-lattice"></a>
### `fill_lattice`

THE (k, m) BOX, FILLED AND REFUSED HERE. The derivation is `rola/ops/lattice.py`'s
`box_shape`, evaluated on runtime widths instead of template arguments -- decode takes
its lattice as data, because one built binary serves every topology the producer can
emit. Every constraint is that header's `static_assert` restated as a runtime check.
docs/internals/decode/decode_lattice.md#the-refusals

<a id="floor-inner"></a>
### `floor_inner`

the lattice's section 2.5 as the atom-rectangle law generalizes it: the INNERMOST
level takes as much capacity as it can carry -- never less than its balanced share,
never less than an atom asks, never more than its own width admits -- and the
rest of `j` is spread balanced over the outer levels, innermost first.  The atom
floor is a PREFERENCE and the width ceiling is the law: a level narrower than an
atom spills its capacity outward and the atom then spans more than one level.

<a id="fill-lattice-note-l960"></a>
### in `fill_lattice`, near line 960

THE UNIT, and the atom it is reconciled with. The unit is the innermost level's run
-- ALWAYS, at every topology -- and the two shifts below are how an atom is named in
terms of it: a run at or above `kAtomLeaves` holds whole atoms, a run below it means
an atom is that many consecutive units. Exactly one of the two is non-zero, and only
the CANDIDATE enumeration reads them; the walk is the same three loops either way.

<a id="rola-decode-producer-width-mirror"></a>
### `rola_decode_producer_width_mirror`

The MIRRORED copy of the producer's `MAX_BRANCH_WIDTH`, bound separately from the
capacity because the two are DIFFERENT CLAIMS -- the second is what stops the
`static_assert` being satisfied by a stale mirror:
docs/internals/decode/decode.md#producer-width-mirror

<a id="rola-decode-residency"></a>
### `rola_decode_residency`

THE STEP'S DECLARED RESIDENCY, in CTAs per SM, read off the same `constexpr` the
`__launch_bounds__` are built from. The grid policy needs it to size one wave, and a
python mirror of a launch bound is exactly the constant that drifts:
docs/internals/decode/decode_api.md#residency

<a id="note-l1083"></a>
### near line 1083

THE TWO BACKINGS, validated where they differ and nowhere else. A page table names
the leaf space (`[BH, N / kAtomLeaves]`) and the plane holds only the slots a plan
committed; without one the plane IS the leaf space. Slot VALUES are not
bounds-checked -- that would be a device-side read of the table, and the plan owns
the bound. docs/internals/decode/decode_api.md#paged-state

<a id="note-l1131"></a>
### near line 1131

THE ADMISSION BUFFERS, checked whichever backing this is: the step's own verdict
lands in them under a page table and they are untouched without one, and validating
them either way is what keeps the two backings ONE ABI:
docs/internals/decode/decode_api.md#the-step

<a id="torch-check"></a>
### `STD_TORCH_CHECK`

THE ATOM'S PLACE IN THE LATTICE, required of a PAGED backing only, and now ONE
condition where the interleaved order needed two: a unit is a contiguous aligned run,
so an atom is a whole number of units or a unit is a whole number of atoms, and only
the second makes the candidate enumeration a partition. The interleave's `k^D >= 16`
and `log2(k) | 4` are discharged by the mixed-radix order itself (`rola/ops/lattice.py`).
A dense backing never asks an atom a liveness question and carries no constraint:
docs/internals/decode/decode.md#candidate-atoms

## Comments moved from source

Verbatim `//:` prose blocks from `csrc/rola/src/decode/decode.cuh`, in source order, one heading per declaration. Source keeps only a short decl block pointing here.

<a id="kmaxlevels"></a>
### `kMaxLevels`

THE PAGE GRANULE AND THE OWNER'S SHAPE COME FROM THE BOX, not from a mirror of it:
`rola/ops/lattice.py` is where the lattice's addressing constants and its
`box_shape` arithmetic live, and the two families write and read ONE plane in ONE order.
What this path adds is the RUNTIME carrier below -- decode takes `(k, m)` and the
per-level widths as data rather than as template arms, so it holds the lattice's fields
as scalars under the lattice's own names and derives them by its formulas.

<a id="kproducermaxbranchwidth"></a>
### `kProducerMaxBranchWidth`

THE LEVEL-WIDTH CAPACITY, a GATE and not a convenience: below the producer's
`MAX_BRANCH_WIDTH` it would accept a topology the producer can emit and then
over-read a staged row. Compile-time half here, runtime half in a bound accessor:
docs/internals/decode/decode.md#level-width-capacity

<a id="kdecodethreads"></a>
### `kDecodeThreads`

THE STEP LAUNCHES ONE CTA PER SM, 16 warps of up to 128 registers, and the width is a
REGISTER decision: a warp holds a quarter of an atom's leaves a lane group in flight, which a
64-register cap (1024 threads) could not, and 32 warps each issuing the prologue's ~250
instructions cost it ~2K cycles of issue. At 1024 threads four arms spilled; at 512 none does.
docs/internals/decode/decode.md#decode-threads

<a id="kunitblockwarps"></a>
### `kUnitBlockWarps`

THE ROTATION'S BLOCK: the low `kUnitBlockWarpsBits` bits of a write unit's index are
rotated into the bottom of the finest outer level's field, so eight consecutive write units --
eight consecutive positions of a whole-atom write term -- are consecutive atoms:
docs/internals/decode/decode_lattice.md#krotlevel

<a id="kdecodemaxspan"></a>
### `kDecodeMaxSpan`

THE LATTICE'S ONE DECLARED CAPACITY, refused loudly at the host entry: an owner's
per-level span is read as a SINGLE 32-bit slice, both when a level's owner grid is
built and when a unit's innermost run is resolved. The interleaved order's second and
third ceilings (a sub-box's leaves, the capacity multiplier) died with it -- under
mixed radix a unit is a contiguous run and there is no sub-box scan to bound.
docs/internals/decode/decode_lattice.md#capacities

<a id="kdecoderesidentthreads"></a>
### `kDecodeResidentThreads`

THE DECLARED RESIDENCY of the step kernel, in threads an SM, and it is a BUDGET rather
than a hope: `__launch_bounds__` turns it into a register cap the assembler must meet, and the
census is what proves it met it without spilling. `kDecodeResidentThreads = 512` is ONE CTA of
`kDecodeThreads = 512` an SM, so the register file admits 128 registers a thread. The walk's
registers are what residency is traded for (few warps, full registers -- the ideal regime); a
second resident CTA would halve them. docs/internals/decode/decode.md#declared-residency

<a id="katomshift"></a>
### `kAtomShift`

The atom as a SHIFT and a MASK. `kAtomLeaves` itself is the box's
(`rola::carry::kAtomLeaves`, imported above): it is geometry-owned and fixed across
every config forever, which is what makes the paged address BC-free.
docs/internals/decode/decode.md#the-atom

<a id="decodeparams-note-l152"></a>
### in `DecodeParams`, near line 152

THE RAW PER-TOKEN OPERANDS, ADDRESSED RATHER THAN COPIED: both sides' routing
levels, the write gain and the token's value, each as the caller's own view. There
is no `[BH, b_l]` plane between the producer and the walk -- the step folds these
into shared memory itself: docs/internals/decode/decode_fold.md

<a id="decodeparams-note-l174"></a>
### in `DecodeParams`, near line 174

THE (k, m) BOX, the lattice's fields as runtime scalars under `rola/ops/lattice.py`'s names:
`lat_s[l] = k * m_l` is an owner's span at level `l`, `lat_g[l] = B_l / lat_s[l]`
its owner-grid extent, and a leaf's lattice index is
`lambda = O * BC + sum_l r_l << lat_run_shift[l]` -- the owner in mixed radix over
`lat_g`, the OWNER-LOCAL index in mixed radix over the runs `lat_s`, most
significant level first. `lat_run_shift[l] = log2 prod_{l' > l} s_l'` is
lattice run_shift, and every suffix product is a SHIFT because every span is a
power of two. docs/internals/decode/decode_lattice.md

<a id="decodeparams-note-l190"></a>
### in `DecodeParams`, near line 190

THE WALK'S UNIT, and the whole of it: a unit is `(owner, r_0 .. r_{D-2})` -- an owner
together with every run offset but the innermost -- so its leaves are the innermost
level's whole run, `1 << lat_inner_bits` of them, CONTIGUOUS and aligned in lattice
order. At the flagship spans `(8, 16)` that run IS an atom, at full occupancy, and
the unit's 16 slots are level 1's digits.
THERE IS NO SECOND UNIT SHAPE. A topology whose innermost span is thinner than an
atom does not get a deeper walk -- it gets SEVERAL UNITS PER ATOM, which the
candidate enumeration absorbs (`lat_units_per_atom_shift`) and the walk never sees.
docs/internals/decode/decode_lattice.md#the-unit

<a id="decodeparams-note-l221"></a>
### in `DecodeParams`, near line 221

THE COMPACTED LISTS THE BITMASK PRODUCES, and they are read exactly as they were
before D1: `olist[kind][level][i]` is the i-th live owner, one indexed load. The
bitmask is the BUILDER, not the reader -- a find-nth-set at use time was measured at
+25 % of the dense step, because the admission resolves one rank per candidate atom
per CTA: docs/internals/decode/decode_lattice.md#the-owner-lists

<a id="decodeparams-note-l233"></a>
### in `DecodeParams`, near line 233

THE ATOM AND THE UNIT, RECONCILED, and PAGED BACKING ONLY. Exactly one of these is
non-zero: a unit holds `1 << atoms_per_unit_shift` atoms, or an atom holds
`1 << lat_units_per_atom_shift` consecutive units. The candidate enumeration is a
PARTITION of the write side's atoms either way -- which is what the pool claim's
canonical ranks need -- and the WALK is the same three loops in both cases.
The host refuses a page table under a lattice whose OWNER is thinner than an atom,
because then an atom would straddle two owners and no grouping of units recovers it:
docs/internals/decode/decode.md#candidate-atoms

<a id="decodeparams-note-l250"></a>
### in `DecodeParams`, near line 250

`[BH, atoms_per_bh]` int32, `-1` = ABSENT; `nullptr` = the dense backing, where the
slot IS the atom. ONE base translation for both:
docs/internals/decode/decode.md#paged-address
MUTABLE because the head's owner writes a pool claim into it, before any other visit
reads it: docs/internals/decode/decode.md#queue-claim-pool

<a id="decodeparams-note-l265"></a>
### in `DecodeParams`, near line 265

`[BH]` int32, THE DONE FLAGS -- `1` where a batch-head has already deposited this
step. A replay re-walks a done batch-head for its `y` and deposits NOTHING, which
is what makes "exactly once per sequence" hold across the admission the replay
exists for: docs/internals/decode/decode.md#done-flags

<a id="decodeparams-note-l275"></a>
### in `DecodeParams`, near line 275

THE SLACK POOL -- physical slots pre-committed and pre-zeroed by the host, from
which a growth step admits ITSELF. `pool_slots` is `[BH, pool_cap]` slot ids,
`pool_cursor` `[BH]` how many of each batch-head's are spent.
`pool_cap = 0` (and two null pointers) is EXACT-COMMIT: every growth step goes to
the host, which is the shipped default: docs/internals/decode/decode.md#the-claim

<a id="extract-bits-2"></a>
### `extract_bits`

EXTRACT `count` bits of a packed bitmap starting at BIT index `start`, returned in
the low `count` bits (`count <= 32`) -- the shift-and-merge every lattice slice is
read through, written once because it is the one place a funnel shift can be off by
one. docs/internals/decode/decode.md#extract-bits

## More comments moved from source

Verbatim trailing/banner `//` comments from `csrc/rola/src/decode/decode.cu` condensed at their call site into a short pointer.

<a id="decode-step-kernel-note-l95"></a>
### in `decode_step_kernel`, near line 95

---- SMEM ledger ----------------------------------------------------------
  sa_r, sa_w   [total_rows] fp32   both sides' amplitude rows, UNCOMPACTED and
                                   indexed by DIGIT: the digit IS the index.
  sd           [total_rows] fp32   this head's decay dials; DECAY only
  v_tok        [DV]         fp32   the token's value, resident
  fold  [warps][cols+3]     fp32   a warp's partials, mass and position count, bank-padded
  mask_r,mask_w [mask_total] u32   one bit per digit, per side
  olist  [4][olist_total]   i32    the four kinds' live owners per level, compacted
  dlist  [dlist_total]      i32    the WRITE side's live digits, OUTER levels only

## More comments moved from source

Verbatim trailing/banner `//` comments from `csrc/rola/src/decode/decode.cuh` condensed at their call site into a short pointer.

<a id="note-l3"></a>
### near line 3

THE T=1 DECODE PATH -- the parameter block and the objects both decode TUs share.

T = 1 HAS NO STAGE 2: every stage-2 decision is degenerate, so this path bypasses
selection ARCHITECTURALLY and its config freezes at the prefill->decode boundary.
What is left per step is data plane only, in the ratified shape: a
TOKEN-STATIONARY, STATE-STREAMING gather-GEMV.

Three facts a reader must not miss:
  1. THE LEAF ORDER IS THE LATTICE, in the ONE order `rola/ops/lattice.py` defines: a leaf is
     named by its owner and by its per-level RUN OFFSETS in mixed radix, and the step
     reaches it by WALKING those factors -- there is no leaf map, no bitmap over the
     leaf space, and no term in `N` in the step's fixed cost.
  2. supp(R) = prod_l supp_l(R) at T = 1, so the touched set is a PRODUCT at every
     lattice granularity and the walk prunes at each one.
  3. PAGED AND DENSE ARE ONE WALK. The page granule is the ATOM (16 leaves), which is
     what keeps the address BC-free; a null page table means the slot IS the atom and
     the bytes are the dense path's, exactly.

The lattice, the walk's cost accounting and the paged base translation:
docs/internals/decode/decode.md

<a id="near-line-51"></a>
### near line 51

NOT an anonymous namespace: nvcc mangles anonymous-namespace kernels with a hash that
depends on the BUILD PATH, so a manifest ratified in one worktree certifies a name no
other checkout's build produces. A ratifiable kernel needs a named symbol.

<a id="decodestampkernel"></a>
### `decode_stamp_kernel`

THE STEP'S OWN FOOTPRINT, recompiled with this translation unit, so a test that reads
it back is reading a fact only the loaded binary can produce:
docs/internals/decode/decode_api.md#build-stamp

<a id="cols"></a>
### `cols`

The mass column is UNCONDITIONAL (raw is dead), which is why the shape is a
template axis and not a runtime test: docs/internals/decode/decode.md#template-axes

<a id="near-line-122"></a>
### near line 122

One unit count per TERM: each term's rank space is laid out as positions on its own
(docs/internals/decode/decode.md#queue-positions).

<a id="near-line-169"></a>
### near line 169

`sum_l B_l` bits and at most `sum_l g_l` entries: nothing in this prologue is sized
in the leaf space: docs/internals/decode/decode_lattice.md

<a id="candfld"></a>
### `cand_fld`

THE CANDIDATE SLOT'S ROW of the same packed term table -- the ADMISSION's rank space,
which is the write product in the OLD owner-and-run order because only that order is
monotone in atom id: docs/internals/decode/decode_lattice.md#the-candidate-slot

<a id="ncandidates"></a>
### `n_candidates`

FROM THE CANDIDATE SLOT, not from the walk's write term: the two enumerate the same
set and only this one is monotone in atom id, which is what the claim's canonical
ranks are defined in: docs/internals/decode/decode_lattice.md#the-write-grain

<a id="log1pf"></a>
### `log1pf`

THE CLOCK IS THE STORED WRITE ALLOCATION ITSELF -- no side gain, which is
what makes "stored != 0 => live != 0" an identity:
docs/internals/decode/decode.md#detached-clock

<a id="seen"></a>
### `seen`

THE REDUCTION THE HOST READS, and the done flags' self-reset. Nothing needy means
every batch-head is done, so the two answers are ONE test and no launch stands
between two steps: docs/internals/decode/decode.md#growth-reduction

<a id="roladecodearms"></a>
### `rola_decode_arms`

THE BUILT SET, READ BACK OFF THE SAME LIST: `[d_v, D, decay]` per arm, in the order
the binary carries them, so a subset build is SELF-IDENTIFYING rather than merely
smaller: docs/internals/decode/decode_api.md#arm-subset

<a id="torchcheck"></a>
### `STD_TORCH_CHECK`

The instantiation matrix is `DV(2) x D(4) x DECAY(2) = 16`. What is NOT an axis --
`BT`, `BC`, the lattice's `(k, m)` and every operand's storage type -- is the
interesting part: docs/internals/decode/decode.md#instantiation-matrix

<a id="launch"></a>
### `launch`

ONE LAUNCH SITE. Each axis arrives as an `integral_constant`, so the three
`::value`s are template arguments and the `<<<...>>>` configuration cannot drift
between arms: docs/internals/dispatch_switch.md#one-launch

<a id="isbf16"></a>
### `is_bf16`

ONE FLOATING AXIS OVER THE CLOSED SET THE PRODUCER AND THE VALUE PROJECTION EMIT: an
unmatched dtype is a REFUSAL rather than a silent widening on the host.
docs/internals/decode/decode_fold.md#the-dtype-arms

<a id="spanof"></a>
### `span_of`

The `[B, 1, H, W] -> [B * H, W]` fold as an ADDRESS. Each operand's `B` and `H` are checked
against the value's, not only their product (a transposed pair would fold to the right count
and the wrong heads). `T != 1` is refused -- the fold is defined on one token and a second
one would be silently dropped:
docs/internals/decode/decode_fold.md#the-identity

<a id="gsuf"></a>
### `gsuf`

lattice run_shift: level `l`'s run offset sits at `log2 prod_{l\' > l} s_l\'` in the
owner-local index, so the whole mixed radix is shifts and masks.

<a id="filllevels"></a>
### `fill_levels`

The per-level geometry every decode entry validates the same way, filled into `p` and
reported back as `total_rows`.

<a id="roladecodebuildstamp"></a>
### `rola_decode_build_stamp`

THE BUILD STAMP: a path or hash check cannot catch a stale `.so` at the right path,
and only a device-side fact can: docs/internals/decode/decode_api.md#build-stamp

<a id="y"></a>
### `y`

`empty`, not `zeros`: every element is written, so a zero-fill is a launch the
launch-latency-bound small-batch regime pays for nothing:
docs/internals/decode/decode.md#empty-not-zeros

<a id="pooled"></a>
### `pooled`

THE SLACK POOL, present or absent as ONE decision -- two buffers and a capacity,
or three zeros and exact-commit. There is no half-configured pool:
docs/internals/paging/paging.md#the-slack-pool

<a id="kmaxlevels-2"></a>
### `kMaxLevels`

`1 <= D <= 4`. The same bound the consumer's `kMaxLevels` states, restated
rather than included -- the decode TUs share no other object with the tiled
consumer. docs/internals/decode/decode.md#max-levels

<a id="span"></a>
### `Span`

ONE OPERAND'S ADDRESS, as the producer left it: the base and the three strides the
fold walks over a `[B, 1, H, W]` view. `sw` is `0` for the per-token scalars, which
have no last axis. docs/internals/decode/decode_fold.md#the-identity

<a id="decodeparams"></a>
### `DecodeParams`

EVERY POINTER AND EXTENT ONE DECODE STEP NEEDS, and nothing the kernel could cheaply
recompute. Field by field: docs/internals/decode/decode.md#params-amps

<a id="near-line-98"></a>
### near line 98

Per READ level: does it still carry its own mass, so the fold divides it by its row
sum? The write side is never normalized.

<a id="rowsf16"></a>
### `rows_f16`

The coefficient rows' storage type, `1` = f16 and `0` = u8. Branching on the DECLARED
type selects a typed gather loop once per side, never a test per element:
docs/internals/decode/decode_fold.md#the-dtype-arms

<a id="near-line-129"></a>
### near line 129

`r_l`'s shift inside the unit index: `lat_run_shift[l]` re-based on the innermost run,
so a unit decomposes by shifts alone and no device division exists on this path.

<a id="near-line-132"></a>
### near line 132

The digit masks' word layout: level `l` owns `mask_words[l]` words at `mask_off[l]`
of a `mask_total`-word plane, one plane per side.

<a id="near-line-150"></a>
### near line 150

THE WRITE SIDE'S LIVE-DIGIT LIST, OUTER LEVELS ONLY -- term 0's own grain, compacted
from `mask_w` by the same prefix popcount:
docs/internals/decode/decode_lattice.md#the-write-grain

<a id="state"></a>
### `state`

DENSE `[BH, N / 16, page bytes]` or PAGED `[slots, page bytes]` uint8, one int8-format page a slot
(`common/state_page.md#geometry`).
Read AND written in place by the SAME lane in program order:
docs/internals/decode/decode.md#state-in-place

<a id="growth-2"></a>
### `growth`

`[BH]` int32, THE PER-BATCH-HEAD VERDICT: `1` where this step writes an atom no
table entry and no pool slot can address, `0` where it advanced. Written by the
head's finisher, or a needy head's owner: docs/internals/decode/decode.md#per-bh-verdict

<a id="growthany"></a>
### `growth_any`

`[1]` int32, the OR of `growth` over batch-heads, written by the last head to finish,
which the self-resetting `growth_ctr` elects:
docs/internals/decode/decode.md#growth-reduction

<a id="atombits"></a>
### `atom_bits`

`[BH, atoms_per_bh]` bool, this step's own write set condensed to atoms. It is what
a growth step's `plan_exact` admits from, so the host never re-derives the set the
step has already expanded: docs/internals/decode/decode.md#atom-bits

<a id="int8-state"></a>
### The int8 state page, and the unit that reads one

THE STATE'S PAGE IS THE INT8 STATE FORMAT -- `[16 x DV hi s16][16 x DV lo u8][16 x DV/32 exponent s8][16 mass fp32]`,
`common/state_page.md` -- and the plane count is the leaf's own fact for this step:

* a leaf this step WRITES is read in full, `(hi << 8 | lo) * 2^e`, updated in fp32, and stored: a lane holds
  eight consecutive channels ([`#leaf-lanes`](#leaf-lanes)), so a GROUP is four lanes, and its exponent is their
  max of `|v|` (eight maxima in the lane, then two shuffles); each value is rounded once to its group's s24 quantum, and
  the group's first lane stores the exponent and the row's first lane the fp32 mass beside it. The readout then reads the STORED value, so the next step reads exactly what this
  one wrote. The plane choice keys on `in_w`, the step's own write set, and NOT on whether THIS PASS performs the
  deposit: a gated step's third outcome is "re-read without depositing" (`done[bh]`), and a re-read of a leaf the
  step wrote must see the same bytes the depositing pass saw, or two admission orders would compute different `y`
  for the same step. `deposit` gates the STORE and nothing else;
* a leaf this step only READS pulls the **hi plane alone**, `hi * 2^(e + 8)` -- the s16 operand every readout
  takes, a DECLARED statement of the arithmetic rather than an approximation of it -- and the fp32 mass. The
  gates mirror the declared store in fp64 (`tests/oracle/fixtures.py`, `state_rounded`) and allow its quantum.

The address is a BYTE offset now: `slot * Blocks<DV>::kBytes` under a page table, and
`(bh * atoms_per_bh + atom) * kBytes` dense -- the dense backing is the same
atom-major page sequence with every page resident, which is what makes the two
backings one addressing law rather than two.

PREFILL AND DECODE SHARE ONE ADMISSION LAW (Blake, 2026-08-29): the descriptor. `N`
comes from the level widths and must be a whole number of pages, which a lawful routing
makes it by construction; a ragged `N` is REFUSED at this seam rather than padded, and
decode carries no shape, tolerance or topology prefill does not. The dense plane is
`[BH, N / 16, page_bytes]` — the same page sequence with every page resident.

<a id="leaf-lanes"></a>
### The leaf's lanes: a warp walks `32 / (DV / 8)` units at once

A leaf row is `DV / 8` lanes of eight channels each (`LeafLanes<DV>`): one 16-byte load of
its hi plane, one 8-byte load of its lo plane, its group's exponent byte and (the row's first
lane) the mass. A warp is therefore `kGroups = 32 / (DV / 8)` lane groups -- four at `DV = 64`
-- and each group resolves and walks its OWN unit, so a warp holds `kGroups` leaves' loads in
flight where it held one, and a unit's resolve and a leaf's address, loop control and
amplitude reads are paid once for `kGroups` leaves. The channels are unpacked by a byte
permute (`state_page.cuh`'s `mantissa8`: the permute's sign replication extends the s24) and
the stored row is packed back the same way (`pack_hi8`, `pack_lo8`).

Why: the one-leaf-a-warp walk measured (sm_86, N 64K, d_v 64, H 16 at k ~27, ncu): DRAM 25%,
0.88 eligible warps a scheduler, 66% of the stall samples on the leaf's loads, and ~144 warp
instructions a leaf -- latency- and issue-bound at once, each warp holding one leaf's
loads in flight and paying the whole leaf's overhead per 2-channel slice.

The warp's partial is summed across its groups by a fixed butterfly of `__shfl_xor_sync`
(offsets `DV / 8, 2 DV / 8, ... 16`) after the walk, so every lane of a channel slice ends
holding the warp's sum and the first group deposits it into the CTA's fold
([the finish](#queue-finish)).

<a id="prologue"></a>
### The prologue: three phases, three barriers

Every visit derives its head's tables itself before it walks
([`#per-cta-prologue`](#per-cta-prologue)), in three phases:

- **Phase A -- every operand's load at once.** The gain, a value channel's digits and scale
  (the arithmetic deferred past the digit pass, so its load does not queue ahead of the
  digits'), the decay dials, and the digit pass: a warp a mask word, each lane loading its
  digit and its row's scale together, staging `raw * scale / full_scale` (or the raw digit,
  for a level the fold normalizes) and balloting the word
  ([`decode_fold.md#fold-digits`](decode_fold.md#fold-digits)).
- **Phase B -- one warp a job.** Warp `l` the level's owner lists, warp `D + l` its write
  digit list, warp `2 D - 1 + l` a normalized level's declared sum
  ([`decode_lattice.md#build-lists`](decode_lattice.md#build-lists)).
- **Phase C.** A normalized level's amplitudes and the term layout, whose barrier ends it.

The visit then lays out the head's positions and takes its admission
([the admission](#queue-admission)).

MEASURED (sm_86, N 64K at (256, 256), d_v 64, H 16, k ~5, the phase ledger): the prologue it
replaced was 16.5K cycles a warp -- the digit masks 5.8K (sixteen threads each reading 32
digits one shared load at a time), the operand fold 5.0K (a global round trip per level
and side, one after another), the owner masks, compaction, term layout and admission 5.2K
over four more barriers. This one is 5.85K: phase A 2.9K, B 1.2K, C 0.9K, the dense
admission 0.8K. Its floor is the per-warp instruction count: phase A issues ~250
instructions in each of 32 warps, and an empty barrier costs ~400 cycles across 32 warps
(both measured) -- so the next cut is fewer, fuller warps, not a shorter chain.

<a id="whole-atoms"></a>
### The write term by whole atoms

When the innermost level's WRITE side is dense by declaration (its activation has no exact
zeros; `DecodeGeometry.write_dense`, from `Topology.activation_properties`) and the innermost
run covers an atom, every write-term unit is a whole atom, and the step walks it as one: a warp
takes a unit, and its `kGroups` lane groups take `16 / kGroups` of the atom's leaves each
(leaf `k * kGroups + grp`). Every leaf's loads are issued before any leaf's math, and for a
fixed `k` the warp's 32 lanes read one contiguous run of the page -- the hi plane in 512-byte
pieces at `DV = 64` -- so the page is read whole, the pattern the microbenchmarks measured at
715-820 GB/s against 446-508 for a leaf's four scattered pieces. Every other term, and a write
term whose innermost level is sparse, keeps a leaf a lane group
([`#leaf-lanes`](#leaf-lanes)); both forms run one statement of the leaf's arithmetic
([`decode_leaf.md`](decode_leaf.md)).

ONLY THE WRITE SUPPORT IS THE TERM'S. A dense declaration does not make every stored digit
nonzero -- a softmax digit can quantize to a zero u8 -- and a leaf outside the write support
belongs to a read term, which reads it. So the atom walk reads and deposits a leaf only under
its write bit (`Unit::cw`), and stores nothing else; reading it here as well double-counted it
in `y` (caught by the oracle's sparse-write cells).

The decision is the declaration's, not the data's: branching on how many of an atom's leaves a
token writes would be a fuzzy threshold (KERNEL_STANDARDS R9). A topology that declares a sparse
innermost write side walks its write term a leaf a lane group, correctly, at the pieces' rate.

MEASURED (sm_86, N 64K at (256, 256), d_v 64, B 1, 16 warps of <= 128 registers, no spill): H 16
at k ~27, 159 -> 116 us dense (124 paged) -- the write term moves its 43.8 MB of pages at ~650
GB/s against the page pattern's ~820, the read terms their ~19 MB of pieces at ~475 against the
pieces' ~480; H 4 at k ~5, 23 -> 16 us. The read terms are at their pattern's ceiling: the next
lever there is the page's layout, not the walk.

<a id="coefficient-rows"></a>
### The coefficient rows, as the bundle holds them

The step reads the bundle's rows as they are (ruled 2026-10-03/04): u8 unit-max digits, a dense
write level's f16 at 1 with its floor stored, each row beside its fp32 max; the write gain fp32 or bf16; the value in f16 against its
largest channel, and the fp32 token scale. The host entry refuses any other type -- the bundle
refuses the same at construction, so a refusal here
names a direct caller of the ABI. How the fold turns each into a staged amplitude:
[`decode_fold.md#raw`](decode_fold.md#raw).

<a id="queue"></a>
### The work queue

A head's work is known only on the device, after its own prologue, and routing makes it
uneven: at the probe cells one head's write term is a few units and another's hundreds. So the
grid is not cut per head. `ctas` CTAs -- one wave of the declared residency
([§21b](#the-split-policy)) -- stay resident and VISIT heads: a CTA starts on head
`blockIdx.x % BH`, takes an unstarted head from a cursor when it finishes one, and once none is
left steals the head with the most work outstanding ([the next head](#queue-next)). A visit
runs the prologue for its head, walks the POSITIONS its warps claim, and adds its partial into
the head's sums.

The queue is `[4 BH + 2]` int32 (`DecodeScratch.queue`): per head the positions CLAIMED, the
positions FINISHED, the position COUNT (`-1` until published) and the admission STATE, then the
unstarted-head CURSOR and the EXIT count. The sums are `yacc`, `[BH, cols]` fp32. Both are
allocated once per sequence: a head's finisher zeroes its row of sums, and the last CTA to exit
resets every word, so no `memset` launch stands between two steps and the step stays
capturable ([§4d](#capture)).

Nothing waits on a CTA that is not running: a follower waits only on its head's owner, which is
running because it won the swap ([the admission](#queue-admission)), and no CTA waits for
another to exit. The one-wave grid is for cost, not for progress.

<a id="queue-positions"></a>**THE POSITIONS.** A head's positions are its terms' shares, laid
end to end, term by term ([`decode_lattice.md#the-terms`](decode_lattice.md#the-terms)): a
whole-atom write term ([`#whole-atoms`](#whole-atoms)) gives one position a unit, since a warp
walks the unit whole, and every other term `ceil(units / kGroups)`, since a warp's lane groups
take `kGroups` consecutive units ([`#leaf-lanes`](#leaf-lanes)). Thread 0 writes the terms'
first positions to `s_first` and the count past the last. A head whose support is empty still
has one position, which walks nothing, so its finish runs and writes its `y` as
`0 / (0 + eps)`; a needy head has none.

<a id="queue-claim"></a>**THE CLAIM.** Lane 0 of a warp claims a position with one `atomicAdd`
on the head's claimed word, and the warp walks it; a claim at or past the count ends the warp's
walk of the head. The NEXT claim is issued before the current position is walked, so its round
trip hides behind the walk. Each warp therefore overshoots the count by one claim a visit,
which is harmless: the word is only ever compared with the count, and reset at exit.

<a id="queue-admission"></a>**THE ADMISSION.** A paged head's admission -- the table test, the
pool claim and the condensation (§4b, [§4e](#queue-claim-pool)) -- is decided once and
published. The first visit to swap the head's state from 0 to 1 is its OWNER: it runs all
three and writes the count (0 if needy); every thread fences its own writes; then one swap
publishes the state word `kQueueReady`, with `kQueueNeedy` if the head is needy. Every other
visit is a FOLLOWER: thread 0 polls the state (`atomicAdd` of 0, `__nanosleep(64)` between
polls) until it is ready, and after the barrier that follows every thread fences and reads the
state and the count itself (`KERNEL_STANDARDS.md` §12: a fence orders only its own thread). A
stealing visit never owns: it only takes a head whose count is published. An unpaged head, and
a paged head a replay re-reads (its done flag set), has nothing to decide: every visit derives
the same count from its own prologue and writes it.

The split-K grid replicated the admission in every CTA of the head instead, which was sound
because those CTAs ran together from identical inputs. A queue's visits to one head arrive
at different times -- a stealer arrives mid-walk -- so the decision has one owner, and the
claim can write the page table itself ([§4e](#queue-claim-pool)).

<a id="queue-finish"></a>**THE FINISH.** After its last claim each warp sums its lane groups'
partials by the butterfly ([`#leaf-lanes`](#leaf-lanes)) and writes a row of `fold`: its
partials, its mass and the positions it walked ([§21](#fold-bank-padding)). After one barrier,
warp 0 sums the rows a column a lane and adds the CTA's partial into the head's sums, one
`atomicAdd` a column a visit; then every lane fences its adds and lane 0 adds the visit's
positions to the head's finished word. The visit that brings that word to the count is the
FINISHER: it fences, reads the sums through `__ldcg` (past L1, which the other SMs' adds never
reached), writes `y = sums / (den + eps)`, zeroes the sums for the next step, and on a paged
step sets the head's done flag and clears its growth flag before
[the verdict's reduction](#growth-reduction). A needy head's owner finishes that head instead,
with its growth flag set. A visit that walked nothing adds nothing and counts nothing.

ONE SET OF ATOMICS A VISIT, NOT A WARP. Each warp adding its own partial measured a 12.9K-cycle
combine at `H = 4`: sixteen warps' adds to the same 65 addresses serialize at L2. The fold costs
one barrier and a sixteen-row sum.

<a id="queue-next"></a>**THE NEXT HEAD.** After the finish and a barrier, warp 0 picks the CTA's
next head: an unstarted one from the cursor, while the grid is smaller than `BH`; otherwise the
head with the most positions left (`count - claimed`, read through `__ldcg`, a warp-wide argmax
per 32 heads, the lower index on a tie) among those whose count is published. It steals that
head only if at least `kDecodeWarps` positions are left: a steal pays a whole prologue, so it
takes a head only where every warp gets a position. With nothing to take, the CTA exits, and
the last one out resets the queue.

MEASURED (sm_86, N 64K at (256, 256), `d_v = 64`, B 1, the split-K grid at `4aed3f17` against
the queue):

| cell | split-K grid | work queue |
|---|---|---|
| k ~27, H 16, dense | 116 us | 105.4 us |
| k ~27, H 16, paged | 124 us | 116.0 us |
| k ~5, H 16, dense | 31 us | 30.2 us |
| k ~5, H 16, paged | -- | 36.0 us |
| k ~5, H 4, dense | 16.3 us | 18.2 us |

The dense cells gain where heads are uneven; the thin `H = 4` cell loses 1.9 us, because a visit
costs more than a split CTA did: a single visit's prologue-to-walk span is ~5.3K cycles against
~3.8K (the positions, the admission's barrier, and the fold's and the next head's barriers), a
sparse head pays a claim's round trip for every few leaves, and a paged head's followers wait
out its owner's admission -- and then fence and read its publication thread by thread, ~0.8 us
of each paged cell.
