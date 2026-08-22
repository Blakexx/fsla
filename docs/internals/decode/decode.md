# `csrc/rola/src/decode/decode.cuh` + `csrc/rola/src/decode/decode.cu` — the `T = 1` decode path

This doc covers **two** source files, because they are one component: the header
carries the parameter block and the objects both decode translation units share,
and the `.cu` carries the step's TWO GEMV kernels — one per side
([§25](#per-side-enumeration)) — their dispatch and its host entry. The
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
this repository, decode included); `n_split` the split-K degree; `k`, `m`,
`s_l`, `g_l`, `BC` the `(k, m)` lattice's own symbols
([`decode_lattice.md`](decode_lattice.md), spec `K31_R0_ABI_RIPPLE_SPEC.md`
§§2.1–2.5); `BT` is the tiled consumer's token tile, which appears here only to
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
current doc for; the prefill arm is deleted pending K31 R2's rebuild (queue K31
card).

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

and the same statement with `W` for `R` is the write support. Each is a
Cartesian product at EVERY granularity the `(k, m)` box names: an owner
coordinate, a fixed run offset, an innermost digit. That is the one fact each
pass's walk rests on, and it prunes at each granularity in turn rather than at
one:

- an owner coordinate dead on that pass's side kills every unit under it — it
  never enters the side's owner list at all
  ([`decode_lattice.md#the-owner-lists`](decode_lattice.md#the-owner-lists));
- a UNIT whose fixed levels have an absent digit on that side is dead WHOLE —
  one register test kills every leaf it would have held
  (`Unit.live`, [`decode_lattice.md#the-unit`](decode_lattice.md#the-unit));
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
(`carry/box.cuh`, K3/K4). One condition survives, `16 | BC` (an atom lies inside one
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

Both passes skip it, and the readout's skip is EXACT rather than an approximation of a
zero row: `acc + ar * 0` is `acc` in fp32 at every rounding mode, so declining to touch
the row and reading zeros out of it are the same number
([§25](#per-side-enumeration)).

## <a id="growth"></a>4b. Growth — the step's own new work, and its own question

A step may deposit into an atom that is not resident. At `T = 1` the write support is
a Cartesian product ([§2](#kronecker-support)), so the step's write-ATOM set is the
per-level exact-nonzero indicators' product ORed 16-deep.

<a id="absorbed-verdict"></a>**THE STEP ASKS THE QUESTION ITSELF, IN ITS OWN PROLOGUE.**
The exact admission is dear against the launch it precedes, and a sequence whose routing
has settled writes atoms that are already resident. What makes the question nearly free is
that the DEPOSIT PASS has already built the objects it is asked about: the write-side
factor tables ([`decode_lattice.md`](decode_lattice.md)) are the ones that pass's own walk
resolves units from, so the residency test is a pass over the write side's candidate atoms
([§4b](#candidate-atoms)) and a page-table indirection each, and no second launch, no second
fold and no second factor pass exist at all. There is nothing left for a probe kernel to be.

So the order per step is the step's two launches ([§25](#per-side-enumeration)), and then,
only where the answer was yes and no pool covered it, `rola_op`'s order at `L = 1`:

```
decode_step        ->  the deposit pass: fold, factor, THE VERDICT, the write walk
                       and the write-ATOM set, per batch-head; then the readout
                       pass, which reads the verdict and takes y
growth_host.copy_  ->  the ONE 4-byte read, after the launches that produced it
  no  -> done                      ->  the walk ran, against the residency it had
  yes -> state._decode_entry_arena ->  plan_exact(scratch.atom_bits, fresh=False)
         arena.wait()              ->  the ONE event the launch waits on
         _decode_step              ->  the same step again, and the batch-heads that
                                       already ran do not run twice ([§4c](#done-flags))
```

<a id="the-table-test"></a>**THE TABLE TEST.** The count of atoms that are written and unmapped
is a block reduction (`cub::BlockScan::ExclusiveSum`) over one `page_row[a] < 0` test per
candidate atom. Every CTA of a batch-head computes it, independently and identically: the
two objects it reads are the CTA's own factor tables and a page table that no launch
mutates until every CTA of that batch-head has arrived ([§4c](#the-last-arriver)).
Recomputing beat publishing when it was measured — the split CTAs' recompute of the fold
and the factor tables is what removed the launch — and the same argument governs here.

<a id="candidate-atoms"></a>**THE CANDIDATE ATOMS ARE ENUMERATED FROM THE SAME LATTICE
PRODUCT THE WALK USES, RESTRICTED TO THE WRITE LIST.** A unit's leaves are a CONTIGUOUS
ALIGNED RUN ([`decode_lattice.md#the-unit`](decode_lattice.md#the-unit)), so an atom and a
unit nest one way or the other and the index says which: the low `atoms_per_unit_shift`
bits of `i` pick an atom INSIDE a unit, and `lat_units_per_atom_shift` consecutive units
make ONE atom. Exactly one of the two shifts is ever non-zero, and both are PAGED-ONLY
quantities the host derives once (`decode_lattice.md#the-refusals`).

`candidate_atom(i)` therefore resolves the unit (or the `1 <<
lat_units_per_atom_shift` consecutive ones) through `resolve_unit` over the DEPOSIT PASS'S
OWN owner list — the write side's, which is the only list that pass holds — and an atom is
live where a resolved unit is `valid && live` and the
ALIGNED WINDOW of its innermost write slice that the atom covers is non-zero. At the
flagship spans `(8, 16)` a unit IS an atom, both shifts are zero, and the test is "the
unit's 16-bit write slice is non-zero".

The whole enumeration is
`n_candidates = (units >> lat_units_per_atom_shift) << atoms_per_unit_shift` —
a PARTITION mapping monotonically onto ascending atom ids by construction, because unit
rank and the atom index inside a unit are BOTH ascending in the lattice's own leaf order.
That
ordering is not a convenience: the pool claim's canonical ranks ([§4e](#the-claim)) are
defined in ascending atom id and this enumeration is where that order comes from. The same
enumeration serves the table test, the claim and the condensation below.

<a id="atom-bits"></a>**THE CONDENSATION IS PUBLISHED, and it is the only thing that is.**
The test already has the write-atom set in hand, so writing it out costs one byte per atom
and removes the host derivation from a growth step entirely: `plan_exact` plans from
`scratch.atom_bits` and derives no support. One CTA per batch-head of the DEPOSIT pass
(`seg == 0`) publishes it, UNCONDITIONALLY — a dead atom's `false` is as much of the set as a live one's `true`,
and publishing only the growing batch-heads would leave the buffer carrying an older step's
answer for the others, which is the very buffer the admission then plans from. It is two
passes, a clear over the row and a set over the candidate rectangle, because "everything
outside the write support is false" is a statement about the whole row while the support
is not. The clear is the one remaining term in a steady step that scales with `N`, and it
is one byte per atom on one CTA per batch-head.

The leaf-granular map is published by NOTHING. The factor tables live and die inside the
CTA that built them ([§12](#factor-tables)); the step's one leaf-level output is this
16-deep condensation of the write side's.

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

**THE VERDICT IS PUBLISHED BY THE DEPOSIT PASS AND READ BY THE READOUT PASS.** The deposit
pass derives it (the table test below) and its last-arriving CTA writes it; the readout
pass, ordered after that launch ([§25](#per-side-enumeration)), reads `growth[bh]` and
returns immediately where it is set. So the needy verdict is derived ONCE per step rather
than once per pass, and the two passes cannot disagree about which batch-heads ran.

The skip is PER BATCH-HEAD and not all-or-nothing, and the thing that used to make that
unsafe is the done flags below. A batch that shares one verdict wastes every resident
sequence's step on the rarest one's growth; a batch that skips per sequence must answer
what a replay does to the sequences that already deposited.

<a id="done-flags"></a>**THE DONE FLAGS.** `DecodeScratch.done` is `[BH]` int32, zeroed
ONCE at build. A batch-head that advances sets its flag. On the caller's replay a done
batch-head **deposits nothing and reads everything** — its deposit pass declines, its
readout pass runs — and reproduces its `y` BIT FOR BIT
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
by that batch-head's LAST-ARRIVING CTA of the DEPOSIT PASS — elected from `ctr[bh]`, the
split-K counter the readout pass's combine also elects from, which the needy path
increments too so that the election is the same statement in both branches. The deposit
pass resets `ctr[bh]` at the election, so the readout pass finds it at zero and the one
counter serves both.
A CTA arrives only after it has finished reading, so no CTA of
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
keeps the admission inside the deposit pass and keeps it capturable. It is ALL-OR-NOTHING: a demand larger
than the batch-head's remaining pool is a needy batch-head and the host path exactly
([§4c](#per-bh-verdict)), because a partial claim would deposit part of a step and leave
the replay to deposit the rest twice.

**IT WRITES A SHADOW ROW, NOT THE TABLE, AND THAT IS FORCED.** The claim's predicate is
`page_tbl[a] < 0`. A prologue write into the table would change a sibling CTA's predicate,
which would change its ranks, and two CTAs would then assign the same slot to different
atoms — the ordering ambiguity that makes replicated assignment unsound. So the claim
writes `pool_map[bh]`: a WHOLE copy of that batch-head's table row with the claim merged
in, every atom present, claimed or resident or absent. The DEPOSIT WALK then addresses
through
THAT ROW INSTEAD of the table, by one pointer chosen once in the prologue, and the walk's
own body knows nothing about pools — no extra test, no extra load, no register held across
it. The batch-head's last-arriving CTA folds the row's new entries into the real table and
advances the cursor ([§4c](#the-last-arriver)), so by the next step the claim is ordinary
residency and the shadow row is dead.

**THE FOLD LANDS BEFORE THE READOUT LAUNCH, WHICH IS WHY THAT PASS CARRIES NO POOL
POINTER AT ALL.** By the time the readout pass runs, the page table holds exactly what the
shadow row held, so the readout addresses through `p.page_tbl` and nothing else: one map,
chosen by the backing rather than by whether this step claimed
([§25](#per-side-enumeration)). The shadow row's whole life is inside the deposit pass.

**AN ADMITTED ATOM IS AN ADMITTED ATOM.** Pool slots are ordinary arena slots, committed
and ZEROED by the host before they are named, so an atom the device admitted is
indistinguishable from one `plan_exact` admitted: same bytes, same table format, no
migration and no copy anywhere. That is what keeps the dense-vs-paged `torch.equal` gate a
statement about ADDRESSING, and it is asserted directly — a pooled arena and a
host-admitted one reach the same state and the same `y`.

## <a id="capture"></a>4d. CUDA-graph capture — `decode_step_gated`

With no host read inside it the step is capturable, and `rola.engine.dags.decode_dag.decode_step_gated`
is that entry: the same two kernels in the same order as `decode_forward` minus the verdict
read. BOTH LAUNCHES ARE CAPTURED — the ordering the semantics rest on is stream order, and
a captured graph preserves it as a dependency edge rather than as a wall-clock accident
([§25](#per-side-enumeration)); `tests/integration/test_decode_graph_step.py` is the gate.
The CALLER owns the verdict, at whatever synchronization it already has (a decode
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
that kernel and a one-way include would have implied they did. There is no prefill
consumer left to include it from at all today (`DELETIONS.md`), which makes the same
choice load-bearing for a different reason now: the SPEC is the shared authority here,
a header is not, and a header cannot be even where one exists.

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
written in place: the DEPOSIT PASS loads a row, decays it, adds the token's deposit and
stores it back — one lane, one row, in program order, with no second plane and no
staging buffer anywhere. The READOUT PASS only loads. The ordering between the two is the
launch boundary ([§25](#per-side-enumeration)) and its semantics are
[§27](#update-then-read).

`y` is `[BH, d_v]` — the FUSED ratio readout, `num / (den + eps)`.

## <a id="factor-tables"></a>12. The factor tables never leave the CTA that builds them

[§2](#kronecker-support)'s exact support is served by the `(k, m)` lattice's factor tables
— ONE side's digit mask and ONE side's owner-coordinate list per pass
([`decode_lattice.md`](decode_lattice.md)) — built in SHARED MEMORY by the CTA that is
about to walk them, out of the amplitudes that CTA folded. A pass holds its own side's
tables and no others: the deposit pass has no read mask, the readout pass has no write
one. NOTHING PUBLISHES ANYTHING: the step's one leaf-level output is the
write map's 16-deep condensation ([§4b](#atom-bits)), and the tables themselves die with
the CTA.

The walk is COUNTED, NOT SCANNED, in the same sense a leaf-space bitmap walk never was:
`resolve_unit` decomposes a unit index by shifts into an owner rank and the fixed run
offsets directly ([`decode_lattice.md#the-unit`](decode_lattice.md#the-unit)), so there is no
prefix-sum structure over the leaf or row space left to consult here at all — the row
count the tiled consumer's `cum`/coarse-index apparatus existed to answer is not a
question this design asks. What replaces it is the owner-list's own realized count `n_o`
per level — a single small array the block scan in
[`build_owner_list`](decode_lattice.md#the-owner-lists) already produces, consulted
directly at unit-resolve time rather than through a second index over it.

## <a id="splitk-workspace"></a>13. The split-K workspace

`ws` is `[BH, n_split, cols]` and `ctr` is `[BH]`. The workspace is SLOT-ORDERED
and summed by the last-arriving CTA, which is what makes decode's `y`
RUN-REPRODUCIBLE. Why that diverges from the tiled consumer on purpose is
[§16](#determinism).

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

The STATE is unconditionally reproducible for a stronger reason: the deposit pass's WRITE
unit partition is a PARTITION, so every written row has exactly one owning CTA, warp
and lane, and no atomic touches `state` at all.

## <a id="no-barrier"></a>17. No barrier per LEAF, and that is the axis that matters

Each pass's fold and factor-table prologue costs `O(D)` barriers — the fixed ones (the
readout pass's row sums then its rows; the deposit pass's single fold barrier, since the
write side is never normalized and waits on nothing), the mask barrier, plus ONE PER LEVEL
inside [`build_owner_list`](decode_lattice.md#the-owner-lists), because each level's
owner-grid block scan needs its own barrier before the next level's can reuse
the same `TempStorage`. A pass builds ONE side's list, so that term is at most
`kMaxLevels = 4` per pass — one barrier per level, not one per (side, level) — and the
step's total is the two passes' sums (plus three more per pass where `n_split > 1`). That is a function of `D` and never
of `N` or of a
leaf-space bitmap's word count, which is the axis the tiled consumer's
per-candidate-tile barriers scaled on. Once a pass's tables are built its WALK
itself needs none: the deposit's load, decay and store of a row are one
statement in program order, the readout's load is one, and UPDATE-THEN-READ is the LAUNCH
ORDER between the two ([§25](#per-side-enumeration), [§27](#update-then-read)) — the one
ordering in this design that a barrier could never have expressed, since it is grid-wide.

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

## <a id="template-axes"></a>20. Why there is no `GLOBAL` axis, and what `NC` is

Raw normalization is dead everywhere in this repository (revival = git;
[`DELETIONS.md`](DELETIONS.md)): the mass column and the `num/den` divide are
UNCONDITIONAL, so `cols = DV + 1` at every instantiation and there is nothing left
for a `GLOBAL` template axis to select between. `decode_deposit_kernel<DV, D, DECAY>`
takes exactly three template parameters and `decode_readout_kernel<DV, D>` takes two:
**DECAY IS NOT AN AXIS OF THE READOUT PASS**, because decay applies at the deposit and
nowhere else, so that pass carries no dials, no clock and no exponential to select
between ([§28](#decay-at-read)).

`NC = DV / 32` is columns per lane. The row load is `NC` fully coalesced 128 B
transactions and the lane's accumulators are its own output columns — the whole
reason no reduction is needed on the dot ([§15](#gemv-shape)).

## <a id="smem-ledger"></a>21. The SMEM ledger, and the fold's `+3` pitch

**TWO LEDGERS, ONE PER PASS, and each holds ONE SIDE.** `decode_deposit_smem_bytes` and
`decode_readout_smem_bytes` size them; neither pass stages the other's amplitudes, mask or
owner list, and there is no `sa_r`/`sa_w` pair anywhere.

The DEPOSIT pass:

| block | extent | what it holds |
|---|---|---|
| `sa_w` | `[total_rows]` fp32 | the write amplitude rows, UNCOMPACTED and indexed by DIGIT |
| `sd` | `[total_rows]` fp32 | this head's dial row; `DECAY` only |
| `v_tok` | `[DV]` fp32 | the token's value, resident |
| `mask_w` | `[mask_total]` u32 | one bit per write digit ([`decode_lattice.md#the-digit-mask`](decode_lattice.md#the-digit-mask)) |
| `olist` | `[olist_total]` i32 | the write owner-coordinate list, per level ([`decode_lattice.md#the-owner-lists`](decode_lattice.md#the-owner-lists)) |

The READOUT pass:

| block | extent | what it holds |
|---|---|---|
| `sa_r` | `[total_rows]` fp32 | the read amplitude rows, UNCOMPACTED and indexed by DIGIT |
| `fold` | `[warps][cols + 3]` fp32 | the cross-warp partials |
| `mask_r` | `[mask_total]` u32 | one bit per read digit |
| `olist` | `[olist_total]` i32 | the read owner-coordinate list, per level |

The host sizes both and checks the LARGER against the per-block budget
([§35](#smem-refusal)), because the two launches are two occupancy questions and only the
worse of them can refuse the step.

Every one of them is FILLED BY THE CTA THAT WALKS IT, from the raw operands and from nothing else: the
step allocates no plane, reads no `[BH, b_l]` intermediate and publishes no bitmap. That is
the whole of [§21b](#per-cta-prologue).

The amplitude rows are indexed by DIGIT and are not compacted: **the digit mask
carries membership, so the digit IS the index**, and no compaction, id list or
flag packing exists on this path.

<a id="fold-bank-padding"></a>The `+3` pads the fold's row stride to `cols + 3` so the eight warps' stores land
on distinct banks. The fold reads DOWN the warp axis, so an unpadded stride
sharing a factor with 32 would collide. `gcd(cols + 3, 32)` is 4 at `cols = 65`
and 1 at `cols = 32` and at `cols = 64 + 3`.

## <a id="per-cta-prologue"></a>21b. The prologue is per-CTA, and that is the design

Each pass is three phases — fold, factor, walk — and each grid is
`(n_split, BH)`, so the first two run once per SPLIT-K CTA rather than once per batch-head.
Every CTA re-derives the same amplitude rows and the same factor tables for its `bh`, then
walks only its own share of the unit rank space. **THE PROLOGUE IS THEREFORE PAID TWICE
PER STEP, ONCE PER PASS** — but each payment folds ONE SIDE: the deposit pass reads the
write levels, the value and the dials and never touches a read level; the readout pass
reads the read levels and their row sums and never touches a write level, the value or the
dials. The per-pass prologue is strictly smaller than a both-sides one, and its cost is
the honest half of [§25](#per-side-enumeration)'s accounting.

That redundancy is the price of two properties:

- **The factor tables never reach DRAM.** A shared per-`bh` table would have to be written
  by someone and read by everyone, which is a global round trip on the walk's hottest
  index — or a launch boundary to order the write against the reads, which is the launch
  this design exists to delete.
- **The split partition stays.** Collapsing the grid to one CTA per batch-head
  (`n_split = 1`) costs no recompute at all, but the walk is row-starved without the
  split — a topology's `N` is exactly what decides how starved, so there is no regime
  where collapsing the grid is the better shape.

**Nothing in the recompute's fixed cost is a term in `N`.** Building a pass's mask is
`sum_l ceil(B_l/32)` words; building its owner list is a block scan per
level, `D` levels, over an array at most `lat_g[l] <= kDecodeThreads` wide
([`decode_lattice.md`](decode_lattice.md)) — bounded by the LEVEL WIDTHS and the OWNER-GRID
EXTENTS, never by the leaf space and never by the state, which is why paying it `n_split`
times still lands ahead of paying one extra launch once. This is a structural property of
the walk itself, not a tuning outcome: the leaf-space bitmap this design replaced (its
construction, its cost and its removal are recorded in `DELETIONS.md`) DID carry a term in
`N`, which is the gap this design exists to close.

<a id="the-split-policy"></a>**The split is ONE WAVE of the declared residency.** Because
the prologue is per-CTA, a grid deeper than the machine can hold at once re-pays that
whole prologue in a second wave for work the first wave could have finished, and the
`beyond-realized CTAs exit early` argument does not cover it: a CTA cannot exit before it
knows its own range, and knowing its range means running the prologue. So
`rola.ops.decode.derive_n_split` takes `SMs * residency / BH`, with the residency read
back off the launch bound itself
([`decode_api.md` §2](decode_api.md#residency)), bounded below by the UNIT SUPPLY so a
small topology does not launch CTAs there is no work for. Both bounds are static facts —
an occupancy fact and a geometry fact. The remaining slack is not: the wall-optimal split
sits somewhat under one wave and its position depends on the REALIZED work, which is a
device fact and reading it back is the one thing a decode step must not do.

**`n_split` IS ONE NUMBER AND BOTH PASSES TAKE IT.** The unit supply it is bounded by is
the STRUCTURAL ceiling `N / s_{D-1}` — the lattice's whole unit count, which neither side
can exceed — so each pass's own realized unit count is at most that ceiling and the ONE
grid over-provisions for both. A CTA past its pass's realized `units` runs an empty walk
loop and exits. There is no per-pass sizing and no density threshold, and there could not
be: a side's realized unit count is a fact only the device holds, and it is exactly the
fact `derive_n_split` may not read.

## <a id="row-scales"></a>22. `sr` and `sw` — the two per-token row scales

At `T = 1` there is no owner rectangle, so the per-LEVEL factors the tiled kernel
hoists into `sr`/`sw` for the fixed levels `l < j` stay in the per-LEAF product
instead. The split between `R[0,s]` and `sr` is an owner-rectangle artefact, and
the product is the same product either way. What remains here is exactly the
per-TOKEN part: the side gains. The read gain is fixed at `1` (`global` fixes
`g_read := 1`; raw is dead, so there is no arm where it is anything else).

## <a id="per-side-enumeration"></a>25. The two side products are enumerated SEPARATELY, and the launch boundary IS the ordering

A step has two obligations over two different leaf sets: it deposits into
`supp(W) = prod_l supp_l(W)` and it reads `supp(R) = prod_l supp_l(R)`
([§2](#kronecker-support)). One walk serving both must enumerate the UNION product
`prod_l (R_l u W_l)`, and **THE UNION PRODUCT DOES NOT SHRINK.** Under the ALTERNATING
declaration ([`carry/box.md`](../carry/box.md)) every level is DENSE FOR ONE SIDE, so
`R_l u W_l` is the whole of level `l` at every level, and their product is the DENSE
product — the entire lattice — however sparse each side is on its own. A routing that
touches `k_tok` of `B` digits per side per level still enumerates every unit there is.

So the two products are enumerated separately, by TWO KERNELS launched back to back ON ONE
STREAM: `decode_deposit_kernel<DV, D, DECAY>` walks `prod_l W_l`,
`decode_readout_kernel<DV, D>` walks `prod_l R_l`. Each folds its own side, builds its own
digit mask and its own owner list ([§21b](#per-cta-prologue),
[`decode_lattice.md`](decode_lattice.md)), and resolves units against that one side alone.

**THE ORDERING IS THE LAUNCH BOUNDARY, AND THAT IS THE WHOLE CORRECTNESS ARGUMENT.**
Stated as the law it is:

- **The deposited leaf set is unchanged.** It is `prod_l supp_l(W)` either way — the
  deposit pass enumerates it directly instead of filtering it out of a union.
- **The read leaf set is unchanged.** It is `prod_l supp_l(R)`, likewise.
- **UPDATE-THEN-READ IS PRESERVED BY ORDERING, NOT BY ONE VISIT PER LEAF.** A leaf in
  `supp(R) n supp(W)` is deposited by the deposit pass, and the readout pass — running
  strictly after that kernel, on the same stream — reads the row AFTER the deposit landed
  in it. A leaf in `supp(R) \ supp(W)` has `keep == 1` and no deposit at all, and is read
  exactly as it stands. Both are [§27](#update-then-read)'s two cases, reached by stream
  order rather than by program order inside one thread.
- **THE STATE STAYS ATOMIC-FREE AND BYTE-IDENTICAL ACROSS `n_split`.** The WRITE unit rank
  space is a PARTITION over write owners ([§26](#the-unit-partition)), so every deposited
  row has exactly one owning CTA, warp and lane, and the deposit is that one lane's
  read-modify-write. Nothing about the split can reorder it.
- **`y` IS BYTE-REPRODUCIBLE AT FIXED `n_split`** — the 8-way fold order and the slot order
  are what that rests on ([§30](#slot-order)) and neither moves. `y` is NOT `torch.equal`
  across different `n_split`, and never was: fp32 addition is not associative across
  partitions. The ACCUMULATION ORDER OVER LEAVES is a property of the rank space being
  walked, so `y` matches a union-walking binary to the ORACLE TOLERANCE rather than
  bitwise.
- **AN ABSENT ATOM stays exact.** The deposit pass skips it and the readout pass skips it,
  and `acc + ar * 0 == acc` in fp32, so skipping is arithmetically identical to reading
  the zeros a dense backing holds there ([§4](#absent-atom)).

### What it costs, measured

The enumeration does what it is shaped to do. Units enumerated per batch-head at the
flagship, union against `W + R`:

| cell | union | deposit `W` | readout `R` | `W + R` vs union |
|---|---|---|---|---|
| dense | 4096 | 4096 | 4096 | 8192 — 0.50x |
| ALT k16 scattered | 4096 | 2048 | 4096 | 6144 — 0.67x |
| ALT k4 scattered | 4096 | 512 | 1024 | 1536 — 2.67x fewer |
| ALT k16 clustered | 4096 | 512 | 512 | 1024 — 4.00x fewer |
| ALT k4 clustered | 4096 | 128 | 256 | 384 — 10.67x fewer |
| ALT k16 aligned | 4096 | 256 | 256 | 512 — 8.00x fewer |

And the wall, RTX 3080 Ti, `BH = 8`, paged, captured graph, median µs, against the
union-walking binary on a host reproducing the health check (402/66/36/57/30 µs):

| cell | union | per-side | |
|---|---|---|---|
| dense | 402.4 | 577.8 | +43.6% |
| ALT k16 scattered | 66.4 | 76.4 | +15.0% |
| ALT k4 scattered | 36.0 | 34.7 | −3.6% |
| ALT k16 clustered | 57.3 | 58.8 | +2.4% |
| ALT k4 clustered | 30.0 | 27.4 | −8.5% |
| ALT k16 aligned | 57.3 | 54.1 | −5.6% |
| ALT k4 aligned | 29.6 | 27.0 | −8.6% |

**TWO MECHANISMS PAY FOR THE ENUMERATION, both measured by `ncu`:**

1. **THE DOUBLE STATE READ OVER `supp(R) n supp(W)`.** A leaf both sides reach is loaded by
   the deposit pass AND by the readout pass, where one walk's load served both. On the
   dense cell `supp(R) = supp(W) =` everything, so `dram__bytes_read` is two passes of the
   plane instead of one: 136.7 MB deposit plus 141.9 MB readout. Where the two supports
   barely overlap — the split-level ALT cells, where `supp(R) n supp(W) = W_0 x R_1` is
   `k_tok^2` leaves against `k_tok * B` on each side — there is no traffic penalty at all.
2. **THE READ PRODUCT DOES NOT SHRINK AT A SCATTERED SPARSE LEVEL.** `ALT k16 scattered`
   puts 16 live digits on level 1's read side but SCATTERS them over all 16 owner spans, so
   `|R_1|` is 16 of 16 and `|R_0|` is 32 of 32 (the dense side): the read product IS the
   full 4096, and adding the write product on top makes the enumeration 1.5x LARGER than
   the union's. Sparsity that does not concentrate into owner spans buys the owner-list
   pruning nothing, which is a property of the LATTICE and not of the split.

**THE VERDICT, stated where it must be stated:** the enumeration change does what it was
designed to do — units fall 2.7x to 10.7x on four cells — but the step is PROLOGUE-BOUND
rather than unit-scan-bound at the production split, so the saving is only 4–9% there, and
it is paid for by (1) and (2) on the other cells.

Registers, both architectures, ZERO SPILL at every arm: the deposit pass is 48 (`DECAY =
false`) / 56 (`DECAY = true`) at `D <= 3` and 64 at `D = 4` with decay; the readout pass is
48 at every `D` except 57 at `D = 4`. The decay arm's headroom against its declared cap is
8 registers at the flagship `D = 2` ([§30b](#declared-residency)) rather than none, because
the pass carrying the dials no longer carries the readout's accumulators too.

## <a id="aligned-subranges"></a>26. The unit's leaves are walked in ascending order, over the innermost run's set bits

<a id="the-unit-partition"></a>Each pass's grid partitions ITS OWN side's UNIT INDEX SPACE
directly — `for
(unit = seg*kDecodeWarps + warp; unit < units; unit += n_split*kDecodeWarps)` — so there
is no separate row-range-to-word resolution step: a unit IS the warp's own share, by
construction, and `resolve_unit`
([`decode_lattice.md#the-unit`](decode_lattice.md#the-unit)) is what turns a unit index
into its `(owner, r_0 .. r_{D-2})` pair before any leaf inside it is touched. In the
DEPOSIT pass that rank space is the write product, so every DEPOSITED leaf has exactly one
owning CTA, warp and lane — the write unit partition is a
PARTITION, so no atomic ever touches `state` ([§16](#determinism)). The readout pass reads
only, so its partition is a work split and nothing more.

**A WARP TAKES A WHOLE UNIT, NOT A WHOLE OWNER, AND THAT IS A MEASURED CHOICE.** A unit is
already the DRAM granule: under the mixed-radix order its leaves are the innermost level's
contiguous run, which at the flagship spans is one 16-leaf ATOM, so owner-contiguity
across units buys nothing on top of it — the sectors are whole either way. Partitioning at
OWNER grain instead caps the walk's parallelism at the LIVE OWNER COUNT and hands each
warp the entire `BC`-leaf box to walk serially; measured on the wide sparse arm that cost
roughly a factor of two, and at `N = 256` (two live owners against eight warps) rather
more. The CTA's warps take CONSECUTIVE units, so a CTA still covers one contiguous stretch
of the plane per pass.

Within one resolved unit the walk visits the SET BITS of THAT SIDE'S slice `c`,
ascending, by `__ffs` — ascending bit index is ascending LEAF order, because the unit's
leaves ARE the innermost level's run in order. A dead innermost digit costs one `__ffs`
step and no address at all, and the side's membership comes off the already-resolved
`c` bitfield with no division anywhere. That ascending order is what makes
`cached_atom` — the paged base translation's one piece of reuse ([§4](#paged-address)) —
correct and cheap at once: at the flagship spans the whole unit is ONE atom, so the
translation is resolved once per unit.

The trip count per unit is `popcount(c) <= s_{D-1} <= kDecodeMaxSpan`
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
No second GEMM, no reconstructed self term: the causal structure is exact because the
readout sees `S_exit`.

**THE ORDER IS ENFORCED BY THE LAUNCH BOUNDARY** ([§25](#per-side-enumeration)): every
deposit of the step lands before any readout of it begins, so a row in
`supp(R) n supp(W)` is read after its own deposit however the two rank spaces are
partitioned. It is an ORDERING property, not a one-visit-per-leaf property, and it is the
grid-wide one a barrier could not have expressed.

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

Both kernels carry
`__launch_bounds__(kDecodeThreads, decode_step_blocks_per_sm(DECAY, D))`, and that second
argument is not decoration: it is a REGISTER CAP the assembler must meet, and the census
is what proves it met it without spilling. **THE READOUT PASS TAKES THE `decay = false` ARM
AT EVERY TOPOLOGY** — decay applies at the deposit and nowhere else — so the STEP'S
declared residency is the DEPOSIT pass's, which is the binding one and the one
`rola_decode_residency` returns. `decode_step_blocks_per_sm` is
`(decay || levels >= kMaxLevels) ? 4 : 5` — TWO TEMPLATE AXES move the declared residency,
each by exactly one CTA:

- **DECAY** carries a second per-leaf product (the decay rate) and a survival
  exponential through the same loop as the undecayed arm's amplitude product;
- **DEEPEST** (`D == kMaxLevels == 4`) resolves one more level per unit — one more owner
  coordinate, one more fixed digit and one more amplitude factor folded into `Unit<D>`
  ([`decode_lattice.md#the-unit`](decode_lattice.md#the-unit)) — held live across the
  same loop.

The measured census is what settles both numbers, and it did move once in wave 2c: an
earlier draft of the unit carried a RUNTIME loop level between the unit and its slots, and
spilled 16–30 stores at `D = 2` and `D = 3` against the 48-register cap the `5` implies.
Collapsing the unit to the innermost run — one shape, uniformly
([`decode_lattice.md#the-unit`](decode_lattice.md#the-unit)) — returned every arm to zero
spill at the SAME declared residency, which is the outcome the *"ideal regime is not
negotiable"* rule asks for: the budget was not the thing to relax.

Both are compile-time functions of a template axis rather than a threshold read off a
measurement, which is what the standing *"branch on structure, not thresholds"* rule asks
for: a config that is DECAY or `D == 4` gets one fewer CTA/SM, unconditionally, and every
other config gets the wider budget.

**The claim's SIZE comes out of the fold that applies it** — an atom the shadow row maps
and the table does not IS a claimed atom ([§4e](#the-claim)) — rather than being carried
across the walk or re-derived from the write-support enumeration afterwards. Both
alternatives hold state alive across the loop the register budget is tight in; counting it
where the fold already touches every atom holds nothing at all.

## <a id="instantiation-matrix"></a>31. The instantiation matrix

`DV(2) x D(4) x DECAY(2) = 16` deposit kernels plus `DV(2) x D(4) = 8` readout kernels =
**24 device entries** ([§25](#per-side-enumeration): the readout pass has no decay arm to
instantiate). Note what is **not** an axis: no `BT` (one token), no `BC`
(no owner-blocked amortization — [§3](#bc-absent)), no `MmaPrec`/precision split (there is
no MMA, and the tf32x3/bf16 split died with the `PREC` axis everywhere else —
`DELETIONS.md`), no `GLOBAL` (raw normalization is dead; the mass column is unconditional
— [§20](#template-axes)), no OPERAND STORAGE TYPE (the fold reads three independent dtype
families as runtime flags selecting a typed loop instead — a template axis there would
multiply the matrix by eight for a few hundred bytes of gather,
[`decode_fold.md#the-dtype-arms`](decode_fold.md#the-dtype-arms)), and — the axis this rung
removed — no ADDRESS-SPACE arm: the factor tables are shared memory unconditionally, at
every shipped topology, so there is no second arm for a template parameter to select
between. `d_v` is restricted to `{32, 64}` by a `TORCH_CHECK`, and `D` outside `[1, 4]` is
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

Every element is written by the `n_split == 1` CTA or by the last-arriving CTA of
the combine, so a zero-fill would be a launch that the launch-latency-bound
small-batch regime pays for nothing — and an unwritten element showing as garbage
rather than as a plausible zero is what a gate wants.

## <a id="smem-refusal"></a>35. The staged-amplitude SMEM budget is a declared capacity

The host entry queries `cudaDevAttrMaxSharedMemoryPerBlock` and `TORCH_CHECK`s the LARGER
of `decode_deposit_smem_bytes(...)` and `decode_readout_smem_bytes(...)` against it, naming
`sum_l width_l` in the message. The max is the right question because the two launches are
two independent occupancy questions and the step is admissible exactly when the worse of
them fits. Between them the two totals name everything either prologue puts in shared
memory — the staged amplitude rows, the dial row, the token value, the cross-warp fold
buffer and each pass's ONE mask and ONE owner list (`mask_total + olist_total` words per
pass, [`decode_lattice.md`](decode_lattice.md)) — and nothing else exists to size. It is **refused loudly
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

The deposit pass publishes `growth_any` at the end of its own launch, so the eager path's
read is behind it BY DATA DEPENDENCE and there is nothing left to enqueue it in front of:
the step's own two launches are the whole stream either way. It is one `[1]` int32 into a workspace-owned PINNED buffer,
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
object `rola.engine.facts.planes.atom_bits` derives from the facts pass and the same one
the step condenses, reached by neither's route.
