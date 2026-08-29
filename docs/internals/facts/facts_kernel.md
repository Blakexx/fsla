# `csrc/rola/src/facts/facts_kernel.cuh` — the facts kernels' device world

The union-table pass (`chunk_facts_tables`), the atom-grain stats reduction
(`atom_bitmap_kernel`), the block-grain liveness epilogue (`block_bitmap_kernel`),
and `LevelPlan` — the geometry law they and the (deleted) chunk consumer share.
Extracted from the consumer's header in the K31 pre-build deletion batch
(baseline = tag `baseline/pre-k31`); the sections below moved here verbatim from
`docs/internals/chunk/chunk_kernel.md`, which mirrored the file they used to
live in. The namespace stays `rola::chunk` so the ratification manifest's
mangled keys do not move with the file.

## The geometry law

<a id="geometry-law"></a>**The geometry law.** `SpanPlan` quantifies it over `D`
levels of one uniform power-of-two width `B` (the compile matrix is 8 widths × 4 depths
= 32 topologies, and `N = B^D` is the reachable grid). It is stated on the quantity it
is about: THE PER-LEVEL SPANS `s_l`, GIVEN. Each span is an aligned run that divides
its level; `BC = prod_l s_l` is a consequence rather than an input, which makes "`BC`
is a rectangle in digit space" true by construction; and the levels that carry table
rows (`s_l < B`) must be a PREFIX of the levels, because the two bitmap reductions
address the innermost clause as `prefix_entries(clauses - 1)` and carve their grid on
`suffix_groups`. A `(widths, spans)` tuple that breaks a clause does not compile.

<a id="lex-policy"></a>**`LevelPlan<D, B, BC>` is the LEX POLICY, an alias.** It is one
way of CHOOSING spans from a `BC`: reading levels most-significant first, every level
above the boundary is spanned at exactly one digit, at most one level is spanned
partially, and every level below is spanned entirely — so `BC = B^whole * partial_span`
and at most one span is neither `1` nor `B`. That shape is what K31-i-d ran into: the
box lattice's owner runs are `D` spans (the flagship's `(8, 16)`), which the
policy forbids arithmetically and not accidentally. The plan now takes the spans, the
policy computes them, and every consumer reads `span_at` / `entries_at` / `col_base`
exactly as it did.

The WIDTHS POLICY is an accessor, not a constant: every consumer reads level widths
through `width(l)`, which the uniform policy answers with `B`. A jagged policy answers
from a table later, and the loop nests are already written against the accessor.

<a id="template-level"></a>**A level is a TEMPLATE parameter, never an argument.**
`off_at`'s divisor is a recursive `constexpr` call, and a level passed as an argument
leaves the recursion for the compiler to unroll at run time — measured: a loop and a
reciprocal division reached the SASS. `base_term` and the clause resolve both state
their divisors as `constexpr` locals for that reason.


<a id="leaf-order-law"></a>**The leaf-order law.** A block's `BC` leaves are a contiguous, `BC`-aligned run
in canonical leaf order, so the block is named by its first leaf and an atom
index is that leaf over sixteen — `rola.ops.paging.AtomKeying.atom_of`. Levels
below the partial one are spanned whole and least significant, the partial run is
aligned and divides its level, and levels above contribute one digit each; the
per-level statement of that is `SpanPlan::base_aligned`, `static_assert`ed by the two
reductions that walk canonical blocks (`atom_bitmap_kernel`, `block_bitmap_kernel`) —
it is a claim about CANONICAL leaf order, so it belongs to those consumers and not to
every plan; the box lattice's owners are aligned runs in LATTICE order and are not
canonical rectangles. `SpanPlan::block_base` is the map. The law is what the
final-state gate checks end to end: a misordered base shows up there as a
permutation, not as noise.

**Fragment ownership.** For `mma.m16n8k16`, accumulator element `k` sits at row
`(lane>>2) + 8*(k>>1)` and column `(lane&3)*2 + (k&1)`. The fold's A operand is
`V^T` and its B operand is the write panel, so `st[m][j][k]` is channel
`m*16 + (lane>>2) + 8*(k>>1)` at the warp's PRIVATE leaf `j*8 + (lane&3)*2 + (k&1)`
(`docs/internals/carry/carry_kernel.md#the-carve-one-digit-group-of-the-innermost-level-per-warp`);
the mass tiles carry the channel only in row 0, i.e. lanes 0..3 and `k in {0,1}`,
which is the restriction the denominator store already makes. A page therefore
holds `kAtomLeaves/s_{D-1}` outer runs times every digit of the carve level, and
each warp's share of it is `s_{D-1}/kWarps` of those digits — a compile-time set of
`(tile, column)` pairs. The atom loop and the loops inside it are fully unrolled
because a runtime index into a register array is a stack frame.

**Staging.** A page is 16 rows of 65 floats and it moves through the output
staging buffer, whose rows are `d_v + kPadF = 68` floats and which is dead at
both ends of the kernel — so no shared memory, staging depth or CTA target
changes. The store runs after the final drain has consumed that buffer, behind
one extra rendezvous; both groups iterate the same atom count against their own
copy, so every barrier stays CTA-uniform.


<a id="outer-radix"></a>**The composite outer index is a radix, and the plan owns
it.** A block's slot is an innermost digit plus a composite outer index over levels
`0..D-2`, whose place values are the outer spans with level `D-2` fastest.
`LevelPlan::outer_digit<Lv>` is that decomposition; the level is a template parameter
for the same reason `base_term`'s is, because a runtime divisor does not fold. Two
residues in it are provably redundant and are not emitted: a single-digit level has
one digit, and the MOST significant outer level cannot wrap. The reverse pass reads
it — its producer columns and its leave-one-out staging both
([leave-one-out](backward.md#leave-one-out)). The FORWARD's producer chain still
spells the same decomposition out inline: dropping those two residues moves twenty
`compacted_kernel` bodies, several of them upward in registers, so repointing it is a
measured change of its own and not a tidy-up.


## The facts pass, one launch over the amplitude planes

<a id="the-facts-pass"></a>**`chunk_facts_tables<PLAN, MASK>` emits BOTH ballot
row sets from one pass.** It is templated on the PLAN TYPE, so the pass is
parameterized by the SPANS it votes and not by a `BC` some policy derived them from;
its mangled name — the ratification manifest's key — carries `(D, B, s_0..s_3)`
accordingly, and `tools/ratify.py:parse_shape` is the one decoder. A lockstep unit takes thirty-two consecutive tokens, one per
lane, and votes once per entry; the vote IS the word, so nothing is packed afterwards
and nothing is zeroed first. A token past the end reads token zero and votes dead —
its bit has no consumer, and reading in range is what keeps the unit converged.

The two row sets are the same expression at two grains. The OWNER rows, both sides,
come out of `PLAN` and are what the consumer's clause pointers address; `PLAN` is
`LevelPlan<D, B, BC>` for the lex arms and the lattice's `SpanPlan<D, B, s_0..s_3>` for
the `rola::carry` family's owner rows (`launch_span_facts_arm`).
The ATOM rows, write side only, come out of `LevelPlan<D, B, kAtomLeaves>` and are what
the atom bitmap reduces into residency (`docs/internals/paging/stats.md`). A lane's
amplitude row is a fact about the token and not about any block geometry — the packed
column layout is the widths policy's — so the two grains read the SAME bytes, and
reading them once instead of twice is the whole of this fusion. The plane is
`BH · L · sumW · 2` bytes and it grows with `T`; that is the axis the saving is stated
on.

**`MASK` is the CALL CLASS, never a bool per fact.** It carries two of the four masks
(`rola/engine/types.py`) — the stateless prefill class compiles no atom-grain emission at
all, so a stateless call pays exactly what a call with no residency to plan should pay.
Within the stateful class, whether there is an arena to admit into is a runtime-null
OUTPUT skip: a null atom table, one predicate outside the loops, no second arm. The
host selects the class with `int_switch` over the instantiated set
(`docs/internals/dispatch_switch.md`), so an unmatched value is a refusal.

**The published atom bitmap stays BC-free.** The transient atom TABLE acquires this
pass' `BC` in its instantiation; the bitmap that reduces it does not — its launch is
keyed `(D, B)` alone, and only the input stride it reads is this template's. That is
what keeps residency an OUTPUT of the plan rather than a consequence of it, and it is
gated by `tests/oracle/test_atom_bitmap.py::test_the_bitmap_is_bc_free`, which requires
the same bitmap at every built `BC` of a topology.


## The liveness epilogue, at the launch's block grain

<a id="block-bitmap"></a>**`block_bitmap_kernel` answers, per `(bh, block)`, the three
questions a launch needs before it can skip a block.** It is `atom_bitmap_kernel`'s
reduction re-instantiated at `BC` instead of `kAtomLeaves`, and everything structural
transfers: the prefix carve (blocks sharing an upper-digit prefix share every clause row
but the innermost, so one CTA serves a prefix and the shared rows are ANDed into a word
stream once), the per-row ballot over the token axis, and the CTA-uniform all-live early
exit that makes the DENSE case the cheap one. Two things are added. The reduction runs
BOTH SIDES against one token tile — the read rows sit at `2·bh + 0` and the write rows at
`2·bh + 1` of the same union table, so the tile is read once and the two upper streams are
formed together. And the page table, when there is one, is OR-condensed over the block's
`BC / 16` atoms into a third bit.

It consumes an artifact that ALREADY EXISTS. The `BC`-grain union table is built for the
launch anyway, with exactly the rows the consumer's clause pointers address, so this is a
pure reduction and not a second producer pass. It is nonetheless a SEPARATE LAUNCH, on the
union table's own rule: a per-call cost is reported on its own line and never folded
into a kernel delta. It runs on EVERY call.

**Why three bits and not one predicate.** `READ` and `WRITE` are exact at the block's own
grain — a block is an aligned digit-range rectangle, so any combination of digits live at
one token, one per level, inside the rectangle IS a leaf of the rectangle, which is what
makes the AND-over-levels-then-OR-over-tokens evaluation exact rather than a bound. What a
skipped block would have COST, however, is not a property of the routing: it differs by
call class (stateless, state-in, state-out) and by backing (dense, paged). So the facts
are emitted separately and the predicate over them is the host's
(`docs/internals/engine/rules.md#run-table`).

`STATE` is the residency witness for "this block may carry a nonzero entry state": an atom
with no page has provably zero entry fragments, so an unmapped block cannot read anything
out. The dense backing carries no residency map, and the bit is then ZERO everywhere
rather than guessed — the host's dense predicate is weakened to compensate, which is a
recorded weakening rather than a hidden one.


## Comments moved from source (S5b)

Verbatim `//:` prose blocks from `csrc/rola/src/facts/facts_kernel.cuh`, in source order, one heading per declaration. Source keeps only a short decl block pointing here.

<a id="katomleaves"></a>
### `kAtomLeaves`

THE INVARIANT QUANTUM: the leaves one page granule holds, and the rectangle
the stats pass keys residency to.  It is the MMA k-extent the state's own
tiling was derived at (the host mirror is `rola.ops.paging.MMA_K_QUANTUM`,
and decode's `kAtomLeaves` states the same 16 independently;
docs/internals/paging/stats.md).

<a id="note-l30"></a>
### near line 30

THE GEOMETRY LAW over D levels of one uniform power-of-two width B, stated on
the quantity it is actually about: THE PER-LEVEL SPANS `s_l`.  Every clause
below is the law's, and a (widths, spans) tuple breaking one does not compile.
Widths are read through `width(l)` so a jagged policy can answer from a table
without touching a loop nest -- docs/internals/facts/facts_kernel.md#geometry-law

THE SPANS ARE GIVEN, NOT DERIVED (K31-i-d).  The lex policy below -- `singles`
levels at one digit, at most one partial run, whole levels beneath -- is ONE way
of choosing them, and it admits at most one span that is neither `1` nor `B`.
The box lattice's owner runs are `D` spans (the flagship's `(8, 16)`),
which that shape forbids arithmetically rather than accidentally.  So the plan
TAKES the spans and the lex policy becomes `LevelPlan`, an alias that computes
them: one plan type, two ways of arriving at it, and every consumer below reads
`span_at` / `entries_at` / `col_base` exactly as it did.

<a id="static-assert"></a>
### `static_assert`

THE WIDTH FLOOR IS THE ATOM'S, not a routing preference: the atom plan
(`LevelPlan<D, B, kAtomLeaves>`, below) exists only where the topology holds a
whole page granule.  At `D = 4` that admits `B = 4`, which is the width the box
`[4, 4, 4, 4]` -- the multi-level carve's own topology -- is stated at.

<a id="clauses-prefix"></a>
### `clauses_prefix`

THE CLAUSE LEVELS ARE A PREFIX -- a property of the TWO BITMAP REDUCTIONS and
not of the table, so it is asserted where it is used (`atom_bitmap_kernel`,
`block_bitmap_kernel`): they address the innermost clause row as
`prefix_entries(clauses - 1)` and carve their grid on `suffix_groups`.  It holds
by construction under the lex policy, which is the only policy those two are
instantiated at; a GIVEN-span plan states an arbitrary per-side partition (K35
S1b: the write carve `[1, 4, 2, 4]` tests levels 0 and 2 and spans 1 and 3 whole)
and neither reduction is built for it.

<a id="base-term"></a>
### `base_term`

A block's leaves are a CONTIGUOUS, BC-ALIGNED run in canonical leaf order, so
the block is named by its first leaf -- docs/internals/facts/facts_kernel.md#leaf-order-law
THE LEVEL IS A TEMPLATE PARAMETER, not an argument: `off_at`'s recursive
constexpr divisor does not fold otherwise -- docs/internals/facts/facts_kernel.md#template-level

<a id="levelplan"></a>
### `LevelPlan`

THE LEX PLAN, an ALIAS.  `LevelPlan<D, B, BC>` is the span plan whose spans the
lex policy derives from `BC`, so every existing consumer keeps its spelling and
every existing instantiation keeps its geometry -- what moved is that the spans
are now the plan's input rather than its private derivation.

<a id="ktokbits"></a>
### `kTokBits`

(G4) The D = 2 compatibility view -- BlockPlan -- is RETIRED.  Its geometry
law and its union-table entry accounting live in LevelPlan above, where
they are stated once for every depth; nothing below this line speaks
lead/tail.

<a id="ktokbits-2"></a>
### `kTokBits`

THE UNION TABLE'S ONE STRUCTURAL CONSTANT.  Tokens are the table's minor axis
and they are packed thirty-two to a word, which is the lockstep unit's width,
so the cook reads one word per unit per side and needs no re-packing step.  A
sequence that does not fill its last word leaves the tail bits ZERO.

<a id="chunk-facts-tables"></a>
### `chunk_facts_tables`

THE TABLES ARE BUILT ONCE FOR THE WHOLE GRID, BOTH ROW SETS IN ONE PASS.  A
lockstep unit takes thirty-two consecutive tokens, one per lane, and votes once
per entry; the vote IS the word, so nothing is packed afterwards and nothing is
zeroed first.

THE TWO SIDES CARRY THEIR OWN PARTITION (K35 S1b addendum).  A bitlist is per side,
per level, per run of THAT SIDE's partition, voted from that side's own factors, so
the two plans are independent and the table is the read plan's rows followed by the
write plan's -- `[BH, entries_read + entries_write, tok_words]`.  Holding ONE
partition for both (the finer, with the coarser side ORing) achieved nothing and
cost every side a table refined by the other's carve.  The two plans coincide for
every caller that wants one grain, and the layout is then the old `[BH, 2, E, .]`.

The two grains are one partition refining another -- the atom rectangle is
`kAtomLeaves` leaves and the owner rectangle is `BC` of them -- so the write
side is voted at the ATOM grain and each owner row is the OR of the atom rows
inside it.  The digits are read once and voted once; nothing is evaluated twice.
Its cost does not depend on `BC`: a lane reads its token's amplitude row once
per level that has entries, whatever the spans partition it into.
docs/internals/facts/facts_kernel.md#the-facts-pass

<a id="note-l339"></a>
### near line 339

THE ATOM ROWS ARE THE WRITE SIDE'S: which atoms this routing writes is what
keys residency, and reads of never-written atoms are inert.  A stateless call
compiles none of this, so it pays a stateless call's price; a stateful call
with no arena to admit into is the runtime-null OUTPUT skip, and takes the
owner-grain emission alone -- which is the cheaper one when nothing needs the
finer rows.

<a id="katomthreads"></a>
### `kAtomThreads`

THE STATS PASS' REDUCTION.  Which atoms does this call's routing WRITE?  The
answer keys the state's residency, so it is taken at the invariant quantum
(`kAtomLeaves`) and never at `BC` -- the derivation, the exactness argument and
the rejected coarser reductions are in docs/internals/paging/stats.md.

how many threads and how much of the token axis one CTA of the reduction
holds resident.  The tile is a RESIDENCY quantity, not an axis: it bounds the
shared word stream so the CTA's footprint is independent of `L`.

<a id="atom-bitmap-kernel"></a>
### `atom_bitmap_kernel`

THE REDUCTION, one CTA per (bh, upper prefix).  A prefix's atoms share every
clause row but the innermost, so the shared rows are ANDed into a word stream
ONCE and each innermost row is then ANDed against that stream and OR-reduced
over the token axis.  An atom is live as soon as one word survives, so the CTA
leaves the moment all of its atoms are live -- which makes the DENSE case the
cheap one.

<a id="pin-address"></a>
### `pin_address`

ADDRESSING DISCIPLINE, THE FORM'S: ONE 64-bit address per (bh), formed
here and never re-derived; every row and every word after it is a 32-bit
BYTE displacement that zero-extends, so no `pointer + int` widening
multiply reaches a loop body.  The displacement's bound is one `bh`'s
slice of the table, `entries * tok_words(L) * 4` bytes.

<a id="block-bitmap-kernel"></a>
### `block_bitmap_kernel`

THE REDUCTION'S ROW GROUP: at most one innermost row per warp per side.  The
prefix carve amortizes the UPPER-clause stream, so a block with no upper
clause has nothing to amortize and one CTA per prefix would be one CTA for
the whole reduction; re-reading the upper stream per group buys the grid
back (docs/internals/facts/facts_kernel.md#block-bitmap).

<a id="block-bitmap-kernel-2"></a>
### `block_bitmap_kernel`

THE LIVENESS EPILOGUE, at the LAUNCH's block grain: `atom_bitmap_kernel`'s
reduction re-instantiated at `BC`, over BOTH sides, with the page table's
residency condensed beside them.  The three facts are emitted SEPARATELY --
one byte per (bh, block), bits {READ, WRITE, STATE} -- because what a skipped
block would have contributed differs by call class and by backing, so the
predicate over them is the host's (docs/internals/facts/facts_kernel.md#block-bitmap).
