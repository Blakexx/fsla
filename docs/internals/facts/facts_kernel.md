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
box lattice's owner runs are `D` BALANCED spans (the flagship's `(16, 8)`), which the
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
`V^T` and its B operand is the write panel, so `st[j][k]` is channel
`gwarp*16 + (lane>>2) + 8*(k>>1)` at leaf `j*8 + (lane&3)*2 + (k&1)`; the mass
tiles carry the channel only in row 0, i.e. lanes 0..3 and `k in {0,1}`, which is
the restriction the denominator store already makes. One atom is therefore the
`st` tile pair `(2i, 2i+1)` across all four units plus ONE unit's mass pair, at
`gwarp = i % kMassUnits` and `q = i / kMassUnits`. The atom loop is fully
unrolled because a runtime index into a register array is a stack frame.

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

