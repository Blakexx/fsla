# `csrc/rola/src/chunk/chunk_kernel.cuh` — the chunk consumer's device world

The ladder rung-6G kernel, adopted whole: `LevelPlan<D,B,BC>` (the
single geometry authority — spans, entries, clause count, the mixed-radix
block→entry map, all-int), `Geom` (SMEM layout and the derived staging depth),
the compacted kernel (form/consume schedule, per-side compaction, K-windows as
folded displacements, level-generic produce with the decoupled write gain) and
the union-table kernel.

Where its reasons live: the ladder repo's rung ladder (5A→6G) is the
derivation record — seven refuted alternative designs, the placement/mutual-
cover/wall-coupling laws, and the wall prices — and the development journal
carries the rulings. This tree's own gates are the operative discipline:
`tools/chunk_sass_gate.py` (zero spill, zero banned body opcodes over the
shipped fatbin) and the conformance battery against `naive_rola`.

Capability layer: `arch_caps.cuh`/`design.cuh`/`ops.cuh` ride with the family
(same directory); dedup against this tree's `mma.cuh`/`cp_async.cuh` lineage
is deliberately deferred to the e2e cleanup step, so the adoption stayed a
verbatim behavioral port (ladder parity gate: tables bit-equal, one-block
arms exact).

<a id="state-axis"></a>
## The carried state: the store epilogue and the entry load

The kernel's state is register-resident for the whole launch, so a call has two
ends to give it: `s_out`, the plane the register fragments are stored into, and
`s_in`, the plane they are seeded from.

**It is a COMPILE-TIME axis, `compacted_kernel`'s trailing `STATE`, and the
reason is measured.** The state I/O first shipped as a pair of runtime null
tests, on the argument that a CTA-uniform predicate at the ends of the kernel is
free. It was not: both blocks then needed a plane base formed at KERNEL SCOPE
(the addressing note below says why that was the only place a widening multiply
could go), so a runtime test still paid for that addressing, and for the register
census that came with it, on the path that does no state I/O at all. The bill was
**+1.2% wall on two of five measured cells** with both pointers null. That is the
same placement law the decay build hit, and it got the same answer: the variant
that carries no state compiles none of it. `chunk.cu` picks the variant on
`(s_in || s_out)`, on the host, where a branch costs nothing; the OFF variant
assembles to the same bodies — arm for arm and architecture for architecture —
as the kernel that predates the state entirely.

The slot resolve that the paged backing brought later removed the kernel-scope
base entirely (the addressing note), so the axis no longer rests on that
placement. What it guarantees is now structural: an OFF variant that contains no
state code cannot be taxed by the feature, and the body diff says so arm for arm
instead of a wall measurement having to.

The state pointers and the page table are the LAST kernel parameters, after `L`,
and that is load-bearing rather than cosmetic: a parameter inserted mid-list
moves every later parameter's constant-bank displacement and with it every
instruction that reads one, which would put byte identity out of reach however
the code was guarded. Unused trailing parameters emit nothing.

`ChunkParams` carries `s_in`/`s_out`/`page_tbl` for both variants — it is one
shared ABI — and `launch_chunk_arm` REFUSES any of them addressed to the OFF
variant rather than dropping them silently. `Geom` and `reg_floor` stay
state-free: the ring is the launch's shared-memory ABI, and the register floor is
one declaration for both kernels. It briefly was not — the ON variant declared 242
at `C = 32, BC >= 128` against the shared 236 — and that row existed only because
the plane bases were then pinned at kernel scope; with them moved inside the state
blocks the ON variants measure at or under the shared floor on every arm and both
architectures.

**The plane, in either backing.** DENSE is `[BH, N/16, 16, d_v+1]` fp32: the
atom-major view of `[BH, N, d_v+1]`, byte for byte, with the last column the mass
channel global normalization divides by. PAGED is `[slots, 16, d_v+1]` — the same
page, as many of them as the plan committed. The caller allocates either one
(`rola.ops.chunk.state_plane`, or a `PageArena`'s storage), because a carried
state is the caller's object. `d_v + 1 = 65` is 260 bytes and not sixteen-byte
aligned, so nothing here vectorizes; the accesses are plain floats, coalesced
across the group.

<a id="leaf-order-law"></a>**The leaf-order law.** A block's `BC` leaves are a contiguous, `BC`-aligned run
in canonical leaf order, so the block is named by its first leaf and an atom
index is that leaf over sixteen — `rola.ops.paging.AtomKeying.atom_of`. Levels
below the partial one are spanned whole and least significant, the partial run is
aligned and divides its level, and levels above contribute one digit each; the
per-level statement of that is `LevelPlan::base_aligned`, `static_assert`ed at
every instantiation, and `LevelPlan::block_base` is the map. The law is what the
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

<a id="state-addressing"></a>**Addressing: NO widening multiply, anywhere in the state I/O.** Each block pins
two PLAIN ORIGINS — the plane's and the table's, neither of them displaced — and
everything per atom is a 32-bit byte offset from one of them: the atom's table
entry is `atom * 4` bytes in, and its page is `slot * 4160` bytes in. That is the
same discipline the gathered rows keep (`gather_ptr`), and the caller owns the
bound: `chunk.cu` refuses a plane of 4 GiB or more, which is a little over a
million pages. It binds the DENSE backing first, and a paged plane holding only
the atoms a plan realized is precisely the answer to that.

<a id="plane-bound"></a>**NAMED FUTURE WORK: lifting the 4 GiB bound**, if a campaign cell
ever reaches it. The two backings need different answers, and that asymmetry is the whole
of the note. A DENSE plane is `bh`-major, so ONE per-`bh` 64-bit base formed at kernel
scope lifts it -- affordable now in a way it was not when the bound was set, because the
OFF variant is specialized and carries none of this code, so the cost lands only where the
capability is used. A PAGED plane has no such structure: slots are pool-global by
construction, so the answer there is ARENA SHARDING (a pool per shard, each under the
bound, the table naming its shard), which is a plan change and not a kernel one. Nothing
is built; the refusal is loud, and the number to compare against is
`BH * (N/16)` pages for the dense backing and the plan's popcount for the paged one.

The alternative — a 64-bit `plane + slot * kPageElems` per atom — was built and
measured: ptxas lays the store epilogue out at addresses INSIDE the main loop's
backward-branch span, so the gate counted ninety `IMAD.WIDE` in loop bodies, and
one more per arm for a table row base formed the same way. The earlier
dense-only code kept its one widening multiply out of that span only by forming
it at kernel scope and pinning it, which cost two live pointers across the whole
body; with the slot resolve those bases moved INSIDE the two state blocks, and
the ON variant's registers fell with them (the `C=32, BC=128, D=4` arm by 34).

Two more consequences the SASS gate had to teach: the page sweep carries a
`(row, column)` cursor rather than dividing a flat index by 65 (a magic-number
division is a widening multiply), and `LevelPlan::base_term` takes its level as a
TEMPLATE parameter, because `off_at`'s recursive constexpr divisor does not fold
when the level arrives as an argument — it reached the SASS as a loop and a
reciprocal.

<a id="entry-state"></a>**Exactness of the seeding.** With no decay in the operator the recurrence is
affine in the state and the readout is inclusive-after-write, so setting the
fragments to the entry state is precisely the oracle's `initial_cache` seeding
and no other term changes. The fragments are SET, never accumulated: they are
provably zero at that point in the prologue.

<a id="base-translation"></a>**The base translation: dense and paged are ONE kernel.** Per atom the kernel
resolves `slot = page_tbl ? page_tbl[atom] : atom` and addresses
`plane + slot * 16 * cols`. A null table is the dense plane, where a slot IS its
atom; a table is the paged one. The branch is on a POINTER uniform over the whole
grid — a fact of the plan, not of a dimension — and it costs one `int32` load per
atom, of which a group has at most eight. An identity table would remove even
that branch and cost `BH * (N/16) * 4` bytes of bandwidth that grows with `N`;
refused.

`slot < 0` means the plan committed no page for the atom, and skipping it is
EXACT rather than defensive: absence means the atom is outside the call's write
set, which (the stats pass is exact, `docs/internals/paging/stats.md`) means no
token ever deposited into it, so its fragment is identically zero — and a zero
fragment stored or skipped is the same state. It stays correct under any SUPERSET
bitmap, since a superset never omits a written atom, and
`test_chunk_paging_equivalence.py` gates exactly that. What the skip omits is the
MEMORY TRAFFIC only: both groups iterate the same atom count and every rendezvous
sits outside it, so a barrier stays CTA-uniform however the two groups' residency
differs.

## The device world's constants, and why each one is the number it is

<a id="k-window"></a>**The K-window, and what actually bounds `L`.** An index-list
entry is one 16-bit token index, WINDOW-RELATIVE: a super-chunk lives inside one
65536-token window, because the walk forces a cut at every window line and empty
stretches slide, so sixteen local bits name any of its tokens at any `L`. Below one
window no line is ever crossed, and the partition — and the output, bit for bit — is
exactly what it was when 65536 was a refusal.

The window is a DISPLACEMENT, not a table. Each plane's per-window byte stride is a
per-launch constant, so a super-chunk's window number is one uniform multiply-add into
a column term the consumer already carries, and the per-index addressing keeps its exact
shape against kernel-wide pinned bases. What bounds `L` is therefore the 32-bit gather
displacement itself: the widest plane row is the fp32 output at `kD * 4` bytes, so
`L * kD * 4` must sit inside the 4 GiB zero-extension window. That is `kMaxGatherL`.
`V`'s row is the one that is not fixed by `d_v` alone (below), so the entry bounds it
in the same window explicitly.

<a id="v-layout"></a>**`V` is read TOKEN-MAJOR, where the caller holds it.** The value
stream arrives `[B, T, H, d_v]` — a head's `d_v` columns are contiguous, and its rows
are a whole token (`H * d_v`) apart. That stride is the launch's `v_row_bytes`, and it
is the ONLY thing about `V`'s addressing that is not compile-time: the head count is
`v_row_bytes / d_v` and is not carried twice. Nothing else changes, because `V` is
gathered by the WRITE LIST — the rows a staged chunk wants are already scattered across
the plane, so a row stride wider than the row costs the gather no coalescing, and the
128-byte row remains one aligned vector pass. What it buys is the transpose the host
used to pay for the kernel's benefit alone.

<a id="the-carve"></a>**The carve is derived, not chosen.** One MMA unit owns a
private 16-row slice of the transposed state's `m` axis, which is `d_v`, so the unit
count is `d_v` over the MMA's `m` quantum. Carving the SLOT axis across units instead
would give every unit a partial output over the whole of `d_v`, and therefore a
cross-unit reduction of the output accumulator — the very thing a private accumulator
exists to avoid. A GROUP is the rung-5D CTA: four units of sixteen `d_v` rows, each
holding its slice of the state for the group's slots.

The per-config rule is: warps = the stall floor inside the register window, filled from
the CHEAP axis first. Value units go to their cap (`d_v/16`; a token split is forbidden
for state-holding warps, since it would duplicate the register-resident state), then
slot groups. At `d_v = 64` the value axis gives four warps, and a second slot group is
added exactly when the block is wide enough that one group's k-run would exceed the
measured `KD = 8` shape — `BC >= 256`, which is also where the register window narrows
to the 8-warp point.

<a id="geometry-law"></a>**The geometry law.** `LevelPlan` quantifies it over `D`
levels of one uniform power-of-two width `B` (the compile matrix is 8 widths × 4 depths
= 32 topologies, and `N = B^D` is the reachable grid). Reading levels
most-significant first: every level ABOVE the boundary is spanned at EXACTLY ONE digit;
AT MOST ONE level is spanned PARTIALLY, by an aligned run whose width divides `B`; every
level BELOW is spanned ENTIRELY; and `BC = B^whole * partial_span`. A `(widths, BC)`
pair that breaks a clause does not compile — every clause is a `static_assert` on the
plan.

The WIDTHS POLICY is an accessor, not a constant: every consumer reads level widths
through `width(l)`, which the uniform policy answers with `B`. A jagged policy answers
from a table later, and the loop nests are already written against the accessor.

<a id="template-level"></a>**A level is a TEMPLATE parameter, never an argument.**
`off_at`'s divisor is a recursive `constexpr` call, and a level passed as an argument
leaves the recursion for the compiler to unroll at run time — measured: a loop and a
reciprocal division reached the SASS. `base_term` and the clause resolve both state
their divisors as `constexpr` locals for that reason.

<a id="form-ring"></a>**The formed ring's depth is a residency quantity, not an axis.**
What depth buys is distance between the parallel former and the serial walk, and the
walk consumes `C/32` words per super-chunk against a former that produces `kThreads` of
them in one step — so any depth at all puts the producer orders of magnitude ahead.
What depth must NOT do is cost a CTA, and that is what pins it: `2 * kFormWords * 4`
bytes must leave every arm's staging depth and CTA target where rung 5B had them, and
`arms()` reports both, so it is CHECKED rather than asserted. It is a power of two
because the ring index is a mask, and a remainder by anything else would put a
reciprocal in the walk's only loop. The shipped depth is `kThreads`, so one parallel
step refills the ring whole; it is overridable AT BUILD TIME because the wait is a
ring's ordinary behaviour and a gate must be able to make it fire — at depth one the
walk waits on almost every word, and the answer must still be bit for bit rung 5B's.

<a id="reg-floor"></a>**The register floor is a DECLARATION, not a derivation.**
ptxas's allocation for this body does not follow from the operand counts, and the
number's only job is to stop the shared-memory ceiling from asking for more CTAs than
the register file holds. It is FALSIFIABLE rather than asserted: the build fails on any
nonzero `localSizeBytes`, so a floor set too low cannot ship quietly. The values are the
measured census of this body, which carries the cook's cursor and the staircase's
thresholds on top of rung 5A's set. ONE floor serves both state variants — see the
carried-state section for the row that briefly existed and why it went.

## The two streams: cook, form, walk

<a id="cook-distance"></a>**The schedule stream and the value stream run at different
depths**, and that is the whole reason the index ring is separate from the value ring.

Compaction creates a chain. Without it every address is affine and the only global round
trip in flight is the value stage. With it the chain is schedule word → cook → index →
gather issue → gather lands → mma, which is TWO round trips in series, and in a
latency-bound arm that is the wrong shape. Running the cook further ahead than the value
ring collapses it back to one: by the time a super-chunk is staged its indices have been
ready for `kAheadExtra` iterations, so only the gather is outstanding.

The schedule is NOT staged asynchronously. A super-chunk's schedule is one word per side
per thirty-two tokens — eight bytes at `C = 64` — against a value tile of `C × d_v`.
There is nothing to stream: what the cook needs is DISTANCE, not bandwidth, and distance
is free here because the cook's carried state is a cursor and four induction pointers.
It also keeps the staging FIFO single-stream, so `stage_wait` still means what it means:
one implicit group, retired in issue order, with nothing behind it that a later consumer
needs first.

`kAheadExtra` is ZERO, and that is a measurement rather than a default. Swept at 0 and 2
on the flagship arms, the two agree within two points and the sign flips across arms.
What the cook actually costs is the latency of its OWN dependent loads, which sits inside
one call immediately before the rendezvous that follows it, and no amount of distance to
its CONSUMER moves that. So the depth is the pipeline's minimum, derived: the cook must
be exactly as far ahead as the stage it feeds. The index ring then holds every
super-chunk in flight — the one being drained, the one computing, and the `AHEAD` cooked
but not yet consumed — which is also what fences it: the slot the cook overwrites belongs
to the super-chunk drained a whole iteration earlier, across the rendezvous.

<a id="clause-rows"></a>**The block's entry rows, per side.** Level 0's entry is the
aligned run of digits the block spans there, and level 1's is its sub-range; the trailing
row DISAPPEARS when the block spans that level entirely, exactly as the clause does, and
a block spanning BOTH levels entirely — the whole slot space, so there is exactly one of
them — has no clause at all. Rung 5A OR-ed the sides to get a group bit; the cook keeps
them APART, which is the whole of per-side compaction at the schedule's end.

There is one pinned row per SURVIVING clause per side, in level order — at `D = 2`
exactly the old `tR0`/`tR1` pair, at depth at most `D` rows. They are NAMED SCALARS,
never an array: the clause loop's callees carry inline asm, which blocks unrolling, and a
runtime-indexed pointer array is demoted to a STACK FRAME. The gate caught exactly that
(`STL [R1+..]`) on the two-clause arms. Unused clauses alias clause 0 and are never read.
The form is a MAP over words rather than a walk, so a thread's word is an index rather
than the previous address plus one, and `gather_ptr` is what keeps that index a
zero-extended byte displacement instead of a widening multiply.

<a id="the-form"></a>**The form is the membership AND and nothing else.** Every thread
takes whole words of the token axis, ANDs the block's surviving clauses on each side, and
publishes one word per side into the resident ring — affine, dependency-free, over the
whole CTA. The ring's head is the walk's cursor plus its depth, so the former stops of its
own accord when it catches its tail: backpressure, with no test for it anywhere but the
loop bound.

<a id="the-walk"></a>**The walk.** One lockstep unit consumes the formed bitmap a word at
a time, keeps a running count on each side, and closes the super-chunk at the FIRST TOKEN
that would push either count past its buffer.

The closing rule is TOKEN-granular, not word-granular, and the difference is not cosmetic:
a chunk whose read list is full keeps ABSORBING write-live tokens until a read-live one
arrives, and a stretch dead on both sides is swallowed however many words it spans.
Stopping at the word boundary where a count happened to fill would put those write tokens
in the next chunk and make the partition depend on the packing — an implementation detail
the sizing model does not have. So the loop runs on the CUT, and the counts appear only in
the per-lane test that produces it.

**THE CLOSING RULE HAS ONE IMPLEMENTATION.** It lives in `Ascent`, a component in
this file, and the forward drives it through four one-line lambdas. Scan A drives
the same object: its partition is its OWN (a chunk boundary is an association of a
sum, not a semantic object) but the rule that closes a chunk is the same law, and a
second copy of that law is what this arrangement exists to prevent. Its mirror for
the reverse pass is `Descent` in `backward.cuh`.

The extraction is behaviour-neutral and was accepted on that basis rather than on
byte identity. Measured across all 580 ratified instantiations, both architectures:
**148 bodies re-assembled, 20 register counts moved by at most 9, and zero changes
to spill bytes, zero to in-loop local accesses, zero to CTA targets.** The component
DEFINED but unused moves nothing, so what moved is the allocator re-landing a
rewritten kernel, not a change in what it does. The forward's wall was checked
against its committed cross-session baseline over two sessions: 0 of 44 arms over
limit, the pinned cell at 0.7905 / 0.7946 ms against a 0.8135 ms limit.

Byte identity was the criterion first proposed, and it does not transfer here: the
`Descent` precedent held the forward's bytes because that extraction never edited
the forward's translation unit. Extracting the forward's OWN walk rewrites the
forward, so the behavioural columns are the honest gate.

It CLEARS WHAT IT TOOK: on a mid-word close the remainder is written back, so re-entry
reads what is left and its prefix counts ARE the counts of what is left, which is why
there is no `inrange` and no mask on any access. It stops at the FORMED REGION's edge, not
the token axis's: running out of formed words leaves the super-chunk OPEN, the CTA forms
more and calls again, and the partition is unchanged. Its state is register-resident,
exactly as rung 5B's cursor was; shared memory carries only what another warp needs — the
cursor, so the former knows how far it may run, and the open flag, so the whole CTA takes
the same path — and it carries them by STORE ONLY. The slot count is four rather than two
so the buffer behind it stays sixteen-byte aligned.

<a id="side-liveness"></a>**Per-side liveness gates per-side work**, and that is not an
optimisation bolted on: it is what having two lists MEANS. The readout, the Gram, the
output and its drain are read-side work; the value staging and the fold are write-side
work; the intra term is the one that needs both, and it needs both to be PRESENT rather
than merely nonzero — a super-chunk with no write entry stages no value tile, so a term
that reads one must not run. On a routing whose sides are separated (`split`) this is
where the mma and the staging the union schedule could not skip actually go away.

## Production and the state plane's addressing

<a id="produce"></a>**The production.** One trailing 16-byte vector is held at a time and
reused across every leading digit a thread owns, so the register cost does not grow with
the block width; each coefficient is one `mul_bf16x2`. A panel row past the gathered count
is exactly zero, and it is zeroed at the LEADING amplitude rather than at the product —
one select per leading digit instead of one per coefficient — and the zero propagates
through a multiplication whose other operand is a finite simplex amplitude. A zero row
contributes nothing to the Gram, nothing to the fold and nothing the drain will publish,
which is what makes a partly filled buffer exact.

<a id="page-sweep"></a>**The page sweep carries a cursor rather than dividing.** `cols` is
`d_v + 1` and therefore not a power of two, so recovering a row from a flat index costs a
magic-number division — a WIDENING MULTIPLY, the one instruction shape this kernel's
bodies do not emit. A step of `kGroupThreads` is one row plus a remainder, so the row
advances by a compare instead. `cols` is 260 bytes and not sixteen-byte aligned, so a page
row cannot move by a vector access either; these are plain float accesses, coalesced
across the group.

<a id="atom-index"></a>**A group's first logical atom** is the leaf-order law's: a block's
leaves are one aligned run starting at `LevelPlan::block_base`, so the index is that base
over the quantum, which is `AtomKeying.atom_of` verbatim. It is derived INSIDE each state
block, so nothing about the state I/O is live across the main loop — which is what the
plane bases cost before there was a slot to resolve. It is ALL-INT, like the plan's own
suffix products: the atom index is bounded by the plane's page count, which the 32-bit
displacement already caps at about a million, so a 64-bit index would be a widening
multiply bought for nothing.

<a id="arm-set"></a>**The arm set.** An arm is `(C, SWZ, BC, D, B)`. `C` and `SWZ` are the
two measured axes; `BC` carries rung 2's measured optimum and is not a tuned axis; the
width configurations are not axes at all but problem sizes — `(8,8)` is `N = 64` with ONE
block, the same-work control; `(64,64)` is the flagship `N = 4096`; `(16,16)` is
`N = 256`, a real fan-in cheap enough for an fp64 reference to gate at full length. The
two-level arms are rung 5A's, arm for arm, so every delta this rung reports is a same-arm
delta; the deep arms are G4's — `(16,16,16)` and `(8,8,8,8)` at the flagship's matched
`N = 4096`, and a `BC = 32` depth-4 arm whose block leaves TWO single-digit levels above a
partial one, so the three-clause form and the two-scalar upper chain are both exercised.
The list itself is GENERATED: `tools/gen_shards.py` owns it, and the `.inc` include is the
whole of the kernel's knowledge of it.

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
union table's own rule: a per-call cost the sparse grid pays is reported on its own line
and never folded into a kernel delta.

**Why three bits and not one predicate.** `READ` and `WRITE` are exact at the block's own
grain — a block is an aligned digit-range rectangle, so any combination of digits live at
one token, one per level, inside the rectangle IS a leaf of the rectangle, which is what
makes the AND-over-levels-then-OR-over-tokens evaluation exact rather than a bound. What a
skipped block would have COST, however, is not a property of the routing: it differs by
call class (stateless, state-in, state-out) and by backing (dense, paged). So the facts
are emitted separately and the predicate over them is the host's
(`docs/internals/chunk/chunk.md#run-table`).

`STATE` is the residency witness for "this block may carry a nonzero entry state": an atom
with no page has provably zero entry fragments, so an unmapped block cannot read anything
out. The dense backing carries no residency map, and the bit is then ZERO everywhere
rather than guessed — the host's dense predicate is weakened to compensate, which is a
recorded weakening rather than a hidden one.

## The work loop, and where it opens

<a id="work-loop"></a>**`PERSIST` compiles the sparse grid's work loop in or out**, on exactly the
rule the carried state's I/O follows: a call that hands over no liveness bitmap runs the
kernel that has no loop in it at all, launched one CTA per block as it always was. With a
bitmap the grid is FIXED — this arm's resident CTAs times the device's SM count, capped at
the block count — and the loop strides over the block axis, testing one byte per work item
and skipping the dead ones. `bh` stays on `blockIdx.y` in both, which is what keeps every
`bh`-keyed base (`v`, the amplitude planes, `o`, `den`, the table rows) a launch constant.

**Why an axis and not a runtime null test.** A loop is not free: its back edge holds live
everything computed above it. Measured, a runtime loop around the hoisted setup cost +20 to
+34 registers and spilled five instantiations inside the main loop; the same body with the
loop degenerated to one iteration cost +2. Compiling the loop out restores the full-grid
kernel exactly, so the launch that does not use the feature does not pay for it — which is
the same reason `STATE` is an axis.

**The loop opens ABOVE the addressing, and that is deliberate.** Everything from the
shared-memory carve down — the operand addresses, the fan-in's rows, the production
geometry, the amplitude columns, the clause rows, the accumulators and every cursor — is
re-derived per work item rather than hoisted, because hoisting is precisely what puts a
value on the back edge. It is the same trade the state I/O already takes: the atom index is
derived inside each state block so that nothing about it is live across the main loop.

**The one hazard the loop introduces is the asynchronous copy pipeline.** The steady loop
issues a `cp.async` group per super-chunk and only ever waits at `STAGES - 2`, so up to
`STAGES - 1` groups are still in flight when it breaks. A one-shot CTA exits and those
copies are discarded; under the loop they land in the V ring slots the NEXT work item is
already staging into. `rendezvous()` does not fence them — `__syncthreads` orders shared
memory accesses, not outstanding asynchronous copies — so each item ends with
`stage_wait<0>()` before its barrier, and only when another item follows.

**The regime is measured, and the plan is what respects it.** On sm_86 at
`BH = 32`, `N = 4096`, the persistent grid beats the full one by 12–47 % of kernel
time at `ρ ≤ 0.125` and LOSES by 13–47 % at `ρ = 1`, crossing over near
`ρ* ≈ 0.25`. The loss at high density is not tail quantization: it is serialization
— a CTA's work items cannot overlap each other's prologues and epilogues the way
the hardware overlaps whole CTAs across waves, and the re-use fence drains the
copy pipeline between every pair. So the sparse grid is a PLAN CHOICE, taken by
handing the launch a bitmap, and there is no density test anywhere in the kernel
or the dispatch. The full-grid path is unaffected either way, because it is a
different compiled variant.

**An arm the loop does not fit carries no persistent variant.** Five do not, for two measured reasons the generator states. `(C=64, BC=128, D=2, B=64)`
declares `reg_floor` 255, the ISA's whole per-thread budget; a loop counter added to it
spills inside the main loop whatever else is trimmed. The four `C=16, BC=64` arms have the
shortest super-chunk loop and the lowest register floor in the manifest, and `ptxas`
answers the loop's added pressure by sinking `(bh)`-keyed widening multiplies INTO that
loop, where the addressing discipline forbids them. The answer to a configuration a
strategy costs too much on is a different configuration, never a special-cased path, so
those arms run the full grid and asking one for the sparse grid is refused by name. The
set is the generator's (`CHUNK_NO_PERSIST`), a measured declaration of the same kind as
`reg_floor` and falsifiable the same way: the spill manifest and the SASS gate both fail
on an arm listed as fitting that does not.
