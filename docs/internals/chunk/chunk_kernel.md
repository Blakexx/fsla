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

**The leaf-order law.** A block's `BC` leaves are a contiguous, `BC`-aligned run
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

**Addressing: NO widening multiply, anywhere in the state I/O.** Each block pins
two PLAIN ORIGINS — the plane's and the table's, neither of them displaced — and
everything per atom is a 32-bit byte offset from one of them: the atom's table
entry is `atom * 4` bytes in, and its page is `slot * 4160` bytes in. That is the
same discipline the gathered rows keep (`gather_ptr`), and the caller owns the
bound: `chunk.cu` refuses a plane of 4 GiB or more, which is a little over a
million pages. It binds the DENSE backing first, and a paged plane holding only
the atoms a plan realized is precisely the answer to that.

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

**Exactness of the seeding.** With no decay in the operator the recurrence is
affine in the state and the readout is inclusive-after-write, so setting the
fragments to the entry state is precisely the oracle's `initial_cache` seeding
and no other term changes. The fragments are SET, never accumulated: they are
provably zero at that point in the prologue.

**The base translation: dense and paged are ONE kernel.** Per atom the kernel
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
