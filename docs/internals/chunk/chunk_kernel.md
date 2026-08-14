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
`s_in`, the plane they are seeded from. Both are RUNTIME NULLS on `ChunkParams`,
tested once and outside every loop — the arm matrix is generated, and doubling
it to carry a state-I/O flag would buy nothing a CTA-uniform predicate at the
ends of the kernel does not.

**The plane.** `[BH, N/16, 16, d_v+1]` fp32: the atom-major view of
`[BH, N, d_v+1]`, byte for byte, with the last column the mass channel global
normalization divides by. The caller allocates it (`rola.ops.chunk.state_plane`),
because a carried state is the caller's object. `d_v + 1 = 65` is 260 bytes and
not sixteen-byte aligned, so nothing here vectorizes; the accesses are plain
floats, coalesced across the group.

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

**Addressing.** One widening multiply per CTA, formed beside the output plane's
base and pinned there, with 32-bit byte displacements after it — a CTA's atoms
span `BC * 65 * 4` bytes, 66 kB at the widest block. Two consequences the SASS
gate had to teach: the page sweep carries a `(row, column)` cursor rather than
dividing a flat index by 65 (a magic-number division is a widening multiply),
and `LevelPlan::base_term` takes its level as a TEMPLATE parameter, because
`off_at`'s recursive constexpr divisor does not fold when the level arrives as an
argument — it reached the SASS as a loop and a reciprocal.

**Exactness of the seeding.** With no decay in the operator the recurrence is
affine in the state and the readout is inclusive-after-write, so setting the
fragments to the entry state is precisely the oracle's `initial_cache` seeding
and no other term changes. The fragments are SET, never accumulated: they are
provably zero at that point in the prologue.

**The page table** is on the ABI and refused: this kernel addresses the dense
plane, where a slot is its atom. Resolving an atom-to-slot map arrives with the
paged backing (`docs/internals/paging/paging.md`).
