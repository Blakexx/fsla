# `box.cuh` — the `(k, m)` box lattice

The mirror of `csrc/rola/src/carry/box.cuh`. This file is the R0 spec's LATTICE-1
linearization made executable (`development/plans/K31_R0_ABI_RIPPLE_SPEC.md` §§2.1–2.5);
the spec is normative and this doc records what the header does with it, not a second
derivation of it.

**Symbols.** `D` levels, `B` digits per level, `K` the uniform sub-box span per level,
`M` the capacity multiplier with `M = Π_l m_l`, `s_l = K·m_l` an owner's span at level
`l`, `g_l = B/s_l` the owner-grid extent at level `l`, `BC = Π_l s_l` the leaves one
owner holds, `N = B^D`. A leaf's canonical (MSB-first mixed-radix) index is `ell`; its
lattice index is `lambda`.

## Why nothing here divides and nothing here tabulates

`pi: ell -> lambda` is a fixed BIT PERMUTATION. In `ell`, level `l` occupies a
contiguous `log2 B`-bit field and inside it `(o_l | r_l)` occupy `(log2 g_l | log2 s_l)`
bits, most significant first — `r_l = d_l - o_l·s_l` is the digit's offset inside the
owner's run. In `lambda` the same fields are regrouped as
`[o_0 … o_{D-1}][r_0 … r_{D-1}]`. Every field keeps its width; only its position moves.

**The intra-box order is MIXED RADIX over the runs** — `local = Σ_l r_l · Π_{l'>l} s_{l'}`,
level `D-1` fastest — which is the K33 enabler (queue card, enabler 2) and supersedes
the R0 spec's `[u_0 … u_{D-1}][c_0 … c_{D-1}]` interleave. Under the interleave each
level's run was SPLIT across two fields (`u_l` high, `c_l` low) and no level owned a
contiguous run of leaves; under mixed radix a bucket keyed on level `l`'s digit owns
`Π_{l'>l} s_{l'}` leaves, and for the OUTERMOST level that run is contiguous and
`BC / s_0` wide — two `kMmaN` tiles at the flagship `(8, 16)`, so a fold bucket
accumulates directly into whole state tiles with no padding. `run_of(l, local)` is now a
single field extract instead of `u_of·K + c_of`, and `u_of`/`c_of` are read back out of
it for the sub-box arithmetic that still wants them.

Three consequences the header spends its whole surface on:

* the owner index is the top `Σ_l log2 g_l` bits of `lambda`, so `o_l` is a bit-field
  extract and never a division chain;
* an owner's leaves are `lambda ∈ [O·BC, (O+1)·BC)` — one contiguous, `BC`-aligned run
  **by construction**, where the shipped kernel proves the same property per level with
  a `static_assert` on `SpanPlan::base_aligned`;
* the atom keying expression is unchanged from `rola.ops.paging.AtomKeying`; only its
  argument is re-read as `lambda`.

## The constraints, and which one each `static_assert` is

The `(k, m)` solver's admissibility system is spec §3.3. The header asserts the four
that are properties of the type rather than of a machine:

| assert | constraint | why |
|---|---|---|
| `kBC % kAtomLeaves == 0` | K2 | a page atom must not straddle two owners, i.e. two CTAs |
| `kLocalBits >= 4` | K3 | a 16-aligned 16-run must lie inside one owner |
| powers of two | K1 | `s_l | B` must hold, and the bit-permutation theorem requires it |
| `s_l <= B`, `B % s_l == 0` | K6 | an owner's run at a level is an aligned run INSIDE that level |
| `sum_l a_l == j` | K6 | a capacity the levels cannot carry is refused, never truncated |

K4 — "the atom is a digit RECTANGLE" — is DISCHARGED BY THE ORDER and no longer needs
an assert. Sixteen consecutive `local` values at a multiple of sixteen are exactly the
set agreeing on every bit above bit 3, and under mixed radix that set is: the levels
whose fields lie above are fixed, the level straddling bit 4 keeps an aligned contiguous
sub-run, and the levels below are free. That is an aligned digit rectangle for EVERY
admissible span vector, where the interleaved order needed `k^D >= 16` and
`4 % kappa == 0` to get it.

K13 — "the innermost span covers a whole atom" — is NOT an assert. It is a PREFERENCE
in the capacity convention below, because it is unsatisfiable at a level narrower than
an atom and satisfiability is not the same question as correctness.

## What the host mirrors, and why it is a second derivation

`rola/ops/carry.py`'s `box_shape` and `lattice_of_canonical` recompute the same shape
and the same permutation in python, independently. That duplication is deliberate and
is the oracle's own rule (`rola/ops/naive.py`'s `_radix_strides`): a host seam that
imported the device's map would make a wrong convention agree with itself in every gate
the repository has. `tests/k31/test_carry_vs_oracle.py::test_arms_are_the_declared_solver_rows`
is where the two are made to agree as a tested claim.

## The atom rectangle, and the distribution of `m` (spec §2.5)

### The atom is a rectangle at every span vector

Let `P(t) = prod_{l >= t} s_l` and let `t*` be the smallest `t` with `P(t) <= A`, where
`A = kAtomLeaves`. An atom holds levels `t*..D-1` WHOLE, plus an aligned contiguous
sub-run of `A / P(t*)` digits at level `t*-1`; every level above is fixed. `kAtomBase`,
`kAtomWhole` and `kAtomPartial` are those three quantities.

This is the general form of what K13 asked for, and it is a THEOREM about the order
rather than a constraint on `(k, m)` — it follows from K1 (powers of two) and K3 (the
atom lies inside one owner), both already asserted.

### What the rectangle costs, and where

A token FILLS an atom exactly when every level of the rectangle is dense for its side.
A ONE-LEVEL rectangle asks that of one level; a rectangle spanning two asks it of two,
and under an alternating declaration at most one of any adjacent pair is dense per side,
so full occupancy on both sides is then impossible. That is why the convention pushes
capacity INNERMOST: it is buying a one-level rectangle wherever one is reachable.

### The convention

`a_{D-1}` is raised to whatever `s_{D-1} >= A` requires, never below its balanced share,
and never above `beta - kappa` — K6, the level's own width. The rest of `j = log2 M` is
spread balanced over the outer levels, innermost first. The width ceiling is what makes
the convention TOTAL: a level narrower than an atom cannot absorb the floor, and the
capacity spills outward rather than the plan being refused.

The solved rows:

| topology | `k` | `m` | `a_l` | `s_l` | `g_l` | `BC` | atom rectangle | walk unit |
|---|---|---|---|---|---|---|---|---|
| `(256,256)` | 4 | 8 | (1,2) | (8,16) | (32,16) | 128 | 1 level x 1 digit | 16 leaves = 1 atom |
| `(64,64)` | 4 | 8 | (1,2) | (8,16) | (8,4) | 128 | 1 level x 1 digit | 16 leaves = 1 atom |
| `(16,16,16)` | 2 | 16 | (0,1,3) | (2,4,16) | (8,4,1) | 128 | 1 level x 1 digit | 16 leaves = 1 atom |
| `(8,8,8,8)` | 2 | 8 | (0,0,1,2) | (2,2,4,8) | (4,4,2,1) | 128 | 1 level x 2 digits | 32 leaves = 2 atoms |

`(16,16,16)` reaches a one-level rectangle only at `k = 2`, which the INTERLEAVED order
forbade (its K3 was `kappa*D >= 4`); the order change is what makes the depth-three arm
satisfiable at all. `(8,8,8,8)` cannot reach one at any `(k, m)`: every span there is at
most `B = 8`, narrower than an atom. Its touched atoms are 50-65% occupied under an
alternating declaration — an occupancy cost, never an error, and one no span vector can
remove. The fix if it ever matters is a wider level, not a different `(k, m)`.

### The walk unit

`kUnitBase` names the shallowest suffix of levels holding a whole number of atoms, and
`kUnitLeaves`/`kUnitAtoms` its size. K2 makes `l = 0` always qualify, so the search is
total. The decode walk enumerates `(owner, r_0 .. r_{kUnitBase-1})`, and the unit is one
atom wherever the rectangle is one level and several atoms where it is not.

## The stream box: ORDER BY MODE, THEN CARVE

`BoxPlan` names the OWNER's box. One STREAM's box inside it is a per-level span vector
`S_l | s_l` — the digits of level `l` one stream holds — with a level past `D` carrying
`B` (the plan's identity). A stream box is an AFFINE RECTANGLE of digits: contiguity in
leaf order is a property of carving a prefix, never a requirement, so `[1, 4, 2, 4]` over
`[4, 4, 4, 4]` is admissible and gives eight streams of 32 leaves. The stream grid is
`g_l = s_l / S_l` per level and `Π_l g_l` streams, and a stream's coordinate at a level is
a bit field of its index, MSB-first exactly as the owner grid's — so every use site is a
shift and a mask and nothing tabulates.

**THE VECTOR IS NOT DECLARED. It is derived from the MODES and a STREAM COUNT** (P76,
ruled 2026-08-26; K35 S1b). `ModeOrder<D, MO, SIDE>` classes each level from the
declaration alone:

| class | meaning on this side | position |
|---|---|---|
| `DENSE` | not sparse on this side | innermost |
| `SPARSE` | sparse on this side only | middle |
| `BOTH_SPARSE` | sparse on BOTH sides | outermost |

with lattice order inside a class and nothing pinned. Dense innermost makes an eight-slot
n-tile full along the dense digits; sparse next lets those levels index WHOLE tiles, where
the digit skip is exact; BOTH_SPARSE outermost is where the two sides' carves AGREE, so
the exchange is block-diagonal on those digits and vanishes when they agree everywhere.
`BOTH_SPARSE` says both sides are sparse there and NOTHING MORE — each side votes its own
bitlists from its own factors and the two supports never have to coincide.

`StreamSpans<D, B, K, M, MO, SIDE, NS>` then walks that order from the OUTERMOST level
inward, carving each level whole (`S_l = 1`) and then partially (halving) until the stream
grid's product is `NS`; a `static_assert` refuses a count the box cannot reach. The write
side's count is `kStreams`, the read side's is `kWarps` — one reader per warp — and the
EXECUTOR MAP is the second, independent declaration over the same counts (`WPS`, the warps
per write stream): `warps = write streams x WPS = read streams`.

`CarveRows<SV>` is ONE SIDE's ballot table, at THAT SIDE's own partition (K35 S1b
addendum). A bitlist is per side, per level, per run of that side's partition, voted from
that side's own factors, so nothing is gained by holding the finer of the two partitions
for both — the coupling only refined each side's table by the other's carve. A level a
side spans whole emits no rows: its clause is true for every token and drops out of the
conjunction on its own. A stream's own test therefore reads EXACTLY ONE row per tested
level, and only the OWNER's whole-run OR spans several (`g_l` of them). The table stays
sigma-shaped — `Σ_l B/S_l` — so it does not grow with `N`.

Neither side's tested levels need be a PREFIX of the levels: the write carve `[1,4,2,4]`
tests levels 0 and 2 and spans 1 and 3 whole. The prefix law is a property of the two
BITMAP REDUCTIONS (`atom_bitmap_kernel`, `block_bitmap_kernel`), which are instantiated at
the lex policy only, and it is asserted there rather than on the table.

THE BUILT ARMS' DERIVED VECTORS. `MO` is the per-level declaration (`ALT` at depth `D` is
write-sparse on the even levels and read-sparse on the odd, i.e. 6 / 38 / 102 at
`D = 2/3/4`); `3` is BOTH-SPARSE at level zero and dense at level one.

| topology | `k` | `m` | `s_l` | `MO` | W str | R str | WPS | write spans | read spans |
|---|---|---|---|---|---|---|---|---|---|
| `(256,256)` | 4 | 8 | (8,16) | 0 | 1 | 1 | 1 | (8,16) | (8,16) |
| `(256,256)` | 4 | 8 | (8,16) | 6 | 4 | 4 | 1 | (2,16) | (8,4) |
| `(64,64)` | 4 | 8 | (8,16) | 6 | 4 | 4 | 1 | (2,16) | (8,4) |
| `(16,16,16)` | 2 | 16 | (2,4,16) | 38 | 1 | 1 | 1 | (2,4,16) | (2,4,16) |
| `(8,8,8,8)` | 2 | 8 | (2,2,4,8) | 102 | 4 | 4 | 1 | (1,2,2,8) | (2,1,4,4) |
| `(4,4,4,4)` | 4 | 1 | (4,4,4,4) | 102 | 8 | 8 | 1 | (1,4,2,4) | (4,1,4,2) |
| `(256,256)` | 4 | 8 | (8,16) | 6 | 4 | 8 | 2 | (2,16) | (8,2) |
| `(256,256)` | 4 | 8 | (8,16) | 6 | 8 | 8 | 1 | (1,16) | (8,2) |
| `(256,256)` | 4 | 4 | (4,16) | 6 | 4 | 4 | 1 | (1,16) | (4,4) |
| `(256,256)` | 4 | 8 | (8,16) | 3 | 4 | 4 | 1 | (2,16) | (2,16) |

The two 1x8 rows are the CONTAINER arms (K35 S3a): the SAME flagship box reached with
eight executors by the two available routes — four write streams column-split two ways,
and eight write streams one warp each — so the container question the card leaves to
measurement has both of its answers built. The `(4,4,4,4)` row is the MULTI-LEVEL carve:
each side carves TWO levels, and it is the only box among the built arms whose owner is
the whole topology (`g_l = 1`, one owner). `k = 4, m = 1` is the only `(k, m)` reaching
`s = (4, 4, 4, 4)`: the capacity convention's atom FLOOR pushes `m` innermost, so any
`m > 1` at `D = 4` unbalances the vector, and at `B = 8` the ceiling `K6` then refuses
`k = 4, m = 1` outright.

THE LAST ROW IS THE TIED ARM. Every level is BOTH_SPARSE or dense on both sides, so the
two orders coincide; at equal stream counts the derivation then hands the two sides the
SAME vector, the exchange's permutation `leaf -> (reader, slot)` is the IDENTITY, and the
publish and the pull are elided at compile time — `hi` never leaves the register file and
the transit owns no shared memory at all (`CarryGeom::kTied`, P76 "THE EXCHANGE MOVES ONLY
THE DISORDER"). `tests/k31/test_segment_layout.py` computes the exchange volume from the
two derived vectors at every arm and asserts the kernel's transit is exactly that.

## The staged factor row (K35 S2)

What a fetcher needs of one token on one side is the OWNER's digit run at each level —
`Σ_l s_l` amplitudes — and NOT the `kWtot = D·B` packed row, which is the width of the
whole topology. `BoxPlan` therefore names that compressed row itself, because the
alignment argument is the box's and not the kernel's:

* `stage_el(l) = min(s_l, kStageAsync)` — the amplitudes one asynchronous copy moves at
  level `l`; `kStageAsync = 8` is the sixteen-byte quantum in amplitudes.
* `stage_off(l)` — level `l`'s run, placed after level `l-1`'s and rounded up to
  `stage_el(l)`, so every copy's SHARED address is naturally aligned.
* `kStageW` — the row's width, rounded to `kStageAsync` so a row base is too.
* `stage_chunks(l)`, `stage_chunk_base(l)`, `kStageChunks` — the copy count per level and
  the flat chunk index a thread claims.

The GLOBAL address is aligned by the lattice rather than by rounding: a level's column
base is `l·B` and an owner's run starts at a multiple of `s_l`, so `runCol(l)·2` is a
multiple of `min(s_l, 8)·2` at every admissible box. `K ≥ 2` makes the narrowest run four
bytes, which is the narrowest asynchronous copy there is; that is the one clause the
layout asserts.

Widths at the built arms: `s = (8,16)` → `kStageW = 24` (offsets 0, 8; three chunks);
`s = (2,2,4,8)` → `16` (offsets 0, 2, 4, 8; four chunks); `s = (4,4,4,4)` → `16`
(offsets 0, 4, 8, 12; four chunks). The `[BH, L, kWtot]` packed row would be 512, 32 and
16 amplitudes for the same three.
