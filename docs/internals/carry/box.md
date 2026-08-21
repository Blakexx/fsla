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
