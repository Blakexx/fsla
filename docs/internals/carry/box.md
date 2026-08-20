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
| `s_{D-1} >= kAtomLeaves` | K13 | a token's run over a level that is dense for its side must FILL atoms |

K4 — "the atom is a digit RECTANGLE" — is DISCHARGED BY THE ORDER and no longer needs
an assert. Sixteen consecutive `local` values at a multiple of sixteen are exactly the
set agreeing on every bit above bit 3, and under mixed radix that set is: the levels
whose fields lie above are fixed, the level straddling bit 4 keeps an aligned contiguous
sub-run, and the levels below are free. That is an aligned digit rectangle for EVERY
admissible span vector, where the interleaved order needed `k^D >= 16` and
`4 % kappa == 0` to get it.

K6 (`s_l <= B`, an owner inside a level) is enforced where the arm is declared, because
its failure is a bad arm rather than a bad type.

## What the host mirrors, and why it is a second derivation

`rola/ops/carry.py`'s `box_shape` and `lattice_of_canonical` recompute the same shape
and the same permutation in python, independently. That duplication is deliberate and
is the oracle's own rule (`rola/ops/naive.py`'s `_radix_strides`): a host seam that
imported the device's map would make a wrong convention agree with itself in every gate
the repository has. `tests/k31/test_carry_vs_oracle.py::test_arms_are_the_declared_solver_rows`
is where the two are made to agree as a tested claim.

## K13 and the distribution of `m` (spec §2.5)

`a_{D-1}` — the innermost level's exponent — is raised to whatever `s_{D-1} >= kAtomLeaves`
requires and is never below its balanced share; the rest of `j = log2 M` is spread
balanced over the outer levels, innermost first. At both built topologies (`D = 2`,
`k = 4`, `m = 8`, so `j = 3`) the floor and the balanced share coincide at `a_1 = 2`,
giving `a = (1, 2)`, `s = (8, 16)` and `g = (32, 16)` at `B = 256` / `(8, 4)` at `B = 64`,
with `BC = 128` either way.

K13 is a DECODE fact stated in the carve, and it is what makes the leaf order serve both
consumers at once. The order already puts the levels that are dense for the walking side
fastest-varying (per-level support law: at most one side is sparse per level, so the
sparse levels sit above the dense ones). What the order alone could not give is
OCCUPANCY: with `s_{D-1} = 8` a token's dense-level run is half an atom, an atom
therefore straddles two outer digits, and under a sparse outer level the walk fetches
each atom at half occupancy whatever the order. `s_{D-1} = 16` makes the run exactly one
atom. The fold keeps every property it had — it still keys level 0, and its bucket grows
from one `kMmaN` tile to two.

A `(k, m)` whose innermost span cannot reach an atom is REFUSED at the type (there is no
fallback span vector), which is why K13 is an assert and not a solver preference. The
refusal binds at depth: `D = 4`, `B = 8`, `k = 2` has `s_3 = 16 > B`, so K13 and K6
cannot both hold and that topology needs a re-solve — spec §7.4's `D = 3` and `D = 4`
rows are open on this, and none of them is a built arm.
