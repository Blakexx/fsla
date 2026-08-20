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

`pi: ell -> lambda` is a fixed BIT PERMUTATION (spec §2.4). In `ell`, level `l` occupies
a contiguous `log2 B`-bit field and inside it `(o_l | u_l | c_l)` occupy
`(log2 g_l | log2 m_l | log2 K)` bits, most significant first. In `lambda` the same
fields are regrouped as `[o_0 … o_{D-1}][u_0 … u_{D-1}][c_0 … c_{D-1}]`. Every field
keeps its width; only its position moves.

Three consequences the header spends its whole surface on:

* the owner index is the top `Σ_l log2 g_l` bits of `lambda`, so `o_l` is a bit-field
  extract and never a division chain;
* an owner's leaves are `lambda ∈ [O·BC, (O+1)·BC)` — one contiguous, `BC`-aligned run
  **by construction**, where the shipped kernel proves the same property per level with
  a `static_assert` on `LevelPlan::base_aligned`;
* the atom keying expression is unchanged from `rola.ops.paging.AtomKeying`; only its
  argument is re-read as `lambda`.

## The constraints, and which one each `static_assert` is

The `(k, m)` solver's admissibility system is spec §3.3. The header asserts the four
that are properties of the type rather than of a machine:

| assert | constraint | why |
|---|---|---|
| `kBC % 16 == 0` | K2 | a page atom must not straddle two owners, i.e. two CTAs |
| `kKappa * D >= 4` | K3 | a 16-aligned 16-run must lie inside one sub-box |
| `4 % kKappa == 0` | K4 | the atom must be a digit RECTANGLE, so `k ∈ {2, 4, 16}` |
| powers of two | K1 | `s_l | B` must hold, and the bit-permutation theorem requires it |

K6 (`s_l <= B`, an owner inside a level) is enforced where the arm is declared, because
its failure is a bad arm rather than a bad type.

## What the host mirrors, and why it is a second derivation

`rola/ops/carry.py`'s `box_shape` and `lattice_of_canonical` recompute the same shape
and the same permutation in python, independently. That duplication is deliberate and
is the oracle's own rule (`rola/ops/naive.py`'s `_radix_strides`): a host seam that
imported the device's map would make a wrong convention agree with itself in every gate
the repository has. `tests/k31/test_carry_vs_oracle.py::test_arms_are_the_declared_solver_rows`
is where the two are made to agree as a tested claim.
