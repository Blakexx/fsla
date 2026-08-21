# `carry_api.cuh` — the carry family's host declarations

The mirror of `csrc/rola/src/carry/carry_api.cuh`. Declarations only, so the pybind
translation unit never sees a device body — the `chunk_api.cuh` convention.

Three entries:

* `forward_inter(pr, pw, gw, v, widths, k, m, window, chunk, carve, level_modes, s_in)` — the INTER
  term only, returned UNDIVIDED as `(num, den)` plus the final state plane in LATTICE
  leaf order. Undivided because the gate compares against an fp64 reference and a ratio
  would fold the readout floor into a conformance number that is about the recurrence.
* `build_stamp(...)` — the device-side build identity and the arm's census.
* `arms()` — the built `(D, B, k, m, W, C, carve, level_modes, BC, owners, smem)` tuples.

`carve` is `0` for the state-split and `1` for the channel-split, matching
`Carve` in `carry_kernel.cuh`.

`level_modes` is architecture law 3's per-level support declaration, two bits per level
with level 0 in the low pair: bit 0 of a pair says the READ side is value-sparse at that
level, bit 1 says the WRITE side is. It is the intra family's encoding, so one statement
serves both kernels. It is a DECLARATION and never a switch — the produce synthesizes
exactly the coefficients the support plane marks live, so no declaration can move a
number and `DENSE_BOTH` everywhere is always correct. A declaration the family has no
compacted arm for is resolved AT THIS SEAM to `DENSE_BOTH`; the topology itself is never
degraded, an unbuilt `(D, B, k, m, W, C)` being refused.

This surface is deliberately outside the shipped dispatch; see
`docs/internals/carry/carry.md`.
