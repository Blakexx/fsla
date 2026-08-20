# `carry_api.cuh` — the carry family's host declarations

The mirror of `csrc/rola/src/carry/carry_api.cuh`. Declarations only, so the pybind
translation unit never sees a device body — the `chunk_api.cuh` convention.

Three entries:

* `forward_inter(pr, pw, gw, v, widths, k, m, window, chunk, carve, s_in)` — the INTER
  term only, returned UNDIVIDED as `(num, den)` plus the final state plane in LATTICE
  leaf order. Undivided because the gate compares against an fp64 reference and a ratio
  would fold the readout floor into a conformance number that is about the recurrence.
* `build_stamp(...)` — the device-side build identity and the arm's census.
* `arms()` — the built `(D, B, k, m, W, C, carve, BC, owners, smem)` tuples.

`carve` is `0` for the state-split and `1` for the channel-split, matching
`Carve` in `carry_kernel.cuh`.

This surface is deliberately outside the shipped dispatch; see
`docs/internals/carry/carry.md`.
