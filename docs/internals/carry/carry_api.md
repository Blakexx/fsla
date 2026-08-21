# `carry_api.cuh` — the carry family's host declarations

The mirror of `csrc/rola/src/carry/carry_api.cuh`. Declarations only, so the pybind
translation unit never sees a device body — the `chunk_api.cuh` convention.

Five entries:

* `forward_inter(pr, pw, gw, v, widths, k, m, window, chunk, carve, s_in)` — the INTER
  term only, returned UNDIVIDED as `(num, den)` plus the final state plane in LATTICE
  leaf order. Undivided because the gate compares against an fp64 reference and a ratio
  would fold the readout floor into a conformance number that is about the recurrence.
* `build_stamp(...)` — the device-side build identity and the arm's census.
* `owner_rows(pr, pw, widths, k, m, carve)` — the ballot rows as a tensor, at the
  lattice's owner spans or at the CARVE grain the kernel consumes; the gate that the
  owner word is the OR of a group's rows reads both.
* `backward_inter(...)` / `backward_build_stamp(...)` — the reverse family's two scans
  and their per-kernel census.
* `arms()` — the built `(D, B, k, m, W, C, carve, BC, owners, smem)` tuples.

`carve` names the ONE ruled carve — `1`, the innermost level's leaf split
(`carry_kernel.md`). A caller asking for anything else is refused rather than
served something else.

This surface is deliberately outside the shipped dispatch; see
`docs/internals/carry/carry.md`.
