# `carry_api.cuh` — the carry family's host declarations

The mirror of `csrc/rola/src/carry/carry_api.cuh`. Declarations only, so the pybind
translation unit never sees a device body — the `chunk_api.cuh` convention.

Five entries:

* `forward_inter(pr, pw, gw, v, widths, k, m, window, chunk, carve, level_modes, s_in)` —
  the INTER term only, returned UNDIVIDED as `(num, den)` plus the final state plane in
  LATTICE leaf order. Undivided because the gate compares against an fp64 reference and a
  ratio would fold the readout floor into a conformance number that is about the
  recurrence.
* `build_stamp(...)` — the device-side build identity and the arm's census.
* `owner_rows(pr, pw, widths, k, m, carve)` — the ballot rows as a tensor, at the
  lattice's owner spans or at the CARVE grain the kernel consumes; the gate that the
  owner word is the OR of a group's rows reads both.
* `backward_inter(...)` / `backward_build_stamp(...)` — the reverse family's two scans
  and their per-kernel census.
* `arms()` — the built `(D, B, k, m, W, C, carve, level_modes, BC, owners, smem,
  carve_level)` tuples.

`carve` names the ONE ruled carve — `1`, the innermost level's leaf split
(`carry_kernel.md`). A caller asking for anything else is refused rather than
served something else.

`level_modes` is K33 law 3's per-level declaration — two bits per level, level 0 in the
low pair, bit 0 = the READ side is value-sparse there and bit 1 = the WRITE side — and
it is the vocabulary `rola.ops.intra.pack_modes` already speaks. It CHOOSES THE BODY
(`carry_kernel.md`), so it is a correctness-neutral performance declaration: DENSE_BOTH
is always right and only ever slower, and a declaration the family has no carved body
for is served DENSE_BOTH rather than refused. The TOPOLOGY is not treated that way — an
unbuilt `(D, B, k, m, W, C)` is refused outright.

This surface is deliberately outside the shipped dispatch; see
`docs/internals/carry/carry.md`.

## Comments moved from source (S5b)

Verbatim `//:` prose blocks from `csrc/rola/src/carry/carry_api.cuh`, in source order, one heading per declaration. Source keeps only a short decl block pointing here.

<a id="note-l14"></a>
### near line 14

the INTER term only: `(num, den)`, undivided.  THE STATE PLANES ARE THE CALLER'S,
which is the shipped contract (docs/internals/state.md section 4): `s_in` and `s_out`
are `[BH, N/16, 16, d_v+1]` fp32 in CANONICAL leaf order (the atom-major view of
`[BH, N, cols]`), a continuation hands the SAME
tensor as both and it advances in place, `page_tbl` is the `[BH, N/16]` int32 slot
table (null = the dense backing, where the slot IS the atom), and BOTH PLANES NULL is
the null-state call: a runtime null check inside the one compiled body (K41; no
STATE template axis) skips the sweep at each null end, so nothing is read,
written, or allocated there.
`carve` names the built (carve, executor map): 1 = the ruled
leaf split, 2/3 = the two 1x8 containers.  The arm's `d_v` is READ OFF `v` and is
part of the key.  `level_modes` is law 3's per-level declaration, two bits
per level (bit 0 of a level's pair = the READ side is value-sparse there, bit 1
= the WRITE side), and it CHOOSES THE BODY; DENSE_BOTH is always correct.

<a id="note-l35"></a>
### near line 35

the BALLOT ROWS as a tensor: `[BH, 2, entries, L/32]` packed bits, read rows
then write rows, one row per (level, group) -- at the lattice's owner spans
`s_l`, or at the CARVE grain the kernel consumes (`carve`), which refines the
innermost level into the four warps' groups.

<a id="note-l42"></a>
### near line 42

device-side build identity plus the built arm's register/SMEM census, the carve
level the declaration named (-1 = the channel-split body), the WRITE stream count
and the arm's `d_v`.  `d_v` is part of the arm key (K35 S3b), so it is an argument
here as well: the same declaration at the two built widths is two arms.
