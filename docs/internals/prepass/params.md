# `prepass/params.cuh` — the pre-pass's parameter block

Mirrors `csrc/rola/src/prepass/params.cuh`.

One `struct` by value, `PrepassParams`, reaching the kernel as a grid constant so every inlined
component reads its fields as constant-bank operands: the tall owner set's addressing block
(`CarryGeomRT`, derived on the host by the tall law, `common/geom.md#tall-owner-set`); the two
`sparse_mask` words, bit `l` set when level `l` is sparse on the read (0) or write (1) side —
the levels whose digit rows the liveness words are staged for, a dense level's digits being
live wherever the token is; the bundle's u8 rows on both sides, `[BH, L, sum_l B_l]`, with the
per-level row maxima `[BH, L, D]`; the write gain and the value scale `[BH, L]`; the two outputs
`den` and `scale`, `[BH, L]` fp32, REDUCED into; the class-1 liveness words
`[BH, 2, sum_l B_l, ceil(L / 32)]`; and the ledger and trace bindings.

`PrepassPhase` names the ledger's rows, one a component of the skeleton — frame, words, lists,
fill, fold, publish, rescale, readout, drain — so the frame's fixed cost is read per component
(`prepass_kernel.md#frame`).
