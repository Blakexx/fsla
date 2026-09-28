# `prepass/prepass_host.cuh` — the call's derivation

Mirrors `csrc/rola/src/prepass/prepass_host.cuh`.

`derive_prepass_call` is the pre-pass boundary: it refuses shapes and dtypes by name — the rows
`[BH, L, sum_l B_l]` u8 on both sides, the maxima `[BH, L, D]` fp32, the gain, the value scale and
both outputs `[BH, L]` fp32, the liveness words `[BH, 2, sum_l B_l, ceil(L / 32)]` int32, every
width a power of two at or above the page atom — and fills the parameter block but for the
ledger. The addressing block comes from the geometry block's own entry for the tall set,
`tall_block` (`common/geom_api.md#tall-block`): the mode word decides the spans, both stream
counts are the CTA's eight warps, run placement, no warp-level page floor. The two sparse masks
are read off the mode word's two bits a level.
