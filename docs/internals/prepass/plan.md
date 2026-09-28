# `prepass/plan.cuh` — the tall plan and its shared ledger

Mirrors `csrc/rola/src/prepass/plan.cuh`.

`TallPlan<D>` is the compile-time shape of the int8 carry's skeleton, which the pre-pass is at two
columns a leaf: eight warps, 1024 leaves a CTA and 128 a warp (`common/geom.md#tall-owner-set`),
the 512-token window in sixteen rounds and sixteen 32-token tiles, the K block of 32 leaves and
four blocks a read warp.

<a id="pool"></a>
## The pool

A chunk of 64 tokens in two barriered slots. A row is the write-sparse level's entries over the
region's digits (128 bytes: the widest span the tall law derives), the write-dense entries (16),
and four floats — the two write row scales, `g_t` and `s_t` (16). Eight chunks a window.

<a id="ring"></a>
## The read ring

A warp's two slots of a tile's read rows: the dense sub-block's 32 entries, the sparse entries
and two scales, 48 bytes a token. The ring runs a tile ahead of the tile being read.

<a id="transfer"></a>
## The transfer

Every leaf's two columns as s16 against the leaf's max, plus that max as a float: 4 KB of values
and 8 KB of maxima. The fold warps write it in the read layout; the read warps rescale their
blocks from it.

<a id="panel"></a>
## The panel

Each warp's own row of a 128-token range's two fp32 sums, plain stores; the eight rows are summed
a token at the drain. Shared float atomics are a CAS loop on this architecture, which the first
frame measurement exposed at 1.6K cycles a tile; the rows cost 8 KB and no atomic.

<a id="ledger"></a>
## The shared ledger

Words (two parities), the pool, the rings, the transfer, the panel, the barriers (pool full and
empty a slot, a ring barrier a warp-slot, the words landed, the publish and the done a parity),
the phase ledger, the list counts: about 85 KB at depth two, held under the architecture's
per-CTA maximum by the plan's own assertion.
