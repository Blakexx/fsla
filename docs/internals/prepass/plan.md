# `prepass/plan.cuh` — the tall plan and its shared ledger

Mirrors `csrc/rola/src/prepass/plan.cuh`.

`TallPlan<D>` is the compile-time shape of the int8 carry's skeleton, which the pre-pass is at two
columns a leaf: eight warps, 1024 leaves a CTA and 128 a warp (`common/geom.md#tall-owner-set`),
the 512-token window in sixteen rounds and sixteen 32-token tiles, the K block of 32 leaves and
four blocks a read warp.

<a id="pool"></a>
## The pool

A chunk of 64 tokens in two barriered slots. The write-sparse level's entries over the region's
digits (128 bytes a token, the widest span the tall law derives) are token-major rows at a 144-byte
stride, so one digit's bytes across consecutive tokens spread over the banks for the fold's gathers;
the write-dense entries (8 bytes a token), the two write scales (8), `g_t` (4) and `s_t` (4) are
planes, tokens contiguous, as they are in global memory.

THE COPY RANKING (measured on sm_86, 2026-09-28, a probe of one warp instruction's patterns with
eight warps in flight, clock cycles a warp instruction; the calibration harness carries the durable
rows): 32 lanes over 512 contiguous bytes 81; a lane quad a 64-byte run of one row, eight rows 84
(the same into 160-byte-stride rows 85); a quad of 8-byte copies a 32-byte sector, eight rows 99;
the carry fill's row pattern (two lanes a row, sixteen rows) 128; a lane pair a 32-byte sector into
two planes 258; one lane a row, 32 rows 268; eight lanes a 128-byte row into eight planes 239. The
shared side's stride or padding moves a pattern by 3%; which lanes share a line and how much of a
line an instruction takes move it 3x. ncu's "shared wavefronts" on an asynchronous copy is 8 for the
fast patterns and 32 for the slow ones whatever its "ideal" column says, and its theoretical L2
sectors count a 16-byte lane copy as a full sector. So the sparse rows are copied by quads, a 64-byte
run of one token a quad, eight tokens an instruction; the per-token fields by whole warps over 32
consecutive tokens (contiguous in global memory), a warp pair a field. The frame's first plane layout
(one lane a token) cost 3x on every copy. Eight chunks a window.

<a id="ring"></a>
## The read ring

A warp's two slots of a tile's read rows, 48 bytes a token: the dense sub-block's 32 entries, the
sparse entries (4) and the two read scales (8). The dense entries are copied by quads of 8-byte
copies (one token's 32-byte sector a quad, eight tokens an instruction), the rest a lane a token; the
48-byte stride puts eight consecutive tokens on eight distinct bank slices, so the readout's A
fragment (a token a lane quad, four consecutive leaves a lane) loads conflict-free. The ring runs a
tile ahead of the tile being read.

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
