# `prepass/plan.cuh` — the tall plan and its shared ledger

Mirrors `csrc/rola/src/prepass/plan.cuh`.

`TallPlan<D>` is the compile-time shape of the int8 carry's skeleton, which the pre-pass is at two
columns a leaf: eight warps, 1024 leaves a CTA and 128 a warp (`common/geom.md#tall-owner-set`),
the 512-token window in sixteen rounds and sixteen 32-token tiles, the K block of 32 leaves and
four blocks a read warp.

<a id="pool"></a>
## The pool

A chunk of 32 tokens in two barriered slots, DIGIT-MAJOR: a digit's 32 token bytes are one row, a
word a token quad, so the fold's operands are single words (four consecutive tokens of one digit)
and the pool is the transpose of the bundle's token rows -- as FA keeps V transposed for the
contraction whose reduction axis is the token. The write-sparse digits are 128 rows at a 36-byte
stride (an m-tile half's two lists are eight digits apart: eight banks apart) plus a zero row a
padding list reads; the write-dense digits 128 rows at a 48-byte stride (eight consecutive digits
cover the 32 banks exactly); then the factors `[column][token]`, formed by the fill from the token's gain, value scale
and write maxima. A chunk is one liveness word a list: sixteen a window.

THE COPY RANKING (measured on sm_86, 2026-09-28, a probe of one warp instruction's patterns with
eight warps in flight, clock cycles a warp instruction; the calibration harness carries the durable
rows): 32 lanes over 512 contiguous bytes 81; a lane quad a 64-byte run of one row, eight rows 84
(the same into 160-byte-stride rows 85); a quad of 8-byte copies a 32-byte sector, eight rows 99;
the carry fill's row pattern (two lanes a row, sixteen rows) 128; a lane pair a 32-byte sector into
two planes 258; one lane a row, 32 rows 268; eight lanes a 128-byte row into eight planes 239. The
shared side's stride or padding moves a pattern by 3%; which lanes share a line and how much of a
line an instruction takes move it 3x. ncu's "shared wavefronts" on an asynchronous copy is 8 for the
fast patterns and 32 for the slow ones whatever its "ideal" column says, and its theoretical L2
sectors count a 16-byte lane copy as a full sector. The ring's copies follow it (a quad a token's
sector). The pool's digit rows cannot come from the copy engine (it does not transpose bytes): the
fill loads a token's 16 bytes a lane (a quad four consecutive tokens at one 16-byte piece: 64
contiguous bytes of one line a quad, the fast pattern) and transposes in registers
(`prepass_kernel.md#fill`).

<a id="fold"></a>
## The fold's tiles

A warp's 128 leaves are sixteen tiles: a list (a write-sparse digit) x an n-tile of eight write-dense
digits. At a write-dense span of 8 Q the warp holds 16 / Q lists; the fold puts four lists' digit
rows on an m-tile and the chunk's tokens on K (`prepass_kernel.md#fold`), so its accumulators are
one an (m-tile, n-tile) and its operands come straight from the pool rows: the token's factors
`P_g(t)`, `P_gs(t)` sit in the sparse row's pad past the 128 digits, followed by eight zero bytes a
padding row reads.

<a id="state"></a>
## The state

A warp its 128 leaves x 2 columns as fp32, 1 KB, accumulated once a window from the IMMA
accumulators and read by the publish; dumped by canonical leaf when the call binds a state plane.

<a id="ring"></a>
## The read ring

A warp's two slots of a tile's read rows, 56 bytes a token: the dense sub-block's 32 entries, the
region's whole read-sparse row (8 or 16 digits; the warp's four are interleaved in it) and the two
read scales (8). The dense entries are copied by quads of 8-byte copies (one token's 32-byte sector a
quad, eight tokens an instruction), the sparse row by two 8-byte copies a lane, the scales a lane a
token; the stride puts eight consecutive tokens on distinct bank words, so the readout's A fragment
(a token a lane quad, four consecutive leaves a lane) loads conflict-free. The ring runs a tile ahead
of the tile being read.

<a id="transfer"></a>
## The transfer

The s16 digits (2 bytes a leaf a column) and each leaf's power-of-two scale as an s8 exponent
(1 byte a leaf a column), 6 KB in all.

<a id="transfer-old"></a>
## The transfer, as first planned

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
