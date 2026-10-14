# `carry/parts.cuh` — the carry's components

Mirrors `csrc/rola/src/carry/parts.cuh`. Each component is at its real signature, tenants and copies, and keeps the
work's counts; what a component leaves to its build is named in its section and in
`carry_kernel.md#kernel`. The fold is `fold.md`'s.

<a id="place"></a>
## The CTA's place

`place_of` is the CTA's batch-head (`blockIdx.y`) and its channel group (the box index's innermost digit);
`origin_of` a level's first box digit. Before the windows, thread 0 works out every address that is the same each
window into the `Addr` tenant (`addr_in`): each side's STATICS (its union group, the 32-digit group holding the box's
list digits; the box's first list digit; its lists; whether it lists) and its operand SLICES (each level's plane at
the CTA's batch and head, displaced to the box's first digit, its token stride and bytes), the values at the box's
channels, the channel group. The divisions run once: at nl64k-alt-k4 the per-fill address arithmetic they replaced was
a sixth of the carry's samples, and the carry ran 16% faster without it. The window loop reads the tenant at each use
(volatile), so none of it holds a register across the windows. `window_of` is a side's window: its tokens from the
window index, its statics from the tenant.

<a id="lists-in"></a>
## Lists in

A window ahead (`carry_kernel.md#staging`), into the window's parity of the tenants: a side's union tokens whole
(512 slots, 1 KB, 64 copies), its box lists' pass starts (one 16-byte copy a list) and its count (one 4-byte copy).
`facts_in` copies the write lists' exponent pairs (128 bytes) and the window's token units (512 bytes). `union_of`,
`at_of` and `slots_of` read a window's tenants back.

<a id="heads-in"></a>
## Heads in

A warp's own 512 bytes, 16 a lane: its 4 lists' (or digits') first `kHeadEntries` entries for the next window, copied
once the warp has read this window's (its readout's, or its fold's), so the heads take one buffer a side.

<a id="swizzle"></a>
## The pool rows' swizzle

A row of 128 bytes XORs its 16-byte chunk by the slot's residue (`slot & 7`); a row of an odd number of 64-byte quarters
(64 or 192 bytes) starts on alternating halves of the 128-byte bank line, so it XORs its chunk within each four by the
residue's high bits (`chunk_at`). Either way eight slots of distinct residues meet eight bank groups at one chunk: a
dense pass's step is 8 consecutive slots; a list's step is its next 8 rank slots, distinct residues only where they run
consecutively (the facts pass's [ranks](../facts/facts.md#ranks)). `chunk_addr` is a logical
chunk's address from a row's chunk 0 with the swizzle folded in: an XOR inside the swizzle's group, the group's offset
added. An entry carries the row's chunk-0 offset, so a lane addresses its chunk with one XOR.

<a id="fill"></a>
## The fill

A ring fill: a range of union slots, each its token's row -- the side's operand slices at 64 bytes each, then on the
write side its values at the box's channels -- swizzled, by `cp.async` over all 256 threads (a slice's 16-byte runs, 8
or 4 under 16, `stage_bytes`), each thread arriving on the slot's barrier when its copies land
(`cp.async.mbarrier.arrive.noinc`). A GAP slot (its token `kGap`, [`carry.md`](carry.md#lists)) is zero-filled: no list
entry names it but a pad may, whose coefficient 0 then multiplies zeros, and its fill reads nothing.

<a id="maxima"></a>
## The digit maxima

The hand-off's first step (the f16 image at a power of two a digit). A lane's registers hold four digits of each
of its warp's lists, each list in its own running unit ([`fold.md#units`](fold.md#units)), so the lists meet on one
scale by exponent: `post_maxima` takes, a digit, each list's largest |accumulator|'s `ceil_exp` plus the list's unit,
the largest over the lists (exactly the exponent of the digit's largest |state| on one scale, whatever the units'
spread), joins the digit row's four lanes by two shuffles and posts it (128 FMNMX, 16 exponent steps and 8 SHFL a lane
a window). `digit_exp` is a digit's exponent over the 8 warps' posts: the smallest `e` with every value at most `2^e`,
at least `kImageTop - 127`. The image holds a digit at `2^(kImageTop - e)`: its largest value at most 2^10, so a
32-list sum of image times read row (each at most 1) stays within 2^15, inside an fp16 accumulator, and one placement
serves the fp32- and fp16-accumulate readouts. Measured (2026-10-12, the image's quality model, per token against fp64
on the flagship cells): the placement at 2^10 and at 2^15 give the same error, as does a block prescale; a leaf keeps 11
bits down to 24 octaves under its digit's largest.

<a id="image"></a>
## The image

A digit's plane is its 32 lists x 32 channels at f16, 2 KB, its 16-byte units at `unit_of(l, k, d)`: the list's four
chunks, the chunk XORed by the list's second and third bits, the unit XORed by the digit's residue. A part's planes
hold its digits (`d % kPlanes`). The fold's stores (a lane a list row's 4-byte word) and the readout's `ldmatrix.trans`
(8 lists' 16-byte chunks) are both conflict-free on it.

<a id="store"></a>
## A part's stores

`store_part<P, Part>`: every warp stores its 4 lists' rows of the part's digits (whole m-tiles), each value times
`2^(u + kImageTop - e)` for its list's unit `u` and its digit's exponent `e`, packed f16 (32 stores a lane a part on
sm_86); the factor is held to a normal float's range (below it the f16 is zero; above it the list's accumulators are
under 2^-117 of its unit). Warp 0 records the part's exponents for the readers (`kExps`). The part is compile-time: the
state's registers are indexed by it.

<a id="pair"></a>
## The exact pair

`s16_pair`: two outputs times the coefficient to the integer `hi x 2^16 + lo` on the FMA pipe (two FFMAs with the magic
offsets, one LEA), exact under u32 adds while each final fits s16.

<a id="readout"></a>
## The readout

THE OUTPUT: each token's partials at its unit, `P(t, c) x unit(t) = sum_{d, r} x(t, d) a(t, r) S(d, r, c) / kappa`
over the CTA's box -- `S` the state before the token's window, `x` the read coefficient of the image's digit level as
stored (a sparse level's byte / 256, a dense level's f16), `a` the read row of the list level as stored (f16), `kappa`
the state's byte scale (`kByteScale`); `unit(t) = 2^u(t)`, `u` the den pass's unit byte. The read side's per-token
scales and `kappa` are common to `num` and `den` and cancel in the epilogue. Gated per slot against
`tests/oracle/readout_reference.py` (`tests/oracle/test_carry_readout.py`): on the even box over four windows, every
slot within half its bound in fp32 (~4e-4 relative, the image's rounding) and a fifth of it in fp16. The dense-dense
arm's branch waits on its fold's reference (the fold, build order 5).

`readout_part`, a warp's: each of its digits in the part, its state fragments from the image (the A operand: 32 channels
x 32 lists, four `ldmatrix.trans`) and its exponent, then its tiles over the pass. A TILE is 8 tokens on N: their read
rows by one `ldmatrix.x4` (B: 32 lists x 8 tokens), four MMAs accumulating in the device's form (fp16 on GeForce, fp32
where fp32 accumulate runs at full rate:
[`carry_api.md#carry-readout-accumulate`](carry_api.md#carry-readout-accumulate); the fp16 accumulator converted to fp32
for the pairs), each lane's two tokens' outputs as exact pairs added by `red.shared` at their panel words. Lane `l`
holds entry `l & 7` (its token's row, so the B load needs no shuffle) and scales its coefficient to the token's unit: `c
2^(e - kImageTop - kCoefTop - u)`, the exponent added to the fp32 coefficient's bits (`e` the digit's exponent, `u` the
token's unit byte, looked up by the token's window place the entry carries); each lane takes its two tokens' scales and
panel offsets by four shuffles. A listed side's tiles are pass 0's first two from the warp's heads, then the rest in
place; a side with no list level takes every slot of the pass, its entry implied -- the slot's row, its token (the
pass's first plus the slot), its panel row (a slot past the pass's end its place's dummy row), its coefficient the
digit's f16 in the row's coefficient slice (its second). Each source is its own loop, so each step is one block
(KERNEL_STANDARDS §27), guarded by a vote and closed by one.


The PANEL is the pass's slots' 64-byte rows after 8 dummy rows (`kPadBytes`, one a residue), laid out in pairs: slots
`2k` and `2k + 1` share a 128-byte line, each slot's two 32-byte halves at the segments its pair's residue bits XOR
(`csrc/rola/src/carry/carry.cuh`'s `panel_row`). A panel add touches 4 tokens' 8 words at one half; a tile of 8
distinct residues (a dense pass's consecutive slots) meets 32 banks an add. Measured by the bank model (2026-10-12):
1.00 wavefronts an add on dense tiles against 4.00 for plain 64-byte rows (every add's 4 tokens on one half), and 2.1
against 2.75 on a list's rank slots; the drain 1.00 a request either way. An entry's panel offset is from
the dummy rows, so it fits its 16 bits on every card; a lane XORs its half (`32 m`) and adds its word (`4 g`).


<a id="drain"></a>
## The drain

A pass's panel to `num`: each slot's 16 words added as u32 atomics at its token (two tokens a warp request, a slot
pair's one line), each word cleared for the next pass. A thread keeps one word of every 16th slot (its word's panel
offset and its output column worked out once) and takes four slots a step, loaded and cleared before their adds, so the
step's loads and atomics overlap; a gap slot's word is zero and adds nothing. CTAs join only here (R22: independent
units, joined by atomics). The panel's word pairs channels (c, c + 8); the output's format follows it. At nl64k-alt-k4
the drain is ~3.7K cycles a warp-window, near the chip's global-atomic rate (~0.15 chip clocks a warp request) when
every SM drains at once: what remains of it is hidden under compute or not at all.
