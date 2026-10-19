# `carry/parts.cuh` — the carry's components

Mirrors `csrc/rola/src/carry/parts.cuh`. Each component is at its real signature, tenants and copies, and keeps the
work's counts; what a component leaves to its build is named in its section and in
`carry_kernel.md#kernel`. The fold is `fold.md`'s.

<a id="place"></a>
## The CTA's place

`place_of` is the CTA's batch and write head (`blockIdx.y`) and its channel group (the box index's innermost digit);
`origin_of` a level's first box digit. Before the windows, thread 0 works out every address that is the same each
window into the `Addr` tenant (`addr_in`): each side's STATICS (its union group, the 32-digit group holding the box's
list digits; the box's first list digit; its lists; whether it lists) and its operand SLICES (each level's plane at
the CTA's batch and head, displaced to the box's first digit, its token stride and bytes), the values at the box's
channels, the channel group. The divisions run once: at nl64k-alt-k4 the per-fill address arithmetic they replaced was
a sixth of the carry's samples, and the carry ran 16% faster without it. The window loop reads the tenant at each use
(volatile), so none of it holds a register across the windows. `window_of` is a side's window: its tokens from the
window index, its statics from the tenant.

<a id="window"></a>
`ops::opaque` (`../common/ops.md#opaque`) and `parity_of` pass the window index (and a thread's) through an opaque
move, so a window-invariant offset --
a window's first token, a list's row base, a tenant by parity, the drain's cell -- is formed at each use: computed once
before a window's steps, the compiler holds it in a register across the readout, the kernel's register peak, and at
255 a thread one such value spills. The fill's quad lane and first slot come from the thread index the same way, and
the kernel passes each part its warp and lane opaque (`carry_kernel.md#kernel`).

<a id="lists-in"></a>
## Lists in

A window ahead (`carry_kernel.md#staging`), into the window's parity of the tenants -- the read side's in the call's
read tables (`plan.md#read-tables`): a side's union tokens whole (512 slots a query head, 1 KB), its group's boundaries
in prefix form (33 rows of `cols_of` columns, 8 a query head) and its count (one 4-byte copy), its copies over the CTA's
threads; the planner alone reads the exact totals. `facts_in` copies the write lists' exponent pairs (128 bytes) and
each query head's token units (512 bytes a head; head `j` of write head `bh` is the call's `bh g + j`). `union_of`,
`pre_of` and `slots_of` read a window's tenants back (`pre_of` forms its parity at each use, `#window`); `chunk_cols`
names a chunk's columns in the boundaries.

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

A chunk's fill issues its copies; the caller arrives on the buffer's landed barrier (`carry_kernel.md#barriers`).
`fill` copies a range of union slots, each its token's row -- the side's operand slices at 64 bytes each, then on the
write side its values at the box's channels -- swizzled, by `cp.async` over all 256 threads (a slice's 16-byte runs, 8
or 4 under 16, `stage_bytes`). A read slot's row names its query head (`carry.md#sides`): its source is the slice at
that head, a head stride on (`Slices::sh`), as a 32-bit offset from the slice's base at the window's first token (the
host holds a window's token and head strides under 2^31 together). A GAP slot (`kGap`, [`carry.md`](carry.md#lists)) is zero-filled and reads
nothing. `read_fill` adds the read buffer's dummy panel rows and its panel's whole 128-byte lines, zero-filled (a zero
quantum reads nothing), and the chunk's entries; `fold_fill` (`fold.md#chunk`) the write chunk's rows and entries. Each
part sits at the chunk's own slots (`read_pads`, `read_ents`, `fold_ents`; `schedule.md#buffers`).

<a id="entries"></a>
## The entries

A buffer holds its chunk's entries as the group's lists' runs in list order: `chunk_runs` gives lane `l` list `l`'s
first entry of the chunk, its count and its place (the group's lists' counts before it) by four loads of the prefix
rows (`pre_at`), each a difference within the chunk modulo 2^16, the same in every warp; `entries_fill` copies a warp's
own lists' runs (`warp + 8 j`), 8 bytes a lane, as ONE stream: entry `e` of the four runs in turn finds its run by
three compares against the runs' firsts and takes that run's source and place as offsets from `e`, so every full trip
of 32 copies unpredicated and only the stream's tail branches, once a warp-chunk. Copied run by run, each run paid its
shuffles, a convergence region around its loop and its predicated copy and 64-bit address arithmetic -- ~40
instructions a run, most runs one copy (2026-10-19: 2,559 -> 1,392 instructions a CTA-window at nl64k-alt-k4, the
fill phase 5,401 -> 4,772 cycles a warp-window, a call 2.7% faster there and at g 4, cohort-k16's long runs unchanged).
A consumer takes its list's place and count from its own loads, so no warp waits on another for them.

<a id="maxima"></a>
## The digit maxima

The hand-off's first step (the f16 image at a power of two a digit). A lane's registers hold four digits of each
of its warp's lists, each list in its own running unit ([`fold.md#units`](fold.md#units)), so the lists meet on one
scale through an integer KEY: `post_keys` takes, a digit row, each list's largest |accumulator| (seven FMNMX from the
row's eight values), its fp32 bits plus (the list's unit less the box's top unit, `box_top`) x 2^23 -- the exponent
field shifted, the mantissa kept, a zero `INT_MIN` -- so the largest over the lists is one integer max a list; the
row's four lanes are joined by two shuffles and each lane posts one of its quad's four rows by one shared atomic max,
at the window's parity in lane order (digit `16 M + 8 H + g` at `4 g + 2 M + H`). A key is exact while the live units
span under `kKeySpan` (a live unit lies between `kUnitMin` and 128; a never-live list's state is zero). After the
barrier `post_exps` turns lane `i`'s key into its digit's exponent: `(key + 0x7fffff) >> 23`, the exponent field
rounded up past a nonzero mantissa, plus the top less 127 -- exactly `ceil_exp` of the digit's largest |state| on one
scale -- at least `kImageTop - 127`, recorded at `kExps` in lane order (digit `d` at `exp_slot(d)`) for the stores and
the readers, and the other parity's keys reset for the next window (a warp a window ahead posts into the other
parity while a slower one still reads these). Every warp records the same values. The image holds a digit at
`2^(kImageTop - e)`: its largest value at most 2^10, so a 32-list sum of image times read row (each at most 1) stays
within 2^15, inside an fp16 accumulator, and one placement serves the fp32- and fp16-accumulate readouts. Measured
(2026-10-12, the image's quality model, per token against fp64 on the flagship cells): the placement at 2^10 and at
2^15 give the same error, as does a block prescale; a leaf keeps 11 bits down to 24 octaves under its digit's
largest. The form was labbed first ([`docs/lab/handoff.md`](../../lab/handoff.md), form 4).

<a id="image"></a>
## The image

A digit's plane is its 32 lists x 32 channels at f16, 2 KB, its 16-byte units at `unit_of(l, k, d)`: the list's four
chunks, the chunk XORed by the list's second and third bits, the unit XORed by `digit_swizzle(d)` (the digit's low bit
moved to bit 2). Chunk `k` holds a writer lane's channels, `8 n + 2 k + b` at position `2 n + b` -- the four channel
tiles' pairs of the lane `k` of a fold quad -- so a lane's row of a list is one 16-byte store. A part's planes hold a
digit at plane `d & 31` (the whole image) or `d & 15` (a half). The fold's stores (a quarter-warp two digits' four
chunks: eight bank groups) and the readout's `ldmatrix.trans` (8 lists' 16-byte chunks) are both conflict-free on it.
The readout's m-tile of chunks `2 m, 2 m + 1` then pairs channels `(c, c + 2)` in an output word, `c = 8 (g >> 1) + 4 m
+ (g & 1)` for its row `g` ([`#drain`](#drain)).

<a id="store"></a>
## A batch's stores

`store_batches<P, Mask>`: every warp stores its 4 lists' rows of the batches' digits (a batch half an m-tile of each
list: 8 digits), each value times `2^(u + kImageTop - e)` for its list's unit `u` and its digit's exponent `e`, packed
f16, a lane's row of a list one 16-byte store (4 a lane a batch). A lane's unit for list `w + 8 j` is its unit for list
`w`, 32 units on (`units_step_by_list`), so a lane's stores are one held base and compile-time offsets (`Mask` picks
the plane); its four exponents are one 16-byte load from `kExps`. The factor (`list_scale`) is two integer
multiply-adds on the FMA pipe and two clamps, its multiplier and bias loaded (`kBits`) so ptxas cannot turn them into
ALU shifts; it is held to a normal float's range (below it the f16 is zero; above it the list's accumulators are under
2^-117 of its unit). Each batch's code is compiled and selected by a uniform test: the state's registers are indexed by
the batch.

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

`read_chunk`, a warp's: each of its digits in the batches, its state fragments from its image plane (the A operand: 32
channels x 32 lists, four `ldmatrix.trans`) and its exponent (`exp_slot`), then its tiles over the chunk. A TILE is 8 tokens on N: their read
rows by one `ldmatrix.x4` (B: 32 lists x 8 tokens), four MMAs accumulating in the device's form (fp16 on GeForce, fp32
where fp32 accumulate runs at full rate:
[`carry_api.md#carry-readout-accumulate`](carry_api.md#carry-readout-accumulate); the fp16 accumulator converted to fp32
for the pairs), each lane's two tokens' outputs as exact pairs added by `red.shared` at their panel words. Lane `l`
holds entry `l & 7` (its token's row, so the B load needs no shuffle) and scales its coefficient to the token's unit: `c
2^(e - kImageTop - kCoefTop - u)`, the exponent added to the fp32 coefficient's bits (`e` the digit's exponent, `u` the
token's unit byte, looked up by the row the entry carries -- its query head's units, `carry.md#sides`); each lane
takes its two tokens' scales and panel offsets by four shuffles. A listed side's tiles take the digit's run from the
buffer's entries (`#entries`); the entries' offsets are the union's, so the chunk's pool is taken back by its first
slot's (`first x row` bytes) and each entry's panel offset made the chunk's by subtracting `64 first` from its high half,
modulo 2^16 as a grouped union's offsets are written, and a lane past the run holds `masked_entry` -- its place's row, coefficient 0, a dummy panel row --
so its products add an exact zero. A side with no list level takes every slot of the chunk, its entry implied -- the
slot's row, its token (the chunk's first plus the slot), its panel row (a slot past the chunk's end its place's dummy
row), its coefficient the digit's f16 in the row's coefficient slice (its second). Each source is its own loop, so each
step is one block (KERNEL_STANDARDS §27), guarded by a vote and closed by one. Where a warp reads two digits a batch
(the image whole or in halves) a listed side steps them together: a step is both digits' tiles, their operands
(`tile_in`), then their MMAs (`tile_mma`), then their adds (`tile_add`) -- two independent chains in one block, each
entry loaded at a clamped place (`masked_entry` past its run), so nothing branches. Measured at nl64k-alt-k4: the
readout 4.0K to 3.6K cycles a warp-window.

A SIDE WITH NO LIST LEVEL reads its two digits as ONE GROUP (the unified readout's output scaling at m = 2, by
structure: every token is in every digit's list, so the group's union is each digit's own and grouping adds no MMAs): a
tile loads its 8 rows once (`tile_in_m`), takes each digit's coefficient from the row and scales it to the token's unit,
runs 4 MMAs a digit against its own state, and leaves by one exit (`tile_add_m`): each digit's exact pair formed as
alone (`s16_pair`) and the pairs summed as u32 words before one `red.shared` a token and channel pair -- the panel's
words bit for bit the per-digit form's. NOT AN FFMA CHAIN: a pair is exact while its products lie within 2^22 of the
pair's offset, a bound on each digit's products, not on their sum (the s16 bound is the final sum's over 32 digits);
chaining two digits' products onto one offset left that range on production's dense cells (most of `num` wrong,
2026-10-19), though the lab's inputs (token units 0-2) never reached it. Measured (sm_86 locked, against the per-digit
form, `num` bit-identical): the dense readout 96.5K -> 67.0K cycles a warp-window at nl64k-dense, its step loop's cover
42% -> 67-70%; a call nl64k-dense 28,795 -> 24,176 us, nl16k-dense 3,639 -> 3,062, flagship-dense 492 -> 419.


The PANEL is the chunk's slots' 64-byte rows after 8 dummy rows (`kPadBytes`, one a residue), laid out in pairs: slots
`2k` and `2k + 1` share a 128-byte line, each slot's two 32-byte halves at the segments its pair's residue bits XOR
(`csrc/rola/src/carry/carry.cuh`'s `panel_row`). A panel add touches 4 tokens' 8 words at one half; a tile of 8
distinct residues (a dense chunk's consecutive slots) meets 32 banks an add. Measured by the bank model (2026-10-12):
1.00 wavefronts an add on dense tiles against 4.00 for plain 64-byte rows (every add's 4 tokens on one half), and 2.1
against 2.75 on a list's rank slots; the drain 1.00 a request either way. An entry's panel offset is from
the dummy rows, so it fits its 16 bits on every card; a lane XORs its half (`32 m`) and adds its word (`4 g`).


<a id="drain"></a>
## The drain

A chunk's panel to `num`, once every warp has read the chunk: each slot's 16 words added as u32 atomics at its row's
query head and token (two tokens a warp request, a slot pair's one line); the next fill zeroes the panel. A thread
keeps one word of every 16th slot -- 16 slots on, its panel cell is 1 KB on, the swizzle's bits unchanged, so its
cells are one address and an add -- and takes two slots a step, their words and union tokens loaded unguarded (the
union array and the panel lie in shared memory past the chunk's slots) before their adds. An add is never predicated:
ptxas puts a predicated `red.global` in a branch and a convergence region (BSSY, BRA, BSYNC around each add), so a
step's second slot past the chunk adds zero at the thread's first slot's row -- a row of the CTA's own, never one every
CTA shares (lab [`drain`](../../lab/drain.md): a shared dummy line serialized every CTA 6x). A slot's row is a 32-bit
word offset, one widening multiply-add from `num`'s base (the host holds a write head's rows under 2^32 words). CTAs
join only here (R22: independent units, joined by atomics). The panel's word `8 m + g` pairs channels `(c, c + 2)`,
`c = 8 (g >> 1) + 4 m + (g & 1)` ([`#image`](#image)); the output's format follows it
(`tests/oracle/readout_reference.py`'s `unpack_num`).

Its floor is the SM's global request path, ~5 cycles a line request with every SM draining (the lab). Measured
2026-10-19 (sm_86 locked, a warp-window): nl64k-alt-k4 2,135 -> 1,389 cycles (216 requests, ~1.1K at the floor), g 4
7,842 -> 5,175, the predicated form's 59 instructions a two-slot step (a guarded token load, 64-bit address
arithmetic, a convergence region an add) against this one's; a call 3.0-4.4% faster on every listed cell. Taken in
slices between the fold's lists it serialized with them rather than overlapping (2026-10-13).
