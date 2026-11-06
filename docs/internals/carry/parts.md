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
`pre_of` reads a window's tenants back (`union_of` its tokens; `pre_of` forms its parity at each use, `#window`);
`chunk_cols` names a chunk's columns in the boundaries.

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

A chunk's fill issues its copies as one commit group; the caller waits the group (`carry_kernel.md#staging`).
`fill` copies a range of union slots, each its token's row -- the side's operand slices at 64 bytes each, then on the
write side its values at the box's channels -- swizzled, by `cp.async` over all 256 threads (a slice's 16-byte runs, 8
or 4 under 16, `stage_bytes`): a quad of lanes a slot, thread `4 q + r` slots `q + 64 k` of each 256. A trip of 256
slots is one block with no branch: its four tokens loaded on either side and selected (an unlisted side's slot is its
token), then every source (`row_at`: the slice's base plus the token's and the head's products, each one wide
multiply-add), then the copies (`copy16`, issued where the slot is under the trip's room in rows' bytes, its size zero
for a gap), so no copy's address register is rewritten while an earlier copy may still read it. A token load past the
chunk is not clamped: it reads the tenant after the union (the prefix rows), in bounds, and its copy is not issued. A
row's swizzle is its slot's low bits (`chunk_at`), the same at every `k` and every trip, so a thread's places are one
base and immediates. The first trip is straight-line; a chunk over 256 slots loops the rest. A read slot's row names
its query head (`carry.md#sides`): its source is the slice at that head, a head stride on (`Slices::sh`), as a 32-bit
offset from the slice's base at the window's first token (the host holds a window's token and head strides under 2^31
together). A GAP slot (`kGap`, [`carry.md`](carry.md#lists)) -- its token negative as 16 bits -- is zero-filled and
reads nothing. `read_fill` adds the chunk's panel and den words (`panel.md`), zeroed by 16-byte stores four a thread
(each predicated on the quanta left, the first four straight-line, so the chunk's landing waits on its rows alone), and
the chunk's entries; `fold_fill` (`fold.md#chunk`) the write chunk's rows and entries, or on a side with no list level
each slot's token's `(F, m)` in the entries' place (8-byte copies from the facts pass's `fm`). Each part sits at the
chunk's own slots (`read_panel`, `read_dens`, `read_ents`, `fold_ents`; `schedule.md#buffers`).

THE FILL'S FLOOR is its bytes through the SM's global path, not its instructions: a CTA-window at nl64k-alt-k4 copies
~51 KB (the read rows 13.8 KB, the fold rows 27.6 KB, the entries 4 KB, the staging ~5 KB), and a `cp.async.cg` stream
of L2 hits runs at ~18 bytes a cycle an SM ([`smem_rates`](../../lab/smem_rates.md), `common/hardware.json`'s line
price) -- ~2.85K cycles a window. The limit is the SM's, not the L2's: the grid's second wave (48 SMs busy) fills in the
first's cycles (80). The copies' issue backs up into the MIO queue (the copy's samples are half MIO throttle).

<a id="entries"></a>
## The entries

A buffer holds its chunk's entries as the group's lists' runs in list order: `chunk_runs` gives lane `l` list `l`'s
first entry of the chunk, its count, its count below the readout's token-group split (`group_split`,
[`readout.md#warps`](readout.md#warps): the prefix rows' count at the split's column) and its place (the group's lists'
counts before it) from its two prefix rows (list `l` and `l + 1`): the chunk's end in both, its start where the chunk
is not the first, each a difference within the chunk modulo 2^16, the same in every warp -- worked out once a chunk by
the kernel (`carry_kernel.md#kernel`) and passed to the chunk's fill and its readout or fold. `entries_fill` copies a warp's own lists' runs (`warp + 8
j`), eight lanes a run: lane `l` takes run `l >> 3`'s first, count and place by one shuffle each from `run` and copies
its entries `l & 7`, `+ 8`, ..., 8 bytes a lane (`ops::stage_run_lanes`: a lane past its run issues nothing), two a
trip, the trips the warp's longest run's (`__reduce_max_sync`), the first two straight-line. A consumer takes its
list's place and count from its own loads, so no warp waits on another for them.

<a id="maxima"></a>
## The digit maxima

The hand-off's first step (the f16 image at a power of two a digit). A lane's registers hold its warp's lists' state,
each list in its own running unit ([`fold.md#units`](fold.md#units)), so the lists meet on one scale through an integer
KEY: a largest |accumulator| at its list's unit as its fp32 bits plus (the list's unit less the box's top unit,
`box_top`) x 2^23 -- the exponent field shifted, the mantissa kept, a zero `INT_MIN` -- so the largest over the lists is
one integer max. A key is exact while the live units span under `kKeySpan` (a live unit lies between `kUnitMin` and
128; a never-live list's state is zero). `post_keys` posts each digit's key at the window's parity, at the digit's
place in lane order (`exp_slot`: digit `16 M + 8 H + g` at `4 g + 2 M + H`), in the hand-off's form (`#same-split`):

- THE ALTERNATING FORM (`post_row_keys`): a digit is a fold row, across every warp's lists. A digit row's largest over
  the lane's eight values a list (seven FMNMX), the bits joined across the row's four lanes by a quad reduce-scatter (a
  list's lanes share its unit, so the max commutes with the shift and a zero stays zero: the zero test and the shift
  run once a lane a row, on its own list `t`), the four lists' keys by two more shuffles, and one shared atomic max a
  lane.
- THE SAME SPLIT (`post_list_keys`): a digit is a fold list, every row of it its warp's. A list's largest over the
  lane's 32 values and the warp (`__reduce_max_sync`), keyed at its unit; lane `j` stores list `w + 8 j`'s key, the
  one writer, no atomic.

After the barrier `post_exps` turns lane `i`'s key into its digit's exponent: `(key + 0x7fffff) >> 23`, the exponent
field rounded up past a nonzero mantissa, plus the top less 127 -- exactly `ceil_exp` of the digit's largest |state| on
one scale -- at least `kImageTop - 127`, recorded in the warp's OWN ROW of `kExps` (`exps_of`; digit `d` at
`exp_slot(d)`) for its stores and its readout. Every warp records the same values and reads only its own: the readout
reads every digit's on every warp, and one shared row written by all eight was an unsynchronized write beside every
read (benign, the same values, but a racecheck record a byte). Warp 0 resets the other parity's keys for the next
window (a warp a window ahead posts into the other parity while a slower one still reads these). The image holds a digit at `2^(kImageTop - e)`: its largest
value at most 2^10, so a 32-list sum of image times read row (each at most 1) stays within 2^15, inside an fp16
accumulator, and one placement serves the fp32- and fp16-accumulate readouts. Measured (2026-10-12, the image's quality
model, per token against fp64 on the flagship cells): the placement at 2^10 and at 2^15 give the same error, as does a
block prescale; a leaf keeps 11 bits down to 24 octaves under its digit's largest. The form was labbed first
([`docs/lab/handoff.md`](../../lab/handoff.md), form 4).

<a id="image"></a>
## The image

The value-carve readout's (`readout.md`): digit `d`'s plane is a SLICE a readout channel group -- 32 lists x 8 channels
f16, 512 bytes (`kSliceBlock`) -- list `l`'s 16-byte row at `l ^ (d & 7)` (`slice_row`), the four slices a 2 KB plane.
The swizzle serves both sides: a hand-off store is a lane's channel pair (4 bytes) and a warp's 32 lanes are 8 lists'
rows x 4 words, distinct rows by `l ^ (d & 7)` -- 32 banks; a readout's `ldmatrix.x4.trans` reads 8 lists' rows of
one digit, their `l ^ (d & 7)` distinct -- 32 banks. A part's planes hold a digit at plane `d & 31` (the whole image) or
`d & 15` (a half: `schedule.md#planner`). The channel pair a store holds is channels `8 n + 2 t, + 1` of slice `n`, so
the readout's MMA output pairs a token's channels `(c, c + 1)` in natural order, the panel's and `num`'s
(`panel.md#words`).

THE DEN IMAGE (`kDenImage`, 2 KB): the den state at its digit's scale, digit `d` at half-word `2 (d & 7)` of block `d >>
3`'s list rows (`den_row`: 32 lists x 8 digits f16, list `l`'s row 16 bytes), so a readout's one `ldmatrix.x4.trans`
of the block gives the digit's column beside its block's (`readout.md#chunk`).

<a id="same-split"></a>
## The same split

The hand-off has two forms, the arm key's `same_split` (`Plan::kSameSplit`; `rola.ops.carry.same_split`, refused at the
seam when the call's levels disagree, `carry.md#refusals`): where the sides list different levels (the alternating box)
the fold's lists are the readout's K and its rows the digits; where both list the same level the fold's lists ARE the
readout's digits and its rows the K. A dense fold side takes its coefficients from the read side's list level
([`engine/carve.md#sides`](../engine/carve.md#sides)), so its box is the same split too. The image, the readout and
the den's readout are the same in both: only the keys (`#maxima`), the stores (`#store`) and the den's tables
(`#den`) read the fold's state the other way round. The readout reads every leaf for its channels, so the hand-off is
always the image -- at dense its stores are ~1.0K cycles of a ~70K window (`tools/design_floor.py`).

ARMS: `[32, 1, 32, 1, 32, 0]` the alternating box, `[32, 1, 32, 1, 32, 1]` both sides listing one level (`[SS, DD]`),
`[32, 2, 32, 1, 32, 1]` a dense write side over a listed read side (`[SD, DD]`).

<a id="store"></a>
## A batch's stores

`store_batches<P, Mask>`: every warp stores its state of the batches' digits (a batch 8 digits) into the image, each
value times `2^(u + kImageTop - e)` for its list's unit `u` and its digit's exponent `e`, packed f16, a lane's channel
pair of a slice row one 4-byte store at its held base and a compile-time offset (`Mask` picks the plane). Its frame
(`store_frame`): the base, opaque so its terms are formed at each call rather than hoisted across the windows; its four
exponents, one 16-byte load from its warp's row of `kExps`; its lists' units. The factor (`list_scale`) is two integer multiply-adds on
the FMA pipe and two clamps, its multiplier and bias loaded (`kBits`) so ptxas cannot turn them into ALU shifts; it is
held to a normal float's range (below it the f16 is zero; above it the list's accumulators are under 2^-117 of its
unit). Each batch's code is compiled and selected by a uniform test: the state's registers are indexed by the batch.

- THE ALTERNATING FORM (`store_row_batches`): batch `2 M + H` is rows `16 M + 8 H + g` of every list; list `w + 8 j`'s
  value goes to its digit's K row `(w ^ g) + 8 j` (digit `g`'s swizzle), the lane's four exponents its rows'.
- THE SAME SPLIT (`store_list_batches`): batch `j` is list `w + 8 j`, its plane the list's; row `16 M + 8 H + g` goes
  to K row `16 M + 8 H + (g ^ w)` (the list's swizzle, its low bits the warp), the lane's four exponents its lists'.

<a id="den"></a>
## The den's tables

The den -- each token's read product against the MASS of the windows before, the v+1 channel ([`fold.md#den`](fold.md#den))
-- is read out beside the channels (`readout.md#chunk`). THE DEN'S TABLES, a window's, from the den's registers
(`State::den`, `fold.md#state`): a window's read chunks run before its fold chunks, so the registers hold the state
before the window's writes, each word at its place (`den_place`: a fold list and a fold row). A word's readout digit
and K are its fold row and list in the alternating form, its list and row in the same split (`den_digit`, `den_k`).
Before the keys each digit's largest goes into `kDenMax` by an integer max (`den_tables_max`: a non-negative float's
bits order as integers; cleared before the window's top barrier); after the exponents each digit's FACTOR `2^(dE -
kImageTop)` (`kDenFactor`; `pow2_wide`) and the den image at `2^(kImageTop - dE)` (`den_tables_out`, four 16-bit stores
a lane). Every den term is ABSOLUTE: its coefficient times the digit's factor times the slice's products. While the
window folds the den image is free (its readers are done at the window's last drain), so a dense fold exchanges its
den tiles through it (`fold.md#dense`).

| buffer | written by | read by | shape | banks |
|---|---|---|---|---|
| `kDenMax` u32 `[32]` | `den_tables_max`: a lane its four words' digits, every warp, `red.shared.max` | `den_tables_out`: a lane its words' digits, lane `l` digit `l`'s factor | scalar | 16 digits two lanes each (once a window) |
| `kDenImage` f16 `[4][32][8]` | `den_tables_out`: a lane its four words, 16-bit | the readout: a digit's block, one `ldmatrix.x4.trans` | 16-bit / 16-byte rows | the stores once a window; the loads 8 rows of 16 bytes: one wavefront |

THE CROSS-CTA ADDS (`red_add_h2`, `red_add_f32`) and a token's row (`token_row`) are the drain's (`panel.md#drain`).
