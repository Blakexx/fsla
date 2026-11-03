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
reads nothing. `read_fill` adds the read buffer's dummy panel rows and its panel's whole 128-byte lines, zeroed by
stores four a thread (each predicated on the quanta left, the first four straight-line, so the chunk's landing waits on
its rows alone), and the chunk's entries; `fold_fill` (`fold.md#chunk`) the write chunk's rows and entries, or on a side
with no list level each slot's token's `(F, m)` in the entries' place (8-byte copies from the facts pass's `fm`). Each
part sits at the chunk's own slots (`read_pads`, `read_ents`, `fold_ents`; `schedule.md#buffers`).

THE FILL'S FLOOR is its bytes through the SM's global path, not its instructions: a CTA-window at nl64k-alt-k4 copies
~51 KB (the read rows 13.8 KB, the fold rows 27.6 KB, the entries 4 KB, the staging ~5 KB), and a `cp.async.cg` stream
of L2 hits runs at ~18 bytes a cycle an SM ([`smem_rates`](../../lab/smem_rates.md), `common/hardware.json`'s line
price) -- ~2.85K cycles a window. The limit is the SM's, not the L2's: the grid's second wave (48 SMs busy) fills in the
first's cycles (80). The copies' issue backs up into the MIO queue (the copy's samples are half MIO throttle).

<a id="entries"></a>
## The entries

A buffer holds its chunk's entries as the group's lists' runs in list order: `chunk_runs` gives lane `l` list `l`'s
first entry of the chunk, its count and its place (the group's lists' counts before it) from its two prefix rows (list
`l` and `l + 1`): the chunk's end in both, its start where the chunk is not the first, each a difference within the
chunk modulo 2^16, the same in every warp -- worked out once a chunk by the kernel (`carry_kernel.md#kernel`) and
passed to the chunk's fill, its units and its tiles or fold. `entries_fill` copies a warp's own lists' runs (`warp + 8
j`), eight lanes a run: lane `l` takes run `l >> 3`'s first, count and place by one shuffle each from `run` and copies
its entries `l & 7`, `+ 8`, ..., 8 bytes a lane (`ops::stage_run_lanes`: a lane past its run issues nothing), two a
trip, the trips the warp's longest run's (`__reduce_max_sync`), the first two straight-line. A consumer takes its
list's place and count from its own loads, so no warp waits on another for them.

<a id="maxima"></a>
## The digit maxima

The hand-off's first step (the f16 image at a power of two a digit). A lane's registers hold four digits of each
of its warp's lists, each list in its own running unit ([`fold.md#units`](fold.md#units)), so the lists meet on one
scale through an integer KEY: a list's largest |accumulator| at a digit row, its fp32 bits plus (the list's unit
less the box's top unit, `box_top`) x 2^23 -- the exponent field shifted, the mantissa kept, a zero `INT_MIN` -- so
the largest over the lists is one integer max a list. `post_keys` takes, a digit row, each list's largest over the
lane's eight values (seven FMNMX), joins those bits across the row's four lanes by the leaves' reduce-scatter
([below](#leaves)) -- a list's lanes share its unit, so the max commutes with the shift and a zero stays zero, and
the zero test and the shift run once a lane a row, on its own list, not once a list -- and the four lists' keys by
two more shuffles; each lane posts one of its quad's four rows by one shared atomic max, at the window's parity in
lane order (digit `16 M + 8 H + g` at `4 g + 2 M + H`). A key is exact while the live units
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
+ (g & 1)` for its row `g` ([`#drain`](#drain)). A warp reads digits `warp + 8 i`, whose swizzle depends on `d & 7`
only, so a lane's four `ldmatrix.trans` units of every digit it reads are one base (`image_lane(lane, warp)`, formed
once a chunk), the base `+ 1024` and both `^ 32` -- the base lies under 1024, so the add never carries
(`image_units_step` asserts it over every lane, digit and unit): two adds a digit where the units were four XORs and
four adds, and the lane's units were four `unit_of`s a chunk.

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
the state's byte scale (`kByteScale`); `unit(t) = 2^u(t)`, `u` the token's unit byte IN THIS CTA ([`#units`](#units)):
the power of two that holds the CTA's bound on the token's partial in 16 bits. The read side's per-token
scales and `kappa` are common to `num` and `den` and cancel in the epilogue. Gated per slot against
`tests/oracle/readout_reference.py` (`tests/oracle/test_carry_readout.py`): on the even box over four windows, every
slot within half its bound in fp32 (~4e-4 relative, the image's rounding) and a fifth of it in fp16. The dense-dense
arm's branch waits on its fold's reference (the fold, build order 5).

`read_chunk`, a warp's: each of its digits in the batches, its state fragments from its image plane (the A operand: 32
channels x 32 lists, four `ldmatrix.trans`) and its exponent (`exp_slot`), then its tiles over the chunk. A TILE is 8 tokens on N: their read
rows by one `ldmatrix.x4` (B: 32 lists x 8 tokens), four MMAs accumulating in the device's form (fp16 on GeForce, fp32
where fp32 accumulate runs at full rate:
[`carry_api.md#carry-readout-accumulate`](carry_api.md#carry-readout-accumulate); the fp16 accumulator converted to fp32
for the pairs), each lane's two tokens' outputs as exact pairs added by `red.shared` at their panel words. The panel
is FIXED POINT because shared memory's native atomic add is 32-bit integer on sm_80/86 -- a float add there compiles
to a compare-and-swap loop -- which is what each token's unit (below) and the units pass exist for; a float shared
panel is not a design option (event:2026-11-01-p87-no-floating-point-atomics-in-shared-memory-the-r). Lane `l`
holds entry `l & 7` (its token's row, so the B load needs no shuffle) and scales its coefficient to the token's unit: `c
2^(e - kImageTop - kCoefTop - u)`, the exponent added to the fp32 coefficient's bits (`e` the digit's exponent, `u` the
token's unit byte, looked up by the row the entry carries -- its query head's units, `carry.md#sides`); each lane
takes its two tokens' scales and panel offsets by four shuffles. A listed side's tiles take the digit's run from the
buffer's entries (`#entries`); the entries' offsets are the union's, so the chunk's pool is taken back by its first
slot's (`first x row` bytes) and each entry's panel offset made the chunk's by subtracting `64 first` from its high half,
modulo 2^16 as a grouped union's offsets are written, and a lane past the run holds `masked_entry` -- its place's row, coefficient 0, a dummy panel row --
so its products add an exact zero. A side with no list level takes every slot of the chunk, its entry implied -- the
slot's row, its token (the chunk's first plus the slot), its panel row (a slot past the chunk's end its place's dummy
row), its coefficient the digit's f16 in the row's coefficient slice (its second) -- each stepped by a constant a tile
(a slot 8 on: its row 8 rows on, both swizzles keeping `s & 7` and `(s >> 1) & 3`; its panel row 512 bytes on;
`slot_step_is_tile_rows` asserts both), so a tile's addressing is four adds and a select, not the row and panel
functions again. The dense readout is bound by its math pipes (the tensor pipe and the FMA pipe's conversions, pair
FFMAs and IMADs, near equal) and issue, so its time moves with ptxas's interleaving: work taken off a tile shows at
dense (-8% of its readout, -3.6% of the carry at 64K x 4 heads). Each source is its own loop, so each
step is one block (KERNEL_STANDARDS §27), guarded by a vote and closed by one. Where a warp reads two digits a batch
(the image whole or in halves) a listed side steps them together: a step is both digits' tiles, their operands
(`tile_in`), then their MMAs (`tile_mma`), then their adds (`tile_add`) -- two independent chains in one block, each
entry loaded unclamped (a place past the digit's run is the next run's, or the shared memory above the buffer: the read
tables) and replaced past its run by `masked_entry`, so nothing branches; `tile_add` takes the dummy rows' address with
the lane's word added once a call, a panel row's second channel half the first's XOR 32 (rows 32-byte aligned, the word
under 32). The readout is bound by the shared-memory pipe, not the tensor pipe (lab
[`readout_alone`](../../lab/readout_alone.md)).

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


<a id="leaves"></a>
## The leaves' maxima

Beside the digit keys, `post_keys` keeps each LEAF's own key -- a list's largest |value| over the CTA's 32 channels at
a digit row -- by a quad reduce-scatter of the bits (lane `t` ends with list `t`'s leaf of each of its four rows: two
shuffles across `xor 2`, one across `xor 1`), keyed at the lane's list's unit, and stores it as an exponent byte (`leaf_byte`: the smallest `e` with the leaf's
largest |value| at most `2^(e + top)`, -128 a zero leaf) at `kLeafExp`, digit-major. After the exponents,
`leaf_scales` turns the bytes into each leaf's f16 scale `2^(e_leaf - e_digit)` at `kLeafScale` (a digit's 32 lists
64 bytes; at least 2^-14, rounded up so still a bound; 0 for a zero leaf), a thread four leaves. Why leaves and not
digits: a token's partial is bounded by the sum over its read leaves of weight x leaf maximum; a digit's maximum in
their place fails the concentrated distributions at 16 bits (modeled 2026-10-19 and 2026-10-23: p99 3-7%), as do
8-leaf groups and the box's top; a leaf's maximum rounded to a power of two reads the same as its exact value.

<a id="units"></a>
## The units

Each CTA sets its OWN unit a token before its first panel add, from its leaves' maxima: the token's bound in the CTA,
`B_cta(t) = sum_d x(t, d) sum_r a(t, r) M(d, r)` over the box, `M` the leaf scales, and `u(t)` the smallest byte with
`1.02 B_cta(t) <= 2^(u + 15)` (`unit_byte`; `kUnitByteNone` where the token reads nothing). The 1.02 covers the
readout MMA's own overshoot (at most 2^-10 with fp16 accumulate) and the pieces' rounding. The bound's arithmetic is
raised by `kF16BoundMargin` (1 + 1/64): a listed bound's dot product runs as two packed f16 chains of `kBoundChain` = 8
fused products, each product and partial rounded by at most 2^-11, and the assertion holds the margin to the chain's
length. The units are written to `units_at` (a byte a query head's token) and the drain's factor beside them.

A LISTED read side (`units_listed_add`, then `units_close` after a barrier): a thread an entry across the CTA -- the
chunk's runs are contiguous in digit order, so an entry's digit is the last run starting at or before it (five
shuffles); its row against its digit's 64-byte scale row (8 `ld.shared.v4`, 16 HFMA2 in two chains), times its
coefficient and `2^(e_d - kCoefTop)`, quantized up at 2^-24 of the window's reference `2^(emax + 5)` and added into
the token's counter by `red.shared.add.u32`; after the barrier each counter becomes its token's byte (counters in
quanta of `2^(emax - 19)`: a token more than ~2^19 below the box's hottest digit rounds up to one quantum -- sound,
coarse). A DENSE read side (`units_dense`): every token reads every leaf, so the bound is 4 f16-accumulate MMAs a
tile of 8 tokens (A the leaf-scale table, 32 digits x 32 lists; B the tokens' rows as the readout loads them), each
digit's result times its coefficient (from the row's coefficient slice at a lane-constant offset) and `2^(e_d -
kCoefTop)`, summed over the digits by three shuffles, two tiles a warp in flight; a slot past the chunk stores to the
dead word. `bounds_clear` resets the counters each window. The pass runs between the chunk's landing and its first
tile, with its barriers; its cost is placement, not arithmetic (measured 2026-10-22: +2.0K cycles a warp-window at
nl64k-alt-k4 as a per-entry chain, unchanged by three placements); the ruled line is +6-9% a listed call by the dense
side's MMA form for listed sides (2026-10-24).

Measured (2026-10-20, sm_86, per token against fp64): fp32 across CTAs 0.005-0.028%, f16 pairs 0.037-0.107%
(corner-concentrated p99 0.33% / 0.36%), the per-slot gate passed on every runnable cell, worst 0.45 of its allowance;
HEAD's one unit a token for every CTA read 0.13-0.34%. Cost at nl64k-alt-k4: +18.5% a call (the bound pass +2.0K, the
leaf maxima +0.8K, the drain's conversion +0.8K cycles a warp-window), dense +5.5%.

<a id="den"></a>
## The den row

The den -- each token's read product against the MASS of the windows before, the v+1 channel ([`fold.md#den`](fold.md#den))
-- is read out beside the channels, in the readout's own tiles (ruled 2026-11-02: inline, not a pass; rebuilt to its
work the same day). THE DEN'S TABLES, a window's, from the den's registers (`State::den`, `fold.md#state`; ruled
2026-11-02: the den as one more channel of the state): a window's read chunks run before its fold chunks, so the
registers hold the state before the window's writes, each word at its place (`den_place`, `fold.md#state`). After the leaves' scales each digit's largest into `kDenMax` by an integer max (`den_tables_max`: a
lane's four words at their digits; a non-negative float's bits order as integers; cleared before the window's top
barrier); and at the first read chunk's units, past the barrier after the leaves'
scales (the rows lie over their exponent bytes, dead until the next window's keys), each digit's FACTOR `2^(dE -
kImageTop)` (`kDenFactor`; `pow2_wide`) and its DEN ROW at `2^(kImageTop - dE)` as f16 B words (`kDenRows`, 64 bytes a
digit, a list pair `p` at word `2 (p & 7) + (p >> 3)`; `den_tables_out`, four 16-bit stores a lane). A tile's den is
then one 8-byte load (`den_b`: the lane's two words, 8 bytes on by `lane & 7`) and two m16n8k8 MMAs at the channels'
accumulation (`den_mma`; fp16 on the f16 arm, `hmma_k8_f16`): the tile's own read rows, already loaded as the
channels' B, are A -- lists `0..7` its rows, `8..15` its rows 8 on, then `16..` and `24..` -- against the den row as
B's columns 0 and 1, so token `r`'s den is `D[r][0] + D[r + 8][1]` on lane `4 r`; one shuffle gives each lane its own
entry's (`lane & 7`). Times the entry's coefficient at its unit (the lane's own, before the tile's shuffles) and the
digit's factor it is the token's den term.

THE DEN'S ADDS go straight to `den` (`[BH heads, L]`, the call's), fire and forget (ruled 2026-11-02: fp32 atomics
for the den alone -- its size against a token's unit varies with the writers' value scales, so it cannot ride the
panel's fixed point, and a float atomic in shared memory is a compare-and-swap loop): `DenAdd`, a lane one term of a
GROUP -- `32 / (8 kStep)` tiles of a digit pair's run, lane `8 q + r` the term of tile `q / kStep`'s entry `r` in digit
`q % kStep` (two selects a tile, no branch) -- so the group's 32 terms are 32 distinct entries and go as ONE
unpredicated `red.global.add.f32` (`token_row`: the entry's units index, its first word's high half, as the head's
row and the window's token). A run's tiles run in SEGMENTS that end where the group fills; the add is between
segments, outside the steps (KERNEL_STANDARDS §29 item 7), on trips taken from a vote (`__any_sync`, so ptxas fences
no collective after it). A group spans the chunk's digits; its last terms go at the readout's end, and a lane holding
none adds an exact zero at the window's token `lane` (held to the call: a predicated reduction is a branch and a
convergence region in ptxas, PTX-level predication included -- checked 2026-11-02). Every den term is ABSOLUTE: its
raw coefficient times the digit's factor `2^(dE - kImageTop)` times its row's products (a term formed at the token's
unit through the channels' exponent `e` overflowed where a digit's state was zero and its den was not). Gated per
token against the inter term's denominator (`test_carry_vs_oracle.py`, the den gate's tolerance).

A SIDE WITH NO LIST LEVEL reads its den the same way (ruled 2026-11-02: the foundation's UNIFORM den): every tile of
every digit pair takes its two m16n8k8 against the digit's den row, each digit's term times the slot's coefficient (the
row's coefficient slice) and the digit's factor, the step's two digits summed in the lane, a lane its token of a group
of 4 tiles added as the listed side's are -- 16 adds a token a block (8 warps, two image halves). The token-carved
form it stands in for, a candidate: a warp's 8-token tiles against all 32 digits' den rows held in 16 registers, 4 MMAs
a tile, one add a token a block (MEASURED +4K in the units at nl64k-dense, den-as-channel 91c6110c).

THE BUDGET, a tile and digit on a listed side: the row's load, two MMAs, an add and a shuffle, the factor's multiply,
two selects, and a 32nd of the group's address and add -- ~8 instructions and a quarter of the channels' tensor work.
The first build's form took ~30 and two m16n8k16 (the row as row 0 of an A whose other rows four selects zeroed, lanes
`0..3` two tokens each, each word's slot by inverting `panel_row`, a den state whose 32 lists of a digit lay in one
bank). MEASURED (phase ledger, nl64k-alt-k4, sm_86, cycles a warp-window, the per-warp shared rows' form): the readout
3.62K, the units 1.86K, the carry 20.86K, against 3.00K, 1.62K and 18.76K with no den (the first build's 5.82K, 1.82K
and 25.0K). THE ACCESS TABLE (§30.1):

| buffer | written by | read by | shape | banks |
|---|---|---|---|---|
| `kDenMax` u32 `[32]` | `den_tables_max`: a lane its four words' digits, every warp, `red.shared.max` | `den_tables_out`: a lane its words' digits, lane `l` digit `l`'s factor | scalar | 16 digits two lanes each (once a window) |
| `kDenRows` f16, 64 bytes a digit | `den_tables_out`: a lane its four words, 16-bit | the tiles: 8 bytes at `64 d + 8 (lane & 7)` | 16-/64-bit | the stores up to 8-way (rows 64 bytes apart, once a window); the loads 8 addresses broadcast to 4 lanes each: one wavefront |

THE DEN'S TENANTS (`plan.md#tenants`): each digit's factor and largest (256 bytes); its rows (2 KB) over the leaves'
exponent bytes (1 KB) and 1 KB past them. Nothing else: the state is registers (`fold.md#den`), the sum is `den`'s.
The forms it replaced: a 4 KB shared den state, then the call's global workspace (one line of adds a list, a load a
window); a read buffer's per-warp fp32 den rows (`8 (8 + C') x 4` bytes, ~8.4 KB at 256 slots, a whole-image cell to
halves) and the drain's sum of them.

<a id="drain"></a>
## The drain

A chunk's panel to `num` (the den went to `den` from the readout, [`#den`](#den)), once every warp has read the
chunk, by the call's cross-CTA dtype
([`carry_api.md#carry-reduce`](carry_api.md#carry-reduce)): each slot's words at the CTA's unit ([`#units`](#units))
converted by the token's factor and added into HBM where the CTAs holding the token's leaves meet; the next fill clears
the panel. A word's two s16 halves come apart by `pair_magics` (offset magics: the fill clears a word to
`kPanelClear` = `0x80008000`, each half's zero biased by 2^15, so a word accumulated from it holds the low half biased,
its borrow taken from the high half's bias, the high half's top bit flipped; one byte permute a half places its 16 bits
under the magic's 2^23 + 2^22 -- two instructions where the masks and shift were five, and the bias's add at every
drained word is gone). F16 PAIRS: the factor is `2^(u - s - 1)`, `s` the token's scale byte (the call's `scale`:
every CTA the same, so the pairs meet at one scale; a bound on |num(t)|, which f16 absorbs with slack to ~2^27), each
half times it packed to f16 and added by `red.global.add.noftz.f16x2` -- the f16 value is bounded because the true
partial is at most the token's bound, which the scale puts under 2^14; the factor itself may be 2. A thread
keeps one word of every 16th slot -- 16 slots on, its panel cell is 1 KB on, the swizzle's bits unchanged, so its
cells are one address and an add -- and takes four slots a step, their words loaded unguarded (the panel lies in
shared memory past the chunk's slots) before their adds. A slot's FACTS -- its word offset in `num` and its factor
-- are formed once a slot, not once a word: a warp's two halves hold two slots a step, and lane `16 h + j` forms half
`h`'s slot of step `j` of a batch of 16 steps (its union token, row and factor), handed to the half's 16 lanes by two
width-16 shuffles a step (the source index the step's, warp-uniform); the step count is shuffled from lane 0, so
ptxas sees the loops warp-uniform ([`ops.md#uniform-warp`](../common/ops.md#uniform-warp): a loop it cannot prove so
fences and outlines every shuffle after it -- here 84 more guarded collectives, a fold step duplicated into a fallback).
In a chunk's last group of four steps a step past the warp's last is not issued (a warp-uniform branch; the requests
are the drain's floor, [lab `drain`](../../lab/drain.md)). FP32: the factor is `2^u`, a warp a slot in turn, four in
flight, a lane a channel (a token's full line for the box's 32 channels), the slot's facts formed by lane `j` for the
warp's step `j` of 32 and broadcast by two shuffles a slot, the lane's half one byte permute, added by
`red.global.add.f32`. A factor is exact over the float range (`pow2_wide`: a subnormal's bit below 2^-126, zero
below 2^-149). An add is never predicated:
ptxas puts a predicated `red.global` in a branch and a convergence region (BSSY, BRA, BSYNC around each add), so a
slot past the chunk takes the warp's first slot's row at factor 0 and adds exact zeros (a finite magic times 0) -- a
row of the CTA's own, never one every CTA shares (lab [`drain`](../../lab/drain.md): a shared dummy line serialized every CTA 6x). A slot's row is a 32-bit
word offset, one widening multiply-add from `num`'s base (the host holds a write head's rows under 2^32 words). CTAs
join only here (R22: independent units, joined by atomics). The panel's word `8 m + g` pairs channels `(c, c + 2)`,
`c = 8 (g >> 1) + 4 m + (g & 1)` ([`#image`](#image)); the f16 output's format follows it, the fp32 output is a
channel a word (`tests/oracle/readout_reference.py`'s `unpack_num`, `num_in_truth`). Measured 2026-10-19 (lab
[`drain`](../../lab/drain.md)): the f16 add alone runs at the integer drain's speed -- its +26-45% a drain is the
conversion's latency; a 32-bit global (fp32 or s32) doubles the drain's L2 sectors.

Its floor is the SM's global request path, ~5 cycles a line request with every SM draining (the lab). Measured
2026-10-19 (sm_86 locked, a warp-window): nl64k-alt-k4 2,135 -> 1,389 cycles (216 requests, ~1.1K at the floor), g 4
7,842 -> 5,175, the predicated form's 59 instructions a two-slot step (a guarded token load, 64-bit address
arithmetic, a convergence region an add) against this one's; a call 3.0-4.4% faster on every listed cell. Taken in
slices between the fold's lists it serialized with them rather than overlapping (2026-10-13).
