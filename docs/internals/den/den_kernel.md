# `csrc/rola/src/den/den_kernel.cuh` — the den kernel

Mirror doc for the kernel that computes, beside the carry, each token's window-aligned denominator and readout
bound: one CTA a tile of 64 level-S rows x 64 level-R digits of a write head, window by window, in two ROLES -- write
warps holding the write mass of the windows before as fp32 MMA fragments across the whole call, den warps running the
window's reads against it. Its constants and tenants are [`den.md`](den.md), its host seam [`den_api.md`](den_api.md);
the window lab that priced it is `docs/lab/den_sweep.md`; the reference is `inter_reference` and the gate
`tests/oracle/test_den_vs_oracle.py`.

Vocabulary: level S = the read side's list level, level R = the other; `a` a level-S row of the tile, `r` a level-R
digit of its span; a BAND = a den warp's eight rows; the MASS `M_ch(a, r)` = the fragments, channel 0 the write gain
`m`, channel 1 `F = m s`; an ENTRY = one (token, digit) of a facts-pass list; a SLAB = 16 rows of 64 f16 digits (2 KB)
staged for one MMA tile or k-step; the two FORMS = LISTS (the read side lists S, the write side lists R) and DENSE
(neither side lists).

## What it computes

For window `w` and a token `t` of a query head in it:

    den(t) += Σ_a x_S(t, a) Σ_r x_R(t, r) M_m(a, r)          M as of the windows before w
    num(t) += Σ_a x_S(t, a) Σ_r x_R(t, r) M_F(a, r)
    M_ch(a, r) += Σ_{t' in w} c_ch(t', r) w_S(t', a)          after the window's reads

`x_S` is the token's level-S read coefficient (the list form's entry coefficient `f16(byte / 256)`; the dense form's
stored f16), `x_R` its level-R read row (stored f16, the dense level), `w_S` the writing token's level-S write row
(stored f16), `c_ch` its level-R coefficient times the channel's gain: the list form's entry coefficient `f16(b m
2^(4 - E))` at the list's exponent, the dense form's stored f16 times the window's gain `m 2^shift` or `F 2^shift`.
The outputs are in these units -- the reader's per-token scales left out, as the carry leaves them, and the mass in
write bytes where level R is u8 -- and the gate's `in_truth` applies the factor.

<a id="kernel"></a>

## The window and the roles

`den_kernel<Dense>` (grid `(tiles_s x tiles_r, BH)`, 512 threads, one CTA an SM) runs the call's windows in one
fixed order; a partial last window has `live` tokens, a whole number of 64. Warps 0-7 are the WRITE role
(`write_role`), warps 8-15 the DEN role (`den_role`) -- KERNEL_STANDARDS section 5's form on sm_80: both roles fold
work (each its own MMAs), neither is a fetcher, and neither needs more than the 128 registers 16 warps leave a
thread. The den of a window and its writes are independent -- the den reads the mass of the windows BEFORE, the
writes fold the window's own -- so a window's den runs beside its writes (ruled 2026-11-01):

1. **the top**: every warp at `__syncthreads` -- the den warps done with the window before's slabs (they sit in the
   mass's region) and its drain, the write warps with its writes and the inputs they issued under them landed;
2. the write warps: **the publish** ([below](#publish)) at the held scales, handed over by `bar.arrive kBarPublish`;
   **the rescale** (one trip on a vote); the next window's inputs issued (the list form's unions and exponents into
   parity `(w + 1) & 1`, the dense form's next gains exponent read); **the writes** ([lists](#writes),
   [dense](#dense)) into the fragments, the issued inputs waited;
3. the den warps: `bar.sync kBarPublish`; **each band's B** ([below](#den-b)); `bar.sync kBarDen` among the 256 den
   threads -- the region is theirs; **the den** ([lists](#den-list), [dense](#den-dense)) into the accumulator;
   `bar.sync kBarDen`; **the drain** ([below](#drain)).

One CTA barrier a window, the publish's hand-over, two den-role barriers (and in the dense form one write-role
barrier a chunk). THE UNION IS AT TWO PARITIES because a window's is issued under the window before's writes (the
2026-10-25 parity, from a non-deterministic cohort-k4 den when it was issued under the top's barrier).

<a id="rows"></a>

## A window's rows

`WindowRows`: a plane's batch, the write head's first query head and the window's first token at the tile's digit
are one 64-bit base a window; a row is the base plus `token x st + head x sh` in 32 bits (the host refuses a call
whose window of a write head's rows spans 2^32 bytes). The pre-split kernel formed each row's address from the call's
token by three 64-bit products off a shared-memory load, the chain the sm_80 step cover found under the den's tile
loop (2026-11-01).

<a id="phases"></a>

## The phases

`DenPhase`, the ledger's columns, booked by each role: a write warp's `kPhaseStage` (from its writes' end to the top:
the issued inputs waited), `kPhaseWait` (the top), `kPhasePublish` (scales, publish, hand-over) and `kPhaseWrites`
(the rescale and the writes); a den warp's `kPhaseWait` (the top and the publish's hand-over, then the den barrier
before the drain), `kPhasePublish` (B and the den barrier after it), `kPhaseDen` and `kPhaseDrain`. The clock is the
carry's `PhaseClock`, gated: unbound it issues nothing.

<a id="fragments"></a>

## The fragments and their scales

A write warp's mass is `acc[4][2][4]` (`Mass`): four m-tiles of 16 rows (the tile's 64 rows) by two n-tiles of 8
columns -- n-tile `J`'s column `n` is channel `n >> 2` of digit `8 ww + 4 J + (n & 3)`, so a write warp owns eight
digits of the span -- in the m16n8k16 accumulator layout: lane `(g, tq)` holds rows `16 mt + g` and `+ 8` of columns
`2 tq` and `2 tq + 1` (`columns_of`: channel `tq >> 1`, digits `8 ww + 4 J + 2 (tq & 1)` and the next). A column's
values are held at a power of two -- the list form's `2^(4 - E_ch(r))`, the dense form's `2^shift_ch` -- and move to
the next window's by exact powers of two (`scales_lists`, `scales_dense`; `pow2f` held to fp32's normals).

THE HELD EXPONENT (`kHeld`, an int16 a column, each lane writing its own two): the list form's exponent for a window
where no token writes the list is -126 (the facts pass's "a list no token writes"); taking it literally would swing
the fragments by 2^-123 and back by 2^+123 across the empty window, clamped at fp32's edge each way -- a factor 2 left
on the mass, seen as a 6% den error at flagship-cohort-k4, whose cohorts leave lists unwritten for a window. So a
column's exponent moves only where the window WRITES the list; an empty window keeps the held one, and the publish
and rescale use the held one. The dense form's window gains are never empty.

<a id="region"></a>
<a id="publish"></a>

## The publish and the region

Once a window the mass as of the windows before crosses from the write warps to the den warps through shared memory:
`publish` stores each lane's two columns of each of its eight rows and two n-tiles as one 8-byte fp32 store at the
column's held scale, 16 stores a lane, into `[2 channels][64 rows][64 digits]` fp32 (the 32 KB `kMass`), n-tile `J`'s
columns in the row's 16-byte chunk `2 ww + J`. fp32, not f16: the mass grows without bound over a call's windows (a
sum of positive terms), past f16's 65,504 within a few thousand tokens. `mass_at` swizzles the row's sixteen 16-byte
chunks by `(row + 4 channel) & 7`, so a quarter-warp's four stores (two rows x two channels, 8 bytes each) and the B
build's loads (two rows of one channel) meet distinct banks.

<a id="den-b"></a>

## The den's B

`den_b`: a den warp's band of eight rows is two n-tiles of the MMA's eight columns -- lane `g`'s column of n-tile `J`
is channel `g >> 2`, row `8 band + 4 J + (g & 3)` -- and its K the span's 64 digits in four k16 steps (positions `2 tq
..` and `2 tq + 8 ..` of step `kk` the digits `16 kk + 2 tq ..` and `+ 8 ..`, read as two 8-byte loads a step). The
lane finds the row's largest over the quad (two shuffles) and converts the row to f16 at the power of two putting
that largest in `[2^14, 2^15)`: 11 bits of every value against the row's top, the carry's image's own precision
class, and f16's range is never a question. `inv` = `2^(e - 14)` undoes it in the add (`den_inv`: each lane's two
columns' factors of each n-tile, from the lanes holding them); a zero row gives B = 0 and `inv` = 0. Measured against
an exact mass on flagship-cohort-k4 (a row's digits spread 1e299 : 1 where a cohort leaves most unwritten) the single
f16 plane holds every token's den within 1e-4.

<a id="slabs"></a>
<a id="den-tile"></a>

## The slabs: operands staged, never loaded into registers

Every MMA operand that comes from global memory goes through a SLAB (KERNEL_STANDARDS §20: a step's operands are
staged by `cp.async`, never `LDG` inside an MMA loop): 16 rows of 64 f16 digits, 128 bytes a row, its 16-byte chunks
XORed with the row's residue (`slab_at`), so the eight rows an `ldmatrix` matrix addresses meet distinct banks by
construction, whatever the tokens' residues. `stage_row`: lane `l` copies the four chunks `4 (l & 1) ..` of row `l >>
1` -- one 64-bit source a lane and four copies at immediate offsets, into the four shared addresses `slab_lane` forms
once a lane -- predicated on `live`: a step's last tile stages nothing ahead, branch-free (a dead copy reads nothing
and zero-fills the slab the tile before consumed; an empty group keeps `cp.async.wait_group 1`'s count). Each warp of
a role owns two slabs in its role's region (the list den's in `kMass` once every band has its B, the write warps' in
`kWriteRegion`): a tile's rows are staged one ahead into the other slab, from an offset or token read a tile before;
the warp waits, `bar.warp.sync`s (`ops::warp_sync`), and reads them by `ldmatrix` (`den_mma`: row-major A, lane `l`
row `(l & 7) + 8 ((l >> 3) & 1)` at chunk `2 kk + (l >> 4)`, against both n-tiles: eight MMAs a tile; `write_mma`:
k-major A by `ldmatrix.trans`, lane `l` slab row `8 (l >> 4) + (l & 7)` at chunk `2 mt + ((l >> 3) & 1)`, against both
n-tiles: eight MMAs a step).

THE STEPS HOLD NOTHING ELSE (KERNEL_STANDARDS §20, §27; the SASS gate's step cover, which until 2026-11-01 could not
see these loops): no shared fp32 add (on sm_80/86 a compare-and-swap loop), no forward branch (a guard is a
predicated copy or store, `ops::store_shared_f32_if`), no global load and no division.

<a id="den-list"></a>

## The den, list form

`den_lists`: a band's eight lists (level S digits `s0 + 8 band ..`, inside one 32-digit group) are one STREAM of
entries (`runs_of`: their lengths from the group's prefix rows' last column; `list_of`: an entry's list by seven
compares). The stream is staged a ROUND of 64 at a time: each entry's word -- `place | list << kPlaceBits | coef <<
16`, its token place (query head x W + token), its row of the band and its f16 coefficient -- into the warp's row of
`kWords`, and its row's 32-bit offset in the window (`WindowRows`) into the first of its two RESULT slots (`kResults`),
which hold it until its tile is staged. A round costs one warp sync, and a band of any length (the cohort's lumpiest
holds ~3,000 entries) runs in rounds.

A TILE is 16 entries: lane `g` holds entries `16 i + g` and `+ 8`, their words read a tile ahead; the tile's rows are a
slab ([above](#slabs)) and its four k-steps against both n-tiles run into fresh fragments (`den_mma`, a burst). An
entry's result is its row's two columns -- n-tile `row >> 2`, columns `row & 3` of each channel -- held by the lanes
whose `tq & 1` is `(row >> 1) & 1`, in register `row & 1`: each stores it, times `inv` and the entry's coefficient,
into the entry's result slot of its channel by one predicated store.

THE DEPOSIT, after the round's tiles and outside them: each entry's two results added into its token's ACCUMULATOR
row ([below](#drain)) by `red.shared.add.f32` (`acc_add`), its den and its num. A token's entries in different bands
meet in its row, so the add is atomic -- on sm_80/86 a compare-and-swap loop, which is why it is no step's.

<a id="den-dense"></a>

## The den, dense form

`den_dense`: no lists -- every live token of every query head against the band, in BATCHES of four tiles (64 tokens
of one query head; a window's `live` is a whole number of them). A batch's 64 level-R read rows and 64 level-S read
rows (the tile's digits, 128 bytes each) are staged ONCE for all eight bands into one of two parities of `kMass`, four
copies a den thread, the next batch's issued as the batch begins. One den-role barrier a batch: every copy of the
batch landed, every band done with the batch before (whose parity the next is then issued into) and its partials
complete. Then each band, for each of the batch's tiles: its MMAs (`den_mma`), its eight rows summed with each
token's level-S coefficients read from the staged rows (a lane's four, the other four across `xor 1`), and the band's
sum for each token and channel stored in the batch's PARTIALS `[tile][token x channel][band]` (`kPartials`, two
parities), each value's band slot rotated by `(band + value / 4) & 7` so a warp's 32 stores meet distinct banks. The
batch before's partials are summed over the eight bands (two threads a value, four slots each, a shuffle) into the
accumulator: each token's row is written once a window, so the dense den issues no atomic. The pre-split kernel had
every den warp stage its own copy of every tile's rows -- eight times the L2 traffic the rows need -- and add every
token's band sum into the accumulator by a shared atomic.

<a id="drain"></a>

## The accumulator and the drain

Each token's den and num of the window accumulate in shared memory -- `kAcc`, fp32 `[2 channels][heads x W]`, zeroed
at the kernel's start -- and go to global memory ONCE a window (`drain`, by the den warps after their second den
barrier): for each channel and query head the 256 den threads take the window's `live` tokens in turn, a warp's lanes
consecutive tokens, add each by `red.global.add.f32` into the token's row of `den` or `num` and zero it behind the
read. The lab's den added every entry's result to global memory where it fell: ~2 scattered adds an entry, each its
own L2 request (~5 SM cycles, `global-red-is-per-request`), ~5K cycles of its 8.45K a CTA-window; the drain's adds are
32 consecutive tokens a warp -- one line a channel -- and the per-entry work is a shared atomic (~2 cycles). It is
also §20's rule: a reduction to global memory belongs to a drain outside the steps.

<a id="writes"></a>

## The writes, list form

`writes_lists`: a write warp's eight level-R digits are its eight lists of the window (one group's), their entries
one stream as the den's; 64 entries a chunk into the warp's SLOTS (the token from the group's staged union, each
channel's coefficient `c_m`, `c_v` with the entry's list) -- then k-steps of 16: the entries' tokens' level-S rows
staged a step ahead into the warp's two slabs ([above](#slabs)), A by `ldmatrix.trans` (`write_mma`, a burst), B for
each n-tile `J` the lane's column's channel of the two entries at its k positions WHERE their list is the column's
digit `4 J + (g & 3)`, else zero (`pick`): an entry is live in two of the sixteen columns, its digit's at each
channel. Eight fp32-accumulate MMAs a k-step. The slab's rows are the step's entries in stream order, so the
`ldmatrix` rows are conflict-free by construction.

<a id="dense"></a>

## The writes, dense form

`writes_dense`: `live / 64` CHUNKS of 64 tokens a window, the `q`-th chunk of the call in parity `q & 1`, each chunk's
level-S rows and level-R rows staged by the write warps (`stage_chunk`: two 16-byte copies of each row set a write
thread, predicated on there being a next chunk). A chunk is waited, then ONE write-role barrier (`kBarWrite`): every
write warp's copies of this chunk landed and every write warp done with the chunk before, whose parity the next
chunk -- the window's, or the next window's first -- is then issued into (the pre-split kernel issued it before its
barrier, so a fast warp could overwrite a chunk a slow one was still reading). THE WINDOW'S GAINS -- its live tokens'
`(m, F)` f16 pairs -- are staged once a window outside the steps (`stage_gains`, the next window's under this one's
writes, two parities in `kGainsW`). A k-step's B is the chunk's level-R rows by one `ldmatrix.trans` over 32 tokens of
the warp's eight digits (the 16-byte chunk `ww`), each lane taking its column's digit `4 J + (g & 3)` of each n-tile by
one shuffle (`4 (4 J + (g & 3)) + tq`), times the tokens' gains of its column's channel (`m` for the first four
columns, `F` for the last four: the staged pairs permuted into token pairs by `prmt`). A as the list form's, from the
chunk's level-S rows (already a swizzled slab). Eight MMAs a k-step, 32 k-steps a whole window: 256 MMAs a write warp,
the tensor floor at 32 cycles each 8K cycles a window on sm_86.

## Against the lab, and the budget

`docs/lab/den_sweep.md` measured the 64 x 128 u8 form at 12.7K cycles a CTA-window at alt-k4 (den 8.45K, writes 8.83K;
the publish and staging 4.4K), 68.9K at dense, with every operand loaded straight into registers and every result
added straight to global memory. The single-role kernel (d1ad1aba) ran 27K at alt-k4, every warp's den then its
writes, each a chain of exposed round trips. THE SPLIT'S BUDGET (the design of 2026-11-01, `tools/budgets/den.json`):
at nl64k-alt-k4 a CTA-window gathers ~512 read entries' and ~512 write entries' 128-byte rows, ~128 KB at ~9 B/cycle
an SM = ~14K cycles, against ~4K of tensor time on sm_86 (512 MMAs) and ~2.5K of issue: bytes-bound, the line 1.3x =
~18.5K. The dense form: ~4,096 MMAs a CTA-window, 32K cycles of tensor time on sm_86.

MEASURED (`tools/phase_ledger.py --kernel den`, sm_86, 2026-11-01): nl64k-alt-k4 ~19.4K cycles a warp-window (den
warps: den 14.1K, B 1.3K, drain 0.9K, waits 3.2K; write warps: writes 11.5K, waits 6.9K) against the single-role
kernel's 27K -- 1.39x the bytes floor at 9 B/cycle, 1.21x at the 8 B/cycle the intra_stream probe measured beside
MMAs; nl64k-dense ~40.6K (den 30.2K, writes 39.6K) against ~68K, 1.27x its tensor floor. The step cover on sm_80: list
form >= 87%, dense 87.2%.
