# `csrc/rola/src/den/den_kernel.cuh` — the den kernel

Mirror doc for the kernel that computes, beside the carry, each token's window-aligned denominator and readout
bound: one CTA a tile of 64 level-S rows x 64 level-R digits of a write head, window by window, the write mass of the
windows before held as fp32 MMA fragments across the whole call. Its constants and tenants are [`den.md`](den.md),
its host seam [`den_api.md`](den_api.md); the window lab that priced it is `docs/lab/den_sweep.md`; the reference is
`inter_reference` and the gate `tests/oracle/test_den_vs_oracle.py`.

Vocabulary: level S = the read side's list level, level R = the other; `a` a level-S row of the tile, `r` a level-R
digit of its span; a BAND = a den warp's four rows; the MASS `M_ch(a, r)` = the fragments, channel 0 the write gain
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

## The window

`den_kernel<Dense>` (grid `(tiles_s x tiles_r, BH)`, 512 threads, one CTA an SM) runs the call's windows in one
fixed order, every warp the same; a partial last window has `live` tokens, a whole number of 64:

1. the window's inputs issued -- the list form's two write unions and list exponents into parity `w & 1`, waited
   (the exponents set the scales); the dense form's first chunk -- then a rendezvous: every warp done with the window
   before (its slabs sit where the publish writes) and the inputs visible to all;
2. **the publish** ([below](#publish)): the fragments at their held scales into the region as fp32; a rendezvous;
3. **each band's B** ([below](#den-b)) from its four rows; a rendezvous -- the region is free;
4. **the den** ([lists](#den-list), [dense](#den-dense)): each warp's band against the window's read entries or
   tokens, every result added into the accumulator ([below](#drain)); a rendezvous;
5. **the drain**: the accumulator to `den` and `num`;
6. **the rescale**: the fragments to this window's scales where a column's power of two moved (one trip on a vote,
   `rescale`);
7. **the writes** ([below](#writes)): the window's entries folded into the fragments.

Four barriers a window in the list form (the dense form adds one a chunk), all `__syncthreads`: every warp runs the
same order, so a barrier where a consumer starts means every producer is done. THE UNION IS AT TWO PARITIES because
the top's staging is issued before the top's barrier: a warp finishing the window before's writes still reads its
union (found as a non-deterministic cohort-k4 den, 2026-10-25, until the parity).

<a id="phases"></a>

## The phases

`DenPhase`, the ledger's columns: `kPhaseStage` (copies issued and, in the list form, waited), `kPhasePublish` (the
publish to its barrier), `kPhaseDen` (the band's tiles), `kPhaseDrain`, `kPhaseWrites` (the window's writes),
`kPhaseWait` (the window's top barrier and the two after the publish and the den). The clock is the carry's
`PhaseClock`, gated: unbound it issues nothing.

<a id="fragments"></a>

## The fragments and their scales

The mass is `acc[4][4]`: four m-tiles of 16 rows (the tile's 64 rows) by one n-tile of 8 columns -- channel `n >> 2`
of digit `4 warp + (n & 3)`, so a warp owns four digits of the span -- in the m16n8k16 accumulator layout: lane `(g,
tq)` holds rows `16 mt + g` and `+ 8` of columns `2 tq` and `2 tq + 1` (`columns_of`: channel `tq >> 1`, digits `4
warp + 2 (tq & 1)` and the next). A column's values are held at a power of two -- the list form's `2^(4 - E_ch(r))`,
the dense form's `2^shift_ch` -- and move to the next window's by exact powers of two (`scales_lists`,
`scales_dense`; `pow2f` held to fp32's normals).

THE HELD EXPONENT (`kHeld`, an int16 a column, each lane writing its own two): the list form's exponent for a window
where no token writes the list is -126 (the facts pass's "a list no token writes"); taking it literally would swing
the fragments by 2^-123 and back by 2^+123 across the empty window, clamped at fp32's edge each way -- a factor 2 left
on the mass, seen as a 6% den error at flagship-cohort-k4, whose cohorts leave lists unwritten for a window. So a
column's exponent moves only where the window WRITES the list; an empty window keeps the held one, and the publish
and rescale use the held one. The dense form's window gains are never empty.

<a id="region"></a>
<a id="publish"></a>

## The publish and the region

Once a window the mass as of the windows before crosses lanes through shared memory: `publish` stores each lane's
two columns of each of its eight rows as one 8-byte fp32 store at the column's held scale, 16 stores a lane, into `[2
channels][64 rows][64 digits]` fp32 (32 KB of the region's 64). fp32, not f16: the mass grows without bound over a
call's windows (a sum of positive terms), past f16's 65,504 within a few thousand tokens; the lab published f16 over
32 windows and would have overflowed at 64K. `mass_at` swizzles the row's sixteen 16-byte chunks by `(row + 4 channel)
& 7`, so a quarter-warp's four stores (two rows x two channels, 8 bytes each) and the B build's loads (two rows of one
channel) meet distinct banks.

<a id="den-b"></a>

## The den's B

`den_b`: a den warp's band is the MMA's eight columns -- lane `g`'s column is channel `g >> 2`, row `4 band + (g &
3)` -- and its K the span's 64 digits in four k16 steps (positions `2 tq ..` and `2 tq + 8 ..` of step `kk` the
digits `16 kk + 2 tq ..` and `+ 8 ..`, read as two 8-byte loads a step). The lane finds the row's largest over the
quad (two shuffles) and converts the row to f16 at the power of two putting that largest in `[2^14, 2^15)`: 11 bits of
every value against the row's top, the carry's image's own precision class, and f16's range is never a question. `inv`
= `2^(e - 14)` undoes it in the add; a zero row gives B = 0 and `inv` = 0. Measured against an exact mass on
flagship-cohort-k4 (a row's digits spread 1e299 : 1 where a cohort leaves most unwritten) the single f16 plane holds
every token's den within 1e-4; a second plane for the residue was not needed.

<a id="slabs"></a>
<a id="den-tile"></a>

## The slabs: operands staged, never loaded into registers

Every MMA operand that comes from global memory goes through a SLAB (KERNEL_STANDARDS §20: a step's operands are
staged by `cp.async`, never `LDG` inside an MMA loop): 16 rows of 64 f16 digits, 128 bytes a row, its 16-byte chunks
XORed with the row's residue (`slab_at`), so the eight rows an `ldmatrix` matrix addresses meet distinct banks by
construction, whatever the tokens' residues. `stage_slab`: lane `l` copies four chunks of row `l >> 1` (128 copies a
slab, four a lane). Each warp owns two slabs in the region (`2 x 2 KB x 16 warps` = the region's 64 KB, free once the
B is built): a tile's rows are staged one ahead into the other slab, the warp waits `cp.async.wait_group 1` and a
`__syncwarp`, and reads them by `ldmatrix` (`den_mma`: row-major A, lane `l` row `(l & 7) + 8 ((l >> 3) & 1)` at chunk
`2 kk + (l >> 4)`; `write_mma`: k-major A by `ldmatrix.trans`, lane `l` slab row `8 (l >> 4) + (l & 7)` at chunk `2 mt
+ ((l >> 3) & 1)`). The dense form's den has one slab a warp (its chunk buffers hold the region's other half), waited
before each tile.

<a id="den-list"></a>

## The den, list form

`den_lists`: a band's four lists (level S digits `s0 + 4 band ..`) are one STREAM of entries, a lane's entry's list by
three compares against the lists' running counts (`chunk_runs`' prefix idea, `parts.md#entries`), their lengths from
the group's prefix rows' last column. The stream is staged a ROUND of 128 at a time into the warp's row of the
scratch as packed words -- `place | list << 14 | coef << 16`, the entry's token place (query head x 512 + token),
its row of the band and its f16 coefficient -- so a tile reads two 4-byte words a lane from shared memory instead
of two 8-byte entries from L2; the round buffer is the warp's own, so a round costs one warp sync, no CTA barrier,
and a band of any length (the cohort's lumpiest holds ~3,000 entries) runs in rounds.

A TILE is 16 entries: lane `g` holds entries `16 i + g` and `+ 8`; the tile's rows are a slab ([above](#slabs)) and
its four k-steps run into a fresh fragment (`den_mma`, a burst). A slot past the round's count takes the last entry's
row with no value.

THE ADD: an entry's result is its row's two columns, held by the lanes whose `tq & 1` is its row's `>> 1` -- in
register `row & 1` -- times `inv` and the entry's coefficient, added into the ACCUMULATOR ([below](#drain)) by a
predicated `red.shared.add.f32` (`acc_add`): exactly two adds an entry, its den by a lane of channel 0 and its num by
one of channel 1.

<a id="den-dense"></a>

## The den, dense form

`den_dense`: no lists -- every live token of every query head against the band, a tile 16 tokens (a head's tokens
then the next head's, `place_of`), their rows a slab. The token's four level-S coefficients of the band are the
stored f16 rows: a lane loads the two of its columns' rows as one 4-byte `ld.global.nc` (the one load the gate admits
outside a slab: a coefficient, not an operand), forms `x_S(a) C(a) inv(a)` over its two rows and adds the other two
across `xor 1`; the band's sum for the tile's first token is added by the quad's even lanes, the second's by its odd
lanes -- one shared add a lane a tile, all live.

<a id="drain"></a>

## The accumulator and the drain

Each token's den and num of the window accumulate in shared memory -- `kAcc`, fp32 `[2 channels][heads x 512]`,
zeroed at the kernel's start -- and go to global memory ONCE a window (`drain`): after the den's barrier every thread
takes rows of the window's `live` tokens in turn, a warp's lanes consecutive tokens of one channel, adds each by
`red.global.add.f32` into the token's row of `den` or `num` and zeroes it behind the read. The lab's den added every
entry's result to global memory where it fell: ~2 scattered adds an entry, each its own L2 request (~5 SM cycles,
`global-red-is-per-request`), ~5K cycles of its 8.45K a CTA-window; the drain's adds are 32 consecutive tokens a
warp -- one line a channel -- and the per-entry work is a shared atomic (~2 cycles). It is also §20's rule: a
reduction to global memory belongs to a drain outside the steps.

<a id="writes"></a>

## The writes, list form

`writes_lists`: a warp's four level-R digits are its four lists of the window (one group's), their entries one
stream as the den's; 64 entries a chunk into the warp's SLOTS (the scratch, free once the den is drained: the token
from the group's staged union, each channel's coefficient `c_m`, `c_v` with the entry's list) -- then k-steps of 16:
the entries' tokens' level-S rows staged a step ahead into the warp's two slabs ([above](#slabs)), A by
`ldmatrix.trans` (`write_mma`, a burst), B the lane's column's channel of the two entries at its k positions WHERE
their list is its digit, else zero (`pick`): an entry is live in two of the n-tile's eight columns, its digit's at each
channel. Four fp32-accumulate MMAs a k-step. The slab's rows are the step's entries in stream order, so the
`ldmatrix` rows are conflict-free by construction (the lab arranged entries on distinct token residues to the same
end; the facts pass's token order needs no arrangement here).

<a id="dense"></a>

## The writes, dense form

`writes_dense`: `live / 64` CHUNKS of 64 tokens a window, each chunk's level-S rows, level-R rows and gains staged
into one of two parities (`stage_chunk`: one 16-byte copy of each row set a thread and the gains by sixteen threads)
while the chunk before folds. A k-step's B is the chunk's level-R rows by one `ldmatrix.trans` over 32 tokens of the
PAIR of warps' eight digits (the 16-byte chunk `warp >> 1`), each lane then taking its own warp's digit across the
pair by one shuffle (`4 (4 (warp & 1) + (g & 3)) + tq`), times the tokens' gains of its column's channel (`m` for the
first four columns, `F` for the last four: the staged `(m, F)` pairs permuted into token pairs by `prmt`). A as the
list form's, from the chunk's level-S rows (already a swizzled slab). Four MMAs a k-step, 32 k-steps a whole window:
128 MMAs a warp, the tensor floor at 32 cycles each 4K cycles a window on sm_86.

## Against the lab, and the budget

`docs/lab/den_sweep.md` measured the 64 x 128 u8 form at 12.7K cycles a CTA-window at alt-k4 (den 8.45K, writes
8.83K; the publish and staging 4.4K), 68.9K at dense, with every operand loaded straight into registers and every
result added straight to global memory. This kernel's budget (`tools/budgets/den.json`) is stated from this form's
operation counts and re-stated when its region ledger first runs; the pass's whole cost beside the carry is the
comparison that matters (the carry at alt-k4 runs ~18.4K cycles a CTA-window over 128 CTAs a head; 16 den CTAs at
~8K each would be ~5% of it).
