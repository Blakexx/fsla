# `csrc/rola/src/den/den.cuh` / `den.cu` — the den pass's constants, operand block and translation unit

Mirror doc for the header every den source includes and for the unit that validates a call, resolves its form and
launches the kernel. The kernel is [`den_kernel.md`](den_kernel.md); the host entries with their refusals
[`den_api.md`](den_api.md). The pass is the carry's companion ruled 2026-10-06 ("the window-aligned den is a pass of
8 CTAs a head beside the carry"; Blake: "If it's small I'm ok to just use that as the actual design") with its den
operand ruled f16 on 2026-10-12: per token `t`, `den(t)` is its read leaf product against the WRITE MASS of the windows
before -- `Σ_a Σ_r x_S(t, a) x_R(t, r) M(a, r)`, `M(a, r) = Σ_{t' in earlier windows} m(t') w_S(t', a) w_R(t', r)` --
and `num(t)` the same product against the mass of `F = m s` (`s` the writing token's value scale), so `num / den`
bounds every readout channel of `t` (`carry_api.md#carry-reduce`). The epilogue divides the carry's numerator by
`den` plus the intra's own; the reference is `tests/oracle/reference.py`'s `inter_reference` on the bundle's
dequantized operands, held per slot by `tests/oracle/test_den_vs_oracle.py`.

<a id="constants"></a>

## The constants

A CTA is `kWarps` = 16 warps over a TILE of `kRows` = 64 level-S rows x `kSpan` = 64 level-R digits, in two ROLES
(ruled 2026-11-01, `den_kernel.md#kernel`): `kWriteWarps` = 8 write warps of `kDigits` = 8 digits each and `kDenWarps`
= 8 den warps of `kBand` = 8-row bands; each role's MMAs are `kBandTiles` = `kDigitTiles` = 2 n-tiles wide (a band's
eight rows or a warp's eight digits x `kChannels` = 2). The den's K is the span in `kSteps` = 4 steps of 16; a den
tile is `kTileRows` = 16 read entries (one m16), a write step 16 entries, each a SLAB of `kSlabBytes` = 2,048 (16 f16
rows of the span); a den warp stages `kRound` = 64 read entries a round; the writes take `kChunk` = 64 entries a chunk
(the dense form 64 tokens); the dense den runs `kBatchTiles` = 4 tiles (64 tokens) a BATCH; `kMaxHeads` = 4 query heads a write head is what the accumulator holds. A den WORD packs
an entry's place (`kPlaceBits` = 13: query head x W + token, under 4 x 1,024), its list in the band (`kListBits` = 3)
and its f16 coefficient. `kBarPublish`, `kBarDen` and `kBarWrite` are the named barriers past the CTA's: the
publish handed to the den warps, the den warps' own, the write warps' own. `kCoefTop` and `kUnitMin` are the facts
pass's (`facts.md#lists-out`): a list's coefficients are `f16(b m 2^(4 - E))`, `E = max(K, -123)`.

WHY 64 x 64 WHERE THE RULING SAID 64 x 128. The den sweep (`docs/lab/den_sweep.md`) priced the pass with its den as
u8 x u8 integer MMAs over u8 read rows, 128 digits of K a CTA, every operand loaded straight into registers. The
bundle stores a DENSE level's rows as f16 on both sides (`facts/liveness_contract.md#floor`), and level R on the read
side is the dense one, so the den's A operand is f16 and the ruling of 2026-10-12 made the den an f16 MMA with fp32
accumulate. KERNEL_STANDARDS §20 puts every MMA operand through shared memory by `cp.async`, so a tile's rows are a
slab a warp stages ahead: at 128 digits a slab is 4 KB and two a warp for 16 warps are 128 KB; at 64 digits they are
the region's 64 KB exactly, and a (256, 256) head is 16 CTAs instead of 8 -- more CTAs a head, which on a card with
few heads is the parallelism the pass lacked (one head's 8 CTAs ran 0.98 ms a call at L 64K on sm_86, a tenth of
the SMs busy). The publish and the staging replicate over twice the CTAs; the per-token work does not.

Per architecture: sm_80 the same form (its fp32-accumulate HMMA four times sm_86's rate a scheduler; its 164 KB of
shared memory would hold 128-digit slabs); sm_90 replaces the staging with TMA and the drain's adds with
`cp.reduce.async.bulk`; the warp count and tile are the same on all three. Scaling: O(L k) list work a head plus
O(L N) MMAs at a dense level, as the carry's.

<a id="params"></a>

## The operand block

`DenParams`: `read_s`, `read_r`, `write_s`, `write_r` the sides' rows at the two levels as `Plane`s (the carry's:
base and byte strides); the list form reads `read_r` and `write_s` as planes and takes `read_lists` (level S, over the
`heads` query heads of a write head) and `write_lists` (level R) in the facts pass's final form (`carry.md#lists`)
with `list_exp` their exponents; the dense form reads all four planes and the facts pass's window gains `gains16` and
`gains_exp`. `den` and `num` are the outputs, fp32 `[BH heads, L]`, zeroed by the caller and added into by every CTA
holding one of the token's leaves. `L`, `windows`, `heads`, `write_heads`, `width_s`, `width_r`, `tiles_s` (level S's
tiles, the grid's fast index) and `level_s` are the call's extents; `ledger` the bound phase ledger or null.

<a id="tenants"></a>

## The shared tenants

`Tenants` lays the dynamic shared memory out in byte offsets, 100,224 bytes at W = 512 against sm_86's 101,376-byte
opt-in, and more at W = 1,024 (sm_80's 164 KB). The two roles run a window at once, so each role's tenants are its own:

| tenant | bytes | holds |
|---|---|---|
| `kMass` | 32,768 | the published mass, fp32 `[2][64][64]` swizzled (`den_kernel.md#region`), until every band has its B (`kBarDen`); then the list form's two slabs a den warp (`den_kernel.md#slabs`) or the dense form's two batches of 64 tokens' level-R and level-S read rows (`den_kernel.md#den-dense`) |
| `kWriteRegion` | 32,768 | the write warps' two slabs a warp (the list form's steps), or the dense form's chunk buffers (`kChunkWs`, `kChunkWr`: two parities of 64 tokens' level-S and level-R rows) |
| `kWords`, `kResults` | 6,144 | each den warp's round of 64 words and their results, a result's first slot holding its row's offset until its tile is staged (`den_kernel.md#den-list`) |
| `kSlotToken`, `kSlotC` | 6,400 | each write warp's slots: 64 entries' tokens, each channel's coefficient with its digit (`den_kernel.md#writes`) |
| `kAcc` | 16,384 | the accumulator: each query head's token's den and num of the window, fp32 `[2][4 x 512]`, drained once a window (`den_kernel.md#drain`) |
| `kUnion` | 4,096 | the span's two write unions at two parities (a slot's token) |
| `kExps` | 512 | the span's list exponents at two parities, `[64][2]` int16 |
| `kHeld` | 256 | each column's held exponent, `[64][2]` int16 (`den_kernel.md#fragments`) |
| `kClock` | 896 | the phase clock's rows |
| `kPartials`, `kGainsW` | 8,192 + 4,096 | the dense form's, over the list form's words, results and slots: its batches' band partials at two parities, `[4 tiles][16 tokens x 2][8 bands]` fp32 (`den_kernel.md#den-dense`), and two windows' gains `[W][2]` f16 (their own tenant where they do not fit) |

The mass region is time-shared because the mass is read once a window (into each band's B registers) and the den
warps' slabs are theirs alone; the publish of the next window waits for the den warps at the window's top barrier.

## The translation unit

`den.cu` holds the host entry (`den_api.md#den-forward`): it checks the planes the form reads are f16 and 16-byte
aligned (`plane_of`), the lists are the facts pass's shapes (`lists_of`), the outputs the caller's, the call a whole
number of 64-token chunks and at most `kMaxHeads` query heads a write head, and launches
`den_kernel<true>` (both sides dense) or `den_kernel<false>` (the list form) on grid `(tiles_s x tiles_r, BH)` with
the tenants' bytes of dynamic shared memory, the attribute raised once.
