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

A CTA is `kWarps` = 16 warps over a TILE of `kRows` = 64 level-S rows x `kSpan` = 64 level-R digits; `kBand` = 4
rows a den warp (the MMA's eight columns: four rows x `kChannels` = 2), `kDigits` = 4 digits a write warp (one n-tile:
four digits x two channels). The den's K is the span in `kSteps` = 4 steps of 16; a den tile is `kTileRows` = 16 read
entries (one m16), a write step 16 entries, each a SLAB of `kSlabBytes` = 2,048 (16 f16 rows of the span); a warp
stages `kRound` = 128 read entries a round; the writes take `kChunk` = 64 entries a chunk (the dense form 64 tokens);
`kMaxHeads` = 4 query heads a write head is what the accumulator holds. `kCoefTop` and `kUnitMin` are the facts
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

`Tenants` lays the dynamic shared memory out in byte offsets, 98.8 KB on sm_86's 99 KB opt-in:

| tenant | bytes | holds |
|---|---|---|
| `kRegion` | 65,536 | the published mass, fp32 `[2][64][64]` swizzled (`den_kernel.md#region`), from the window's top until every band has its B; then each warp's two slabs of staged rows (`den_kernel.md#slabs`) -- the den's tiles, then the writes' steps -- or the dense form's one slab a warp and, past the mass, its chunk buffers (`kChunkWs`, `kChunkWr`: two parities of 64 tokens' level-S and level-R rows) |
| `kScratch` | 12,800 | each warp's round of 128 packed read entries during the den; each warp's write slots (64 entries' tokens, each channel's coefficient with its digit) during the writes |
| `kAcc` | 16,384 | the accumulator: each query head's token's den and num of the window, fp32 `[2][4 x 512]`, drained once a window (`den_kernel.md#drain`) |
| `kUnion` | 4,096 | the span's two write unions at two parities (a slot's token) |
| `kExps` | 512 | the span's list exponents at two parities, `[64][2]` int16 |
| `kGains` | 512 | the dense form's chunk gains at two parities, `[64][2]` f16 |
| `kHeld` | 256 | each column's held exponent, `[64][2]` int16 (`den_kernel.md#fragments`) |
| `kClock` | 896 | the phase clock's rows |

The region is time-shared because the mass is read once a window (into each band's B registers) and the slabs are
each warp's own; the scratch because the den's rounds end before the writes' slots begin (the drain's barrier
between them).

## The translation unit

`den.cu` holds the host entry (`den_api.md#den-forward`): it checks the planes the form reads are f16 and 16-byte
aligned (`plane_of`), the lists are the facts pass's shapes (`lists_of`), the outputs the caller's, the call a whole
number of 64-token chunks and at most `kMaxHeads` query heads a write head, and launches
`den_kernel<true>` (both sides dense) or `den_kernel<false>` (the list form) on grid `(tiles_s x tiles_r, BH)` with
the tenants' bytes of dynamic shared memory, the attribute raised once.
