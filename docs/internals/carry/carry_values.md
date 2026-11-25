# `carry/carry_values.cuh` — the carry's value lines

Mirrors `csrc/fsla/src/carry/carry_values.cuh`: a fold warp's own supply of its group's value lines
([`../../architecture/hopper-carry.md`](../../architecture/hopper-carry.md#what-a-cta-makes)), the second of the
skeleton's components ([`carry_kernel.md`](carry_kernel.md)), fed by the listing ([`carry_list.md`](carry_list.md)).
No role calls it yet; the lab checks and times it with the listing
([`../../lab/carry_fetch.md`](../../lab/carry_fetch.md)).

- <a id="tile"></a>A STEP'S VALUE TILE. The tensor step reads its 16 tokens' values as bricks of 8 channels, a line
  (16 bytes) one token's 8 channels of a brick, a brick's 16 tokens consecutive lines
  ([`../common/atoms/tensor.md`](../common/atoms/tensor.md#operand)). THE BRICKS ARE 17 LINES APART, one more than
  they hold (`kBrickPitch`), so that a token's pieces fall on different banks: a token's 128 bytes arrive as 8 copies
  of 16 bytes, one a lane, and at a pitch of a whole number of 128 bytes all 8 would write the same four banks. With
  one instruction's 32 lanes copying 4 consecutive tokens, the 17-line pitch puts exactly four of them on every bank
  -- what 512 contiguous bytes would ([`../../KERNEL_STANDARDS.md` section 30](../../KERNEL_STANDARDS.md#s30)). Whether
  an asynchronous copy's write pays for a bank it shares is NOT MEASURED; the layout does not depend on it.
- <a id="fetch"></a>`fetch_round<CHANNELS>(tiles, values, token)` copies a round of the listing, 32 places, into two
  steps' tiles. A PLACE PAST THE LIST'S TOTAL COPIES NOTHING AND ITS LINES ARE ZERO (`stage_tile_if`), so a short step
  needs no clearing and no branch, and a pad's value is finite whatever the tile held. THE ROUND IS ONE COMMITTED
  GROUP: its caller waits for it, and may hold a second round in flight in the tiles beside it.
- THE LAYOUT'S BANKS ARE HELD BY A TEST (`tests/unit/test_carry_values_layout.py`): every instruction of a round puts
  exactly four of its 32 lanes on each bank, at this pitch and not at a pitch of 16 lines. What the layout costs on a
  device is the lab's ([`../../lab/value_fetch.md`](../../lab/value_fetch.md), its row of bricks 17 lines apart).

THE BUDGET, stated before the code ([`../../KERNEL_STANDARDS.md` section 19](../../KERNEL_STANDARDS.md#s19)): a round
of 32 tokens, a warp -- 8 copy instructions a lane, each behind a shuffle and an address, about 50 instructions a
lane; 4 KB landed. The floor is the copies' own, measured alone at 16 warps: 2.56 to 3.79 cycles a 128-byte line a
CTA ([`../../lab/value_fetch.md`](../../lab/value_fetch.md)). Shared memory: two tiles a round, and a second round's
while the first is read. Registers: the round's token.
