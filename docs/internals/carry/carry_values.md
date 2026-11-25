# `carry/carry_values.cuh` — the carry's value lines

Mirrors `csrc/fsla/src/carry/carry_values.cuh`: a fold warp's own supply of its group's value lines
([`../../architecture/hopper-carry.md`](../../architecture/hopper-carry.md#what-a-cta-makes)), the second of the
skeleton's components ([`carry_kernel.md`](carry_kernel.md)), fed by the listing ([`carry_list.md`](carry_list.md)).
No role calls it yet; the lab checks and times it with the listing
([`../../lab/carry_fetch.md`](../../lab/carry_fetch.md)).

- <a id="tile"></a>A STEP'S VALUE TILE. The tensor step reads its 16 tokens' values as bricks of 8 channels, a line
  (16 bytes) one token's 8 channels of a brick, a brick's 16 tokens consecutive lines
  ([`../common/atoms/tensor.md`](../common/atoms/tensor.md#operand)); the header lays the bricks 17 lines apart
  (`kBrickPitch`). THIS LAYOUT IS NOT THE FORM: on a device a line whose 8 pieces land apart costs 16.6 cycles against
  2.70 for a line that lands as its 128 bytes whole, in any order of its pieces
  ([`../../lab/value_fetch.md`](../../lab/value_fetch.md)); the banks its lanes fall on, which the pitch was chosen
  by, are not what an asynchronous copy's write costs. The header stays as measured until a tile of whole rows is
  shown to be one a step reads ([`../../lab/step_swizzle.md`](../../lab/step_swizzle.md)).
- <a id="fetch"></a>`fetch_round<CHANNELS>(tiles, values, token)` copies a round of the listing, 32 places, into two
  steps' tiles. A PLACE PAST THE LIST'S TOTAL COPIES NOTHING AND ITS LINES ARE ZERO (`stage_tile_if`), so a short step
  needs no clearing and no branch, and a pad's value is finite whatever the tile held. THE ROUND IS ONE COMMITTED
  GROUP: its caller waits for it, and may hold a second round in flight in the tiles beside it.

THE BUDGET, stated before the code ([`../../KERNEL_STANDARDS.md` section 19](../../KERNEL_STANDARDS.md#s19)): a round
of 32 tokens, a warp -- 8 copy instructions a lane, each behind a shuffle and an address, about 50 instructions a
lane; 4 KB landed. ITS FLOOR WAS TAKEN FROM THE WRONG FORM: 2.55 to 3.80 cycles a line is a line landing whole, and
this layout's own price is 16.6 ([`../../lab/value_fetch.md`](../../lab/value_fetch.md)); with the listing, 13.3 to
13.9 ([`../../lab/carry_fetch.md`](../../lab/carry_fetch.md)). Shared memory: two tiles a round, and a second round's
while the first is read. Registers: the round's token.
