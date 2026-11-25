# `carry/carry_values.cuh` — the carry's value lines

Mirrors `csrc/fsla/src/carry/carry_values.cuh`: a fold warp's own supply of its group's value lines
([`../../architecture/hopper-carry.md`](../../architecture/hopper-carry.md#what-a-cta-makes)), the second of the
skeleton's components ([`carry_kernel.md`](carry_kernel.md)), fed by the listing ([`carry_list.md`](carry_list.md)).
No role calls it yet; the lab checks and times it with the listing
([`../../lab/carry_fetch.md`](../../lab/carry_fetch.md)).

- <a id="tile"></a>A ROUND'S TILE IS 32 ROWS, a token's value line each as it lies in global memory: 128 bytes, its
  eight 16-byte pieces in the order the tensor step's 128-byte swizzle reads a row
  ([`../common/atoms/tensor.md`](../common/atoms/tensor.md#rows)). WHY ROWS: an asynchronous copy is cheap where an
  instruction moves whole 128-byte rows at both ends, in any order within a row (2.7 cycles a line), and six times
  that where a row is in pieces at either end (16.5: the step's unswizzled brick layout, tried two ways) --
  [`../../lab/value_fetch.md`](../../lab/value_fetch.md). The step reads such rows and computes what it computes from
  bricks ([`../../lab/step_swizzle.md`](../../lab/step_swizzle.md)). A step of the round is 16 consecutive rows
  (`round_step`).
- <a id="fetch"></a>`fetch_round<CHANNELS>(tiles, values, token)` copies a round of the listing into its 32 rows, 4
  consecutive tokens an instruction, so every instruction moves 4 whole rows. A PLACE PAST THE LIST'S TOTAL COPIES
  NOTHING AND ITS ROW IS ZERO (`stage_tile_if`), so a short step needs no clearing and no branch, and a pad's value is
  finite whatever the tile held. THE ROUND IS ONE COMMITTED GROUP: its caller waits for it, and may hold a second
  round in flight in the rows beside it. A value line of another width than 64 channels is not a row of 128 bytes:
  the header does not compile for it (NOT BUILT: the den's channel, a narrower value).

THE BUDGET, stated before the code ([`../../KERNEL_STANDARDS.md` section 19](../../KERNEL_STANDARDS.md#s19)): a round
of 32 tokens, a warp -- 8 copy instructions a lane, each behind a shuffle and an address, about 50 instructions a
lane; 4 KB landed. The floor is the copies' own with rows whole at both ends, 2.55 to 3.80 cycles a line a CTA at 16
warps ([`../../lab/value_fetch.md`](../../lab/value_fetch.md)). Shared memory: a round's 4 KB, and a second round's
while the first is read. Registers: the round's token.
