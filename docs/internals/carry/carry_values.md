# `carry/carry_values.cuh` — the carry's value lines

Mirrors `csrc/fsla/src/carry/carry_values.cuh`: a fold warp's own supply of its group's value lines
([`../../architecture/hopper-carry.md`](../../architecture/hopper-carry.md#what-a-cta-makes)), the second of the
skeleton's components ([`carry_kernel.md`](carry_kernel.md)), fed by the listing ([`carry_list.md`](carry_list.md)).
The fold's supply calls it ([`carry_kernel.md`](carry_kernel.md#supply)); the lab checks and times it with the listing
([`../../lab/carry_fetch.md`](../../lab/carry_fetch.md)).

- <a id="tile"></a>A ROUND'S TILE IS 32 ROWS, a token's value line each as it lies in global memory: 128 bytes, its
  eight 16-byte pieces in the order the tensor step's 128-byte swizzle reads a row
  ([`../common/atoms/tensor.md`](../common/atoms/tensor.md#rows)). WHY ROWS: an asynchronous copy is cheap where an
  instruction moves whole 128-byte rows at both ends, in any order within a row (2.7 cycles a line), and six times
  that where a row is in pieces at either end (16.5: the step's unswizzled brick layout, tried two ways) --
  [`../../lab/value_fetch.md`](../../lab/value_fetch.md). THE STEP READS SUCH ROWS, in the time it reads
  bricks ([`../../lab/step_swizzle.md`](../../lab/step_swizzle.md)). A step of the round is 16
  consecutive rows (`round_step`), and THE TILE STARTS ON A 1,024-BYTE BOUNDARY, the condition of the step's operand
  ([`../common/atoms/tensor.md`](../common/atoms/tensor.md#rows)).
- <a id="fetch"></a>`fetch_round<CHANNELS>(tiles, values, stride, tokens, live)` issues, uncommitted ([`carry_kernel.md`](carry_kernel.md#supply): the stage commits them with its bytes'), the copies of a round of the warp's list -- where `live` is false (a set with no round after) every place copies nothing and its row is written zero, as a place past the list's total is -- (`tokens`, its 32 places, [`carry_list.md`](carry_list.md#lay)) into its 32 rows -- a token's
  line `stride` bytes on from the token before's, its first 128 bytes copied, so a CTA's slice of a wider value starts
  `values` at its slice -- 4
  consecutive tokens an instruction, so every instruction moves 4 whole rows. A PLACE PAST THE LIST'S TOTAL COPIES
  NOTHING AND ITS ROW IS ZERO (`stage_tile_if`), so a short step needs no clearing and no branch, and a pad's value is
  finite whatever the tile held. THE ROUND'S 8 TOKENS ARE LOADED FIRST, then its 8 copies issued: a warp issues in order, and a token loaded where it is used holds the warp until it lands, eight times a round. A copy's row is 4 rows on from the copy before's, so the rows of every other copy turn their pieces alike under the swizzle: a lane forms two places a round (`step_row_piece`), a copy's destination is one add, and its source a 32-bit multiply (a line's place in a window fits 32 bits). THE ROUND'S COPIES JOIN THEIR STAGE'S COMMITTED GROUP: the stage's issuer commits and waits for it, and may hold a second
  round in flight in the rows beside it. A value line of another width than 64 channels is not a row of 128 bytes:
  the header does not compile for it (NOT BUILT: the den's channel, a narrower value).

THE BUDGET, stated before the code ([`../../KERNEL_STANDARDS.md` section 19](../../KERNEL_STANDARDS.md#s19)): a round
of 32 tokens, a warp -- 8 shared loads of its tokens and 8 copy instructions a lane, each behind a multiply and two adds, about 35 instructions a
lane; 4 KB landed. The floor is the copies' own with rows whole at both ends, 2.55 to 3.80 cycles a line a CTA at 16
warps ([`../../lab/value_fetch.md`](../../lab/value_fetch.md)); MEASURED in the select-fed form the lab's `carry_fetch` keeps, with the listing, 2.96 a line at 64 live tokens a
window and 2.10 with every token live ([`../../lab/carry_fetch.md`](../../lab/carry_fetch.md)). Shared memory: a round's 4 KB, and a second round's
while the first is read. Registers: the round's 8 tokens and a lane's two row places.
