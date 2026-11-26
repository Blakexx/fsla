# The leaf groups' masks' place (`csrc/fsla/src/common/plan/mask_prepare_layout.h`)

Where [the mask-preparation pass](../../mask_prepare/mask_prepare.md) writes a side's leaf groups' mask words and counts, and where the carry (its CTAs, both sides) reads
them. Pure, constexpr and total, so the host sizes the workspace and the device addresses a block from the same arithmetic (`FSLA_CARVE_FN`, as `cta_carve.h`). THE HOST
NEVER READS A DENSITY ([`prefill_launch.md`](prefill_launch.md)): the workspace is sized for the most leaf groups the call's shape allows
([`leaf_groups.md`](leaf_groups.md#most)), and the kernels, which derive the call's carve, use the front of it.

Symbols: a window is the card's (`window_of_cc`, [`levels.md`](levels.md); tokens), `words` mask words of 32 tokens, an ARGUMENT of every function here and a template
constant of the kernel (the hopper card's window is `kHopperWindow` = 1,024 tokens, 32 words); a SIDE is 0, the read side, or 1, the write side (`kCarveSides` = 2); a
side has `1 << bits` LEAF GROUPS, `bits` the carve's ([`leaf_groups.md`](leaf_groups.md#order)), in the order that page gives; `most` is the most `bits` the call's shape
allows the side (`leaf_group_bits_most`).

## What a leaf group holds

- ITS LINE: `words` mask words of the window, a bit a token. A token's bit is set where the token is one of THE CALL'S OWN (its index is under the call's tokens) AND, on
  every level the side lists, it is live on a digit of the leaf group's group of that level. A level the side walks whole asks nothing. A side that lists no level has
  ONE leaf group, whose line is the call's own tokens.
- ITS COUNT: the line's set bits, the leaf group's live tokens of the window, a uint16.

A level is LISTED by a side where its group is smaller than the level (`group_bits[l] < width_bits[l]`); a group's digits are `group << group_bits[l]` to
`((group + 1) << group_bits[l]) - 1`.

A LEVEL-SIDE WITH NO MASK IS WALKED WHOLE, WHATEVER ITS DENSITY BYTE. The mask's absence is the dense flag, and every kernel forms the carve with that level-side's
density at the whole (256 256ths) before anything else: the carve's groups, the leaf groups and their count are the same in the pass and in the carry, and no density
byte can ask for more leaf groups than the workspace, which is sized from the masks that are there, holds. So a listed level always has a mask. THE PASS ALWAYS WRITES, a
dense side as one leaf group of the call's own tokens, and the kernels are told which level-sides have masks (a bit a level a side): the owner, 2026-11-26, to the account of
what is built against his assumption that a dense side does nothing in the pass and the carry reads no mask and takes it as dense: "ok thats fine makes sense"
([`../../mask_prepare/mask_prepare.md`](../../mask_prepare/mask_prepare.md)).

## The layout

- A BLOCK is a (batch-head, window) of one side, `leaf_block_bytes(bits, words)` bytes at the side's `bits`: `uint16 counts[1 << bits]` at byte 0, then, at
  `leaf_lines_at(bits, words)` -- the counts' bytes rounded up to a whole number of lines -- `uint32 lines[1 << bits][words]`, a leaf group's line `leaf_line_bytes(words)`
  = `4 * words` bytes (128 on the hopper card), leaf groups in index order. At 64 leaf groups and 32 words a block is 128 + 64 x 128 bytes; at 4,096, 8,192 + 4,096 x 128.
- A SIDE's blocks are in (batch-head, window) order, batch-heads as `B * H` with `H` the side's heads inner (the write side has its own, fewer or equal), ONE BLOCK AFTER
  ANOTHER AT THE SIDE'S OWN `bits`: the stride is the carve's, which a kernel derives and the host does not know.
- THE SIDES' REGIONS are in side order, the read side's first, each sized at its `most`: `prepared_layout(levels, width_bits, masked, heads, windows, words)` gives
  `offset[side]` and the total `bytes`, from the call's level widths, each side's masked levels (`masked[side]`, a bit a level it has a mask on), each side's batch-heads,
  the call's windows and the window's `words`. A side's region is `heads[side] * windows * leaf_block_bytes(most, words)` bytes, of which its blocks fill the first
  `heads[side] * windows * leaf_block_bytes(bits, words)`; the rest is not written.
- A window past the call's last token has no own token: its lines are zero and its counts zero. A word of the mask past the mask's last reads as zero.

`mask_prepare_bytes(widths, masked, heads, tokens, device)` is the layout's size from a call's level widths, each side's masked levels and batch-heads, its tokens a head
and the card's window (the op derives the window from the device's card).

## Proofs

`layout_proof` holds the header's static assertions: one leaf group is a line of counts and its line (256 bytes at 32 words); the cell's 64 leaf groups' counts fill one
line; 4,096 leaf groups have 8,192 bytes of counts.
