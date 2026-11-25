# The prepared masks' place (`csrc/fsla/src/common/plan/mask_prepare_layout.h`)

Where [the mask-preparation pass](../../mask_prepare/mask_prepare.md) writes a level-side's group words and totals, and where the carry (its CTAs, both sides) reads them. Pure, constexpr and total, so the host sizes the workspace and the
device addresses a block from the same arithmetic (`FSLA_CARVE_FN`, as `cta_carve.h`). It is sized from the call's shape alone, for the group size that needs the most: THE HOST NEVER READS A DENSITY
([`prefill_launch.md`](prefill_launch.md)).

Symbols: a window is the card's (`window_of_cc`, [`levels.md`](levels.md); tokens), `words` mask words of 32 tokens, an ARGUMENT of every function here and a template constant of the kernel (the hopper card's window is `kHopperWindow` = 1,024 tokens, 32 words); a level-side is `slot = side * kCarveMaxLevels + level`, `kLevelSides` = 8 of them, side 0 the read side; `width` a level-side's digits, `groups = width / g` for its group `g` in digits.

## The layout

- A BLOCK is a (batch-head, window) of one level-side, `prepared_block_bytes(width_bits, words)` bytes WHATEVER ITS GROUP: the larger of `prepared_words_at(1 << width_bits)` (a total a digit, which is what a group of one digit needs) and, where `width_bits >= 2`, `prepared_words_at(G) + 4 * words * G` with `G = 1 << (width_bits - 2)` (groups of 4 digits: the totals, then a word a mask word a group). The block fits every group. At 256 digits and 32 words it is 8,320 bytes.
- A LEVEL-SIDE's blocks are in (batch-head, window) order, batch-heads as `B * H` with `H` the side's heads inner (the write side has its own, fewer or equal); level-sides are in slot order and a level-side with no mask takes nothing (`prepared_layout`: `offset[s]` and the total `bytes`, from each side's `width_bits`, its batch-heads (0 where it has no mask), the call's windows and the window's `words`).
- IN A BLOCK of `groups` groups: `uint16 totals[groups]` at byte 0 (the window's live tokens of the group), then
  `uint32 words[words][groups]` at `prepared_words_at(groups)` = `(2 * groups + 3) & ~3`, word-major (the OR of the group's digits' words `j`; NONE at a group of one digit, where the word is the mask's own).
- NO PER-WORD COUNT IS STORED: a carry CTA counts inline, a popcount and a running sum over the window's words, and the total is what a reader of the window's end needs.
- The bytes of a block the layout does not use at the level-side's group are UNSPECIFIED, and a level-side at density byte 255 (dense) is not written: its region keeps what it held.

A group of one digit stores a total a digit alone (its words are the mask's own); a group of 4 digits or more stores a word a group, at most 1 byte a mask word. `mask_prepare_bytes` is the layout's size from a call's widths, batch-heads, words and the card's window (`mask_prepare_bytes(width_bits, heads, words, device)`: the op derives the window from the device's card).
