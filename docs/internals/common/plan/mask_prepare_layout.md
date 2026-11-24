# The prepared masks' place (`csrc/fsla/src/common/plan/mask_prepare_layout.h`)

Where [the mask-preparation pass](../../mask_prepare/mask_prepare.md) writes a level-side's group words, counts and totals, and where the gather and the carry read them. Pure, constexpr and total, so the host sizes the workspace and the
device addresses a block from the same arithmetic (`FSLA_CARVE_FN`, as `cta_carve.h`). It is sized from the call's shape alone, for the group size that needs the most: THE HOST NEVER READS A DENSITY
([`prefill_launch.md`](prefill_launch.md)).

Symbols: a window is `kMaskWindowWords` = 32 mask words, 1,024 tokens; a level-side is `slot = side * kCarveMaxLevels + level`, `kLevelSides` = 8 of them, side 0 the read side; `width` a level-side's digits, `groups = width / g` for its group `g` in digits.

## The layout

- A BLOCK is a (batch-head, window) of one level-side, `prepared_block_bytes(width_bits)` = `66 * width` bytes WHATEVER ITS GROUP: a total and 32 counts of 2 bytes each a digit, which is what a group of one digit (the most groups) needs, so the block fits every group.
- A LEVEL-SIDE's blocks are in (batch-head, window) order, batch-heads as `B * H` with `H` the side's heads inner (the write side has its own, fewer or equal); level-sides are in slot order and a level-side with no mask takes nothing (`prepared_layout`: `offset[s]` and the total `bytes`, from each side's `width_bits`, its batch-heads (0 where it has no mask) and the call's windows).
- IN A BLOCK of `groups` groups: `uint16 totals[groups]` at byte 0 (the window's live tokens of the group), `uint16 counts[32][groups]` at `prepared_counts_at(groups)` = `2 * groups`, word-major (the group's live tokens of the window BEFORE word `j`), and
  `uint32 words[32][groups]` at `prepared_words_at(groups)` = `(66 * groups + 3) & ~3`, word-major (the OR of the group's digits' words `j`; NONE at a group of one digit, where the word is the mask's own).
- A WINDOW'S TOTAL IS STORED beside its counts (`totals`): a count is of the tokens before a word, so the last word's own live tokens are in no count and the total is what a reader of the window's end needs.
- The bytes of a block the layout does not use at the level-side's group are UNSPECIFIED, and a level-side at density byte 255 (dense) is not written: its region keeps what it held.

A group of one digit stores 2 bytes a mask word, half the level-side's mask; a group of 4 digits or more stores a word and a count a group, at most 1.5 bytes a mask word. `mask_prepare_bytes` is the layout's size from a call's widths, batch-heads and words.
