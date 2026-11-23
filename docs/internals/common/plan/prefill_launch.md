# The prefill launch function's carve (`csrc/fsla/src/common/plan/prefill_launch.h`)

`prefill_carve(const Descriptor&)` returns a `PrefillCarve`: how a prefill call's levels fall into boxes of 256 leaves, each side's roles in a box, the CTA's pair of
boxes, the level its two folds split along, the counts of a batch-head and the window. It is the first half of the prefill launch function
([`../../../architecture/hopper-carry.md#the-order-of-the-build`](../../../architecture/hopper-carry.md#the-order-of-the-build), step 1), written like
[`decode_launch.md`](decode_launch.md)'s args half: from EXACT facts only (a range aborts), reading `levels`, `width_bits_1..4`, `read_listed_1..4`,
`write_listed_1..4` (through the per-level lookups of [`levels.md`](levels.md)), `tokens` and `cc`, and deciding nothing from data (R24: the op reads the descriptor and the masks, never a flag of the data).
`prefill_carves(d)` is whether the call has a carve; `prefill_carve` aborts on one that has none. The prefill stays an OWED descriptor
([`../../hardware/registry.md#the-owed-folder`](../../hardware/registry.md#the-owed-folder)): no arm is declared here, and which carves have a built kernel is the arms' business.

Levels are 0-based. A level is LISTED on a side when that side's `*_listed` fact is 1 (the side walks it by its lists), WHOLE when 0.

## The rules as built

- R1, THE LEVELS IN A BOX (`prefill_box_levels`; CONSTITUTION R10, every pattern at every depth has a carve). A box is 256 leaves: 8 address bits; `box_bits[l]` is level `l`'s
  bits in a box and they sum to 8. A box's two halves of 4 bits are each drawn from a level (`prefill_box_bits`): the first level the write side walks by its lists and
  the first it walks whole, or the first two levels when the write side has not both; every other level has 0. A call of one level draws both halves from it (`{8}`).
- R2, EACH SIDE'S ROLES. Of a box's 8 bits a side takes 4 as LIST bits (16 lists) and 4 as COLUMN bits (16 columns): `read_list_bits[l]` and `write_list_bits[l]` (4 on the
  list level, 0 elsewhere; at one level 4 of the level's 8), the column bits `box_bits[l] - list_bits[l]`. One level: the level's 4 high bits are list bits. Two box levels
  `lo < hi`: when exactly one is listed on the side, it is the side's list level and the other its column level; when both or neither, `lo` is the list level
  and `hi` the column level.
- R3, THE PAIR (CONSTITUTION R20: the region gives the read side its larger extent along a level it walks whole, so a CTA's two boxes neighbour along one). Boxes along level `l` are
  `2^(width_bits[l] - box_bits[l])`, so a level outside the box (0 bits) has `2^width_bits[l]` of them. `pair_level` is the first that holds: (1) the read side's column level, if
  it walks it whole with 2 or more boxes; (2) the lowest other level it walks whole with 2 or more boxes (its list level where that is whole, or a level outside the box);
  (3) where no level it walks whole has 2 boxes, its column level if that has 2 or more boxes, else its list level if that has 2 or more, else the lowest other level with 2 or more;
  (4) else -1 (a call of one level has that level in both roles). `boxes_per_cta` is 2 with a pair level, else 1. A pair level outside the box's levels doubles neither the lists nor
  the columns: the CTA's two boxes are the same box digits at two neighbouring digits of that level.
- R4, THE FOLDS' SPLIT (CONSTITUTION R20: along a write-sparse level first, otherwise write dense). `split_level` is the write side's list level when the write side walks
  it by its lists, else the pair level (the write side's list level where there is no pair). The CTA's region is 16 digits of each level in the box, 32 along the pair
  level; a fold is half the region along the split level. `fold_lists` and `fold_columns` are the fold's extents along the write side's list and column levels; `fold_outer` is its
  extent in boxes along a pair level that is not one of the box's two levels: 2 where there is one and the folds do not split along it, else 1. A fold's leaves are
  `fold_lists * fold_columns * fold_outer`: 256 with a pair, 128 without.
  Where the write side's lists and columns are one level's bits (a call of one level), the pair and the split take its lists, the high bits: 16 and 16 with a pair, 8 and 16
  without. So a split along the pair level gives 16 by 16, a split off it with a pair 8 by 32 (a pair outside the box: 8 by 16, outer 2), no pair 8 by 16.
- R5, THE COUNTS of a batch-head. `boxes_per_head` is the product over levels of the boxes along each; `ctas_per_head = boxes_per_head / boxes_per_cta`;
  `units_per_head` is the product, over every level the write side walks by its lists, of the CTA positions along it (the boxes along it, halved where it is the pair level),
  1 where it lists none. A UNIT is the CTAs that share a write-listed position: all the readers of its value streams
  ([`../../../architecture/hopper-carry.md#the-packets`](../../../architecture/hopper-carry.md#the-packets)). `unit_ctas = ctas_per_head / units_per_head`.
- R6, THE WINDOW. `window` is `kHopperWindow` (1024) at `cc` 900, any other `cc` aborts; it equals `build.window` of `common/hardware.json`'s sm_90a, which only the JSON holds, so
  `tests/unit/test_prefill_carve.py` asserts the equality. `windows = ceil(tokens / window)`.

NOT CARVED (`prefill_carves`): a call with a level narrower than its bits in a box -- one level under 256 digits, or a box level under 16. Nothing else is refused.
THE CONTRACT ACCEPTS SUCH CALLS: `hardware/parts/calls/depth1.json` takes one level of 16 to 256 digits and only 256 has a carve. A box of fewer leaves, or a narrower
contract, is owed ([`../../../open-work.md`](../../../open-work.md) row A-f).

## The table (widths 256 x 256, `cc` 900, `tokens` 65,536: window 1024, 64 windows)

| write | read | write list lvl | read list lvl | pair | split | fold lists x cols | outer | boxes | CTAs | units | unit CTAs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1,0 | 0,1 | 0 | 1 | 0 | 0 | 16 x 16 | 1 | 256 | 128 | 8 | 16 |
| 0,1 | 1,0 | 1 | 0 | 1 | 1 | 16 x 16 | 1 | 256 | 128 | 8 | 16 |
| 1,0 | 1,0 | 0 | 0 | 1 | 0 | 8 x 32 | 1 | 256 | 128 | 16 | 8 |
| 1,1 | 1,1 | 0 | 0 | 1 | 0 | 8 x 32 | 1 | 256 | 128 | 128 | 1 |
| 0,0 | 0,0 | 0 | 0 | 1 | 1 | 16 x 16 | 1 | 256 | 128 | 1 | 128 |

Every row has box bits 4,4. One level of 65,536 digits: box bits 8, 256 boxes paired along level 0 (128 CTAs), split 0, 16 x 16; write listed 128 units of one CTA, write whole one unit of 128. `tests/unit/test_prefill_carve.py` holds these and the depth-3 and depth-4 cells (the pair on a level outside the box: 8 x 16, outer 2), the pair-falls-to-the-list-level cell and a cell for each of rule 3's clauses, each worked by hand in its comments and each asserting the fold's leaves.

## What is NOT here

The grid's split between the gather and the carry CTAs, the ring's size, the workspace's layout and the segment length wait on
[the hand-off's design](../../../architecture/hopper-carry.md#the-hand-off) and stay owed ([`../../../open-work.md`](../../../open-work.md) rows A-f and A-g). No Python API and no op:
`fsla.ops.prefill` keeps refusing by name. `plan_tool` prints a prefill call's carve as JSON ([`plan_tool.md`](plan_tool.md)).
