# A side's leaf groups (`csrc/fsla/src/common/plan/leaf_groups.h`)

How many leaf groups a call's carve gives a side, the order they are laid in, and the most a call's shape allows whatever its densities. Pure, constexpr and total, so the
host and the device count and index them by the same arithmetic (`FSLA_CARVE_FN`, as [`cta_carve.h`](cta_carve.md)). Its readers: the mask-preparation pass, which
writes a leaf group's mask words and count ([`mask_prepare.md`](../../mask_prepare/mask_prepare.md)), its layout ([`mask_prepare_layout.md`](mask_prepare_layout.md)) and
the carry's fold, whose sets are consecutive leaf groups (the owner, 2026-11-25: "an actual per group mask with counts would work too"; "we can concer mask prep pass
into just materializing these group masks").

Symbols: a level's GROUP is `1 << group_bits[l]` of its digits, the side's (`write_group_bits`, `read_group_bits` of the carve); a level `width_bits[l]` wide has
`1 << (width_bits[l] - group_bits[l])` groups along it; a BOX is what one consumer holds of a level -- a fold on the write side (`fold_bits`), the CTA on the read side
(`carve_bits`); a SET is a group inside a box.

## <a id="order"></a>Order

A LEAF GROUP is one group on each of the side's levels. A side has `1 << bits` of them, `bits` the sum over its levels of `width_bits[l] - group_bits[l]`: a level walked
whole is one group and adds nothing.

`leaf_groups(carve, read)` gives the order. A level's group index is cut in two fields of the leaf group's index:

- its INNER bits, `inner[l] = max(0, box_bits[l] - group_bits[l])`: the low bits of the group index, those that vary inside one box. They are placed at `inner_at[l]`, the
  side's densest level's lowest (the side's loop order, innermost first), and together they are the index's low `inner_bits`.
- its OUTER bits, `outer[l]`, the rest of the group index: which box, or which run of boxes that share the group. They are placed above every inner field, at
  `outer_at[l]`, the lowest level's lowest.

WHY THIS ORDER: a box's sets are then CONSECUTIVE leaf groups. A fold (or the CTA, on the read side) holds `1 << inner_bits` sets -- the carve's `fold_sets` and
`read_sets` -- and its set `i` is leaf group `(outer << inner_bits) + i`, `outer` its levels' outer fields, formed once. The inner fields' places are the places the fold
gives a set's bits a level (`fold_levels` of [`carry_groups.cuh`](../../carry/carry_groups.md), the write side's densest level's lowest), so a set's number IS the low bits.

`leaf_group_of(g, group)` forms a leaf group's index from one group index a level, and `leaf_group_levels(g, leaf, group)` undoes it: one to one over the side's leaf groups.

## <a id="most"></a>Most

`leaf_group_bits_most(levels, width_bits, masked)` is the most leaf groups a side can have, as log2, from the call's level widths and the levels the side has a mask on
(`masked`, a bit a level) -- no density. It is the smaller of two bounds:

- THE LISTED LEVELS' DIGITS. A level-side with no mask is walked whole, whatever its density byte ([the layout](mask_prepare_layout.md#what-a-leaf-group-holds)): one
  group. The side has at most a group a digit on every level it has a mask on.
- THE LEAVES OVER 16. A set is raised to 16 leaves wherever a box holds as many (`carve_least_sets`), and a leaf group holds its set, so a leaf group is 16 leaves at
  least. A fold holds 16 leaves in every call of 32 leaves or more; a smaller call is bounded by its leaves.

Two levels of 256 digits: 4,096 leaf groups at most with both masked, 256 with one, one with neither. It is what a workspace sized before any density is known must hold,
and what the layer's memory handle prices a segment at (the owner, 2026-11-25: "we assume the group size is the minimum of 16 right, and so the number of groups per side
is N/16"; "if you know a few levels are dense, that can be info for the planner? And it will still assume minimum group for the rest?").

## Proofs

`leaf_proof` holds the header's static assertions: the measured cell has 64 leaf groups a side; two levels at 4 live digits of 256 have 4,096; a leaf group's index and its
levels' groups are one to one on three carves; the most of 65,536 leaves is 4,096, 256 and 1. `tests/unit/test_leaf_groups.py` runs the properties over the calls of depth
1 to 4 through `plan_tool --cta-carve`, which prints each side's leaf groups beside the carve.
