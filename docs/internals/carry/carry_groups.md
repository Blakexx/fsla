# `carry/carry_groups.cuh` — a call's operands and a group's words

Mirrors `csrc/fsla/src/carry/carry_groups.cuh`: what a role of the carry
([`carry_kernel.md`](carry_kernel.md)) reads its groups from. No role calls it yet.

- <a id="operands"></a>`Operands`: the call as the kernel is handed it -- each level-side's mask (null where it has
  none: in the bundle's slot order, the read side's four, then the write side's, `side_at`), the workspace and each level-side's place in it for
  the prepared words ([`../common/plan/mask_prepare_layout.md`](../common/plan/mask_prepare_layout.md)), the density
  bytes, the values (a head's tokens' lines one after another, `line_bytes` a token), the call's tokens a head, the leaf
  CTAs a head (`ctas`, the host's count), the levels' width bits, and the mask's words and windows. The host fills it from the call's shape and reads
  no density.
- <a id="carve"></a>`carve_of(operands)`: THE KERNEL DERIVES THE CARVE, from the densities as the mask-preparation
  pass reads them (a byte `b` is `b + 1` 256ths), by the one rule the planner states
  ([`../common/plan/cta_carve.md`](../common/plan/cta_carve.md)); that rule is written so that a kernel's thread can
  run it in registers ([`../common/plan/cta_carve.md#register-shaped`](../common/plan/cta_carve.md#register-shaped)).
- <a id="whole"></a>`whole(carve, read, level)`: a level-side is WHOLE where ITS GROUP IS THE LEVEL'S SIZE, and that
  is the carry's one test -- the owner: "use the group size equals level as the check, and that check is all you
  need in carry, you don't even need to look at the mask activation flag at all, which shouldn't even reach carry in
  the first place". A group the size of the level is every token whatever the mask says ("mathematically ignoring
  the mask has no difference if the group is level size"): the level-side adds nothing to a role's AND, its mask is
  never read, and the mask-preparation pass prepares nothing for it
  ([`../mask_prepare/mask_prepare.md`](../mask_prepare/mask_prepare.md)). A level-side with no mask is whole by its
  density, a byte of 255 by the declaration ([`../../api.md`](../../api.md) section 2.6); a density below the whole
  with a mask of all ones is read as it is given. IT IS THE CALL'S FACT, asked once a level-side a call: a role's
  AND runs over the levels that are not whole.
- <a id="window-word"></a>`window_word<WORDS>(operands, window, word)`: the call's own tokens in a word of a window,
  the first of 32 where the call ends inside it. A role's AND starts from it, so that a side whose levels are all whole
  lists the call's tokens and no place past its last.
- <a id="word"></a>`group_word<WORDS>(operands, level-side, group bits, head, window, word, group)`: a lane's word of a
  group's window on a level-side that is not whole -- at a group of one digit the mask's own word of the digit,
  otherwise the prepared word ([`../common/plan/mask_prepare_layout.md`](../common/plan/mask_prepare_layout.md)). NO
  BRANCH: both addresses are formed and one is read, because a role calls it inside its loop over sets, where a branch
  on a loaded value costs every lane a join ([`../../lab/collective_branch.md`](../../lab/collective_branch.md)). A
  word past the mask's last reads the mask's first word and counts for nothing, the mask-preparation pass's own way,
  so nothing is read past the mask.
- `listed_levels(carve, read)`: the levels a side lists, a bit a level -- those that are not whole. A role takes it
  once a call as a value every lane holds and `set_word` branches on that.
- <a id="place"></a>`place_of(operands, carve, at, slices)`: A CTA'S PLACE IN THE CALL from its index in the grid: its
  slice of the value channels (the slices of one leaf carve are neighbours in the grid, having one token schedule),
  its head, and its index along each level in the order the schedule takes a head's CTAs
  ([`../common/plan/cta_carve.md`](../common/plan/cta_carve.md#schedule)).
- <a id="sets"></a>`fold_groups(carve, place, fold, set, group)`: A FOLD'S SETS. A fold's box is the CTA's carve, halved
  along the split level; a set is one write group a level inside it, and `group[l]` its index among level `l`'s groups.
  A level whose group is no smaller than the fold's extent gives every set the one group the box lies in (a group may
  span folds and CTAs); a level whose group is smaller gives the set's index its bits, the write side's densest level
  varying fastest. The sets' leaves tile the leaf space once, which the lab checks on the host
  ([`../../lab/carry_supply.md`](../../lab/carry_supply.md)). EACH OF THE CARVE'S ARRAYS IS READ BEFORE THE LEVEL IS
  COMPARED WITH ANYTHING: written as `l == split_level ? fold << fold_bits[l] : 0`, the read was compiled as one
  indexed by the split level at run time, which put the whole carve in the kernel's stack frame (208 bytes; the lab's
  checker refused the kernel for it).
- <a id="set-word"></a>`set_word<WORDS>(operands, carve, place, fold, set, window, word, listed)`: a lane's word of a
  set's window on the write side -- the call's own tokens (`window_word`), ANDed with the set's group's word on each
  level the side lists.
