# `carry/carry_groups.cuh` — a call's operands and a group's words

Mirrors `csrc/fsla/src/carry/carry_groups.cuh`: what a role of the carry
([`carry_kernel.md`](carry_kernel.md)) reads its groups from. No role calls it yet.

- <a id="operands"></a>`Operands`: the call as the kernel is handed it -- each level-side's mask (null where it has
  none: in the bundle's slot order, the read side's four, then the write side's, `side_at`), the workspace and each level-side's place in it for
  the prepared words ([`../common/plan/mask_prepare_layout.md`](../common/plan/mask_prepare_layout.md)), the density
  bytes, the call's tokens, the levels' width bits, and the mask's words and windows. The host fills it from the call's shape and reads
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
- <a id="word"></a>`group_word<WORDS>(operands, level-side, group bits, head, window, word, group)`: a lane's word of
  a group's window, ONE FUNCTION FOR EVERY GROUP SIZE so that a role's AND across a side's levels is the same code
  whatever each is, for a level-side that is not whole: zero past the mask's last word (a ragged last window); the
  mask's own word at a group of one digit, which the pass stores no words for; the prepared group word otherwise.
