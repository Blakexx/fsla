# `carry/carry_groups.cuh` — a call's operands and a group's words

Mirrors `csrc/fsla/src/carry/carry_groups.cuh`: what a role of the carry
([`carry_kernel.md`](carry_kernel.md)) reads its groups from. No role calls it yet.

- <a id="operands"></a>`Operands`: the call as the kernel is handed it -- each level-side's mask (null where it has
  none: the write side's four, then the read side's, `side_at`), the workspace and each level-side's place in it for
  the prepared words ([`../common/plan/mask_prepare_layout.md`](../common/plan/mask_prepare_layout.md)), the density
  bytes, the levels' width bits, and the mask's words and windows. The host fills it from the call's shape and reads
  no density.
- <a id="carve"></a>`carve_of(operands)`: THE KERNEL DERIVES THE CARVE, from the densities as the mask-preparation
  pass reads them (a byte `b` is `b + 1` 256ths), by the one rule the planner states
  ([`../common/plan/cta_carve.md`](../common/plan/cta_carve.md)); that rule is written so that a kernel's thread can
  run it in registers ([`../common/plan/cta_carve.md#register-shaped`](../common/plan/cta_carve.md#register-shaped)).
- <a id="word"></a>`group_word<WORDS>(operands, level-side, group bits, head, window, word, group)`: a lane's word of
  a group's window, ONE FUNCTION FOR EVERY GROUP SIZE so that a role's AND across a side's masked levels is the same
  code whatever each is: all ones where the level-side has no mask (every token of the call is live in its one
  group); zero past the mask's last word (a ragged last window); the mask's own word at a group of one digit, which
  the pass stores no words for; the prepared group word otherwise.
