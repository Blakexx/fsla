# `carry/carry_groups.cuh` — a call's operands and a side's leaf groups' lines

Mirrors `csrc/fsla/src/carry/carry_groups.cuh`: what a role of the carry
([`carry_kernel.md`](carry_kernel.md)) reads its groups from. The fold's supply does ([`carry_kernel.md`](carry_kernel.md#supply)).

- <a id="operands"></a>`Operands`: the call as the kernel is handed it -- the workspace and each side's place in it for
  the leaf groups' lines and counts (`prepared[plan::kCarveSides]`, the read side first,
  [`../common/plan/mask_prepare_layout.md`](../common/plan/mask_prepare_layout.md)), the density bytes, and which level-sides have a mask
  (`masked[plan::kCarveSides]`, a bit a level a side; `plan::kWriteSide` is the write side's index). It holds also the values (a head's tokens' lines one
  after another, `line_bytes` a token), each level-side's rows (`rows`, in the bundle's slot order, the read side's four then the write side's; a head's tiles one
  after another: [`carry_coefs.md`](carry_coefs.md#rows)), the call's tokens a head, the leaf CTAs a head (`ctas`, the host's count), the levels' width bits, and the
  mask's windows. It holds no mask: the carry reads the pass's lines alone. The host fills it from the call's shape and reads no density.
- <a id="carve"></a>`carve_of(operands)`: THE KERNEL DERIVES THE CARVE, from the densities as the mask-preparation
  pass reads them (a byte `b` is `b + 1` 256ths), by the one rule the planner states
  ([`../common/plan/cta_carve.md`](../common/plan/cta_carve.md)); that rule is written so that a kernel's thread can
  run it in registers ([`../common/plan/cta_carve.md#register-shaped`](../common/plan/cta_carve.md#register-shaped)). A level-side's density is `plan::carve_density(masked, level, byte)`
  ([`../common/plan/cta_carve.md#carve-density`](../common/plan/cta_carve.md#carve-density)): the whole density where its bit in `masked` is clear, whatever its byte. The pass calls the same function,
  so the carve's groups, the leaf groups and their count are the same in the pass and in the carry
  ([`../common/plan/mask_prepare_layout.md`](../common/plan/mask_prepare_layout.md#what-a-leaf-group-holds)).
- <a id="lines"></a>`leaf_lines<WORDS>(operands, side, bits, head)`: A SIDE'S LEAF GROUPS' LINES of a batch-head, from its first window: leaf group `leaf`'s
  line is `WORDS` words from `leaf * WORDS`, a word a mask word of the window and a bit a token, set where the token is one of the call's own and live in the leaf
  group ([`../common/plan/leaf_groups.md`](../common/plan/leaf_groups.md#order), [`../mask_prepare/mask_prepare.md`](../mask_prepare/mask_prepare.md)). `bits` is the
  side's leaf groups as log2, the carve's; the block of a window is `leaf_block_bytes(bits, WORDS)` bytes, the counts first and the lines after
  (`leaf_lines_at`). `leaf_step<WORDS>(bits)` is the lines a window on, in words: a role forms a line's address once and advances it by the step a window. A line holds
  the call's own tokens alone, so a reader masks nothing, and a side that lists no level has one leaf group whose line is the call's own tokens, so a reader has one
  path whatever the side lists.
- <a id="place"></a>`place_of(operands, carve, at, slices)`: A CTA'S PLACE IN THE CALL from its index in the grid: its
  slice of the value channels (the slices of one leaf carve are neighbours in the grid, having one token schedule),
  its head, and its index along each level in the order the schedule takes a head's CTAs
  ([`../common/plan/cta_carve.md`](../common/plan/cta_carve.md#schedule)).
- <a id="sets"></a>`fold_levels(carve, place, fold, base, shift, mask)`: A FOLD'S SETS. A fold's box is the CTA's
  carve, halved along the split level; a set is one write group a level inside it, and the set's group on level `l` is
  `base[l] + ((set >> shift[l]) & mask[l])`. A level whose group is no smaller than the fold's extent gives every set
  the one group the box lies in (a group may span folds and CTAs: its mask is 0); a level whose group is smaller gives
  the set's index its bits, the write side's densest level's lowest. The three are formed once a fold, outside a
  role's loop over sets, and the fold's first leaf group is `leaf_group_of(leaves, base)`, `leaves` the write side's `leaf_groups` of the carve: a fold's sets are
  consecutive leaf groups from it. The sets' leaves tile the leaf space once, which the lab checks on the host
  ([`../../lab/carry_supply.md`](../../lab/carry_supply.md)). EACH OF THE CARVE'S ARRAYS IS READ BEFORE THE LEVEL IS
  COMPARED WITH ANYTHING: written as `l == split_level ? fold << fold_bits[l] : 0`, the read was compiled as one
  indexed by the split level at run time, which put the whole carve in the kernel's stack frame (208 bytes; the lab's
  checker refused the kernel for it).
