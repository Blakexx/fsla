# The prefill launch function (`csrc/fsla/src/common/plan/prefill_launch.h`, `cta_carve.h`)

The prefill's launch function has TWO HALVES. Which half is called by whom follows from two rulings (the owner, 2026-11-24: "switching uniformly to precomputed things defined for those patterns. Like switching to a part of the code that is meant for group size 4 fully for the whole
call" -- "We are just templating in the code instead of at the arm"; and "I almost think weights should be outside the masks? They need not be coupled, and are independent variables really" -- "It's also more clear what the contract is if the weights are their own variable vs hidden in the masks"): the densities are
the prefill call's own input beside the masks, `density [2, D]` (row 0 the read side, row 1 the write side; [`../../../api.md`](../../../api.md)), so THE HOST NEVER READS THEM. They are no fact of the prefill's descriptor, and the kernels derive the groups and the carve from them with the one rule
this library states.

| half | function | input | called by | result |
|---|---|---|---|---|
| THE LAUNCH (the host's) | `prefill_launch(const PrefillDescriptor&)` (`prefill_launch.h`) | the descriptor: `levels`, `width_bits_1..4`, `tokens`, `cc` | the host, right before the launch | `PrefillLaunch`: `levels`, `width_bits[l]`, `leaves`, `cta_leaves`, `ctas_per_head`, `window`, `windows` |
| THE CARVE | `prefill_carve(levels, width_bits, write_density, read_density)` (`cta_carve.h`, [`cta_carve.md`](cta_carve.md)) | the widths and the two density lists alone | the kernels of the prefill (the same function: its group rule is annotated for the device and one kernel calls it, [the mask-preparation pass's](../../mask_prepare/mask_prepare.md#the-kernel); the rest is host text) and `plan_tool --cta-carve` | `PrefillCarve`: the groups, `carve_bits`, `cta_leaves`, `split_level`, `folds`, `fold_bits`, `fold_leaves`, `fold_set_leaves`, `fold_sets`, `read_set_leaves`, `read_sets`, `unit_ctas` |

The launch is the first half of the prefill launch function ([`../../../architecture/hopper-carry.md#the-order-of-the-build`](../../../architecture/hopper-carry.md#the-order-of-the-build), step 1), written like [`decode_launch.md`](decode_launch.md)'s args half: from EXACT facts only
(a range aborts), reading `levels`, `width_bits_1..4` (through the per-level lookup of [`levels.md`](levels.md)), `tokens` and `cc`, and deciding nothing from data (R24: the op reads the descriptor and the masks, never a flag of the data). Every call of the contract has a launch. The prefill stays an OWED descriptor
([`../../hardware/registry.md#the-owed-folder`](../../hardware/registry.md#the-owed-folder)): no arm is declared here, and which launches have a built kernel is the arms' business. `plan_tool --kernel prefill ... --launch json` prints the launch ([`plan_tool.md`](plan_tool.md)).

## The launch

Levels are 0-based. `leaves` is the call's leaves a batch-head, the product of the levels' widths; `cta_leaves` is a CTA's, the smaller of 512 and `leaves`; `ctas_per_head = leaves / cta_leaves`. `ctas_per_head` does not depend on how the carve falls across the levels (whatever share each level takes, the shares multiply to
`cta_leaves`: 512 where the call has that many leaves, else every leaf), so the host states it without the densities; `tests/unit/test_prefill_launch.py` checks it against the carve of every density pattern of small calls.

- <a id="window"></a>THE WINDOW. `window` is what `prefill_window(descriptor)` derives from the device's facts, as the arm is: it calls `window_of_cc(cc)` (`levels.h`, [`levels.md`](levels.md), which both launch functions include, beside `kHopperCc` (900) and `kHopperWindow` (1024)), which gives 1024 at `kHopperCc` and 0 at any other `cc`, and a card with no window aborts. The mask-preparation pass's host half (`card_of(device).cc`) and its kernel (`Bt::build.cc`) call the same rule. It is not a fact of the registry: the owner (2026-11-23): "why? shouldnt the matchers be mapping the hardware facts etc to the arm with the correct window. where else would we derive window". The device card (`fsla.ops.window.window_of`) takes it from the
  same function (`device_card_op`, `fsla_api.cpp`, over the card's compute capability). `windows = ceil(tokens / window)`.

## <a id="the-carve"></a>The carve

A side's DENSITY on a level is the share of the level's digits a token of that side is live on, in 256ths (the owner, 2026-11-24: "each weight is like the sparsity of that level on its own, and we have an indepence assumption of sparsity between levels, and with that we derive everything else"):
256 where the side walks the level whole, 1 where it walks it by its lists, a measured sparsity in between. The carve takes the densities as the call's input states them (the integer form in 256ths is the kernels' rule), and clamps one outside 1..256 into the range: the carve is total. The call's own form is a `torch.uint8` byte `b`, the share `b / 255` (255 every digit live; [`api.md`](../../../api.md) section 2.6), and the kernels read it as the integer `b + 1` in 256ths: every byte is a valid density, so the rule needs no clamp, no rounding and no NaN case.

- <a id="groups"></a>R1, THE GROUPS (`prefill_group_bits`, `cta_carve.h`; the owner: "the rule is just grouping, where by default dense level digits are always grouped, and sparse level digits are not ... carry doesn't read the metadata, it just reads
  group size per legel"; "Group size should be like 1,4,16,32,64,128,256 right? ... it's basically multiply the sparsity by the actual level digits and round that to our allows range"). One rule for both sides. The ALLOWED group
  sizes are 1, 4, 16, 32, 64, 128 and 256 digits (`kGroupSizeBits`), those no larger than the level's width. A side's group on a level is the allowed size nearest to `density * width / 256` digits, a tie to the smaller
  size, compared exactly in integers (the distance of a size `a` is `|a * 256 - density * width|`). On a level of 256 digits, density to group: 1 to 1, 2 to 1, 3 to 4, 5 to 4, 10 to 4, 11 to 16, 24 to 16, 25 to 32, 48 to 32, 49 to 64, 96 to 64,
  97 to 128, 192 to 128, 193 to 256, 256 to 256. The groups are per call ("per window is not possible to implement efficiently from what we found"), and "the producer can be improved to choose them better without any
  carry changes": the kernel reads the groups: the mask-preparation pass derives each level-side's group on the device from its density byte by this rule. A group and a CTA's carve size on a level nest, and a group may be the larger ("the more important thing is the multiple rule, which we can guarantee via padding").
- R2, THE CTA'S CARVE ([`cta_carve.md`](cta_carve.md)): `carve_bits[l]` of the level's densities (`write_density[l] + read_density[l]`), the CTA's `cta_leaves` the product of the carve sizes (at most 512, the call's leaves where fewer).
- R3, THE FOLDS' SPLIT (CONSTITUTION R20, the owner, 2026-11-23: "take any write sparse level first, otherwise write dense"). `split_level` is the first level (lowest index) whose write group is smaller than its width and whose
  carve size is at least 2 digits; where there is none, the first level whose carve size is at least 2; -1 where the CTA's carve is one leaf. `folds` is 2 with a split level, else 1; `fold_bits[l]` is `carve_bits[l]`, less 1 on the
  split level; `fold_leaves = cta_leaves / folds`.
- R4, THE SIDES' SETS. THE WRITE SIDE of a fold: `fold_set_leaves` is the product over levels of `2^min(write_group_bits[l], fold_bits[l])`, the leaves a token is expanded over at once, and `fold_sets = fold_leaves / fold_set_leaves`, the
  separate sets of tokens a fold holds. THE READ SIDE over the whole carve (readout does not partition): `read_set_leaves` is the product of `2^min(read_group_bits[l], carve_bits[l])`, `read_sets = cta_leaves / read_set_leaves`.
- R5, THE UNITS. `unit_ctas` is the product of `2^max(0, write_group_bits[l] - carve_bits[l])`, the CTAs inside one combination of write groups, which see one token schedule and are taken together ([the schedule](cta_carve.md#schedule))
  ([`../../../architecture/hopper-carry.md`](../../../architecture/hopper-carry.md#what-a-cta-makes)). The units a head, `units_per_head = ctas_per_head / unit_ctas`, join the carve with the launch's count: the carve does not state it, and `plan_tool --cta-carve` prints it. A DENSITY OF THE WHOLE (256, the byte 255) IS THE LEVEL ITSELF whatever its width, so that a whole level-side is the one whose group is the level's size and no kernel asks anything else (the owner: "use the group size equals level as the check, and that check is all you need in carry").

## Worked cases (`cc` 900, `tokens` 65,536: window 1024, 64 windows)

THE LAUNCH: two levels of 256 digits are 65,536 leaves, a CTA of 512 and 128 CTAs a head; one level of 256 digits is 256 leaves and one CTA; one level of 16 digits is 16 leaves and one CTA (`tests/unit/test_prefill_launch.py`).

THE CARVE: a level is `write density / read density`; widths 256 digits unless stated; carve sizes, then the groups written write / read. The CTAs a head are the launch's.

| levels | carve | groups (write / read) | split | fold | fold set leaves x sets | read set leaves x sets | CTAs, unit CTAs, units |
|---|---|---|---|---|---|---|---|
| A 5 / 256, B 256 / 1 | 32 x 16 | 4, 256 / 256, 1 | 0 | 16 x 16 | 64 x 4 | 32 x 16 | 128, 16, 8 |
| A 1 / 256, B 256 / 1 | 32 x 16 | 1, 256 / 256, 1 | 0 | 16 x 16 | 16 x 16 | 32 x 16 | 128, 16, 8 |
| A 256 / 1, B 1 / 256 | 16 x 32 | 256, 1 / 1, 256 | 1 | 16 x 16 | 16 x 16 | 32 x 16 | 128, 16, 8 |
| A, B 256 / 256 | 32 x 16 | 256, 256 / 256, 256 | 0 | 16 x 16 | 256 x 1 | 512 x 1 | 128, 128, 1 |
| A 5 / 256, B 256 / 1, C 256 / 256 | 4 x 4 x 32 | 4, 256, 256 / 256, 1, 256 | 0 | 2 x 4 x 32 | 256 x 1 | 128 x 4 | 32768, 512, 64 |
| one level, 5 / 256 | 256 | 4 / 256 | 0 | 128 | 4 x 32 | 256 x 1 | 1, 1, 1 |

A 256 / 1 against B 1 / 256 weigh alike (257 each): the larger read density (B's) takes the first doubling, so B takes the ninth, 16 x 32. A density of 0 or 300 is clamped to 1 or 256: the carve of densities 0 and 300 is the carve of 1 and 256. `tests/unit/test_prefill_carve.py` holds these and the
other cases (a density of 128, four levels, two levels of 16 digits, one level of 16), every field against the tool's answer.

## What is NOT here

The segment length and the workspace beyond the prepared masks wait on the carry's kernel
([what a CTA makes](../../../architecture/hopper-carry.md#what-a-cta-makes)) and stay owed ([`../../../open-work.md`](../../../open-work.md) rows A-f and A-g). The workspace's part for the prepared masks is built ([the prepared masks' layout](mask_prepare_layout.md)). The density input reaches the prefill's kernels once they are built: the call carries it
(`fsla/engine/prefill_call.py`, [`../../engine/prefill_call.md`](../../engine/prefill_call.md)) and the mask-preparation pass is the one kernel that reads it. `fsla.ops.prefill` keeps refusing by name.
