# The prefill's carve (`csrc/fsla/src/common/plan/cta_carve.h`)

`prefill_carve(levels, width_bits, write_density, read_density)` is a function of the level widths and each side's densities ALONE: no descriptor, no hardware fact. It returns the groups (`write_group_bits[l]`, `read_group_bits[l]`: each level's size from its density, `prefill_group_bits`, raised where a set would hold fewer than 16 leaves, [`prefill_launch.md`](prefill_launch.md#least)), `carve_bits[l]` (the log2 of the
CTA's carve size on level `l`), `cta_leaves`, `split_level`, `folds`, `fold_bits`, `fold_leaves`, `fold_set_leaves`, `fold_sets`, `read_set_leaves`, `read_sets`, `write_order`, `read_order`, `unit_bits[l]` (the bits of level `l`'s CTA index its write group spans, `max(0, write_group_bits - carve_bits)`) and `unit_ctas` (`2 ** sum(unit_bits)`). A density is a side's share of a level's digits a token is live on, in 256ths of the level;
the densities are the prefill call's own input, `density [2, D]` uint8 (a byte `b` is read as `b + 1` in 256ths, [`prefill_launch.md`](prefill_launch.md)), built by the library's helpers from the rows or the masks (the owner, 2026-11-24: "I almost think weights should be outside the masks? They need not be coupled, and are independent variables really"), so the host never reads them ([`prefill_launch.md`](prefill_launch.md)). The kernels of the prefill call
this same function: EVERY FUNCTION OF THE RULE is annotated for the device (`FSLA_CARVE_FN`: `__host__ __device__` where the file is compiled by the CUDA compiler, nothing in a host tool): the mask-preparation pass's kernel calls the group rule
([`../../mask_prepare/mask_prepare.md#the-kernel`](../../mask_prepare/mask_prepare.md#the-kernel)) and the carry's device code the carve ([`../../carry/carry_groups.md#carve`](../../carry/carry_groups.md#carve)). Host and device have one statement of the rule. It is `constexpr`, uses integers and fixed arrays alone (no `std::abort`, no allocation, no stream, no container), and is TOTAL: it has no failure path.
A density outside 1..256 is CLAMPED into 1..256, the range of 256ths (the carve of 0 and 300 is the carve of 1 and 256); `levels` is likewise held within 0..4 and a width within 0..8 bits (so that no input reaches outside the arrays).
`units_per_head` is not the carve's: it is `ctas_per_head / unit_ctas`, and the head's count of CTAs is the launch's (the carve does not depend on it), so the tool joins them (`plan_tool --cta-carve`, [`plan_tool.md`](plan_tool.md)). The apportionment below is `cta_carve(width_bits, write_density, read_density, levels)`, which `prefill_carve` calls on the clamped densities.

THE CTA'S LEAVES. A CTA owns 512 leaves (`kCtaLeaves`, `kCtaLeafBits` = 9 doublings from one digit a level; the owner, 2026-11-24: "The cta actually owns 512 leaves, not 256, with the exception of scaling down, and we are just
partioning them during fold due to register limits. During readout we don't partition, it's just one by one warpgroup"). A call with fewer leaves takes every leaf ("A box need not be 256 leaves we will scale them down if we are forced to").

The allocation is an apportionment (the owner, 2026-11-23: "What about like the US Congress apportionment. We need to ensure each one gets at least one assignment, and then after you assign based on weight?"):

- every level starts at a carve size of one digit ("Obviously the minimum is just one");
- EACH OF THE NINE DOUBLINGS (`kCtaLeafBits`) GOES TO THE LEVEL THAT SAVES THE MOST FETCHES (the owner, 2026-11-26: "it failed because it was overly biasing one side's carve evne though it hadnt saturated the other one"; "so the goal is save as much traffic as possible"; "so you just pick the double the one that has the highest left at each step"; "its just the algorithm picking based on the read and write remaining"). A token's value line is fetched once for each set of a side it is live in, and a side's group on a level that is larger than the CTA's share there is chopped across group / share CTAs, each of which fetches it. A side's LEFT is that product over its levels, size over share where the size is still the larger: the CTAs one of its leaf groups is chopped across. Doubling a level halves the left of each side whose size there is still above the share, so its SAVING is half of each such side's left, added, and the level of the largest saving doubles. `cta_carve` compares savings without forming the products: the side with more left counts first, the other second, and sides with as much left count alike.
- TIES: the smaller share; then a level the read side needs (the owner: "we wanted to bias for the read dense side cause it's better"; "That tie break is fine"); then the lower level index. WHERE NO SIDE NEEDS ANY LEVEL every saving is nothing and the smallest share takes the doubling.
- THE SIZES, NOT THE RAISED GROUPS: the carve reads each level's size from its density ([`prefill_launch.md`](prefill_launch.md#groups)); a set's least raises a group after it and never past a share ([`prefill_launch.md`](prefill_launch.md#least)), so it changes no side's left and the carve and the groups do not depend on each other in a loop.
- NOT IN THE RULE: the sides are weighed alike (how many leaf groups a token is live in is not known from a density), and the write side's fold is half the CTA along one level. NOT MEASURED on record: a call under this rule's carve against the same call under another.
- a level is eligible when its doubled carve size stays within its width; when none is, the CTA has fewer than 512 leaves.

The fold's 64 columns are no limit on the carve (the owner: "The fold limit of 64 refers to its wgmma sizes ... its wgmma execution is decoupled from the data layout"). `plan_tool --cta-carve` ([`plan_tool.md`](plan_tool.md))
prints the carve's full result from the width bits and densities; `tests/unit/test_cta_carve.py` holds the hand-worked cases and the properties over every call of depth 1 to 3 and a sample of depth 4.

THE GROUPS, THE FOLDS' SPLIT, THE SETS and THE UNITS are the rules R1, R3, R4 and R5 of [`prefill_launch.md`](prefill_launch.md#the-carve), whose cases `tests/unit/test_prefill_carve.py` holds, every field, against the tool.

<a id="loop-order"></a>A SIDE'S LOOP ORDER (`carve_loop_order`; `write_order`, `read_order`) is the levels by that side's density, the sparsest
outermost and the densest innermost, equal densities keeping the lower level outside (the owner, 25 November 2026:
"Obviously each side has its own loop order. That's why publish has to order the smem state in a different orientation
than the registers").

<a id="schedule"></a>THE SCHEDULE (`prefill_cta_at(carve, place, cta)`) is the order a head's CTAs are taken in, a place to a CTA's index
along each level (the owner, 25 November 2026: "Write density ignoring read was group scheduling. right. In a group the
CTAs are by read"). A CTA's index along a level has two parts: the low bits, where the level's write group spans
several CTAs (`write_group_bits - carve_bits` of them), and the bits above, which say which group. A place's low
digits fill the first part, level by level in the read side's loop order, so inside a group the read side's sparsest
level varies fastest; its high digits fill the second, in the write side's loop order with its densest level varying
fastest. The header's proof holds the measured cell: a carve of 32 by 16, a write group of 16 CTAs taken together
along the level the read side lists, then the next group. `plan_tool` does not print the schedule function; it prints the two orders (`write_order`, `read_order`) and `unit_bits` it is made from.

<a id="register-shaped"></a>NO ARRAY OF THE CARVE IS INDEXED AT RUN TIME, because a kernel runs the rule: a carry CTA
derives its carve and its place from the call's densities, and in a kernel an array indexed by a run-time value is a
stack frame, which the assembler gate refuses. Every function of the header is the device's as well as the host's
(`FSLA_CARVE_FN`) and its loops over the levels are unrolled there (`FSLA_CARVE_UNROLL`); its loops over the doublings stay loops (`FSLA_CARVE_TRIPS`), so that a kernel holds one trip's values and not nine. What the rule gives is
unchanged, held by the hand-worked cases and the properties over every call (`tests/unit/test_cta_carve.py`,
`tests/unit/test_prefill_carve.py`).
