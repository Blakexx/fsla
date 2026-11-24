# The prefill's carve (`csrc/fsla/src/common/plan/cta_carve.h`)

`prefill_carve(levels, width_bits, write_density, read_density)` is a function of the level widths and each side's densities ALONE: no descriptor, no hardware fact. It returns the densities' groups (`write_group_bits[l]`, `read_group_bits[l]`), `carve_bits[l]` (the log2 of the
CTA's carve size on level `l`), `cta_leaves`, `split_level`, `folds`, `fold_bits`, `fold_leaves`, `fold_set_leaves`, `fold_sets`, `read_set_leaves`, `read_sets` and `unit_ctas`. A density is a side's share of a level's digits a token is live on, in 256ths of the level;
the densities are the prefill call's own input, `density [2, D]`, built by the library's helpers from the rows or the masks (the owner, 2026-11-24: "I almost think weights should be outside the masks? They need not be coupled, and are independent variables really"), so the host never reads them ([`prefill_launch.md`](prefill_launch.md)). The kernels of the prefill call
this same function (it is not yet annotated for the device, and no kernel includes it yet): host and device have one statement of the rule. It is `constexpr`, uses integers and fixed arrays alone (no `std::abort`, no allocation, no stream, no container), and is TOTAL: it has no failure path.
A density outside 1..256 is CLAMPED into 1..256, the range of 256ths (the carve of 0 and 300 is the carve of 1 and 256); `levels` is likewise held within 0..4 and a width within 0..8 bits (so that no input reaches outside the arrays).
`units_per_head` is not the carve's: it is `ctas_per_head / unit_ctas`, and the head's count of CTAs is the launch's (the carve does not depend on it), so the tool joins them (`plan_tool --cta-carve`, [`plan_tool.md`](plan_tool.md)). The apportionment below is `cta_carve(width_bits, write_density, read_density, levels)`, which `prefill_carve` calls on the clamped densities.

THE CTA'S LEAVES. A CTA owns 512 leaves (`kCtaLeaves`, `kCtaLeafBits` = 9 doublings from one digit a level; the owner, 2026-11-24: "The cta actually owns 512 leaves, not 256, with the exception of scaling down, and we are just
partioning them during fold due to register limits. During readout we don't partition, it's just one by one warpgroup"). A call with fewer leaves takes every leaf ("A box need not be 256 leaves we will scale them down if we are forced to").

The allocation is an apportionment (the owner, 2026-11-23: "What about like the US Congress apportionment. We need to ensure each one gets at least one assignment, and then after you assign based on weight?"):

- every level starts at a carve size of one digit ("Obviously the minimum is just one");
- a level's WEIGHT (the one place the apportionment keeps that word) is its two sides' densities summed, `write_density[l] + read_density[l]`; both are in 256ths of the level, which is the owner's "The weight should be as percentage of the level size", so the sums compare exactly as they stand;
- nine times (`kCtaLeafBits`) the level of the highest priority `weight / (carve_bits + 1)` doubles, compared exactly in integers by cross-multiplying;
- TIES: of two levels of equal priority the one with the larger READ density takes the doubling (the owner: "we wanted to bias for the read dense side cause it's better"; "That tie break is fine"); if the read densities are equal too, the lower level index;
- a level is eligible when its doubled carve size stays within its width; when none is, the CTA has fewer than 512 leaves.

The fold's 64 columns are no limit on the carve (the owner: "The fold limit of 64 refers to its wgmma sizes ... its wgmma execution is decoupled from the data layout"). `plan_tool --cta-carve` ([`plan_tool.md`](plan_tool.md))
prints the carve's full result from the width bits and densities; `tests/unit/test_cta_carve.py` holds the hand-worked cases and the properties over every call of depth 1 to 3 and a sample of depth 4.

THE GROUPS, THE FOLDS' SPLIT, THE SETS and THE UNITS are the rules R1, R3, R4 and R5 of [`prefill_launch.md`](prefill_launch.md#the-carve), whose cases `tests/unit/test_prefill_carve.py` holds, every field, against the tool.
