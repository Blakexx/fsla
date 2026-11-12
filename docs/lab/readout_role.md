# `readout_role`

Mirrors `measure/lab/experiments/readout_role.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

A CTA's two boxes share their read digits, and each leaves its state as an image in shared memory
([`state_image`](state_image.md)). What does a token cost when ONE readout warpgroup reads both images into one
accumulator -- the two boxes added in fp32 by the steps themselves -- stores the token's partial to a panel, and
drain warps issue it as one line of f16 pairs ([`drain_role`](drain_role.md) priced two panels merged by the drain)?

THE FORM. A CTA an SM: a readout warpgroup and 1, 2 or 4 drain warps. The images (16 tiles a box) and a coefficient
tile a read digit are written once. A chunk is 4 read digits of 16 tokens: 8 `m64n16k16` steps in one commit group,
a digit's step on box 0's tile then on box 1's into the same accumulator -- the first writing it whole, so no
register is cleared between chunks and the steps are not serialized (ptxas C7520, which a cleared accumulator in a
role's branch draws) -- then the 64 tokens' fp32 rows stored to the panel, rows 272 bytes apart so a warp's lanes
fall on 32 banks. The drain warps, a chunk behind, load a token's row, round it to f16 pairs and add one line into
its output row. Every output row is checked exactly.

UNMEASURED: no card has run it.
