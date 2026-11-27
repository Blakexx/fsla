# `step_swizzle`

Mirrors `measure/lab/experiments/step_swizzle.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

Does the tensor step read its A operand from TOKEN-MAJOR ROWS -- each of its 16 K entries a contiguous 128 bytes
holding that entry's 64 M values, the row's eight 16-byte pieces turned by the row's place (the descriptor's 128-byte
swizzle) -- and compute what it computes from the unswizzled brick layout the lab's kernels use? It is what would let
a carry CTA's value line land as its 128 bytes whole, the one form of the fetch measured cheap
([`value_fetch`](value_fetch.md)).

THE FORM. One warpgroup. The host lays one A [64 x 16] of small integers in both layouts and one B [16 x 16]; the
kernel copies the three tiles into shared memory, the swizzled tile at a 1,024-byte boundary, issues one step on each
A against the same B, and writes both accumulators out. CHECKED: every thread's 8 sums are the same from both layouts,
and their total over the warpgroup is the product's total, computed on the host. THE LAYOUT IS READ OFF THE VENDOR'S
PUBLIC LIBRARY (its sm_90 tensor-step traits), not off a statement of the instruction's: the experiment is the check.

THE STEP'S TIME BY A'S LAYOUT is its second question: 8 steps a commit group on one accumulator and the same B, the
group waited, 2,048 groups, with A in bricks 16 lines apart, in bricks 17 apart (the pitch a thread's own stores
want), and in token-major rows. CHECKED: a thread's sum of its 8 sums is the first kernel's step's, 16,384 times
over. The rows' descriptor is the library's (`step_operand_rows`,
[`internals/common/atoms/tensor.md`](../internals/common/atoms/tensor.md#rows)).

MEASURED (NVIDIA H100 80GB HBM3, 132 SMs, a Nebius instance, driver 580.196.07; 2026-11-25, the experiment at
73620f1a; 7 repetitions). THE STEP READS THE ROWS: 0 of 135,168 sums differ between the two layouts, every one of the
132 warpgroups' totals is the product's, and every tile is on its boundary. Its time by A's layout, the sums of each
row's 16,896 threads checked (cycles a step, the median):

| A's layout | cycles a step |
|---|---:|
| bricks 16 lines apart | 27.639 |
| bricks 17 lines apart | 27.637 |
| token-major rows, the 128-byte swizzle | 27.636 |

PREDICTED, before the run: 15 to 40 each, from a fold's step read at 23 to 25 cycles in [`step_queue`](step_queue.md), with the hypothesis
that the three cost the same, the step's read of its operand not being a thread's banked load -- HELD: the three
medians are within 0.003 cycles. WHAT IT IS NOT: a statement about B's layout, about a step issued while threads
store into the tile it reads, or about rows off a 1,024-byte boundary.

Record: `data:lab/step_swizzle/sm_90a.json`.
