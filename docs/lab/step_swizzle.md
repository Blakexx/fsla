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

NOT MEASURED: no run of it is recorded.
