# `fold_mchan`

Mirrors `measure/lab/experiments/fold_mchan.cu`, a part harness of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The value carve's fold (M = channel: a warp 16 channels of a box's 256 leaves, 4-warp CTAs, two an SM) on the same
lists, lab-local code: a 16 x 16 and an 8 x 32 box, composed, steady, the MMAs alone, whole loops with a barrier a
window, with heads prefetched, with lane-ready step operands (a lane's two row slots and two c pairs, 16 bytes), and
free-running on landed and drained mbarriers. Per-burst timing reads below the floor at two CTAs an SM (each CTA's
burst can have the tensor pipe alone); only the whole-loop modes are honest.

Measured (sm_86): the 16 x 16 box 1.45x its floor free-running (1.69x with a barrier a window), 8 x 32 1.26x. Not the
operand chain: 10.7 shared-memory wavefronts an HMMA at 16 x 16 (ncu) against the 8 an SM issues an fp32-accumulate
HMMA in, 28% of them bank conflicts over gathered rows, every warp reloading the same slice; 8 x 32 7.45.

Record: `measure/lab/results/fold_mchan/sm_86.json`.
