# `fold_alone`

Mirrors `measure/lab/experiments/fold_alone.cu`, a part harness of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

THE QUESTION. What does the carry's fold phase cost a warp-window alone -- production's `fold_chunk`
([`fold.md`](../internals/carry/fold.md#chunk)), nothing else in the window -- against its tensor floor and against the
composed kernel's fold phase? Asked 2026-11-01 under the ruling that each phase is lean and at its own floor before
any composition, after three forms built in the kernel to take the fold toward its floor all measured slower.

THE METHOD. One CTA an SM with arm 0's plan tenants; each of 64 windows a box's 32 lists drawn as nl64k-alt-k4's are:
Binomial(16, 1/2) entries a list over a 216-slot union, which gives the carve model's alt-k4 steps (40% of the lists
a 16-token step, the rest one 8-token step: 12.9 and 19.2 a CTA-window against its 13.0 and 19.1), and ~45% of the
lists rescaled (the composed kernel's multiply count). A warp writes its own lists' entries and scales, then a
rendezvous (as the carry runs it), then the timed fold. The tensor floor is computed on the host from the same draw:
each scheduler's two warps' MMA cycles (a 16-token step 8 x 32, an 8-token one 8 x 16). The traced variant passes a
recorder as the fold's laps (`fold.md#laps`: the kernel's are none, and its SASS is identical with them).

THE MEASUREMENT (sm_86, carry-k5, cycles a warp-window, median of 7, spread under 0.3%):

| variant | cycles | of the tensor floor (1,434) |
|---|---|---|
| as the carry runs it: a rendezvous a window | 2,489 | 1.74x |
| the windows back to back (no rendezvous) | 1,939 | 1.35x |
| no list rescaled | 2,422 | 1.69x |
| the composed kernel's fold phase (`phase_ledger.py`, nl64k-alt-k4) | 2,859 | 1.99x |

The traced run (2,665: ~20 cycles a lap) splits it: every list's scale, count and first step's lanes 607; each list's
rescale guard ~62; each list's steps 541 / 432 / 411 / 409 against the two warps' pipe time of ~358 a list.

WHAT IT SETTLES.
- The fold is not at its floor alone: 1.74x in the carry's shape. Composition adds only ~0.37K a warp-window on top.
- The rendezvous before each window's fold costs ~0.55K: the scheduler's two warps start together, their operand
  chains coincide, and nothing covers them; run back to back the warps drift apart and cover each other (1.35x).
- The first steps' lanes are the largest part (~0.6K): after the rendezvous every warp issues ~20 shared loads and
  shuffles at once (the lists' counts and places, their scales and first entries, the entries' c pairs).
- Moving each list's step shuffles up front with the entries leaves the composed shape where it was (2,502) and takes
  the back-to-back one to 1,879: shorter chains help warps that are out of phase, not warps in lockstep.

Record: `measure/lab/results/fold_alone/sm_86.json`.
