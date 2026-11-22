# `collective_branch`

Mirrors `measure/lab/experiments/collective_branch.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

Eight shuffle chains in a loop, one shuffling warp a scheduler: with no role-dependent control flow the shuffle is
inline (8.3 cycles a shuffle), and so under a branch on the block index or a kernel argument (8.3); under a branch on
the warp id (13.6) or with the role setting the trip count (13.9) it is a CALL into a convergence subroutine, ~1.65x.
Record: `measure/lab/results/collective_branch/sm_86.json`.

ON HOPPER THE ROW NO LONGER REPRODUCES. The record above is sm_86 (RTX 3080 Ti, 2026-10-04): there the role-dependent
forms compiled to a CALL into a convergence subroutine. This tree's toolkit (CUDA 13.5, `tools/cuda_toolkit_pin.json`) builds the same source for sm_90a
with no CALL in any of the five forms: the shuffle stays an inline `SHFL`, and the forms under a role's branch (the warp-id
branch and the trip-count role) hold one inline `WARPSYNC.COLLECTIVE` before each of the 8 shuffles, the others none. The
experiment declares that form (CALL 0, WARPSYNC 8 under a role, 0 otherwise). The 1.65x is the cost of the sm_86 subroutine
call; whether the inline `WARPSYNC.COLLECTIVE` form costs anything on an H100 is unmeasured, so the fact the sm_86 row
recorded does not carry to Hopper.
