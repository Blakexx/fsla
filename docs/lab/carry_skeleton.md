# `carry_skeleton`

Mirrors `measure/lab/experiments/carry_skeleton.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

Does the LIBRARY's carry skeleton ([`internals/carry/carry_kernel.md`](../internals/carry/carry_kernel.md)) -- its
three roles at their registers, their hand-offs and their order of work -- run a call's windows to the end on every
multiprocessor, and what does a window of nothing but its hand-offs cost? THE CALL HAS NO TOKENS AND NO LEVELS, so
the folds' supply, the one component built, lists nothing.

THE FORM. This file compiles the library's device code and puts a clock around `carry_cta`: a CTA an SM of 512
threads, a call 256 windows. Every warp adds its windows to its CTA's word when its role has returned; a CTA that
hangs or leaves early fails the check. Its second row stamps every warp's hand-offs and windows through the kernel's
tracer (64 windows), and the record holds two CTAs' timelines of it.

MEASURED (NVIDIA H100 80GB HBM3, 132 SMs, a Nebius instance, driver 580.196.07; 2026-11-25, the experiment at
34b9a08d; 7 repetitions, every warp of all 132 CTAs returning in both rows; cycles a window a CTA, the median):

| the carry on a call of no tokens | cycles |
|---|---:|
| as the library compiles it, its tracer nothing | 653 |
| every warp's hand-offs and phases stamped | 1460 |

PREDICTED, before the run: the plain row 560 to 640 (596 at b64785d8) -- MISSED, 13 over: the range took the code as
unchanged, and a fold warp's window runs its set's listing, which now lays the set's tokens in its list too, a list of
none here. The stamped row 1,330 to 1,480 (1,408 at b64785d8) -- HELD. A
STAMPED ROW OF THIS CALL IS A TIMELINE'S ORDER, NOT ITS DURATIONS: on a window that is nothing but 16 hand-offs the
stamps are the larger cost.

THE HAND-OFF'S FORM was chosen on the experiment at 819a33c0, which held two (the same card, the same day): a
window's 16 hand-offs 849 cycles as shared barriers (`mbarrier`) with suspending waits, 528 as named barriers of the
two roles' threads. The suspending row's prediction held (760 to 940); the named row's missed (650 to 1,050): it took
[`carry_window`](carry_window.md)'s finding, a fold warp's wait at its own warpgroup's rendezvous the same in both
forms, for a hand-off between roles, where the shared barrier's waiter is suspended and woken. The kernel keeps the
named form and the other is gone from it (the owner, 25 November 2026: "I don't want to maintain two forms, so we need
to pick one after the barrier result comes in"). A WAIT THAT SPINS was not run (the owner, the same day: "the spinning
will look free until it contends with real work, so probably the only worth doing now for sure if it's better is
named barriers, and otherwise just keep spin in mind as something we can try if we see barriers as a problem in the
final kernel").

WHAT IT IS NOT: a floor for the built kernel -- there a role arrives while the one that waits on it is still at
work, and the waits overlap that work.

Record: `measure/lab/results/carry_skeleton/sm_90a.json`.
