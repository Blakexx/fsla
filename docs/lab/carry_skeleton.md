# `carry_skeleton`

Mirrors `measure/lab/experiments/carry_skeleton.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

Does the LIBRARY's carry skeleton ([`internals/carry/carry_kernel.md`](../internals/carry/carry_kernel.md)) -- its three roles at
their registers, their barriers and their order of work, every component a stub -- run a call's windows to the end on
every multiprocessor, and what does a window of nothing but its barriers cost?

THE FORM. This file compiles the library's device code and puts a clock around `carry_cta`: a CTA an SM of 512
threads, a call 256 windows. Every warp adds its windows to its CTA's word when its role has returned; a CTA that
hangs or leaves early fails the check.

MEASURED (NVIDIA H100 80GB HBM3, 132 SMs, a Nebius instance, driver 580.196.07; 2026-11-25, the experiment at
2ec94cc6; 7 repetitions): every warp of all 132 CTAs returned from its role after 256 windows, and
a window of the skeleton's barriers alone is 849 cycles a CTA (the median).

PREDICTED, before the run: 150 to 3,000 cycles, wide because no lab row prices a suspended wait's wake -- HELD.

WHAT IT SHOWS. The roles' protocol runs to the end on every multiprocessor at the registers of the lab's split
(176, 104 and 56 a thread). A window's 16 waits and arrivals, with no work between them, cost about a sixth of the
lab's composed window ([`carry_token`](carry_token.md): some 5,400 cycles): the chain of suspended waits is no
free edge. WHAT IT IS NOT: a floor for the built kernel -- there a role arrives while the one that waits on it is
still at work, and the waits overlap that work.

Record: `measure/lab/results/carry_skeleton/sm_90a.json`.
