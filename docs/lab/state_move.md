# `state_move`

Mirrors `measure/lab/experiments/state_move.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

[`cluster_sum`](cluster_sum.md) showed that a token's partial costs as much into a cluster-mate's shared memory as
straight to global memory, so a carry whose CTAs exchange something must exchange less of it. A CTA's STATE is
that: it changes once a window, not once a token. What does a kilobyte of state cost a CTA to move -- into its
cluster-mates' shared memory by bulk copies, or out to global memory and back in by one bulk copy each way? And
what does a token's 128-byte value cost, loaded by itself?

THE FORM. Clusters of 8 CTAs, 4 warps each, as many clusters as the card's SMs hold. A round moves 8 blocks a CTA,
of 1 KB or of 8 KB each (8 KB or 64 KB a round).
- `stage`: the round's words stored into the CTA's own shared memory -- the base every moving form carries.
- `mates`: the stage, then a bulk copy of one block to each CTA of the cluster (itself among them), whose bytes
  complete the receiver's barrier; the receiver reads its image, fences at cluster scope and releases its mates into
  the next round.
- `out`: the stage, then one bulk copy of it to the CTA's own span of global memory, waited.
- `out and in`: that, then one bulk copy of the span back into the image.
- `in`: one bulk copy from global memory into the image, with no stage.
- `rows`: 64 loads of 128 bytes a round, from scattered rows of an 8 MB table, issued by one thread or by 64.
The last round's image is checked exactly, word for word.

WHAT THE SASS SAYS (sm_90a, the pinned toolkit; no device): a bulk copy into a mate is one `UBLKCP.S.S`, out to
global memory one `UBLKCP.G.S`, in from it one `UBLKCP.S.G`.

MEASURED (H100 SXM, 132 SMs, vast.ai instance 60254914, driver 580.173.02; 2026-11-11, the experiment at
93eae8f4; 7 repetitions, every image exact; SM-cycles a kilobyte a CTA, the median, the spread under 1.0%):

| the form | 1 KB blocks (8 KB a round) | 8 KB blocks (64 KB a round) |
|---|---|---|
| the stage's stores alone | 87.2 | 75.6 |
| a bulk copy a cluster-mate | 1,011.0 | 228.3 |
| one bulk copy out to global memory | 130.5 | 109.3 |
| one bulk copy out to global memory and one back in | 204.1 | 127.9 |
| one bulk copy in from global memory | 61.0 | 32.8 |

and a 128-byte load of a scattered row, SM-cycles a load a CTA: 32.8 with a thread issuing each, 103.7 with one
thread issuing all 64.

WHAT IT SHOWS.
- MOVING A KILOBYTE COSTS TENS OF CYCLES BY EVERY ROUTE. Less the stage, at 64 KB a round: 34 out to global memory,
  33 in from it, 52 for both (they overlap), and to the cluster's mates 43 a kilobyte beside 7,050 a round -- the
  two block sizes' split: (228.3 - 75.6) x 64 = 9,770 cycles a round at 64 KB, (1,011.0 - 87.2) x 8 = 7,390 at
  8 KB. That round is the barriers' cost [`cluster_sum`](cluster_sum.md) read, and a CTA pays it every round
  whatever it sends.
- THE STAGE COSTS AS MUCH AS THE MOVE: 76 to 87 cycles a kilobyte to store the words a CTA will send. The loop here
  is one store a trip with its own arithmetic; what a kernel's own narrowing store costs is UNMEASURED.
- GLOBAL MEMORY IS NO SLOWER THAN A MATE'S SHARED MEMORY, and it needs no cluster barrier: 52 a kilobyte out and
  back against 43 plus the round. What orders a writer and a reader in different CTAs on that route is not in this
  experiment.
- A VALUE LOADED BY ITSELF COSTS EIGHT TIMES ITS BYTES: 32.8 cycles a 128-byte request against 4.1 for 128 bytes of
  a 64 KB copy, and 104 when one thread issues them. Values reach a CTA cheaply only packed together. Both rates
  are read on a working set the cache holds (an 8 MB table, a 64 KB span a CTA); a set it cannot hold is UNMEASURED.

Record: `measure/lab/results/state_move/sm_90a.json`.
