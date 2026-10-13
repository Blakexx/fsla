# `gang_ring`

Mirrors `measure/lab/experiments/gang_ring.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

Whether sidecar CTAs can build each window's per-list slot lists from the bitlists into a ring of R windows fast
enough for the carry CTAs consuming them, in ONE PERSISTENT COOPERATIVE GRID (a CTA an SM, every CTA resident by
`cudaLaunchCooperativeKernel`, so spin-waits cannot starve a consumer). A synthetic head: 64 windows, each token live
in k of 256 level-R digits; carry CTA c owns group c % 8 (32 digits, 4 lists a warp) and steps them in `list_fold`
form 5's shape over a static pool (runtime list lengths, two operand slots, the odd step peeled), each warp staging
its lists a window ahead by cp.async and freeing the slot; a sidecar builds a (window, group)'s 32 lists -- the
union's ranks, each digit's entries round-robin by rank & 7, steps of 16, each entry its pool row offset and f16
coefficient pair (the level-R byte, staged from the rows, times the token's (F, m)) -- after the gang freed the slot
(window w - R staged by every carry warp). Spans are each CTA's own (SMs' clocks do not compare), the slowest CTA's
taken; floor = the busiest scheduler's paired warps' MMAs at 32 cycles. sm_86, cycles a window (the build as four
teams of two warps a sidecar, each team an item under its own named barrier; still issue-heavy, ~6K warp-instructions
an item at k 4):

| k | lists given (76 carry CTAs) | build alone, 8 sidecars (a sidecar) | ring of 3, 8 sidecars | ring of 3, 4 sidecars | ring of 2 / 4, 4 sidecars |
|---|---|---|---|---|---|
| 4 | 4.00K (1.47x) | 4.29K (32.1K) | 4.92K (1.80x) | 9.50K (3.48x) | 9.54K / 9.51K |
| 16 | 7.43K (1.04x) | 9.54K (71.9K) | 10.4K (1.46x) | 20.9K (2.91x) | 21.1K / 21.3K |

The ring works -- no deadlock, depth 2-4 alike, the consumers waiting only on the build -- and the build binds: a
sidecar's throughput 4.0K (k 4) / 9.0K (k 16) cycles an item. The first cut (a CTA an item, serial thread-0 prefixes,
scattered 2-byte global stores) ran 17.6K / 33K; the second (a lane a digit, a warp a word pair, the union a warp OR,
prefixes by shuffles, lists assembled in shared memory and copied out whole) 7.7K / 11.3K; the third (four teams of
two warps, the coefficients gathered four entries a thread) above. The ncu source profile of the third: fixed-latency
chains 28%, the gathers 17%, the per-entry round-robin position (~35 instructions) and a serial pad loop. The
comparison here is the fold's steps alone; a carry window (fold, readout, fill) is ~20K (alt-k4 target) to
~61K (alt-k16). Record: `measure/lab/results/gang_ring/sm_86.json`.
