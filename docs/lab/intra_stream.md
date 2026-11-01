# `intra_stream`

Mirrors `measure/lab/experiments/intra_stream.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

THE QUESTION: the fp16 intra ([`docs/internals/intra/intra_kernel.md`](../internals/intra/intra_kernel.md)) moves ~5
bytes a cycle an SM, ~6.8 with its Gram skipped, where `cp.async` streams 18 ([`smem_rates`](smem_rates.md)). What does
its stage ring deliver an SM beside a Gram, and which way of issuing the refills keeps the copies under the MMAs?

THE FORM: one CTA an SM (99 KB of dynamic shared memory), 600 stages over L2-resident rows (4 MB: 4,096 tokens of 1
KB, the dense plane's token stride), a stage the 64 rows' 128-byte slice (the intra's 8 KB stage), a ring of 3, 6 or 10
buffers. Eight compute warps each await a stage, run `SLABS` slabs of the dense Gram on it -- a slab two k-steps, an
`ldmatrix.x4` of a static A and two of the stage's B a k-step, 8 fp16-accumulate MMAs a slab (the 8-warp walk's 4
n-tiles a warp) -- and release it. The issue forms: the LAST RELEASER of a stage refills its buffer a ring on (the
intra's); EVERY WARP, at a stage's start, awaits the previous stage's `empty` mbarrier and copies its share of the stage
two on; a PRODUCER warp, a ninth, copies every stage on each buffer's `empty`. Controls: the Gram alone on resident
stages (no copies), and a stage's rows one contiguous 8 KB run. A sample is the CTA's span over the bytes moved and over
the MMAs issued (cycles an MMA a scheduler), the mean over SMs.

THE RECORD: `measure/lab/results/intra_stream/sm_86.json` (RTX 3080 Ti). The A100's (SXM4 40 GB, a rented host whose
counters are closed, so the driver's ncu pass cannot run there) is the same binary at three buffers run directly,
2026-11-01:

| form, 3 buffers | 0 slabs, B/cycle | 2 slabs (dense f16): B/cycle, cycles/MMA | 4 slabs (u8): B/cycle, cycles/MMA |
|---|---|---|---|
| the Gram alone (no copies) | -- | --, **16.9** | --, 17.0 |
| last releaser refills | 10.3 (A100 10.3) | 6.4, 40.2 (A100 6.3, 40.4) | 4.9, 26.2 (A100 5.0, 25.7) |
| every warp a share | 13.6 (13.8) | 6.9, 37.1 (7.8, 33.0) | 4.8, 26.5 (5.8, 22.3) |
| a producer warp | 10.7 (10.9) | 7.9, 32.3 (7.8, 32.6) | 6.8, 18.9 (6.6, 19.5) |
| the producer, contiguous rows | 10.4 | 7.9, 32.2 | -- |

| a producer warp, by ring depth | 0 slabs, B/cycle | 2 slabs: B/cycle, cycles/MMA | 4 slabs |
|---|---|---|---|
| 3 buffers (24 KB) | 10.7 | 7.9, 32.3 | 6.8, 18.9 |
| 6 buffers (48 KB) | 10.8 | 7.9, 32.2 | -- |
| 10 buffers (80 KB) | 10.6 | 8.0, 32.0 | 6.7, 19.0 |

WHAT IT SETTLES: the Gram's code runs at the tensor rate (16.9 cycles an MMA against sm_86's 16), the token stride costs
nothing, and the bytes in flight do not bind: a ring of 3, 6 or 10 stages delivers the same rate. An SM's copy path
beside a Gram is capped near 8 bytes a cycle (the producer warp the best of the forms; ~10.7 alone, where
[`smem_rates`](smem_rates.md)'s 18 takes eight warps streaming and nothing else). The dense f16 stage feeds a 64-reader
block 128 MMAs an 8 KB, ~16 bytes a cycle at the tensor rate, so its Gram runs at ~32 cycles an MMA, twice the floor;
the u8 stage, half the bytes a digit, at 19. What lowers it is fewer bytes an MMA -- more readers a writer byte, or
narrower rows -- not a deeper ring (above) nor the readers' A fragments in registers to make room for one (built
2026-11-01 and slower: 683 against 505 us at `nl64k-dense`, spilling at depth two).
