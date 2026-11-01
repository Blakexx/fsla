# `intra_stream`

Mirrors `measure/lab/experiments/intra_stream.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

THE QUESTION: the fp16 intra ([`docs/internals/intra/intra_kernel.md`](../internals/intra/intra_kernel.md)) moves ~5
bytes a cycle an SM, ~6.8 with its Gram skipped, where `cp.async` streams 18 ([`smem_rates`](smem_rates.md)). What does
its stage ring deliver an SM beside a Gram, and which way of issuing the refills keeps the copies under the MMAs?

THE FORM: one CTA an SM (90 KB of dynamic shared memory), 512 stages over L2-resident rows (4 MB: 4,096 tokens of 1 KB,
the dense plane's token stride), a stage the 64 rows' 128-byte slice (the intra's 8 KB stage), three buffers. Eight
compute warps each await a stage, run `SLABS` slabs of the dense Gram on it -- a slab two k-steps, an `ldmatrix.x4` of
a static A and two of the stage's B a k-step, 8 fp16-accumulate MMAs a slab (the 8-warp walk's 4 n-tiles a warp) --
and release it. The issue forms: the LAST RELEASER of a stage refills its buffer three stages on (the intra's); EVERY
WARP, at a stage's start, awaits the previous stage's `empty` mbarrier and copies its share of the stage two on; a
PRODUCER warp, a ninth, copies every stage on each buffer's `empty`. Controls: the Gram alone on resident stages (no
copies), and a stage's rows one contiguous 8 KB run. A sample is the CTA's span over the bytes moved and over the MMAs
issued (cycles an MMA a scheduler), the mean over SMs.

THE RECORD: `measure/lab/results/intra_stream/sm_86.json` (RTX 3080 Ti). The A100's (SXM4 40 GB, a rented host whose
counters are closed, so the driver's ncu pass cannot run there) is the same binary run directly, 2026-11-01:

| form | 0 slabs, B/cycle | 2 slabs (dense f16): B/cycle, cycles/MMA | 4 slabs (u8): B/cycle, cycles/MMA |
|---|---|---|---|
| the Gram alone (no copies) | -- | --, **16.9** (sm_80 not run) | --, 17.0 |
| last releaser refills | 10.3 (A100 10.3) | 6.4, 40.2 (A100 6.3, 40.4) | 4.9, 26.2 (A100 5.0, 25.7) |
| every warp a share | 13.6 (13.8) | 6.9, 37.1 (7.8, 33.0) | 4.8, 26.5 (5.8, 22.3) |
| a producer warp | 10.7 (10.9) | 7.9, 32.3 (7.8, 32.6) | 6.8, 18.9 (6.6, 19.5) |
| the producer, contiguous rows | 10.4 | 7.9, 32.2 | -- |

WHAT IT SETTLES: the Gram's code runs at the tensor rate (16.9 cycles an MMA against sm_86's 16), and the token stride
costs nothing; what binds is the ring's bytes in flight. Three 8 KB stages hold at most ~24 KB in flight, and an SM
then sustains ~10-14 bytes a cycle with nothing beside it -- Little's law at ~2K cycles of loaded L2 latency -- and
~6-8 beside a Gram, where the dense f16 stage needs ~16 at the tensor rate (128 MMAs an 8 KB stage at a 64-reader
block); [`smem_rates`](smem_rates.md)'s 18 had ~64 KB in flight. No issue form recovers it (the producer warp is the
best, 7.9). The intra's three stages beside its resident reader rows are the area's on both cards, so the bytes in
flight have to come from freeing the reader rows' area: the readers' A fragments in registers
([`docs/internals/intra/intra_kernel.md#where-the-time-goes`](../internals/intra/intra_kernel.md#where-the-time-goes)).
