# `mask_pack`

Mirrors `measure/lab/experiments/mask_pack.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The producer writes a level-side's mask as 32-token words, a bit the u8 digit is nonzero, from the epilogue's registers. What does the packer cost a 64-token block of 256 digits in a `wgmma` m64n256 accumulator's layout?

THE FORM. A warpgroup holds its block's quantized values in that layout, loaded before the clock; a digit is nonzero where its value rounds above 0. A ballot a register index (8 tokens x 4 columns); lane q keeps its column group's four ballots; each of its 8 columns' 16 token bits are every fourth bit of two ballots, gathered by three shift-and-mask rounds; a warp's halves go to shared memory as one 16-byte store a lane; a named barrier joins the warp pair of a 32-token half; each thread writes 16 bytes of the half's 256 words. Every word is checked; densities 10%, 50%, 100%.

PREDICTION, run 20261128T014617Z-889657-f2c62487 (written 2026-11-27, before it ran; its basis: a warp's 16 tokens x 256 columns at ~560 instructions (128 compares, 128 ballots, ~160 keeps, 8 columns x 2 gathers x 8, ~20 for the halves' store, the barrier and the words) at one a clock, a warpgroup's 4 warps on 4 schedulers; no branch on the data, so flat in density):

- 256 columns, 10% of digits nonzero: a 64-token block: 500.00-900.00 cycles a block.
- 256 columns, 50% of digits nonzero: a 64-token block: 500.00-900.00 cycles a block.
- 256 columns, 100% of digits nonzero: a 64-token block: 500.00-900.00 cycles a block.
- the densest block over the sparsest: 0.97-1.03 times.

PREDICTION, run 20261128T034805Z-927513-666b225e (written 2026-11-28, before it ran; its basis: the first run's 1,130 cycles a block are 500 instructions a warp at 0.44 a clock (the SASS: 352 of the ballots and their keeps, 120 of the gathers): latency, not count, so no form with fewer instructions is in the SASS; what it shows removable is the trip's second barrier (the halves double-buffered, a buffer the trip's parity's), one rendezvous of the pair's 64 threads, ~40-150 cycles):

- 256 columns, 10% of digits nonzero: a 64-token block: 1,074.00-1,187.00 cycles a block.
- 256 columns, 50% of digits nonzero: a 64-token block: 1,074.00-1,187.00 cycles a block.
- 256 columns, 100% of digits nonzero: a 64-token block: 1,074.00-1,187.00 cycles a block.
- the double-buffered block over the one-buffer, 50%: 0.85-0.97 times.
- the double-buffered densest over its sparsest: 0.97-1.03 times.

MEASURED (NVIDIA H100 80GB HBM3, Nebius preemptible VM e00fyq0107h186k284, driver 580.196.07; 2026-11-28, the experiment at
b7215a57, run 20261128T034805Z-927513-666b225e; 7 repetitions, the 6 checked variants exact; the median, the widest
spread of a row 0.2%; the clock as the card left it):

| the figure | measured | the prediction | held |
|---|---:|---|---|
| 256 columns, 10% of digits nonzero: a 64-token block | 1,130.50 cycles a block | 1,074.00-1,187.00 cycles a block | held |
| 256 columns, 50% of digits nonzero: a 64-token block | 1,130.30 cycles a block | 1,074.00-1,187.00 cycles a block | held |
| 256 columns, 100% of digits nonzero: a 64-token block | 1,130.67 cycles a block | 1,074.00-1,187.00 cycles a block | held |
| the double-buffered block over the one-buffer, 50% | 0.97 times | 0.85-0.97 times | held |
| the double-buffered densest over its sparsest | 1.00 times | 0.97-1.03 times | held |

The record's rows:
- 256 columns, 10% of digits nonzero: a 64-token block: 1,130.50 cycles a block.
- 256 columns, 50% of digits nonzero: a 64-token block: 1,130.30 cycles a block.
- 256 columns, 100% of digits nonzero: a 64-token block: 1,130.67 cycles a block.
- 256 columns, 10% of digits nonzero, the halves double-buffered: a 64-token block: 1,094.30 cycles a block.
- 256 columns, 50% of digits nonzero, the halves double-buffered: a 64-token block: 1,095.15 cycles a block.
- 256 columns, 100% of digits nonzero, the halves double-buffered: a 64-token block: 1,094.30 cycles a block.

Record: `data:records/fsla/measure/lab/mask_pack.run/20261128T034805Z-927513-666b225e/output.json`.
