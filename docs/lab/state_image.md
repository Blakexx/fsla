# `state_image`

Mirrors `measure/lab/experiments/state_image.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

Where a box's two sides are sparse on different levels, the fold accumulates its state along one axis -- an
accumulator a fold digit, 16 leaves each -- and the readout reads it along the other, a read digit's 16 leaves
one from each accumulator. A wgmma takes its A from one warpgroup's registers in one layout or from shared memory.
What does it cost a window to store the state to shared memory as the tiles the readout's steps read, and to read
them?

THE FORM. A CTA an SM, two warpgroups, each a box of 16 x 16 leaves x 64 channels in registers (fp32). A window,
every thread stores its 128 values, truncated to bf16, into the tile of each value's read digit -- M|N-major, a line
one leaf's 8 channels, tiles 137 lines apart and bricks 17, so a warp's lanes fall on distinct banks.
- `halves`: a value a 2-byte store.
- `pairs`: a lane takes its neighbour's value of the same line by one shuffle and stores 4 bytes.
- `store and read` (by halves): then 16 `m64n16k16` steps a box, A a tile of the image and B a coefficient tile,
  into one accumulator, checked exactly.
THE REGISTERS BIND: with the state's 128 a thread, the 16 steps' descriptors held at once spill to the stack, so the
steps run in a rolled loop; and the pairs form beside the steps does not fit at all and is not measured.

MEASURED (H100 SXM, 132 SMs, vast.ai instance 60297524, driver 615.71.09; 2026-11-12, the experiment at
8c03d7d9; 7 repetitions, the readout exact; SM-cycles a window a CTA of two boxes, the median, the spread
under 0.1%):

| the form | cycles a window |
|---|---|
| a value a 2-byte store, the image alone | 1,102 |
| a pair a 4-byte store after a shuffle, the image alone | 1,204 |
| a value a 2-byte store, and the 16 steps a box that read it | 2,218 |

WHAT IT SHOWS.
- THE IMAGE COSTS A CTA 1,102 CYCLES A WINDOW, 2.2 a stored value's warp instruction: once a window, whatever the
  window holds. At 1,024 tokens a window and 128 CTAs that is 138 SM-cycles a token.
- THE SHUFFLE BUYS NOTHING: a pair by one 4-byte store costs 1.09 times a value by a 2-byte store.
- THE READOUT READS THE IMAGE EXACTLY, a box's 16 steps 558 cycles (35 a step): the M|N-major tile a read digit
  is what a step takes as its A.

Record: `measure/lab/results/state_image/sm_90a.json`.
