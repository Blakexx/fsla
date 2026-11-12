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

UNMEASURED: no card has run it.
