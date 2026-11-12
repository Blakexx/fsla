# `carry_window`

Mirrors `measure/lab/experiments/carry_window.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The carry's roles were each measured apart: the fold's operands ([`pool_fill`](pool_fill.md)), the state's image
([`state_image`](state_image.md)), the readout and its drain ([`readout_role`](readout_role.md)). What does a window
cost a CTA when they run TOGETHER, under one register file, one tensor pipe and one shared memory?

THE FORM. A CTA an SM of five warpgroups, 640 threads.
- TWO FOLD warpgroups, a 16 × 16 box each, its state 16 accumulators in registers. At a window's start each
  PUBLISHES its state as a bf16 image of 16 tiles (the first window's is cleared memory), once the readout has left
  the window before; then, a slot of 8 lists at a time, it runs the slot's 8 steps on the tiles its producers filled.
- FOUR PRODUCER warps, two a box, fill a slot's 8 lists of 16 entries from scattered rows, four entries a value
  instruction and one instruction of 32 entries' dense bytes ([`pool_fill`](pool_fill.md)'s form) -- with ONE batch
  of 32 entries in registers where that experiment held two, the register file's share.
- ONE READOUT warpgroup runs each read digit's step on box 0's tile and box 1's into one accumulator, 4 digits a
  chunk, and stores the chunk's 64 tokens' rows to a panel.
- FOUR DRAIN warps add each token's row to its output row as one line of f16 pairs.
A window is a box's 256 entries, and 256 tokens read the pair. Shared memory: two slots of pools a box 87 KB, the two
images 70 KB, the read coefficients 8 KB (written once: their fill is not a role here), two panels 35 KB.

THE REGISTERS (`setmaxnreg`, each role's body in its own branch): a fold thread 160, a readout thread 64, a producer
80, a drain 48 -- 65,536 in all, the SM's whole file. At 152 the fold spills to local memory and at 48 the toolkit
serializes the readout's steps; a producer holding two batches needs 112. NO REGISTER IS LEFT for a sixth
warpgroup: one beside these roles takes its registers from them.

ROWS: the fold's side alone (producers and steps), the read side alone (the image, the readout and the drain), and
both. With both, the fold's state and every output row are checked exactly against the host's: the state summed in
integers, the image its truncation to bf16, a line its rounding to f16.

MEASURED: not yet run.
