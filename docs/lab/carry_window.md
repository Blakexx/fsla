# `carry_window`

Mirrors `measure/lab/experiments/carry_window.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The carry's roles were each measured apart: the fold's operands ([`pool_fill`](pool_fill.md)), the state's image
([`state_image`](state_image.md)), the readout and its drain ([`readout_role`](readout_role.md)). What does a window
cost a CTA when they run TOGETHER, under one register file, one tensor pipe and one shared memory?

THE FORM. A CTA an SM of four warpgroups, 512 threads.
- TWO FOLD warpgroups, a 16 × 16 box each, its state 16 accumulators in registers. At a window's start each
  PUBLISHES its state as a bf16 image of 16 tiles (the first window's is cleared memory), once the readout has left
  the window before; then, a slot of 8 lists at a time, it runs the slot's 8 steps on the tiles its producer filled.
- TWO PRODUCER warps, one a box, fill a slot's 8 lists of 16 entries from scattered rows, four entries a value
  instruction and one instruction of 32 entries' dense bytes, two batches in registers
  ([`pool_fill`](pool_fill.md)'s form, which had two producers a box).
- ONE READOUT warpgroup runs each read digit's step on box 0's tile and box 1's into one accumulator, 4 digits a
  chunk, and stores the chunk's 64 tokens' rows to a panel.
- TWO DRAIN warps add each token's row to its output row as one line of f16 pairs.
A window is a box's 256 entries, and 256 tokens read the pair. Shared memory: two slots of pools a box 87 KB, the two
images 70 KB, the read coefficients 8 KB (written once: their fill is not a role here), two panels 35 KB.

THE REGISTERS (`setmaxnreg`, each role's body in its own branch). 512 threads launch at 128 registers a thread, the
SM's whole file of 65,536. A fold thread raises to 160, a readout thread lowers to 64, and the producers and drains,
one warpgroup, stay at 128: the folds' 32 more each are exactly the readout's 64 fewer. At 152 the fold spills to
local memory, and at 48 the toolkit serializes the readout's steps. NO REGISTER IS LEFT for a fifth warpgroup: one
beside these roles takes its registers from them. A form of five warpgroups (640 threads: four producers, four
drain warps) launches at 96 a thread, 61,440 in all, and its roles' counts summed to the whole file hung on the
device at the folds' raise.

ROWS: the fold's side alone (producers and steps), the read side alone (the image, the readout and the drain), and
both. With both, the fold's state and every output row are checked exactly against the host's: the state summed in
integers, the image its truncation to bf16, a line its rounding to f16.

MEASURED: not yet run.
