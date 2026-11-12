# `carry_window`

Mirrors `measure/lab/experiments/carry_window.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The carry's roles were each measured apart: the fold's operands ([`pool_fill`](pool_fill.md)), the state's image
([`state_image`](state_image.md)), the readout and its drain ([`readout_role`](readout_role.md)). What does a window
cost a CTA when they run TOGETHER, under one register file, one tensor pipe and one shared memory?

THE FORM. A CTA an SM of four warpgroups, 512 threads.
- TWO FOLD warpgroups, a 16 × 16 box each, its state 16 accumulators in registers. Each PUBLISHES its state as a
  bf16 image of 16 tiles (the first window's is cleared memory), a group of 4 tiles at a time, each once the readout
  has read that group of the window before: the readout reads a window's first groups while its last are
  published. Then, a slot of 8 lists at a time, it runs the slot's 8 steps on the tiles its producers filled.
- FOUR PRODUCER warps, two a box, fill a slot's 8 lists of 16 entries from scattered rows, four entries a value
  instruction and one instruction of 32 entries' dense bytes ([`pool_fill`](pool_fill.md)'s form; the value words
  two batches ahead in registers, the dense bytes one).
- ONE READOUT warpgroup is the read side whole, a chunk of 4 read digits of 16 tokens at a time. It EXPANDS the 64
  readers' coefficients itself -- a thread loads a reader's 16 bytes on one box's leaves, multiplies by the
  reader's byte on the digit, rounds to bf16 and stores a line of its own tile, nothing staged between -- runs each
  digit's step on box 0's image tile and box 1's into one accumulator, stores the 64 tokens' rows to its panel, and
  its own four warps ADD each token's row to the output as one whole line of f16 pairs. No drain warp. (Adding
  from the registers themselves was run and is slower: a warp holds 16 of a token's 64 channels, so a token's line
  was four partial adds -- the read side alone 7,316 cycles a window where the panel and two drain warps read
  6,780.)
A window is a box's 256 entries, and 256 tokens read the pair. Shared memory: two slots of pools a box 87 KB, the two
images 70 KB, a chunk's coefficient tiles 4 KB, the panel 17 KB.

THE REGISTERS (`setmaxnreg`, each role's body in its own branch). 512 threads launch at 128 registers a thread, the
SM's whole file of 65,536. A fold thread raises to 160, a readout thread lowers to 64, and the producers stay at
128: the folds' 32 more each are exactly the readout's 64 fewer. At 152 the fold spills to local memory, and at 48
the toolkit serializes the readout's steps. NO REGISTER IS LEFT for a fifth warpgroup: one beside these roles takes
its registers from them. A form of five warpgroups (640 threads) launches at 96 a thread, 61,440 in all; one whose
roles' counts summed to the whole file hung on the device, and whether at the folds' raise no run separated.

ROWS: the fold's side alone (producers and steps), the read side alone (the image and the readout), both, and both
with one access taken out to find what the sides contend for: the producers' global loads (their words made in
registers), the readout's global adds (its lines folded into a sink), the image's stores. With both, the fold's
state and every output row are checked exactly against the host's: the state summed in integers, the image its
truncation to bf16, a line its rounding to f16.

MEASURED: not yet run in this form.
