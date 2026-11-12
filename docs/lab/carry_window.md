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
  published. Then, a slot of 8 lists at a time, it runs the slot's 8 steps on the tiles its producer filled.
- TWO PRODUCER warps, one a box, fill a slot's 8 lists of 16 entries from scattered rows, four entries a value
  instruction and one instruction of 32 entries' dense bytes, two batches in registers
  ([`pool_fill`](pool_fill.md)'s form, which had two producers a box).
- ONE READOUT warpgroup, a chunk of 4 read digits of 16 tokens at a time. It EXPANDS the 64 readers' coefficients
  itself -- a thread loads a reader's 16 bytes on one box's leaves, multiplies by the reader's byte on the digit,
  rounds to bf16 and stores a line of its own tile, nothing staged between -- runs each digit's step on box 0's
  image tile and box 1's into one accumulator, and stores the chunk's 64 tokens' rows to a panel.
- TWO DRAIN warps add each token's row to its output row as one whole line of f16 pairs.
A window is a box's 256 entries, and 256 tokens read the pair. Shared memory: two slots of pools a box 87 KB, the two
images 70 KB, a chunk's coefficient tiles 4 KB, two panels 35 KB.

THE REGISTERS (`setmaxnreg`, each role's body in its own branch). 512 threads launch at 128 registers a thread, the
SM's whole file of 65,536. A fold thread raises to 160, a readout thread lowers to 64, and the producers and drains,
one warpgroup, stay at 128: the folds' 32 more each are exactly the readout's 64 fewer. At 152 the fold spills to
local memory, and at 48 the toolkit serializes the readout's steps. NO REGISTER IS LEFT for a fifth warpgroup: one
beside these roles takes its registers from them. A form of five warpgroups (640 threads) launches at 96 a thread,
61,440 in all; one whose roles' counts summed to the whole file hung on the device, and whether at the folds' raise
no run separated.

ROWS: the fold's side alone (producers and steps), the read side alone (the image, the readout and the drain),
both, and both with one access taken out: the producers' global loads (their words made in registers), the drains'
global adds (their lines folded into a sink), the image's stores. EVERY WAIT AT A BARRIER IS CLOCKED in the fold,
the readout and the drain warps, and the composed row reports each: a role's own time is its span less its waits.
With both, the fold's state and every output row are checked exactly against the host's: the state summed in
integers, the image its truncation to bf16, a line its rounding to f16.

MEASURED (H100 SXM, 132 SMs, vast.ai instance 60335581, driver 595.91.07; 2026-11-12, the experiment at
6f85c2d4; 7 repetitions, the composed row's 4,325,376 state values and 8,650,752 output values exact; the median,
the spread under 3%; SM-cycles a token are the window's cycles over 8 -- 128 CTAs of two boxes, 1,024 tokens):

| the row | cycles a window a CTA | SM-cycles a token |
|---|---|---|
| the fold's side alone: producers and steps | 4,318 | 540 |
| the read side alone: the image, the readout and the drain | 7,775 | 972 |
| the carry: both sides | 10,221 | 1,278 |
| both sides, the producers' global loads taken out | 8,808 | 1,101 |
| both sides, the drains' global adds taken out | 8,120 | 1,015 |
| both sides, the image's stores taken out | 8,794 | 1,099 |

The composed row's waits, cycles a window a warp of the role:

| the role's wait | cycles | of the window |
|---|---|---|
| a fold warp, on the readout's read of a tile group | 2,149 | 21% |
| a fold warp, on its producer | 910 | 9% |
| a readout warp, on a tile group's publish | 528 | 5% |
| a readout warp, on the panel's drain | 3,283 | 32% |
| a drain warp, on the readout's panel | 1,379 | 13% |

WHAT IT SHOWS.
- THE CARRY COMPOSES, EXACTLY: two boxes' folds, the image a window, one readout warpgroup over both -- its
  coefficients expanded by itself from the readers' bytes -- and one f16 line a token, in one CTA of 512 threads
  at the SM's whole register file.
- A TOKEN COSTS 1,278 SM-CYCLES COMPOSED, 0.62 of FlashAttention's tensor floor at this cell (2,049,
  [`carry_floor`](../internals/tools/carry_floor.md); a cell with its sides sparse on opposite levels, 256 x 256
  leaves -- the model prices a dense side higher).
- THE DRAIN WARPS ARE THE BUSIEST ROLE AND THE READOUT WAITS ON THEM: a drain warp is out of its wait 87% of the
  window, 8,842 cycles for its 128 lines, 69 a line; a readout warp waits on the panel's drain a third of the
  window and a fold warp on the readout's read a fifth.
- A DRAIN WARP'S LINE COSTS IT TWICE WHAT IT DID ALONE: [`readout_role`](readout_role.md) read a token at 15.6
  cycles a CTA by two drain warps, 31 a warp. Taking the producers' global loads out gives back 1,413 cycles a
  window and the drains' adds 2,101; that the two streams share the path to global memory is not separated here
  from the other things the rows change.
- THE TENSOR PIPE IS MOSTLY IDLE: a window is 64 steps of some 20 cycles
  ([`wgmma_width`](wgmma_width.md)), about 1,300 of its 10,221.
- The readout's own coefficients cost the read side about 1,000 cycles a window against the form that wrote them
  once (6,780 then; that form's record is not kept).

Record: `measure/lab/results/carry_window/sm_90a.json`.
