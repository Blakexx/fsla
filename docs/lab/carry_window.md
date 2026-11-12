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
  instruction and one instruction of 32 entries' dense bytes ([`pool_fill`](pool_fill.md)'s form, which had two
  producers a box and two batches in registers where this has one).
- ONE READOUT warpgroup, a chunk of 4 read digits of 16 tokens at a time, A DIGIT A WARP'S ROWS. A warp EXPANDS ITS
  TOKENS' COEFFICIENTS IN ITS REGISTERS -- a thread loads two words of a reader's bytes on a box's leaves,
  multiplies by the reader's byte on the digit and rounds to bf16 pairs -- and they are the A OPERAND of the
  digit's step, whose B is the box's image tile as the fold published it; a step is 64 tokens × 64 channels, every
  other warp's rows nothing. One accumulator holds the chunk's 64 tokens, box 0's and box 1's steps added in it,
  and a token's whole row is one warp's: the readout stores it to a panel as the f16 pairs its line holds, a
  warp's lanes on 32 banks. No coefficient reaches shared memory.
- TWO DRAIN warps add each token's row to its output row as one whole line of f16 pairs: a load of the row as the
  panel holds it and the add.
A window is a box's 256 entries, and 256 tokens read the pair. Shared memory: two slots of pools a box 87 KB, the two
images 70 KB, two panels 35 KB.

THE REGISTERS (`setmaxnreg`, each role's body in its own branch). 512 threads launch at 128 registers a thread, the
SM's whole file of 65,536. A fold thread raises to 160; a readout thread lowers to 96, and the producers and drains,
one warpgroup, to 96: the folds' 32 more each are those 32 fewer. At 152 the fold spills to local memory; at 80 the
toolkit serializes the readout's steps of 64 columns with A in registers; a producer holding two batches needs
some 125. THE FOURTH WARPGROUP'S REGISTERS CAN BE SPLIT BY WARP (a row): a drain warp, which holds 16 words, lowers
to 64 and a producer keeps the 128 it entered with and holds two batches; the compiler takes both counts (a release
to 64 and one to 96, the raise to 160). NO REGISTER IS LEFT for a fifth warpgroup: one beside these roles takes its registers from them. A form
of five warpgroups (640 threads) launches at 96 a thread, 61,440 in all; one whose roles' counts summed to the
whole file hung on the device, and whether at the folds' raise no run separated.

ROWS: the fold's side alone (producers and steps), the read side alone (the image, the readout and the drain),
both, and both with one access taken out: the producers' global loads (their words made in registers), the drains'
global adds (their lines folded into a sink), the image's stores. Then WHAT THE READOUT'S STEPS WAIT BEHIND: the
readout and the drains with no fold warp running (the image as it lies, never rewritten, no fold warp at a
barrier); every barrier's wait by the instruction that may suspend its thread (`mbarrier.try_wait`) where the
other rows poll (`test_wait` in a loop, an instruction every trip of every waiting warp); and the registers by
warp. THE FOLD'S, THE READOUT'S AND THE DRAIN WARPS'
WAITS at their barriers are clocked, and the readout's cycles FORMING its steps' A (the coefficients' rounds and
the 32 words held) apart from those IN ITS STEPS (the fence, 8 steps, the commit, the next chunk's 8 loads issued,
the wait). THE CARD'S CLOCK is read from the composed row: a
long call's span in SM cycles over its wall time. With both sides and nothing taken out, the fold's state and
every output row are checked exactly against the host's: the state summed in integers, the image its truncation to
bf16, a line its rounding to f16.

MEASURED (H100 SXM, 132 SMs, vast.ai instance 60394533, driver 610.57.04; 2026-11-12, the experiment at
18aee859; 7 repetitions, the composed row's 4,325,376 state values and 8,650,752 output values exact; the median,
the spread under 8%; SM-cycles a token are the window's cycles over 8 -- 128 CTAs of two boxes, 1,024 tokens):

| the row | cycles a window a CTA | SM-cycles a token |
|---|---|---|
| the fold's side alone: producers and steps | 5,755 | 719 |
| the read side alone: the image, the readout and the drain | 7,679 | 960 |
| the carry: both sides | 8,803 | 1,100 |
| both sides, the producers' global loads taken out | 7,931 | 991 |
| both sides, the drains' global adds taken out | 7,459 | 932 |
| both sides, the image's stores taken out | 8,007 | 1,001 |

THE CARD'S CLOCK, the composed row's long call: 1.78 GHz. At it a window's 8,803 cycles are 4.94 microseconds, and
64 windows -- 65,536 tokens' carry -- 0.316 ms.

A warp's clocked cycles a window:

| the role's count | both sides | the read side alone |
|---|---|---|
| a fold warp waits on the readout's read of a tile group | 451 | 4,038 |
| a fold warp waits on its producer | 3,430 | |
| a readout warp waits on a tile group's publish | 1,818 | 317 |
| a readout warp waits on the panel's drain | 360 | 785 |
| a readout warp in its steps | 3,023 | 4,206 |
| a drain warp waits on the panel | 3,090 | 2,476 |

WHAT IT SHOWS.
- THE CARRY COMPOSES, EXACTLY: two boxes' folds, the image a window, one readout warpgroup over both -- its
  coefficients expanded in its registers, its steps' A operand -- and one f16 line a token, in one CTA of 512
  threads at the SM's whole register file.
- A TOKEN COSTS 1,100 SM-CYCLES COMPOSED at this cell (its sides sparse on opposite levels, 256 x 256 leaves;
  [`carry_floor`](../internals/tools/carry_floor.md) prices a dense side higher).
- THE READOUT'S STEPS RUN FAR OVER THEIR OWN TIME, the cause not isolated by this record: a chunk's 8 steps are
  756 cycles of a readout warp's time composed and 1,052 with the read side alone, where 8 steps of 64 columns
  into one accumulator, both operands shared memory and nothing else running, are 319
  ([`wgmma_width`](wgmma_width.md)), and a register A is no slower than a shared one
  ([`wgmma_rate`](wgmma_rate.md)). The clocked span here also holds the 32 words of A formed and 8 loads issued.
- THE FOLD WAITS ON ITS PRODUCER 3,430 cycles of the window: a producer holds one batch in registers.
- THE TENSOR PIPE'S OWN WORK is a window's 32 steps of 16 columns and 32 of 64, some 1,700 cycles at their
  shared-memory rates.

Record: `measure/lab/results/carry_window/sm_90a.json`.
