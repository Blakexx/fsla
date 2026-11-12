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
some 125. NO REGISTER IS LEFT for a fifth warpgroup: one beside these roles takes its registers from them. A form
of five warpgroups (640 threads) launches at 96 a thread, 61,440 in all; one whose roles' counts summed to the
whole file hung on the device, and whether at the folds' raise no run separated.

ROWS: the fold's side alone (producers and steps), the read side alone (the image, the readout and the drain),
both, and both with one access taken out: the producers' global loads (their words made in registers), the drains'
global adds (their lines folded into a sink), the image's stores. Then WHAT THE READOUT'S STEPS WAIT BEHIND: the
readout and the drains with no fold warp running (the image as it lies, never rewritten, no fold warp at a
barrier); every barrier's wait by the instruction that may suspend its thread (`mbarrier.try_wait`) where the
other rows poll (`test_wait` in a loop, an instruction every trip of every waiting warp). THE FOLD'S, THE READOUT'S AND THE DRAIN WARPS'
WAITS at their barriers are clocked, and the readout's cycles FORMING its steps' A (the coefficients' rounds and
the 32 words held) apart from those IN ITS STEPS (the fence, 8 steps, the commit, the next chunk's 8 loads issued,
the wait). THE CARD'S CLOCK is read from the composed row: a
long call's span in SM cycles over its wall time. With both sides and nothing taken out, the fold's state and
every output row are checked exactly against the host's: the state summed in integers, the image its truncation to
bf16, a line its rounding to f16.

MEASURED (H100 SXM, 132 SMs, vast.ai instance 60398450, driver 595.91.07; 2026-11-12, the experiment at
bb5fa556; 7 repetitions, the composed rows' 4,325,376 state values and 8,650,752 output values exact; the median,
the spread under 3%; SM-cycles a token are the window's cycles over 8 -- 128 CTAs of two boxes, 1,024 tokens):

| the row | cycles a window a CTA | SM-cycles a token |
|---|---|---|
| the fold's side alone: producers and steps | 5,782 | 723 |
| the read side alone: the image, the readout and the drain | 7,812 | 977 |
| the carry: both sides | 8,765 | 1,096 |
| both sides, the producers' global loads taken out | 7,666 | 958 |
| both sides, the drains' global adds taken out | 7,363 | 920 |
| both sides, the image's stores taken out | 7,674 | 959 |
| the read side alone, no fold warp running | 4,209 | 526 |
| the read side alone, every wait suspended | 7,161 | 895 |
| both sides, every wait suspended | 8,486 | 1,061 |

THE CARD'S CLOCK, the composed row's long call, did not hold on this box: 1.56 falling to 1.28 GHz over five calls.
This record gives cycles and no time.

A warp's clocked cycles a window:

| the role's count | both sides | both, waits suspended | the read side alone | alone, waits suspended | alone, no fold warp |
|---|---|---|---|---|---|
| a fold warp waits on the readout's read of a tile group | 448 | 403 | 4,071 | 3,682 | |
| a fold warp waits on its producer | 3,193 | 3,331 | | | |
| a readout warp waits on a tile group's publish | 1,761 | 1,703 | 303 | 291 | |
| a readout warp waits on the panel's drain | 373 | 329 | 975 | 768 | 260 |
| a readout warp forming its steps' A | 1,446 | 1,465 | 1,055 | 1,093 | 1,079 |
| a readout warp in its steps | 2,351 | 2,390 | 3,727 | 2,962 | 2,191 |
| a drain warp waits on the panel | 2,896 | 2,887 | 2,634 | 2,281 | 697 |

WHAT IT SHOWS.
- THE CARRY COMPOSES, EXACTLY: two boxes' folds, the image a window, one readout warpgroup over both -- its
  coefficients expanded in its registers, its steps' A operand -- and one f16 line a token, in one CTA of 512
  threads at the SM's whole register file.
- A TOKEN COSTS 1,096 SM-CYCLES COMPOSED at this cell (its sides sparse on opposite levels, 256 x 256 leaves;
  [`carry_floor`](../internals/tools/carry_floor.md) prices a dense side higher).
- THE FOLD WARPS' PUBLISHING HALVES THE READ SIDE: with no fold warp running the read side is 4,209 cycles a
  window where it is 7,812 with them, and there the fold warps do nothing but store the image -- 32 stores of two
  bytes a thread a tile group -- and wait. The readout's steps are 2,191 without them and 3,727 with; the drain's
  wait on the panel 697 and 2,634.
- THE READOUT FORMS ITS STEPS' A IN 1,050 TO 1,450 CYCLES A WINDOW, 260 to 360 a chunk: the coefficients' rounds
  and the 32 words held, a third of what its steps and their forming take together.
- THE STEPS THEMSELVES ARE STILL OVER THEIR OWN TIME with nothing beside them but the drains: 548 cycles a chunk
  (the fence, 8 steps, the commit, 8 loads issued, the wait) where 8 steps of 64 columns into one accumulator from
  shared memory are 319 ([`wgmma_width`](wgmma_width.md)). Not isolated here.
- A SUSPENDED WAIT IS WORTH 3% COMPOSED (8,486 against 8,765) and 8% with the read side alone: the polling warps
  are not what holds the steps.
- THE FOLD WAITS ON ITS PRODUCER 3,193 cycles of the window: a producer holds one batch in registers.
- THE TENSOR PIPE'S OWN WORK is a window's 32 steps of 16 columns and 32 of 64, some 1,700 cycles at their
  shared-memory rates.

Record: `measure/lab/results/carry_window/sm_90a.json`.
