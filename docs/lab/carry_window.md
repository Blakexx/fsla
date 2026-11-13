# `carry_window`

Mirrors `measure/lab/experiments/carry_window.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The carry's roles were each measured apart: the fold's operands ([`pool_fill`](pool_fill.md)), the state's image
([`state_image`](state_image.md)), the readout and its drain ([`readout_role`](readout_role.md)). What does a window
cost a CTA when they run TOGETHER, under one register file, one tensor pipe and one shared memory?

THE FORM. A CTA an SM of four warpgroups, 512 threads.
- TWO FOLD warpgroups, a 16 × 16 box each, its state 16 accumulators in registers. Each PUBLISHES its state as a
  bf16 image of 16 tiles (the first window's is cleared memory), K-MAJOR -- a line of a tile is one channel's 8
  fold digits, one fold thread's own values, so a thread stores a tile group as 4 lines of 16 bytes, a warp's lanes on
  8 consecutive lines of 4 tiles (every bank four times, the fewest passes 512 bytes take); the readout's step reads
  B K-major -- a group of 4 tiles at a time, each once the readout
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
one warpgroup, to 96: the folds' 32 more each are those 32 fewer. At 152 the fold spills to local memory; with every step's A in its own registers (32 words) the toolkit serializes
the readout's steps at 80, and with A in two sets (16 words, the lean row) at 72; a producer holds two batches at
112. NO REGISTER IS LEFT for a fifth warpgroup: one beside these roles takes its registers from them. A form
of five warpgroups (640 threads) launches at 96 a thread, 61,440 in all, and its roles' counts sum to that
([`reg_five`](reg_five.md)): one whose counts summed to the SM's whole file hung on the device, a raise waiting
for registers the kernel never entered with.

ROWS: the fold's side alone (producers and steps), the read side alone (the image, the readout and the drain),
both, and both with one access taken out: the producers' global loads (their words made in registers), the drains'
global adds (their lines folded into a sink), the image's stores; and, both and each side alone, with THE READOUT
DRAINING ITS OWN ROWS -- a readout warp loads its 16 tokens' lines back from the panel and adds them itself, the
adds one whole line each as a drain warp's are, no drain warp and no panel barrier -- and all four warps of the
fourth warpgroup producers, two a box, each two of a slot's four batches; and, both and each side alone, with THE
READOUT LEAN -- a chunk's four digits take turns in TWO sets of A registers, a digit's two steps a commit group and
a set written again once the group two back is done, 16 words of A where every step's own are 32; the readout then
lowers to 80 registers a thread and the fourth warpgroup to 112, where a producer holds two batches, its loads two
batches ahead (at 72 the toolkit serializes the readout's steps); and, both and each side alone, with FIVE
WARPGROUPS -- 640 threads, which enter with 96 registers a thread ([`reg_five`](reg_five.md)): the folds at 160,
the lean readout at 80, a fourth warpgroup of four producers and a fifth of four drain warps at 40 each. At 40 a
producer holds one value instruction's four entries at a time and a drain warp one batch of 4 tokens, and neither
holds a clock or a count (a producer holding two instructions' entries fits at 48 and four's at 56; a drain warp
holding 8 tokens at 48); and, both and each side alone, with THE LONG WAITS ASLEEP: a fold's wait on its
producer, the readout's on a tile group's publish and a drain warp's on the panel are at NAMED barriers
(`bar.sync`), where a waiting warp issues nothing, in place of the polled test and branch a waiting warp issues
every trip; the short waits -- a slot freed, a tile group read, a panel drained -- stay polled. Then WHAT THE
READOUT'S STEPS WAIT BEHIND: the
readout and the drains with no fold warp running (the image as it lies, never rewritten, no fold warp at a
barrier); every barrier's wait by the instruction that may suspend its thread (`mbarrier.try_wait`) where the
other rows poll (`test_wait` in a loop, an instruction every trip of every waiting warp). THE FOLD'S, THE READOUT'S AND THE DRAIN WARPS'
WAITS at their barriers are clocked, and the readout's cycles FORMING its steps' A (the coefficients' rounds and
the 32 words held) apart from those IN ITS STEPS (the fence, 8 steps, the commit, the next chunk's 8 loads issued,
the wait). THE CARD'S CLOCK is read from the composed row: a
long call's span in SM cycles over its wall time. With both sides and nothing taken out, the fold's state and
every output row are checked exactly against the host's: the state summed in integers, the image its truncation to
bf16, a line its rounding to f16.

MEASURED (H100 SXM, 132 SMs, vast.ai instance 60433783, driver 580.173.02; 2026-11-13, the experiment at
760cfb1a, one session; 7 repetitions, the composed rows' 4,325,376 state values and 8,650,752 output values exact;
the median; SM-cycles a token are the window's cycles over 8 -- 128 CTAs of two boxes, 1,024 tokens):

| the row | cycles a window a CTA | SM-cycles a token |
|---|---|---|
| the fold's side alone: producers and steps | 5,983 | 748 |
| the read side alone: the image, the readout and the drain | 7,334 | 917 |
| the carry: both sides | 8,199 | 1,025 |
| both sides, the producers' global loads taken out | 8,242 | 1,030 |
| both sides, the drains' global adds taken out | 7,064 | 883 |
| both sides, the image's stores taken out | 7,995 | 999 |
| the read side alone, no fold warp running | 4,380 | 548 |
| the read side alone, every wait suspended | 7,075 | 884 |
| both sides, every wait suspended | 8,059 | 1,007 |
| THE READOUT DRAINING ITS OWN ROWS, FOUR PRODUCERS: the fold's side alone | 3,676 | 460 |
| the same: the read side alone | 7,247 | 906 |
| the same: both sides | 9,244 | 1,155 |
| THE READOUT LEAN, TWO BATCHES A PRODUCER: the fold's side alone | 5,688 | 711 |
| the same: the read side alone | 7,526 | 941 |
| the same: both sides | 8,836 | 1,105 |
| FIVE WARPGROUPS, FOUR PRODUCERS AND FOUR DRAIN WARPS AT 40: the fold's side alone | 6,330 | 791 |
| the same: the read side alone | 6,841 | 855 |
| the same: both sides | 9,109 | 1,139 |
| THE LONG WAITS ASLEEP AT NAMED BARRIERS: the fold's side alone | 5,880 | 735 |
| the same: the read side alone | 7,009 | 876 |
| the same: both sides | 8,129 | 1,016 |

THE CARD'S CLOCK, the composed row's long call: 1.79 to 1.80 GHz over five calls. At it a window's 8,199 cycles
are 4.55 microseconds, and 64 windows -- 65,536 tokens' carry -- 0.291 ms.

A warp's clocked cycles a window (the sleeping row's are not tabled: a clock read is not held to its side of a named
barrier, and its waits read as its neighbours' time):

| the role's count | both sides | the read side alone | alone, no fold warp | both, readout draining | both, readout lean | both, five warpgroups |
|---|---|---|---|---|---|---|
| a fold warp waits on the readout's read of a tile group | 432 | 4,203 |  | 3,480 | 632 | 905 |
| a fold warp waits on its producer | 5,191 |  |  | 734 | 4,291 | 5,651 |
| a readout warp waits on a tile group's publish | 2,100 | 292 |  | 437 | 885 | 2,526 |
| a readout warp waits on the panel's drain | 336 | 882 | 246 |  | 364 | 386 |
| a readout warp draining its own rows |  |  |  | 3,676 |  |  |
| a readout warp forming its steps' A | 1,391 | 1,098 | 1,262 | 1,196 |  |  |
| a readout warp in its steps (a lean readout's hold the forming) | 2,243 | 3,402 | 2,207 | 2,488 | 4,620 | 4,070 |
| a drain warp waits on the panel | 3,450 | 2,043 | 832 |  | 2,904 |  |

WHAT IT SHOWS.
- THE CARRY COMPOSES, EXACTLY: two boxes' folds, the image a window, one readout warpgroup over both -- its
  coefficients expanded in its registers, its steps' A operand -- and one f16 line a token, in one CTA of 512
  threads at the SM's whole register file.
- A TOKEN COSTS 1,025 SM-CYCLES COMPOSED at this cell (its sides sparse on opposite levels, 256 x 256 leaves;
  [`carry_floor`](../internals/tools/carry_floor.md) prices a dense side higher).
- THE FILL WANTS WARPS THAT EACH HOLD A BATCH: two producers a box, a batch of 32 entries in registers each, take
  the fold's side alone to 3,676 cycles a window where one a box is 5,983; one a box holding TWO batches is
  5,688; and two a box at 40 registers a thread, each holding one value instruction's four entries, are
  6,330 -- slower than one a box that holds a batch.
- A WAITING WARP'S POLLING IS NOT WHAT HOLDS THE OTHERS: with the three long waits asleep at named barriers the
  window is 8,129 against 8,199, the fold's side alone 5,880 against 5,983 and the read side alone 7,009
  against 7,334.
- FIVE WARPGROUPS LOSE COMPOSED, 9,109 against 8,199: 640 threads leave four producers and four drain warps 40
  registers a thread, the fold waits on its producers 5,651 cycles a window, and the lean readout the count
  forces is 4,070 in its steps and their forming where one commit group's are 3,635. Four drain warps do
  help the read side alone, 6,841 against 7,334.
- THE LEAN READOUT WITH TWO-BATCH PRODUCERS IS 8,836 COMPOSED against 8,199: a readout warp is 4,620 cycles
  a window in its steps and their forming. The two changes are not separated.
- THE READOUT DRAINING ITS OWN ROWS LOSES COMPOSED, 9,244 against 8,199: a readout warp is 3,676 cycles a
  window in its 64 adds, and the window becomes the readout's -- the fold waits on its read 3,480. A warp that
  issues an add is held until the SM passes it: adds issued by the readout hold the readout.
- THE DRAINS' ADDS ARE THE LARGEST SINGLE ACCESS: 7,064 with them out; the producers' loads out, 8,242; the
  image's stores out, 7,995.
- THE READ SIDE IS HELD BY THE FOLD WARPS BESIDE IT, by more than their stores or their polling: alone it is 7,334
  cycles a window with the fold warps publishing and waiting, 7,009 with their waits asleep, and 4,380 with no
  fold warp running -- a row that also takes out the readout's wait on the publish. Not isolated.
- THE READOUT FORMS ITS STEPS' A IN 1,098 TO 1,391 CYCLES A WINDOW, 274 to 348 a chunk: the coefficients' rounds
  and the 32 words held.
- THE STEPS THEMSELVES ARE OVER THEIR OWN TIME with nothing beside them but the drains: 552 cycles a chunk (the
  fence, 8 steps, the commit, 8 loads issued, the wait) where 8 steps of 64 columns into one accumulator from
  shared memory are 319 ([`wgmma_width`](wgmma_width.md)). Not isolated here.
- EVERY ROLE ISSUES AN INSTRUCTION EVERY 3.5 TO 7 CYCLES of its busy time (the static ledger's counts a trip: a
  producer 168 a batch, a drain warp 339 a chunk, the readout 284 a chunk, a fold warp 379 a window): no role is
  bound by the instructions it issues, and what holds each is not isolated.

Record: `measure/lab/results/carry_window/sm_90a.json`.
