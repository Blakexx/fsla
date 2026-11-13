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
producer and on the readout's read of a tile group, the readout's on a tile group's publish and a drain warp's on
the panel are at NAMED barriers (`bar.sync`), where a waiting warp issues nothing, in place of the polled test and
branch a waiting warp issues every trip; a slot freed and a panel drained stay polled; and each side alone with
one more access out: the producers' loads, their coefficient arithmetic (the bytes stored as they are), both, and
the readout's loads of its readers' bytes (the words made in registers), with the fold warps and without, and
both sides with those byte loads out and with the drains' adds out beside them; and with those bytes ONE WIDE LOAD
a readout warp a chunk: a lane fetches a whole row's 16 bytes (the warp's 16 tokens on the two boxes, 32 rows an
instruction), the warp stores them to its own scratch in shared memory and each thread reads its 8 words back --
one instruction where 8 are, each of whose 32 lanes takes 4 bytes of one of 8 rows; and the read side alone with
the fold's image stores, its fence behind them, or both taken out, and both sides with that fence out. Two rows
keep the fold warps PRESENT AND IDLE beside the readout and the drains for the whole call -- parked at a named
barrier, or spinning at a polled one -- where the row with no fold warp running has them leave. One row RESTORES
the fold's state dump to every call, as every record before this form had it: the A/B that prices the dump in one
session. Rows whose TABLES THE CACHE CANNOT HOLD read every access's row at a replica of the table picked by
the access's own position -- a replica's row holds its token's values, so the check stands -- as a large table's
rows are each read once; 16 replicas, 128 MB of values (composed, the winner's form, and the fold's side alone). The long call and its wall milliseconds (scaled to 65,536 tokens) run for the headline's
rows -- the baseline, the winner's form and the two big-table composed rows -- each record named by its row. The
once-read price is R22.2's bound and a cold-latency fact rather than the expected operating point: a streamed
carry reads a token's rows in its arrival window, so ONE CTA's window working set is small at any length --
though 132 CTAs' phases can drift apart (the open-work list carries that), so the resident and once-read rows
bracket production rather than naming its point. The suspended and the sleeping waits also run combined with the lean readout and the wide byte load. Then WHAT THE
READOUT'S STEPS WAIT BEHIND: the
readout and the drains with no fold warp running (the image as it lies, never rewritten, no fold warp at a
barrier); every barrier's wait by the instruction that may suspend its thread (`mbarrier.try_wait`) where the
other rows poll (`test_wait` in a loop, an instruction every trip of every waiting warp). THE FOLD'S, THE READOUT'S AND THE DRAIN WARPS'
WAITS at their barriers are clocked, and the readout's cycles FORMING its steps' A (the coefficients' rounds and
the 32 words held) apart from those IN ITS STEPS (the fence, 8 steps, the commit, the next chunk's 8 loads issued,
the wait). THE CARD'S CLOCK and the long call's WALL TIME are read from each
headline row's own long calls -- the baseline, the winner's form and the big-table composed rows -- each record
named by its row. With both sides and nothing taken out, the fold's state and
every output row are checked exactly against the host's: the state summed in integers, the image its truncation to
bf16, a line its rounding to f16. The fold's state is stored out for that checked call alone: a timed call leaves it in
registers, since its 128 stores a thread -- each instruction's lanes on 32 lines -- would be issued while the
readout and the drains are in their last windows and be clocked as theirs.

MEASURED (H100 SXM, 132 SMs, vast.ai instance 60450651, driver 595.91.07; 2026-11-13, the experiment at
1c2621b3, one session; 7 repetitions, the composed rows' 4,325,376 state values and 8,650,752 output values exact;
the median; SM-cycles a token are the window's cycles over 8 -- 128 CTAs of two boxes, 1,024 tokens. THE FOLD'S STATE
DUMP STAYS OUT OF THE TIMED CALLS, and one row restores it to price it in this session):

| the row | cycles a window a CTA | SM-cycles a token |
|---|---|---|
| the fold's side alone: producers and steps | 5,063 | 633 |
| the read side alone: the image, the readout and the drain | 5,559 | 695 |
| the carry: both sides | 7,022 | 878 |
| both sides, the producers' global loads taken out | 6,439 | 805 |
| both sides, the drains' global adds taken out | 6,675 | 834 |
| both sides, the image's stores taken out | 6,634 | 829 |
| the read side alone, no fold warp running | 4,281 | 535 |
| the read side alone, every wait suspended | 5,256 | 657 |
| both sides, every wait suspended | 6,821 | 853 |
| THE READOUT DRAINING ITS OWN ROWS, FOUR PRODUCERS: the fold's side alone | 3,048 | 381 |
| the same: the read side alone | 5,380 | 672 |
| the same: both sides | 7,332 | 916 |
| THE READOUT LEAN, TWO BATCHES A PRODUCER: the fold's side alone | 4,589 | 574 |
| the same: the read side alone | 5,760 | 720 |
| the same: both sides | 6,910 | 864 |
| FIVE WARPGROUPS, FOUR PRODUCERS AND FOUR DRAIN WARPS AT 40: the fold's side alone | 5,928 | 741 |
| the same: the read side alone | 5,039 | 630 |
| the same: both sides | 7,661 | 958 |
| THE LONG WAITS ASLEEP AT NAMED BARRIERS: the fold's side alone | 5,062 | 633 |
| the same: the read side alone | 5,055 | 632 |
| the same: both sides | 6,847 | 856 |
| the fold's side alone, the producers' global loads taken out | 2,440 | 305 |
| the fold's side alone, their coefficient arithmetic taken out | 4,652 | 581 |
| the fold's side alone, both taken out | 2,338 | 292 |
| the read side alone, the readout's byte loads taken out | 5,002 | 625 |
| the read side alone, no fold warp running, the byte loads taken out | 4,049 | 506 |
| both sides, the readout's byte loads taken out | 6,728 | 841 |
| both sides, the drains' adds and the readout's byte loads taken out | 6,537 | 817 |
| THE READOUT'S BYTES ONE WIDE LOAD A WARP: the read side alone | 5,045 | 631 |
| the same: the read side alone, no fold warp running | 4,198 | 525 |
| the same: both sides | 6,962 | 870 |
| the same: both sides, the long waits asleep | 6,978 | 872 |
| the read side alone, the image's stores taken out | 4,993 | 624 |
| the read side alone, the fold's fence behind its image taken out | 5,483 | 685 |
| the read side alone, both taken out | 4,982 | 623 |
| both sides, that fence taken out | 6,920 | 865 |
| the read side alone, no fold warp running, the fold warps parked asleep | 4,353 | 544 |
| the same, spinning | 4,489 | 561 |
| THE DUMP RESTORED: both sides, the fold's state stored out in every call | 8,163 | 1,020 |
| THE WINNERS COMBINED: both sides, suspended waits and the lean readout | 6,750 | 844 |
| the same, suspended waits and the wide byte load | 6,764 | 845 |
| the same, suspended waits, the lean readout and the wide byte load (the fourth warpgroup at 104) | 6,872 | 859 |
| the same, sleeping waits, the lean readout and the wide byte load | 7,091 | 886 |
| THE TABLES ONCE-READ, 16 replicas the cache cannot hold: both sides | 15,379 | 1,922 |
| the same, the winner's form | 11,629 | 1,454 |
| the same: the fold's side alone | 8,951 | 1,119 |

THE WALL TIME, measured on this box and immune to its clock (which drifted, 1.30 to 1.58 GHz): the long call
scaled to 65,536 tokens is 0.330 ms for the baseline, 0.313 ms for the winner's form, and 0.568 ms at the
once-read bound (0.627 without the winner's changes). THE FLASH REFERENCES, THE SAME SESSION on this box
(`rola_bench.measure.attention`, causal, 1 head, dv 64, bf16; the records in rola-results at `flash/reference`):
FlashAttention-3, built on the box at Dao's pinned commit, 1.775 ms at 65,536 tokens (0.300 at 16,384; 0.121 at
1,024); torch's FlashAttention-2 3.072 ms (0.323; 0.040). WALL AGAINST WALL IN ONE SESSION the winner's carry is
5.7x under FA3 and 9.8x under FA2, and 3.1x under FA3 at the once-read bound -- the carry alone: the intra, the
facts pass, the den and the join are not in it.

A warp's clocked cycles a window (the sleeping row's are not tabled: a clock read is not held to its side of a named
barrier, and its waits read as its neighbours' time):

| the role's count | both sides | the read side alone | alone, byte loads out | alone, no fold warp | alone, no fold warp, byte loads out | both, readout draining | both, five warpgroups |
|---|---|---|---|---|---|---|---|
| a fold warp waits on the readout's read of a tile group | 379 | 4,182 | 3,800 |  |  | 3,615 | 410 |
| a fold warp waits on its producer | 4,561 |  |  |  |  | 434 | 5,192 |
| a readout warp waits on a tile group's publish | 1,771 | 291 | 323 |  |  | 300 | 2,102 |
| a readout warp waits on the panel's drain | 334 | 287 | 273 | 253 | 268 |  | 328 |
| a readout warp draining its own rows |  |  |  |  |  | 2,742 |  |
| a readout warp forming its steps' A | 1,390 | 1,101 | 1,108 | 1,169 | 1,022 | 1,174 |  |
| a readout warp in its steps (a lean readout's hold the forming) | 2,190 | 2,579 | 2,128 | 2,190 | 2,081 | 1,963 | 3,892 |
| a drain warp waits on the panel | 3,011 | 2,048 | 1,513 | 737 | 668 |  |  |

WHAT IT SHOWS.
- THE CARRY COMPOSES, EXACTLY: two boxes' folds, the image a window, one readout warpgroup over both -- its
  coefficients expanded in its registers, its steps' A operand -- and one f16 line a token, in one CTA of 512
  threads at the SM's whole register file.
- A TOKEN COSTS 878 SM-CYCLES COMPOSED at this cell (its sides sparse on opposite levels, 256 x 256 leaves;
  [`carry_floor`](../internals/tools/carry_floor.md) prices a dense side higher).
- THE FILL WANTS WARPS THAT EACH HOLD A BATCH: two producers a box, a batch of 32 entries in registers each, take
  the fold's side alone to 3,048 cycles a window where one a box is 5,063; one a box holding TWO batches is
  4,589; and two a box at 40 registers a thread, each holding one value instruction's four entries, are
  5,928 -- slower than one a box that holds a batch.
- SUSPENDED WAITS WITH THE LEAN READOUT ARE THE FASTEST COMPOSED FORM: 6,750 cycles a window against the
  baseline's 7,022, with the suspended waits beside the wide byte load a near tie at 6,764. The parts: every wait
  suspended alone 6,821, the long waits asleep at named barriers 6,847, the lean readout alone within 1.5%
  of the baseline, the wide load alone 6,962. ALL THREE AT ONCE LOSE the pairing's gain (6,872: the lean
  readout beside the wide load's scratch needs 88 registers a thread and the fourth warpgroup drops to 104), and
  the sleeping triple is 7,091.
- THE TABLES ONCE-READ MORE THAN DOUBLE THE WINDOW: 15,379 cycles composed against 7,022, the winner's form
  11,629 against 6,750, the fold's side alone 8,951 against 5,063 -- every access a fresh row of 128 MB of
  values, the model of a large table whose rows are each read once. Locality of a token's reads is worth a
  window's half.
- THE DUMP IS PRICED IN THIS SESSION: the fold's state stored out in every call, as every record before 79ec1b08
  ran, puts the composed row at 8,163 against 7,022 -- one binary, one box, one session.
- FIVE WARPGROUPS LOSE COMPOSED, 7,661 against 7,022: 640 threads leave four producers and four drain warps 40
  registers a thread, the fold waits on its producers 5,192 cycles a window, and the lean readout the count
  forces is 3,892 in its steps and their forming where one commit group's are 3,580. Four drain warps do
  help the read side alone, 5,039 against 5,559.
- THE LEAN READOUT WITH TWO-BATCH PRODUCERS IS 6,910 COMPOSED against 7,022: a readout warp is 4,368 cycles
  a window in its steps and their forming. The two changes are not separated.
- THE READOUT DRAINING ITS OWN ROWS LOSES COMPOSED, 7,332 against 7,022: a readout warp is 2,742 cycles a
  window in its 64 adds, and the window becomes the readout's -- the fold waits on its read 3,615. A warp that
  issues an add is held until the SM passes it: adds issued by the readout hold the readout.
- COMPOSED, THE PRODUCERS' LOADS ARE NOW THE LARGEST SINGLE ACCESS: with them out the window is 6,439 against 7,022, and
  those loads are three fifths of the fold's side alone -- 2,440 cycles a window without them against 5,063, some
  164 cycles a batch of a producer's 9 load instructions; their coefficient arithmetic is 411 of it
  (4,652 without). The drains' adds out, the window is 6,675; the image's stores out, 6,634.
- A PRESENT FOLD WARP COSTS THE READ SIDE NOTHING, A PUBLISHING ONE 1,278 CYCLES: the read side alone is 5,559
  with the fold warps publishing and 4,281 with no fold warp running, while one parked asleep beside it costs
  72 cycles and one spinning 208. What of the publishing fold's hand-off holds it is NOT ISOLATED
  (its stores are 567 of it; its fence nothing).
- THE READOUT'S OWN BYTE LOADS ARE A SIXTH OF ITS STEPS' SPAN: 8 instructions a chunk, each lane's 4 bytes of a
  different row; without them a readout warp's steps are 2,128 cycles a window against 2,579. The read side alone
  is then the drain's -- 5,002 against 5,559, the readout waiting on the drain 273 -- at some
  27 cycles an add a drain warp, where with no fold warp running an add is 26.
- ONE WIDE LOAD A WARP FOR THE READOUT'S BYTES is the read side alone without those loads -- 5,045 cycles a window
  against 5,002 with them out and 5,559 as they were -- and little composed: 6,962 against 7,022, where the
  byte loads out are 6,728 and the drains' adds out with them 6,537.
- THE READOUT FORMS ITS STEPS' A IN 1,101 TO 1,390 CYCLES A WINDOW, 275 to 348 a chunk: the coefficients' rounds
  and the 32 words held.
- THE STEPS THEMSELVES ARE OVER THEIR OWN TIME with nothing beside them but the drains: 548 cycles a chunk (the
  fence, 8 steps, the commit, 8 loads issued, the wait) where 8 steps of 64 columns into one accumulator from
  shared memory are 319 ([`wgmma_width`](wgmma_width.md)). Not isolated here.
- EVERY ROLE ISSUES AN INSTRUCTION EVERY 3.5 TO 7 CYCLES of its busy time (the static ledger's counts a trip: a
  producer 168 a batch, a drain warp 339 a chunk, the readout 284 a chunk, a fold warp 379 a window): no role is
  bound by the instructions it issues, and what holds each is not isolated.

Record: `measure/lab/results/carry_window/sm_90a.json`.
