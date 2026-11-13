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
  8 consecutive lines of 4 tiles (every bank four times, the fewest passes 512 bytes take, at a tile pitch of an odd count of lines: the lanes' four tiles are two apart, and an even pitch puts two of them on the same banks); the readout's step reads
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
images 82 KB (room for tiles 161 lines apart; 70 at the num's 137), two panels 35 KB.

THE REGISTERS (`setmaxnreg`, each role's body in its own branch). 512 threads launch at 128 registers a thread, the
SM's whole file of 65,536. A fold thread raises to 160; a readout thread lowers to 96, and the producers and drains,
one warpgroup, to 96: the folds' 32 more each are those 32 fewer. At 152 the fold spills to local memory; with every step's A in its own registers (32 words) the toolkit serializes
the readout's steps at 80, and with A in two sets (16 words, the lean row) at 72; a producer holds two batches at
112. The den's 4 fold accumulators fit the fold's 160 (a 17th set of 8 would not: the toolkit keeps one in local
memory and serializes the steps); the den's 4 readout accumulators take the lean readout to 88 (serialized at
80) and the fourth warpgroup to 104, its producers still two batches. NO REGISTER IS LEFT for a fifth warpgroup: one beside these roles takes its registers from them. A form
of five warpgroups (640 threads) launches at 96 a thread, 61,440 in all, and its roles' counts sum to that
([`reg_five`](reg_five.md)): one whose counts summed to the SM's whole file hung on the device, a raise waiting
for registers the kernel never entered with.

THE DEN (its own rows). The den is the num with every value 1: a box's den state is a value a (list, digit), 256
where the num's is 16,384.
- FOLDED with the coefficients as the step's A. A step's 64 rows are too many for it as a value row -- a step a
  list with a tile of ones in one row is 16 steps a window for 1 row in 64 -- so A is 8 lists' coefficient bricks
  on 8 digits, read where the producers stored them (a descriptor whose bricks are a pool tile apart), and B a
  constant brick of ones in one column. TWO STEPS A SLOT, four a window, add into one accumulator set of 4
  registers a thread, a column a step.
- PUBLISHED as a NINTH BRICK of each image tile (the den's rows' tiles are 155 lines apart, the others' 137; rows
  that move the pitch alone price it): its first channel's two lines are the den's, a fold thread that holds the column storing one word of two lists a tile group.
- READ as column 64 of the readout's steps, 72 columns wide where the num's are 64: no step is added. A readout
  thread holds 4 more accumulators, and a token's den is an fp32 behind its panel row.
- DRAINED by one more load and one more add a drain warp a chunk, a lane a token.
The den's state and every token's den sum are checked exactly against the host's.

THE ALIGNED CELL (its own rows). Where both sides list the same level, a reader reads one list on the operand
level's 16 digits, so an image tile is a LIST and its K the digits: the image lies along the fold's own lists.
A line is then four fold threads' values, a word each -- a thread stores a tile group as 16 words where the other
cell's is 4 lines, a warp's lanes on 128 consecutive bytes -- and the den's line is 8 threads' values, 2 bytes
each. The readout, the panel and the drains are the same code. It prices the image form where the carves are
equal, in place of a form of its own.

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
once-read rows are the price (R22.2: a cache that may not hold is priced as if it does not) and the cached-table
rows the upside: a streamed carry reads a token's rows in its arrival window, so one CTA's window working set is
small at any length, while 132 CTAs' phases can drift apart (the open-work list carries that). The suspended and the sleeping waits also run combined with the lean readout and the wide byte load. Then WHAT THE
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

MEASURED (H100 80GB HBM3, 132 SMs, vast.ai instance 60471067, driver 615.71.09; 2026-11-13, the experiment at
bfb1f87b, one session; 7 repetitions, every checked row's state and output values exact;
the median; SM-cycles a token are the window's cycles over 8 -- 128 CTAs of two boxes, 1,024 tokens. THE FOLD'S STATE
DUMP STAYS OUT OF THE TIMED CALLS, and one row restores it to price it in this session):

| the row | cycles a window a CTA | SM-cycles a token |
|---|---|---|
| the fold's side alone: producers and steps | 5,079 | 635 |
| the read side alone: the image, the readout and the drain | 5,564 | 696 |
| the carry: both sides | 7,091 | 886 |
| both sides, the producers' global loads taken out | 6,445 | 806 |
| both sides, the drains' global adds taken out | 6,776 | 847 |
| both sides, the image's stores taken out | 6,741 | 843 |
| the read side alone, no fold warp running | 4,391 | 549 |
| the read side alone, every wait suspended | 5,270 | 659 |
| both sides, every wait suspended | 6,903 | 863 |
| THE READOUT DRAINING ITS OWN ROWS, FOUR PRODUCERS: the fold's side alone | 3,095 | 387 |
| the same: the read side alone | 5,378 | 672 |
| the same: both sides | 7,576 | 947 |
| THE READOUT LEAN, TWO BATCHES A PRODUCER: the fold's side alone | 4,667 | 583 |
| the same: the read side alone | 5,767 | 721 |
| the same: both sides | 6,966 | 871 |
| FIVE WARPGROUPS, FOUR PRODUCERS AND FOUR DRAIN WARPS AT 40: the fold's side alone | 5,952 | 744 |
| the same: the read side alone | 5,045 | 631 |
| the same: both sides | 7,848 | 981 |
| THE LONG WAITS ASLEEP AT NAMED BARRIERS: the fold's side alone | 5,071 | 634 |
| the same: the read side alone | 5,062 | 633 |
| the same: both sides | 6,976 | 872 |
| the fold's side alone, the producers' global loads taken out | 2,440 | 305 |
| the fold's side alone, their coefficient arithmetic taken out | 4,674 | 584 |
| the fold's side alone, both taken out | 2,338 | 292 |
| the read side alone, the readout's byte loads taken out | 5,003 | 625 |
| the read side alone, no fold warp running, the byte loads taken out | 4,049 | 506 |
| both sides, the readout's byte loads taken out | 6,791 | 849 |
| both sides, the drains' adds and the readout's byte loads taken out | 6,571 | 821 |
| THE READOUT'S BYTES ONE WIDE LOAD A WARP: the read side alone | 5,045 | 631 |
| the same: the read side alone, no fold warp running | 4,222 | 528 |
| the same: both sides | 7,050 | 881 |
| the same: both sides, the long waits asleep | 7,085 | 886 |
| the read side alone, the image's stores taken out | 4,993 | 624 |
| the read side alone, the fold's fence behind its image taken out | 5,505 | 688 |
| the read side alone, both taken out | 4,982 | 623 |
| both sides, that fence taken out | 7,004 | 875 |
| the read side alone, no fold warp running, the fold warps parked asleep | 4,371 | 546 |
| the same, spinning | 4,505 | 563 |
| THE DUMP RESTORED: both sides, the fold's state stored out in every call | 8,159 | 1,020 |
| THE WINNERS COMBINED: both sides, suspended waits and the lean readout | 6,763 | 845 |
| the same, suspended waits and the wide byte load | 6,855 | 857 |
| the same, suspended waits, the lean readout and the wide byte load (the fourth warpgroup at 104) | 6,849 | 856 |
| the same, sleeping waits, the lean readout and the wide byte load | 7,017 | 877 |
| THE TABLES ONCE-READ, 16 replicas the cache cannot hold: both sides | 17,457 | 2,182 |
| the same, the winner's form | 16,101 | 2,013 |
| the same: the fold's side alone | 11,679 | 1,460 |
| THE DEN, folded, published, read and drained: the fold's side alone | 5,359 | 670 |
| the same: both sides, the winner's form | 7,080 | 885 |
| the same, the tables once-read | 16,312 | 2,039 |
| THE IMAGE'S TILES MOVED APART, nothing else changed: the winner's form at 154 lines (137 as it is) | 7,123 | 890 |
| the winner's form with the den at 161 lines (155 as it is) | 7,082 | 885 |
| THE ALIGNED CELL, an image tile a list: the winner's form | 6,742 | 843 |
| the same with the den | 7,185 | 898 |

THE WALL TIME, measured on this box and immune to its clock (1.78 to 1.81 GHz over the session): the long call
scaled to 65,536 tokens is 0.251 ms for the baseline, 0.244 ms for the winner's form, and 0.524 ms at the
once-read bound (0.589 without the winner's changes). THE FLASH REFERENCES, from the session of instance 60450651 ON ANOTHER BOX (a different offer and driver, whose
once-read rows were a third fewer cycles than this box's)
(`rola_bench.measure.attention`, causal, 1 head, dv 64, bf16; the records in rola-results at `flash/reference`):
FlashAttention-3, built on the box at Dao's pinned commit, 1.775 ms at 65,536 tokens (0.300 at 16,384; 0.121 at
1,024); torch's FlashAttention-2 3.072 ms (0.323; 0.040). THESE ARE ADJACENT MEASUREMENTS, NOT A COMPARISON: the
carry's wall time is this lab's and the flash times the bench's attention entry's, two processes on a box with no
clock lock, and the carry is one part of the op (no intra, facts pass, den or join). Flash is a LAYER entry, so
the comparison is a layer comparison in the declared suite ([`measurement.md`](../measurement.md)), owed when a
layer of this op is a session member beside it. THE PRICE (R22.2) is the once-read wall time -- a cache that may
not hold the working set is priced as if it does not -- and the cached-table time is the upside when the cache
holds; whether a streamed carry's arrivals keep it there is unmeasured.

A warp's clocked cycles a window (the sleeping row's are not tabled: a clock read is not held to its side of a named
barrier, and its waits read as its neighbours' time):

| the role's count | both sides | the read side alone | alone, byte loads out | alone, no fold warp | alone, no fold warp, byte loads out | both, readout draining | both, five warpgroups |
|---|---|---|---|---|---|---|---|
| a fold warp waits on the readout's read of a tile group | 361 | 4,184 | 3,803 |  |  | 3,704 | 410 |
| a fold warp waits on its producer | 4,656 |  |  |  |  | 422 | 5,373 |
| a readout warp waits on a tile group's publish | 1,842 | 291 | 323 |  |  | 300 | 2,258 |
| a readout warp waits on the panel's drain | 330 | 287 | 273 | 245 | 268 |  | 327 |
| a readout warp draining its own rows |  |  |  |  |  | 2,817 |  |
| a readout warp forming its steps' A | 1,395 | 1,104 | 1,108 | 1,274 | 1,022 | 1,135 |  |
| a readout warp in its steps (a lean readout's hold the forming) | 2,192 | 2,582 | 2,130 | 2,206 | 2,082 | 1,980 | 3,923 |
| a drain warp waits on the panel | 3,101 | 2,050 | 1,517 | 839 | 668 |  |  |

WHAT IT SHOWS.
- THE CARRY COMPOSES, EXACTLY: two boxes' folds, the image a window, one readout warpgroup over both -- its
  coefficients expanded in its registers, its steps' A operand -- and one f16 line a token, in one CTA of 512
  threads at the SM's whole register file.
- A TOKEN IS PRICED AT 2,013 SM-CYCLES COMPOSED, its tables read once (R22.2), in the winner's form; 845 when the
  cache holds them (886 in the first form) at this cell (its sides sparse on opposite levels, 256 x 256 leaves;
  [`carry_floor`](../internals/tools/carry_floor.md) prices a dense side higher).
- THE FILL WANTS WARPS THAT EACH HOLD A BATCH: two producers a box, a batch of 32 entries in registers each, take
  the fold's side alone to 3,095 cycles a window where one a box is 5,079; one a box holding TWO batches is
  4,667; and two a box at 40 registers a thread, each holding one value instruction's four entries, are
  5,952 -- slower than one a box that holds a batch.
- SUSPENDED WAITS WITH THE LEAN READOUT ARE THE FASTEST COMPOSED FORM: 6,763 cycles a window against the
  baseline's 7,091, with the suspended waits beside the wide byte load a near tie at 6,855. The parts: every wait
  suspended alone 6,903, the long waits asleep at named barriers 6,976, the lean readout alone within 1.5%
  of the baseline, the wide load alone 7,050. ALL THREE AT ONCE LOSE the pairing's gain (6,849: the lean
  readout beside the wide load's scratch needs 88 registers a thread and the fourth warpgroup drops to 104), and
  the sleeping triple is 7,017.
- THE DEN COSTS 4.7% OF THE COMPOSED WINDOW AT THIS CELL, the tables cached: 7,080 cycles in the winner's form against
  6,763 without it; with the tables once-read 16,312 against 16,101; the fold's side alone 5,359 against 5,079. One
  cell (its sides sparse on opposite levels); a dense write side, where the fold's side binds, is unmeasured. WHAT
  OF IT IS WHICH PART IS NOT ISOLATED: the fold's four steps a window, the readout's steps 72 columns wide where
  they were 64, its 88 registers and the fourth warpgroup's 104 where they were 80 and 112, the den's stores and
  the drain's add.
- THE FOLD'S FORM WAS A CONTEST. A step a list with ones in one row of A is 16 steps a window for 1 row in 64 and
  a 17th accumulator set of 8, which the toolkit will not hold at 160 registers: at a 168-register fold with
  one-batch producers it cost an eighth of the composed window in its session (the experiment at ac342446, whose
  record this one replaced). The coefficients as A are four steps into 4 registers.
- AN IMAGE TILE'S PITCH MUST BE AN ODD COUNT OF LINES: a warp's lanes store to four tiles two apart, and at an
  even pitch two of the four land on the same banks. The winner's form at 154 lines is 7,123 against 6,763 at 137;
  the den's at 161 is 7,082 against 7,080 at 155.
- THE IMAGE FORM COSTS THE SAME WHERE THE CARVES ARE EQUAL: the aligned cell's winner's form is 6,742 cycles a
  window against 6,763 at the cell whose sides list opposite levels -- 16 word stores a thread a tile group where
  that cell's are 4 lines, at no cost -- and with the den 7,185 against 7,080, the den's lines there 2-byte
  stores of 8 threads. A form that reads the fold's own registers has nothing of the window to win back but the
  image's stores; it is not built.
- THE TABLES ONCE-READ COST 1.7 TO 2.2 TIMES THE WINDOW: 17,457 cycles composed against 7,091, the winner's form
  16,101 against 6,763, the fold's side alone 11,679 against 5,079 -- every access a fresh row of 128 MB of
  values, the model of a large table whose rows are each read once.
- THE DUMP IS PRICED IN THIS SESSION: the fold's state stored out in every call, as every record before 79ec1b08
  ran, puts the composed row at 8,159 against 7,091 -- one binary, one box, one session.
- FIVE WARPGROUPS LOSE COMPOSED, 7,848 against 7,091: 640 threads leave four producers and four drain warps 40
  registers a thread, the fold waits on its producers 5,373 cycles a window, and the lean readout the count
  forces is 3,923 in its steps and their forming where one commit group's are 3,587. Four drain warps do
  help the read side alone, 5,045 against 5,564.
- THE LEAN READOUT WITH TWO-BATCH PRODUCERS IS 6,966 COMPOSED against 7,091: a readout warp is 4,468 cycles
  a window in its steps and their forming. The two changes are not separated.
- THE READOUT DRAINING ITS OWN ROWS LOSES COMPOSED, 7,576 against 7,091: a readout warp is 2,817 cycles a
  window in its 64 adds, and the window becomes the readout's -- the fold waits on its read 3,704. A warp that
  issues an add is held until the SM passes it: adds issued by the readout hold the readout.
- COMPOSED, THE PRODUCERS' LOADS ARE NOW THE LARGEST SINGLE ACCESS: with them out the window is 6,445 against 7,091, and
  those loads are three fifths of the fold's side alone -- 2,440 cycles a window without them against 5,079, some
  165 cycles a batch of a producer's 9 load instructions; their coefficient arithmetic is 405 of it
  (4,674 without). The drains' adds out, the window is 6,776; the image's stores out, 6,741.
- A PRESENT FOLD WARP COSTS THE READ SIDE NOTHING, A PUBLISHING ONE 1,173 CYCLES: the read side alone is 5,564
  with the fold warps publishing and 4,391 with no fold warp running, while one parked asleep beside it costs
  -19 cycles and one spinning 114. What of the publishing fold's hand-off holds it is NOT ISOLATED
  (its stores are 571 of it; its fence nothing).
- THE READOUT'S OWN BYTE LOADS ARE A SIXTH OF ITS STEPS' SPAN: 8 instructions a chunk, each lane's 4 bytes of a
  different row; without them a readout warp's steps are 2,130 cycles a window against 2,582. The read side alone
  is then the drain's -- 5,003 against 5,564, the readout waiting on the drain 273 -- at some
  27 cycles an add a drain warp, where with no fold warp running an add is 26.
- ONE WIDE LOAD A WARP FOR THE READOUT'S BYTES is the read side alone without those loads -- 5,045 cycles a window
  against 5,003 with them out and 5,564 as they were -- and little composed: 7,050 against 7,091, where the
  byte loads out are 6,791 and the drains' adds out with them 6,571.
- THE READOUT FORMS ITS STEPS' A IN 1,104 TO 1,395 CYCLES A WINDOW, 276 to 349 a chunk: the coefficients' rounds
  and the 32 words held.
- THE STEPS THEMSELVES ARE OVER THEIR OWN TIME with nothing beside them but the drains: 552 cycles a chunk (the
  fence, 8 steps, the commit, 8 loads issued, the wait) where 8 steps of 64 columns into one accumulator from
  shared memory are 319 ([`wgmma_width`](wgmma_width.md)). Not isolated here.
- EVERY ROLE ISSUES AN INSTRUCTION EVERY 3.5 TO 7 CYCLES of its busy time (the static ledger's counts a trip: a
  producer 168 a batch, a drain warp 339 a chunk, the readout 284 a chunk, a fold warp 379 a window): no role is
  bound by the instructions it issues, and what holds each is not isolated.

Record: `measure/lab/results/carry_window/sm_90a.json`.
