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

THE DEN'S PARTS (a chain of rows, no clock inside a role). Each row adds one part to the row before, so each
part is one subtraction: the fold's steps; the image's ninth brick and its store; the register split; the
72-column steps; and, last, the panel's word and the drain's add, which is the whole den. One more row moves the
winner's tiles to the den's pitch, 155 lines, without the den. The den's image store is issued by every
lane with no branch around it: the lanes that hold no den value store to one place a warp -- channel 65's line,
which no kept column reads -- so their stores merge, on banks the holders' stores do not use.

THE VALUES TILE-MAJOR (its own rows, no clock inside a role). A plane's order in global memory follows its readers'
measured access ([`ARCHITECTURE.md`](../ARCHITECTURE.md#1-what-the-kernel-computes)), and the intra wants a tile of 64
tokens laid block by block ([`intra_window`](intra_window.md)). The carry's producers read a token's values as 8
lines of 16 bytes: token-major they are one run of 128 bytes, tile-major a kilobyte apart. Rows read the values
tile-major with the tables cached and once-read, each beside its token-major row.

ENTRIES FROM LISTS (its own rows; the fold's side). The facts pass turns a sparse level's mask into a window's tokens by
digit ([`facts_lists`](facts_lists.md)), an entry the token's place in its window and no byte. In these rows a
producer loads a batch's 32 places from such lists, a batch ahead of the loads they address; a window's tokens are
its own 1,024, so their rows are one stretch of each table where the other rows' tokens are spread over 65,536;
and an entry's byte on its list's digit and its 16 dense bytes are each a block of a level's tile-major row. The
producers hold one batch: the places and the sparse words are 4 registers a thread that two batches do not leave.
The readout's side still reads the synthetic readers. A second pair of rows gives every window the SAME 1,024 tokens, so
their rows stay in cache: what of the listed rows is their loads' code and what their rows' arrival. A third row ASKS FOR
A WINDOW'S ROWS A WINDOW AHEAD: for the batch a window on, a prefetch of each entry's values, dense bytes and sparse
word, so the loads that use them find them in cache.

EXPANDED VALUES (its own rows; the fold's side). An entry's values are 128 of the 148 bytes its CTA gathers, and
the 16 boxes of a sparse digit's column gather the same ones. In these rows a pass before the carry is taken to
have written each column's lists' values a window, contiguous, in the layout a step reads -- the host lays that
buffer here -- and a producer lands a list's value tile with one bulk copy, 8 a slot; it still makes the list's
coefficients from the lists and the levels' rows. A window's values are new each window, and a buffer is shared by 16
sources. A second pair has the pass write an entry's COEFFICIENT BYTES beside its values -- a list's ninth brick,
a line an entry's 16 dense bytes and the 17th the list's sparse bytes, raw, nothing computed -- so a list is one
bulk copy of 2,448 bytes and its producer loads nothing from global memory: it expands the landed bytes where
they lie.

NO CLOCK INSIDE A ROLE (its own rows). A role's waits and phases are clocked by reads of the cycle counter, and
the toolkit compiles those reads differently from row to row -- into ordinary registers in the winner's kernel,
through uniform registers in the den's. Rows of the winner's form and the den's, at both cells, and the den's chain run with no
clock read inside a role: their windows are the roles' own work, and their difference is the den's.

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

MEASURED (H100 80GB HBM3, 132 SMs, vast.ai instance 60617433, driver 595.91.07; 2026-11-14, the experiment at
eac7c8df, one session; 7 repetitions, every checked row's state and output values exact;
the median; SM-cycles a token are the window's cycles over 8 -- 128 CTAs of two boxes, 1,024 tokens. THE FOLD'S STATE
DUMP STAYS OUT OF THE TIMED CALLS, and one row restores it to price it in this session):

| the row | cycles a window a CTA | SM-cycles a token |
|---|---|---|
| the fold's side alone: producers and steps | 5,091 | 636 |
| the read side alone: the image, the readout and the drain | 5,559 | 695 |
| the carry: both sides | 6,948 | 868 |
| both sides, the producers' global loads taken out | 6,418 | 802 |
| both sides, the drains' global adds taken out | 6,626 | 828 |
| both sides, the image's stores taken out | 6,575 | 822 |
| the read side alone, no fold warp running | 4,262 | 533 |
| the read side alone, every wait suspended | 5,263 | 658 |
| both sides, every wait suspended | 6,823 | 853 |
| THE READOUT DRAINING ITS OWN ROWS, FOUR PRODUCERS: the fold's side alone | 3,046 | 381 |
| the same: the read side alone | 5,366 | 671 |
| the same: both sides | 7,395 | 924 |
| THE READOUT LEAN, TWO BATCHES A PRODUCER: the fold's side alone | 4,538 | 567 |
| the same: the read side alone | 5,770 | 721 |
| the same: both sides | 6,816 | 852 |
| FIVE WARPGROUPS, FOUR PRODUCERS AND FOUR DRAIN WARPS AT 40: the fold's side alone | 5,894 | 737 |
| the same: the read side alone | 5,043 | 630 |
| the same: both sides | 7,591 | 949 |
| THE LONG WAITS ASLEEP AT NAMED BARRIERS: the fold's side alone | 5,120 | 640 |
| the same: the read side alone | 5,059 | 632 |
| the same: both sides | 6,892 | 861 |
| the fold's side alone, the producers' global loads taken out | 2,400 | 300 |
| the fold's side alone, their coefficient arithmetic taken out | 4,682 | 585 |
| the fold's side alone, both taken out | 2,279 | 285 |
| the read side alone, the readout's byte loads taken out | 4,999 | 625 |
| the read side alone, no fold warp running, the byte loads taken out | 4,015 | 502 |
| both sides, the readout's byte loads taken out | 6,594 | 824 |
| both sides, the drains' adds and the readout's byte loads taken out | 6,395 | 799 |
| THE READOUT'S BYTES ONE WIDE LOAD A WARP: the read side alone | 5,059 | 632 |
| the same: the read side alone, no fold warp running | 4,177 | 522 |
| the same: both sides | 6,872 | 859 |
| the same: both sides, the long waits asleep | 7,007 | 876 |
| the read side alone, the image's stores taken out | 4,988 | 624 |
| the read side alone, the fold's fence behind its image taken out | 5,502 | 688 |
| the read side alone, both taken out | 4,974 | 622 |
| both sides, that fence taken out | 6,832 | 854 |
| the read side alone, no fold warp running, the fold warps parked asleep | 4,371 | 546 |
| the same, spinning | 4,512 | 564 |
| THE DUMP RESTORED: both sides, the fold's state stored out in every call | 8,026 | 1,003 |
| THE WINNERS COMBINED: both sides, suspended waits and the lean readout | 6,606 | 826 |
| the same, suspended waits and the wide byte load | 6,776 | 847 |
| the same, suspended waits, the lean readout and the wide byte load (the fourth warpgroup at 104) | 6,722 | 840 |
| the same, sleeping waits, the lean readout and the wide byte load | 6,964 | 870 |
| THE TABLES ONCE-READ, 16 replicas the cache cannot hold: both sides | 15,705 | 1,963 |
| the same, the winner's form | 14,299 | 1,787 |
| the same: the fold's side alone | 9,195 | 1,149 |
| THE DEN, folded, published, read and drained: the fold's side alone | 4,878 | 610 |
| the same: both sides, the winner's form | 7,041 | 880 |
| the same, the tables once-read | 14,550 | 1,819 |
| THE IMAGE'S TILES MOVED APART, nothing else changed: the winner's form at 154 lines (137 as it is) | 6,997 | 875 |
| the winner's form with the den at 161 lines (155 as it is) | 7,042 | 880 |
| THE ALIGNED CELL, an image tile a list: the winner's form | 6,631 | 829 |
| the same with the den | 7,083 | 885 |
| NO CLOCK INSIDE A ROLE: the winner's form | 6,534 | 817 |
| the same with the den | 6,868 | 859 |
| the same at the aligned cell | 6,612 | 826 |
| the same at the aligned cell with the den | 6,989 | 874 |
| THE DEN'S PARTS, a chain with no clock inside a role: the fold's steps | 6,592 | 824 |
| then the image's ninth brick and its store | 6,671 | 834 |
| then the register split | 6,767 | 846 |
| then the 72-column steps | 6,837 | 855 |
| the winner's form with its tiles at the den's pitch, 155 lines | 6,535 | 817 |
| THE VALUES TILE-MAJOR in global memory, no clock inside a role: the winner's form | 12,016 | 1,502 |
| the tables once-read, the values token-major | 14,650 | 1,831 |
| the tables once-read, the values tile-major | 48,402 | 6,050 |
| THE FOLD'S ENTRIES FROM LISTS, a window's own tokens: the fold's side alone | 8,750 | 1,094 |
| the same: both sides, no clock inside a role | 9,857 | 1,232 |
| the same with every window the same tokens: the fold's side alone | 6,261 | 783 |
| the same: both sides | 8,024 | 1,003 |
| a window's own tokens, their rows asked for a window ahead: both sides | 11,703 | 1,463 |
| THE VALUES LANDED BY BULK COPY from expanded lists, a window's own tokens: the fold's side alone | 8,410 | 1,051 |
| the same: both sides | 8,599 | 1,075 |
| THE WHOLE ENTRY LANDED, values and coefficient bytes: the fold's side alone | 5,133 | 642 |
| the same: both sides | 5,813 | 727 |
| the whole entry, EACH BOX ITS OWN expanded windows, past the cache: the fold's side alone | 5,715 | 714 |
| the same: both sides | 7,753 | 969 |
| the whole entry, its 16 columns' windows NEVER READ BEFORE, 16 boxes a window's readers: the fold's side alone | 5,162 | 645 |
| the same: both sides | 5,785 | 723 |

THE WALL TIME, measured on this box and immune to its clock (1.12 to 1.37 GHz, its rows' medians): the long call
scaled to 65,536 tokens is 0.329 ms for the baseline, 0.310 ms for the winner's form, and 0.597 ms at the
once-read bound (0.664 without the winner's changes). THE FLASH REFERENCES, from the session of instance 60450651 ON ANOTHER BOX (a different offer and driver)
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
| a fold warp waits on the readout's read of a tile group | 378 | 4,182 | 3,797 |  |  | 3,754 | 403 |
| a fold warp waits on its producer | 4,528 |  |  |  |  | 431 | 5,436 |
| a readout warp waits on a tile group's publish | 1,747 | 290 | 324 |  |  | 302 | 2,046 |
| a readout warp waits on the panel's drain | 329 | 286 | 275 | 253 | 291 |  | 329 |
| a readout warp draining its own rows |  |  |  |  |  | 2,811 |  |
| a readout warp forming its steps' A | 1,342 | 1,103 | 1,107 | 1,161 | 1,010 | 1,097 |  |
| a readout warp in its steps (a lean readout's hold the forming) | 2,197 | 2,580 | 2,126 | 2,183 | 2,042 | 1,944 | 3,843 |
| a drain warp waits on the panel | 3,065 | 2,055 | 1,527 | 744 | 706 |  |  |

WHAT IT SHOWS.
- THE CARRY COMPOSES, EXACTLY: two boxes' folds, the image a window, one readout warpgroup over both -- its
  coefficients expanded in its registers, its steps' A operand -- and one f16 line a token, in one CTA of 512
  threads at the SM's whole register file.
- A TOKEN IS PRICED AT 1,787 SM-CYCLES COMPOSED, its tables read once (R22.2), in the winner's form; 826 when the
  cache holds them (868 in the first form) at this cell (its sides sparse on opposite levels, 256 x 256 leaves;
  [`carry_floor`](../internals/tools/carry_floor.md) prices a dense side higher).
- THE FILL WANTS WARPS THAT EACH HOLD A BATCH: two producers a box, a batch of 32 entries in registers each, take
  the fold's side alone to 3,046 cycles a window where one a box is 5,091; one a box holding TWO batches is
  4,538; and two a box at 40 registers a thread, each holding one value instruction's four entries, are
  5,894 -- slower than one a box that holds a batch.
- SUSPENDED WAITS WITH THE LEAN READOUT ARE THE FASTEST COMPOSED FORM: 6,606 cycles a window against the
  baseline's 6,948, with the suspended waits beside the wide byte load a near tie at 6,776. The parts: every wait
  suspended alone 6,823, the long waits asleep at named barriers 6,892, the lean readout alone 1.9%
  under the baseline, the wide load alone 6,872. ALL THREE AT ONCE LOSE the pairing's gain (6,722: the lean
  readout beside the wide load's scratch needs 88 registers a thread and the fourth warpgroup drops to 104), and
  the sleeping triple is 6,964.
- THE DEN COSTS 5.1% OF THE COMPOSED WINDOW AT THIS CELL AND 5.7% AT THE ALIGNED ONE, the tables cached and no
  clock inside a role: 6,868 cycles against 6,534, and 6,989 against 6,612. THE ROLES' CLOCKS ARE NOT FREE AND NOT EQUAL: they are
  72 cycles of the winner's clocked window and 173 of the den's, and which is larger has changed between
  records, so a clocked row does not price the den (clocked: 7,041 against 6,606; why the clocks differ is
  not isolated); with the tables once-read 14,550 against 14,299; the fold's side alone 4,878 against 5,091. One
  cell (its sides sparse on opposite levels); a dense write side, where the fold's side binds, is unmeasured. ITS
  PARTS, each a row less the row before, no clock inside a role: the fold's four steps a window 58 cycles;
  the image's ninth brick and its store 79 (the pitch itself nothing: the winner's form at 155 lines is 6,535
  against 6,534); the register split, readout 88 and fourth warpgroup 104, 96; the steps 72 columns wide 70;
  the panel's word and the drain's add 31. No part is a third of it. The split cannot be avoided in this
  form: the 72-column readout serializes its steps at 80 registers with no clock read too.
- THE FOLD'S FORM WAS A CONTEST. A step a list with ones in one row of A is 16 steps a window for 1 row in 64 and
  a 17th accumulator set of 8, which the toolkit will not hold at 160 registers: at a 168-register fold with
  one-batch producers it cost an eighth of the composed window in its session (the experiment at ac342446, whose
  record this one replaced). The coefficients as A are four steps into 4 registers.
- AN IMAGE TILE'S PITCH MUST BE AN ODD COUNT OF LINES: a warp's lanes store to four tiles two apart, and at an
  even pitch two of the four land on the same banks. The winner's form at 154 lines is 6,997 against 6,606 at 137;
  the den's at 161 is 7,042 against 7,041 at 155.
- THE IMAGE FORM COSTS 1.2% MORE WHERE THE CARVES ARE EQUAL, no clock inside a role (6,612 cycles a window against
  6,534); clocked, the aligned cell's winner's form is 6,631 cycles a
  window against 6,606 at the cell whose sides list opposite levels -- 16 word stores a thread a tile group where
  that cell's are 4 lines -- and with the den 7,083 against 7,041, the den's lines there 2-byte
  stores of 8 threads. A form that reads the fold's own registers has nothing of the window to win back but the
  image's stores; it is not built.
- THE CARRY'S PRODUCERS WANT A TOKEN'S VALUES AS ONE RUN: with the values tile-major in global memory -- a token's 8
  lines a kilobyte apart -- the winner's form is 12,016 cycles a window against 6,534, and with the tables once-read
  48,402 against 14,650. The intra's window moves 5% with the values' order ([`intra_window`](intra_window.md)), so
  the values stay token-major and the levels, which the carry reads 16 bytes at a time, go tile-major.
- FROM LISTS, WITH A WINDOW'S OWN TOKENS, THE FOLD'S SIDE ALONE IS 8,750 CYCLES A WINDOW where the synthetic entries'
  is 5,091 cached and 9,195 once-read; both sides 9,857 (the readout's side still on synthetic, cached readers) against
  6,534. Two things are in it, and the rows with every window the same tokens part them: THE LISTS' CODE -- a
  batch's places loaded a batch ahead, a shuffle a value instruction, the sparse word, tile-major addresses, and one
  batch held where the winner holds two -- is 6,261 against 5,091 alone and 8,024 against 6,534 composed; and ROWS THE
  CACHE HAS NOT SEEN, a window's own being new each window as a call's are, are the rest, 2,489 alone and
  1,833 composed. A window's rows are loaded one batch ahead of their use. ASKING FOR THEM A WINDOW AHEAD BY
  PREFETCH LOSES: 11,703 composed against 9,857 -- six prefetches an entry-batch are more requests of the same
  memory, not less waiting. Whether loads issued two batches ahead hide the rows' arrival is not measured (the
  producers' registers hold one batch from lists). The check of these rows is weak: every table's byte is its token's parity.
- LANDING THE VALUES ALONE FROM EXPANDED LISTS is 8,599 cycles a window composed against 9,857 with every entry's
  values gathered (8,024 for the gather with rows the cache holds), and the fold's side alone 8,410 against 8,750: a
  list's values are one bulk copy of 2 KB from a buffer laid contiguous a window (the host's here; 16 sources
  share a buffer), new each window, and its producers still gather an entry's 16 dense bytes and its sparse byte,
  a line of each level's rows an entry. What that leaves is the next finding's.
- WITH THE WHOLE ENTRY LANDED -- its coefficient bytes beside its values, raw, a list one bulk copy and no global load
  in a producer's trip -- the window is 5,813 cycles composed and the fold's side alone 5,133, against 6,534 and
  5,091 for the synthetic entries whose rows the cache holds.
- WHAT THAT ROW NEEDS IS ITS BYTES SHARED, NOT HELD BEFOREHAND: its lists are 16 columns' expanded windows,
  40 MB that 16 boxes each read and every launch reads again. With EACH BOX ITS OWN -- 660 MB no other box read --
  the window is 7,753 cycles composed against 5,813 and the fold's side alone 5,715 against 5,133. And
  with the 16 columns kept but A WINDOW'S BYTES NEVER READ BEFORE -- a launch of 128 windows of their own
  addresses, 80 MB, each read by its 16 boxes at once, what a stream written up front for a segment is -- it is
  5,785 composed and 5,162 alone.
  So one box's miss serves its 15 mates, and a box alone with its bytes costs a third more: 16 times the bytes
  from global memory, on every multiprocessor at once. (That 80 MB leave a cache of 50 between a launch's two
  visits to a window is taken from the sizes, not measured.)
- THE TABLES ONCE-READ COST 1.8 TO 2.3 TIMES THE WINDOW ON THIS BOX (in cycles, which for a row bound by global
  memory follow the card's clock): 15,705 cycles composed against 6,948, the winner's form
  14,299 against 6,606, the fold's side alone 9,195 against 5,091 -- every access a fresh row of 128 MB of
  values, the model of a large table whose rows are each read once.
- THE DUMP IS PRICED IN THIS SESSION: the fold's state stored out in every call, as every record before 79ec1b08
  ran, puts the composed row at 8,026 against 6,948 -- one binary, one box, one session.
- FIVE WARPGROUPS LOSE COMPOSED, 7,591 against 6,948: 640 threads leave four producers and four drain warps 40
  registers a thread, the fold waits on its producers 5,436 cycles a window, and the lean readout the count
  forces is 3,843 in its steps and their forming where one commit group's are 3,539. Four drain warps do
  help the read side alone, 5,043 against 5,559.
- THE LEAN READOUT WITH TWO-BATCH PRODUCERS IS 6,816 COMPOSED against 6,948: a readout warp is 4,456 cycles
  a window in its steps and their forming. The two changes are not separated.
- THE READOUT DRAINING ITS OWN ROWS LOSES COMPOSED, 7,395 against 6,948: a readout warp is 2,811 cycles a
  window in its 64 adds, and the window becomes the readout's -- the fold waits on its read 3,754. A warp that
  issues an add is held until the SM passes it: adds issued by the readout hold the readout.
- COMPOSED, THE PRODUCERS' LOADS ARE NOW THE LARGEST SINGLE ACCESS: with them out the window is 6,418 against 6,948, and
  those loads are three fifths of the fold's side alone -- 2,400 cycles a window without them against 5,091, some
  168 cycles a batch of a producer's 9 load instructions; their coefficient arithmetic is 408 of it
  (4,682 without). The drains' adds out, the window is 6,626; the image's stores out, 6,575.
- A PARKED FOLD WARP COSTS THE READ SIDE 108 CYCLES, A PUBLISHING ONE 1,297: the read side alone is 5,559
  with the fold warps publishing and 4,262 with no fold warp running, while one parked asleep beside it costs
  108 cycles and one spinning 249. What of the publishing fold's hand-off holds it is NOT ISOLATED
  (its stores are 571 of it; its fence nothing).
- THE READOUT'S OWN BYTE LOADS ARE A SIXTH OF ITS STEPS' SPAN: 8 instructions a chunk, each lane's 4 bytes of a
  different row; without them a readout warp's steps are 2,126 cycles a window against 2,580. The read side alone
  is then the drain's -- 4,999 against 5,559, the readout waiting on the drain 275 -- at some
  27 cycles an add a drain warp, where with no fold warp running an add is 26.
- ONE WIDE LOAD A WARP FOR THE READOUT'S BYTES is the read side alone without those loads -- 5,059 cycles a window
  against 4,999 with them out and 5,559 as they were -- and little composed: 6,872 against 6,948, where the
  byte loads out are 6,594 and the drains' adds out with them 6,395.
- THE READOUT FORMS ITS STEPS' A IN 1,103 TO 1,342 CYCLES A WINDOW, 276 to 336 a chunk: the coefficients' rounds
  and the 32 words held.
- THE STEPS THEMSELVES ARE OVER THEIR OWN TIME with nothing beside them but the drains: 546 cycles a chunk (the
  fence, 8 steps, the commit, 8 loads issued, the wait) where 8 steps of 64 columns into one accumulator from
  shared memory are 319 ([`wgmma_width`](wgmma_width.md)). Not isolated here.
- EVERY ROLE ISSUES AN INSTRUCTION EVERY 3.5 TO 7 CYCLES of its busy time (the static ledger's counts a trip: a
  producer 168 a batch, a drain warp 339 a chunk, the readout 284 a chunk, a fold warp 379 a window): no role is
  bound by the instructions it issues, and what holds each is not isolated.

Record: `measure/lab/results/carry_window/sm_90a.json`.
