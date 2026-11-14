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

WHERE THOSE BYTES ARE (two pairs more). The whole-entry rows' lists are 16 columns' expanded windows, 40 MB that 16
boxes each read and every launch reads again. One pair gives EACH BOX ITS OWN, 660 MB no other box read. The other
keeps the 16 columns and runs 128 windows of their own addresses, 80 MB: a window's bytes are read by its 16
boxes at once and, by the sizes, are not in a cache of 50 MB when the first asks.

THE STREAM AS RULED (a pair; [`hopper-carry.md`](../architecture/hopper-carry.md#the-gather-stream)). The same 256
entries a box a window fall UNEVENLY on its 16 lists, 8 to 24 a list. A list's full steps are whole-entry tiles; its
last, part-filled step is stored TRUNCATED -- each value brick only the lines of the entries it has, then their
dense lines and a line of their sparse bytes -- so nothing of a pad crosses global memory. A window's steps are
taken a round at a time (every list's first, then every list's second), 8 a slot, a slot's laid in the order of
their lists; the buffer's head holds, a window, a record a step (its first line, its entries) and a header a slot
(each list's steps in it, which list's last is truncated, whether the slot is the window's last). A producer
loads a window's records and headers a window ahead; a slot is a lane a step, one bulk copy of the step's own
size, then the slot's header to its fold, then its entries expanded where they landed -- a pad's coefficient lines
zero, and every line a truncated step's operand reaches past its entries zeroed behind the read. The fold runs a
slot's steps list by list, a list's a loop of its own bottom-tested on a vote; a truncated step is used where it
lies, its operand's bricks as far apart as it has entries. A second pair gives a box FOUR SLOTS OF 4 TILES where
it has two of 8 -- the same pool -- and its producer issues a slot's copies TWO SLOTS AHEAD of expanding it, so a
slot's bytes land behind two slots' work where the two-slot row's producer waits out its own copy. One more row, the
fold's side alone, gives that form TWO PRODUCERS A BOX -- the four serve warps all producing, the call's even
slots one's and its odd ones the other's. The whole-entry rows' fold alone also record what a fold warp waits on
its producer.

THE FOLD EXPANDS (a pair, even lists). A producer only issues a slot's 8 copies, when the fold frees the slot; the
slot's fold warpgroup waits on their bytes and expands them itself -- its 128 threads the slot's 128 entries at
once, a thread an entry, where one producer warp takes four trips of 32 -- then its barrier and its 8 steps.

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

MEASURED (H100 80GB HBM3, 132 SMs, vast.ai instance 60629359, driver 580.173.02; 2026-11-14, the experiment at
e26c5dea, one session; 7 repetitions, every checked row's state and output values exact;
the median; SM-cycles a token are the window's cycles over 8 -- 128 CTAs of two boxes, 1,024 tokens. THE FOLD'S STATE
DUMP STAYS OUT OF THE TIMED CALLS, and one row restores it to price it in this session):

| the row | cycles a window a CTA | SM-cycles a token |
|---|---|---|
| the fold's side alone: producers and steps | 5,114 | 639 |
| the read side alone: the image, the readout and the drain | 5,561 | 695 |
| the carry: both sides | 7,047 | 881 |
| both sides, the producers' global loads taken out | 6,429 | 804 |
| both sides, the drains' global adds taken out | 6,698 | 837 |
| both sides, the image's stores taken out | 6,687 | 836 |
| the read side alone, no fold warp running | 4,374 | 547 |
| the read side alone, every wait suspended | 5,272 | 659 |
| both sides, every wait suspended | 6,922 | 865 |
| THE READOUT DRAINING ITS OWN ROWS, FOUR PRODUCERS: the fold's side alone | 3,088 | 386 |
| the same: the read side alone | 5,366 | 671 |
| the same: both sides | 7,434 | 929 |
| THE READOUT LEAN, TWO BATCHES A PRODUCER: the fold's side alone | 4,574 | 572 |
| the same: the read side alone | 5,771 | 721 |
| the same: both sides | 6,790 | 849 |
| FIVE WARPGROUPS, FOUR PRODUCERS AND FOUR DRAIN WARPS AT 40: the fold's side alone | 5,921 | 740 |
| the same: the read side alone | 5,046 | 631 |
| the same: both sides | 7,772 | 971 |
| THE LONG WAITS ASLEEP AT NAMED BARRIERS: the fold's side alone | 5,155 | 644 |
| the same: the read side alone | 5,061 | 633 |
| the same: both sides | 6,992 | 874 |
| the fold's side alone, the producers' global loads taken out | 2,400 | 300 |
| the fold's side alone, their coefficient arithmetic taken out | 4,727 | 591 |
| the fold's side alone, both taken out | 2,279 | 285 |
| the read side alone, the readout's byte loads taken out | 4,999 | 625 |
| the read side alone, no fold warp running, the byte loads taken out | 4,016 | 502 |
| both sides, the readout's byte loads taken out | 6,678 | 835 |
| both sides, the drains' adds and the readout's byte loads taken out | 6,422 | 803 |
| THE READOUT'S BYTES ONE WIDE LOAD A WARP: the read side alone | 5,064 | 633 |
| the same: the read side alone, no fold warp running | 4,235 | 529 |
| the same: both sides | 6,973 | 872 |
| the same: both sides, the long waits asleep | 7,135 | 892 |
| the read side alone, the image's stores taken out | 4,988 | 624 |
| the read side alone, the fold's fence behind its image taken out | 5,502 | 688 |
| the read side alone, both taken out | 4,973 | 622 |
| both sides, that fence taken out | 6,936 | 867 |
| the read side alone, no fold warp running, the fold warps parked asleep | 4,385 | 548 |
| the same, spinning | 4,505 | 563 |
| THE DUMP RESTORED: both sides, the fold's state stored out in every call | 8,151 | 1,019 |
| THE WINNERS COMBINED: both sides, suspended waits and the lean readout | 6,662 | 833 |
| the same, suspended waits and the wide byte load | 6,926 | 866 |
| the same, suspended waits, the lean readout and the wide byte load (the fourth warpgroup at 104) | 6,749 | 844 |
| the same, sleeping waits, the lean readout and the wide byte load | 7,001 | 875 |
| THE TABLES ONCE-READ, 16 replicas the cache cannot hold: both sides | 17,725 | 2,216 |
| the same, the winner's form | 16,118 | 2,015 |
| the same: the fold's side alone | 11,817 | 1,477 |
| THE DEN, folded, published, read and drained: the fold's side alone | 5,095 | 637 |
| the same: both sides, the winner's form | 7,027 | 878 |
| the same, the tables once-read | 16,500 | 2,063 |
| THE IMAGE'S TILES MOVED APART, nothing else changed: the winner's form at 154 lines (137 as it is) | 7,045 | 881 |
| the winner's form with the den at 161 lines (155 as it is) | 7,032 | 879 |
| THE ALIGNED CELL, an image tile a list: the winner's form | 6,665 | 833 |
| the same with the den | 7,054 | 882 |
| NO CLOCK INSIDE A ROLE: the winner's form | 6,586 | 823 |
| the same with the den | 6,908 | 864 |
| the same at the aligned cell | 6,667 | 833 |
| the same at the aligned cell with the den | 7,058 | 882 |
| THE DEN'S PARTS, a chain with no clock inside a role: the fold's steps | 6,643 | 830 |
| then the image's ninth brick and its store | 6,724 | 840 |
| then the register split | 6,813 | 852 |
| then the 72-column steps | 6,873 | 859 |
| the winner's form with its tiles at the den's pitch, 155 lines | 6,585 | 823 |
| THE VALUES TILE-MAJOR in global memory, no clock inside a role: the winner's form | 12,170 | 1,521 |
| the tables once-read, the values token-major | 16,264 | 2,033 |
| the tables once-read, the values tile-major | 48,358 | 6,045 |
| THE FOLD'S ENTRIES FROM LISTS, a window's own tokens: the fold's side alone | 8,731 | 1,091 |
| the same: both sides, no clock inside a role | 9,984 | 1,248 |
| the same with every window the same tokens: the fold's side alone | 6,208 | 776 |
| the same: both sides | 8,091 | 1,011 |
| a window's own tokens, their rows asked for a window ahead: both sides | 11,897 | 1,487 |
| THE VALUES LANDED BY BULK COPY from expanded lists, a window's own tokens: the fold's side alone | 8,462 | 1,058 |
| the same: both sides | 9,030 | 1,129 |
| THE WHOLE ENTRY LANDED, values and coefficient bytes: the fold's side alone | 5,168 | 646 |
| the same: both sides | 5,987 | 748 |
| the whole entry, EACH BOX ITS OWN expanded windows, past the cache: the fold's side alone | 6,471 | 809 |
| the same: both sides | 11,193 | 1,399 |
| the whole entry, its 16 columns' windows NEVER READ BEFORE, 16 boxes a window's readers: the fold's side alone | 5,174 | 647 |
| the same: both sides | 5,983 | 748 |
| THE STREAM AS RULED, packed by digit, its lists uneven (8 to 24 entries), last steps truncated: the fold's side alone | 10,124 | 1,265 |
| the same: both sides | 10,580 | 1,322 |
| the stream as ruled, FOUR SLOTS OF 4 TILES a box: the fold's side alone | 10,716 | 1,339 |
| the same: both sides | 11,497 | 1,437 |
| the whole entry, THE FOLD WARPS EXPANDING their own slot: the fold's side alone | 3,026 | 378 |
| the same: both sides | 5,977 | 747 |

THE WALL TIME, measured on this box and immune to its clock (1.77 to 1.81 GHz, its rows' medians): the long call
scaled to 65,536 tokens is 0.255 ms for the baseline, 0.244 ms for the winner's form, and 0.519 ms at the
once-read bound (0.588 without the winner's changes). THE FLASH REFERENCES, from the session of instance 60450651 ON ANOTHER BOX (a different offer and driver)
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
| a fold warp waits on the readout's read of a tile group | 367 | 4,182 | 3,798 |  |  | 3,768 | 413 |
| a fold warp waits on its producer | 4,692 |  |  |  |  | 434 | 5,616 |
| a readout warp waits on a tile group's publish | 1,859 | 290 | 324 |  |  | 301 | 2,221 |
| a readout warp waits on the panel's drain | 327 | 286 | 275 | 246 | 291 |  | 328 |
| a readout warp draining its own rows |  |  |  |  |  | 2,839 |  |
| a readout warp forming its steps' A | 1,341 | 1,104 | 1,107 | 1,264 | 1,010 | 1,103 |  |
| a readout warp in its steps (a lean readout's hold the forming) | 2,198 | 2,581 | 2,125 | 2,203 | 2,042 | 1,949 | 3,864 |
| a drain warp waits on the panel | 3,232 | 2,057 | 1,528 | 840 | 706 |  |  |

WHAT IT SHOWS.
- THE CARRY COMPOSES, EXACTLY: two boxes' folds, the image a window, one readout warpgroup over both -- its
  coefficients expanded in its registers, its steps' A operand -- and one f16 line a token, in one CTA of 512
  threads at the SM's whole register file.
- A TOKEN IS PRICED AT 2,015 SM-CYCLES COMPOSED, its tables read once (R22.2), in the winner's form; 833 when the
  cache holds them (881 in the first form) at this cell (its sides sparse on opposite levels, 256 x 256 leaves;
  [`carry_floor`](../internals/tools/carry_floor.md) prices a dense side higher).
- THE FILL WANTS WARPS THAT EACH HOLD A BATCH: two producers a box, a batch of 32 entries in registers each, take
  the fold's side alone to 3,088 cycles a window where one a box is 5,114; one a box holding TWO batches is
  4,574; and two a box at 40 registers a thread, each holding one value instruction's four entries, are
  5,921 -- slower than one a box that holds a batch.
- SUSPENDED WAITS WITH THE LEAN READOUT ARE THE FASTEST COMPOSED FORM: 6,662 cycles a window against the
  baseline's 7,047, with the suspended waits beside the wide byte load a near tie at 6,926. The parts: every wait
  suspended alone 6,922, the long waits asleep at named barriers 6,992, the lean readout alone 3.7%
  under the baseline, the wide load alone 6,973. ALL THREE AT ONCE LOSE the pairing's gain (6,749: the lean
  readout beside the wide load's scratch needs 88 registers a thread and the fourth warpgroup drops to 104), and
  the sleeping triple is 7,001.
- THE DEN COSTS 4.9% OF THE COMPOSED WINDOW AT THIS CELL AND 5.9% AT THE ALIGNED ONE, the tables cached and no
  clock inside a role: 6,908 cycles against 6,586, and 7,058 against 6,667. THE ROLES' CLOCKS ARE NOT FREE AND NOT EQUAL: they are
  76 cycles of the winner's clocked window and 119 of the den's, and which is larger has changed between
  records, so a clocked row does not price the den (clocked: 7,027 against 6,662; why the clocks differ is
  not isolated); with the tables once-read 16,500 against 16,118; the fold's side alone 5,095 against 5,114. One
  cell (its sides sparse on opposite levels); a dense write side, where the fold's side binds, is unmeasured. ITS
  PARTS, each a row less the row before, no clock inside a role: the fold's four steps a window 57 cycles;
  the image's ninth brick and its store 80 (the pitch itself nothing: the winner's form at 155 lines is 6,585
  against 6,586); the register split, readout 88 and fourth warpgroup 104, 89; the steps 72 columns wide 61;
  the panel's word and the drain's add 35. No part is a third of it. The split cannot be avoided in this
  form: the 72-column readout serializes its steps at 80 registers with no clock read too.
- THE FOLD'S FORM WAS A CONTEST. A step a list with ones in one row of A is 16 steps a window for 1 row in 64 and
  a 17th accumulator set of 8, which the toolkit will not hold at 160 registers: at a 168-register fold with
  one-batch producers it cost an eighth of the composed window in its session (the experiment at ac342446, whose
  record this one replaced). The coefficients as A are four steps into 4 registers.
- AN IMAGE TILE'S PITCH MUST BE AN ODD COUNT OF LINES: a warp's lanes store to four tiles two apart, and at an
  even pitch two of the four land on the same banks. The winner's form at 154 lines is 7,045 against 6,662 at 137;
  the den's at 161 is 7,032 against 7,027 at 155.
- THE IMAGE FORM COSTS 1.2% MORE WHERE THE CARVES ARE EQUAL, no clock inside a role (6,667 cycles a window against
  6,586); clocked, the aligned cell's winner's form is 6,665 cycles a
  window against 6,662 at the cell whose sides list opposite levels -- 16 word stores a thread a tile group where
  that cell's are 4 lines -- and with the den 7,054 against 7,027, the den's lines there 2-byte
  stores of 8 threads. A form that reads the fold's own registers has nothing of the window to win back but the
  image's stores; it is not built.
- THE CARRY'S PRODUCERS WANT A TOKEN'S VALUES AS ONE RUN: with the values tile-major in global memory -- a token's 8
  lines a kilobyte apart -- the winner's form is 12,170 cycles a window against 6,586, and with the tables once-read
  48,358 against 16,264. The intra's window moves 5% with the values' order ([`intra_window`](intra_window.md)), so
  the values stay token-major and the levels, which the carry reads 16 bytes at a time, go tile-major.
- FROM LISTS, WITH A WINDOW'S OWN TOKENS, THE FOLD'S SIDE ALONE IS 8,731 CYCLES A WINDOW where the synthetic entries'
  is 5,114 cached and 11,817 once-read; both sides 9,984 (the readout's side still on synthetic, cached readers) against
  6,586. Two things are in it, and the rows with every window the same tokens part them: THE LISTS' CODE -- a
  batch's places loaded a batch ahead, a shuffle a value instruction, the sparse word, tile-major addresses, and one
  batch held where the winner holds two -- is 6,208 against 5,114 alone and 8,091 against 6,586 composed; and ROWS THE
  CACHE HAS NOT SEEN, a window's own being new each window as a call's are, are the rest, 2,523 alone and
  1,894 composed. A window's rows are loaded one batch ahead of their use. ASKING FOR THEM A WINDOW AHEAD BY
  PREFETCH LOSES: 11,897 composed against 9,984 -- six prefetches an entry-batch are more requests of the same
  memory, not less waiting. Whether loads issued two batches ahead hide the rows' arrival is not measured (the
  producers' registers hold one batch from lists). The check of these rows is weak: every table's byte is its token's parity.
- LANDING THE VALUES ALONE FROM EXPANDED LISTS is 9,030 cycles a window composed against 9,984 with every entry's
  values gathered (8,091 for the gather with rows the cache holds), and the fold's side alone 8,462 against 8,731: a
  list's values are one bulk copy of 2 KB from a buffer laid contiguous a window (the host's here; 16 sources
  share a buffer), new each window, and its producers still gather an entry's 16 dense bytes and its sparse byte,
  a line of each level's rows an entry. What that leaves is the next finding's.
- WITH THE WHOLE ENTRY LANDED -- its coefficient bytes beside its values, raw, a list one bulk copy and no global load
  in a producer's trip -- the window is 5,987 cycles composed and the fold's side alone 5,168, against 6,586 and
  5,114 for the synthetic entries whose rows the cache holds.
- WHAT THAT ROW NEEDS IS ITS BYTES SHARED, NOT HELD BEFOREHAND: with a window's bytes never read before and its
  16 boxes its readers, the window is 5,983 cycles composed against 5,987 with the cache holding them, and
  the fold's side alone 5,174 against 5,168: one box's miss serves its 15 mates. With EACH BOX ITS OWN it is
  11,193 composed, 1.87 times, and 6,471 alone, 1.25 times -- 16 times the bytes from global memory, on
  every multiprocessor at once.
- THE STREAM AS RULED IS EXACT, AND ITS UNEVEN LISTS COST THE FOLD 1.96 TIMES: 10,124 cycles a window the
  fold's side alone against 5,168, 10,580 composed against 5,987. The same 256 entries a box a window, as
  lists of 8 to 24 where the row above has 16 each, are 24 steps where it has 16 -- three slots of 8 tiles where it
  has two -- and a truncated step, used where it landed with its bricks as far apart as it has entries, folds
  exactly. In four slots of 4 tiles, a slot's copies issued two slots ahead, the same stream is 10,716 alone
  and 11,497 composed: more slots cost more, so a slot's price is not its bytes' landing.
- THE FOLD'S SIDE WAITS ON ITS PRODUCER. With even lists a fold warp waits 4,538 of the fold's side's 5,168 cycles a
  window on its producer: its own steps are what is left. One producer warp a box expands every entry. On the
  packed stream in four slots the wait is 4,172 of 10,716, and with TWO PRODUCERS A BOX 1,497 of 8,452: the wait
  falls and the fold's own time is then the larger part. That fold runs a list's steps as a loop of its own, a
  vote a list a slot, where the even rows' steps are one block; no row has the uneven lists' steps as one block,
  so the form's share of that time is not isolated.
- WITH THE FOLD WARPS EXPANDING THEIR OWN SLOT (even lists; a producer only issues copies) the fold's side alone
  is 3,026 cycles a window against 5,168, a fold warp's wait -- now on its slot's bytes -- 1,896 against 4,538.
  THE COMPOSED WINDOW DOES NOT MOVE: 5,977 against 5,987. At even lists the window is not the fold's side's; no row of
  this pair says what of it is the read side's and what the two sides' waits on each other. Where the fold's
  side is the larger -- the uneven stream's 10,124 -- it is. NOT ASKED: this form on the uneven stream; a truncated
  step's pads left out of the expanding.
- THE TABLES ONCE-READ COST 2.3 TO 2.5 TIMES THE WINDOW ON THIS BOX (in cycles, which for a row bound by global
  memory follow the card's clock): 17,725 cycles composed against 7,047, the winner's form
  16,118 against 6,662, the fold's side alone 11,817 against 5,114 -- every access a fresh row of 128 MB of
  values, the model of a large table whose rows are each read once.
- THE DUMP IS PRICED IN THIS SESSION: the fold's state stored out in every call, as every record before 79ec1b08
  ran, puts the composed row at 8,151 against 7,047 -- one binary, one box, one session.
- FIVE WARPGROUPS LOSE COMPOSED, 7,772 against 7,047: 640 threads leave four producers and four drain warps 40
  registers a thread, the fold waits on its producers 5,616 cycles a window, and the lean readout the count
  forces is 3,864 in its steps and their forming where one commit group's are 3,539. Four drain warps do
  help the read side alone, 5,046 against 5,561.
- THE LEAN READOUT WITH TWO-BATCH PRODUCERS IS 6,790 COMPOSED against 7,047: a readout warp is 4,476 cycles
  a window in its steps and their forming. The two changes are not separated.
- THE READOUT DRAINING ITS OWN ROWS LOSES COMPOSED, 7,434 against 7,047: a readout warp is 2,839 cycles a
  window in its 64 adds, and the window becomes the readout's -- the fold waits on its read 3,768. A warp that
  issues an add is held until the SM passes it: adds issued by the readout hold the readout.
- COMPOSED, THE PRODUCERS' LOADS ARE NOW THE LARGEST SINGLE ACCESS: with them out the window is 6,429 against 7,047, and
  those loads are three fifths of the fold's side alone -- 2,400 cycles a window without them against 5,114, some
  170 cycles a batch of a producer's 9 load instructions; their coefficient arithmetic is 387 of it
  (4,727 without). The drains' adds out, the window is 6,698; the image's stores out, 6,687.
- A PARKED FOLD WARP COSTS THE READ SIDE 11 CYCLES, A PUBLISHING ONE 1,187: the read side alone is 5,561
  with the fold warps publishing and 4,374 with no fold warp running, while one parked asleep beside it costs
  11 cycles and one spinning 131. What of the publishing fold's hand-off holds it is NOT ISOLATED
  (its stores are 572 of it; its fence nothing).
- THE READOUT'S OWN BYTE LOADS ARE A SIXTH OF ITS STEPS' SPAN: 8 instructions a chunk, each lane's 4 bytes of a
  different row; without them a readout warp's steps are 2,125 cycles a window against 2,581. The read side alone
  is then the drain's -- 4,999 against 5,561, the readout waiting on the drain 275 -- at some
  27 cycles an add a drain warp, where with no fold warp running an add is 26.
- ONE WIDE LOAD A WARP FOR THE READOUT'S BYTES is the read side alone without those loads -- 5,064 cycles a window
  against 4,999 with them out and 5,561 as they were -- and little composed: 6,973 against 7,047, where the
  byte loads out are 6,678 and the drains' adds out with them 6,422.
- THE READOUT FORMS ITS STEPS' A IN 1,104 TO 1,341 CYCLES A WINDOW, 276 to 335 a chunk: the coefficients' rounds
  and the 32 words held.
- THE STEPS THEMSELVES ARE OVER THEIR OWN TIME with nothing beside them but the drains: 551 cycles a chunk (the
  fence, 8 steps, the commit, 8 loads issued, the wait) where 8 steps of 64 columns into one accumulator from
  shared memory are 319 ([`wgmma_width`](wgmma_width.md)). Not isolated here.
- EVERY ROLE ISSUES AN INSTRUCTION EVERY 3.5 TO 7 CYCLES of its busy time (the static ledger's counts a trip: a
  producer 168 a batch, a drain warp 339 a chunk, the readout 284 a chunk, a fold warp 379 a window): no role is
  bound by the instructions it issues, and what holds each is not isolated.

Record: `measure/lab/results/carry_window/sm_90a.json`.
