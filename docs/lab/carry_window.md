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

ENTRIES FROM LISTS (its own rows; the fold's side). A pass turns a sparse level's mask into a window's tokens by
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
Further rows have THE FOLD ISSUE ITS OWN COPIES -- the window after's, as it leaves the slot: no producer warp
runs and no slot is handed back; a fold thread is then at 168 registers and a serve thread 16 fewer (96 beside
the lean readout), the copies' addresses held beside the state; alone and clocked the form does not fit a fold
thread's registers, so it has no fold-alone row -- composed,
and composed with the serve warpgroup FOUR DRAIN WARPS,
16 of a chunk's 64 tokens each. The composed rows of these forms run once more with every role's waits clocked:
the role that does not wait is the window's.

THE READOUT A SUM A BOX (a pair, composed, with and without clocks; the fold expanding, a producer issuing). A
chunk's 8 steps all add into the readout's one accumulator, a chain. Here the steps on box 0's tiles add into one
accumulator and those on box 1's into another -- two chains of 4 -- and a thread sums the two before its row is
rounded: one panel, the drain as it was. A readout thread is at 112 registers (at 104 the toolkit serializes its
steps) and a serve thread at 80; its A is one set, a digit's pair of steps done before the next pair's is
written; the next chunk's bytes are loaded behind the steps, not under them; and its folds and serve warps mark
their clocks in global memory (the readout holds its own), a held clock's start being the value such a kernel
spills.

THE MIRRORED READOUT (a pair, composed, with and without clocks; the fold expanding, a producer issuing). The
readout's step has its tokens on its 64 rows, and one read digit's 16 readers are a step: 48 of a step's rows are
zeros. Here the readout is laid as the fold is: its coefficients in SHARED MEMORY and its tokens on the step's
COLUMNS. A thread expands one (token, box)'s 16 bytes into two lines of that digit's tile (16 tokens by 16 fold
digits, K-major); the warpgroup's barrier; then a step a (digit, box) of 64 channels by 16 tokens -- the image its
A as it lies, the tile its B, every row and column live -- a digit's two steps into its own accumulator, four
chains of 2. A thread's 32 sums are a channel's values on two tokens each: it stores each as a half of its token's
panel line, 32 stores of 2 bytes where the other form stores 16 words. These rows' folds are at 168 registers a
thread and their serve warps at 96 (at 160 the fold's own expansion spills a product), and every role marks its
clock in global memory.
A second pair gives the mirrored readout A CHUNK OF 8 DIGITS: a pass is two tile groups -- both awaited, both
expanded, their 16 steps one commit group, both panels written -- so a window is two passes of the readout's
fixed work, not four; 64 sums a thread, the readout at 104 registers, the folds at 160 and the serve warps at 88.
Two pairs more put the output path on four warps: the chunk-of-8 form with THE FOLD ISSUING ITS OWN COPIES (no
producer; its folds at 160 registers a thread, its readout at 104, its serve warps at 88 -- every role's clock
marked in global memory, the self-issuing fold fits at 160) and the serve warpgroup FOUR DRAIN WARPS; and that form with the readout's sums stored AS THEY ARE,
fp32, and ROUNDED BY THE DRAIN WARPS as they load them -- a row's 64 values are 256 of its 272 bytes, so the panel
is no larger.

THE BLOCK ROWS (the fold alone; composed, with and without clocks, beside the mirrored readout at a chunk of 8
digits and four drain warps). THE QUESTION: what uneven lists cost the carry when the stream says what to do and
the fold walks nothing. The fold is fed A BLOCK A LIST A WINDOW, its lists uneven, 8 to 24 entries, a slot ONE
LANDED IMAGE the host laid: a word an entry -- the line of its dense bytes and the line of its second coefficient
run; the entries' sparse bytes; the second runs' pad lines; a step's four halves -- its A's first line and brick
spacing, its B's line and its two bricks' spacing, or all ones for a step that is absent; the pads' count; then
each list's 8 runs of value lines, a brick a run, a line an entry, as long as the list; then each list's dense
lines, padded with zero lines to its steps. A window's PLAN is 16 bytes -- each slot's image's first line in the
stream and its bytes -- loaded a window ahead by one thread a box. A fold lands a slot with one bulk copy, issued
a window ahead as it leaves the slot; a thread reads its entry's word, expands the entry's dense line by its
sparse byte where it lies and to its second run -- behind the landed image, in shared memory only -- zeroes a pad
line and, a thread of 16, makes a step's two operands from the image's halves; and a slot's steps are ONE BLOCK
of 16, two rounds of its 8 lists, an absent step multiplying a box's tile of zeros. A fold thread holds no count
and walks none. The lab's bounds: a slot's 8 lists are 128 entries, no list is past two rounds, and a slot is
1,488 lines. These rows' folds are at 168 registers a thread, their readout at 104 and their drain warps at 72.
The even-list row beside them is the fold issuing whole-entry tiles under the same readout and drains, its folds
at 160 and its drain warps at 88.
The mirrored readout's rows clock a third phase: its sums' rounding, their stores to the panel and its arrivals,
recorded with its wait on the drain in it.
The clocked rows of the block form and of the even lists' self-issuing form clock THE FOLD'S PHASES too: the
image's publishing (its wait on the readout's read in it); its entries' expansion, from a slot's bytes landed;
the block form's pads and steps' operands; the slot's fence; the warpgroup's barrier; and its steps with the
next copy's issue, the two slots' fences and rendezvous apart. A clocked fold holds nine more registers, and its phase laps' readings are each thread's own
([`LEARNINGS.md`](../internals/LEARNINGS.md#lap-in-uniform)).
A CONTROL, both forms, with and without clocks: the fold stores its landed bytes AS THEY ARE to its coefficient
lines -- no conversion, no product, no sparse byte. Its sums are not the carry's: an isolation, unchecked.
A SECOND, the even lists' form: BOTH its next window's copies ISSUED AT THE WINDOW'S END, behind its second
slot's steps, where the form issues a slot's as it leaves the slot -- so the second slot's fence has no copy of
the fold's own issued since the window began, where the form's has the first slot's issued a slot before; at the
first slot's fence both copies have had the image's publishing to land, and the second's is not waited on
there. THE QUESTION: is a fence's time a wait on the copies in flight? Checked, exact.
THE MEETING ROWS, both forms, with and without clocks: a warpgroup's rendezvous behind its coefficient lines --
a fold's a slot, the readout's a pass -- is AN MBARRIER'S: a warp arrives and every thread waits as the row waits,
where the other rows sync a named barrier. THE QUESTION: what the named barrier costs a warpgroup. Checked,
exact. Their clocked even-list row's folds are at 168 registers and its drain warps at 72.
A PAIR WITH EVERY WAIT POLLED (the even lists' form): a wait is a loop of tests, where the form's suspends its
warp. THE QUESTION: whether a fold warp's time at its rendezvous is its warpgroup's warps returning from a
suspended wait at different times. Checked, exact.
A PAIR WITH THE COPIES ISSUED TWO A WARP (the even lists' form): a slot's 8 copies are its four warps' first two
lanes', where the form's first warp issues all 8. THE QUESTION: whether the rendezvous is the warpgroup waiting
for the warp that issues. Checked, exact. A clocked fold's sites are recorded for its warpgroups' first warps
alone beside every warp's mean.

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

MEASURED (H100 80GB HBM3, 132 SMs, vast.ai instance 60715693, driver 595.91.07; 2026-11-15, the experiment at
ecdd6b1c, one session; 7 repetitions, every checked row's state and output values exact;
the median; SM-cycles a token are the window's cycles over 8 -- 128 CTAs of two boxes, 1,024 tokens. THE FOLD'S STATE
DUMP STAYS OUT OF THE TIMED CALLS, and one row restores it to price it in this session):

| the row | cycles a window a CTA | SM-cycles a token |
|---|---|---|
| the fold's side alone: producers and steps | 5,077 | 635 |
| the read side alone: the image, the readout and the drain | 5,565 | 696 |
| the carry: both sides | 6,948 | 868 |
| both sides, the producers' global loads taken out | 6,412 | 801 |
| both sides, the drains' global adds taken out | 6,588 | 824 |
| both sides, the image's stores taken out | 6,511 | 814 |
| the read side alone, no fold warp running | 4,240 | 530 |
| the read side alone, every wait suspended | 5,259 | 657 |
| both sides, every wait suspended | 6,811 | 851 |
| THE READOUT DRAINING ITS OWN ROWS, FOUR PRODUCERS: the fold's side alone | 3,042 | 380 |
| the same: the read side alone | 5,362 | 670 |
| the same: both sides | 7,397 | 925 |
| THE READOUT LEAN, TWO BATCHES A PRODUCER: the fold's side alone | 4,528 | 566 |
| the same: the read side alone | 5,769 | 721 |
| the same: both sides | 6,813 | 852 |
| FIVE WARPGROUPS, FOUR PRODUCERS AND FOUR DRAIN WARPS AT 40: the fold's side alone | 5,871 | 734 |
| the same: the read side alone | 5,050 | 631 |
| the same: both sides | 7,589 | 949 |
| THE LONG WAITS ASLEEP AT NAMED BARRIERS: the fold's side alone | 5,105 | 638 |
| the same: the read side alone | 5,064 | 633 |
| the same: both sides | 6,886 | 861 |
| the fold's side alone, the producers' global loads taken out | 2,414 | 302 |
| the fold's side alone, their coefficient arithmetic taken out | 4,664 | 583 |
| the fold's side alone, both taken out | 2,282 | 285 |
| the read side alone, the readout's byte loads taken out | 5,001 | 625 |
| the read side alone, no fold warp running, the byte loads taken out | 4,020 | 503 |
| both sides, the readout's byte loads taken out | 6,591 | 824 |
| both sides, the drains' adds and the readout's byte loads taken out | 6,392 | 799 |
| THE READOUT'S BYTES ONE WIDE LOAD A WARP: the read side alone | 5,053 | 632 |
| the same: the read side alone, no fold warp running | 4,163 | 520 |
| the same: both sides | 6,872 | 859 |
| the same: both sides, the long waits asleep | 6,994 | 874 |
| the read side alone, the image's stores taken out | 4,994 | 624 |
| the read side alone, the fold's fence behind its image taken out | 5,506 | 688 |
| the read side alone, both taken out | 4,983 | 623 |
| both sides, that fence taken out | 6,831 | 854 |
| the read side alone, no fold warp running, the fold warps parked asleep | 4,341 | 543 |
| the same, spinning | 4,496 | 562 |
| THE DUMP RESTORED: both sides, the fold's state stored out in every call | 8,116 | 1,015 |
| THE WINNERS COMBINED: both sides, suspended waits and the lean readout | 6,616 | 827 |
| the same, suspended waits and the wide byte load | 6,771 | 846 |
| the same, suspended waits, the lean readout and the wide byte load (the fourth warpgroup at 104) | 6,727 | 841 |
| the same, sleeping waits, the lean readout and the wide byte load | 6,967 | 871 |
| THE TABLES ONCE-READ, 16 replicas the cache cannot hold: both sides | 15,707 | 1,963 |
| the same, the winner's form | 14,252 | 1,782 |
| the same: the fold's side alone | 8,629 | 1,079 |
| THE DEN, folded, published, read and drained: the fold's side alone | 4,805 | 601 |
| the same: both sides, the winner's form | 7,068 | 883 |
| the same, the tables once-read | 14,579 | 1,822 |
| THE IMAGE'S TILES MOVED APART, nothing else changed: the winner's form at 154 lines (137 as it is) | 7,034 | 879 |
| the winner's form with the den at 161 lines (155 as it is) | 7,070 | 884 |
| THE ALIGNED CELL, an image tile a list: the winner's form | 6,642 | 830 |
| the same with the den | 7,095 | 887 |
| NO CLOCK INSIDE A ROLE: the winner's form | 6,540 | 818 |
| the same with the den | 6,889 | 861 |
| the same at the aligned cell | 6,617 | 827 |
| the same at the aligned cell with the den | 7,000 | 875 |
| THE DEN'S PARTS, a chain with no clock inside a role: the fold's steps | 6,598 | 825 |
| then the image's ninth brick and its store | 6,691 | 836 |
| then the register split | 6,785 | 848 |
| then the 72-column steps | 6,849 | 856 |
| the winner's form with its tiles at the den's pitch, 155 lines | 6,552 | 819 |
| THE VALUES TILE-MAJOR in global memory, no clock inside a role: the winner's form | 11,997 | 1,500 |
| the tables once-read, the values token-major | 14,638 | 1,830 |
| the tables once-read, the values tile-major | 48,593 | 6,074 |
| THE FOLD'S ENTRIES FROM LISTS, a window's own tokens: the fold's side alone | 8,746 | 1,093 |
| the same: both sides, no clock inside a role | 9,845 | 1,231 |
| the same with every window the same tokens: the fold's side alone | 6,246 | 781 |
| the same: both sides | 8,024 | 1,003 |
| a window's own tokens, their rows asked for a window ahead: both sides | 11,712 | 1,464 |
| THE VALUES LANDED BY BULK COPY from expanded lists, a window's own tokens: the fold's side alone | 8,437 | 1,055 |
| the same: both sides | 8,578 | 1,072 |
| THE WHOLE ENTRY LANDED, values and coefficient bytes: the fold's side alone | 5,147 | 643 |
| the same: both sides | 5,808 | 726 |
| the whole entry, EACH BOX ITS OWN expanded windows, past the cache: the fold's side alone | 5,745 | 718 |
| the same: both sides | 8,060 | 1,007 |
| the whole entry, its 16 columns' windows NEVER READ BEFORE, 16 boxes a window's readers: the fold's side alone | 5,144 | 643 |
| the same: both sides | 5,778 | 722 |
| THE STREAM AS RULED, packed by digit, its lists uneven (8 to 24 entries), last steps truncated: the fold's side alone | 10,078 | 1,260 |
| the same: both sides | 10,443 | 1,305 |
| the stream as ruled, FOUR SLOTS OF 4 TILES a box: the fold's side alone | 10,719 | 1,340 |
| the same: both sides | 11,481 | 1,435 |
| the whole entry, THE FOLD WARPS EXPANDING their own slot: the fold's side alone | 2,698 | 337 |
| the same: both sides | 5,898 | 737 |

THE WALL TIME, measured on this box and immune to its clock (1.12 to 1.40 GHz, its rows' medians): the long call
scaled to 65,536 tokens is 0.319 ms for the baseline, 0.308 ms for the winner's form, and 0.603 ms at the
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
| a fold warp waits on the readout's read of a tile group | 380 | 4,194 | 3,802 |  |  | 3,750 | 406 |
| a fold warp waits on its producer | 4,539 |  |  |  |  | 435 | 5,422 |
| a readout warp waits on a tile group's publish | 1,747 | 290 | 324 |  |  | 303 | 2,039 |
| a readout warp waits on the panel's drain | 329 | 286 | 273 | 256 | 293 |  | 330 |
| a readout warp draining its own rows |  |  |  |  |  | 2,812 |  |
| a readout warp forming its steps' A | 1,345 | 1,102 | 1,108 | 1,128 | 1,004 | 1,096 |  |
| a readout warp in its steps (a lean readout's hold the forming) | 2,197 | 2,595 | 2,131 | 2,184 | 2,050 | 1,943 | 3,843 |
| a drain warp waits on the panel | 3,042 | 2,070 | 1,519 | 705 | 679 |  |  |

WHAT IT SHOWS.
- THE CARRY COMPOSES, EXACTLY: two boxes' folds, the image a window, one readout warpgroup over both -- its
  coefficients expanded in its registers, its steps' A operand -- and one f16 line a token, in one CTA of 512
  threads at the SM's whole register file.
- A TOKEN IS PRICED AT 1,782 SM-CYCLES COMPOSED, its tables read once (R22.2), in the winner's form; 827 when the
  cache holds them (868 in the first form) at this cell (its sides sparse on opposite levels, 256 x 256 leaves;
  [`carry_floor`](../internals/tools/carry_floor.md) prices a dense side higher).
- THE FILL WANTS WARPS THAT EACH HOLD A BATCH: two producers a box, a batch of 32 entries in registers each, take
  the fold's side alone to 3,042 cycles a window where one a box is 5,077; one a box holding TWO batches is
  4,528; and two a box at 40 registers a thread, each holding one value instruction's four entries, are
  5,871 -- slower than one a box that holds a batch.
- SUSPENDED WAITS WITH THE LEAN READOUT ARE THE FASTEST COMPOSED FORM: 6,616 cycles a window against the
  baseline's 6,948, with the suspended waits beside the wide byte load a near tie at 6,771. The parts: every wait
  suspended alone 6,811, the long waits asleep at named barriers 6,886, the lean readout alone 1.9%
  under the baseline, the wide load alone 6,872. ALL THREE AT ONCE LOSE the pairing's gain (6,727: the lean
  readout beside the wide load's scratch needs 88 registers a thread and the fourth warpgroup drops to 104), and
  the sleeping triple is 6,967.
- THE DEN COSTS 5.3% OF THE COMPOSED WINDOW AT THIS CELL AND 5.8% AT THE ALIGNED ONE, the tables cached and no
  clock inside a role: 6,889 cycles against 6,540, and 7,000 against 6,617. THE ROLES' CLOCKS ARE NOT FREE AND NOT EQUAL: they are
  76 cycles of the winner's clocked window and 179 of the den's, and which is larger has changed between
  records, so a clocked row does not price the den (clocked: 7,068 against 6,616; why the clocks differ is
  not isolated); with the tables once-read 14,579 against 14,252; the fold's side alone 4,805 against 5,077. One
  cell (its sides sparse on opposite levels); a dense write side, where the fold's side binds, is unmeasured. ITS
  PARTS, each a row less the row before, no clock inside a role: the fold's four steps a window 58 cycles;
  the image's ninth brick and its store 93 (the pitch itself nothing: the winner's form at 155 lines is 6,552
  against 6,540); the register split, readout 88 and fourth warpgroup 104, 94; the steps 72 columns wide 64;
  the panel's word and the drain's add 40. No part is a third of it. The split cannot be avoided in this
  form: the 72-column readout serializes its steps at 80 registers with no clock read too.
- THE FOLD'S FORM WAS A CONTEST. A step a list with ones in one row of A is 16 steps a window for 1 row in 64 and
  a 17th accumulator set of 8, which the toolkit will not hold at 160 registers: at a 168-register fold with
  one-batch producers it cost an eighth of the composed window in its session (the experiment at ac342446, whose
  record this one replaced). The coefficients as A are four steps into 4 registers.
- AN IMAGE TILE'S PITCH MUST BE AN ODD COUNT OF LINES: a warp's lanes store to four tiles two apart, and at an
  even pitch two of the four land on the same banks. The winner's form at 154 lines is 7,034 against 6,616 at 137;
  the den's at 161 is 7,070 against 7,068 at 155.
- THE IMAGE FORM COSTS 1.2% MORE WHERE THE CARVES ARE EQUAL, no clock inside a role (6,617 cycles a window against
  6,540); clocked, the aligned cell's winner's form is 6,642 cycles a
  window against 6,616 at the cell whose sides list opposite levels -- 16 word stores a thread a tile group where
  that cell's are 4 lines -- and with the den 7,095 against 7,068, the den's lines there 2-byte
  stores of 8 threads. A form that reads the fold's own registers has nothing of the window to win back but the
  image's stores; it is not built.
- THE CARRY'S PRODUCERS WANT A TOKEN'S VALUES AS ONE RUN: with the values tile-major in global memory -- a token's 8
  lines a kilobyte apart -- the winner's form is 11,997 cycles a window against 6,540, and with the tables once-read
  48,593 against 14,638. The intra's window moves 5% with the values' order ([`intra_window`](intra_window.md)), so
  the values stay token-major and the levels, which the carry reads 16 bytes at a time, go tile-major.
- FROM LISTS, WITH A WINDOW'S OWN TOKENS, THE FOLD'S SIDE ALONE IS 8,746 CYCLES A WINDOW where the synthetic entries'
  is 5,077 cached and 8,629 once-read; both sides 9,845 (the readout's side still on synthetic, cached readers) against
  6,540. Two things are in it, and the rows with every window the same tokens part them: THE LISTS' CODE -- a
  batch's places loaded a batch ahead, a shuffle a value instruction, the sparse word, tile-major addresses, and one
  batch held where the winner holds two -- is 6,246 against 5,077 alone and 8,024 against 6,540 composed; and ROWS THE
  CACHE HAS NOT SEEN, a window's own being new each window as a call's are, are the rest, 2,500 alone and
  1,821 composed. A window's rows are loaded one batch ahead of their use. ASKING FOR THEM A WINDOW AHEAD BY
  PREFETCH LOSES: 11,712 composed against 9,845 -- six prefetches an entry-batch are more requests of the same
  memory, not less waiting. Whether loads issued two batches ahead hide the rows' arrival is not measured (the
  producers' registers hold one batch from lists). The check of these rows is weak: every table's byte is its token's parity.
- LANDING THE VALUES ALONE FROM EXPANDED LISTS is 8,578 cycles a window composed against 9,845 with every entry's
  values gathered (8,024 for the gather with rows the cache holds), and the fold's side alone 8,437 against 8,746: a
  list's values are one bulk copy of 2 KB from a buffer laid contiguous a window (the host's here; 16 sources
  share a buffer), new each window, and its producers still gather an entry's 16 dense bytes and its sparse byte,
  a line of each level's rows an entry. What that leaves is the next finding's.
- WITH THE WHOLE ENTRY LANDED -- its coefficient bytes beside its values, raw, a list one bulk copy and no global load
  in a producer's trip -- the window is 5,808 cycles composed and the fold's side alone 5,147, against 6,540 and
  5,077 for the synthetic entries whose rows the cache holds.
- WHAT THAT ROW NEEDS IS ITS BYTES SHARED, NOT HELD BEFOREHAND: with a window's bytes never read before and its
  16 boxes its readers, the window is 5,778 cycles composed against 5,808 with the cache holding them, and
  the fold's side alone 5,144 against 5,147: one box's miss serves its 15 mates. With EACH BOX ITS OWN it is
  8,060 composed, 1.39 times, and 5,745 alone, 1.12 times -- 16 times the bytes from global memory, on
  every multiprocessor at once.
- THE STREAM AS RULED IS EXACT, AND ITS UNEVEN LISTS COST THE FOLD 1.96 TIMES: 10,078 cycles a window the
  fold's side alone against 5,147, 10,443 composed against 5,808. The same 256 entries a box a window, as
  lists of 8 to 24 where the row above has 16 each, are 24 steps where it has 16 -- three slots of 8 tiles where it
  has two -- and a truncated step, used where it landed with its bricks as far apart as it has entries, folds
  exactly. In four slots of 4 tiles, a slot's copies issued two slots ahead, the same stream is 10,719 alone
  and 11,481 composed: more slots cost more, so a slot's price is not its bytes' landing.
- THE FOLD'S SIDE WAITS ON ITS PRODUCER. With even lists a fold warp waits 4,518 of the fold's side's 5,147 cycles a
  window on its producer: its own steps are what is left. One producer warp a box expands every entry. On the
  packed stream in four slots the wait is 4,194 of 10,719, and with TWO PRODUCERS A BOX 1,501 of 8,485: the wait
  falls and the fold's own time is then the larger part. That fold runs a list's steps as a loop of its own, a
  vote a list a slot, where the even rows' steps are one block; the block rows below run the uneven lists' steps so.
- WITH THE FOLD WARPS EXPANDING THEIR OWN SLOT (even lists; a producer only issues copies) the fold's side alone
  is 2,698 cycles a window against 5,147, a fold warp's wait -- now on its slot's bytes -- 1,494 against 4,518.
  THE COMPOSED WINDOW DOES NOT MOVE: 5,898 against 5,808; with the fold issuing its own copies too, no
  producer running, 5,780.
- THE WINDOW IS THE READOUT'S. That form with every role's waits clocked (6,270 cycles a window): a readout warp
  waits 291 on the image and 278 on the drain and is 4,203 in its coefficients and steps; a fold
  warp waits 1,794 on the readout's read and 178 on its bytes; a drain warp waits 2,054 on the
  panel. The one role that does not wait is the readout's warpgroup.
- FOUR DRAIN WARPS BUY NOTHING: 5,774 against 5,780, each drain warp then waiting 4,034 of a window on the
  panel, and a readout warp is 3,793 in its coefficients and steps against 4,203. The drain is not the
  window's bound.
- THE MIRRORED READOUT (coefficients in shared memory, tokens on the step's columns; the fold expanding, a producer
  issuing): the window is 5,551 cycles against 5,898. Clocked (5,937 against 6,455): a readout warp is
  1,089 forming its coefficients and 1,921 in its steps, against 3,846 for both in the register form; it
  waits 343 on the image and 240 on the drain; a fold warp waits 2,642 on the readout's read against 2,799;
  a drain warp 2,236 on the panel against 2,445.
  What its clocks do not name is 2,344 cycles a window: its rows' rounding, their stores and its arrivals. Its 32
  steps are a quarter the register form's multiply-adds and take 1,921: a step's price here is not its size.
- THE BLOCK ROWS -- uneven lists, a block a list, the stream saying what to do: the window is 6,828 cycles
  composed (6,813 clocked), against 5,275 for even lists under the same readout and drains (whole-entry
  tiles, the fold issuing) -- 1.29 times it -- and
  10,443 for the uneven lists' truncated steps behind a producer and a loop a list.
  The fold's side alone is 3,747,
  a fold warp waiting 265 of it on its bytes, against 2,698 for the even lists' fold expanding its own slot. Clocked: a fold warp waits 249 on the
  readout's read and 187 on its bytes; a readout warp 1,778 on the image (323 with even lists) and 250 on the drain; a drain warp
  4,349 on the panel.
  THE FOLD'S PHASES, clocked, the block form against the even lists' (a clocked fold holds nine more
  registers; the windows are 6,813 and 5,579 with them): publishing the image 1,299 against 1,804,
  its wait on the readout's read in both (249 and 380); expanding its entries' lines 732 against 682; its pads and its
  steps' operands 748, the block form's alone; its slots' fences 57 and 67 against 65
  and 53; AT ITS SLOTS' RENDEZVOUS -- between a reading before the warpgroup's sync and one behind it --
  693 and 563 against 400 and 588; its steps and its next copy's issue 2,138 against 1,557.
  A FENCE IS SOME 60 CYCLES, AND A FOLD WARP IS LONGER AT A SLOT'S RENDEZVOUS THAN EXPANDING THE SLOT. With no
  arithmetic in the fold (the control: windows 6,609 against 6,828 and 5,195 against
  5,275; its entries' lines 449 and 327) its rendezvous are 743
  and 600, and 304 and 604: they do not follow the lines' work. THE
  FOLD'S ARITHMETIC IS NOT WHAT THE WINDOW WAITS ON.
  WITH BOTH COPIES ISSUED AT THE WINDOW'S END (even lists) the window is 5,456 against 5,275,
  its fences 62 and 63: a fence does not wait on a copy in flight, and the later
  copies cost the window.
  ITS RENDEZVOUS MOVE WITH THE ISSUE: 827 at its first slot and 162 at its second, where the form
  that issues a slot's copies as it leaves the slot is 400 and 588 -- the rendezvous behind an issue
  is the long one. A WARPGROUP WAITS THERE FOR ITS FIRST WARP, WHOSE THREADS ISSUE THE SLOT'S 8 COPIES.
  THE MEETING ROWS -- the rendezvous an mbarrier's -- DO NOT MOVE IT: the window is 5,337 against
  5,275 with even lists and 6,887 against 6,828 in the block form; clocked, a fold warp
  is 492 and 597 at its rendezvous (that row's folds at 168 registers), and
  685 and 590 in the block form. The time is not the named barrier's own.
  WITH EVERY WAIT POLLED, not suspended (even lists): the window is 5,501 against 5,275;
  clocked (5,817), a fold warp's rendezvous are 416 and 583, its waits 325 on the readout's
  read and 362 on its bytes.
  A slot's steps are a
  fixed two rounds: ISSUING ONLY THE STEPS ITS LISTS HAVE, a second-round step a trip of one or none, WAS MEASURED
  AND LOST (the experiment at 758f0d43, its record that commit's: 7,022 against 6,822 composed, the fold's side
  alone 4,385 against 3,741), and its rows are not kept. NOT ASKED: the drain warps at 88 registers under this
  fold (they are at 72 here); lists past two rounds.
- THE MIRRORED READOUT'S SUMS TO THE PANEL ARE ITS LARGEST PHASE (the even lists' form, clocked): a readout warp is
  1,980 cycles a window rounding its sums, storing them a half each and arriving -- its
  2,259 less its 279 waiting on the drain -- against 1,236 in its steps and
  1,116 forming its coefficients. NOT ASKED: the panel laid so a thread's two sums are one word's store.
- THE OUTPUT PATH ON FOUR WARPS (the chunk-of-8 form, the fold issuing its own copies, four drain warps): the window
  is 5,275 cycles against 5,530 with two drain warps and a producer; clocked 5,579 against 5,627: a readout
  warp waits 279 on the drain against 657, a drain warp 3,480 on the panel against 610 -- and A FOLD WARP WAITS
  587 of the window, on the readout's read and on its bytes, against 2,127: the folds, which waited a third
  of a window, now wait a tenth; no clock says in what.
- WITH THE DRAIN WARPS ROUNDING the window is 5,307 against 5,275: nothing. Clocked (5,481), a readout
  warp's unclocked remainder is 1,815 against 2,626 and it waits 1,079 on the image against 323: what the
  readout sheds it waits out on the folds' image.
- THE MIRRORED READOUT WITH A CHUNK OF 8 DIGITS: the window is 5,530 cycles against 5,551 at 4. Clocked (5,627
  against 5,937): a readout warp is 960 forming and 1,532 in its steps against 1,089 and 1,921 -- half the
  passes, three quarters of the time -- and what its clocks do not name is 2,225 against 2,344: its
  rows' rounding and stores are a value's, not a pass's. It waits 657 on the drain against 240, and a drain
  warp waits 610 on the panel against 2,236: the two drain warps are near the window's whole length at
  work (their warpgroup at 88 registers here, 96 there; no row parts that from the two panels handed at once).
- WITH THE READOUT A SUM A BOX (the fold expanding, a producer issuing) THE WINDOW IS LONGER: 6,269 cycles against
  5,898. Clocked (6,485 against 6,455): a readout warp is 3,358 in its coefficients and steps against 3,846 --
  two chains of 4 steps are a tenth under one of 8, so the chain was not what its steps cost -- and it waits
  856 on the drain against 243, a drain warp waiting 344 on the panel against 2,445. The row's serve
  warps are at 80 registers where the other's are at 112, and its drain is what the window then waits on; no row
  parts the drain's registers from the readout's second sum. NOT ASKED: the readout's steps by what they
  multiply -- a step is 64 rows of which one digit's 16 are not zero; its coefficients formed by warps that
  wait; the fold-expands form on the uneven stream; a truncated step's pads left out of the expanding.
- THE TABLES ONCE-READ COST 1.7 TO 2.3 TIMES THE WINDOW ON THIS BOX (in cycles, which for a row bound by global
  memory follow the card's clock): 15,707 cycles composed against 6,948, the winner's form
  14,252 against 6,616, the fold's side alone 8,629 against 5,077 -- every access a fresh row of 128 MB of
  values, the model of a large table whose rows are each read once.
- THE DUMP IS PRICED IN THIS SESSION: the fold's state stored out in every call, as every record before 79ec1b08
  ran, puts the composed row at 8,116 against 6,948 -- one binary, one box, one session.
- FIVE WARPGROUPS LOSE COMPOSED, 7,589 against 6,948: 640 threads leave four producers and four drain warps 40
  registers a thread, the fold waits on its producers 5,422 cycles a window, and the lean readout the count
  forces is 3,843 in its steps and their forming where one commit group's are 3,541. Four drain warps do
  help the read side alone, 5,050 against 5,565.
- THE LEAN READOUT WITH TWO-BATCH PRODUCERS IS 6,813 COMPOSED against 6,948: a readout warp is 4,451 cycles
  a window in its steps and their forming. The two changes are not separated.
- THE READOUT DRAINING ITS OWN ROWS LOSES COMPOSED, 7,397 against 6,948: a readout warp is 2,812 cycles a
  window in its 64 adds, and the window becomes the readout's -- the fold waits on its read 3,750. A warp that
  issues an add is held until the SM passes it: adds issued by the readout hold the readout.
- COMPOSED, THE PRODUCERS' LOADS ARE NOW THE LARGEST SINGLE ACCESS: with them out the window is 6,412 against 6,948, and
  those loads are three fifths of the fold's side alone -- 2,414 cycles a window without them against 5,077, some
  166 cycles a batch of a producer's 9 load instructions; their coefficient arithmetic is 413 of it
  (4,664 without). The drains' adds out, the window is 6,588; the image's stores out, 6,511.
- A PARKED FOLD WARP COSTS THE READ SIDE 101 CYCLES, A PUBLISHING ONE 1,325: the read side alone is 5,565
  with the fold warps publishing and 4,240 with no fold warp running, while one parked asleep beside it costs
  101 cycles and one spinning 256. What of the publishing fold's hand-off holds it is NOT ISOLATED
  (its stores are 572 of it; its fence nothing).
- THE READOUT'S OWN BYTE LOADS ARE A SIXTH OF ITS STEPS' SPAN: 8 instructions a chunk, each lane's 4 bytes of a
  different row; without them a readout warp's steps are 2,131 cycles a window against 2,595. The read side alone
  is then the drain's -- 5,001 against 5,565, the readout waiting on the drain 273 -- at some
  27 cycles an add a drain warp, where with no fold warp running an add is 26.
- ONE WIDE LOAD A WARP FOR THE READOUT'S BYTES is the read side alone without those loads -- 5,053 cycles a window
  against 5,001 with them out and 5,565 as they were -- and little composed: 6,872 against 6,948, where the
  byte loads out are 6,591 and the drains' adds out with them 6,392.
- THE READOUT FORMS ITS STEPS' A IN 1,102 TO 1,345 CYCLES A WINDOW, 275 to 336 a chunk: the coefficients' rounds
  and the 32 words held.
- THE STEPS THEMSELVES ARE OVER THEIR OWN TIME with nothing beside them but the drains: 546 cycles a chunk (the
  fence, 8 steps, the commit, 8 loads issued, the wait) where 8 steps of 64 columns into one accumulator from
  shared memory are 319 ([`wgmma_width`](wgmma_width.md)). Not isolated here.
- EVERY ROLE ISSUES AN INSTRUCTION EVERY 3.5 TO 7 CYCLES of its busy time (the static ledger's counts a trip: a
  producer 168 a batch, a drain warp 339 a chunk, the readout 284 a chunk, a fold warp 379 a window): no role is
  bound by the instructions it issues, and what holds each is not isolated.

Record: `measure/lab/results/carry_window/sm_90a.json`.
