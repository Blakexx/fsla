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

THE PACKED STREAM (a pair; [`hopper-carry.md`](../architecture/hopper-carry.md#the-gather-stream)). The same 256
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
next copy's issue, the two slots' fences and rendezvous apart, and its steps' issue to their commit apart from
their wait (the mirrored readout's likewise, its next bytes' loads in the issue). A clocked fold holds nine more registers, and its phase laps' readings are each thread's own
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
for the warp that issues. Checked as the form's rows are. A clocked fold's sites are recorded for its warpgroups' first warps
alone beside every warp's mean.
THE BLOCK FORM'S SHARED PAIR: its plans and its one copy a slot are its second warp's first thread's, its 16
steps' operands four a warp, its pads a warp in turn -- no warp of the warpgroup holds more of the slot's lone
jobs than another. Checked as the form's rows are.
THE READY PAIR (the block form; [`hopper-carry.md`](../architecture/hopper-carry.md#the-gather-stream)): a slot
LANDS WITH ITS OPERANDS MADE -- its head the 16 steps' two operands as a slot at address 0 has them, a tile of
zeros for an absent step's, its lists' value runs, and its coefficients bf16, a brick a run, padded with zero lines
to the lists' steps -- and the fold waits on its bytes, adds its slot's address to each operand and steps: it
writes no shared memory but its window's plan, so it has no fence and no rendezvous; a step's two operands are one
landed line, the slot's line added to each one's first word. THE QUESTION: what the
fold's expansion, pads, operands, fence and rendezvous cost the uneven lists' window. The host lays the stream; a
slot's copy is some 14% more bytes. Checked as the form's rows are.
THE READ-READY ROWS (the ready pair's form and the even lists', with and without clocks): the mirrored readout's
coefficient tiles LAND MADE too -- a pass's 16 tiles one copy from a CTA's stream of its passes, issued by the
warpgroup's first thread behind the steps of the pass before and under its panel's stores, into the one buffer
the steps read (shared memory holds no second) -- and the readout waits on its bytes and steps: no expansion, no
fence, no rendezvous. THE QUESTION: what the readout's coefficient forming and its rendezvous cost the window
it bounds. The host lays the stream, 8.7 KB a pass. Checked as the form's rows are. The copy's edge behind the
old tiles' last read is its issuing thread's own wait on the steps, TAKEN as the warpgroup's one operation
complete for its four warps at once: no lab measures that apart. In these rows the site recorded as the readout
forming its coefficients is its wait on its tiles' bytes.
THE OWN ROWS (the read-ready ready pair's form, with and without clocks): EVERY BOX LANDS ITS OWN STREAM, a copy
of its column's, where the form's 16 boxes share a column's. In the design a box's coefficient part is its own
and its values are its column's ([`hopper-gather.md`](../architecture/hopper-gather.md)): the form's rows and
these bracket what a box's own bytes cost its window. Checked as the form's rows are.

THE TOKEN ROWS (the block form;
[`hopper-carry.md`](../architecture/hopper-carry.md#the-gather-stream)): THE FOLD LAYS ITS OWN SLOT from a view in token order. THE QUESTION: the design's stream holds a
sparse block's live tokens once each as they come, so a token's value lines do not land where its digit's step reads
them -- what does the fold moving them cost the window, beside the block form, whose value runs land in place? A
slot's view is its 128 entries in token order -- each entry's places, its dense line and its 8 value lines, a brick a
run -- one copy into the box's second pool slot, issued a slot ahead, behind the rendezvous of the slot before; a
thread expands its entry's dense line into the box's first pool slot and moves its 8 value lines there, a pad's two
lines are zeroed, and the steps are the block form's. The view is the block image's lists and bytes; what differs is
the order its entries land in, who lays them, and the copy issued one slot ahead where the block form's is a window.
THE THROUGH ROW asks what the view costs with NO SHARED MEMORY TO STAGE IT: the view lands without its value lines and
a thread loads its entry's 8 from global memory, two in registers at a time. It has no clocked row, whose fold spills
at its 168 registers. A slot of those rows is 8 of the block's 16 digits, as the block form's is. THE ONE-VIEW ROWS ask what the ruled fill's STRUCTURE is worth -- one view a window and one pass, where the block form
is two slots -- and what the view costs through registers: the block's 16 lists' 256 entries in token order, a thread
two of them, each entry's dense line and 8 value lines loaded from global memory and laid across the box's two pool
slots, and the window's 32 steps one block behind one fence and one rendezvous, where the block form's are two of
each. Their PLACES ARE THE HOST'S: each entry's lines and the window's 32 steps are a table the window lands, where
the design's CTA counts a rank from the mask and its steps from the window's counts -- these rows price the fill's moves and its steps, not that counting. All
checked as the block form's rows are.

THE JOINED ROW (the block form, clocked: its unclocked fold spills an address at its 168 registers) asks what ONE RENDEZVOUS A WINDOW is worth where the values
land in place: the block form's two slots are both expanded before one fence and one barrier, the window's 32 steps
are one block, and both of the next window's copies are issued behind them -- the one-view rows' structure on the
block form's landed images. It also issues both copies at the window's end, where the block form issues a slot's as
it leaves the slot: the row does not part the two. Checked as the block form's rows are; its window is read beside
the block form's clocked one.

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

MEASURED (H100 PCIe 80GB, 114 SMs, Hyperstack spot VM 1161570; a role's steps are its issue and its wait, which the record holds apart, driver 580.196.07; 2026-11-15, the experiment at
2b496b8d, one session; 7 repetitions, every checked row's state and output values exact;
the median; SM-cycles a token are the window's cycles over 8 -- a CTA of two boxes is 8 tokens a window. THE FOLD'S STATE
DUMP STAYS OUT OF THE TIMED CALLS, and one row restores it to price it in this session):

| the row | cycles a window a CTA | SM-cycles a token |
|---|---|---|
| the fold's side alone: producers and steps | 5,011 | 626 |
| the read side alone: the image, the readout and the drain | 5,550 | 694 |
| the carry: both sides | 6,935 | 867 |
| both sides, the producers' global loads taken out | 6,421 | 803 |
| both sides, the drains' global adds taken out | 6,647 | 831 |
| both sides, the image's stores taken out | 6,588 | 823 |
| the read side alone, no fold warp running | 4,272 | 534 |
| the read side alone, every wait suspended | 5,259 | 657 |
| both sides, every wait suspended | 6,813 | 852 |
| THE READOUT DRAINING ITS OWN ROWS, FOUR PRODUCERS: the fold's side alone | 3,024 | 378 |
| the same: the read side alone | 5,347 | 668 |
| the same: both sides | 7,389 | 924 |
| THE READOUT LEAN, TWO BATCHES A PRODUCER: the fold's side alone | 4,454 | 557 |
| the same: the read side alone | 5,756 | 719 |
| the same: both sides | 6,766 | 846 |
| FIVE WARPGROUPS, FOUR PRODUCERS AND FOUR DRAIN WARPS AT 40: the fold's side alone | 5,727 | 716 |
| the same: the read side alone | 5,036 | 630 |
| the same: both sides | 7,599 | 950 |
| THE LONG WAITS ASLEEP AT NAMED BARRIERS: the fold's side alone | 4,838 | 605 |
| the same: the read side alone | 5,051 | 631 |
| the same: both sides | 6,868 | 858 |
| the fold's side alone, the producers' global loads taken out | 2,412 | 301 |
| the fold's side alone, their coefficient arithmetic taken out | 4,557 | 570 |
| the fold's side alone, both taken out | 2,281 | 285 |
| the read side alone, the readout's byte loads taken out | 5,003 | 625 |
| the read side alone, no fold warp running, the byte loads taken out | 4,017 | 502 |
| both sides, the readout's byte loads taken out | 6,563 | 820 |
| both sides, the drains' adds and the readout's byte loads taken out | 6,375 | 797 |
| THE READOUT'S BYTES ONE WIDE LOAD A WARP: the read side alone | 5,018 | 627 |
| the same: the read side alone, no fold warp running | 4,053 | 507 |
| the same: both sides | 6,851 | 856 |
| the same: both sides, the long waits asleep | 6,979 | 872 |
| the read side alone, the image's stores taken out | 4,989 | 624 |
| the read side alone, the fold's fence behind its image taken out | 5,494 | 687 |
| the read side alone, both taken out | 4,975 | 622 |
| both sides, that fence taken out | 6,825 | 853 |
| the read side alone, no fold warp running, the fold warps parked asleep | 4,252 | 532 |
| the same, spinning | 4,446 | 556 |
| THE DUMP RESTORED: both sides, the fold's state stored out in every call | 7,923 | 990 |
| THE WINNERS COMBINED: both sides, suspended waits and the lean readout | 6,588 | 823 |
| the same, suspended waits and the wide byte load | 6,742 | 843 |
| the same, suspended waits, the lean readout and the wide byte load (the fourth warpgroup at 104) | 6,726 | 841 |
| the same, sleeping waits, the lean readout and the wide byte load | 6,990 | 874 |
| THE TABLES ONCE-READ, 16 replicas the cache cannot hold: both sides | 20,495 | 2,562 |
| the same, the winner's form | 19,228 | 2,404 |
| the same: the fold's side alone | 12,165 | 1,521 |
| THE DEN, folded, published, read and drained: the fold's side alone | 4,989 | 624 |
| the same: both sides, the winner's form | 7,061 | 883 |
| the same, the tables once-read | 19,474 | 2,434 |
| THE IMAGE'S TILES MOVED APART, nothing else changed: the winner's form at 154 lines (137 as it is) | 7,016 | 877 |
| the winner's form with the den at 161 lines (155 as it is) | 7,055 | 882 |
| THE ALIGNED CELL, an image tile a list: the winner's form | 6,610 | 826 |
| the same with the den | 7,088 | 886 |
| NO CLOCK INSIDE A ROLE: the winner's form | 6,534 | 817 |
| the same with the den | 6,894 | 862 |
| the same at the aligned cell | 6,611 | 826 |
| the same at the aligned cell with the den | 7,004 | 875 |
| THE DEN'S PARTS, a chain with no clock inside a role: the fold's steps | 6,594 | 824 |
| then the image's ninth brick and its store | 6,688 | 836 |
| then the register split | 6,782 | 848 |
| then the 72-column steps | 6,847 | 856 |
| the winner's form with its tiles at the den's pitch, 155 lines | 6,541 | 818 |
| THE VALUES TILE-MAJOR in global memory, no clock inside a role: the winner's form | 11,811 | 1,476 |
| the tables once-read, the values token-major | 19,534 | 2,442 |
| the tables once-read, the values tile-major | 64,173 | 8,022 |
| THE FOLD'S ENTRIES FROM LISTS, a window's own tokens: the fold's side alone | 8,570 | 1,071 |
| the same: both sides, no clock inside a role | 9,812 | 1,226 |
| the same with every window the same tokens: the fold's side alone | 6,113 | 764 |
| the same: both sides | 7,951 | 994 |
| a window's own tokens, their rows asked for a window ahead: both sides | 11,478 | 1,435 |
| THE VALUES LANDED BY BULK COPY from expanded lists, a window's own tokens: the fold's side alone | 7,638 | 955 |
| the same: both sides | 8,152 | 1,019 |
| THE WHOLE ENTRY LANDED, values and coefficient bytes: the fold's side alone | 4,972 | 622 |
| the same: both sides | 5,808 | 726 |
| the whole entry, EACH BOX ITS OWN expanded windows, past the cache: the fold's side alone | 7,091 | 886 |
| the same: both sides | 12,364 | 1,545 |
| the whole entry, its 16 columns' windows NEVER READ BEFORE, 16 boxes a window's readers: the fold's side alone | 4,965 | 621 |
| the same: both sides | 5,814 | 727 |
| THE PACKED STREAM, packed by digit, its lists uneven (8 to 24 entries), last steps truncated: the fold's side alone | 9,791 | 1,224 |
| the same: both sides | 10,316 | 1,290 |
| the packed stream, FOUR SLOTS OF 4 TILES a box: the fold's side alone | 10,678 | 1,335 |
| the same: both sides | 11,474 | 1,434 |
| the whole entry, THE FOLD WARPS EXPANDING their own slot: the fold's side alone | 2,462 | 308 |
| the same: both sides | 5,874 | 734 |

THE WALL TIME, measured on this box and immune to its clock (1.56 to 1.64 GHz, its rows' medians): the long call
scaled to 65,536 tokens is 0.321 ms for the baseline, 0.310 ms for the winner's form, and 0.761 ms at the
once-read bound (0.833 without the winner's changes). THE FLASH REFERENCES, from the session of instance 60450651 ON ANOTHER BOX (a different offer and driver)
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
| a fold warp waits on the readout's read of a tile group | 379 | 4,191 | 3,802 |  |  | 3,752 | 407 |
| a fold warp waits on its producer | 4,530 |  |  |  |  | 425 | 5,450 |
| a readout warp waits on a tile group's publish | 1,741 | 291 | 323 |  |  | 301 | 2,057 |
| a readout warp waits on the panel's drain | 329 | 286 | 273 | 253 | 282 |  | 329 |
| a readout warp draining its own rows |  |  |  |  |  | 2,805 |  |
| a readout warp forming its steps' A | 1,338 | 1,089 | 1,109 | 1,166 | 1,007 | 1,096 |  |
| a readout warp in its steps (a lean readout's hold the forming) | 2,196 | 2,592 | 2,131 | 2,183 | 2,049 | 1,943 | 3,843 |
| a drain warp waits on the panel | 3,033 | 2,048 | 1,521 | 731 | 700 |  |  |

WHAT IT SHOWS.
- THE CARRY COMPOSES, EXACTLY: two boxes' folds, the image a window, one readout warpgroup over both -- its
  coefficients expanded in its registers, its steps' A operand -- and one f16 line a token, in one CTA of 512
  threads at the SM's whole register file.
- A TOKEN IS PRICED AT 2,404 SM-CYCLES COMPOSED, its tables read once (R22.2), in the winner's form; 823 when the
  cache holds them (867 in the first form) at this cell (its sides sparse on opposite levels, 256 x 256 leaves;
  [`carry_floor`](../internals/tools/carry_floor.md) prices a dense side higher).
- THE FILL WANTS WARPS THAT EACH HOLD A BATCH: two producers a box, a batch of 32 entries in registers each, take
  the fold's side alone to 3,024 cycles a window where one a box is 5,011; one a box holding TWO batches is
  4,454; and two a box at 40 registers a thread, each holding one value instruction's four entries, are
  5,727 -- slower than one a box that holds a batch.
- SUSPENDED WAITS WITH THE LEAN READOUT ARE THE FASTEST COMPOSED FORM: 6,588 cycles a window against the
  baseline's 6,935, with the suspended waits beside the wide byte load a near tie at 6,742. The parts: every wait
  suspended alone 6,813, the long waits asleep at named barriers 6,868, the lean readout alone 2.4%
  under the baseline, the wide load alone 6,851. ALL THREE AT ONCE LOSE the pairing's gain (6,726: the lean
  readout beside the wide load's scratch needs 88 registers a thread and the fourth warpgroup drops to 104), and
  the sleeping triple is 6,990.
- THE DEN COSTS 5.5% OF THE COMPOSED WINDOW AT THIS CELL AND 5.9% AT THE ALIGNED ONE, the tables cached and no
  clock inside a role: 6,894 cycles against 6,534, and 7,004 against 6,611. THE ROLES' CLOCKS ARE NOT FREE AND NOT EQUAL: they are
  54 cycles of the winner's clocked window and 167 of the den's, and which is larger has changed between
  records, so a clocked row does not price the den (clocked: 7,061 against 6,588; why the clocks differ is
  not isolated); with the tables once-read 19,474 against 19,228; the fold's side alone 4,989 against 5,011. One
  cell (its sides sparse on opposite levels); a dense write side, where the fold's side binds, is unmeasured. ITS
  PARTS, each a row less the row before, no clock inside a role: the fold's four steps a window 60 cycles;
  the image's ninth brick and its store 93 (the pitch itself nothing: the winner's form at 155 lines is 6,541
  against 6,534); the register split, readout 88 and fourth warpgroup 104, 94; the steps 72 columns wide 65;
  the panel's word and the drain's add 47. No part is a third of it. The split cannot be avoided in this
  form: the 72-column readout serializes its steps at 80 registers with no clock read too.
- THE FOLD'S FORM WAS A CONTEST. A step a list with ones in one row of A is 16 steps a window for 1 row in 64 and
  a 17th accumulator set of 8, which the toolkit will not hold at 160 registers: at a 168-register fold with
  one-batch producers it cost an eighth of the composed window in its session (the experiment at ac342446, whose
  record this one replaced). The coefficients as A are four steps into 4 registers.
- AN IMAGE TILE'S PITCH MUST BE AN ODD COUNT OF LINES: a warp's lanes store to four tiles two apart, and at an
  even pitch two of the four land on the same banks. The winner's form at 154 lines is 7,016 against 6,588 at 137;
  the den's at 161 is 7,055 against 7,061 at 155.
- THE IMAGE FORM COSTS 1.2% MORE WHERE THE CARVES ARE EQUAL, no clock inside a role (6,611 cycles a window against
  6,534); clocked, the aligned cell's winner's form is 6,610 cycles a
  window against 6,588 at the cell whose sides list opposite levels -- 16 word stores a thread a tile group where
  that cell's are 4 lines -- and with the den 7,088 against 7,061, the den's lines there 2-byte
  stores of 8 threads. A form that reads the fold's own registers has nothing of the window to win back but the
  image's stores; it is not built.
- THE CARRY'S PRODUCERS WANT A TOKEN'S VALUES AS ONE RUN: with the values tile-major in global memory -- a token's 8
  lines a kilobyte apart -- the winner's form is 11,811 cycles a window against 6,534, and with the tables once-read
  64,173 against 19,534. The intra's window moves 5% with the values' order ([`intra_window`](intra_window.md)), so
  the values stay token-major and the levels, which the carry reads 16 bytes at a time, go tile-major.
- FROM LISTS, WITH A WINDOW'S OWN TOKENS, THE FOLD'S SIDE ALONE IS 8,570 CYCLES A WINDOW where the synthetic entries'
  is 5,011 cached and 12,165 once-read; both sides 9,812 (the readout's side still on synthetic, cached readers) against
  6,534. Two things are in it, and the rows with every window the same tokens part them: THE LISTS' CODE -- a
  batch's places loaded a batch ahead, a shuffle a value instruction, the sparse word, tile-major addresses, and one
  batch held where the winner holds two -- is 6,113 against 5,011 alone and 7,951 against 6,534 composed; and ROWS THE
  CACHE HAS NOT SEEN, a window's own being new each window as a call's are, are the rest, 2,457 alone and
  1,861 composed. A window's rows are loaded one batch ahead of their use. ASKING FOR THEM A WINDOW AHEAD BY
  PREFETCH LOSES: 11,478 composed against 9,812 -- six prefetches an entry-batch are more requests of the same
  memory, not less waiting. Whether loads issued two batches ahead hide the rows' arrival is not measured (the
  producers' registers hold one batch from lists). The check of these rows is weak: every table's byte is its token's parity.
- LANDING THE VALUES ALONE FROM EXPANDED LISTS is 8,152 cycles a window composed against 9,812 with every entry's
  values gathered (7,951 for the gather with rows the cache holds), and the fold's side alone 7,638 against 8,570: a
  list's values are one bulk copy of 2 KB from a buffer laid contiguous a window (the host's here; 16 sources
  share a buffer), new each window, and its producers still gather an entry's 16 dense bytes and its sparse byte,
  a line of each level's rows an entry. What that leaves is the next finding's.
- WITH THE WHOLE ENTRY LANDED -- its coefficient bytes beside its values, raw, a list one bulk copy and no global load
  in a producer's trip -- the window is 5,808 cycles composed and the fold's side alone 4,972, against 6,534 and
  5,011 for the synthetic entries whose rows the cache holds.
- WHAT THAT ROW NEEDS IS ITS BYTES SHARED, NOT HELD BEFOREHAND: with a window's bytes never read before and its
  16 boxes its readers, the window is 5,814 cycles composed against 5,808 with the cache holding them, and
  the fold's side alone 4,965 against 4,972: one box's miss serves its 15 mates. With EACH BOX ITS OWN it is
  12,364 composed, 2.13 times, and 7,091 alone, 1.43 times -- 16 times the bytes from global memory, on
  every multiprocessor at once.
- THE PACKED STREAM IS EXACT, AND ITS UNEVEN LISTS COST THE FOLD 1.97 TIMES: 9,791 cycles a window the
  fold's side alone against 4,972, 10,316 composed against 5,808. The same 256 entries a box a window, as
  lists of 8 to 24 where the row above has 16 each, are 24 steps where it has 16 -- three slots of 8 tiles where it
  has two -- and a truncated step, used where it landed with its bricks as far apart as it has entries, folds
  exactly. In four slots of 4 tiles, a slot's copies issued two slots ahead, the same stream is 10,678 alone
  and 11,474 composed: more slots cost more, so a slot's price is not its bytes' landing.
- THE FOLD'S SIDE WAITS ON ITS PRODUCER. With even lists a fold warp waits 4,339 of the fold's side's 4,972 cycles a
  window on its producer: its own steps are what is left. One producer warp a box expands every entry. On the
  packed stream in four slots the wait is 4,168 of 10,678, and with TWO PRODUCERS A BOX 1,481 of 8,467: the wait
  falls and the fold's own time is then the larger part. That fold runs a list's steps as a loop of its own, a
  vote a list a slot, where the even rows' steps are one block; the block rows below run the uneven lists' steps so.
- WITH THE FOLD WARPS EXPANDING THEIR OWN SLOT (even lists; a producer only issues copies) the fold's side alone
  is 2,462 cycles a window against 4,972, a fold warp's wait -- now on its slot's bytes -- 1,266 against 4,339.
  THE COMPOSED WINDOW DOES NOT MOVE: 5,874 against 5,808; with the fold issuing its own copies too, no
  producer running, 5,801.
- THE WINDOW IS THE READOUT'S. That form with every role's waits clocked (6,246 cycles a window): a readout warp
  waits 290 on the image and 278 on the drain and is 4,171 in its coefficients and steps; a fold
  warp waits 1,755 on the readout's read and 170 on its bytes; a drain warp waits 1,977 on the
  panel. The one role that does not wait is the readout's warpgroup.
- FOUR DRAIN WARPS BUY NOTHING: 5,751 against 5,801, each drain warp then waiting 4,014 of a window on the
  panel, and a readout warp is 3,756 in its coefficients and steps against 4,171. The drain is not the
  window's bound.
- THE MIRRORED READOUT (coefficients in shared memory, tokens on the step's columns; the fold expanding, a producer
  issuing): the window is 5,524 cycles against 5,874. Clocked (5,908 against 6,428): a readout warp is
  1,092 forming its coefficients and 152 in its steps, against 3,826 for both in the register form; it
  waits 300 on the image and 234 on the drain; a fold warp waits 2,617 on the readout's read against 2,791;
  a drain warp 2,228 on the panel against 2,408.
  What its clocks do not name is 4,129 cycles a window: its rows' rounding, their stores and its arrivals. Its 32
  steps are a quarter the register form's multiply-adds and take 152: a step's price here is not its size.
- THE BLOCK ROWS -- uneven lists, a block a list, the stream saying what to do: the window is 6,828 cycles
  composed (6,817 clocked), against 5,267 for even lists under the same readout and drains (whole-entry
  tiles, the fold issuing) -- 1.30 times it -- and
  10,316 for the uneven lists' truncated steps behind a producer and a loop a list.
  The fold's side alone is 3,746,
  a fold warp waiting 264 of it on its bytes, against 2,462 for the even lists' fold expanding its own slot. Clocked: a fold warp waits 249 on the
  readout's read and 189 on its bytes; a readout warp 1,794 on the image (344 with even lists) and 252 on the drain; a drain warp
  4,320 on the panel.
  THE FOLD'S PHASES, clocked, the block form against the even lists' (a clocked fold holds nine more
  registers; the windows are 6,817 and 5,722 with them): publishing the image 1,304 against 1,892,
  its wait on the readout's read in both (249 and 266); expanding its entries' lines 729 against 772; its pads and its
  steps' operands 753, the block form's alone; its slots' fences 56 and 73 against 67
  and 57; AT ITS SLOTS' RENDEZVOUS -- between a reading before the warpgroup's sync and one behind it --
  689 and 549 against 354 and 503; its steps and its next copy's issue 2,143 against 1,668.
  A FENCE IS SOME 60 CYCLES, AND A FOLD WARP IS LONGER AT A SLOT'S RENDEZVOUS THAN EXPANDING THE SLOT. With no
  arithmetic in the fold (the control: windows 6,597 against 6,828 and 5,185 against
  5,267; its entries' lines 472 and 317) its rendezvous are 748
  and 557, and 348 and 581: they do not follow the lines' work. THE
  FOLD'S ARITHMETIC IS NOT WHAT THE WINDOW WAITS ON.
  WITH BOTH COPIES ISSUED AT THE WINDOW'S END (even lists) the window is 5,458 against 5,267,
  its fences 66 and 58: a fence does not wait on a copy in flight, and the later
  copies cost the window.
  ITS RENDEZVOUS MOVE WITH THE ISSUE: 850 at its first slot and 156 at its second, where the form
  that issues a slot's copies as it leaves the slot is 354 and 503 -- the rendezvous behind an issue
  is the long one.
  A WARPGROUP WAITS AT ITS RENDEZVOUS FOR ITS FIRST WARP, whose lanes issue the slot's 8 copies: the first warp
  alone is 95 and 94 at its two rendezvous where the warps' mean is 354 and 503, and
  2,617 in its steps and its issue where the mean is 1,668.
  WITH THE COPIES ISSUED TWO A WARP every warp is 226 and 241 at its rendezvous and 1,681 in its
  steps and issue; the window is 5,213 against 5,267, and a fold warp then waits 805 on the
  readout's read against 266: THE FOLD'S TIME SAVED IS ITS WAIT ON THE READOUT, WHICH IS THE WINDOW.
  THE BLOCK FORM'S FIRST WARP IS LAST TOO: 102 and 97 at its rendezvous against 689 and
  549; its plans, its one copy a slot, 16 threads' operands and the first pads are its own -- 1,135 at
  its pads and operands against 753, 858 expanding against 729, 2,544 in its steps and
  issue against 2,143. WITH THOSE JOBS SHARED AMONG ITS WARPS THE WINDOW DOES NOT MOVE: 6,856 against
  6,828. Its rendezvous fall to 472 and 293 and every warp's pads and operands rise to
  1,227: the phase is some 53 instructions a thread, and a warp is as long in it with a pad and four operands
  as the first warp was with all of them. NOT ISOLATED: what a warp waits on in that phase.
  THE READY PAIR -- the block form's slots landed with their operands made, the fold only landing and stepping:
  the window is 5,667 against 6,828 for the block form and 5,267 for even
  lists; clocked (5,787), a fold warp waits 217 on its bytes and
  903 on the readout's read, is 2,179 issuing its 32 steps and
  846 in their wait and its copy's issue, and A READOUT WARP WAITS 296 ON THE
  IMAGE against 1,794 with the fold expanding and 344 with even lists: with its operands made the
  uneven lists' fold is no longer what the window waits on. What the window is over the even lists' by, no row
  parts: the fold issues twice the steps beside the same readout.
  THE READ-READY ROWS -- the readout's coefficient tiles landed made as well: the window is 4,867 against
  5,267 with even lists, and 5,189 against 5,667 for the ready pair's form. Clocked
  (5,218 and 5,306), a readout warp is 337 and 284 from its
  image's wait to its tiles landed, where forming them itself with its rendezvous is 1,472 and
  1,558; it waits 1,314 and 1,140 on the image, and a fold warp
  356 and 294 on the readout's read.
  THE OWN ROWS -- every box its own stream, where 16 share a column's -- ARE THREE TIMES THE WINDOW: 15,145
  against 5,189. Their streams are some 680 MB, past the card's cache, where the form's are under it; a
  window's bytes a CTA are the same. Clocked (15,147): A READOUT WARP WAITS 11,576 ON ITS
  TILES' BYTES against 284 -- its copy is issued one panel's stores ahead of its use -- and
  a fold warp, whose copies are issued a window ahead, 701 on its own against 210; the
  fold then waits 9,624 on the readout's read. WITH ITS OPERANDS FROM THE CARD'S MEMORY THE CARRY IS
  THAT MEMORY'S: the ready stream is a window's 110 KB a CTA, and past the cache it is this price. NOT ASKED: the
  readout's copy issued further ahead; the design's mix, a box's own coefficients and its column's values.
  THE TOKEN ROWS -- the fold laying its own operands from a view in token order -- ARE A QUARTER OVER THE BLOCK FORM, HOWEVER THE VIEW
  ARRIVES: the window is 8,474 with a half-window view staged by one copy and its value lines moved in shared memory,
  8,526 with ONE VIEW A WINDOW loaded through registers and its 32 steps one block, and 9,409 with the half-window
  view through registers, against 6,828 for the block form, whose value runs land in place, and 5,667
  with its operands made. Clocked (8,932 and 9,280; the block form 6,817):
  THE MOVE IS THE FOLD'S FILL -- a fold warp is 2,473 expanding and moving its entries where the block form's is
  729 expanding them, 16 value lines a thread a window through the shared-memory pipe twice; and THROUGH REGISTERS
  THE FILL IS 5,380, two entries a thread of 9 lines each from global memory, the stream in the cache: a warp's
  loads of 16 bytes are a fifth of a bulk copy's bytes a cycle (KERNEL_STANDARDS 33). THE ONE-VIEW ROW'S STEPS AND RENDEZVOUS ARE SHORT, AND NOT BY ITS STRUCTURE (the joined row, below): its 32 steps
  issue in 1,029 where the block form's two slots' are 1,681, its one rendezvous is
  332 where two are 689 and 549, and its pads and operands
  875 against 753. A readout warp waits 3,026 and 4,455
  on the image against 1,794: the fold is the window in both. NOT ASKED: one view a window staged by
  small bulk copies; a rank counted from the mask, where these rows' places are the host's.
  THE JOINED ROW -- the block form's two slots laid before ONE RENDEZVOUS, its 32 steps one block -- DOES NOT MOVE THE WINDOW:
  6,795 against 6,817, both clocked. Its one rendezvous is 996 where the two are 689 and
  549, its steps' issue 1,490 against 1,681, with their wait 1,781 against
  2,143; its expansion is 926 against 729 and its pads and operands 842 against
  753. The row also issues both copies at the window's end, which the late-issue rows price apart; the two are
  not parted here. SO THE ONE-VIEW ROWS' SHORT ISSUE AND RENDEZVOUS ARE NOT THEIR STRUCTURE'S. READ AS, not isolated: their
  fold is alone at its steps while the readout waits on the image, where this row's steps run beside the readout's.
  THE MEETING ROWS -- the rendezvous an mbarrier's -- DO NOT MOVE IT: the window is 5,321 against
  5,267 with even lists and 6,878 against 6,828 in the block form; clocked, a fold warp
  is 427 and 532 at its rendezvous (that row's folds at 168 registers), and
  678 and 560 in the block form. The time is not the named barrier's own.
  WITH EVERY WAIT POLLED, not suspended (even lists): the window is 5,498 against 5,267;
  clocked (5,896), a fold warp's rendezvous are 385 and 523, its waits 295 on the readout's
  read and 350 on its bytes.
  A slot's steps are a
  fixed two rounds: ISSUING ONLY THE STEPS ITS LISTS HAVE, a second-round step a trip of one or none, WAS MEASURED
  AND LOST (the experiment at 758f0d43, its record that commit's: 7,022 against 6,822 composed, the fold's side
  alone 4,385 against 3,741), and its rows are not kept. NOT ASKED: the drain warps at 88 registers under this
  fold (they are at 72 here); lists past two rounds.
- THE MIRRORED READOUT'S SUMS TO THE PANEL ARE ITS LARGEST PHASE (the even lists' form, clocked): a readout warp is
  1,747 cycles a window rounding its sums, storing them a half each and arriving -- its
  2,018 less its 271 waiting on the drain -- against 1,243 in its steps and
  1,472 forming its coefficients. NOT ASKED: the panel laid so a thread's two sums are one word's store.
- THE OUTPUT PATH ON FOUR WARPS (the chunk-of-8 form, the fold issuing its own copies, four drain warps): the window
  is 5,267 cycles against 5,513 with two drain warps and a producer; clocked 5,722 against 5,618: a readout
  warp waits 271 on the drain against 427, a drain warp 3,105 on the panel against 635 -- and A FOLD WARP WAITS
  466 of the window, on the readout's read and on its bytes, against 2,120: the folds, which waited a third
  of a window, now wait a tenth; no clock says in what.
- WITH THE DRAIN WARPS ROUNDING the window is 5,299 against 5,267: nothing. Clocked (5,506), a readout
  warp's unclocked remainder is 1,771 against 2,393 and it waits 1,326 on the image against 344: what the
  readout sheds it waits out on the folds' image.
- THE MIRRORED READOUT WITH A CHUNK OF 8 DIGITS: the window is 5,513 cycles against 5,524 at 4. Clocked (5,618
  against 5,908): a readout warp is 1,032 forming and 131 in its steps against 1,092 and 152 -- half the
  passes, three quarters of the time -- and what its clocks do not name is 3,779 against 4,129: its
  rows' rounding and stores are a value's, not a pass's. It waits 427 on the drain against 234, and a drain
  warp waits 635 on the panel against 2,228: the two drain warps are near the window's whole length at
  work (their warpgroup at 88 registers here, 96 there; no row parts that from the two panels handed at once).
- WITH THE READOUT A SUM A BOX (the fold expanding, a producer issuing) THE WINDOW IS LONGER: 6,224 cycles against
  5,874. Clocked (6,461 against 6,428): a readout warp is 3,304 in its coefficients and steps against 3,826 --
  two chains of 4 steps are a tenth under one of 8, so the chain was not what its steps cost -- and it waits
  943 on the drain against 244, a drain warp waiting 319 on the panel against 2,408. The row's serve
  warps are at 80 registers where the other's are at 112, and its drain is what the window then waits on; no row
  parts the drain's registers from the readout's second sum. NOT ASKED: the readout's steps by what they
  multiply -- a step is 64 rows of which one digit's 16 are not zero; its coefficients formed by warps that
  wait; the fold-expands form on the uneven stream; a truncated step's pads left out of the expanding.
- THE TABLES ONCE-READ COST 2.4 TO 3.0 TIMES THE WINDOW ON THIS BOX (in cycles, which for a row bound by global
  memory follow the card's clock): 20,495 cycles composed against 6,935, the winner's form
  19,228 against 6,588, the fold's side alone 12,165 against 5,011 -- every access a fresh row of 128 MB of
  values, the model of a large table whose rows are each read once.
- THE DUMP IS PRICED IN THIS SESSION: the fold's state stored out in every call, as every record before 79ec1b08
  ran, puts the composed row at 7,923 against 6,935 -- one binary, one box, one session.
- FIVE WARPGROUPS LOSE COMPOSED, 7,599 against 6,935: 640 threads leave four producers and four drain warps 40
  registers a thread, the fold waits on its producers 5,450 cycles a window, and the lean readout the count
  forces is 3,843 in its steps and their forming where one commit group's are 3,535. Four drain warps do
  help the read side alone, 5,036 against 5,550.
- THE LEAN READOUT WITH TWO-BATCH PRODUCERS IS 6,766 COMPOSED against 6,935: a readout warp is 4,451 cycles
  a window in its steps and their forming. The two changes are not separated.
- THE READOUT DRAINING ITS OWN ROWS LOSES COMPOSED, 7,389 against 6,935: a readout warp is 2,805 cycles a
  window in its 64 adds, and the window becomes the readout's -- the fold waits on its read 3,752. A warp that
  issues an add is held until the SM passes it: adds issued by the readout hold the readout.
- COMPOSED, THE PRODUCERS' LOADS ARE NOW THE LARGEST SINGLE ACCESS: with them out the window is 6,421 against 6,935, and
  those loads are three fifths of the fold's side alone -- 2,412 cycles a window without them against 5,011, some
  162 cycles a batch of a producer's 9 load instructions; their coefficient arithmetic is 454 of it
  (4,557 without). The drains' adds out, the window is 6,647; the image's stores out, 6,588.
- A PARKED FOLD WARP COSTS THE READ SIDE -19 CYCLES, A PUBLISHING ONE 1,279: the read side alone is 5,550
  with the fold warps publishing and 4,272 with no fold warp running, while one parked asleep beside it costs
  -19 cycles and one spinning 174. What of the publishing fold's hand-off holds it is NOT ISOLATED
  (its stores are 561 of it; its fence nothing).
- THE READOUT'S OWN BYTE LOADS ARE A SIXTH OF ITS STEPS' SPAN: 8 instructions a chunk, each lane's 4 bytes of a
  different row; without them a readout warp's steps are 2,131 cycles a window against 2,592. The read side alone
  is then the drain's -- 5,003 against 5,550, the readout waiting on the drain 273 -- at some
  27 cycles an add a drain warp, where with no fold warp running an add is 26.
- ONE WIDE LOAD A WARP FOR THE READOUT'S BYTES is the read side alone without those loads -- 5,018 cycles a window
  against 5,003 with them out and 5,550 as they were -- and little composed: 6,851 against 6,935, where the
  byte loads out are 6,563 and the drains' adds out with them 6,375.
- THE READOUT FORMS ITS STEPS' A IN 1,089 TO 1,338 CYCLES A WINDOW, 272 to 335 a chunk: the coefficients' rounds
  and the 32 words held.
- THE STEPS THEMSELVES ARE OVER THEIR OWN TIME with nothing beside them but the drains: 546 cycles a chunk (the
  fence, 8 steps, the commit, 8 loads issued, the wait) where 8 steps of 64 columns into one accumulator from
  shared memory are 319 ([`wgmma_width`](wgmma_width.md)). Not isolated here.
- EVERY ROLE ISSUES AN INSTRUCTION EVERY 3.5 TO 7 CYCLES of its busy time (the static ledger's counts a trip: a
  producer 168 a batch, a drain warp 339 a chunk, the readout 284 a chunk, a fold warp 379 a window): no role is
  bound by the instructions it issues, and what holds each is not isolated.
- THE COUNTERS (the profiler's, each kernel's checked launch of 4 windows a CTA, which also stores the fold's state
  out -- 128 stores a thread, each over 32 lines, which the timed windows leave out; a CTA a window, its 16 warps
  together) OF THE BLOCK FORM, THE FORM WITH THE FOLD'S OPERANDS MADE, AND WITH BOTH SIDES': 13,123, 12,127 and
  10,886 warp-instructions -- over four schedulers 3,281, 3,032 and 2,722 cycles of issue, about half of each window;
  384 wgmma issues (96 steps, a warpgroup's four warps each); 4,972, 4,504 and 4,439 shared-memory wavefronts
  (1,908 loads' and 2,243 stores' in the block form, 598 of them bank conflicts) -- NOT PRICED on this
  architecture, which declares no rate: at the A100's one a cycle (`hardware.json`, sm_80) they would be 73% of the block
  form's window (6,828) and 86% of the both-sides form's (5,189); 256, 128 and 0
  conversion-unit instructions, the expansions' alone; 256 add requests, a token's line each; 88 KB read from the cache.
  STALLED, of a warp's waiting cycles: the block form 42% on the global-memory queue -- the checked launch's state stores
  among them -- 14% on global data and 11% on the shared-memory queue; with both sides' operands made 41% on the
  shared-memory queue, 29% on the global-memory queue and 10% on global data. What a warp asleep at a barrier's wait is
  counted as is NOT KNOWN.
  SO, BY COUNTS: issue some half of each window, the tensor pipe 96 steps of 20 cycles ([`wgmma_width`](wgmma_width.md)),
  the adds 256 lines of some 8.5 ([`add_rate`](add_rate.md)); THE SHARED-MEMORY PIPE, unpriced here, is the one the
  counts put nearest the whole window, and a wavefront's price on this card is what decides it.
Record: `measure/lab/results/carry_window/sm_90a.json`.
