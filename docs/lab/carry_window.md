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

THE CUT ROWS (the block form and the read-ready ready pair's, no clock inside a role) ask what each of the phase
ledger's three removable shared-memory costs is worth to the window, alone and together: THE PANEL BY STMATRIX -- the
readout's sums stored as a token's 8 channels a 16-byte row, `stmatrix.trans`, where the form stores 16-bit halves
(512 wavefronts a window against some 128); THE STEPS' OPERANDS SHUFFLED -- a slot's 16 steps' lines loaded once, lane
s step s's, and taken at each step by four shuffles, where every lane of the form loads every step's line (512); ONLY
THE STEPS A LIST REACHES -- a second-round step issued in a one-trip loop where its list reaches it, where the form
issues it on a tile of zeros (some 640 of the operands' reads). Checked as the block form's rows are; the phase ledger
reads each form with and without all three.

THE AHEAD ROWS (the block form and the read-ready ready pair's, both with the panel by stmatrix, no clock inside a
role) ask whether the fold's steps wait on their operand fields' loads, as the timed launch's ledger reads: a step's
line is loaded two steps ahead of its issue, two lines held in turn, where the form loads it at the step. Checked as
the block form's rows are.

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

MEASURED (H100 80GB HBM3, 132 SMs, Nebius preemptible VM zw5c2248, driver 580.173.02; 2026-11-15, the experiment at
cbdb81f2, one session; 7 repetitions, every checked row's state and output values exact;
the median; SM-cycles a token are the window's cycles over 8 -- a CTA of two boxes is 8 tokens a window. THE FOLD'S STATE
DUMP STAYS OUT OF THE TIMED CALLS, and one row restores it to price it in this session):

| the row | cycles a window a CTA | SM-cycles a token |
|---|---|---|
| the fold's side alone: producers and steps | 5,114 | 639 |
| the read side alone: the image, the readout and the drain | 5,564 | 695 |
| the carry: both sides | 7,047 | 881 |
| both sides, the producers' global loads taken out | 6,431 | 804 |
| both sides, the drains' global adds taken out | 6,721 | 840 |
| both sides, the image's stores taken out | 6,714 | 839 |
| the read side alone, no fold warp running | 4,416 | 552 |
| the read side alone, every wait suspended | 5,271 | 659 |
| both sides, every wait suspended | 6,921 | 865 |
| THE READOUT DRAINING ITS OWN ROWS, FOUR PRODUCERS: the fold's side alone | 3,094 | 387 |
| the same: the read side alone | 5,370 | 671 |
| the same: both sides | 7,441 | 930 |
| THE READOUT LEAN, TWO BATCHES A PRODUCER: the fold's side alone | 4,576 | 572 |
| the same: the read side alone | 5,771 | 721 |
| the same: both sides | 6,794 | 849 |
| FIVE WARPGROUPS, FOUR PRODUCERS AND FOUR DRAIN WARPS AT 40: the fold's side alone | 5,915 | 739 |
| the same: the read side alone | 5,047 | 631 |
| the same: both sides | 7,766 | 971 |
| THE LONG WAITS ASLEEP AT NAMED BARRIERS: the fold's side alone | 5,156 | 645 |
| the same: the read side alone | 5,065 | 633 |
| the same: both sides | 6,992 | 874 |
| the fold's side alone, the producers' global loads taken out | 2,411 | 301 |
| the fold's side alone, their coefficient arithmetic taken out | 4,722 | 590 |
| the fold's side alone, both taken out | 2,281 | 285 |
| the read side alone, the readout's byte loads taken out | 5,003 | 625 |
| the read side alone, no fold warp running, the byte loads taken out | 4,017 | 502 |
| both sides, the readout's byte loads taken out | 6,680 | 835 |
| both sides, the drains' adds and the readout's byte loads taken out | 6,423 | 803 |
| THE READOUT'S BYTES ONE WIDE LOAD A WARP: the read side alone | 5,053 | 632 |
| the same: the read side alone, no fold warp running | 4,243 | 530 |
| the same: both sides | 6,989 | 874 |
| the same: both sides, the long waits asleep | 7,155 | 894 |
| the read side alone, the image's stores taken out | 4,994 | 624 |
| the read side alone, the fold's fence behind its image taken out | 5,508 | 689 |
| the read side alone, both taken out | 4,980 | 623 |
| both sides, that fence taken out | 6,943 | 868 |
| the read side alone, no fold warp running, the fold warps parked asleep | 4,388 | 549 |
| the same, spinning | 4,521 | 565 |
| THE DUMP RESTORED: both sides, the fold's state stored out in every call | 8,121 | 1,015 |
| THE WINNERS COMBINED: both sides, suspended waits and the lean readout | 6,672 | 834 |
| the same, suspended waits and the wide byte load | 6,921 | 865 |
| the same, suspended waits, the lean readout and the wide byte load (the fourth warpgroup at 104) | 6,759 | 845 |
| the same, sleeping waits, the lean readout and the wide byte load | 7,031 | 879 |
| THE TABLES ONCE-READ, 16 replicas the cache cannot hold: both sides | 17,586 | 2,198 |
| the same, the winner's form | 16,171 | 2,021 |
| the same: the fold's side alone | 11,904 | 1,488 |
| THE DEN, folded, published, read and drained: the fold's side alone | 5,105 | 638 |
| the same: both sides, the winner's form | 7,075 | 884 |
| the same, the tables once-read | 16,481 | 2,060 |
| THE IMAGE'S TILES MOVED APART, nothing else changed: the winner's form at 154 lines (137 as it is) | 7,071 | 884 |
| the winner's form with the den at 161 lines (155 as it is) | 7,067 | 883 |
| THE ALIGNED CELL, an image tile a list: the winner's form | 6,669 | 834 |
| the same with the den | 7,064 | 883 |
| NO CLOCK INSIDE A ROLE: the winner's form | 6,590 | 824 |
| the same with the den | 6,936 | 867 |
| the same at the aligned cell | 6,681 | 835 |
| the same at the aligned cell with the den | 7,078 | 885 |
| THE DEN'S PARTS, a chain with no clock inside a role: the fold's steps | 6,654 | 832 |
| then the image's ninth brick and its store | 6,749 | 844 |
| then the register split | 6,836 | 854 |
| then the 72-column steps | 6,896 | 862 |
| the winner's form with its tiles at the den's pitch, 155 lines | 6,598 | 825 |
| THE VALUES TILE-MAJOR in global memory, no clock inside a role: the winner's form | 12,196 | 1,524 |
| the tables once-read, the values token-major | 16,290 | 2,036 |
| the tables once-read, the values tile-major | 48,621 | 6,078 |
| THE FOLD'S ENTRIES FROM LISTS, a window's own tokens: the fold's side alone | 8,741 | 1,093 |
| the same: both sides, no clock inside a role | 9,982 | 1,248 |
| the same with every window the same tokens: the fold's side alone | 6,245 | 781 |
| the same: both sides | 8,112 | 1,014 |
| a window's own tokens, their rows asked for a window ahead: both sides | 11,909 | 1,489 |
| THE VALUES LANDED BY BULK COPY from expanded lists, a window's own tokens: the fold's side alone | 8,462 | 1,058 |
| the same: both sides | 9,065 | 1,133 |
| THE WHOLE ENTRY LANDED, values and coefficient bytes: the fold's side alone | 5,164 | 645 |
| the same: both sides | 5,993 | 749 |
| the whole entry, EACH BOX ITS OWN expanded windows, past the cache: the fold's side alone | 6,479 | 810 |
| the same: both sides | 11,124 | 1,390 |
| the whole entry, its 16 columns' windows NEVER READ BEFORE, 16 boxes a window's readers: the fold's side alone | 5,172 | 646 |
| the same: both sides | 5,976 | 747 |
| THE PACKED STREAM, packed by digit, its lists uneven (8 to 24 entries), last steps truncated: the fold's side alone | 10,074 | 1,259 |
| the same: both sides | 10,583 | 1,323 |
| the packed stream, FOUR SLOTS OF 4 TILES a box: the fold's side alone | 10,707 | 1,338 |
| the same: both sides | 11,483 | 1,435 |
| the whole entry, THE FOLD WARPS EXPANDING their own slot: the fold's side alone | 3,038 | 380 |
| the same: both sides | 5,992 | 749 |

THE WALL TIME, measured on this box and immune to its clock (1.78 to 1.82 GHz, its rows' medians): the long call
scaled to 65,536 tokens is 0.249 ms for the baseline, 0.238 ms for the winner's form, and 0.505 ms at the
once-read bound (0.572 without the winner's changes). THE FLASH REFERENCES, from the session of instance 60450651 ON ANOTHER BOX (a different offer and driver)
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
| a fold warp waits on the readout's read of a tile group | 378 | 4,183 | 3,802 |  |  | 3,772 | 407 |
| a fold warp waits on its producer | 4,687 |  |  |  |  | 433 | 5,615 |
| a readout warp waits on a tile group's publish | 1,858 | 291 | 323 |  |  | 300 | 2,216 |
| a readout warp waits on the panel's drain | 326 | 287 | 273 | 245 | 282 |  | 328 |
| a readout warp draining its own rows |  |  |  |  |  | 2,840 |  |
| a readout warp forming its steps' A | 1,343 | 1,105 | 1,110 | 1,296 | 1,007 | 1,103 |  |
| a readout warp in its steps (a lean readout's hold the forming) | 2,197 | 2,581 | 2,131 | 2,207 | 2,049 | 1,953 | 3,863 |
| a drain warp waits on the panel | 3,214 | 2,033 | 1,521 | 870 | 701 |  |  |

WHAT IT SHOWS.
- THE CARRY COMPOSES, EXACTLY: two boxes' folds, the image a window, one readout warpgroup over both -- its
  coefficients expanded in its registers, its steps' A operand -- and one f16 line a token, in one CTA of 512
  threads at the SM's whole register file.
- A TOKEN IS PRICED AT 2,021 SM-CYCLES COMPOSED, its tables read once (R22.2), in the winner's form; 834 when the
  cache holds them (881 in the first form) at this cell (its sides sparse on opposite levels, 256 x 256 leaves;
  [`carry_floor`](../internals/tools/carry_floor.md) prices a dense side higher).
- THE FILL WANTS WARPS THAT EACH HOLD A BATCH: two producers a box, a batch of 32 entries in registers each, take
  the fold's side alone to 3,094 cycles a window where one a box is 5,114; one a box holding TWO batches is
  4,576; and two a box at 40 registers a thread, each holding one value instruction's four entries, are
  5,915 -- slower than one a box that holds a batch.
- SUSPENDED WAITS WITH THE LEAN READOUT ARE THE FASTEST COMPOSED FORM: 6,672 cycles a window against the
  baseline's 7,047, with the suspended waits beside the wide byte load a near tie at 6,921. The parts: every wait
  suspended alone 6,921, the long waits asleep at named barriers 6,992, the lean readout alone 3.6%
  under the baseline, the wide load alone 6,989. ALL THREE AT ONCE LOSE the pairing's gain (6,759: the lean
  readout beside the wide load's scratch needs 88 registers a thread and the fourth warpgroup drops to 104), and
  the sleeping triple is 7,031.
- THE DEN COSTS 5.3% OF THE COMPOSED WINDOW AT THIS CELL AND 5.9% AT THE ALIGNED ONE, the tables cached and no
  clock inside a role: 6,936 cycles against 6,590, and 7,078 against 6,681. THE ROLES' CLOCKS ARE NOT FREE AND NOT EQUAL: they are
  82 cycles of the winner's clocked window and 139 of the den's, and which is larger has changed between
  records, so a clocked row does not price the den (clocked: 7,075 against 6,672; why the clocks differ is
  not isolated); with the tables once-read 16,481 against 16,171; the fold's side alone 5,105 against 5,114. One
  cell (its sides sparse on opposite levels); a dense write side, where the fold's side binds, is unmeasured. ITS
  PARTS, each a row less the row before, no clock inside a role: the fold's four steps a window 64 cycles;
  the image's ninth brick and its store 95 (the pitch itself nothing: the winner's form at 155 lines is 6,598
  against 6,590); the register split, readout 88 and fourth warpgroup 104, 87; the steps 72 columns wide 60;
  the panel's word and the drain's add 40. No part is a third of it. The split cannot be avoided in this
  form: the 72-column readout serializes its steps at 80 registers with no clock read too.
- THE FOLD'S FORM WAS A CONTEST. A step a list with ones in one row of A is 16 steps a window for 1 row in 64 and
  a 17th accumulator set of 8, which the toolkit will not hold at 160 registers: at a 168-register fold with
  one-batch producers it cost an eighth of the composed window in its session (the experiment at ac342446, whose
  record this one replaced). The coefficients as A are four steps into 4 registers.
- AN IMAGE TILE'S PITCH MUST BE AN ODD COUNT OF LINES: a warp's lanes store to four tiles two apart, and at an
  even pitch two of the four land on the same banks. The winner's form at 154 lines is 7,071 against 6,672 at 137;
  the den's at 161 is 7,067 against 7,075 at 155.
- THE IMAGE FORM COSTS 1.4% MORE WHERE THE CARVES ARE EQUAL, no clock inside a role (6,681 cycles a window against
  6,590); clocked, the aligned cell's winner's form is 6,669 cycles a
  window against 6,672 at the cell whose sides list opposite levels -- 16 word stores a thread a tile group where
  that cell's are 4 lines -- and with the den 7,064 against 7,075, the den's lines there 2-byte
  stores of 8 threads. A form that reads the fold's own registers has nothing of the window to win back but the
  image's stores; it is not built.
- THE CARRY'S PRODUCERS WANT A TOKEN'S VALUES AS ONE RUN: with the values tile-major in global memory -- a token's 8
  lines a kilobyte apart -- the winner's form is 12,196 cycles a window against 6,590, and with the tables once-read
  48,621 against 16,290. The intra's window moves 5% with the values' order ([`intra_window`](intra_window.md)), so
  the values stay token-major and the levels, which the carry reads 16 bytes at a time, go tile-major.
- FROM LISTS, WITH A WINDOW'S OWN TOKENS, THE FOLD'S SIDE ALONE IS 8,741 CYCLES A WINDOW where the synthetic entries'
  is 5,114 cached and 11,904 once-read; both sides 9,982 (the readout's side still on synthetic, cached readers) against
  6,590. Two things are in it, and the rows with every window the same tokens part them: THE LISTS' CODE -- a
  batch's places loaded a batch ahead, a shuffle a value instruction, the sparse word, tile-major addresses, and one
  batch held where the winner holds two -- is 6,245 against 5,114 alone and 8,112 against 6,590 composed; and ROWS THE
  CACHE HAS NOT SEEN, a window's own being new each window as a call's are, are the rest, 2,496 alone and
  1,870 composed. A window's rows are loaded one batch ahead of their use. ASKING FOR THEM A WINDOW AHEAD BY
  PREFETCH LOSES: 11,909 composed against 9,982 -- six prefetches an entry-batch are more requests of the same
  memory, not less waiting. Whether loads issued two batches ahead hide the rows' arrival is not measured (the
  producers' registers hold one batch from lists). The check of these rows is weak: every table's byte is its token's parity.
- LANDING THE VALUES ALONE FROM EXPANDED LISTS is 9,065 cycles a window composed against 9,982 with every entry's
  values gathered (8,112 for the gather with rows the cache holds), and the fold's side alone 8,462 against 8,741: a
  list's values are one bulk copy of 2 KB from a buffer laid contiguous a window (the host's here; 16 sources
  share a buffer), new each window, and its producers still gather an entry's 16 dense bytes and its sparse byte,
  a line of each level's rows an entry. What that leaves is the next finding's.
- WITH THE WHOLE ENTRY LANDED -- its coefficient bytes beside its values, raw, a list one bulk copy and no global load
  in a producer's trip -- the window is 5,993 cycles composed and the fold's side alone 5,164, against 6,590 and
  5,114 for the synthetic entries whose rows the cache holds.
- WHAT THAT ROW NEEDS IS ITS BYTES SHARED, NOT HELD BEFOREHAND: with a window's bytes never read before and its
  16 boxes its readers, the window is 5,976 cycles composed against 5,993 with the cache holding them, and
  the fold's side alone 5,172 against 5,164: one box's miss serves its 15 mates. With EACH BOX ITS OWN it is
  11,124 composed, 1.86 times, and 6,479 alone, 1.25 times -- 16 times the bytes from global memory, on
  every multiprocessor at once.
- THE PACKED STREAM IS EXACT, AND ITS UNEVEN LISTS COST THE FOLD 1.95 TIMES: 10,074 cycles a window the
  fold's side alone against 5,164, 10,583 composed against 5,993. The same 256 entries a box a window, as
  lists of 8 to 24 where the row above has 16 each, are 24 steps where it has 16 -- three slots of 8 tiles where it
  has two -- and a truncated step, used where it landed with its bricks as far apart as it has entries, folds
  exactly. In four slots of 4 tiles, a slot's copies issued two slots ahead, the same stream is 10,707 alone
  and 11,483 composed: more slots cost more, so a slot's price is not its bytes' landing.
- THE FOLD'S SIDE WAITS ON ITS PRODUCER. With even lists a fold warp waits 4,534 of the fold's side's 5,164 cycles a
  window on its producer: its own steps are what is left. One producer warp a box expands every entry. On the
  packed stream in four slots the wait is 4,213 of 10,707, and with TWO PRODUCERS A BOX 1,486 of 8,471: the wait
  falls and the fold's own time is then the larger part. That fold runs a list's steps as a loop of its own, a
  vote a list a slot, where the even rows' steps are one block; the block rows below run the uneven lists' steps so.
- WITH THE FOLD WARPS EXPANDING THEIR OWN SLOT (even lists; a producer only issues copies) the fold's side alone
  is 3,038 cycles a window against 5,164, a fold warp's wait -- now on its slot's bytes -- 1,909 against 4,534.
  THE COMPOSED WINDOW DOES NOT MOVE: 5,992 against 5,993; with the fold issuing its own copies too, no
  producer running, 5,888.
- THE WINDOW IS THE READOUT'S. That form with every role's waits clocked (6,331 cycles a window): a readout warp
  waits 286 on the image and 278 on the drain and is 4,274 in its coefficients and steps; a fold
  warp waits 1,818 on the readout's read and 178 on its bytes; a drain warp waits 2,097 on the
  panel. The one role that does not wait is the readout's warpgroup.
- FOUR DRAIN WARPS BUY NOTHING: 5,839 against 5,888, each drain warp then waiting 4,028 of a window on the
  panel, and a readout warp is 3,826 in its coefficients and steps against 4,274. The drain is not the
  window's bound.
- THE MIRRORED READOUT (coefficients in shared memory, tokens on the step's columns; the fold expanding, a producer
  issuing): the window is 5,599 cycles against 5,992. Clocked (6,038 against 6,520): a readout warp is
  1,081 forming its coefficients and 156 in its steps, against 3,842 for both in the register form; it
  waits 368 on the image and 238 on the drain; a fold warp waits 2,704 on the readout's read against 2,811;
  a drain warp 2,410 on the panel against 2,536.
  What its clocks do not name is 4,194 cycles a window: its rows' rounding, their stores and its arrivals. Its 32
  steps are a quarter the register form's multiply-adds and take 156: a step's price here is not its size.
- THE BLOCK ROWS -- uneven lists, a block a list, the stream saying what to do: the window is 6,816 cycles
  composed (6,837 clocked), against 5,294 for even lists under the same readout and drains (whole-entry
  tiles, the fold issuing) -- 1.29 times it -- and
  10,583 for the uneven lists' truncated steps behind a producer and a loop a list.
  The fold's side alone is 3,744,
  a fold warp waiting 265 of it on its bytes, against 3,038 for the even lists' fold expanding its own slot. Clocked: a fold warp waits 250 on the
  readout's read and 189 on its bytes; a readout warp 1,792 on the image (335 with even lists) and 252 on the drain; a drain warp
  4,337 on the panel.
  THE FOLD'S PHASES, clocked, the block form against the even lists' (a clocked fold holds nine more
  registers; the windows are 6,837 and 5,794 with them): publishing the image 1,309 against 1,954,
  its wait on the readout's read in both (250 and 263); expanding its entries' lines 723 against 782; its pads and its
  steps' operands 740, the block form's alone; its slots' fences 55 and 71 against 69
  and 56; AT ITS SLOTS' RENDEZVOUS -- between a reading before the warpgroup's sync and one behind it --
  709 and 559 against 355 and 481; its steps and its next copy's issue 2,140 against 1,686.
  A FENCE IS SOME 60 CYCLES, AND A FOLD WARP IS LONGER AT A SLOT'S RENDEZVOUS THAN EXPANDING THE SLOT. With no
  arithmetic in the fold (the control: windows 6,608 against 6,816 and 5,230 against
  5,294; its entries' lines 476 and 347) its rendezvous are 777
  and 573, and 319 and 601: they do not follow the lines' work. THE
  FOLD'S ARITHMETIC IS NOT WHAT THE WINDOW WAITS ON.
  WITH BOTH COPIES ISSUED AT THE WINDOW'S END (even lists) the window is 5,491 against 5,294,
  its fences 65 and 59: a fence does not wait on a copy in flight, and the later
  copies cost the window.
  ITS RENDEZVOUS MOVE WITH THE ISSUE: 821 at its first slot and 155 at its second, where the form
  that issues a slot's copies as it leaves the slot is 355 and 481 -- the rendezvous behind an issue
  is the long one.
  A WARPGROUP WAITS AT ITS RENDEZVOUS FOR ITS FIRST WARP, whose lanes issue the slot's 8 copies: the first warp
  alone is 93 and 93 at its two rendezvous where the warps' mean is 355 and 481, and
  2,638 in its steps and its issue where the mean is 1,686.
  WITH THE COPIES ISSUED TWO A WARP every warp is 220 and 224 at its rendezvous and 1,703 in its
  steps and issue; the window is 5,248 against 5,294, and a fold warp then waits 836 on the
  readout's read against 263: THE FOLD'S TIME SAVED IS ITS WAIT ON THE READOUT, WHICH IS THE WINDOW.
  THE BLOCK FORM'S FIRST WARP IS LAST TOO: 103 and 96 at its rendezvous against 709 and
  559; its plans, its one copy a slot, 16 threads' operands and the first pads are its own -- 1,155 at
  its pads and operands against 740, 826 expanding against 723, 2,544 in its steps and
  issue against 2,140. WITH THOSE JOBS SHARED AMONG ITS WARPS THE WINDOW DOES NOT MOVE: 6,839 against
  6,816. Its rendezvous fall to 490 and 297 and every warp's pads and operands rise to
  1,224: the phase is some 53 instructions a thread, and a warp is as long in it with a pad and four operands
  as the first warp was with all of them. NOT ISOLATED: what a warp waits on in that phase.
  THE READY PAIR -- the block form's slots landed with their operands made, the fold only landing and stepping:
  the window is 5,681 against 6,816 for the block form and 5,294 for even
  lists; clocked (5,789), a fold warp waits 219 on its bytes and
  850 on the readout's read, is 2,209 issuing its 32 steps and
  848 in their wait and its copy's issue, and A READOUT WARP WAITS 289 ON THE
  IMAGE against 1,792 with the fold expanding and 335 with even lists: with its operands made the
  uneven lists' fold is no longer what the window waits on. What the window is over the even lists' by, no row
  parts: the fold issues twice the steps beside the same readout.
  THE READ-READY ROWS -- the readout's coefficient tiles landed made as well: the window is 5,336 against
  5,294 with even lists, and 5,191 against 5,681 for the ready pair's form. Clocked
  (5,377 and 5,298), a readout warp is 979 and 401 from its
  image's wait to its tiles landed, where forming them itself with its rendezvous is 1,494 and
  1,471; it waits 868 and 1,094 on the image, and a fold warp
  430 and 296 on the readout's read.
  THE OWN ROWS -- every box its own stream, where 16 share a column's -- ARE THREE TIMES THE WINDOW: 13,196
  against 5,191. Their streams are some 680 MB, past the card's cache, where the form's are under it; a
  window's bytes a CTA are the same. Clocked (13,118): A READOUT WARP WAITS 9,508 ON ITS
  TILES' BYTES against 401 -- its copy is issued one panel's stores ahead of its use -- and
  a fold warp, whose copies are issued a window ahead, 781 on its own against 211; the
  fold then waits 7,686 on the readout's read. WITH ITS OPERANDS FROM THE CARD'S MEMORY THE CARRY IS
  THAT MEMORY'S: the ready stream is a window's 110 KB a CTA, and past the cache it is this price. NOT ASKED: the
  readout's copy issued further ahead; the design's mix, a box's own coefficients and its column's values.
  THE TOKEN ROWS -- the fold laying its own operands from a view in token order -- ARE A QUARTER OVER THE BLOCK FORM, HOWEVER THE VIEW
  ARRIVES: the window is 8,488 with a half-window view staged by one copy and its value lines moved in shared memory,
  8,596 with ONE VIEW A WINDOW loaded through registers and its 32 steps one block, and 9,557 with the half-window
  view through registers, against 6,816 for the block form, whose value runs land in place, and 5,681
  with its operands made. Clocked (8,953 and 9,444; the block form 6,837):
  THE MOVE IS THE FOLD'S FILL -- a fold warp is 2,399 expanding and moving its entries where the block form's is
  723 expanding them, 16 value lines a thread a window through the shared-memory pipe twice; and THROUGH REGISTERS
  THE FILL IS 5,674, two entries a thread of 9 lines each from global memory, the stream in the cache: a warp's
  loads of 16 bytes are a fifth of a bulk copy's bytes a cycle (KERNEL_STANDARDS 33). THE ONE-VIEW ROW'S STEPS AND RENDEZVOUS ARE SHORT, AND NOT BY ITS STRUCTURE (the joined row, below): its 32 steps
  issue in 1,007 where the block form's two slots' are 1,676, its one rendezvous is
  344 where two are 709 and 559, and its pads and operands
  812 against 740. A readout warp waits 3,068 and 4,653
  on the image against 1,792: the fold is the window in both. NOT ASKED: one view a window staged by
  small bulk copies; a rank counted from the mask, where these rows' places are the host's.
  THE JOINED ROW -- the block form's two slots laid before ONE RENDEZVOUS, its 32 steps one block -- DOES NOT MOVE THE WINDOW:
  6,816 against 6,837, both clocked. Its one rendezvous is 1,025 where the two are 709 and
  559, its steps' issue 1,479 against 1,676, with their wait 1,773 against
  2,140; its expansion is 917 against 723 and its pads and operands 824 against
  740. The row also issues both copies at the window's end, which the late-issue rows price apart; the two are
  not parted here. SO THE ONE-VIEW ROWS' SHORT ISSUE AND RENDEZVOUS ARE NOT THEIR STRUCTURE'S. READ AS, not isolated: their
  fold is alone at its steps while the readout waits on the image, where this row's steps run beside the readout's.
  THE CUT ROWS: OF THE PHASE LEDGER'S THREE SHARED-MEMORY COSTS ONLY THE PANEL'S CUT IS WORTH TIME. The block form's
  window is 6,485 with the panel by stmatrix against 6,816, 7,354 with the steps' operands shuffled,
  7,130 issuing only the steps a list reaches, and 7,694 with all three; the read-ready ready pair's
  4,944, 5,828, 5,501 and 5,984 against 5,191. All three cut the
  windows' shared-memory wavefronts by some 1,020 a CTA-window (the phase ledger: 4,238 to 3,216 and 3,758 to 2,735),
  and two of them lengthen the window: A WINDOW WITH A QUARTER LESS SHARED-MEMORY TRAFFIC IS NOT SHORTER, SO THAT PIPE
  IS NOT WHAT HOLDS IT. The shuffles put four instructions before each step's issue and the skip a branch around it,
  both on the fold's steps' issue; the panel's stores are the readout's, beside its steps. READ AS, not isolated: the
  window is held by the fold's and the readout's chain through the image, its waits and issue, not by a pipe's bytes.
  THE AHEAD ROWS -- the fold's steps' operand lines loaded two steps ahead -- DO NOT MOVE THE WINDOW: 6,491
  against 6,485 for the block form with the panel by stmatrix, and 4,947 against 4,944 for the
  read-ready ready pair's. THE FOLD'S STEPS DO NOT WAIT ON THEIR OPERAND LINES' LOADS -- the timed launch's short
  scoreboard at them is not that -- though the fold is the read-ready ready pair's path (`lab path`: of its clocked
  row's roles a fold warp waits least, and its steps' issue is its largest phase). What holds that issue is NOT
  ISOLATED; no clocked ahead row is run.
  THE MEETING ROWS -- the rendezvous an mbarrier's -- DO NOT MOVE IT: the window is 5,361 against
  5,294 with even lists and 6,881 against 6,816 in the block form; clocked, a fold warp
  is 431 and 527 at its rendezvous (that row's folds at 168 registers), and
  693 and 567 in the block form. The time is not the named barrier's own.
  WITH EVERY WAIT POLLED, not suspended (even lists): the window is 5,524 against 5,294;
  clocked (6,002), a fold warp's rendezvous are 391 and 500, its waits 294 on the readout's
  read and 353 on its bytes.
  A slot's steps are a
  fixed two rounds: ISSUING ONLY THE STEPS ITS LISTS HAVE, a second-round step a trip of one or none, WAS MEASURED
  AND LOST (the experiment at 758f0d43, its record that commit's: 7,022 against 6,822 composed, the fold's side
  alone 4,385 against 3,741), and its rows are not kept. NOT ASKED: the drain warps at 88 registers under this
  fold (they are at 72 here); lists past two rounds.
- THE MIRRORED READOUT'S SUMS TO THE PANEL ARE ITS LARGEST PHASE (the even lists' form, clocked): a readout warp is
  1,797 cycles a window rounding its sums, storing them a half each and arriving -- its
  2,068 less its 270 waiting on the drain -- against 1,240 in its steps and
  1,494 forming its coefficients. NOT ASKED: the panel laid so a thread's two sums are one word's store.
- THE OUTPUT PATH ON FOUR WARPS (the chunk-of-8 form, the fold issuing its own copies, four drain warps): the window
  is 5,294 cycles against 5,550 with two drain warps and a producer; clocked 5,794 against 5,645: a readout
  warp waits 270 on the drain against 431, a drain warp 3,145 on the panel against 713 -- and A FOLD WARP WAITS
  466 of the window, on the readout's read and on its bytes, against 2,131: the folds, which waited a third
  of a window, now wait a tenth; no clock says in what.
- WITH THE DRAIN WARPS ROUNDING the window is 5,310 against 5,294: nothing. Clocked (5,520), a readout
  warp's unclocked remainder is 1,790 against 2,454 and it waits 1,318 on the image against 335: what the
  readout sheds it waits out on the folds' image.
- THE MIRRORED READOUT WITH A CHUNK OF 8 DIGITS: the window is 5,550 cycles against 5,599 at 4. Clocked (5,645
  against 6,038): a readout warp is 1,037 forming and 126 in its steps against 1,081 and 156 -- half the
  passes, three quarters of the time -- and what its clocks do not name is 3,794 against 4,194: its
  rows' rounding and stores are a value's, not a pass's. It waits 431 on the drain against 238, and a drain
  warp waits 713 on the panel against 2,410: the two drain warps are near the window's whole length at
  work (their warpgroup at 88 registers here, 96 there; no row parts that from the two panels handed at once).
- WITH THE READOUT A SUM A BOX (the fold expanding, a producer issuing) THE WINDOW IS LONGER: 6,263 cycles against
  5,992. Clocked (6,570 against 6,520): a readout warp is 3,472 in its coefficients and steps against 3,842 --
  two chains of 4 steps are a tenth under one of 8, so the chain was not what its steps cost -- and it waits
  792 on the drain against 244, a drain warp waiting 337 on the panel against 2,536. The row's serve
  warps are at 80 registers where the other's are at 112, and its drain is what the window then waits on; no row
  parts the drain's registers from the readout's second sum. NOT ASKED: the readout's steps by what they
  multiply -- a step is 64 rows of which one digit's 16 are not zero; its coefficients formed by warps that
  wait; the fold-expands form on the uneven stream; a truncated step's pads left out of the expanding.
- THE TABLES ONCE-READ COST 2.3 TO 2.5 TIMES THE WINDOW ON THIS BOX (in cycles, which for a row bound by global
  memory follow the card's clock): 17,586 cycles composed against 7,047, the winner's form
  16,171 against 6,672, the fold's side alone 11,904 against 5,114 -- every access a fresh row of 128 MB of
  values, the model of a large table whose rows are each read once.
- THE DUMP IS PRICED IN THIS SESSION: the fold's state stored out in every call, as every record before 79ec1b08
  ran, puts the composed row at 8,121 against 7,047 -- one binary, one box, one session.
- FIVE WARPGROUPS LOSE COMPOSED, 7,766 against 7,047: 640 threads leave four producers and four drain warps 40
  registers a thread, the fold waits on its producers 5,615 cycles a window, and the lean readout the count
  forces is 3,863 in its steps and their forming where one commit group's are 3,540. Four drain warps do
  help the read side alone, 5,047 against 5,564.
- THE LEAN READOUT WITH TWO-BATCH PRODUCERS IS 6,794 COMPOSED against 7,047: a readout warp is 4,469 cycles
  a window in its steps and their forming. The two changes are not separated.
- THE READOUT DRAINING ITS OWN ROWS LOSES COMPOSED, 7,441 against 7,047: a readout warp is 2,840 cycles a
  window in its 64 adds, and the window becomes the readout's -- the fold waits on its read 3,772. A warp that
  issues an add is held until the SM passes it: adds issued by the readout hold the readout.
- COMPOSED, THE PRODUCERS' LOADS ARE NOW THE LARGEST SINGLE ACCESS: with them out the window is 6,431 against 7,047, and
  those loads are three fifths of the fold's side alone -- 2,411 cycles a window without them against 5,114, some
  169 cycles a batch of a producer's 9 load instructions; their coefficient arithmetic is 392 of it
  (4,722 without). The drains' adds out, the window is 6,721; the image's stores out, 6,714.
- A PARKED FOLD WARP COSTS THE READ SIDE -27 CYCLES, A PUBLISHING ONE 1,148: the read side alone is 5,564
  with the fold warps publishing and 4,416 with no fold warp running, while one parked asleep beside it costs
  -27 cycles and one spinning 105. What of the publishing fold's hand-off holds it is NOT ISOLATED
  (its stores are 570 of it; its fence nothing).
- THE READOUT'S OWN BYTE LOADS ARE A SIXTH OF ITS STEPS' SPAN: 8 instructions a chunk, each lane's 4 bytes of a
  different row; without them a readout warp's steps are 2,131 cycles a window against 2,581. The read side alone
  is then the drain's -- 5,003 against 5,564, the readout waiting on the drain 273 -- at some
  27 cycles an add a drain warp, where with no fold warp running an add is 26.
- ONE WIDE LOAD A WARP FOR THE READOUT'S BYTES is the read side alone without those loads -- 5,053 cycles a window
  against 5,003 with them out and 5,564 as they were -- and little composed: 6,989 against 7,047, where the
  byte loads out are 6,680 and the drains' adds out with them 6,423.
- THE READOUT FORMS ITS STEPS' A IN 1,105 TO 1,343 CYCLES A WINDOW, 276 to 336 a chunk: the coefficients' rounds
  and the 32 words held.
- THE STEPS THEMSELVES ARE OVER THEIR OWN TIME with nothing beside them but the drains: 552 cycles a chunk (the
  fence, 8 steps, the commit, 8 loads issued, the wait) where 8 steps of 64 columns into one accumulator from
  shared memory are 319 ([`wgmma_width`](wgmma_width.md)). Not isolated here.
- EVERY ROLE ISSUES AN INSTRUCTION EVERY 3.5 TO 7 CYCLES of its busy time (the static ledger's counts a trip: a
  producer 168 a batch, a drain warp 339 a chunk, the readout 284 a chunk, a fold warp 379 a window): no role is
  bound by the instructions it issues, and what holds each is not isolated.
- THE COUNTERS: this record's run read none; the phase ledger below holds the windows' counts.
Record: `measure/lab/results/carry_window/sm_90a.json`; the phase ledger's dumps:
`measure/lab/results/carry_window/ledger_sm_90a/ledger.json` and the dumps beside it.

THE PHASE LEDGER (H100 SXM on Nebius, the pack at a8a55a64, `lab ledger-dumps` there and `lab ledger --packed --dump
--per 528` here: each kernel's checked launch, 4 windows on 132 CTAs, profiled by ncu's SourceCounters; a CTA-window).
SHARED-MEMORY WAVEFRONTS: the block form 4,238 and with the three cuts 3,216; both sides' operands made 3,758 and with
the cuts 2,735 -- the bulk copies' writes not among them. WGMMA'S OPERAND READS ARE COUNTED AND ARE HALF OF IT: 1,920
in each uncut form, 20 a step (its 2 KB of A and 512 B of B at 128 bytes a wavefront), 1,280 of them the fold's 64
steps and 640 the readout's 32; the fold's two steps a list are issued whatever the list holds, so at lists of 8 to 24
entries part of its operands' reads are of pads. THEN, both sides made: the steps' operand fields loaded at each
step's issue, 512 (16 bytes a thread a step, every lane the same line); the readout's sums to the panel as 16-bit
halves, 512 (256 by stmatrix); the image's stores, 384; the drains' loads of the panel, 256; the images cleared at the
launch's start, 161 (once a launch). The block form adds its expansion (some 290) and the readout's coefficient tiles
(128). Bank conflicts: none in the made forms, 56 in the block form's expansion and pads. The launch profiled stores
the fold's state out too, so its stall samples are the dump's, not the window's.

THE TIMED LAUNCH'S LEDGER (the pack at c9e374fd, `ledger-dumps --skip 1`: each kernel's first timed launch, 64 windows
on 132 CTAs, `--per 8448`; its dumps `measure/lab/results/carry_window/ledger_timed_sm_90a/ledger.json` and beside
it). Of all warps' stall samples, THE FOLD'S STEPS' LOOP HOLDS THE MOST: 20% in the block form with the panel by
stmatrix (its reasons the warpgroup's barrier 53%, a short scoreboard 24%) and 27% in both sides' made with it (short
scoreboard 32%, barrier 25%) -- some half of the fold warps' samples, its 8 of the CTA's 16 warps; then the drain
warps' (long scoreboard, waiting), the readout's coefficient loads (long scoreboard 73-81%). A short scoreboard at a
step is a step's operand fields' shared-memory load waited on at its issue: the steps' loop loads each step's line at
the step, behind an edge that keeps the loads from one register set. READ AS, not isolated: the fold's steps wait out
one shared-memory load's latency each, 32 a window.