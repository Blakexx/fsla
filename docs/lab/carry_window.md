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

THE NO-READ-STEPS ROW (the read-ready ready pair's, clocked: its unclocked kernel spills an address) asks what holds the fold's steps' issue,
the path's largest phase (`lab path`): the readout issues no step -- its tiles' waits, its panel and its arrivals
kept, its sums zeros written each pass -- so the fold's steps meet none of the readout's on the tensor path.
Unchecked: an isolation.

THE TRACED ROWS -- the digit form with the panel by stmatrix, the same with the drain warps expanding (the serve row),
the operands landed made with their coefficient tiles, and that form with every box its own stream from HBM -- each
its unclocked row with every warp's timeline stamped (`lab trace`; the traces at
`measure/lab/results/carry_window/sm_90a.trace/` (warpline's native records, a row each), and at 8 CTAs `sm_90a.blocks8.trace/` beside it, that record
rola 99807df0's; the full grid's are rola 44e2597c's; at ONE CTA, `sm_90a.blocks1.json` and `sm_90a.blocks1.trace/`, whose own section THE LAUNCH'S EDGES is below. `kFine` ALONE is an EDGES row, `, the launch's two edges stamped`: the plain kernel with its two edges and nothing else stamped). The figures here are rounded: a session moves each by a few
cycles, the record holds them. Measured on an H100 SXM (Nebius): A FINE TRACE (every wait, arrival and issue beside
the phases' starts) costs the first three some 15%, 25% and 22% of their windows and the own-stream row 2%, so its
cycles are read for structure; A DETAIL ROW (the kernel `window_detail`: the coarse row's body with ONE group of waits and arrivals kept) spends a
stamp budget on one question, and four of them are every wait between the own-stream row's roles. At one CTA
(`sm_90a.blocks1.json`, the experiment at 4094bdcf; `bazel run //tools:lab -- trace carry_window` reads the rows'
records) they cost 4.9% to 7.6% of the plain window where the coarse row costs 3.6%, and they give: a fold warp
waits 555 to 592 cycles a window on its tile groups read (a tenth of the window); a readout warp 1,153 on the tile
groups published (a fifth) and 500 on the panels drained; a drain warp 1,380 on the panels filled (a quarter). THE
FINE ROW HAD EACH WRONG, in both directions (430 for the readout's wait on the folds, some 1,400 for the folds' on the
readout, 3,664 for the drain's): its stamps slow the readout most. A role's waits on its own steps and on its copies
landing are in none of the four groups. A COARSE TRACE (a phase's start and the launch's two edges, each warp's entry at the
kernel's top and its exit at its role's end, on both clocks: the rows `, every warp's phases stamped`; a fine row holds no
edges, its kernels being at their registers' limit) costs
them 3.0%, 5.5% and 2.2% and the own-stream row nothing a session resolves, so its times are the kernel's to within
that. THE COST IS THE STAMPS' NUMBER: a fine fold warp takes some 25 to 31 stamps a window, a readout warp 34 to 40 and
a drain warp 16 to 32, a coarse one 3 to 8 (the trace record's counts over its windows), at about 13 instructions a
stamp site as compiled here (the drain's chunk loop is 181 instructions plain and 237 fine-traced), where `stamp_cost`
prices a lone warp's added instruction at about two cycles. ON THE LAB'S SHARED STREAMS (16 columns, mostly the L2's) THE WINDOW IS
THE FOLD'S OWN CHAIN: at 8 CTAs the windows are the full grid's (some 6,500, 6,650 and 4,950 both ways), and the
path's copies bind by only 110-155 cycles over the fold's own work beside them -- in the digit form a fold warp's
traced window is its expansion with its pads, operands, fence and rendezvous wait (some 2,700 cycles), its steps'
issue (some 1,950) and its image (some 1,350), with some 1,400 more of waits; the readout and the drains are off it (a
drain warp idles some 4,300 a window). In the serve row the drain warps' expansion (some 4,000 a warp) holds the
fold's 3,500-4,300 waits on its slots. WITH EVERY BOX ITS OWN HBM STREAM THE CARD'S HBM BINDS: the window is some
13,100 at the full grid against 5,345 at 8 CTAs; the readout waits some 9,000 cycles a window on its coefficient
tiles' copy, the fold some 8,000-8,300 on the readout. A CTA's window moves some 123 KB (four slot copies of some 22
KB and four readout passes of 8.7 KB): some 16 MB a window card-wide, some 2.1 TB/s at that window.

THE FOLD-ALONE ROWS (the digit form with the panel by stmatrix, the fold's side alone, clocked and unclocked, the fold
holding the idle readout's registers, 184 a thread, where its phase clocks spill at 168) and the same form's clocked
full row ask what holds the fold's window when the read side runs beside it. PREDICTED before the run: the shared-
memory pipe if the fold alone issues its steps near 40 cycles a step, the schedulers' issue if every phase falls
alike. Measured (H100 SXM, Nebius; rounded): the fold alone is some 3,600 cycles a window (clocked some 4,250) against
some 6,500 with both sides (clocked some 6,750). EVERY PHASE FALLS ALIKE, by a fifth to a third -- a fold warp's
steps' issue some 1,880 to 1,550 (59 to 48 cycles a step), its expansion some 750 to 590, its pads and operands some
700 to 500, its steps' wait and copies some 465 to 325, its two rendezvous some 1,275 to 800 -- and its image (some
1,050) and its wait on the readout's read (some 230) leave with the read side. By the prediction that is the
schedulers' issue, not the shared-memory pipe alone -- but the SM issues at some 46% of its slots (the issue count
below), so what the read side takes from the fold is NOT ISOLATED: issue, the shared-memory pipe and the barriers it
shares are each still open. The 16 registers the fold alone holds do not move its window: unclocked at 168 a thread it
was 3,610 (the record at rola 1c811b5d), at 184 some 3,615. A slot's expansion is some 300-380 cycles a warp; the
traced phase of 2,700 a window is expansion, pads, operands, fence and the rendezvous wait together.

THE HELD-STEP ROW (the fold alone, clocked, every step of a slot issued with the slot's first step's operands loaded
once and held; its sums wrong, unchecked) asked why a fold step issues at 48 cycles alone against some 19 of its own
tensor time: in its loop each step's operands are a shared load whose four moves to the uniform registers the next
step waits on, one load in flight. PREDICTED: if that load's latency, near 20-25 cycles a step (some 700 a window); if
the step's own issue, near 48. Measured: THE LOAD -- the held row issues its 32 steps in some 480 cycles (15 a step)
against some 1,550, and its window is some 3,680 against 4,250 clocked: the steps' wait grows (some 325 to 580) and
the rendezvous with it, the tensor pipe taking what the issue gave. BUT THE COMPOSED FORM DOES NOT TAKE IT: its row
with the operands loaded two steps ahead, whose binary holds several loads in flight and its steps back to back, is
some 6,510 against 6,500 (the ahead rows above). The clocked ahead row below shows why: in the composed form its
steps' issue does not fall at all.

THE ISSUE COUNT (the timed ledger, `measure/lab/results/carry_window/ledger_timed_sm_90a/`, the digit form with the
panel by stmatrix: its executed warp instructions by role, the role's code the span between its register raise or
release and the next): a scheduler holds one warp of each warpgroup -- two fold warps, a readout warp, a drain warp --
and issues some 3,000 instructions a window: the fold's two some 1,320 (some 660 a warp), the readout some 940, the
drain some 725. Against a window of some 6,500 that is some 46% of its issue: THE SM IS NOT ISSUE-BOUND; a fold warp
issues some 660 instructions in its window and waits out the rest -- its loads, its rendezvous, its copies -- the
other three warps' some 2,330 issue cycles on its scheduler the read side's share of its slots. At one instruction a
cycle a scheduler, the count alone is a floor of some 3,000 cycles a window.

THE SHARED-MEMORY COUNT (the same timed ledger, the same form, its shared wavefronts by role): a CTA window asks the
shared-memory pipe some 3,950 wavefronts -- the fold some 2,660 (its steps' operands 1,280, its stores 716, its loads
664, the steps' operand loads some 256 of those), the readout some 1,020 (its steps' operands 640, its sums by
stmatrix 256, its stores 128), the drain some 260 (its panel loads) -- some 3,950 cycles at one a cycle
([smem_rates.md](smem_rates.md)), the binding unit of the window; its output's adds are 256 lines, some 2,560 cycles
card-wide at the H100's some 10 cycles a line (`add_rate`, [add_rate.md](add_rate.md)).

THE CLOCKED AHEAD ROW (the digit form with the panel by stmatrix and its operands loaded two steps ahead, clocked)
asked where the fold's faster issue goes in the composed form. PREDICTED: its steps' issue falls toward the fold
alone's (some 1,550 a window, or below), and the window does not. Measured on an H100 SXM (Nebius; rounded): THE ISSUE DOES
NOT FALL -- some 1,840 cycles a window against some 1,880 (58 cycles a step), every other phase within a few percent,
the window some 6,730 against 6,750 clocked. In the composed form a fold step's issue is not its operands' load: alone
it is (48 a step, 15 held), beside the read side something each step shares holds it near 58 whatever its load. NOT
ISOLATED; the candidates are what the read side's warps use beside each step -- the tensor pipe's queue (the readout's
own steps: with them out the fold's issue fell only 2,263 to 1,991 in the operands-made form, the no-read-steps row),
the uniform datapath each step's operands cross (the readout issues some 400 uniform instructions a scheduler a
window), and the shared-memory pipe the steps' operands are read through.

THE STEPS' STALLS (`measure/lab/results/carry_window/ledger_steps_sm_90a/`: the timed launches of the digit form with
the panel by stmatrix and of its ahead row, one binary, profiled on an H100 SXM): over the fold's step loops both are
alike -- 57% of their stall samples the short scoreboard, 15-16% issuing, 9-10% a fixed wait, 7% unselected, 4-6% the
tensor pipe -- and 87-88% of the samples are at the moves of a step's operands into the uniform registers, waiting on
the shared load that brought them. With the load two steps ahead the move still waits: in the composed form an
operands' load takes longer than two steps. READ AS, not isolated: queued behind the shared-memory pipe's other
traffic, the steps' own operand reads among it. Alone held, no load at all, the issue was 15 a step: what each step
needs is its operands without a shared load in front of it.

THE COUNTED ROW (the digit form with the panel by stmatrix, a slot's 8 lists' counts a byte each in its image's head
line's second 8-byte word, loaded once at the slot's start beside its expansion's loads, every step's operands made
from their prefix sums -- a list's values after the tables and the lists before it, its dense lines after the slot's
128 entries' values, its second run the lists' padded runs further on -- and no step table) asks whether a step
without a shared load in front of it issues as the fold alone held did. Checked exactly. PREDICTED: the composed
window some 5,300-5,900 against some 6,500, the fold's chain shorter by most of its steps' wait on their operands.
Measured on an H100 SXM (Nebius): IT LOSES -- some 6,775 cycles a window against some 6,500, exact. Its binary holds
no shared load among a slot's steps, but its operands' arithmetic is made beside each step in the vector registers,
each step's four moves to the uniform registers still its own: a slot's 16 steps are 190 instructions against 79. READ
AS, not isolated (a static count, no clock or sample behind it): the load left the chain and more than its cost came
into it. (Its clocked twin spills 32 bytes: the phases are not split.)

THE UNIFORM ROW (the counted row with the counts shuffled from the warp's first lane, a slot's operands made once
before its steps) is in the binary as the toolkit makes it: its operands' arithmetic stays in the vector registers, a
slot's 16 steps 114 instructions (37 moves to the uniform registers, no shared load) against the plain form's 79 and
the counted row's 190. Measured beside the next rows: some 6,770 cycles a window against some 6,500 -- it loses, as
the counted row does.

THE READ SIDE'S ACCESSES TAKEN OUT of the digit form with the panel by stmatrix, one a row -- the drains' global adds,
the image's stores (each unchecked) -- ask which of the read side's accesses costs the fold the some 1,400 cycles a
window it loses beside them. PREDICTED, if what it loses is the shared-memory pipe: the image's stores out moves the
window most (some 300-600 cycles), the adds on the global path little. Measured on an H100 SXM (Nebius; rounded): the
image's stores out some 5,960 against some 6,500 (some 540 less, the fold's own image phase among it), THE ADDS OUT
some 6,190 (some 310 less) -- the drains' global adds slow the fold, which never waits on them. READ AS, not isolated:
a shared load and a global add take the same load-store queue of their sub-partition, so the fold's steps' operand
loads wait behind the drains' adds as behind the shared traffic.

THE VECTOR-DRAIN ROWS (the digit form and the token-order form landed whole, each with the panel by stmatrix, and each
with the drains' loads and adds 16 bytes a lane -- a lane an eighth of a token's row, four tokens an instruction, the
same lines) ask whether the drains' share of the load-store queue, not their lines, is what slows the fold: 4 adds and
4 loads a drain warp a chunk against 16 and 16. Checked exactly. PREDICTED: the digit form some 100-300 cycles a
window under its own, toward the some 6,190 with the adds out; the token form by a like amount under its own with the
panel by stmatrix -- the read side's cut carries to the form the design is. Measured on an H100 SXM (Nebius; rounded):
NO GAIN -- the digit form some 6,510 against some 6,505, the token-order form some 8,375 against some 8,420, both
exact. The drains' instructions on the queue are not what slows the fold; only their adds' going (some 6,200) moves it
-- READ AS, not isolated: their lines through the L1, not their count.

THE WIDE-FOLD ROWS (the digit form with the panel by stmatrix, the drain warps at 40 registers a thread and the folds
at 184; plain and with the uniform row's operands made once a slot, each clocked and unclocked) ask where the uniform
row's loss was: [`step_queue`](step_queue.md) measured a fold step holding its operands at some 27 cycles beside every
kind of traffic, against 58 composed, yet the uniform row lost; at 168 registers its clocked twin spills. PREDICTED:
the plain form near its own (the drains' registers are not on the path); the uniform form some 5,500-6,100 cycles a
window, its steps' issue some 900-1,100 a window against some 1,880, and the clocks naming what holds it if not.
Measured on an H100 SXM (Nebius; rounded): the plain form some 6,370 against some 6,500, the uniform some 6,490, both
exact. THE DRAINS AT 40 BIND: clocked, the plain form's readout waits some 3,380 cycles a window on the drain (some
270 at 72) and its sums to the panel some 3,980 with that wait in it, and the fold some 1,250 on the readout's read;
the drains need their 72. THE UNIFORM FORM'S STEPS ARE NOT THE HELD TABLE: its steps' issue is some 2,130 a window
(some 66 a step) against the plain form's some 1,520 at 184 -- its slot's operands made and a round's offset and zeros
chosen inside the phase, where step_queue's held table moves one held word a step (some 25).

THE PACKED-TABLE ROW (the digit form with the panel by stmatrix at its own registers, 168 and 72; the counted rows'
counts; a slot's 16 steps' operand addresses made once in the expansion's phase -- a step's A and B address fields and
its round's liveness one held word -- and a step's operands unpacked from its word) asks whether the carry's fold can
hold its operands as [`step_queue`](step_queue.md)'s held table does, some 25-27 cycles a step beside every traffic.
Its binary: no shared load among a slot's steps and no local memory, 131 instructions a slot's 16 steps (some 8 a
step: the unpacking, the pitch chosen, the moves to the uniform registers) against the plain form's 79 and the held
table's some 5 a step. Checked exactly. PREDICTED: its steps some 27-32 cycles each, the window some 5,600-6,000
cycles against some 6,500; its clocked twin spills, so the window alone is read. Measured on an H100 SXM (Nebius;
rounded): IT LOSES TOO -- some 6,580 cycles a window against some 6,500, exact. No form that takes the shared load out
of the fold's steps has shortened the composed window; where its time goes is read from its profile, below.

THE PACKED ROW'S PROFILE (`measure/lab/results/carry_window/ledger_packed_sm_90a/`: the timed launches of the plain
digit form and of the packed-table row, one binary, profiled on an H100 SXM; a role's stall samples are its warps'
time). The fold's warps: 5,299 instructions a CTA window plain, 8,662 packed -- a held word is made by every thread,
so each of the 8 fold warps makes the slot's whole table; the short scoreboard 26% of the fold's samples plain and 11%
packed, its wait on its steps' commit group 4% and 13%, the tensor pipe 3% and 10%, issuing or unselected 15% and 30%.
READ AS: the packed fold's steps issue without their load and then wait on the tensor pipe, where the plain fold's
wait on its operands' load hid the same wait -- the two fold warpgroups issue a slot's 16 steps each at once, beside
the readout's; and the table's making is some 3,400 instructions a window the plain form does not issue. What a held
table saved, its making and the steps' own completion took.

THE LIGHT-LOAD RECORD (`measure/lab/results/carry_window/sm_90a.blocks8.json`, the whole experiment at 8 CTAs, `lab
run --blocks 8`) asked whether a copy's flight is a fixed latency or the card's bandwidth shared. PREDICTED, if
shared: at 8 CTAs the flight falls to some 1,000-1,500 and the window toward 3,000-3,500. Measured: neither on the
shared streams -- their windows do not move, because the fold's own chain, not the copy, holds them; on the own stream
the window falls 2.5 times (some 13,100 to 5,345), which is the card's HBM bandwidth.

THE SERVE ROWS (the block form with the panel by stmatrix, with and without clocks) ask whether the fold's own
expansion -- the 1,500-odd cycles a window its expansion, pads, operands, fences and rendezvous cost its path --
leaves that path when the four drain warps do it a slot ahead: before a window's drain opens they expand its second
slot (its copy issued at the window before's end), and half through it the next window's first (its copy issued as the
fold left this window's first); a slot's expansion is each box's landed image awaited, a thread's entry expanded, its
pad zeroed, a thread of 16 its step's operands into the slot's own table, the fence and a warp's arrival; the fold
waits on that arrival and steps. Checked as the block form's rows are. PREDICTED: some 5,700 cycles a window (the form
with the fold's operands made), and the drain warps then busy some 70% of it.

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

MEASURED (H100 80GB HBM3, 132 SMs, Nebius preemptible VM 3dt0t7sr, driver 580.173.02; 2026-11-18, the experiment at
44e2597c, one session; 7 repetitions, every checked row's state and output values exact;
the median; SM-cycles a token are the window's cycles over 8 -- a CTA of two boxes is 8 tokens a window. THE FOLD'S STATE
DUMP STAYS OUT OF THE TIMED CALLS, and one row restores it to price it in this session):

| the row | cycles a window a CTA | SM-cycles a token |
|---|---|---|
| the fold's side alone: producers and steps | 5,091 | 636 |
| the read side alone: the image, the readout and the drain | 5,560 | 695 |
| the carry: both sides | 7,032 | 879 |
| both sides, the producers' global loads taken out | 6,432 | 804 |
| both sides, the drains' global adds taken out | 6,716 | 840 |
| both sides, the image's stores taken out | 6,705 | 838 |
| the read side alone, no fold warp running | 4,381 | 548 |
| the read side alone, every wait suspended | 5,268 | 659 |
| both sides, every wait suspended | 6,902 | 863 |
| THE READOUT DRAINING ITS OWN ROWS, FOUR PRODUCERS: the fold's side alone | 3,075 | 384 |
| the same: the read side alone | 5,367 | 671 |
| the same: both sides | 7,432 | 929 |
| THE READOUT LEAN, TWO BATCHES A PRODUCER: the fold's side alone | 4,556 | 569 |
| the same: the read side alone | 5,768 | 721 |
| the same: both sides | 6,784 | 848 |
| FIVE WARPGROUPS, FOUR PRODUCERS AND FOUR DRAIN WARPS AT 40: the fold's side alone | 5,880 | 735 |
| the same: the read side alone | 5,045 | 631 |
| the same: both sides | 7,736 | 967 |
| THE LONG WAITS ASLEEP AT NAMED BARRIERS: the fold's side alone | 5,109 | 639 |
| the same: the read side alone | 5,065 | 633 |
| the same: both sides | 6,972 | 872 |
| the fold's side alone, the producers' global loads taken out | 2,412 | 301 |
| the fold's side alone, their coefficient arithmetic taken out | 4,693 | 587 |
| the fold's side alone, both taken out | 2,281 | 285 |
| the read side alone, the readout's byte loads taken out | 5,003 | 625 |
| the read side alone, no fold warp running, the byte loads taken out | 4,017 | 502 |
| both sides, the readout's byte loads taken out | 6,665 | 833 |
| both sides, the drains' adds and the readout's byte loads taken out | 6,414 | 802 |
| THE READOUT'S BYTES ONE WIDE LOAD A WARP: the read side alone | 5,050 | 631 |
| the same: the read side alone, no fold warp running | 4,217 | 527 |
| the same: both sides | 6,977 | 872 |
| the same: both sides, the long waits asleep | 7,145 | 893 |
| the read side alone, the image's stores taken out | 4,990 | 624 |
| the read side alone, the fold's fence behind its image taken out | 5,505 | 688 |
| the read side alone, both taken out | 4,976 | 622 |
| both sides, that fence taken out | 6,934 | 867 |
| the read side alone, no fold warp running, the fold warps parked asleep | 4,375 | 547 |
| the same, spinning | 4,503 | 563 |
| THE DUMP RESTORED: both sides, the fold's state stored out in every call | 8,173 | 1,022 |
| THE WINNERS COMBINED: both sides, suspended waits and the lean readout | 6,659 | 832 |
| the same, suspended waits and the wide byte load | 6,919 | 865 |
| the same, suspended waits, the lean readout and the wide byte load (the fourth warpgroup at 104) | 6,752 | 844 |
| the same, sleeping waits, the lean readout and the wide byte load | 7,028 | 878 |
| THE TABLES ONCE-READ, 16 replicas the cache cannot hold: both sides | 17,739 | 2,217 |
| the same, the winner's form | 16,111 | 2,014 |
| the same: the fold's side alone | 11,936 | 1,492 |
| THE DEN, folded, published, read and drained: the fold's side alone | 5,099 | 637 |
| the same: both sides, the winner's form | 7,068 | 884 |
| the same, the tables once-read | 16,589 | 2,074 |
| THE IMAGE'S TILES MOVED APART, nothing else changed: the winner's form at 154 lines (137 as it is) | 7,074 | 884 |
| the winner's form with the den at 161 lines (155 as it is) | 7,082 | 885 |
| THE ALIGNED CELL, an image tile a list: the winner's form | 6,661 | 833 |
| the same with the den | 7,068 | 884 |
| NO CLOCK INSIDE A ROLE: the winner's form | 6,590 | 824 |
| the same with the den | 6,935 | 867 |
| the same at the aligned cell | 6,678 | 835 |
| the same at the aligned cell with the den | 7,072 | 884 |
| THE DEN'S PARTS, a chain with no clock inside a role: the fold's steps | 6,654 | 832 |
| then the image's ninth brick and its store | 6,746 | 843 |
| then the register split | 6,833 | 854 |
| then the 72-column steps | 6,891 | 861 |
| the winner's form with its tiles at the den's pitch, 155 lines | 6,598 | 825 |
| THE VALUES TILE-MAJOR in global memory, no clock inside a role: the winner's form | 12,201 | 1,525 |
| the tables once-read, the values token-major | 16,385 | 2,048 |
| the tables once-read, the values tile-major | 48,903 | 6,113 |
| THE FOLD'S ENTRIES FROM LISTS, a window's own tokens: the fold's side alone | 8,782 | 1,098 |
| the same: both sides, no clock inside a role | 9,975 | 1,247 |
| the same with every window the same tokens: the fold's side alone | 6,268 | 784 |
| the same: both sides | 8,105 | 1,013 |
| a window's own tokens, their rows asked for a window ahead: both sides | 11,883 | 1,485 |
| THE VALUES LANDED BY BULK COPY from expanded lists, a window's own tokens: the fold's side alone | 8,476 | 1,059 |
| the same: both sides | 9,032 | 1,129 |
| THE WHOLE ENTRY LANDED, values and coefficient bytes: the fold's side alone | 5,171 | 646 |
| the same: both sides | 5,983 | 748 |
| the whole entry, EACH BOX ITS OWN expanded windows, past the cache: the fold's side alone | 6,482 | 810 |
| the same: both sides | 11,166 | 1,396 |
| the whole entry, its 16 columns' windows NEVER READ BEFORE, 16 boxes a window's readers: the fold's side alone | 5,165 | 646 |
| the same: both sides | 5,967 | 746 |
| THE PACKED STREAM, packed by digit, its lists uneven (8 to 24 entries), last steps truncated: the fold's side alone | 10,026 | 1,253 |
| the same: both sides | 10,549 | 1,319 |
| the packed stream, FOUR SLOTS OF 4 TILES a box: the fold's side alone | 10,707 | 1,338 |
| the same: both sides | 11,473 | 1,434 |
| the whole entry, THE FOLD WARPS EXPANDING their own slot: the fold's side alone | 2,992 | 374 |
| the same: both sides | 5,966 | 746 |

THE WALL TIME, measured on this box and immune to its clock (1.76 to 1.82 GHz, its rows' medians): the long call
scaled to 65,536 tokens is 0.250 ms for the baseline, 0.237 ms for the winner's form, and 0.493 ms at the
once-read bound (0.570 without the winner's changes). THE FLASH REFERENCES, from the session of instance 60450651 ON ANOTHER BOX (a different offer and driver)
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
| a fold warp waits on the readout's read of a tile group | 375 | 4,181 | 3,802 |  |  | 3,769 | 419 |
| a fold warp waits on its producer | 4,672 |  |  |  |  | 431 | 5,579 |
| a readout warp waits on a tile group's publish | 1,844 | 291 | 323 |  |  | 300 | 2,184 |
| a readout warp waits on the panel's drain | 327 | 287 | 273 | 246 | 282 |  | 328 |
| a readout warp draining its own rows |  |  |  |  |  | 2,834 |  |
| a readout warp forming its steps' A | 1,341 | 1,101 | 1,109 | 1,267 | 1,007 | 1,103 |  |
| a readout warp in its steps (a lean readout's hold the forming) | 2,197 | 2,579 | 2,131 | 2,203 | 2,049 | 1,951 | 3,864 |
| a drain warp waits on the panel | 3,186 | 2,034 | 1,522 | 833 | 701 |  |  |

WHAT IT SHOWS.
- THE CARRY COMPOSES, EXACTLY: two boxes' folds, the image a window, one readout warpgroup over both -- its
  coefficients expanded in its registers, its steps' A operand -- and one f16 line a token, in one CTA of 512
  threads at the SM's whole register file.
- A TOKEN IS PRICED AT 2,014 SM-CYCLES COMPOSED, its tables read once (R22.2), in the winner's form; 832 when the
  cache holds them (879 in the first form) at this cell (its sides sparse on opposite levels, 256 x 256 leaves;
  [`carry_floor`](../internals/tools/carry_floor.md) prices a dense side higher).
- THE FILL WANTS WARPS THAT EACH HOLD A BATCH: two producers a box, a batch of 32 entries in registers each, take
  the fold's side alone to 3,075 cycles a window where one a box is 5,091; one a box holding TWO batches is
  4,556; and two a box at 40 registers a thread, each holding one value instruction's four entries, are
  5,880 -- slower than one a box that holds a batch.
- SUSPENDED WAITS WITH THE LEAN READOUT ARE THE FASTEST COMPOSED FORM: 6,659 cycles a window against the
  baseline's 7,032, with the suspended waits beside the wide byte load a near tie at 6,919. The parts: every wait
  suspended alone 6,902, the long waits asleep at named barriers 6,972, the lean readout alone 3.5%
  under the baseline, the wide load alone 6,977. ALL THREE AT ONCE LOSE the pairing's gain (6,752: the lean
  readout beside the wide load's scratch needs 88 registers a thread and the fourth warpgroup drops to 104), and
  the sleeping triple is 7,028.
- THE DEN COSTS 5.2% OF THE COMPOSED WINDOW AT THIS CELL AND 5.9% AT THE ALIGNED ONE, the tables cached and no
  clock inside a role: 6,935 cycles against 6,590, and 7,072 against 6,678. THE ROLES' CLOCKS ARE NOT FREE AND NOT EQUAL: they are
  69 cycles of the winner's clocked window and 133 of the den's, and which is larger has changed between
  records, so a clocked row does not price the den (clocked: 7,068 against 6,659; why the clocks differ is
  not isolated); with the tables once-read 16,589 against 16,111; the fold's side alone 5,099 against 5,091. One
  cell (its sides sparse on opposite levels); a dense write side, where the fold's side binds, is unmeasured. ITS
  PARTS, each a row less the row before, no clock inside a role: the fold's four steps a window 64 cycles;
  the image's ninth brick and its store 91 (the pitch itself nothing: the winner's form at 155 lines is 6,598
  against 6,590); the register split, readout 88 and fourth warpgroup 104, 88; the steps 72 columns wide 57;
  the panel's word and the drain's add 44. No part is a third of it. The split cannot be avoided in this
  form: the 72-column readout serializes its steps at 80 registers with no clock read too.
- THE FOLD'S FORM WAS A CONTEST. A step a list with ones in one row of A is 16 steps a window for 1 row in 64 and
  a 17th accumulator set of 8, which the toolkit will not hold at 160 registers: at a 168-register fold with
  one-batch producers it cost an eighth of the composed window in its session (the experiment at ac342446, whose
  record this one replaced). The coefficients as A are four steps into 4 registers.
- AN IMAGE TILE'S PITCH MUST BE AN ODD COUNT OF LINES: a warp's lanes store to four tiles two apart, and at an
  even pitch two of the four land on the same banks. The winner's form at 154 lines is 7,074 against 6,659 at 137;
  the den's at 161 is 7,082 against 7,068 at 155.
- THE IMAGE FORM COSTS 1.3% MORE WHERE THE CARVES ARE EQUAL, no clock inside a role (6,678 cycles a window against
  6,590); clocked, the aligned cell's winner's form is 6,661 cycles a
  window against 6,659 at the cell whose sides list opposite levels -- 16 word stores a thread a tile group where
  that cell's are 4 lines -- and with the den 7,068 against 7,068, the den's lines there 2-byte
  stores of 8 threads. A form that reads the fold's own registers has nothing of the window to win back but the
  image's stores; it is not built.
- THE CARRY'S PRODUCERS WANT A TOKEN'S VALUES AS ONE RUN: with the values tile-major in global memory -- a token's 8
  lines a kilobyte apart -- the winner's form is 12,201 cycles a window against 6,590, and with the tables once-read
  48,903 against 16,385. The intra's window moves 5% with the values' order ([`intra_window`](intra_window.md)), so
  the values stay token-major and the levels, which the carry reads 16 bytes at a time, go tile-major.
- FROM LISTS, WITH A WINDOW'S OWN TOKENS, THE FOLD'S SIDE ALONE IS 8,782 CYCLES A WINDOW where the synthetic entries'
  is 5,091 cached and 11,936 once-read; both sides 9,975 (the readout's side still on synthetic, cached readers) against
  6,590. Two things are in it, and the rows with every window the same tokens part them: THE LISTS' CODE -- a
  batch's places loaded a batch ahead, a shuffle a value instruction, the sparse word, tile-major addresses, and one
  batch held where the winner holds two -- is 6,268 against 5,091 alone and 8,105 against 6,590 composed; and ROWS THE
  CACHE HAS NOT SEEN, a window's own being new each window as a call's are, are the rest, 2,514 alone and
  1,869 composed. A window's rows are loaded one batch ahead of their use. ASKING FOR THEM A WINDOW AHEAD BY
  PREFETCH LOSES: 11,883 composed against 9,975 -- six prefetches an entry-batch are more requests of the same
  memory, not less waiting. Whether loads issued two batches ahead hide the rows' arrival is not measured (the
  producers' registers hold one batch from lists). The check of these rows is weak: every table's byte is its token's parity.
- LANDING THE VALUES ALONE FROM EXPANDED LISTS is 9,032 cycles a window composed against 9,975 with every entry's
  values gathered (8,105 for the gather with rows the cache holds), and the fold's side alone 8,476 against 8,782: a
  list's values are one bulk copy of 2 KB from a buffer laid contiguous a window (the host's here; 16 sources
  share a buffer), new each window, and its producers still gather an entry's 16 dense bytes and its sparse byte,
  a line of each level's rows an entry. What that leaves is the next finding's.
- WITH THE WHOLE ENTRY LANDED -- its coefficient bytes beside its values, raw, a list one bulk copy and no global load
  in a producer's trip -- the window is 5,983 cycles composed and the fold's side alone 5,171, against 6,590 and
  5,091 for the synthetic entries whose rows the cache holds.
- WHAT THAT ROW NEEDS IS ITS BYTES SHARED, NOT HELD BEFOREHAND: with a window's bytes never read before and its
  16 boxes its readers, the window is 5,967 cycles composed against 5,983 with the cache holding them, and
  the fold's side alone 5,165 against 5,171: one box's miss serves its 15 mates. With EACH BOX ITS OWN it is
  11,166 composed, 1.87 times, and 6,482 alone, 1.25 times -- 16 times the bytes from global memory, on
  every multiprocessor at once.
- THE PACKED STREAM IS EXACT, AND ITS UNEVEN LISTS COST THE FOLD 1.94 TIMES: 10,026 cycles a window the
  fold's side alone against 5,171, 10,549 composed against 5,983. The same 256 entries a box a window, as
  lists of 8 to 24 where the row above has 16 each, are 24 steps where it has 16 -- three slots of 8 tiles where it
  has two -- and a truncated step, used where it landed with its bricks as far apart as it has entries, folds
  exactly. In four slots of 4 tiles, a slot's copies issued two slots ahead, the same stream is 10,707 alone
  and 11,473 composed: more slots cost more, so a slot's price is not its bytes' landing.
- THE FOLD'S SIDE WAITS ON ITS PRODUCER. With even lists a fold warp waits 4,540 of the fold's side's 5,171 cycles a
  window on its producer: its own steps are what is left. One producer warp a box expands every entry. On the
  packed stream in four slots the wait is 4,216 of 10,707, and with TWO PRODUCERS A BOX 1,484 of 8,474: the wait
  falls and the fold's own time is then the larger part. That fold runs a list's steps as a loop of its own, a
  vote a list a slot, where the even rows' steps are one block; the block rows below run the uneven lists' steps so.
- WITH THE FOLD WARPS EXPANDING THEIR OWN SLOT (even lists; a producer only issues copies) the fold's side alone
  is 2,992 cycles a window against 5,171, a fold warp's wait -- now on its slot's bytes -- 1,856 against 4,540.
  THE COMPOSED WINDOW DOES NOT MOVE: 5,966 against 5,983; with the fold issuing its own copies too, no
  producer running, 5,872.
- THE WINDOW IS THE READOUT'S. That form with every role's waits clocked (6,313 cycles a window): a readout warp
  waits 286 on the image and 278 on the drain and is 4,246 in its coefficients and steps; a fold
  warp waits 1,792 on the readout's read and 178 on its bytes; a drain warp waits 2,051 on the
  panel. The one role that does not wait is the readout's warpgroup.
- FOUR DRAIN WARPS BUY NOTHING: 5,824 against 5,872, each drain warp then waiting 4,019 of a window on the
  panel, and a readout warp is 3,813 in its coefficients and steps against 4,246. The drain is not the
  window's bound.
- THE MIRRORED READOUT (coefficients in shared memory, tokens on the step's columns; the fold expanding, a producer
  issuing): the window is 5,576 cycles against 5,966. Clocked (6,016 against 6,509): a readout warp is
  1,083 forming its coefficients and 156 in its steps, against 3,830 for both in the register form; it
  waits 360 on the image and 238 on the drain; a fold warp waits 2,691 on the readout's read against 2,800;
  a drain warp 2,381 on the panel against 2,523.
  What its clocks do not name is 4,179 cycles a window: its rows' rounding, their stores and its arrivals. Its 32
  steps are a quarter the register form's multiply-adds and take 156: a step's price here is not its size.
- THE BLOCK ROWS -- uneven lists, a block a list, the stream saying what to do: the window is 6,859 cycles
  composed (6,845 clocked), against 5,289 for even lists under the same readout and drains (whole-entry
  tiles, the fold issuing) -- 1.30 times it -- and
  10,549 for the uneven lists' truncated steps behind a producer and a loop a list.
  The fold's side alone is 3,779,
  a fold warp waiting 264 of it on its bytes, against 2,992 for the even lists' fold expanding its own slot. Clocked: a fold warp waits 251 on the
  readout's read and 190 on its bytes; a readout warp 1,763 on the image (317 with even lists) and 251 on the drain; a drain warp
  4,337 on the panel.
  THE FOLD'S PHASES, clocked, the block form against the even lists' (a clocked fold holds nine more
  registers; the windows are 6,845 and 5,816 with them): publishing the image 1,309 against 1,955,
  its wait on the readout's read in both (251 and 263); expanding its entries' lines 722 against 782; its pads and its
  steps' operands 734, the block form's alone; its slots' fences 54 and 71 against 69
  and 54; AT ITS SLOTS' RENDEZVOUS -- between a reading before the warpgroup's sync and one behind it --
  722 and 572 against 347 and 480; its steps and its next copy's issue 2,130 against 1,716.
  A FENCE IS SOME 60 CYCLES, AND A FOLD WARP IS LONGER AT A SLOT'S RENDEZVOUS THAN EXPANDING THE SLOT. With no
  arithmetic in the fold (the control: windows 6,641 against 6,859 and 5,225 against
  5,289; its entries' lines 441 and 335) its rendezvous are 738
  and 595, and 293 and 623: they do not follow the lines' work. THE
  FOLD'S ARITHMETIC IS NOT WHAT THE WINDOW WAITS ON.
  WITH BOTH COPIES ISSUED AT THE WINDOW'S END (even lists) the window is 5,481 against 5,289,
  its fences 66 and 58: a fence does not wait on a copy in flight, and the later
  copies cost the window.
  ITS RENDEZVOUS MOVE WITH THE ISSUE: 809 at its first slot and 154 at its second, where the form
  that issues a slot's copies as it leaves the slot is 347 and 480 -- the rendezvous behind an issue
  is the long one.
  A WARPGROUP WAITS AT ITS RENDEZVOUS FOR ITS FIRST WARP, whose lanes issue the slot's 8 copies: the first warp
  alone is 91 and 92 at its two rendezvous where the warps' mean is 347 and 480, and
  2,661 in its steps and its issue where the mean is 1,716.
  WITH THE COPIES ISSUED TWO A WARP every warp is 218 and 208 at its rendezvous and 1,697 in its
  steps and issue; the window is 5,241 against 5,289, and a fold warp then waits 872 on the
  readout's read against 263: THE FOLD'S TIME SAVED IS ITS WAIT ON THE READOUT, WHICH IS THE WINDOW.
  THE BLOCK FORM'S FIRST WARP IS LAST TOO: 103 and 97 at its rendezvous against 722 and
  572; its plans, its one copy a slot, 16 threads' operands and the first pads are its own -- 1,167 at
  its pads and operands against 734, 832 expanding against 722, 2,533 in its steps and
  issue against 2,130. WITH THOSE JOBS SHARED AMONG ITS WARPS THE WINDOW DOES NOT MOVE: 6,854 against
  6,859. Its rendezvous fall to 495 and 301 and every warp's pads and operands rise to
  1,247: the phase is some 53 instructions a thread, and a warp is as long in it with a pad and four operands
  as the first warp was with all of them. NOT ISOLATED: what a warp waits on in that phase.
  THE READY PAIR -- the block form's slots landed with their operands made, the fold only landing and stepping:
  the window is 5,589 against 6,859 for the block form and 5,289 for even
  lists; clocked (5,798), a fold warp waits 218 on its bytes and
  920 on the readout's read, is 2,186 issuing its 32 steps and
  834 in their wait and its copy's issue, and A READOUT WARP WAITS 286 ON THE
  IMAGE against 1,763 with the fold expanding and 317 with even lists: with its operands made the
  uneven lists' fold is no longer what the window waits on. What the window is over the even lists' by, no row
  parts: the fold issues twice the steps beside the same readout.
  THE READ-READY ROWS -- the readout's coefficient tiles landed made as well: the window is 5,244 against
  5,289 with even lists, and 5,198 against 5,589 for the ready pair's form. Clocked
  (5,317 and 5,296), a readout warp is 833 and 379 from its
  image's wait to its tiles landed, where forming them itself with its rendezvous is 1,470 and
  1,539; it waits 983 and 1,109 on the image, and a fold warp
  399 and 295 on the readout's read.
  THE OWN ROWS -- every box its own stream, where 16 share a column's -- ARE THREE TIMES THE WINDOW: 12,921
  against 5,198. Their streams are some 680 MB, past the card's cache, where the form's are under it; a
  window's bytes a CTA are the same. Clocked (12,815): A READOUT WARP WAITS 9,234 ON ITS
  TILES' BYTES against 379 -- its copy is issued one panel's stores ahead of its use -- and
  a fold warp, whose copies are issued a window ahead, 739 on its own against 211; the
  fold then waits 7,327 on the readout's read. WITH ITS OPERANDS FROM THE CARD'S MEMORY THE CARRY IS
  THAT MEMORY'S: the ready stream is a window's ~123 KB a CTA, and past the cache it is this price. NOT ASKED: the
  readout's copy issued further ahead; the design's mix, a box's own coefficients and its column's values.
  THE TOKEN ROWS -- the fold laying its own operands from a view in token order -- ARE A QUARTER OVER THE BLOCK FORM, HOWEVER THE VIEW
  ARRIVES: the window is 8,497 with a half-window view staged by one copy and its value lines moved in shared memory,
  8,608 with ONE VIEW A WINDOW loaded through registers and its 32 steps one block, and 9,531 with the half-window
  view through registers, against 6,859 for the block form, whose value runs land in place, and 5,589
  with its operands made. Clocked (8,943 and 9,398; the block form 6,845):
  THE MOVE IS THE FOLD'S FILL -- a fold warp is 2,383 expanding and moving its entries where the block form's is
  722 expanding them, 16 value lines a thread a window through the shared-memory pipe twice; and THROUGH REGISTERS
  THE FILL IS 5,591, two entries a thread of 9 lines each from global memory, the stream in the cache: a warp's
  loads of 16 bytes are a fifth of a bulk copy's bytes a cycle (KERNEL_STANDARDS 33). THE ONE-VIEW ROW'S STEPS AND RENDEZVOUS ARE SHORT, AND NOT BY ITS STRUCTURE (the joined row, below): its 32 steps
  issue in 973 where the block form's two slots' are 1,672, its one rendezvous is
  349 where two are 722 and 572, and its pads and operands
  835 against 734. A readout warp waits 2,901 and 4,530
  on the image against 1,763: the fold is the window in both. NOT ASKED: one view a window staged by
  small bulk copies; a rank counted from the mask, where these rows' places are the host's.
  THE JOINED ROW -- the block form's two slots laid before ONE RENDEZVOUS, its 32 steps one block -- DOES NOT MOVE THE WINDOW:
  6,787 against 6,845, both clocked. Its one rendezvous is 1,041 where the two are 722 and
  572, its steps' issue 1,471 against 1,672, with their wait 1,774 against
  2,130; its expansion is 878 against 722 and its pads and operands 806 against
  734. The row also issues both copies at the window's end, which the late-issue rows price apart; the two are
  not parted here. SO THE ONE-VIEW ROWS' SHORT ISSUE AND RENDEZVOUS ARE NOT THEIR STRUCTURE'S. READ AS, not isolated: their
  fold is alone at its steps while the readout waits on the image, where this row's steps run beside the readout's.
  THE CUT ROWS: OF THE PHASE LEDGER'S THREE SHARED-MEMORY COSTS ONLY THE PANEL'S CUT IS WORTH TIME. The block form's
  window is 6,499 with the panel by stmatrix against 6,859, 7,385 with the steps' operands shuffled,
  7,182 issuing only the steps a list reaches, and 7,671 with all three; the read-ready ready pair's
  4,942, 5,825, 5,472 and 5,956 against 5,198. The two read-ready rows that issue only the steps a list reaches were taken with the steps serialized (ptxas C7520): their numbers are of a serialized form. All three cut the
  windows' shared-memory wavefronts by some 1,020 a CTA-window (the phase ledger: 4,238 to 3,216 and 3,758 to 2,735),
  and two of them lengthen the window: A WINDOW WITH A QUARTER LESS SHARED-MEMORY TRAFFIC IS NOT SHORTER, SO THAT PIPE
  IS NOT WHAT HOLDS IT. The shuffles put four instructions before each step's issue and the skip a branch around it,
  both on the fold's steps' issue; the panel's stores are the readout's, beside its steps. READ AS, not isolated: the
  window is held by the fold's and the readout's chain through the image, its waits and issue, not by a pipe's bytes.
  THE AHEAD ROWS -- the fold's steps' operand lines loaded two steps ahead -- DO NOT MOVE THE WINDOW: 6,506
  against 6,499 for the block form with the panel by stmatrix, and 4,947 against 4,942 for the
  read-ready ready pair's. Alone, a fold step's issue IS its operands' load
  (the held-step row below: some 15 cycles a step held against 48); in the composed rows the loads ahead -- several in
  flight in their binary -- leave the issue where it was (the clocked ahead row: some 58 cycles a step either way).
  THE NO-READ-STEPS ROW -- the readout issuing no step -- SHOWS THE FOLD'S STEPS ARE HELD BY THEIR OWN: clocked, the window
  is 4,600 against 5,296 with the readout's steps, and a fold warp's steps' issue 1,992 against
  2,264: some 62 cycles a step with no readout's step beside it. The two fold warpgroups' 64 steps
  a window are 16 columns each, reading their 2.5 KB of operands at 20 cycles ([`wgmma_width`](wgmma_width.md)): 1,280
  cycles of the shared memory's reads between them. READ AS, not isolated: the fold's issue is its two warpgroups'
  narrow steps' operand reads, and a step's own issue beside them.
  THE SERVE ROWS -- the drain warps expanding the fold's slots a slot ahead -- LOSE: the window is 6,695 against
  6,499 for the same form with the fold expanding its own, where some 5,700 was predicted. Clocked (6,896), the
  path moves to the drain warps: a drain warp waits 1,243 of it (`lab path`: busy 82%), a fold warp
  3,405 on its slots' expansion -- the fold left the path and the serve warps took it. The expansion
  cost them some 2,600 cycles a window where some 1,000 was predicted: a slot is expanded a box at a time, an entry a
  thread, its load, products, stores, fence and arrival in turn, with nothing in flight across boxes or slots.
  THE MEETING ROWS -- the rendezvous an mbarrier's -- DO NOT MOVE IT: the window is 5,354 against
  5,289 with even lists and 6,937 against 6,859 in the block form; clocked, a fold warp
  is 463 and 533 at its rendezvous (that row's folds at 168 registers), and
  717 and 588 in the block form. The time is not the named barrier's own.
  WITH EVERY WAIT POLLED, not suspended (even lists): the window is 5,520 against 5,289;
  clocked (6,018), a fold warp's rendezvous are 377 and 486, its waits 292 on the readout's
  read and 355 on its bytes.
  A slot's steps are a
  fixed two rounds: ISSUING ONLY THE STEPS ITS LISTS HAVE, a second-round step a trip of one or none, WAS MEASURED
  AND LOST (the experiment at 758f0d43, its record that commit's: 7,022 against 6,822 composed, the fold's side
  alone 4,385 against 3,741), and its rows are not kept. NOT ASKED: the drain warps at 88 registers under this
  fold (they are at 72 here); lists past two rounds.
- THE MIRRORED READOUT'S SUMS TO THE PANEL ARE ITS LARGEST PHASE (the even lists' form, clocked): a readout warp is
  1,802 cycles a window rounding its sums, storing them a half each and arriving -- its
  2,164 less its 362 waiting on the drain -- against 1,240 in its steps and
  1,470 forming its coefficients. NOT ASKED: the panel laid so a thread's two sums are one word's store.
- THE OUTPUT PATH ON FOUR WARPS (the chunk-of-8 form, the fold issuing its own copies, four drain warps): the window
  is 5,289 cycles against 5,532 with two drain warps and a producer; clocked 5,816 against 5,662: a readout
  warp waits 362 on the drain against 664, a drain warp 3,177 on the panel against 649 -- and A FOLD WARP WAITS
  467 of the window, on the readout's read and on its bytes, against 2,120: the folds, which waited a third
  of a window, now wait a tenth; no clock says in what.
- WITH THE DRAIN WARPS ROUNDING the window is 5,309 against 5,289: nothing. Clocked (5,531), a readout
  warp's unclocked remainder is 1,790 against 2,427 and it waits 1,253 on the image against 317: what the
  readout sheds it waits out on the folds' image.
- THE MIRRORED READOUT WITH A CHUNK OF 8 DIGITS: the window is 5,532 cycles against 5,576 at 4. Clocked (5,662
  against 6,016): a readout warp is 1,024 forming and 127 in its steps against 1,083 and 156 -- half the
  passes, three quarters of the time -- and what its clocks do not name is 3,592 against 4,179: its
  rows' rounding and stores are a value's, not a pass's. It waits 664 on the drain against 238, and a drain
  warp waits 649 on the panel against 2,381: the two drain warps are near the window's whole length at
  work (their warpgroup at 88 registers here, 96 there; no row parts that from the two panels handed at once).
- WITH THE READOUT A SUM A BOX (the fold expanding, a producer issuing) THE WINDOW IS LONGER: 6,248 cycles against
  5,966. Clocked (6,565 against 6,509): a readout warp is 3,460 in its coefficients and steps against 3,830 --
  two chains of 4 steps are a tenth under one of 8, so the chain was not what its steps cost -- and it waits
  802 on the drain against 244, a drain warp waiting 330 on the panel against 2,523. The row's serve
  warps are at 80 registers where the other's are at 112, and its drain is what the window then waits on; no row
  parts the drain's registers from the readout's second sum. NOT ASKED: the readout's steps by what they
  multiply -- a step is 64 rows of which one digit's 16 are not zero; its coefficients formed by warps that
  wait; the fold-expands form on the uneven stream; a truncated step's pads left out of the expanding.
- THE TABLES ONCE-READ COST 2.3 TO 2.5 TIMES THE WINDOW ON THIS BOX (in cycles, which for a row bound by global
  memory follow the card's clock): 17,739 cycles composed against 7,032, the winner's form
  16,111 against 6,659, the fold's side alone 11,936 against 5,091 -- every access a fresh row of 128 MB of
  values, the model of a large table whose rows are each read once.
- THE DUMP IS PRICED IN THIS SESSION: the fold's state stored out in every call, as every record before 79ec1b08
  ran, puts the composed row at 8,173 against 7,032 -- one binary, one box, one session.
- FIVE WARPGROUPS LOSE COMPOSED, 7,736 against 7,032: 640 threads leave four producers and four drain warps 40
  registers a thread, the fold waits on its producers 5,579 cycles a window, and the lean readout the count
  forces is 3,864 in its steps and their forming where one commit group's are 3,537. Four drain warps do
  help the read side alone, 5,045 against 5,560.
- THE LEAN READOUT WITH TWO-BATCH PRODUCERS IS 6,784 COMPOSED against 7,032: a readout warp is 4,467 cycles
  a window in its steps and their forming. The two changes are not separated.
- THE READOUT DRAINING ITS OWN ROWS LOSES COMPOSED, 7,432 against 7,032: a readout warp is 2,834 cycles a
  window in its 64 adds, and the window becomes the readout's -- the fold waits on its read 3,769. A warp that
  issues an add is held until the SM passes it: adds issued by the readout hold the readout.
- COMPOSED, THE PRODUCERS' LOADS ARE NOW THE LARGEST SINGLE ACCESS: with them out the window is 6,432 against 7,032, and
  those loads are three fifths of the fold's side alone -- 2,412 cycles a window without them against 5,091, some
  167 cycles a batch of a producer's 9 load instructions; their coefficient arithmetic is 398 of it
  (4,693 without). The drains' adds out, the window is 6,716; the image's stores out, 6,705.
- A PARKED FOLD WARP COSTS THE READ SIDE -6 CYCLES, A PUBLISHING ONE 1,179: the read side alone is 5,560
  with the fold warps publishing and 4,381 with no fold warp running, while one parked asleep beside it costs
  -6 cycles and one spinning 121. What of the publishing fold's hand-off holds it is NOT ISOLATED
  (its stores are 570 of it; its fence nothing).
- THE READOUT'S OWN BYTE LOADS ARE A SIXTH OF ITS STEPS' SPAN: 8 instructions a chunk, each lane's 4 bytes of a
  different row; without them a readout warp's steps are 2,131 cycles a window against 2,579. The read side alone
  is then the drain's -- 5,003 against 5,560, the readout waiting on the drain 273 -- at some
  27 cycles an add a drain warp, where with no fold warp running an add is 26.
- ONE WIDE LOAD A WARP FOR THE READOUT'S BYTES is the read side alone without those loads -- 5,050 cycles a window
  against 5,003 with them out and 5,560 as they were -- and little composed: 6,977 against 7,032, where the
  byte loads out are 6,665 and the drains' adds out with them 6,414.
- THE READOUT FORMS ITS STEPS' A IN 1,101 TO 1,341 CYCLES A WINDOW, 275 to 335 a chunk: the coefficients' rounds
  and the 32 words held.
- THE STEPS THEMSELVES ARE OVER THEIR OWN TIME with nothing beside them but the drains: 551 cycles a chunk (the
  fence, 8 steps, the commit, 8 loads issued, the wait) where 8 steps of 64 columns into one accumulator from
  shared memory are 319 ([`wgmma_width`](wgmma_width.md)). Not isolated here.
- EVERY ROLE ISSUES AN INSTRUCTION EVERY 3.5 TO 7 CYCLES of its busy time (the static ledger's counts a trip: a
  producer 168 a batch, a drain warp 339 a chunk, the readout 284 a chunk, a fold warp 379 a window): no role is
  bound by the instructions it issues, and what holds each is not isolated.
- THE COUNTERS: this record's run read none; the phase ledger below holds the windows' counts.

THE LAUNCH'S EDGES (one CTA, `sm_90a.blocks1.json` and `sm_90a.blocks1.trace/`, the experiment at
4094bdcf, 7 repetitions; the row whose name with `, the launch's two edges stamped` exists, its plain
and coarse twins by its name with that suffix removed and replaced): each warp marks its entry at the kernel's top and its exit at its role's end, on both clocks.
- THE WINDOW, cycles a window a CTA: the plain row's 5,288; the edges row's 5,300, the same as the plain row's
  (+12 cycles); the coarse row's (its phases' starts and the edges) 5,479, 3.6% over the plain row's
  (+192 cycles).
- THE LAUNCH, CTA 0's, from the trace (the first warp's entry to the last warp's exit): the edges row's
  340,901 cycles, 198,272 ns, 1,719.4 cycles a microsecond; the coarse row's 355,725 cycles,
  206,336 ns, 1,724.0 cycles a microsecond. The warps enter within 40 (the coarse row's: 47) cycles
  and exit within 1,040 (1,409).

Record: `measure/lab/results/carry_window/sm_90a.json` (and, at one CTA with the edges row, `sm_90a.blocks1.json` and its trace `sm_90a.blocks1.trace/`); the phase ledger's dumps:
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