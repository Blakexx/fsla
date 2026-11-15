# `gather_pass`

Mirrors `measure/lab/experiments/gather_pass.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The carry lands a step's whole entries by bulk copy from a buffer some pass arranged
([`carry_window`](carry_window.md)). [`facts_lists`](facts_lists.md) built that pass in its plainest form -- a
live (token, digit) pair's lines moved by its token's thread, a load and a store of global memory a line -- and it
cost twice the carry's window. What does the same re-arrangement cost when every byte crosses global memory in a
bulk copy and a line is moved in shared memory, and how far is that from the floor: the same copies, no line moved?

THE FORM. A window of 1,024 tokens, one side's sparse level of 256 digits. The pass is handed the window's STEP
LISTS -- the lists pass's output padded to steps of 16 entries, an entry its token and its digit's low 4 bits, a
pad the token 1,024 -- and each sparse block's first step. It writes two planes:
- THE VALUES PLANE: a step is a tile of 137 lines, 8 bricks of 17 lines -- line `k` of brick `b` entry k's
  channels `8 b` on -- then a line of the 16 entries' bytes on the sparse digit. A CTA holds the window's values
  (128 KB, 16 copies in) and, a sparse block of 16 digits at a time, that block's rows of the window (16 KB); a
  unit is 8 steps of one sparse block, a line a (step, brick, entry).
- THE DENSE PLANE, block-major: a dense block's 16 lines a step, a block's steps side by side, so a carry CTA's
  dense block of consecutive steps is one run. A CTA holds one dense block's rows of the window (16 KB) and walks
  every step, 192 a unit, once a dense block.
A pad's coefficient lines are zero -- token 1,024 is a zero line after each table -- so a short list's last step
adds nothing. A unit is a trip: one thread issues the copy out of the unit before and the copies in of the unit
after; every thread waits on its unit's barriers and moves its lines shared to shared; three buffers out in turn.
Every writer fences its own lines before the barrier its copy out follows. Each plane at 256, 512 and 1,024
threads, its lines moved and as A FLAT STREAM of the same copies, its rounds the same windows or a fresh window a
round, at 4 to 5 live digits a token and at 32. TWO CONTROLS at 512 threads say what binds the copies: the same
kernels with CTAs on A QUARTER of the multiprocessors, and the dense plane with SIX buffers out of half the size
(four copies out in flight where three buffers allow one). Every line and pad of each moved form's first round is
checked on every multiprocessor, each byte of the tables its own position's.

THE STREAM, TOKEN-ORDERED (not the design's stream, which is laid as the carry's blocks --
[`hopper-carry.md`](../architecture/hopper-carry.md#the-gather-stream) -- and is the block stream's rows below). The same
two kernels fed lists BY SPARSE BLOCK, not by digit: a block's live tokens once each, in token order, padded to 16
only at the block's end. The values kernel writes RECORDS -- a token's 8 value lines and then its line of the
block's 16 sparse bytes, 9 lines a token -- that the block's CTAs share; the dense plane is the block-major one, so a
CTA's dense block of its sparse block's tokens is one run. No digit's step is padded in global memory. At 512
threads, moved and flat, on every multiprocessor and on a quarter. THE RING: a carry's CTAs read one window's
stream at a time, so the pass's planes can be a ring the cache holds. The stream's kernels on 16 and on 8
multiprocessors, each CTA rewriting its own region every round, beside the same CTAs writing a new region a
round: the same reads, eight times the planes' footprint (at the sparse cell 16 CTAs' rings are some 30 MB of a
cache of 50 and their new regions 240; at 32 live digits neither fits).

THE BLOCK STREAM, ITS LISTS THE PASS'S OWN (the design's stream's shared part, its coefficients' bytes --
[`hopper-carry.md`](../architecture/hopper-carry.md#the-gather-stream); 4 to 5 live digits a token, 512 threads).
THE QUESTION: every row above is fed lists the host made, and the one pass that made its own cost twice the
carry. What does the block stream cost when the pass is handed THE MASK ALONE and writes what the carry lands --
a slot's tables, its lists' value runs as long as the lists, its sparse bytes, its plan? The mask is digit-major,
a digit's 1,024 bits of a window; the pass turns it around in shared memory, a fixed count of trips whatever is
live, and no list crosses global memory. Checked on every multiprocessor against lists the host makes from the
same mask. The lab's bounds: a list is at most 32 entries and a slot 150 (this cell's are 16 to 19 and 128 to
143). NOT IN THE ROW: the dense lines, a run a CTA's dense block.

THE COEFFICIENT RUNS, IN THE STEP'S FORMAT (the design's stream's other part --
[`hopper-carry.md`](../architecture/hopper-carry.md#the-gather-stream); 4 to 5 live digits a token, 512 threads).
THE QUESTION: the carry lands its coefficients made; what does the pass that makes them cost? A CTA takes a
window a dense block at a time -- that block's rows of the window, 16 KB -- and for each slot writes its lists'
entries' coefficients: an entry's 16 dense digits of the block by its sparse digit, rounded to bf16, a line a
brick, the first bricks' runs and then the second's, each list's padded with zero lines to its steps; 32 bytes an
entry a dense block where its bytes are 16. The window's lists are made from the mask once, an entry kept as its
token and its sparse byte, and every slot's layout once; a dense block's rows are asked a dense block ahead. Every line is checked on every multiprocessor against the host's own product and
rounding. NOT IN THE ROW: the scales, the steps' operands' fields, the read side.

THE RULED WRITE SIDE ([`hopper-gather.md`](../architecture/hopper-gather.md)): an entry's sparse byte on its
VALUES and its coefficient line its dense digits WIDENED ALONE. Three rows more: the block stream's shared part
with an entry's 64 values multiplied by its sparse byte as they are written (a value whose bits are no finite
normal number is not compared); the coefficient runs FOUR SLOTS A TRIP, a buffer out their four rooms and its
copy four stores of one group, the product formed as above -- what the trips cost; and those four-slot runs with
no product, a line the entry's dense digits rounded to bf16 alone.

MEASURED (H100 PCIe 80GB, 114 SMs, Hyperstack spot VM 1161228, driver 580.196.07; 2026-11-15, the experiment at
e4bf499f; 7 repetitions; every arranged line and pad of every moved form, and every word, byte, line,
pad, step and plan of the block stream, exact on every multiprocessor; the median, SM-cycles a token for one
side's sparse level; every figure below is this one run's):

THE BLOCK STREAM, its lists the pass's own (4 to 5 live digits a token, 512 threads, a fresh window a round):

| every multiprocessor | a quarter |
|---|---|
| 158 | 147 |

THE COEFFICIENT RUNS in the step's format, every dense block's (4 to 5 live digits a token, 512 threads, a fresh window a round):

| every multiprocessor | a quarter |
|---|---|
| 722 | 697 |

THE RULED WRITE SIDE's rows (every multiprocessor, a fresh window a round):

| the row | SM-cycles a token |
|---|---|
| the shared part, an entry's values multiplied by its sparse byte | 162 |
| the coefficient runs, the product formed, four slots a trip | 613 |
| the coefficient runs, the dense digits widened alone, four slots a trip | 617 |

THE ARRANGED STEPS, by digit:

| the cell | the plane | threads | a flat stream, a fresh window a round | its lines moved, a fresh window | moved, the same windows |
|---|---|---|---|---|---|
| 4 to 5 live digits a token | values | 256 | 162 | 173 | 173 |
| 4 to 5 live digits a token | values | 512 | 163 | 173 | 172 |
| 4 to 5 live digits a token | values | 1024 | 165 | 178 | 178 |
| 4 to 5 live digits a token | dense | 256 | 227 | 223 | 220 |
| 4 to 5 live digits a token | dense | 512 | 225 | 225 | 213 |
| 4 to 5 live digits a token | dense | 1024 | 222 | 233 | 225 |
| 32 live digits a token | values | 256 | 554 | 623 | 623 |
| 32 live digits a token | values | 512 | 556 | 625 | 625 |
| 32 live digits a token | values | 1024 | 563 | 687 | 687 |
| 32 live digits a token | dense | 256 | 1,020 | 1,034 | 1,035 |
| 32 live digits a token | dense | 512 | 1,020 | 1,035 | 1,033 |
| 32 live digits a token | dense | 1024 | 1,021 | 1,040 | 1,037 |

BOTH FORMS at 512 threads, a fresh window a round:

| the cell | the plane | flat, every multiprocessor | flat, a quarter | moved, every multiprocessor | moved, a quarter |
|---|---|---|---|---|---|
| 4 to 5 live digits a token | values plane | 163 | 83 | 173 | 130 |
| 4 to 5 live digits a token | dense plane | 225 | 87 | 225 | 110 |
| 4 to 5 live digits a token | dense plane, six buffers out | 232 | 99 | 240 | 139 |
| 4 to 5 live digits a token | stream's values and sparse lines | 118 | 60 | 122 | 83 |
| 4 to 5 live digits a token | stream's dense lines | 142 | 60 | 153 | 83 |
| 32 live digits a token | values plane | 556 | 296 | 625 | 578 |
| 32 live digits a token | dense plane | 1,020 | 316 | 1,035 | 420 |
| 32 live digits a token | dense plane, six buffers out | 1,057 | 332 | 1,082 | 523 |
| 32 live digits a token | stream's values and sparse lines | 282 | 141 | 291 | 194 |
| 32 live digits a token | stream's dense lines | 431 | 144 | 437 | 173 |

THE STREAM ON A FEW MULTIPROCESSORS, 512 threads, a fresh window a round:

| the cell | the plane | CTAs | flat, a ring | flat, a new region a round | moved, a ring | moved, a new region a round |
|---|---|---|---|---|---|---|
| 4 to 5 live digits a token | stream's values and sparse lines | 16 | 56.7 | 58.6 | 81.0 | 81.9 |
| 4 to 5 live digits a token | stream's values and sparse lines | 8 | 55.7 | 56.8 | 80.2 | 80.9 |
| 4 to 5 live digits a token | stream's dense lines | 16 | 55.8 | 56.8 | 81.7 | 82.0 |
| 4 to 5 live digits a token | stream's dense lines | 8 | 55.2 | 55.6 | 81.2 | 81.6 |
| 32 live digits a token | stream's values and sparse lines | 16 | 133.4 | 133.4 | 191.5 | 191.6 |
| 32 live digits a token | stream's values and sparse lines | 8 | 125.2 | 130.3 | 190.6 | 191.1 |
| 32 live digits a token | stream's dense lines | 16 | 142.8 | 142.9 | 172.1 | 172.2 |
| 32 live digits a token | stream's dense lines | 8 | 142.6 | 142.6 | 171.7 | 171.9 |

WHAT IT SHOWS.
- MAKING ITS OWN LISTS, THE BLOCK STREAM'S SHARED PART COSTS 158 SM-CYCLES A TOKEN at 4 to 5 live digits a token
  with every multiprocessor at work (147 on a quarter of them): the mask turned around in the kernel, the tables,
  the value runs and the sparse bytes, one copy out a slot. The arranged values plane, its lists the host's, is
  173 -- the same entries' lines moved, a digit's steps padded: THE PASS'S OWN LISTS AND TABLES ARE -15 ON IT
  (the token-ordered stream's values and sparse lines, 122, move a token's lines once a sparse block, not
  once a digit). It falls little with the card unloaded: its own CTA binds it. NOT IN IT: the dense lines (the arranged dense plane is
  225, the stream's 153); a cell past its bounds -- a list over 32 entries, a slot over 150;
  32 live digits a token.
- THE COEFFICIENTS IN THE STEP'S FORMAT COST 722 SM-CYCLES A TOKEN, every dense block's (697 on a quarter of
  the multiprocessors: the kernel's own work binds it), where the arranged dense plane -- the same entries' bytes
  moved, no product -- is 225: THE BLOCK STREAM'S WRITE SIDE IS 880 with its shared part, more than
  the carry's window a token. A unit here is one slot's runs for one dense block, 512 trips a window, each with
  its fence, its barrier and its copy out; the dense plane's unit is 192 steps. NOT ISOLATED: the trips from the
  products (the rows below part them). NOT IN IT: the scales; the steps' operands' fields; the read side.
- THE ENTRY'S SCALAR ON ITS VALUES COSTS +4: the shared part is 162 with an entry's 64 values multiplied
  by its sparse byte, 158 without. THE COEFFICIENT RUNS ARE NEITHER THEIR PRODUCT NOR THEIR TRIPS: four slots a
  trip are 613 where one is 722, and with no product at all -- the dense digits widened alone, the ruled
  form -- 617. THE RULED WRITE SIDE IS 778 SM-CYCLES A TOKEN. NOT ISOLATED: what the runs' kernel is
  bound by -- its lines are twice the arranged dense plane's bytes, read through a list's entries.
- THE TOKEN-ORDERED STREAM COSTS 274 SM-CYCLES A TOKEN a side at 4 to 5 live digits a token with every multiprocessor at work
  (122 its values and sparse lines, 153 its dense lines) and 728 at 32; the arranged steps 397 and 1,660.
  At 32 live digits it is 0.44 of the arranged steps: a token is in at most 16 sparse blocks however many of
  its digits are live.
- WITH THE CARD UNLOADED IT IS 163, AND THERE IT IS ITS OWN CTA THAT BINDS. On 16 or on 8 multiprocessors the stream is
  163 and 161 at the sparse cell (166 on a quarter of them), its flat form 113: the cost stops falling with
  the load, and the rest of it is the moving of lines.
- A RING THE CACHE HOLDS DOES NOT MOVE THE PASS'S WRITES ON AN UNLOADED CARD, at the sparse cell: there 16 CTAs'
  rings fit the cache and their new regions do not, and the moved forms cost the same (the table's sparse rows).
  NOT ASKED: a ring on a loaded card, where the card's memory binds; a reader of the ring; a discard.
- SIX BUFFERS OUT AGAINST THREE (the arranged dense plane): 240 against 225 moved at the sparse cell,
  1,082 against 1,035 at 32 live digits.
- WHAT IT DOES NOT SAY: the block stream's read side -- no values -- is not built; its head's operands' fields are not written; the
  token-ordered stream's and the arranged steps' lists are the host's; a dense level and a clustered one are
  unmeasured; the pass beside a running carry is unmeasured (the carry filling from a stream the host laid is
  [`carry_window`](carry_window.md)'s); the same kernels on another rented card are not compared.

Record: `measure/lab/results/gather_pass/sm_90a.json`.
