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

THE STREAM, TOKEN-ORDERED (not the ruled stream, which is laid as the carry's steps --
[`hopper-carry.md`](../architecture/hopper-carry.md#the-gather-stream) -- and whose pass is not built). The same
two kernels fed lists BY SPARSE BLOCK, not by digit: a block's live tokens once each, in token order, padded to 16
only at the block's end. The values kernel writes RECORDS -- a token's 8 value lines and then its line of the
block's 16 sparse bytes, 9 lines a token -- that the block's CTAs share; the dense plane is the block-major one, so a
CTA's dense block of its sparse block's tokens is one run. No digit's step is padded in global memory. At 512
threads, moved and flat, on every multiprocessor and on a quarter. THE RING: a carry's CTAs read one window's
stream at a time, so the pass's planes can be a ring the cache holds. The stream's kernels on 16 and on 8
multiprocessors, each CTA rewriting its own region every round, beside the same CTAs writing a new region a
round: the same reads, eight times the planes' footprint (at the sparse cell 16 CTAs' rings are some 30 MB of a
cache of 50 and their new regions 240; at 32 live digits neither fits).

THE BLOCK STREAM, ITS LISTS THE PASS'S OWN (the ruled stream's shared part --
[`hopper-carry.md`](../architecture/hopper-carry.md#the-gather-stream); 4 to 5 live digits a token, 512 threads).
The pass is handed the sparse level's MASK digit-major -- a digit's 1,024 bits of a window, 32 words -- and no
list. A block of 16 digits at a time, a warp a digit and a lane 32 tokens' bits: a lane's live tokens go to their
ranks in the digit's list, the lanes' counts before it by the warp's scan and the bits under each by a count, 32
fixed trips whatever is live. A UNIT IS A SLOT, 8 digits' lists: every warp scans the slot's layout, and a thread
a (list, place) writes the carry's image -- an entry's 8 value lines to its list's runs, a brick a run, as long as
the list; its word (its dense line and its second run's line in a CTA's dense run); its sparse byte; a pad's
line; 16 threads a step's four halves; the head -- then one copy out and the slot's plan, its first line and
its bytes. Every word, byte, line, pad, step and plan of the first round is checked on every multiprocessor
against lists the host makes from the same mask. The lab's bounds: a list is at most 32 entries and a slot 150
(this cell's are 16 to 19 and 128 to 143). NOT IN THE ROW: the dense lines, a run a CTA's dense block.

MEASURED (H100 80GB HBM3, 132 SMs, vast.ai instance 60614444, driver 595.91.07; 2026-11-14, the experiment at
46f4e03a; 7 repetitions; every arranged line and pad of every moved form exact on every multiprocessor; the
median, SM-cycles a token for one side's sparse level; every figure below is this one run's):

THE ARRANGED STEPS, by digit:

| the cell | the plane | threads | a flat stream, a fresh window a round | its lines moved, a fresh window | moved, the same windows |
|---|---|---|---|---|---|
| 4 to 5 live digits a token | values | 256 | 141 | 155 | 155 |
| 4 to 5 live digits a token | values | 512 | 134 | 153 | 152 |
| 4 to 5 live digits a token | values | 1024 | 136 | 164 | 164 |
| 4 to 5 live digits a token | dense | 256 | 130 | 141 | 140 |
| 4 to 5 live digits a token | dense | 512 | 151 | 137 | 140 |
| 4 to 5 live digits a token | dense | 1024 | 132 | 147 | 147 |
| 32 live digits a token | values | 256 | 428 | 583 | 594 |
| 32 live digits a token | values | 512 | 360 | 597 | 598 |
| 32 live digits a token | values | 1024 | 391 | 679 | 679 |
| 32 live digits a token | dense | 256 | 550 | 594 | 551 |
| 32 live digits a token | dense | 512 | 480 | 577 | 551 |
| 32 live digits a token | dense | 1024 | 474 | 593 | 585 |

BOTH FORMS at 512 threads, a fresh window a round:

| the cell | the plane | flat, every multiprocessor | flat, a quarter | moved, every multiprocessor | moved, a quarter |
|---|---|---|---|---|---|
| 4 to 5 live digits a token | values plane | 134 | 85 | 153 | 130 |
| 4 to 5 live digits a token | dense plane | 151 | 89 | 137 | 111 |
| 4 to 5 live digits a token | dense plane, six buffers out | 157 | 102 | 166 | 138 |
| 4 to 5 live digits a token | stream's values and sparse lines | 97 | 61 | 101 | 83 |
| 4 to 5 live digits a token | stream's dense lines | 87 | 63 | 98 | 84 |
| 32 live digits a token | values plane | 360 | 270 | 597 | 578 |
| 32 live digits a token | dense plane | 480 | 315 | 577 | 421 |
| 32 live digits a token | dense plane, six buffers out | 608 | 342 | 667 | 528 |
| 32 live digits a token | stream's values and sparse lines | 244 | 130 | 215 | 191 |
| 32 live digits a token | stream's dense lines | 234 | 145 | 225 | 174 |

THE STREAM ON A FEW MULTIPROCESSORS, 512 threads, a fresh window a round:

| the cell | the plane | CTAs | flat, a ring | flat, a new region a round | moved, a ring | moved, a new region a round |
|---|---|---|---|---|---|---|
| 4 to 5 live digits a token | stream's values and sparse lines | 16 | 57.0 | 57.7 | 81.0 | 81.5 |
| 4 to 5 live digits a token | stream's values and sparse lines | 8 | 55.9 | 56.9 | 80.5 | 81.0 |
| 4 to 5 live digits a token | stream's dense lines | 16 | 55.8 | 56.6 | 81.5 | 81.9 |
| 4 to 5 live digits a token | stream's dense lines | 8 | 55.2 | 55.9 | 81.0 | 81.8 |
| 32 live digits a token | stream's values and sparse lines | 16 | 129.4 | 134.2 | 191.3 | 192.5 |
| 32 live digits a token | stream's values and sparse lines | 8 | 122.6 | 132.2 | 190.9 | 191.7 |
| 32 live digits a token | stream's dense lines | 16 | 143.5 | 143.6 | 172.8 | 172.7 |
| 32 live digits a token | stream's dense lines | 8 | 144.2 | 144.4 | 172.6 | 172.8 |

WHAT IT SHOWS.
- THE STREAM COSTS 199 SM-CYCLES A TOKEN a side at 4 to 5 live digits a token with every multiprocessor at work
  (101 its values and sparse lines, 98 its dense lines) and 441 at 32; the arranged steps 290 and 1,174.
  At 32 live digits it is 0.38 of the arranged steps: a token is in at most 16 sparse blocks however many of
  its digits are live.
- WITH THE CARD UNLOADED IT IS 163, AND THERE IT IS ITS OWN CTA THAT BINDS. On 16 or on 8 multiprocessors the stream is
  163 and 162 at the sparse cell (167 on a quarter of them), its flat form 113: the cost stops falling with
  the load, and a third of it is the moving of lines.
- A RING THE CACHE HOLDS DOES NOT MOVE THE PASS'S WRITES ON AN UNLOADED CARD, at the sparse cell: there 16 CTAs'
  rings fit the cache and their new regions do not, and the moved forms cost the same (the table's sparse rows).
  At 32 live digits neither footprint fits; the values kernel's flat rings read a few cycles a token under its
  new regions and the dense kernel's the same as theirs. NOT ASKED: a ring on a loaded card, where the card's memory binds; a reader of the ring; a discard.
- SIX BUFFERS OUT LOSE TO THREE everywhere (the arranged dense plane): a copy out half the size, twice as often,
  costs more, and more of them in flight buys nothing. With the flat form at 114 thousand cycles a window for
  some 2.7 MB in and out, a unit's trip -- its barrier, its list a unit ahead, its copy out waited on -- is the
  suspect; no row here parts those.
- WHAT IT DOES NOT SAY: the lists are the host's -- no pass that makes a block's live tokens, their indices and
  their mask slices is built or priced anywhere -- and the stream's kernels write neither; the read side's stream
  -- no values -- is not built; a dense level and a clustered one are unmeasured; the pass beside a running carry
  is unmeasured (the carry filling from a stream the host laid is [`carry_window`](carry_window.md)'s); the same kernels on another rented card are not compared.

Record: `measure/lab/results/gather_pass/sm_90a.json`.
