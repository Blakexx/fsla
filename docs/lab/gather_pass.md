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

MEASURED (H100 80GB HBM3, 132 SMs, vast.ai instance 60703552, driver 580.173.02; 2026-11-14, the experiment at
f6a6f250; 7 repetitions; every arranged line and pad of every moved form, and every word, byte, line,
pad, step and plan of the block stream, exact on every multiprocessor; the median, SM-cycles a token for one
side's sparse level; every figure below is this one run's):

THE BLOCK STREAM, its lists the pass's own (4 to 5 live digits a token, 512 threads, a fresh window a round):

| every multiprocessor | a quarter |
|---|---|
| 161 | 154 |

THE ARRANGED STEPS, by digit:

| the cell | the plane | threads | a flat stream, a fresh window a round | its lines moved, a fresh window | moved, the same windows |
|---|---|---|---|---|---|
| 4 to 5 live digits a token | values | 256 | 142 | 161 | 161 |
| 4 to 5 live digits a token | values | 512 | 143 | 160 | 160 |
| 4 to 5 live digits a token | values | 1024 | 147 | 171 | 171 |
| 4 to 5 live digits a token | dense | 256 | 195 | 199 | 198 |
| 4 to 5 live digits a token | dense | 512 | 195 | 199 | 198 |
| 4 to 5 live digits a token | dense | 1024 | 196 | 200 | 199 |
| 32 live digits a token | values | 256 | 493 | 611 | 611 |
| 32 live digits a token | values | 512 | 498 | 615 | 615 |
| 32 live digits a token | values | 1024 | 514 | 688 | 687 |
| 32 live digits a token | dense | 256 | 863 | 880 | 879 |
| 32 live digits a token | dense | 512 | 864 | 880 | 879 |
| 32 live digits a token | dense | 1024 | 867 | 887 | 886 |

BOTH FORMS at 512 threads, a fresh window a round:

| the cell | the plane | flat, every multiprocessor | flat, a quarter | moved, every multiprocessor | moved, a quarter |
|---|---|---|---|---|---|
| 4 to 5 live digits a token | values plane | 143 | 85 | 160 | 130 |
| 4 to 5 live digits a token | dense plane | 195 | 90 | 199 | 111 |
| 4 to 5 live digits a token | dense plane, six buffers out | 202 | 102 | 207 | 140 |
| 4 to 5 live digits a token | stream's values and sparse lines | 103 | 61 | 112 | 83 |
| 4 to 5 live digits a token | stream's dense lines | 128 | 63 | 133 | 84 |
| 32 live digits a token | values plane | 498 | 300 | 615 | 579 |
| 32 live digits a token | dense plane | 864 | 331 | 880 | 428 |
| 32 live digits a token | dense plane, six buffers out | 905 | 358 | 956 | 529 |
| 32 live digits a token | stream's values and sparse lines | 248 | 144 | 264 | 194 |
| 32 live digits a token | stream's dense lines | 377 | 149 | 380 | 176 |

THE STREAM ON A FEW MULTIPROCESSORS, 512 threads, a fresh window a round:

| the cell | the plane | CTAs | flat, a ring | flat, a new region a round | moved, a ring | moved, a new region a round |
|---|---|---|---|---|---|---|
| 4 to 5 live digits a token | stream's values and sparse lines | 16 | 57.1 | 58.3 | 81.2 | 81.6 |
| 4 to 5 live digits a token | stream's values and sparse lines | 8 | 56.0 | 57.2 | 80.6 | 81.1 |
| 4 to 5 live digits a token | stream's dense lines | 16 | 56.1 | 57.1 | 81.8 | 82.2 |
| 4 to 5 live digits a token | stream's dense lines | 8 | 55.3 | 56.1 | 81.1 | 81.8 |
| 32 live digits a token | stream's values and sparse lines | 16 | 134.2 | 134.3 | 192.1 | 192.2 |
| 32 live digits a token | stream's values and sparse lines | 8 | 123.7 | 132.2 | 191.0 | 191.8 |
| 32 live digits a token | stream's dense lines | 16 | 143.9 | 143.7 | 172.8 | 172.8 |
| 32 live digits a token | stream's dense lines | 8 | 144.3 | 144.4 | 172.5 | 172.9 |

WHAT IT SHOWS.
- MAKING ITS OWN LISTS, THE BLOCK STREAM'S SHARED PART COSTS 161 SM-CYCLES A TOKEN at 4 to 5 live digits a token
  with every multiprocessor at work (154 on a quarter of them): the mask turned around in the kernel, the tables,
  the value runs and the sparse bytes, one copy out a slot. The arranged values plane, its lists the host's, is
  160 -- the same entries' lines moved, a digit's steps padded: THE PASS'S OWN LISTS AND TABLES ARE +1 ON IT
  (the token-ordered stream's values and sparse lines, 112, move a token's lines once a sparse block, not
  once a digit). It falls little with the card unloaded: its own CTA binds it. NOT IN IT: the dense lines (the arranged dense plane is
  199, the stream's 133); a cell past its bounds -- a list over 32 entries, a slot over 150;
  32 live digits a token.
- THE TOKEN-ORDERED STREAM COSTS 245 SM-CYCLES A TOKEN a side at 4 to 5 live digits a token with every multiprocessor at work
  (112 its values and sparse lines, 133 its dense lines) and 645 at 32; the arranged steps 359 and 1,495.
  At 32 live digits it is 0.43 of the arranged steps: a token is in at most 16 sparse blocks however many of
  its digits are live.
- WITH THE CARD UNLOADED IT IS 163, AND THERE IT IS ITS OWN CTA THAT BINDS. On 16 or on 8 multiprocessors the stream is
  163 and 162 at the sparse cell (167 on a quarter of them), its flat form 113: the cost stops falling with
  the load, and the rest of it is the moving of lines.
- A RING THE CACHE HOLDS DOES NOT MOVE THE PASS'S WRITES ON AN UNLOADED CARD, at the sparse cell: there 16 CTAs'
  rings fit the cache and their new regions do not, and the moved forms cost the same (the table's sparse rows).
  NOT ASKED: a ring on a loaded card, where the card's memory binds; a reader of the ring; a discard.
- SIX BUFFERS OUT AGAINST THREE (the arranged dense plane): 207 against 199 moved at the sparse cell,
  956 against 880 at 32 live digits.
- WHAT IT DOES NOT SAY: the block stream's dense lines and its read side -- no values -- are not built; the
  token-ordered stream's and the arranged steps' lists are the host's; a dense level and a clustered one are
  unmeasured; the pass beside a running carry is unmeasured (the carry filling from a stream the host laid is
  [`carry_window`](carry_window.md)'s); the same kernels on another rented card are not compared.

Record: `measure/lab/results/gather_pass/sm_90a.json`.
