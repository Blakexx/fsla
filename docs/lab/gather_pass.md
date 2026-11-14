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

THE STREAM (the ruled form, [`hopper-carry.md`](../architecture/hopper-carry.md#the-gather-stream)). The same
two kernels fed lists BY SPARSE BLOCK, not by digit: a block's live tokens once each, in token order, padded to 16
only at the block's end. The values kernel writes RECORDS -- a token's 8 value lines and then its line of the
block's 16 sparse bytes, 9 lines a token -- that the block's CTAs share; the dense plane is the block-major one, so a
CTA's dense block of its sparse block's tokens is one run. No digit's step is padded in global memory. At 512
threads, moved and flat, on every multiprocessor and on a quarter.

MEASURED (H100 80GB HBM3, 132 SMs, vast.ai instance 60612875, driver 580.173.02; 2026-11-14, the experiment at
27b94614; 7 repetitions; every arranged line and pad of every moved form exact; the median, SM-cycles a
token for one side's sparse level; every figure below is this one run's):

THE ARRANGED STEPS, by digit:

| the cell | the plane | threads | a flat stream, a fresh window a round | its lines moved, a fresh window | moved, the same windows |
|---|---|---|---|---|---|
| 4 to 5 live digits a token | values | 256 | 141 | 160 | 160 |
| 4 to 5 live digits a token | values | 512 | 142 | 160 | 160 |
| 4 to 5 live digits a token | values | 1024 | 146 | 170 | 170 |
| 4 to 5 live digits a token | dense | 256 | 194 | 197 | 196 |
| 4 to 5 live digits a token | dense | 512 | 194 | 197 | 196 |
| 4 to 5 live digits a token | dense | 1024 | 195 | 199 | 198 |
| 32 live digits a token | values | 256 | 490 | 606 | 606 |
| 32 live digits a token | values | 512 | 495 | 614 | 614 |
| 32 live digits a token | values | 1024 | 510 | 687 | 687 |
| 32 live digits a token | dense | 256 | 857 | 874 | 873 |
| 32 live digits a token | dense | 512 | 858 | 875 | 872 |
| 32 live digits a token | dense | 1024 | 862 | 881 | 879 |

BOTH FORMS at 512 threads, a fresh window a round:

| the cell | the plane | flat, every multiprocessor | flat, a quarter | moved, every multiprocessor | moved, a quarter |
|---|---|---|---|---|---|
| 4 to 5 live digits a token | values plane | 142 | 84 | 160 | 130 |
| 4 to 5 live digits a token | dense plane | 194 | 89 | 197 | 111 |
| 4 to 5 live digits a token | dense plane, six buffers out | 200 | 102 | 206 | 139 |
| 4 to 5 live digits a token | stream's values and sparse lines | 102 | 60 | 111 | 83 |
| 4 to 5 live digits a token | stream's dense lines | 127 | 62 | 132 | 84 |
| 32 live digits a token | values plane | 495 | 299 | 614 | 581 |
| 32 live digits a token | dense plane | 858 | 330 | 875 | 426 |
| 32 live digits a token | dense plane, six buffers out | 898 | 357 | 950 | 526 |
| 32 live digits a token | stream's values and sparse lines | 247 | 143 | 263 | 194 |
| 32 live digits a token | stream's dense lines | 374 | 148 | 378 | 175 |

WHAT IT SHOWS.
- THE STREAM COSTS 243 SM-CYCLES A TOKEN a side at 4 to 5 live digits a token (111 its values and sparse lines,
  132 its dense lines) and 641 at 32; the arranged steps 357 and 1,489. On a quarter of the multiprocessors
  the stream is 167 and 369. At 32 live digits it is 0.43 of the arranged steps: a token is in at most 16
  sparse blocks however many of its digits are live.
- THE COPIES ARE THE PRICE, in both forms. A flat stream of the same copies, no line moved, is nine tenths of
  the moved form on every multiprocessor; the threads' count does not move it, and neither does a window the
  cache has seen -- a window's tables and planes are past the cache in either mode.
- THE CARD'S MEMORY BINDS THEM, NOT A CTA'S QUEUE. On a quarter of the multiprocessors a CTA's flat stream costs
  0.59 and 0.46 of its cost with every multiprocessor at work (the arranged planes; 0.59 and 0.49 the
  stream's). Six buffers out lose to three everywhere. With the card less loaded the moving of lines shows:
  23 and 21 over the stream's flat forms.
- THE STREAM IS NOT ITS BYTES' SHARE OF THE STEPS. At the sparse cell it writes 0.6 of the steps the arranged form
  does (a block's tokens once each, no digit padded) and costs 0.68: both read the same tables, some 0.9 MB a
  window, which the stream's smaller planes leave the larger part.
- A PASS BOUND BY A CARD'S MEMORY IS PRICED BY ITS CARD: rented cards have read these kernels 20 to 40% apart, in
  runs of different builds that this page does not compare.
- WHAT IT DOES NOT SAY: the lists are the host's (a counts pass and the lists' making are priced by
  [`facts_lists`](facts_lists.md), not here); the stream's token indices and mask slices are not written; the read
  side's stream -- no values -- is not built; a dense level and a clustered one are unmeasured; the pass beside a
  running carry, and the carry filling from the stream, are unmeasured.

Record: `measure/lab/results/gather_pass/sm_90a.json`.
