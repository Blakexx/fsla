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

MEASURED (H100 80GB HBM3, 132 SMs, vast.ai instance 60601585, driver 610.57.04; 2026-11-14, the experiment at
ad82c249; 7 repetitions; every arranged line and pad exact; the median, SM-cycles a token for one side's
sparse level):

| the cell | the plane | threads | a flat stream, a fresh window a round | its lines moved, a fresh window | moved, the same windows |
|---|---|---|---|---|---|
| 4 to 5 live digits a token | values | 256 | 141 | 160 | 160 |
| 4 to 5 live digits a token | values | 512 | 142 | 159 | 159 |
| 4 to 5 live digits a token | values | 1024 | 145 | 170 | 170 |
| 4 to 5 live digits a token | dense | 256 | 194 | 197 | 196 |
| 4 to 5 live digits a token | dense | 512 | 194 | 197 | 196 |
| 4 to 5 live digits a token | dense | 1024 | 195 | 199 | 198 |
| 32 live digits a token | values | 256 | 488 | 606 | 605 |
| 32 live digits a token | values | 512 | 493 | 614 | 613 |
| 32 live digits a token | values | 1024 | 508 | 687 | 687 |
| 32 live digits a token | dense | 256 | 856 | 874 | 871 |
| 32 live digits a token | dense | 512 | 858 | 874 | 873 |
| 32 live digits a token | dense | 1024 | 860 | 881 | 879 |

| the cell | the plane (512 threads, a fresh window a round) | flat, every multiprocessor | flat, a quarter | moved, every multiprocessor | moved, a quarter |
|---|---|---|---|---|---|
| 4 to 5 live digits a token | values | 142 | 85 | 159 | 130 |
| 4 to 5 live digits a token | dense | 194 | 89 | 197 | 111 |
| 4 to 5 live digits a token | dense, six buffers out | 200 | 102 | 206 | 139 |
| 32 live digits a token | values | 493 | 298 | 614 | 581 |
| 32 live digits a token | dense | 858 | 329 | 874 | 426 |
| 32 live digits a token | dense, six buffers out | 898 | 358 | 950 | 527 |

WHAT IT SHOWS.
- THE PASS IS EXACT AND COSTS 356 SM-CYCLES A TOKEN a side at 4 to 5 live digits a token -- its two planes' best
  forms summed -- and 1,479 at 32: a fifth of [`facts_lists`](facts_lists.md)'s form at the sparse cell, a quarter at
  32. It is not the 65 to 150 that was estimated before it was built.
- ITS COPIES ARE ITS PRICE. A flat stream of the same copies, no line moved, costs nine tenths of the moved form
  on every multiprocessor; the threads' count does not move it, and neither does a window the cache has seen: a
  window's tables and planes are past the cache in either mode.
- THE CARD'S MEMORY BINDS IT, NOT A CTA'S QUEUE. On a quarter of the multiprocessors a CTA's flat stream costs
  0.60 and 0.46 of its cost with every multiprocessor at work (values, dense; 0.61 and 0.38 at 32
  live digits): what is shared by the CTAs is what slows them. Six buffers out lose to three everywhere: a copy
  out half the size, twice as often, costs more, and more of them in flight buys nothing.
- SO ITS PRICE IS ITS BYTES. A window of the sparse cell reads some 0.9 MB and writes some 3.0 MB -- 3 KB a token a
  side written where the tables hold 0.6 KB a token -- and near half of what it writes is the pads of each digit's
  last step (17 entries a digit a window is two steps of 16). With the card to itself a CTA's moved form is
  130 and 111 (values, dense): there the moving of lines shows, 46 and 21 over the flat
  stream.
- TWO BOXES DISAGREE. The first run of the unfenced form (e34189c2, instance 60600636, exact) read 152 and 137 for
  the two planes' moved forms at the sparse cell where this box reads 159 and 197: a pass bound by a
  card's memory is priced by the card it ran on, and a rented card's is not one figure.
- WHAT IT DOES NOT SAY: the step lists are the host's (the lists pass emitting them padded is priced at its lists,
  not measured); the read side's pass -- no values -- is not built; the pass beside a running carry is unmeasured.

Record: `measure/lab/results/gather_pass/sm_90a.json`.
