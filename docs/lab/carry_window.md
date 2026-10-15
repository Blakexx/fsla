# `carry_window`

Mirrors `measure/lab/experiments/carry_window.cu`, a part harness of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

What the carry's window costs a CTA on sm_86 in each form the window's critical-path design admits, phase by phase in
cycles and live registers, against the production kernel on the same inputs. A form is a window written by
hand from the production kernel's own parts (`csrc/rola/src/carry/parts.cuh`, `fold.cuh`), so its registers and
instructions are the parts' own.

The inputs are a real call (`carry_window.inputs.py`): the measurements' pipeline over a cell's first 32 windows (a
window's cost is flat in L), launched once with `rola.ops.carry.capture_bind` bound
([`../internals/carry/carry_api.md#carry-capture-bind`](../internals/carry/carry_api.md#carry-capture-bind)), and every
buffer the operand block points into saved beside the capture. The experiment re-hosts the buffers and moves each
pointer word into its copy. Two cells: nl64k-alt-k4, and the same with each (digit, token) of both list levels kept
with probability 1/16 (lists under one write a window, as at ~1M leaves). Every form's readout output is checked bit for
bit against the captured call's; the phase ledger (`carry_kernel.md#phases`'s phases) gives each form's cycles, the
busiest warp's sum over a window and each phase's mean warp.

The forms:
- 0, the production kernel: its plan pass, then the carry (the item machine).
- 1, option A by hand: the maxima, each image half stored then read out (two digits a warp stepped together), the
  fold's rows over the image once every warp has read it, the drain, the next window's read rows at the area's top, the
  fold. One chunk a side; async copies by `cp.async` groups, a CTA barrier at each hand-off. No address is held across
  phases (the thread index re-read at each), and the phase clock holds no register.
- 3, form 1 with the next window's panel zeroed by stores, its entries before the fold, its rows in four slices, one
  before each of a warp's lists in the fold.
- 4, form 1 with the window's read buffer issued at its top (under the maxima and the first half's stores, the panel by
  stores), and the drain inside the fold, a warp's share before its list `(warp >> 1) & 3`.
- 5, form 4 with its fills' loads ahead (a thread's four slots' tokens loaded before any copy) and each warp's entries
  copied in one pass over its four lists' runs.

Cycles a CTA-window, sm_86 (RTX 3080 Ti), median of 7 (a phase's column is the mean warp's; with a barrier at both ends
it is the slowest warp's work):

| | total | landed | maxima | store | readout | drain | fill | fold | wait |
|---|---|---|---|---|---|---|---|---|---|
| production, alt-k4 | 25,904 | 3,902 | 1,750 | 4,007 | 3,572 | 1,889 | 3,958 | 3,912 | 2,907 |
| form 1, alt-k4 | 17,554 | 1,073 | 1,847 | 2,506 | 3,215 | 1,427 | 4,034 | 3,145 | 290 |
| form 3, alt-k4 | 17,859 | 1,099 | 1,832 | 2,492 | 3,177 | 1,444 | 3,439 | 4,049 | 299 |
| form 4, alt-k4 | 17,047 | 761 | 1,884 | 2,433 | 3,265 | -- | 3,984 | 4,201 | 553 |
| form 5, alt-k4 | 16,645 | 734 | 1,791 | 2,477 | 3,233 | -- | 3,730 | 4,136 | 544 |
| production, thinned | 18,463 | 3,355 | 1,796 | 2,222 | 2,084 | 697 | 2,679 | 2,568 | 3,044 |
| form 1, thinned | 12,876 | 1,115 | 1,842 | 2,489 | 2,090 | 406 | 2,718 | 1,862 | 261 |
| form 5, thinned | 12,171 | 880 | 1,776 | 2,520 | 2,113 | -- | 2,416 | 1,937 | 514 |

Live registers (`tools/life_ranges.py`): production peaks at 251 of 253 usable and holds ~198 at an item start; form 1
peaks at 233 in the readout's tiles, its fold steps at most 188, its fills 184-210, its base ~157-165; form 5 is
allocated 223. Static instructions: form 1 6,215 against 11,736.

What it decides:
- The item machine costs ~8.3K a CTA-window at alt-k4 (its waits, its landed checks, its per-item glue); a window
  written by hand from the same parts recovers it and holds ~40 fewer registers at every phase.
- The window is bound by latency, not issue: form 1 executes ~3.3K warp instructions a warp-window in 17.6K cycles
  (`lab.py ledger`), and the stall samples are "wait" (a fixed-latency dependence) and "short_sb" (a shared-memory
  result) in every phase, issue ~20%. Two warps a scheduler hide nothing a part leaves serial; the barriers themselves
  are ~6% of the samples.
- Moving work under the fold hides little while the fold is serial too: the rows' slices in form 3 and the drain in
  form 4 each moved most of their cost into the fold (+0.9K, +1.1K). Loading a fill's tokens ahead and copying a
  warp's entries in one pass took 0.25K off the fills.
- Each part runs well over the lab stand-in the design was priced on (`digit_box`): the readout 3.2K (2.1K), the fold
  ~3.1K (1.9K), the two halves' stores 2.5K (~0.4K), the maxima 1.8K (~0.4K, posts and a barrier).
