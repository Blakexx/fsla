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
- 6, form 5 with each half's stores as independent chains (`store_half`: a lane's units and exponents read first, its
  eight scales formed, then its 32 pairs scaled, packed and stored).

Each form is read twice: its launch timed with every clock off -- a lab form's warps stamping only their start and end,
so its span in cycles is free of the SM's clock (which moves ~9% between processes here), the production kernel's
ledger unbound -- and then with its clock on, for the phases. The clock costs ~11% of a lab form's time and ~16% of the
production kernel's (a clock read is a point ptxas does not schedule across), so the phases attribute, the span times.
sm_86 (RTX 3080 Ti), median of 7; the production kernel's clock-off cycles taken from its time at form 5's cycles a
microsecond in the same run:

| | us, clock off | vs production | cycles a CTA-window, clock off | clocked | landed | maxima | store | readout | drain | fill | fold | wait |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| production, alt-k4 | 902 | 1.00 | ~22,160 | 25,908 | 3,915 | 1,753 | 4,002 | 3,571 | 1,888 | 3,956 | 3,907 | 2,907 |
| form 1, alt-k4 | 660 | 0.73 | 16,213 | 18,229 | 1,036 | 1,918 | 2,115 | 3,142 | 1,409 | 3,675 | 3,099 | 1,813 |
| form 3, alt-k4 | 682 | 0.76 | 16,699 | 18,703 | 1,111 | 1,910 | 2,106 | 3,144 | 1,467 | 3,110 | 4,029 | 1,815 |
| form 4, alt-k4 | 664 | 0.74 | 16,225 | 18,031 | 454 | 1,895 | 2,164 | 3,238 | -- | 3,975 | 4,207 | 2,078 |
| form 6, alt-k4 | -- | 0.70 | 15,463 | -- | -- | -- | 1,928 | -- | -- | -- | -- | -- |
| form 5, alt-k4 | 634 | 0.70 | 15,572 | 17,379 | 355 | 1,798 | 2,100 | 3,154 | -- | 3,683 | 4,133 | 2,116 |
| production, thinned | 546 | 1.00 | ~15,140 | 18,503 | 3,385 | 1,794 | 2,225 | 2,089 | 696 | 2,684 | 2,587 | 3,045 |
| form 1, thinned | 414 | 0.76 | 11,610 | 13,580 | 1,147 | 1,922 | 2,116 | 2,035 | 382 | 2,364 | 1,822 | 1,770 |
| form 6, thinned | -- | 0.73 | 11,122 | -- | -- | -- | 1,953 | -- | -- | -- | -- | -- |
| form 5, thinned | 403 | 0.74 | 11,194 | 12,929 | 490 | 1,732 | 2,127 | 2,039 | -- | 2,456 | 1,952 | 2,118 |

A phase is the mean warp's cycles with the clock on; "wait" is the warps' time at the barriers that end the maxima, the
stores and the readout's halves.

Live registers (`tools/life_ranges.py`): production peaks at 251 of 253 usable and holds ~198 at an item start; form 1
peaks at 233 in the readout's tiles, its fold steps at most 188, its fills 184-210, its base ~157-165; form 5 is
allocated 223. Static instructions: form 1 6,215 against 11,736.

What it decides:
- The item machine costs ~6K a CTA-window at alt-k4, clock off (its waits, its landed checks, its per-item glue); a
  window written by hand from the same parts recovers it (form 1: 0.73 of the production kernel's time) and holds ~40
  fewer registers at every phase.
- The window is bound by latency, not issue: form 1 executes ~3.3K warp instructions a warp-window in ~16K cycles
  (`lab.py ledger`), and the stall samples are "wait" (a fixed-latency dependence) and "short_sb" (a shared-memory
  result) in every phase, issue ~20%. Two warps a scheduler hide nothing a part leaves serial; the barriers themselves
  are ~6% of the samples.
- Moving work under the fold hides little while the fold is serial too: the rows' slices in form 3 and the drain in
  form 4 each moved most of their cost into the fold (+0.9K, +1.1K). Loading a fill's tokens ahead and copying a
  warp's entries in one pass took 0.25K off the fills.
- Each part runs well over the lab stand-in the design was priced on (`digit_box`): the readout 3.2K (2.1K), the fold
  ~3.1K (1.9K), the two halves' stores 2.5K (~0.4K), the maxima 1.8K (~0.4K, posts and a barrier).
- The window is mostly fixed cost: the thinned cell, with 1/16 of the entries, still costs ~11.1K of alt-k4's ~15.5K
  (form 6), so ~70% of the window at nl64k-alt-k4 does not shrink with its writes. Form 6 executes ~2,980 warp
  instructions a warp-window with its clock off (~6K cycles of issue at two warps a scheduler), against ~80 MMAs.
- The stores' floor is the hand-off's format, not their order: chained, they run ~0.17K under form 5's; their ~400
  instructions are near the ~260 their scaling needs (each value to its digit's exponent, then packed to f16 and
  stored), where `digit_box`'s stand-in stored bf16 straight from the accumulators (~130).
