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
microsecond in the same run, every form with its check a warp's (below):

| | us, clock off | vs production | cycles a CTA-window, clock off | clocked | landed | maxima | store | readout | drain | fill | fold | wait |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| production, alt-k4 | 860 | 1.00 | ~25,809 | 25,631 | 3,678 | 1,753 | 4,033 | 3,543 | 1,878 | 3,961 | 3,868 | 2,905 |
| form 1, alt-k4 | 614 | 0.71 | 15,820 | 17,875 | 662 | 1,910 | 2,113 | 3,168 | 1,406 | 3,687 | 3,100 | 1,812 |
| form 3, alt-k4 | 628 | 0.73 | 16,176 | 18,385 | 2,600 | 6,508 | 4,667 | 5,185 | 1,487 | 3,161 | 8,762 | 2,699 |
| form 4, alt-k4 | 617 | 0.72 | 15,944 | 17,789 | 200 | 1,893 | 2,156 | 3,237 | -- | 3,916 | 4,202 | 2,136 |
| form 5, alt-k4 | 507 | 0.59 | 15,215 | 17,053 | 203 | 1,789 | 2,087 | 3,154 | -- | 3,590 | 4,098 | 2,107 |
| form 6, alt-k4 | 503 | 0.58 | 15,086 | 17,049 | 205 | 1,788 | 1,919 | 3,142 | -- | 3,590 | 4,101 | 2,313 |
| production, thinned | 509 | 1.00 | ~15,160 | 18,210 | 3,212 | 1,800 | 2,181 | 2,049 | 688 | 2,679 | 2,567 | 3,022 |
| form 1, thinned | 377 | 0.74 | 11,246 | 13,235 | 787 | 1,916 | 2,114 | 2,035 | 382 | 2,394 | 1,819 | 1,770 |
| form 3, thinned | 387 | 0.76 | 11,576 | 13,608 | 781 | 1,912 | 2,119 | 2,045 | 395 | 2,173 | 2,396 | 1,771 |
| form 4, thinned | 374 | 0.73 | 11,152 | 12,984 | 214 | 1,860 | 2,154 | 2,086 | -- | 2,707 | 1,960 | 1,988 |
| form 5, thinned | 362 | 0.71 | 10,828 | 12,577 | 216 | 1,716 | 2,098 | 2,039 | -- | 2,455 | 1,952 | 2,086 |
| form 6, thinned | 360 | 0.71 | 10,737 | 12,505 | 213 | 1,716 | 1,922 | 2,030 | -- | 2,452 | 1,955 | 2,204 |

A phase is the mean warp's cycles with the clock on; "wait" is the warps' time at the barriers that end the maxima, the
stores and the readout's halves.

<a id="fixed-call"></a>
The fixed cost a call. Each form's launch timed with every clock off over the call's first 1, 2, 4 ... 32 windows; the
line through them, its intercept the call's fixed cost (the plan pass where there is one, the prologue and epilogue, the
last wave's tail) and its slope a window's (two waves of CTAs at one head). us, sm_86, median of 7:

| | 1 window | 32 windows | fixed a call | a window |
|---|---|---|---|---|
| production, alt-k4 | 46.1 | 860.2 | 21.1 | 26.3 |
| form 6, alt-k4 | 27.6 | 581.6 | 11.2 | 17.9 |
| production, thinned | 32.8 | 510.0 | 17.2 | 15.4 |
| form 6, thinned | 19.5 | 359.4 | 8.2 | 11.0 |

Every form adds its check once a warp (`../internals/carry/carry_kernel.md#check`): added once a thread, its 32,768
fp32 adds a call on one address serialized at their L2 slice for ~40-47 us -- most of a call's fixed cost (the
production kernel's intercept was ~50 us a call at nl64k before its epilogue joined a warp's sum first). An empty launch
of 128 CTAs costs ~4 us here.

Live registers (`tools/life_ranges.py`): production peaks at 251 of 253 usable and holds ~198 at an item start; form 1
peaks at 233 in the readout's tiles, its fold steps at most 188, its fills 184-210, its base ~157-165; form 5 is
allocated 223. Static instructions: form 1 6,215 against 11,736.

What it decides:
- The item machine costs ~6K a CTA-window at alt-k4, clock off (its waits, its landed checks, its per-item glue); a
  window written by hand from the same parts recovers it (form 1: 0.71 of the production kernel's time, form 6 0.58)
  and holds ~40 fewer registers at every phase.
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
