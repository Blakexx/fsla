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
- 6, form 5 with each half's stores as independent chains (`store_half`), WITHDRAWN from the runs 2026-10-24: its
  stores predate the 16-byte image, which production's `store_batches` now writes in the lean form (2026-10-18), so it
  no longer matches the call.

Every form sets each read chunk's units as the production window does (`parts.md#units`: the CTA's own, from its
leaves' maxima, before the chunk's first add) and initializes the tenants production's prologue does (the keys, the
bounds' counters, the units, the scales' bits). The check against the captured call is per token since the cross-CTA
sum went floating (`carry_api.md#carry-reduce`): the sum's order is the hardware's, so each value is held within 2^-5
of the token's largest |channel| (or 2^-12 absolute); the production kernel's own replay reads within 0.4%.

Each form is read twice: its launch timed with every clock off -- a lab form's warps stamping only their start and end,
so its span in cycles is free of the SM's clock (which moves ~9% between processes here), the production kernel's
ledger unbound -- and then with its clock on, for the phases. The clock costs ~11% of a lab form's time and ~16% of the
production kernel's (a clock read is a point ptxas does not schedule across), so the phases attribute, the span times.
sm_86 (RTX 3080 Ti), SM clock locked at 1.665 GHz, median of 7, carry-k5 e53bc28d (the per-leaf output form) on
2026-10-24; the production kernel's clock-off cycles from its time at form 5's cycles a microsecond in the same run (the
2026-10-14 table, at 790dfac6's parts: production 860 / 509 us, form 6 503 / 360):

| | us, clock off | vs production | cycles a CTA-window, clock off | clocked | landed | maxima | store | readout | drain | fill | fold | wait | units |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| production, alt-k4 | 895 | 1.00 | ~22,871 | 25,246 | 1,038 | 2,036 | 3,146 | 3,812 | 2,121 | 4,747 | 3,287 | 2,989 | 2,063 |
| form 1, alt-k4 | 660 | 0.74 | 16,849 | 18,862 | 666 | 1,905 | 1,445 | 3,288 | 1,512 | 3,049 | 3,143 | 1,998 | 1,839 |
| form 3, alt-k4 | 675 | 0.75 | 17,235 | 19,290 | 722 | 1,926 | 1,451 | 3,283 | 1,526 | 2,509 | 4,039 | 1,980 | 1,837 |
| form 4, alt-k4 | 662 | 0.74 | 16,904 | 18,384 | 222 | 2,270 | 1,393 | 3,236 | -- | 3,246 | 4,069 | 2,223 | 1,717 |
| form 5, alt-k4 | 650 | 0.73 | 16,617 | 18,151 | 217 | 1,843 | 1,480 | 3,212 | -- | 3,486 | 4,051 | 2,121 | 1,721 |
| production, thinned | 652 | 1.00 | ~16,543 | 18,679 | 1,272 | 2,042 | 1,600 | 1,940 | 859 | 3,858 | 2,141 | 3,382 | 1,569 |
| form 1, thinned | 476 | 0.73 | 12,049 | 14,012 | 740 | 1,908 | 1,444 | 2,178 | 566 | 2,083 | 1,891 | 1,880 | 1,301 |
| form 3, thinned | 486 | 0.75 | 12,367 | 14,367 | 742 | 1,928 | 1,447 | 2,171 | 579 | 1,859 | 2,468 | 1,855 | 1,300 |
| form 4, thinned | 485 | 0.74 | 12,310 | 13,869 | 231 | 2,273 | 1,395 | 2,164 | -- | 2,344 | 2,153 | 2,089 | 1,207 |
| form 5, thinned | 470 | 0.72 | 11,926 | 13,572 | 227 | 1,772 | 1,480 | 2,145 | -- | 2,455 | 2,120 | 2,154 | 1,206 |

A phase is the mean warp's cycles with the clock on; "wait" is the warps' time at the barriers that end the maxima, the
stores and the readout's halves.

<a id="fixed-call"></a>
The fixed cost a call. Each form's launch timed with every clock off over the call's first 1, 2, 4 ... 32 windows; the
line through them, its intercept the call's fixed cost (the plan pass where there is one, the prologue and epilogue, the
last wave's tail) and its slope a window's (two waves of CTAs at one head). us, sm_86, median of 7:

| | 1 window | 32 windows | fixed a call | a window |
|---|---|---|---|---|
| production, alt-k4 | 47.1 | 894.0 | 22.9 | 27.3 |
| form 5, alt-k4 | 30.7 | 647.2 | 12.7 | 19.9 |
| production, thinned | 38.9 | 653.2 | 19.7 | 19.8 |
| form 5, thinned | 24.6 | 469.0 | 10.2 | 14.3 |

Every form adds its check once a warp (`../internals/carry/carry_kernel.md#check`): added once a thread, its 32,768
fp32 adds a call on one address serialized at their L2 slice for ~40-47 us -- most of a call's fixed cost (the
production kernel's intercept was ~50 us a call at nl64k before its epilogue joined a warp's sum first). An empty launch
of 128 CTAs costs ~4 us here.

Live registers (`tools/life_ranges.py`): production peaks at 251 of 253 usable and holds ~198 at an item start; form 1
peaks at 233 in the readout's tiles, its fold steps at most 188, its fills 184-210, its base ~157-165; form 5 is
allocated 223. Static instructions: form 1 6,215 against 11,736.

What it decides (2026-10-24, the lab re-run at e53bc28d; the 2026-10-14 findings below stand):
- The hand-written window still runs 0.73x the production kernel (form 5: 650 against 895 us at alt-k4, 470 against
  652 thinned), with the units pass in both: the step machine and its statics re-read at each use cost ~6.3K cycles a
  CTA-window clock off (~7.1K clocked) at either cell -- the largest single item of the window's fixed work.
- The window is 73% fixed at this build: production thinned 652 / 895 us; form 5 thinned 470 / 650 (72%).
- Production's phases against form 5's (alt-k4, a warp-window, clock on): landed 1.0K vs 0.2K, store 3.1K vs 1.5K
  (production's image in halves with a rendezvous each: every alt-k4 window's whole image does not fit beside its read
  chunk), fill 4.7K vs 3.5K, wait 3.0K vs 2.1K, drain 2.1K vs 0 (inside the fold: fold 3.3K vs 4.1K), readout 3.8K
  vs 3.2K, maxima 2.0K vs 1.8K, units 2.1K vs 1.7K.

What it decided (2026-10-14):
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
