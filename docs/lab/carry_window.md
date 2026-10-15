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

Cycles a CTA-window, sm_86 (RTX 3080 Ti), median of 5:

| | total | landed | maxima | store | readout | drain | fill | fold | wait |
|---|---|---|---|---|---|---|---|---|---|
| production, alt-k4 | 25,902 | 3,905 | 1,751 | 4,012 | 3,567 | 1,888 | 3,959 | 3,909 | 2,907 |
| form 1, alt-k4 | 17,556 | 1,065 | 1,847 | 2,507 | 3,214 | 1,427 | 4,032 | 3,145 | 292 |
| production, thinned | 18,478 | 3,372 | 1,796 | 2,218 | 2,085 | 697 | 2,682 | 2,575 | 3,041 |
| form 1, thinned | 12,900 | 1,113 | 1,842 | 2,490 | 2,089 | 406 | 2,719 | 1,862 | 264 |

The total is the busiest warp's; a phase is the mean warp's. Live registers (`tools/life_ranges.py`): production peaks
at 251 of 253 usable and holds ~198 at an item start; form 1 peaks at 233 in the readout's tiles, its fold steps at most
188, its fills 184-210, its base ~157-165. Static instructions: 6,215 against 11,736.

What it decides: the item machine costs ~8.3K a CTA-window at alt-k4 (its waits, its landed checks and its per-item
glue); a window written by hand from the same parts recovers it and holds ~40 fewer registers at every phase. The fills
(4.0K a warp-window, the copies' issue) are now the largest phase, then the store and the maxima: the next forms move
the fills under the fold, zero the panel by stores, and post the maxima in the fold's tail.
