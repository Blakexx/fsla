# `carry_token`

Mirrors `measure/lab/experiments/carry_token.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The carry's token-stream form as a PACKET CONTRACT: the carry blindly consumes a window's packet -- a block of value
lines and a header -- and the gather lays the packet out however is best. What does a window cost a CTA, which role
waits on which, what do the fold's fill and its steps cost apart, and what does the read side cost the fold?

THE FORM. A CTA an SM of four warpgroups, 512 threads, registers traded by role: two fold warpgroups at 168, the
readout at 104, four drain warps at 72.
- A PACKET is a box's window: its HEADER (an entry's word and sparse byte; where each of the 32 steps' A tile starts
  among the value lines, a byte a step; which of a step's 16 places hold an entry; which steps are there) and its
  VALUE LINES, a brick a run of 256. It lands by two bulk copies behind one barrier, issued behind the warpgroup's barrier that follows the wait on the
  window before's steps. THE ONE RULE on it: a step's 16 tokens are consecutive positions -- then they are an
  M|N-major tile the step reads as it landed. Order, a token written twice, two lists' tiles overlapping on a token
  they share, and padding are the layout's.
- THE FOLD: a thread loads its two entries' dense lines from global memory and expands each by its sparse byte to the
  two lines of its step's coefficient tile; the places of a tile no entry fills are zeroed; no value line is moved. A
  step's A is the landed lines at the header's start, its B its tile at the step's index: nothing is loaded a step
  and the 32 steps are one commit group. The image is published a tile group at a time behind the readout.
- THE READOUT is mirrored: a (token, box) a thread, its 16 bytes expanded to two lines of its digit's coefficient tile
  in shared memory, a pass of 8 digits, 16 steps of 64 channels by 16 tokens one commit group, its sums stored to the
  panel as halves.
- THE DRAIN WARPS add each token's row as one f16 line to global memory.
The kernel is `window<TRACE, DETAIL, PROBE>`. `TRACE` 0 is the plain kernel, 1 its launch's two edges stamped, 2 every
warp's phases stamped (the coarse row: the fold's window, its expanding and its steps), 3 every stamp; `DETAIL` names
a coarse row's one more group -- the waits on the tile groups read or published or on the panels filled or drained,
the fold's steps by eighths, the fold's fill by phases (its entries' coefficient stores, the dead places zeroed, the
tile starts loaded, then the wait at the fill's barrier: what a thread does and what it waits) or the waits on the
packets landed (a wait's begin and end at the barrier, and the copy's issue where it is issued); `PROBE` is an
isolation row's change, the read side idle. Fourteen rows: plain, edges, coarse, the seven detail rows, every stamp;
the coarse kernel with the read side idle (not checked: its kernel is not the carry's); and the plain and coarse
kernels on packets whose lists `2 p` and `2 p + 1` share four tokens, written once. The state and every output row are
checked exactly in the other thirteen. The plain row also runs one long call, timed by the host's clock: its span in
the card's cycles over its wall time, and the call by wall time scaled to 65,536 tokens (a window counted as 8 tokens
a CTA, as `carry_window` counts it). THE PACKETS ARE THE HOST'S, playing the gather.

MEASURED (NVIDIA H100 80GB HBM3, 132 SMs, Nebius preemptible VMs 26cmmpjn and 8sredw87, driver 580.173.02, the clock locked; 2026-11-19 00:54 and 01:04 UTC (the one-CTA and the full-grid records), the
experiment at f4189c7e, 7 repetitions; the labels name the boxes of the two records, which hold no box's id; the 11 checked rows pass at one CTA
and at 132, 1 row is unchecked (the isolation row); `sm_90a.blocks1.json` and `sm_90a.blocks132.json`, each with its traced rows' records beside it). Cycles a window a CTA:

| row | one CTA | 132 CTAs |
|---|---:|---:|
| plain | 6,174 | 6,108 |
| the launch's two edges stamped | 6,105 | 6,132 |
| coarse | 6,545 | 6,497 |
| coarse, the waits on the tile groups read | 6,566 | 6,550 |
| coarse, the waits on the tile groups published | 6,789 | 6,756 |
| coarse, the waits on the panels filled | 6,757 | 6,719 |
| coarse, the waits on the panels drained | 6,679 | 6,649 |
| coarse, the fold's steps by eighths | 6,897 | 6,795 |
| every warp's timeline stamped | 7,420 | 7,345 |
| coarse, the read side idle (not checked) | 4,685 | 4,490 |
| lists overlapping on shared tokens, plain | 6,131 | 6,108 |
| lists overlapping on shared tokens, coarse | 6,561 | 6,506 |

WHAT IT DECIDES. THE WINDOW AT ONE CTA IS THE WINDOW AT THE FULL GRID (6,174 and 6,108, 1.1% apart; called the same within 2.0%): 132 CTAs each landing two packets a window do not bind on global memory at this cell. THE LAYOUT IS THE GATHER'S: lists overlapping on tokens written once cost the carry nothing (at one CTA 6,131 against 6,174: the same; at 132 CTAs 6,108 against 6,108: the same; called the same within 2.0%) and are exact (the checks of both overlapping rows, at both grids). THE FOLD'S WINDOW, from the coarse row's record at 132 CTAs (`bazel run //tools:lab -- trace carry_token`; CTA 0, the mean of its 2 fold warpgroups, a window of 6,540 cycles in their own windows): its image 1,810 cycles (27.7%), its fill 1,928 (29.5%), its 32 steps 2,802 (42.8%); the steps phase is the longest and the image phase the shortest (the longest phase is 1.55 times the shortest). THE READ SIDE COSTS THE FOLDS 30.9% OF THE WINDOW at 132 CTAs (28.4% at one CTA): the coarse row's window is 6,497 cycles and the same kernel with the read side idle 4,490 (the isolation row, not checked: its kernel is not the carry's); the fold's 32 steps are 2,802 cycles and, with it idle, 1,442 (48.5% under the coarse row's). THE MARKS' COST: the coarse row's window is 6,497 against the plain row's 6,108 at 132 CTAs (6.4% over, its phases' starts and the launch's edges on both clocks); the figures above are the coarse row's.
BY THE LAUNCH'S OWN CLOCKS the coarse row's launch at 132 CTAs (CTA 0) is 418,598 cycles in 229,728 ns: the cycle counter runs 1,822 cycles a microsecond (1,685 at one CTA, CTA 0); the plain row's window of 6,108 cycles is 3.35 microseconds, and 65,536 tokens (62 windows a CTA, a window counted as 8 tokens a CTA as `carry_window`'s wall-clock rows count it) some 0.21 ms. NOT MEASURED: the gather that writes the packets; a cell where the packets' bytes grow with duplication; and, in the experiment and not yet in a record: the fold's fill by phases (the fill's thread work against its wait at the barrier); the fold's wait on its packet's landing (the waits and the copies' issues); the plain row's long call by wall time (the card's clock, and the call scaled to 65,536 tokens).

THE PHASES, in each role's own window of the coarse row's record at 132 CTAs (CTA 0), cycles:
- the fold: image 1,810, fill 1,928, steps 2,802.
- the readout, its phases in its own window of 6,537 cycles (duration from start): issuing steps 680 from 1,430; storing the panel 529 from 2,002; storing the panel 966 from 2,566; issuing steps 1,189 from 3,582; storing the panel 807 from 4,667; storing the panel 2,366 from 5,511; storing the panel 2,480 from 5,947.
- the drain, its phases in its own window of 6,537 cycles (duration from start): adding 1,465 from 1,444; adding 1,520 from 2,908; adding 1,549 from 4,404; adding 1,828 from 5,793; adding 2,448 from 6,425; adding 2,743 from 6,879.

Record: `measure/lab/results/carry_token/sm_90a.blocks1.json` and `sm_90a.blocks132.json`, each with its traced rows' records in `sm_90a.blocks1.trace/` and `sm_90a.blocks132.trace/`.
