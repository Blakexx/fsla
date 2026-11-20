# `carry_token`

Mirrors `measure/lab/experiments/carry_token.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The carry's token-stream form as a PACKET CONTRACT: the carry blindly consumes a window's packet -- a block of value
lines and a header -- and the gather lays the packet out however is best. What does a window cost a CTA, which role
waits on which, what do the fold's fill and its steps cost apart, and what does the read side cost the fold?

THE FORM. A CTA an SM of four warpgroups, 512 threads, registers traded by role: two fold warpgroups at 168, the
readout at 120, four drain warps at 56.
- A PACKET is a box's window: its HEADER (an entry's word and sparse byte; where each of the 32 steps' A tile starts
  among the value lines, a byte a step; for a place of a step that holds none a PAD, a word naming its two tile lines; which steps are there) and its
  VALUE LINES, a brick a run of 256. It lands by two bulk copies behind one barrier, issued behind the warpgroup's barrier that follows the wait on the
  window before's steps. THE ONE RULE on it: a step's 16 tokens are consecutive positions -- then they are an
  M|N-major tile the step reads as it landed. Order, a token written twice, two lists' tiles overlapping on a token
  they share, and padding are the layout's.
- THE FOLD: a thread loads its two entries' dense lines from global memory at the window's start, before the image,
  and expands each by its sparse byte to the
  two lines of its step's coefficient tile; a place of a tile no entry fills is a pad's, its two lines written zero by a thread's third word; no value line is moved. A
  step's A is the landed lines at the header's start, its B its tile at the step's index: nothing is loaded a step
  and the 32 steps are one commit group. The image is published a tile group at a time behind the readout.
- THE READOUT is mirrored: a (token, box) a thread, its 16 bytes -- loaded two passes ahead -- expanded to two lines of its digit's coefficient tile
  in shared memory, a pass of 8 digits, 16 steps of 64 channels by 16 tokens one commit group, its sums stored to the
  panel as four transposed 8 x 8 matrices a digit.
- THE DRAIN WARPS add each token's row as one f16 line to global memory.
The kernel is `window<TRACE, DETAIL, PROBE>`. `TRACE` 0 is the plain kernel, 1 its launch's two edges stamped, 2 every
warp's phases stamped (the coarse row: the fold's window, its expanding and its steps), 3 every stamp; `DETAIL` names
a coarse row's one more group -- the waits on the tile groups read or published or on the panels filled or drained,
the fold's steps by eighths, the fold's fill by phases (its entries' coefficient stores, its pad written, the
tile starts loaded, then the wait at the fill's barrier: what a thread does and what it waits) or the waits on the
packets landed (a wait's begin and end at the barrier, and the copy's issue where it is issued); `PROBE` is an
isolation row's change, the read side idle. Fourteen rows: plain, edges, coarse, the seven detail rows, every stamp;
the coarse kernel with the read side idle (not checked: its kernel is not the carry's); and the plain and coarse
kernels on packets whose lists `2 p` and `2 p + 1` share four tokens, written once. The state and every output row are
checked exactly in the other thirteen. The plain row also runs one long call, timed by the host's clock: its span in
the card's cycles over its wall time, and the call by wall time scaled to 65,536 tokens (a window counted as 8 tokens
a CTA, as `carry_window` counts it). THE PACKETS ARE THE HOST'S, playing the gather.

MEASURED (NVIDIA H100 80GB HBM3, 132 SMs, Nebius preemptible VM zsfq8jye, driver 580.196.07, the clock locked; 2026-11-20 21:11 and 21:05 UTC (the one-CTA and the full-grid records), the
experiment at a32d0424, 7 repetitions; the labels name the boxes of the two records, which hold no box's id; the 13 checked rows pass at one CTA
and at 132; `sm_90a.blocks1.json` and `sm_90a.blocks132.json`, each with its traced rows' records beside it). Cycles a window a CTA:

| row | one CTA | 132 CTAs |
|---|---:|---:|
| plain | 5,433 | 5,445 |
| the launch's two edges stamped | 5,311 | 5,315 |
| coarse | 5,749 | 5,759 |
| coarse, the waits on the tile groups read | 5,891 | 5,908 |
| coarse, the waits on the tile groups published | 5,936 | 5,939 |
| coarse, the waits on the panels filled | 5,726 | 5,707 |
| coarse, the waits on the panels drained | 5,701 | 5,740 |
| coarse, the fold's steps by eighths | 6,145 | 6,182 |
| coarse, the fold's fill by phases | 5,835 | 5,860 |
| coarse, the waits on the packets landed | 5,748 | 5,777 |
| every warp's timeline stamped | 6,374 | 6,368 |
| coarse, the read side idle (not checked) | 4,110 | 4,114 |
| lists overlapping on shared tokens, plain | 5,458 | 5,455 |
| lists overlapping on shared tokens, coarse | 5,754 | 5,757 |

WHAT IT DECIDES. THE WINDOW AT ONE CTA IS THE WINDOW AT THE FULL GRID (5,433 and 5,445, 0.2% apart; called the same within 2.0%): 132 CTAs each landing two packets a window do not bind on global memory at this cell. THE LAYOUT IS THE GATHER'S: lists overlapping on tokens written once cost the carry nothing (at one CTA 5,458 against 5,433: the same; at 132 CTAs 5,455 against 5,445: the same; called the same within 2.0%) and are exact (the checks of both overlapping rows, at both grids). THE FOLD'S WINDOW, from the coarse row's record at 132 CTAs (`bazel run //tools:lab -- trace carry_token`; CTA 0, the mean of its 2 fold warpgroups, a window of 5,724 cycles in their own windows): its image 2,252 cycles (39.3%), its fill 1,377 (24.1%), its 32 steps 2,096 (36.6%); the image phase is the longest and the fill phase the shortest (the longest phase is 1.64 times the shortest). THE READ SIDE COSTS THE FOLDS 28.6% OF THE WINDOW at 132 CTAs (28.5% at one CTA): the coarse row's window is 5,759 cycles and the same kernel with the read side idle 4,114 (the isolation row, not checked: its kernel is not the carry's); the fold's 32 steps are 2,096 cycles and, with it idle, 2,094 (0.1% under the coarse row's). THE MARKS' COST: the coarse row's window is 5,759 against the plain row's 5,445 at 132 CTAs (5.8% over, its phases' starts and the launch's edges on both clocks); the figures above are the coarse row's.
BY THE LAUNCH'S OWN CLOCKS the coarse row's launch at 132 CTAs (CTA 0) is 368,343 cycles in 203,616 ns: the cycle counter runs 1,809 cycles a microsecond (1,696 at one CTA, CTA 0); the plain row's window of 5,445 cycles is 3.01 microseconds, and 65,536 tokens (62 windows a CTA, a window counted as 8 tokens a CTA as `carry_window`'s wall-clock rows count it) some 0.19 ms. NOT MEASURED: the gather that writes the packets; a cell where the packets' bytes grow with duplication.

THE PHASES, in each role's own window of the coarse row's record at 132 CTAs (CTA 0), cycles:
- the fold: image 2,252, fill 1,377, steps 2,096.
- the readout, its phases in its own window of 5,731 cycles (duration from start): issuing steps 628 from 1,223; storing the panel 233 from 1,851; storing the panel 857 from 2,084; issuing steps 883 from 2,941; storing the panel 301 from 3,824; storing the panel 2,828 from 4,125.
- the drain, its phases in its own window of 5,731 cycles (duration from start): adding 799 from 1,786; adding 1,307 from 2,586; adding 1,252 from 3,893; adding 2,363 from 5,121; adding 2,472 from 5,460.

- THE FILL'S PHASES, from the fill row's record at 132 CTAs (the mean of the fold warpgroups of CTA 0): expanding 833; writing its pad 284; loading tile starts 146; at the fill's barrier 346 cycles; the fill's own work is the phases before the barrier and its wait is `at the fill's barrier`.
- THE FOLD'S WAIT ON ITS PACKET, from the packets row's record at 132 CTAs (CTA 0, the mean of its 8 fold warps): 144 cycles a window (2.5% of the window).
- THE LONG CALL BY WALL TIME at 132 CTAs: 0.185 ms for 65,536 tokens (8 tokens a window a CTA), the call's span over its wall time 1.823 GHz (the card's clock), against the launch-derived 0.187 ms above.

Record: `measure/lab/results/carry_token/sm_90a.blocks1.json` and `sm_90a.blocks132.json`, each with its traced rows' records in `sm_90a.blocks1.trace/` and `sm_90a.blocks132.trace/`.
