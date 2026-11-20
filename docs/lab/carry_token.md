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

MEASURED (NVIDIA H100 80GB HBM3, 132 SMs, Nebius preemptible VM hjq8b4c0, driver 580.173.02, the clock locked; 2026-11-19 02:55 and 02:55 UTC (the one-CTA and the full-grid records), the
experiment at 3ef04b63, 7 repetitions; the labels name the boxes of the two records, which hold no box's id; the 13 checked rows pass at one CTA
and at 132; `sm_90a.blocks1.json` and `sm_90a.blocks132.json`, each with its traced rows' records beside it). Cycles a window a CTA:

| row | one CTA | 132 CTAs |
|---|---:|---:|
| plain | 5,765 | 5,769 |
| the launch's two edges stamped | 5,984 | 5,979 |
| coarse | 5,970 | 5,933 |
| coarse, the waits on the tile groups read | 6,251 | 6,262 |
| coarse, the waits on the tile groups published | 6,043 | 6,039 |
| coarse, the waits on the panels filled | 5,940 | 5,929 |
| coarse, the waits on the panels drained | 5,882 | 5,913 |
| coarse, the fold's steps by eighths | 6,504 | 6,509 |
| coarse, the fold's fill by phases | 6,243 | 6,275 |
| coarse, the waits on the packets landed | 6,213 | 6,216 |
| every warp's timeline stamped | 6,652 | 6,622 |
| coarse, the read side idle (not checked) | 4,394 | 4,340 |
| lists overlapping on shared tokens, plain | 5,784 | 5,775 |
| lists overlapping on shared tokens, coarse | 5,953 | 5,934 |

WHAT IT DECIDES. THE WINDOW AT ONE CTA IS THE WINDOW AT THE FULL GRID (5,765 and 5,769, 0.1% apart; called the same within 2.0%): 132 CTAs each landing two packets a window do not bind on global memory at this cell. THE LAYOUT IS THE GATHER'S: lists overlapping on tokens written once cost the carry nothing (at one CTA 5,784 against 5,765: the same; at 132 CTAs 5,775 against 5,769: the same; called the same within 2.0%) and are exact (the checks of both overlapping rows, at both grids). THE FOLD'S WINDOW, from the coarse row's record at 132 CTAs (`bazel run //tools:lab -- trace carry_token`; CTA 0, the mean of its 2 fold warpgroups, a window of 5,981 cycles in their own windows): its image 2,290 cycles (38.3%), its fill 1,691 (28.3%), its 32 steps 1,999 (33.4%); the three are near thirds of it (the longest phase is 1.35 times the shortest). THE READ SIDE COSTS THE FOLDS 26.8% OF THE WINDOW at 132 CTAs (26.4% at one CTA): the coarse row's window is 5,933 cycles and the same kernel with the read side idle 4,340 (the isolation row, not checked: its kernel is not the carry's); the fold's 32 steps are 1,999 cycles and, with it idle, 1,856 (7.2% under the coarse row's). THE MARKS' COST: the coarse row's window is 5,933 against the plain row's 5,769 at 132 CTAs (2.8% over, its phases' starts and the launch's edges on both clocks); the figures above are the coarse row's.
BY THE LAUNCH'S OWN CLOCKS the coarse row's launch at 132 CTAs (CTA 0) is 384,484 cycles in 210,496 ns: the cycle counter runs 1,827 cycles a microsecond (1,741 at one CTA, CTA 0); the plain row's window of 5,769 cycles is 3.16 microseconds, and 65,536 tokens (62 windows a CTA, a window counted as 8 tokens a CTA as `carry_window`'s wall-clock rows count it) some 0.20 ms. NOT MEASURED: the gather that writes the packets; a cell where the packets' bytes grow with duplication.

THE PHASES, in each role's own window of the coarse row's record at 132 CTAs (CTA 0), cycles:
- the fold: image 2,290, fill 1,691, steps 1,999.
- the readout, its phases in its own window of 5,995 cycles (duration from start): issuing steps 640 from 1,253; storing the panel 332 from 1,860; storing the panel 653 from 2,204; issuing steps 636 from 2,869; storing the panel 845 from 3,479; storing the panel 2,914 from 4,326; storing the panel 1,381 from 5,607.
- the drain, its phases in its own window of 5,995 cycles (duration from start): adding 1,155 from 1,953; adding 1,166 from 3,108; adding 1,088 from 4,275; adding 2,617 from 5,361; adding 1,644 from 5,839.

- THE FILL'S PHASES, from the fill row's record at 132 CTAs (the mean of the fold warpgroups of CTA 0): expanding 762; zeroing dead places 906; loading tile starts 77; at the fill's barrier 134 cycles; the fill's own work is the phases before the barrier and its wait is `at the fill's barrier`.
- THE FOLD'S WAIT ON ITS PACKET, from the packets row's record at 132 CTAs (CTA 0, the mean of its 8 fold warps): 149 cycles a window (2.4% of the window).
- THE LONG CALL BY WALL TIME at 132 CTAs: 0.196 ms for 65,536 tokens (8 tokens a window a CTA), the call's span over its wall time 1.822 GHz (the card's clock), against the launch-derived 0.196 ms above.

Record: `measure/lab/results/carry_token/sm_90a.blocks1.json` and `sm_90a.blocks132.json`, each with its traced rows' records in `sm_90a.blocks1.trace/` and `sm_90a.blocks132.trace/`.
