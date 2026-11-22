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
- THE GROUPED FOLD (the kernel's `kWide`): a GROUP of four digits is a layout, a token's value lines laid once a group with its
  coefficients for the group's digits. The fold steps a group, not a list: a step is 16 tokens' value lines (the A operand,
  as landed) against a tile of 16 tokens by four digits' 64 columns, so a group whose tokens are on all four digits of it is
  one step of 16 tokens a group (`kFull`) and a group whose tokens are on one digit is four steps of 16 entries. The kernel
  runs at 176 / 112 / 48 registers a thread (the folds, the readout, the drain); the per-list fold at those registers is a row
  of its own (`kSplit`). The one-step-a-group fold (`kFull`: four steps a window, every line of a group's tile an entry's) clears nothing;
  the two-pass fold (a token on one digit) clears each pass's eight tiles, then stores its entries, and its warpgroup's barrier
  follows every pass's wait.
- THE PER-COLUMN FORM (`kColumn`): one kernel holding both folds, the per-list fold and the group-of-four fold, the box's WIDTH
  WORD -- a word of the packet's header, the device plan's -- read before the window loop choosing between them: a step's
  shape is chosen where no step is in flight; a width word that is neither 1 nor 4 traps.
- THE 16-DIGIT PACKETS (layout 5): a token on all 16 digits of its box, the box's 16 tokens' value lines once and every group's
  step reading that one run of value lines as its A operand: the box as one group is four steps of 64 columns on one A, its
  width data the packet holds and no step shape of its own.
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
isolation row's change, the read side idle. Twenty-four rows: plain, edges, coarse, the seven detail rows, every stamp;
the coarse kernel with the read side idle (not checked: its kernel is not the carry's); the plain and coarse
kernels on packets whose lists `2 p` and `2 p + 1` share four tokens, written once; and ten grouping rows (the plain kernel
on groups of four digits and its folds alone, the per-list fold at the grouped kernel's registers, a token on all four digits
of its group in groups of four and in a list a digit, each with its folds alone, a token on all 16 digits of its box, and the
per-column form on the two packets). The state and every output row are checked exactly in all but the four isolation rows. The plain row also runs one long call, timed by the host's clock: its span in
the card's cycles over its wall time, and the call by wall time scaled to 65,536 tokens (a window counted as 8 tokens
a CTA, as `carry_window` counts it). THE PACKETS ARE THE HOST'S, playing the gather.

ACCESS TABLE (KERNEL_STANDARDS [§30.1](../KERNEL_STANDARDS.md#s30): each instruction that touches a shared buffer, by lanes, addresses and banks; the pad stores', the grouped folds' tile clear and entry stores' rows; the fold's other accesses are not yet in it).

| buffer | instruction | lanes and addresses | banks | measured | owed |
|---|---|---|---|---|---|
| the fold's coefficient tiles | the pad stores: 6 `STS.128` of the fold, a pad's two zero lines written by a thread's third word | a warp's 32 stores, a thread's lines named by its packet's words, so the data picks the addresses | where the data puts them: the stores meet on banks by chance, conflict-free for no shape | 43.5 excess shared wavefronts a CTA-window (367,276 over the timed launch's 8,448 CTA-windows, 132 CTAs of 64 windows, the launch's dump), accepted as declared `BANK_CONFLICT <= 44` | the gather places each pad's lines by bank, a placement rule in the packet's format (KERNEL_STANDARDS §30.2: priced against the fold's cost before it enters the format), then the declaration drops to the conflict-free count |
| the grouped folds' coefficient tiles (the two-pass fold, groups of four digits) | the tile clear: a pass's 8 tiles cleared before its entries are stored, 16 stores a thread of 8 bytes, behind the warpgroup's barrier | a warp's 32 lanes store 32 consecutive 8-byte words an instruction (a thread's word `(x & 31) * 8` bytes into its warp's 4,096-byte quarter of the tiles, an instruction `256` bytes on) | every bank twice, the 256 bytes an instruction in two wavefronts: conflict-free by the lanes' addresses (the source's own count; no profile of this kernel) | nothing: the record holds no counters (`counters` empty: not yet read on this box); the declared `BANK_CONFLICT <= 44` bounds the kernel's excess, the pad stores' | the dump of this kernel (`lab -- ncu`, then `ledger`), the clear's wavefronts counted against the conflict-free 2 an instruction |
| the grouped folds' coefficient tiles (both grouped folds) | the entry stores: two 16-byte lines an entry, `STS.128`, at the lines its word names (`word & 0xffff` and `word >> 16`), two entries a thread in the one-step-a-group fold, one an entry a pass in the two-pass fold | the host lays the lines: in the one-step-a-group fold (a token on all four digits) a warp's lanes are 32 consecutive entries, so each quarter-warp phase of 8 lanes stores 8 consecutive 16-byte lines (`128 q + 32 digit + place`); in the two-pass fold the lane's line is its entry's position in its list (lists of 8 to 24), so the data picks the addresses | the one-step fold: conflict-free by the host's layout (each phase 8 consecutive lines); the two-pass fold: by chance, as the pad stores | nothing beyond the same declaration `BANK_CONFLICT <= 44` and no counters: the one-step fold's by the layout's arithmetic only | the dump of both kernels; for the two-pass fold a placement rule by bank in the packet's format (KERNEL_STANDARDS §30.2), as the pad stores owe |

MEASURED (NVIDIA H100 80GB HBM3, 132 SMs, Nebius preemptible VMs 6vmzys6s and xd529n2w, driver 580.196.07, toolkit 13.5, the clock locked; 2026-11-22 21:18 and 21:13 UTC (the one-CTA and the full-grid records), the
experiment at 4a2ba99b, 7 repetitions; the labels name the boxes of the two records, which hold no box's id; the 20 checked rows pass at one CTA
and at 132, the 4 isolation rows are unchecked (their kernels are not the carry's); `sm_90a.blocks1.json` and `sm_90a.blocks132.json`, each with its traced rows' records beside it). Cycles a window a CTA:

| row | one CTA | 132 CTAs |
|---|---:|---:|
| plain | 5,475 | 5,463 |
| the launch's two edges stamped | 5,271 | 5,290 |
| coarse | 5,702 | 5,725 |
| coarse, the waits on the tile groups read | 5,942 | 5,957 |
| coarse, the waits on the tile groups published | 5,973 | 5,989 |
| coarse, the waits on the panels filled | 5,685 | 5,706 |
| coarse, the waits on the panels drained | 5,664 | 5,734 |
| coarse, the fold's steps by eighths | 6,112 | 6,186 |
| coarse, the fold's fill by phases | 6,014 | 5,918 |
| coarse, the waits on the packets landed | 5,786 | 5,788 |
| every warp's timeline stamped | 6,184 | 6,195 |
| coarse, the read side idle (not checked) | 4,075 | 4,027 |
| lists overlapping on shared tokens, plain | 5,473 | 5,471 |
| lists overlapping on shared tokens, coarse | 5,695 | 5,728 |
| groups of four digits, plain | 6,228 | 6,193 |
| registers 176 / 112 / 48, plain | 5,375 | 5,405 |
| a token on all four digits of its group, groups of four, plain | 4,240 | 4,288 |
| a token on all four digits of its group, groups of four, coarse, the read side idle (not checked) | 2,632 | 2,545 |
| a token on all four digits of its group, a list a digit, plain | 5,434 | 5,428 |
| a token on all four digits of its group, a list a digit, coarse, the read side idle (not checked) | 4,053 | 4,010 |
| groups of four digits, coarse, the read side idle (not checked) | 4,092 | 4,017 |
| a token on all 16 digits of its box, groups of four on one run of value lines, plain | 4,254 | 4,288 |
| a token on all four digits of its group, a list a digit, the box's width word choosing its fold, plain | 5,291 | 5,308 |
| a token on all four digits of its group, groups of four, the box's width word choosing its fold, plain | 4,218 | 4,261 |

WHAT IT DECIDES. THE WINDOW AT ONE CTA IS THE WINDOW AT THE FULL GRID (5,475 and 5,463, 0.2% apart; called the same within 2.0%): 132 CTAs each landing two packets a window do not bind on global memory at this cell. THE LAYOUT IS THE GATHER'S: lists overlapping on tokens written once cost the carry nothing (at one CTA 5,473 against 5,475: the same; at 132 CTAs 5,471 against 5,463: the same; called the same within 2.0%) and are exact (the checks of both overlapping rows, at both grids). THE FOLD'S WINDOW, from the coarse row's record at 132 CTAs (`bazel run //tools:lab -- trace carry_token`; CTA 0, the mean of its 2 fold warpgroups, a window of 5,744 cycles in their own windows): its image 2,330 cycles (40.6%), its fill 1,377 (24.0%), its 32 steps 2,037 (35.5%); the image phase is the longest and the fill phase the shortest (the longest phase is 1.69 times the shortest). THE READ SIDE COSTS THE FOLDS 29.7% OF THE WINDOW at 132 CTAs (28.5% at one CTA): the coarse row's window is 5,725 cycles and the same kernel with the read side idle 4,027 (the isolation row, not checked: its kernel is not the carry's); the fold's 32 steps are 2,037 cycles and, with it idle, 2,035 (the same as the coarse row). THE MARKS' COST: the coarse row's window is 5,725 against the plain row's 5,463 at 132 CTAs (4.8% over, its phases' starts and the launch's edges on both clocks); the figures above are the coarse row's.
BY THE LAUNCH'S OWN CLOCKS the coarse row's launch at 132 CTAs (CTA 0) is 369,257 cycles in 202,368 ns: the cycle counter runs 1,825 cycles a microsecond (1,701 at one CTA, CTA 0); the plain row's window of 5,463 cycles is 2.99 microseconds, and 65,536 tokens (62 windows a CTA, a window counted as 8 tokens a CTA as `carry_window`'s wall-clock rows count it) some 0.19 ms. NOT MEASURED: the gather that writes the packets; a cell where the packets' bytes grow with duplication.

THE GROUPED FOLD (a group's tokens once, a step 16 tokens' value lines against a tile of 16 tokens by four digits' 64 columns), cycles a window a CTA at 132 CTAs unless said, every row exact:
- (a) A STEP SHAPE CHOSEN BEFORE THE WINDOW LOOP IS FREE: the per-column kernel (one kernel holding both folds, the box's width word read before its window loop) is, at 132 CTAs, 5,308 on per-list packets against the per-list kernel's 5,428 (2.2% under; at one CTA 5,291 against 5,434) and 4,261 on group-of-four packets against the group-of-four kernel's 4,288 (the same; at one CTA 4,218 against 4,240), exact.
- (b) A GROUP WIDER THAN FOUR DIGITS NEEDS NO STEP SHAPE OF ITS OWN: a token on all 16 digits of its box, executed as groups of four on one run of value lines, is 4,288 cycles against the four-digit row's 4,288 (the same; at one CTA 4,254 against 4,240), exact.
- (c) WHERE A TOKEN IS ON ONE DIGIT OF ITS GROUP THE 64-COLUMN STEP IS WASTE, AND ON FOUR IT IS THE FOLD'S GAIN: groups of four digits (a token on one digit) are 6,193 cycles against the per-list row's 5,463 (13.4% over), their folds alone 4,017 against the per-list folds' 4,027 (the same), and the per-list fold at the wide kernel's registers (176 / 112 / 48) 5,405 (the same as the per-list row: the registers are not the excess). A token on all four digits of its group: the folds alone are 2,545 against 4,010 a list a digit (36.5% under), and the composed window 4,288 against 5,428 (21.0% under): the fold's gain is paid in the composed window.
- (d) NOT MEASURED: a run-time count of steps a group -- every grouped fold of the record issues a fixed number of steps a window.

THE FOLDS' ISSUE WORK IS PAID COMPOSED, ONE POINT, AND THE READ SIDE IS NOT A CONSTANT: an earlier reading of this page, that the composed window at four digits a token is bound by the read side and keeps a share of it, is WRONG. Between the records of 8e909ad0 and of this page's commit the four-digit fold stopped clearing its tiles and the barrier that followed the clear: the folds alone went 2,814 to 2,545 (269 cycles) and the composed window 4,705 to 4,288 (417 cycles), so the window is not its longer chain plus a constant; a fold's issue work (stores, fences, barriers) is paid composed at 1.55 times its alone time, sixteen warps sharing the SM's issue ports. It is ONE point: the two-pass rows (groups of four digits, which gained one barrier) agree within their spread: one barrier more is +19 alone and +29 composed, and a second point is owed before it is used as a price ([`open-work.md`](../open-work.md)).

THE BYTES A LAYOUT HOLDS, by arithmetic from the packet's layout (a token's value lines are 8 lines, 128 bytes; a window is 256 memberships a box; `kValueBytes` is 32,768): per list at one digit a token, 256 tokens, 32,768 bytes of value lines; grouped at four digits a token, 64 tokens, 8,192 bytes of value lines; grouped at 16 digits a token, 16 tokens, 2,048 bytes of value lines. Grouping's larger purpose is the stream's bytes when dense: a token on k digits of a group is laid once, with its coefficients for those digits, so its value lines are k times fewer than a list a digit's.

NOT MEASURED HERE: this kernel lands the whole `kValueBytes` of every packet whatever the layout (`issue_one`), and the cell is not bound on global memory (one CTA against 132), so no row shows the traffic a grouped layout saves; the 21.0% at four digits a token is the tensor steps' side alone. OWED: a row whose packet lands only its tokens' value lines; and the gather's cost of writing a token once. ALSO OWED, as an observation: every membership holds its own dense line (16 bytes) though a token's dense line is the same on all its digits; a grouped layout could hold it once a token.

THE PHASES, in each role's own window of the coarse row's record at 132 CTAs (CTA 0), cycles:
- the fold: image 2,330, fill 1,377, steps 2,037.
- the readout, its phases in its own window of 5,756 cycles (duration from start): issuing steps 676 from 1,118; storing the panel 164 from 1,794; storing the panel 964 from 1,958; issuing steps 799 from 2,922; storing the panel 197 from 3,721; storing the panel 2,955 from 3,918.
- the drain, its phases in its own window of 5,756 cycles (duration from start): adding 834 from 1,840; adding 1,220 from 2,675; adding 1,105 from 3,895; adding 2,574 from 4,991; adding 3,071 from 5,114.

- THE FILL'S PHASES, from the fill row's record at 132 CTAs (the mean of the fold warpgroups of CTA 0): expanding 777; writing its pad 221; loading tile starts 117; at the fill's barrier 260 cycles; the fill's own work is the phases before the barrier and its wait is `at the fill's barrier`.
- THE FOLD'S WAIT ON ITS PACKET, from the packets row's record at 132 CTAs (CTA 0, the mean of its 8 fold warps): 164 cycles a window (2.8% of the window).
- THE LONG CALL BY WALL TIME at 132 CTAs: 0.186 ms for 65,536 tokens (8 tokens a window a CTA), the call's span over its wall time 1.814 GHz (the card's clock), against the launch-derived 0.186 ms above.

Record: `measure/lab/results/carry_token/sm_90a.blocks1.json` and `sm_90a.blocks132.json`, each with its traced rows' records in `sm_90a.blocks1.trace/` and `sm_90a.blocks132.trace/`.
