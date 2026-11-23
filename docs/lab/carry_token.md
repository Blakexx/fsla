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
  of its own (`kSplit`). The one-step-a-group fold (`kFull`: four steps a window, every line of a group's tile an entry's) clears nothing unless the cleared probe asks it to;
  the two-pass fold (a token on one digit) clears each pass's eight tiles, then stores its entries, and its warpgroup's barrier
  follows every pass's wait.
- THE CLEARED PROBE (`kCleared`, on the one-step fold): the tiles' room cleared and the warpgroup's barrier behind the clear before
  the entries are stored -- issue work that fold does not need, since every line of its tiles is an entry's. Its rows against the
  uncleared fold's price a fold's issue work composed and alone within one record.
- THE COUNTED FOLD (`kCounted`, with the cleared probe): a group's steps are a count of the packet's, a byte a group of the header,
  one or two. A count is a choice between two straight blocks, one step or two, each its own fence and commit group, made outside
  them with the group's accumulators held across the choice; a group's steps take the group's own two places of the tiles' room and
  of the header's starts, so every operand of a step is a constant, and any other count traps. A loop on the loaded count, each thread deciding for itself, was
  not an option: ptxas 13.5 serializes the steps (C7519, "serialized due to program dependence on compiler-inserted WG.AR in divergent
  path"), which the lab's checks refuse -- a compile fact of this kernel.
- THE TABLED PROBE (`kTabled`; `kSpread` with it): the packet holds an entry's token index, two bytes in the header (contiguous by
  entry), where it held the entry's 16-byte dense line; the fold loads the index from the packet in global memory at the window's start
  and then the token's row from the tokens' table by it -- two chained global loads an entry, the second scattered by token, issued
  under the image as the packed line's load is. The table is 1 MB (the lab's dense table), or with `kSpread` the same rows 32 KB apart,
  a 2 GB table. Measured about these loads: only the rows' cycles a window, against the packed-line rows ((h)); no counter, and the
  access table below is the shared-memory accesses'. Rows: the four-digit grouped fold plain and its folds alone, and the per-list fold,
  on each table.
- THE WINDOW-AHEAD PROBE (`kAhead`, with the tabled probe): a window's rows are loaded a window ahead, behind the entries of the window
  before, and the fold holds the next window's two lines through its steps. Its kernels are 176 / 112 / 48 registers a thread (the
  grouped fold's), also the per-list rows': a compile fact of this kernel, the per-list fold spills at 168 with the next window's
  two lines held through its steps. (Under the tabled probe alone, and in the packed forms, the lines are loaded at the window's
  start, before the image.) Rows: the same six as the tabled probe's, each loaded a window ahead.
- THE VOTED LOOP (`kLooped`, with the cleared probe): a group's steps are a loop a group on the packet's count, any count, whose test is a
  warp vote (one answer a warp: the thread-by-thread loop above is what ptxas serialized); a step's tile and its A's start are the window's
  next in the order issued, so the operands are run-time values. ptxas 13.5 does not serialize this loop (no C7519), and its steps are
  rolled, which the checker's `step_rolled` rule names and the rows declare ([`step_rolled`](step_rolled.md) prices a rolled step alone).
  THE UNEVEN PACKET (layout 8): a window's four groups take four, two, one and one steps -- a token on one digit of the first group (a
  step a digit), on two digits of the second, on all four of the third and of the fourth -- the eight steps' tiles and starts the
  window's next in that order, the header counting each group's steps.
- THE TWO NEW PACKETS: a token on two digits of its group (layout 6: a group's 32 tokens are two steps of 64 columns, a token's value
  lines once, an entry a (token, digit) with the even places on the group's digits 0 and 2 and the odd on 1 and 3, so half of each
  tile's lines are no entry's and the clear writes them), and the four-digit tokens laid in a group's two tile places (layout 7: a
  step a group, each group's step in the first of its two places).
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
isolation row's change, the read side idle. Forty-seven rows: plain, edges, coarse, the seven detail rows, every stamp;
the coarse kernel with the read side idle (not checked: its kernel is not the carry's); the plain and coarse
kernels on packets whose lists `2 p` and `2 p + 1` share four tokens, written once; and thirty-three grouping rows: the plain kernel
on groups of four digits and its folds alone, the per-list fold at the grouped kernel's registers, a token on all four digits
of its group in groups of four and in a list a digit, each with its folds alone, a token on all 16 digits of its box, the
per-column form on the two packets, six rows of the counted fold -- the four-digit fold with its tiles cleared and its folds alone,
and a group's steps counted at run time on the four-digit tokens in a group's two tile places and on a token on two digits of its group,
each plain and with the read side idle -- and five rows of the voted loop (on the four-digit packets, plain and with the read side idle;
on the two-digit packets; on the uneven packet, plain and with the read side idle) and six rows of the tabled fold (the four-digit grouped fold from the tokens' table and from the 2 GB table, each plain and with the read side idle, and the per-list fold from each) and the same six rows loaded a window ahead. The state and every output row are checked exactly in
all but the thirteen isolation rows. The plain row also runs one long call, timed by the host's clock: its span in
the card's cycles over its wall time, and the call by wall time scaled to 65,536 tokens (a window counted as 8 tokens
a CTA, as `carry_window` counts it). THE PACKETS ARE THE HOST'S, playing the gather.

ACCESS TABLE (KERNEL_STANDARDS [§30.1](../KERNEL_STANDARDS.md#s30): each instruction that touches a shared buffer, by lanes, addresses and banks; the pad stores', the grouped folds' tile clear and entry stores', and the cleared and counted folds' clear and entry stores', and the voted loop's entry stores' rows; the fold's other accesses are not yet in it).

| buffer | instruction | lanes and addresses | banks | measured | owed |
|---|---|---|---|---|---|
| the fold's coefficient tiles | the pad stores: 6 `STS.128` of the fold, a pad's two zero lines written by a thread's third word | a warp's 32 stores, a thread's lines named by its packet's words, so the data picks the addresses | where the data puts them: the stores meet on banks by chance, conflict-free for no shape | 43.5 excess shared wavefronts a CTA-window (367,276 over the timed launch's 8,448 CTA-windows, 132 CTAs of 64 windows, the launch's dump), accepted as declared `BANK_CONFLICT <= 44` | the gather places each pad's lines by bank, a placement rule in the packet's format (KERNEL_STANDARDS §30.2: priced against the fold's cost before it enters the format), then the declaration drops to the conflict-free count |
| the grouped folds' coefficient tiles (the two-pass fold, groups of four digits) | the tile clear: a pass's 8 tiles cleared before its entries are stored, 16 stores a thread of 8 bytes, behind the warpgroup's barrier | a warp's 32 lanes store 32 consecutive 8-byte words an instruction (a thread's word `(x & 31) * 8` bytes into its warp's 4,096-byte quarter of the tiles, an instruction `256` bytes on) | every bank twice, the 256 bytes an instruction in two wavefronts: conflict-free by the lanes' addresses (the source's own count; no profile of this kernel) | nothing: the record holds no counters (`counters` empty: not yet read on this box); the declared `BANK_CONFLICT <= 44` bounds the kernel's excess, the pad stores' | the dump of this kernel (`lab -- ncu`, then `ledger`), the clear's wavefronts counted against the conflict-free 2 an instruction |
| the grouped folds' coefficient tiles (both grouped folds) | the entry stores: two 16-byte lines an entry, `STS.128`, at the lines its word names (`word & 0xffff` and `word >> 16`), two entries a thread in the one-step-a-group fold, one an entry a pass in the two-pass fold | the host lays the lines: in the one-step-a-group fold (a token on all four digits) a warp's lanes are 32 consecutive entries, so each quarter-warp phase of 8 lanes stores 8 consecutive 16-byte lines (`128 q + 32 digit + place`); in the two-pass fold the lane's line is its entry's position in its list (lists of 8 to 24), so the data picks the addresses | the one-step fold: conflict-free by the host's layout (each phase 8 consecutive lines); the two-pass fold: by chance, as the pad stores | nothing beyond the same declaration `BANK_CONFLICT <= 44` and no counters: the one-step fold's by the layout's arithmetic only | the dump of both kernels; for the two-pass fold a placement rule by bank in the packet's format (KERNEL_STANDARDS §30.2), as the pad stores owe |
| the cleared and counted folds' coefficient tiles (`kCleared`, `kCounted`) | the tile clear: the tiles' room cleared before the entries are stored, 16 stores a thread of 8 bytes, behind the warpgroup's barrier (the two-pass fold's shape) | a warp's 32 lanes store 32 consecutive 8-byte words an instruction (a thread's word `(x & 31) * 8` bytes into its warp's 4,096-byte quarter of the tiles, an instruction 256 bytes on) | every bank twice, 256 bytes an instruction in two wavefronts: conflict-free by the lanes' addresses (the source's own count; no profile of these kernels) | nothing: the record holds no counters; the declared `BANK_CONFLICT <= 44` bounds each kernel's excess, the pad stores' | the dumps of these kernels (`lab -- ncu`, then `ledger`), the clear's wavefronts counted against the conflict-free 2 an instruction |
| the counted folds' coefficient tiles (layouts 6 and 7) | the entry stores: two 16-byte lines an entry, `STS.128`, at the lines its word names (`line` and `line + 16`), two entries a thread; layout 6 leaves half of each tile's lines to the clear | the host lays the lines. Layout 7 (a token on four digits, a step a group in the first of its two places): a warp's lanes are 32 consecutive entries, so a quarter-warp phase of 8 lanes stores 8 consecutive lines (`256 q + 32 digit + place`). Layout 6 (a token on two digits): entry `64 q + 32 s + 16 j + k` stores line `128 step + 32 (2 j + (k & 1)) + k`, so a phase of 8 lanes (`k` 0 to 7 or 8 to 15) stores lines `0, 33, 2, 35, 4, 37, 6, 39` from its base (or from 8): the digit alternates across lanes | layout 7: 8 consecutive 16-byte lines a phase, conflict-free; layout 6: the 8 lines of a phase fall on the 8 distinct 16-byte bank groups (the lines modulo 8 are `0` to `7`), conflict-free by the layout's arithmetic. Neither is profiled | nothing beyond the same declaration `BANK_CONFLICT <= 44` and no counters: both by the layout's arithmetic only | the dumps of both kernels, to turn the arithmetic into a count |
| the voted loop's coefficient tiles (layout 8, `kLooped`) | the entry stores of the uneven packet, two 16-byte lines an entry (`line` and `line + 16`), two entries a thread; the clear writes every line no entry does | the host lays the lines. First group, a token on one digit: step `s`'s tile (tiles 0 to 3) is written at digit `s` only, 32 of its 128 lines, `base + 32 s + k` (lane `k` of 16), the other 96 the clear's. Second group, a token on two digits: tiles 4 and 5, digits `2 j + (k & 1)` as layout 6, 64 of 128 lines each, the other 64 the clear's. Third and fourth groups, a token on all four: tiles 6 and 7, every line an entry's | first and third and fourth groups: a quarter-warp phase of 8 lanes stores 8 consecutive 16-byte lines, conflict-free; second group: the phase's lines modulo 8 are `0` to `7`, conflict-free (layout 6's arithmetic). None profiled | nothing beyond the same declaration `BANK_CONFLICT <= 44` and no counters: by the layout's arithmetic only | the dump of the kernel, to turn the arithmetic into a count |

MEASURED (NVIDIA H100 80GB HBM3, 132 SMs, Nebius preemptible VMs grvz4frf and 16rnj0ng, driver 580.196.07, toolkit 13.5, the clock locked; 2026-11-23 02:11 and 02:06 UTC (the one-CTA and the full-grid records), the
experiment at 1f5acda6, 7 repetitions; the labels name the boxes of the two records, which hold no box's id; the 34 checked rows pass at one CTA
and at 132, the 13 isolation rows are unchecked (their kernels are not the carry's); `sm_90a.blocks1.json` and `sm_90a.blocks132.json`, each with its traced rows' records beside it). Cycles a window a CTA:

| row | one CTA | 132 CTAs |
|---|---:|---:|
| plain | 5,417 | 5,419 |
| the launch's two edges stamped | 5,269 | 5,278 |
| coarse | 5,712 | 5,727 |
| coarse, the waits on the tile groups read | 5,955 | 5,964 |
| coarse, the waits on the tile groups published | 5,974 | 5,995 |
| coarse, the waits on the panels filled | 5,704 | 5,717 |
| coarse, the waits on the panels drained | 5,701 | 5,748 |
| coarse, the fold's steps by eighths | 6,256 | 6,196 |
| coarse, the fold's fill by phases | 5,888 | 5,931 |
| coarse, the waits on the packets landed | 5,771 | 5,795 |
| every warp's timeline stamped | 6,169 | 6,197 |
| coarse, the read side idle (not checked) | 4,026 | 4,024 |
| lists overlapping on shared tokens, plain | 5,425 | 5,427 |
| lists overlapping on shared tokens, coarse | 5,722 | 5,733 |
| groups of four digits, plain | 6,234 | 6,183 |
| registers 176 / 112 / 48, plain | 5,352 | 5,382 |
| a token on all four digits of its group, groups of four, plain | 4,262 | 4,278 |
| a token on all four digits of its group, groups of four, coarse, the read side idle (not checked) | 2,646 | 2,535 |
| a token on all four digits of its group, a list a digit, plain | 5,381 | 5,379 |
| a token on all four digits of its group, a list a digit, coarse, the read side idle (not checked) | 4,014 | 4,015 |
| groups of four digits, coarse, the read side idle (not checked) | 4,141 | 4,056 |
| a token on all 16 digits of its box, groups of four on one run of value lines, plain | 4,256 | 4,278 |
| a token on all four digits of its group, a list a digit, the box's width word choosing its fold, plain | 5,273 | 5,296 |
| a token on all four digits of its group, groups of four, the box's width word choosing its fold, plain | 4,229 | 4,252 |
| a token on all four digits of its group, groups of four, its tiles cleared, plain | 4,704 | 4,706 |
| a token on all four digits of its group, groups of four, coarse, its tiles cleared, the read side idle (not checked) | 2,872 | 2,810 |
| a token on all four digits of its group, groups of four in a group's two tile places, its tiles cleared, a group's steps counted at run time, plain | 4,977 | 4,890 |
| a token on all four digits of its group, groups of four in a group's two tile places, coarse, its tiles cleared, a group's steps counted at run time, the read side idle (not checked) | 2,919 | 2,916 |
| a token on two digits of its group, groups of four, its tiles cleared, a group's steps counted at run time, plain | 5,150 | 5,076 |
| a token on two digits of its group, groups of four, coarse, its tiles cleared, a group's steps counted at run time, the read side idle (not checked) | 3,124 | 3,098 |
| a token on all four digits of its group, groups of four, its tiles cleared, a group's steps a voted loop on its count, plain | 4,991 | 4,949 |
| a token on all four digits of its group, groups of four, coarse, its tiles cleared, a group's steps a voted loop on its count, the read side idle (not checked) | 3,196 | 3,202 |
| a token on two digits of its group, groups of four, its tiles cleared, a group's steps a voted loop on its count, plain | 5,419 | 5,387 |
| groups of four taking four, two, one and one steps, its tiles cleared, a group's steps a voted loop on its count, plain | 5,458 | 5,399 |
| groups of four taking four, two, one and one steps, coarse, its tiles cleared, a group's steps a voted loop on its count, the read side idle (not checked) | 3,655 | 3,680 |
| a token on all four digits of its group, groups of four, an entry's dense line from the tokens' table, plain | 4,525 | 4,555 |
| a token on all four digits of its group, groups of four, coarse, an entry's dense line from the tokens' table, the read side idle (not checked) | 2,811 | 2,978 |
| a token on all four digits of its group, groups of four, an entry's dense line from a 2 GB table of the tokens, plain | 4,647 | 4,723 |
| a token on all four digits of its group, groups of four, coarse, an entry's dense line from a 2 GB table of the tokens, the read side idle (not checked) | 2,893 | 3,112 |
| an entry's dense line from the tokens' table, plain | 5,805 | 5,806 |
| an entry's dense line from a 2 GB table of the tokens, plain | 6,155 | 6,505 |
| a token on all four digits of its group, groups of four, an entry's dense line from the tokens' table, loaded a window ahead, plain | 4,554 | 4,592 |
| a token on all four digits of its group, groups of four, coarse, an entry's dense line from the tokens' table, loaded a window ahead, the read side idle (not checked) | 2,859 | 3,138 |
| a token on all four digits of its group, groups of four, an entry's dense line from a 2 GB table of the tokens, loaded a window ahead, plain | 4,754 | 4,800 |
| a token on all four digits of its group, groups of four, coarse, an entry's dense line from a 2 GB table of the tokens, loaded a window ahead, the read side idle (not checked) | 2,968 | 3,227 |
| an entry's dense line from the tokens' table, loaded a window ahead, plain | 6,256 | 6,145 |
| an entry's dense line from a 2 GB table of the tokens, loaded a window ahead, plain | 6,453 | 6,498 |

WHAT IT DECIDES. THE WINDOW AT ONE CTA IS THE WINDOW AT THE FULL GRID (5,417 and 5,419, 0.0% apart; called the same within 2.0%): 132 CTAs each landing two packets a window do not bind on global memory at this cell. THE LAYOUT IS THE GATHER'S: lists overlapping on tokens written once cost the carry nothing (at one CTA 5,425 against 5,417: the same; at 132 CTAs 5,427 against 5,419: the same; called the same within 2.0%) and are exact (the checks of both overlapping rows, at both grids). THE FOLD'S WINDOW, from the coarse row's record at 132 CTAs (`bazel run //tools:lab -- trace carry_token`; CTA 0, the mean of its 2 fold warpgroups, a window of 5,690 cycles in their own windows): its image 2,419 cycles (42.5%), its fill 1,297 (22.8%), its 32 steps 1,973 (34.7%); the image phase is the longest and the fill phase the shortest (the longest phase is 1.86 times the shortest). THE READ SIDE COSTS THE FOLDS 29.7% OF THE WINDOW at 132 CTAs (29.5% at one CTA): the coarse row's window is 5,727 cycles and the same kernel with the read side idle 4,024 (the isolation row, not checked: its kernel is not the carry's); the fold's 32 steps are 1,973 cycles and, with it idle, 2,036 (3.2% over the coarse row). THE MARKS' COST: the coarse row's window is 5,727 against the plain row's 5,419 at 132 CTAs (5.7% over, its phases' starts and the launch's edges on both clocks); the figures above are the coarse row's.
BY THE LAUNCH'S OWN CLOCKS the coarse row's launch at 132 CTAs (CTA 0) is 365,767 cycles in 200,128 ns: the cycle counter runs 1,828 cycles a microsecond (1,711 at one CTA, CTA 0); the plain row's window of 5,419 cycles is 2.96 microseconds, and 65,536 tokens (62 windows a CTA, a window counted as 8 tokens a CTA as `carry_window`'s wall-clock rows count it) some 0.18 ms. NOT MEASURED: the gather that writes the packets; a cell where the packets' bytes grow with duplication.

THE GROUPED FOLD (a group's tokens once, a step 16 tokens' value lines against a tile of 16 tokens by four digits' 64 columns), cycles a window a CTA at 132 CTAs unless said, every row exact:
- (a) A STEP SHAPE CHOSEN BEFORE THE WINDOW LOOP IS FREE: the per-column kernel (one kernel holding both folds, the box's width word read before its window loop) is, at 132 CTAs, 5,296 on per-list packets against the per-list kernel's 5,379 (the same; at one CTA 5,273 against 5,381) and 4,252 on group-of-four packets against the group-of-four kernel's 4,278 (the same; at one CTA 4,229 against 4,262), exact.
- (b) A GROUP WIDER THAN FOUR DIGITS NEEDS NO STEP SHAPE OF ITS OWN: a token on all 16 digits of its box, executed as groups of four on one run of value lines, is 4,278 cycles against the four-digit row's 4,278 (the same; at one CTA 4,256 against 4,262), exact.
- (c) WHERE A TOKEN IS ON ONE DIGIT OF ITS GROUP THE 64-COLUMN STEP IS WASTE, AND ON FOUR IT IS THE FOLD'S GAIN: groups of four digits (a token on one digit) are 6,183 cycles against the per-list row's 5,419 (14.1% over), their folds alone 4,056 against the per-list folds' 4,024 (the same), and the per-list fold at the wide kernel's registers (176 / 112 / 48) 5,382 (the same as the per-list row: the registers are not the excess). A token on all four digits of its group: the folds alone are 2,535 against 4,015 a list a digit (36.9% under), and the composed window 4,278 against 5,379 (20.5% under): the fold's gain is paid in the composed window.
- (d) THE PADDED FORM IS NO ROW HERE: a fold that always issues a group's two steps with the absent one on zeros is not measured (see (f)).
- (e) A FOLD'S ISSUE WORK, A PAIR OF ROWS: the four-digit grouped fold with its tiles' room cleared and the clear's barrier (4,706 composed, 2,810 with the read side idle) against the same fold without them (4,278 and 2,535) is 427 cycles more composed and 275 more with the read side idle, 1.55 times; at one CTA 442 and 226, 1.95 times. The clear is shared-memory stores and a warpgroup barrier, which every entry of that fold's tiles makes needless: they cost more composed than alone. The record names no cause (it holds no counter).
- (f) A GROUP'S STEPS COUNTED AT RUN TIME: the count is a byte a group in the packet's header, and a group's steps are a choice between two straight blocks (one step or two, each its own fence and commit group, the group's accumulators held across the choice) made outside them, in the group's own two places of the tiles' room and of the header's starts; any other count traps. (A loop on the loaded count, each thread deciding for itself, was serialized by ptxas 13.5, C7519 "serialized due to program dependence on compiler-inserted WG.AR in divergent path", which the lab's checks refuse: a compile fact of this kernel; (g)'s loop, whose test is a warp vote, is not.) At one step a group the counted fold is 4,890 composed and 2,916 with the read side idle against the cleared fixed-block fold's 4,706 and 2,810: 184 more composed and 106 more alone, 1.73 times (at one CTA 273 and 47, 5.83 times): not the clear's 1.55, so the two kinds of issue work are not one price. At two steps a group (a token on two digits of its group) it is 5,076 and 3,098, 187 more composed and 182 more alone than the one-step rows (at one CTA 173 and 206); the rows are exact. What follows from those two differences alone: for a count of one or two, the choice costs about what a second step does. NOT MEASURED: a fold that always issues two steps a group with the absent one on zeros (counts over two a group are (g)'s).
- (g) A GROUP'S STEPS AS A VOTED LOOP ON ITS COUNT: a loop a group on the packet's count whose test is a warp vote, a step's tile and start the window's next in the order issued: any count. On the four-digit packets (one step a group) it is 4,949 composed and 3,202 with the read side idle against the cleared fixed block's 4,706 and 2,810: 243 more composed and 392 more alone (at one CTA 287 and 324); against the choice of straight blocks on the same four-digit tokens (the counted one-step row, 4,890 and 2,916) it is 59 more composed and 285 more alone (at one CTA 14 and 277). On a token on two digits of its group it is 5,387 against the counted two-step row's 5,076 (310 more; at one CTA 5,419 against 5,150, 269 more). On the packet whose groups take four, two, one and one steps (a token on one digit of the first group, on two of the second, on all four of the third and the fourth) it is 5,399 composed and 3,680 with the read side idle (at one CTA 5,458 and 3,655). Every row is exact, the uneven one included. A compile fact of this kernel: ptxas 13.5 does not serialize the voted loop (no C7519), and its steps are rolled, which the lab's `step_rolled` rule names and the rows declare ([`step_rolled`](step_rolled.md) prices a rolled 16-column step alone). NOT MEASURED: a window that needs more steps than the tiles' room holds in one pass (eight a box), a second pass, an outer loop on the count of passes; what carries a rolled step's cost (the vote, the branch, the run-time operands, the overlap between a group's steps).
- (h) WHO READS A MEMBERSHIP'S DENSE ROW: the packet's packed 16-byte line, or the fold's own load from the tokens' table by a two-byte index in the packet (two chained global loads an entry at the window's start; the rows 16 bytes; the table 1 MB, or the same rows 32 KB apart, 2 GB). The four-digit grouped fold on the table is 4,555 composed and 2,978 with the read side idle against the packed line's 4,278 and 2,535: 277 and 443 more (at one CTA 263 and 165); on the 2 GB table 4,723 and 3,112, 445 and 577 more (at one CTA 385 and 248). The per-list fold on the table is 5,806 against the packed line's 5,419, 388 more (at one CTA 388); on the 2 GB table 6,505, 1,086 more (at one CTA 738). Every row is exact. Two facts, and nothing beyond them: the table's own load is not free even from a cached table (the 1 MB rows above); and it grows with the table's size by the DISTINCT rows a packet names (the four-digit packet's 256 entries a box name 64 tokens, the per-list packet's name 256): the 2 GB table over the 1 MB one costs the grouped fold 167 cycles and the per-list fold 699, 4.2 times (at one CTA 122 and 349, 2.9 times). THE ROWS LOADED A WINDOW AHEAD (behind the entries of the window before; the fold then holds the next window's two lines through its steps) against the same rows loaded at the window's start: the grouped fold on the table 4,592 composed and 3,138 with the read side idle against 4,555 and 2,978 (at one CTA 4,554 and 2,859 against 4,525 and 2,811); on the 2 GB table 4,800 and 3,227 against 4,723 and 3,112 (at one CTA 4,754 and 2,968 against 4,647 and 2,893). The per-list fold on the table 6,145 against 5,806 (at one CTA 6,256 against 5,805); on the 2 GB table 6,498 against 6,505 (at one CTA 6,453 against 6,155). The window-ahead kernels run at 176 / 112 / 48 registers a thread and the tabled per-list rows at 168 / 120 / 56: the per-list fold at 176 / 112 / 48 with its packed lines is 5,382 (at one CTA 5,352), so (a compile fact of this kernel, from its header: the per-list fold spills at 168 registers a thread with the next window's two lines held through its steps) the per-list window-ahead rows are 762 and 1,116 over that row, and 338 over and 7 under the tabled rows at the window's start. What the figures show and nothing beyond: loading the rows a window ahead takes nothing off the window and is a little more where it differs, so the table load's cost is not the rows' latency on the window's path. The record names no cause for where the cost is paid (it holds no counter). NOT MEASURED: what the gather pays to write a packed 16-byte line a membership in its current form ([`gather_group`](gather_group.md)'s row is values only, and writes the entries' token indices, no dense line), so the two forms are not compared end to end here.

THE BYTES A LAYOUT HOLDS, by arithmetic from the packet's layout (a token's value lines are 8 lines, 128 bytes; a window is 256 memberships a box; `kValueBytes` is 32,768): per list at one digit a token, 256 tokens, 32,768 bytes of value lines; grouped at four digits a token, 64 tokens, 8,192 bytes of value lines; grouped at 16 digits a token, 16 tokens, 2,048 bytes of value lines. Grouping's larger purpose is the stream's bytes when dense: a token on k digits of a group is laid once, with its coefficients for those digits, so its value lines are k times fewer than a list a digit's.

NOT MEASURED HERE: this kernel lands the whole `kValueBytes` of every packet whatever the layout (`issue_one`), and the cell is not bound on global memory (one CTA against 132), so no row shows the traffic a grouped layout saves; the 20.5% at four digits a token is the tensor steps' side alone. OWED: a row whose packet lands only its tokens' value lines; and the gather's cost of writing a token once. ALSO OWED, as an observation: every membership holds its own dense line (16 bytes) though a token's dense line is the same on all its digits; a grouped layout could hold it once a token.

THE PHASES, in each role's own window of the coarse row's record at 132 CTAs (CTA 0), cycles:
- the fold: image 2,419, fill 1,297, steps 1,973.
- the readout, its phases in its own window of 5,709 cycles (duration from start): issuing steps 716 from 1,127; storing the panel 163 from 1,844; storing the panel 981 from 2,007; issuing steps 751 from 2,987; storing the panel 211 from 3,738; storing the panel 2,894 from 3,949.
- the drain, its phases in its own window of 5,709 cycles (duration from start): adding 862 from 1,804; adding 1,189 from 2,666; adding 1,090 from 3,855; adding 2,497 from 4,913; adding 3,261 from 5,112.

- THE FILL'S PHASES, from the fill row's record at 132 CTAs (the mean of the fold warpgroups of CTA 0): expanding 773; writing its pad 235; loading tile starts 119; at the fill's barrier 269 cycles; the fill's own work is the phases before the barrier and its wait is `at the fill's barrier`.
- THE FOLD'S WAIT ON ITS PACKET, from the packets row's record at 132 CTAs (CTA 0, the mean of its 8 fold warps): 156 cycles a window (2.7% of the window).
- THE LONG CALL BY WALL TIME at 132 CTAs: 0.184 ms for 65,536 tokens (8 tokens a window a CTA), the call's span over its wall time 1.823 GHz (the card's clock), against the launch-derived 0.184 ms above.

Record: `measure/lab/results/carry_token/sm_90a.blocks1.json` and `sm_90a.blocks132.json`, each with its traced rows' records in `sm_90a.blocks1.trace/` and `sm_90a.blocks132.trace/`.
