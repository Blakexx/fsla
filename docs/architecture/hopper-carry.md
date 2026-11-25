# The hopper carry: the design its kernel is built from

STATUS: THE BUILD'S DESIGN. No carry kernel is built on this line ([`open-work.md`](../open-work.md)); the kernel is
built from this page and then optimized as a hill climb (Blake, 2026-11-23: "if we know the general design then we
should just implement it and optimize it as a hill climb rather than optimizing a lab kernel"). Every form says what
it stands on: MEASURED (a lab's page holds the record), RULED, THE SKELETON'S (a choice the lab left open, made so the
skeleton has every component in the shape it will live in, [`KERNEL_STANDARDS.md` section 19](../KERNEL_STANDARDS.md#s19);
the climb changes it inside that shape), NOT DESIGNED, or OPEN.

What the carry computes is [`ARCHITECTURE.md` section 6.3](../ARCHITECTURE.md#63-the-carry--the-inter-window-term): a
window at a time, the readout of the state as of the window's start by the window's readers, into `num` and `den`,
then the fold of the window's writes into the state.

## The call's shape

- A CTA OWNS A CARVE OF 512 LEAVES ([`CONSTITUTION.md` R20](../CONSTITUTION.md#r20)), in registers for the call: a
  power-of-two extent a level, APPORTIONED across the levels by the two sides' densities
  ([`cta_carve.md`](../internals/common/plan/cta_carve.md)). The measured cell -- the write side grouping 4 digits of one
  level and walking the other whole, the read side the reverse -- comes out 32 x 16, the 32 along the level the read side
  walks whole. Its two fold warpgroups hold half each, 16 x 16, split along a level the write side walks by groups smaller
  than the level first, otherwise one it walks whole; the readout warpgroup takes the carve whole. A fold's half of the
  carve is called a BOX below.
- A SIDE'S DENSITY on a level is its share of live digit-token pairs there, the prefill call's own input beside the
  masks (the library's helpers measure it from a level-side's mask; a level-side whose activation cannot make exact
  zeros has no mask, is dense, and its byte is 255 by the declaration: [`api.md`](../api.md) section 2.6); ITS GROUP
  SIZE is the density times the level's digits, rounded to a power of two from 1 to 256 (R10). The kernels read the densities, derive the groups, the carve and the split with
  that one rule, and switch once a call to the code compiled for the pattern (`uniform_switch`; R9). The host reads no density: its half of the launch is the grid's counts and the window
  ([`prefill_launch.md`](../internals/common/plan/prefill_launch.md); R24; [`open-work.md`](../open-work.md) row A-f).
  NOT BUILT: the kernels.
- A SIDE'S LEAF GROUP is the box of one group a level (Blake, 2026-11-24: "per level side groups. Each of those group
  sizes get multiplied to get the overall leaf group size per side. CTAs own their own leaf segments per side. So often
  times, a cta will own a segment of many leaf groups"). Leaf groups tile the leaf space across CTAs; a group and a CTA's
  extent on a level nest, so on a level where the group is the larger the CTAs along it are in one group, and where it is
  the smaller a CTA holds several. A GROUP'S TOKENS are those live in its group on every masked level -- the AND of the
  level-sides' group words; a level-side whose group is the level's size is WHOLE and adds nothing, its
  mask never read (one with no mask is whole by its byte of 255). "Within a group, the
  CTAs all share the exact same token schedule. That's the definition of a group."
- EACH SIDE HAS ITS OWN LOOP ORDER over the levels, its sparsest outermost (Blake, 2026-11-25: "Obviously each side has
  its own loop order. That's why publish has to order the smem state in a different orientation than the registers"):
  the fold's by the write side's densities, the readout's by the read side's, and the image a fold publishes is laid in
  the readout's.
- ONE CTA A MULTIPROCESSOR (R15), a CTA a (carve, batch-head). Where a call has more CTAs than the card has
  multiprocessors they run in waves, each CTA its carve's whole segment; carry CTAs meet each other only at the output's
  adds (R22). THE SCHEDULE (Blake, 2026-11-25: "Write density ignoring read was group scheduling. right. In a group the
  CTAs are by read"): the write side's leaf groups are taken in the order of its loop, so the groups active together
  share the most tokens, head by head; inside a group its CTAs are ordered by the read side, the level sparsest for the
  read side varying fastest, so CTAs active together add into different output lines. MEASURED
  ([`lab/drain_rate.md`](../lab/drain_rate.md)): a line of adds is 19.2 cycles into a CTA's own rows, 24.8 spread over a
  window, 46.7 with every CTA on the same 64 rows. NOT BUILT: the schedule.
- THE WINDOW is the card's (1,024 tokens on hopper, provisional), global to every CTA.
- THE SKELETON'S CELL: two levels of 256 digits, each side walking one level by its lists and the other whole, 64
  value channels, a segment a whole number of windows, the state zero at entry and not stored. What a wider contract
  needs beyond it is under [The order of the build](#the-order-of-the-build).

## What a CTA makes

THERE IS NO GATHER KERNEL (Blake, 2026-11-25: "if we can drop the gather that would be great. If it is actually
possible to do the in cta scatter loads for both sides it's ideal"; "Today's thinking is my preferred design. Should we
just build it for real"). A carry CTA makes everything it uses, on both sides, from the bundle's planes and the
prepared group words ([the mask-preparation pass](mask-preparation.md#the-mask-preparation-pass)).

- ITS LISTS. For each of its groups, a side: the group's live tokens of the window, in order -- the group words ANDed
  across the side's masked levels and counted inline, a popcount and a running sum over the window's words. One form
  whether a side has one masked level or several (Blake, 2026-11-25: "the counts are useless anyway? We need to find a
  way to do the counting inline anwyay"). NOT BUILT. MEASURED of a lab's read-side listing, its counts handed to it
  ([`lab/carry_token.md`](../lab/carry_token.md), paragraphs (k) and (l)): 107 to 238 cycles a window a CTA over the same
  row without it at one CTA, 614 to 700 at 132, the growth at the full grid not explained; an inline scan of four digits
  at once on drain warps of 56 registers did not fit.
- ITS VALUE LINES. A write group's live tokens' value lines, fetched by token index straight into the step's tile in
  shared memory by the threads' asynchronous copies, 8 lanes a line, in step order: the data picks where a line comes
  from and the layout where it lands. MEASURED ALONE ([`lab/value_fetch.md`](../lab/value_fetch.md)): 16
  warps fetch a scattered line in 2.70 cycles where it lands as its 128 bytes whole (2.55 where a unit's 16 CTAs ask
  for the same lines), and in 16.6 in either form tried that lands it as a step's tile -- its 8 pieces in bricks, or a
  piece of 32 lines an instruction. SO A LINE LANDS WHOLE, a row of the step's tile with its pieces in the order the
  step's 128-byte swizzle reads ([`internals/carry/carry_values.md`](../internals/carry/carry_values.md#tile)). NOT
  MEASURED: the fetch while the CTA folds.
- ITS COEFFICIENTS. A token's rows on the CTA's digits, loaded by the token's index into registers and expanded there,
  on both sides (RULED 2026-11-23: "the reason that coefficnets can afford to scatter is because we already need to load
  them into registers to expand them anyway"). The fold then stores a thread's expanded lines into its step's
  coefficient tile in shared memory, at lines the list's order fixes: the step's two operands are both shared memory,
  the value lines on the 64 channels and the coefficient tile on the leaves, as the lab's composed fold ran; a step fed
  from registers is the climb's. MEASURED ([`lab/carry_token.md`](../lab/carry_token.md)): the row's own load 6 to 10%
  of the grouped window over a packed line.
- A STEP is 16 tokens by up to 64 of a group's leaves (R25); a group wider than a step is more steps on the same value
  lines, and a short step is padded on the coefficient side, a zero byte, never by zero value lines (RULED 2026-11-19).
- THE READ SIDE'S RESULTS. A reader's sums are stored once, in the list's order, into a PANEL a read group in shared
  memory; the drain warps walk the panel as a ring and add each line at its token's output line (Blake, 2026-11-25: "keep
  a contiguous panel of data to drain per group, in like a ring, that drain just loops through in the background and
  sends as atomics"; "instead of having to read and write back, we just store the results and let drain take them"). The
  adds follow the live readers, so the traffic follows the density, and a full ring holds the readout back (R22).
  MEASURED ([`lab/drain_rate.md`](../lab/drain_rate.md)): an SM takes a 128-byte line of adds about every 9.6 cycles,
  whatever the warps issuing. NOT BUILT in this form.
- SHARED MEMORY IS CONFLICT-FREE BY CONSTRUCTION ([`KERNEL_STANDARDS.md` section 30](../KERNEL_STANDARDS.md#s30)): the
  layout picks every address a value line, a coefficient tile or a panel entry is stored at. NOT DESIGNED: the
  listing's compaction, where a lane's place follows the counts before it.

## The CTA

512 threads at the multiprocessor's whole register file ([`lab/carry_token.md`](../lab/carry_token.md)):

| the role | threads | registers a thread | what it holds |
|---|---|---|---|
| a FOLD warpgroup a box, two | 128 each | 176 | the box's state, 16 accumulator sets of 8; it lists its groups, issues its value lines' copies, loads and expands its entries' coefficients and steps them |
| the READOUT warpgroup | 128 | 104 to 112 | a pass's 8 slots' sums, 64 a thread, and two passes of entries and rows on their way; who lists its readers is the climb's |
| DRAIN warps, four | 32 each | 48 to 56 | a batch of the panel's lines; a reader's output line is the CTA's lists' |

The registers are the lab kernel's split; the kernel's own is the climb's, the listing included.

A warpgroup's registers are set once, at its role's head; two counts in one warpgroup hang the CTA. A hand-off
between roles is a named barrier, the producer arriving and the waiter syncing ([`internals/carry/carry_kernel.md`](../internals/carry/carry_kernel.md#handoff)). Shared memory, through the one layout ([`internals/common/smem_layout.md`](../internals/common/smem_layout.md)): a fold's step tiles of value lines, the lists, the two folds' images, the readout's coefficient
tiles, the panels' ring.

A STEP IS NEVER WIDER THAN 64 COLUMNS (R25). THE SKELETON'S fold is the group of four digits alone: a token's value
lines once, its row the coefficients of the group's digits it is live on and zeros, a step 16 tokens by 64 columns,
and a group's steps a loop on its list's count whose test is a warp vote. The group a step lays out is its level's, from the side's density
([the call's shape](#the-calls-shape)); the step shapes that execute a group are the climb's.

## A window

1. THE IMAGE. Each fold warpgroup publishes its state as of the window's start, bf16, a tile group at a time behind
   the readout's read of the window before.
2. THE READOUT, a half of the read digits at a time: the wait on the half's tile groups, then its passes, a loop on
   its lists' count (a warp vote). A pass: a thread expands a reader's 16 bytes on a box by its byte on its digit
   to its two lines of a coefficient tile; 16 steps, a slot's image tile by its coefficient tile on each box, into
   one accumulator; the sums to a panel as f16 pairs; the drain warps add each reader's line at its token's.
3. THE FOLD, a group at a time: its list (made a window ahead); the wait on its value lines' landing; a thread's two
   entries expanded; the group's steps; the next lines' copies issued. A step tile is a tenant the copies write and
   the steps read (R21): its completion edge is the wait on the landing, and its release edge the barrier behind the
   last step that reads it, before which no copy into that tile is issued -- two tiles, so the next group's lines land
   while this one's steps run. NOT BUILT.

MEASURED of the composed window ([`lab/carry_token.md`](../lab/carry_token.md), the two readouts' timelines and the
readout's pass by phases): the folds wait on the readout's read of their tile groups; with the fixed readout the drain
waits on a panel filled about half a window, and with the listed readout the drain is busy most of it and the readout
waits longer on a panel drained -- the read side's chain is the readout and the drain together. The listed forms run
more instructions in both roles (the page's static counts) and are slower in each phase the page splits.

## Entry and exit

A CTA loads its boxes from the paged state at entry and stores what it wrote at exit
([`ARCHITECTURE.md` section 3](../ARCHITECTURE.md#3-the-state)). MEASURED apart from the carry
([`lab/state_move.md`](../lab/state_move.md), [`lab/state_image.md`](../lab/state_image.md)); not composed with it.

## The scales, the den and the output's line

- THE SCALES. A row's bytes are multiplied by the row's fp32 scale where its coefficient is formed, a value by its
  token's; every sum leaves the reader's own factor out ([`fsla/ops/prefill.py`](../../fsla/ops/prefill.py)). No lab
  ran on scaled operands: it is built with the kernel and checked against the oracle.
- THE DEN is the same recurrence on ones: the fold's mass by steps whose A is the coefficient tile, a brick more of
  each image tile, a column more of the readout's step, a word more of a panel row and an add more of the drain.
  MEASURED in the digit form at 5 to 6% of a window ([`lab/carry_window.md`](../lab/carry_window.md)); not in the
  packet form.
- THE OUTPUT'S LINE across CTAs is fp32 or one f16 line a token scaled by a power of two from a coarse bound, as the
  plan states (R17).

## The order of the build

Skeleton first ([`KERNEL_STANDARDS.md` section 19](../KERNEL_STANDARDS.md#s19)): every component below as a stub with
its real signature, shared-memory tenants and barriers, the composition correct in shape; then one component at a
time, each exact against the fp64 oracle at the skeleton's cell before the next.

1. The prefill launch function (its carve, units and window are built: [`internals/common/plan/prefill_launch.md`](../internals/common/plan/prefill_launch.md)): the carve, the folds' split, each side's loop order, the schedule, the window and
   the grid, from the descriptor's facts alone, and the workspace's layout.
2. The mask-preparation pass ([`mask-preparation.md`](mask-preparation.md#the-mask-preparation-pass)), reduced to the group
   words and totals.
3. The carry's skeleton at the cell, then its components, the state zero at entry: the lists; the value lines' fetch;
   the coefficients; the fold's steps; the image; the readout; the panel and the drain -- `num` exact at the skeleton's
   cell; then the scales; then the den.
4. The intra beside it ([`lab/intra_window.md`](../lab/intra_window.md)), and the op's two halves against
   `tests/oracle/test_prefill_vs_oracle.py`; then the producer.
5. The state at entry and exit, paged and dense.
6. The contract's width: value widths, level widths, depth 3 and 4, every pattern of listed and whole levels (R10).
7. The climb.

## Answered in the kernel, not before it

- OVERLAP INSIDE A WARPGROUP: the next pass's or packet's work begun under the steps in flight (the lab's declared
  wait-after-issue debt).
- UNEVEN HALVES: what a fold waits where its box's lists are quiet and its neighbour's busy; the split along a level
  walked whole is its alternative (R20).
- THE LISTING: who runs it, in what registers, its inline count and a compaction free of bank conflicts; what its
  cost at the full grid is made of. Reader lists written once a head by another kernel are NOT A FALLBACK the build
  may take (Blake, 2026-11-23: "if we run into that you need to stop nad let me know before doing it").
- THE MEMORY PATH'S BUDGET: the value lines' fetch, the coefficient rows' loads and the drain's adds share an SM's one
  path; by the lab's rates the drain's adds are the largest of them.
- THE REGISTERS' SPLIT between the folds, the readout and the drain, the listing included.
- A PATTERN THAT IS SLOW (Blake, 2026-11-25: "we shouldn't overfit on one pattern, it's supposed to be generic. And if
  some patterns are bad, it's because those patterns themselves are bad, not the kenrel"): the kernel is one generic
  form for every pattern of masked and whole levels; how slow a pattern is decides whether it is worked on.
- THE READ SIDE'S COEFFICIENT FORM (RULED 2026-11-14, chosen by measurement): a reader's byte on its digit multiplied
  into its coefficients, as built, or applied to its sums after the step.

## Blackwell

NOT STATED. No blackwell carry is designed and no blackwell card measured (R25's step shape is hopper's).
