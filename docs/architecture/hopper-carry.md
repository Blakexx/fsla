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
  SIZE is the density times the level's digits, rounded to 1, 4, 16,
  32, 64, 128 or 256 (R10). The kernels read the densities, derive the groups, the carve and the split with
  that one rule, and switch once a call to the code compiled for the pattern (`uniform_switch`; R9). The host reads no density: its half of the launch is the grid's counts and the window
  ([`prefill_launch.md`](../internals/common/plan/prefill_launch.md); R24; [`open-work.md`](../open-work.md) row A-f).
  NOT BUILT: the kernels.
- ONE CTA A MULTIPROCESSOR (R15), a CTA a (carve, batch-head). Where a call has more CTAs than the card has
  multiprocessors they run in waves, each CTA its carve's whole segment; carry CTAs meet each other only at the output's
  adds (R22). A carry CTA also takes its packets from the gather's CTAs: [the hand-off](#the-hand-off).
- A UNIT is the CTAs inside one combination of the write side's groups: where a group is larger than the carve on its
  level, the CTAs along it read one value stream a fold -- in the measured cell the 16 CTAs along the level the write
  side walks whole. They are all the readers of the unit's value streams, and units are taken head by head (Blake,
  2026-11-23: "looping by head, by sparse box, so the gather is servign as many ctas as possible at once").
- THE WINDOW is the card's (1,024 tokens on hopper, provisional), global to every CTA.
- THE SKELETON'S CELL: two levels of 256 digits, each side walking one level by its lists and the other whole, 64
  value channels, a segment a whole number of windows, the state zero at entry and not stored. What a wider contract
  needs beyond it is under [The order of the build](#the-order-of-the-build).

## The packets

RULED 2026-11-18: the gather and the carry are decoupled by a packet contract -- "the carry just blindly consumes what
comes in, the gather lays out data in what ever way is optimal" -- and a packet's layout is implementation, not a ruled
form. What the carry is handed, a (CTA, window):

- A FOLD PACKET a box: the box's live tokens' VALUE LINES, a token once a group of digits it is live in (RULED
  2026-11-19, the grouped digit stream; R25), and a header -- each membership (a token on a digit) as the two
  coefficient lines of its step's tile it expands into, its byte on its digit, and ITS TOKEN'S INDEX, by which the
  fold loads the token's row on the box's other level itself (RULED 2026-11-23, one form for coefficients on both
  sides: "the reason that coefficnets can afford to scatter is because we already need to load them into registers to
  expand them anyway"); each group's count of steps; each step's start among the value lines. MEASURED
  ([`lab/carry_token.md`](../lab/carry_token.md)): the packet bulk-copied into shared memory a window ahead; the
  row's own load 6 to 10% of the grouped window over a packed line.
- A PACKET HOLDS 256 MEMBERSHIPS (two a fold thread) and 8 steps of 64 columns. A window whose box has more is MORE
  PACKETS, each the same trip; the fold loops on the window's count of them. NOT BUILT in the lab, which ran one
  packet a window.
- THE READ SIDE HAS NO PACKET. Fanned along the level the write side walks whole, a unit's CTAs each hold a different
  block of read digits, so a reader list is one CTA's alone, and that CTA makes it (Blake, 2026-11-23: "each cta read
  step just does the packet formation itself, since its largely going to be unique work per cta anyway"; "if the
  repeated work is cheap and saves memory, then its worth doing"). A CARRY CTA LISTS ITS OWN READERS from the
  prepared read-side words and counts: a pass is 8 slots, a slot 16 readers of one read digit -- the digit's live
  tokens of the window, in order; a reader's output line is its token's and a slot's tile its digit's, each one add
  from use; a slot no reader holds is a pad, a zero byte. THE CARRY READS NO MASK AND COUNTS NOTHING: a prepared
  word's count is the group's live tokens of the window before it
  ([the mask-preparation pass](hopper-gather.md#the-mask-preparation-pass)).
  MEASURED ([`lab/carry_token.md`](../lab/carry_token.md), paragraphs (i) to (l)): the lists consumed, with no read
  packet -- each fold warp listing two read digits' readers of the next window behind its image, the readout and the
  drain taking a reader's line from the lists, the listing reading a digit's own mask words with counts handed to
  it -- exact, 5,645 cycles a window at 132 CTAs and 5,110 at one, against
  the fixed readout's 4,278 and 4,251 and the best packet-driven row's 4,945 and 4,871. NOT BUILT, each named on that
  page: the fold's words and counts loaded a window ahead (the lab's listing loads and uses them at once, and the form
  is 10.5% dearer at 132 CTAs than at one); the drain's adds trimmed (its chunk is 181 instructions against the fixed
  drain's 63). NOT MEASURED: a count of passes from the lists; a digit of more than 16 readers a window; a reader's
  byte on its digit loaded.
- NOT BUILT: the gather that writes the fold's packets from the prepared write-side words
  and counts ([`hopper-gather.md`](hopper-gather.md)); the carry's listing from prepared GROUP words; the lab's host
  played both.

## The hand-off

THE GATHER IS THE WRITE SIDE'S (Blake, 2026-11-23: "our cta layout is hold a few write digits, fan out over read. and
that comes from teh fact that gather needs to layout values, and that read can use atomics to write back. so thats why
a ring is possible ... so gather basically just becomes write side only"). A block's value stream -- its windows' fold
packets -- is A RING of a few windows (RULED 2026-11-15), written by gather CTAs beside the carry's in one grid and read
by the unit's 16 CTAs, which are all of its readers; the gather advances when the slowest of them has ("isnt it
basically just a ring and the ctas need to signal when the ring can continue"). No barrier crosses CTAs without a
cluster, so the signals are words in global memory a waiting CTA polls. The workspace is the op's own
([the mask-preparation pass](hopper-gather.md#the-mask-preparation-pass)): it holds the ring and the prepared words and
counts, and its size is the descriptor's, the same whatever the densities are, which the host never reads.

NOT DESIGNED PAST THAT, and designed with the gather: the split of a card's multiprocessors between gather CTAs and
units; how a unit's 16 CTAs start together on persistent CTAs; what the ring's waits cost where a unit's CTAs drift
apart, their read sides unequal (R22: shown before the form is composed). MEASURED: a stream written whole before the
carry costs the carry the same ([`lab/carry_window.md`](../lab/carry_window.md)), and is what every lab ran; the pass
at a token list's values, an eighth of the carry's work ([`lab/gather_pass.md`](../lab/gather_pass.md)).

## The CTA

512 threads at the multiprocessor's whole register file ([`lab/carry_token.md`](../lab/carry_token.md)):

| the role | threads | registers a thread | what it holds |
|---|---|---|---|
| a FOLD warpgroup a box, two | 128 each | 176 | the box's state, 16 accumulator sets of 8; it lands its packets, expands its entries' coefficients and steps them; behind its image a warp lists two read digits' readers of the next window from the prepared words and counts |
| the READOUT warpgroup | 128 | 104 to 112 | a pass's 8 slots' sums, 64 a thread, and two passes of entries and rows on their way |
| DRAIN warps, four | 32 each | 48 to 56 | a batch of the panel's lines; a reader's output line is the CTA's lists' |

A warpgroup's registers are set once, at its role's head; two counts in one warpgroup hang the CTA. Every barrier
wait is a suspending poll. Shared memory: a fold's packet slots, the two folds' images, the readout's coefficient
tiles, two panels.

A STEP IS NEVER WIDER THAN 64 COLUMNS (R25). THE SKELETON'S fold is the group of four digits alone: a token's value
lines once, its row the coefficients of the group's digits it is live on and zeros, a step 16 tokens by 64 columns,
and a group's steps a loop on the packet's count whose test is a warp vote. The group a step lays out is its level's, from the side's density
([the call's shape](#the-calls-shape)); the step shapes that execute a group are the climb's.

## A window

1. THE IMAGE. Each fold warpgroup publishes its state as of the window's start, bf16, a tile group at a time behind
   the readout's read of the window before.
2. THE READOUT, a half of the read digits at a time: the wait on the half's tile groups, then its passes, a loop on
   its lists' count (a warp vote). A pass: a thread expands a reader's 16 bytes on a box by its byte on its digit
   to its two lines of a coefficient tile; 16 steps, a slot's image tile by its coefficient tile on each box, into
   one accumulator; the sums to a panel as f16 pairs; the drain warps add each reader's line at its token's.
3. THE FOLD, a packet at a time: the wait on its landing; a thread's two entries expanded; a group's steps; the next
   packet's copy issued.

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

1. The prefill launch function (its carve, units and window are built: [`internals/common/plan/prefill_launch.md`](../internals/common/plan/prefill_launch.md)): the carve, the folds' split, the units, the window, the grid, the packets'
   capacities and the ring's size, from the descriptor's facts alone, and the workspace's layout (the prepared
   words and counts' place in it is built: [`internals/common/plan/mask_prepare_layout.md`](../internals/common/plan/mask_prepare_layout.md)).
2. The mask-preparation pass ([`hopper-gather.md`](hopper-gather.md#the-mask-preparation-pass)).
3. The gather from the prepared write-side words and counts ("gather first isnt it simple and an easy confimration of
   risk"): fold packets through the ring ([`hopper-gather.md`](hopper-gather.md)), and what of
   [the hand-off](#the-hand-off) is not designed.
4. The carry on real packets and prepared read-side words, the state zero at entry: `num` exact at the skeleton's
   cell; then the scales; then the den.
5. The intra beside it ([`lab/intra_window.md`](../lab/intra_window.md)), and the op's two halves against
   `tests/oracle/test_prefill_vs_oracle.py`; then the producer.
6. The state at entry and exit, paged and dense.
7. The contract's width: value widths, level widths, depth 3 and 4, every pattern of listed and whole levels (R10).
8. The climb.

## Answered in the kernel, not before it

- OVERLAP INSIDE A WARPGROUP: the next pass's or packet's work begun under the steps in flight (the lab's declared
  wait-after-issue debt).
- UNEVEN HALVES: what a fold waits where its box's lists are quiet and its neighbour's busy; the split along a level
  walked whole is its alternative (R20).
- THE LISTING'S COSTS the lab left unbuilt: the fold's prepared words and counts loaded a window ahead, and the drain's adds at a
  load, a shift and an address an add; a count of passes from the lists; a denser mask. If the form cannot be brought
  under the packet-driven one, the fallback is reader lists the gather would then write once a head and keep (Blake,
  2026-11-23: "I recommend the first, tuned in the real kernel, with the second as the fallback." "ok go") -- taken
  on his word alone ("if we run into that you need to stop nad let me know before doing it").
- THE READ SIDE'S COEFFICIENT FORM (RULED 2026-11-14, chosen by measurement): a reader's byte on its digit multiplied
  into its coefficients, as built, or applied to its sums after the step.

## Blackwell

NOT STATED. No blackwell carry is designed and no blackwell card measured (R25's step shape is hopper's).
