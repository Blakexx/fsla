# The hopper gather: its design before it is built

STATUS: PROPOSED, NOT LOCKED. No gather kernel is built on this line; this page is the design the first one is built
from. A form a lab measured on an H100 quotes the lab's page; a form RULED and not yet measured says so; where
nothing is decided the page says OPEN.

The gather is the hopper op's own kernel work (RULED 2026-11-15: a first-class kernel with an explicit design): it
turns a side's masks and planes into the stream the carry lands and steps
([`hopper-carry.md`](hopper-carry.md#the-gather-stream)). Whatever the carry would otherwise decide or compute
between a slot's landing and its steps is done here, off the carry's synchronized path.

## What it is handed

The bundle's planes as the producer wrote them ([`ARCHITECTURE.md` section 1](../ARCHITECTURE.md#1-what-the-kernel-computes)):
a side's levels, u8 rows tile-major, each row's fp32 scale; the values, bf16 token-major, each token's fp32 scale; a
sparse level's MASK, a bit a digit a token; and THE PLAN (CONSTITUTION R24), which says which levels a side walks
by their lists and which are a box's operand -- the gather does what the plan says and reads no data to decide it. THE MASK'S ORDER IS OPEN: the lab reads it DIGIT-MAJOR -- a digit's
bits of a window, 32 words -- so that a warp turns a digit around from one contiguous load; the public surface's
order is the producer's readers' measured access, and this reader is the first measured.

## What it writes

A ring of a few windows a CTA group (RULED 2026-11-15), a window its slots:
- THE SHARED PART, a sparse block's, read by the 16 carry CTAs that share the block: a slot's head -- its steps'
  operands' fields -- and its lists' value runs, a brick a run, as long as the list. AN ENTRY'S VALUES CARRY ITS
  SCALAR (RULED 2026-11-15): whatever is the same for every leaf of the box -- its digits on its sparse levels,
  the scales -- is multiplied into its 64 values, once.
- THE COEFFICIENT PART, a (sparse block, dense block)'s, read by one carry CTA: a slot's lists' entries'
  coefficients in the step's format, a line a brick, each list's runs padded with zero lines to its steps. Where a
  box spans ONE dense level a coefficient line is that level's 16 digits widened to bf16, exact, no product; where
  it spans several, the product of their slices, 16 an entry a box.
- A window's PLAN: each slot's place and bytes.
- THE READ SIDE'S: the same without values, with each entry's token index. NOT DESIGNED past that.

## The CTA

OPEN. What the labs ran is one CTA a window of 512 threads, every thread producing and its first thread issuing
every copy. The carry's learnings, to be applied where a measurement shows this kernel bound by its own work:
- A LOAD IS ISSUED A STAGE AHEAD OF ITS USE, two stages in flight (KERNEL_STANDARDS 32): a sparse block's mask and
  rows a block ahead, a dense block's rows a dense block ahead.
- WHAT DOES NOT CHANGE IS MADE ONCE: a window's lists and every slot's layout depend on the window and the sparse
  block alone; an entry is kept as its token and its sparse byte.
- NO DECISION ON DATA IN A STEP'S PATH: a list is turned around in a fixed count of trips whatever is live; a token
  not live goes to a spare place, not past a branch.
- NO WARP OF A GROUP THAT MEETS HOLDS THE GROUP'S LONE JOBS: the carry's fold warpgroup waited at its rendezvous
  for the one warp that issued its copies ([`lab/carry_window.md`](../lab/carry_window.md)); whether this kernel
  wants its issuing on warps of their own is OPEN, and KERNEL_STANDARDS 5 asks a measurement for it.
- THE COPY OUT in the labs is three buffers in turn and one copy in flight; its depth is OPEN.

## Its order of work

A window's lists come first, because everything after is addressed by them; then the shared part, a sparse block
at a time; then the coefficient part, a dense block at a time, since a dense block's rows of a window are what a
CTA can hold beside its lists. The lab kernels' own timelines are their sources' headers
([`lab/gather_pass.md`](../lab/gather_pass.md)).

## Beside the carry

RULED 2026-11-15: persistent gather CTAs beside the carry's, in one grid, a gather CTA a multiprocessor of its own.
NOT DESIGNED: the hand-off between a gather CTA and its readers (no barrier crosses CTAs without a cluster), which
gather CTA serves which carry CTAs, the split's rule, what the split leaves the intra; CONSTITUTION R22's
independent units against the ring's waits.

## What is measured

All at 4 to 5 live digits a token, one side's sparse level of 256 digits, lists of 16 to 19 entries; the figures
are [`lab/gather_pass.md`](../lab/gather_pass.md)'s and follow the card a session ran on.
- The shared part with tables for a carry that expands bytes, its lists made from the mask: exact, and at the
  price of the same lines moved from the host's lists.
- The coefficient part with sparse by dense formed here, a window's lists made once: exact, and three times the
  same entries' bytes moved; a slot a dense block a trip, 512 trips a window. With the shared part it is more than
  the carry's window a token.
- NOT MEASURED: the scalar on the values and the coefficient widened alone (the ruled form); larger units; the
  head's operands' fields; the scales; the read side; the ring and the hand-off; a list past 32 entries; more live
  digits.

## To be ruled before it is built

1. The read side: its coefficients sparse by dense, made here, or its dense digits alone with the reader's scalar
   applied to its sums after the step.
2. The mask's order on the public surface.
3. The split of a card's multiprocessors between gather and carry CTAs, and the hand-off.

## Blackwell

NOT STATED. No blackwell gather is designed and no blackwell card measured.
