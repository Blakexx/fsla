# The hopper gather: its design before it is built

STATUS: PROPOSED, NOT LOCKED. No gather kernel is built on this line; this page is the design the first one is built
from. A form a lab measured on an H100 quotes the lab's page; a form RULED and not yet measured says so; where
nothing is decided the page says OPEN.

The gather is the hopper op's own kernel work (RULED 2026-11-15: a first-class kernel with an explicit design): it
turns a side's masks and planes into the stream a group of carry CTAs shares
([`hopper-carry.md`](hopper-carry.md#the-packets)). What one carry CTA alone uses, that CTA makes.

## What it is handed

The bundle's planes as the producer wrote them ([`ARCHITECTURE.md` section 1](../ARCHITECTURE.md#1-what-the-kernel-computes)):
a side's levels, u8 rows tile-major, each row's fp32 scale; the values, bf16 token-major, each token's fp32 scale; a
sparse level's MASK, a bit a digit a token; and THE PLAN (CONSTITUTION R24), which says which levels a side walks
by their lists and which are a box's operand -- the gather does what the plan says and reads no data to decide it. THE MASK'S ORDER IS OPEN: the lab reads it DIGIT-MAJOR -- a digit's
bits of a window, 32 words -- so that a warp turns a digit around from one contiguous load; the public surface's
order is the producer's readers' measured access, and this reader is the first measured.

## What it writes

THE CARRY'S PACKETS ([`hopper-carry.md`](hopper-carry.md#the-packets)), under the packet contract (RULED 2026-11-18:
the carry consumes them blindly; their layout is the gather's, whatever is best). ONLY WHAT A GROUP OF CTAS SHARES
(RULED 2026-11-15: "the expansion is unique to each final cta, not something shared. each final cta should own it, to
distribute load"): a coefficient is one carry CTA's, and that CTA forms it.

- FOLD PACKETS, a (block of 16 digits a side walks by its lists, window), read by the 16 CTAs that hold a box of the
  block. THE GROUPED DIGIT STREAM (RULED 2026-11-19): values only; the block's digits in groups, a group's run the
  tokens live on any of its digits, each written once in it; a step reads 16 consecutive lines of a run; a short step
  is padded on the coefficient side, a zero byte, never by zero value lines. With each membership: its place in its
  step's tile, its byte on its digit, its token's index. THE MERGE RULE (RULED 2026-11-19, the starting rule): never
  add a tensor step to save a line; among the layouts of fewest steps, the fewest lines.
- A BOX'S ROWS ON ITS OTHER LEVEL are not the pass's: the carry loads them by the token's index (RULED 2026-11-23).
- THE READ SIDE IS NOT THE PASS'S: a carry CTA lists its own readers from the read mask
  ([`hopper-carry.md`](hopper-carry.md#the-packets)).
- A WINDOW'S COUNT of a block's packets is the call's own data, decided inside the call
  ([`CONSTITUTION.md` R24](../CONSTITUTION.md#r24)).
- WHAT SAYS WHERE A TOKEN GOES is the input masks (2026-11-13); the pass reads no data to decide a walk.
- NOT THE FORM, each measured: a list a digit with its value runs, and coefficient runs in the step's format laid here
  -- the pass that made them cost as much as the carry ([`lab/gather_pass.md`](../lab/gather_pass.md)); a token-list
  stream the fold lays its own slots from ([`lab/carry_window.md`](../lab/carry_window.md)).
- NOT MEASURED: the pass writing the packets from the write masks; its bytes a token in the packet form.

## The CTA

OPEN. What the labs ran is one CTA a window of 512 threads, every thread producing and its first thread issuing
every copy. The carry's learnings, to be applied where a measurement shows this kernel bound by its own work:
- A LOAD IS ISSUED A STAGE AHEAD OF ITS USE, two stages in flight (KERNEL_STANDARDS 32): a sparse block's mask and
  rows a block ahead, a dense block's rows a dense block ahead.
- WHAT DOES NOT CHANGE IS MADE ONCE: which tokens a window's are depends on the window and the sparse block alone.
- NO DECISION ON DATA IN A STEP'S PATH: a list is turned around in a fixed count of trips whatever is live; a token
  not live goes to a spare place, not past a branch.
- NO WARP OF A GROUP THAT MEETS HOLDS THE GROUP'S LONE JOBS: the carry's fold warpgroup waited at its rendezvous
  for the one warp that issued its copies ([`lab/carry_window.md`](../lab/carry_window.md)); whether this kernel
  wants its issuing on warps of their own is OPEN, and KERNEL_STANDARDS 5 asks a measurement for it.
- THE COPY OUT in the labs is three buffers in turn and one copy in flight; its depth is OPEN.

## Its order of work

OPEN. The passes the labs ran moved a window's live tokens' lines from lists the host made; lists from the masks, as
the digit lists' kernels make theirs, are not run for the packets. The lab kernels' timelines are their sources'
headers ([`lab/gather_pass.md`](../lab/gather_pass.md)).

## Beside the carry

RULED 2026-11-15: persistent gather CTAs beside the carry's, in one grid, a gather CTA a multiprocessor of its own. A
block's packets are a ring read by a unit's 16 carry CTAs ([`hopper-carry.md`](hopper-carry.md#the-hand-off)); the
split of a card's multiprocessors, a unit's start and the ring's waits are not designed. MEASURED: a token list's
values and sparse lines are 122 SM-cycles a token a side against the carry's some 850 expanding its own -- a
multiprocessor in eight the pass's, as that kernel stood.

## What is measured

All at 4 to 5 live digits a token, one side's sparse level of 256 digits, lists of 16 to 19 entries; the figures
are [`lab/gather_pass.md`](../lab/gather_pass.md)'s and follow the card a session ran on.
- The shared part with tables for a carry that expands bytes, its lists made from the mask: exact, and at the
  price of the same lines moved from the host's lists.
- THE TOKEN LIST'S PASS, its lists the host's: exact; its lines moved, 122 SM-cycles a token for a side's values and
  sparse lines and 155 for its dense segments (a box's rows on its other level, the carry's own loads since 2026-11-23), 291 and 448 at 32 live digits; its copies alone are nearly all of it
  (118 and 147).
- NOT THE FORM: the coefficient runs with sparse by dense formed here, a window's lists made once: exact, and three
  times the same entries' bytes moved; a slot a dense block a trip, 512 trips a window. With the shared part it is
  more than the carry's window a token. Widened alone, the entry's scalar on its values, they cost the same; larger
  units take a sixth off. What binds that kernel is NOT ISOLATED.
- MEASURED IN THE CARRY ([`lab/carry_window.md`](../lab/carry_window.md)): the fold placing a token view's value lines is a
  quarter over the block form's window, whose value runs land in place. NOT MEASURED: the boundaries' fields; the
  scales; the hand-off; the pass making the packets from the masks; a denser cell in the carry.

## To be ruled before it is built

1. The ring where the card's memory binds the pass: the labs' ring rows ran on 8 and 16 CTAs, where a CTA's copies do.

2. The mask's order on the public surface.
3. The split of a card's multiprocessors between gather CTAs and the carry's units, and a unit's start.
4. What the ring does on a card whose cache does not hold it: the token list cost the carry the same read before or
   not while its boxes read it together; a ring's waits and a denser cell's bytes are not measured.

## Blackwell

NOT STATED. No blackwell gather is designed and no blackwell card measured.
