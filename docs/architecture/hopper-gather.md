# The hopper gather: its design before it is built

STATUS: PROPOSED, NOT LOCKED. No gather kernel is built on this line; this page is the design the first one is built
from. A form a lab measured on an H100 quotes the lab's page; a form RULED and not yet measured says so; where
nothing is decided the page says OPEN.

The gather is the hopper op's own kernel work (RULED 2026-11-15: a first-class kernel with an explicit design): a
VALUE RE-ORDERING KERNEL that turns a side's planes, by the words and counts the
[mask-preparation pass](#the-mask-preparation-pass) prepared, into the stream a group of carry CTAs shares
([`hopper-carry.md`](hopper-carry.md#the-packets)). What one carry CTA alone uses, that CTA makes.

## The mask-preparation pass

RULED 2026-11-24. ONE MASK-PREPARATION PASS, run by the op before its gather and its carry, reads the call's masks and
writes into the op's workspace, for BOTH SIDES (read and write) of every level that has a mask. The owner: "why not
also prepare write side. if it does that, do we even need gather to do that for us. then gather can just read those
and just be a value re-ordering machine just from the blind masks"; of the whole form: "it seems fine".
- A GROUP'S WORD: a word of 32 tokens, the group's digits' mask words ORed -- a token's bit is set where it is live on
  any digit of the group; at a group of one digit the word is the mask's own.
- A WORD'S COUNT: the group's live tokens of the WINDOW before that word, counted from the window's start.
- THE GATHER AND THE CARRY READ THEM BLIND: neither reads a mask its own way and neither counts. The gather reads the
  prepared write-side words and counts and places each live token's values and coefficient bytes at its place, the
  count before the token's word plus the popcount of the word's lower bits; it reads no mask and makes no list. The
  carry takes its readers from the prepared read-side words and counts ([`hopper-carry.md`](hopper-carry.md#the-packets)).
- THE MASK'S FORMAT IS PLAIN BITS: a bit a token, a digit, a side, a level. It carries no counts.
- THE GROUP SIZES ARE NOT WRITTEN OUT: every kernel that needs a level-side's group size computes it from the call's
  density by the one rule (`prefill_carve`, [`cta_carve.md`](../internals/common/plan/cta_carve.md)), as the pass does.
  The density stays the call's input, which is why the pass is one read of the masks and not two.
- THE WORKSPACE IS THE OP'S OWN (the owner: "the workspace jsut holds the workpsace, we own it and its op internal,
  its whatever want"): it holds the gather's ring AND the prepared words and counts, and whatever else the op wants.
- MEASURED ([`lab/mask_prep.md`](../lab/mask_prep.md), an H100, 132 CTAs, every word and count exact): the mask
  once-read, 4.93, 3.46 and 2.60 SM-cycles a token a sparse level-side at groups of 1, 4 and 16 digits, at either
  density measured; the mask in cache, about 1.0 at every group size, what bounds it there being open.
- BUILT, DEVICE-UNTESTED UNTIL ITS FIRST DEVICE RUN: THE PASS IS ITS OWN KERNEL (`mask_prepare_kernel`,
  [`internals/mask_prepare/mask_prepare.md`](../internals/mask_prepare/mask_prepare.md)), one launch over every level-side that has a mask, with its own contract registry (`hardware/kernels/mask_prepare/`)
  and its own op (`mask_prepare`; the Python face `fsla.ops.mask_prepare.prepare_masks`). The library kernel is NOT MEASURED: the lab's
  numbers above are a lab kernel's, groups of 1, 4 and 16 digits with its last two words peeled; groups of 32 to 256 digits are measured nowhere.
- NOT BUILT: the gather; the carry's listing from prepared GROUP words (the lab's listing,
  [`lab/carry_token.md`](../lab/carry_token.md) paragraphs (i) to (l), read a digit's own mask words with counts
  handed to it).
- ITS PLACE IN THE WORKSPACE IS SIZED WITHOUT THE DENSITIES, which the host never reads
  ([`prefill_launch.md`](../internals/common/plan/prefill_launch.md)): the host sizes it for the group size that needs
  the most. At a word of 4 bytes and a count of 2 (the lab's widths), that is groups of one digit, where the word is
  the mask's own and only the counts are stored: 2 bytes a mask word, half the level-side's mask. A group of 4 digits
  or more stores a word and a count a group, at most 1.5 bytes a mask word. THE LAYOUT IS `mask_prepare_layout.h`
  ([`internals/common/plan/mask_prepare_layout.md`](../internals/common/plan/mask_prepare_layout.md)): a block a (batch-head, window) of `66 * width`
  bytes, the totals, the counts and the words in it, A WINDOW'S TOTAL A GROUP IS STORED beside the counts.
- THE INTRA, in the owner's words, NOT YET CHECKED against the kernel: "the intra doesnt use makss, so it can start
  alongisde that mask pass".

## What it is handed

The bundle's planes as the producer wrote them ([`ARCHITECTURE.md` section 1](../ARCHITECTURE.md#1-what-the-kernel-computes)):
a side's levels, u8 rows tile-major, each row's fp32 scale; the values, bf16 token-major, each token's fp32 scale; a
sparse level's MASK, plain bits, a bit a digit a token, which the [mask-preparation pass](#the-mask-preparation-pass)
turns into the words and counts the gather reads; and THE PLAN (CONSTITUTION R24), which says which levels a side
walks by their lists and which are a box's operand -- the gather does what the plan says and reads no mask and no
data to decide it. THE MASK'S ORDER IS OPEN: the lab reads it DIGIT-MAJOR -- a digit's
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
- A BOX'S ROWS ON ITS OTHER LEVEL are not the gather's: the carry loads them by the token's index (RULED 2026-11-23).
- THE READ SIDE IS NOT THE GATHER'S: a carry CTA lists its own readers from the prepared read-side words and counts
  ([`hopper-carry.md`](hopper-carry.md#the-packets)).
- A WINDOW'S COUNT of a block's packets is the call's own data, decided inside the call
  ([`CONSTITUTION.md` R24](../CONSTITUTION.md#r24)); a window's total a group is stored beside the prepared
  counts ([the pass](#the-mask-preparation-pass)).
- WHAT SAYS WHERE A TOKEN GOES is the prepared write-side words and counts (2026-11-24): the count before the token's
  word plus the popcount of the word's lower bits; the gather reads no mask and no data to decide a walk.
- NOT THE FORM, each measured: a list a digit with its value runs, and coefficient runs in the step's format laid here
  -- the pass that made them cost as much as the carry ([`lab/gather_pass.md`](../lab/gather_pass.md)); a token-list
  stream the fold lays its own slots from ([`lab/carry_window.md`](../lab/carry_window.md)).
- NOT MEASURED: the gather writing the packets from the prepared write-side words and counts; its bytes a token in the packet form.

## The CTA

OPEN. What the labs ran is one CTA a window of 512 threads, every thread producing and its first thread issuing
every copy. The carry's learnings, to be applied where a measurement shows this kernel bound by its own work:
- A LOAD IS ISSUED A STAGE AHEAD OF ITS USE, two stages in flight (KERNEL_STANDARDS 32): a sparse block's prepared
  words and counts and rows a block ahead, a dense block's rows a dense block ahead.
- WHAT DOES NOT CHANGE IS MADE ONCE: which tokens a window's are depends on the window and the sparse block alone.
- NO DECISION ON DATA IN A STEP'S PATH: a token's place is computed in a fixed count of trips whatever is live; a
  token not live goes to a spare place, not past a branch.
- NO WARP OF A GROUP THAT MEETS HOLDS THE GROUP'S LONE JOBS: the carry's fold warpgroup waited at its rendezvous
  for the one warp that issued its copies ([`lab/carry_window.md`](../lab/carry_window.md)); whether this kernel
  wants its issuing on warps of their own is OPEN, and KERNEL_STANDARDS 5 asks a measurement for it.
- THE COPY OUT in the labs is three buffers in turn and one copy in flight; its depth is OPEN.

## Its order of work

OPEN. The passes the labs ran moved a window's live tokens' lines from lists the host made; a gather placing them
from the prepared words and counts is not run for the packets. The lab kernels' timelines are their sources'
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
  scales; the hand-off; the gather making the packets from the prepared words and counts; a denser cell in the carry.

## To be ruled before it is built

1. The ring where the card's memory binds the pass: the labs' ring rows ran on 8 and 16 CTAs, where a CTA's copies do.

2. The order of the mask's words on the public surface (the format is plain bits).
3. The split of a card's multiprocessors between gather CTAs and the carry's units, and a unit's start.
4. What the ring does on a card whose cache does not hold it: the token list cost the carry the same read before or
   not while its boxes read it together; a ring's waits and a denser cell's bytes are not measured.

## Blackwell

NOT STATED, for the gather and for the mask-preparation pass alike: neither has a blackwell design and no blackwell
card is measured.
