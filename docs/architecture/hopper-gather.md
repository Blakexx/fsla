# The hopper gather: its design before it is built

STATUS: PROPOSED, NOT LOCKED. No gather kernel is built on this line; this page is the design the first one is built
from. A form a lab measured on an H100 quotes the lab's page; a form RULED and not yet measured says so; where
nothing is decided the page says OPEN.

The gather is the hopper op's own kernel work (RULED 2026-11-15: a first-class kernel with an explicit design): it
turns a side's masks and planes into the stream a group of carry CTAs shares
([`hopper-carry.md`](hopper-carry.md#the-gather-stream)). What one carry CTA alone uses, that CTA makes.

## What it is handed

The bundle's planes as the producer wrote them ([`ARCHITECTURE.md` section 1](../ARCHITECTURE.md#1-what-the-kernel-computes)):
a side's levels, u8 rows tile-major, each row's fp32 scale; the values, bf16 token-major, each token's fp32 scale; a
sparse level's MASK, a bit a digit a token; and THE PLAN (CONSTITUTION R24), which says which levels a side walks
by their lists and which are a box's operand -- the gather does what the plan says and reads no data to decide it. THE MASK'S ORDER IS OPEN: the lab reads it DIGIT-MAJOR -- a digit's
bits of a window, 32 words -- so that a warp turns a digit around from one contiguous load; the public surface's
order is the producer's readers' measured access, and this reader is the first measured.

## What it writes

ONLY WHAT A GROUP OF CTAS SHARES (RULED 2026-11-15, the day's later ruling: "the expansion is unique to each final
cta, not something shared. each final cta should own it, to distribute load"). A coefficient is a (sparse block, dense
block)'s -- one carry CTA's alone -- and that CTA forms it ([`hopper-carry.md`](hopper-carry.md#the-gather-stream)).
The stream is THE TOKEN LIST (RULED 2026-11-13, in force again): a sparse block's live tokens once each, in token
order, packed -- a token's values as the bundle holds them and its sparse bytes on the block's digits -- read by the
carry CTAs that share the block. It is a ring of a few windows a CTA group (RULED 2026-11-15).
- A TOKEN LIVE ON SEVERAL DIGITS OF A BLOCK IS WRITTEN ONCE, where a list a digit writes it once a digit. ESTIMATED,
  live digits falling independently: 1.14 digits a live token a block at 4.5 live digits of 256, 2.3 at 32, 16 dense
  -- the token list's bytes grow with the tokens, never with their digits.
- A LIVE TOKEN'S DENSE DIGITS are a carry CTA's own as well. OPEN, by measurement: a segment of the stream a dense
  block (2026-11-13: 16 bytes a live token a dense block, written here and read by one CTA), or the bundle's dense
  rows landed by the carry itself (16 bytes a token a box, read by every box of the dense block, nothing written).
- WHAT SAYS WHERE A TOKEN GOES is the input masks (2026-11-13). Whether the pass also writes each digit's list as
  indices into the token list -- the stream saying what to do, as the carry's block form's tables do
  ([`lab/carry_window.md`](../lab/carry_window.md)) -- is OPEN.
- WHAT A CTA NEEDS TO TAKE ITS VIEW is the pass's to write, the group's to read (Blake, 2026-11-15: "the gather has
  metadata about boundaries for all the ctas to consume like window edges and such"). PROPOSED, its fields: a window's
  place in the stream and its live tokens; each of the block's 16 digits' count in the window, which is the window's
  steps; a live token's 16 bits -- the block's digits it is live on, the masks' slice for the token -- so a CTA reads
  neither the masks nor a list; on the read side each entry's token index; the segment's edges.
- ITS SIZE, ESTIMATED: a live token's record is 9 lines a sparse block it is live in -- some 0.6 KB a token a side
  at 4 to 5 live digits, 40 MB a side at 65,536 tokens, 2.3 KB a token at most. Written whole before the carry it costs the carry what a ring the cache holds does
  ([`lab/carry_window.md`](../lab/carry_window.md)); the ring is ruled for global memory, and what it asks of the
  hand-off is under "Beside the carry".
- THE READ SIDE'S: the same without values, with each entry's token index. NOT DESIGNED past that.
- NOT THE FORM, each measured: a list a digit with its value runs, and the coefficient runs in the step's format laid
  here. With its operands made the carry's window is 5,188 cycles against 6,826 expanding its own, and the pass that
  made them cost as much as the carry ([`lab/gather_pass.md`](../lab/gather_pass.md)); their 110 KB a window a CTA
  cost three times the window once past the card's cache.

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

OPEN. The token list's pass the lab ran moved a window's live tokens' lines from lists the host made; its lists from
the masks, as the digit lists' kernels make theirs, are not run for it. The lab kernels' timelines are their sources'
headers ([`lab/gather_pass.md`](../lab/gather_pass.md)).

## Beside the carry

RULED 2026-11-15: persistent gather CTAs beside the carry's, in one grid, a gather CTA a multiprocessor of its own.
NOT DESIGNED: the hand-off between a gather CTA and its readers (no barrier crosses CTAs without a cluster), which
gather CTA serves which carry CTAs, the split's rule, what the split leaves the intra; CONSTITUTION R22's
independent units against the ring's waits. MEASURED: the token list's values and sparse lines are 122
SM-cycles a token a side against the carry's 853 expanding its own (6,826 cycles a window) -- a multiprocessor in
eight the pass's, as that kernel stands.

## What is measured

All at 4 to 5 live digits a token, one side's sparse level of 256 digits, lists of 16 to 19 entries; the figures
are [`lab/gather_pass.md`](../lab/gather_pass.md)'s and follow the card a session ran on.
- The shared part with tables for a carry that expands bytes, its lists made from the mask: exact, and at the
  price of the same lines moved from the host's lists.
- THE TOKEN LIST'S PASS, its lists the host's: exact; its lines moved, 122 SM-cycles a token for a side's values and
  sparse lines and 155 for its dense segments, 291 and 448 at 32 live digits; its copies alone are nearly all of it
  (118 and 147).
- NOT THE FORM: the coefficient runs with sparse by dense formed here, a window's lists made once: exact, and three
  times the same entries' bytes moved; a slot a dense block a trip, 512 trips a window. With the shared part it is
  more than the carry's window a token. Widened alone, the entry's scalar on its values, they cost the same; larger
  units take a sixth off. What binds that kernel is NOT ISOLATED.
- NOT MEASURED: the boundaries' fields; the scales; the read side; the hand-off; the pass making the token list from
  the masks; the carry placing a token list's tokens.

## To be ruled before it is built

1. Where a carry CTA's dense digits come from: the stream's segment or the bundle's rows.
2. Whether the pass writes the digits' lists as indices, or the carry reads the masks.
3. The read side's stream: its token indices.
4. The mask's order on the public surface.
5. The split of a card's multiprocessors between gather and carry CTAs, and the hand-off.
6. What the ring does on a card whose cache does not hold it: the token list cost the carry the same read before or
   not while its boxes read it together; a ring's waits and a denser cell's bytes are not measured.

## Blackwell

NOT STATED. No blackwell gather is designed and no blackwell card measured.
