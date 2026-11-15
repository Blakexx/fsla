# The hopper carry: its design before it is built

STATUS: PROPOSED, NOT LOCKED. No carry kernel is built on this line ([`open-work.md`](../open-work.md)); this page is
the design the first one is built from. A form a lab measured on an H100 quotes the lab's number, and the lab's
page holds the record; a form RULED and not yet measured says so; where nothing is decided the page says OPEN.

What the carry computes is [`ARCHITECTURE.md` section 6.3](../ARCHITECTURE.md#63-the-carry--the-inter-window-term): a
window at a time, the readout of the state as of the window's start by the window's readers, into `num` and `den`,
then the fold of the window's writes into the state.

## The gather stream

RULED; NOT BUILT. The gather pass writes, for each sparse block of 16 digits -- the group of CTAs that share it -- A
STREAM OF WHAT THE GROUP SHARES ([`hopper-gather.md`](hopper-gather.md)): the block's live tokens once each, in token
order, packed, a token's values and its sparse bytes on the block's digits (RULED 2026-11-13; in force again
2026-11-15). EACH CARRY CTA MAKES ITS OWN OPERANDS (RULED 2026-11-15: "the expansion is unique to each final cta, not
something shared. each final cta should own it, to distribute load"): it lands its window's view by bulk copies, and
its fold's and readout's warps place each token's value lines on its digits' steps and form its coefficients -- an
entry's sparse digits, the scales and the box's dense digits, their product, bf16 -- in shared memory. THE FILL IS SLICES (RULED 2026-11-15; NOT MEASURED): ONE VIEW A WINDOW for all 16 digits of the block and ONE PASS
over it in token order -- a token's lines stored to its digit's slice at its rank, the rank a count of the digit's
mask before the token -- then all the window's steps, a digit's last padded with zero lines; the steps are counted
once from the window's counts and run with no decision on data. A window's slices are the room two half-window slots of landed operands took. HOW THE VIEW REACHES THE SLICES IS OPEN.
Through registers -- ruled "if they actually fit" -- they fit, two value lines at a time, and are NOT THE FORM:
MEASURED, a fold's fill is then seven times the block form's, a warp's loads a fifth of a bulk copy's bytes a cycle,
as KERNEL_STANDARDS 33 has it for contiguous rows. A view landed by bulk copy and moved in shared memory is MEASURED
on half-window views at a quarter over values that land in place by digit -- the move doubles the fold's shared-memory
traffic -- and one view a window staged so does not fit beside its slices. MEASURED too: one view's structure, 32
steps one block behind one rendezvous, takes the fold's steps' issue to three fifths and
its rendezvous to a quarter of the two-slot form's ([`lab/carry_window.md`](../lab/carry_window.md)). A token live on two digits of one block is one token of the list and an entry of
each digit's slice. Where a CTA's dense digits come from -- a segment of the stream a dense block, or the bundle's
dense rows -- is OPEN. The stream is WORKSPACE, and A RING (RULED 2026-11-15): a few windows a CTA group, rewritten as
the carry consumes them, so global memory holds windows of it and never a segment's. IT IS FED BY PERSISTENT GATHER
CTAS BESIDE THE CARRY'S, in one grid (RULED 2026-11-15): a carry CTA holds a multiprocessor's whole register file and
nearly all its shared memory, so a gather CTA is a multiprocessor's own, and the card's multiprocessors are split
between the two for the call. NOT DESIGNED: the hand-off between a gather CTA and its readers (no barrier crosses CTAs
without a cluster), which gather CTA serves which carry CTAs, the split's rule, and what the split leaves the intra,
which is launched beside the carry on the multiprocessors the carry leaves. A carry CTA then waits on a gather CTA and
a gather CTA on its readers: CONSTITUTION R22's independent units hold between carry CTAs, and what the ring's waits
cost that rule is to be shown before the form is built. A stream written whole before the carry runs costs the carry
the same ([`lab/carry_window.md`](../lab/carry_window.md)), and is what the labs ran. The read side's stream is the
same without values, with each entry's token index, where the drain adds. A window's stream is read by the boxes that
share it and costs the carry the same whether the cache held it before or not -- WHILE those boxes read it close
together: boxes that drift apart, as their read sides' counts differ, are not measured.
MEASURED, in the carry, from streams the host laid BY DIGIT at lists of 8 to 24 entries and no list past two steps
([`lab/carry_window.md`](../lab/carry_window.md)) -- none of them the token list: the fold landing a slot with its
operands made and stepping it, 5,188 cycles a window with the readout's tiles made as well, which is the window with
the carry's own operands free; and the fold expanding an entry's 16 dense bytes and its sparse byte where they land,
its head tables for each entry's lines and each step's operands, 6,826. MEASURED, the pass
([`lab/gather_pass.md`](../lab/gather_pass.md)): the token list's lines moved from the host's lists, 122 SM-cycles a
token for a side's values and sparse lines and 155 for its dense segments.
NOT MEASURED: the carry placing a token list's tokens; slices; the read side's stream; the ring; lists past two steps.

## What it is handed

- A side's levels, u8 rows TILE-MAJOR, and each row's fp32 scale; the values, bf16 TOKEN-MAJOR, and each token's fp32
  scale. The order of each plane is its readers' measured access
  ([`ARCHITECTURE.md` section 1](../ARCHITECTURE.md#1-what-the-kernel-computes)).
- A sparse level's MASK, a bit a digit a token, used blindly; the gather pass reads it in its own
  CTAs and writes the stream above. There is no separate facts pass on hopper.
- A PLAN (CONSTITUTION R24): the box, the window, each side's walk of each level.

## The CTA

One CTA a multiprocessor, two boxes a CTA, 512 threads at the multiprocessor's whole register file
([`lab/carry_window.md`](../lab/carry_window.md)):

| the role | threads | registers a thread | what it holds |
|---|---|---|---|
| a FOLD warpgroup a box, two | 128 each | 160 | the box's state, 16 accumulator sets of 8; it issues its own copies of the stream, makes its steps' operands where they land and steps them |
| the READOUT warpgroup | 128 | 104 | a pass's two tile groups' sums, 64 a thread; its coefficients are in shared memory |
| DRAIN warps, four | 32 each | 88 | batches of the panel's lines on their way to the output |

NO PRODUCER WARP: a fold warpgroup waited on its one producer for most of a window, and issuing its own copies it
does not. The den in this form is NOT MEASURED: it was measured with the readout's A in registers (a
ninth brick of each image tile, a 65th column of the readout's step), and is owed here.

Every barrier wait is a suspending poll. Shared memory: two slots of pools a box, the two boxes' images (a tile 155
lines, an odd count), the readout's coefficient tiles, two panels; under pressure the buffers' depths give way before the box
([`ARCHITECTURE.md` section 9](../ARCHITECTURE.md#9-portability)).

## A window

1. THE IMAGE. Each fold warpgroup publishes its state as of the window's start, bf16, a tile group at a time behind
   the readout's read of the window before: across its sparse digits where the two sides' carves differ, along one
   sparse digit's dense digits where they are equal (CONSTITUTION R17). The den's state is a ninth brick of each
   tile.
2. THE READOUT, MIRRORED: its coefficients in shared memory and its tokens on the step's columns, as the fold's
   are. A pass is 8 of the read side's sparse digits, 16 readers each, two tile groups: a thread expands a
   reader's 16 dense bytes on a box by its byte on its sparse digit to a coefficient tile, and a digit's step is
   an image tile by that tile. The pass's sums go to two panels as f16 pairs, and the drain warps add them to the
   output.
3. THE FOLD. A fold warpgroup issues its view's copies from the stream; when they have landed its warps place the
   tokens' value lines and form their coefficients on their digits' steps, and the warpgroup runs the steps, one a
   sparse digit's 16 entries (the gather stream, above). MEASURED from a stream laid by digit: a slot's copy issued a
   window ahead, the fold expanding its entries' coefficients where they land.

## What the lab did not have, and the kernel must

The lab's entries and readers were synthetic until its last rows: every sparse digit held exactly 16 entries a
window, and each entry's token and byte were computed, not read.

- LISTS OF ANY LENGTH. A sparse digit's entries in a window are its list, which the CTA lays from the token list. A digit is
  `ceil(count / 16)` steps into the same accumulator set, the last padded with entries whose coefficient lines are
  zero -- exact, since a zero coefficient adds nothing. A window's steps are one block (RULED 2026-11-15); the labs before it ran slots of 8 lists. The readout likewise: a sparse digit's readers are `ceil(count / 16)` steps, the
  last padded with readers whose coefficient lines are zero. The steps are the CTA's to lay, from the counts the stream holds (the
  section above).
  MEASURED in the lab at lists of 8 to 24 entries: a slot more a window ([`lab/carry_window.md`](../lab/carry_window.md)).
- ROWS THAT ARRIVE COLD. With real lists a window's tokens are its own, new to the cache each window, and a
  carry that gathers its entries from the tables itself pays for it; asking for a window's rows a window ahead by
  prefetch loses. Whole entries landed from a stream do not: a window's stream is read by the 16 boxes that share
  it, and it costs the carry the same whether the cache held it before or not; only a stream no box shares costs
  more. The figures are [`lab/carry_window.md`](../lab/carry_window.md)'s.
- THE SCALES. Every lab ran on small integers. The carry CTA multiplies a row's bytes by the row's fp32 scale
  into the coefficient it forms; the kernel multiplies a value by its token's, carries a token's factor product, and issues one f16 line a token scaled by a power of two
  from a coarse bound (CONSTITUTION R17). None of this is measured on hopper; it is designed with the kernel and
  checked against the oracle.
- THE READOUT'S LISTS. Only the fold's side has run from lists.

## Its price

Every figure is [`lab/carry_window.md`](../lab/carry_window.md)'s, at the hardest cell measured -- each side sparse
on a different level, 256 by 256 leaves, 4 live digits a token -- with the readout on synthetic readers; that page
holds the numbers, which follow the card a session ran on. Their order: whole entries landed from a stream its 16
boxes share, read before or not, cost the least; the carry gathering its entries from lists itself some two
thirds more; a stream no box shares more again. THOSE ARE AT 16 ENTRIES A LIST. With lists uneven, 8 to 24, the window is MEASURED some 7% over the even lists'
where a slot lands with its operands made, and some 30% over where the fold expands bytes, under the same readout
and drains: the lab page holds the figures. At even lists
the form above -- no producer, the readout mirrored, four drain warps -- is the least of the composed forms
measured; the lab page holds what is and is not isolated on the uneven stream. To each the stream's pass adds ([`lab/gather_pass.md`](../lab/gather_pass.md)).
The den was 5% of the window with the readout's coefficients in registers, and is not measured in the form above; the intra, a kernel of its own launched beside the carry, a fifth of it
([`lab/intra_window.md`](../lab/intra_window.md), [`lab/call_pair.md`](../lab/call_pair.md)). A dense side is
unmeasured on hopper.

## Blackwell

NOT STATED. No blackwell carry is designed and no blackwell card measured.

## To be ruled before it is built

1. How the token view reaches the slices, or whether the stream is laid by digit again: the measured price of the
   move is above.
2. The read side's step tables: who makes them, from what.
3. The carry's price taken at the cold figure, the kernel built to that and the open staging questions measured in
   it.
4. The build's order: the gather pass from masks; the carry skeleton with real lists, exact against the oracle at one
   cell; the scales; the readout's lists; then the open questions above.
