# The hopper carry: its design before it is built

STATUS: PROPOSED, NOT LOCKED. No carry kernel is built on this line ([`open-work.md`](../open-work.md)); this page is
the design the first one is built from. A form a lab measured on an H100 quotes the lab's number, and the lab's
page holds the record; a form RULED and not yet measured says so; where nothing is decided the page says OPEN.

What the carry computes is [`ARCHITECTURE.md` section 6.3](../ARCHITECTURE.md#63-the-carry--the-inter-window-term): a
window at a time, the readout of the state as of the window's start by the window's readers, into `num` and `den`,
then the fold of the window's writes into the state.

## The gather stream

RULED; its pass NOT BUILT. The gather pass writes, for each sparse block of 16 digits -- the group of CTAs that
share it -- A STREAM, and the stream SAYS WHAT TO DO: the carry computes no boundary and makes no operand. A
window's stream is its slots, a slot 8 sparse digits' lists. A list is A BLOCK AS LONG AS THE LIST: 8 runs of its
entries' value lines, a brick a run, a line an entry, and its entries' COEFFICIENTS IN THE STEP'S FORMAT -- an
entry's dense digits by its sparse digit, the rows' scales in them, bf16, a line a brick, each brick's run padded
with zero lines to the list's steps. A step's operand is pointed at any 16 entries of a block, its bricks as far
apart as the list is long; a slot's head holds each step's two operands' fields, or that the step is absent, when
its second operand is a tile of zeros in the slot. Blocks stay in the order of their digits. A slot is ONE COPY,
and a window's PLAN is each slot's place in the stream and its bytes. THE CARRY LANDS A SLOT AND STEPS IT: it
expands nothing and writes no operand, so it has no fence and no rendezvous before its steps. A token live on two
digits of one block is two entries. The pass consumes the masks; a counts pass sizes the streams. The stream is
WORKSPACE, and A RING (RULED 2026-11-15): a few windows a CTA group, rewritten as the carry consumes them, so
global memory holds windows of it and never a segment's -- the coefficients it holds are live entries' staged for
the carry, never the stored representation. IT IS FED BY PERSISTENT GATHER CTAS BESIDE THE CARRY'S, in one grid (RULED 2026-11-15): a
carry CTA holds a multiprocessor's whole register file and nearly all its shared memory, so a gather CTA is a
multiprocessor's own, and the card's multiprocessors are split between the two for the call. NOT DESIGNED: the
hand-off between a gather CTA and its readers (no barrier crosses CTAs without a cluster), which gather CTA
serves which carry CTAs, and the split's rule. NOT MEASURED on hopper. A stream written whole before the carry runs costs
the carry the same ([`lab/carry_window.md`](../lab/carry_window.md)), and is what the labs ran
([`factored-representation.md`](factored-representation.md)). The read side's stream is the same without values,
with each entry's token index, where the drain adds; its head is not designed. A window's stream is read by the 16
boxes that share it and costs the carry the same whether the cache held it before or not
([`lab/carry_window.md`](../lab/carry_window.md)) -- WHILE those boxes read it close together: boxes that drift
apart, as their read sides' counts differ, are not measured.
WHAT IS MEASURED IS A STREAM OF BYTES: a slot whose coefficients are an entry's 16 dense bytes and its sparse
byte, expanded by the fold where they land, its head tables that say where each entry's lines and each step's
operands are -- the carry filling from it at lists of 8 to 24 entries and no list past two steps
([`lab/carry_window.md`](../lab/carry_window.md)), and the pass writing its shared part from the mask alone at
lists of 16 to 19 ([`lab/gather_pass.md`](../lab/gather_pass.md)). NOT MEASURED: the carry fed coefficients in
the step's format; the pass writing them; its dense part in either form; the read side's stream; lists past two
steps.

## What it is handed

- A side's levels, u8 rows TILE-MAJOR, and each row's fp32 scale; the values, bf16 TOKEN-MAJOR, and each token's fp32
  scale. The order of each plane is its readers' measured access
  ([`ARCHITECTURE.md` section 1](../ARCHITECTURE.md#1-what-the-kernel-computes)).
- A sparse level's MASK, a bit a digit a token, used blindly; the gather pass reads it, makes its lists in its own
  kernel and writes the stream above. There is no separate facts pass on hopper.
- A PLAN (CONSTITUTION R24): the box, the window, each side's walk of each level.

## The CTA

One CTA a multiprocessor, two boxes a CTA, 512 threads at the multiprocessor's whole register file
([`lab/carry_window.md`](../lab/carry_window.md)):

| the role | threads | registers a thread | what it holds |
|---|---|---|---|
| a FOLD warpgroup a box, two | 128 each | 160 | the box's state, 16 accumulator sets of 8; it issues its own slots' copies and steps them as they land |
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
3. THE FOLD. A fold warpgroup issues a slot's copy from the stream a window ahead, as it leaves the slot; when it
   has landed the warpgroup runs the slot's steps, one a sparse digit's 16 entries, their operands the landed
   lines (the gather stream, above).

## What the lab did not have, and the kernel must

The lab's entries and readers were synthetic until its last rows: every sparse digit held exactly 16 entries a
window, and each entry's token and byte were computed, not read.

- LISTS OF ANY LENGTH. A sparse digit's entries in a window are its list in the stream. A digit is
  `ceil(count / 16)` steps into the same accumulator set, the last padded with entries whose coefficient lines are
  zero -- exact, since a zero coefficient adds nothing. A slot is 8 steps, whichever digits they are; a window is as
  many slots as its steps need. The readout likewise: a sparse digit's readers are `ceil(count / 16)` steps, the
  last padded with readers whose coefficient lines are zero. On the fold's side the stream's pass lays the steps and each slot's tables (the
  section above); the read side's tables are not designed.
  MEASURED in the lab at lists of 8 to 24 entries: a slot more a window ([`lab/carry_window.md`](../lab/carry_window.md)).
- ROWS THAT ARRIVE COLD. With real lists a window's tokens are its own, new to the cache each window, and a
  carry that gathers its entries from the tables itself pays for it; asking for a window's rows a window ahead by
  prefetch loses. Whole entries landed from a stream do not: a window's stream is read by the 16 boxes that share
  it, and it costs the carry the same whether the cache held it before or not; only a stream no box shares costs
  more. The figures are [`lab/carry_window.md`](../lab/carry_window.md)'s.
- THE SCALES. Every lab ran on small integers. The gather pass multiplies a row's bytes by the row's fp32 scale
  into the coefficient it writes; the kernel multiplies a value by its token's, carries a token's factor product, and issues one f16 line a token scaled by a power of two
  from a coarse bound (CONSTITUTION R17). None of this is measured on hopper; it is designed with the kernel and
  checked against the oracle.
- THE READOUT'S LISTS. Only the fold's side has run from lists.

## Its price

Every figure is [`lab/carry_window.md`](../lab/carry_window.md)'s, at the hardest cell measured -- each side sparse
on a different level, 256 by 256 leaves, 4 live digits a token -- with the readout on synthetic readers; that page
holds the numbers, which follow the card a session ran on. Their order: whole entries landed from a stream its 16
boxes share, read before or not, cost the least; the carry gathering its entries from lists itself some two
thirds more; a stream no box shares more again. THOSE ARE AT 16 ENTRIES A LIST. With the lab's stream -- its coefficients bytes the fold expands -- and its
lists uneven, 8 to 24, the window is MEASURED at 1.29 times the even lists' under the same readout and drains. At even lists
the form above -- no producer, the readout mirrored, four drain warps -- is the least of the composed forms
measured; the lab page holds what is and is not isolated on the uneven stream. To each the stream's pass adds ([`lab/gather_pass.md`](../lab/gather_pass.md)).
The den was 5% of the window with the readout's coefficients in registers, and is not measured in the form above; the intra, a kernel of its own launched beside the carry, a fifth of it
([`lab/intra_window.md`](../lab/intra_window.md), [`lab/call_pair.md`](../lab/call_pair.md)). A dense side is
unmeasured on hopper.

## Blackwell

NOT STATED. No blackwell carry is designed and no blackwell card measured.

## To be ruled before it is built

1. The padding form: zero coefficient lines for a short list's last step, and for a short reader list's.
2. The read side's step tables: who makes them, from what.
3. The carry's price taken at the cold figure, the kernel built to that and the open staging questions measured in
   it.
4. The build's order: the gather pass from masks; the carry skeleton with real lists, exact against the oracle at one
   cell; the scales; the readout's lists; then the open questions above.
