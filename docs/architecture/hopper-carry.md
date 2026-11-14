# The hopper carry: its design before it is built

STATUS: PROPOSED, NOT LOCKED. No carry kernel is built on this line ([`open-work.md`](../open-work.md)); this page is
the design the first one is built from, and every form in it is one a lab measured on an H100. Where a number is
quoted it is the lab's and the lab's page holds the record; where nothing was measured the page says OPEN.

What the carry computes is [`ARCHITECTURE.md` section 6.3](../ARCHITECTURE.md#63-the-carry--the-inter-window-term): a
window at a time, the readout of the state as of the window's start by the window's readers, into `num` and `den`,
then the fold of the window's writes into the state.

## What it is handed

- A side's levels, u8 rows TILE-MAJOR, and each row's fp32 scale; the values, bf16 TOKEN-MAJOR, and each token's fp32
  scale. The order of each plane is its readers' measured access
  ([`ARCHITECTURE.md` section 1](../ARCHITECTURE.md#1-what-the-kernel-computes)).
- A sparse level's MASK, a bit a digit a token, used blindly; and from it the facts pass's lists -- a window's tokens
  by sparse digit, in token order, an entry the token's place in its window, with each digit's count
  ([`lab/facts_lists.md`](../lab/facts_lists.md)). A token's byte on a digit is read from its row.
- A PLAN (CONSTITUTION R24): the box, the window, each side's walk of each level.

## The CTA

One CTA a multiprocessor, two boxes a CTA, 512 threads at the multiprocessor's whole register file
([`lab/carry_window.md`](../lab/carry_window.md)):

| the role | threads | registers a thread | what it holds |
|---|---|---|---|
| a FOLD warpgroup a box, two | 128 each | 160 | the box's state, 16 accumulator sets of 8, and the den's set of 4 |
| the READOUT warpgroup | 128 | 88 | a chunk's accumulator, 36 with the den's column; its A in two sets |
| PRODUCER warps, one a box | 32 each | 104 | a batch of 32 entries |
| DRAIN warps, two | 32 each | 104 | two batches of 8 tokens' lines |

Every barrier wait is a suspending poll. Shared memory: two slots of pools a box, the two boxes' images (a tile 155
lines, an odd count), two panels; under pressure the buffers' depths give way before the box
([`ARCHITECTURE.md` section 9](../ARCHITECTURE.md#9-portability)).

## A window

1. THE IMAGE. Each fold warpgroup publishes its state as of the window's start, bf16, a tile group at a time behind
   the readout's read of the window before: across its sparse digits where the two sides' carves differ, along one
   sparse digit's dense digits where they are equal (CONSTITUTION R17). The den's state is a ninth brick of each
   tile.
2. THE READOUT. A chunk is 4 of the read side's sparse digits, 16 readers each. A readout warp expands its readers'
   coefficients in its registers -- a reader's 16 dense bytes on a box by its byte on its sparse digit -- and they are
   the A of the digit's step on each box's image tile, 72 columns wide; the 65th column is the den. The chunk's 64
   rows go to a panel as f16 pairs and one fp32, and the drain warps add them to the output.
3. THE FOLD. A producer warp fills a slot of 8 sparse digits' tiles: an entry's 64 values, and its 16 dense bytes by
   its byte on its sparse digit. A fold warpgroup then runs the slot's steps, one a sparse digit's 16 entries, and
   the den's two.

## What the lab did not have, and the kernel must

The lab's entries and readers were synthetic until its last rows: every sparse digit held exactly 16 entries a
window, and each entry's token and byte were computed, not read.

- LISTS OF ANY LENGTH. A sparse digit's entries in a window are its count from the facts pass. A digit is
  `ceil(count / 16)` steps into the same accumulator set, the last padded with entries whose coefficient lines are
  zero -- exact, since a zero coefficient adds nothing. A slot is 8 steps, whichever digits they are; a window is as
  many slots as its steps need. The readout likewise: a sparse digit's readers are `ceil(count / 16)` warp-rows, the
  last padded with a zero A. The step tables are a prefix sum of the counts, made once a window by the producers.
  OPEN: the cost of a window whose digits are unevenly full; a digit with no entry costs nothing.
- ROWS THAT ARRIVE COLD. With real lists a window's tokens are its own, new to the cache each window. MEASURED on the
  fold's side: 8,295 cycles a window against 6,273 with rows the cache holds, and 5,111 for the synthetic entries;
  composed 9,650 with the readout still synthetic. Asking for a window's rows a window ahead by prefetch LOSES
  (11,563). OPEN: the real loads issued two batches ahead, which needs registers the producers do not have at 104;
  and two producers a box, which the lab measured as the faster fill on synthetic entries.
- THE SCALES. Every lab ran on small integers. The kernel multiplies a row's bytes by the row's fp32 scale and a
  value by its token's, carries a token's factor product, and issues one f16 line a token scaled by a power of two
  from a coarse bound (CONSTITUTION R17). None of this is measured on hopper; it is designed with the kernel and
  checked against the oracle.
- THE READOUT'S LISTS. Only the fold's side has run from lists.

## Its price

At the hardest cell measured -- each side sparse on a different level, 256 by 256 leaves, 4 live digits a token --
with no clock inside a role: 823 SM-cycles a token for the synthetic entries with every table in cache, the upside;
1,206 with the fold's entries from lists and a window's own tokens, the readout still synthetic. The price is the
second or more (CONSTITUTION R22.2). The den is 5% of the window; the intra, a kernel of its own launched beside
the carry, a fifth of it ([`lab/intra_window.md`](../lab/intra_window.md), [`lab/call_pair.md`](../lab/call_pair.md)).
A dense side is unmeasured on hopper.

## Blackwell

NOT STATED. No blackwell carry is designed and no blackwell card measured.

## To be ruled before it is built

1. The padding form: zero coefficient lines for a short list's last step, a zero A for a short reader row.
2. The step tables made by the producer warps once a window from the facts pass's counts.
3. The carry's price taken at the cold figure, the kernel built to that and the open staging questions measured in
   it.
4. The build's order: the facts pass from masks; the carry skeleton with real lists, exact against the oracle at one
   cell; the scales; the readout's lists; then the open questions above.
