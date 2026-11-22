# `fsla/engine/facts/liveness.py` — the class-1 words' contract

Mirror doc for the Python module that states the class-1 words: their layout, the two folds over them and the pure-torch
MODEL of both, which `tests/unit/test_liveness_contract.py` gates. THE PRESENT FORM: no kernel writes the words on this
line -- the facts pass was deleted whole (Blake, 2026-11-21: "lets just delete facts"; 2026-11-13: "With the gather pass
there is no separate facts pass on hopper"), its entry raises `fsla.ops._ext.KernelNotBuilt`, and the Hopper design
reads the digits in its gather pass. The layout stands as the model's definition, the engine's shape law and the
reference a Hopper schedule is graded against; it carries no C++ twin. What is written down here is the shape of the
words, the folds over them, and why each is the grain it is.

Vocabulary, defined here and used unqualified below: `D` = routing depth;
`B_l` = level `l`'s PADDED width (a power of two at or above 16 — the descriptor's,
K50); `b_l` = its LOGICAL width, which never leaves the host; `L` = tokens in the call;
`BH` = batch × heads; `N = prod_l B_l` = leaves; **atom** = 16 consecutive leaves in
canonical order, which is the page granule; **box** = one warp sub-box, an aligned
rectangle in digit space that the geometry block derives
([`common/geom.md`](../common/geom.md)).

## Two classes, and nothing between them

Blake's ruling of 2026-08-29 (the foundation card, "ONE GEOMETRY-INDEPENDENT
LIVENESS PASS") admits exactly two classes of fact.

* **CLASS 1 — the raw bitlists**, `O(Σ_l B_l × L)` per side, stored once. One bit per
  (side, token, level, digit), in canonical digit order. No span, no carve, no `B`
  in the pass's template: this is the whole reason there is ONE pass rather than one
  per consumer, and it is what makes the words reusable across kernels and across
  calls whose routes are unchanged.
* **CLASS 2 — the `O(N)`-class facts**, aggregated over tokens: the PRIMARY per-box
  READ/WRITTEN bits over the WHOLE CALL, and the DERIVED per-page bits the host ORs
  out of them. Nothing sits between: a box's liveness *for one token* is the AND over
  levels, computed on the fly by the owning warp and never stored, because
  `O(#boxes × L)` has no consumer.

The int8 rebuild's ruling of 2026-09-28 widened what the facts kernel owns to anything
determinable from the raw produced values, so one per-grain aggregate sits beside
the classes and is written by the same pass: the [tile unions](#tile-unions), per 64 tokens. It is not a class-2 fact —
it is neither over the whole call nor `O(N)` — and it is not derived by a consumer: it is the OR
of what the pass already holds when it holds it.

## <a id="max"></a>The digits: each against its row's max

Python's statement of the digits is `fsla.routing.quant` (`AMP_LEVELS`, the `ROWS` form);
the decode fold's `kRowMaxDigit` (`csrc/fsla/src/decode/decode_fold.cuh`) is the same 255, which `tests/unit/test_api_contract.py` holds.

Every level's row is stored unit-max: u8 digits with the row's largest at 255 (`fsla.routing.quant.AMP_LEVELS`), the row's max beside
it as an fp32 scale. A digit `b` is the amplitude `b / 255 × scale`. An all-zero row, its scale 0, is a dead token at
that level.

## <a id="alignment"></a>The row alignment law

Every level's rows start on 32 bytes and step batch, token and head by multiples of 32 bytes (a dimension of length one
is free), in `RouteFactors`' check; `quant.ROW_ALIGN` states it. The producer and `RouteFactors.from_levels` pack each side's levels in 32-byte slots
(`fsla.routing.quant.row_slot`: a level's width rounded up), the slot's tail zero — at power-of-two widths of 32 and up
the slots are the widths, so the law costs nothing there. It is what lets a kernel move a token's run as 16-byte
asynchronous quanta whose source is 16-byte aligned and whose 32-byte run is one sector. A level whose width is not a multiple of 16 is copied whole
up to its last digit and no further (a partial quantum zero-fills the rest).

## <a id="digits"></a>The digits, dense or sparse

Every level's rows are u8 against the row max (`AMP_LEVELS` = 255), on either side, and a zero digit is an exact
zero: there is no floor under a row ([`docs/CONSTITUTION.md` R23](../../CONSTITUTION.md#r23)). A DENSE level is one
whose activation has no exact zeros (softmax, not entmax); it differs from a sparse
one only in that its support is not worded -- no pass reads or writes its rows, and its bits are its
width mask over the live tokens. A dead row (scale 0) is zero everywhere, including where a consumer divides the row by
its own sum (the decay's shares).

<a id="layout"></a>

## The class-1 word layout

    [BH][side][row][word]        side: 0 = read, 1 = write
    row  = row_base(level) + digit,  row_base(l) = Σ_{i<l} B_i
    word = token / 32,  bit = token % 32

`rows = Σ_l B_l`, `words = ceil(L / 32)`, and the words are `uint32_t` (torch carries
them as `int32`, the same 32 bits, exactly as the union table already did). Per side
per batch-head that is `Σ_l B_l × L / 8` bytes whenever `L` is a whole number of
words, and `ceil(L/32) × 4 × Σ_l B_l` in general — it does NOT grow with `N`.

**A row is a level's digit, in canonical order** — the column a packed plane of the
levels holds it at, which is how the producer writes the bundle (`[B, T, H, Σ_l b_l]`,
the levels on `level_offsets`). Tokens are the minor axis: a word is 32 consecutive
tokens of one row, so a consumer walking a window reads a row's words in order, and a
warp's 32 tokens are one word.

**A token past the end votes dead.** The tail word's high bits are zero by
construction, so a fold may OR whole words without masking.

**The nonzero test is on the digit** (`digit_live`): a u8 digit is live when it is
nonzero. The model applies `!= 0` to whatever rows it is given, which is the same test on
u8 digits and lets a drawn call state its support from its floating levels (`-0` dead like
`+0`).

<a id="statics"></a>

## Every level has a mask: the words are the supplied bits

The words are the op's MASKS (ruled 2026-11-13, 2026-11-22; [`api.md` §2.6](../../api.md)): the producer emits a bit per side,
level, digit and token beside the rows, token-packed in the layout the pass reads (bit `t % 32` of word `t // 32` at a digit,
`[B, H, ceil(L / 32), w_l]` a level), and `fsla.engine.facts.liveness.gather_words` lays them out in the pass's table -- the
model of the table the gather pass reads and no test of any value. There is no row test, no dense flag and no static width
mask: a level whose activation has no exact zeros has an all-live mask like any other, a digit past a level's width holds
a zero bit, and every fold (`box_activity`, `atom_activity`, `page_bits`) reads the words alone. A mask that does not cover a
row's nonzero digits changes the result; that is the supplier's job, and the oracle states it (`r * m`).

<a id="tile-unions"></a>

## The tile unions

    [BH][side][group][tile]      group: a level's digits in 32s, group_base(l) = Σ_{i<l} ceil(B_i / 32)
    tile = token / 64,  bit i of (group g of level l) = digit 32 g + i

A word per (side, 32-digit group of a level, 64-token tile): bit `i` set when that digit is
live at some token of the tile — the OR of the tile's two class-1 words, a row a bit. A
level narrower than 32 digits is one group with its high bits clear; `groups = Σ_l
ceil(B_l / 32)` a side, `tiles = ceil(L / 64)`, and a tile past `L`'s last word ORs only
the words that exist. `LivenessLayout.union_index` is the table's address from its extents; `tile_unions` in the Python module is the model, gated against the per-token bits.

An intra's slab gate is the consumer: the table in place of a run of each of a window's rows'
words, which lie `L / 8` bytes apart. No intra is built on this line, so what the table saves
its reader is owed to the first one's measurement.

<a id="folds"></a>

## The folds, and why the AND is exact

**Per box, per side, over the whole call** (the PRIMARY bits):

    live(box, token) = AND over levels l of ( OR over d in box.run(l) of bit(side, l, d, token) )
    READ/WRITTEN(box) = OR over tokens of live(box, token)

The AND is not a relaxation. A token's live leaf set is the PRODUCT of its per-level
live digit sets — the routing is a Kronecker product — so "some live leaf inside the
box" and "every level has a live digit inside its run" are the same statement. The
consuming side therefore ORs its own span's digits per level and then ANDs the levels;
that word count is the fold's floor term (the note).

The bits are per box over the WHOLE CALL and never per window: the box lives in
registers from the call's entry to its exit, so the gate is `LOAD on resident ∧ (read ∨
written)`, `STORE on written`, once each. A window with nothing live for a box is a
schedule fact read off these same rows while walking, never a stored bit.

**Per page** (the DERIVED bits), on the host:

    page.READ    = OR over read-side  boxes intersecting the page of READ(box)
    page.WRITTEN = OR over write-side boxes intersecting the page of WRITTEN(box)

through the box→page map the descriptor gives (a box's leaves are the product of its
digit runs; the page is the trailing four canonical bits). The byte is the one
`fsla/engine/facts/planes.py` already publishes — `ATOM_READ | ATOM_WRITTEN`, never
pre-ORed — so `plan_exact` allocates on the written half, residency checks read the
read half, and a decode step consumes this set because its unit is the page. Being a
box-grain OR it may over-report a page whose live leaves belong to a box that
straddles it; it never under-reports, which is the property allocation depends on and
which the gate asserts against the leaf-level truth.

## What padding does, and does not, reach

The kernel never learns that padding exists. Pad digits lie past a level's rows (the
bundle holds `b_l` digits a level) and the mask never names them, so their class-1 bits
are zero on a sparse level and zero by the width mask on a dense one; a box of nothing but pad digits therefore has both call-level
bits clear and dies at the prologue, and a live box that straddles the pad boundary
folds zeros on its pad slots. There is no descriptor-keyed dense shortcut in the
kernel and no host-side grid compaction.
