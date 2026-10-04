# `csrc/rola/src/owner/owner.cuh` — the owner carry's constants, carve and operand block

Mirror doc for the header the owner carry's components share. The owner carry is the rebuild's fp16 carry: a CTA OWNS a box of leaves and, window by window, folds the window's writes into the box's state and
reads the box's state out to the window's readers. It is being built as a SKELETON in the part harness first
([below](#skeleton)), priced across two carves and two step forms; the carve chosen from that, it replaces
`csrc/rola/src/carry/` in place.

Vocabulary, defined here and used unqualified in this directory's docs: **level S** = the read-sparse level, **level
R** = the read-dense level (in the split routing the cells declare, level R is write-sparse and level S write-dense);
**box** = a CTA's leaves, `RS` digits of level S by `RD` digits of level R, for one channel group of 32; **window** =
512 tokens; **write list** = the tokens live at one of the box's level-R write digits in a window (the fold's unit);
**read digit** = one of the box's level-S read digits (the readout's unit); **pool slot** = a live token's rank among
the window's live tokens of a side.

<a id="constants"></a>

## The constants

`kWindow` 512 tokens, the intra's and the facts pass's, so a row of the facts pass's words is `kWindowWords` = 16 words
a window. `kWarps` = 8 warps a CTA, `kChannels` = 32 a channel group, `kChannelGroups` = 2 at `d_v` 64.
`kListCapacity` = 512 entries: a warp's list buffer a side, one window's tokens.

<a id="carve"></a>

## The carve

`Carve<RS, RD>`: a CTA owns `RS` level-S digits and `RD` level-R digits, so a warp folds `RD / 8` write lists and reads
out `RS / 8` read digits; the staged word rows are the `RD` write rows then the `RS` read rows. The two carves priced
are `(16, 64)` and `(32, 32)` — 1,024 leaves a CTA either way, 128 CTAs a head at `(256, 256)` with two channel groups.

<a id="step"></a>

`Step` names a warp's loop step for a window: `kStepPerList` compacts each list and steps over its own tokens (every
IMMA is a live pair); `kStepMerged` compacts the warp's union once and steps over it, the operand that does not depend
on the list loaded once a step for all the warp's lists; `kStepChosen` lets the cost comparison over the window's exact
counts pick ([`owner_lists.md`](owner_lists.md#step-cost)). A measurement forces one form to price it.

<a id="params"></a>

## `OwnerParams` — the operand block, a grid constant

The facts pass's words (`words`, contiguous `[BH, 2, rows, L / 32]` int32, with a side's and a row's word counts), which
level is S and which R, each level's first row in the layout and its width, whether level R's write side and level S's
read side are sparse (a dense side's words are not read: its union is every token and its form merged), `H`, `L`, the
window count, a forced step form, the lists' dump for the harness's model check, and the phase ledger; for the fold,
the facts pass's list exponents (`list_exp`, a `(K_m, K_v)` word a window and level-R digit, `list_width` digits a
window) and whether level S's write side is dense (`floor_s`, its digits' floor).

<a id="skeleton"></a>

## The skeleton in the part harness

`measure/harness/owner/owner_skeleton.cu` builds the components into the parts module (`torch.ops.rola_owner`) and
`measure/harness/bench_owner_skeleton.py` checks each against a pure model and prices it. The plan, the order and the
pricing table are on the rebuild's plan card, in its skeleton block. Built so far:

* `lists_part` — the words and the lists ([`owner_lists.md`](owner_lists.md)) over every window of every box, the next window's
  words staged before this window's lists; with a dump it writes each CTA-window's unions, row counts, forms, totals
  and entries, which `--check lists` holds bit for bit against the model at both carves and every step setting.
  Checked: `flagship-alt-k4`, `flagship-dense`, `flagship-cohort-k16`, `flagship-alt-k4-L4096`.
* `fill_part` — the lists, then each window's write and read passes ([`owner_fill.md`](owner_fill.md)); with a dump
  it writes every pool row to its slot, which `--check fill` holds byte for byte against the model.
* `drain_part` — the lists, then each window's read passes' panels drained ([`owner_drain.md`](owner_drain.md)).
* `readout_part` — the read passes, then the readout ([`owner_readout.md`](owner_readout.md)) in a forced or chosen
  form, a checksum so nothing is dead.
* `fold_part` — the lists and units, then each window's write passes filled, restaged and folded
  ([`owner_fold.md`](owner_fold.md)) on either arm; it writes the final state, mass and units, which `--check fold`
  holds per slot against `fold_reference.fold_state`; `--ledger fold|fold-h` books its phases.

Parts are priced per head at <= 256K token-heads (64K x 4 or 16K x 16): above that this box's GPU address translation
slows every scattered read 10-30x, the kernel unchanged.
