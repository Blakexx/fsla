# The mask-preparation pass

STATUS: THE PASS IS BUILT as its own kernel, writing the group words and the totals alone, exact against its host reference on an H100 and measured at each of its group sizes. This page holds the pass alone: there is no gather kernel
([No gather kernel](#no-gather-kernel)). A form a lab measured on an H100 quotes the lab's page; a form RULED and not
yet built says so.

## The mask-preparation pass

RULED 2026-11-24. ONE MASK-PREPARATION PASS, run by the op before its carry, reads the call's masks and
writes into the op's workspace, for BOTH SIDES (read and write) of every level that has a mask. The owner (2026-11-24): "why not
also prepare write side. if it does that, do we even need gather to do that for us. then gather can just read those
and just be a value re-ordering machine just from the blind masks"; of the whole form: "it seems fine".
- A GROUP'S WORD: a word of 32 tokens, the group's digits' mask words ORed -- a token's bit is set where it is live on
  any digit of the group; at a group of one digit the word is the mask's own.
- A GROUP'S TOTAL: the group's live tokens of the WINDOW, a uint16 a group.
- THE CARRY CTAS READ THEM BLIND, BOTH SIDES: a carry CTA reads no mask its own way; it makes its lists, a side, from
  the prepared group words ANDed across the side's masked levels ([`hopper-carry.md`](hopper-carry.md#what-a-cta-makes)).
- THE PASS WRITES NO PER-WORD COUNTS: it writes the group words and the totals only, because each CTA counts inline, a
  popcount and a running sum over the window's words (the owner, 2026-11-25: "the counts are useless anyway? We
  need to find a way to do the counting inline anwyay").
- THE MASK'S FORMAT IS PLAIN BITS: a bit a token, a digit, a side, a level. It carries no counts.
- THE MASK'S ORDER ON THE PUBLIC SURFACE IS OPEN: the lab reads it DIGIT-MAJOR, a digit's bits of a window, 32 words, so
  that a warp turns a digit around from one contiguous load; the order is the producer's readers' measured access.
- THE GROUP SIZES ARE NOT WRITTEN OUT: every kernel that needs a level-side's group size computes it from the call's
  densities and its levels' widths by the one rule (`prefill_carve`, [`cta_carve.md`](../internals/common/plan/cta_carve.md)), as the pass does.
  The density stays the call's input, which is why the pass is one read of the masks and not two.
- THE WORKSPACE IS THE OP'S OWN (the owner: "the workspace jsut holds the workpsace, we own it and its op internal,
  its whatever want"): it holds the prepared words and totals, and whatever else the op wants.
- MEASURED ([`lab/mask_prep.md`](../lab/mask_prep.md), an H100, 132 CTAs, every word and count exact; the lab's kernel, which wrote counts): the mask
  once-read, 4.93, 3.46 and 2.60 SM-cycles a token a sparse level-side at groups of 1, 4 and 16 digits, at either
  density measured; the mask in cache, about 1.0 at every group size, what bounds it there being open.
- BUILT, AND EXACT AGAINST ITS HOST REFERENCE ON AN H100 (`tests/oracle/test_mask_prepare.py`): THE PASS IS ITS OWN KERNEL (`mask_prepare_kernel`,
  [`internals/mask_prepare/mask_prepare.md`](../internals/mask_prepare/mask_prepare.md)), one launch over every level-side that has a mask, with its own contract registry (`hardware/kernels/mask_prepare/`)
  and its own op (`mask_prepare`; the Python face `fsla.ops.mask_prepare.prepare_masks`). The library kernel was MEASURED at each of the eight group sizes it prepares
  ([`lab/mask_prepare_time.md`](../lab/mask_prepare_time.md)). Both are of the pass that took a level-side's group from its own density; the pass as it stands derives the call's
  carve, whose least set of 16 leaves raises a group, and is NOT RUN on a device.
- NOT BUILT: the carry's listing from prepared GROUP words (the lab's listing,
  [`lab/carry_token.md`](../lab/carry_token.md) paragraphs (i) to (l), read a digit's own mask words with counts
  handed to it).
- ITS PLACE IN THE WORKSPACE IS SIZED WITHOUT THE DENSITIES, which the host never reads
  ([`prefill_launch.md`](../internals/common/plan/prefill_launch.md)): the host sizes it for the group size that needs
  the most. At a word of 4 bytes and a total of 2, that is the larger of a total a digit (groups of one digit, where the word is
  the mask's own) and a word a group at groups of 2 digits (the totals, then a word a mask word a group): at 256 digits
  and 32 words, 16,640 bytes. THE LAYOUT IS `mask_prepare_layout.h`
  ([`internals/common/plan/mask_prepare_layout.md`](../internals/common/plan/mask_prepare_layout.md)): a block a (batch-head, window) of `prepared_block_bytes(width_bits, words)`
  bytes (`words` the card's window in mask words), the totals, then the words in it.
- THE INTRA, in the owner's words, NOT YET CHECKED against the kernel: "the intra doesnt use makss, so it can start
  alongisde that mask pass".

## No gather kernel

RULED 2026-11-25: there is no gather kernel. A carry CTA makes its own lists, fetches its own value lines by token
index and loads its own coefficients, on both sides ([`hopper-carry.md`](hopper-carry.md#what-a-cta-makes)); the fetch
of the value lines is measured alone in [`lab/value_fetch.md`](../lab/value_fetch.md). The lab pages that measured the
gather's forms are the record of why: [`lab/gather_pass.md`](../lab/gather_pass.md),
[`lab/gather_group.md`](../lab/gather_group.md) and [`lab/carry_window.md`](../lab/carry_window.md).

## Blackwell

NOT STATED, for the mask-preparation pass: it has no blackwell design and no blackwell card is measured.
