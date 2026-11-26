# The mask-preparation pass

STATUS: THE PASS WRITES each side's LEAF GROUPS' mask words and counts and nothing else. It is exact against its host reference on an H100 (`tests/oracle/test_mask_prepare.py`);
its time has no record yet ([`lab/mask_prepare_time.md`](../lab/mask_prepare_time.md)). This page holds
the pass alone: there is no gather kernel ([No gather kernel](#no-gather-kernel)). A form a lab measured on an H100 quotes the lab's page; a form RULED and not yet built
says so.

## The mask-preparation pass

RULED 2026-11-24, ITS PRODUCT RULED AGAIN 2026-11-25. ONE MASK-PREPARATION PASS, run by the op before its carry, reads the call's masks and writes into the op's
workspace, for BOTH SIDES (read and write). The owner (2026-11-24): "why not also prepare write side. if it does that, do we even need gather to do that for us. then
gather can just read those and just be a value re-ordering machine just from the blind masks"; of the whole form: "it seems fine".
- WHAT IT WRITES IS A SIDE'S LEAF GROUPS, ruled for each side on the owner's word (2026-11-25: "Oh so an actual per group mask with counts would work too? [...] Ok,
  I think it also simplies the decode and it's examples, let's just do it, and then also the segmentation planner, and the per group bitlists and counts."): for each
  side the pass writes one mask word a window word a leaf group -- the side's listed levels' group words ANDed, the call's own tokens masked in -- and each leaf group's
  live count a window, and the fold and the readout read one line whatever the side's listed levels, counting and compacting inline. And the pass is nothing but
  that (2026-11-25: "And I guess we can concer mask prep pass into just materializing these group masks", 'concer' being the dictation's 'convert'): its product is
  one thing a side, a leaf group's mask words and live count, and the per-level group words and totals are not a product the carry reads. How the pass computes them
  is its own to build and measure. A LEAF GROUP is one group on each of the side's levels ([`internals/common/plan/leaf_groups.md`](../internals/common/plan/leaf_groups.md)).
  For each, a window:
  - ITS LINE: a bit a token of the window, set where the token is one of the call's own and, on every level the side lists, is live on a digit of the leaf group's group
    of that level -- the listed levels' group words ANDed. On the hopper card a line is 32 words, 128 bytes.
  - ITS COUNT: the line's set bits, a uint16.
  A side that lists no level has one leaf group, the call's own tokens.
- THE CARRY CTAS READ THEM BLIND, BOTH SIDES: a carry CTA reads no mask. A fold's sets are consecutive leaf groups of the write side, so a set's tokens are one line; the
  fold counts and compacts them inline ([`hopper-carry.md`](hopper-carry.md#what-a-cta-makes)). WHY: a word a listed level a set would need a row, a step and
  three group numbers a level held between sets, about 27 of the 71 registers a fold thread holds there
  ([`internals/carry/carry_kernel.md`](../internals/carry/carry_kernel.md#registers)); with a line it holds one address and one step. NOT MEASURED: what that leaves the
  fold beside its accumulators.
- NO PER-WORD COUNT IS WRITTEN: each CTA counts inline, a popcount and a running sum over the window's words (the owner, 2026-11-25: "the counts are useless anyway? We
  need to find a way to do the counting inline anwyay"). A leaf group's count is the window's.
- THE MASK'S FORMAT IS PLAIN BITS: a bit a token, a digit, a side, a level. It carries no counts.
- THE MASK'S ORDER ON THE PUBLIC SURFACE IS OPEN: the lab reads it DIGIT-MAJOR, a digit's bits of a window, 32 words, so
  that a warp turns a digit around from one contiguous load; the order is the producer's readers' measured access.
- THE GROUP SIZES ARE DERIVED BY EVERY KERNEL TODAY: each kernel that needs a side's groups computes them from the call's
  densities and its levels' widths by the one rule (`prefill_carve`, [`cta_carve.md`](../internals/common/plan/cta_carve.md)), as the pass does. The density stays the
  call's input, which is why the pass is one read of the masks and not two. A LEVEL-SIDE WITH NO MASK is taken at the whole density by every kernel, whatever its
  density byte; the host tells each kernel which level-sides have a mask, a bit a level a side.
- RULED, NOT BUILT: THE CARVE'S RESULTS MAY BE WRITTEN ONCE A CALL into the workspace for every kernel to read. The owner, 2026-11-26, asked "whether the carve's results
  may be written once per call into the workspace": "i dont see why not". Measured the same day: deriving the carve in a CTA costs about 6,100 cycles, and the mask
  pass, a CTA a window, pays it a window. The ruling changes the earlier text (2026-11-24) that every kernel derives the group sizes itself and they are never written
  out; it stands with the ruling (2026-11-24) that the workspace is the op's own and holds facts. The intended form: one small kernel, or the pass's first step, derives
  the carve and each side's leaf-group facts on the device, by the planner's one C++ function, and writes them at the workspace's head, and the pass's CTAs and the
  carry read them. NOT BUILT in this tree.
- THE PASS ALWAYS WRITES, A DENSE SIDE AS ONE LEAF GROUP OF THE CALL'S OWN TOKENS (the owner, 2026-11-26, answering "ok thats fine makes sense" to the account of what is
  built against his assumption that a fully dense side does nothing in the pass): the pass writes both sides whatever the masks, so the fold always reads a line, has one
  path and keeps no bookkeeping of the call's end; a level-side with no mask is taken whole in the carve ([`internals/common/plan/mask_prepare_layout.md`](../internals/common/plan/mask_prepare_layout.md#what-a-leaf-group-holds)).
- THE WORKSPACE IS THE OP'S OWN (the owner: "the workspace jsut holds the workpsace, we own it and its op internal,
  its whatever want"; 2026-11-25: "We don't need any workspace changes right? We still own its allocation"): it holds the leaf groups' lines and counts, and whatever
  else the op wants.
- ITS PLACE IN THE WORKSPACE IS SIZED WITHOUT THE DENSITIES, which the host never reads ([`prefill_launch.md`](../internals/common/plan/prefill_launch.md)): at the MOST
  leaf groups the call's shape allows a side. The owner (2026-11-25): "we assume the group size is the minimum of 16 right, and so the number of groups per side is N/16,
  so N/16 times L bits"; and a level-side with no mask is walked whole, which the host knows from the declaration ("if you know a few levels are dense, that can be info
  for the planner? And it will still assume minimum group for the rest?"). So the most is the smaller of the masked levels' digits multiplied and the leaves over 16
  ([`leaf_groups.md`](../internals/common/plan/leaf_groups.md#most)): two levels of 256 digits, 4,096 leaf groups a side with both masked (512 bytes a token a side), 256
  with one (32), one with neither. The kernels use the front of a side's region, at the leaf groups the carve gives. THE LAYOUT IS `mask_prepare_layout.h`
  ([`internals/common/plan/mask_prepare_layout.md`](../internals/common/plan/mask_prepare_layout.md)).
- THE PASS IS ITS OWN KERNEL (`mask_prepare_kernel`, [`internals/mask_prepare/mask_prepare.md`](../internals/mask_prepare/mask_prepare.md)): one launch, a CTA a (side,
  batch-head, window); it reads the window's masks once, forms each listed level's group words in shared memory and ANDs them a leaf group. Its own contract registry
  (`hardware/kernels/mask_prepare/`) and its own op (`mask_prepare`; the Python face `fsla.ops.mask_prepare.prepare_masks`).
- MEASURED, OF THE FORMS BEFORE THIS ONE: the lab's own kernel, which wrote counts ([`lab/mask_prep.md`](../lab/mask_prep.md), an H100, 132 CTAs, every word and count
  exact): the mask once-read, 4.93, 3.46 and 2.60 SM-cycles a token a sparse level-side at groups of 1, 4 and 16 digits; the mask in cache, about 1.0 at every group
  size. The library kernel that wrote a level-side's group words and totals ([`lab/mask_prepare_time.md`](../lab/mask_prepare_time.md)): 3.63 SM-cycles a token at a
  group of one digit, 2.67 to 3.85 at the others. NOT MEASURED: this form, whose work a window grows with the side's leaf groups (a line a leaf group: 512 kilobytes a
  window a side at 4,096).
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
