---
name: rola-shared-layout
description: Load when designing or changing a shared-memory buffer, or data that picks shared addresses (slots, pools, panels): the access table first, conflict-free by construction, checked by the bank model before the GPU.
---
# Shared-memory layouts

The law: `docs/KERNEL_STANDARDS.md#s30`. The model: `docs/internals/tools/bank_model.md`.

0. Stage the data at all only for a reason `docs/KERNEL_STANDARDS.md#s33` names (other warps read it, it is
   rearranged, buffered past the registers, or the MMA reads it from shared memory); else it stays in registers.

1. In the component's `docs/internals/` page (or the lab's header), write the access table §30.1 asks for and choose
   the layout from it; name the instruction shapes it is conflict-free for.
2. Where data picks the addresses, a placement enters the producer's format only once the composed consumer is timed
   with and without it (§30.2); the measured colorings: `docs/lab/color_union.md`, `docs/lab/facts_color.md`.
3. Place pads by §30.3.
4. Before any GPU run, model the producer's emission with `tools/bank_model.py`'s `wavefronts` and `ldmatrix` from the
   lab's `.inputs.py`, written beside its data as `banks.json` (as `read_image.inputs.py` does); its command line
   answers one instruction (`docs/internals/tools/bank_model.md`).
5. Put the `wavefronts` line in the budget (§30.4; `rola-budget`) and price it at every density (§30.6).
