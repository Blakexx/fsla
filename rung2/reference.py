"""fp64 references for the CAPACITY-SPLIT routed chunked linear attention.

Rung 1's references unchanged (they are already written for arbitrary level
widths) plus the two things capacity adds:

``block_plan``  -- the block geometry law, evaluated in Python from the widths
                   and BC alone, and the slot range each block owns. Written
                   from the LAW rather than from the kernel's constants, so a
                   kernel whose rectangle drifted would disagree with it.
``split``       -- the modelled-rounding recurrence run PER BLOCK and summed,
                   which is what the kernel does: each block rounds its own
                   state to bf16 and contributes a complete partial output.

The exact fp64 references need no split at all -- every term of the recurrence
is linear in the slot axis, so the sum over blocks of the exact partials IS the
exact whole. That the two agree is itself checked.
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from rung1.reference import (  # noqa: E402,F401
    amplitudes,
    chunked,
    chunked_bf16_state,
    digits,
    kron,
    kron_bf16,
    quadratic,
    split_levels,
    strides,
)


def block_plan(widths, bc: int):
    """The law, in Python: ``[(lead_off, tail_off, slot0)]`` per block.

    Raises unless the (widths, BC) pair is a rectangle in digit space: every
    level above the boundary at exactly one digit, exactly one level partially
    spanned by an aligned sub-range dividing its width, every level below spanned
    entirely. Two levels are built, so the law reads as ``lead_span`` digits of
    level 0 by ``tail_span`` digits of level 1.
    """
    if len(widths) != 2:
        raise ValueError(f"the built kernel spans two levels, got widths={widths}")
    w0, w1 = widths
    lead_span = 1 if bc <= w1 else bc // w1
    tail_span = bc if bc <= w1 else w1
    if lead_span * tail_span != bc:
        raise ValueError(f"BC={bc} is not a rectangle in digit space for {widths}")
    if bc > w1 and bc % w1:
        raise ValueError(f"BC={bc} spans the trailing level partially AND wraps it")
    if w1 % tail_span or w0 % lead_span:
        raise ValueError(f"BC={bc}'s spans do not tile the levels of {widths}")
    if lead_span != 1 and tail_span != w1:
        raise ValueError(f"BC={bc} would span two levels partially in {widths}")
    tail_groups = w1 // tail_span
    out = []
    for k in range(w0 * w1 // bc):
        lead_off = (k // tail_groups) * lead_span
        tail_off = (k % tail_groups) * tail_span
        out.append((lead_off, tail_off, lead_off * w1 + tail_off))
    return out, lead_span, tail_span


def block_slots(widths, bc: int, device="cuda"):
    """[blocks, BC] int table of the GLOBAL slot each block-local slot names.

    Built from the mixed-radix definition with an explicit divide and modulo over
    the block's digit rectangle -- the leaf-major construction the kernel's loop
    nest deliberately never executes.
    """
    w0, w1 = widths
    plan, lead_span, tail_span = block_plan(widths, bc)
    out = torch.empty(len(plan), bc, dtype=torch.long, device=device)
    for k, (lead_off, tail_off, _) in enumerate(plan):
        for j in range(bc):
            d0 = lead_off + j // tail_span
            d1 = tail_off + j % tail_span
            out[k, j] = d0 * w1 + d1
    return out


def split(rb, wb, v, chunk: int, bc: int):
    """The kernel's own arithmetic: per block, bf16 state and bf16 Gram, summed.

    ``rb``/``wb`` are the bf16-rounded coefficients over ALL N slots. Each block
    runs the whole recurrence on its own slot columns and the partials are added
    in the accumulator's precision (fp64 here; fp32 in the kernel, which is
    ~1e-7 and therefore far inside the bf16 floor this models).
    """
    n = rb.shape[-1]
    out = None
    for s0 in range(0, n, bc):
        part = chunked_bf16_state(rb[..., s0:s0 + bc], wb[..., s0:s0 + bc], v, chunk)
        out = part if out is None else out + part
    return out
