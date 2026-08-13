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


def level_plan(widths, bc: int):
    """The law at ANY depth: ``(offs, spans)`` -- the LevelPlan, in Python.

    ``spans[l]`` is how many digits of level ``l`` a block spans: reading
    levels most-significant first, every level above the boundary at exactly
    ONE digit, at most one level PARTIALLY by an aligned run dividing its
    width, every level below spanned ENTIRELY.  ``offs[k][l]`` is block ``k``'s
    first digit at level ``l`` -- the same mixed-radix map the kernel's
    ``off_at`` computes.  Raises unless (widths, BC) is a rectangle in digit
    space.
    """
    D = len(widths)
    spans, rem, l = [1] * D, bc, D - 1
    while l >= 0 and rem >= widths[l]:
        if rem % widths[l]:
            raise ValueError(f"BC={bc} spans level {l} partially AND wraps it "
                             f"in {widths}")
        spans[l] = widths[l]
        rem //= widths[l]
        l -= 1
    if rem > 1:
        if l < 0:
            raise ValueError(f"BC={bc} exceeds the slot space of {widths}")
        if widths[l] % rem:
            raise ValueError(f"BC={bc}'s partial run does not tile level {l} "
                             f"of {widths}")
        spans[l] = rem
    prod = 1
    for s in spans:
        prod *= s
    if prod != bc:
        raise ValueError(f"BC={bc} is not a rectangle in digit space for {widths}")
    groups = [w // s for w, s in zip(widths, spans)]
    sufs = [1] * (D + 1)
    for i in range(D - 1, -1, -1):
        sufs[i] = sufs[i + 1] * groups[i]
    offs = [tuple(((k // sufs[i + 1]) % groups[i]) * spans[i] for i in range(D))
            for k in range(sufs[0])]
    return offs, spans


def block_plan(widths, bc: int):
    """The TWO-LEVEL VIEW of ``level_plan``: ``[(lead_off, tail_off, slot0)]``.

    Kept because a body of two-level rungs consumes this exact shape; the law
    itself lives in ``level_plan`` and this derives from it.
    """
    if len(widths) != 2:
        raise ValueError(f"the two-level view spans two levels, got widths={widths}")
    offs, (lead_span, tail_span) = level_plan(widths, bc)
    out = [(o0, o1, o0 * widths[1] + o1) for o0, o1 in offs]
    return out, lead_span, tail_span


def block_slots(widths, bc: int, device="cuda"):
    """[blocks, BC] int table of the GLOBAL slot each block-local slot names.

    Built from the mixed-radix definition with an explicit divide and modulo over
    the block's digit rectangle -- the leaf-major construction the kernel's loop
    nest deliberately never executes.
    """
    offs, spans = level_plan(widths, bc)
    D = len(widths)
    wsuf = [1] * (D + 1)          # slot stride of a digit at each level
    ssuf = [1] * (D + 1)          # block-local stride of the span rectangle
    for i in range(D - 1, -1, -1):
        wsuf[i] = wsuf[i + 1] * widths[i]
        ssuf[i] = ssuf[i + 1] * spans[i]
    out = torch.empty(len(offs), bc, dtype=torch.long, device=device)
    for k, off in enumerate(offs):
        for j in range(bc):
            slot = 0
            for l in range(D):
                slot += (off[l] + (j // ssuf[l + 1]) % spans[l]) * wsuf[l + 1]
            out[k, j] = slot
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
