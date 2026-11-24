# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The public op.

    prefill(routes, values, masks, density, *, state=None, decay=None) -> (y, state)

That is the whole functional surface. It is linear attention's contract with the
feature map named: ``routes`` is the ``(q, k)`` pair AFTER the map -- a
:class:`~fsla.routing.factors.RouteFactors` bundle -- ``values`` is the value stream in
the internal format -- a :class:`~fsla.values.Values` bundle -- ``masks`` is the DATA beside the rows, like attention's
mask (:class:`~fsla.routing.masks.Masks`: a bit a side, level, digit and token; a bit of 0 is the caller's guarantee that the row is zero),
``density`` is how full each level is (a uint8 ``[2, D]``, 255 for every digit live), the
``state`` is its own argument, in and out, and ``decay`` is the recurrence dial: the data changes the mathematics.
The call carries only its inputs: it builds its own validated descriptor from the bundles, the dial, the state and the operands' device,
and the workspace it passes around inside (docs/api.md section 1).

ONE SEGMENT a call: the op runs the prefill call (:mod:`fsla.engine.prefill_call`)
over the segment it is handed and continues the state it is handed. Splitting a long
sequence into segments is the LAYER's (:class:`fsla.FSLA`), because only the layer can
bound the projections as well as the op. The op is forward-only until the native
backward is rebuilt, and refuses a gradient by name; the fp64 oracle
(:mod:`fsla.ops.naive`) remains the executable spec.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fsla._state import FSLAState
from fsla.engine.facts.operands import operands_of, requires_backward
from fsla.engine.types import BackwardNotImplemented

if TYPE_CHECKING:
    from fsla.routing.factors import RouteFactors
    from fsla.routing.masks import Masks
    from fsla.routing.types import DecayConfig
    from fsla.values import Values

__all__ = [
    "BackwardNotImplemented",
    "operands_of",
    "prefill",
    "requires_backward",
]


def prefill(
    routes: RouteFactors,
    values: Values,
    masks: Masks,
    density: torch.Tensor,
    *,
    state: FSLAState | None = None,
    decay: DecayConfig = None,
) -> tuple[torch.Tensor, FSLAState | None]:
    """Run FSLA over one segment: ``(y, state)``.

    ``routes``: a :class:`~fsla.routing.factors.RouteFactors`; ``values``: a :class:`~fsla.values.Values` with
    the same ``[B, T, H]``; ``masks``: the PREFILL-layout :class:`~fsla.routing.masks.Masks` the producer emitted beside
    them (validated for shape and dtype, never values; a bit of 0 is YOUR GUARANTEE that the entry's row is zero, a bit of 1 guarantees
    nothing); ``density``: a ``torch.uint8`` tensor ``[2, D]`` on the rows' device, row 0 the read side and row 1 the write side, an entry a
    level -- a byte ``b`` for the share ``b / 255`` of the level's digits a token is live on, 255 every digit and 0 none
    (:func:`fsla.masks_and_density` and :func:`fsla.density_of_masks` build it). The call checks its type, shape, dtype and device
    and NOTHING about its values (every byte is a valid density); it is carried with the call's other inputs to where the prefill's kernels will read it, to choose how the work is laid out. The GPU
    prefill is not built (it is refused by name) and the reference path does not use the density: as a fact of the contract, the density
    changes no defined result. ``state``: ``None`` for a stateless
    call, or a :class:`~fsla.FSLAState` from :func:`fsla.state` -- an unbound one binds from this call, a bound
    one is continued; ``decay``: the recurrence dial, or ``None``. Returns ``y: [B, T, H, d_v]`` fp32 and the SAME state object,
    mutated in place (``None`` when the call was stateless).

    A call outside the built envelope -- a CPU operand, decay, a topology or value
    width no built arm carries, ``T`` off the window -- RAISES, naming the clause. A gradient raises too, until
    the native backward is rebuilt.
    """
    from fsla.engine.prefill_call import PrefillContext, descriptor_of, validate_context

    ctx = PrefillContext(routes=routes, values=values, masks=masks, density=density, decay=decay, state=state,
                         want_state=state is not None)
    #: in the CALLER's grad mode: inside the node's forward autograd has turned it off, and the gradient
    #: clause would never speak.
    validate_context(ctx)
    descriptor = descriptor_of(ctx)
    y, plane = _PrefillNode.run(ctx, descriptor)
    if state is None:
        return y, None
    state._kernel_commit(plane)
    return y, state


class _PrefillNode(torch.autograd.Function):
    """The prefill as an autograd node whose backward refuses, loudly.

    The envelope refuses a gradient before any launch; this node is what makes it LOUD if that refusal is ever
    bypassed -- a first line and a last line. ``forward`` takes every tensor the operation consumes
    POSITIONALLY, because autograd inspects the flat argument list and an operand inside the context would be
    invisible to it.
    """

    @staticmethod
    def run(ctx, descriptor):
        box = {}
        y = _PrefillNode.apply(ctx, descriptor, box, *operands_of(ctx.routes, ctx.values, ctx.decay), ctx.density)
        return y, box.get("state_out")

    @staticmethod
    def forward(_node, ctx, descriptor, box, *_operands):  # type: ignore[override]
        from fsla.engine.prefill_call import run_prefill

        y, box["state_out"] = run_prefill(ctx, descriptor)
        return y

    @staticmethod
    def backward(_node, *_grads):  # type: ignore[override]
        raise BackwardNotImplemented(
            "the FSLA prefill has no reverse pass until the native backward is rebuilt; the envelope should have "
            "refused this call before it launched.")
