# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The public op.

    fsla_op(routes, values, masks, descriptor, *, workspace=None, state=None, decay=None)

That is the whole functional surface. It is linear attention's contract with the
feature map named: ``routes`` is the ``(q, k)`` pair AFTER the map -- a
:class:`~fsla.routing.factors.RouteFactors` bundle -- ``values`` is the value stream in
the internal format -- a :class:`~fsla.values.Values` bundle -- ``masks`` is the DATA beside the rows, like attention's
mask (:class:`~fsla.routing.masks.Masks`: a bit a side, level, digit and token; the op computes with ``r * m``), the
``state`` is its own argument, in and out, and ``decay`` is the recurrence dial: the data changes the mathematics.
``descriptor`` is the call's validated run-time context (:func:`fsla.ops.produce.prefill_descriptor`): every device fact and every call fact.
``workspace`` is everything derived during the call, one buffer a caller lays out with :func:`fsla.build_workspace`: the op runs no derivation
pass of its own.

ONE SEGMENT a call: the op runs the prefill DAG (:mod:`fsla.engine.dags.prefill_dag`)
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
from fsla.engine import BackwardNotImplemented
from fsla.engine.facts.operands import operands_of, requires_backward

if TYPE_CHECKING:
    from fsla.ops.descriptor import Descriptor
    from fsla.routing.factors import RouteFactors
    from fsla.routing.masks import Masks
    from fsla.routing.types import DecayConfig
    from fsla.values import Values

__all__ = [
    "BackwardNotImplemented",
    "fsla_op",
    "operands_of",
    "requires_backward",
]


def fsla_op(
    routes: RouteFactors,
    values: Values,
    masks: Masks,
    descriptor: Descriptor,
    *,
    workspace: torch.Tensor | None = None,
    state: FSLAState | None = None,
    decay: DecayConfig = None,
) -> tuple[torch.Tensor, FSLAState | None]:
    """Run FSLA over one segment: ``(y, state)``.

    ``routes``: a :class:`~fsla.routing.factors.RouteFactors`; ``values``: a :class:`~fsla.values.Values` with
    the same ``[B, T, H]``; ``masks``: the PREFILL-layout :class:`~fsla.routing.masks.Masks` the producer emitted beside
    them (validated for shape and dtype, never values); ``descriptor``: the prefill's validated
    :class:`~fsla.ops.descriptor.Descriptor` (:func:`fsla.ops.produce.prefill_descriptor`), checked against the bundles
    (:func:`fsla.ops.prefill.check_descriptor`); ``workspace``: the buffer :func:`fsla.build_workspace` laid out for this call.
    ``state``: ``None`` for a stateless
    call, or a :class:`~fsla.FSLAState` from :func:`fsla.state` -- an unbound one binds from this call, a bound
    one is continued; ``decay``: the recurrence dial, or ``None``. Returns ``y: [B, T, H, d_v]`` fp32 and the SAME state object,
    mutated in place (``None`` when the call was stateless).

    A call outside the built envelope -- a CPU operand, decay, a topology or value
    width no built arm carries, ``T`` off the window -- RAISES, naming the clause. A gradient raises too, until
    the native backward is rebuilt.
    """
    from fsla.engine.dags.prefill_dag import PrefillContext, validate_context

    ctx = PrefillContext(routes=routes, values=values, masks=masks, descriptor=descriptor, workspace=workspace, decay=decay, state=state,
                         want_state=state is not None)
    #: in the CALLER's grad mode: inside the node's forward autograd has turned it off, and the gradient
    #: clause would never speak.
    validate_context(ctx)
    y, plane = _PrefillNode.run(ctx)
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
    def run(ctx):
        box = {}
        y = _PrefillNode.apply(ctx, box, *operands_of(ctx.routes, ctx.values, ctx.decay))
        return y, box.get("state_out")

    @staticmethod
    def forward(_node, ctx, box, *_operands):  # type: ignore[override]
        from fsla.engine.dags.prefill_dag import run_prefill

        y, box["state_out"] = run_prefill(ctx)
        return y

    @staticmethod
    def backward(_node, *_grads):  # type: ignore[override]
        raise BackwardNotImplemented(
            "the FSLA prefill has no reverse pass until the native backward is rebuilt; the envelope should have "
            "refused this call before it launched.")
