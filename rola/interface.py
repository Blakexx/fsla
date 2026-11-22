# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The public op.

    rola_op(routes, values, masks, decay, plan=plan)

That is the whole functional surface. It is linear attention's contract with the
feature map named: ``routes`` is the ``(q, k)`` pair AFTER the map -- a
:class:`~rola.routing.factors.RouteFactors` bundle -- ``values`` is the value stream in
the internal format -- a :class:`~rola.values.Values` bundle -- ``masks`` is the DATA beside the rows, like attention's
mask (:class:`~rola.routing.masks.Masks`: a bit a side, level, digit and token; the op computes with ``r * m``) and
``decay`` is the recurrence dial. ``plan`` is a static :class:`~rola.engine.plan.PrefillPlan`: what to do, never what
you are giving, and never changing the mathematical output -- the data does.

ONE SEGMENT a call: the op runs the prefill DAG (:mod:`rola.engine.dags.prefill_dag`)
over the segment it is handed and continues the state it is handed. Splitting a long
sequence into segments is the LAYER's (:class:`rola.RoLA`), because only the layer can
bound the projections as well as the op. The op is forward-only until the native
backward is rebuilt, and refuses a gradient by name; the fp64 oracle
(:mod:`rola.ops.naive`) remains the executable spec.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rola._state import RoLAState
from rola.engine import BackwardNotImplemented
from rola.engine.facts.operands import operands_of, requires_backward

if TYPE_CHECKING:
    from rola.engine import PrefillPlan
    from rola.routing.factors import RouteFactors
    from rola.routing.masks import Masks
    from rola.routing.types import DecayConfig
    from rola.values import Values

__all__ = [
    "BackwardNotImplemented",
    "operands_of",
    "requires_backward",
    "rola_op",
]


def rola_op(
    routes: RouteFactors,
    values: Values,
    masks: Masks,
    decay: DecayConfig = None,
    *,
    plan: PrefillPlan,
    state: RoLAState | None = None,
) -> tuple[torch.Tensor, RoLAState | None]:
    """Run RoLA over one segment: ``(y, state)``.

    ``routes``: a :class:`~rola.routing.factors.RouteFactors`; ``values``: a :class:`~rola.values.Values` with
    the same ``[B, T, H]``; ``masks``: the PREFILL-layout :class:`~rola.routing.masks.Masks` the producer emitted beside
    them (validated for shape and dtype, never values); ``decay``: the recurrence dial, or ``None``; ``plan``: the
    static :class:`~rola.engine.plan.PrefillPlan` (:func:`rola.ops.produce.prefill_plan`). ``state``: ``None`` for a stateless
    call, or a :class:`~rola.RoLAState` from :func:`rola.state` -- an unbound one binds from this call, a bound
    one is continued. Returns ``y: [B, T, H, d_v]`` fp32 and the SAME state object, mutated in place (``None``
    when the call was stateless).

    A call outside the built envelope -- a CPU operand, an unratified activation, decay, a topology or value
    width no built arm carries, ``T`` off the window -- RAISES, naming the clause. A gradient raises too, until
    the native backward is rebuilt.
    """
    from rola.engine.dags.prefill_dag import PrefillContext, validate_context

    ctx = PrefillContext(routes=routes, values=values, masks=masks, plan=plan, decay=decay, state=state,
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
        from rola.engine.dags.prefill_dag import run_prefill

        y, box["state_out"] = run_prefill(ctx)
        return y

    @staticmethod
    def backward(_node, *_grads):  # type: ignore[override]
        raise BackwardNotImplemented(
            "the RoLA prefill has no reverse pass until the native backward is rebuilt; the envelope should have "
            "refused this call before it launched.")
