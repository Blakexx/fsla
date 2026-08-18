# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The public op, and the one autograd boundary in the package.

    rola_op(routes, v, decay)

That is the whole functional surface. It is linear attention's contract with the
feature map named: ``routes`` is the ``(q, k)`` pair AFTER the map -- a
:class:`~rola.routing.factors.RouteFactors` bundle -- ``v`` is the value stream, and
``decay`` is the recurrence dial. No plan, no configuration, no statistics appear on
it, because none of them are model facts: they are how the op runs, and the op decides
how it runs.

**THE BUILT ENVELOPE IS THE WHOLE SURFACE** (docs/api.md §1): a call the built kernel
arm cannot run is REFUSED here, by name, and nothing else is reachable through this
module. The fp64 oracle (:mod:`rola.ops.naive`) is the executable spec and the thing
tests and ratification call directly; it is not an arm of this op.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from rola._state import RoLAState
from rola.engine import BackwardNotImplemented, rules
from rola.engine.facts.operands import operands_of, requires_backward

if TYPE_CHECKING:
    from rola.routing.factors import RouteFactors
    from rola.routing.types import DecayConfig

__all__ = [
    "BackwardNotImplemented",
    "ChunkConsumer",
    "operands_of",
    "requires_backward",
    "rola_op",
]

# ---------------------------------------------------------------------------
# The public op
# ---------------------------------------------------------------------------

def rola_op(
    routes: RouteFactors,
    v: torch.Tensor,
    decay: DecayConfig = None,
    *,
    state: RoLAState | None = None,
    expert: Any = None,
) -> tuple[torch.Tensor, RoLAState | None]:
    """Run RoLA over one routing bundle and one value stream.

    ``routes``: a :class:`~rola.routing.factors.RouteFactors`, from
    :class:`~rola.routing.producer.RouteProducer` or built directly. Anything else is
    refused by type: the routing sides are the feature-mapped ``(q, k)`` pair and a
    loose tuple of tensors is not one.
    ``v``: ``[B, T, H, d_v]``. THE VALUE WIDTH IS READ HERE, off ``v``: the routing
    bundle carries no value geometry, so ``d_v`` is a fact about this operand and the
    op is where it enters.
    ``decay``: the recurrence dial, or ``None``. Build one from a decay source
    (:mod:`rola.routing.decay`), which is the slot that decides where rates come from.

    ``state``: ``None`` for a stateless call, or a :class:`~rola.RoLAState` from
    :func:`rola.state`. THE PREDICATE IS ``state is not None``: it is the whole
    engagement signal for the carried state, there is no second flag, and a stateless
    call runs the kernel variant that has no state I/O compiled into it at all. An
    unbound state binds its keying, its device and its backing from this call; a bound
    one is checked for MATH-COMPATIBILITY only and continued.

    Returns ``(y, state)`` -- ``y: [B, T, H, d_v]`` in the operand dtype, and the SAME
    state object that was handed in, mutated in place (``None`` when the call was
    stateless). Read the state with ``state.materialize() -> [B, H, N, d_v + 1]``; that
    is a bridge for the oracle and for a test, never a measured path.

    ``expert``: :class:`rola.expert.PlanOverrides`, the quarantined advanced surface.
    It is a keyword argument with a default of ``None`` because the op does not need
    it: every axis it pins is one the op decides for itself.

    A call outside the built envelope -- a CPU operand, an unratified activation,
    decay, a jagged topology, ``d_v != 64``, an unbuilt ``(D, B)``, ``T`` off the
    chunk -- RAISES, naming the clause that has no arm. A GRADIENT IS NO LONGER ONE
    OF THEM: the two-pass reverse pass is built, so a grad-bearing call trains.
    Carrying a gradient AND a ``state`` still raises -- training is stateless in v1
    (P65 R-D).
    """
    from rola.engine.dags.chunk_dag import ChunkContext, build_chunk_plan

    plan = build_chunk_plan(ChunkContext(
        routes=routes, v=v, decay=decay, state=state, expert=expert,
        want_state=state is not None))
    y = ChunkConsumer.run(plan, v)
    if state is None:
        return y, None
    state._kernel_commit(plan.state_out)
    return y, state

class ChunkConsumer(torch.autograd.Function):
    """The CHUNK consumer as an autograd node whose backward refuses, loudly.

    ``rola.ops.chunk.execute`` is a bare extension call: reached from a gradient-requiring
    context it returns a perfectly good tensor with no ``grad_fn``, and the symptom is
    ``x.grad is None`` several files away (the layer-dispatch autograd gap).
    :func:`rola.engine.facts.call.require_envelope` is what keeps that from happening;
    this node is what makes it LOUD if the refusal is ever bypassed -- a first line and
    a last line, not one line twice.

    ``forward`` takes the routing tensors it does not read. That is the mechanism, not
    an oversight: autograd inspects the FLAT argument list for grad-requiring tensors,
    so an operand passed inside the plan would be invisible to it and the node would
    attach to nothing. Everything the operation consumes is therefore passed
    positionally, and :meth:`run` is the one place that is assembled.
    """

    @staticmethod
    def run(plan, v):
        """``y`` -- one :class:`~rola.engine.plan.ChunkPlan` executed through the node.

        ``v`` is passed beside the plan rather than on it because it is this node's
        own OPERAND and autograd owns its versioning; ``plan.v`` is the kernel's
        bf16-contiguous copy of it, which is a fact.

        The final state plane is ``plan.state_out`` and does not come back through
        this call: it is the CALLER's allocation (the arena's, when paged), and an
        autograd node that returned it would stamp graph metadata onto storage the
        state object owns and reuses across steps.
        """
        return ChunkConsumer.apply(plan, v, *plan.routes.tensors())

    @staticmethod
    def forward(ctx, plan, v, *operands):  # type: ignore[override]
        # Deliberately NOT `ctx.mark_non_differentiable`: marking the output would make
        # `backward` unreachable, which is the same silence this class exists to break.
        from rola.engine.plan import BackwardPlan
        from rola.ops import chunk as _chunk

        y, den = _chunk.execute(plan)
        #: THE CTX IS TAKEN ON THE GRADIENT'S OWN CONDITION, and `needs_input_grad`
        #: is that condition rather than a guess at it: an inference call retains
        #: nothing, so the two-pass activation cost is paid by training alone
        #: (docs/internals/chunk/backward.md).
        if rules.retention(any(ctx.needs_input_grad)):
            #: `v` and `y` ride `save_for_backward` because they are this node's own
            #: operand and output and autograd owns their versioning; the O(L)
            #: byproducts are not graph tensors and ride beside them.
            ctx.save_for_backward(v, y)
            #: THE CTX IS THE PLAN, BY REFERENCE, plus the one thing the forward
            #: produced rather than configured. RETENTION IS THE WHOLE DIFFERENCE
            #: between this call and an inference one: no fact is re-derived, no
            #: tensor is copied and no launch is added here.
            ctx.saved_plan = BackwardPlan(forward=plan, den=den)
        return y

    @staticmethod
    def backward(ctx, *grad_outputs):  # type: ignore[override]
        """The two-pass reverse pass: ONE walk of the backward DAG over the ctx.

        The ctx IS the forward's plan (:class:`~rola.engine.plan.BackwardPlan`), so
        there is nothing to re-derive and nothing to thread here -- the seeds, the two
        scans and the un-packing are declared nodes over that plan
        (:mod:`rola.engine.dags.backward_dag`), and the routing bundle the un-packing needs
        for its layouts is the plan's own.

        What comes back is ordered by `RouteFactors.tensors()`, which `forward`'s flat
        argument list is also ordered by; `tests/unit/test_interface.py` asserts that
        correspondence rather than trusting it.
        """
        from rola.engine.dags.backward_dag import run_backward

        plan = getattr(ctx, "saved_plan", None)
        if plan is None:
            raise BackwardNotImplemented(
                "the RoLA chunk consumer reached backward with no saved ctx: the "
                "forward retains one only when the call carries a gradient, so this "
                "node was applied outside a gradient context and then differentiated.")
        v, y = ctx.saved_tensors
        #: the leading `None` is the plan, which is not a tensor
        return (None, *run_backward(plan, grad_outputs[0], y, v))
