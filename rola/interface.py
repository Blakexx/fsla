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

if TYPE_CHECKING:
    from rola.routing.factors import RouteFactors
    from rola.routing.types import DecayConfig

__all__ = [
    "BackwardNotImplemented",
    "ChunkConsumer",
    "envelope_refusal",
    "operands_of",
    "require_envelope",
    "requires_backward",
    "rola_op",
]

#: The one sentence every no-backward refusal in the package ends with.
BACKWARD_ARRIVES_WITH = (
    "the backward arrives with the native backward (docs/open-work.md W9); until then "
    "there is no training path through this surface")

class BackwardNotImplemented(NotImplementedError):
    """Raised when a gradient is demanded of the forward-only kernel surface."""

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
    from rola.facts.chunk_dag import ChunkContext, build_chunk_plan

    plan = build_chunk_plan(ChunkContext(
        routes=routes, v=v, decay=decay, state=state, expert=expert,
        want_state=state is not None))
    y = ChunkConsumer.run(plan, v)
    if state is None:
        return y, None
    state._kernel_commit(plan.state_out)
    return y, state

def _paging_preference(expert: Any) -> bool:
    """Whether a state bound by this call gets a PAGED backing. Default: yes.

    The expert may set ``paging=False`` and only False -- ``paging=True`` is refused
    by :class:`rola.expert.PlanOverrides` itself, because there is nothing to turn on.
    That is the "expert overrides the MAPPING only" ruling, made mechanical.
    """
    if expert is None:
        return True
    from rola.expert import PlanOverrides

    return PlanOverrides.of(expert).paging is not False

# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------

def operands_of(routes: RouteFactors, v: torch.Tensor, decay: Any = None) -> tuple:
    """Every tensor the OPERATION consumes, flat -- the predicate's argument list."""
    return (v, *routes.tensors(), *(() if decay is None else tuple(decay.dials)))

def requires_backward(*tensors: Any) -> bool:
    """Whether a call over these operands must produce a differentiable result.

    Evaluated on the tensors the OPERATION consumes -- ``v``, the routing factors, the
    side gains, the decay dials -- and never on a module's parameters. A
    frozen-parameter forward under ``torch.no_grad()`` has no graph to sever and
    correctly takes the kernel; a dial-only unfrozen finetune has exactly one
    grad-requiring operand and must not.
    """
    if not torch.is_grad_enabled():
        return False
    return any(_any_requires_grad(t) for t in tensors)

def envelope_refusal(routes: RouteFactors, v: torch.Tensor, decay: Any = None) -> str | None:
    """Why the built kernel arm has no arm for this call, or ``None`` if it runs.

    THE ONE AUTHORITY on the envelope's CONFIGURATION clauses, so a caller can ask
    before calling and get the sentence the refusal will carry. The gradient and
    device clauses are facts about the operands rather than the configuration and live
    in :func:`require_envelope` beside them.
    """
    unratified = tuple(routes.topology.unratified_activations)
    if unratified:
        return (f"unratified activations {unratified}; no kernel was measured for them "
                "(docs/ratification.md)")
    if routes.device.type != "cuda":
        return f"the operands are on {routes.device}; the arm is a CUDA kernel"
    from rola.ops import chunk as _chunk

    return _chunk.unsupported(routes, decay, d_v=v.shape[-1])

def require_envelope(routes: RouteFactors, v: torch.Tensor, decay: Any = None,
                     state: Any = None) -> None:
    """Refuse anything the built arm cannot run. THE FAST-FAIL, in one place.

    There is no second arm to route to: the fp64 oracle is the executable spec, not a
    fallback, and a caller who wants it calls it (:func:`rola.ops.naive.naive_rola`).

    **THE GRADIENT IS NO LONGER A CLAUSE HERE.** It was, while the operator was
    forward-only; the two-pass backward is built and gated, so a grad-bearing call
    now runs. What remains is the CONFIGURATION envelope, which refuses whether or
    not a gradient is attached, plus one clause that exists only on the grad path:
    a carried STATE. Training in v1 is stateless (P65 R-D) -- the state plane is
    caller-owned and mutated in place, and `d initial_state` is not in the scans --
    so `state=` and a gradient are mutually exclusive, by name, at the boundary
    rather than silently.
    """
    grad = requires_backward(*operands_of(routes, v, decay))
    if grad and state is not None:
        raise BackwardNotImplemented(
            "rola_op trains STATELESS ONLY and this call carries both a gradient and "
            "a state plane. The state is caller-owned and mutated in place, and the "
            "reverse pass has no `d initial_state`: it rebuilds from zero. Detach the "
            "operands for a stateful step, or drop `state=` to train.")
    if grad and routes.device.type == "cuda":
        from rola.ops import chunk as _chunk

        #: THE REVERSE PASS' ARM MATRIX IS NARROWER THAN THE FORWARD'S, so a call
        #: that would run forward and fail backward is refused HERE rather than
        #: three launches later with the operands already consumed.
        back = _chunk.backward_unsupported(routes, decay, d_v=v.shape[-1])
        if back is not None:
            raise BackwardNotImplemented(
                f"rola_op can run this call but cannot differentiate it: {back}. Run "
                "it under torch.no_grad() with the operands detached, or move to an "
                "arm the reverse pass carries.")
    reason = envelope_refusal(routes, v, decay)
    if reason is not None:
        raise NotImplementedError(
            f"rola_op runs the BUILT envelope only and there is no arm for this call: "
            f"{reason}. Nothing else is reachable through this surface "
            "(docs/api.md §1).")

def _any_requires_grad(obj: Any) -> bool:
    if isinstance(obj, torch.Tensor):
        return obj.requires_grad
    if isinstance(obj, (tuple, list)):
        return any(_any_requires_grad(item) for item in obj)
    return False

class ChunkConsumer(torch.autograd.Function):
    """The CHUNK consumer as an autograd node whose backward refuses, loudly.

    ``rola.ops.chunk.execute`` is a bare extension call: reached from a gradient-requiring
    context it returns a perfectly good tensor with no ``grad_fn``, and the symptom is
    ``x.grad is None`` several files away (the layer-dispatch autograd gap).
    :func:`require_envelope` is what keeps that from happening; this node is what makes
    it LOUD if the refusal is ever bypassed -- a first line and a last line, not one
    line twice.

    ``forward`` takes the routing tensors it does not read. That is the mechanism, not
    an oversight: autograd inspects the FLAT argument list for grad-requiring tensors,
    so an operand passed inside the plan would be invisible to it and the node would
    attach to nothing. Everything the operation consumes is therefore passed
    positionally, and :meth:`run` is the one place that is assembled.
    """

    @staticmethod
    def run(plan, v):
        """``y`` -- one :class:`~rola.facts.plan.ChunkPlan` executed through the node.

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
        from rola.ops import chunk as _chunk

        y, den = _chunk.execute(plan)
        #: THE CTX IS TAKEN ON THE GRADIENT'S OWN CONDITION, and `needs_input_grad`
        #: is that condition rather than a guess at it: an inference call retains
        #: nothing, so the two-pass activation cost is paid by training alone
        #: (docs/internals/chunk/backward.md).
        if any(ctx.needs_input_grad):
            #: `v` and `y` ride `save_for_backward` because they are this node's own
            #: operand and output and autograd owns their versioning; the O(L)
            #: byproducts are not graph tensors and ride beside them.
            ctx.save_for_backward(v, y)
            ctx.saved_plan = _chunk.SavedForward.of(plan, den)
            #: the bundle, for its LAYOUTS alone: `backward` needs each level's width
            #: and dtype to un-pack the digit gradients, and those are the bundle's.
            ctx.saved_routes = plan.routes
        return y

    @staticmethod
    def backward(ctx, *grad_outputs):  # type: ignore[override]
        """The two-pass reverse pass, returned in `RouteFactors.tensors()`' ORDER.

        That ordering is load-bearing rather than incidental: `forward` receives the
        operands flat and in exactly that order, so `backward` must return them the
        same way or every gradient lands on the wrong tensor SILENTLY -- the shapes
        agree. `tests/unit/test_interface.py` asserts the correspondence rather than
        trusting it.

        Everything the scans emit is fp32 in the kernels' layouts; each gradient is
        narrowed to its operand's dtype and un-permuted to its public layout here,
        because autograd requires `grad.dtype == input.dtype` and the packing that
        produced the kernel layout was this node's own.
        """
        from rola.ops import chunk as _chunk

        saved = getattr(ctx, "saved_plan", None)
        if saved is None:
            raise BackwardNotImplemented(
                "the RoLA chunk consumer reached backward with no saved ctx: the "
                "forward retains one only when the call carries a gradient, so this "
                "node was applied outside a gradient context and then differentiated.")
        v, y = ctx.saved_tensors
        routes = ctx.saved_routes
        dv, dpr, dpw, dgw, dm = _chunk.backward(saved, grad_outputs[0], y)

        def _levels(packed_side, levels):
            """`[B, H, T, sumW]` fp32 back to the per-level `[B, T, H, width_l]`."""
            out, off = [], 0
            for level in levels:
                w = level.shape[-1]
                out.append(packed_side[..., off:off + w].permute(0, 2, 1, 3)
                           .to(level.dtype).contiguous())
                off += w
            return out

        grads = [dv.permute(0, 2, 1, 3).to(v.dtype).contiguous()]
        grads += _levels(dpr, routes.read)
        grads += _levels(dpw, routes.write)
        #: `g_read` does not exist on this surface, so the gains are the write gain
        #: alone -- the same one clause `tensors()` builds.
        grads.append(dgw.permute(0, 2, 1).to(routes.g_write.dtype).contiguous())
        if routes.read_mass is not None:
            #: SURPRISE 2: the m-carry is forward-exact but NOT gradient-neutral.
            grads.append(dm.permute(0, 2, 1).to(routes.read_mass.dtype).contiguous())
        #: the leading `None` is the plan, which is not a tensor
        return (None, *grads)
