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

from dataclasses import dataclass
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


@dataclass
class _ChunkCall:
    """The non-tensor half of a chunk launch, and the plane it hands back.

    Everything :func:`rola.ops.chunk.run` needs that autograd must NOT see as an
    operand, plus the out-parameter the state plane returns through.
    """

    routes: Any
    state_in: torch.Tensor | None
    want_state: bool
    arena: Any
    plane: torch.Tensor | None


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
    :class:`~rola.routing.factors.RouteGateway` or built directly. Anything else is
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

    A call outside the built envelope -- a gradient anywhere in it, a CPU operand, an
    unratified activation, decay, a jagged topology, ``d_v != 64``, an unbuilt
    ``(D, B)``, ``T`` off the chunk -- RAISES, naming the clause that has no arm.
    """
    _validate_call(routes, v, decay)
    if state is not None and type(state) is not RoLAState:
        raise TypeError(
            "rola_op's `state` argument is a rola.RoLAState from rola.state(), got "
            f"{type(state).__name__}. A raw tensor is not one: the state owns its own "
            "backing (dense or paged) and the shape it is bound to.")
    require_envelope(routes, v, decay)
    from rola.expert import PlanOverrides
    from rola.ops import chunk as _chunk

    paging = _paging_preference(expert)
    #: The chunk arm's `BC` is fixed by the manifest, not host-selectable, so an
    #: expert pin over it is refused BY NAME rather than accepted and ignored.
    if expert is not None and PlanOverrides.of(expert).state_block is not None:
        raise NotImplementedError(
            "chunk consumer (kernel arm): PlanOverrides(state_block=...) pins the "
            "TILED consumer's owner block. The chunk arm's BC belongs to the built arm "
            "the manifest carries for this topology and is not host-selectable; pass "
            "state_block=None.")
    if state is None:
        #: No state object, no state I/O: the variant with none of it compiled in
        #: runs (docs/internals/chunk/chunk_kernel.md#state-axis).
        y, _ = ChunkConsumer.run(routes, v)
        return y, None
    # THE HOST ORDER is why these four lines are not folded into `chunk.run`
    # (docs/internals/state.md#host-order).
    atom_bits = _chunk.stats(routes) if state._wants_pages(paging) else None
    s_in, arena = state._kernel_entry(
        routes, atom_bits, d_v=v.shape[-1], paging=paging, BC=_chunk.arm(routes)[1])
    y, plane = ChunkConsumer.run(routes, v, state_in=s_in, want_state=True, arena=arena)
    state._kernel_commit(plane)
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


def require_envelope(routes: RouteFactors, v: torch.Tensor, decay: Any = None) -> None:
    """Refuse anything the built arm cannot run. THE FAST-FAIL, in one place.

    There is no second arm to route to: the fp64 oracle is the executable spec, not a
    fallback, and a caller who wants it calls it (:func:`rola.ops.naive.naive_rola`).
    """
    if requires_backward(*operands_of(routes, v, decay)):
        raise BackwardNotImplemented(
            f"rola_op is forward-only and this call carries a gradient: "
            f"{BACKWARD_ARRIVES_WITH}. Run under torch.no_grad() with the operands "
            "detached.")
    reason = envelope_refusal(routes, v, decay)
    if reason is not None:
        raise NotImplementedError(
            f"rola_op runs the BUILT envelope only and there is no arm for this call: "
            f"{reason}. Nothing else is reachable through this surface "
            "(docs/api.md §1).")


def _validate_call(routes, v, decay) -> None:
    from rola.routing.factors import RouteFactors as _RouteFactors

    if type(routes) is not _RouteFactors:
        raise TypeError(
            "rola_op's first argument must be a RouteFactors bundle, got "
            f"{type(routes).__name__}. The routing sides are the feature-mapped (q, k) pair; "
            "build one with RouteGateway, or construct a RouteFactors directly -- its "
            "invariants are checked either way.")
    #: THE GAIN LAYOUT AGAINST THE ARM, structurally: architecture §4.2 fixes
    #: `g_read := 1`, so a supplied one would be silently ignored. Checked here rather
    #: than trusted from the gateway because the bundle is hand-constructible.
    if routes.g_read is not None:
        raise ValueError(
            "this bundle carries a g_read tensor: architecture §4.2 fixes g_read := 1 "
            "under the self-normalizing readout, and there is no read-gain column")
    topology = routes.topology
    if not isinstance(v, torch.Tensor) or v.ndim != 4:
        raise ValueError(f"v must be [B, T, H, d_v], got {getattr(v, 'shape', type(v).__name__)}")
    #: THE LAST DIM IS NOT CHECKED, IT IS READ. `d_v` is the value stream's own width
    #: and the routing carries no value geometry, so there is nothing to agree with;
    #: what must match the bundle is the token addressing, `[B, T, H]`.
    expected = (routes.batch, routes.tokens, routes.heads)
    if tuple(v.shape[:3]) != expected:
        raise ValueError(
            f"v must be [B, T, H, d_v] with [B, T, H]={expected} to match the routing "
            f"bundle, got {tuple(v.shape)}")
    if v.device != routes.device:
        raise ValueError(f"v is on {v.device} but the routing bundle is on {routes.device}")
    if decay is not None:
        decay.validate_against(topology)


def _any_requires_grad(obj: Any) -> bool:
    if isinstance(obj, torch.Tensor):
        return obj.requires_grad
    if isinstance(obj, (tuple, list)):
        return any(_any_requires_grad(item) for item in obj)
    return False


class ChunkConsumer(torch.autograd.Function):
    """The CHUNK consumer as an autograd node whose backward refuses, loudly.

    ``rola.ops.chunk.run`` is a bare extension call: reached from a gradient-requiring
    context it returns a perfectly good tensor with no ``grad_fn``, and the symptom is
    ``x.grad is None`` several files away (the layer-dispatch autograd gap).
    :func:`require_envelope` is what keeps that from happening; this node is what makes
    it LOUD if the refusal is ever bypassed -- a first line and a last line, not one
    line twice.

    ``forward`` takes the routing tensors it does not read. That is the mechanism, not
    an oversight: autograd inspects the FLAT argument list for grad-requiring tensors,
    so an operand passed inside the ``call`` payload would be invisible to it and the
    node would attach to nothing. Everything the operation consumes is therefore passed
    positionally, and :meth:`run` is the one place that is assembled.
    """

    @staticmethod
    def run(routes, v, *, state_in=None, want_state=False, arena=None):
        """``(y, final state or None)`` -- ``rola.ops.chunk.run`` through the node."""
        call = _ChunkCall(routes=routes, state_in=state_in, want_state=want_state,
                          arena=arena, plane=None)
        y = ChunkConsumer.apply(call, v, *routes.tensors())
        return y, call.plane

    @staticmethod
    def forward(ctx, call, v, *operands):  # type: ignore[override]
        # Deliberately NOT `ctx.mark_non_differentiable`: marking the output would make
        # `backward` unreachable, which is the same silence this class exists to break.
        from rola.ops import chunk as _chunk

        y, call.plane = _chunk.run(
            call.routes, v, state_in=call.state_in, want_state=call.want_state,
            arena=call.arena)
        #: The state plane leaves through the payload rather than as a second output:
        #: it is the CALLER's allocation (the arena's, when paged), and an autograd
        #: node that returned it would stamp graph metadata onto storage the state
        #: object owns and reuses across steps.
        return y

    @staticmethod
    def backward(ctx, *grad_outputs):  # type: ignore[override]
        raise BackwardNotImplemented(
            f"the RoLA chunk consumer is forward-only: {BACKWARD_ARRIVES_WITH}. Reaching "
            "this means the kernel was invoked on grad-requiring operands, bypassing "
            "`rola.rola_op`, which refuses them.")
