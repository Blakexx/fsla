# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The public op, and the one autograd boundary in the package.

    rola_op(routes, v, decay)

That is the whole functional surface. It is linear attention's contract with the
feature map named: ``routes`` is the ``(q, k)`` pair AFTER the map -- a
:class:`~rola.routing.factors.RouteFactors` bundle, obtainable only from
:class:`~rola.routing.factors.RouteProducer` -- ``v`` is the value stream, and
``decay`` is the recurrence dial. No plan, no configuration, no statistics appear on
it, because none of them are model facts: they are how the op runs, and the op decides
how it runs.

**The private seam.** Inside, the work splits in two:

``_derive_plan(routes, ...) -> Plan``
    Derive the launch geometry. Everything it reads is launch-static -- the topology,
    the cohort count, the device's own facts -- so it is host-side arithmetic over
    :func:`~rola.planner.derive.derive_bc`'s three ceilings, never a pass over the
    tensors.

``_execute(plan, routes, v, ...)``
    Build the §4 schedule from the realized masks and run the kernel.

The seam is deliberately private: how a launch is shaped is not a model fact, and the
reduction theorem is what the public surface is allowed to say. :mod:`rola.expert` is
the quarantined door to it, and it is also what lets a conformance test drive
``_execute`` with a SYNTHETIC plan -- a stale one, one straddling a boundary -- which
is the adversarial handle a plan-derived-then-immediately-used pipeline would not have.

**Why the boundary sits here** (`[[layer-dispatch-autograd-gap]]`): a forward-only
kernel reachable from a gradient-requiring context does not raise. It returns a
perfectly good tensor with no ``grad_fn``, the graph is severed, and the symptom is
``x.grad is None`` several files away. ``rola_op`` is the ONE place that rule is
enforced: a call that wants a gradient runs the whole operation on ``naive_rola``,
forward and backward from one graph; a call that does not reaches a CUDA consumer
wrapped in an autograd node -- :class:`RoLAConsumer` for the tiled arm,
:class:`ChunkConsumer` for the chunk one -- whose backward raises by name rather than
severing anything. The dispatch above is the FIRST line of that defence and the nodes
are the LAST: the dispatch decides, and the node is what makes a decision that was
somehow bypassed loud instead of silent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from rola._state import RoLAState
from rola.ops.consumer import _consumer_forward, sm_shared_bytes
from rola.ops.naive import naive_rola
from rola.planner.derive import MachineFacts, derive_bc
from rola.planner.schedule import Plan, build_plan, build_schedule

if TYPE_CHECKING:
    from rola.planner.schedule import Schedule
    from rola.routing.factors import RouteFactors
    from rola.routing.types import DecayConfig

__all__ = ["BackwardNotImplemented", "ChunkConsumer", "RoLAConsumer", "rola_op"]

#: Tier-H facts, cached on the DEVICE they are facts about. No member can move within a
#: process for a fixed device, and no routing statistic lives here.
_MACHINE: dict[torch.device, MachineFacts] = {}


class BackwardNotImplemented(NotImplementedError):
    """Raised when a gradient is demanded of the forward-only CUDA consumer."""


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
    _arm: str | None = None,
) -> tuple[torch.Tensor, RoLAState | None]:
    """Run RoLA over one routing bundle and one value stream.

    ``routes``: a :class:`~rola.routing.factors.RouteFactors` from
    :class:`~rola.routing.factors.RouteProducer`. Anything else is refused -- a bundle
    the gateway did not derive is a bundle whose statistics need not describe its own
    tensors, and every trustless property of this surface rests on that being
    impossible.
    ``v``: ``[B, T, H, d_v]``.
    ``decay``: the recurrence dial, or ``None``. Build one from a decay source
    (:mod:`rola.routing.decay`), which is the slot that decides where rates come from.

    ``state``: ``None`` for a stateless call, or a :class:`~rola.RoLAState` from
    :func:`rola.state`. THE PREDICATE IS ``state is not None``: it is the whole
    engagement signal for the carried state, there is no second flag, and a stateless
    call runs the kernel variant that has no state I/O compiled into it at all. An
    unbound state binds its keying, its device and its backing from this call; a bound
    one is checked for MATH-COMPATIBILITY only and continued.

    Returns ``(y, state)`` -- ``y: [B, T, H, d_v]``, and the SAME state object that was
    handed in, mutated in place (``None`` when the call was stateless). Read the state
    with ``state.materialize() -> [B, H, N, d_v + (norm == 'global')]``; that is a
    bridge for the oracle and for a test, never a measured path.

    ``y``'s dtype differs by arm: the differentiable arm is the fp64 oracle and hands
    back fp64 whatever it was given, the kernel arm accumulates in fp32 and hands back
    the operand dtype. A carried state may cross arms between steps -- the facade
    bridges it -- so cast ``y`` if your downstream needs a particular dtype; neither
    arm guesses, because a reference that silently rounded its own result to an
    operand's precision would make an accuracy gate compare against a rounded
    reference.

    ``expert``: :class:`rola.expert.PlanOverrides`, the quarantined advanced surface.
    It is a keyword argument with a default of ``None`` because the op does not need
    it: every axis it pins is one the planner derives.
    """
    _validate_call(routes, v, decay)
    if state is not None and type(state) is not RoLAState:
        raise TypeError(
            "rola_op's `state` argument is a rola.RoLAState from rola.state(), got "
            f"{type(state).__name__}. A raw tensor is not one: the state owns its own "
            "backing (dense or paged) and the keying it is bound to.")
    paging = _paging_preference(expert)
    operands = (v, *routes.tensors(), *(() if decay is None else tuple(decay.dials)))
    # `_arm` is how a MODULE contributes its own condition -- `RoLA` gates the kernel on
    # `self.training` as well, because reentrant checkpointing's first pass runs under
    # `no_grad` with the module still training. It is private and undocumented on
    # purpose: "training mode" has no meaning for a free function, so it is not a knob.
    #: the decay retirement: decay is not implemented in the kernel arm; like an
    #: unratified activation, it DISPATCHES to the reference arm (the fp64
    #: oracle carries the semantics).  An explicit _arm="kernel" still gets
    #: the loud refusal below -- dispatch is not a lie, override is.
    if decay is not None and _arm is None:
        _arm = "reference"
    if (_arm or _backend_for(routes, *operands)) == "reference":
        return _reference_arm(routes, v, decay, state, paging=paging)
    # operator adoption (docs/internals/chunk/).
    #: THE OVERRIDE, AND ONLY THE OVERRIDE. `dispatch_for` has already sent every
    #: bundle outside the built chunk matrix to the reference arm, so the only way to
    #: arrive here with a refusal in hand is `_arm="kernel"` -- a caller overruling
    #: the dispatch and asking for a kernel that does not exist for this shape. That
    #: is refused by name, on the decay precedent: dispatch is not a lie, override is.
    from rola.ops import chunk as _chunk

    reason = _chunk.unsupported(routes, decay)
    if reason is not None:
        raise NotImplementedError(
            f"chunk consumer (kernel arm): {reason}. The dispatch would have run this "
            "on the reference arm; _arm=\"kernel\" asked for the kernel by name and "
            "there is none for this configuration.")
    #: THE EXPERT'S `BC` PIN IS AN AXIS THIS ARM DOES NOT HAVE. The chunk consumer's
    #: `BC` is a slot tile INSIDE a built arm, fixed by the manifest and not
    #: host-selectable, so a pin over it is refused by name rather than accepted and
    #: ignored: a knob that silently does nothing is how a caller comes to believe a
    #: launch was shaped a way it never was.
    from rola.expert import PlanOverrides

    if expert is not None and PlanOverrides.of(expert).state_block is not None:
        raise NotImplementedError(
            "chunk consumer (kernel arm): PlanOverrides(state_block=...) pins the "
            "TILED consumer's owner block. The chunk arm's BC belongs to the built arm "
            "the manifest carries for this topology and is not host-selectable; pass "
            "state_block=None, or run the reference arm.")
    if state is None:
        #: no state object, no state I/O: both of the kernel's state pointers are
        #: null and the variant with none of it compiled in runs
        #: (docs/internals/chunk/chunk_kernel.md).
        y, _ = ChunkConsumer.run(routes, v)
        return y, None
    # THE HOST ORDER (design §A.4/§C.3), and it is the reason these four lines are
    # not folded into `chunk.run`: the stats pass is `BC`-free and precedes the arm,
    # its bitmap plans the arena on the arena's own side stream, and the launch's
    # `arena.wait()` is the single join. The only host synchronization in the whole
    # sequence is `plan_exact`'s one size read, which also sizes the allocation.
    atom_bits = _chunk.stats(routes) if state._wants_pages(paging) else None
    s_in, arena = state._kernel_entry(
        routes, atom_bits, paging=paging, BC=_chunk.arm(routes)[1])
    y, plane = ChunkConsumer.run(routes, v, state_in=s_in, want_state=True, arena=arena)
    state._kernel_commit(plane)
    return y, state


def _reference_arm(routes, v, decay, state, *, paging: bool):
    """The fp64 oracle, which keeps RAW TENSORS -- its signature is the spec.

    So the facade is what crosses the boundary in both directions: an incoming state
    materializes to ``[B, H, N, cols]`` fp64, and the result is stored back into the
    state object as dense contents. A paged state that comes back to the kernel arm
    afterwards re-admits from those contents' own atom occupancy.
    """
    initial = None if state is None else state._dense_entry(routes, paging=paging)
    y, final = naive_rola(
        v, routes.read, routes.write, routes.g_read, routes.g_write, routes.topology,
        decay, routes.normalization, initial_state=initial,
        output_final_state=state is not None)
    if state is None:
        return y, None
    state.adopt_dense_contents(final)
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
# The private seam
# ---------------------------------------------------------------------------


def _derive_plan(
    routes: RouteFactors,
    *,
    decay: DecayConfig = None,
    expert: Any = None,
    device: torch.device | None = None,
) -> Plan:
    """Derive this launch's geometry from launch-static facts.

    Two tiers, and the split is what keeps the derivation a pure function of recorded
    inputs rather than something that queries the machine ambiently: the model facts
    come off the routing bundle, and tier H is measured once per device.
    :func:`~rola.planner.derive.derive_bc` closes the form over them.
    """
    from rola.expert import PlanOverrides

    overrides = PlanOverrides() if expert is None else PlanOverrides.of(expert)
    device = routes.device if device is None else device
    topology = routes.topology
    # THE MODEL FIELDS COME OFF THE BUNDLE, not from a second path out of the caller:
    # one boolean computed twice by two different expressions is how the `decay` flag
    # once came to disagree with itself.
    return build_plan(
        topology,
        derive_bc(topology, _machine_facts(device), BH=routes.batch * routes.heads,
                  decay=decay is not None, pinned=overrides.state_block),
        normalization=routes.normalization, decay=decay is not None,
        # Recorded beside the geometry it bounded: the plan carries the machine it was
        # derived for rather than depending on the one it is replayed on.
        sm_shared_bytes=sm_shared_bytes())


def _execute(
    plan: Plan,
    routes: RouteFactors,
    v: torch.Tensor,
    *,
    decay: DecayConfig = None,
    initial_state: torch.Tensor | None = None,
    schedule: Schedule | None = None,
    paging=None,
    continue_paging: bool = False,
) -> tuple[torch.Tensor, Any, Schedule]:
    """Run one plan over one routing bundle.

    The schedule is assembled here, from the plan's tiling and the bundle's REALIZED
    masks, so a synthetic plan is executable on its own -- which is what gives the
    conformance families their adversarial handle. Nothing here re-decides anything:
    a plan is obeyed exactly as given, and an impossible one raises out of the
    kernel's own checks rather than being quietly repaired.
    """
    if schedule is None:
        schedule = build_schedule(
            routes.read, routes.write, routes.g_read, routes.g_write, plan,
            views=routes.views())
    return RoLAConsumer.apply(
        v, routes.read, routes.write, routes.g_read, routes.g_write, plan,
        dict(decay=decay, initial_state=initial_state, schedule=schedule, paging=paging,
             continue_paging=continue_paging))


def _machine_facts(device) -> MachineFacts:
    """Tier H: the SM count and the binary's own arm carve, measured once per device.

    Both are facts about the machine and the install, so they enter the derivation as
    an INPUT and are cached on the device they describe -- never queried from inside
    the formula, which is what keeps it replayable.
    """
    device = torch.device(device)
    facts = _MACHINE.get(device)
    if facts is None:
        facts = MachineFacts.for_device(device)
        _MACHINE[device] = facts
    return facts


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Dispatch:
    """What the op decided, and why -- so a caller can ASK instead of infer.

    ``not_kerneled`` names the cells whose registry row says nobody measured a kernel
    for them, so the call ran on the reference path as a LABELED RESEARCH SURFACE. It
    is a property of the CONFIGURATION, which is why it can be answered before the call
    and without a device.

    ``no_kernel_arm`` is the OTHER way a call lands on the reference path with nothing
    having gone wrong: the chunk consumer's own refusal for this bundle, in its own
    words, or ``None`` when it has an arm. It is the answer to "why did my call run on
    the oracle" for a topology outside the built matrix -- the same labelled-surface
    treatment the unratified activations get, for the shape axes rather than the
    activation ones. It is only ever set when the arm would otherwise have been the
    kernel, which is also the only case in which the built matrix can be read.
    """

    backend: str
    not_kerneled: tuple[tuple[int, str, Any], ...]
    no_kernel_arm: str | None = None


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


def select_backend(*operands: Any) -> str:
    """``'reference'`` or ``'cuda'`` for a call over these operands. TWO conditions.

    1. **Gradient reachability.** The consumer is forward-only; the reference is
       differentiable and is the only implemented training path. Calling it a fallback
       would be wrong in both directions: nothing failed to reach it, and there is
       nothing else to fall back from.
    2. **Device.** The consumer is a CUDA kernel. On CPU tensors the reference is the
       only implementation that exists FOR THAT DEVICE -- a device dispatch, not a
       degradation, because there is no CPU lowering to decline.

    Deliberately NOT conditions: an un-instantiated ``d_v`` or ``BC``, an unrealized
    warp tile. Those raise loudly downstream; folding them in would turn a loud
    unsupported configuration into a silent slow path.
    """
    if requires_backward(*operands):
        return "reference"
    if not all(t.is_cuda for t in _tensors(operands)):
        return "reference"
    return "cuda"


def dispatch_for(routes: RouteFactors, *operands: Any) -> Dispatch:
    """The arm the op will take for this bundle and these operands.

    THE UNSUPPORTED-TOPOLOGY RETIREMENT (2026-08-14) is decided here, in the one
    place, so the op and every caller that asks first get the same answer. The kernel
    arm is the chunk consumer, and the chunk consumer is a BUILT MATRIX: uniform
    widths, ``d_v = 64``, ``(D, B)`` in the manifest's combinations, ``T`` a multiple
    of the chunk. Everything outside it -- a jagged topology, ``d_v = 32``, ``raw`` or
    per-state normalization, an unbuilt ``(D, B)`` -- DISPATCHES to the reference arm,
    exactly as decay and an unratified activation already do, and is named in
    ``no_kernel_arm`` rather than left to be inferred. This NARROWS the shipping
    kernel envelope to the built matrix and the reference arm's throughput is the
    price; ``docs/open-work.md`` carries the card that widens the matrix.

    ``decay`` is not an argument here: a decay-configured call is routed by ``rola_op``
    before this is consulted, so asking the chunk arm about it again would be a second
    authority on the same retirement.
    """
    not_kerneled = tuple(routes.topology.unratified_activations)
    backend = "reference" if not_kerneled else select_backend(*operands)
    no_kernel_arm = None
    if backend == "cuda":
        from rola.ops import chunk as _chunk

        no_kernel_arm = _chunk.unsupported(routes, None)
        if no_kernel_arm is not None:
            backend = "reference"
    return Dispatch(backend=backend, not_kerneled=not_kerneled, no_kernel_arm=no_kernel_arm)


def _backend_for(routes: RouteFactors, *operands: Any) -> str:
    return dispatch_for(routes, *operands).backend


def _validate_call(routes, v, decay) -> None:
    from rola.routing.factors import RouteFactors as _RouteFactors

    if type(routes) is not _RouteFactors:
        raise TypeError(
            "rola_op's first argument must be a RouteFactors bundle from RouteProducer, got "
            f"{type(routes).__name__}. The routing sides are the feature-mapped (q, k) pair, "
            "and the gateway is what derives their statistics from their tensors -- there is "
            "no supported way to assemble the bundle by hand.")
    topology = routes.topology
    if not isinstance(v, torch.Tensor) or v.ndim != 4:
        raise ValueError(f"v must be [B, T, H, d_v], got {getattr(v, 'shape', type(v).__name__)}")
    expected = (routes.batch, routes.tokens, routes.heads, topology.d_v)
    if tuple(v.shape) != expected:
        raise ValueError(
            f"v must be [B, T, H, d_v]={expected} to match the routing bundle, got "
            f"{tuple(v.shape)}")
    if v.device != routes.device:
        raise ValueError(f"v is on {v.device} but the routing bundle is on {routes.device}")
    if decay is not None:
        decay.validate_against(topology)


def _tensors(obj: Any) -> list[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (tuple, list)):
        return [t for item in obj for t in _tensors(item)]
    return []


def _any_requires_grad(obj: Any) -> bool:
    if isinstance(obj, torch.Tensor):
        return obj.requires_grad
    if isinstance(obj, (tuple, list)):
        return any(_any_requires_grad(item) for item in obj)
    return False


class RoLAConsumer(torch.autograd.Function):
    """The CUDA consumer as an autograd node whose backward refuses, loudly.

    The node exists so the kernel's output is never a graph-severing leaf by accident.
    ``rola_op`` guarantees this arm is only taken when no gradient is wanted, so
    ``backward`` is unreachable through the public surface; if it is ever reached,
    something called the private path directly, and a named exception is the correct
    answer to that.
    """

    @staticmethod
    def forward(ctx, v, read, write, g_read, g_write, plan, kwargs):  # type: ignore[override]
        # Deliberately NOT `ctx.mark_non_differentiable`: marking the output would make
        # `backward` unreachable, which is the same silence this class exists to break.
        return _consumer_forward(v, read, write, g_read, g_write, plan, **kwargs)

    @staticmethod
    def backward(ctx, *grad_outputs):  # type: ignore[override]
        raise BackwardNotImplemented(
            "the RoLA CUDA consumer is forward-only: there is no native backward for it yet "
            "(docs/open-work.md M-2). Reaching this means the consumer was invoked directly on "
            "grad-requiring operands, bypassing `rola.rola_op`, which routes such calls to the "
            "differentiable reference path instead. Call `rola_op`, or run under "
            "`torch.no_grad()`.")


class ChunkConsumer(torch.autograd.Function):
    """The CHUNK consumer as an autograd node whose backward refuses, loudly.

    :class:`RoLAConsumer`'s claim, for the arm that replaced it. ``rola.ops.chunk.run``
    is a bare extension call: reached from a gradient-requiring context it returns a
    perfectly good tensor with no ``grad_fn``, and the symptom is ``x.grad is None``
    several files away (`[[layer-dispatch-autograd-gap]]`). ``rola_op``'s dispatch is
    what keeps that from happening; this node is what makes it LOUD if the dispatch is
    ever bypassed -- a first line and a last line, not one line twice.

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
            "the RoLA chunk consumer is forward-only: there is no native backward for it "
            "yet (docs/open-work.md M-2). Reaching this means the kernel arm was invoked "
            "on grad-requiring operands, bypassing `rola.rola_op`, which routes such calls "
            "to the differentiable reference path instead. Call `rola_op`, or run under "
            "`torch.no_grad()`.")
