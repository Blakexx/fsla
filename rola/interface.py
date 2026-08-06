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
    Choose the launch geometry. Everything it reads is already resident -- the
    summaries the routing bundle carries, the topology, the device's own facts -- so it
    is host-side arithmetic plus a plan assembly, not a pass over the tensors.

``_execute(plan, routes, v, ...)``
    Build the §4 schedule from the realized masks and run the kernel.

The seam is where the SELECTOR is swapped, and it is deliberately private: a caller
choosing a cost model is not describing a model, and the reduction theorem is what the
public surface is allowed to say. :mod:`rola.expert` is the quarantined door to it,
and it is also what lets a conformance test drive ``_execute`` with a SYNTHETIC plan --
a stale one, one straddling a boundary -- which is the adversarial handle a
plan-derived-then-immediately-used pipeline would not have.

**Why the boundary sits here** (`[[layer-dispatch-autograd-gap]]`): a forward-only
kernel reachable from a gradient-requiring context does not raise. It returns a
perfectly good tensor with no ``grad_fn``, the graph is severed, and the symptom is
``x.grad is None`` several files away. ``rola_op`` is the ONE place that rule is
enforced: a call that wants a gradient runs the whole operation on ``naive_rola``,
forward and backward from one graph; a call that does not reaches the CUDA consumer
wrapped in :class:`RoLAConsumer`, whose backward raises by name rather than severing
anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from rola.ops.consumer import _consumer_forward, sm_shared_bytes
from rola.ops.naive import naive_rola
from rola.planner.schedule import Plan, build_schedule, consumer_occupancy
from rola.planner.select import (
    DEFAULT_SELECTOR,
    SUPPORTED_BT,
    HardwareFacts,
    StaticPlanInputs,
    StatisticsBundle,
    feasible_bc,
    select_with,
)

if TYPE_CHECKING:
    from rola.planner.schedule import Schedule
    from rola.routing.factors import RouteFactors
    from rola.routing.types import DecayConfig

__all__ = ["BackwardNotImplemented", "RoLAConsumer", "rola_op"]

#: Tier-H facts, cached on the DEVICE they are facts about. Neither member can move
#: within a process for a fixed device, and no routing statistic lives here, so this is
#: not the per-step-selection staleness hazard a cached PLAN would be.
_HARDWARE: dict[torch.device, HardwareFacts] = {}


class BackwardNotImplemented(NotImplementedError):
    """Raised when a gradient is demanded of the forward-only CUDA consumer."""


# ---------------------------------------------------------------------------
# The public op
# ---------------------------------------------------------------------------


def rola_op(
    routes: RouteFactors,
    v: torch.Tensor,
    decay: DecayConfig = None,
    *,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    expert: Any = None,
    _arm: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run RoLA over one routing bundle and one value stream.

    ``routes``: a :class:`~rola.routing.factors.RouteFactors` from
    :class:`~rola.routing.factors.RouteProducer`. Anything else is refused -- a bundle
    the gateway did not derive is a bundle whose statistics need not describe its own
    tensors, and every trustless property of this surface rests on that being
    impossible.
    ``v``: ``[B, T, H, d_v]``.
    ``decay``: the recurrence dial, or ``None``. Build one from a decay source
    (:mod:`rola.routing.decay`), which is the slot that decides where rates come from.

    Returns ``(y, final_state)`` -- ``y: [B, T, H, d_v]`` and, when
    ``output_final_state``, ``final_state: [B, H, N, d_v + (norm == 'global')]``. The
    differentiable arm and the kernel arm return the same shapes with the same
    meaning; only the accumulation dtype differs (fp64 reference, fp32 kernel) and both
    accept either on input, so a carried state may cross arms between steps.

    ``expert``: :class:`rola.expert.PlanOverrides`, the quarantined advanced surface.
    It is a keyword argument with a default of ``None`` because the op does not need
    it: every axis it pins is one the planner selects.
    """
    _validate_call(routes, v, decay)
    operands = (v, *routes.tensors(), *(() if decay is None else tuple(decay.dials)))
    # `_arm` is how a MODULE contributes its own condition -- `RoLA` gates the kernel on
    # `self.training` as well, because reentrant checkpointing's first pass runs under
    # `no_grad` with the module still training. It is private and undocumented on
    # purpose: "training mode" has no meaning for a free function, so it is not a knob.
    if (_arm or _backend_for(routes, *operands)) == "reference":
        return naive_rola(
            v, routes.read, routes.write, routes.g_read, routes.g_write, routes.topology,
            decay, routes.normalization, initial_state=initial_state,
            output_final_state=output_final_state)
    plan = _derive_plan(routes, decay=decay, expert=expert, device=v.device)
    y, final_state, _schedule = _execute(
        plan, routes, v, decay=decay, initial_state=initial_state)
    return y, (final_state if output_final_state else None)


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
    """Choose this launch's geometry from resident facts.

    Three tiers, and the split is what keeps selection a pure function of its recorded
    inputs rather than something that queries the machine ambiently: tier 0 is fixed at
    construction (the topology, the feasible ``BC`` ladder, the declared density), tier
    H is measured once per device, and the per-step tier is the summary the routing
    bundle already carries.

    **Selection is PER STEP and never cached per shape.** The statistics it prices are
    exact over the REALIZED routing, so a cached plan prices a support distribution
    that no longer exists the moment training moves it.
    """
    from rola.expert import PlanOverrides

    overrides = PlanOverrides() if expert is None else PlanOverrides.of(expert)
    device = routes.device if device is None else device
    static = _static_inputs(routes, decay=decay, overrides=overrides)
    hardware = _hardware_facts(device)
    summaries = routes.summaries(BC=static.feasible_BC[0], BT=min(SUPPORTED_BT))
    selector = overrides.selector or DEFAULT_SELECTOR
    choice, evidence = select_with(
        selector, StatisticsBundle(static=static, hw=hardware, summaries=summaries))
    return Plan(
        # THE MODEL FIELDS COME FROM TIER 0, not from a second path out of the caller.
        # They used to be written into the plan from the layer while the planner was
        # handed its own copy, which is how one boolean came to be computed twice by
        # two different expressions.
        topology=static.topology, normalization=static.normalization,
        decay=static.decay, mma_precision=static.mma_precision,
        BT=choice.BT, BC=choice.BC,
        # The realized warp tile the consumer implements, stated rather than defaulted
        # so the driver's check has something to check.
        warp_tile=(1, max(1, static.d_v // 32)),
        # Recorded beside the decision they drove: the plan carries the machine it
        # reasoned about rather than depending on the one it is replayed on.
        sm_shared_bytes=sm_shared_bytes(),
        # A selector that WEIGHED nothing returns no evidence, and the plan must say so
        # with the not-recorded value rather than with a plausible number: the residency
        # field means "the residency this decision was reasoned at", and a fixed choice
        # was not reasoned at one.
        occupancy_blocks_per_sm=0 if evidence is None else evidence.blocks_per_sm,
        selector_name=selector.name, selector_status=selector.status)


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
    sequence_parallel=None,
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
             continue_paging=continue_paging, sequence_parallel=sequence_parallel))


def _static_inputs(routes: RouteFactors, *, decay: DecayConfig, overrides) -> StaticPlanInputs:
    """Tier 0, assembled from the bundle's own construction-time facts."""
    return _static_tier0(
        routes.topology, routes.normalization, routes.density,
        decay=decay is not None, BH=routes.batch * routes.heads, overrides=overrides)


def _static_tier0(topology, normalization, density, *, decay: bool, BH: int,
                  overrides) -> StaticPlanInputs:
    """Tier 0 from the model facts alone, for callers with no bundle in hand.

    ``feasible_BC`` is the alignment-admissible set intersected with the instantiated
    ladder; an expert pin SHRINKS that set to itself rather than bypassing the check,
    which is what makes it a pin over a selected axis and not an escape from one.
    """
    pinned = overrides.state_block
    feasible = (feasible_bc(topology.widths) if pinned is None
                else feasible_bc(topology.widths, pinned=int(pinned)))
    return StaticPlanInputs(
        topology=topology, normalization=normalization, decay=bool(decay),
        mma_precision=overrides.mma_precision, density=density,
        feasible_BC=tuple(feasible), BH=int(BH),
        pinned_BC=None if pinned is None else int(pinned))


def _hardware_facts(device) -> HardwareFacts:
    """Tier H: the driver's occupancy oracle and the SM count, measured once per device.

    The occupancy oracle is ``cudaOccupancyMaxActiveBlocksPerMultiprocessor`` on the
    real templated kernel at its real dynamic shared-memory size -- the sole authority
    for physical facts. It enters selection as an INPUT and is recorded next to the
    decision it drove.
    """
    device = torch.device(device)
    facts = _HARDWARE.get(device)
    if facts is None:
        facts = HardwareFacts(
            sm_count=torch.cuda.get_device_properties(device).multi_processor_count,
            occupancy=consumer_occupancy())
        _HARDWARE[device] = facts
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
    """

    backend: str
    not_kerneled: tuple[tuple[int, str, Any], ...]


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
    """The arm the op will take for this bundle and these operands."""
    not_kerneled = tuple(routes.topology.unratified_activations)
    backend = "reference" if not_kerneled else select_backend(*operands)
    return Dispatch(backend=backend, not_kerneled=not_kerneled)


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
