# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The public functional op, and the one autograd boundary in the package.

Three layers, in the shape flash-attention's `flash_attn_interface.py` uses
(extraction design §4.3)::

    rola/ops/consumer.py :: _consumer_forward(...)   private, forward-only CUDA
    rola/ops/naive.py    :: naive_rola(...)          the differentiable reference
    rola/interface.py    :: RoLAConsumer(autograd.Function)
    rola/interface.py    :: rola_func(...)           PUBLIC, autograd-aware
    rola/interface.py    :: rola_forward(...)        PUBLIC, no-grad / inference

WHY THIS EXISTS, and it is not tidiness. `[[layer-dispatch-autograd-gap]]`: a
forward-only kernel reachable from a gradient-requiring context does not raise. It
returns a perfectly good tensor with no `grad_fn`, the graph is severed at that
call, and the symptom is `x.grad is None` several files away. That is how the
fleet-smoke `L=64` failure happened, and the class of bug is structural, not a bug
in any one call site: every caller that reaches the kernel directly has to remember
the rule, and eventually one does not.

`rola_func` is the place the rule is enforced ONCE. It asks, of the tensors the
OPERATION actually consumes, whether a gradient is wanted; and:

* **gradient wanted** -> the whole operation runs on `naive_rola`, forward and
  backward from ONE graph. This is the documented training path until a native
  backward lands (`docs/open-work.md` M-2).
* **no gradient wanted** -> the CUDA consumer, wrapped in `RoLAConsumer`.

The dispatch is EXPLICIT -- `requires_backward()` is a public predicate a caller can
ask directly, and `rola_func` returns which arm it took -- rather than implicit in
whether some tensor happened to have a `grad_fn`. `tests/test_interface.py` and
`tests/layers/test_layer_dispatch.py` gate it from both sides.

**Why backward is not "recompute on the reference".** An autograd `Function` whose
forward is the kernel and whose backward differentiates the oracle would return a
value from one implementation and a gradient from another, agreeing only to `RTOL`.
`rola/layers/rola.py` already rejects exactly that arrangement, in prose, as the
reason training mode gates the CUDA path independently of `torch.is_grad_enabled()`
(the reentrant-checkpointing case). Dispatching the WHOLE operation keeps forward
and backward on one implementation. `RoLAConsumer.backward` therefore exists only as
a tripwire: it raises, by name, and it is reachable only if something bypassed
`rola_func`. It never silently severs the graph, which is the failure being closed.

**`torch.library` registration and `_fake` meta kernels are NOT here yet** -- see
`docs/open-work.md` W2b for why and what it would take. The op's signature carries
structured Python objects (`Plan`, `Topology`, `Schedule`), which no `torch.library`
schema can express; registering it needs a flattening layer that does not exist, and
inventing one inside this unit would be a 40-argument schema written blind.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from rola.ops.constants import READOUT_EPS
from rola.ops.consumer import _consumer_forward
from rola.ops.naive import naive_rola

if TYPE_CHECKING:
    from rola.planner.schedule import Plan, Schedule
    from rola.routing.types import DecayConfig

__all__ = [
    "Dispatch",
    "RoLAConsumer",
    "dispatch_for",
    "requires_backward",
    "rola_forward",
    "rola_func",
    "select_backend",
]


class BackwardNotImplemented(NotImplementedError):
    """Raised when a gradient is demanded of the forward-only CUDA consumer."""


def requires_backward(*tensors: Any) -> bool:
    """Whether a call over these operands must produce a differentiable result.

    The predicate the whole dispatch turns on, exposed so that a caller can ask the
    same question `rola_func` asks instead of reimplementing it slightly differently.

    It is evaluated on the tensors the OPERATION consumes -- `v`, the producer's
    per-level routing allocations, the side gains, and the decay dials -- and NOT on
    a module's parameters. A frozen-parameter forward under `torch.no_grad()` has no
    graph to sever and correctly takes the kernel; a dial-only unfrozen finetune has
    exactly one grad-requiring operand and must not (`wiring-review.md` §2: that case
    once reached the CUDA consumer and silently produced `decay_theta.grad = None`).

    Nested tuples and lists are walked, because the routing allocations arrive as
    per-level tuples and a predicate that only looked at the top level would miss
    every one of them.
    """
    if not torch.is_grad_enabled():
        return False
    return any(_any_requires_grad(t) for t in tensors)


def select_backend(*operands: Any) -> str:
    """`'reference'` or `'cuda'` for a call over these operands. TWO conditions.

    1. **Gradient reachability** (`requires_backward`). The consumer is forward-only;
       the reference is differentiable and is the only implemented training path.
       Calling it a fallback would be wrong in both directions: nothing failed to
       reach it, and there is nothing else to fall back from.
    2. **Device.** The consumer is a CUDA kernel. On CPU tensors the reference is the
       only implementation that exists *for that device* -- a device dispatch, not a
       degradation, because there is no CPU lowering to decline.

    Deliberately NOT conditions: an un-instantiated `d_v` or `BC`, an unrealized warp
    tile, an inadmissible decay panel. Those raise loudly downstream. Folding them in
    would turn a loud unsupported configuration into a silent slow path, which is
    the pattern `[[kernel-engineering-rules]]` forbids.

    `RoLA.forward` adds a THIRD condition on top of these -- `self.training` -- which
    is a property of a module and has no meaning for a free function, so it lives on
    the layer. See `RoLA.select_execution_backend`.
    """
    if requires_backward(*operands):
        return "reference"
    if not all(t.is_cuda for t in _tensors(operands)):
        return "reference"
    return "cuda"


@dataclass(frozen=True)
class Dispatch:
    """What `rola_func` decided, and why -- so a caller can ASK instead of infer.

    `not_kerneled` is §4.6's explicit flag: the configuration names an activation
    whose registry row says `kerneled=False`, so it ran on the reference path as a
    LABELED RESEARCH SURFACE. It is a property of the configuration, which is why it
    can be answered before the call and without a device.
    """

    backend: str
    not_kerneled: tuple[tuple[int, str, Any], ...]


def dispatch_for(plan: Plan, *operands: Any) -> Dispatch:
    """The arm `rola_func` will take for this plan and these operands."""
    not_kerneled = tuple(plan.topology.unratified_activations)
    backend = "reference" if not_kerneled else select_backend(*operands)
    return Dispatch(backend=backend, not_kerneled=not_kerneled)


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

    The node exists so that the kernel's output is never a graph-severing leaf by
    accident. `rola_func` guarantees this arm is only taken when no gradient is
    wanted, so `backward` is unreachable through the public API; if it is ever
    reached, something called the private path directly, and a named exception is
    the correct answer to that.
    """

    @staticmethod
    def forward(ctx, v, read_levels, write_levels, g_read, g_write, plan, kwargs):  # type: ignore[override]
        # Deliberately NOT `ctx.mark_non_differentiable`: marking the output would
        # make `backward` unreachable, which is the same silence this class exists
        # to break. The output is non-differentiable in practice because this arm is
        # only taken when nothing upstream wants a gradient; if that ever stops
        # being true, the tripwire below must be what fires.
        return _consumer_forward(
            v, read_levels, write_levels, g_read, g_write, plan, **kwargs)

    @staticmethod
    def backward(ctx, *grad_outputs):  # type: ignore[override]
        raise BackwardNotImplemented(
            "the RoLA CUDA consumer is forward-only: there is no native backward for it "
            "yet (docs/open-work.md M-2). Reaching this means the consumer was invoked "
            "directly on grad-requiring operands, bypassing `rola.rola_func`, which "
            "routes such calls to the differentiable reference path instead. Call "
            "`rola_func`, or run under `torch.no_grad()`.")


def rola_func(
    v: torch.Tensor,
    read_levels: tuple[torch.Tensor, ...],
    write_levels: tuple[torch.Tensor, ...],
    g_read: torch.Tensor | None,
    g_write: torch.Tensor,
    plan: Plan,
    *,
    decay: DecayConfig = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    write_levels_stored: tuple[torch.Tensor, ...] | None = None,
    eps: float = READOUT_EPS,
    schedule: Schedule | None = None,
    **consumer_kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run RoLA, differentiably when a gradient is wanted and on the kernel when not.

    `v` is `[B, T, H, d_v]`; `read_levels` / `write_levels` are one
    `[B, T, H, width_l]` simplex per routing level; `g_read` / `g_write` are the
    per-side gains (`g_read` must be `None` under `plan.normalization == "global"`,
    which fixes it to 1). `plan` names the topology and the launch geometry --
    build one with `rola.select_plan` or `rola.plan_and_schedule`.

    Returns `(y, final_state)`, `y: [B, T, H, d_v]`, `final_state:
    [B, H, N, d_v + (norm == "global")]` when `output_final_state`, else `None`.
    Both arms return the same shapes with the same meaning; only the DTYPE differs,
    and it differs for both returned tensors together: the reference arm is the fp64
    oracle and hands back fp64 `y` and fp64 state whatever it was given, the kernel
    arm accumulates in fp32 and hands back the operand dtype. Both accept either on
    input, so a carried state may cross arms between steps. Cast the output if your
    downstream needs a particular dtype -- neither arm guesses, because a reference
    that silently rounded its own result to an operand's precision would make an
    accuracy gate compare against a rounded reference.

    `consumer_kwargs` (`paging`, `continue_paging`, `sequence_parallel`, ...) are
    kernel-arm options. Passing one on a call that dispatches to the reference arm
    is a `ValueError` rather than a silent no-op -- a flag honoured on one arm and
    ignored on the other is `[[layer-dispatch-autograd-gap]]`'s shape of failure.

    `.last_backend` is not a thing on a function; ask `requires_backward(...)` with
    the same operands if you need to know which arm a call will take.
    """
    dial_operands = () if decay is None else tuple(decay.dials)
    operands = (v, read_levels, write_levels, g_read, g_write, dial_operands)
    # §4.6's third dispatch condition, and the only one that is a property of the
    # CONFIG rather than of the operands: an activation whose registry row says
    # `kerneled=False` has no measured kernel, so the call runs on the reference path
    # carrying `not_kerneled`. A labeled research surface, not a kernel fallback --
    # `launch_consumer` is never asked and never declines.
    dispatch = dispatch_for(plan, *operands)
    if dispatch.backend == "reference":
        if consumer_kwargs:
            raise ValueError(
                f"{sorted(consumer_kwargs)} are CUDA-consumer options, and this call "
                "dispatched to the differentiable reference path, which does not "
                "implement them. Silently ignoring them would make a flag mean one "
                "thing on one arm and nothing on the other; put the operands on a CUDA "
                "device under `torch.no_grad()` to reach the consumer, or drop the "
                "options.")
        return naive_rola(
            v, read_levels, write_levels, g_read, g_write, plan.topology, decay,
            plan.normalization, write_levels_stored=write_levels_stored, eps=eps,
            initial_state=initial_state, output_final_state=output_final_state)

    y, final_state, _schedule = RoLAConsumer.apply(
        v, read_levels, write_levels, g_read, g_write, plan,
        dict(decay=decay, initial_state=initial_state,
             write_levels_stored=write_levels_stored, eps=eps, schedule=schedule,
             **consumer_kwargs))
    return y, (final_state if output_final_state else None)


def rola_forward(
    v: torch.Tensor,
    read_levels: tuple[torch.Tensor, ...],
    write_levels: tuple[torch.Tensor, ...],
    g_read: torch.Tensor | None,
    g_write: torch.Tensor,
    plan: Plan,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The inference entry point: same signature, and no gradient, ever.

    A caller that means "inference" says so here rather than by remembering a
    `torch.no_grad()` somewhere up the stack. The difference from `rola_func` is
    that this one cannot be wrong: it establishes the no-grad context itself, so it
    always reaches the kernel arm and its result never carries a graph.
    """
    with torch.no_grad():
        return rola_func(v, read_levels, write_levels, g_read, g_write, plan, **kwargs)
