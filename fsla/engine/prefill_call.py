# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The prefill call's one plain function, its fast-fail, and the context it runs over.

ONE SEGMENT in, its readout out: the carry and the intra over the segment's masks, then the post pass
``y = num / (den + eps / r)``. The stateless and stateful calls are one function. Segmentation is the LAYER's
(docs/internals/engine/prefill_call.md): this call runs whatever segment it is handed, against whatever state it is handed. The call carries only its
inputs: :func:`descriptor_of` builds its descriptor from them and :func:`run_prefill` builds the workspace that passes between its steps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from fsla.engine.facts import planes
from fsla.engine.facts.call import refuse_backward
from fsla.engine.plan import PrefillLaunch, PrefillOperands, PrefillWorkspace
from fsla.engine.trace import span
from fsla.routing.masks import PREFILL, Masks


@dataclass(frozen=True, slots=True)
class PrefillContext:
    """One segment's RAW inputs -- everything before a single fact is found.

    `state` is the facade's :class:`~fsla.FSLAState`, which owns its own residency;
    `state_in` and `arena` are the same two facts handed in directly, which is how a
    caller drives the kernels without the state object.
    """

    routes: Any
    values: Any
    masks: Any
    density: Any
    decay: Any = None
    state: Any = None
    state_in: torch.Tensor | None = None
    arena: Any = None
    want_state: bool = False


def backing(arena, state_in, want_state, routes, d_v):
    """The two state planes a launch is given -- and for a CONTINUATION they are ONE.

    A FRESH DENSE PLANE IS ZEROED, and that is the price of the store
    predicate: the exit sweep writes the atoms this call WRITES and no others, so the
    atoms it skips must already hold the zero an absent paged atom reads as. It is one
    fill per SEQUENCE on the expert-override backing, never per call.

    A continued sequence advances its state IN PLACE, in either backing: the arena has
    always handed the same plane in and out, and the dense one does now
    (docs/internals/state.md#the-dense-continuation). Only a FRESH dense bind allocates,
    and that plane is uninitialized because it is written whole.
    """
    with span("backing"):
        if arena is not None:
            return arena.page_table, (arena.state if want_state else None)
        if not want_state:
            return None, None
        return None, (planes.state_plane(routes, d_v=d_v, zeros=True) if state_in is None else state_in)


def launch(descriptor, workspace, routes, values, masks, density, decay, state_in, page_table, state_out):
    """The carry at its bytes and the intra into the same planes: ``num [BH, T, d_v]`` and
    ``den [BH, T]``, fp32, each over the reader factor (`fsla.ops.prefill.prefill`), over the launch's three named
    parts -- the descriptor, the operands and the workspace (:class:`fsla.engine.plan.PrefillLaunch`)."""
    from fsla.ops import prefill as prefill_ops

    with span("launch"):
        bundle = PrefillLaunch(
            descriptor=descriptor,
            operands=PrefillOperands(routes=routes, values=values, masks=masks, density=density, decay=decay,
                                     state_in=state_in, page_table=page_table, state_out=state_out),
            workspace=PrefillWorkspace(buffer=workspace))
        return prefill_ops.prefill(bundle)


def post(num, den, routes):
    """THE POST PASS: ``y = num / (den + eps)``, ``[B, T, H, d_v]`` fp32 (`fsla.ops.prefill.readout`)."""
    from fsla.ops import prefill as prefill_ops

    with span("post"):
        return prefill_ops.readout(num, den, routes)


def descriptor_of(ctx: PrefillContext):
    """THE CALL'S DESCRIPTOR, from what the call is handed: the declaration and the bundles' extents, whether a decay is handed, whether a state is
    handed, and the operands' device. Memoized by the call and the device (:func:`fsla.ops.produce._prefill_descriptor`). The device refusal is
    `fsla.ops._ext.device_index`'s and the unbuilt-kernel refusal the owed descriptor's, both when the descriptor is made."""
    from fsla.ops._ext import device_index
    from fsla.ops.produce import _prefill_descriptor

    routes = ctx.routes
    return _prefill_descriptor(routes.topology, ctx.values.d_v, ctx.decay is not None, routes.tokens, routes.batch * routes.heads,
                               ctx.want_state or ctx.state is not None, device_index(routes.device))


def validate_context(ctx: PrefillContext) -> None:
    """THE FAST-FAIL, in ONE call: the operand-type, mask, density and gradient refusals of a prefill call (the descriptor's refusals -- the device's and
    the unbuilt kernel's -- are :func:`descriptor_of`'s, which the call runs after this).

    It runs before the call's first step, so nothing here can refuse a call whose
    operands a launch has already consumed.
    """
    from fsla._state import FSLAState
    from fsla.routing.factors import RouteFactors
    from fsla.values import Values

    routes, values, decay = ctx.routes, ctx.values, ctx.decay
    if type(routes) is not RouteFactors:
        raise TypeError(
            "prefill's first argument must be a RouteFactors bundle, got "
            f"{type(routes).__name__}. The routing sides are the feature-mapped (q, k) pair; "
            "a layer builds one with its producer, and RouteFactors.from_levels builds one from any levels -- "
            "its invariants are checked either way.")
    if type(values) is not Values:
        raise TypeError(
            "prefill's second argument must be a Values bundle, got "
            f"{type(values).__name__}. The value stream reaches the op in the internal format; "
            "build one with Values.from_projection(v).")
    #: THE LAST DIM IS NOT CHECKED, IT IS READ. `d_v` is the value stream's own width
    #: and the routing carries no value geometry, so there is nothing to agree with;
    #: what must match the bundle is the token addressing, `[B, T, H_w]`, a write head's.
    expected = (routes.batch, routes.tokens, routes.write_heads)
    if (values.batch, values.tokens, values.heads) != expected:
        raise ValueError(
            f"values must be [B, T, H_w, d_v] with [B, T, H_w]={expected} (a write head's) to match the routing "
            f"bundle, got {tuple(values.v.shape)}")
    if values.device != routes.device:
        raise ValueError(f"values are on {values.device} but the routing bundle is on {routes.device}")
    if decay is not None:
        decay.validate_against(routes.topology)
    if ctx.state is not None and type(ctx.state) is not FSLAState:
        raise TypeError(
            "prefill's `state` argument is a fsla.FSLAState from fsla.state(), got "
            f"{type(ctx.state).__name__}. A raw tensor is not one: the state owns its own "
            "backing (dense or paged) and the shape it is bound to.")
    if type(ctx.masks) is not Masks:
        raise TypeError(f"prefill's `masks` is a fsla.routing.masks.Masks, got {type(ctx.masks).__name__}; the producer "
                        "emits it beside the bundle (docs/api.md section 2.6)")
    if ctx.masks.consumer != PREFILL:
        raise ValueError(f"prefill takes the {PREFILL!r}-layout masks, got {ctx.masks.consumer!r}")
    ctx.masks.check(routes)
    _check_density(ctx.density, routes)
    refuse_backward(routes, values, decay)


def _check_density(density, routes) -> None:
    """The density's type, shape, floating dtype and device against the rows, and nothing about its values: a caller may pass any."""
    D = routes.topology.D
    if not isinstance(density, torch.Tensor):
        raise TypeError(f"prefill's `density` is a floating torch.Tensor [2, D], got {type(density).__name__}; build it with "
                        "fsla.masks_and_density or fsla.density_of_masks (docs/api.md section 2.6)")
    if tuple(density.shape) != (2, D):
        raise ValueError(f"prefill's `density` is [2, D] with D={D} (row 0 the read side, row 1 the write side, an entry a level), "
                         f"got {tuple(density.shape)}")
    if not density.is_floating_point():
        raise ValueError(f"prefill's `density` is a floating tensor, got {density.dtype}")
    if density.device != routes.device:
        raise ValueError(f"prefill's `density` is on {density.device} but the rows are on {routes.device}")


def run_prefill(ctx: PrefillContext, descriptor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the prefill over a VALIDATED `ctx` and its `descriptor` (:func:`validate_context`, which the caller runs in its own
    grad mode): ``(y, state_out)``, the plane the call stored into or ``None``.

    The intra reads the bundle; the carry leg's floating operands are `fsla.ops.prefill`'s,
    dequantized there once a call until the carry reads the rows.
    """
    from fsla.ops import prefill as prefill_ops

    page_table, state_out = backing(ctx.arena, ctx.state_in, ctx.want_state, ctx.routes, ctx.values.d_v)
    workspace = prefill_ops._build_workspace(descriptor)
    num, den = launch(descriptor, workspace, ctx.routes, ctx.values, ctx.masks, ctx.density, ctx.decay, ctx.state_in,
                      page_table, state_out)
    return post(num, den, ctx.routes), state_out
