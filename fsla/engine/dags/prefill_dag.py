# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The prefill family's DAG, read top to bottom, and the context it runs over.

ONE SEGMENT in, its readout out: the carry and the intra over the segment's masks, then the post pass
``y = num / (den + eps / r)``. The stateless and stateful bands are one tuple. Segmentation is the LAYER's
(docs/internals/engine/prefill_dag.md): this DAG runs whatever segment it is handed, against whatever state it is handed, over the descriptor
and the workspace its caller built (:func:`fsla.ops.prefill.build_workspace`): it derives nothing of its own.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from fsla.engine import (
    SUPPLIABLE,
    PrefillLaunch,
    PrefillOperands,
    PrefillWorkspace,
    node,
    rules,
    run,
    validate_dag,
)
from fsla.engine.facts import nodes
from fsla.engine.facts.call import require_envelope
from fsla.routing.masks import PREFILL, Masks

_SEEDS = ("routes", "values", "masks", "decay", "state", "descriptor", "workspace", "want_state", "given_state_in", "given_arena")


@dataclass(frozen=True, slots=True)
class PrefillContext:
    """One segment's RAW inputs -- everything before a single fact is found.

    `state` is the facade's :class:`~fsla.FSLAState`, which owns its own residency;
    `state_in` and `arena` are the same two facts handed in directly, which is how a
    caller drives the kernels without the state object. `descriptor` is the call's validated
    :class:`~fsla.ops.descriptor.Descriptor` and `workspace` the buffer :func:`fsla.ops.prefill.build_workspace` wrote for it.
    """

    routes: Any
    values: Any
    masks: Any
    descriptor: Any
    workspace: Any = None
    decay: Any = None
    state: Any = None
    state_in: torch.Tensor | None = None
    arena: Any = None
    want_state: bool = False
    supplied: Mapping[str, Any] = field(default_factory=dict)


@node(gives=("num", "den"),
      needs=("descriptor", "workspace", "routes", "values", "masks", "decay", "given_state_in", "page_table", "state_out"),
      stream="main")
def launch(f):
    """The carry at its bytes and the intra into the same planes: ``num [BH, T, d_v]`` and
    ``den [BH, T]``, fp32, each over the reader factor (`fsla.ops.prefill.prefill`), over the launch's three named
    parts -- the descriptor, the operands and the workspace (:class:`fsla.engine.plan.PrefillLaunch`)."""
    from fsla.ops import prefill as prefill_ops
    from fsla.ops.descriptor import check_workspace

    check_workspace(f["workspace"], f["descriptor"])
    bundle = PrefillLaunch(
        descriptor=f["descriptor"],
        operands=PrefillOperands(routes=f["routes"], values=f["values"], masks=f["masks"], decay=f["decay"],
                                 state_in=f["given_state_in"], page_table=f["page_table"], state_out=f["state_out"]),
        workspace=PrefillWorkspace(buffer=f["workspace"]))
    return prefill_ops.prefill(bundle)


@node(gives=("y",), needs=("num", "den", "routes"), stream="main")
def post(f):
    """THE POST PASS: ``y = num / (den + eps)``, ``[B, T, H, d_v]`` fp32 (`fsla.ops.prefill.readout`)."""
    from fsla.ops import prefill as prefill_ops

    return prefill_ops.readout(f["num"], f["den"], f["routes"])


PREFILL_DAG = (
    rules.paging_preference.node,
    nodes.topology,
    nodes.classify_call,
    rules.variant.node,
    nodes.backing,
    launch,
    post,
)

validate_dag(PREFILL_DAG, _SEEDS)


def validate_context(ctx: PrefillContext) -> None:
    """THE FAST-FAIL, in ONE call: everything a prefill call can be refused for.

    It runs before the DAG's first node, so nothing here can refuse a call whose
    operands a launch has already consumed.
    """
    from fsla._state import FSLAState
    from fsla.ops.prefill import check_descriptor
    from fsla.routing.factors import RouteFactors
    from fsla.values import Values

    illegal = sorted(set(ctx.supplied) - SUPPLIABLE)
    if illegal:
        raise ValueError(
            f"a PrefillContext may not supply {illegal}: the suppliable set is closed at "
            f"{sorted(SUPPLIABLE)}. Everything below the amplitude boundary is one "
            f"canonical implementation, gated once, and substituting it would fork "
            f"fact production.")
    routes, values, decay = ctx.routes, ctx.values, ctx.decay
    if type(routes) is not RouteFactors:
        raise TypeError(
            "fsla_op's first argument must be a RouteFactors bundle, got "
            f"{type(routes).__name__}. The routing sides are the feature-mapped (q, k) pair; "
            "a layer builds one with its producer, and RouteFactors.from_levels builds one from any levels -- "
            "its invariants are checked either way.")
    if type(values) is not Values:
        raise TypeError(
            "fsla_op's second argument must be a Values bundle, got "
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
            "fsla_op's `state` argument is a fsla.FSLAState from fsla.state(), got "
            f"{type(ctx.state).__name__}. A raw tensor is not one: the state owns its own "
            "backing (dense or paged) and the shape it is bound to.")
    check_descriptor(ctx.descriptor, routes, values, decay=decay, stateful=ctx.want_state or ctx.state is not None)
    if type(ctx.masks) is not Masks:
        raise TypeError(f"fsla_op's `masks` is a fsla.routing.masks.Masks, got {type(ctx.masks).__name__}; the producer "
                        "emits it beside the bundle (docs/api.md section 2.6)")
    if ctx.masks.consumer != PREFILL:
        raise ValueError(f"fsla_op takes the {PREFILL!r}-layout masks, got {ctx.masks.consumer!r}")
    ctx.masks.check(routes)
    require_envelope(routes, values, decay, ctx.state)


def run_prefill(ctx: PrefillContext) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Walk the prefill DAG over a VALIDATED `ctx` (:func:`validate_context`, which the caller runs in its own
    grad mode): ``(y, state_out)``, the plane the call stored into or ``None``.

    The intra reads the bundle; the carry leg's floating operands are `fsla.ops.prefill`'s,
    dequantized there once a call until the carry reads the rows.
    """
    facts = {"routes": ctx.routes, "values": ctx.values, "masks": ctx.masks, "decay": ctx.decay, "state": ctx.state,
             "descriptor": ctx.descriptor, "workspace": ctx.workspace, "want_state": ctx.want_state,
             "given_state_in": ctx.state_in, "given_arena": ctx.arena}
    facts.update(ctx.supplied)
    done = run(PREFILL_DAG, facts)
    return done["y"], done["state_out"]
