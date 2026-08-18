# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The decode family's DAG, read top to bottom, its context, and its four entries.

M3 (decode-dense) and M4 (decode-paged) are ONE tuple: M3 is M4 with the growth band
masked out. A GROWTH STEP IS NOT A THIRD GRAPH -- it is this same tuple walked twice
around an admission, which is what makes the replay self-correcting and the step
capturable (docs/internals/engine/decode_dag.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rola.engine import (
    DecodePlan,
    Mask,
    node,
    rules,
    run,
    validate_capturable,
    validate_dag,
)
from rola.ops import decode as bodies

_SEEDS = ("routes", "v", "decay", "state", "expert")

#: BOTH decode masks. A node carrying only `Mask.M4` is one the dense backing has no
#: work for; there is no node the paged backing skips.
_DECODE = (Mask.M3, Mask.M4)


@dataclass(frozen=True, slots=True)
class DecodeContext:
    """One decode step's RAW inputs -- everything before a single fact is found.

    `state` is the facade's :class:`~rola.RoLAState`, which owns its own backing and
    therefore answers the one question the mask folds.
    """

    routes: Any
    v: torch.Tensor
    state: Any
    decay: Any = None
    expert: Any = None


@node(gives=("widths", "d_v", "has_decay", "BH", "device", "paged"),
      needs=("routes", "v", "decay", "state"), masks=_DECODE, capturable=True)
def decode_call(f):
    routes, state = f["routes"], f["state"]
    return (tuple(routes.topology.widths), f["v"].shape[-1], f["decay"] is not None,
            routes.batch * routes.heads, routes.device, bool(state.paged))


@node(gives=("geometry",),
      needs=("routes", "widths", "d_v", "has_decay", "BH", "device"),
      pure_of=("widths", "d_v", "has_decay", "BH", "device"), masks=_DECODE,
      capturable=True)
def geometry(f):
    """The frozen carrier. Memoized on the CONFIG half alone, which is what it is."""
    return bodies.derive_decode_geometry(
        f["routes"].topology, d_v=f["d_v"], decay=f["has_decay"], BH=f["BH"],
        device=f["device"])


@node(gives=("scratch",), needs=("state", "geometry", "BH", "device"),
      per_state=True, pure_of=("geometry", "BH", "device"), masks=_DECODE,
      capturable=True)
def scratch(f):
    """The per-sequence buffers, held WEAKLY against the state that owns them.

    The weakness is required rather than incidental: a dropped sequence must drop its
    scratch, and two concurrent sequences must never share a buffer. Being memoized is
    also what hoists the allocation and the one-time zeroing of `ctr`/`probe_ctr` above
    any captured region (docs/internals/decode/decode.md#capture).
    """
    return bodies.DecodeScratch.build(f["geometry"], f["BH"], f["device"])


@node(gives=("gate",), needs=("routes", "geometry", "state", "scratch"),
      stream="main", masks=(Mask.M4,), capturable=True)
def growth_probe(f):
    """The step's own stats pass, and the GATE is its output the step reads.

    It reads the producer's RAW write side and folds it itself, with the same device
    function the step folds with (docs/internals/decode/decode_fold.md).
    """
    routes, workspace = f["routes"], f["scratch"]
    bodies.growth_probe(routes.write, routes.heads, f["geometry"],
                        f["state"].page_table, workspace)
    return workspace.growth_any


@node(gives=("carried_plane", "page_table"), needs=("state", "routes"), stream="main",
      masks=_DECODE, capturable=True)
def backing(f):
    """The state's own plane and table, over the residency the step already has.

    A step admits NOTHING inside the walk: the admission a growth verdict calls for is
    :func:`admit_growth`, between two walks, because the verdict is a host fact.
    """
    return f["state"]._decode_entry(f["routes"])


@node(gives=("state_plane",),
      needs=("v", "geometry", "routes", "decay", "carried_plane", "page_table"),
      masks=_DECODE, capturable=True)
def state_plane(f):
    """The plane the kernel indexes, and the per-step agreement checks that find it."""
    return bodies.step_plane(f["v"], f["geometry"], f["routes"].g_read, f["decay"],
                             f["carried_plane"], f["page_table"])


@node(gives=("plan",),
      needs=("geometry", "routes", "v", "decay", "state_plane", "page_table", "scratch",
             "gate"),
      stream="main", masks=_DECODE, capturable=True)
def build_plan(f):
    """The launch's arguments, over the producer's OWN views.

    It carries a stream because the decay arm's dials are derived here: a per-HEAD
    constant, the one operand the step kernel does not fold
    (docs/internals/decode/decode_fold.md).
    """
    routes, geometry = f["routes"], f["geometry"]
    return DecodePlan(
        config=geometry, heads=routes.heads, read=routes.read, write=routes.write,
        normalize=tuple(routes.read_needs_normalization()),
        dials=bodies.leaf_rate_dials_for(f["decay"], geometry, f["v"].device),
        g_write=routes.g_write, v=f["v"], state_plane=f["state_plane"],
        page_table=f["page_table"], scratch=f["scratch"], gate=f["gate"])


@node(gives=("y",), needs=("plan",), stream="main", masks=_DECODE, capturable=True)
def step(f):
    """THE STEP: fold, expand and walk, in one launch."""
    return bodies.step(f["plan"])


@node(needs=("scratch",), stream="main", masks=(Mask.M4,))
def request_verdict(f):
    """Enqueue the verdict's read BEFORE the step it gates, and record the event.

    THE HOST ORDER, and the whole of P67-a: the step is enqueued BEHIND ITS OWN VERDICT,
    gated so a growth step is a device-side no-op, and the host's wait for the 4-byte
    flag is spent on the launches instead of in front of them
    (docs/internals/decode/decode.md#the-gate).
    """
    workspace = f["scratch"]
    workspace.growth_host.copy_(workspace.growth_any, non_blocking=True)
    workspace.verdict.record()


@node(gives=("grow",), needs=("scratch", "y"), masks=(Mask.M4,), host_sync=True)
def read_verdict(f):
    """The ONE 4-byte device-to-host read, after the launches it was enqueued ahead of."""
    workspace = f["scratch"]
    workspace.verdict.synchronize()
    return bool(workspace.growth_host[0])


@node(gives=("atom_bits",), needs=("routes", "geometry"), masks=_DECODE,
      reference_only=True)
def write_atom_bitmap(f):
    """The gate's INDEPENDENT second derivation of the write-atom set.

    Declared so that `validate_dag`'s refusal of a reference_only node has something to
    refuse: the shipped path derives this set on the device, inside the probe, and a
    gate that compared the probe against itself would prove nothing
    (docs/internals/decode/decode.md#second-derivation).
    """
    return bodies._write_atom_bitmap(f["routes"].write, f["geometry"])


#: THE CAPTURED RECIPE. `decode_step_gated` is exactly this tuple, and a CUDA graph
#: captured over it replays correctly across an admission because nothing in it reads a
#: device value on the host and nothing in it allocates.
DECODE_STEP = (
    decode_call,
    rules.decode_mask.node,
    geometry,
    scratch,
    growth_probe,
    backing,
    state_plane,
    build_plan,
    step,
)

#: The same tuple with the host-visible verdict spliced in: the copy and the event are
#: enqueued AHEAD of the step, the read follows it. Two nodes and not one, because a
#: single epilogue node would enqueue the copy behind the launches it is meant to
#: overlap.
DECODE_VERDICT = (
    DECODE_STEP[:DECODE_STEP.index(backing)]
    + (request_verdict,)
    + DECODE_STEP[DECODE_STEP.index(backing):]
    + (read_verdict,)
)

validate_dag(DECODE_STEP, _SEEDS)
validate_dag(DECODE_VERDICT, _SEEDS)
validate_capturable(DECODE_STEP, "DECODE_STEP")


def validate_context(ctx: DecodeContext) -> None:
    """THE FAST-FAIL, in ONE call: everything a decode step can be refused for here.

    It runs before the DAG's first node, so nothing here can refuse a step whose
    operands a launch has already consumed. The per-step SHAPE agreements are NOT here:
    they are :func:`rola.ops.decode.step_plane`'s, checked against the plane the step
    actually indexes.
    """
    ctx.state._require_carrying()
    if ctx.routes.tokens != 1:
        raise ValueError(
            f"the decode op consumes exactly one token, got T={ctx.routes.tokens}")
    bodies.guard_no_grad((ctx.v, ctx.routes.g_read, ctx.routes.g_write,
                          *ctx.routes.read, *ctx.routes.write,
                          *(ctx.decay.dials if ctx.decay is not None else ())))


def _facts(ctx: DecodeContext) -> dict:
    return {"routes": ctx.routes, "v": ctx.v, "decay": ctx.decay, "state": ctx.state,
            "expert": ctx.expert}


def _scratch_of(state):
    """The buffers a step of `state` left behind, or `None` before its first step."""
    table = scratch.memo.get(state)
    if not table:
        return None
    if len(table) > 1:
        raise RuntimeError(
            f"{len(table)} decode scratches are alive for one state: the launch config "
            "is FROZEN for the sequence, so a second one means the decay arm changed "
            "between two steps of the same sequence. Run the sequence with one arm.")
    return next(iter(table.values()))


def decode_step_gated(routes, v: torch.Tensor, state, *, decay=None, expert=None):
    """ONE decode step with NO host participation -- the CAPTURE-SAFE entry.

    Same kernels in the same order as :func:`decode_forward`, minus the verdict read:
    the step is gated on the probe's device-side answer, so a step whose write set is
    not resident is a NO-OP that leaves the state exactly as the admission needs it.
    The CALLER owns the verdict, at whatever synchronization it already has:

        y, state = decode_step_gated(...)   # or graph.replay()
        if growth_pending(state):
            admit_growth(state)
            y, state = decode_step_gated(...)   # or graph.replay(); now it runs

    A REPLAY IS SELF-CORRECTING because the probe re-runs inside it: after the
    admission its verdict is 0 and the same captured graph performs the step.
    docs/internals/decode/decode.md#capture
    """
    ctx = DecodeContext(routes=routes, v=v, state=state, decay=decay, expert=expert)
    validate_context(ctx)
    return run(DECODE_STEP, _facts(ctx))["y"], state


def growth_pending(state) -> bool:
    """Did the last gated step skip itself for want of an atom? ONE 4-byte read."""
    workspace = _scratch_of(state)
    if workspace is None or not state.paged:
        return False
    workspace.growth_host.copy_(workspace.growth_any)
    return bool(workspace.growth_host[0])


def admit_growth(state) -> None:
    """Admit the pending step's write atoms, from the probe's own condensation."""
    workspace = _scratch_of(state)
    if workspace is None:
        raise RuntimeError("admit_growth wants the workspace a gated step leaves behind")
    state._decode_entry_arena(workspace.atom_bits).wait()


def decode_forward(routes, v: torch.Tensor, state, *, decay=None, expert=None):
    """ONE decode step over a routing bundle and a carried state -- the whole surface.

    ``routes`` is a one-token :class:`~rola.routing.factors.RouteFactors`; ``v`` is
    ``[B, 1, H, d_v]``; ``state`` is a :class:`~rola.RoLAState` that CARRIES
    (a step continues a sequence, so an empty state has nothing to decode from).
    Returns ``(y, state)`` -- ``y`` shaped ``[B, 1, H, d_v]`` and the SAME state
    object, updated in place, exactly as :func:`rola.rola_op` returns it.

    This is :data:`DECODE_STEP` with the verdict read spliced in, and a growth step is
    that same walk again after the admission the verdict called for -- never a second
    graph (docs/internals/decode/decode.md#growth).
    """
    ctx = DecodeContext(routes=routes, v=v, state=state, decay=decay, expert=expert)
    validate_context(ctx)
    facts = run(DECODE_VERDICT, _facts(ctx))
    if not facts["grow"]:
        return facts["y"], state
    #: The gated step deposited nothing and read nothing, so the state is exactly where
    #: the admission needs it, and `scratch.atom_bits` is the probe's own condensation --
    #: the host derives no support (docs/internals/decode/decode.md#second-derivation).
    admit_growth(state)
    return run(DECODE_STEP, _facts(ctx))["y"], state
