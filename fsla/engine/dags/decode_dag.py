# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The decode family's DAG, read top to bottom, its context, and its four entries.

DECODE_DENSE and DECODE_PAGED are ONE tuple, and after the step absorbed its
own residency question they are the same tuple. The host-visible verdict is not a node:
:func:`growth_pending` reads it at the caller's own synchronization (the layer's one-token
``forward``, or a serving loop replaying a captured step). A GROWTH STEP IS NOT A THIRD
GRAPH -- it is this same tuple walked twice around an admission, which is what makes the
replay self-correcting and the step capturable (docs/internals/engine/decode_dag.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fsla.engine import (
    DecodeLaunch,
    DecodeOperands,
    DecodeWorkspace,
    Variant,
    node,
    rules,
    run,
    validate_capturable,
    validate_dag,
)
from fsla.ops import decode as bodies
from fsla.ops.descriptor import Descriptor, check_workspace

_SEEDS = ("routes", "values", "masks", "decay", "state", "descriptor")

#: BOTH decode variants. A node carrying only `Variant.DECODE_PAGED` is one the dense backing has no
#: work for; there is no node the paged backing skips.
_DECODE = (Variant.DECODE_DENSE, Variant.DECODE_PAGED)


@dataclass(frozen=True, slots=True)
class DecodeContext:
    """One decode step's RAW inputs -- everything before a single fact is found.

    `state` is the facade's :class:`~fsla.FSLAState`, which owns its own backing and
    therefore answers the one question the variant folds.
    """

    routes: Any
    values: Any
    masks: Any
    state: Any
    descriptor: Any
    decay: Any = None
    workspace: Any = None


@node(gives=("BH", "device", "paged"), needs=("routes", "state"), variants=_DECODE, capturable=True)
def decode_call(f):
    routes, state = f["routes"], f["state"]
    routes.refuse_grouped("the decode step")
    return routes.batch * routes.heads, routes.device, bool(state.paged)


@node(gives=("scratch",), needs=("state", "descriptor", "BH", "device"),
      per_state=True, pure_of=("descriptor", "BH", "device"), variants=_DECODE,
      capturable=True)
def scratch(f):
    """The per-sequence buffers, held WEAKLY against the state that owns them.

    The weakness is required rather than incidental: a dropped sequence must drop its
    scratch, and two concurrent sequences must never share a buffer. Being memoized is
    also what hoists the allocation and the one-time zeroing of the counters and flags above
    any captured region (docs/internals/decode/decode.md#capture).
    """
    return bodies.DecodeScratch.build(f["descriptor"], f["BH"], f["device"])


@node(gives=("carried_plane", "page_table", "pool"), needs=("state", "routes"),
      stream="main", variants=_DECODE, capturable=True)
def backing(f):
    """The state's own plane, table and slack pool, over the residency it already has.

    A step admits from its POOL inside the kernel and from nowhere else: the admission a
    verdict the pool could not cover calls for is :func:`admit_growth`, between two walks,
    because that verdict is a host fact.
    """
    return f["state"]._decode_entry(f["routes"])


@node(gives=("state_plane",),
      needs=("values", "descriptor", "decay", "carried_plane", "page_table"),
      variants=_DECODE, capturable=True)
def state_plane(f):
    """The plane the kernel indexes, and the per-step agreement checks that find it."""
    return bodies.step_plane(f["values"].v, f["descriptor"], f["decay"],
                             f["carried_plane"], f["page_table"])


@node(gives=("launch",),
      needs=("descriptor", "routes", "values", "masks", "decay", "state", "state_plane", "page_table", "scratch", "pool"),
      stream="main", variants=_DECODE, capturable=True)
def build_launch(f):
    """The launch's three named parts, over the bundles' OWN rows, scales, values and masks: the descriptor, the
    operands and the workspace (:class:`fsla.engine.plan.DecodeLaunch`).

    It carries a stream because the decay arm's dials are derived here: a per-HEAD
    constant, the one operand the step kernel does not fold
    (docs/internals/decode/decode_fold.md). The state plane's leaf order is derived from the state's format
    (:func:`fsla.ops.decode.lattice_of`): the state does not carry it.
    """
    routes, values, masks, descriptor = f["routes"], f["values"], f["masks"], f["descriptor"]
    return DecodeLaunch(
        descriptor=descriptor,
        operands=DecodeOperands(
            read=routes.amp_read, write=routes.amp_write, read_scale=routes.scale_read,
            write_scale=routes.scale_write, read_mask=masks.read, write_mask=masks.write,
            dials=bodies.leaf_rate_dials_for(f["decay"], descriptor, values.device), v=values.v, v_scale=values.scale,
            state_plane=f["state_plane"], state_lattice=bodies.lattice_of(tuple(f["state"].format.B)),
            page_table=f["page_table"]),
        workspace=DecodeWorkspace(scratch=f["scratch"], pool=f["pool"]))


@node(gives=("y",), needs=("launch",), stream="main", variants=_DECODE, capturable=True)
def step(f):
    """THE STEP: fold, expand, walk and the residency verdict, in one launch."""
    return bodies.step(f["launch"])


@node(gives=("atom_bits",), needs=("masks", "descriptor"), variants=_DECODE,
      reference_only=True)
def write_atom_bitmap(f):
    """The gate's INDEPENDENT second derivation of the write-atom set.

    Declared so that `validate_dag`'s refusal of a reference_only node has something to
    refuse: the shipped path derives this set on the device, inside the step, and a gate
    that compared the step against itself would prove nothing
    (docs/internals/decode/decode.md#second-derivation).
    """
    masks = f["masks"]
    widths = bodies.decode_launch(f["descriptor"]).widths
    live = tuple(masks.bits("write", level, 1, width) for level, width in enumerate(widths))
    return bodies._write_atom_bitmap(live, widths, bodies.lattice_of(widths))


#: THE CAPTURED RECIPE. `decode_step_gated` is exactly this tuple, and a CUDA graph
#: captured over it replays correctly across an admission because nothing in it reads a
#: device value on the host and nothing in it allocates.
DECODE_STEP = (
    decode_call,
    rules.decode_variant.node,
    scratch,
    backing,
    state_plane,
    build_launch,
    step,
)

#: The same tuple with the host-visible verdict read appended. ONE node, and it is the
#: only place a decode step touches the host at all.
validate_dag(DECODE_STEP, _SEEDS)
validate_capturable(DECODE_STEP, "DECODE_STEP")


def validate_context(ctx: DecodeContext) -> None:
    """THE FAST-FAIL, in ONE call: everything a decode step can be refused for here.

    It runs before the DAG's first node, so nothing here can refuse a step whose
    operands a launch has already consumed. The per-step SHAPE agreements are NOT here:
    they are :func:`fsla.ops.decode.step_plane`'s, checked against the plane the step
    actually indexes.
    """
    ctx.state._require_carrying()
    if ctx.routes.tokens != 1:
        raise ValueError(
            f"the decode op consumes exactly one token, got T={ctx.routes.tokens}")
    if type(ctx.descriptor) is not Descriptor:
        raise TypeError(f"the decode step takes a fsla.Descriptor, got {type(ctx.descriptor).__name__}; "
                        "fsla.ops.produce.decode_descriptor builds one from the declaration")
    launch = bodies.decode_launch(ctx.descriptor)
    widths = tuple(ctx.routes.topology.widths)
    if widths != launch.widths:
        raise ValueError(f"the descriptor is for widths {launch.widths}, the bundle's are {widths}")
    if ctx.descriptor.device != ctx.values.device.index:
        raise ValueError(f"the descriptor was made for device {ctx.descriptor.device}; the values are on {ctx.values.device}")
    check_workspace(ctx.workspace, ctx.descriptor)
    bodies.check_decode_masks(ctx.masks, ctx.routes)
    bodies.guard_no_grad((*ctx.values.tensors(), *ctx.routes.tensors(),
                          *(ctx.decay.dials if ctx.decay is not None else ())))


def _facts(ctx: DecodeContext) -> dict:
    return {"routes": ctx.routes, "values": ctx.values, "masks": ctx.masks, "decay": ctx.decay, "state": ctx.state,
            "descriptor": ctx.descriptor}


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


def decode_step_gated(routes, values, masks, state, descriptor, *, workspace=None, decay=None):  #: reserved: decode needs no per-call scratch (fsla.workspace_bytes is 0); the op's planner interface, checked when given
    """ONE decode step with NO host participation -- the decode entry, and CAPTURE-SAFE.

    Each batch-head is gated on the step's own device-side answer, so one whose write set
    it can neither address nor admit is a NO-OP that leaves the state exactly as the
    admission needs it. The CALLER owns the verdict, at whatever synchronization it
    already has (the layer's one-token ``forward`` does exactly this):

        y, state = decode_step_gated(...)   # or graph.replay()
        if growth_pending(state):
            admit_growth(state)
            y, state = decode_step_gated(...)   # or graph.replay(); now it runs

    A REPLAY IS SELF-CORRECTING because the verdict is re-derived inside it: after the
    admission it is 0 and the same captured graph performs the step. It advances every
    sequence EXACTLY ONCE across the pair -- a batch-head that ran in the first walk
    carries a done flag into the second and re-reads for its `y` without depositing again
    (docs/internals/decode/decode.md#done-flags).

    ``descriptor`` is the validated context of the call (:func:`fsla.ops.produce.decode_descriptor`, or :func:`fsla.descriptor`: the device's
    facts and the call's), the planner's launch function reads it for the arm and its arguments. ``workspace`` is the call's own scratch
    (:func:`fsla.workspace_bytes`: decode needs none), checked for size and alignment when given.
    """
    ctx = DecodeContext(routes=routes, values=values, masks=masks, state=state, descriptor=descriptor,
                        decay=decay, workspace=workspace)
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
    """Admit the pending step's write atoms, from the step's own condensation."""
    workspace = _scratch_of(state)
    if workspace is None:
        raise RuntimeError("admit_growth wants the workspace a gated step leaves behind")
    state._decode_entry_arena(workspace.atom_bits).wait()


def pool_headroom(state) -> int:
    """Slack slots the thinnest batch-head still has. A HOST READ, AT LEISURE.

    Pool-covered growth raises no verdict -- the step admitted itself and there is nothing
    urgent for the host -- so this is the level a serving loop polls to decide when to
    :func:`refill_pool`, on its own schedule and off the step's path.
    """
    return 0 if state._arena is None else state._arena.pool_headroom()


def refill_pool(state) -> int:
    """Replace the slack slots the device claimed, and return how many. HOST, OFF PATH."""
    if state._arena is None:
        raise RuntimeError("a pool refill wants a paged state; this one is dense")
    return state._arena.refill_pool()
