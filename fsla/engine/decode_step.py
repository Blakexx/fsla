# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The decode step's one plain function, its fast-fail, and the host-side verdict entries.

:func:`decode_step_gated` runs the steps below in order: :func:`decode_call`, :func:`scratch`, :func:`backing`,
:func:`state_plane`, :func:`build_launch`, :func:`step`. The host-visible verdict is not a step: :func:`growth_pending` reads it at the
caller's own synchronization (the layer's one-token ``forward``, or a serving loop replaying a captured step). A GROWTH STEP IS NOT A
SECOND RECIPE -- it is the same call run twice around an admission, which is what makes the replay self-correcting and the step capturable
(docs/internals/engine/decode_step.md).
"""
from __future__ import annotations

from weakref import WeakKeyDictionary

from fsla.engine.plan import DecodeLaunch, DecodeOperands, DecodeWorkspace
from fsla.engine.trace import span
from fsla.ops import decode as bodies
from fsla.ops.descriptor import Descriptor, check_workspace

#: ONE scratch per sequence: state -> {(descriptor, BH, device): DecodeScratch}, weak on the state so a dropped sequence drops its buffers and
#: two live sequences never share one.
_SCRATCH: WeakKeyDictionary = WeakKeyDictionary()


def decode_call(routes):
    """``(BH, device)`` of the step, after refusing a grouped bundle."""
    routes.refuse_grouped("the decode step")
    return routes.batch * routes.heads, routes.device


def scratch(state, descriptor, BH, device):
    """The per-sequence buffers, held WEAKLY against the state that owns them.

    Being held is what hoists the allocation and the one-time zeroing of the counters and flags above any captured region
    (docs/internals/decode/decode.md#capture).
    """
    held = _SCRATCH.setdefault(state, {})
    key = (descriptor, BH, device)
    if key not in held:
        with span("scratch"):
            held[key] = bodies.DecodeScratch.build(descriptor, BH, device)
    return held[key]


def backing(state, routes):
    """The state's own plane, table and slack pool, over the residency it already has.

    A step admits from its POOL inside the kernel and from nowhere else: the admission a verdict the pool could not cover calls for is
    :func:`admit_growth`, between two runs of the step, because that verdict is a host fact.
    """
    with span("backing"):
        return state._decode_entry(routes)


def state_plane(values, descriptor, decay, carried_plane, page_table):
    """The plane the kernel indexes, and the per-step agreement checks that find it."""
    with span("state_plane"):
        return bodies.step_plane(values.v, descriptor, decay, carried_plane, page_table)


def build_launch(descriptor, routes, values, masks, decay, state, plane, page_table, held, pool):
    """The launch's three named parts, over the bundles' OWN rows, scales, values and masks: the descriptor, the
    operands and the workspace (:class:`fsla.engine.plan.DecodeLaunch`).

    The decay arm's dials are derived here: a per-HEAD constant, the one operand the step kernel does not fold
    (docs/internals/decode/decode_fold.md). The state plane's leaf order is derived from the state's format
    (:func:`fsla.ops.decode.lattice_of`): the state does not carry it.
    """
    with span("build_launch"):
        return DecodeLaunch(
            descriptor=descriptor,
            operands=DecodeOperands(
                read=routes.amp_read, write=routes.amp_write, read_scale=routes.scale_read,
                write_scale=routes.scale_write, read_mask=masks.read, write_mask=masks.write,
                dials=bodies.leaf_rate_dials_for(decay, descriptor, values.device), v=values.v, v_scale=values.scale,
                state_plane=plane, state_lattice=bodies.lattice_of(tuple(state.format.B)),
                page_table=page_table),
            workspace=DecodeWorkspace(scratch=held, pool=pool))


def step(launch):
    """THE STEP: fold, expand, walk and the residency verdict, in one launch."""
    with span("step"):
        return bodies.step(launch)


def write_atom_bitmap(masks, descriptor):
    """The gate's INDEPENDENT second derivation of the write-atom set.

    No call invokes it: the shipped path derives this set on the device, inside the step, and a gate that compared the step against itself
    would prove nothing (docs/internals/decode/decode.md#second-derivation).
    """
    widths = bodies.decode_launch(descriptor).widths
    live = tuple(masks.bits("write", level, 1, width) for level, width in enumerate(widths))
    return bodies._write_atom_bitmap(live, widths, bodies.lattice_of(widths))


def validate_context(routes, values, masks, state, descriptor, decay, workspace) -> None:
    """THE FAST-FAIL, in ONE call: everything a decode step can be refused for here.

    It runs before the call's first step, so nothing here can refuse a step whose
    operands a launch has already consumed. The per-step SHAPE agreements are NOT here:
    they are :func:`fsla.ops.decode.step_plane`'s, checked against the plane the step
    actually indexes.
    """
    state._require_carrying()
    if routes.tokens != 1:
        raise ValueError(
            f"the decode op consumes exactly one token, got T={routes.tokens}")
    if type(descriptor) is not Descriptor:
        raise TypeError(f"the decode step takes a fsla.Descriptor, got {type(descriptor).__name__}; "
                        "fsla.ops.produce.decode_descriptor builds one from the declaration")
    bodies.check_descriptor(descriptor, routes, values.device)
    check_workspace(workspace, descriptor)
    bodies.check_decode_masks(masks, routes)
    bodies.guard_no_grad((*values.tensors(), *routes.tensors(),
                          *(decay.dials if decay is not None else ())))


def _scratch_of(state):
    """The buffers a step of `state` left behind, or `None` before its first step."""
    held = _SCRATCH.get(state)
    if not held:
        return None
    if len(held) > 1:
        raise RuntimeError(
            f"{len(held)} decode scratches are alive for one state: the launch config "
            "is FROZEN for the sequence, so a second one means the decay arm changed "
            "between two steps of the same sequence. Run the sequence with one arm.")
    return next(iter(held.values()))


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
    sequence EXACTLY ONCE across the pair -- a batch-head that ran in the first run
    carries a done flag into the second and re-reads for its `y` without depositing again
    (docs/internals/decode/decode.md#done-flags).

    ``descriptor`` is the validated context of the call (:func:`fsla.ops.produce.decode_descriptor`, or :func:`fsla.descriptor`: the device's
    facts and the call's), the planner's launch function reads it for the arm and its arguments. ``workspace`` is the call's own scratch
    (:func:`fsla.workspace_bytes`: decode needs none), checked for size and alignment when given.
    """
    validate_context(routes, values, masks, state, descriptor, decay, workspace)
    BH, device = decode_call(routes)
    held = scratch(state, descriptor, BH, device)
    carried_plane, page_table, pool = backing(state, routes)
    plane = state_plane(values, descriptor, decay, carried_plane, page_table)
    launch = build_launch(descriptor, routes, values, masks, decay, state, plane, page_table, held, pool)
    return step(launch), state


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
