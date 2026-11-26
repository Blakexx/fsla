# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The decode step's one plain function and its fast-fail.

:func:`decode_step_gated` runs the steps below in order: :func:`decode_call`, :func:`descriptor_of`, :func:`scratch`, :func:`backing`,
:func:`state_plane`, :func:`build_launch`, :func:`step`. It has no host participation, which is what makes it capturable in a CUDA
graph (docs/internals/engine/decode_step.md).
"""
from __future__ import annotations

from weakref import WeakKeyDictionary

from fsla.engine.plan import DecodeLaunch, DecodeOperands, DecodeWorkspace
from fsla.engine.trace import span
from fsla.ops import decode as bodies
from fsla.ops._ext import device_index
from fsla.ops.produce import _decode_descriptor

#: ONE scratch per sequence: state -> {(descriptor, BH, device): DecodeScratch}, weak on the state so a dropped sequence drops its buffers and
#: two live sequences never share one.
_SCRATCH: WeakKeyDictionary = WeakKeyDictionary()


def decode_call(routes):
    """``(BH, device)`` of the step, after refusing a grouped bundle."""
    routes.refuse_grouped("the decode step")
    return routes.batch * routes.heads, routes.device


def descriptor_of(routes, values, decay, BH, device):
    """THE STEP'S DESCRIPTOR, from what the step is handed: the declaration, the value width, whether a decay is handed, ``BH`` and the device.
    Memoized by the call and the device (:func:`fsla.ops.produce._decode_descriptor`): after a call's first, a host dictionary lookup, no device read and
    no allocation, which keeps the step capturable."""
    return _decode_descriptor(routes.topology, values.d_v, decay is not None, BH, device_index(device))


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
    """The state's own plane, which the kernel updates in place."""
    with span("backing"):
        return state._decode_entry(routes)


def state_plane(values, descriptor, decay, carried_plane):
    """The plane the kernel indexes, and the per-step agreement checks that find it."""
    with span("state_plane"):
        return bodies.step_plane(values.v, descriptor, decay, carried_plane)


def build_launch(descriptor, routes, values, masks, decay, plane, held):
    """The launch's three named parts, over the bundles' OWN rows, scales, values and masks: the descriptor, the
    operands and the workspace (:class:`fsla.engine.plan.DecodeLaunch`).

    The decay arm's dials are derived here: a per-HEAD constant, the one operand the step kernel does not fold
    (docs/internals/decode/decode_fold.md).
    """
    with span("build_launch"):
        return DecodeLaunch(
            descriptor=descriptor,
            operands=DecodeOperands(
                read=routes.amp_read, write=routes.amp_write, read_scale=routes.scale_read,
                write_scale=routes.scale_write, read_mask=masks.read, write_mask=masks.write,
                dials=bodies.leaf_rate_dials_for(decay, descriptor, values.device), v=values.v, v_scale=values.scale,
                state_plane=plane),
            workspace=DecodeWorkspace(scratch=held))


def step(launch):
    """THE STEP: fold, expand, walk and the residency verdict, in one launch."""
    with span("step"):
        return bodies.step(launch)


def validate_context(routes, values, masks, state, decay) -> None:
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


def decode_step_gated(routes, values, masks, state, *, decay=None):
    """ONE decode step with NO host participation -- the decode entry, and CAPTURE-SAFE.

    The step reads and updates the state's plane in place and returns ``(y, state)``. The step builds its own
    descriptor from its arguments (:func:`descriptor_of`); the call carries only its inputs.
    """
    validate_context(routes, values, masks, state, decay)
    BH, device = decode_call(routes)
    descriptor = descriptor_of(routes, values, decay, BH, device)
    held = scratch(state, descriptor, BH, device)
    carried_plane = backing(state, routes)
    plane = state_plane(values, descriptor, decay, carried_plane)
    launch = build_launch(descriptor, routes, values, masks, decay, plane, held)
    return step(launch), state
