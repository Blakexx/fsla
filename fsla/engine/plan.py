# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The bundles a launch travels in.

NO FAMILY HAS A PLAN. An operator is ``op(data...)``: the data is what is given at the boundary and changes the
mathematics; the call builds the validated descriptor (:class:`fsla.ops.descriptor.Descriptor`), every device fact and every call fact, and the workspace,
what the call derives. Everything derived from the descriptor's facts is the planner's launch function's, computed right before the launch and
never stored (docs/internals/common/plan/decode_launch.md).

A LAUNCH is one internal bundle per kernel family (:class:`DecodeLaunch`, :class:`PrefillLaunch`), built by the call's launch step (``build_launch`` in ``fsla/engine/decode_step.py``, ``launch`` in ``fsla/engine/prefill_call.py``) and passed to
the launcher, never public. It holds three named parts: the ``descriptor``, the ``operands`` (the data -- rows, scales, values, masks, dials, the
state and its page table -- what the oracle also takes) and the ``workspace`` (what the call derives for itself). The public op takes the operands as arguments;
the launch is how they and the call's descriptor travel inside it (docs/internals/engine/engine.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class PrefillOperands:
    """The prefill's data: the bundle, the values, the masks, the density, the decay, and the state's entry plane, page table and the plane it stores into."""

    routes: object
    values: object
    masks: object
    density: torch.Tensor
    decay: object
    state_in: torch.Tensor | None
    page_table: torch.Tensor | None
    state_out: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class PrefillWorkspace:
    """What the prefill call derives: ``buffer``, the one byte buffer the call's private workspace builder (:func:`fsla.ops.prefill._build_workspace`) laid out: the mask-preparation pass's prepared leaf groups' mask words and counts and whatever the carry's kernel needs.
    The state is not in it: the state is the op's own argument, in and out."""

    buffer: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class PrefillLaunch:
    """One prefill launch's whole execution data: the ``descriptor`` (the validated context the planner's launch function reads,
    :class:`fsla.ops.descriptor.Descriptor`), the ``operands`` and the ``workspace``."""

    descriptor: object
    operands: PrefillOperands
    workspace: PrefillWorkspace


@dataclass(frozen=True, slots=True)
class DecodeOperands:
    """The decode step's data. The bundle's own ``[B, 1, H, width_l]`` unit-max coefficient rows (u8) and their fp32
    ``[B, 1, H]`` maxima, per side; the masks' per-level words, DIGIT-packed int32 ``[B, H, ceil(width_l / 32)]`` per
    side (the step computes with ``r * m``; :mod:`fsla.routing.masks`); the dials; the value ``[B, 1, H, d_v]`` bf16
    against the token's largest channel and its fp32 ``[B, 1, H]`` scale; the state plane and the page table."""

    read: tuple[torch.Tensor, ...]
    write: tuple[torch.Tensor, ...]
    read_scale: tuple[torch.Tensor, ...]
    write_scale: tuple[torch.Tensor, ...]
    read_mask: tuple[torch.Tensor, ...]
    write_mask: tuple[torch.Tensor, ...]
    dials: torch.Tensor | None
    v: torch.Tensor
    v_scale: torch.Tensor
    state_plane: torch.Tensor
    page_table: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class DecodeWorkspace:
    """What the step allocates for itself: ``scratch``, the per-sequence buffers, and ``pool``, the arena's slack slots a
    growth step admits itself into -- present exactly on a paged step that has one, and the kernel's own per-batch-head
    verdict makes a step it cannot cover a device-side no-op (docs/internals/decode/decode.md#per-bh-verdict)."""

    scratch: object
    pool: object


@dataclass(frozen=True, slots=True)
class DecodeLaunch:
    """Everything one decode launch reads, in three named parts: the ``descriptor`` (the validated context the planner's launch
    function reads, :class:`fsla.ops.descriptor.Descriptor`), the ``operands`` and the ``workspace``. NO FIELD IS DERIVED AT THE LAUNCH SITE."""

    descriptor: object
    operands: DecodeOperands
    workspace: DecodeWorkspace
