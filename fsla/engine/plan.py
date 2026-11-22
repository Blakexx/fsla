# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The prefill plan, and the bundles a launch travels in.

PREFILL'S PLAN is WHAT TO DO and nothing else: frozen, static, hashable -- the arm and the cache key. The plan says what to do,
not what you are giving (ruled 2026-11-11), and it changes at most rounding within the oracle's tolerance, never the
mathematical output: that is the DATA's (ruled 2026-11-22). Nothing static reads an operand. DECODE HAS NO PLAN: its op takes
the validated descriptor (:class:`fsla.ops.descriptor.Descriptor`), and everything derived from the descriptor's facts is the
planner's launch function's, computed right before the launch and never stored (docs/internals/common/plan/decode_launch.md).

A LAUNCH is one internal bundle per kernel family (:class:`DecodeLaunch`, :class:`PrefillLaunch`), built at a DAG's
terminal node and passed to the launcher, never public. It holds three named parts: prefill's ``plan`` or decode's
``descriptor``, the ``operands`` (the data -- rows, scales, values, masks, dials, the state and its page table -- what the
oracle also takes) and the ``workspace`` (what the op allocates for itself). The public op takes the operands as arguments
and the plan or descriptor; the launch is how they travel inside it (docs/internals/engine/engine.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from fsla.engine.types import Carve


@dataclass(frozen=True, slots=True)
class PrefillPlan:
    """What the prefill does: ``carve``, the carry's box and each side's operation plan (the lists a side walks are
    the carve's; docs/internals/engine/carve.md), and ``paging``, the state's BACKING mapping.

    ``paging``: the plan overrides the MAPPING ONLY. A stateful call pages by default -- one page is one MMA atom, pages
    are admitted by the write-side plan's exact bitmap, and an atom with no page reads as zeros, so paging is BITWISE
    INVISIBLE and there is nothing to turn on. ``None`` (the default) is that default; ``False`` pins the DENSE plane,
    which is what the bit-identity gate compares against; ``True`` is REFUSED, because a knob whose only value is the
    default is a knob that lies about being a decision.

    Built by the producer's plan rule (:func:`fsla.ops.produce.prefill_plan`), which checks the carve against the
    declaration.
    """

    carve: Carve
    paging: bool | None = None

    def __post_init__(self) -> None:
        if type(self.carve) is not Carve:
            raise TypeError(f"carve must be a Carve, got {type(self.carve).__name__}")
        if self.paging is not None and type(self.paging) is not bool:
            raise TypeError(f"paging must be None or a bool, got {self.paging!r}")
        if self.paging is True:
            raise ValueError(
                "paging is ON by default for every stateful call, so there is nothing to turn on: pass paging=None "
                "(or omit it) for the paged backing, or paging=False to pin the dense plane. The plan overrides the "
                "MAPPING only (docs/internals/state.md).")


@dataclass(frozen=True, slots=True)
class PrefillOperands:
    """The prefill's data: the bundle, the values, the masks, the decay and the state's entry plane and page table."""

    routes: object
    values: object
    masks: object
    decay: object
    state_in: torch.Tensor | None
    page_table: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class PrefillWorkspace:
    """What the prefill call allocates for itself: the arena and the plane it stores the state into."""

    arena: object
    state_out: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class PrefillLaunch:
    """One prefill launch's whole execution data: the static plan, the operands, the workspace."""

    plan: PrefillPlan
    operands: PrefillOperands
    workspace: PrefillWorkspace


@dataclass(frozen=True, slots=True)
class DecodeOperands:
    """The decode step's data. The bundle's own ``[B, 1, H, width_l]`` unit-max coefficient rows (u8) and their fp32
    ``[B, 1, H]`` maxima, per side; the masks' per-level words, DIGIT-packed int32 ``[B, H, ceil(width_l / 32)]`` per
    side (the step computes with ``r * m``; :mod:`fsla.routing.masks`); the dials; the value ``[B, 1, H, d_v]`` bf16
    against the token's largest channel and its fp32 ``[B, 1, H]`` scale; the state plane, its leaf order and the page table."""

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
    #: The `(k, m)` BOX the state plane's leaves are ordered by: the leaf order travels with the plane
    #: (`fsla.ops.lattice`), an attribute of the state operand and not a value of any plan.
    state_lattice: tuple[int, int]
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
