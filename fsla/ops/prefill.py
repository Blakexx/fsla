# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE PREFILL COMPOSITION: one segment's kernels over the bundle and the epilogue that joins them.

THE KERNELS ARE ONE OPERATOR on one window grid: the carry computes every pair whose write fell in a strictly earlier
window than its read, the intra every pair inside one window. Together they are the whole recurrence, no pair counted
twice and none dropped (`tests/oracle/test_carry_vs_oracle.py`'s decomposition test states it in fp64). Neither is
built on this line, so :func:`prefill` raises :class:`fsla.ops._ext.KernelNotBuilt`.

THE UNITS. Every kernel leaves each token's READER FACTOR ``r`` (its read levels' scales) out of its sums, and the
readout divides by it (:func:`readout`): ``y = num / (den + eps / r)`` is the defined readout exactly.

Symbols, each at first use: ``L`` tokens, ``BH`` batch x heads, ``d_v`` value channels.
"""
from __future__ import annotations

import torch

from fsla.ops._ext import not_built
from fsla.ops.descriptor import Descriptor

__all__ = ["build_workspace", "check_descriptor", "prefill", "prefill_workspace", "reader_factor", "readout"]


def prefill(launch, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """``launch`` is the :class:`fsla.engine.plan.PrefillLaunch`. ``(num, den)`` fp32 ``[BH, L, d_v]`` and ``[BH, L]`` over ``r`` (:func:`readout` divides by it), the carry's and
    the intra's reduced into the same planes: not built on this line (:func:`fsla.ops._ext.not_built`)."""
    not_built("prefill")


def prefill_workspace(descriptor: Descriptor) -> tuple[int, int]:
    """The workspace ``(bytes, alignment)`` of a prefill call: the prefill launch function's layout, whose first region is the device plan. Not built on this
    line (:func:`fsla.ops._ext.not_built`): the launch function, which derives the carve, owns the layout."""
    not_built("prefill")


def check_descriptor(descriptor: Descriptor, routes, values, *, decay=None, stateful: bool | None = None) -> None:
    """THE ONE CHECK OF A PREFILL'S DESCRIPTOR against the call it is handed: a :class:`~fsla.ops.descriptor.Descriptor` of a prefill (the ``prefill`` fact 1),
    made for the device the operands are on, stating the bundles' own extents (``tokens``, ``batch_heads``, ``d_v``) and the declaration's (the depth,
    each level's padded width and each side's walk of it). ``decay`` is the call's dial, and ``stateful`` whether a state is handed in or out: each is
    checked against its fact unless ``stateful`` is ``None`` (a caller that has none to compare, the workspace's builder)."""
    from fsla.ops.produce import level_facts

    if type(descriptor) is not Descriptor:
        raise TypeError(f"a prefill's `descriptor` is a fsla.Descriptor, got {type(descriptor).__name__}; "
                        "fsla.ops.produce.prefill_descriptor builds one from the declaration")
    if not descriptor["prefill"]:
        raise ValueError("the descriptor is a decode step's (its `prefill` fact is 0); fsla.ops.produce.prefill_descriptor builds a prefill's")
    if routes.device.type == "cuda" and descriptor.device != routes.device.index:
        raise ValueError(f"the descriptor was made for device {descriptor.device}; the operands are on {routes.device}")
    stated = {"tokens": routes.tokens, "batch_heads": routes.batch * routes.heads, "d_v": values.d_v,
              **level_facts(routes.topology, tuple(routes.topology.padded_widths))}
    if stateful is not None:
        stated.update(decay=int(decay is not None), stateful=int(stateful))
    for fact, want in stated.items():
        if descriptor[fact] != want:
            raise ValueError(f"the descriptor states {fact}={descriptor[fact]}; this call's is {want}: build the descriptor from this call "
                             "(fsla.ops.produce.prefill_descriptor)")


def build_workspace(routes, values, masks, descriptor: Descriptor,
                    workspace: torch.Tensor | None = None) -> torch.Tensor:  #: reserved: the caller's own buffer to write into, once the layout (open-work A-g) is written
    """THE PUBLIC BUILDER of a prefill call's workspace: the one buffer ``fsla_op(..., workspace=)`` takes, laid out by the prefill launch function
    (``common/plan/``), whose first region is the DEVICE PLAN -- the decisions of the call, each box's counts and the step-shape word -- written
    from the masks. ``workspace`` is a CUDA uint8 buffer of :func:`fsla.workspace_bytes` to write into, or ``None`` to allocate one; the buffer is
    returned. The op runs no derivation pass of its own: this is the only place the device plan is derived.

    THE SURFACE ONLY. It validates its arguments -- the bundles' types, the masks' layout, a ``descriptor`` that is a prefill's
    (:func:`check_descriptor`) and made for the operands' device, a ``workspace`` that is a one-dimensional byte tensor -- and then REFUSES BY NAME
    (:class:`~fsla.ops._ext.KernelNotBuilt`, kernel ``prefill``): the device plan needs the carve, which the prefill launch function derives and which
    is not written yet (docs/open-work.md rows A-f and A-g)."""
    from fsla.routing.factors import RouteFactors
    from fsla.routing.masks import PREFILL, Masks
    from fsla.values import Values

    if type(routes) is not RouteFactors or type(values) is not Values or type(masks) is not Masks:
        raise TypeError("build_workspace takes a RouteFactors, a Values and the PREFILL-layout Masks of the producer, got "
                        f"{type(routes).__name__}, {type(values).__name__} and {type(masks).__name__}")
    if masks.consumer != PREFILL:
        raise ValueError(f"build_workspace takes the {PREFILL!r}-layout masks, got {masks.consumer!r}")
    masks.check(routes)
    check_descriptor(descriptor, routes, values)
    if workspace is not None and (not isinstance(workspace, torch.Tensor) or workspace.dtype != torch.uint8 or workspace.dim() != 1):
        raise TypeError("workspace is a contiguous 1-D CUDA uint8 tensor (or None, for the builder to allocate)")
    not_built("prefill")


def reader_factor(routes) -> torch.Tensor:
    """``[BH, L]``: each token's read scales multiplied -- the per-token factor the kernels leave out of ``num`` and
    ``den``."""
    r = routes.scale_read[0]
    for scale in routes.scale_read[1:]:
        r = r * scale
    return r.permute(0, 2, 1).reshape(routes.batch * routes.heads, routes.tokens)


def readout(num: torch.Tensor, den: torch.Tensor, routes) -> torch.Tensor:
    """THE READOUT, ``y = num / (den + eps)`` of the full sums, ``[B, T, H, d_v]`` fp32. The kernels' sums leave out
    each token's reader factor ``r`` (it cancels in the ratio), so the floor divides by it: ``num' / (den' + eps / r)``
    is the defined readout exactly, and a token that reads nothing (``r = 0``) reads zero."""
    from fsla.ops.constants import READOUT_EPS

    y = num / (den + READOUT_EPS / reader_factor(routes)).unsqueeze(-1)
    return y.reshape(routes.batch, routes.heads, routes.tokens, -1).permute(0, 2, 1, 3)
