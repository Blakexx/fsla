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

__all__ = ["launch_bytes_per_token", "prefill", "reader_factor", "readout"]


def prefill(launch, facts, layout, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """``launch`` is the :class:`fsla.engine.plan.PrefillLaunch`. ``(num, den)`` fp32 ``[BH, L, d_v]`` and ``[BH, L]`` over ``r`` (:func:`readout` divides by it), the carry's and
    the intra's reduced into the same planes: not built on this line (:func:`fsla.ops._ext.not_built`)."""
    not_built("prefill")


def launch_bytes_per_token(plan, topology, num_heads: int, d_v: int, hidden_size: int) -> int:
    """A segment's working set per batch-token in bytes: the sum of what the launch builder allocates for a
    :class:`fsla.engine.plan.PrefillLaunch`'s operands and workspace, by the same function. Not built on this line
    (:func:`fsla.ops._ext.not_built`): the builder exists with the prefill kernel."""
    not_built("prefill launch builder")


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
