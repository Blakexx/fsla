# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The state plane: the dense plane's allocation."""
from __future__ import annotations

import torch

from fsla.engine.types import MMA_K_QUANTUM


def dense_plane(batch_heads: int, widths, *, d_v: int, device, zeros: bool = False) -> torch.Tensor:
    """A `[BH, N/16, page_bytes]` uint8 state plane of ``batch_heads`` batch-heads over levels of ``widths`` digits: one
    int8-format page an atom, in canonical atom order (`fsla.ops.paging.page_layout`). An all-zero page is the zero state.

    Allocation is the caller's, not the kernel entry's -- which is what lets a
    carried state be handed straight back in, and a CONTINUATION does exactly that
    rather than reaching here at all (docs/internals/state.md#the-dense-continuation).
    """
    from fsla.ops.paging import page_layout

    n = 1
    for w in widths:
        n *= w
    build = torch.zeros if zeros else torch.empty
    return build((batch_heads, n // MMA_K_QUANTUM, page_layout(d_v).bytes), device=device, dtype=torch.uint8)


def state_plane(routes, *, d_v: int, zeros: bool = False) -> torch.Tensor:
    """The state plane of this routing's batch-heads, levels and device (:func:`dense_plane`)."""
    return dense_plane(routes.batch * routes.heads, tuple(level.width for level in routes.topology.levels), d_v=d_v,
                       device=routes.device, zeros=zeros)
