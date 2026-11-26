# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The state plane: the dense plane's allocation."""
from __future__ import annotations

import torch

from fsla.engine.types import MMA_K_QUANTUM


def state_plane(routes, *, d_v: int, zeros: bool = False) -> torch.Tensor:
    """A `[BH, N/16, page_bytes]` uint8 state plane for this routing: one int8-format page an atom, in canonical
    atom order (`fsla.ops.paging.page_layout`). An all-zero page is the zero state.

    Allocation is the caller's, not the kernel entry's -- which is what lets a
    carried state be handed straight back in, and a CONTINUATION does exactly that
    rather than reaching here at all (docs/internals/state.md#the-dense-continuation).
    """
    from fsla.ops.paging import page_layout

    widths = tuple(level.width for level in routes.topology.levels)
    n = 1
    for w in widths:
        n *= w
    shape = (routes.batch * routes.heads, n // MMA_K_QUANTUM, page_layout(d_v).bytes)
    build = torch.zeros if zeros else torch.empty
    return build(shape, device=routes.device, dtype=torch.uint8)
