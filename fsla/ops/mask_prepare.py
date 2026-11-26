# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE PREFILL'S MASK-PREPARATION PASS, its Python face (csrc/fsla/src/mask_prepare/mask_prepare.cu, docs/internals/mask_prepare/mask_prepare.md).

:func:`prepare_masks` writes, for EACH SIDE of a call (0 the read side, 1 the write side), each of its LEAF GROUPS' mask line (the window's mask words: the call's own tokens ANDed, on every level the side
lists and has a mask on, with the OR of the leaf group's group's digits) and count (uint16, the line's set bits) into one uint8 workspace, at the layout of
``csrc/fsla/src/common/plan/mask_prepare_layout.h`` (docs/internals/common/plan/mask_prepare_layout.md); the leaf groups are the carve's
(docs/internals/common/plan/leaf_groups.md). Both sides are always written, whether or not the bundle holds a mask: a side that lists no level has one leaf group, the call's own tokens.
"""
from __future__ import annotations

import torch

from fsla.ops._ext import extension
from fsla.routing.masks import PREFILL, Masks

__all__ = ["prepare_masks"]

#: a density's levels at most, as the carve holds them (`plan::kCarveMaxLevels`): a level-side's slot is `side * LEVELS + level`
LEVELS = 4


def prepare_masks(masks: Masks, density: torch.Tensor, widths: tuple[int, ...] | list[int], heads: tuple[int, int] | list[int], tokens: int) -> torch.Tensor:
    """The workspace the pass wrote for ``masks``: uint8 on the density's device, ``mask_prepare_bytes`` long. The slot of a present entry is ``side * 4 + level`` (side 0 the read side).
    ``widths`` are the digits of each of the call's levels, masked or not: a side's leaf groups are the call's carve's, which the pass derives from them and the density. ``heads`` is
    ``(read batch-heads, write batch-heads)``, batch times the side's heads, and ``tokens`` the call's tokens a head; a present mask is int32 ``[B, H, ceil(tokens / 32), width]`` and the
    op checks it against both. Both sides are written even where no entry is present (the density's card is the call's); the bytes of a side's region past its blocks, and of a block between
    its counts and its lines, are left as the workspace was made (uninitialised). The window is the card's, which the op derives from the density's device."""
    if masks.consumer != PREFILL:
        raise ValueError(f"the mask-preparation pass is the prefill's: the masks are {masks.consumer!r}-layout, not {PREFILL!r}")
    if not density.is_cuda:
        raise ValueError("the mask-preparation pass runs on a card: the density is on the host")
    if len(heads) != 2:
        raise ValueError(f"heads is (read batch-heads, write batch-heads), got {tuple(heads)}")
    present = [(side * LEVELS + level, mask)
               for side, name in enumerate(("read", "write"))
               for level, mask in enumerate(getattr(masks, name)) if mask is not None]
    masked = [sum(1 << level for level, mask in enumerate(getattr(masks, name)) if mask is not None) for name in ("read", "write")]
    ext = extension()
    workspace = torch.empty(int(ext.mask_prepare_bytes([int(w) for w in widths], masked, [int(h) for h in heads], int(tokens), density.device.index)), dtype=torch.uint8, device=density.device)
    ext.mask_prepare([mask for _, mask in present], [slot for slot, _ in present], [int(w) for w in widths], [int(h) for h in heads], int(tokens), density, workspace)
    return workspace
