# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE PREFILL'S MASK-PREPARATION PASS, its Python face (csrc/fsla/src/mask_prepare/mask_prepare.cu, docs/internals/mask_prepare/mask_prepare.md).

:func:`prepare_masks` writes, for every level-side of a PREFILL-layout :class:`fsla.routing.masks.Masks` bundle that has a mask, its group words
and totals into one uint8 workspace, at the layout of ``csrc/fsla/src/common/plan/mask_prepare_layout.h`` (docs/internals/common/plan/mask_prepare_layout.md).
"""
from __future__ import annotations

import torch

from fsla.ops._ext import extension
from fsla.routing.masks import PREFILL, Masks

__all__ = ["prepare_masks"]

#: a density's levels at most, as the carve holds them (`plan::kCarveMaxLevels`): a level-side's slot is `side * LEVELS + level`
LEVELS = 4


def prepare_masks(masks: Masks, density: torch.Tensor, widths: tuple[int, ...] | list[int]) -> torch.Tensor:
    """The workspace the pass wrote for ``masks``: uint8 on the masks' device, ``mask_prepare_bytes`` long. The slot of a present entry
    is ``side * 4 + level`` (side 0 the read side); a bundle with no present entry returns an empty tensor and launches nothing. ``widths`` are the digits of each of the
    call's levels, masked or not: a level-side's group is the call's carve's, which the pass derives from them and the density. A level-side whose
    group is the level's size is whole (byte 255 always) and is left as the workspace was made (uninitialised). The window is the card's, which the op derives from the masks' device."""
    if masks.consumer != PREFILL:
        raise ValueError(f"the mask-preparation pass is the prefill's: the masks are {masks.consumer!r}-layout, not {PREFILL!r}")
    present = [(side * LEVELS + level, mask)
               for side, name in enumerate(("read", "write"))
               for level, mask in enumerate(getattr(masks, name)) if mask is not None]
    if not present:
        return torch.empty(0, dtype=torch.uint8, device=density.device)
    width_bits = [0] * (2 * LEVELS)
    heads = [0] * (2 * LEVELS)
    for slot, mask in present:
        width_bits[slot] = int(mask.shape[3]).bit_length() - 1
        heads[slot] = int(mask.shape[0] * mask.shape[1])
    ext = extension()
    words = int(present[0][1].shape[2])
    workspace = torch.empty(int(ext.mask_prepare_bytes(width_bits, heads, words, present[0][1].device.index)), dtype=torch.uint8, device=present[0][1].device)
    ext.mask_prepare([mask for _, mask in present], [slot for slot, _ in present], [int(w) for w in widths], density, workspace)
    return workspace
