# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE WINDOW W, a card's: the device's card as the binary selects it (``csrc/rola/src/common/select.cuh``) -- its
compute capability, product, build variant, family and W.

Every kernel of a call shares the window -- the facts pass builds each window's lists, the carry reads each window's
state out and folds its writes, the den pass and the intra each window's pairs -- and W is per-card configuration
(ruled 2026-10-31), so a host shape or check is built from the device's W, never a module constant.
"""
from __future__ import annotations

import dataclasses
import functools

import torch

from rola.ops._ext import extension

__all__ = ["Card", "card_of", "window_of"]

#: the families in `Family`'s order (`common/hardware.cuh`)
FAMILIES = ("ampere", "hopper", "blackwell")


@dataclasses.dataclass(frozen=True)
class Card:
    """A device's card: its compute capability (x10), its product row's index in the generated `kProducts` (-1: none),
    the build variant it runs, its family and its window W."""

    cc: int
    product: int
    variant: int
    family: str
    window: int


@functools.cache
def _card_of(index: int) -> Card:
    cc, product, variant, family, window = (int(x) for x in extension().device_card(index))
    return Card(cc, product, variant, FAMILIES[family], window)


def _index(device) -> int:
    if isinstance(device, int):
        return device
    index = None if device is None else torch.device(device).index
    return torch.cuda.current_device() if index is None else index


def card_of(device=None) -> Card:
    """``device``'s card -- a device, its index, or ``None`` for the current device; refused by name where the binary
    carries no build for it."""
    return _card_of(_index(device))


def window_of(device=None) -> int:
    """``device``'s window in tokens -- a device, its index, or ``None`` for the current device."""
    return card_of(device).window
