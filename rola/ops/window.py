# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE WINDOW W, a card's: the binary's table at a device's compute capability (``csrc/rola/src/common/window.cuh``).

Every kernel of a call shares the window -- the facts pass builds each window's lists, the carry reads each window's
state out and folds its writes, the den pass and the intra each window's pairs -- and W is per-card configuration
(ruled 2026-10-31), so a host shape or check is built from the device's W, never a module constant.
"""
from __future__ import annotations

import functools

import torch

from rola.ops._ext import extension

__all__ = ["window_of"]


@functools.cache
def _window_of(index: int) -> int:
    return int(extension().device_window(index))


def window_of(device=None) -> int:
    """``device``'s window in tokens -- a device, its index, or ``None`` for the current device."""
    if isinstance(device, int):
        return _window_of(device)
    index = None if device is None else torch.device(device).index
    return _window_of(torch.cuda.current_device() if index is None else index)
