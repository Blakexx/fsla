# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE DESCRIPTOR: one device and one call as the planner (``csrc/fsla/src/common/plan/``, docs/internals/common/plan/environment.md) holds
them, validated against the contract before any launch. It is the op's argument beside its data: immutable, hashable, reusable by every
call of the same shape on the same device. Python reads no hardware fact: the device's are the planner's read, the call's are the
caller's, and what a launch takes from them is the planner's launch function's (:func:`fsla.ops.decode.decode_launch`).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import torch

from fsla.ops._ext import device_index, extension

__all__ = ["Descriptor", "check_workspace", "descriptor", "workspace_bytes"]


@dataclass(frozen=True, slots=True)
class Descriptor:
    """A validated context: ``device`` the CUDA index of its device, ``call`` the call's facts by name, and ``words`` every fact as the
    planner carries it (an operator's argument)."""

    device: int
    call: tuple[tuple[str, int], ...]
    words: tuple[int, ...]

    def __getitem__(self, fact: str) -> int:
        for name, value in self.call:
            if name == fact:
                return value
        raise KeyError(f"the descriptor holds no call fact {fact!r}; its call facts are {[name for name, _ in self.call]}")


@cache
def _descriptor(device: int, call: tuple[tuple[str, int], ...]) -> Descriptor:
    ext = extension()
    names = tuple(ext.fsla_call_facts().split(","))
    given = dict(call)
    if set(given) != set(names):
        raise TypeError(f"a call supplies the facts {list(names)}; this one gives {sorted(given)}")
    words = ext.fsla_descriptor(device, [given[name] for name in names])
    return Descriptor(device=device, call=tuple((name, given[name]) for name in names), words=tuple(int(w) for w in words))


def descriptor(*, device=None, **call) -> Descriptor:
    """THE VALIDATED DESCRIPTOR of a call on ``device`` (the current device when ``None``): the device's facts, read once by the
    planner, and the call's (``d_v``, ``levels``, ``decay`` for decode). A context outside the contract is refused here, before any launch,
    naming the facts."""
    return _descriptor(device_index(device), tuple(sorted((name, int(value)) for name, value in call.items())))


def workspace_bytes(descriptor: Descriptor) -> int:
    """The scratch a call of ``descriptor`` needs beside its operands, in bytes: the planner's layout. Decode needs none."""
    from fsla.ops.decode import decode_launch

    return decode_launch(descriptor).workspace_bytes


def check_workspace(workspace: torch.Tensor | None, descriptor: Descriptor) -> None:
    """A caller's own workspace, refused by name when it is not a CUDA byte tensor, is too small or is misaligned for ``descriptor``."""
    if workspace is None:
        return
    from fsla.ops.decode import decode_launch

    launch = decode_launch(descriptor)
    size, alignment = launch.workspace_bytes, launch.workspace_alignment
    if not isinstance(workspace, torch.Tensor) or workspace.dtype != torch.uint8 or workspace.dim() != 1 or not workspace.is_cuda:
        raise TypeError("workspace is a contiguous 1-D CUDA uint8 tensor (or None, for the op to allocate)")
    if not workspace.is_contiguous():
        raise ValueError("workspace must be contiguous")
    if workspace.numel() < size:
        raise ValueError(f"workspace holds {workspace.numel()} bytes; the descriptor's call needs {size} (fsla.workspace_bytes)")
    if workspace.data_ptr() % alignment:
        raise ValueError(f"workspace's address is not a multiple of {alignment} bytes (fsla.workspace_bytes)")
    if workspace.device.index != descriptor.device:
        raise ValueError(f"workspace is on device {workspace.device.index}; the descriptor was made for device {descriptor.device}")
