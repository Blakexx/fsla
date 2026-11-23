# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE DESCRIPTOR: one device and one call of a KERNEL as the planner (``csrc/fsla/src/common/plan/``, docs/internals/common/plan/environment.md) holds
them, validated against the kernel's contract before any launch. Each call builds its own, internally: immutable, hashable, reusable by every
call of the same shape on the same device. Python reads no hardware fact: the device's are the planner's read, the call's are the
caller's, and what a launch takes from them is the planner's launch function's (:func:`fsla.ops.decode.decode_launch`).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from fsla.ops._ext import device_index, extension, not_built

__all__ = ["Descriptor", "descriptor"]


@dataclass(frozen=True, slots=True)
class Descriptor:
    """A validated context: ``kernel`` the kernel it is of (``decode``, ``prefill``: a registry of ``hardware/kernels/``), ``device`` the CUDA index of its
    device, ``call`` the call's facts by name and ``words`` every fact as the planner carries it (an operator's argument)."""

    kernel: str
    device: int
    call: tuple[tuple[str, int], ...]
    words: tuple[int, ...]

    def __getitem__(self, fact: str) -> int:
        for name, value in self.call:
            if name == fact:
                return value
        raise KeyError(f"the {self.kernel} descriptor holds no call fact {fact!r}; its call facts are {[name for name, _ in self.call]}")


@cache
def call_facts(kernel: str) -> tuple[str, ...]:
    """The names of the call facts ``kernel`` states, as the planner's generated definition of it holds them, in its order."""
    return tuple(extension().fsla_call_facts(kernel).split(","))


@cache
def _descriptor(kernel: str, device: int, call: tuple[tuple[str, int], ...]) -> Descriptor:
    ext = extension()
    names = call_facts(kernel)
    given = dict(call)
    if set(given) != set(names):
        raise TypeError(f"a {kernel} call supplies the facts {list(names)}; this one gives {sorted(given)}")
    values = [given[name] for name in names]
    owed = ext.fsla_owed_kernels(kernel, device, values)
    if owed:
        not_built(*owed.split(","))
    words = ext.fsla_descriptor(kernel, device, values)
    return Descriptor(kernel=kernel, device=device, call=tuple((name, given[name]) for name in names), words=tuple(int(w) for w in words))


def descriptor(*, kernel: str, device=None, **call) -> Descriptor:
    """THE VALIDATED DESCRIPTOR of a call of ``kernel`` on ``device`` (the current device when ``None``): the device's facts, read once by the
    planner, and the call's (every call fact of the kernel's registry: its extents, its value width and decay, each level's ``width_bits_k`` and its
    walks, which :func:`fsla.ops.produce.decode_descriptor` and :func:`fsla.ops.produce.prefill_descriptor` state from a declaration). A context outside
    the kernel's contract is refused here, before any launch, naming the facts; one the contract holds as OWED (its kernels are not built) is refused as
    :class:`~fsla.ops._ext.KernelNotBuilt` naming each kernel it waits on."""
    return _descriptor(kernel, device_index(device), tuple(sorted((name, int(value)) for name, value in call.items())))
