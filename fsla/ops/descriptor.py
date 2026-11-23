# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE DESCRIPTOR: one device and one call as the planner (``csrc/fsla/src/common/plan/``, docs/internals/common/plan/environment.md) holds
them, validated against the contract before any launch. Each call builds its own, internally: immutable, hashable, reusable by every
call of the same shape on the same device. Python reads no hardware fact: the device's are the planner's read, the call's are the
caller's, and what a launch takes from them is the planner's launch function's (:func:`fsla.ops.decode.decode_launch`).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from fsla.ops._ext import device_index, extension, not_built

__all__ = ["Descriptor", "descriptor", "workspace_bytes"]


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
    values = [given[name] for name in names]
    owed = ext.fsla_owed_kernels(device, values)
    if owed:
        not_built(*owed.split(","))
    words = ext.fsla_descriptor(device, values)
    return Descriptor(device=device, call=tuple((name, given[name]) for name in names), words=tuple(int(w) for w in words))


def descriptor(*, device=None, **call) -> Descriptor:
    """THE VALIDATED DESCRIPTOR of a call on ``device`` (the current device when ``None``): the device's facts, read once by the
    planner, and the call's (every call fact of the schema: the call's kind ``prefill``, its extents, its value width and decay, each level's
    ``width_bits_k``, ``read_listed_k`` and ``write_listed_k``, which :func:`fsla.ops.produce.decode_descriptor` and
    :func:`fsla.ops.produce.prefill_descriptor` state from a declaration). A context outside the contract is refused here, before any
    launch, naming the facts; one the contract holds as OWED (its kernels are not built) is refused as :class:`~fsla.ops._ext.KernelNotBuilt`
    naming each kernel it waits on."""
    return _descriptor(device_index(device), tuple(sorted((name, int(value)) for name, value in call.items())))


def _need(descriptor: Descriptor) -> tuple[int, int]:
    """The call's workspace ``(bytes, alignment)``: the planner's layout of its call kind (``prefill``)."""
    if descriptor["prefill"]:
        from fsla.ops.prefill import prefill_workspace

        return prefill_workspace(descriptor)
    from fsla.ops.decode import decode_launch

    launch = decode_launch(descriptor)
    return launch.workspace_bytes, launch.workspace_alignment


def workspace_bytes(descriptor: Descriptor) -> int:
    """The workspace a call of ``descriptor`` needs beside its operands, in bytes: the planner's layout of its kind (the ``prefill`` fact
    chooses). Decode needs none; a prefill's layout is the prefill launch function's, which is not built, so asking for one is refused by name."""
    return _need(descriptor)[0]
