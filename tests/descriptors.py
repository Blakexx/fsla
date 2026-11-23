# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The descriptor of a call, made the way the call makes it, for a test that states it by hand.

A call builds its own descriptor through the library's cached builders (`fsla.ops.produce`); these two are the tests' handle on them, with
the device defaulted to the current one. A width or an extent outside the contract is refused when the descriptor is made, naming the fact; a
prefill is refused as `KernelNotBuilt` while no carry is built.
"""
from __future__ import annotations

from fsla.ops._ext import device_index
from fsla.ops.descriptor import Descriptor
from fsla.ops.produce import _decode_descriptor, _prefill_descriptor


def decode_descriptor(topology, *, d_v: int, decay: bool, batch_heads: int, device=None) -> Descriptor:
    """The decode step's validated descriptor on ``device`` (the current device for ``None``): the level widths and write walks are the
    declaration's, ``batch_heads`` is B times H."""
    return _decode_descriptor(topology, int(d_v), bool(decay), int(batch_heads), device_index(device))


def prefill_descriptor(topology, *, d_v: int, decay: bool, tokens: int, batch_heads: int, stateful: bool, device=None) -> Descriptor:
    """The prefill's validated descriptor on ``device`` (the current device for ``None``): the walks and padded widths are the declaration's;
    ``tokens`` is the call's length, ``batch_heads`` B times H, ``stateful`` whether a state is handed in or out."""
    return _prefill_descriptor(topology, int(d_v), bool(decay), int(tokens), int(batch_heads), bool(stateful), device_index(device))
