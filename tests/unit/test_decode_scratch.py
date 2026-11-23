# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The decode step's per-sequence scratch, WITHOUT a device.

The scratch is held one per sequence, weak on the state, and keyed by what a rebuild is a function of
(docs/internals/engine/decode_step.md#the-scratch). Both are host-only properties and this battery holds them
that way: no CUDA, no extension, no kernel.
"""
from __future__ import annotations

import gc

import pytest
import torch

from fsla.engine import decode_step
from fsla.ops import decode as bodies


class _Sequence:
    """A weak-referenceable stand-in for the state object a sequence is keyed by."""


class _Poison:
    """A sequence whose every attribute read is a failure."""

    def __getattr__(self, item):
        raise AssertionError(f"the scratch read the sequence: state.{item}")


@pytest.fixture
def builds(monkeypatch):
    """`DecodeScratch.build` replaced by a recorder of its arguments, so a held scratch and a rebuilt one are told apart."""
    calls: list[tuple] = []

    def build(descriptor, BH, device):
        calls.append((descriptor, BH, device))
        return object()

    monkeypatch.setattr(bodies.DecodeScratch, "build", staticmethod(build))
    return calls


def test_a_scratch_is_built_once_per_key(builds):
    sequence = _Sequence()

    first = decode_step.scratch(sequence, "d64", 2, "cpu")
    again = decode_step.scratch(sequence, "d64", 2, "cpu")
    other = decode_step.scratch(sequence, "d16", 2, "cpu")

    assert first is again, "an identical key rebuilt the scratch"
    assert other is not first, "a different key returned the held scratch"
    assert len(builds) == 2, f"the builder ran {len(builds)} times over two distinct keys"


def test_two_live_sequences_never_share_a_scratch(builds):
    one, two = _Sequence(), _Sequence()

    a = decode_step.scratch(one, "d64", 2, "cpu")
    a_again = decode_step.scratch(one, "d64", 2, "cpu")
    b = decode_step.scratch(two, "d64", 2, "cpu")

    assert a is a_again, "one sequence rebuilt its own scratch"
    assert b is not a, "two concurrent sequences shared one buffer"
    assert len(builds) == 2


def test_a_dropped_sequence_drops_its_scratch(builds):
    """The weakness is required, not incidental: a table keyed on dead sequences is a leak of exactly the device
    allocations paging exists to bound."""
    sequence = _Sequence()
    decode_step.scratch(sequence, "d64", 2, "cpu")
    assert sequence in decode_step._SCRATCH
    before = len(decode_step._SCRATCH)
    del sequence
    gc.collect()
    assert len(decode_step._SCRATCH) == before - 1, "the scratch outlived the sequence that owned it"


def test_a_second_scratch_for_one_sequence_is_refused(builds):
    sequence = _Sequence()
    assert decode_step._scratch_of(sequence) is None
    decode_step.scratch(sequence, "d64", 2, "cpu")
    assert decode_step._scratch_of(sequence) is not None
    decode_step.scratch(sequence, "d16", 2, "cpu")
    with pytest.raises(RuntimeError, match="alive for one state"):
        decode_step._scratch_of(sequence)


def _device_free_launch(monkeypatch):
    """The planner's launch function of a descriptor is the extension's, which reads a device: this battery is device-free by charter, so the
    one launch it sizes a scratch by is given as the planner would answer it (depth 2, 16-wide levels, a 64-wide arm)."""
    launch = bodies.DecodeLaunchFacts(d_v=64, levels=2, decay=0, pad=0, widths=(16, 16))
    monkeypatch.setattr(bodies, "decode_launch", lambda descriptor: launch)


def test_the_scratch_reads_the_state_as_a_KEY_and_never_as_an_INPUT(monkeypatch):
    """THE DISCIPLINE: the held scratch is REBUILDABLE TRANSIENTS and nothing else.

    The body runs with the state replaced by an object that raises on any attribute read, so nothing about the
    sequence can enter what is held, and two scratches built for one geometry are interchangeable BY CONSTRUCTION
    rather than by inspection. The CROSS-STEP half -- that a reused scratch carries nothing from one step to the
    next -- is gated where a real step runs: `DecodeScratch`'s counters are self-resetting and
    `tests/integration/test_decode_graph_step.py` replays a captured step across an admission against eager.
    """
    from fsla.ops.descriptor import Descriptor

    _device_free_launch(monkeypatch)
    descriptor = Descriptor(kernel="decode", device=0, call=(), words=())
    one = decode_step.scratch(_Poison(), descriptor, 2, "cpu")
    two = decode_step.scratch(_Poison(), descriptor, 2, "cpu")

    assert type(one) is type(two)
    for name in one.__slots__:
        a, b = getattr(one, name), getattr(two, name)
        assert type(a) is type(b), name
        if isinstance(a, torch.Tensor):
            assert (a.shape, a.dtype, a.device) == (b.shape, b.dtype, b.device), (
                f"{name} is not a function of the declared keys")
            #: A ZERO-ELEMENT buffer owns no bytes, so "shared" is not a property it can have -- torch gives every
            #: empty tensor the null pointer.
            assert a.numel() == 0 or a.data_ptr() != b.data_ptr(), f"{name} is SHARED between two sequences' scratches"
