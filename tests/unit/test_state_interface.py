# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE STATE IS AN INTERFACE WITH ONE IMPLEMENTATION, the dense state, pinned without a kernel or a device.

`fsla.FSLAState` is the interface a caller and the ops hold; `fsla.dense_state()` builds the one implementation,
`fsla._state.DenseState` (docs/internals/state.md#interface). The dense state takes no argument: the three the
old constructor had (`strategy`, `pool_slack`, `backing`) were paging's or had one built value, and `fsla.state`
is gone.
"""
from __future__ import annotations

import inspect

import pytest
import torch

import fsla
from fsla._state import DenseState, FSLAState

#: what a caller may ask of a state, and the seam the ops take the plane in and hand it back through.
_ASKED = ("format", "bound", "shape", "device", "dtype", "carries", "materialize", "clone")
_SEAM = ("_kernel_entry", "_kernel_commit", "_require_carrying", "_decode_entry")


def test_the_interface_declares_what_a_caller_asks_and_the_seam_the_ops_use():
    assert FSLAState.__abstractmethods__ == frozenset(_ASKED + _SEAM)


def test_the_interface_is_not_constructible():
    with pytest.raises(TypeError, match="abstract"):
        FSLAState()  # type: ignore[abstract]


def test_the_dense_state_satisfies_the_interface():
    state = fsla.dense_state()
    assert type(state) is DenseState and isinstance(state, FSLAState)
    assert not getattr(DenseState, "__abstractmethods__", frozenset())


def test_the_public_constructor_takes_no_argument():
    assert list(inspect.signature(fsla.dense_state).parameters) == []
    assert "dense_state" in fsla.__all__ and "FSLAState" in fsla.__all__


def test_the_old_constructor_is_gone_without_an_alias():
    assert not hasattr(fsla, "state") and "state" not in fsla.__all__
    import fsla._state as module

    assert not hasattr(module, "state")


@pytest.mark.parametrize("keyword", ["backing", "pool_slack", "strategy"])
def test_the_old_arguments_are_refused_as_unknown_keywords(keyword):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        fsla.dense_state(**{keyword: "dense"})
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        DenseState(**{keyword: "dense"})


def test_a_fresh_state_is_unbound_and_carries_nothing():
    state = fsla.dense_state()
    assert not state.bound and not state.carries
    assert state.format is None and state.shape is None and state.device is None and state.dtype is None
    assert state.materialize() is None
    with pytest.raises(RuntimeError, match="carries nothing"):
        state._require_carrying()


def test_a_clone_is_an_independent_unbound_state_when_the_original_is_unbound():
    state = fsla.dense_state()
    copy = state.clone()
    assert copy is not state and isinstance(copy, FSLAState) and not copy.bound


def test_a_clone_of_a_carrying_state_owns_its_own_plane():
    state = fsla.dense_state()
    state._shape, state._device = (1, 1, 16, 65), torch.device("cpu")
    state._plane = torch.zeros(1, 1, 3168, dtype=torch.uint8)
    copy = state.clone()
    assert copy.bound and copy.carries and copy.shape == state.shape
    copy._plane.fill_(1)
    assert int(state._plane.sum()) == 0, "a clone aliased the original's plane"


def test_a_state_is_weak_referenceable_because_the_decode_scratch_is_keyed_on_it():
    import weakref

    state = fsla.dense_state()
    assert weakref.ref(state)() is state
