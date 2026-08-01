# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The native stateful surface (`rola.state`), and the optional adapters onto it.

`RoLAState` is a FIXED-SIZE recurrent state, not a KV cache, and most of what is
asserted here is about that distinction being structural: nothing in the object
grows with `T` except an `int` counter, and the object needs no third-party package
to exist.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from rola.state import LayerRecurrentState, RoLAState, as_layer_state


def _state(n: int = 4, cols: int = 5) -> torch.Tensor:
    return torch.arange(n * cols, dtype=torch.float32).reshape(1, 1, n, cols)


def test_the_state_object_needs_nothing_but_torch():
    """The whole reason it is native: `rola.state` must import with no optional
    package present. Checked by inspecting the module's own imports rather than by
    uninstalling anything, which a test cannot do."""
    import rola.state as module

    source = module.__file__
    with open(source) as handle:
        text = handle.read()
    for forbidden in ("import transformers", "from transformers", "import fla", "from fla"):
        assert forbidden not in text, f"rola/state.py must not reference {forbidden!r}"


def test_an_empty_state_reports_no_prior_entry_rather_than_raising():
    state = RoLAState()
    assert len(state) == 0
    assert state.get_seq_length(0) == 0
    with pytest.raises(IndexError):
        state[0]


def test_the_first_write_creates_the_layer_entry_and_reads_back_by_attribute():
    state = RoLAState()
    recurrent = _state()
    conv = (torch.zeros(1, 3, 4), None, None)
    entry = state.update(layer_idx=0, recurrent_state=recurrent, conv_state=conv, offset=7)
    assert len(state) == 1
    assert entry is state[0]
    assert state[0].recurrent_state is recurrent
    assert state[0].conv_state is conv
    assert state.get_seq_length(0) == 7


def test_a_none_field_leaves_the_stored_one_alone():
    """A layer without short convolutions must not have to invent a `conv_state`
    to avoid clobbering the one it does not use."""
    state = RoLAState()
    conv = (torch.zeros(1, 3, 4),)
    state.update(layer_idx=0, recurrent_state=_state(), conv_state=conv, offset=1)
    second = _state() + 1.0
    state.update(layer_idx=0, recurrent_state=second, conv_state=None, offset=1)
    assert state[0].conv_state is conv
    assert state[0].recurrent_state is second


def test_the_token_tally_counts_the_step_once_not_once_per_layer():
    """The counter is a sequence position. Counting it per layer would multiply the
    position by the depth of the model."""
    state = RoLAState()
    for step in range(3):
        for layer_idx in range(4):
            state.update(layer_idx=layer_idx, recurrent_state=_state(), offset=5)
        assert state.get_seq_length(0) == 5 * (step + 1)
    assert len(state) == 4


def test_writing_a_layer_out_of_order_is_refused_rather_than_leaving_a_gap():
    state = RoLAState()
    state.update(layer_idx=0, recurrent_state=_state())
    with pytest.raises(ValueError, match="skips past"):
        state.update(layer_idx=2, recurrent_state=_state())
    with pytest.raises(ValueError, match="nonnegative"):
        state.update(layer_idx=-1, recurrent_state=_state())


def test_nothing_in_the_state_grows_with_the_sequence():
    """The claim the class is named for. Two steps of different token counts leave
    every stored tensor at exactly the shape the first step produced."""
    state = RoLAState()
    conv = (torch.zeros(1, 3, 4),)
    state.update(layer_idx=0, recurrent_state=_state(), conv_state=conv, offset=128)
    shapes = (state[0].recurrent_state.shape, state[0].conv_state[0].shape)
    state.update(layer_idx=0, recurrent_state=_state(), conv_state=conv, offset=1024)
    assert (state[0].recurrent_state.shape, state[0].conv_state[0].shape) == shapes


def test_reset_makes_the_object_new_again():
    state = RoLAState()
    state.update(layer_idx=0, recurrent_state=_state(), offset=9)
    state.reset()
    assert len(state) == 0 and state.get_seq_length(0) == 0


def test_detach_severs_the_graph_without_moving_the_values():
    state = RoLAState()
    leaf = _state().requires_grad_(True)
    state.update(layer_idx=0, recurrent_state=leaf * 2, conv_state=(leaf * 3,), offset=1)
    detached = state.detach()
    assert state[0].recurrent_state.requires_grad
    assert not detached[0].recurrent_state.requires_grad
    assert not detached[0].conv_state[0].requires_grad
    assert torch.equal(detached[0].recurrent_state, state[0].recurrent_state.detach())
    assert detached.get_seq_length(0) == state.get_seq_length(0)


# --- the ONE adaptation point ------------------------------------------------

def test_a_foreign_dict_entry_is_normalized_to_the_native_entry():
    """`flash-linear-attention`'s cache stores a layer's state as a dict. That is the
    only shape `as_layer_state` accepts besides the native one, and the native one
    passes through by identity so the native path costs one isinstance."""
    native = LayerRecurrentState(recurrent_state=_state())
    assert as_layer_state(native) is native
    assert as_layer_state(None) is None

    recurrent, conv = _state(), (torch.zeros(1, 3, 4),)
    adapted = as_layer_state(
        {"recurrent_state": recurrent, "conv_state": conv,
         "attn_state": "ignored", "ffn_state": "ignored"})
    assert adapted.recurrent_state is recurrent and adapted.conv_state is conv

    with pytest.raises(TypeError, match="LayerRecurrentState or a mapping"):
        as_layer_state(object())


def test_a_dict_entry_missing_the_rola_keys_reads_as_no_prior_state():
    adapted = as_layer_state({})
    assert adapted.recurrent_state is None and adapted.conv_state is None


# --- optional extras ---------------------------------------------------------

_HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
_HAS_FLA = importlib.util.find_spec("fla") is not None


@pytest.mark.skipif(not _HAS_TRANSFORMERS, reason="the [transformers] extra is not installed")
def test_the_hf_adapter_is_an_hf_cache_that_speaks_the_rola_update_signature():
    from transformers.cache_utils import Cache as HFCache

    from rola.integrations.hf import RoLAHFCache

    cache = RoLAHFCache()
    assert isinstance(cache, HFCache), "generate() type-checks past_key_values against Cache"
    recurrent = _state()
    cache.update(layer_idx=0, recurrent_state=recurrent, conv_state=None, offset=6)
    assert cache[0].recurrent_state is recurrent
    assert cache.get_seq_length(0) == 6
    assert cache.get_max_cache_shape() is None, "a fixed-size state has no capacity limit"
    cache.reset()
    assert len(cache) == 0


@pytest.mark.skipif(not _HAS_FLA, reason="the [fla] extra is not installed")
def test_an_fla_cache_reaches_the_layer_body_through_the_same_normalization():
    """FLA's `Cache` already accepts RoLA's update keywords and stores dict entries,
    so the adapter is `as_layer_state` and nothing else. Asserted against the real
    class rather than a stand-in, because "it already fits" is the claim."""
    from fla.models.utils import Cache as FLACache

    cache = FLACache()
    recurrent = _state()
    conv = (torch.zeros(1, 3, 4), None, None)
    cache.update(layer_idx=0, recurrent_state=recurrent, conv_state=conv, offset=4)
    entry = as_layer_state(cache[0])
    assert entry.recurrent_state is recurrent and entry.conv_state is conv
