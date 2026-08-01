# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
#
# The layer-indexing shape (a `layer_idx` slot, `update(layer_idx=..., **kwargs)`)
# follows flash-linear-attention's `fla/layers/utils.py`, Copyright (c) 2023-2025
# Songlin Yang, Yu Zhang, MIT licensed. See NOTICE. No code is copied: the state
# object being addressed is `rola.state.RoLAState`, which is this package's own.

"""Reading and writing one layer's slot in a :class:`~rola.state.RoLAState`.

Two functions, on the layer's unconditional forward path, kept in their own module
because they are the entire coupling between `RoLA.forward` and the stateful
surface. The state object is duck-typed on purpose -- `len()`, `__getitem__`, and
`update(layer_idx=..., recurrent_state=..., conv_state=..., offset=...)` -- which is
what lets the optional `rola/integrations/` adapters present a `transformers` or
`flash-linear-attention` cache to the same layer body without the layer importing
either package. `rola.state.as_layer_state` normalizes whatever `__getitem__`
returns, so the layer reads attributes and never a dict key.
"""

from __future__ import annotations

from typing import Any

from rola.state import LayerRecurrentState, as_layer_state

_LAYER_IDX_REQUIRED_MSG = "{cls} requires `layer_idx` when `past_key_values` is provided."


def require_cache_layer_idx(module: Any, past_key_values: Any) -> int | None:
    """The module's `layer_idx`, insisting it exists whenever a state is in play.

    A state object is indexed per layer; a module constructed without a `layer_idx`
    cannot address its own slot, and silently writing to slot 0 would corrupt
    another layer's state.
    """
    layer_idx = getattr(module, "layer_idx", None)
    if past_key_values is not None and layer_idx is None:
        raise ValueError(_LAYER_IDX_REQUIRED_MSG.format(cls=module.__class__.__name__))
    return layer_idx


def get_layer_cache(module: Any, past_key_values: Any) -> LayerRecurrentState | None:
    """This layer's carried state, or None when there is none or no entry yet."""
    layer_idx = require_cache_layer_idx(module, past_key_values)
    if past_key_values is not None and len(past_key_values) > layer_idx:
        return as_layer_state(past_key_values[layer_idx])
    return None


def update_layer_cache(module: Any, past_key_values: Any, **kwargs: Any) -> Any:
    """Write this layer's state back; a no-op when there is no state object."""
    layer_idx = require_cache_layer_idx(module, past_key_values)
    if past_key_values is not None:
        return past_key_values.update(layer_idx=layer_idx, **kwargs)
    return None


__all__ = ["require_cache_layer_idx", "get_layer_cache", "update_layer_cache"]
