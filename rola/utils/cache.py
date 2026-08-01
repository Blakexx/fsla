# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from flash-linear-attention (`fla/layers/utils.py`),
# Copyright (c) 2023-2025 Songlin Yang, Yu Zhang, MIT licensed. See NOTICE.

"""Layer-cache plumbing: reading and updating one layer's slot in a KV-cache object.

Vendored rather than imported, and the reasoning is recorded because vendoring is
normally the wrong answer (extraction design §1.3 prefers cutting the dependency).
The two functions the RoLA layer calls are on its **unconditional** forward path,
so they cannot be hidden behind the optional `[fla]` import guard that
`ShortConvolution` uses. Their source file in the fork is 218 lines, of which RoLA
touches the last 19 — the rest is unpad/pad machinery with its own transitive
dependencies that RoLA never calls. Copying 25 self-contained lines is cheaper and
more honest than taking a hard dependency on a package to reach them.

These helpers are cache-implementation agnostic: they require only that the cache
object support `len()`, `__getitem__`, and `update(layer_idx=..., **kwargs)` —
which is what `transformers`' cache classes and the fork's `Cache` both provide.
"""

from __future__ import annotations

from typing import Any

_LAYER_IDX_REQUIRED_MSG = "{cls} requires `layer_idx` when `past_key_values` is provided."


def require_cache_layer_idx(module: Any, past_key_values: Any) -> int | None:
    """The module's `layer_idx`, insisting it exists whenever a cache is in play.

    A cache is indexed per layer; a module that was constructed without a
    `layer_idx` cannot address its own slot, and silently writing to slot 0 would
    corrupt another layer's state.
    """
    layer_idx = getattr(module, "layer_idx", None)
    if past_key_values is not None and layer_idx is None:
        raise ValueError(_LAYER_IDX_REQUIRED_MSG.format(cls=module.__class__.__name__))
    return layer_idx


def get_layer_cache(module: Any, past_key_values: Any) -> Any:
    """This layer's cached state, or None when there is no cache or no entry yet."""
    layer_idx = require_cache_layer_idx(module, past_key_values)
    if past_key_values is not None and len(past_key_values) > layer_idx:
        return past_key_values[layer_idx]
    return None


def update_layer_cache(module: Any, past_key_values: Any, **kwargs: Any) -> Any:
    """Write this layer's state back into the cache; a no-op when there is no cache."""
    layer_idx = require_cache_layer_idx(module, past_key_values)
    if past_key_values is not None:
        return past_key_values.update(layer_idx=layer_idx, **kwargs)
    return None


__all__ = ["require_cache_layer_idx", "get_layer_cache", "update_layer_cache"]
