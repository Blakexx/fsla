# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""`transformers` adapter: a `Cache` that is really a :class:`rola.state.RoLAState`.

Requires the optional `[transformers]` extra. Import this module only from code that
already depends on `transformers`; `rola` never does.

`transformers.generate()` type-checks `past_key_values` against its own `Cache` base
and calls `get_seq_length()` on it to build position ids. That is the ENTIRE contract
a RoLA stack needs from it -- there is no key/value tensor to concatenate, no window
to roll, no maximum length. So the adapter is a subclass that satisfies the base
class and delegates the state itself to the native object, rather than a
reimplementation of `RoLAState` in HF's vocabulary.

`update()` deliberately keeps the RoLA keyword signature (`recurrent_state`,
`conv_state`, `offset`) rather than HF's `(key_states, value_states, layer_idx)`.
The base class's signature is for growing key/value caches; a fixed-size recurrent
state has no key or value states to pass, and pretending otherwise by stuffing the
recurrent state into `key_states` would be a lie that the next `transformers`
release gets to break.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import Cache as _HFCache

from rola.state import LayerRecurrentState, RoLAState

__all__ = ["RoLAHFCache"]


class RoLAHFCache(_HFCache):
    """A `transformers.cache_utils.Cache` carrying RoLA's fixed-size state."""

    is_compileable = True

    def __init__(self, seen_tokens: int = 0) -> None:
        super().__init__()
        self.state = RoLAState(seen_tokens)

    # -- the duck type `rola/utils/cache.py` uses ---------------------------
    def __len__(self) -> int:
        return len(self.state)

    def __getitem__(self, layer_idx: int) -> LayerRecurrentState:
        return self.state[layer_idx]

    def __iter__(self):
        return iter(self.state)

    def update(  # type: ignore[override]
        self,
        *,
        layer_idx: int = 0,
        recurrent_state: torch.Tensor | None = None,
        conv_state: tuple[torch.Tensor | None, ...] | None = None,
        offset: int = 1,
    ) -> LayerRecurrentState:
        return self.state.update(
            layer_idx=layer_idx, recurrent_state=recurrent_state,
            conv_state=conv_state, offset=offset)

    # -- what `generate()` asks of a Cache ----------------------------------
    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        return self.state.get_seq_length(layer_idx)

    def get_max_length(self) -> int | None:
        """Unbounded: a fixed-size recurrent state has no capacity to run out of."""
        return None

    def get_max_cache_shape(self) -> int | None:
        return None

    def reset(self) -> None:
        self.state.reset()
