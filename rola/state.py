# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The package's own recurrent-state object.

**A fixed-size recurrent state is not a KV cache**, and this module exists because
the difference is the architecture's whole point. A KV cache grows with the
sequence; RoLA's carried state is `[B, H, N, d_v + (norm == 'global')]` and
`[B, H*d, conv_size]` -- shapes that do not mention `T` at all. Every abstraction
`transformers` provides for caching (`DynamicCache`, `StaticCache`, window rolling,
`get_max_cache_shape`, legacy key/value tuples) is built for the growing case, and
borrowing one to hold a fixed-size state means inheriting concatenation, eviction,
and window bookkeeping that have no meaning here -- plus a hard dependency on
`transformers` for a kernel package (extraction design §3.9, ratified decision O-6,
which pins the hard dependencies at `torch` + `einops`).

So `RoLAState` is native, and it is deliberately the smallest object that can carry
one decode step across a call boundary. What it holds, and why each field is here:

* **`recurrent_state`** -- the routed-slot state itself, `[B, H, N, cols]`. This is
  the thing being carried. Its size is fixed by the topology, not by how much text
  has been read; that is the value proposition (`[[rola-competes-with-kv-cache-not-nc1]]`).
* **`conv_state`** -- the three short-convolution ring buffers `(v, write-route,
  read-route)`, each `conv_size` wide. Also fixed-size, and *also* genuinely part of
  the recurrence: a continuation that drops them re-starts the causal conv from
  zeros and computes a different function for the first `conv_size - 1` tokens. It
  is `None` when the layer runs without short convolutions.
* **`_seen_tokens`** -- a scalar step counter, advanced by `offset` on each update.
  The only quantity here that grows with the sequence, and it is an `int`.
  Generation loops ask a state object how many tokens it has consumed
  (`get_seq_length`) to build position ids; without it, a caller has to keep that
  tally beside the state, which is exactly the kind of split bookkeeping that goes
  wrong.

What is deliberately ABSENT, so that the absence is a decision rather than an
omission: `attn_state` / `ffn_state` (RoLA has neither), window sizes and rolling
eviction (nothing grows), `to_legacy_cache` / `from_legacy_cache` in the
key/value-tuple sense (there is no legacy format to be compatible with -- the
constructor takes no arguments and that is the whole "conversion"), and quantization
or offload hooks (a fixed-size state is small; if that changes it is a measured
increment, not a speculative hook).

Interoperating with `transformers` or `flash-linear-attention` is a THIN ADAPTER
behind an optional extra -- see `rola/integrations/`. The layer itself never imports
either package.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch

__all__ = ["LayerRecurrentState", "RoLAState", "as_layer_state"]


@dataclass(slots=True)
class LayerRecurrentState:
    """One layer's carried state. Attribute access only -- there is one spelling."""

    recurrent_state: torch.Tensor | None = None
    conv_state: tuple[torch.Tensor | None, ...] | None = None

    def detach(self) -> LayerRecurrentState:
        """A graph-free copy of this entry, for handing a state across a step boundary."""
        return LayerRecurrentState(
            recurrent_state=None if self.recurrent_state is None else self.recurrent_state.detach(),
            conv_state=None if self.conv_state is None else tuple(
                None if t is None else t.detach() for t in self.conv_state),
        )


class RoLAState:
    """The carried recurrent state of a stack of `RoLA` layers.

    Indexed by `layer_idx`, one :class:`LayerRecurrentState` per layer, in the order
    the layers first wrote to it. `len(state)` is the number of layers that have
    written, which is what makes `state[layer_idx]` on a never-written layer a
    "no prior state" answer rather than an error.
    """

    __slots__ = ("_layers", "_seen_tokens")

    def __init__(self, seen_tokens: int = 0) -> None:
        self._layers: list[LayerRecurrentState] = []
        self._seen_tokens = int(seen_tokens)

    # -- container protocol -------------------------------------------------
    def __len__(self) -> int:
        return len(self._layers)

    def __getitem__(self, layer_idx: int) -> LayerRecurrentState:
        if 0 <= layer_idx < len(self._layers):
            return self._layers[layer_idx]
        raise IndexError(
            f"RoLAState holds {len(self._layers)} layer(s); layer_idx={layer_idx} has no entry")

    def __iter__(self) -> Iterator[LayerRecurrentState]:
        return iter(self._layers)

    def __repr__(self) -> str:
        return f"RoLAState(layers={len(self._layers)}, seen_tokens={self._seen_tokens})"

    # -- the two operations the layer performs ------------------------------
    def update(
        self,
        *,
        layer_idx: int = 0,
        recurrent_state: torch.Tensor | None = None,
        conv_state: tuple[torch.Tensor | None, ...] | None = None,
        offset: int = 1,
    ) -> LayerRecurrentState:
        """Write layer `layer_idx`'s state back, and advance the token tally.

        `offset` is the number of tokens this call consumed. It is counted ONCE per
        step rather than once per layer: the tally advances on the first layer to
        write a fresh state object and on the last layer of an established one --
        the same rule the fork's cache used, kept because a per-layer tally would
        multiply the sequence position by the depth of the model.

        A `None` field leaves the stored one alone, so a layer without short
        convolutions never has to invent a `conv_state` to avoid clobbering one.
        """
        if layer_idx < 0:
            raise ValueError(f"layer_idx must be nonnegative, got {layer_idx}")
        if layer_idx > len(self._layers):
            raise ValueError(
                f"layer_idx={layer_idx} skips past the {len(self._layers)} layer(s) already "
                "written; RoLAState is filled in layer order, and a gap would make "
                "`len(state)` stop meaning 'layers with a prior state'")
        if layer_idx == len(self._layers):
            if layer_idx == 0:
                self._seen_tokens += int(offset)
            self._layers.append(LayerRecurrentState(recurrent_state, conv_state))
            return self._layers[layer_idx]
        if layer_idx == len(self._layers) - 1:
            self._seen_tokens += int(offset)
        entry = self._layers[layer_idx]
        if recurrent_state is not None:
            entry.recurrent_state = recurrent_state
        if conv_state is not None:
            entry.conv_state = conv_state
        return entry

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        """Tokens consumed so far; 0 before `layer_idx` has any entry."""
        if layer_idx is not None and len(self._layers) <= layer_idx:
            return 0
        return self._seen_tokens

    # -- lifecycle ----------------------------------------------------------
    def reset(self) -> None:
        """Drop every layer's state and the tally: the object is new again."""
        self._layers.clear()
        self._seen_tokens = 0

    def detach(self) -> RoLAState:
        """A graph-free copy, for carrying a state out of a step that built a graph."""
        out = RoLAState(self._seen_tokens)
        out._layers = [entry.detach() for entry in self._layers]
        return out


def as_layer_state(entry: Any) -> LayerRecurrentState | None:
    """Normalize one foreign cache entry to :class:`LayerRecurrentState`.

    The ONE adaptation point. `flash-linear-attention`'s cache stores a layer's
    state as a `dict` with `recurrent_state` / `conv_state` keys; this maps that
    onto the native object so the layer body has a single spelling. Anything that
    is already a `LayerRecurrentState` passes through untouched, so the native path
    costs one `isinstance`.
    """
    if entry is None or isinstance(entry, LayerRecurrentState):
        return entry
    if isinstance(entry, dict):
        return LayerRecurrentState(
            recurrent_state=entry.get("recurrent_state"),
            conv_state=entry.get("conv_state"))
    raise TypeError(
        f"a RoLA layer's cache entry must be a LayerRecurrentState or a mapping with "
        f"'recurrent_state'/'conv_state', got {type(entry).__name__}")
