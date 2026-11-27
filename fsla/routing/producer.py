# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE ROUTING A LAYER COMPUTES, DECLARED: its topology, its activations, the write gain and the projection bias.

FSLA is linear attention under a structured feature map, and the feature map is declared, not supplied: a
:class:`Routing` is DATA, the layer owns the weights it implies, and the producer (:func:`fsla.ops.produce.produce_prefill`)
computes the bundle from them in one call. The format and its producer are ours, so a routing the registry does not
hold is added to the registry -- its activation, its solve, its reference and its oracle -- never plugged in beside
it (docs/api.md section 2).

**THE TOPOLOGY IS DECLARED ONCE.** ``Routing(topology=(64, 64), write=(entmax(1.5), softmax()), read=(softmax(),
entmax(1.5)))``: the widths, outermost level first, then one flat activation a write level and the read heads' tuples --
one a read head of the group, or one for every head -- each entry an activation or :func:`~fsla.routing.types.tied`.
Depth, leaf count, the packed router width and the logit offsets are all DERIVED -- none of them is a second input that
could disagree with the declaration.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from functools import cache

import torch
import torch.nn as nn

from fsla.routing.projection import RouterLayout
from fsla.routing.side_gain import GainConfig
from fsla.routing.types import RoutingActivation, Tied, Topology

__all__ = ["Routing"]


@dataclass(frozen=True)
class Routing:
    """THE DECLARATION: the topology, its activations, the write gain and the projection bias. Data only.

    Args:
        topology: one width a level, outermost first, a tuple of ints; it becomes the
            :class:`~fsla.routing.types.Topology` the declaration states, read back as ``routing.topology``.
        write: one flat activation a level (``softmax()``, ``entmax(alpha)``); none is chained.
        read: one tuple a read head of the group (``G`` of them, ``G`` the layer's query heads a write head), or ONE
            tuple every head reads by; each entry a level's activation or ``tied()``, which reads the write side's
            routing on that level. The read side states no widths: they are the topology's.
        gain: whether the write-side gain is learned. ``False`` removes its projection
            column rather than pinning it to 1, so no dead parameter is optimized.
        gain_bias_init: initialize the gain bias so an all-zero-logit gain starts at
            this value. ``None`` leaves it at zero (unit gain at zero logits).
        gain_config: the research surface behind ``gain``, importable
            from :mod:`fsla.routing.side_gain` and deliberately not part of the contract.
        bias: a learned bias on the routing projection.

    Construction validates the whole declaration -- every activation a registry row, nonnegative and one the producer
    solves, every read tuple the topology's depth and a width of at least 2 -- so an unbuildable model fails here and
    never inside a forward. A level whose untied read and whose write both make exact zeros is two solves with two
    supports, which nothing makes overlap; it is allowed. The read
    count against the layer's group is :meth:`layout`'s.
    """

    topology: Topology
    write: InitVar[tuple]
    read: InitVar[tuple]
    gain: bool = True
    gain_bias_init: float | None = None
    gain_config: GainConfig | None = None
    bias: bool = False

    def __post_init__(self, write, read) -> None:
        widths = self.topology
        if not isinstance(widths, tuple) or not all(type(w) is int for w in widths):
            raise TypeError(f"Routing.topology is the level widths, a tuple of ints, outermost first; got {widths!r}")
        if not isinstance(write, tuple):
            raise TypeError(f"Routing.write is one activation a level, a tuple; got {write!r}")
        if not isinstance(read, tuple) or not read:
            raise TypeError(f"Routing.read is one tuple a read head of the group, or one tuple for every head; got {read!r}")
        if not isinstance(read[0], tuple):
            raise TypeError(f"Routing.read holds TUPLES, one a read head: a single head's tuple is `read=({read!r},)`")
        topology = Topology(widths=widths, write=write, read=read)
        if any(width < 2 for width in widths):
            raise ValueError(f"a routed level is at least 2 wide, got {widths!r}")
        for level, activation in enumerate(topology.write):
            _solved(activation, f"Routing.write[{level}]")
        for head, entries in enumerate(topology.read):
            for level, entry in enumerate(entries):
                if type(entry) is Tied:
                    continue
                _solved(entry, f"Routing.read[{head}][{level}]")
        if type(self.gain) is not bool:
            raise TypeError(f"gain must be a bool, got {self.gain!r}")
        if type(self.bias) is not bool:
            raise TypeError(f"bias must be a bool, got {self.bias!r}")
        object.__setattr__(self, "topology", topology)

    @property
    def widths(self) -> tuple[int, ...]:
        """One branch width per level, in level order."""
        return self.topology.widths

    def layout(self, group: int) -> RouterLayout:
        """The packed projection layout at ``group`` read heads a write head, the layer's ``G``. Refuses a read count
        that is neither ``G`` nor one, naming both."""
        return _layout(self.topology, group)

    @property
    def gain_layout(self) -> GainConfig:
        return self.gain_config if self.gain_config is not None else GainConfig.from_public(self.gain)

    @property
    def gain_columns(self) -> int:
        """The projection columns the write gain takes, per write head: its trailing span of ``route_W``."""
        return len(self.gain_layout.layout)

    def columns(self, group: int) -> int:
        """``route_W``'s columns a write head at ``group`` read heads: every level's slots, then the gain's."""
        return self.layout(group).packed_router_width + self.gain_columns


@cache
def _layout(topology: Topology, group: int) -> RouterLayout:
    return RouterLayout(topology=topology, group=group)


def _solved(activation, owner: str) -> None:
    if not isinstance(activation, RoutingActivation):
        raise TypeError(f"{owner} is an activation the producer solves (softmax(), entmax(alpha)) or, on the read side, "
                        f"tied(); got {activation!r}")


def _init_packed_router(W: nn.Parameter) -> None:
    """Init ``[H, hidden, packed_router_width]`` like an ``nn.Linear(hidden, cols)``.

    The routing logit is ``h @ W[head]``, so the init must be a TRANSPOSED linear
    weight with ``fan_in == hidden``. ``kaiming_uniform_(a=sqrt(5))`` on ``[out,
    hidden]`` draws ``U(-1/sqrt(hidden), +1/sqrt(hidden))`` per element -- the bound
    does not depend on ``out`` -- so every level gets exactly the per-level fan-in
    semantics the projection GEMM has, one dot product over the full hidden dim per
    column.
    """
    H, hidden, cols = W.shape
    wt = torch.empty(H * cols, hidden, dtype=W.dtype, device=W.device)
    nn.init.kaiming_uniform_(wt, a=5 ** 0.5)
    W.data.copy_(wt.view(H, cols, hidden).transpose(-1, -2))
