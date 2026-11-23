# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The engine's value types: the MMA k-quantum and the refusal every no-backward clause raises (docs/internals/engine/engine.md)."""
from __future__ import annotations

#: the MMA k-extent, and the page granule. The Python mirror of
#: the MMA k-extent (16, the ampere atom's `m16n8k16`, `common/atoms/mma.cuh`), which a C++
#: ``constexpr`` cannot supply here; :meth:`fsla.ops.paging.PageArena.__init__` checks
#: it against the atom geometry it addresses.
MMA_K_QUANTUM = 16


class BackwardNotImplemented(NotImplementedError):
    """Raised when a gradient is demanded of the forward-only kernel surface."""
