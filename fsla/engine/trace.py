# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The profiler range of one step of a call (docs/internals/engine/engine.md#the-profiler-range)."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def span(name: str) -> Iterator[None]:
    """An NVTX range named ``name``, only where a CUDA context exists: a profiling aid, never a correctness dependency."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()
