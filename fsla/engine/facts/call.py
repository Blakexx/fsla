# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The gradient clause of a prefill call, read off the operands a caller holds.

The rules fold VALUES -- that is what makes them testable with no device present --
so something has to read those values off the objects a caller actually holds. This
is that adapter and nothing else: it finds the facts and decides nothing of its own
(docs/internals/engine/engine.md). The prefill's other refusals are made when its descriptor is
(docs/internals/hardware/registry.md#the-owed-folder).
"""
from __future__ import annotations

from typing import Any

from fsla.engine.facts.operands import operands_of, requires_backward
from fsla.engine.types import BackwardNotImplemented


def refuse_backward(routes, values, decay: Any = None) -> None:
    """A GRADIENT IS REFUSED, loudly: the prefill has no reverse pass until the native backward is rebuilt, and
    a forward reached from a grad context must raise rather than sever the graph silently (the layer-dispatch
    autograd gap). THE ONE GRADIENT CLAUSE of a prefill, called by the layer before it makes a descriptor and by
    the op's fast-fail."""
    if requires_backward(*operands_of(routes, values, decay)):
        raise BackwardNotImplemented(
            "a FSLA PREFILL carrying a gradient: the prefill has no reverse pass until the native backward is "
            "rebuilt, and running it would sever the graph silently. Run under torch.no_grad(), freeze the "
            "operands, or train through the fp64 oracle (fsla.ops.naive.naive_fsla).")
