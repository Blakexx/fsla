# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""EnvelopeRule -- the operand-side clause of the built surface: the arm is a CUDA kernel (docs/api.md §1).
The one device refusal a call meets, whichever surface it comes through (docs/internals/engine/rules.md)."""
from __future__ import annotations

from fsla.engine import rule


@rule(gives=("envelope",), needs=("device",))
def envelope(device) -> str | None:
    """The operand-side clause: a CPU tensor is named BEFORE any question about the built matrix is asked of a
    device that may not have one. No activation is refused here: the op is agnostic to what produced its operands."""
    if device.type != "cuda":
        return f"the operands are on {device}; the arm is a CUDA kernel"
    return None
