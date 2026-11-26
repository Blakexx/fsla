# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""SEGMENTATION: how a layer cuts a long prefill into steps, each one call of the op, the state carried between them.

The op runs one segment and knows nothing of segmentation (docs/api.md section 4.2). The layer is handed a SEGMENT
HANDLE, a function from the call's context to a MAXIMUM of tokens a step may hold, and the plan of a call is made once,
by the planner (``csrc/fsla/src/common/plan/segments.h``), from the sequence's length, that maximum and the card's window::

    layer = fsla.FSLA(routing, hidden_size=..., num_heads=..., segments=fsla.max_tokens(8192))
    layer = fsla.FSLA(routing, hidden_size=..., num_heads=..., segments=fsla.max_memory(1 << 30))
    plan = fsla.segment_plan(length, maximum)     # the steps and the tokens a step

A caller never needs the card's window, and this module reads none: a handle states a maximum, and the planner, which
reads the card, takes the largest whole number of windows under it. A sequence of any length is served: one no longer than the maximum is ONE step, a partial window
included (the kernels run a partial last window; the caller pads nothing). A sequence that must be split is split on
window boundaries only -- tokens of one window interact, and a boundary on a window boundary moves no pair -- so a
maximum below one window cannot serve it, and that plan is REFUSED, naming the window.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

__all__ = ["SegmentContext", "SegmentHandle", "SegmentPlan", "max_memory", "max_tokens", "segment_plan", "step_bytes"]


@dataclass(frozen=True, slots=True)
class SegmentContext:
    """WHAT A HANDLE IS TOLD of a call, none of it data: the digits of each level, which levels each side can hold a
    mask on (a bit a level, the read side first: a level-side whose activation cannot make exact zeros has none), each
    side's batch-heads (batch times the side's heads), the values' channels a head, and the device."""

    widths: tuple[int, ...]
    masked: tuple[int, int]
    heads: tuple[int, int]
    value_width: int
    device: torch.device


#: A SEGMENT HANDLE: the call's context in, the most tokens a step may hold out (``None``: no limit, one step)
SegmentHandle = Callable[[SegmentContext], "int | None"]


@dataclass(frozen=True, slots=True)
class SegmentPlan:
    """THE PLAN OF A CALL: ``steps`` calls of the op, each of ``tokens`` tokens but the last, which holds the rest."""

    steps: int
    tokens: int
    length: int

    @property
    def last(self) -> int:
        """The last step's tokens."""
        return self.length - (self.steps - 1) * self.tokens

    def bounds(self) -> list[tuple[int, int]]:
        """Each step's ``(start, end)`` in the sequence."""
        return [(s * self.tokens, min((s + 1) * self.tokens, self.length)) for s in range(self.steps)]


def _positive(name: str, value) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} is a positive exact int, got {value!r}")
    return value


def segment_plan(length: int, maximum: int | None, *, device=None) -> SegmentPlan:
    """THE PLAN, made once a call: ``length`` tokens under ``maximum`` tokens a step (``None``: no limit) on ``device``
    (the current CUDA device where left out).

    The planner's answer (``fsla_segment_plan``): a sequence no longer than the maximum is one step; a longer one is cut
    into steps of the largest whole number of the card's windows under the maximum, the last step shorter; where the
    maximum is below one window no cut exists, and the planner refuses the plan, naming the window."""
    _positive("length", length)
    if maximum is None:
        return SegmentPlan(steps=1, tokens=length, length=length)
    from fsla.ops._ext import device_index, extension

    steps, tokens = extension().fsla_segment_plan(length, _positive("maximum", maximum), device_index(device))
    return SegmentPlan(steps=int(steps), tokens=int(tokens), length=length)


@dataclass(frozen=True, slots=True)
class _MaxTokens:
    tokens: int

    def __call__(self, context: SegmentContext) -> int:
        return self.tokens


@dataclass(frozen=True, slots=True)
class _MaxMemory:
    limit: int

    def __call__(self, context: SegmentContext) -> int:
        """The most whole windows of a step whose bytes fit the limit: the planner's answer (``fsla_step_tokens``), from the
        call's shape alone, so the bound holds whatever the data."""
        from fsla.ops._ext import device_index, extension

        return int(extension().fsla_step_tokens(*_shape(context), self.limit, device_index(context.device)))


def _shape(context: SegmentContext) -> tuple[list[int], list[int], list[int], int]:
    return ([int(w) for w in context.widths], [int(m) for m in context.masked], [int(h) for h in context.heads],
            int(context.value_width))


def step_bytes(context: SegmentContext, tokens: int) -> tuple[int, int]:
    """WHAT A STEP of ``tokens`` tokens of this call holds beside the state, in bytes, as the planner counts it
    (``fsla_step_bytes``, in whole windows): the op's workspace at its worst case, then its projected inputs (each side's
    rows and scales, the masks, the values)."""
    from fsla.ops._ext import device_index, extension

    workspace, operands = extension().fsla_step_bytes(*_shape(context), _positive("tokens", tokens), device_index(context.device))
    return int(workspace), int(operands)


def max_tokens(tokens: int) -> SegmentHandle:
    """A handle that allows a step at most ``tokens`` tokens."""
    return _MaxTokens(_positive("tokens", tokens))


def max_memory(limit_bytes: int) -> SegmentHandle:
    """A handle that allows a step whose bytes are at most ``limit_bytes``: the op's workspace at its worst case (the
    prepared masks of the most leaf groups the call's shape allows) and the step's projected inputs (:func:`step_bytes`).
    The state, the same however the call is cut, and the call's output, assembled whole, are beside it. Where one window
    does not fit, the call is refused."""
    return _MaxMemory(_positive("limit_bytes", limit_bytes))
