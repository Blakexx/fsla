# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""SEGMENTATION: how a layer cuts a long prefill into steps, each one call of the op, the state carried between them.

The op runs one segment and knows nothing of segmentation (docs/api.md section 4.2). The layer is handed a SEGMENT
HANDLE, a function from the call's context to a MAXIMUM of tokens a step may hold, and the plan of a call is made once,
by one host function, from the sequence's length and that maximum::

    layer = fsla.FSLA(routing, hidden_size=..., num_heads=..., segments=fsla.max_tokens(8192))
    layer = fsla.FSLA(routing, hidden_size=..., num_heads=..., segments=fsla.max_memory(1 << 30))
    plan = fsla.segment_plan(length, maximum)     # the steps and the tokens a step

A caller never needs the card's window: a handle states a maximum, and the plan takes the largest whole number of
windows under it. A sequence of any length is served: one no longer than the maximum is ONE step, a partial window
included (the kernels run a partial last window; the caller pads nothing). A sequence that must be split is split on
window boundaries only -- tokens of one window interact, and a boundary on a window boundary moves no pair -- so a
maximum below one window cannot serve it, and that plan is REFUSED, naming the window.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

__all__ = ["SegmentContext", "SegmentHandle", "SegmentPlan", "max_memory", "max_tokens", "segment_plan"]


@dataclass(frozen=True, slots=True)
class SegmentContext:
    """WHAT A HANDLE IS TOLD of a call, none of it data: the digits of each level, which levels each side can hold a
    mask on (a bit a level, the read side first: a level-side whose activation cannot make exact zeros has none), each
    side's batch-heads (batch times the side's heads), the device, and the card's window in tokens."""

    widths: tuple[int, ...]
    masked: tuple[int, int]
    heads: tuple[int, int]
    device: torch.device
    window: int


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


def segment_plan(length: int, maximum: int | None, *, window: int | None = None) -> SegmentPlan:
    """THE PLAN, made once a call: ``length`` tokens under ``maximum`` tokens a step (``None``: no limit).

    A sequence no longer than the maximum is one step. A longer one is cut into steps of the largest whole number of
    windows under the maximum, the last step shorter; where the maximum is below one window no cut exists, and the plan
    is refused. ``window`` is the card's window in tokens; left out, it is the current device's
    (:func:`fsla.ops.window.window_of`)."""
    _positive("length", length)
    if maximum is None or length <= _positive("maximum", maximum):
        return SegmentPlan(steps=1, tokens=length, length=length)
    if window is None:
        from fsla.ops.window import window_of

        window = window_of()
    _positive("window", window)
    tokens = (maximum // window) * window
    if tokens == 0:
        raise ValueError(
            f"a sequence of {length} tokens under a maximum of {maximum} must be split, and a split sits on a window "
            f"boundary: the window is {window} tokens, so no step of at most {maximum} exists. The least maximum that "
            f"serves a sequence longer than itself is {window}.")
    return SegmentPlan(steps=-(-length // tokens), tokens=tokens, length=length)


@dataclass(frozen=True, slots=True)
class _MaxTokens:
    tokens: int

    def __call__(self, context: SegmentContext) -> int:
        return self.tokens


@dataclass(frozen=True, slots=True)
class _MaxMemory:
    limit: int

    def __call__(self, context: SegmentContext) -> int:
        """The most whole windows whose WORST-CASE workspace fits the limit: the workspace is sized from the call's shape
        alone, at the most leaf groups the shape allows, so the bound holds whatever the data."""
        one = workspace_bytes(context, context.window)
        if one > self.limit:
            raise ValueError(
                f"no segmentation serves a memory limit of {self.limit} bytes on this call: one window of {context.window} "
                f"tokens needs a workspace of {one} bytes at its worst case (levels of {list(context.widths)} digits, "
                f"masked levels {list(context.masked)}, batch-heads {list(context.heads)}).")
        windows = self.limit // one
        #: the layout rounds a block to a line, so the bytes of `windows` windows are checked, not assumed linear
        while windows > 1 and workspace_bytes(context, windows * context.window) > self.limit:
            windows -= 1
        return windows * context.window


def workspace_bytes(context: SegmentContext, tokens: int) -> int:
    """THE OP'S WORKSPACE for a segment of ``tokens`` tokens of this call, in bytes, at its worst case: the planner's own
    figure (``mask_prepare_bytes``), a function of the shape alone."""
    from fsla.ops._ext import extension

    return int(extension().mask_prepare_bytes([int(w) for w in context.widths], [int(m) for m in context.masked],
                                              [int(h) for h in context.heads], int(tokens), context.device.index))


def max_tokens(tokens: int) -> SegmentHandle:
    """A handle that allows a step at most ``tokens`` tokens."""
    return _MaxTokens(_positive("tokens", tokens))


def max_memory(limit_bytes: int) -> SegmentHandle:
    """A handle that allows a step whose workspace, at its worst case, is at most ``limit_bytes`` bytes. It bounds what
    the op allocates for a segment (the prepared masks of the most leaf groups the call's shape allows); the state and
    the segment's own projected inputs are beside it. Where one window does not fit, the call is refused."""
    return _MaxMemory(_positive("limit_bytes", limit_bytes))
