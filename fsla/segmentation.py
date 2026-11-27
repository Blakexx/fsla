# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""SEGMENTATION: how a layer cuts a long prefill into steps, each one call of the op, over head blocks and token ranges.

The op runs one segment and knows nothing of segmentation (docs/api.md section 4.2). The layer is handed a SEGMENT POLICY,
the most tokens and the most write heads a step may hold (``None`` on an axis: no limit there), or a SEGMENT HANDLE, a
function from the call's context to one, and the plan of a call is made once, by the planner
(``csrc/fsla/src/common/plan/segments.h``), from the sequence's length, its batch and write heads, the policy and the
card's window::

    layer = fsla.FSLA(routing, hidden_size=..., num_heads=..., segments=fsla.SegmentPolicy(max_tokens=8192, max_heads=16))
    layer = fsla.FSLA(routing, hidden_size=..., num_heads=..., segments=fsla.max_memory(1 << 30))
    plan = fsla.segment_plan(length, policy, batch=..., heads=...)     # the head blocks, and the token steps in each

A caller never needs the card's window, and this module reads none: a policy states maxima, and the planner, which reads
the card, takes the largest whole number of windows under the token maximum. A sequence of any length is served: one no
longer than the maximum is ONE step, a partial window included (the kernels run a partial last window; the caller pads
nothing). A sequence that must be split is split on window boundaries only -- tokens of one window interact, and a
boundary on a window boundary moves no pair -- so a maximum below one window cannot serve it, and that plan is REFUSED,
naming the window. Head blocks are independent: the state is batch-major with write heads inside a sequence, so a block is
a run of whole sequences, or a range of one sequence's write heads, and its rows of the state are contiguous.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

__all__ = ["HeadBlock", "SegmentContext", "SegmentHandle", "SegmentPlan", "SegmentPolicy", "max_memory", "max_tokens",
           "segment_plan", "step_bytes"]


def _positive(name: str, value) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} is a positive exact int, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class SegmentPolicy:
    """THE MOST A STEP MAY HOLD: ``max_tokens`` tokens and ``max_heads`` WRITE heads (the state's owners; a step takes
    every query head reading them and never splits a write head's query group), each a positive exact int or ``None``,
    no limit on that axis."""

    max_tokens: int | None = None
    max_heads: int | None = None

    def __post_init__(self) -> None:
        for name in ("max_tokens", "max_heads"):
            value = getattr(self, name)
            if value is not None:
                _positive(name, value)


#: THE LAYER'S POLICY when it is given none: 64 windows a step of every head, revisited when the prefill is first timed
DEFAULT_SEGMENTS = SegmentPolicy(max_tokens=65_536)


@dataclass(frozen=True, slots=True)
class SegmentContext:
    """WHAT A HANDLE IS TOLD of a call, none of it data: the digits of each level, which levels each side can hold a
    mask on (a bit a level, the read side first: a level-side whose activation cannot make exact zeros has none), the
    batch, the write heads and the query heads a sequence, the values' channels a head, and the device."""

    widths: tuple[int, ...]
    masked: tuple[int, int]
    batch: int
    write_heads: int
    query_heads: int
    value_width: int
    device: torch.device


#: A SEGMENT HANDLE: the call's context in, the policy of its steps out (``None``: no limit, one op call)
SegmentHandle = Callable[[SegmentContext], "SegmentPolicy | None"]


@dataclass(frozen=True, slots=True)
class HeadBlock:
    """ONE HEAD BLOCK of a call: the sequences ``batch`` and, of each, the write heads ``heads``, each ``(start, end)``.
    A block is whole sequences or a range of one sequence's write heads, so its rows of the batch-major state are one
    contiguous run."""

    batch: tuple[int, int]
    heads: tuple[int, int]


@dataclass(frozen=True, slots=True)
class SegmentPlan:
    """THE PLAN OF A CALL of ``length`` tokens of ``batch`` sequences of ``heads`` write heads: head blocks of
    ``block_sequences`` whole sequences, or of ``block_heads`` write heads of one sequence; in each, ``steps`` calls of the
    op of ``tokens`` tokens but the last, which holds the rest. Blocks are independent; a block's steps run in order,
    carrying its state."""

    length: int
    batch: int
    heads: int
    steps: int
    tokens: int
    block_sequences: int
    block_heads: int

    @classmethod
    def whole(cls, length: int, batch: int, heads: int) -> SegmentPlan:
        """The plan of one op call: one block of every head, one step of every token."""
        return cls(length=length, batch=batch, heads=heads, steps=1, tokens=length, block_sequences=batch, block_heads=heads)

    @property
    def last(self) -> int:
        """The last step's tokens."""
        return self.length - (self.steps - 1) * self.tokens

    def bounds(self) -> list[tuple[int, int]]:
        """Each step's ``(start, end)`` in the sequence."""
        return [(s * self.tokens, min((s + 1) * self.tokens, self.length)) for s in range(self.steps)]

    def blocks(self) -> list[HeadBlock]:
        """Each head block, batch-major: the sequences first, then a sequence's write heads."""
        return [HeadBlock(batch=(b, min(b + self.block_sequences, self.batch)), heads=(h, min(h + self.block_heads, self.heads)))
                for b in range(0, self.batch, self.block_sequences) for h in range(0, self.heads, self.block_heads)]


def segment_plan(length: int, policy: SegmentPolicy, *, batch: int, heads: int, device=None) -> SegmentPlan:
    """THE PLAN, made once a call: ``length`` tokens of ``batch`` sequences of ``heads`` write heads under ``policy`` on
    ``device`` (the current CUDA device where left out).

    The planner's answer (``fsla_segment_plan``). Heads: where the head maximum holds one sequence's write heads a block is
    the most whole sequences under it, else the most of one sequence's write heads, the last block of a sequence holding
    the rest. Tokens: a sequence no longer than the token maximum is one step; a longer one is cut into steps of the
    largest whole number of the card's windows under it, the last step shorter; where the maximum is below one window no
    cut exists, and the planner refuses the plan, naming the window. A call no maximum cuts is one op call and asks no card."""
    for name, value in (("length", length), ("batch", batch), ("heads", heads)):
        _positive(name, value)
    if type(policy) is not SegmentPolicy:
        raise TypeError(f"a segment plan is made under a fsla.SegmentPolicy, got {type(policy).__name__}")
    whole_tokens = policy.max_tokens is None or length <= policy.max_tokens
    whole_heads = policy.max_heads is None or batch * heads <= policy.max_heads
    if whole_tokens and whole_heads:
        return SegmentPlan.whole(length, batch, heads)
    from fsla.ops._ext import device_index, extension

    steps, tokens, sequences, block_heads = extension().fsla_segment_plan(
        length, policy.max_tokens or 0, policy.max_heads or 0, batch, heads, device_index(device))
    return SegmentPlan(length=length, batch=batch, heads=heads, steps=int(steps), tokens=int(tokens),
                       block_sequences=int(sequences), block_heads=int(block_heads))


@dataclass(frozen=True, slots=True)
class _MaxMemory:
    limit: int

    def __call__(self, context: SegmentContext) -> SegmentPolicy:
        """The planner's policy under the limit (``fsla_step_policy``), from the call's shape alone, so the bound holds
        whatever the data: tokens first, the most whole windows of every head; only where one window of every head does
        not fit, one window and the most write heads."""
        from fsla.ops._ext import device_index, extension

        tokens, heads = extension().fsla_step_policy(*_shape(context), self.limit, device_index(context.device))
        return SegmentPolicy(max_tokens=int(tokens), max_heads=int(heads) or None)


def _shape(context: SegmentContext) -> tuple[list[int], list[int], list[int], int]:
    """The planner's step shape of a call: each side's batch-heads, the read (query) side first."""
    return ([int(w) for w in context.widths], [int(m) for m in context.masked],
            [context.batch * context.query_heads, context.batch * context.write_heads], int(context.value_width))


def step_bytes(context: SegmentContext, tokens: int) -> tuple[int, int]:
    """WHAT A STEP of ``tokens`` tokens of every head of this call holds beside the state, in bytes, as the planner counts
    it (``fsla_step_bytes``, in whole windows): the op's workspace at its worst case, then its projected inputs (each
    side's rows and scales, the masks, the values)."""
    from fsla.ops._ext import device_index, extension

    workspace, operands = extension().fsla_step_bytes(*_shape(context), _positive("tokens", tokens), device_index(context.device))
    return int(workspace), int(operands)


def max_tokens(tokens: int) -> SegmentPolicy:
    """The policy of at most ``tokens`` tokens a step, every head."""
    return SegmentPolicy(max_tokens=tokens)


def max_memory(limit_bytes: int) -> SegmentHandle:
    """A handle that allows a step whose bytes are at most ``limit_bytes``: the op's workspace at its worst case (the
    prepared masks of the most leaf groups the call's shape allows) and the step's projected inputs (:func:`step_bytes`).
    Tokens first: the most whole windows whose step of every head fits; only where one window of every head does not
    fit, one window of the most write heads that fit. The state, the same however the call is cut, and the call's output,
    assembled whole, are beside it. Where one window of one write head does not fit, the call is refused."""
    return _MaxMemory(_positive("limit_bytes", limit_bytes))
