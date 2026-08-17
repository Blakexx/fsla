# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The plan objects -- one per kernel family, built once at a DAG's terminal node.

A plan holds two kinds of field and no third: a HOST field is some rule's verdict,
and a TENSOR field is a reference to a terminal fact. A field that is neither means
something is still being decided at the launch site, which is the condition the facts
engine exists to remove (docs/internals/facts/facts.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from rola.facts.types import Arm, CallClass


@dataclass(frozen=True, slots=True)
class ChunkPlan:
    """Everything one chunk launch reads, and nothing it does not.

    NO FIELD IS `O(N * L)`. The two-pass reverse pass replays the schedule from the
    union table rather than from per-chunk snapshots, and that claim is a property of
    this object's field list (docs/internals/chunk/backward.md).
    """

    call_class: CallClass
    arm: Arm
    widths: tuple[int, ...]
    d_v: int
    v_row_bytes: int
    eps: float
    #: the OPERAND's dtype, which the divide narrows `y` back to. `v` below is the
    #: kernel's own bf16-contiguous copy and cannot answer for it.
    y_dtype: torch.dtype
    #: the bundle, for its LAYOUTS alone: the autograd node passes the routing
    #: tensors flat so the graph attaches to them, and the reverse pass needs each
    #: level's width and dtype to un-pack the digit gradients.
    routes: object
    read_plane: torch.Tensor
    write_plane: torch.Tensor
    g_write: torch.Tensor
    read_mass: torch.Tensor | None
    v: torch.Tensor
    union_table: torch.Tensor
    block_bits: torch.Tensor
    page_table: torch.Tensor | None
    state_in: torch.Tensor | None
    state_out: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class BackwardPlan:
    """THE CTX CONTRACT: a REFERENCE to the forward's plan, plus the one output kept.

    The reverse pass finds no facts of its own -- every fact it reads is one the
    forward already produced, so its plan IS the forward's plan, held by reference
    (docs/internals/facts/backward_dag.md). `den` is the second field because it is
    the forward's OUTPUT rather than one of its launch arguments: `execute` returns
    it, so it cannot be a `ChunkPlan` field without either un-freezing the plan or
    building the plan after the launch it configures. It is fed here, at retention
    time, which is also the moment the retention decision is actually made.

    THE NUMERATOR IS NOT HERE. `num` is `O(T * d_v)` per head and the two-pass design
    exists so that it need not be kept: the scans rebuild the state and REPLAY the
    schedule off the union table, so what the forward hands them is `O(L)` tables and
    `O(1)` scalars and nothing per-leaf-per-token. That claim is a property of this
    object's field list and of :meth:`replayed_from`, and
    `tests/integration/test_backward_ctx.py` is where it is gated rather than argued.
    """

    forward: ChunkPlan
    den: torch.Tensor

    def replayed_from(self) -> tuple[torch.Tensor | None, ...]:
        """The tensors the reverse pass READS -- the schedule it replays from.

        `read_mass` is `None` on a bundle that carries none; every other entry is
        present on any plan that ran.
        """
        f = self.forward
        return (f.read_plane, f.write_plane, f.g_write, f.read_mass, f.union_table,
                self.den)

    def retained_bytes(self) -> int:
        """What a grad-bearing call costs OVER an inference one.

        `v` and `y` are the op's own operand and output, alive either way and held by
        `save_for_backward`, so neither is counted. The plan reference also pins the
        forward's `block_bits` -- `[BH, N / BC]` bytes, carrying no `T` at all -- which
        is named and gated in `tests/integration/test_backward_ctx.py` rather than
        folded in here: this number is the `O(L)` closed form, and a term with no `L`
        in it would make it a number no closed form matches.
        """
        return sum(t.numel() * t.element_size()
                   for t in self.replayed_from() if t is not None)
