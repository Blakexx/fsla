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
