# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE GRADIENT REFERENCE FOR A BACKWARD THAT READS STORED VALUES: the operator's mathematical vector-Jacobian product,
evaluated in fp64 at the SAME stored values the kernel's backward reads (the forward's own bf16 outputs, widened), so
both sides evaluate one nonlinear function at one point and what remains is the kernel's fp32 arithmetic and the
rounding of its result. The gradient against the oracle's autograd at the unrounded logits is REPORTED, never gated.

The closed forms are the published ones, not the kernels': they are held to the fp64 oracle's own autograd at unrounded
values by `tests/unit/test_stored_vjp.py`.

- softmax (``p = softmax(z)``): ``dz = p * (dp - sum(p * dp))``.
- alpha-entmax (Peters, Niculae, Martins 2019, "Sparse Sequence-to-Sequence Models", Prop. 2; Blondel, Martins,
  Niculae 2020, "Learning with Fenchel-Young losses", the entmax Jacobian): on the support ``s = p^(2 - alpha)`` and
  ``dz = s * (dp - sum(s * dp) / sum(s))``, zero off the support.
"""
from __future__ import annotations

import torch

from fsla.routing.projection import RouterLayout
from fsla.routing.types import EntmaxActivation, SoftmaxActivation


def activation_vjp(stored, cotangent, activation):
    """``dz`` of one activation from its stored output ``p`` and the incoming ``dp``, in fp64."""
    p, dp = stored.double(), cotangent.double()
    if isinstance(activation, SoftmaxActivation):
        return p * (dp - (p * dp).sum(dim=-1, keepdim=True))
    assert isinstance(activation, EntmaxActivation), activation
    s = torch.where(p > 0, p.pow(2.0 - activation.alpha), torch.zeros_like(p))
    total = s.sum(dim=-1, keepdim=True)
    return s * (dp - (s * dp).sum(dim=-1, keepdim=True) / torch.where(total > 0, total, torch.ones_like(total)))


def gradient_from_stored(layout: RouterLayout, reads, writes, read_cotangents, write_cotangents):
    """``(d logits, covered)``: the gradient into the projection's logits, fp64 ``[S, T, packed_router_width]``, a solve
    at a time from the stored per-level outputs (``[S, T, width]`` a write level, ``[S * G, T, width]`` a read level,
    read head ``g`` at rows ``s * G + g``), and ``covered``: the column spans of the solves this reference states. A
    tied read entry is the write solve's: its cotangent adds to the write side's on the one stored value."""
    S, T, G = writes[0].shape[0], writes[0].shape[1], layout.group

    def head(t, g):
        return t.reshape(S, G, *t.shape[1:])[:, g]

    d_logits = torch.zeros((S, T, layout.packed_router_width), dtype=torch.float64, device=writes[0].device)
    covered = []
    for index in range(layout.D):
        span = layout.write_columns(index)
        cotangent = write_cotangents[index].double()
        for g in layout.tied_heads(index):
            cotangent = cotangent + head(read_cotangents[index], g).double()
        d_logits[..., span] = activation_vjp(writes[index], cotangent, layout.topology.write[index])
        covered.append(span)
        for g, entries in enumerate(layout.heads):
            span = layout.read_columns(g, index)
            if span is not None:
                d_logits[..., span] = activation_vjp(head(reads[index], g), head(read_cotangents[index], g),
                                                     entries[index])
                covered.append(span)
    return d_logits, covered
