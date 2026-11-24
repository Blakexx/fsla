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

from fsla.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    ResolvedRouting,
    SoftmaxActivation,
    TiedRouting,
)


def activation_vjp(stored, cotangent, activation):
    """``dz`` of one activation from its stored output ``p`` and the incoming ``dp``, in fp64."""
    p, dp = stored.double(), cotangent.double()
    if isinstance(activation, SoftmaxActivation):
        return p * (dp - (p * dp).sum(dim=-1, keepdim=True))
    assert isinstance(activation, EntmaxActivation), activation
    s = torch.where(p > 0, p.pow(2.0 - activation.alpha), torch.zeros_like(p))
    total = s.sum(dim=-1, keepdim=True)
    return s * (dp - (s * dp).sum(dim=-1, keepdim=True) / torch.where(total > 0, total, torch.ones_like(total)))


def gradient_from_stored(routing: ResolvedRouting, reads, writes, read_cotangents, write_cotangents):
    """The gradient planes ``(d read logits, d write logits)``, fp64 ``[BH, T, packed_logit_width]``, a level at a time
    from the stored per-level outputs, and ``covered``: the logit spans of the levels this reference states. A tied
    level is one solve: its two cotangents add on the one stored value and land on the write plane; the read plane's
    span is zero."""
    shape = (*reads[0].shape[:2], routing.packed_logit_width)
    d_read = torch.zeros(shape, dtype=torch.float64, device=reads[0].device)
    d_write = torch.zeros_like(d_read)
    covered = []
    for index, level in enumerate(routing.levels):
        span = routing.level_logit_slice(index)
        if isinstance(level, TiedRouting):
            d_write[..., span] = activation_vjp(writes[index], read_cotangents[index].double()
                                                + write_cotangents[index].double(), level.activation("write"))
        elif isinstance(level, IndependentRouting):
            d_read[..., span] = activation_vjp(reads[index], read_cotangents[index], level.read)
            d_write[..., span] = activation_vjp(writes[index], write_cotangents[index], level.write)
        else:
            continue
        covered.append(span)
    return d_read, d_write, covered
