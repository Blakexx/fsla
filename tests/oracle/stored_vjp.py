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

THE UNION LEVEL HAS NO PUBLISHED CLOSED FORM: `union_vjp` is DERIVED from the fp64 reference forward's definition
(`fsla/routing/reference.py::_union_split_with_support`), not published and not the kernel's, and is held to the oracle's
autograd at unrounded values by `tests/unit/test_stored_vjp.py`. With ``z = (r + w) / 2``, ``p = entmax_alpha(z)`` on its
support ``S``, ``delta = r - w``, ``sig = sigmoid(delta)``, ``read = p * sig`` and
``write = softmax_S(log p + logsigmoid(-delta))``:

1. ``da = write * (d_write - sum_S(write * d_write))`` on ``S``, zero off it;
2. ``dp = d_read * sig + da / p`` on ``S``; ``d_delta = d_read * read * (1 - sig) - da * sig``;
3. ``s = p^(2 - alpha)`` on ``S``, ``dz = s * (dp - sum_S(s * dp) / sum_S(s))``;
4. ``d_r = dz / 2 + d_delta``, ``d_w = dz / 2 - d_delta``;

zero off the support and in a row with no support; ``d_delta`` is zero where the reference's clamp of ``delta`` saturates
(the clamp's own adjoint). THE STORED-VALUES EVALUATION is what "the same inputs" means for an operator whose backward
reads its stored outputs: ``read`` and ``write`` are the kernel's STORED bf16 outputs (widened), ``p`` is an ARGUMENT
(`union_vjp`'s), and ``delta`` and ``sig`` come from the same logits. WHICH ``p``: the unit test at unrounded values
passes the fp64 reference's own ``entmax_alpha(z)`` (`reference_midpoint`), so its comparison against autograd is exact;
a DEVICE gate passes the midpoint the KERNEL's forward saved (`saved_union_midpoints`), upcast, because that is the
backward kernel's actual input. ``da / p`` divides by ``p``, and near the support's boundary ``p`` is tiny, so the
kernel's fp32 rounding of it (about 1e-7 absolute) is a large RELATIVE difference from the fp64 ``p`` that nothing
damps at alpha 2.0 (``s = 1``); a reference dividing by its own ``p`` would not be reading the kernel's inputs.
"""
from __future__ import annotations

import torch

from fsla.routing.entmax.reference import entmax, entmax_forward
from fsla.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    ResolvedRouting,
    SoftmaxActivation,
    TiedRouting,
    UnionRouting,
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


def reference_midpoint(read_logits, write_logits, alpha):
    """The fp64 reference's own midpoint ``entmax_alpha((r + w) / 2)``: the ``p`` the unit test at unrounded values
    passes `union_vjp`."""
    return entmax(0.5 * read_logits.double() + 0.5 * write_logits.double(), alpha, dim=-1)


def _saved_node(outputs):
    """The autograd node of `_ProductionLevelsFn` behind one of the CUDA solve's outputs, found by walking ``grad_fn``
    and ``next_functions`` to the node that holds ``midpoint_offsets``."""
    pending = [output.grad_fn for output in outputs if getattr(output, "grad_fn", None) is not None]
    seen = set()
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        if hasattr(node, "midpoint_offsets"):
            return node
        pending.extend(next_node for next_node, _ in node.next_functions if next_node is not None)
    raise AssertionError("no `_ProductionLevelsFn` node behind the outputs: they are not the CUDA solve's outputs")


def saved_union_midpoints(routing, outputs):
    """``{level index: p}``, fp64, the midpoint the KERNEL's forward saved for each union level of ``routing``.

    ``outputs`` are the CUDA solve's graph outputs (its reads or writes). The forward returns no midpoint and the
    backward reads the saved one, so this reads what the backward reads: the node's ``saved_tensors`` hold
    ``(*read levels, *write levels, support words, read logits, write logits, midpoints)`` and the node's
    ``midpoint_offsets`` give each union level's column offset into ``midpoints`` (fp32, in union-level order)."""
    union_levels = [index for index, level in enumerate(routing.levels) if isinstance(level, UnionRouting)]
    if not union_levels:
        return {}
    node = _saved_node(outputs)
    midpoints = node.saved_tensors[2 * routing.D + 3]
    assert len(union_levels) == len(node.midpoint_offsets), "the node's union levels differ from the routing's"
    return {index: midpoints[..., offset:offset + routing.branches[index]].detach().double()
            for index, offset in zip(union_levels, node.midpoint_offsets)}


def union_vjp(read_logits, write_logits, alpha, p, d_read, d_write, read=None, write=None):
    """``(d r, d w)`` of the union split, fp64, from the derivation in the module docstring. ``p`` is the midpoint
    steps 1 to 3 divide by and raise to ``2 - alpha`` (which one a gate passes: the module docstring). ``read`` and
    ``write`` are the values steps 1 and 2 read: the stored outputs of the kernel, or (None) the reference forward's
    own, which is what the unit test holds to autograd."""
    r, w = read_logits.double(), write_logits.double()
    d_read, d_write = d_read.double(), d_write.double()
    z = 0.5 * r + 0.5 * w
    p = p.double()
    _, support, _ = entmax_forward(z, alpha, dim=-1)
    finfo = torch.finfo(torch.float64)
    raw = r - w
    delta = raw.clamp(min=-finfo.max, max=finfo.max)
    sig = torch.sigmoid(delta)
    safe_p = torch.where(support, p, torch.ones_like(p))
    if read is None:
        read = p * sig
    if write is None:
        log_write = (safe_p.log() + torch.nn.functional.logsigmoid(-delta)).masked_fill(~support, float("-inf"))
        has = support.any(dim=-1, keepdim=True)
        write = torch.where(has & support, torch.softmax(torch.where(has, log_write, torch.zeros_like(log_write)), -1),
                            torch.zeros_like(p))
    read, write = read.double(), write.double()
    zero = torch.zeros_like(p)
    wd = torch.where(support, write * d_write, zero)
    da = torch.where(support, write * d_write - write * wd.sum(dim=-1, keepdim=True), zero)
    dp = torch.where(support, d_read * sig + da / safe_p, zero)
    d_delta = torch.where(support, d_read * read * (1.0 - sig) - da * sig, zero)
    d_delta = torch.where(raw.abs() <= finfo.max, d_delta, zero)
    s = torch.where(support, safe_p.pow(2.0 - alpha), zero)
    total = s.sum(dim=-1, keepdim=True)
    dz = s * (dp - (s * dp).sum(dim=-1, keepdim=True) / torch.where(total > 0, total, torch.ones_like(total)))
    return 0.5 * dz + d_delta, 0.5 * dz - d_delta


def gradient_from_stored(routing: ResolvedRouting, reads, writes, read_cotangents, write_cotangents, read_logits=None,
                         write_logits=None, midpoints=None):
    """The gradient planes ``(d read logits, d write logits)``, fp64 ``[BH, T, packed_logit_width]``, a level at a time
    from the stored per-level outputs, and ``covered``: the logit spans of the levels this reference states. A tied
    level is one solve: its two cotangents add on the one stored value and land on the write plane; the read plane's
    span is zero. A union level is covered when the logits (bf16-exact, the ones the forward was handed) are given:
    `union_vjp` at the stored read and write, with ``midpoints[index]`` as its ``p`` (`saved_union_midpoints` on a
    device, `reference_midpoint` at unrounded values)."""
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
        elif isinstance(level, UnionRouting):
            if read_logits is None:
                continue
            d_read[..., span], d_write[..., span] = union_vjp(
                read_logits[..., span], write_logits[..., span], level.alpha, midpoints[index],
                read_cotangents[index], write_cotangents[index], reads[index], writes[index])
        else:
            continue
        covered.append(span)
    return d_read, d_write, covered
