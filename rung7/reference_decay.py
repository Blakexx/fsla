"""THE DECAY-AWARE FLOOR MODEL — the rung-7 kernel's own arithmetic, in fp64.

Mirrors the kernel's rounding sites exactly: bf16 level-chained coefficients
(kron_bf16), the UNGAINED clock read from the bf16 write coefficient, fp64
decay-to-date running products standing in for the kernel's fp32/EX2 ones
(that approximation is ~1e-7 relative and sits far under the bf16 floor this
model measures), dressed panels rounded to bf16, bf16 Gram, per-slot end
scale on the accumulator.  The PARTITION is the walk's own, via `cook`.

This is a FLOOR model, not the correctness reference — `naive_rola` with
`LeafMassDecay` is the oracle; this measures what the kernel's designed
quantization is expected to cost against it.  Rung-local until the static
clock lands in the shared reference (then both move together).
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from rung1.reference import kron_bf16  # noqa: E402
from rung5b.reference import plans_for, with_ones  # noqa: E402


def compacted_decay_state(rb, wu, g, v1, plan, rate):
    """One block's decayed recurrence over a compacted partition.

    ``rb``: bf16-rounded read coefficients (fp64), block's slot columns.
    ``wu``: bf16-rounded UNGAINED write coefficients — the clock's input.
    ``g``: per-token gain (fp64, [.., L]).  ``rate``: scalar fp64 per-head.
    """
    S = torch.zeros(*rb.shape[:-2], rb.shape[-1], v1.shape[-1],
                    dtype=torch.float64, device=rb.device)
    out = torch.zeros(*rb.shape[:-1], v1.shape[-1], dtype=torch.float64,
                      device=rb.device)
    gam = 1.0 - rate
    for r_idx, w_idx, thr in plan:
        if not r_idx and not w_idx:
            continue
        if w_idx:
            wuc = wu[..., w_idx, :]                     # [.., nw, S]
            #: the decay-to-date over the COMPACTED write list, inclusive.
            T = torch.pow(gam, wuc.cumsum(dim=-2))      # [.., nw, S]
            gc = g[..., w_idx].unsqueeze(-1)
            wd = (gc * wuc / T).bfloat16().double()     # dressed W~, bf16
            vc = v1[..., w_idx, :]
        if r_idx:
            rc = rb[..., r_idx, :]
            if w_idx:
                #: T at each read's causal threshold (1 where the threshold
                #: is zero) -- the merged scan's gather, batched.
                idx = torch.tensor(thr, device=rb.device).clamp_min(1) - 1
                Tr = T[..., idx, :]
                Tr = torch.where(
                    torch.tensor(thr, device=rb.device).view(
                        *([1] * (T.dim() - 2)), -1, 1) > 0,
                    Tr, torch.ones_like(Tr))
                rd = (rc * Tr).bfloat16().double()      # dressed R~, bf16
            else:
                rd = rc
            o = rd @ S.bfloat16().double()
            if w_idx:
                col = torch.arange(len(w_idx), device=rb.device)
                keep = col.unsqueeze(0) < torch.tensor(
                    thr, device=rb.device).unsqueeze(1)
                gr = torch.where(keep, rd @ wd.transpose(-1, -2),
                                 0.0).bfloat16().double()
                o = o + gr @ vc
            out[..., r_idx, :] = o
        if w_idx:
            #: S_next = T_end * (S_a + sum W~ v) -- one per-slot scale.
            S = T[..., -1:, :].transpose(-1, -2) * (S + wd.transpose(-1, -2) @ vc)
    return out


def model_compacted_decay(pr, pw, gw, v, widths, chunk, bc, rate):
    """The kernel's arithmetic over the whole cell, d_v+1 columns, decay on.

    ``rate`` is fp32 [H]; ``gw`` bf16 [B,H,L].
    """
    rb = kron_bf16(pr, widths).double()
    wu = kron_bf16(pw, widths).double()   # UNGAINED: the clock reads this
    g64 = gw.double()
    v1 = with_ones(v)
    B, H = pr.shape[0], pr.shape[1]
    out = torch.empty(B, H, pr.shape[2], v.shape[-1] + 1, dtype=torch.float64,
                      device=pr.device)
    n = rb.shape[-1]
    for b in range(B):
        for h in range(H):
            plans = plans_for(pr, pw, widths, bc, chunk, b, h)
            acc = None
            for k, s0 in enumerate(range(0, n, bc)):
                part = compacted_decay_state(
                    rb[b:b + 1, h:h + 1, :, s0:s0 + bc],
                    wu[b:b + 1, h:h + 1, :, s0:s0 + bc],
                    g64[b:b + 1, h:h + 1], v1[b:b + 1, h:h + 1], plans[k],
                    float(rate[h]))
                acc = part if acc is None else acc + part
            out[b:b + 1, h:h + 1] = acc
    return out
