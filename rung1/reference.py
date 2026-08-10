"""fp64 references for ROUTED-KRONECKER chunked linear attention.

The base's three references, extended at exactly one point: the coefficient.
A slot index n in [0, N) is a mixed-radix number over the level widths, most
significant digit first, and

    coeff[t][n] = prod_l p[l][t][digit_l(n)] ,  digit_l(n) = (n / stride_l) % W_l

with stride_l = prod_{m > l} W_m.  ``kron`` builds that by the DEFINITION (an
explicit divide and modulo over slots, i.e. the leaf-major form) so that it is
an independent check of the kernel's digit-major loop nest, not a restatement
of it.

Two coefficient models, because the routed form introduces a quantisation the
dense base did not have -- the product of two bf16 amplitudes is not a bf16:

  ``kron``       -- the exact fp64 product.  The DESIGN's coefficient.
  ``kron_bf16``  -- the product rounded to bf16, which is what any bf16 mma
                    operand must be.  The kernel's coefficient.
"""
from __future__ import annotations

import torch


def strides(widths):
    s, acc = [], 1
    for w in reversed(widths):
        s.append(acc)
        acc *= w
    return list(reversed(s)), acc


def digits(widths, device):
    """[N, D] int table of digit_l(n), built from the definition (div then mod)."""
    st, n_slots = strides(widths)
    n = torch.arange(n_slots, device=device)
    return torch.stack([(n // s) % w for s, w in zip(st, widths)], dim=-1), n_slots


def split_levels(p, widths):
    """Packed [..., sum(W_l)] -> list of [..., W_l]."""
    out, off = [], 0
    for w in widths:
        out.append(p[..., off:off + w])
        off += w
    return out


def kron(p, widths):
    """Exact fp64 routed coefficient [B,H,T,N] from packed amplitudes."""
    dg, n_slots = digits(widths, p.device)
    levels = split_levels(p.double(), widths)
    c = torch.ones(*p.shape[:-1], n_slots, dtype=torch.float64, device=p.device)
    for l, pl in enumerate(levels):
        c = c * pl[..., dg[:, l]]
    return c


def kron_bf16(p, widths):
    """The kernel's coefficient: the level product rounded to bf16, then fp64."""
    dg, n_slots = digits(widths, p.device)
    levels = split_levels(p, widths)
    c = None
    for l, pl in enumerate(levels):
        sel = pl[..., dg[:, l]]
        c = sel if c is None else (c.float() * sel.float()).bfloat16()
    return c


def quadratic(r, w, v):
    r, w, v = (t.double() for t in (r, w, v))
    s = (r @ w.transpose(-1, -2)).tril()
    return s @ v


def chunked(r, w, v, chunk: int):
    r, w, v = (t.double() for t in (r, w, v))
    B, H, L, N = r.shape
    S = torch.zeros(B, H, N, v.shape[-1], dtype=torch.float64, device=r.device)
    out = torch.empty_like(v)
    tri = torch.ones(chunk, chunk, dtype=torch.bool, device=r.device).tril()
    for c in range(0, L, chunk):
        rc, wc, vc = r[..., c:c + chunk, :], w[..., c:c + chunk, :], v[..., c:c + chunk, :]
        o = rc @ S
        o = o + torch.where(tri, rc @ wc.transpose(-1, -2), 0.0) @ vc
        out[..., c:c + chunk, :] = o
        S = S + wc.transpose(-1, -2) @ vc
    return out


def chunked_bf16_state(r, w, v, chunk: int):
    r, w, v = (t.double() for t in (r, w, v))
    B, H, L, N = r.shape
    S = torch.zeros(B, H, N, v.shape[-1], dtype=torch.float64, device=r.device)
    out = torch.empty_like(v)
    tri = torch.ones(chunk, chunk, dtype=torch.bool, device=r.device).tril()
    for c in range(0, L, chunk):
        rc, wc, vc = r[..., c:c + chunk, :], w[..., c:c + chunk, :], v[..., c:c + chunk, :]
        o = rc @ S.bfloat16().double()
        g = torch.where(tri, rc @ wc.transpose(-1, -2), 0.0).bfloat16().double()
        o = o + g @ vc
        out[..., c:c + chunk, :] = o
        S = S + wc.transpose(-1, -2) @ vc
    return out


def amplitudes(B, H, L, widths, device="cuda", seed=0):
    """Packed [B,H,L,sum(W_l)] bf16 amplitudes, simplex-normalised per level.

    Real data the kernel must load: positive, per-level normalised, and varying
    per token, so nothing here is foldable into a constant.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    parts = []
    for w in widths:
        x = torch.rand(B, H, L, w, generator=g, device=device, dtype=torch.float32) + 0.05
        parts.append(x / x.sum(-1, keepdim=True))
    return torch.cat(parts, dim=-1).bfloat16().contiguous()
