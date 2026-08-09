"""fp64 references for dense causal linear attention.

Three of them, deliberately independent:

* ``quadratic``   -- the definition, O[t] = sum_{s<=t} (q_t.k_s) v_s, as one
  masked matmul. No chunking anywhere.
* ``chunked``     -- the algorithm the kernel implements, with an explicit
  carried state. Agreement with ``quadratic`` is the ALGORITHM check.
* ``chunked_bf16_state`` -- the same, but with the state rounded to bf16 at
  every readout, exactly as any bf16 kernel must. Agreement with the kernel is
  the IMPLEMENTATION check; the gap between the two is the price of the design.
"""
from __future__ import annotations

import torch


def quadratic(q, k, v):
    q, k, v = (t.double() for t in (q, k, v))
    B, H, L, _ = q.shape
    s = q @ k.transpose(-1, -2)
    s = s.tril()
    return s @ v


def chunked(q, k, v, chunk: int):
    q, k, v = (t.double() for t in (q, k, v))
    B, H, L, D = q.shape
    S = torch.zeros(B, H, D, v.shape[-1], dtype=torch.float64, device=q.device)
    out = torch.empty_like(v)
    tri = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril()
    for c in range(0, L, chunk):
        qc, kc, vc = q[..., c:c + chunk, :], k[..., c:c + chunk, :], v[..., c:c + chunk, :]
        o = qc @ S
        o = o + torch.where(tri, qc @ kc.transpose(-1, -2), 0.0) @ vc
        out[..., c:c + chunk, :] = o
        S = S + kc.transpose(-1, -2) @ vc
    return out


def chunked_bf16_state(q, k, v, chunk: int):
    q, k, v = (t.double() for t in (q, k, v))
    B, H, L, D = q.shape
    S = torch.zeros(B, H, D, v.shape[-1], dtype=torch.float64, device=q.device)
    out = torch.empty_like(v)
    tri = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril()
    for c in range(0, L, chunk):
        qc, kc, vc = q[..., c:c + chunk, :], k[..., c:c + chunk, :], v[..., c:c + chunk, :]
        o = qc @ S.bfloat16().double()
        g = torch.where(tri, qc @ kc.transpose(-1, -2), 0.0).bfloat16().double()
        o = o + g @ vc
        out[..., c:c + chunk, :] = o
        S = S + kc.transpose(-1, -2) @ vc
    return out


def errors(got, ref):
    got, ref = got.double(), ref.double()
    d = (got - ref).abs()
    denom = ref.abs().clamp_min(1e-12)
    return dict(max_abs=d.max().item(), max_rel=(d / denom).max().item(),
                rms_rel=((d / denom) ** 2).mean().sqrt().item(),
                ref_absmax=ref.abs().max().item())
