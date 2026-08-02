# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#
# CANONICAL fp64 reference for the decomposed entmax/sparsemax routing primitive (#8, M0).
#
# Independently re-derived from the standard entmax-family closed form (Peters, Niculae & Martins 2019,
# "Sparse Sequence-to-Sequence Models"; Blondel et al. 2019 backward-VJP) — NOT copied from rola-cute's
# `_entmax15`/`_sparsemax` (chunk.py there). Per the "matching-the-naive anti-pattern" memory: every gate
# in M0-M4 validates against THIS independent fp64 ground truth, never kernel-vs-kernel and never
# kernel-vs-rola-cute's-own-torch-fallback.
#
# Three stages (mirrors scoping.md section 1, and the in-kernel decomposition every later milestone
# builds toward):
#   Stage 1 - threshold tau(z, alpha): the unique tau such that
#             sum_i [(alpha-1)(z_i - tau)]_+^{1/(alpha-1)} == 1
#   Stage 2 - support = {i : z_i > tau}          (free byproduct of stage 1)
#   Stage 3 - values   p_i = [(alpha-1)(z_i - tau)]_+^{1/(alpha-1)}     (alpha=2 -> relu(z-tau), i.e. sparsemax)
#
# alpha=2 additionally gets a closed-form (sort-based) tau/support/values as the one "super-trivial free
# win" the user confirmed (no general sparsemax-specific kernel path — this closed form exists ONLY to
# cross-check bisection, per the user's decision that BISECTION is the uniform path for all alpha).
#
# BACKWARD (dz, fixed alpha - no learned-alpha here, that's M5/out of scope): the standard sparse entmax
# Jacobian-vector product. With s_i = p_i^(2-alpha) for i in support, 0 otherwise:
#     dz_i = s_i * dp_i  -  s_i * ( sum_j s_j * dp_j / sum_j s_j )
# (alpha=2 special case: s_i = 1 for i in support, so this reduces to dz_i = dp_i - mean_{j in support}(dp_j),
#  exactly the softmax fold's structural twin with the support indicator replacing the softmax weight fr and
#  a plain mean replacing the fr-weighted sum — see chunk.py's `_fold_level_weights`.)
#
# This module is pure torch, fp64, no Triton, no dependency on chunk.py — the standalone ground truth
# M1-M4's kernel primitives gate against.

from __future__ import annotations

import torch

__all__ = [
    "EntmaxFn",
    "entmax",
    "entmax_forward",
    "entmax_tau_bisection",
    "sparsemax_forward_closed_form",
]


def _support_mask(z: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Stage 2: support = {i : z_i > tau}, tau broadcast over the last dim."""
    return z > tau


@torch.no_grad()
def entmax_tau_bisection(
    z: torch.Tensor,
    alpha: float,
    dim: int = -1,
    n_iter: int = 60,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Stage 1, general alpha, bisection (the uniform path for ALL alpha, including alpha=2 — user
    decision: bisection everywhere, closed forms are only opportunistic free wins).

    z: [..., b] logits (any leading shape). alpha: python float > 1 (alpha->1 is softmax, NOT handled
    here — this primitive is for the strictly-sparse alpha>1 regime). mask: optional bool [..., b],
    True = valid entry (False entries are excluded from the support/threshold solve, mirroring the
    kernel's `bmask` padding past b). Returns tau: [..., 1] (fp64 if z is fp64).

    Bracket: tau in [max(z)/(alpha-1) - 1/(alpha-1), max(z)] is the standard tight bracket (at tau=max(z)
    only the argmax survives with p=0 in the limit; at the lower bound at least one full-mass solution
    exists). We use the simpler, always-valid bracket [min(z) - 1, max(z)] widened by `alpha` so bisection
    provably contains the root regardless of alpha, then refine with n_iter halvings — correctness of the
    bracket matters far more than its tightness for a REFERENCE implementation.
    """
    if mask is not None:
        neg_inf = torch.full_like(z, float("-inf"))
        z = torch.where(mask, z, neg_inf)
    else:
        mask = torch.ones_like(z, dtype=torch.bool)

    z_max = z.amax(dim=dim, keepdim=True)
    # hi=z_max always gives mass=0<=1 (support collapses to the argmax, whose clamp hits exactly 0).
    # lo must give mass>=1; the required gap scales with 1/(alpha-1) (small alpha-1 -> the pow exponent
    # 1/(alpha-1) is large -> need tau much lower before the sum of large-exponent terms exceeds 1).
    # The fixed gap below is a proved lower bracket even near alpha=1; no data-dependent widening is
    # required.
    hi = z_max.clone()
    lo = z_max - (1.0 + 10.0 / (alpha - 1.0))

    def _mass(tau: torch.Tensor) -> torch.Tensor:
        base = (alpha - 1.0) * (z - tau)
        p = torch.where(mask, torch.clamp(base, min=0.0).pow(1.0 / (alpha - 1.0)), torch.zeros_like(base))
        return p.sum(dim=dim, keepdim=True)

    # The initial bracket is already sufficient: its argmax term is
    # ((alpha-1) + 10)^(1/(alpha-1)) > 1 for every alpha > 1, hence mass(lo) > 1. The former
    # defensive widening loop could never update `lo`; in eager CUDA it nevertheless evaluated one
    # full mass tensor and synchronized it to the host through bool(any()). Keep the proved bracket as
    # the single production/reference contract and avoid that semantically dead work.

    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        too_much = _mass(mid) > 1.0
        lo = torch.where(too_much, mid, lo)
        hi = torch.where(too_much, hi, mid)
    return 0.5 * (lo + hi)


def entmax_forward(
    z: torch.Tensor,
    alpha: float,
    dim: int = -1,
    n_iter: int = 60,
    mask: torch.Tensor | None = None,
):
    """Full 3-stage forward (bisection tau), detached (no autograd) — the canonical fp64 forward.
    Returns (p, support, tau) with p/support the same shape as z, tau broadcastable ([...,1])."""
    tau = entmax_tau_bisection(z, alpha, dim=dim, n_iter=n_iter, mask=mask)
    support = _support_mask(z, tau)
    if mask is not None:
        support = support & mask
    base = (alpha - 1.0) * (z - tau)
    p = torch.where(support, torch.clamp(base, min=0.0).pow(1.0 / (alpha - 1.0)), torch.zeros_like(z))
    return p, support, tau


@torch.no_grad()
def sparsemax_forward_closed_form(z: torch.Tensor, dim: int = -1, mask: torch.Tensor | None = None):
    """alpha=2 closed form (sort-based), the ONE super-trivial free win (user decision). Used ONLY as a
    cross-check against the bisection path at alpha=2 — not a general kernel path.

    tau = (cumsum_k(z_sorted) - 1) / k  where k = support size found by the standard condition
    1 + k*z_(k) > cumsum_k (z_(k) the k-th largest, 1-indexed)."""
    if mask is not None:
        neg_inf = torch.full_like(z, float("-inf"))
        z_m = torch.where(mask, z, neg_inf)
    else:
        z_m = z
        mask = torch.ones_like(z, dtype=torch.bool)
    b = z.shape[dim]
    z_sorted, _ = torch.sort(z_m, dim=dim, descending=True)
    rho = torch.arange(1, b + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.dim()
    shape[dim if dim >= 0 else z.dim() + dim] = b
    rho = rho.reshape(shape)
    cumsum = z_sorted.cumsum(dim=dim)
    cond = 1.0 + rho * z_sorted > cumsum
    # k = count of True in cond (cond is a prefix of Trues by construction since z_sorted descending)
    k = cond.to(z.dtype).sum(dim=dim, keepdim=True).clamp(min=1.0)
    cumsum_k = torch.gather(cumsum, dim, (k.long() - 1).clamp(min=0))
    tau = (cumsum_k - 1.0) / k
    support = (z > tau) & mask
    p = torch.where(support, torch.clamp(z - tau, min=0.0), torch.zeros_like(z))
    return p, support, tau


class EntmaxFn(torch.autograd.Function):
    """Custom autograd.Function: forward = bisection-converged (p, support); backward = the ANALYTIC
    sparse entmax Jacobian-vector product (dz only — alpha is a fixed python float here, not a tensor
    requiring grad; learned-alpha's dalpha is M5, explicitly out of scope for M0-M4).

    This is deliberately NOT autograd-through-the-bisection-loop (that only approximates the true implicit
    derivative at finite n_iter and is O(n_iter) more expensive) — matches rola-cute's own documented
    caveat ("not differentiable through the bisection loop... gradient must come from the analytic sparse
    Jacobian"), independently re-derived here rather than reused.
    """

    @staticmethod
    def forward(ctx, z, alpha, dim, n_iter, mask):
        p, support, tau = entmax_forward(z, alpha, dim=dim, n_iter=n_iter, mask=mask)
        # s_i = p_i^(2-alpha) for i in support, 0 otherwise. alpha=2 -> s_i = p_i^0 = 1 on support
        # (well-defined by convention even where p_i==0 only at the support boundary, which cannot
        # happen strictly inside the support since p_i>0 there by construction).
        s = torch.where(support, p.clamp(min=0.0).pow(2.0 - alpha), torch.zeros_like(p))
        ctx.save_for_backward(support, s)
        ctx.dim = dim
        return p

    @staticmethod
    def backward(ctx, dp):
        support, s = ctx.saved_tensors
        dim = ctx.dim
        num = (s * dp).sum(dim=dim, keepdim=True)
        den = s.sum(dim=dim, keepdim=True).clamp(min=1e-30)
        dz = torch.where(support, s * dp - s * (num / den), torch.zeros_like(dp))
        return dz, None, None, None, None


def entmax(z: torch.Tensor, alpha: float, dim: int = -1, n_iter: int = 60, mask: torch.Tensor | None = None):
    """Public entry point: differentiable entmax/sparsemax with the analytic sparse Jacobian backward.
    alpha=2.0 is sparsemax. Forward uses bisection (uniform path); dtype follows z (use fp64 for gates)."""
    return EntmaxFn.apply(z, alpha, dim, n_iter, mask)
