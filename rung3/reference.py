"""fp64 references for the SCHEDULED routed chunked linear attention.

Rung 2's references unchanged (the recurrence, the block plan and the per-block
split) plus the three things a schedule adds:

``support_amplitudes`` -- amplitudes with EXACT ZEROS, a support of `k` digits
                          of each level drawn per token and renormalised over
                          the survivors. This is the shape an entmax producer
                          emits: a simplex row whose support is small and whose
                          survivors still sum to one.
``schedule``           -- the kernel's own membership test, written from the LAW
                          rather than from the kernel's constants: for each
                          (block, group), the AND over the levels the block does
                          not span entirely of "some spanned digit is nonzero",
                          OR-ed over the read and write sides and over the
                          group's tokens.
``schedule_exact``     -- the test that enumerates the block's slots and asks
                          whether ANY coefficient is nonzero. The prefix test
                          may only ever be a SUPERSET of this one; that is the
                          claim the gate checks, and the gap between them is
                          what the untested whole levels cost in skips.

A skipped group contributes the additive identity to every term, so the fp64
reference of a scheduled kernel IS rung 2's -- there is nothing to model. That
is the point of the rung and it is why `split` is imported rather than rewritten.
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from rung1.reference import (  # noqa: E402,F401
    amplitudes,
    chunked,
    chunked_bf16_state,
    digits,
    kron,
    kron_bf16,
    quadratic,
    split_levels,
    strides,
)
from rung2.reference import block_plan, block_slots, split  # noqa: E402,F401


def support_amplitudes(B, H, L, widths, support, device="cuda", seed=0,
                       support_seed=None):
    """Packed [B,H,L,sum(W_l)] bf16 amplitudes with a support of `support` digits.

    Per token and per level, `support` digits are drawn and the rest are EXACTLY
    ZERO; the survivors are renormalised so the row still sums to one.
    `support >= W_l` gives that level back dense, so one generator covers the
    dense end of the sweep too.

    `support_seed` names the draw that chooses WHICH digits survive, separately
    from the one that chooses their amplitudes. Passing the same `support_seed`
    to the read and write sides makes the routing TIED -- the two agree on where
    a token points and disagree only on how much -- which is a real producer
    configuration and the one under which a block's membership is scarcest.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    gs = torch.Generator(device=device).manual_seed(
        seed if support_seed is None else support_seed)
    parts = []
    for w in widths:
        x = torch.rand(B, H, L, w, generator=g, device=device, dtype=torch.float32) + 0.05
        k = min(support, w)
        if k < w:
            pick = torch.rand(B, H, L, w, generator=gs, device=device)
            keep = torch.zeros_like(x, dtype=torch.bool)
            keep.scatter_(-1, pick.topk(k, dim=-1).indices, True)
            x = torch.where(keep, x, torch.zeros_like(x))
        parts.append(x / x.sum(-1, keepdim=True))
    return torch.cat(parts, dim=-1).bfloat16().contiguous()


#: THE TOPOLOGY REGIMES, and why they are expressed at TWO levels.
#:
#: A slot index is a mixed-radix number and DOES NOT KNOW where the level
#: boundaries are: `D = 4` with widths (8,8,8,8) addresses exactly the same 4096
#: slots as `D = 2` with widths (64,64), and a level of width 64 whose amplitudes
#: are the outer product of two width-8 rows IS a pair of routing levels. So a
#: deeper topology enters here as a SPARSITY PATTERN on the two-level interface
#: -- the pattern that is itself a product -- and every reference, gate and
#: kernel path above stays exactly as it is.
#:
#: What that expression does NOT reproduce is amplitude TRAFFIC: a native
#: four-level kernel would load 8+8+8+8 amplitudes per token per side where this
#: one loads 64+64. Skip fractions, membership, mma counts and the arithmetic are
#: exactly the deeper topology's; wall is pessimistic by that traffic, and both
#: kernels of a comparison pay it identically, so a DELTA is still clean.
#:
#: A regime is, per kernel level, a list of `(width, read_support,
#: write_support)` sub-levels. WHERE a side is sparse relative to the block
#: boundary is the whole question: sparsity ABOVE the boundary confines a side to
#: few blocks, sparsity BELOW spreads it across every block at one slot each.
REGIMES = {
    "dense":      [[(64, 64, 64)], [(64, 64, 64)]],
    "lead1":      [[(64, 1, 1)], [(64, 64, 64)]],
    "lead4":      [[(64, 4, 4)], [(64, 64, 64)]],
    "split":      [[(64, 64, 1)], [(64, 1, 64)]],
    "d4":         [[(8, 8, 1), (8, 1, 8)], [(8, 8, 1), (8, 1, 8)]],
}


def regime_amplitudes(B, H, L, regime, device="cuda", seed=0):
    """(pr, pw) packed [B,H,L,128] bf16 for a named topology regime."""
    spec = REGIMES[regime] if isinstance(regime, str) else regime
    g = torch.Generator(device=device).manual_seed(seed)
    parts = [[], []]
    for level in spec:
        acc = [None, None]
        for (w, kr, kw) in level:
            #: ONE support draw per sub-level, shared by the two sides. Where
            #: both are sparse the routing is TIED -- the sides agree on which
            #: digit a token points at and differ only in how much -- which is
            #: the configuration under which a block's membership is scarcest.
            pick = torch.rand(B, H, L, w, generator=g, device=device)
            for i, k in enumerate((kr, kw)):
                x = torch.rand(B, H, L, w, generator=g, device=device,
                               dtype=torch.float32) + 0.05
                if k < w:
                    keep = torch.zeros_like(x, dtype=torch.bool)
                    keep.scatter_(-1, pick.topk(k, dim=-1).indices, True)
                    x = torch.where(keep, x, torch.zeros_like(x))
                x = (x / x.sum(-1, keepdim=True)).bfloat16().float()
                acc[i] = x if acc[i] is None else \
                    (acc[i].unsqueeze(-1) * x.unsqueeze(-2)).flatten(-2)
        for i in range(2):
            parts[i].append(acc[i])
    return tuple(torch.cat(p, dim=-1).bfloat16().contiguous() for p in parts)


def _nonzero(p):
    """bf16 nonzero, by the magnitude bits -- the kernel's own test."""
    return p.view(torch.int16).to(torch.int32).bitwise_and(0x7FFF) != 0


def schedule_sides(pr, pw, widths, bc: int):
    """(read, write), each [B,H,blocks,L] bool: the PREFIX test, PER SIDE.

    Written from the law: a level the block spans ENTIRELY is not tested (its
    clause is a property of the token, and a simplex row always has a survivor),
    a level spanned at one digit or at an aligned sub-range is tested by OR over
    that span, and the levels are AND-ed. The kernel schedules on the UNION of
    the two; they are returned separately because WHICH side is live decides
    which TERMS a live visit actually needs, which is what says whether a
    per-side schedule would be worth building.
    """
    plan, lead_span, tail_span = block_plan(widths, bc)
    _, w1 = widths
    B, H, L, _ = pr.shape
    out = []
    for p in (pr, pw):
        a0, a1 = (_nonzero(x) for x in split_levels(p, widths))
        t = torch.empty(B, H, len(plan), L, dtype=torch.bool, device=pr.device)
        for k, (lead_off, tail_off, _) in enumerate(plan):
            live = a0[..., lead_off:lead_off + lead_span].any(-1)
            if tail_span < w1:
                live = live & a1[..., tail_off:tail_off + tail_span].any(-1)
            t[:, :, k] = live
        out.append(t)
    return tuple(out)


def schedule_tokens(pr, pw, widths, bc: int):
    """[B,H,blocks,L] bool: the union the kernel actually schedules on."""
    r, w = schedule_sides(pr, pw, widths, bc)
    return r | w


def schedule(pr, pw, widths, bc: int, chunk: int):
    """[B,H,blocks,groups] bool: a group is live when any of its tokens is."""
    tok = schedule_tokens(pr, pw, widths, bc)
    B, H, nb, L = tok.shape
    return tok.reshape(B, H, nb, L // chunk, chunk).any(-1)


def schedule_exact(pr, pw, widths, bc: int, chunk: int):
    """[B,H,blocks,groups] bool: ANY nonzero coefficient over the block's slots.

    The definition, by slot enumeration over the exact fp64 coefficient. What
    the kernel skips must be a subset of what this says is dead.
    """
    B, H, L, _ = pr.shape
    tab = block_slots(widths, bc, pr.device)
    R, W = kron(pr, widths), kron(pw, widths)
    out = torch.empty(B, H, tab.shape[0], L // chunk, dtype=torch.bool, device=pr.device)
    for k in range(tab.shape[0]):
        tok = (R[..., tab[k]] != 0).any(-1) | (W[..., tab[k]] != 0).any(-1)
        out[:, :, k] = tok.reshape(B, H, L // chunk, chunk).any(-1)
    return out
