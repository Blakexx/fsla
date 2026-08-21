# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""CHECK 1(c): the off-support cotangent seam, on the shipped binary.

Runs the shipped prefill reverse (carry scans + intra reverse) on a flagship ALT draw,
captures the digit-gradient planes d_pr / d_pw at the producer seam, zeroes the
OFF-SUPPORT entries of the two entmax-fed planes, and pushes both the raw and the
zeroed cotangent through the shipped entmax VJP kernel. Equality of the resulting logit
gradients is the seam confirmation. A softmax duty is the negative control.
"""
from __future__ import annotations

import torch

from rola.ops import carry as carry_ops
from rola.ops import intra as intra_ops
from rola.ops.prefill import prefill_autograd
from rola.routing.entmax.production import production_factor

D, B, WINDOW, T, DV = 2, 256, 512, 1024, 64
K, M = 4, 8
MODES = (intra_ops.WRITE_SPARSE, intra_ops.READ_SPARSE)   # ALT
WIDTHS = (B,) * D


def stamps():
    got = (carry_ops.build_stamp(WIDTHS, K, M, WINDOW, 32, carry_ops.CHANNEL_SPLIT)[0],
           carry_ops.backward_stamp(WIDTHS, K, M, WINDOW, 32)[0])
    assert got == (carry_ops.BUILD_MAGIC, carry_ops.BWD_BUILD_MAGIC), f"stale binary: {got}"
    print(f"device stamps: carry fwd {got[0]}, carry bwd {got[1]}, "
          f"intra fwd {intra_ops.build_stamp()}, intra bwd {intra_ops.backward_stamp()}")


def sparse_logits(gen, scale):
    return (scale * torch.randn(1, T, B, device="cuda", generator=gen)).requires_grad_(True)


def main():
    stamps()
    gen = torch.Generator(device="cuda").manual_seed(31)

    #: the two ENTMAX duties under ALT: level 0's WRITE side, level 1's READ side.
    z_w0 = sparse_logits(gen, 4.35)
    z_r1 = sparse_logits(gen, 4.35)
    p_w0, sw_w0 = production_factor(z_w0, "entmax", 1.5)
    p_r1, sw_r1 = production_factor(z_r1, "entmax", 1.5)
    for name, p in (("write level 0", p_w0), ("read level 1", p_r1)):
        k = float((p > 0).sum(-1).double().mean())
        print(f"{name}: entmax support = {k:.2f} of {B} digits per token (target ~4)")

    #: the two SOFTMAX duties: full support by construction.
    p_r0 = torch.softmax(torch.randn(1, T, B, device="cuda", generator=gen), -1)
    p_w1 = torch.softmax(torch.randn(1, T, B, device="cuda", generator=gen), -1)

    #: the seam: the producer's outputs enter the operator here, and the reverse pass
    #: hands the digit-gradient planes back at exactly this cut.
    leaves = {}
    def leaf(x):
        y = x.detach().clone().requires_grad_(True)
        leaves[id(y)] = y
        return y

    q_r0, q_r1, q_w0, q_w1 = leaf(p_r0), leaf(p_r1), leaf(p_w0), leaf(p_w1)
    read = tuple(q.reshape(1, T, 1, B) for q in (q_r0, q_r1))
    write = tuple(q.reshape(1, T, 1, B) for q in (q_w0, q_w1))
    g = (0.5 + torch.rand(1, T, 1, device="cuda", generator=gen)).to(torch.bfloat16)
    v = torch.randn(1, T, 1, DV, device="cuda", generator=gen).to(torch.bfloat16)

    num, den, _ = prefill_autograd(read, write, g, v, WIDTHS, k=K, m=M, window=WINDOW,
                                   modes=MODES)
    sn = torch.randn_like(num)
    sd = torch.randn_like(den)
    ((num.float() * sn.float()).sum() + (den.float() * sd.float()).sum()).backward()

    def seam(name, p, z, q, support):
        cot = q.grad
        off = ~support
        n_off_nonzero = int((cot[off] != 0).sum())
        print(f"\n{name}: cotangent plane {tuple(cot.shape)}, off-support entries "
              f"{int(off.sum())}, of which NONZERO {n_off_nonzero} "
              f"({100.0 * n_off_nonzero / max(1, int(off.sum())):.1f}%)")
        assert n_off_nonzero > 0, "vacuous: the kernel left the off-support plane at zero"
        masked = cot.masked_fill(off, 0.0)
        dz_full, = torch.autograd.grad(p, z, cot, retain_graph=True)
        dz_mask, = torch.autograd.grad(p, z, masked, retain_graph=True)
        same = torch.equal(dz_full, dz_mask)
        rel = float((dz_full - dz_mask).abs().max()) / max(1e-30, float(dz_full.abs().max()))
        print(f"{name}: shipped entmax VJP, torch.equal = {same}, rel max diff = {rel:.3e}")
        assert same, "the off-support cotangent is NOT annihilated by the producer VJP"

    seam("d_pw level 0 (entmax write)", p_w0, z_w0, q_w0, p_w0 > 0)
    seam("d_pr level 1 (entmax read)", p_r1, z_r1, q_r1, p_r1 > 0)

    #: NEGATIVE CONTROL, the other half of the ALT declaration: a softmax duty has no
    #: off-support set, and truncating its digit gradient to a 4-digit window changes the
    #: logit gradient outright.
    cot = q_r0.grad
    keep = torch.zeros_like(cot, dtype=torch.bool)
    keep.scatter_(-1, cot.abs().topk(4, dim=-1).indices, True)
    dense = -(p_r0 * cot).sum(-1, keepdim=True) * p_r0 + p_r0 * cot
    trunc_cot = cot.masked_fill(~keep, 0.0)
    trunc = -(p_r0 * trunc_cot).sum(-1, keepdim=True) * p_r0 + p_r0 * trunc_cot
    rel = float((dense - trunc).abs().max()) / max(1e-30, float(dense.abs().max()))
    print(f"\nd_pr level 0 (SOFTMAX read): truncating to 4 of {B} digits moves the logit "
          f"gradient by rel {rel:.3e} -- softmax has no off-support set; its digit "
          f"gradient is needed on the whole axis.")
    assert rel > 1e-3

    print("\nSEAM CONFIRMED for d_pw_0 and d_pr_1; REFUTED (by construction) for the "
          "softmax duties d_pr_0 and d_pw_1.")


if __name__ == "__main__":
    main()
