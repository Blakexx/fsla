# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""K31 R4 — the reverse pass's measured table, beside the forward's own wall.

NOT A GATE. The rung's claim is the ratio, not a bar: three reverse kernels against
three forward ones, two of them replaying the carry's own walk and one recomputing the
intra's Grams at roughly three times its MMA count.

EVERYTHING IS MEASURED IN ONE PROCESS ON ONE QUIET CARD, on PRE-PACKED planes, so the
forward and reverse rows are the same measurement of the same bytes.

    python benchmarks/bench_prefill_backward.py                 # L = 65536
    python benchmarks/bench_prefill_backward.py --length 8192   # a smoke

Symbols at first use: ``L`` the sequence length, ``W`` the global window length in
tokens, ``k_tok`` a token's nonzero digits per sparse level, ``d_v = 64`` value channels.
"""
from __future__ import annotations

import argparse
import statistics

import torch

from benchmarks.bench_prefill import ALT, D_V, DENSE, WIDTHS, K, M, draw, median_ms, seed_of
from rola.ops import carry as carry_ops
from rola.ops import intra as intra_ops
from rola.ops._ext import extension

CELLS = [
    ("dense", DENSE, None, None),
    ("ALT k_tok=4 scattered", ALT, 4, None),
    ("ALT k_tok=16 scattered", ALT, 16, None),
]


def walls(read, write, g, v, window, modes):
    """``(forward, backward, scans, intra_bwd)`` in ms, on pre-packed planes."""
    pread = carry_ops.pack_side(read)
    pwrite = carry_ops.pack_side(write)
    gwrite = g.permute(0, 2, 1).contiguous().to(torch.bfloat16)
    gwrite = gwrite.reshape(gwrite.shape[0] * gwrite.shape[1], -1)
    value = v.to(torch.bfloat16).contiguous()
    packed = intra_ops.pack_modes(modes)
    bh, T, _ = pread.shape
    ext = extension()

    d_num = torch.randn((bh, T, D_V), device=pread.device, dtype=torch.float32)
    d_den = torch.randn((bh, T), device=pread.device, dtype=torch.float32)
    vt = carry_ops.pack_cols(value.permute(0, 2, 1, 3).reshape(bh, T, D_V), 1.0)
    dn = carry_ops.pack_cols(d_num, d_den)
    dnum_bf = d_num.to(torch.bfloat16).contiguous()

    def forward():
        num, den, _ = ext.carry_forward_inter(pread, pwrite, gwrite, value, list(WIDTHS), K, M,
                                              window, 32, carry_ops.LEAF_SPLIT, None)
        ext.intra_forward(pread, pwrite, gwrite, value, num, den, len(WIDTHS), packed, window)

    def scans():
        ext.carry_backward_inter(pread, pwrite, gwrite, vt, dn, list(WIDTHS), K, M, window, 32,
                                 None, None, False)

    planes = ext.carry_backward_inter(pread, pwrite, gwrite, vt, dn, list(WIDTHS), K, M, window,
                                      32, None, None, False)

    def intra_bwd():
        ext.intra_backward(pread, pwrite, gwrite, value, dnum_bf, d_den, planes[0], planes[1],
                           planes[2], planes[3], len(WIDTHS), packed, window)

    def backward():
        out = ext.carry_backward_inter(pread, pwrite, gwrite, vt, dn, list(WIDTHS), K, M, window,
                                       32, None, None, False)
        ext.intra_backward(pread, pwrite, gwrite, value, dnum_bf, d_den, out[0], out[1], out[2],
                           out[3], len(WIDTHS), packed, window)

    return (median_ms(forward), median_ms(backward), median_ms(scans), median_ms(intra_bwd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=65536)
    ap.add_argument("--window", type=int, default=carry_ops.WINDOW_WIDE)
    a = ap.parse_args()
    T, window = a.length, a.window
    if T % window:
        raise SystemExit(f"L = {T} is not a whole number of {window}-token windows")
    stamp = carry_ops.backward_stamp(WIDTHS, K, M, window, 32)
    print(f"# device {torch.cuda.get_device_name(0)}   L = {T}   W = {window}")
    print(f"# reverse stamp {stamp[0]}   scan A regs {stamp[3]} local {stamp[4]}   "
          f"scan B regs {stamp[9]} local {stamp[10]}")
    print(f"{'cell':26s} {'fwd ms':>9s} {'bwd ms':>9s} {'bwd/fwd':>8s} "
          f"{'scans ms':>9s} {'intra ms':>9s}")
    ratios = []
    for name, modes, k_tok, cohort in CELLS:
        read, write, g, v = draw(T, modes, k_tok, cohort, seed_of(name))
        fwd, bwd, sc, ib = walls(read, write, g, v, window, modes)
        ratios.append(bwd / fwd)
        print(f"{name:26s} {fwd:9.2f} {bwd:9.2f} {bwd / fwd:8.2f} {sc:9.2f} {ib:9.2f}")
        del read, write, g, v
        torch.cuda.empty_cache()
    print(f"# median backward/forward ratio {statistics.median(ratios):.2f}")


if __name__ == "__main__":
    main()
