# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""CHECK 2: the ncu subject -- ONE shipped intra FORWARD at ALT k_tok=4 scattered.

Flagship shapes: L = 65536, W = 512, BH = 1, D = 2, B = 256, d_v = 64. Pre-packed planes,
so the profiled launch is the kernel and nothing else.
"""
from __future__ import annotations

import torch

from benchmarks.bench_prefill import D_V, PATTERNS, TOPOLOGIES, draw, seed_of
from rola.ops import carry as carry_ops
from rola.ops import intra as intra_ops
from rola.ops._ext import extension

L = 65536
NAME = "ALT k_tok=4 scattered"
WIDTHS, K, M, WINDOW = TOPOLOGIES["flat"]


def main():
    ext = extension()
    modes = PATTERNS["ALT"](len(WIDTHS))
    assert modes == (intra_ops.WRITE_SPARSE, intra_ops.READ_SPARSE)
    read, write, g, v = draw(L, WIDTHS, modes, 4, None, seed_of(NAME))
    pread = carry_ops.pack_side(read)
    pwrite = carry_ops.pack_side(write)
    gwrite = g.permute(0, 2, 1).contiguous().to(torch.bfloat16)
    gwrite = gwrite.reshape(gwrite.shape[0] * gwrite.shape[1], -1)
    value = v.to(torch.bfloat16).contiguous()
    packed = intra_ops.pack_modes(modes)
    bh, T, _ = pread.shape
    num = torch.zeros((bh, T, D_V), device=pread.device, dtype=torch.float32)
    den = torch.zeros((bh, T), device=pread.device, dtype=torch.float32)

    def run():
        ext.intra_forward(pread, pwrite, gwrite, value, num, den, len(WIDTHS), WIDTHS[0], packed, WINDOW)

    run()
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    run()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


if __name__ == "__main__":
    main()
