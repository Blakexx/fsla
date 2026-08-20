# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""K31 integration I3 — the combined prefill's measured table, with a same-session
flash-attention row.

NOT A GATE. The acceptance bars live on the queue card; this script is what
produces the numbers they are read against.

EVERYTHING IS MEASURED IN ONE PROCESS ON ONE QUIET CARD. The flash-attention row
in particular is taken here rather than quoted, because a number from another
session carries that session's clock, driver state and card temperature as
alternative explanations for any difference.

THE SPLIT IS REPORTED, not just the total. The combined wall is what a caller
pays; the carry-alone and intra-alone walls are what tells you which half moved,
and their sum is not the combined wall — the two kernels are separate launches on
one stream, so the difference is launch and tail overlap.

    python benchmarks/bench_prefill.py                 # the full table, L = 65536
    python benchmarks/bench_prefill.py --length 8192   # a smoke

Symbols at first use: ``N`` the leaf capacity, ``L`` the sequence length, ``W``
the global window length in tokens, ``k_tok`` a token's nonzero digits per sparse
level, ``cohort`` the run of consecutive tokens sharing one contiguous digit
window, ``d_v = 64`` value channels.
"""
from __future__ import annotations

import argparse
import math
import statistics
import zlib

import torch

from rola.ops import carry as carry_ops
from rola.ops import intra as intra_ops
from rola.ops.prefill import prefill

REPS = 6
WIDTHS = (256, 256)
K, M = 4, 8
D_V = 64

DENSE = (intra_ops.DENSE_BOTH, intra_ops.DENSE_BOTH)
ALT = (intra_ops.WRITE_SPARSE, intra_ops.READ_SPARSE)


def median_ms(fn, warmup=2, reps=REPS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        stop.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(stop))
    return statistics.median(samples)


def sparse_plane(T, width, k_tok, cohort, gen, device):
    """A row-normalized nonnegative plane, ``k_tok`` nonzeros per row.

    ``cohort`` draws ONE contiguous digit window per run of that many consecutive
    tokens, which is the CLUSTERED structure both the intra's slab gate and the
    carry's whole-window skip exist to exploit; ``None`` scatters independently.
    """
    x = torch.rand(1, T, 1, width, device=device, generator=gen)
    if k_tok is not None and k_tok < width:
        if cohort:
            starts = torch.randint(0, width, (1, T // cohort, 1, 1), device=device,
                                   generator=gen)
            starts = starts.expand(1, T // cohort, cohort, 1).reshape(1, T, 1, 1)
            index = (starts + torch.arange(k_tok, device=device).view(1, 1, 1, k_tok)) % width
        else:
            index = torch.argsort(torch.rand(1, T, 1, width, device=device, generator=gen),
                                  dim=-1)[..., :k_tok]
        keep = torch.zeros_like(x)
        keep.scatter_(-1, index, 1.0)
        x = x * keep
    return (x / x.sum(-1, keepdim=True)).to(torch.bfloat16)


#: THE CELLS, AND A SEED THAT IS A PROPERTY OF THE CELL.  `hash(str)` is salted
#: per PROCESS, so a table seeded from it is a different draw every run and two
#: "identical" measurements are two different cells -- which is how a 2x DRAM
#: difference gets read as noise.  `crc32` is a function of the name and nothing
#: else, so a row is reproducible from its label alone.
CELLS = [
    ("dense", DENSE, None, None),
    ("ALT k_tok=16 scattered", ALT, 16, None),
    ("ALT k_tok=4 scattered", ALT, 4, None),
    ("ALT k_tok=16 cohort=64", ALT, 16, 64),
    ("ALT k_tok=4 cohort=64", ALT, 4, 64),
]


def seed_of(name: str) -> int:
    return zlib.crc32(name.encode()) & 0xFFFF


def draw(T, modes, k_tok, cohort, seed, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    read, write = [], []
    for level, mode in enumerate(modes):
        width = WIDTHS[level]
        read.append(sparse_plane(
            T, width, k_tok if mode == intra_ops.READ_SPARSE else None, cohort, gen, device))
        write.append(sparse_plane(
            T, width, k_tok if mode == intra_ops.WRITE_SPARSE else None, cohort, gen, device))
    g = (0.5 + torch.rand(1, T, 1, device=device, generator=gen)).to(torch.bfloat16)
    v = torch.randn(1, T, 1, D_V, device=device, generator=gen).to(torch.bfloat16)
    return tuple(read), tuple(write), g, v


def split_walls(read, write, g, v, window, modes):
    """``(combined, carry_alone, intra_alone, with_packing)`` in ms.

    THE FIRST THREE ARE MEASURED ON PRE-PACKED PLANES, so the split and the total
    are the same measurement of the same bytes and their difference is the two
    launches' non-overlap and nothing else. `with_packing` is the fourth column
    because a caller who hands in per-level tensors pays the pack too, and hiding
    a per-call `permute`+copy of two 64 MB planes inside a kernel wall would make
    the kernel look like it did work it did not do.
    """
    pread = carry_ops.pack_side(read)
    pwrite = carry_ops.pack_side(write)
    gwrite = g.permute(0, 2, 1).contiguous().to(torch.bfloat16)
    gwrite = gwrite.reshape(gwrite.shape[0] * gwrite.shape[1], -1)
    value = v.to(torch.bfloat16).contiguous()
    packed_modes = intra_ops.pack_modes(modes)
    from rola.ops._ext import extension
    ext = extension()

    def carry_only():
        ext.carry_forward_inter(pread, pwrite, gwrite, value, list(WIDTHS), K, M,
                                window, 32, carry_ops.CHANNEL_SPLIT, None)

    o = torch.zeros((1, pread.shape[1], D_V), dtype=torch.float32, device=pread.device)
    den = torch.zeros((1, pread.shape[1]), dtype=torch.float32, device=pread.device)

    def intra_only():
        ext.intra_forward(pread, pwrite, gwrite, value, o, den, len(WIDTHS),
                          packed_modes, window)

    def combined():
        num, dn, _ = ext.carry_forward_inter(pread, pwrite, gwrite, value, list(WIDTHS),
                                             K, M, window, 32, carry_ops.CHANNEL_SPLIT, None)
        ext.intra_forward(pread, pwrite, gwrite, value, num, dn, len(WIDTHS),
                          packed_modes, window)

    def with_packing():
        prefill(read, write, g, v, WIDTHS, k=K, m=M, window=window, modes=modes)

    return (median_ms(combined), median_ms(carry_only), median_ms(intra_only),
            median_ms(with_packing))


def flash_ms(seq, heads, device="cuda"):
    """`scaled_dot_product_attention`, flash backend, causal, bf16.

    Reported EXACTLY as run. The capacity-fair convention is `N = L` and the same
    `d_v`; the head count is the one axis where the two operators are not
    naturally matched, so both the brief's `H = 8` shape and a per-head `H = 1`
    shape are taken and both are printed.
    """
    q, k, v = (torch.randn(1, heads, seq, D_V, device=device, dtype=torch.bfloat16)
               for _ in range(3))
    backend = torch.nn.attention.SDPBackend.FLASH_ATTENTION

    def go():
        with torch.nn.attention.sdpa_kernel(backend):
            torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    try:
        return median_ms(go)
    except RuntimeError as exc:
        print(f"# flash backend refused H={heads}: {exc}")
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=65536)
    ap.add_argument("--window", type=int, default=carry_ops.WINDOW_WIDE)
    a = ap.parse_args()
    T, window = a.length, a.window
    if T % window:
        raise SystemExit(f"L = {T} is not a whole number of {window}-token windows")
    N = math.prod(WIDTHS)
    print(f"# K31 integration I3 -- {torch.cuda.get_device_name()}, N = {N}, L = {T}, "
          f"N == L: {N == T}, d_v = {D_V}, BH = 1, W = {window}, medians of {REPS}")
    stamp = carry_ops.build_stamp(WIDTHS, K, M, window, 32, carry_ops.CHANNEL_SPLIT)
    print(f"# carry stamp {stamp[0]}  regs {stamp[3]}  local {stamp[8]}  smem {stamp[6]}  "
          f"BC {stamp[9]}  owners {stamp[10]}  stages {stamp[11]}")
    print(f"# intra stamp {intra_ops.build_stamp()}  arms {intra_ops.arms()}")

    rows = []
    for name, modes, k_tok, cohort in CELLS:
        read, write, g, v = draw(T, modes, k_tok, cohort, seed=seed_of(name))
        rows.append((name, *split_walls(read, write, g, v, window, modes)))
        del read, write, g, v
        torch.cuda.empty_cache()

    fa8 = flash_ms(T, 8)
    fa1 = flash_ms(T, 1)

    print(f"\n{'cell':<24} {'combined':>10} {'carry':>10} {'intra':>10} {'sum':>10} "
          f"{'+pack':>10} {'vs FA/head':>11}")
    for name, combined, carry, intra, packed in rows:
        print(f"{name:<24} {combined:10.2f} {carry:10.2f} {intra:10.2f} "
              f"{carry + intra:10.2f} {packed:10.2f} {combined / (fa8 / 8):10.2f}x")
    print(f"\nflash-attention causal bf16, B=1 H=8 d=64 seq={T}: {fa8:.2f} ms "
          f"({fa8 / 8:.2f} ms/head)")
    print(f"flash-attention causal bf16, B=1 H=1 d=64 seq={T}: {fa1:.2f} ms")
    print("# 'vs FA/head' divides by the H=8 row's per-head cost: RoLA is measured at "
          "BH = 1 and\n#  flash amortizes its launch and its L2 traffic across heads, "
          "so the H=1 row is the\n#  honest floor and the per-head H=8 figure the "
          "honest ceiling for that division.")


if __name__ == "__main__":
    main()
