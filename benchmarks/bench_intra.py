# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""K31 R2's perf snapshot: the intra kernel at the N = L = 65536 anchor.

Reports wall time, tile-pairs per millisecond, and the achieved fraction of the
device's mma.sync bf16/fp32 ceiling. The issued MAC count is exact: the window's
lower-triangular tile grid is ``n(n+1)/2`` of its ``n = W/64`` squared [64x64]
tiles, each contracting ``sum_l B_l`` digits for the Gram and ``d_v`` for the
value product.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess

import torch

from rola.ops.intra import DENSE_BOTH, READ_SPARSE, VALUE_WIDTH, WINDOW, WRITE_SPARSE, intra_forward

TILE = 64
#: sm_86 consumer parts issue 256 bf16 MACs per clock per SM with an fp32
#: accumulator (half the fp16-accumulate rate) -- the mma.sync ceiling this
#: kernel's fraction is quoted against.
MACS_PER_CLOCK_PER_SM = 256


def peak_flops(device=0):
    """The ceiling, taken from the device's own advertised boost clock and SM count."""
    properties = torch.cuda.get_device_properties(device)
    clock = float(subprocess.run(["nvidia-smi", "--query-gpu=clocks.max.sm",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout.split()[0]) * 1e6
    return 2.0 * MACS_PER_CLOCK_PER_SM * properties.multi_processor_count * clock, clock


def tile_pairs(window):
    tiles = window // TILE
    return tiles * (tiles + 1) // 2


def macs(bh, length, levels, window):
    per_tile = TILE * TILE * (levels * 256 + VALUE_WIDTH)
    return bh * (length // window) * tile_pairs(window) * per_tile


def sparse_plane(bh, length, width, k_tok, cohort, generator, device):
    plane = torch.rand((bh, length, width), device=device, generator=generator)
    if k_tok is not None:
        rows = length // cohort
        starts = torch.randint(0, width, (bh, rows, 1, 1), device=device, generator=generator)
        starts = starts.expand(bh, rows, cohort, 1).reshape(bh, length, 1)
        index = (starts + torch.arange(k_tok, device=device).view(1, 1, k_tok)) % width
        keep = torch.zeros_like(plane)
        keep.scatter_(-1, index, 1.0)
        plane = plane * keep
    return (plane / plane.sum(-1, keepdim=True)).to(torch.bfloat16)


def build(bh, length, levels, modes, k_tok, cohort, device, seed=0):
    generator = torch.Generator(device=device).manual_seed(seed)
    read, write = [], []
    for mode in modes:
        read.append(sparse_plane(bh, length, 256, k_tok if mode == READ_SPARSE else None,
                                 cohort, generator, device))
        write.append(sparse_plane(bh, length, 256, k_tok if mode == WRITE_SPARSE else None,
                                  cohort, generator, device))
    pread = torch.cat(read, dim=-1)
    pwrite = torch.cat(write, dim=-1)
    gwrite = (0.5 + torch.rand((bh, length), device=device, generator=generator)).to(torch.bfloat16)
    v = torch.randn((1, length, bh, VALUE_WIDTH), device=device,
                    generator=generator).to(torch.bfloat16)
    return pread, pwrite, gwrite, v


def time_ms(fn, repeats):
    samples = []
    for _ in range(repeats):
        start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize()
        start.record()
        fn()
        stop.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(stop))
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bh", type=int, default=8)
    parser.add_argument("--length", type=int, default=65536)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--window", type=int, default=WINDOW)
    args = parser.parse_args()

    device = "cuda"
    levels = 2
    window = args.window
    if args.length % window:
        raise SystemExit(f"L = {args.length} is not a whole number of {window}-token windows")
    flops, clock = peak_flops()
    print(f"device {torch.cuda.get_device_name()}  clock {clock / 1e9:.3f} GHz  "
          f"mma.sync ceiling {flops / 1e12:.1f} TFLOP/s")
    windows = args.bh * args.length // window
    print(f"cell BH={args.bh} L={args.length} D={levels} W={window}  "
          f"{windows} windows, {windows * tile_pairs(window)} tile-pairs")
    work = 2.0 * macs(args.bh, args.length, levels, window)
    print(f"{'cell':<24}{'ms':>9}{'tiles/ms':>12}{'TFLOP/s':>10}{'peak':>8}")

    cells = [
        ("dense", (DENSE_BOTH, DENSE_BOTH), None, 1),
        ("alt-k4-scatter", (WRITE_SPARSE, READ_SPARSE), 4, 1),
        ("alt-k16-scatter", (WRITE_SPARSE, READ_SPARSE), 16, 1),
        ("alt-k4-cohort-64", (WRITE_SPARSE, READ_SPARSE), 4, 64),
        ("alt-k16-cohort-64", (WRITE_SPARSE, READ_SPARSE), 16, 64),
    ]
    for name, modes, k_tok, cohort in cells:
        pread, pwrite, gwrite, v = build(args.bh, args.length, levels, modes, k_tok, cohort, device)
        o = torch.zeros((args.bh, args.length, VALUE_WIDTH), dtype=torch.float32, device=device)
        den = torch.zeros((args.bh, args.length), dtype=torch.float32, device=device)

        def run(pread=pread, pwrite=pwrite, gwrite=gwrite, v=v, o=o, den=den, modes=modes,
                window=window):
            intra_forward(pread, pwrite, gwrite, v, modes, o=o, den=den, window=window)

        run()
        ms = time_ms(run, args.repeats)
        tiles = windows * tile_pairs(window)
        achieved = work / (ms * 1e-3)
        print(f"{name:<24}{ms:>9.3f}{tiles / ms:>12.0f}{achieved / 1e12:>10.2f}"
              f"{achieved / flops * 100:>7.1f}%")
        del pread, pwrite, gwrite, v, o, den, run
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
