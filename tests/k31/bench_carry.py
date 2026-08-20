# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""K31 R1 M5 -- the perf snapshot: the channel-split arm beside the shipped baseline.

NOT A GATE.  R1's targets are context, not a bar: the shipped arms stay the
baseline until the new family beats them cell for cell, and this script is what
produces the cell-for-cell table.

BOTH FAMILIES ARE MEASURED FROM THE SAME BINARY, in the same process, on the same
quiet card.  The shipped kernel's source is byte-identical to the main tree's on
this branch (nothing under `csrc/rola/src/chunk/` is touched), so "the main tree
binary" and "this binary's chunk half" are the same codegen -- and measuring them
in one session removes the clock, the driver state and the card temperature as
explanations for a difference.

    python tests/k31/bench_carry.py            # the full table
    python tests/k31/bench_carry.py --quick    # L = 8192, for a smoke
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys

import torch

from rola.ops import carry as carry_ops
from rola.ops._ext import extension

REPS = 6


def median_ms(fn, warmup=2, reps=REPS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return statistics.median(out)


def draw(widths, T, k_tok_read, k_tok_write, seed, d_v=64, device="cuda"):
    """`k_tok_read`/`k_tok_write` are per-level tuples (`None` = dense).

    The ALT structure (level 0 write-sparse/read-dense, level 1 read-sparse/
    write-dense) is what the two-sided schedule exists for, so it is the sparse
    cell here -- a union-routed cell would not exercise it.
    """
    gen = torch.Generator(device=device).manual_seed(seed)

    def side(ks):
        levels = []
        for w, k in zip(widths, ks):
            x = torch.rand(1, T, 1, w, device=device, generator=gen)
            if k is not None and k < w:
                keep = torch.zeros_like(x)
                idx = torch.argsort(torch.rand(1, T, 1, w, device=device, generator=gen),
                                    dim=-1)[..., :k]
                keep.scatter_(-1, idx, 1.0)
                x = x * keep
            levels.append((x / x.sum(-1, keepdim=True)).to(torch.bfloat16))
        return tuple(levels)

    read, write = side(k_tok_read), side(k_tok_write)
    g = (torch.rand(1, T, 1, device=device, generator=gen) + 0.5).to(torch.bfloat16)
    v = torch.randn(1, T, 1, d_v, device=device, generator=gen).to(torch.bfloat16)
    return read, write, g, v


def carry_wall(read, write, g, v, widths, k, m, W, carve, chunk=32):
    def go():
        carry_ops.carry_inter(read, write, g, v, widths, k=k, m=m, window=W,
                              chunk=chunk, carve=carve)
    return median_ms(go)


def shipped_wall(read, write, g, v, widths, chunk=32, bc=128):
    """The shipped chunk consumer's kernel wall, facts pass EXCLUDED.

    The facts table is built once outside the timed region because it is a
    per-call preamble both families would pay and the comparison is between the
    two consumers.  Nothing under `csrc/rola/src/chunk/` is modified on this
    branch; this is the baseline arm as shipped.
    """
    pr = carry_ops.pack_side(read).unsqueeze(0)
    pw = carry_ops.pack_side(write).unsqueeze(0)
    #: the shipped entry wants `[B, H, T, sumW]`.
    pr = pr.reshape(1, 1, pr.shape[-2], pr.shape[-1])
    pw = pw.reshape(1, 1, pw.shape[-2], pw.shape[-1])
    gw = g.permute(0, 2, 1).contiguous()
    tbl = extension().chunk_facts(pr, pw, chunk, bc, list(widths), 1, False)[0]
    args = (pr, pw, gw, v, tbl, chunk, bc, list(widths), 0xFFFFFFFF, None, None, None, None)

    def go():
        extension().chunk_forward_raw(*args)
    return median_ms(go)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    widths = (256, 256)
    T = 8192 if a.quick else 65536
    N = math.prod(widths)
    print(f"# K31 R1 M5 perf snapshot -- {torch.cuda.get_device_name()}, "
          f"N = {N}, L = {T}, d_v = 64, BH = 1, medians of {REPS}")
    stamp = carry_ops.build_stamp(widths, 4, 8, 256, 32, 0)
    print(f"# carry device stamp {stamp[0]}; arm A regs {stamp[3]} local {stamp[8]} "
          f"smem {stamp[6]}")

    cells = {
        "dense": ((None, None), (None, None)),
        "ALT k_tok=16": ((None, 16), (16, None)),
        "ALT k_tok=4": ((None, 4), (4, None)),
    }
    variants = [
        ("B  channel-split BC=128 W=256", 4, 8, 256, carry_ops.CHANNEL_SPLIT),
    ]
    rows = []
    for cell, (kr, kw) in cells.items():
        read, write, g, v = draw(widths, T, kr, kw, seed=hash(cell) & 0xFFFF)
        base = shipped_wall(read, write, g, v, widths)
        rows.append((cell, "shipped chunk C=32 BC=128", base))
        for name, k, m, W, carve in variants:
            try:
                ms = carry_wall(read, write, g, v, widths, k, m, W, carve)
            except RuntimeError as exc:  # an arm this build does not carry
                ms = float("nan")
                print(f"# {cell} / {name}: {exc}", file=sys.stderr)
            rows.append((cell, name, ms))
        del read, write, g, v
        torch.cuda.empty_cache()

    print(f"\n{'cell':<14} {'arm':<30} {'wall (ms)':>10}  {'vs shipped':>10}")
    base_by_cell = {c: w for c, n, w in rows if n.startswith("shipped")}
    for cell, name, ms in rows:
        rel = ms / base_by_cell[cell] if base_by_cell[cell] else float("nan")
        print(f"{cell:<14} {name:<30} {ms:10.2f}  {rel:9.2f}x")


if __name__ == "__main__":
    main()
