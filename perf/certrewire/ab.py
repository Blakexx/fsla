# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The zero-ABI certificate rewire's paired A/B — the same cells on both binaries.

MEASUREMENT ONLY. The script runs unchanged against a tree whose intra takes the
producer's support word and against one whose certificate rebuilds the union from
the staged amplitudes; which tree it is on is a fact it reads off `rola.ops.intra`,
so the two runs are the same script and the same draws.

Symbols at first use: ``BH`` batch x heads, ``L`` the sequence length, ``W`` the
window in tokens, ``D`` routing levels, ``B`` digits per level, ``k_tok`` a token's
nonzero digits at a sparse level, ``cohort`` the run of consecutive tokens sharing
one digit window, ``d_v = 64`` value channels.

    dump   the five cells' o/den/state, for a bit-identity join across the pair
    bench  paired medians: the intra forward alone, and the combined prefill
    ncu    one intra launch on the `k31_offsupport` Check 2 cell, for --launch-skip
"""
from __future__ import annotations

import argparse
import statistics
import zlib

import torch

from rola.ops import carry as carry_ops
from rola.ops import intra as intra_ops
from rola.ops._ext import extension
from rola.ops.prefill import prefill

WIDTHS = (256, 256)
K, M = 4, 8
D_V = 64
WORDS = hasattr(intra_ops, "support_words")

#: `(name, modes, k_tok, cohort)` -- the card's five cells.
CELLS = [
    ("dense", (intra_ops.DENSE_BOTH,) * 2, None, None),
    ("k4-scat", (intra_ops.WRITE_SPARSE, intra_ops.READ_SPARSE), 4, None),
    ("k16-scat", (intra_ops.WRITE_SPARSE, intra_ops.READ_SPARSE), 16, None),
    ("k4-clu", (intra_ops.WRITE_SPARSE, intra_ops.READ_SPARSE), 4, 64),
    ("k16-clu", (intra_ops.WRITE_SPARSE, intra_ops.READ_SPARSE), 16, 64),
]


def sparse_plane(bh, T, width, k_tok, cohort, gen, device):
    x = torch.rand(bh, T, width, device=device, generator=gen)
    if k_tok is not None and k_tok < width:
        if cohort:
            starts = torch.randint(0, width, (bh, T // cohort, 1), device=device, generator=gen)
            starts = starts.expand(bh, T // cohort, cohort).reshape(bh, T, 1)
            index = (starts + torch.arange(k_tok, device=device).view(1, 1, k_tok)) % width
        else:
            index = torch.argsort(torch.rand(bh, T, width, device=device, generator=gen),
                                  dim=-1)[..., :k_tok]
        keep = torch.zeros_like(x)
        keep.scatter_(-1, index, 1.0)
        x = x * keep
    return (x / x.sum(-1, keepdim=True)).to(torch.bfloat16)


def draw(bh, T, modes, k_tok, cohort, seed, device="cuda"):
    """The packed planes, the gain, the value stream and the two support words."""
    gen = torch.Generator(device=device).manual_seed(seed)
    read, write = [], []
    for level, mode in enumerate(modes):
        read.append(sparse_plane(bh, T, WIDTHS[level],
                                 k_tok if mode == intra_ops.READ_SPARSE else None,
                                 cohort, gen, device))
        write.append(sparse_plane(bh, T, WIDTHS[level],
                                  k_tok if mode == intra_ops.WRITE_SPARSE else None,
                                  cohort, gen, device))
    pread = torch.cat(read, dim=-1).contiguous()
    pwrite = torch.cat(write, dim=-1).contiguous()
    gwrite = (0.5 + torch.rand(bh, T, device=device, generator=gen)).to(torch.bfloat16)
    value = torch.randn(bh, T, 1, D_V, device=device, generator=gen).to(torch.bfloat16)
    words = ((intra_ops.support_words(pread), intra_ops.support_words(pwrite))
             if WORDS else ())
    return pread, pwrite, gwrite, value, words


def median_ms(fn, warmup=3, reps=15):
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
    return statistics.median(samples), min(samples)


def arms(bh, T, window, name, modes, k_tok, cohort):
    pread, pwrite, gwrite, value, words = draw(bh, T, modes, k_tok, cohort,
                                               zlib.crc32(name.encode()) & 0xFFFF)
    packed = intra_ops.pack_modes(modes)
    ext = extension()
    o = torch.zeros((bh, T, D_V), dtype=torch.float32, device=pread.device)
    den = torch.zeros((bh, T), dtype=torch.float32, device=pread.device)

    def intra_only():
        ext.intra_forward(pread, pwrite, gwrite, value, *words, o, den, len(WIDTHS),
                          WIDTHS[0], packed, window)

    def combined():
        num, dn, _ = ext.carry_forward_inter(pread, pwrite, gwrite, value, list(WIDTHS), K, M,
                                             window, 32, carry_ops.CHANNEL_SPLIT, None)
        ext.intra_forward(pread, pwrite, gwrite, value, *words, num, dn, len(WIDTHS),
                          WIDTHS[0], packed, window)
    return intra_only, combined, (pread, pwrite, gwrite, value, words, o, den, packed)


def run_bench(bh, T, window, rounds):
    print(f"# A/B  BH={bh} L={T} W={window}  support-word form: {WORDS}")
    print(f"{'cell':<12}{'intra ms':>11}{'intra min':>11}{'combined ms':>13}{'comb min':>11}")
    for name, modes, k_tok, cohort in CELLS:
        intra_only, combined, hold = arms(bh, T, window, name, modes, k_tok, cohort)
        i_med, i_min = median_ms(intra_only, reps=rounds)
        c_med, c_min = median_ms(combined, reps=rounds)
        print(f"{name:<12}{i_med:>11.4f}{i_min:>11.4f}{c_med:>13.4f}{c_min:>11.4f}")
        del intra_only, combined, hold
        torch.cuda.empty_cache()


def run_dump(bh, T, window, path):
    """o/den/state per cell, for the bit-identity join across the pair."""
    assert bh == 1, "the dump feeds `prefill`'s per-level [B, T, H, w] form at B = H = 1"
    out = {}
    for name, modes, k_tok, cohort in CELLS:
        pread, pwrite, gwrite, value, words = draw(bh, T, modes, k_tok, cohort,
                                                   zlib.crc32(name.encode()) & 0xFFFF)
        read = tuple(pread[:, :, i * 256:(i + 1) * 256].reshape(bh, T, 1, 256)
                     for i in range(2))
        write = tuple(pwrite[:, :, i * 256:(i + 1) * 256].reshape(bh, T, 1, 256)
                      for i in range(2))
        kw = dict(sread=words[0], swrite=words[1]) if WORDS else {}
        num, den, state = prefill(read, write, gwrite.reshape(bh, T, 1),
                                  value, WIDTHS, k=K, m=M, window=window, modes=modes, **kw)
        out[name] = (num.cpu(), den.cpu(), state.cpu())
        del num, den, state, pread, pwrite, read, write
        torch.cuda.empty_cache()
    torch.save(out, path)
    print(f"wrote {path}")


def run_ncu(bh, T, window, skip, cell):
    """One cell, warmed, then `skip` more launches so ncu profiles a settled one."""
    name, modes, k_tok, cohort = next(c for c in CELLS if c[0] == cell)
    pread, pwrite, gwrite, value, words = draw(bh, T, modes, k_tok, cohort,
                                               zlib.crc32(name.encode()) & 0xFFFF)
    packed = intra_ops.pack_modes(modes)
    ext = extension()
    o = torch.zeros((bh, T, D_V), dtype=torch.float32, device=pread.device)
    den = torch.zeros((bh, T), dtype=torch.float32, device=pread.device)
    for _ in range(skip + 1):
        ext.intra_forward(pread, pwrite, gwrite, value, *words, o, den, len(WIDTHS), WIDTHS[0],
                          packed, window)
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("bench", "dump", "ncu"), default="bench")
    ap.add_argument("--bh", type=int, default=1)
    ap.add_argument("--length", type=int, default=65536)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--skip", type=int, default=4)
    ap.add_argument("--cell", default="k4-scat", choices=[c[0] for c in CELLS])
    ap.add_argument("--out", default="/tmp/certrewire_dump.pt")
    a = ap.parse_args()
    torch.cuda.init()
    if a.mode == "bench":
        run_bench(a.bh, a.length, a.window, a.rounds)
    elif a.mode == "dump":
        run_dump(a.bh, a.length, a.window, a.out)
    else:
        run_ncu(a.bh, a.length, a.window, a.skip, a.cell)


if __name__ == "__main__":
    main()
