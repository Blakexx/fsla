# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE FACTS PASS ON THE LAB'S `facts_block` CELL, against its line (docs/internals/facts/facts.md#budget): rows of four
level-sides `[65,536 tokens, 4 heads, 256 digits]` u8 (268 MB) -- each side's plane head-major, its two levels' slots
side by side, as the producer writes them -- each token-head live in `k` random digits of each level-side; the write
side lists level 0 and the read side level 1, the other level of each voted (the lab cell's two listed and two voted
level-sides). One launch of the production kernel in the sm_86 even arm's form, timed by CUDA events; the required
bytes (the rows, every output the pass writes, the columns it reads) priced at the lab's recorded flat read.

    python measure/harness/bench_facts_block.py [--k 4 16] [--reps 20] [--ledger]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

TOKENS, HEADS, WIDTH, WINDOW = 65536, 4, 256, 512
#: the line (ruled 2026-10-13): 1.15x the pass's required-bytes floor at the colored lists' places, by k
LINE_US = {4: 516.0, 16: 665.0}
#: the lab's recorded flat read of the cell's 268 MB of rows (measure/lab/results/facts_two/sm_86.json)
FLAT_RECORD = ROOT / "measure/lab/results/facts_two/sm_86.json"


def flat_rate() -> float:
    """Bytes a microsecond, the lab's flat read of every row."""
    rows = json.loads(FLAT_RECORD.read_text())["results"]
    us = next(r["median"] for r in rows if r["name"] == "flat read of all the rows")
    return 4 * TOKENS * HEADS * WIDTH / us


#: the cell's draw
SEED = 0


def cell(k: int):
    """``(read, write, scales, g, vs)``: each side's two levels as ``[1, T, H, 256]`` views of its head-major plane,
    each token-head live in ``k`` random digits of each (u8 1..255); the write levels' scales, the write gain and the
    value scale ``[1, T, H]`` fp32."""
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    sides = []
    for _ in range(2):
        plane = torch.zeros((1, HEADS, TOKENS, 2 * WIDTH), dtype=torch.uint8, device="cuda")
        for level in range(2):
            idx = torch.randint(0, WIDTH, (1, HEADS, TOKENS, k), device="cuda", generator=gen) + level * WIDTH
            val = torch.randint(1, 256, (1, HEADS, TOKENS, k), device="cuda", generator=gen).to(torch.uint8)
            plane.scatter_(-1, idx, val)
        sides.append([plane[..., level * WIDTH:(level + 1) * WIDTH].transpose(1, 2) for level in range(2)])

    def column():
        return (torch.rand((1, HEADS, TOKENS), device="cuda", generator=gen) + 0.1).transpose(1, 2)

    return sides[0], sides[1], [column(), column()], column(), column()


def launch(read, write, scales, g, vs, read_row, outs):
    from rola.engine.facts import liveness as lv
    from rola.ops import facts as facts_ops

    words, unions, list_exp, fm, gains16, gains_exp, wl, rl = outs
    masks = [0xFFFFFFFF] * lv.mask_words(2 * WIDTH)
    facts_ops.extension().rola_facts_forward(
        read, write, scales, g, vs, words, unions, list_exp, fm, gains16, gains_exp, list(wl), list(rl),
        [WIDTH, WIDTH], 0, masks, 0, masks, 0, 1, WINDOW, read_row)


def outputs():
    """The pass's outputs at the cell, allocated once."""
    from rola.ops import facts as facts_ops

    windows, bh = TOKENS // WINDOW, HEADS
    words = torch.empty((bh, 2, 2 * WIDTH, TOKENS // 32), dtype=torch.int32, device="cuda")
    unions = torch.empty((bh, 2, 2 * WIDTH // 32, TOKENS // 64), dtype=torch.int32, device="cuda")
    list_exp = torch.empty((bh, windows, WIDTH, 2), dtype=torch.int16, device="cuda")
    fm = torch.empty((bh, TOKENS, 2), dtype=torch.float32, device="cuda")
    gains16 = torch.empty((bh, TOKENS, 2), dtype=torch.float16, device="cuda")
    gains_exp = torch.empty((bh, windows, 2), dtype=torch.int16, device="cuda")
    wl, rl = (facts_ops._lists(bh, windows, WIDTH, "cuda") for _ in range(2))
    return words, unions, list_exp, fm, gains16, gains_exp, wl, rl


def required_bytes(outs) -> float:
    """The rows, the columns read, and every byte the pass writes: the words and tile unions, each list's entries,
    boundaries, totals, union and extent, the exponents, the factor products and gains."""
    words, unions, list_exp, fm, gains16, gains_exp, wl, rl = outs
    rows = 4 * TOKENS * HEADS * WIDTH
    columns = 4 * TOKENS * HEADS * 4
    entries = sum(int(l.totals[..., -1].long().sum()) for l in (wl, rl)) * 8
    lists = sum(l.pre.numel() * 2 + l.totals.numel() * 4 + l.union.numel() * 2 + l.count.numel() * 4 for l in (wl, rl))
    rest = (words.numel() + unions.numel()) * 4 + list_exp.numel() * 2 + fm.numel() * 4 + gains16.numel() * 2
    return rows + columns + entries + lists + rest + gains_exp.numel() * 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--k", type=int, nargs="+", default=[4, 16])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--ledger", action="store_true", help="also book the phase ledger (a separate, untimed run)")
    args = ap.parse_args()
    from rola_devtools.locks.gpu import gpu_lock

    from rola.ops import facts as facts_ops
    read_row = 64
    rate = flat_rate()
    with gpu_lock():
        for k in args.k:
            read, write, scales, g, vs = cell(k)
            outs = outputs()
            for _ in range(3):
                launch(read, write, scales, g, vs, read_row, outs)
            torch.cuda.synchronize()
            times = []
            for _ in range(args.reps):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                launch(read, write, scales, g, vs, read_row, outs)
                b.record()
                b.synchronize()
                times.append(1000.0 * a.elapsed_time(b))
            us = statistics.median(times)
            floor = required_bytes(outs) / rate
            line = LINE_US.get(k)
            print(f"k {k}: {us:.1f} us (min {min(times):.1f}), required-bytes floor {floor:.1f} us at "
                  f"{rate / 1000:.0f} GB/s: {us / floor:.3f}x" + (f"; the line {line:.0f} us" if line else ""))
            if args.ledger:
                ledger = torch.zeros((facts_ops.LEDGER_WARPS, len(facts_ops.LEDGER_PHASES)), dtype=torch.int64,
                                     device="cuda")
                facts_ops.ledger_bind(ledger)
                try:
                    launch(read, write, scales, g, vs, read_row, outs)
                    torch.cuda.synchronize()
                finally:
                    facts_ops.ledger_bind(None)
                total = ledger.sum(0).double()
                print("  phase share (cycles summed over warps): " + ", ".join(
                    f"{name} {v / total.sum():.2f}" for name, v in zip(facts_ops.LEDGER_PHASES, total.tolist())))
            del read, write, outs
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
