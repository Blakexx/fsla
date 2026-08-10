"""The stop gate's arms only, timed against rung 5A in one session.

    python floor.py [A|B|both]

`bench.py headline` sweeps every arm; this times the four that carry the verdict
plus the control, so an ablation can be priced without a twenty-arm session. Same
stopwatch, same interleaving, same warm-up discard, same VRAM guard.

Set `RUNG5B_ABLATE_COOK=1` to load the instrument build in which the cook is
replaced by the contiguous partition it must produce at dense. The difference
between the two runs is what the COOK costs; the difference between the ablation
and rung 5A is what the GATHERED ADDRESSING and the staircase cost.
"""
from __future__ import annotations

import statistics
import subprocess
import sys

import torch

ROOT = __file__.rsplit("/", 2)[0]
sys.path.insert(0, ROOT)
from rung5 import build as r5a  # noqa: E402
from rung5b import build as r5b  # noqa: E402
from rung5b import reference  # noqa: E402

BLOCKS, REPS, ROUNDS = 12, 20, 3
CELLS = {"A": (2, 8, 4096), "B": (20, 8, 1024)}
FLAG = (64, 64)
ARMS = [(16, 128), (32, 64), (32, 128), (64, 64), (32, 32)]


def stopwatch(fn):
    best = None
    for _ in range(ROUNDS):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        mins = []
        for _ in range(BLOCKS):
            ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(REPS)]
            for a, b in ev:
                a.record()
                fn()
                b.record()
            torch.cuda.synchronize()
            mins.append(min(a.elapsed_time(b) for a, b in ev))
        m = statistics.median(mins)
        best = m if best is None else min(best, m)
    return best


def clocks():
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()


def run(name, B, H, L):
    dev = "cuda"
    torch.manual_seed(0)
    v = torch.randn(B, H, L, 64, device=dev).bfloat16()
    pr = reference.amplitudes(B, H, L, FLAG, dev, seed=2)
    pw = reference.amplitudes(B, H, L, FLAG, dev, seed=3)
    print(f"\n=== CELL {name}: B={B} H={H} L={L}", flush=True)
    tbl0 = r5b.union_table(pr, pw, v, 32, 64, *FLAG)
    stopwatch(lambda: r5b.forward_raw(pr, pw, v, tbl0, 32, False, 64, *FLAG))
    print(f"    clocks before: {clocks()}", flush=True)
    print(f"{'arm':16s}{'rung5A ms':>11s}{'rung5B ms':>11s}{'delta':>9s}", flush=True)
    for c, bc in ARMS:
        if L % c:
            continue
        r5a.assert_build_identity(c, False, bc, *FLAG)
        r5b.assert_build_identity(c, False, bc, *FLAG)
        tbl = r5b.union_table(pr, pw, v, c, bc, *FLAG)
        a1 = stopwatch(lambda: r5a.forward_raw(pr, pw, v, tbl, c, False, bc, *FLAG))
        b1 = stopwatch(lambda: r5b.forward_raw(pr, pw, v, tbl, c, False, bc, *FLAG))
        b2 = stopwatch(lambda: r5b.forward_raw(pr, pw, v, tbl, c, False, bc, *FLAG))
        a2 = stopwatch(lambda: r5a.forward_raw(pr, pw, v, tbl, c, False, bc, *FLAG))
        a, b = min(a1, a2), min(b1, b2)
        print(f"{f'C={c} BC={bc}':16s}{a:11.4f}{b:11.4f}{100*(b-a)/a:+8.1f}%", flush=True)
    free, total = torch.cuda.mem_get_info()
    print(f"    clocks after: {clocks()}   device_used={(total-free)/(1<<30):.3f} GiB",
          flush=True)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    for nm in (CELLS if which == "both" else [which]):
        run(nm, *CELLS[nm])
