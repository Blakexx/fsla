"""Wall clock, on the P51 protocol so the numbers are commensurable.

Stopwatch: block-min-median, 12 blocks x 20 reps, cuda events, best of 3 rounds
(P51 `whole_layer.py`).  Two FA references are timed in the SAME process:

  fa_layer  -- the P51 whole-layer proxy: 4 linear projections + causal SDPA
               (FLASH_ATTENTION backend).
  fa_core   -- causal SDPA alone on the same [B,H,L,64] tensors, no projections.

Every measurement is bracketed by a clock/thermal sample, because the FA proxy
has been observed to swing 2.4x between sessions and a downclock is the leading
hypothesis.  `--fa-repeat N` re-times the FA arms N times, interleaved with the
kernel arms, so drift within a session is visible rather than assumed away.

    python bench.py [--L 4096] [--B 2] [--H 8] [--fa-repeat 1] [--arms all|head]
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from denseref import build  # noqa: E402

BLOCKS, REPS, ROUNDS = 12, 20, 3

CLOCK_Q = ("clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,"
           "power.limit,utilization.gpu,clocks_throttle_reasons.active")


def clocks() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={CLOCK_Q}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().split(", ")
        keys = CLOCK_Q.split(",")
        return dict(zip(keys, out))
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}


def fmt_clocks(c: dict) -> str:
    return (f"sm {c.get('clocks.sm','?')}/{c.get('clocks.max.sm','?')} MHz  "
            f"mem {c.get('clocks.mem','?')} MHz  {c.get('temperature.gpu','?')}C  "
            f"{c.get('power.draw','?')}/{c.get('power.limit','?')} W  "
            f"throttle {c.get('clocks_throttle_reasons.active','?')}")


#: hard gate.  This is a 12 GB card on WSL2, where exhausting VRAM does not
#: OOM -- the driver falls back to system memory over PCIe and the stopwatch
#: silently starts timing the bus.  Any row above this is DISCARDED, not
#: caveated.  (This is what made the B=20 H=8 L=4096 FA proxy swing 2.4x
#: between sessions; that cell is unmeasurable on this device.)
VRAM_CEILING_B = 10 * (1 << 30)


def vram() -> dict:
    free, total = torch.cuda.mem_get_info()
    return dict(alloc=torch.cuda.max_memory_allocated(),
                reserved=torch.cuda.max_memory_reserved(),
                dev_used=total - free, dev_total=total)


def fmt_vram(v: dict) -> str:
    g = 1 << 30
    return (f"peak alloc {v['alloc']/g:.2f} GiB  reserved {v['reserved']/g:.2f} GiB  "
            f"device used {v['dev_used']/g:.2f}/{v['dev_total']/g:.2f} GiB")


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


def make_fa(q, k, v, B, H, L, D, dev):
    def fa_core():
        with torch.no_grad(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    HIDDEN = H * D

    class MatchedAttention(nn.Module):
        def __init__(self):
            super().__init__()
            for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(self, n, nn.Linear(HIDDEN, HIDDEN, bias=False))

        def forward(self, x):
            Bb, Lx, _ = x.shape

            def split(t):
                return t.view(Bb, Lx, H, D).transpose(1, 2)
            o = torch.nn.functional.scaled_dot_product_attention(
                split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x)),
                is_causal=True)
            return self.o_proj(o.transpose(1, 2).reshape(Bb, Lx, HIDDEN))

    layer = MatchedAttention().to(dev, torch.bfloat16).eval()
    for p in layer.parameters():
        p.requires_grad_(False)
    x = torch.randn(B, L, HIDDEN, device=dev, dtype=torch.bfloat16)

    def fa_layer():
        with torch.no_grad(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return layer(x)
    return fa_core, fa_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=4096)
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--fa-repeat", type=int, default=1)
    ap.add_argument("--arms", default="all")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    L, B, H, D, dev = a.L, a.B, a.H, 64, "cuda"

    torch.manual_seed(0)
    q = (torch.randn(B, H, L, D, device=dev) * 0.5).bfloat16()
    k = (torch.randn(B, H, L, D, device=dev) * 0.5).bfloat16()
    v = torch.randn(B, H, L, D, device=dev).bfloat16()
    nsm = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"cell B={B} H={H} L={L} D={D}   grid = B*H = {B*H} CTAs on {nsm} SMs")
    print(f"[clocks pre ] {fmt_clocks(clocks())}")
    print(f"[vram   pre ] {fmt_vram(vram())}   ceiling "
          f"{VRAM_CEILING_B / (1<<30):.0f} GiB")

    arms = build.arms()
    if a.arms == "head":
        arms = [x for x in arms if (x[0], x[1], x[2]) in
                {(16, 4, False), (32, 4, False), (64, 3, False),
                 (16, 3, True), (32, 3, True), (64, 3, True)}]
    fa_core, fa_layer = make_fa(q, k, v, B, H, L, D, dev)

    rows, tele = {}, []

    def timed(name, fn):
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        ms = stopwatch(fn)
        c, vm = clocks(), vram()
        ok = vm["dev_used"] <= VRAM_CEILING_B
        tele.append(dict(name=name, ms=ms, t=t0, valid=ok, **c, **vm))
        print(f"  {name:34s} {ms:9.4f} ms   [{fmt_clocks(c)}] [{fmt_vram(vm)}]"
              f"{'' if ok else '   *** DISCARDED: OVER VRAM CEILING ***'}", flush=True)
        return ms if ok else None

    for r in range(a.fa_repeat):
        rows[f"fa_core#{r}"] = timed(f"fa_core#{r}", fa_core)
        rows[f"fa_layer#{r}"] = timed(f"fa_layer#{r}", fa_layer)
        for c_, s_, w_, sm_ in arms:
            st = build.assert_build_identity(c_, s_, w_)
            key = f"denseref C={c_} S={s_} SWZ={int(w_)}"
            rows[f"{key}#{r}"] = timed(f"{key}#{r}",
                                       lambda c_=c_, s_=s_, w_=w_: build.forward(q, k, v, c_, s_, w_))
            tele[-1]["regs"] = st["regs_per_thread"]
            tele[-1]["smem"] = sm_

    print(f"[clocks post] {fmt_clocks(clocks())}")
    rows = {k_: v_ for k_, v_ in rows.items() if v_ is not None}
    core = min(v_ for k_, v_ in rows.items() if k_.startswith("fa_core"))
    lay = min(v_ for k_, v_ in rows.items() if k_.startswith("fa_layer"))
    print(f"\n{'arm':38s} {'ms':>9s} {'/fa_core':>9s} {'/fa_layer':>10s}")
    for name, ms in sorted(rows.items(), key=lambda kv: kv[1]):
        print(f"{name:38s} {ms:9.4f} {ms/core:9.2f}x {ms/lay:10.2f}x")
    if a.fa_repeat > 1:
        for pre in ("fa_core", "fa_layer"):
            xs = [v_ for k_, v_ in rows.items() if k_.startswith(pre)]
            print(f"\n{pre} across {len(xs)} interleaved timings: min {min(xs):.4f} "
                  f"max {max(xs):.4f} spread {max(xs)/min(xs):.3f}x  values "
                  f"{[round(x,4) for x in xs]}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(dict(cell=dict(B=B, H=H, L=L), rows=rows, tele=tele), fh, indent=1)


if __name__ == "__main__":
    main()
