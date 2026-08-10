"""Wall clock for the ladder: the hardened base and rung 1, ARM FOR ARM, in ONE session.

Stopwatch: P51's protocol unchanged -- block-min-median, 12 blocks x 20 reps,
cuda events, best of 3 rounds -- so every row is commensurable with p57's tables.
The base arm is rebuilt and timed in the SAME PROCESS as the rung arm, which is
what makes the delta a same-session delta rather than a cross-session comparison.

MEASUREMENT HYGIENE, binding:
  * peak VRAM is recorded per cell and a cell above 10 GiB of the 12 GiB card is
    DISCARDED, not caveated: past the cap the driver falls back to system memory
    over PCIe and the stopwatch silently times the bus.
  * the B=20 H=8 L=4096 cell is WITHDRAWN on those grounds and is not offered.
  * SM clock and temperature are sampled before and after each cell; clocks
    cannot be pinned on this box (no permission), so the drift is REPORTED.

    python bench.py [cell]        cell in {A, B, both}
"""
from __future__ import annotations

import statistics
import subprocess
import sys

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = __file__.rsplit("/", 2)[0]
sys.path.insert(0, ROOT)
from denseref import build as dbuild  # noqa: E402
from rung1 import build as rbuild  # noqa: E402
from rung1 import reference  # noqa: E402

BLOCKS, REPS, ROUNDS = 12, 20, 3
VRAM_DISCARD_GIB = 10.0

#: the two VALID cells of p57b.  (B, H, L)
CELLS = {"A": (2, 8, 4096), "B": (20, 8, 1024)}


def clocks():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,clocks_throttle_reasons.active",
         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return out


def vram():
    free, total = torch.cuda.mem_get_info()
    return (total - free) / (1 << 30), total / (1 << 30)


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


def fa_layer_module(H, D, dev):
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
    return layer


def cell(name, B, H, L, D=64):
    dev = "cuda"
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    q = (torch.randn(B, H, L, D, device=dev) * 0.5).bfloat16()
    k = (torch.randn(B, H, L, D, device=dev) * 0.5).bfloat16()
    v = torch.randn(B, H, L, D, device=dev).bfloat16()
    widths = rbuild.WIDTHS
    pr = reference.amplitudes(B, H, L, widths, dev, seed=2)
    pw = reference.amplitudes(B, H, L, widths, dev, seed=3)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"\n=== CELL {name}: B={B} H={H} L={L} D={D}   grid = B*H = {B*H} CTAs on {sms} SMs")
    print(f"    clocks before: {clocks()}")

    arms = rbuild.arms()
    base, rung = {False: {}}, {}
    # INTERLEAVED, arm by arm: this box cannot pin its clocks, so a base block
    # followed by a rung block would put the whole rung on the hotter half of a
    # 2.6% thermal drift and charge it to the feature.
    for c, s, w, pk, _ in arms:
        if pk not in base:
            base[pk] = {}
        rbuild.assert_build_identity(c, s, w, pk)
        if (c, s, w) not in base[False]:
            dbuild.assert_build_identity(c, s, w)
            base[False][(c, s, w)] = stopwatch(
                lambda c=c, s=s, w=w: dbuild.forward(q, k, v, c, s, w))
        rung[(c, s, w, pk)] = stopwatch(
            lambda c=c, s=s, w=w, pk=pk: rbuild.forward(pr, pw, v, c, s, w, pk))

    def fa_core():
        with torch.no_grad(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    core = stopwatch(fa_core)

    layer = fa_layer_module(H, D, dev)
    x = torch.randn(B, L, H * D, device=dev, dtype=torch.bfloat16)

    def fa_l():
        with torch.no_grad(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return layer(x)
    lay = stopwatch(fa_l)

    used, total = vram()
    print(f"    clocks after:  {clocks()}")
    print(f"    VRAM: peak_alloc={torch.cuda.max_memory_allocated()/(1<<30):.3f} GiB  "
          f"device_used={used:.3f}/{total:.3f} GiB  "
          f"{'VALID' if used < VRAM_DISCARD_GIB else 'DISCARDED (>10 GiB)'}")
    if used >= VRAM_DISCARD_GIB:
        return None

    print(f"\n{'arm':24s}{'base ms':>10s}{'WIDEN ms':>10s}{'PACKED ms':>10s}"
          f"{'delta':>9s}{'ratio':>8s}{'/fa_core':>10s}")
    for c, s, w, pk, _ in arms:
        if pk:
            continue
        b = base[False][(c, s, w)]
        r, rp = rung[(c, s, w, False)], rung[(c, s, w, True)]
        print(f"{f'C={c} S={s} SWZ={int(w)}':24s}{b:10.4f}{r:10.4f}{rp:10.4f}"
              f"{r-b:+9.4f}{r/b:7.3f}x{r/core:9.2f}x")
    print(f"{'fa_core':22s}{core:10.4f}")
    print(f"{'fa_layer':22s}{lay:10.4f}")
    bb = min(base[False].items(), key=lambda kv: kv[1])
    rr = min(((k, v) for k, v in rung.items() if not k[3]), key=lambda kv: kv[1])
    print(f"    wall-optimal base  C={bb[0][0]} S={bb[0][1]} SWZ={int(bb[0][2])}  {bb[1]:.4f} ms")
    print(f"    wall-optimal rung1 C={rr[0][0]} S={rr[0][1]} SWZ={int(rr[0][2])}  {rr[1]:.4f} ms"
          f"   ({100*(rr[1]-bb[1])/bb[1]:+.1f}% vs the base's best)")

    el = B * H * L
    wt = sum(widths)
    base_b = el * (3 * D + D) * 2
    rung_b = el * (D + 2 * wt + D) * 2
    print(f"    compulsory global traffic  base {base_b/1e6:8.2f} MB   "
          f"rung1 {rung_b/1e6:8.2f} MB   {100*(rung_b-base_b)/base_b:+.1f}%  "
          f"({(3*D+D)*2} -> {(D+2*wt+D)*2} B/token)")
    # The cell's tensors go out of scope with this frame; the cache is released
    # here so the NEXT cell's peak-VRAM reading is its own and not a carry-over.
    torch.cuda.empty_cache()
    return True


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    for name in (CELLS if which == "both" else [which]):
        cell(name, *CELLS[name])
