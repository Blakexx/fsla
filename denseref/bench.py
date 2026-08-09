"""Wall clock, on the P51 protocol so the numbers are commensurable.

Stopwatch: block-min-median, 12 blocks x 20 reps, cuda events, best of 3 rounds
(P51 `whole_layer.py`). Two FA references are timed in the SAME process:

  fa_layer  -- the P51 whole-layer proxy: 4 linear projections + causal SDPA
               (FLASH_ATTENTION backend). This is the arm P51 reports 0.9170 ms
               for, and is the only apples-to-apples anchor against prior rows.
  fa_core   -- causal SDPA alone on the same [B,H,L,64] tensors, no projections.
               This is the fair comparison for a BARE kernel, which is what the
               dense reference is.

    python bench.py [L] [B] [H]
"""
from __future__ import annotations

import statistics
import sys

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from denseref import build  # noqa: E402

BLOCKS, REPS, ROUNDS = 12, 20, 3


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


def main(L=4096, B=2, H=8, D=64):
    dev = "cuda"
    torch.manual_seed(0)
    q = (torch.randn(B, H, L, D, device=dev) * 0.5).bfloat16()
    k = (torch.randn(B, H, L, D, device=dev) * 0.5).bfloat16()
    v = torch.randn(B, H, L, D, device=dev).bfloat16()
    print(f"cell B={B} H={H} L={L} D={D}   grid = B*H = {B*H} CTAs on "
          f"{torch.cuda.get_device_properties(0).multi_processor_count} SMs")

    rows = {}
    for c in (16, 32, 64):
        st = build.assert_build_identity(c)
        print(f"  build identity chunk={c}: magic={st['magic']} warps={st['warps']} "
              f"regs/thread={st['regs_per_thread']} dyn_smem={st['dynamic_smem']} "
              f"binary=sm_{st['binary_version']}")
        rows[f"denseref C={c}"] = stopwatch(lambda c=c: build.forward(q, k, v, c))

    def fa_core():
        with torch.no_grad(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    rows["fa_core (SDPA flash, causal, same tensors)"] = stopwatch(fa_core)

    HIDDEN = H * D

    class MatchedAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
            self.k_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
            self.v_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
            self.o_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)

        def forward(self, x):
            Bb, Lx, _ = x.shape

            def split(t):
                return t.view(Bb, Lx, H, D).transpose(1, 2)
            qq, kk, vv = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))
            o = torch.nn.functional.scaled_dot_product_attention(qq, kk, vv, is_causal=True)
            return self.o_proj(o.transpose(1, 2).reshape(Bb, Lx, HIDDEN))

    layer = MatchedAttention().to(dev, torch.bfloat16).eval()
    for p in layer.parameters():
        p.requires_grad_(False)
    x = torch.randn(B, L, HIDDEN, device=dev, dtype=torch.bfloat16)

    def fa_layer():
        with torch.no_grad(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return layer(x)
    rows["fa_layer (P51 whole-layer proxy)"] = stopwatch(fa_layer)

    core = rows["fa_core (SDPA flash, causal, same tensors)"]
    lay = rows["fa_layer (P51 whole-layer proxy)"]
    print(f"\n{'arm':46s} {'ms':>9s} {'/fa_core':>9s} {'/fa_layer':>10s}")
    for name, ms in rows.items():
        print(f"{name:46s} {ms:9.4f} {ms/core:9.2f}x {ms/lay:10.2f}x")


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:]]
    main(*(a or [4096]))
