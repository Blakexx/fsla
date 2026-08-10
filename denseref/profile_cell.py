"""One forward, for ncu. arg1 = arm (denseref16|denseref32|denseref64|fa_core|fa_layer)."""
import sys

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from denseref import build  # noqa: E402

ARM = sys.argv[1]
L = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
B = int(sys.argv[3]) if len(sys.argv) > 3 else 2
H = int(sys.argv[4]) if len(sys.argv) > 4 else 8
D = 64
dev = "cuda"
torch.manual_seed(0)
q = (torch.randn(B, H, L, D, device=dev) * 0.5).bfloat16()
k = (torch.randn(B, H, L, D, device=dev) * 0.5).bfloat16()
v = torch.randn(B, H, L, D, device=dev).bfloat16()

if ARM.startswith("denseref"):
    # denseref<C>_<STAGES>_<0|1 swizzle>
    c, stg, swz = (int(x) for x in ARM[len("denseref"):].split("_"))
    st = build.assert_build_identity(c, stg, bool(swz))
    print("BUILD IDENTITY:", st, flush=True)
    def fwd():
        return build.forward(q, k, v, c, stg, bool(swz))
elif ARM == "fa_core":
    def fwd():
        with torch.no_grad(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
elif ARM == "fa_layer":
    HIDDEN = H * D
    class MA(nn.Module):
        def __init__(s):
            super().__init__()
            for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(s, n, nn.Linear(HIDDEN, HIDDEN, bias=False))
        def forward(s, x):
            Bb, Lx, _ = x.shape
            def sp(t):
                return t.view(Bb, Lx, H, D).transpose(1, 2)
            o = torch.nn.functional.scaled_dot_product_attention(
                sp(s.q_proj(x)), sp(s.k_proj(x)), sp(s.v_proj(x)), is_causal=True)
            return s.o_proj(o.transpose(1, 2).reshape(Bb, Lx, HIDDEN))
    layer = MA().to(dev, torch.bfloat16).eval()
    for p in layer.parameters():
        p.requires_grad_(False)
    x = torch.randn(B, L, HIDDEN, device=dev, dtype=torch.bfloat16)
    def fwd():
        with torch.no_grad(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return layer(x)
else:
    raise SystemExit(ARM)

for _ in range(5):
    fwd()
torch.cuda.synchronize()
fwd()
torch.cuda.synchronize()
free, total = torch.cuda.mem_get_info()
print(f"VRAM: peak_alloc={torch.cuda.max_memory_allocated()/(1<<30):.3f} GiB "
      f"device_used={(total-free)/(1<<30):.3f}/{total/(1<<30):.3f} GiB", flush=True)
print("OK", ARM, flush=True)
