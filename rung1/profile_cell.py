"""One forward, for ncu.

    python profile_cell.py <arm> [L] [B] [H]

`arm` is `base<C>_<S>_<0|1 swizzle>`, `rung<C>_<0|1 swizzle>`, `fa_core` or
`fa_layer`. The rung's staging depth is derived, so it is not part of its name.
The build identity is asserted DEVICE-SIDE before the profiled launch, and the
peak VRAM is printed after it, so a row that ran near the 12 GiB cap is visible
in its own log rather than only in the harness that launched it.
"""
import sys

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = __file__.rsplit("/", 2)[0]
sys.path.insert(0, ROOT)
from denseref import build as dbuild  # noqa: E402
from rung1 import build as rbuild  # noqa: E402
from rung1 import reference  # noqa: E402

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


def parse(prefix):
    return (int(x) for x in ARM[len(prefix):].split("_"))


if ARM.startswith("base"):
    c, s, w = parse("base")
    print("BUILD IDENTITY:", dbuild.assert_build_identity(c, s, bool(w)), flush=True)

    def fwd():
        return dbuild.forward(q, k, v, c, s, bool(w))
elif ARM.startswith("rung"):
    c, w = parse("rung")
    print("BUILD IDENTITY:", rbuild.assert_build_identity(c, bool(w)), flush=True)
    pr = reference.amplitudes(B, H, L, rbuild.WIDTHS, dev, seed=2)
    pw = reference.amplitudes(B, H, L, rbuild.WIDTHS, dev, seed=3)

    def fwd():
        return rbuild.forward(pr, pw, v, c, bool(w))
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
