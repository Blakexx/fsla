"""One forward, for ncu.

    python profile_cell.py <arm> [L] [B] [H]

`arm` is `r1<C>_<0|1 swizzle>` for the ladder's rung 1, or
`r2<C>_<0|1>_<BC>_<W0>_<W1>` for this rung. The staging depth and the block plan
are derived, so neither is part of an arm's name. The build identity is asserted
DEVICE-SIDE before the profiled launch, and the peak VRAM is printed after it.
"""
import sys

import torch

ROOT = __file__.rsplit("/", 2)[0]
sys.path.insert(0, ROOT)
from rung1 import build as r1build  # noqa: E402
from rung2 import build as r2build  # noqa: E402
from rung2 import reference  # noqa: E402

ARM = sys.argv[1]
L = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
B = int(sys.argv[3]) if len(sys.argv) > 3 else 1
H = int(sys.argv[4]) if len(sys.argv) > 4 else 2
dev = "cuda"
torch.manual_seed(0)
v = torch.randn(B, H, L, 64, device=dev).bfloat16()

if ARM.startswith("r1"):
    c, w = (int(x) for x in ARM[2:].split("_"))
    print("BUILD IDENTITY:", r1build.assert_build_identity(c, bool(w)), flush=True)
    widths = r1build.WIDTHS
    pr = reference.amplitudes(B, H, L, widths, dev, seed=2)
    pw = reference.amplitudes(B, H, L, widths, dev, seed=3)

    def fwd():
        return r1build.forward(pr, pw, v, c, bool(w))
elif ARM.startswith("r2"):
    c, w, bc, w0, w1 = (int(x) for x in ARM[2:].split("_"))
    print("BUILD IDENTITY:",
          r2build.assert_build_identity(c, bool(w), bc, w0, w1), flush=True)
    pr = reference.amplitudes(B, H, L, (w0, w1), dev, seed=2)
    pw = reference.amplitudes(B, H, L, (w0, w1), dev, seed=3)

    def fwd():
        return r2build.forward(pr, pw, v, c, bool(w), bc, w0, w1)
else:
    raise SystemExit(ARM)

for _ in range(3):
    fwd()
torch.cuda.synchronize()
fwd()
torch.cuda.synchronize()
free, total = torch.cuda.mem_get_info()
print(f"VRAM: peak_alloc={torch.cuda.max_memory_allocated()/(1<<30):.3f} GiB "
      f"device_used={(total-free)/(1<<30):.3f}/{total/(1<<30):.3f} GiB", flush=True)
print("OK", ARM, flush=True)
