"""One forward, for ncu.

    python profile_cell.py <arm> [L] [B] [H]

`arm` is `r2<C>_<0|1 swizzle>_<BC>_<W0>_<W1>` for rung 2 or
`r3<C>_<0|1>_<BC>_<W0>_<W1>_<regime>` for this rung, where `regime` names an
entry of `reference.REGIMES` -- the topology and the placement of each side's
sparsity relative to the block boundary, which is what decides whether the
schedule can fire at all. `dense` is the arm whose cost the rung is really
reporting. The staging depth, the block plan and the schedule's capacity are
derived, so none of them is part of an arm's name.

The build identity is asserted DEVICE-SIDE before the profiled launch, and the
skip fraction the schedule will produce is printed with it, so no report is ever
read without knowing what data it ran on.
"""
import sys

import torch

ROOT = __file__.rsplit("/", 2)[0]
sys.path.insert(0, ROOT)
from rung2 import build as r2build  # noqa: E402
from rung3 import build as r3build  # noqa: E402
from rung3 import reference  # noqa: E402

ARM = sys.argv[1]
L = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
B = int(sys.argv[3]) if len(sys.argv) > 3 else 1
H = int(sys.argv[4]) if len(sys.argv) > 4 else 2
dev = "cuda"
torch.manual_seed(0)
v = torch.randn(B, H, L, 64, device=dev).bfloat16()

if ARM.startswith("r2"):
    c, w, bc, w0, w1 = (int(x) for x in ARM[2:].split("_"))
    print("BUILD IDENTITY:",
          r2build.assert_build_identity(c, bool(w), bc, w0, w1), flush=True)
    pr = reference.amplitudes(B, H, L, (w0, w1), dev, seed=2)
    pw = reference.amplitudes(B, H, L, (w0, w1), dev, seed=3)

    def fwd():
        return r2build.forward(pr, pw, v, c, bool(w), bc, w0, w1)
elif ARM.startswith("r3"):
    c, w, bc, w0, w1, regime = ARM[2:].split("_")
    c, w, bc, w0, w1 = (int(x) for x in (c, w, bc, w0, w1))
    print("BUILD IDENTITY:",
          r3build.assert_build_identity(c, bool(w), bc, w0, w1), flush=True)
    pr, pw = reference.regime_amplitudes(B, H, L, regime, dev, seed=5)
    sched = reference.schedule(pr, pw, (w0, w1), bc, c)
    print(f"SCHEDULE: regime={regime} live={100*sched.double().mean():.2f}% "
          f"skipped={100*(1-sched.double().mean()):.2f}%", flush=True)

    def fwd():
        return r3build.forward(pr, pw, v, c, bool(w), bc, w0, w1)
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
