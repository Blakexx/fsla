"""One forward, for ncu.

    python profile_cell.py <arm> [L] [B] [H]

`arm` is `r4<C>_<0|1 swizzle>_<BC>_<W0>_<W1>_<regime>` for rung 5A, `r5<...>`
for rung 5B and `r6<...>` for this rung, where `regime` names an entry of
`reference.REGIMES`.
`dense` is the arm the STOP GATE is measured on, because compaction is provably
the identity there.

THE TABLE PASS IS NOT IN THE PROFILED LAUNCH. It is a separate kernel, built
before the capture, exactly as the divide pass is -- a profile of the chunk
kernel must not contain it. `r5t<...>` profiles the TABLE PASS on its own.

The build identity is asserted DEVICE-SIDE before the profiled launch, and the
skip fraction the schedule will produce is printed with it, so no report is ever
read without knowing what data it ran on.
"""
import sys

import torch

ROOT = __file__.rsplit("/", 2)[0]
sys.path.insert(0, ROOT)
from rung5 import build as r4build  # noqa: E402
from rung5b import build as r5build  # noqa: E402
from rung5b import reference  # noqa: E402
from rung5d import build as r6build  # noqa: E402

ARM = sys.argv[1]
L = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
B = int(sys.argv[3]) if len(sys.argv) > 3 else 1
H = int(sys.argv[4]) if len(sys.argv) > 4 else 2
dev = "cuda"
torch.manual_seed(0)
v = torch.randn(B, H, L, 64, device=dev).bfloat16()

tag, rest = (ARM[:3], ARM[3:]) if ARM.startswith("r5t") else (ARM[:2], ARM[2:])
c, w, bc, w0, w1, regime = rest.split("_")
c, w, bc, w0, w1 = (int(x) for x in (c, w, bc, w0, w1))
pr, pw = reference.regime_amplitudes(B, H, L, regime, dev, seed=5)
sched = reference.schedule(pr, pw, (w0, w1), bc, c)
plans = reference.plans_for(pr, pw, (w0, w1), bc, c)
sc = sum(len(x) for x in plans)
grp = int(sched[0, 0].sum().item())
print(f"SCHEDULE: regime={regime} live groups={100*sched.double().mean():.2f}%  "
      f"super-chunks={sc} live groups(b0h0)={grp}  grp/sc={grp/max(sc,1):.2f}",
      flush=True)

if tag == "r4":
    print("BUILD IDENTITY:",
          r4build.assert_build_identity(c, bool(w), bc, w0, w1), flush=True)
    tbl = r4build.union_table(pr, pw, v, c, bc, w0, w1)

    def fwd():
        return r4build.forward_raw(pr, pw, v, tbl, c, bool(w), bc, w0, w1)
elif tag == "r5":
    print("BUILD IDENTITY:",
          r5build.assert_build_identity(c, bool(w), bc, w0, w1), flush=True)
    tbl = r5build.union_table(pr, pw, v, c, bc, w0, w1)

    def fwd():
        return r5build.forward_raw(pr, pw, v, tbl, c, bool(w), bc, w0, w1)
elif tag == "r6":
    print("BUILD IDENTITY:",
          r6build.assert_build_identity(c, bool(w), bc, w0, w1), flush=True)
    tbl = r6build.union_table(pr, pw, v, c, bc, w0, w1)

    def fwd():
        return r6build.forward_raw(pr, pw, v, tbl, c, bool(w), bc, w0, w1)
elif tag == "r5t":
    print("BUILD IDENTITY:",
          r5build.assert_build_identity(c, bool(w), bc, w0, w1), flush=True)

    def fwd():
        return r5build.union_table(pr, pw, v, c, bc, w0, w1)
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
