"""Fragment-convention probe: one m16n8k16 per ldmatrix mode, checked against torch.

Modes 0/1 are the padded layout, 2/3 the XOR-swizzled one, so the swizzle's
address algebra is tested against torch exactly as the padded one is.
"""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from denseref import build  # noqa: E402

torch.manual_seed(1)
A = torch.randn(16, 16, device="cuda").bfloat16()
Bs = torch.randn(16, 16, device="cuda").bfloat16()
m = build.module()
NAME = {0: "pad  A[m][k] x B[n][k]", 1: "pad  A^T[k][m] x B[k][n]",
        2: "swz  A[m][k] x B[n][k]", 3: "swz  A^T[k][m] x B[k][n]"}
ok = True
for mode in (0, 1, 2, 3):
    expect = (A.double() @ Bs.double().T) if mode % 2 == 0 else (A.double().T @ Bs.double())
    got = m.probe(A.contiguous(), Bs.contiguous(), mode).double()
    e = (got - expect).abs().max().item()
    good = e < 4e-6 * max(1.0, expect.abs().max().item())
    ok &= good
    print(f"mode {mode} {NAME[mode]}: max abs err {e:.3e}   scale "
          f"{expect.abs().max().item():.3f}   {'OK' if good else 'FAIL'}")
print("PROBE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
