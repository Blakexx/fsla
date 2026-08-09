"""Fragment-convention probe: one m16n8k16 per ldmatrix mode, checked against torch."""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from denseref import build  # noqa: E402

torch.manual_seed(1)
A = torch.randn(16, 16, device="cuda").bfloat16()
Bs = torch.randn(16, 16, device="cuda").bfloat16()
m = build.module()
for mode, expect in ((0, A.double() @ Bs.double().T), (1, A.double().T @ Bs.double())):
    got = m.probe(A.contiguous(), Bs.contiguous(), mode).double()
    e = (got - expect).abs().max().item()
    print(f"mode {mode}: max abs err {e:.3e}   scale {expect.abs().max().item():.3f}")
