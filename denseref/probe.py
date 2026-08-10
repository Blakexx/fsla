"""Convention and capability probes, both checked against torch.

FRAGMENT CONVENTION -- one m16n8k16 per ldmatrix mode. Modes 0/1 are the padded
layout, 2/3 the XOR-swizzled one, so the swizzle's address algebra is tested
against torch exactly as the padded one is.

CAPABILITY CONFORMANCE -- every baseline operation with more than one possible
implementation, in whichever implementation this architecture's capability row
selects. A flag no test exercises is a claim, not a capability. Today that is
`mul_bf16x2`, whose sm_86 implementation is the fp32 detour that `mul.rn.bf16x2`
would replace on sm_90; the contract both must satisfy is bf16 round-to-nearest
of the exact product, which is bit-for-bit what torch does.
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
row = build.arch_row()
sel = "mul.rn.bf16x2" if row["bf16_simd_arith"] else "fp32 detour (1 FMUL + 1/2 pack)"
x = (torch.randn(4096, device="cuda") * 3).bfloat16()
y = (torch.randn(4096, device="cuda") * 3).bfloat16()
got = m.caps_probe(x.contiguous(), y.contiguous())
expect = (x.float() * y.float()).bfloat16()
good = torch.equal(got, expect)
ok &= good
print(f"caps mul_bf16x2 [{sel}]: bit-exact vs torch  {'OK' if good else 'FAIL'}")

print("PROBE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
