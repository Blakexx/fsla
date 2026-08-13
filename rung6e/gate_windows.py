"""THE WINDOW GATES -- what changes past 65536 tokens, gated at the edge, at
one window exactly, and across a genuine line.

  (a) L = 65536: ONE window, the old refusal's edge -- must be BIT-IDENTICAL
      to rung 6 (which still accepts exactly 65536).
  (b) L = 131072 DENSE: two windows; at dense every cut falls at C and
      65536 % C == 0, so the partition is untouched by the line and the fp64
      compacted model applies EXACTLY.
  (c) L = 131072 SPARSE (support 2 of 8): a real line crossing under
      compaction, gated against naive_rola directly (the model's partition
      does not know windows, so only the oracle itself can gate this row).
"""
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from rola.ops.constants import READOUT_EPS  # noqa: E402
from rung5b import reference  # noqa: E402
from rung5d.check import regspec  # noqa: E402
from rung6 import build as b6  # noqa: E402
from rung6e import build as b6e  # noqa: E402
from rung6e.oracle_gate import oracle_ratio, rel  # noqa: E402

dev = "cuda"


def run(mod, pr, pw, v, c, bc, w0, w1):
    tbl = mod.union_table(pr, pw, v, c, bc, w0, w1)
    return mod.forward_raw(pr, pw, v, tbl, c, False, bc, w0, w1)


def main():
    torch.manual_seed(0)
    # (a) the edge: one full window, bit-identical to the refusal-era kernel
    w0, w1, bc, c = 8, 8, 64, 32
    L = 65536
    pr, pw = reference.regime_amplitudes(1, 1, L, regspec((w0, w1), "d4"), dev, seed=5)
    v = torch.randn(1, 1, L, 64, device=dev).bfloat16()
    n0, d0 = run(b6.module(), pr, pw, v, c, bc, w0, w1)
    n1, d1 = run(b6e.module(), pr, pw, v, c, bc, w0, w1)
    assert torch.equal(n0, n1) and torch.equal(d0, d1), "edge L=65536 NOT bit-identical"
    print("(a) L=65536 one-window edge: BIT-IDENTICAL to rung 6")

    # (b) two windows, dense: the partition is line-invariant, model applies
    L = 131072
    pr = reference.amplitudes(1, 1, L, (w0, w1), dev, seed=2)
    pw = reference.amplitudes(1, 1, L, (w0, w1), dev, seed=3)
    v = torch.randn(1, 1, L, 64, device=dev).bfloat16()
    n, d = run(b6e.module(), pr, pw, v, c, bc, w0, w1)
    y = b6e.divide_pass(n, d, READOUT_EPS)
    md = reference.ratio(
        reference.model_compacted(pr, pw, v, (w0, w1), c, bc), READOUT_EPS)
    yo = oracle_ratio(v, pr, pw, (w0, w1))
    r, floor = rel(y, yo), rel(md, yo)
    print(f"(b) L=131072 dense two-window: rel {r:.3e} floor {floor:.3e} "
          f"ratio {r/max(floor,1e-12):.4f}")
    assert r / max(floor, 1e-12) <= 1.05, "dense two-window off floor"

    # (c) two windows, sparse: a genuine line crossing under compaction
    pr, pw = reference.regime_amplitudes(1, 1, L, regspec((w0, w1), "d4"), dev, seed=5)
    n, d = run(b6e.module(), pr, pw, v, c, bc, w0, w1)
    y = b6e.divide_pass(n, d, READOUT_EPS)
    yo = oracle_ratio(v, pr, pw, (w0, w1))
    r = rel(y, yo)
    print(f"(c) L=131072 sparse (d4) two-window: rel_L2 vs naive {r:.3e}")
    assert r < 5e-3, "sparse two-window disagrees with the oracle"
    print("WINDOW GATES: PASS")


if __name__ == "__main__":
    main()
