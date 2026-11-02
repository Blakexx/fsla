#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE SCALE'S SLACK: whether a bound cheaper than the den pass's can stand as the scale of the carry's f16 sum across
CTAs. The carry runs on one bundle at the den pass's scale bytes twice and once at each candidate's, its f16-pair
``num`` taken back by each token's power of two; the sum's order across CTAs is the atomics' arrival, so two runs at
one scale differ by f16's rounding already -- a candidate is harmless where its spread from the den-pass run is that
run-to-run spread. Beside it, each token's slack: log2 of the bound over the exact read product.

    python tools/scale_slack.py nl64k-alt-k4 flagship-cohort-k16-L16384 [--json slack.json]   # carry cells
    python tools/scale_slack.py nl64k-alt-h4 nl64k-alt-h4-g16                                 # layer constructions

THE BOUNDS on ``|num(t)|`` (``x`` a read row, ``w`` a write row, ``g`` the write gain, ``s`` a writing token's value
scale, each sum over the writing tokens of the windows before t's), in the recurrence's units:
  exact  sum_a x_S(t, a) sum_r x_R(t, r) M(a, r),  M(a, r) = sum w_S(a) w_R(r) g s   (the den pass's num)
  write  (sum_r x_R(t, r)) sum_a x_S(t, a) G(a),     G(a) = sum w_S(a) max_r w_R(r) g s
  facts  (sum_a x_S(t, a)) (sum_r x_R(t, r)) sum max_a w_S(a) max_r w_R(r) g s
the scale bytes take each in the den pass's units: the reader's per-token row scales left out (`den_units`).

Docs: docs/internals/tools/scale_slack.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def bounds(routes, values) -> dict:
    """Each bound ``[B H, T]`` fp64, a query head's (its write head's mass), window by window at the device's W."""
    import torch

    from rola.ops.window import window_of

    W = window_of(routes.device)
    ls, lr = routes.level_s, 1 - routes.level_s
    read, write = routes.dequantized()
    x_s, x_r = read[ls].double(), read[lr].double()                               # [B, T, H, w]
    w_s, w_r = write[ls].double(), write[lr].double()                             # [B, T, Hw, w]
    gs = routes.g_write.double() * values.scale.double()                          # [B, T, Hw]
    B, T, H, _ = x_s.shape
    g = routes.group
    out = {k: torch.zeros(B, T, H, dtype=torch.float64, device=x_s.device) for k in ("exact", "write", "facts")}
    M = torch.zeros(B, w_s.shape[2], w_s.shape[-1], w_r.shape[-1], dtype=torch.float64, device=x_s.device)
    G = torch.zeros(M.shape[:-1], dtype=torch.float64, device=x_s.device)
    F = torch.zeros(M.shape[:2], dtype=torch.float64, device=x_s.device)

    def heads(m):
        return m.repeat_interleave(g, 1)                                          # a query head its write head's

    for t0 in range(0, T, W):
        #: the window's tokens read the mass of the windows before, then its writes join it
        sl = slice(t0, min(t0 + W, T))
        xs, xr = x_s[:, sl], x_r[:, sl]
        out["exact"][:, sl] = torch.einsum("btha,bthr,bhar->bth", xs, xr, heads(M))
        out["write"][:, sl] = xr.sum(-1) * torch.einsum("btha,bha->bth", xs, heads(G))
        out["facts"][:, sl] = xs.sum(-1) * xr.sum(-1) * heads(F)[:, None]
        ws, wr, f = w_s[:, sl], w_r[:, sl], gs[:, sl]
        M += torch.einsum("btha,bthr,bth->bhar", ws, wr, f)
        G += torch.einsum("btha,bth->bha", ws, wr.amax(-1) * f)
        F += (ws.amax(-1) * wr.amax(-1) * f).sum(1)
    return {k: v.permute(0, 2, 1).reshape(B * H, T).contiguous() for k, v in out.items()}


def den_units(routes):
    """``[B H, T]`` fp64: what the den pass's sums are multiplied by to be the recurrence's (`test_den_vs_oracle`'s
    ``in_truth``): each read level's per-token row scale, a byte level S's coefficient over 256, a byte write level R's
    mass on the byte scale."""
    ls, lr = routes.level_s, 1 - routes.level_s
    f = routes.scale_read[ls].double() * routes.scale_read[lr].double()          # [B, T, H]
    if routes.sparse[0][ls]:
        f = f * (256.0 / 255.0)
    if routes.sparse[1][lr]:
        f = f / 255.0
    B, T, H = f.shape
    return f.permute(0, 2, 1).reshape(B * H, T)


def operands(name: str):
    """``(routes, values, facts, carve)``: a carry cell's (`measure.subjects.carry_operands`), or a layer
    construction's on its own cell, the producer run once (`measure.subjects._layer_facts`)."""
    from measure.cells import by_name
    from measure.cells.layer import CONSTRUCTIONS, build

    if name in CONSTRUCTIONS:
        from measure.subjects import _layer_facts

        construction = CONSTRUCTIONS[name]
        routes, values, _layout, _statics, carve, facts = _layer_facts(build(by_name(construction.cells[0]),
                                                                             construction))
        return routes, values, facts, carve
    from measure.subjects import carry_operands

    routes, values, facts, carve, _zero = carry_operands(by_name(name))
    return routes, values, facts, carve


def reading(name: str) -> dict:
    """One cell: the den pass's bound against the exact product, each candidate's slack, and each candidate's spread
    from the den-pass run against two den-pass runs' own."""
    import torch

    from rola.ops import carry as carry_ops
    from rola.ops import den as den_ops
    from rola.ops.prefill import scale_bytes, unpack_num

    routes, values, facts, carve = operands(name)
    _den, bound = den_ops.den(routes, facts)
    true = bounds(routes, values)
    unit = den_units(routes)
    exact = true["exact"]
    live = exact > 0

    def num_at(scale):
        num, _den, _ = carry_ops.carry(routes, values, facts, carve, scale, reduce_dtype=torch.float16)
        return unpack_num(num, 32).double() * torch.exp2(scale.double() + 1)[..., None]

    def spread(got, ref):
        r = ((got - ref).abs().amax(-1) / ref.abs().amax(-1).clamp_min(1e-300))[live]
        return {"median": float(r.median()), "max": float(r.max()), "inf": int((~torch.isfinite(got)).sum())}

    def slack(b):
        r = torch.log2(b[live] / exact[live])
        q = torch.quantile(r[:: max(1, r.numel() // (1 << 24))],
                           torch.tensor([0.5, 0.99], dtype=torch.float64, device=r.device))
        return {"min": round(float(r.min()), 3), "median": round(float(q[0]), 2), "p99": round(float(q[1]), 2),
                "max": round(float(r.max()), 2)}

    ref_scale = scale_bytes(routes, bound)
    ref = num_at(ref_scale)
    out = {"cell": name, "tokens": int(live.sum()), "den_pass": slack(bound.double() * unit),
           "run_to_run": spread(num_at(ref_scale), ref), "candidates": {}}
    for key in ("write", "facts"):
        got = num_at(scale_bytes(routes, (true[key] / unit).float()))
        out["candidates"][key] = {"slack": slack(true[key]), "spread": spread(got, ref),
                                  "below_exact": int((true[key][live] < exact[live] * (1 - 1e-9)).sum())}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cells", nargs="+")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    rows = []
    for c in a.cells:
        r = reading(c)
        rows.append(r)
        d, rr = r["den_pass"], r["run_to_run"]
        print(f"{r['cell']}: {r['tokens']} tokens; den pass / exact log2 {d['min']} .. {d['max']};"
              f" run to run median {rr['median']:.2e} max {rr['max']:.2e}")
        for key, c_ in r["candidates"].items():
            s, sp = c_["slack"], c_["spread"]
            print(f"  {key:5s} slack log2 median {s['median']} p99 {s['p99']} max {s['max']} (min {s['min']},"
                  f" {c_['below_exact']} below exact); spread median {sp['median']:.2e} max {sp['max']:.2e}"
                  f" inf {sp['inf']}")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
