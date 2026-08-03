"""Spec §7.1 page commitment: the measurement that justifies the feature.

Two questions, and they are different questions.

**1. Does committed state scale with realized usage or with `N`?** Run a topology
where `N` is far larger than the owner set any step actually writes -- a wide
mixed radix with a genuinely sparse (exact-entmax) write router -- and report
resident MiB against the dense `[BH, N, cols]` counterfactual. If resident MiB
tracks `N`, paging has bought nothing; the claim is that it tracks the REALIZED
owner count instead.

**2. What does paging cost where it can buy nothing?** The flagship cells are
dense: every owner is live, so the page set is the whole page domain and the
resident footprint is exactly the dense one. Paging must cost ~nothing there, and
"must" is not a measurement.

That cost is reported **in isolation** (`plan_cold_ms` / `plan_warm_ms`: the
`prepare -> publish -> admit-mask` plane on its own), not only as a whole-layer
delta. Measured on the first pass, the whole-layer delta at the dense flagship
cell ranged from `-12.7%` to `+37.1%` across four cells -- a spread with a
NEGATIVE member, i.e. the quantity is launch-noise dominated at a 5-7 ms step and
the delta does not measure paging. The isolated plane is a real number; the delta
is reported beside it and should be read as bounded by it, not as an independent
result.

**Cold vs warm (fix pass R5, `workflows/20260730-wiring/paging-review.md`).** The
layer resets the arena on every forward (`c469b117`; `PageArena.plan`'s own
`fresh=True` default since the BLOCKER B1 fix pass), so the plane a real step
pays is the COLD one: full reset, replan, republish. `plan_warm_ms` measures the
SAME arena re-planning the SAME schedule with `fresh=False` (no reset, so after
the first call there are zero new pages to commit) -- a real number for a
genuinely persistent-arena regime, but not the one the shipped layer runs. An
earlier revision of this benchmark measured only the warm number in a no-reset
loop and reported it as "the" paging cost; the two differ by 1.13x at the dense
flagship cell and by **4.74x** at the wide sparse cell (measured: 6.04 ms cold
vs 1.28 ms warm). Both are reported here, labelled, and neither is quoted alone.

The sparse level is a SYNTHETIC stand-in for training. `--logit-gain` sharpens an
untrained exact-entmax router into a real sparse support: the support is genuine
(exact zeros, spec §1.2), its DEGREE is not a trained quantity and is labelled as
such wherever it is reported. Same convention, and the same caveat, as
`benchmark_rola_v3_layer.py`.

Vocabulary: `N` = leaf count = `prod_l width_l`; `BC` = leaves per owner (spec §3);
`owners_total` = `N / BC`; `BH` = `batch * heads`; `cols` = `d_v` plus one mass
column under `state_norm='global'`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from rola.layers.rola import LevelRoutingConfig, RoLA
from rola.planner.schedule import build_schedule
from rola.routing.producer import produce_routing


def _routing_of(layer, x):
    """The layer's own routing allocations, on the layer's own inputs.

    Reproduces the forward's producer call rather than re-deriving it, so the
    schedule the paging plane is timed against is the schedule the layer actually
    hands the arena.
    """
    v = layer.v_proj(x).view(x.shape[0], x.shape[1], layer.num_heads, layer.head_v_dim)
    out = produce_routing(
        x, None, v, layer.route_W, layer.route_bias_param, layer.gain_W, layer.gain_bias,
        layer.topology, layer.state_norm,
        budget=layer.budget, r=None, budget_embed_W=layer.budget_embed_W,
        temp_W=layer.temp_W, temp_b=layer.temp_b,
        usage_bias_W=layer.usage_bias_W, usage_bias_b=layer.usage_bias_b)
    return out.read_levels, out.write_levels, out.g_read, out.g_write

_MiB = 1024 * 1024


def _median_ms(fn, *, reps: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def _layer(*, hidden, H, d_v, branches, router, norm, decay, paging, seed=0):
    side = "sparse" if router == "entmax" else "dense"
    torch.manual_seed(seed)
    layer = RoLA(
        hidden_size=hidden, num_heads=H, head_v_dim=d_v,
        levels_per_head=len(branches), states_per_level=branches,
        decay="leaf_mass" if decay else "none", state_norm=norm,
        level_routing_configs=LevelRoutingConfig(read=side, write=side, tied=False, alpha=1.5),
        paging=paging, layer_idx=0).cuda().eval()
    for param in layer.parameters():
        param.requires_grad_(False)
    return layer


def measure_residency(*, B, L, H, d_v, branches, router, logit_gain, norm, decay,
                      reps, warmup) -> dict:
    hidden = H * d_v
    layer = _layer(hidden=hidden, H=H, d_v=d_v, branches=branches, router=router,
                   norm=norm, decay=decay, paging=True)
    if logit_gain != 1.0:
        with torch.no_grad():
            layer.route_W.mul_(logit_gain)
    x = torch.randn(B, L, hidden, device="cuda")

    def step(module):
        # `torch.no_grad()` is load-bearing, not hygiene: the layer's dispatch sends a
        # grad-enabled forward to the fp64 oracle, which has no page table and RAISES
        # under `paging=True`. This measures the inference path, which is the path
        # paging exists on.
        def run():
            with torch.no_grad():
                module(x)
        return run

    step(layer)()
    assert layer.last_execution_backend == "cuda", (
        f"the paged cell took the {layer.last_execution_backend!r} backend, so this row "
        "would be an oracle timing")
    stats = layer.last_paging_stats

    # The paging plane ALONE: `prepare -> publish -> admit mask` against the exact
    # schedule this cell's routing produces. This is the number that answers "what
    # does paging cost", because the whole-layer delta at these sizes is dominated
    # by launch noise (see the module docstring).
    arena = layer._page_arena
    plan_obj = layer.last_selected_plan
    with torch.no_grad():
        routing = _routing_of(layer, x)
        schedule = build_schedule(routing[0], routing[1], routing[2], routing[3], plan_obj)

    # Fix pass R5 (`workflows/20260730-wiring/paging-review.md`): the layer calls
    # `arena.plan(schedule)` with its default `fresh=True` on EVERY forward (a
    # fresh forward resets the arena; `c469b117` fixed the defect where it did
    # not), so the plane a real step pays is the COLD one -- full reset,
    # replan, republish. The instrument this replaced timed a no-reset loop on
    # one arena (`fresh=False`, i.e. WARM: after rep 1 there is nothing left to
    # commit), which is a different, smaller number and was reported as if it
    # were the shipped cost. Both are measured and reported, labelled, so
    # neither is quoted alone.
    def plan_cold():
        with torch.no_grad():
            arena.plan(schedule)                # fresh=True (default): full reset + replan

    def plan_warm():
        with torch.no_grad():
            arena.plan(schedule, fresh=False)    # explicit continuation: no reset

    # `plan_warm` must start from a resident arena, or its own first call would be
    # doing the same new-page work `plan_cold` measures. One untimed cold call
    # establishes residency; every `fresh=False` call after commits nothing new.
    arena.plan(schedule)
    plan_warm_ms = _median_ms(plan_warm, reps=reps, warmup=warmup)
    plan_cold_ms = _median_ms(plan_cold, reps=reps, warmup=warmup)

    paged_ms = _median_ms(step(layer), reps=reps, warmup=warmup)
    unpaged = _layer(hidden=hidden, H=H, d_v=d_v, branches=branches, router=router,
                     norm=norm, decay=decay, paging=False)
    # `load_state_dict` carries the scaled `route_W` across, so both cells route
    # identically and the timing difference is paging and nothing else.
    unpaged.load_state_dict(layer.state_dict())
    step(unpaged)()
    assert unpaged.last_execution_backend == "cuda"
    unpaged_ms = _median_ms(step(unpaged), reps=reps, warmup=warmup)

    N = layer.topology.N
    return {
        "router": router, "logit_gain": logit_gain, "norm": norm,
        "decay": "leaf_mass" if decay else "none",
        "branches": list(branches), "N": N, "BC": layer.state_block,
        "BH": B * H, "L": L,
        "owners_total": stats.owners_total,
        "realized_owners": stats.realized_owners,
        "resident_pages": stats.resident_pages,
        "peak_pages": stats.peak_pages,
        "resident_MiB": stats.resident_bytes / _MiB,
        "dense_MiB": stats.dense_bytes / _MiB,
        "resident_fraction": stats.resident_fraction,
        "plan_cold_ms": plan_cold_ms,
        "plan_warm_ms": plan_warm_ms,
        "paged_ms": paged_ms,
        "unpaged_ms": unpaged_ms,
        "paging_overhead_pct": 100.0 * (paged_ms - unpaged_ms) / unpaged_ms,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-v-dim", type=int, default=64)
    #: Default is the wide cell: 16^4 = 65536 leaves, where `N` is far larger than
    #: any step's realized owner set. `8,16` (the flagship) is the dense-limit cell.
    ap.add_argument("--branches", default="16,16,16,16")
    ap.add_argument("--router", choices=("softmax", "entmax"), default="entmax")
    ap.add_argument("--logit-gain", type=float, default=8.0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("this benchmark requires CUDA")

    branches = [int(s) for s in args.branches.split(",")]
    rows = []
    for decay in (False, True):
        for norm in ("raw", "global"):
            rows.append(measure_residency(
                B=args.batch, L=args.seq_len, H=args.heads, d_v=args.head_v_dim,
                branches=branches, router=args.router, logit_gain=args.logit_gain,
                norm=norm, decay=decay, reps=args.reps, warmup=args.warmup))

    head = rows[0]
    print(f"\n=== V3 §7.1 page commitment — branches={head['branches']} "
          f"N={head['N']} BC={head['BC']} BH={head['BH']} L={head['L']} "
          f"router={args.router} logit_gain={args.logit_gain} ===")
    for r in rows:
        print(f"decay={r['decay']:9s} norm={r['norm']:7s} | "
              f"owners {r['resident_pages']:6d}/{r['owners_total'] * r['BH']:6d} | "
              f"resident {r['resident_MiB']:8.2f} MiB vs dense {r['dense_MiB']:8.2f} MiB "
              f"({100 * r['resident_fraction']:5.1f}%) | "
              f"plan[cold {r['plan_cold_ms']:6.3f} / warm {r['plan_warm_ms']:6.3f}] ms | "
              f"paged {r['paged_ms']:7.3f} vs unpaged {r['unpaged_ms']:7.3f} ms "
              f"({r['paging_overhead_pct']:+5.1f}%)")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(rows, handle, indent=2)


if __name__ == "__main__":
    main()
