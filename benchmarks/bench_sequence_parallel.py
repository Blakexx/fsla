"""§6.4 sequence parallelism: what a per-owner `P` split actually costs, measured.

`P = 1` IS THE PRIMARY NUMBER (spec §6.4). Every `P > 1` row this script prints is
labelled **supplemental** in the output itself, because a headline quoted at `P > 1`
is a reporting defect and a label attached in a findings doc is a label that can be
detached from the number.

Symbols, defined at first use: `B` batch, `H` heads, `L` sequence length, `d_v`
per-head value dim, `BC` leafs per owner block, `BT` token tile, `P_o` the number of
sequence segments owner `o` is split into, `n_slots = sum_o P_o` over SPLIT owners --
the `P_o - 1` boundaries plus one entry-state slot each (fix pass R1) -- the workspace
slot count, `cols = d_v + 1` under `global` normalization and `d_v` under `raw`.

WHAT THIS GRADES. `sequence_parallel.py`'s bandwidth gate prices a split's roundtrip
at `roundtrip_bytes / state_roundtrip_bytes_per_s(BC * cols * 4)`, with the rate
transcribed from `bench_state_roundtrip`. That price has never been checked against a
running kernel. Here it is: `wall(P) - wall(P=1)` is the time the split actually cost,
and it is printed next to `sp.roundtrip_seconds` plainly and as a ratio.
[[matching-the-naive]]: the model is NOT tuned to fit. If it is wrong the ratio says so.

ONE CONFOUND, MEASURED RATHER THAN ARGUED AWAY. `_consumer_forward` builds the
sequence-parallel work lists and workspace INSIDE the call (`v3_consumer.py:270`), so
a `P > 1` wall-clock contains CPU apportionment plus H2D copies that a `P = 1` call
does not perform, and those are not roundtrip traffic. This script therefore also
times `build_sequence_parallel_tensors` standalone at each `P` and reports the
build-subtracted delta alongside the raw one. The subtracted number is the one the
cost model should be graded against; both are printed, and neither is hidden.

Usage (ALL GPU work belongs under `flock /tmp/rola_gpu.lock`):

    flock /tmp/rola_gpu.lock env PYTHONPATH=<worktree> python \
        benchmarks/ops/benchmark_rola_v3_sequence_parallel.py
"""
from __future__ import annotations

import argparse
import json
import subprocess

import torch

# G16 (THUNDER_RUN_SPEC.md §8b item 3): the COST MODEL moved to
# `benchmarks/planner_calibration/` in the Track P retirement (P4) -- `schedule.py` keeps
# the SCHEDULE, the calibration package owns the pricing. `schedule.py`'s PEP-562 lazy
# proxies still resolve these names, but importing them from their real home is what
# keeps `import rola` free of the calibration package's inductor edge.
from planner_calibration.cost_model import plan_and_schedule

from rola.layers import RoLA
from rola.layers.rola import LevelRoutingConfig
from rola.ops.consumer import _consumer_forward
from rola.planner.schedule import Plan, build_schedule
from rola.routing.producer import produce_routing
from rola.routing.sequence_parallel import (
    MEASURED_HBM_BYTES_PER_S,
    SequenceParallelPlan,
    SequenceParallelPolicy,
    build_sequence_parallel_tensors,
    consumer_blocks_per_sm,
    owner_slot_counts,
    select_sequence_parallelism,
    slot_bytes,
    state_roundtrip_bytes_per_s,
)

#: Extra warmup paid ONCE per process, before anything is timed. The V3 extension
#: JIT-compiles/loads its first template instantiation on first call and the caching
#: allocator grows on first use; `benchmark_rola_v3_consumer.py` measured a 6x
#: inflation on whichever cell happened to be first. Paid, not footnoted.
_FIRST_CELL_EXTRA_WARMUP = 20
_warmed = False


def _stats_ms(fn, reps: int, warmup: int) -> dict[str, float]:
    """CUDA-event median and IQR over `reps` timed reps after `warmup` untimed ones."""
    global _warmed
    if not _warmed:
        for _ in range(_FIRST_CELL_EXTRA_WARMUP):
            fn()
        torch.cuda.synchronize()
        _warmed = True
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        samples.append(e0.elapsed_time(e1))
    samples.sort()
    n = len(samples)
    return {
        "median_ms": samples[n // 2],
        "q1_ms": samples[n // 4],
        "q3_ms": samples[(3 * n) // 4],
        "min_ms": samples[0],
    }


def _forced_plan(base: SequenceParallelPlan, n_owners: int, P_val: int) -> SequenceParallelPlan:
    """A uniform `P = P_val` plan, so `P` can be swept INDEPENDENTLY of the policy.

    The policy is a policy; the cost model is a cost model. Grading the second
    requires points the first would not choose, so occupancy/BC/cols/ledger are copied
    from the policy-selected plan and only `P` (and the `n_slots`/roundtrip terms that
    are functions of it) is overridden. The float evidence fields the policy would
    have filled are recomputed here rather than zeroed, because `roundtrip_seconds` is
    the quantity under test and a zero there would silently grade nothing.
    """
    P = tuple([P_val] * n_owners)
    n_slots = sum(owner_slot_counts(P))
    roundtrip = sum((p - 1) * base.BC * base.cols * 4 * 2 for p in P)
    return SequenceParallelPlan(
        P=P, limiting_gate=(None,) * n_owners, binding_gate=None,
        sms=base.sms, blocks_per_sm=base.blocks_per_sm,
        waves_before=base.waves_before,
        waves_after=sum(P) / max(1, base.sms * base.blocks_per_sm),
        base_traffic_bytes=base.base_traffic_bytes,
        roundtrip_bytes=roundtrip,
        roundtrip_seconds=roundtrip / state_roundtrip_bytes_per_s(base.BC * base.cols * 4),
        base_traffic_seconds=base.base_traffic_seconds,
        ledger=base.ledger, n_slots=n_slots, BC=base.BC, cols=base.cols)


def _gpu_clocks() -> dict[str, str]:
    """Idle clock/temperature, if `nvidia-smi` exists. A thermally throttled run and a
    cold one are different machines, and a wall-clock table that does not say which it
    was cannot be compared to another one."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.max.sm,temperature.gpu,"
             "utilization.gpu,power.draw", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as exc:  # pragma: no cover - reporting only
        return {"nvidia_smi": f"unavailable: {exc}"}
    return {"nvidia_smi": out}


def _fixture(norm: str, decay_on: bool, L: int, BC: int, device: str = "cuda",
             branches: list[int] | None = None, router: str = "dense",
             logit_gain: float = 1.0):
    """The flagship cell's fixture: `B=1 H=8 L d_v=64 branches=(8,16)`, REAL producer.

    `branches`/`router`/`logit_gain` exist so the SAME instrument can be pointed at an
    AT-SCALE topology (`N >= 65536`, sparse entmax routing), which OPERATIONS.md §3
    requires before any `P` conclusion is quoted as more than cell-bound: spec §6.4's
    own measured-status note predicts that at `nc ~ 65K` the liveness-filtered grid
    saturates, the occupancy gate closes and `P` is never selected at all. Defaults
    reproduce the flagship cell exactly, so no previously recorded number moves.

    Routing comes from `produce_routing` on a constructed `RoLA`, not from a
    synthetic density dial, so the owner list and `owner_costs` the `P` apportionment
    reads are the ones the model emits. `decay` is the layer's own `_decay_config()`
    for the same reason.
    """
    B, H, d_v = 1, 8, 64
    branches = [8, 16] if branches is None else list(branches)
    side = "sparse" if router == "entmax" else "dense"
    hidden = H * d_v
    layer = RoLA(
        hidden_size=hidden, num_heads=H, head_v_dim=d_v,
        levels_per_head=len(branches), states_per_level=branches,
        decay="leaf_mass" if decay_on else "none", state_norm=norm,
        level_routing_configs=LevelRoutingConfig(read=side, write=side,
                                                tied=False, alpha=1.5),
        state_block=BC, layer_idx=0).to(device, torch.float32)
    if logit_gain != 1.0:
        with torch.no_grad():
            layer.route_W.mul_(logit_gain)
    topology = layer.topology
    x = torch.randn(B, L, hidden, device=device, dtype=torch.float32)
    v = torch.randn(B, L, H, d_v, device=device, dtype=torch.float32)
    with torch.no_grad():
        ro = produce_routing(
            x, None, v, layer.route_W, layer.route_bias_param, layer.gain_W,
            layer.gain_bias, topology, norm, budget=layer.budget, r=None,
            budget_embed_W=layer.budget_embed_W, temp_W=layer.temp_W, temp_b=layer.temp_b,
            usage_bias_W=layer.usage_bias_W, usage_bias_b=layer.usage_bias_b)
    read_levels = tuple(t.float() for t in ro.read_levels)
    write_levels = tuple(t.float() for t in ro.write_levels)
    g_read = None if norm == "global" else ro.g_read.float()
    g_write = ro.g_write.float()
    decay = layer._decay_config() if decay_on else None
    return dict(v=v, read_levels=read_levels, write_levels=write_levels, g_read=g_read,
                g_write=g_write, topology=topology, decay=decay, layer=layer, BC=BC,
                norm=norm, decay_on=decay_on)


def run_cell(name: str, norm: str, decay_on: bool, *, L: int, BC: int, reps: int,
             warmup: int, forced_ps: tuple[int, ...], branches=None,
             router: str = "dense", logit_gain: float = 1.0) -> dict:
    fx = _fixture(norm, decay_on, L, BC, branches=branches, router=router,
                  logit_gain=logit_gain)
    topology, decay = fx["topology"], fx["decay"]

    # BT comes from the PLANNER's selection, not from a flag: the cell under test is
    # the cell the plan build would produce. `compact` is pinned by this benchmark's
    # spec, and the planner's own choice is reported next to it so an off-plan pin is
    # visible rather than implied. `intra_lowering` was retired at item 6b
    # (2026-08-01, `schedule.py` §6 docstring) -- one topology-agnostic kernel family
    # now, nothing left on that axis to pin.
    selected, axes, _sched0 = plan_and_schedule(
        fx["read_levels"], fx["write_levels"], fx["g_read"], fx["g_write"], topology, BC,
        normalization=norm, decay=decay_on, mma_precision="bf16")
    plan = Plan(topology=topology, normalization=norm, BT=selected.BT, BC=BC,
                  warp_tile=(1, topology.d_v // 32), decay=decay_on,
                  mma_precision="bf16", compact=False)
    sched = build_schedule(fx["read_levels"], fx["write_levels"], fx["g_read"],
                              fx["g_write"], plan)
    n_owners = int(sched.owners.shape[0])

    blocks_per_sm = consumer_blocks_per_sm(plan)
    policy_plan = select_sequence_parallelism(plan, sched, policy=SequenceParallelPolicy())

    def consume(sp):
        return _consumer_forward(fx["v"], fx["read_levels"], fx["write_levels"],
                                   fx["g_read"], fx["g_write"], plan, decay=decay,
                                   schedule=sched, sequence_parallel=sp)

    # Throwaways for BOTH paths, before anything is timed. The split path is a
    # different launch shape (three passes, two launch boundaries) and a first call to
    # it pays its own one-time costs; paying them on the P=1 timing's neighbour would
    # show up as a `P` effect that is really a warmup effect.
    consume(None)
    _warm_sp = _forced_plan(select_sequence_parallelism(plan, sched,
                                                        policy=SequenceParallelPolicy()),
                            n_owners, 2)
    consume(_warm_sp)

    capacity = max(1, policy_plan.sms * blocks_per_sm)
    rows: list[dict] = []

    # --- P = 1: THE PRIMARY NUMBER ------------------------------------------
    base_stats = _stats_ms(lambda: consume(None), reps, warmup)
    rows.append({
        "P_label": "P=1", "role": "PRIMARY", "P": 1, "n_slots": 0,
        "ctas_pass3": n_owners, "ctas_pass1": 0, "ctas_pass2": 0,
        "waves_per_sm": n_owners / capacity,
        "predicted_roundtrip_s": 0.0, "roundtrip_bytes": 0,
        "build_ms": 0.0, **base_stats,
    })
    base_ms = base_stats["median_ms"]

    # --- P > 1: SUPPLEMENTAL ------------------------------------------------
    variants: list[tuple[str, SequenceParallelPlan]] = []
    if policy_plan.split:
        variants.append((f"P=policy({min(policy_plan.P)}..{max(policy_plan.P)})", policy_plan))
    for p in forced_ps:
        variants.append((f"P={p} forced", _forced_plan(policy_plan, n_owners, p)))

    for label, sp in variants:
        stats = _stats_ms(lambda sp=sp: consume(sp), reps, warmup)
        build = _stats_ms(lambda sp=sp: build_sequence_parallel_tensors(sp, sched),
                          reps, warmup)["median_ms"]
        n_split = sum(1 for p in sp.P if p > 1)
        rows.append({
            "P_label": label, "role": "supplemental", "P": list(sp.P) if len(set(sp.P)) > 1
            else sp.P[0], "n_slots": sp.n_slots,
            # Pass 1 summarizes every segment EXCEPT each owner's last, so its grid is
            # `sum(P_o - 1)`. That stopped equalling `n_slots` at the R1 fix, which
            # added the entry-state slot: `n_slots` is now `sum_o P_o` over split
            # owners. Reporting one as the other would over-state the pass-1 grid.
            "ctas_pass3": sp.total_segments,
            "ctas_pass1": sum(p - 1 for p in sp.P), "ctas_pass2": n_split,
            "waves_per_sm": sp.total_segments / capacity,
            "predicted_roundtrip_s": sp.roundtrip_seconds,
            "roundtrip_bytes": sp.roundtrip_bytes,
            "workspace_bytes": sp.n_slots * slot_bytes(sp.BC, sp.cols),
            "build_ms": build,
            "measured_delta_ms": stats["median_ms"] - base_ms,
            "measured_delta_less_build_ms": stats["median_ms"] - base_ms - build,
            **stats,
        })

    cell = {
        "cell": name, "norm": norm, "decay": "LeafMassDecay" if decay_on else "None",
        "L": L, "BC": BC, "BT": plan.BT, "planner_selected_BT": selected.BT,
        "planner_selected_compact": selected.compact,
        "pinned_compact": plan.compact,
        "n_owners": n_owners, "n_tiles": sched.n_tiles,
        "sms": policy_plan.sms, "blocks_per_sm": blocks_per_sm, "capacity_ctas": capacity,
        "policy_P": list(policy_plan.P), "policy_P_max": max(policy_plan.P),
        "binding_gate": policy_plan.binding_gate,
        "limiting_gates": sorted({g for g in policy_plan.limiting_gate if g}),
        "waves_before": policy_plan.waves_before,
        "base_traffic_bytes": policy_plan.base_traffic_bytes,
        "base_traffic_seconds": policy_plan.base_traffic_seconds,
        "state_roundtrip_rate_bytes_per_s":
            state_roundtrip_bytes_per_s(BC * policy_plan.cols * 4),
        "hbm_rate_bytes_per_s": MEASURED_HBM_BYTES_PER_S,
        "cols": policy_plan.cols,
        "rows": rows,
    }
    fx = None  # drop the refs before empty_cache(); `del` would unbind names a closure reads
    torch.cuda.empty_cache()
    print(json.dumps(cell), flush=True)
    return cell


def _print_tables(cells: list[dict]) -> None:
    for c in cells:
        print(f"\n=== {c['cell']}: norm={c['norm']} decay={c['decay']} BT={c['BT']} "
              f"BC={c['BC']} L={c['L']} compact={c['pinned_compact']} ===")
        print(f"    owners={c['n_owners']} tiles={c['n_tiles']} sms={c['sms']} "
              f"blocks/sm={c['blocks_per_sm']} capacity={c['capacity_ctas']} CTAs  "
              f"waves_before={c['waves_before']:.3f}")
        print(f"    policy: binding_gate={c['binding_gate']} P_max={c['policy_P_max']} "
              f"limiting={c['limiting_gates']}")
        print(f"    {'row':22s} {'role':12s} {'median':>9s} {'IQR':>16s} {'CTAs p3/p1/p2':>18s} "
              f"{'waves/SM':>9s} {'pred rt (ms)':>12s} {'meas d (ms)':>12s} {'build':>8s} "
              f"{'-build':>9s} {'ratio':>10s}")
        for r in c["rows"]:
            pred = r["predicted_roundtrip_s"] * 1e3
            meas = r.get("measured_delta_less_build_ms")
            ratio = "-" if not pred or meas is None else f"{meas / pred:.0f}x"
            print(f"    {r['P_label']:22s} {r['role']:12s} {r['median_ms']:8.3f}m "
                  f"[{r['q1_ms']:6.3f},{r['q3_ms']:6.3f}] "
                  f"{r['ctas_pass3']:6d}/{r['ctas_pass1']:5d}/{r['ctas_pass2']:4d} "
                  f"{r['waves_per_sm']:9.3f} {pred:12.5f} "
                  f"{(r.get('measured_delta_ms') or 0.0):12.3f} {r['build_ms']:8.3f} "
                  f"{(meas or 0.0):9.3f} {ratio:>10s}")
    print("\nP=1 IS THE PRIMARY NUMBER (spec §6.4). Every other row is SUPPLEMENTAL and is\n"
          "labelled so above; quoting a headline at P>1 is a reporting defect.\n"
          "`pred rt` is sp.roundtrip_seconds -- the cost model's price for the split.\n"
          "`meas d` is wall(P) - wall(P=1); `-build` additionally subtracts the measured\n"
          "cost of build_sequence_parallel_tensors, which _consumer_forward performs\n"
          "inside the timed call and which is NOT roundtrip traffic. `ratio` is\n"
          "measured-less-build / predicted. The model is NOT tuned to this table.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--BC", type=int, default=16)
    ap.add_argument("--branches", default="8,16",
                    help="routing widths. The default is the flagship cell; an at-scale "
                         "topology (e.g. 256,256 with --BC 64) is what OPERATIONS.md §3 "
                         "requires before a `P` conclusion stops being cell-bound.")
    ap.add_argument("--router", choices=("softmax", "entmax"), default="softmax")
    ap.add_argument("--logit-gain", type=float, default=1.0)
    ap.add_argument("--forced-p", default="2,4",
                    help="comma-separated uniform P values to force, so the cost model is "
                         "graded at points the policy would not necessarily choose")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("this benchmark requires CUDA")
    torch.manual_seed(0)
    forced = tuple(int(s) for s in args.forced_p.split(",") if s.strip())
    branches = [int(s) for s in args.branches.split(",")]

    clocks = _gpu_clocks()
    print(json.dumps({"gpu": clocks}), flush=True)

    cells = [
        run_cell("1 flagship (raw)", "raw", False, L=args.seq_len, BC=args.BC,
                 reps=args.reps, warmup=args.warmup, forced_ps=forced,
                 branches=branches, router=args.router, logit_gain=args.logit_gain),
        run_cell("2 global norm", "global", False, L=args.seq_len, BC=args.BC,
                 reps=args.reps, warmup=args.warmup, forced_ps=forced,
                 branches=branches, router=args.router, logit_gain=args.logit_gain),
        run_cell("3 LeafMassDecay", "raw", True, L=args.seq_len, BC=args.BC,
                 reps=args.reps, warmup=args.warmup, forced_ps=forced,
                 branches=branches, router=args.router, logit_gain=args.logit_gain),
    ]
    print(json.dumps({"gpu_after": _gpu_clocks()}), flush=True)
    _print_tables(cells)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"gpu": clocks, "cells": cells}, fh, indent=1)


if __name__ == "__main__":
    main()
