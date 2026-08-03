"""RoLA V3 consumer benchmark, per-plane, on the Stage 0 matched cell.

Review finding **F6**: `workflows/20260728-wave2/consumer.md` §11's numbers (the
four cells, and the 9.7x `_sort_owners` attribution that fell out of them) existed
only in the findings doc — no script was committed, so nothing in the repo
reproduced them. This is that script.

Matched shape: `B=1, H=8, L=2048, d_v=64, branches=[8,16], BC=16, BT=32` — the same
cell `20260728-stage0/baseline.md` measures V2's whole layer at. Routing comes from
the REAL V3 producer (`produce_routing`) on a constructed layer, so the measured
supports are the ones the model actually emits rather than a synthetic sparsity dial.

**The comparison is NOT matched and the script says so in its own output.** V2's
baseline is a whole-layer forward (projections, short conv, producer, kernel, output
projection); V3 here is a *kernel*. Planes are therefore timed and reported
separately, with the missing term named, rather than divided into one flattering
ratio. The matched whole-layer V3 number this used to say "requires the consumer
wired into the layer" now exists: `benchmark_rola_v3_layer.py`, since the layer wire
(`RoLA.select_execution_backend`). This script keeps its kernel-level scope -- it is
where the off-plan cells the cost model is calibrated against are swept -- and the
whole-layer comparison belongs to that one.

`FILL` (spec §8: read-writes per owner-row) is measured on the same fixtures because
it is the quantity that says how many rows of an MMA flush are useful; a low fill is
the difference between a tensor-core path being worth its operand assembly and not.

Usage (GPU work belongs under `flock /tmp/rola_gpu.lock`):

    PYTHONPATH=<worktree>:<rola-bench> python benchmarks/ops/benchmark_rola_v3_consumer.py
"""
from __future__ import annotations

import argparse
import json

import torch

# G16 (THUNDER_RUN_SPEC.md §8b item 3): the COST MODEL moved to
# `benchmarks/planner_calibration/` in the Track P retirement (P4) -- `schedule.py` keeps
# the SCHEDULE, the calibration package owns the pricing. `schedule.py`'s PEP-562 lazy
# proxies still resolve these names, but importing them from their real home is what
# keeps `import rola` free of the calibration package's inductor edge.
from planner_calibration.cost_model import select_plan

from rola.layers import RoLA
from rola.layers.rola import LevelRoutingConfig
from rola.ops.consumer import _consumer_forward
from rola.planner.schedule import Plan, build_clock_stream, build_schedule
from rola.routing.producer import produce_routing

# V2 whole-layer medians (ms) from `20260728-stage0/baseline.md` §3 at this exact
# cell. Carried as data, not re-measured here: this script benchmarks V3, and
# quoting the baseline inline is what keeps the "not matched" caveat attached to
# the number instead of to a paragraph somebody has to go find.
V2_WHOLE_LAYER_MS = {
    ("rla", "raw"): 2.913, ("rla", "global"): 3.895,
    ("gla_static", "raw"): 5.926, ("gla_static", "global"): 6.812,
    ("gla_mass_detached", "raw"): 5.580, ("gla_mass_detached", "global"): 6.343,
}


#: Extra warmup iterations given to the FIRST timed cell of a process. MEASURED,
#: not defensive: `softmax BC=16 BT=16 raw realized_leaf` timed 10.8 ms as the first
#: cell of a run and 1.74 ms when the same build measured it second (`--lowerings
#: gram,realized_leaf`), on both the pre- and post-increment trees. Something
#: process-level -- module load of the first launched template instantiation, first
#: growth of the caching allocator -- outlives the per-cell warmup, and a 6x
#: inflation on one cell is enough to invert a `BT` comparison. The correct fix is
#: to pay it before anything is timed rather than to explain it in a footnote.
_FIRST_CELL_EXTRA_WARMUP = 20
_warmed = False


def _median_ms(fn, reps: int, warmup: int) -> float:
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
    return samples[len(samples) // 2]


def _fill(read_levels, write_levels, N: int, BC: int) -> dict[str, float]:
    """Spec §8's `fill` = read-writes per owner-row, plus the MMA-relevant reading.

    A read event at token `t` touches leaf `s` iff every level's read amplitude is
    nonzero at `s`'s digit; likewise writes. `A_r`/`A_w` are the per-token active
    leaf counts. `fill = A_r * T / owners_total` is reads per owner over the
    sequence; `active_rows_per_token = A_r / N * BC` is the fraction of an owner's
    `BC` resident leafs a SINGLE READ ROW touches.

    **`active_rows_per_token` is NOT the MMA row fill, and calling it
    `useful_rows_per_flush` -- which this function did until 2026-07-29 -- was
    wrong in a way that misdirected two increments' worth of reasoning.** The
    readout's `k` extent is the owner's per-TILE compacted count `K`: the union of
    the read and write supports over the tile's `BT` tokens, which is what §3 step
    2's gather returns and what the kernel contracts over. A union over 16-32
    tokens saturates far faster than one token's support, so the two numbers are
    not close and are not even correlated in the regime that matters: at
    `BC = 16, BT = 32` this function reports 0.586/16 at entmax gain 8 while the
    measured `K` is 15.91/16. `compacted_rows_per_tile` below reports `K` -- the
    number that actually decides whether an MMA flush is mostly zeros -- computed
    from the same digit lowering `v3_consumer.py`'s `level_descriptors` hands the
    kernel, so it cannot drift from the owner rectangle it describes.

    TWO STATED APPROXIMATIONS -- both measured, neither assumed:

    1. `active_rows_per_token` averages over ALL `(token, owner)` pairs, while the
       kernel only walks owner-tiles that are CANDIDATES. It equals the
       token-conditioned truth exactly when candidate coverage is 100%, which is
       what it measured at the reference cell (`BT=32`, `L=2048`: every
       `(tile, owner)` pair is a candidate) -- a property of THAT CELL, not of the
       formula. At smaller `BT`, shorter `L`, or clustered support, coverage drops
       and this number OVER-STATES emptiness. For contrast, the per-token
       conditioning -- mean fill over the owners a given token actually touches --
       measured 2.5-4x denser at the sharpened cells.
       `compacted_rows_per_tile` has no such caveat: it is computed per
       `(owner, tile)` exactly.
    2. The per-level product below is the mean active leaf count only to within a
       measured 0.02-0.19%; see `active_per_token`.
    """
    def active_per_token(levels):
        # Product over levels of the mean per-token nonzero count. This is the mean
        # active leaf count only APPROXIMATELY: `prod_l mean_t nnz_l(t)` equals
        # `mean_t prod_l nnz_l(t)` only when the per-level counts are uncorrelated
        # across tokens, and they are emitted from one hidden state, so they are
        # not. Measured gap 0.03/0.02/0.15/0.19% at logit gains 1/2/4/8 --
        # numerically irrelevant, but it is an approximation and is labelled one.
        return float(torch.stack([(t != 0).float().sum(-1).mean() for t in levels]).prod())

    a_r = active_per_token(read_levels)
    a_w = active_per_token(write_levels)
    return {
        "A_r": round(a_r, 3), "A_w": round(a_w, 3), "N": N,
        "read_density": round(a_r / N, 4), "write_density": round(a_w / N, 4),
        "active_rows_per_token": round(a_r / N * BC, 3),
    }


def _compacted_rows(read_levels, write_levels, plan) -> dict[str, float]:
    """Mean per-`(owner, tile)` compacted count `K` -- §3 step 2's gather, exactly.

    `K = |{ i in [0,BC) : exists t in the tile, R[t, leaf(i)] != 0 or W[t, leaf(i)] != 0 }|`,
    over ALL `D` levels -- the FIXED levels as a per-TOKEN factor (they cannot
    distinguish two leafs of one owner, but a zero there kills the whole token's
    row scale, so ANDing them into token validity is exact and is what the kernel
    does), the SPANNED levels per leaf -- and over the UNION of the two supports
    (`GRADIENT_PROOFS_WOLFRAM.md` P5.21 item 1). Same definition as
    `v3_consumer.cu`'s compaction site, derived from the same `level_descriptors`
    lowering so the two cannot drift.

    **Restricting this to the SPANNED levels -- which is what both this function
    and the kernel did before 2026-07-30 -- is not a milder approximation, it is a
    strictly weaker gather.** At `branches=[8,16], BC=16` the boundary is `j = 1`,
    so the whole width-8 level was invisible to it, and the flagship entmax-gain-8
    cell read `K = 15.91/16, sat 0.908` ("nothing to compact") against the exact
    `11.13/16, sat 0.017`.

    This function delegates to the PLANNER's `tile_union_stats`, so the number the
    benchmark prints and the number the plan-time compact/`BT` selection prices
    from are one computation, not two that agree today.
    """
    from planner_calibration.cost_model import tile_union_stats  # G16

    stats = tile_union_stats(read_levels, write_levels, plan.topology, plan.BC, plan.BT)
    return {
        "compacted_rows_per_tile": round(stats.mean_K, 3),
        "compacted_rows_saturated_frac": round(stats.saturated_frac, 4),
    }


def run(reps: int = 30, warmup: int = 4, L: int = 2048, router: str = "softmax",
        logit_gain: float = 1.0,
        BC: int = 16, decays: tuple[bool, ...] = (False, True),
        BT: int = 32, compact: bool = True) -> list[dict]:
    """Sweep the four (decay, norm) cells.

    **THE `intra_lowering` AXIS IS GONE** (item 6b, 2026-08-01, `schedule.py` §6
    docstring): `INTRA_LOWERINGS`/`select_intra_lowering`/`crossover_intra_lowering`/
    `dense_limit_lowering`/`Plan.intra_lowering` were all retired -- there is one
    topology-agnostic kernel family now, so this used to sweep `lowerings` within
    each (decay, norm) cell and no longer does; one `Plan` per cell.

    `router` selects the routing ACTIVATION, which is the axis that decides FILL:

    - ``softmax`` (`read=write='dense'`) is what an untrained router emits and it
      is fully dense, `A_r = A_w = N`. Every FILL reading on it is the degenerate
      `BC/BC`, which says nothing about MMA row fill. Kept as the reference cell,
      not as the informative one.
    - ``entmax`` (`read=write='sparse'`, exact alpha=1.5) is the PRODUCTION sparse
      routing path -- not an epsilon floor or a top-k stand-in
      ([[sparsity-comes-from-entmax]]). Untrained, its logits are near-uniform and
      entmax's support is then still near-full, so `logit_gain` scales `route_W`
      to sharpen them. **That gain is a synthetic stand-in for training**, and the
      output labels the cell accordingly: it produces a real entmax support at a
      real sparsity, from the real producer, but the sparsity LEVEL is dialed, not
      learned. A trained MQAR checkpoint would replace only the gain.
    """
    B, H, d_v = 1, 8, 64
    branches = [8, 16]
    hidden = H * d_v
    dev = "cuda"
    rows: list[dict] = []
    side = "sparse" if router == "entmax" else "dense"

    for decay_on in decays:
        for norm in ("raw", "global"):
            layer = RoLA(
                hidden_size=hidden, num_heads=H, head_v_dim=d_v,
                levels_per_head=len(branches), states_per_level=branches,
                decay="leaf_mass" if decay_on else "none", state_norm=norm,
                level_routing_configs=LevelRoutingConfig(read=side, write=side,
                                                        tied=False, alpha=1.5),
                layer_idx=0).to(dev, torch.float32)
            if logit_gain != 1.0:
                with torch.no_grad():
                    layer.route_W.mul_(logit_gain)
            topology = layer.topology
            x = torch.randn(B, L, hidden, device=dev, dtype=torch.float32)
            v = torch.randn(B, L, H, d_v, device=dev, dtype=torch.float32)

            def produce():
                with torch.no_grad():
                    return produce_routing(
                        x, None, v, layer.route_W, layer.route_bias_param, layer.gain_W,
                        layer.gain_bias, topology, norm, budget=layer.budget, r=None,
                        budget_embed_W=layer.budget_embed_W, temp_W=layer.temp_W,
                        temp_b=layer.temp_b, usage_bias_W=layer.usage_bias_W,
                        usage_bias_b=layer.usage_bias_b)

            ro = produce()
            read_levels = tuple(t.float() for t in ro.read_levels)
            write_levels = tuple(t.float() for t in ro.write_levels)
            g_read = None if norm == "global" else ro.g_read.float()
            g_write = ro.g_write.float()
            # The layer's own decay config, not a synthetic dial: this is the dial
            # distribution the model actually trains toward.
            decay = layer._decay_config() if decay_on else None
            # THE PLAN BUILD, run on THIS cell's realized routing. Every row below
            # carries `plan_selected`, so "plan-selected" in a findings doc is a
            # field of the output rather than a label a human attached afterwards
            # (adversarial review 2026-07-30 §3.1: the doc labelled a cell
            # plan-selected that the planner did not select). The sweep itself stays
            # exhaustive -- the selector marks the cell, it does not replace it,
            # because the A/B the constants are calibrated against IS the off-plan
            # cells.
            selected, axes = select_plan(read_levels, write_levels, topology, BC,
                                            normalization=norm, decay=decay_on)
            plan = Plan(topology=topology, normalization=norm, BT=BT, BC=BC,
                          warp_tile=(1, d_v // 32), decay=decay_on, compact=compact)
            sched = build_schedule(read_levels, write_levels, g_read, g_write, plan)

            def consume(plan=plan, sched=sched):
                return _consumer_forward(v, read_levels, write_levels, g_read,
                                           g_write, plan, decay=decay, schedule=sched)

            consume()
            row = {
                "router": router, "logit_gain": logit_gain,
                "decay": "LeafMassDecay" if decay_on else "None", "norm": norm,
                "compact": compact,
                "BT": BT,
                "plan_selected": bool(
                    (BT, compact) == (selected.BT, selected.compact)),
                "selected_BT": selected.BT,
                "selected_compact": selected.compact,
                "selected_cost_per_token": round(axes.cost_per_token, 3),
                "consumer_ms": round(_median_ms(consume, reps, warmup), 4),
                "schedule_ms": round(_median_ms(
                    lambda: build_schedule(read_levels, write_levels, g_read,
                                              g_write, plan), reps, warmup), 4),
                "clock_ms": round(_median_ms(
                    lambda: build_clock_stream(write_levels, write_levels, plan),
                    reps, warmup), 4) if decay_on else 0.0,
                "producer_ms": round(_median_ms(produce, reps, warmup), 4),
                "active_owners": int(sched.owners.shape[0]),
                "owners_total": topology.N // BC * B * H,
                "schedule_kib": round(sched.schedule_bytes() / 1024, 1),
                "stream_mib": round(sched.stream_bytes() / 2 ** 20, 2),
                **_fill(read_levels, write_levels, topology.N, BC),
                **_compacted_rows(read_levels, write_levels, plan),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            layer = None  # drop the refs before empty_cache(); `del` would unbind names a closure reads
            torch.cuda.empty_cache()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--router", choices=("softmax", "entmax"), default="softmax",
                    help="routing activation; 'entmax' is the production sparse path "
                         "(exact alpha=1.5) and is the only one whose FILL is informative")
    ap.add_argument("--logit-gain", type=float, default=1.0,
                    help="scales route_W to sharpen an UNTRAINED entmax router into a "
                         "real sparse support; a synthetic stand-in for training, "
                         "labelled as such in the output")
    ap.add_argument("--BC", type=int, default=16,
                    help="owner leaf-block size; decides HOW MANY LEVELS the owner spans, "
                         "which is what the gram/realized_leaf crossover actually turns on")
    ap.add_argument("--BT", type=int, default=32,
                    help="token tile; spec §6 fixes it to {16, 32}. It is the OTHER half of "
                         "the plan-time compact/BT decision: the tile union is taken over BT "
                         "tokens, so BT is what sets the saturation the gather is priced on")
    ap.add_argument("--compact", type=int, default=1,
                    help="spec §6's plan-time gather axis (rev 2026-07-29.5 Q1): 1 compiles "
                         "§3 step 2's support gather in, 0 compiles it out at its K == BC "
                         "degeneracy. Both are exact; the pair is the A/B the plan-time "
                         "selection is priced from")
    ap.add_argument("--decays", default="0,1", help="comma-separated 0/1 decay arms to run")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("this benchmark requires CUDA")
    torch.manual_seed(0)
    rows = run(reps=args.reps, warmup=args.warmup, L=args.seq_len, router=args.router,
               logit_gain=args.logit_gain,
               BC=args.BC, BT=args.BT, compact=bool(args.compact),
               decays=tuple(bool(int(s)) for s in args.decays.split(",")))

    print(f"\n=== V3 consumer, per plane (ms, median) — router={args.router} "
          f"logit_gain={args.logit_gain} ===")
    for r in rows:
        print(f"decay={r['decay']:14s} norm={r['norm']:7s} "
              f"consumer={r['consumer_ms']:8.3f} "
              f"schedule={r['schedule_ms']:8.3f} clock={r['clock_ms']:6.3f} "
              f"producer={r['producer_ms']:7.3f} | A_r={r['A_r']} A_w={r['A_w']} "
              f"N={r['N']} active_rows/token={r['active_rows_per_token']}/{args.BC} "
              f"K/tile={r['compacted_rows_per_tile']}/{args.BC} "
              f"(sat {r['compacted_rows_saturated_frac']}) "
              f"owners={r['active_owners']}/{r['owners_total']}")
    print("\nREAD THESE TWO NUMBERS AS DIFFERENT QUANTITIES. `active_rows/token` is the\n"
          "PER-TOKEN active fraction of an owner's BC leafs. `K/tile` is what the kernel\n"
          "actually contracts over: the per-TILE UNION of the read and write supports,\n"
          "i.e. §3 step 2's gather, with `sat` the fraction of (owner, tile) pairs where\n"
          "K == BC. A union over BT tokens saturates far faster than one token's support,\n"
          "so `active_rows/token` is NOT MMA row fill and must not be quoted as it --\n"
          "this script called it `useful_rows_per_flush` until 2026-07-29 and that name\n"
          "misdirected two increments' worth of reasoning.")
    if args.router == "softmax":
        print("\nDensity above is DEGENERATE: softmax over an untrained router is fully\n"
              "dense (A_r = A_w = N), so both numbers are trivially BC/BC. Re-run with\n"
              "--router entmax for the informative reading.")
    elif args.logit_gain != 1.0:
        print(f"\nDensity above comes from a REAL exact-entmax(alpha=1.5) support, but at a\n"
              f"sparsity DIALED by --logit-gain {args.logit_gain} rather than learned.\n"
              f"The support is genuine; its level is a synthetic stand-in for training.")
    print("\n=== V2 whole-layer baseline (20260728-stage0/baseline.md §3) ===")
    for (kernel, norm), ms in V2_WHOLE_LAYER_MS.items():
        print(f"  {kernel:20s} {norm:7s} {ms:7.3f} ms")
    print(
        "\nNOT A MATCHED RATIO. V2 above is a WHOLE-LAYER forward (projections, short\n"
        "conv, producer, kernel, output projection) in bf16; V3 above is the consumer\n"
        "KERNEL in fp32, with the producer and the planner reported as separate planes.\n"
        "Dividing them would credit V3 with the projections it does not include. The\n"
        "matched whole-layer number now exists and is NOT this script's: run\n"
        "`benchmark_rola_v3_layer.py`, which times `RoLA.forward` on the CUDA backend at\n"
        "this same cell and attributes it per plane. Its headline is that the whole layer\n"
        "is PLANNER-bound rather than kernel-bound at L=2048 -- the per-step plan build is\n"
        "55-70% of the forward while the consumer is 8-10% -- which is a fact about the\n"
        "layer that no kernel-level cell in this script can show.")


if __name__ == "__main__":
    main()
