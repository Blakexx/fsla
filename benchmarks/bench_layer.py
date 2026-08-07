"""RoLA V3 WHOLE-LAYER benchmark at the Stage 0 matched cell, per plane.

`benchmark_rola_v3_consumer.py` benchmarks the consumer KERNEL and says in its own
output that it cannot be divided against `20260728-stage0/baseline.md`'s V2 numbers,
because those are a whole-layer forward and this was not: "a matched whole-layer V3
number requires the consumer wired into the layer." The consumer is now wired
(`RoLA.select_execution_backend`), so this script is that number.

**What is measured.** One `RoLA.forward` under `torch.no_grad()` on CUDA, which is
the context in which the layer selects the plan-selected consumer -- projections,
producer, per-step plan build, schedule build, kernel, output projection, glue. The
same forward is then decomposed into PLANES by timing each component separately on
the identical inputs, so the number is attributed rather than just reported.

**What is NOT claimed.** V2 and V3 are DIFFERENT MODELS. V2 reads and writes its
state through content-addressed `Q`/`K` at `d_qk = 16` on top of the same routed
hierarchy; V3 has no `Q`, no `K` and no feature map anywhere. This script reports
OPERATIONAL COST at matched `(B, H, L, d_v, state size)` shapes and prints, in its
own output, the columns that say what is and is not comparable -- including the
addressing-entropy columns architecture doc §5.2 makes normative for any
cross-architecture cell. A single V3/V2 ratio is not a quality ratio and is not a
matched-capacity ratio; it is a cost ratio at matched shape.

Usage (GPU work belongs under `flock /tmp/rola_gpu.lock`):

    PYTHONPATH=<worktree> python benchmarks/ops/benchmark_rola_v3_layer.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch

from rola import RoLA, RouteProducer, SimplexProducer, dense_routing, sparse_routing
from rola.ops.consumer import _consumer_forward

# `BC` is derived once, in closed form, over launch-static facts
# (`rola.planner.derive.derive_bc`); `build_plan`/`build_schedule`/`plan_and_schedule`
# in `rola.planner.schedule` take it as an argument rather than choosing among
# candidates.
from rola.planner.schedule import build_plan, build_schedule, plan_and_schedule

#: Extra warmup iterations paid before ANY cell of the process is timed. MEASURED,
#: carried over from `benchmark_rola_v3_consumer.py`: a first-cell inflation of ~6x
#: was observed there (10.8 ms first, 1.74 ms for the same cell measured second),
#: from something process-level -- module load of the first launched template
#: instantiation, first growth of the caching allocator -- that outlives a per-cell
#: warmup. A 6x inflation on one cell is enough to invert a comparison, so it is
#: paid up front rather than footnoted.
_FIRST_CELL_EXTRA_WARMUP = 20
_warmed = False


def _quartiles(samples: list[float]) -> tuple[float, float]:
    samples = sorted(samples)
    n = len(samples)
    return samples[n // 2], samples[(3 * n) // 4] - samples[n // 4]


def _measure(fn, reps: int, warmup: int) -> dict[str, float]:
    """Time one plane on BOTH instruments, because they measure different things.

    ``event_ms`` -- median of ``reps`` ``cuda.Event`` brackets, the requested
    instrument and the right one for a GPU-bound region.

    ``wall_ms`` -- median of ``reps`` synchronized wall-clock brackets
    (``perf_counter`` around the call plus a ``cuda.synchronize()``), i.e.
    end-to-end latency including CPU launch and CPU compute.

    **They disagree, and the disagreement is a finding rather than noise.** Several
    planes of this layer are CPU-BOUND: ``select`` (``tile_union_stats`` is real
    tensor work over ``[B, T, H, width]``, run once per ``BT`` candidate) and
    ``schedule`` spend most of their time with the GPU idle waiting on the host.
    A ``cuda.Event`` pair brackets that stall -- the second event cannot record
    until the queued work reaches it -- so the event timer reports the stall with
    high variance instead of reporting a kernel. Measured here: repeated runs of
    the same cell moved ``select``'s event median between 7.9 and 12.4 ms while its
    consumer plane repeated to 1.141/1.141 ms. So the event number is quoted for
    the consumer and the wall number is the honest one for the host-bound planes,
    and both are in the output rather than one being chosen silently.
    """
    global _warmed
    if not _warmed:
        for _ in range(_FIRST_CELL_EXTRA_WARMUP):
            fn()
        torch.cuda.synchronize()
        _warmed = True
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    events, walls = [], []
    for _ in range(reps):
        e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start = time.perf_counter()
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        walls.append((time.perf_counter() - start) * 1e3)
        events.append(e0.elapsed_time(e1))
    event_ms, event_iqr = _quartiles(events)
    wall_ms, wall_iqr = _quartiles(walls)
    return {"event_ms": event_ms, "event_iqr": event_iqr,
            "wall_ms": wall_ms, "wall_iqr": wall_iqr}


#: Attribute names a Q/K-era or any future content-addressed V3 layer would carry.
#: Fix pass note (wiring-review.md §8 NOTE a): `content_addressing_width` used to be
#: a hardcoded literal `0`, which would keep reporting 0 even if the layer grew a
#: content-addressing path back. `_content_addressing_width` below derives the
#: column from the layer's own attribute surface instead, so it FAILS loudly the
#: moment any of these reappear rather than silently staying stale.
_CONTENT_ADDRESSING_ATTRS = (
    "head_k_dim", "q_proj", "k_proj", "qkv_proj", "q_norm", "k_norm",
    "q_conv1d", "k_conv1d",
)


def _content_addressing_width(layer) -> int:
    """Derive V3's `content_addressing_width` from the layer instead of hardcoding
    it. V3 has no `Q`, no `K` and no feature map (spec §1.2 deletion inventory), so
    today this is always 0 -- but it is computed by asserting NONE of the
    content-addressing attribute names are present, so a future content-addressed
    component (a revived `head_k_dim`/`q_proj`/... ) makes this raise instead of
    silently keeping a stale 0."""
    present = [name for name in _CONTENT_ADDRESSING_ATTRS if hasattr(layer, name)]
    if present:
        raise NotImplementedError(
            f"V3 layer grew content-addressed attribute(s) {present!r}; "
            "content_addressing_width must be derived from them, not left at the "
            "structural 0 (wiring-review.md §8 NOTE a)")
    return 0


def _addressing(layer, branches, d_v, H, norm, read_levels, write_levels) -> dict:
    """Architecture §5.2's addressing columns, plus the realized support.

    §5.2 defines addressing entropy STRUCTURALLY and makes matching on it normative:
    attention's analogue is `d_qk` (the `Theta(log L)` width that buys per-sequence
    exact pointer retrieval), RoLA's is `sum_l b_l` (`~ D * N^(1/D)` at balanced
    widths). Both are logarithmic-class addressing for an exponential space; they
    differ in WHERE the bill lands, and a cell that reports only one of them is not
    admissible as a matched comparison.

    So four columns, and the fourth is the one that is not matched:

    * `address_space_bits = log2(N)` -- the size of the address space itself.
    * `routing_addressing_width = sum_l b_l` -- the grid-located addressing budget.
    * `realized_read_address_bits = log2(A_r)` -- how much of that space one token's
      READ actually resolves, from the realized router. `A_r` is computed as
      `prod_l mean(nnz at level l)`, a PRODUCT OF PER-LEVEL MEANS, not
      `mean(prod_l nnz)` (fix pass note, wiring-review.md §8 NOTE b) -- those differ
      whenever the levels' supports are correlated, which sparse routing generally
      makes them. It is exact at the dense operating point (every level fully
      active, so the product-of-means and the mean-of-products coincide) and only
      an APPROXIMATION at a sparse operating point; `realized_read_address_bits`
      inherits that approximation there.
    * `content_addressing_width` -- `d_qk` for a Q/K architecture, and 0 for V3,
      which has no Q, no K and no feature map; derived by `_content_addressing_width`
      rather than hardcoded. THIS ONE IS NOT MATCHED between V2 and V3 and the
      mismatch is in V2's favour: V2 carries data-located content addressing on top
      of the identical grid-located routing.
    """
    N = math.prod(branches)

    def active(levels):
        # Product of per-level means -- see the `A_r` note above for exactness at
        # the dense point and approximation at a sparse one.
        return float(torch.stack([(t != 0).float().sum(-1).mean() for t in levels]).prod())

    a_r, a_w = active(read_levels), active(write_levels)
    return {
        "N": N,
        "state_floats_per_head": N * (d_v + int(norm == "global")),
        "address_space_bits": round(math.log2(N), 3),
        "routing_addressing_width": sum(branches),
        "content_addressing_width": _content_addressing_width(layer),
        "A_r": round(a_r, 3),
        "A_r_basis": "product-of-per-level-means (exact when dense, approximate when sparse)",
        "A_w": round(a_w, 3),
        "realized_read_address_bits": round(math.log2(a_r), 3),
        "realized_write_address_bits": round(math.log2(a_w), 3),
    }


def run(*, reps: int, warmup: int, L: int, B: int, H: int, d_v: int, branches: list[int],
        router: str, logit_gain: float, dtype: torch.dtype) -> list[dict]:
    hidden = H * d_v
    side = "sparse" if router == "entmax" else "dense"
    rows: list[dict] = []

    for decay_on in (False, True):
        for norm in ("raw", "global"):
            torch.manual_seed(0)
            from rola import LearnedDecay

            routing = sparse_routing(1.5) if side == "sparse" else dense_routing()
            producer = SimplexProducer(
                tuple(branches), routing, hidden_size=hidden, num_heads=H)
            routes = RouteProducer(
                [producer], d_v=d_v, num_heads=H, hidden_size=hidden, normalization=norm)
            decay_source = (LearnedDecay(widths=routes.widths, num_heads=H)
                            if decay_on else None)
            layer = RoLA(
                hidden, H, routes, decay=decay_source, layer_idx=0,
            ).to("cuda", dtype)
            # Fix pass item 2 (wiring-review.md §3): the CUDA path now additionally
            # requires eval mode. This script measures inference-path cost, so eval
            # is correct here, not incidental.
            layer.eval()
            if logit_gain != 1.0:
                with torch.no_grad():
                    producer.route_W.mul_(logit_gain)
            for param in layer.parameters():
                param.requires_grad_(False)
            x = torch.randn(B, L, hidden, device="cuda", dtype=dtype)

            def whole_layer():
                with torch.no_grad():
                    return layer(x)

            # One untimed call to (a) fault in the template instantiation this cell
            # needs and (b) publish `last_execution_backend`, which is the assertion
            # that this row measures the CUDA path and not the oracle.
            whole_layer()
            assert layer.last_execution_backend == "cuda", (
                f"cell decay={decay_on} norm={norm} did not take the CUDA backend; it took "
                f"{layer.last_execution_backend!r}, so this row would be an oracle timing")

            # Plane decomposition, on the identical inputs the whole-layer call uses.
            v = layer.v_proj(x)
            v = v.view(B, L, H, d_v)

            def producer():
                with torch.no_grad():
                    return layer.routes(x)

            with torch.no_grad():
                ro = producer()
            read_levels, write_levels = ro.read, ro.write
            g_read, g_write = ro.g_read, ro.g_write
            decay = None if layer.decay is None else layer.decay()

            def select():
                return build_plan(layer.topology, layer.state_block,
                                   normalization=norm, decay=decay_on)

            plan = select()

            def schedule_build():
                return build_schedule(read_levels, write_levels, g_read, g_write, plan)

            sched = schedule_build()

            # THE plane the layer actually runs (planner increment, 2026-07-30):
            # selection and the §4 schedule build sharing ONE pass over the routing
            # tensors. `select` and `schedule` above are kept as separate rows so
            # the split-path numbers this replaces stay comparable, but they are
            # not a decomposition of `planner` -- `planner` is strictly less than
            # their sum because the shared prefix is paid once, and the difference
            # is the point of the increment.
            def planner():
                return plan_and_schedule(read_levels, write_levels, g_read, g_write,
                                            layer.topology, layer.state_block,
                                            normalization=norm, decay=decay_on)

            planner()

            def consumer():
                with torch.no_grad():
                    return _consumer_forward(v, read_levels, write_levels, g_read, g_write,
                                               plan, decay=decay, schedule=sched)

            consumer()

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            planes = {"total": _measure(whole_layer, reps, warmup)}
            peak_gb = torch.cuda.max_memory_allocated() / 2 ** 30
            for name, fn in (("producer", producer), ("planner", planner),
                             ("select", select), ("schedule", schedule_build),
                             ("consumer", consumer)):
                planes[name] = _measure(fn, reps, warmup)
            plane_ms = {name: stats["wall_ms"] for name, stats in planes.items()}
            total_ms = plane_ms["total"]
            producer_ms, select_ms = plane_ms["producer"], plane_ms["select"]
            schedule_ms, consumer_ms = plane_ms["schedule"], plane_ms["consumer"]
            planner_ms = plane_ms["planner"]

            rows.append({
                "version": "v3", "decay": "LeafMassDecay" if decay_on else "None",
                "norm": norm, "router": router, "logit_gain": logit_gain,
                "dtype": str(dtype).replace("torch.", ""),
                "B": B, "H": H, "L": L, "d_v": d_v, "branches": list(branches),
                "BC": layer.state_block,
                "total_ms": round(total_ms, 4),
                "producer_ms": round(producer_ms, 4),
                "planner_ms": round(planner_ms, 4),
                "select_ms": round(select_ms, 4),
                "schedule_ms": round(schedule_ms, 4),
                "consumer_ms": round(consumer_ms, 4),
                "glue_ms": round(total_ms - producer_ms - planner_ms - consumer_ms, 4),
                "glue_ms_split_path": round(
                    total_ms - producer_ms - select_ms - schedule_ms - consumer_ms, 4),
                # Explicitly labelled ARITHMETIC, not a measurement: what the same
                # forward would cost if the per-step plan build were free. It is the
                # upper bound on what any amortization policy could recover, and it
                # is NOT a configuration this tree runs -- the per-step ruling stands.
                "total_ms_minus_planner_counterfactual": round(total_ms - planner_ms, 4),
                "planes": {name: {k: round(v, 4) for k, v in stats.items()}
                           for name, stats in planes.items()},
                "peak_alloc_gb": round(peak_gb, 4),
                **_addressing(layer, branches, d_v, H, norm, read_levels, write_levels),
            })
            print(json.dumps(rows[-1]), flush=True)
            layer = x = v = ro = read_levels = write_levels = sched = None  # drop the refs before empty_cache(); `del` would unbind names a closure reads
            torch.cuda.empty_cache()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-v-dim", type=int, default=64)
    ap.add_argument("--branches", default="8,16")
    ap.add_argument("--router", choices=("softmax", "entmax"), default="softmax")
    ap.add_argument("--logit-gain", type=float, default=1.0,
                    help="scales route_W to sharpen an UNTRAINED exact-entmax router into a "
                         "real sparse support; the support is genuine, its LEVEL is a "
                         "synthetic stand-in for training and is labelled as such")
    ap.add_argument("--dtype", choices=("float32",), default="float32",
                    help="the V3 consumer's tensors are fp32 (`v3_consumer.cu` TORCH_CHECKs "
                         "at::kFloat), so this axis exists to be stated, not swept")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("this benchmark requires CUDA")
    branches = [int(s) for s in args.branches.split(",")]
    rows = run(reps=args.reps, warmup=args.warmup, L=args.seq_len, B=args.batch,
               H=args.heads, d_v=args.head_v_dim, branches=branches, router=args.router,
               logit_gain=args.logit_gain, dtype=getattr(torch, args.dtype))

    print(f"\n=== V3 whole layer, per plane (ms, synchronized wall-clock median of "
          f"{args.reps}) — router={args.router} logit_gain={args.logit_gain} ===")
    for r in rows:
        print(f"decay={r['decay']:14s} norm={r['norm']:7s} total={r['total_ms']:7.3f} | "
              f"producer={r['producer_ms']:6.3f} planner={r['planner_ms']:6.3f} "
              f"(split: select={r['select_ms']:6.3f} schedule={r['schedule_ms']:6.3f}) "
              f"consumer={r['consumer_ms']:6.3f} "
              f"(event {r['planes']['consumer']['event_ms']:5.3f}) "
              f"glue={r['glue_ms']:7.3f} | peak={r['peak_alloc_gb']:.3f} GB | "
              f"BC={r['BC']} "
              f"| A_r={r['A_r']}/{r['N']}")
    print("\nTIMES ABOVE ARE SYNCHRONIZED WALL-CLOCK. The `cuda.Event` median is carried per\n"
          "plane in the JSON (`planes.<name>.event_ms`) and is quoted inline for the consumer,\n"
          "the one plane that is GPU-bound. For the host-bound planes (`select`, `schedule`)\n"
          "an event pair brackets a CPU stall rather than a kernel and is high-variance: the\n"
          "same cell moved `select`'s event median between 7.9 and 12.4 ms across runs while\n"
          "its consumer plane repeated to 1.141/1.141 ms.")
    print("\nPLANE SUMS DO NOT ADD TO `total` BY CONSTRUCTION. Each plane is timed as its\n"
          "own launch sequence, so per-plane launch and synchronization overhead is counted\n"
          "in the plane AND absent from the fused whole-layer call; `glue` is the residual\n"
          "(projections, rearranges, casts, dispatch) and may be negative when the planes\n"
          "cost more measured apart than together. It is reported as a residual, not as an\n"
          "independently measured quantity.")
    print("\nNOT A MATCHED RATIO. V2 and V3 are DIFFERENT MODELS: V2 addresses its state\n"
          "through content-addressed Q/K at d_qk=16 on top of the same routed hierarchy,\n"
          "V3 has no Q, no K and no feature map. What is matched here is (B, H, L, d_v,\n"
          "state size) and the routing topology, i.e. OPERATIONAL COST AT MATCHED SHAPE.\n"
          "Per architecture §5.2 the addressing-entropy columns are reported so the cell is\n"
          "admissible at all: address_space_bits and routing_addressing_width are IDENTICAL\n"
          "on both sides, content_addressing_width is 0 here against V2's d_qk=16, so the\n"
          "cells are NOT addressing-entropy matched and the mismatch favours V2.")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(rows, handle, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
