"""The latency-vs-`L` curve: V3 whole layer against flash attention at matched shape.

OPERATIONS.md §3 (FLASH-ATTENTION BASELINE, LARGE-REGIME TESTING) makes this the
decisive artifact: attention is quadratic in `L` and RoLA is linear in `L` against a
FIXED state, so the comparison is a CURVE and a single cell is not a result. At small
`L` attention wins and that is expected; the number that matters is the crossover and
the memory axis (attention's KV cache grows with `L`, RoLA's committed pages do not).

**What is measured.** One forward under `torch.no_grad()` on CUDA per arm per `L`:

* `v3`   -- `RoLA.forward`, plan-selected per step, paging optional, `P = 1`
           (spec §6.4's primary number: no sequence-parallel split is requested,
           so the consumer runs one block per owner). **THE READ SIDE IS DENSE AND
           THE WRITE SIDE IS ENTMAX-SPARSE by default** (ruling 2026-08-01):
           `candidate_frac ~ 0.2` sparse-both (item 0's cell) is the easy case and
           is not what the freeze criterion is stated at; a
           dense read is the arm the freeze is defined at, because winning on the
           sparse-both case is the trivial one. `--read-router`/`--write-router`
           override the two sides independently (`'softmax'` = dense, `'entmax'` =
           sparse); the default pair (`softmax`, `entmax`) IS the parity cell.
* `v3_sparse_both` -- the SAME arm with both sides forced `entmax`, i.e. exactly item
           0's original cell. Kept as a DIAGNOSTIC row, labelled `cell_role` in the
           output, never conflated with the `v3` parity row: it is the regime where
           the routing leaves ~20% of `(owner, tile)` pairs candidate, which is a
           different (easier) case than the freeze arm.
* `fa`   -- a matched attention layer: `q/k/v` projections, causal
           `scaled_dot_product_attention`, output projection. Same
           `(B, H, L, d_model)`, `d_head = d_model / H = d_v`.

**Local budget note on the dense-read arm.** A dense read walks every `(owner, tile)`
pair instead of the sparse-both cell's ~20%, so `v3` costs roughly 5x `v3_sparse_both`'s
consumer time at the same `L` -- and, at the planner's dense-degenerate path, can cost
far more than 5x in PLAN time (`rewrite.md` §2.28 measured 39.03 s of plan time at
`candidate_frac = 1`, `N = L = 65,536`). Local runs of `v3` are therefore sized to
`L <= 16,384` under this task's 8 GiB budget, footprint printed before allocation; the
`L = 65,536` dense-read parity cell is a RENTAL cell (A100-class), same as item 0's
§1.8 ruling for the whole-layer call at that `L`. If even `L = 16,384` turns out
pathological on the dense-degenerate planner path locally, that number is reported as
measured and stands as the dense-saturation lever's motivation (§2.x of the roadmap) --
it is not worked around.

The V3 forward is additionally decomposed into PLANES (producer / planner / consumer)
on identical inputs, exactly as `benchmark_rola_v3_layer.py` does, so an at-scale row
is attributed rather than just reported.

**Dtype is NOT matched and the mismatch favours attention.** The SDPA FLASH backend
rejects fp32; the V3 consumer accepts only fp32 (`v3_consumer.cu` `TORCH_CHECK`s
`at::kFloat`). So `fa` is timed in bf16 on the FLASH backend -- its fastest real
configuration -- and `v3` in fp32. A second `fa_fp32` row on the EFFICIENT backend
(the fastest fp32-capable SDPA backend) is emitted so the dtype axis is visible
instead of assumed. Which backend produced each row is recorded in the row itself,
never inferred.

**Memory is reported on the same rows as latency**, because the curve's second axis
is the point: `kv_cache_bytes` for attention (analytic, `2 * B * H * L * d_head *
itemsize`, the bytes a decoder would have to keep), against V3's `resident_bytes`
(measured, `last_paging_stats`, §7.1 committed pages) and `dense_state_bytes`
(analytic, the provisioned `N`). `peak_alloc_gb` is torch's own peak for the whole
forward and is a THIRD, different quantity -- it includes activations and is reported
separately rather than conflated with either.

**OOM is a finding.** Every cell is wrapped: an OOM records a row with `oom` set and
the stage that failed, and the sweep continues at the next arm. Nothing is silently
shrunk.

Usage (GPU work belongs under `flock /tmp/rola_gpu.lock`):

    PYTHONPATH=<worktree> python benchmarks/ops/benchmark_rola_v3_scaling.py \
        --seq-lens 512,1024,2048,4096,8192 --json out.json
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from bench.instruments import settle_clocks
from rola import RoLA, RouteProducer, SimplexProducer, sparse_routing
from rola.expert import PlanOverrides
from rola.ops.consumer import _consumer_forward

# G16 (THUNDER_RUN_SPEC.md §8b item 3): the COST MODEL moved to
# `rola/planner/calibration/` in the Track P retirement (P4) -- `schedule.py` keeps
# the SCHEDULE, the calibration package owns the pricing. `schedule.py`'s PEP-562 lazy
# proxies still resolve these names, but importing them from their real home is what
# keeps `import rola` free of the calibration package's inductor edge.
from rola.planner.calibration.cost_model import plan_and_schedule, select_plan
from rola.planner.schedule import build_schedule


def _select_evidence(layer, x, decay_config):
    """Reconstruct `(ConfigChoice, ConfigEvidence)` on the SAME routing bundle a
    forward would use -- the private seam `rola.interface._derive_plan` walks
    internally.

    MIGRATED (API-contract break, 2026-08-05): `layer.last_plan_axes` (the
    `ConfigEvidence` a forward used to retain beside its `Plan`) is GONE with no
    layer-level replacement -- `rola.interface._derive_plan` computes it and
    discards it now. This walks the same private seam by hand so the reporting
    columns below (`k_hat`, `candidate_frac`, `selected_cost_per_token`,
    `selection_provenance`) stay populated. It ALSO sidesteps a separate,
    suspected rola/ bug (see the migration report): `layer.last_selected_plan`
    itself reads back `None` after the ordinary (unpaged, non-decode) CUDA arm,
    because `RoLA.forward`'s shared oracle/cuda `else` branch unconditionally
    nulls it after calling `rola_op`, which never surfaces the `Plan` it computed
    internally. Re-deriving here is unaffected by that bug.
    """
    from rola.interface import _hardware_facts, _static_inputs
    from rola.planner.select import DEFAULT_SELECTOR, SUPPORTED_BT, StatisticsBundle, select_with

    routes = layer.routes(x)
    static = _static_inputs(routes, decay=decay_config, overrides=layer.expert)
    hardware = _hardware_facts(routes.device)
    summaries = routes.summaries(BC=static.feasible_BC[0], BT=min(SUPPORTED_BT))
    return select_with(DEFAULT_SELECTOR, StatisticsBundle(static=static, hw=hardware, summaries=summaries))

#: Carried verbatim from `benchmark_rola_v3_layer.py`: a ~6x first-cell inflation was
#: MEASURED there (10.8 ms first vs 1.74 ms for the same cell measured second), from
#: something process-level that outlives a per-cell warmup. A 6x inflation on one cell
#: inverts a comparison, and on a CURVE it would invert exactly the smallest `L` --
#: the end the honesty rules say must not be cherry-picked -- so it is paid up front.
_FIRST_CELL_EXTRA_WARMUP = 20
_warmed = False


def _quartiles(samples: list[float]) -> tuple[float, float]:
    samples = sorted(samples)
    n = len(samples)
    return samples[n // 2], samples[(3 * n) // 4] - samples[n // 4]


def _measure(fn, reps: int, warmup: int) -> dict[str, float]:
    """Both instruments, per `wiring.md` §3.1.

    `event_ms` is the `cuda.Event` median -- right for a GPU-bound region.
    `wall_ms` is the synchronized wall-clock median -- the honest one for the
    host-bound planes (`planner`), where an event pair brackets a CPU stall rather
    than a kernel and reports it with large variance. Both are carried on every
    plane so neither is chosen silently.
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


class MatchedAttention(nn.Module):
    """The attention arm, built to be the SAME LAYER SHAPE as the RoLA layer.

    `v_proj` + `o_proj` exist on both sides; attention additionally carries `q_proj`
    and `k_proj`, which is precisely architecture §5.2's `content_addressing_width =
    d_qk` showing up as parameters. Causal, no positional embedding (RoLA has none
    either), no dropout, no mask tensor -- `is_causal=True` so the FLASH backend takes
    its fused path rather than materializing an `L x L` mask, which would make the
    memory column measure the harness instead of the architecture.
    """

    def __init__(self, hidden: int, heads: int, d_head: int):
        super().__init__()
        self.heads, self.d_head = heads, d_head
        self.q_proj = nn.Linear(hidden, heads * d_head, bias=False)
        self.k_proj = nn.Linear(hidden, heads * d_head, bias=False)
        self.v_proj = nn.Linear(hidden, heads * d_head, bias=False)
        self.o_proj = nn.Linear(heads * d_head, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        def split(t):
            return t.view(B, L, self.heads, self.d_head).transpose(1, 2)
        q, k, v = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(o.transpose(1, 2).reshape(B, L, self.heads * self.d_head))


_FA_BACKENDS = {"flash": SDPBackend.FLASH_ATTENTION,
                "efficient": SDPBackend.EFFICIENT_ATTENTION,
                "math": SDPBackend.MATH}


def run_fa_cell(*, L, B, H, d_head, hidden, dtype, backend: str, reps, warmup) -> dict:
    torch.manual_seed(0)
    row = {"arm": f"fa_{backend}", "backend": backend, "L": L, "B": B, "H": H,
           "d_head": d_head, "hidden": hidden,
           "dtype": str(dtype).replace("torch.", ""),
           "reps": reps, "warmup": warmup,
           # ANALYTIC, and it is the decoder's bill, not this forward's allocation:
           # keys plus values for every position and head. This is the quantity that
           # grows with `L` and that RoLA's fixed state replaces.
           "kv_cache_bytes": 2 * B * H * L * d_head * torch.finfo(dtype).bits // 8}
    try:
        # The SDPA FLASH backend does not accept fp32 (atscale-review.md G2, reproduced
        # under `ncu`: FLASH-only context dispatches `pytorch_flash::flash_fwd_kernel`,
        # which is bf16/fp16 only, and fp32 under a FLASH-only context RAISES
        # "No available kernel" rather than silently falling back to another backend).
        # Assert it explicitly here rather than relying on that low-level message: a
        # caller who requests `backend == "flash"` with fp32 gets a benchmark-level
        # failure that names the constraint instead of a CUDA-side error string.
        if backend == "flash":
            assert dtype in (torch.bfloat16, torch.float16), (
                f"SDPA FLASH backend does not accept {dtype}; use fa_efficient for fp32 "
                "(atscale-review.md G2)")
        layer = MatchedAttention(hidden, H, d_head).to("cuda", dtype).eval()
        for param in layer.parameters():
            param.requires_grad_(False)
        x = torch.randn(B, L, hidden, device="cuda", dtype=dtype)

        def forward():
            with torch.no_grad(), sdpa_kernel(_FA_BACKENDS[backend]):
                return layer(x)

        forward()  # faults in the backend; raises here if this backend cannot run
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        stats = _measure(forward, reps, warmup)
        row.update({
            "total_ms": round(stats["wall_ms"], 4),
            "total_iqr_ms": round(stats["wall_iqr"], 4),
            "event_ms": round(stats["event_ms"], 4),
            "event_iqr_ms": round(stats["event_iqr"], 4),
            "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 4),
            "oom": False,
            # §5.2's addressing columns for the attention side: attention's addressing
            # is DATA-located and its width is `d_qk`; it has no routing hierarchy, so
            # the grid-located columns are structurally absent (None), not zero.
            "content_addressing_width": d_head,
            "routing_addressing_width": None,
            "address_space_bits": round(math.log2(L), 3),
            "address_space_basis": "attention addresses TOKENS: the space is L and it GROWS",
        })
        layer = x = None  # drop the refs before empty_cache(); `del` would unbind names a closure reads
    except torch.cuda.OutOfMemoryError as exc:
        row.update({"oom": True, "oom_stage": "forward", "error": str(exc)[:200]})
    except RuntimeError as exc:
        row.update({"oom": False, "unsupported": True, "error": str(exc)[:200]})
    gc.collect()
    torch.cuda.empty_cache()
    return row


def _addressing(branches, d_v, norm, read_levels, write_levels) -> dict:
    """§5.2's addressing columns; same definitions as `benchmark_rola_v3_layer.py`.

    `A_r` is a PRODUCT OF PER-LEVEL MEANS, exact at a dense operating point and an
    approximation at a sparse one (the levels' supports are correlated under sparse
    routing); `realized_read_address_bits` inherits that approximation. Labelled in
    the row itself rather than in this docstring alone.
    """
    N = math.prod(branches)

    def active(levels):
        return float(torch.stack([(t != 0).float().sum(-1).mean() for t in levels]).prod())

    a_r, a_w = active(read_levels), active(write_levels)
    return {
        "N": N,
        "state_floats_per_head": N * (d_v + int(norm == "global")),
        "address_space_bits": round(math.log2(N), 3),
        "address_space_basis": "RoLA addresses LEAFS: the space is N and it is FIXED in L",
        "routing_addressing_width": sum(branches),
        "content_addressing_width": 0,
        "A_r": round(a_r, 4), "A_w": round(a_w, 4),
        "A_r_basis": "product-of-per-level-means (exact when dense, approximate when sparse)",
        "realized_read_address_bits": round(math.log2(a_r), 3),
        "realized_write_address_bits": round(math.log2(a_w), 3),
    }


def run_v3_cell(*, L, B, H, d_v, branches, hidden, read_router, write_router, logit_gain,
                norm, decay_on, paging, planes: bool, reps, warmup, state_block=None,
                arm="v3", cell_role="parity") -> dict:
    """`read_router`/`write_router` are independent (`'softmax'` = dense, `'entmax'`
    = exact entmax sparse), per the 2026-08-01 ruling: the freeze-defining parity
    cell is read-DENSE / write-SPARSE, not the sparse-both cell item 0 measured.
    `arm`/`cell_role` label the row so a dense-read parity row and a sparse-both
    diagnostic row (both produced by this same function) are never confused for
    each other downstream.
    """
    def routing_tag(read_router: str, write_router: str):
        """Old `LevelRoutingConfig(read=side(read_router), write=side(write_router),
        tied=False, alpha=1.5)`. Per the migration mapping: both sides sparse and
        untied -> `UnionRouting(alpha)` (`sparse_routing(1.5)`); otherwise
        `IndependentRouting(tied, ...)` with each side's own activation -- which
        is what a MIXED read/write router (the read-dense/write-sparse parity
        cell) needs, since `IndependentRouting` already takes per-side
        activations independently of `tied`."""
        from rola import entmax, softmax
        from rola.routing.types import IndependentRouting

        def activation(router: str):
            return entmax(1.5) if router == "entmax" else softmax()

        if read_router == "entmax" and write_router == "entmax":
            return sparse_routing(1.5)
        return IndependentRouting(False, activation(read_router), activation(write_router))

    row = {"arm": arm, "cell_role": cell_role, "L": L, "B": B, "H": H, "d_v": d_v,
           "hidden": hidden, "branches": list(branches),
           "read_router": read_router, "write_router": write_router,
           "logit_gain": logit_gain,
           "norm": norm, "decay": "LeafMassDecay" if decay_on else "None",
           "paging": paging, "P": 1, "P_role": "PRIMARY (spec §6.4)",
           "requested_state_block": state_block,
           "dtype": "float32", "reps": reps, "warmup": warmup}
    try:
        torch.manual_seed(0)
        from rola import LearnedDecay

        producer = SimplexProducer(
            tuple(branches), routing_tag(read_router, write_router),
            hidden_size=hidden, num_heads=H)
        routes = RouteProducer(
            [producer], d_v=d_v, num_heads=H, hidden_size=hidden, normalization=norm)
        decay_source = LearnedDecay(widths=routes.widths, num_heads=H) if decay_on else None
        layer = RoLA(
            hidden, H, routes, decay=decay_source, layer_idx=0,
            expert=PlanOverrides(paging=paging, state_block=state_block),
        ).to("cuda", torch.float32).eval()
        if logit_gain != 1.0:
            with torch.no_grad():
                producer.route_W.mul_(logit_gain)
        for param in layer.parameters():
            param.requires_grad_(False)
        x = torch.randn(B, L, hidden, device="cuda", dtype=torch.float32)

        def whole_layer():
            with torch.no_grad():
                return layer(x)

        whole_layer()
        assert layer.last_execution_backend == "cuda", (
            f"cell L={L} took the {layer.last_execution_backend!r} backend, so this row "
            "would be an oracle timing")
        row["BC"] = layer.state_block
        decay_config = None if layer.decay is None else layer.decay()
        # `layer.last_selected_plan`/`layer.last_plan_axes`: see `_select_evidence`'s
        # docstring -- both are re-derived here rather than read off the layer.
        from rola.expert import derive_plan

        _choice, evidence = _select_evidence(layer, x, decay_config)
        plan = derive_plan(
            layer.routes(x), decay=decay_config, expert=layer.expert, device=x.device)
        # `Plan.intra_lowering` and the whole crossover it drove (`INTRA_LOWERINGS`,
        # `select_intra_lowering`) were retired at item 6b (`schedule.py` §6 docstring,
        # 2026-08-01): there is one topology-agnostic kernel family now, so there is
        # nothing left on this axis to record.
        row.update({"selected_BT": plan.BT})
        if evidence is not None:
            # G16 RE-POINT (terminal cycle): `last_plan_axes` carries
            # `rola.planner.select.ConfigEvidence` since the closed-form cut-over --
            # the pre-cut-over `PlanAxes.stats` object (mean_K / mean_K_candidate /
            # saturated_frac) was retired with the runtime argmin. `k_hat` is the
            # SELECTOR'S OWN collapse statistic (the quanta-histogram upper bound
            # over the realized k-extent, spec §4.2) and `candidate_frac` survives by name;
            # both are recorded from the evidence the decision actually used, which
            # is stronger provenance than a re-derived statistic.
            row["k_hat"] = round(evidence.k_hat, 4)
            row["candidate_frac"] = round(evidence.candidate_frac, 6)
            row["selected_cost_per_token"] = round(evidence.cost_per_token, 4)
            row["selection_provenance"] = evidence.provenance
        # Old `layer.last_paging_stats` -> the arena's own `PagingResult`
        # (`layer._page_arena.last_result`), the same fields.
        stats_paging = layer._page_arena.last_result if layer._page_arena is not None else None
        if stats_paging is not None:
            row.update({
                "owners_total": stats_paging.owners_total,
                "realized_owners": stats_paging.realized_owners,
                "resident_pages": stats_paging.resident_pages,
                "peak_pages": stats_paging.peak_pages,
                "resident_bytes": stats_paging.resident_bytes,
                "dense_state_bytes": stats_paging.dense_bytes,
                "resident_fraction": round(stats_paging.resident_fraction, 6),
            })
        else:
            # Unpaged: the provisioned dense state IS the committed state. Analytic,
            # and labelled as such -- there is no page table to measure.
            row["dense_state_bytes"] = (B * H * math.prod(branches)
                                        * (d_v + int(norm == "global")) * 4)
            row["resident_bytes"] = row["dense_state_bytes"]
            row["resident_fraction"] = 1.0
            row["resident_basis"] = "analytic (paging off: every provisioned leaf is resident)"

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        total = _measure(whole_layer, reps, warmup)
        row.update({
            "total_ms": round(total["wall_ms"], 4),
            "total_iqr_ms": round(total["wall_iqr"], 4),
            "event_ms": round(total["event_ms"], 4),
            "event_iqr_ms": round(total["event_iqr"], 4),
            "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 4),
            "oom": False,
        })

        if planes:
            # Plane decomposition on the identical inputs the whole-layer call uses,
            # same construction as `benchmark_rola_v3_layer.py`. `glue` is a RESIDUAL
            # (total minus the planes), not an independent measurement: each plane is
            # timed as its own launch sequence, so its launch/sync overhead is inside
            # the plane and absent from the fused call, and the residual can go negative.
            v = layer.v_proj(x).view(B, L, H, d_v)

            def producer():
                with torch.no_grad():
                    return layer.routes(x)

            with torch.no_grad():
                ro = producer()
            read_levels, write_levels = ro.read, ro.write
            g_read, g_write = ro.g_read, ro.g_write
            decay = decay_config

            # `layer.routes.density` is the ROUTE PRODUCER's own build-time
            # declaration (old: `layer.static_density`), passed here for the same
            # reason `layer.topology`/`layer.state_block` are: a plane that dropped
            # it would time a planner call the whole-layer row above does not make.
            def planner():
                return plan_and_schedule(read_levels, write_levels, g_read, g_write,
                                            layer.topology, layer.state_block,
                                            normalization=norm, decay=decay_on,
                                            static_density=layer.routes.density)

            sel_plan, _axes, sched = planner()

            def select_only():
                return select_plan(read_levels, write_levels, layer.topology,
                                      layer.state_block, normalization=norm,
                                      decay=decay_on,
                                      static_density=layer.routes.density)

            def schedule_only():
                return build_schedule(read_levels, write_levels, g_read, g_write, sel_plan)

            def consumer():
                with torch.no_grad():
                    return _consumer_forward(v, read_levels, write_levels, g_read,
                                               g_write, sel_plan, decay=decay,
                                               schedule=sched)

            consumer()
            plane_stats = {}
            for name, fn in (("producer", producer), ("planner", planner),
                             ("select", select_only), ("schedule", schedule_only),
                             ("consumer", consumer)):
                plane_stats[name] = _measure(fn, reps, warmup)
            row["planes"] = {name: {k: round(v, 4) for k, v in s.items()}
                             for name, s in plane_stats.items()}
            for name in ("producer", "planner", "select", "schedule", "consumer"):
                row[f"{name}_ms"] = round(plane_stats[name]["wall_ms"], 4)
            row["glue_ms"] = round(row["total_ms"] - row["producer_ms"]
                                   - row["planner_ms"] - row["consumer_ms"], 4)
            row["glue_basis"] = "RESIDUAL of total minus (producer, planner, consumer); may be negative"
            row.update(_addressing(branches, d_v, norm, read_levels, write_levels))
            v = ro = read_levels = write_levels = sched = None  # drop the refs before empty_cache(); `del` would unbind names a closure reads
        layer = x = None  # drop the refs before empty_cache(); `del` would unbind names a closure reads
    except torch.cuda.OutOfMemoryError as exc:
        row.update({"oom": True, "oom_stage": "forward-or-planes", "error": str(exc)[:200]})
    except RuntimeError as exc:
        # A LAUNCHABILITY CEILING, not a crash and not an OOM. The consumer rejects
        # an unrealizable configuration loudly rather than degrading to the oracle
        # (`wiring.md` §1.2 item 2 -- launchability is deliberately NOT a dispatch
        # condition), so at some `L` the sweep hits a hard kernel bound. That bound
        # is exactly the arm's max-`L`, so it is RECORDED as a row with its message
        # verbatim and the sweep continues at the next arm.
        row.update({"oom": False, "unsupported": True, "error": str(exc)[:400]})
    gc.collect()
    torch.cuda.empty_cache()
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-lens", default="512,1024,2048,4096,8192")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-v-dim", type=int, default=64)
    ap.add_argument("--branches", default="8,16")
    ap.add_argument("--read-router", choices=("softmax", "entmax"), default="softmax",
                    help="the `v3` arm's read side. Default 'softmax' (dense) -- the "
                         "2026-08-01 ruling: the freeze-defining parity cell is "
                         "read-dense/write-sparse, not the sparse-both cell item 0 "
                         "measured (winning sparse-both is the trivial case).")
    ap.add_argument("--write-router", choices=("softmax", "entmax"), default="entmax",
                    help="the `v3` arm's write side. Default 'entmax' (sparse); see "
                         "--read-router.")
    ap.add_argument("--logit-gain", type=float, default=8.0,
                    help="scales route_W to sharpen an UNTRAINED exact-entmax router into a "
                         "real sparse support; the support is genuine, its LEVEL is a "
                         "synthetic stand-in for training and is labelled as such")
    ap.add_argument("--norm", choices=("raw", "global"), default="global")
    ap.add_argument("--decay", choices=("none", "leaf_mass"), default="none")
    ap.add_argument("--state-block", type=int, default=None,
                    help="`BC`, the leafs per owner block. Default None resolves to the "
                         "innermost level's width -- the STRUCTURAL choice (`wiring.md` "
                         "§1.2 item 1). At-scale topologies need it set explicitly because "
                         "the consumer instantiates only BC in {16, 64} today and an "
                         "innermost width outside that set is a launchability ceiling, "
                         "not a dispatch condition.")
    ap.add_argument("--paging", action="store_true")
    ap.add_argument("--planes", action="store_true",
                    help="decompose the V3 forward per plane (producer/planner/consumer)")
    ap.add_argument("--arms", default="v3,v3_sparse_both,fa_flash,fa_efficient",
                    help="'v3' = the parity cell (--read-router/--write-router, default "
                         "dense-read/sparse-write). 'v3_sparse_both' = the SAME cell "
                         "with both sides forced entmax -- item 0's original cell, kept "
                         "as a labelled diagnostic row, not the freeze arm.")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("this benchmark requires CUDA")
    # Item 0b's protocol, for the reason that item states: this box idles at ~220 MHz
    # of a ~2100 MHz rating, so the FIRST cell of a sweep is timed on a ramping clock
    # and the smallest `L` -- the end the honesty rules forbid cherry-picking -- is
    # exactly the one that inflates. Reported on stderr so the `--json` stdout of a
    # piped run stays what it was.
    clock_info = settle_clocks()
    print(f"clock-settle: {clock_info}", file=sys.stderr)
    if not clock_info["settled"]:
        # Gate-1 remediation finding 3: an unsettled clock must not be a silent
        # precondition failure. Every row this run produces is stamped
        # `clock_settled: False` below so a reader of the JSON (not just the stderr
        # log) can see the run's timings are not comparable to a settled baseline.
        print(f"WARNING: clock did not settle before this sweep: {clock_info}",
              file=sys.stderr)

    branches = [int(s) for s in args.branches.split(",")]
    seq_lens = [int(s) for s in args.seq_lens.split(",")]
    arms = args.arms.split(",")
    H, d_v, B = args.heads, args.head_v_dim, args.batch
    hidden = H * d_v
    rows: list[dict] = []

    for L in seq_lens:
        for arm in arms:
            if arm == "v3":
                row = run_v3_cell(L=L, B=B, H=H, d_v=d_v, branches=branches, hidden=hidden,
                                  read_router=args.read_router, write_router=args.write_router,
                                  logit_gain=args.logit_gain,
                                  norm=args.norm, decay_on=(args.decay == "leaf_mass"),
                                  paging=args.paging, planes=args.planes,
                                  reps=args.reps, warmup=args.warmup,
                                  state_block=args.state_block,
                                  arm="v3", cell_role="parity (read-dense/write-sparse, "
                                  "the freeze-defining arm)")
            elif arm == "v3_sparse_both":
                row = run_v3_cell(L=L, B=B, H=H, d_v=d_v, branches=branches, hidden=hidden,
                                  read_router="entmax", write_router="entmax",
                                  logit_gain=args.logit_gain,
                                  norm=args.norm, decay_on=(args.decay == "leaf_mass"),
                                  paging=args.paging, planes=args.planes,
                                  reps=args.reps, warmup=args.warmup,
                                  state_block=args.state_block,
                                  arm="v3_sparse_both",
                                  cell_role="DIAGNOSTIC: sparse-both (item 0's original "
                                  "cell; the easy regime, not the freeze arm)")
            elif arm.startswith("fa_"):
                backend = arm[3:]
                dtype = torch.bfloat16 if backend == "flash" else torch.float32
                row = run_fa_cell(L=L, B=B, H=H, d_head=d_v, hidden=hidden, dtype=dtype,
                                  backend=backend, reps=args.reps, warmup=args.warmup)
            else:
                raise SystemExit(f"unknown arm {arm!r}")
            row["clock_settled"] = clock_info["settled"]
            rows.append(row)
            print(json.dumps(row), flush=True)
            if args.json:  # write incrementally: an OOM later must not lose earlier rows
                with open(args.json, "w") as handle:
                    json.dump(rows, handle, indent=2)

    print("\n=== latency vs L (synchronized wall-clock median) ===")
    for r in rows:
        if r.get("oom"):
            print(f"L={r['L']:7d} {r['arm']:14s} OOM at {r.get('oom_stage')}")
        elif r.get("unsupported"):
            print(f"L={r['L']:7d} {r['arm']:14s} UNSUPPORTED: {r['error'][:80]}")
        else:
            print(f"L={r['L']:7d} {r['arm']:14s} total={r['total_ms']:9.3f} ms "
                  f"(IQR {r['total_iqr_ms']:7.3f}) event={r['event_ms']:9.3f} "
                  f"peak={r['peak_alloc_gb']:.3f} GB "
                  f"state/kv={r.get('resident_bytes', r.get('kv_cache_bytes', 0)) / 2 ** 20:8.2f} MiB")
    print("\nNOT A MATCHED-CAPACITY RATIO. Attention and RoLA are different architectures:\n"
          "attention's addressing is DATA-located over a token space that GROWS with L\n"
          "(content_addressing_width = d_head); RoLA's is GRID-located over a FIXED leaf\n"
          "space N (routing_addressing_width = sum_l b_l, content width 0). §5.2's columns\n"
          "are on every row so the cell is admissible as what it is: operational cost and\n"
          "committed state at matched (B, H, L, d_model).\n"
          "DTYPE IS NOT MATCHED AND THE MISMATCH FAVOURS ATTENTION: the SDPA FLASH backend\n"
          "rejects fp32 and the V3 consumer accepts only fp32, so fa_flash is bf16 and v3 is\n"
          "fp32. fa_efficient carries the fp32 attention row for that axis.")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(rows, handle, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
