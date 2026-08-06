# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""Where the routing bundle's derived work should be paid: two A/Bs, one harness.

The API contract raises two placement questions that are measurements, not opinions,
and this answers both on the same fixture so the two numbers are comparable.

**A/B 1 -- statistics: fused, separate, or not at all.** The auxiliary statistics
(exact support counts, touched bits) can be produced INSIDE a producer's own solve,
where the values are already in registers, or DERIVED by the gateway from the stored
tensors afterwards. Three arms, because the third is the one the API actually changed:

``stats_unused``   production only. Nothing in the library reads these statistics --
                   they exist for the budget losses and for diagnostics -- so this is
                   what a training step that does not use them now pays.
``stats_gateway``  production, then the gateway derives both statistics from the
                   tensors. The SEPARATE arm.
``stats_fused``    production with ``expose_stats=True``: the producer returns the
                   counts as a byproduct, and the gateway measures only what is left.
                   The FUSED arm.

**A/B 2 -- packing: a view, or a copy.** The solve emits ``[BH, T, width]``, which is
already the layout the kernel's view wants; the public layout is a permute of it. Only
one of the two can be the materialized tensor. ``pack_view`` is the shipped
arrangement -- normalize in the packed domain, hand back a permuted view -- so the
downstream fold is free. ``pack_copy`` is the inverse-copy pair it replaced:
materialize the public layout, then fold it back. The delta is the prize for the
zero-copy hand-off, measured rather than argued.

Run it BARE (no outer ``flock``): the harness takes the GPU lock itself, settles the
clock, and refuses to measure a ramping box.

    python benchmarks/bench_api_stats.py --ledger results/perf/api_contract.jsonl
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

# The worktree's own package, ahead of any installed copy: a benchmark that measured a
# different tree than the one it was run from would be a number about nothing.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rola  # noqa: E402
from bench.ledger import append  # noqa: E402
from probe.discipline import disciplined  # noqa: E402
from rola.planner.schedule import routing_views  # noqa: E402

PROBE_VERSION = 1


def build(cell) -> tuple:
    """One gateway per stat arm, sharing weights so the arms differ only in placement."""
    torch.manual_seed(0)
    shared = rola.SimplexProducer(
        cell["widths"], rola.sparse_routing(1.5), hidden_size=cell["hidden"],
        num_heads=cell["heads"])
    exposing = rola.SimplexProducer(
        cell["widths"], rola.sparse_routing(1.5), hidden_size=cell["hidden"],
        num_heads=cell["heads"], expose_stats=True)
    exposing.route_W.data.copy_(shared.route_W.data)
    gateways = []
    for producer in (shared, exposing):
        gateway = rola.RouteProducer(
            [producer], d_v=cell["d_v"], num_heads=cell["heads"],
            hidden_size=cell["hidden"]).cuda()
        gateways.append(gateway)
    x = torch.randn(cell["B"], cell["L"], cell["hidden"], device="cuda")
    return gateways[0], gateways[1], x


def timed(fn, *, rounds: int, warmup: int = 5) -> list[float]:
    """Per-call duration in ms, CUDA events, raw samples in run order."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        stop.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(stop))
    return samples


def arms(cell, rounds: int) -> dict[str, list[float]]:
    plain, exposing, x = build(cell)

    def stats_unused():
        with torch.no_grad():
            plain(x)

    def stats_gateway():
        with torch.no_grad():
            factors = plain(x)
            factors.support_counts()
            factors.touched_bits()

    def stats_fused():
        with torch.no_grad():
            factors = exposing(x)
            factors.support_counts()
            factors.touched_bits()

    def pack_view():
        with torch.no_grad():
            factors = plain(x)
            routing_views(factors.read, factors.write, factors.topology)

    def pack_copy():
        with torch.no_grad():
            factors = plain(x)
            # THE PAIR THIS REPLACED: materialize the public layout, then fold it back.
            read = tuple(level.contiguous() for level in factors.read)
            write = tuple(level.contiguous() for level in factors.write)
            routing_views(read, write, factors.topology)

    return {name: timed(fn, rounds=rounds)
            for name, fn in (("stats_unused", stats_unused),
                             ("stats_gateway", stats_gateway),
                             ("stats_fused", stats_fused),
                             ("pack_view", pack_view),
                             ("pack_copy", pack_copy))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=4096)
    parser.add_argument("--rounds", type=int, default=21)
    parser.add_argument("--ledger", type=Path, default=None)
    args = parser.parse_args()
    cell = dict(B=2, L=args.L, hidden=1024, heads=8, d_v=64, widths=(64, 64))

    with disciplined(tier="landing", cmd=" ".join(sys.argv)) as run:
        results = arms(cell, args.rounds)
        arm_key = f"B{cell['B']}|L{cell['L']}|h{cell['hidden']}|H{cell['heads']}|w64x64|union1.5"
        for name, values in results.items():
            median = statistics.median(values)
            print(f"{name:>14}  median {median:8.4f} ms   n={len(values)}")
            if args.ledger is not None:
                append(args.ledger, run.row(
                    probe=f"api_contract:{name}", probe_version=PROBE_VERSION,
                    arm=arm_key, source="cuda_events", metric="duration", unit="ms",
                    values=values))
    base = statistics.median(results["stats_unused"])
    for name in ("stats_gateway", "stats_fused"):
        delta = statistics.median(results[name]) - base
        print(f"{name} costs {delta:+.4f} ms over production alone "
              f"({100 * delta / base:+.2f}%)")
    view = statistics.median(results["pack_view"])
    copy = statistics.median(results["pack_copy"])
    print(f"packing: view {view:.4f} ms vs copy {copy:.4f} ms "
          f"({100 * (copy - view) / view:+.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
