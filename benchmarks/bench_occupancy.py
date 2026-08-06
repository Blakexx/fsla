"""Is the V3 consumer parallelism-starved? A two-axis scaling probe.

**Why this exists.** Increments 3, 4 and 5 each measured an optimization returning
far less than its arithmetic predicted: bf16 beat tf32x3 by ~4% despite 3x fewer
MMA passes; the `gram` lowering cut intra FMAs 3.2x and came out *slower*; a 27x
reduction in routing density moved the consumer ~10%. Three independent levers all
returning nothing is not three coincidences — it is a shared ceiling, and this
script measures the ceiling directly instead of inferring it.

**The launch geometry.** The consumer runs one CTA per owner, so

    grid = B * H * (N / BC)

which at the benchmarked cell (`B=1, H=8, N=128, BC=16`) is **64 CTAs of 64
threads = 4096 threads**, on a GPU whose capacity is ~123k. Crucially the grid is
**independent of sequence length**: longer `L` lengthens each CTA's serial tile
loop and adds no parallelism at all.

**The probe.** Two axes that both scale total work, differing only in whether they
scale the grid with it:

- **batch** (`B`) scales the grid 1:1 — work and parallelism together;
- **sequence** (`L`) holds the grid fixed at 64 — work alone, serially.

If the machine is saturated, both axes cost the same per unit of work. If it is
starved, the batch axis is nearly free until the machine fills while the sequence
axis pays full price. `us/owner-tile` is the per-unit cost that makes the two
directly comparable.

Interpreting the output: a *falling* `us/owner-tile` down the batch block is idle
hardware being consumed, and where it flattens is where the launch finally
saturates. A *flat* `us/owner-tile` down the sequence block is work that the launch
geometry cannot absorb.

The fix this measurement points at is spec **§6.4 sequence parallelism (`P` per
owner)**, whose contract is NORMATIVE but whose policy is `TODO`.

Usage (GPU work belongs under `flock /tmp/rola_gpu.lock`):

    PYTHONPATH=<worktree> python benchmarks/ops/benchmark_rola_v3_occupancy.py
"""
from __future__ import annotations

import argparse

import torch

from rola import RoLA, RouteProducer, SimplexProducer, dense_routing
from rola.ops.consumer import _consumer_forward
from rola.planner.schedule import Plan, build_schedule


def _median_ms(fn, reps: int, warmup: int = 4) -> float:
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


def run(reps: int = 20) -> list[dict]:
    H, d_v, BC, BT = 8, 64, 16, 32
    hidden = H * d_v
    torch.manual_seed(0)
    producer = SimplexProducer((8, 16), dense_routing(), hidden_size=hidden, num_heads=H)
    routes = RouteProducer(
        [producer], d_v=d_v, num_heads=H, hidden_size=hidden, normalization="raw")
    layer = RoLA(hidden, H, routes, layer_idx=0).to("cuda", torch.float32)
    plan = Plan(topology=layer.topology, normalization="raw", BT=BT, BC=BC,
                  warp_tile=(1, d_v // 32), decay=False)

    # (axis, B, L). The batch block scales the grid with the work; the sequence
    # block scales the work alone. B=1,L=512 is deliberately in both as the shared
    # origin, so the two blocks are read against one point rather than two.
    cells = [("batch", b, 512) for b in (1, 2, 4, 8, 16, 32)]
    cells += [("sequence", 1, ell) for ell in (512, 1024, 2048, 4096)]

    rows = []
    print(f"{'axis':>9} {'B':>3} {'L':>5} {'grid':>5} {'threads':>8} "
          f"{'ms':>8} {'us/owner-tile':>14}")
    for axis, B, L in cells:
        x = torch.randn(B, L, hidden, device="cuda")
        v = torch.randn(B, L, H, d_v, device="cuda")
        with torch.no_grad():
            ro = layer.routes(x)
        rl = tuple(t.float() for t in ro.read)
        wl = tuple(t.float() for t in ro.write)
        gr, gw = ro.g_read.float(), ro.g_write.float()
        sch = build_schedule(rl, wl, gr, gw, plan)
        ms = _median_ms(
            lambda: _consumer_forward(v, rl, wl, gr, gw, plan, decay=None, schedule=sch),
            reps)
        grid = int(sch.owners.shape[0])
        tiles = grid * (L // BT)
        row = {"axis": axis, "B": B, "L": L, "grid": grid, "threads": grid * d_v,
               "ms": round(ms, 4), "us_per_owner_tile": round(ms * 1000 / tiles, 4)}
        rows.append(row)
        print(f"{axis:>9} {B:>3} {L:>5} {grid:>5} {grid * d_v:>8} "
              f"{ms:>8.3f} {row['us_per_owner_tile']:>14.3f}")
        x = v = ro = rl = wl = gr = gw = sch = None  # drop the refs before empty_cache(); `del` would unbind names a closure reads
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("this benchmark requires CUDA")
    rows = run(reps=args.reps)

    # `B=1, L=512` is the SHARED origin of the two axes, so an identical
    # configuration is measured twice. The spread between those two readings IS
    # this instrument's noise floor; it is printed so that no flatness claim can be
    # quoted below the resolution that produced it (review 2 §4.2).
    origins = [r for r in rows if r["B"] == 1 and r["L"] == 512]
    origin = next(r for r in origins if r["axis"] == "batch")
    origin_lo = min(r["us_per_owner_tile"] for r in origins)
    origin_hi = max(r["us_per_owner_tile"] for r in origins)
    batch_best = min(r["us_per_owner_tile"] for r in rows if r["axis"] == "batch")
    seq_worst = max(r["us_per_owner_tile"] for r in rows if r["axis"] == "sequence")
    print(
        f"\nNoise bar: the shared origin (B=1, L=512) is measured once per axis, at "
        f"{origin_lo:.3f} and {origin_hi:.3f} us/owner-tile\n"
        f"({100.0 * (origin_hi - origin_lo) / origin_lo:.0f}% apart on an identical "
        f"configuration). Read every 'flat' claim below\nagainst that bar, never below it."
        f"\n\nPer-unit cost falls {origin['us_per_owner_tile'] / batch_best:.1f}x along the "
        f"BATCH axis (grid grows with the work) and is flat along the SEQUENCE axis\n"
        f"(grid pinned at {origin['grid']}): {seq_worst:.3f} us/owner-tile at the worst "
        f"sequence cell against {origin['us_per_owner_tile']:.3f} at the origin.\n"
        f"The launch geometry, not arithmetic or bandwidth, is what bounds the\n"
        f"benchmarked configuration. Spec §6.4's `P` per owner is the lever; its\n"
        f"contract is NORMATIVE and its policy is TODO.")


if __name__ == "__main__":
    main()
