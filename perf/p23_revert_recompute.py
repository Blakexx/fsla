"""P23: recompute the U08 price at the revert tip, against the P22-remedy
`8ce8797c` rows (not re-measured here -- those binaries and rows already
exist and are unchanged by this revert).

    python3 perf/p23_revert_recompute.py

Same discipline as `perf/p22_remedy_recompute.py`: min-over-blocks per the p12
harness convention (min over a block's reps, then min-of-blocks and
median-of-block-medians across blocks). Host-only, reads perf/ledger.jsonl.
"""
import json
import statistics
from pathlib import Path

LEDGER = Path(__file__).resolve().parent / "ledger.jsonl"

#: run_ids by (sha, cell). `8ce8797c` rows are the P22-remedy's, reused
#: unchanged; `52782c90` rows are this batch's (topo-D2: `rola-probe duration`,
#: 8 rounds x 30 reps; atscale: `atscale_probe.py`, 2 runs x 30 reps).
RUNS = {
    ("52782c90", "topo-D2"): ["4571d9f0af1041b4"],
    ("8ce8797c", "topo-D2"): ["00c10de26fcd4df2"],
    ("52782c90", "atscale"): ["22ee64ac97c04959", "1f7d664595eb45c0"],
    ("8ce8797c", "atscale"): ["83f35e767d7347fc", "44255997c0a3480e"],
}

rows = [json.loads(l) for l in LEDGER.open()]


def blocks_for(sha_prefix, run_ids, arm_filter):
    out = []
    for d in rows:
        if not (d.get("sha") or "").startswith(sha_prefix):
            continue
        if d.get("run_id") not in run_ids:
            continue
        if arm_filter not in (d.get("arm") or ""):
            continue
        out.append(d["values"])
    return out


def summarize(label, sha_prefix, cell_key, arm_filter):
    run_ids = RUNS[(sha_prefix, cell_key)]
    blocks = blocks_for(sha_prefix, run_ids, arm_filter)
    mins = [min(b) for b in blocks]
    meds = [statistics.median(b) for b in blocks]
    print(f"{label:40s} n_blocks={len(blocks):2d}  min-of-blocks={min(mins):.4f} ms  "
          f"median-of-block-medians={statistics.median(meds):.4f} ms")
    return min(mins), statistics.median(meds)


print("=== topo-D2-w64x64-BC64 | global (8 rounds x 30 reps each sha) ===")
d2_master_min, d2_master_med = summarize("master 8ce8797c", "8ce8797c", "topo-D2", "global")
d2_tip_min, d2_tip_med = summarize("P23 tip 52782c90", "52782c90", "topo-D2", "global")
pct_min = 100 * (d2_tip_min - d2_master_min) / d2_master_min
pct_med = 100 * (d2_tip_med - d2_master_med) / d2_master_med
print(f"  DELTA vs 8ce8797c (min-of-blocks):            {pct_min:+.1f} %")
print(f"  DELTA vs 8ce8797c (median-of-block-medians):  {pct_med:+.1f} %")
print("  (P10 tip 70f607b0 measured +35.4% here; the P23 revert's job is to take that back)")

print()
print("=== atscale-L2048 | global (2 runs x 30 reps each sha, escape-hatch armtime) ===")
at_master_min, at_master_med = summarize("master 8ce8797c", "8ce8797c", "atscale", "")
at_tip_min, at_tip_med = summarize("P23 tip 52782c90", "52782c90", "atscale", "")
pct_min = 100 * (at_tip_min - at_master_min) / at_master_min
pct_med = 100 * (at_tip_med - at_master_med) / at_master_med
print(f"  DELTA vs 8ce8797c (min-of-blocks):            {pct_min:+.1f} %")
print(f"  DELTA vs 8ce8797c (median-of-block-medians):  {pct_med:+.1f} %")
print("  (P10 tip 70f607b0 measured +23.0% here; the P23 revert's job is to take that back)")
