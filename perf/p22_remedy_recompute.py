"""P22 remedy: recompute the U08 headline and the topo-D2 percentage from the
ledger rows THIS session wrote (run_ids listed below), not from the document.

    python3 perf/p22_remedy_recompute.py

Host-only, reads perf/ledger.jsonl. Min-over-blocks per the p12 harness
discipline: a "block" is one measured run (one process launch); within a block
take min over its reps (the floor least contaminated by scheduler jitter), then
report min-of-blocks and median-of-block-medians side by side.
"""
import json
import statistics
from pathlib import Path

LEDGER = Path(__file__).resolve().parent / "ledger.jsonl"

#: run_ids this remedy wrote, by (sha, cell).
RUNS = {
    ("70f607b0", "topo-D2"): ["af4fe07b8149498c"],
    ("8ce8797c", "topo-D2"): ["00c10de26fcd4df2"],
    ("70f607b0", "atscale"): ["5288fd3098404898", "ebd684295d624705"],
    ("8ce8797c", "atscale"): ["83f35e767d7347fc", "44255997c0a3480e"],
}

rows = [json.loads(l) for l in LEDGER.open()]


def blocks_for(sha_prefix, run_ids, arm_filter):
    """One block per (run_id, round) for topo-D2 (8 rounds/run_id), or one block
    per run_id for atscale (armtime has no 'round' field)."""
    out = []
    for d in rows:
        if not (d.get("sha") or "").startswith(sha_prefix):
            continue
        if d.get("run_id") not in run_ids:
            continue
        if arm_filter not in d.get("arm", ""):
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
d2_tip_min, d2_tip_med = summarize("tip 70f607b0", "70f607b0", "topo-D2", "global")
pct_min = 100 * (d2_tip_min - d2_master_min) / d2_master_min
pct_med = 100 * (d2_tip_med - d2_master_med) / d2_master_med
print(f"  DELTA (min-of-blocks):            {pct_min:+.1f} %")
print(f"  DELTA (median-of-block-medians):  {pct_med:+.1f} %")
print("  (doc claimed +44% off a baseline of 0.077 ms, which matches NEITHER "
      "binary's measured global-norm value)")

print()
print("=== atscale-L2048 | global (2 runs x 30 reps each sha, escape-hatch armtime) ===")
at_master_min, at_master_med = summarize("master 8ce8797c", "8ce8797c", "atscale", "")
at_tip_min, at_tip_med = summarize("tip 70f607b0", "70f607b0", "atscale", "")
pct_min = 100 * (at_tip_min - at_master_min) / at_master_min
pct_med = 100 * (at_tip_med - at_master_med) / at_master_med
print(f"  DELTA (min-of-blocks):            {pct_min:+.1f} %")
print(f"  DELTA (median-of-block-medians):  {pct_med:+.1f} %")
print("  (doc claimed +7.1%, 7.989 -> 8.557 ms)")
