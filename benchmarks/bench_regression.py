"""Record the committed latency baseline. **A MEASUREMENT, not an edit.**

    flock /tmp/rola_gpu.lock python benchmarks/bench_regression.py --write

The pytest side (``benchmarks/test_latency_regression.py``) only COMPARES against
what this wrote; it never writes, because a test that can rewrite its own baseline
cannot fail.

The arm set is deliberately SMALL and named here rather than derived from the whole
cell registry: this is a per-commit CI gate, and a gate that takes the full grid is
a gate that gets disabled. The full grid is the nightly `bench` job (§6.6).
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
#: AND the repo root, AHEAD of it: `rola` is installed EDITABLE into the shared
#: venv, so without this a worktree run imports the MAIN tree's extension while
#: stamping this tree's sha (P11's discarded A/B). `probe.discipline` REFUSES if
#: the two still disagree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.cells import CELLS  # noqa: E402
from bench.instruments import settle_clocks  # noqa: E402
from bench.pairing import CANONICAL_INSTRUMENT, WARMUP_LAUNCHES, _timed  # noqa: E402
from bench.regression import (  # noqa: E402
    BASELINE,
    check_instrument,
    load_baseline,
    summarize,
    threshold_ms,
    write_baseline,
)

#: `(cell, norm, decay_temp)`. One small topology cell per structural axis the gate
#: is meant to protect: both normalizations, the decay arm, and a deeper topology at
#: the other `BC`. Each is a plan the planner could actually choose.
ARMS = (
    ("topo-D2-w64x64-BC64", "global", None),
    ("topo-D2-w64x64-BC64", "raw", None),
    ("topo-D2-w64x64-BC64", "global", "warm"),
    ("topo-D4-w8x8x8x16-BC16", "global", None),
)


def arm_id(name, norm, decay_temp) -> str:
    return f"{name}|{norm}|{decay_temp or 'none'}"


def _prepare(name, norm, decay_temp):
    from rola.routing.types import LeafMassDecay

    cell = CELLS[name]
    fx = cell.build()
    decay = None if decay_temp is None else LeafMassDecay(dials=fx["dials"][decay_temp])
    g_read = None if norm == "global" else fx["g_read"]
    plan, sched = cell.plan(norm=norm, decay_on=decay is not None, fx=fx)
    for _ in range(WARMUP_LAUNCHES):
        _, sched = _timed(cell, fx, plan, decay, g_read, sched)
    return cell, fx, plan, decay, g_read, sched


def measure(name, norm, decay_temp, reps=11):
    """ONE arm's per-launch latencies, warmed, through the harness's own timer.

    `_timed` rather than a fresh timing loop: a baseline measured by a different
    stopwatch than the one the drivers use is a baseline for a different quantity.
    It runs on :data:`bench.pairing.CANONICAL_INSTRUMENT` and nothing here may
    override that -- the recorder and the gate must not be able to disagree.
    """
    state = _prepare(name, norm, decay_temp)
    return [_timed(*state)[0] for _ in range(reps)]


def measure_all(arms=ARMS, *, reps=11, blocks=5):
    """Every arm, INTERLEAVED BLOCK BY BLOCK, and the spread taken ACROSS blocks.

    **THIS IS THE FIX FOR A REAL AND MEASURED DEFECT, recorded so it is not undone.**
    The first version timed each arm once, in one pass, and derived the threshold
    from the WITHIN-PASS IQR of individual launches. That IQR is the spread of a
    launch; the quantity the gate actually compares is an arm's MEDIAN, measured in a
    different process on a different day. On this box the two are not the same size:
    the `raw|none|dense` arm's within-pass IQR read 0.010 ms while its median moved
    0.201 -> 0.240 -> 0.247 ms across processes, so an unchanged kernel failed its own
    gate on every run. Widening the sigma would have hidden that; measuring the right
    spread fixes it.

    So: the arms are cycled ROUND-ROBIN across `blocks` blocks -- :mod:`bench.pairing`
    rule 1 applied to the baseline, so a drift step lands inside every arm rather than
    inside one -- each block yields one median per arm, and the reported statistic is
    the MEDIAN OF BLOCK MEDIANS with the IQR OF BLOCK MEDIANS. That IQR is the spread
    of the number being compared, which is what the threshold has to be derived from.
    """
    # Item 0b: settle the clock BEFORE preparing/timing any arm, in every process
    # that measures -- this is the SAME routine `test_latency_regression.py`'s
    # session fixture calls, so the recorder and the gate share one precondition
    # instead of the recorder ramping incidentally (35 * 5 reps) while the gate
    # (11 reps, one process) does not. `--emit-json` subprocesses reach this too
    # (`pooled_across_processes` -> `main` -> here), so every pooled process starts
    # from the same clock state the pool is meant to characterize. Printed to
    # STDERR, never stdout: `--emit-json` mode's stdout is parsed whole as JSON by
    # the parent (`pooled_across_processes`), and a diagnostic line ahead of the
    # payload would break that parse.
    clock_info = settle_clocks()
    print(f"clock-settle: {clock_info}", file=sys.stderr)
    if not clock_info["settled"]:
        # Gate-1 remediation finding 3: don't proceed as though nothing happened --
        # say so loudly, and hand the flag back to the caller so it can be recorded
        # (baseline provenance / comparison row) instead of only living in this
        # stderr line.
        print(f"WARNING: clock did not settle before this measurement: {clock_info}",
              file=sys.stderr)
    prepared = {arm_id(*spec): _prepare(*spec) for spec in arms}
    per_block = {k: [] for k in prepared}
    for _ in range(blocks):
        for key, state in prepared.items():
            per_block[key].append(statistics.median(
                [_timed(*state)[0] for _ in range(reps)]))
    return per_block, clock_info["settled"]


def pooled_across_processes(processes: int, reps: int, blocks: int) -> tuple[dict, bool]:
    """The baseline's spread, measured ACROSS PROCESSES -- because that is the
    comparison the gate makes.

    **SECOND RECORDED FIX, same defect one level up.** Interleaved blocks WITHIN one
    process still understated it: three consecutive `pytest -m bench` runs against a
    block-derived baseline read 7/7, 7/7, then 5/7. A median measured in a fresh
    process is not a draw from the within-process distribution -- clock state, memory
    layout and allocator history all reset -- so the only spread that bounds the
    gate's own comparison is the between-process one, and it has to be MEASURED
    rather than approximated by a bigger sigma.

    So `--write` re-invokes this script `processes` times, pools every block median
    from every process, and takes the median and IQR of that pool. The pool therefore
    contains both sources of drift, and the derived threshold is a statement about
    the quantity actually being compared.
    """
    pool: dict[str, list[float]] = {}
    all_settled = True
    for i in range(processes):
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--emit-json",
             "--reps", str(reps), "--blocks", str(blocks)],
            capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            raise SystemExit(f"baseline process {i} failed; refusing to record a "
                             f"partial measurement")
        payload = json.loads(proc.stdout)
        if not payload["clock_settled"]:
            all_settled = False
            print(f"WARNING: process {i} did not settle its clock before measuring",
                  file=sys.stderr)
        for key, medians in payload["blocks"].items():
            pool.setdefault(key, []).extend(medians)
        print(f"  process {i + 1}/{processes} done")
    return pool, all_settled


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="RE-RECORD the baseline")
    ap.add_argument("--reps", type=int, default=11, help="launches per block")
    ap.add_argument("--blocks", type=int, default=5, help="interleaved blocks")
    ap.add_argument("--processes", type=int, default=7,
                    help="--write only: independent processes to pool over")
    ap.add_argument("--emit-json", action="store_true",
                    help="internal: print this process's block medians and exit")
    a = ap.parse_args()

    if a.emit_json:
        per_block, settled = measure_all(reps=a.reps, blocks=a.blocks)
        print(json.dumps({"clock_settled": settled, "blocks": per_block}))
        return 0

    pool, settled = (pooled_across_processes(a.processes, a.reps, a.blocks) if a.write
                     else measure_all(reps=a.reps, blocks=a.blocks))
    if not settled:
        # Gate-1 remediation finding 3: don't record/compare silently against an
        # unsettled clock -- say so on stderr, and mark every row rather than
        # letting the JSON look identical to a settled run.
        print("WARNING: at least one measurement in this run did not settle its "
              "clock -- the numbers below are not comparable to a settled baseline",
              file=sys.stderr)
    arms = {}
    for key, medians in pool.items():
        s = summarize(medians, source=CANONICAL_INSTRUMENT)
        s["clock_settled"] = settled
        arms[key] = s
        print(f"{key:48s} median {s['median_ms']:.4f} ms  block IQR {s['iqr_ms']:.4f}  "
              f"-> alarm at {threshold_ms(s['median_ms'], s['iqr_ms']):.4f} ms")

    if a.write:
        write_baseline(arms)
        print(f"wrote {BASELINE}")
        return 0

    want = load_baseline()["arms"]
    bad = 0
    for key, got in arms.items():
        exp = want.get(key)
        if exp is None:
            print(f"  NEW ARM (no baseline) {key}")
            continue
        check_instrument(exp, got["source"], key)
        limit = threshold_ms(exp["median_ms"], exp["iqr_ms"])
        flag = "REGRESSED" if got["median_ms"] > limit else "ok"
        bad += got["median_ms"] > limit
        print(f"  {flag:9s} {key}  {got['median_ms']:.4f} vs limit {limit:.4f}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
