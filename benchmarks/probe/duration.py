"""The `duration` subcommand: an interleaved paired A/B, recorded.

It owns no stopwatch and no pairing logic. :func:`bench.pairing._paired_one_kernel`
is the methodology and stays the methodology; what happens here is that a round's
raw samples become ledger rows with their pairing structure intact -- `paired_with`,
`round`, and the instrument's name -- and that the three-gate verdict is computed
against the rows already in the ledger rather than against a number in a document.
"""
from __future__ import annotations

import statistics

from bench.cells import CELLS
from bench.ledger import LEDGER, append, baseline_series, load
from bench.pairing import CANONICAL_INSTRUMENT, _paired_one_kernel
from bench.regression import classify
from bench.stats import MIN_ROUNDS, paired_verdict
from probe.registry import arm_id, get

#: Launches per arm per round. The campaign recipe's count, and the point of the
#: pairing is that the two arms alternate WITHIN it.
REPS_PER_ROUND = 30

#: `light` answers a question during a batch; `landing` collects enough rounds that
#: the paired test can reach its alpha at all (`bench.stats.MIN_ROUNDS`). A light
#: run is stamped, so it can never be quoted as a landing number.
TIER_ROUNDS = {"light": 1, "landing": MIN_ROUNDS}


def run(*, run_ctx, probe_name: str, ab: str, rounds: int | None) -> int:
    probe = get(probe_name)
    try:
        a_name, b_name = ab.split(":")
    except ValueError:
        raise SystemExit("--ab takes two arm names separated by a colon, e.g. "
                         "--ab global:raw") from None
    for arm in (a_name, b_name):
        if arm not in probe.arms:
            raise SystemExit(f"{probe.name} has no arm {arm!r}; it has "
                             f"{', '.join(sorted(probe.arms))}")
    rounds = rounds if rounds is not None else TIER_ROUNDS[run_ctx.tier]

    cell = CELLS[probe.cell]
    fx = cell.build()
    a_id, b_id = arm_id(probe, a_name), arm_id(probe, b_name)

    rows, diffs = [], []
    for r in range(rounds):
        got = _paired_one_kernel(cell, fx, probe.arms[a_name], probe.arms[b_name],
                                 REPS_PER_ROUND, f"round {r + 1}/{rounds}",
                                 instrument=CANONICAL_INSTRUMENT)
        common = dict(probe=probe.name, probe_version=probe.version,
                      source=got["source"], metric="duration", unit="ms", round=r)
        rows.append(run_ctx.row(arm=a_id, values=got["a_times"],
                                paired_with=b_id, **common))
        rows.append(run_ctx.row(arm=b_id, values=got["b_times"],
                                paired_with=a_id, **common))
        diffs.append(statistics.median(got["b_times"])
                     - statistics.median(got["a_times"]))

    append(rows)
    print(f"\n{len(rows)} rows appended to {LEDGER}")

    verdict = paired_verdict(diffs)
    print(f"paired rounds n={len(diffs)}  median per-round difference "
          f"{statistics.median(diffs):+.4f} ms  -> {verdict}")

    #: The candidate is judged against THIS fingerprint's own history, which is the
    #: thing the ledger exists to make possible: no baseline file to hand-edit and no
    #: markdown table to transcribe.
    history = baseline_series(load(), rows[-1]["fingerprint"],
                              source=rows[-1]["source"])
    print(f"three-gate verdict for {b_id}: "
          f"{classify(history, rows[-1], paired_diffs=diffs)}")
    return 0
