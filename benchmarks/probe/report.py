"""The `report` subcommand: the ledger, read back as a table.

Generated and never hand-edited. A hand-maintained performance table is a
transcription of numbers that already exist somewhere else, and this repository has
already recorded one transcription defect found by hand; a generated table retires
the failure mode rather than catching it again.

Derived scalars are computed HERE, on read, and are not stored: a median beside its
own samples is a second copy that can disagree with them.
"""
from __future__ import annotations

import statistics

from bench.ledger import TIMING_SOURCES, load
from bench.regression import iqr


def run(*, fingerprint_prefix: str | None, probe: str | None, fmt: str) -> int:
    rows = [r for r in load()
            if (fingerprint_prefix is None
                or r["fingerprint"].startswith(fingerprint_prefix))
            and (probe is None or r["probe"] == probe)]
    if not rows:
        print("no ledger rows match")
        return 0

    timing = [r for r in rows if r["source"] in TIMING_SOURCES]
    header = ["probe", "arm", "tier", "source", "metric", "n", "median", "IQR",
              "dirty", "sha"]
    lines = [("| " + " | ".join(header) + " |" if fmt == "md" else ",".join(header)),
             ("|---|---|---|---|---|---:|---:|---:|---|---|" if fmt == "md" else None)]
    for r in timing:
        #: A legacy row has NO tree sha, and the table says so rather than inventing
        #: one: which commit a banked markdown number describes is exactly what was
        #: lost, and printing a plausible blank would hide the loss.
        cells = [r["probe"], r["arm"], r["tier"], r["source"], r["metric"],
                 str(len(r["values"])), f"{statistics.median(r['values']):.4f}",
                 f"{iqr(r['values']):.4f}", str(r["dirty"]),
                 r["sha"][:8] if r["sha"] else "unrecorded"]
        if fmt == "md":
            #: The arm id contains `|`, which is the markdown table's own column
            #: separator. Escaped here rather than renamed: the arm's name is shared
            #: with the latency baseline and must stay one string.
            lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
        else:
            lines.append(",".join(f'"{c}"' for c in cells))
    print("\n".join(x for x in lines if x is not None))

    counters = [r for r in rows if r["source"] == "ncu"]
    if counters:
        print(f"\n{len(counters)} counter row(s); these are NOT comparable to the "
              f"durations above -- ncu replays under flushed caches and base clocks.")
    return 0
