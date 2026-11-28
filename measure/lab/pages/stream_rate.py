# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The measured section of `docs/lab/stream_rate.md`, generated from the newest sample of `stream_rate`'s record.

`write(root, data, instance)` reads, in the pinned data `data`, the record's rows (each walk's GB/s the device and bytes a
cycle an SM) and rewrites `docs/lab/stream_rate.md` under `root` from its `MEASURED` line to its end: a table of every
walk the record holds, each against the band the page predicted for it before its run (`BANDS`, the run's), and whether
it held. It REFUSES, naming what is missing, where the record is absent, holds a failed check, or holds a walk `BANDS`
does not price (a walk run without a prediction is no row of this page). `instance` is the rented box's label (`nb...`
Nebius, `hs...` Hyperstack, digits vast.ai), which only words the card's line. It returns the files it wrote (`lab page
stream_rate --instance LABEL`, docs/internals/tools/lab.md).
"""

from __future__ import annotations

import pathlib
import re

from _record import Record, card, read

DEVICE, CTA = ": the device", ": a CTA"
#: THE PREDICTIONS of the newest run, each walk's: the figure it bounds (`DEVICE` in GB/s or `CTA` in bytes a cycle an
#: SM) and its band, as the page states them above its MEASURED line
BANDS = {
    "HBM read, a CTA its own slice of a 1 GiB table": (DEVICE, 2850.0, 3350.0),
    "HBM write, a CTA its own slice of a 1 GiB table": (DEVICE, 2700.0, 3350.0),
    "HBM copy, a CTA its own slice of 512 MB into 512 MB": (DEVICE, 2600.0, 3300.0),
    "L2 read, a CTA its own 128 KB slice 64 times": (CTA, 20.0, 40.0),
    "L2 read, every CTA one 4 MB table 4 times": (CTA, 15.0, 40.0),
}
#: what a walk's figure is, where its name alone does not say it
NOTES = {
    "L2 read, a CTA its own 128 KB slice 64 times": (
        "plain loads are cached in the SM's L1 on sm_90 and a 128 KB slice fits an SM whose kernel carves no shared "
        "memory, so its passes after the first hit L1: the figure is L1's, mostly, and no L2 rate"
    ),
}


def write(root: pathlib.Path, data: pathlib.Path, instance: str) -> list[pathlib.Path]:
    sample = read(data, "stream_rate")
    d = sample.body()
    failed = [c["variant"] for c in d["checks"] if c["lab"] == "check" and not c["ok"]]
    if failed:
        raise SystemExit(f"lab page: the record holds failed checks, {failed}: its figures are not the page's to state")
    R = Record({r["name"]: r["median"] for r in d["results"]})
    walks = [r["name"][: -len(DEVICE)] for r in d["results"] if r["name"].endswith(DEVICE)]
    unpriced = [w for w in walks if w not in BANDS]
    if unpriced:
        raise SystemExit(f"lab page: the record holds walks the page predicted nothing for: {unpriced}")
    spread = max((r["max"] - r["min"]) / r["median"] for r in d["results"])
    rows, held = [], {}
    for w in walks:
        which, lo, hi = BANDS[w]
        figure = R[w + which]
        held[w] = lo <= figure <= hi
        unit = "GB/s" if which == DEVICE else "B/clk/SM"
        rows.append(
            f"| {w} | {R[w + DEVICE]:,.1f} | {R[w + CTA]:.2f} | {lo:,.0f}-{hi:,.0f} {unit} | {'yes' if held[w] else 'MISSED'} |"
        )
    table = "\n".join(
        ["| the walk | GB/s the device | bytes a cycle an SM | the prediction | held |", "|---|---:|---:|---|---|"] + rows
    )
    shows = []
    for w in walks:
        which, lo, hi = BANDS[w]
        figure = R[w + which]
        verdict = "inside" if held[w] else ("over" if figure > hi else "under")
        line = f"- {w.upper()}: {R[w + DEVICE]:,.1f} GB/s the device, {R[w + CTA]:.2f} bytes a cycle an SM, {verdict} its band"
        shows.append(line + (f"; {NOTES[w]}." if w in NOTES else "."))
    checked = sum(1 for c in d["checks"] if c["lab"] == "check" and c["ok"])
    text = f"""MEASURED ({card(d['device'], instance)}, driver {d['driver']}; {d['date'][:10]}, the experiment at
{d['git']['sha'][:8]}; {d['reps']} repetitions, the {checked} checked walks exact; the median, the widest spread of a row
{100 * spread:.1f}%; the clock as the card left it):

{table}

WHAT IT SHOWS.
{chr(10).join(shows)}

Record: `{sample.cited}`.
"""
    p = root / "docs/lab/stream_rate.md"
    s = p.read_text()
    m = re.search(r"^MEASURED\b", s, re.M)
    i = m.start() if m else len(s.rstrip("\n")) + 2
    p.write_text(s[:i].rstrip("\n") + "\n\n" + text)
    return [p]
