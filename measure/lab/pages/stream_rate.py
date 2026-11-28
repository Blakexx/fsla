# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The predictions and the measured section of `docs/lab/stream_rate.md`, generated: `RUNS` states each run's
predictions once, and the newest sample of `stream_rate`'s record is read against its own run's.

`write(root, data, instance)` rewrites `docs/lab/stream_rate.md` under `root` from its first `PREDICTION` line to its
end: each run of `RUNS` in order, its basis and every walk's band, then the newest sample's walks in a table against the
bands of the run that recorded it, and whether each held. It REFUSES, naming what is missing, where the record is absent,
holds a failed check, was recorded by a run `RUNS` does not hold (its predictions not stated before it), or holds a walk
its run did not price. `PENDING` is the run not yet recorded: once its sample is pinned, the entry takes the sample's run
id. `instance` is the rented box's label (`nb...` Nebius, `hs...` Hyperstack, digits vast.ai), which only words the
card's line. It returns the files it wrote (`lab page stream_rate --instance LABEL`, docs/internals/tools/lab.md).
"""

from __future__ import annotations

import pathlib
import re

from _record import Record, card, read

DEVICE, CTA = ": the device", ": a CTA"
PENDING = "pending"
SECOND = "20261128T013030Z-737802-5ab0223a"
HBM_READ = "HBM read, a CTA its own slice of a 1 GiB table"
HBM_WRITE = "HBM write, a CTA its own slice of a 1 GiB table"
HBM_COPY = "HBM copy, a CTA its own slice of 512 MB into 512 MB"
OWN = "L2 read, a CTA its own 128 KB slice 64 times"
SHARED = "L2 read, every CTA one 4 MB table 4 times"
OWN_CG = "L2 read past L1 (ld.global.cg), a CTA its own 128 KB slice 64 times"
SHARED_CG = "L2 read past L1 (ld.global.cg), every CTA one 4 MB table 4 times"
HALF_CG = "L2 read past L1 (ld.global.cg), a CTA its own 128 KB slice 64 times, half the SMs"
SPREAD_CG = "L2 read past L1 (ld.global.cg), a CTA its own 128 KB interleaved over the table 64 times"
SMALL_CG = "L2 read past L1 (ld.global.cg), every CTA one 128 KB slice 64 times"

#: THE PREDICTIONS, each run's, in the order they ran: its run id (the sample's, or `PENDING`), the date they were
#: written (before the run), their basis, and each walk's band -- the figure it bounds (`DEVICE` in GB/s, `CTA` in bytes
#: a cycle an SM) and its range
RUNS = (
    (
        "20261127T232424Z-209524-19f37764",
        "2026-11-27",
        "HBM against the documented 3.35 TB/s; L2 from A100's documented L2 (5,120 bytes a clock over 108 "
        "SMs, 47 an SM) at 50-85% of it",
        {
            HBM_READ: (DEVICE, 2850.0, 3350.0),
            HBM_WRITE: (DEVICE, 2700.0, 3350.0),
            HBM_COPY: (DEVICE, 2600.0, 3300.0),
            OWN: (CTA, 20.0, 40.0),
            SHARED: (CTA, 15.0, 40.0),
        },
    ),
    (
        SECOND,
        "2026-11-27",
        "the first run's two L2 bands missed on wrong prices: the ceiling came from A100's documented L2, where the H100 "
        "gave every SM reading one shared table 43.0 bytes a cycle; and a CTA's own 128 KB slice was taken to miss L1, "
        "which plain loads on sm_90 do not (it fits an SM's L1). From the H100's 43.0: the plain walks repeat within 5%; "
        "past L1 the shared table stays near 43 (its 4 MB a pass overran L1 already, so `.cg` removes nothing it used), "
        "and a CTA's own slice reads up to a fifth under it, never over (no L1 now, its lines over both L2 partitions); "
        "HBM's walks repeat within 3%",
        {
            HBM_READ: (DEVICE, 3068.0, 3258.0),
            HBM_WRITE: (DEVICE, 3090.0, 3281.0),
            HBM_COPY: (DEVICE, 2777.0, 2949.0),
            OWN: (CTA, 62.0, 69.0),
            SHARED: (CTA, 41.0, 45.0),
            OWN_CG: (CTA, 34.0, 48.0),
            SHARED_CG: (CTA, 40.0, 46.0),
        },
    ),
    (
        PENDING,
        "2026-11-28",
        "the second run's own slice past L1 missed under its band (30.95 bytes a cycle an SM, 0.73 of the shared table's "
        "42.38): its price took the far L2 partition to cost at most a fifth. Three walks separate the explanations. IF a "
        "CTA's distinct lines are served from the one partition that holds them (half the SMs' far) while a table every "
        "SM reads is served near from both, as predicted here: the own slice on half the SMs reads as on all (a path an SM, "
        "no aggregate cap), interleaving its units over the table moves nothing (placement by address is already spread), "
        "and every CTA reading one shared 128 KB slice reads as fast as the shared 4 MB table (sharing is what lifts it). "
        "An aggregate cap on distinct lines would lift the half-the-SMs walk instead; a placement effect, the interleaved one; "
        "and the shared small slice under 34 would say sharing is not the cause. Every walk of the second run repeats "
        "within 5% (HBM's within 3%)",
        {
            HBM_READ: (DEVICE, 3077.0, 3267.0),
            HBM_WRITE: (DEVICE, 3094.0, 3286.0),
            HBM_COPY: (DEVICE, 2770.0, 2942.0),
            OWN: (CTA, 62.0, 68.5),
            SHARED: (CTA, 41.4, 45.7),
            OWN_CG: (CTA, 29.4, 32.5),
            SHARED_CG: (CTA, 40.3, 44.5),
            HALF_CG: (CTA, 28.0, 34.0),
            SPREAD_CG: (CTA, 28.0, 34.0),
            SMALL_CG: (CTA, 39.0, 46.0),
        },
    ),
)


def _band(which: str, lo: float, hi: float) -> str:
    if which == DEVICE:
        return f"{lo:,.0f}-{hi:,.0f} GB/s the device"
    return f"{lo:,.1f}-{hi:,.1f} bytes a cycle an SM"


def write(root: pathlib.Path, data: pathlib.Path, instance: str) -> list[pathlib.Path]:
    sample = read(data, "stream_rate")
    d = sample.body()
    failed = [c["variant"] for c in d["checks"] if c["lab"] == "check" and not c["ok"]]
    if failed:
        raise SystemExit(f"lab page: the record holds failed checks, {failed}: its figures are not the page's to state")
    run = re.search(r"/([^/]+)/output\.json$", sample.cited).group(1)
    bands = {r: b for r, _, _, b in RUNS}
    if run not in bands:
        raise SystemExit(f"lab page: the newest sample's run {run} states no predictions in RUNS: none was written before it")
    predicted = bands[run]
    R = Record({r["name"]: r["median"] for r in d["results"]})
    walks = [r["name"][: -len(DEVICE)] for r in d["results"] if r["name"].endswith(DEVICE)]
    unpriced = [w for w in walks if w not in predicted]
    if unpriced:
        raise SystemExit(f"lab page: the record holds walks its run predicted nothing for: {unpriced}")
    paragraphs = []
    for rid, when, basis, walk_bands in RUNS:
        who = "the run to come" if rid == PENDING else f"run {rid}"
        lines = "\n".join(f"- {w}: {_band(*b)}." for w, b in walk_bands.items())
        paragraphs.append(f"PREDICTION, {who} (written {when}, before it ran; its basis: {basis}):\n\n{lines}\n")
    spread = max((r["max"] - r["min"]) / r["median"] for r in d["results"])
    rows, shows = [], []
    for w in walks:
        which, lo, hi = predicted[w]
        figure = R[w + which]
        verdict = "held" if lo <= figure <= hi else "MISSED"
        rows.append(f"| {w} | {R[w + DEVICE]:,.1f} | {R[w + CTA]:.2f} | {_band(which, lo, hi)} | {verdict} |")
        where = "inside" if verdict == "held" else ("over" if figure > hi else "under")
        shows.append(f"- {w.upper()}: {R[w + DEVICE]:,.1f} GB/s the device, {R[w + CTA]:.2f} bytes a cycle an SM, {where} its band.")
    table = "\n".join(
        ["| the walk | GB/s the device | bytes a cycle an SM | the prediction | held |", "|---|---:|---:|---|---|"] + rows
    )
    checked = sum(1 for c in d["checks"] if c["lab"] == "check" and c["ok"])
    text = "\n".join(paragraphs) + f"""
MEASURED ({card(d['device'], instance)}, driver {d['driver']}; {d['date'][:10]}, the experiment at
{d['git']['sha'][:8]}, run {run}; {d['reps']} repetitions, the {checked} checked walks exact; the median, the widest spread
of a row {100 * spread:.1f}%; the clock as the card left it):

{table}

WHAT IT SHOWS.
{chr(10).join(shows)}

Record: `{sample.cited}`.
"""
    p = root / "docs/lab/stream_rate.md"
    s = p.read_text()
    m = re.search(r"^PREDICTION\b", s, re.M)
    i = m.start() if m else len(s.rstrip("\n")) + 2
    p.write_text(s[:i].rstrip("\n") + "\n\n" + text)
    return [p]
