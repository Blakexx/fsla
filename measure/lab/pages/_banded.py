# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""A PAGE OF PREDICTIONS AND THE RECORD READ AGAINST THEM, the generators' common form where an experiment's page states
bands: `RUNS` (each run's id, the date its predictions were written before it, their basis, and its CLAIMS -- a figure
of the record's rows, by a label, with its band) is the one statement of the predictions; `write_banded` rewrites the
page from its first `PREDICTION` line to its end: every run's claims, then the newest sample's figures against the
claims of the run that recorded it, and whether each held. Where no sample is pinned yet, it writes the predictions
alone. It REFUSES where the record holds a failed check, was recorded by a run `RUNS` does not hold, or lacks a row a
claim reads (`Record`)."""

from __future__ import annotations

import pathlib
import re
from collections.abc import Callable
from dataclasses import dataclass

from _record import Record, card, read

#: the run not yet recorded: its predictions stand alone until its sample is pinned and takes its id
PENDING = "pending"


@dataclass(frozen=True)
class Claim:
    #: what the figure is, as the page words it
    label: str
    #: the figure, from the record's rows by name
    figure: Callable[[Record], float]
    lo: float
    hi: float
    unit: str


def _band(c: Claim) -> str:
    return f"{c.lo:,.2f}-{c.hi:,.2f} {c.unit}"


def write_banded(root: pathlib.Path, data: pathlib.Path, instance: str, experiment: str, runs: tuple) -> list[pathlib.Path]:
    """`runs`: `(run id, date written, basis, (Claim, ...))` in the order they ran."""
    paragraphs = []
    for rid, when, basis, cs in runs:
        lines = "\n".join(f"- {c.label}: {_band(c)}." for c in cs)
        who = "the run to come" if rid == PENDING else f"run {rid}"
        paragraphs.append(f"PREDICTION, {who} (written {when}, before it ran; its basis: {basis}):\n\n{lines}\n")
    p = root / f"docs/lab/{experiment}.md"
    s = p.read_text()
    m = re.search(r"^PREDICTION\b", s, re.M)
    i = m.start() if m else len(s.rstrip("\n")) + 2
    try:
        sample = read(data, experiment)
    except SystemExit:
        p.write_text(s[:i].rstrip("\n") + "\n\n" + "\n".join(paragraphs))
        return [p]
    d = sample.body()
    failed = [c["variant"] for c in d["checks"] if c["lab"] == "check" and not c["ok"]]
    if failed:
        raise SystemExit(f"lab page: the record holds failed checks, {failed}: its figures are not the page's to state")
    run = re.search(r"/([^/]+)/output\.json$", sample.cited).group(1)
    claims = {rid: cs for rid, _, _, cs in runs}
    if run not in claims:
        raise SystemExit(f"lab page: the newest sample's run {run} states no predictions in RUNS: none was written before it")
    R = Record({r["name"]: r["median"] for r in d["results"]})
    rows = []
    for c in claims[run]:
        f = c.figure(R)
        verdict = "held" if c.lo <= f <= c.hi else "MISSED"
        rows.append(f"| {c.label} | {f:,.2f} {c.unit} | {_band(c)} | {verdict} |")
    table = "\n".join(["| the figure | measured | the prediction | held |", "|---|---:|---|---|"] + rows)
    spread = max((r["max"] - r["min"]) / r["median"] for r in d["results"] if r["median"])
    checked = sum(1 for c in d["checks"] if c["lab"] == "check" and c["ok"])
    figures = "\n".join(f"- {r['name']}: {r['median']:,.2f} {r['unit']}." for r in d["results"])
    text = "\n".join(paragraphs) + f"""
MEASURED ({card(d['device'], instance)}, driver {d['driver']}; {d['date'][:10]}, the experiment at
{d['git']['sha'][:8]}, run {run}; {d['reps']} repetitions, the {checked} checked variants exact; the median, the widest
spread of a row {100 * spread:.1f}%; the clock as the card left it):

{table}

The record's rows:
{figures}

Record: `{sample.cited}`.
"""
    p.write_text(s[:i].rstrip("\n") + "\n\n" + text)
    return [p]
