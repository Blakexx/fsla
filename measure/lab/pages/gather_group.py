# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The measured section of `docs/lab/gather_group.md`, generated from `data:lab/gather_group/sm_90a.json`.

`write(root, data, instance)` reads that record in the pinned data `data` and rewrites `docs/lab/gather_group.md` under `root` from its
`MEASURED` line to its end (where the page has none, it appends the section after the page's text), every figure the record's own and every comparative word chosen by the figures it compares. It REFUSES, naming what is
missing, where the record is absent, holds none of the experiment's rows, or holds a row whose check failed.
`instance` is the rented box's label (`nb...` Nebius, `hs...` Hyperstack, digits vast.ai), which only words the card's
line. It returns the files it wrote (`lab page gather_group --instance LABEL`, docs/internals/tools/lab.md).
"""

from __future__ import annotations

import json
import pathlib
import re

from _record import Record, against, card

UNIT = "cycles a block's window a CTA"
CELL = "a block's window gathered: every list a run, values only, 4 live digits a token"
LOADS = CELL + ", a thread an entry's eight loads"
LANED = CELL + ", loads and stores, 8 lanes a row"
PHASED = LANED + ", its phases clocked"
PROBES = (
    ("no packet stored", "no packet stored"),
    ("no entry's token written", "no entry's token written"),
)
PHASES = (
    "the lists and their barrier",
    "the entries' tokens and their barrier",
    "the values landing",
    "the values' stores and their barrier",
    "the packet's store",
)
#: the checked forms, in the table's order
FORMS = (
    (LOADS, "a thread an entry's eight loads"),
    (LANED, "loads and stores, 8 lanes a row"),
    (PHASED, "the same, its phases clocked"),
)


def write(root: pathlib.Path, data: pathlib.Path, instance: str) -> list[pathlib.Path]:
    path = data / "lab/gather_group/sm_90a.json"
    if not path.exists():
        raise SystemExit(
            "lab page: no record at data:lab/gather_group/sm_90a.json -- the page's measured section is "
            "generated from a run of the experiment recorded with `lab run gather_group --record` on an sm_90a card"
        )
    d = json.loads(path.read_text())
    rows = {r["name"]: r for r in d["results"]}
    R = Record({n: r["median"] for n, r in rows.items()})
    failed = [c["variant"] for c in d["checks"] if c["lab"] == "check" and not c["ok"]]
    if failed:
        raise SystemExit(f"lab page: the record holds failed checks, {failed}: its figures are not the page's to state")
    for name, _ in FORMS:
        R[name]
        R[name + ": a token a side"]
        R[name + ": an entry"]
    for probe, _ in PROBES:
        R[f"{CELL}, loads and stores, 8 lanes a row, PROBE {probe}"]
    for phase in PHASES:
        R[f"{PHASED}: thread 0, {phase}"]
    checked = sum(1 for c in d["checks"] if c["lab"] == "check" and c["ok"])

    def spread(name: str) -> float:
        r = rows[name]
        return (r["max"] - r["min"]) / r["median"]

    inst = instance
    card_line = card(d['device'], inst)
    table = "\n".join(
        ["| form | cycles a block's window a CTA | SM-cycles a token a side | cycles an entry a CTA |", "|---|---:|---:|---:|"]
        + [
            f"| {label} | {R[n]:,.0f} | {R[n + ': a token a side']:.1f} | {R[n + ': an entry']:.1f} |"
            for n, label in FORMS
        ]
    )
    laned = R[LANED]

    def probe_line(probe: str, what: str) -> str:
        v = R[f"{CELL}, loads and stores, 8 lanes a row, PROBE {probe}"]
        word = against(v, laned)
        by = f"{abs(v - laned):,.0f} cycles {word}" if word != "the same as" else word
        return f"- PROBE, {what.upper()}: {v:,.0f} {UNIT}, {by} the 8-lanes form's {laned:,.0f}."

    probes = "\n".join(probe_line(p, w) for p, w in PROBES)
    trip = sum(R[f"{PHASED}: thread 0, {p}"] for p in PHASES)
    phases = "\n".join(
        f"- THREAD 0, {p.upper()}: {R[f'{PHASED}: thread 0, {p}']:,.0f} cycles a trip, "
        f"{100 * R[f'{PHASED}: thread 0, {p}'] / trip:.0f}% of thread 0's {trip:,.0f}."
        for p in PHASES
    )
    own = R[LOADS]
    word = against(laned, own)
    by = f"{abs(laned - own):,.0f} cycles {word}" if word != "the same as" else word
    text = f"""MEASURED ({card_line}, driver {d['driver']}; {d['date'][:10]}, the experiment at
{d['git']['sha'][:8]}; {d['reps']} repetitions; the {checked} checked rows pass, every packet exact byte for byte; the median, the widest
spread of a row {100 * max(spread(n) for n, _ in FORMS):.1f}%). {UNIT[0].upper()}{UNIT[1:]}:

{table}

WHAT IT SHOWS.
- THE 8-LANES FORM AGAINST A THREAD AN ENTRY'S EIGHT LOADS: {laned:,.0f} {UNIT}, {by} the {own:,.0f} of the thread-an-entry form.
{probes}

THREAD 0'S TRIP BY PHASE (the clocked form; thread 0 alone stores the packet):
{phases}

Record: `data:lab/gather_group/sm_90a.json`.
"""
    p = root / "docs/lab/gather_group.md"
    s = p.read_text()
    m = re.search(r"^MEASURED\b", s, re.M)
    i = m.start() if m else len(s.rstrip("\n")) + 2
    p.write_text(s[:i].rstrip("\n") + "\n\n" + text)
    return [p]
