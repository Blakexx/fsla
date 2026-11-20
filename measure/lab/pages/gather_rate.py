# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The measured section of `docs/lab/gather_rate.md`, generated from `measure/lab/results/gather_rate/sm_90a.json`.

`write(root, instance)` reads that record under `root` and rewrites `docs/lab/gather_rate.md` under `root` from its
`MEASURED` line (the page ends there, at a line saying no record is taken, until a record is written) to its end,
every figure the record's own and every comparative word chosen by the figures it compares. It REFUSES, naming what is
missing, where the record is absent, holds none of the experiment's ten rows, or holds a row whose check failed.
`instance` is the rented box's label (`nb...` Nebius, `hs...` Hyperstack, digits vast.ai), which only words the card's
line. It returns the files it wrote (`lab page gather_rate --instance LABEL`, docs/internals/tools/lab.md).
"""

from __future__ import annotations

import json
import pathlib
import re

from _record import Record

#: two rows within this share of each other (of the lesser) are said to be the same
SAME = 0.02
UNIT = "cycles a 128-byte row a CTA"
RING = "loaded, stored and copied out to a ring of 256 KB a CTA"
#: the experiment's forms: (what the row's name ends in, rows in flight, the table's label)
FORMS = (
    ("loaded and folded", 512, "LOAD: loaded and folded"),
    ("loaded and stored into shared memory", 512, "STORE: and stored into shared memory"),
    (RING, 512, "OUT, a ring: and copied out to a ring of 256 KB a CTA"),
    ("loaded, stored and copied out to fresh memory", 512, "OUT, fresh memory: and copied out to fresh memory"),
    (RING, 256, "OUT, a ring, 256 rows in flight"),
)
TABLES = (("an 8 MB table", "8 MB table"), ("a 2 GB table", "2 GB table"))


def row_name(table: str, what: str, rows: int) -> str:
    """The experiment's name for a row, as `run` in the source builds it."""
    return f"{table}, {rows} rows in flight: {what}"


def against(a: float, b: float) -> str:
    """How `a` stands to `b`: `the same as` within `SAME` of the lesser, else `below` or `above`."""
    if abs(a - b) <= SAME * min(a, b):
        return "the same as"
    return "below" if a < b else "above"


def write(root: pathlib.Path, instance: str) -> list[pathlib.Path]:
    path = root / "measure/lab/results/gather_rate/sm_90a.json"
    if not path.exists():
        raise SystemExit(
            "lab page: no record at measure/lab/results/gather_rate/sm_90a.json -- the page's measured section is "
            "generated from a run of the experiment recorded with `lab run gather_rate --record` on an sm_90a card"
        )
    d = json.loads(path.read_text())
    rows = {r["name"]: r for r in d["results"]}
    R = Record({n: r["median"] for n, r in rows.items()})
    failed = [c["variant"] for c in d["checks"] if c["lab"] == "check" and not c["ok"]]
    if failed:
        raise SystemExit(f"lab page: the record holds failed checks, {failed}: its figures are not the page's to state")
    names = {(t, i): row_name(t, what, rows_in_flight) for t, _ in TABLES for i, (what, rows_in_flight, _) in enumerate(FORMS)}
    for name in names.values():
        R[name]
    checked = sum(1 for c in d["checks"] if c["lab"] == "check" and c["ok"])

    def spread(name: str) -> float:
        r = rows[name]
        return (r["max"] - r["min"]) / r["median"]

    inst = instance
    card = (
        f"{d['device']}, Hyperstack spot VM {inst[2:]}" if inst.startswith("hs")
        else f"{d['device']}, Nebius preemptible VM {inst[2:]}" if inst.startswith("nb")
        else f"{d['device']}, vast.ai instance {inst}"
    )
    small, large = (t for t, _ in TABLES)
    table = "\n".join(
        ["| form | 8 MB table | 2 GB table |", "|---|---:|---:|"]
        + [f"| {label} | {R[names[small, i]]:.2f} | {R[names[large, i]]:.2f} |" for i, (_, _, label) in enumerate(FORMS)]
    )

    def at(t: str, i: int) -> float:
        return R[names[t, i]]

    def line(what: str, t: str, a: int, b: int) -> str:
        """`what`: form `a` against form `b` on table `t`."""
        word = against(at(t, a), at(t, b))
        by = f"{abs(at(t, a) - at(t, b)):.2f} cycles {word}" if word != "the same as" else word
        return f"- {what}: {at(t, a):.2f} {UNIT}, {by} the {at(t, b):.2f} of {FORMS[b][2].split(':')[0]}."

    lines = []
    for t, label in TABLES:
        lines.append(f"- THE {label.upper()}, THE LOAD: {at(t, 0):.2f} {UNIT}.")
        lines.append(line(f"THE {label.upper()}, THE STORE INTO SHARED MEMORY ADDED", t, 1, 0))
        lines.append(line(f"THE {label.upper()}, THE COPY OUT TO A RING ADDED", t, 2, 1))
        lines.append(line(f"THE {label.upper()}, FRESH MEMORY AGAINST THE RING", t, 3, 2))
        lines.append(line(f"THE {label.upper()}, 256 ROWS IN FLIGHT AGAINST 512 (THE RING)", t, 4, 2))
    shown = "\n".join(lines)

    text = f"""MEASURED ({card}, driver {d['driver']}; {d['date'][:10]}, the experiment at
{d['git']['sha'][:8]}; {d['reps']} repetitions; the {checked} checked rows pass, every word exact; the median, the widest
spread of a row {100 * max(spread(n) for n in names.values()):.1f}%). {UNIT[0].upper()}{UNIT[1:]}:

{table}

WHAT IT SHOWS.
{shown}

Record: `measure/lab/results/gather_rate/sm_90a.json`.
"""
    p = root / "docs/lab/gather_rate.md"
    s = p.read_text()
    m = re.search(r"^MEASURED\b", s, re.M)
    i = m.start() if m else len(s.rstrip("\n")) + 2
    p.write_text(s[:i].rstrip("\n") + "\n\n" + text)
    return [p]
