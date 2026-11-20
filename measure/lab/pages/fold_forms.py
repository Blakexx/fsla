# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The measured section of `docs/lab/fold_forms.md`, generated from `measure/lab/results/fold_forms/sm_90a.json`.

`write(root, instance)` reads that record under `root` and rewrites `docs/lab/fold_forms.md` under `root` from its
`MEASURED (` line (the page ends before it, at the end of THE FORM, until a record is written) to its end, every
figure the record's own and every comparative word chosen by the figures it compares. It REFUSES, naming what is
missing, where the record is absent, holds none of the experiment's four rows, or holds a row whose check failed.
`instance` is the rented box's label (`nb...` Nebius, `hs...` Hyperstack, digits vast.ai), which only words the card's
line. It returns the files it wrote (`lab page fold_forms --instance LABEL`, docs/internals/tools/lab.md).
"""

from __future__ import annotations

import json
import pathlib

from _record import Record

#: two windows within this share of each other (of the lesser) are said to be the same
SAME = 0.02
UNIT = "cycles a window a CTA"
#: the experiment's rows by their names, in the table's order
TILE = "the fold's coefficients a tile in shared memory, a step a list"
EVERY = "the fold's coefficients in registers, a step four digits, every class computed"
OWN = "the fold's coefficients in registers, a step four digits, a warp's own classes computed"
TABLE = "the fold's coefficients a tile, its dense lines fetched from the level by token"
ROWS = (
    (TILE, "the tile form: the coefficients a tile in shared memory, a step a list"),
    (EVERY, "the register form, every class computed: a step four digits"),
    (OWN, "the register form, a warp's own classes computed"),
    (TABLE, "the tile form, its dense lines fetched from the level by token"),
)


def against(a: float, b: float) -> str:
    """How `a` stands to `b`: `the same as` within `SAME` of the lesser, else `below` or `above`."""
    if abs(a - b) <= SAME * min(a, b):
        return "the same as"
    return "below" if a < b else "above"


def write(root: pathlib.Path, instance: str) -> list[pathlib.Path]:
    path = root / "measure/lab/results/fold_forms/sm_90a.json"
    if not path.exists():
        raise SystemExit(
            "lab page: no record at measure/lab/results/fold_forms/sm_90a.json -- the page's measured section is "
            "generated from a run of the experiment recorded with `lab run fold_forms --record` on an sm_90a card"
        )
    d = json.loads(path.read_text())
    rows = {r["name"]: r for r in d["results"]}
    R = Record({n: r["median"] for n, r in rows.items()})
    failed = [c["variant"] for c in d["checks"] if c["lab"] == "check" and not c["ok"]]
    if failed:
        raise SystemExit(f"lab page: the record holds failed checks, {failed}: its figures are not the page's to state")
    for name, _ in ROWS:
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
    tile = R[TILE]
    table = "\n".join(
        ["| form | cycles a window a CTA | against the tile form |", "|---|---:|---:|"]
        + [f"| {label} | {R[n]:,.0f} | {R[n] / tile:.2f} |" for n, label in ROWS]
    )
    cheapest = min(ROWS, key=lambda row: R[row[0]])[1]
    least = R[min(ROWS, key=lambda row: R[row[0]])[0]]
    own_saves = R[EVERY] - R[OWN]

    def line(name: str, what: str) -> str:
        word = against(R[name], tile)
        by = f"{abs(R[name] - tile):,.0f} cycles {word}" if word != "the same as" else f"{word}"
        return f"- {what}: {R[name]:,.0f} {UNIT}, {R[name] / tile:.2f} of the tile form's {tile:,.0f}, {by} it."

    text = f"""MEASURED ({card}, driver {d['driver']}; {d['date'][:10]}, the experiment at
{d['git']['sha'][:8]}; {d['reps']} repetitions; the {checked} checked rows pass, every accumulator exact; the median, the widest
spread of a row {100 * max(spread(n) for n, _ in ROWS):.1f}%). {UNIT[0].upper()}{UNIT[1:]}:

{table}

WHAT IT SHOWS.
{line(EVERY, "THE REGISTER FORM, EVERY CLASS COMPUTED")}
{line(OWN, "THE REGISTER FORM, A WARP'S OWN CLASSES COMPUTED")}
- A WARP'S OWN CLASSES AGAINST EVERY CLASS: {abs(own_saves):,.0f} cycles {'fewer' if own_saves > 0 else 'more'} a window a CTA,
  {R[OWN] / R[EVERY]:.2f} of the every-class form's {R[EVERY]:,.0f}.
{line(TABLE, "THE TILE FORM, ITS DENSE LINES FETCHED FROM THE LEVEL")}
- THE CHEAPEST FORM is {cheapest.split(':')[0]}: {least:,.0f} {UNIT}, {least / tile:.2f} of the tile form's.

Record: `measure/lab/results/fold_forms/sm_90a.json`.
"""
    p = root / "docs/lab/fold_forms.md"
    s = p.read_text()
    i = s.index("MEASURED (") if "MEASURED (" in s else len(s.rstrip("\n")) + 2
    p.write_text(s[:i].rstrip("\n") + "\n\n" + text)
    return [p]
