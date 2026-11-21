# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The measured section of `docs/lab/step_rolled.md`, generated from `measure/lab/results/step_rolled/`.

`write(root, instance)` reads `sm_90a.blocks1.json` and `sm_90a.blocks132.json` (one CTA and the full grid) under
`root` and rewrites `docs/lab/step_rolled.md` under `root` from its `MEASURED` line to its end (where the page has none,
it appends the section after the page's text), every figure the records' own and every comparative word chosen by the
figures it compares. It REFUSES, naming what is missing, where a record is absent, holds none of the experiment's three
rows, or holds a row whose check failed. `instance` is the rented box's label (`nb...` Nebius, `hs...` Hyperstack, digits
vast.ai), which only words the card's line. It returns the files it wrote (`lab page step_rolled --instance LABEL`,
docs/internals/tools/lab.md).
"""

from __future__ import annotations

import json
import pathlib
import re

from _record import Record, against, card

UNIT = "cycles a step"
#: the experiment's rows, as `run` in the source names them: (the row's name, its label in the table)
STRAIGHT = "16 steps a straight block behind one fence"
FENCED = "16 steps a straight block, a fence a step"
ROLLED = "16 steps each in a rolled loop of a run-time count of one, a fence a step"
FORMS = (
    (STRAIGHT, "STRAIGHT: a straight block behind one fence"),
    (FENCED, "FENCED: a straight block, a fence a step"),
    (ROLLED, "ROLLED: each step in a loop of a run-time count of one, a fence a step"),
)
GRIDS = (1, 132)


def write(root: pathlib.Path, instance: str) -> list[pathlib.Path]:
    records = {}
    for n in GRIDS:
        path = root / f"measure/lab/results/step_rolled/sm_90a.blocks{n}.json"
        if not path.exists():
            raise SystemExit(
                f"lab page: no record at measure/lab/results/step_rolled/sm_90a.blocks{n}.json -- the page's measured "
                "section is generated from runs of the experiment recorded with `lab run step_rolled --record` at one "
                "CTA and at the full grid on an sm_90a card"
            )
        records[n] = json.loads(path.read_text())
    R = {}
    for n, d in records.items():
        failed = [c["variant"] for c in d["checks"] if c["lab"] == "check" and not c["ok"]]
        if failed:
            raise SystemExit(f"lab page: the record at {n} CTAs holds failed checks, {failed}: its figures are not the page's to state")
        R[n] = Record({r["name"]: r["median"] for r in d["results"]})
        for name, _ in FORMS:
            R[n][name]
    full = records[GRIDS[-1]]
    checked = {n: sum(1 for c in d["checks"] if c["lab"] == "check" and c["ok"]) for n, d in records.items()}
    spread = max(
        (r["max"] - r["min"]) / r["median"] for d in records.values() for r in d["results"] if r["name"] in dict(FORMS)
    )
    card_line = card(full['device'], instance)
    table = "\n".join(
        ["| form | 1 CTA | 132 CTAs |", "|---|---:|---:|"]
        + [f"| {label} | {R[1][name]:.2f} | {R[132][name]:.2f} |" for name, label in FORMS]
    )
    lines = []
    for n in GRIDS:
        s, f, r = (R[n][name] for name, _ in FORMS)
        fence = against(f, s)
        fence_words = (
            f"{f:.2f} {UNIT}, the same as the {s:.2f} of STRAIGHT: a fence a step is free"
            if fence == "the same as"
            else f"{f:.2f} {UNIT}, {abs(f - s):.2f} {UNIT} {fence} the {s:.2f} of STRAIGHT"
        )
        lines.append(f"- AT {n} CTA{'s' if n > 1 else ''}, THE STRAIGHT BLOCK: {s:.2f} {UNIT}.")
        lines.append(f"- AT {n} CTA{'s' if n > 1 else ''}, A FENCE A STEP: {fence_words}.")
        lines.append(
            f"- AT {n} CTA{'s' if n > 1 else ''}, EACH STEP IN A ROLLED LOOP OF A RUN-TIME COUNT: {r:.2f} {UNIT}, "
            f"{r / s:.2f} times the {s:.2f} of STRAIGHT ({abs(r - s):.2f} {UNIT} {against(r, s)})."
        )
    d1 = records[1]
    text = f"""MEASURED ({card_line}, driver {full['driver']}; {d1['date'][:10]} (the one-CTA and the full-grid records), the experiment
at {full['git']['sha'][:8]}; {full['reps']} repetitions; the {checked[1]} checked rows at one CTA and the {checked[132]} at 132 pass, every sum exact; the median, the widest
spread of a row {100 * spread:.1f}%). {UNIT[0].upper()}{UNIT[1:]}:

{table}

WHAT IT SHOWS.
{chr(10).join(lines)}

Record: `measure/lab/results/step_rolled/sm_90a.blocks1.json` and `sm_90a.blocks132.json`.
"""
    p = root / "docs/lab/step_rolled.md"
    s = p.read_text()
    m = re.search(r"^MEASURED\b", s, re.M)
    i = m.start() if m else len(s.rstrip("\n")) + 2
    p.write_text(s[:i].rstrip("\n") + "\n\n" + text)
    return [p]
