# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The measured section of `docs/lab/mask_prep.md`, generated from `measure/lab/results/mask_prep/sm_90a.json`.

`write(root, instance)` reads that record under `root` and rewrites `docs/lab/mask_prep.md` under `root` from its
`MEASURED (` line to its end, every figure the record's own and every comparative word chosen by the figures it
compares. It REFUSES, naming what is missing, where the record is absent, lacks one of the experiment's twelve rows, or
holds a row whose check failed. `instance` is the rented box's label (`nb...` Nebius, `hs...` Hyperstack, digits
vast.ai), which only words the card's line. It returns the files it wrote (`lab page mask_prep --instance LABEL`,
docs/internals/tools/lab.md).
"""

from __future__ import annotations

import json
import pathlib

from _record import Record, against, card

UNIT = "SM-cycles a token"
CELLS = ("4 to 5 live digits a token", "32 live digits a token")
#: a group's size and its rows' words
GROUPS = (
    (1, ", groups of 1 digit, the counts alone"),
    (4, ", groups of 4 digits, the groups' words and counts"),
    (16, ", groups of 16 digits, the groups' words and counts"),
)
CACHED = ", the mask in cache, the grid over 4 windows a CTA's worth"
FRESH = ", the mask once-read"


def write(root: pathlib.Path, instance: str) -> list[pathlib.Path]:
    path = root / "measure/lab/results/mask_prep/sm_90a.json"
    if not path.exists():
        raise SystemExit(
            "lab page: no record at measure/lab/results/mask_prep/sm_90a.json -- the page's measured section is "
            "generated from a run of the experiment recorded on an sm_90a card"
        )
    d = json.loads(path.read_text())
    failed = [c["variant"] for c in d["checks"] if c["lab"] == "check" and not c["ok"]]
    if failed:
        raise SystemExit(f"lab page: the record holds failed checks, {failed}: its figures are not the page's to state")
    R = Record({r["name"]: r["median"] for r in d["results"]})
    checked = sum(1 for c in d["checks"] if c["lab"] == "check" and c["ok"])

    def at(cell: int, group: str, how: str) -> float:
        return R[CELLS[cell] + group + how]

    table = "\n".join(
        [
            "| a group | 4 to 5 live digits: in cache | once-read | 32 live digits: in cache | once-read |",
            "|---|---:|---:|---:|---:|",
        ]
        + [
            f"| {size} digit{'s' if size > 1 else ''} | {at(0, g, CACHED):.2f} | {at(0, g, FRESH):.2f} | {at(1, g, CACHED):.2f} | {at(1, g, FRESH):.2f} |"
            for size, g in GROUPS
        ]
    )
    #: the widest move the denser mask makes on any row, as a share of the sparser one's figure
    moved = max(abs(at(1, g, how) - at(0, g, how)) / at(0, g, how) for _, g in GROUPS for how in (CACHED, FRESH))
    fresh = [at(0, g, FRESH) for _, g in GROUPS]
    cached = [at(0, g, CACHED) for _, g in GROUPS]
    lines = [
        f"- THE MASK'S DENSITY: the denser mask moves no row by more than {100 * moved:.1f}% of the sparser one's figure.",
        f"- ONCE-READ, BY THE GROUP'S SIZE: {fresh[0]:.2f}, {fresh[1]:.2f} and {fresh[2]:.2f} {UNIT} at groups of 1, 4 and 16 digits; "
        f"a group of 16 is {against(fresh[2], fresh[0])} a group of 1, {fresh[2] / fresh[0]:.2f} of it.",
        f"- IN CACHE: {cached[0]:.2f}, {cached[1]:.2f} and {cached[2]:.2f}; a group of 16 is {against(cached[2], cached[0])} a group "
        f"of 1, {cached[2] / cached[0]:.2f} of it.",
    ] + [
        f"- A GROUP OF {size}, ONCE-READ AGAINST IN CACHE: {at(0, g, FRESH) / at(0, g, CACHED):.1f} times."
        for size, g in GROUPS
    ]
    text = f"""MEASURED ({card(d['device'], instance)}, driver {d['driver']}; {d['date'][:10]}, the experiment at
{d['git']['sha'][:8]}; {d['reps']} repetitions; the {checked} checked rows pass, every word, count and total of each kernel's first
round exact; the median). {UNIT[0].upper()}{UNIT[1:]} for one side's sparse level:

{table}

WHAT IT SHOWS.
{chr(10).join(lines)}

WHAT THE IN-CACHE ROWS ARE AND ARE NOT. They run the grid over one working set at every group size: 4 windows a CTA's
worth, so at groups of 4 and 16 digits, where a CTA's round is 16 and 64 windows, several CTAs own the same windows
and store the same words, counts and totals to the same places. The outputs are the same whichever CTA writes them;
the rows' times include those shared stores. At a group of one digit each CTA owns its own windows.

Record: `measure/lab/results/mask_prep/sm_90a.json`.
"""
    p = root / "docs/lab/mask_prep.md"
    s = p.read_text()
    i = s.index("MEASURED (") if "MEASURED (" in s else len(s.rstrip("\n")) + 2
    p.write_text(s[:i].rstrip("\n") + "\n\n" + text)
    return [p]
