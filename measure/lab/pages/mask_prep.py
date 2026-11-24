# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The measured section of `docs/lab/mask_prep.md`, generated from `measure/lab/results/mask_prep/sm_90a.json`.

`write(root, instance)` reads that record under `root` and rewrites `docs/lab/mask_prep.md` under `root` from its
`MEASURED (` line to its end, every figure the record's own and every comparative word chosen by the figures it
compares. It REFUSES, naming what is missing, where the record is absent, lacks one of the experiment's twelve rows,
holds a row whose check failed, or is not the record its predictions were written for (`PREDICTED_FOR`): a new run's
predictions and their basis are written here before it runs. `instance` is the rented box's label (`nb...` Nebius, `hs...` Hyperstack, digits
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
#: THE RECORD THE PREDICTIONS ARE OF, by its experiment's commit: they were written before that run and grade no other
PREDICTED_FOR = "f22d40f0"
#: WHAT EACH RANGE WAS DERIVED FROM, as written before the run
BASIS = {
    FRESH: "the experiment's first run (an H100, 132 CTAs, the experiment at 92c67f2d; its record is not kept, this "
    "one replaced it) measured 5.08, 3.45 and 2.60 at groups of 1, 4 and 16 digits, and the kernel's memory traffic "
    "is the same in both, so each within about 4% of that",
    CACHED: "the compiled walk is 21, 30 and 46 instructions a pass of two words at groups of 1, 4 and 16 digits "
    "(2.9, 1.0 and 0.38 warp-instructions a token), priced at 2.9 warp-instructions a cycle an SM: the rate of "
    "the first run's row at a group of 1 digit, the only one of its rows that was in cache",
}
#: THE PREDICTED RANGES, one a (group, how), the same for both masks (the mask's density changes no instruction)
PREDICTED = {
    (1, FRESH): (4.9, 5.3),
    (4, FRESH): (3.3, 3.6),
    (16, FRESH): (2.5, 2.7),
    (1, CACHED): (0.78, 1.3),
    (4, CACHED): (0.25, 0.6),
    (16, CACHED): (0.08, 0.35),
}


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
    if not d["git"]["sha"].startswith(PREDICTED_FOR):
        raise SystemExit(
            f"lab page: the predictions here were written for the run of the experiment at {PREDICTED_FOR}, and the "
            f"record is of {d['git']['sha'][:8]}: write that run's predictions and their basis (PREDICTED_FOR, BASIS, "
            "PREDICTED) as they stood before it ran, then generate"
        )
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
    verdicts = []
    for (size, how), (low, high) in PREDICTED.items():
        g = next(words for n, words in GROUPS if n == size)
        got = [at(cell, g, how) for cell in range(len(CELLS))]
        held = all(low <= x <= high for x in got)
        verdicts.append(
            f"- a group of {size}, {'once-read' if how == FRESH else 'in cache'}: predicted {low:g} to {high:g}, measured "
            f"{got[0]:.2f} and {got[1]:.2f} -- {'HELD' if held else 'MISSED'}."
        )
    missed = [v for v in verdicts if v.endswith("MISSED.")]
    misses = (
        """
WHAT THE MISSES SAY, AND WHAT IS OPEN. A range priced by the instructions the pass issues, at a rate read off one
row, that its row falls outside says the pass is not bound by issue there. What does bound it is OPEN: the cache's
read rate fits (the pass reads 32 bytes of mask a token at every group size), and so do the shared stores of the
in-cache rows below, and no run of this experiment separates them.
"""
        if missed
        else ""
    )
    text = f"""MEASURED ({card(d['device'], instance)}, driver {d['driver']}; {d['date'][:10]}, the experiment at
{d['git']['sha'][:8]}; {d['reps']} repetitions; the {checked} checked rows pass, every word, count and total of each kernel's first
round exact; the median). {UNIT[0].upper()}{UNIT[1:]} for one side's sparse level:

{table}

WHAT IT SHOWS.
{chr(10).join(lines)}

AGAINST THE PREDICTIONS WRITTEN BEFORE THIS RUN ({len(verdicts) - len(missed)} held, {len(missed)} missed; the two masks' figures each).
Once-read, the basis: {BASIS[FRESH]}. In cache, the basis: {BASIS[CACHED]}.
{chr(10).join(verdicts)}
{misses}
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
