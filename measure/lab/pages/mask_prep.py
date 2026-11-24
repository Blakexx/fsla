# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The measured section of `docs/lab/mask_prep.md`, generated from `measure/lab/results/mask_prep/sm_90a.json`.

`write(root, instance)` reads that record under `root` and rewrites `docs/lab/mask_prep.md` under `root` from its
`MEASURED (` line to its end, every figure a record's own and every comparative word chosen by the figures it
compares. The record is graded against the predictions written before its run, whose basis is the experiment's FIRST
run, the record `measure/lab/results/mask_prep/sm_90a.first.json`. It REFUSES, naming what is missing, where either
record is absent, the record lacks one of the experiment's twelve rows or holds a row whose check failed, the record
is not the one the predictions were written for (`PREDICTED_FOR`), or its misses are not the ones the reading below
speaks of (`MISSED`): a new run's predictions, their basis and the reading of its misses are written here, the
predictions before it runs. `instance` is the rented box's label (`nb...` Nebius, `hs...` Hyperstack, digits vast.ai),
which only words the card's line. It returns the files it wrote (`lab page mask_prep --instance LABEL`,
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
#: the first run's in-cache rows' words: only its row at a group of 1 digit was in cache (the others' working sets
#: were 68 and 270 MB), which is why the record's rows keep one working set at every group size
FIRST_CACHED = ", the mask in cache"
#: THE RECORD THE PREDICTIONS ARE OF, by its experiment's commit: they were written before that run and grade no other
PREDICTED_FOR = "f22d40f0"
#: THE COMPILED WALK'S WARP-INSTRUCTIONS A TOKEN at groups of 1, 4 and 16 digits: 21, 30 and 46 instructions a pass of
#: two words (`lab ledger mask_prep --static`), a thread a group, a pass 64 tokens
ISSUED = (2.9, 1.0, 0.38)
#: THE PREDICTED RANGES, one a (group, how), the same for both masks (the mask's density changes no instruction):
#: once-read, the first run's figure within about 4%; in cache, `ISSUED` at the first run's one in-cache rate
PREDICTED = {
    (1, FRESH): (4.9, 5.3),
    (4, FRESH): (3.3, 3.6),
    (16, FRESH): (2.5, 2.7),
    (1, CACHED): (0.78, 1.3),
    (4, CACHED): (0.25, 0.6),
    (16, CACHED): (0.08, 0.35),
}
#: THE ROWS THE RUN MISSED, which the reading below is of
MISSED = {(4, CACHED), (16, CACHED)}
READING = """WHAT THE MISSES SAY, AND WHAT IS OPEN. The in-cache ranges priced the pass by the instructions it issues, at a rate
read off one row; the rows at groups of 4 and 16 digits falling outside them says the pass in cache is not bound by
issue there. What does bound it is OPEN: the cache's read rate fits (the pass reads 32 bytes of mask a token at every
group size), and so do the shared stores of the in-cache rows below, and no run of this experiment separates them."""


def _load(root: pathlib.Path, name: str, what: str) -> dict:
    path = root / "measure/lab/results/mask_prep" / name
    if not path.exists():
        raise SystemExit(f"lab page: no record at measure/lab/results/mask_prep/{name} -- {what}")
    d = json.loads(path.read_text())
    failed = [c["variant"] for c in d["checks"] if c["lab"] == "check" and not c["ok"]]
    if failed:
        raise SystemExit(f"lab page: the record {name} holds failed checks, {failed}: its figures are not the page's to state")
    return d


def write(root: pathlib.Path, instance: str) -> list[pathlib.Path]:
    d = _load(root, "sm_90a.json", "the page's measured section is generated from a run recorded on an sm_90a card")
    first = _load(root, "sm_90a.first.json", "the experiment's first run, the basis of the predictions the page grades")
    if not d["git"]["sha"].startswith(PREDICTED_FOR):
        raise SystemExit(
            f"lab page: the predictions here were written for the run of the experiment at {PREDICTED_FOR}, and the "
            f"record is of {d['git']['sha'][:8]}: write that run's predictions, their basis and the reading of its "
            "misses (PREDICTED_FOR, PREDICTED, MISSED, READING) as the predictions stood before it ran, then generate"
        )
    R = Record({r["name"]: r["median"] for r in d["results"]})
    F = Record({r["name"]: r["median"] for r in first["results"]})
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
    #: THE BASIS, read off the first run's record: its once-read figures, and the issue rate its one in-cache row gives
    was = [F[CELLS[0] + g + FRESH] for _, g in GROUPS]
    rate = ISSUED[0] / F[CELLS[0] + GROUPS[0][1] + FIRST_CACHED]
    basis = [
        f"- ONCE-READ: the first run (the experiment at {first['git']['sha'][:8]}, the record",
        f"  `measure/lab/results/mask_prep/sm_90a.first.json`) measured {was[0]:.2f}, {was[1]:.2f} and {was[2]:.2f} at groups of 1, 4 and 16",
        "  digits, and the kernel's memory traffic is the same in both runs: each within about 4% of that.",
        f"- IN CACHE: the compiled walk issues {ISSUED[0]:g}, {ISSUED[1]:g} and {ISSUED[2]:g} warp-instructions a token at groups of 1, 4 and 16",
        f"  digits, priced at {rate:.1f} warp-instructions a cycle an SM: the rate of the first run's row at a group of 1 digit,",
        "  the only one of its in-cache rows whose working set was in cache.",
    ]
    verdicts, missed = [], set()
    for (size, how), (low, high) in PREDICTED.items():
        g = next(words for n, words in GROUPS if n == size)
        got = [at(cell, g, how) for cell in range(len(CELLS))]
        held = all(low <= x <= high for x in got)
        if not held:
            missed.add((size, how))
        verdicts.append(
            f"- a group of {size}, {'once-read' if how == FRESH else 'in cache'}: predicted {low:g} to {high:g}, measured "
            f"{got[0]:.2f} and {got[1]:.2f} -- {'HELD' if held else 'MISSED'}."
        )
    if missed != MISSED:
        raise SystemExit(
            f"lab page: the record misses {sorted(missed)} and the reading here is of {sorted(MISSED)}: the reading "
            "of a run's misses (MISSED, READING) is written for that run"
        )
    text = f"""MEASURED ({card(d['device'], instance)}, driver {d['driver']}; {d['date'][:10]}, the experiment at
{d['git']['sha'][:8]}; {d['reps']} repetitions; the {checked} checked rows pass, every word, count and total of each kernel's first
round exact; the median). {UNIT[0].upper()}{UNIT[1:]} for one side's sparse level:

{table}

WHAT IT SHOWS.
{chr(10).join(lines)}

THE PREDICTIONS WRITTEN BEFORE THIS RUN, AND THEIR BASIS.
{chr(10).join(basis)}

AGAINST THEM ({len(verdicts) - len(missed)} held, {len(missed)} missed; the two masks' figures each):
{chr(10).join(verdicts)}

{READING}

WHAT THE IN-CACHE ROWS ARE AND ARE NOT. They run the grid over one working set at every group size: 4 windows a CTA's
worth, so at groups of 4 and 16 digits, where a CTA's round is 16 and 64 windows, several CTAs own the same windows
and store the same words, counts and totals to the same places. The outputs are the same whichever CTA writes them;
the rows' times include those shared stores. At a group of one digit each CTA owns its own windows.

Records: `measure/lab/results/mask_prep/sm_90a.json`, and the first run's, `measure/lab/results/mask_prep/sm_90a.first.json`.
"""
    p = root / "docs/lab/mask_prep.md"
    s = p.read_text()
    i = s.index("MEASURED (") if "MEASURED (" in s else len(s.rstrip("\n")) + 2
    p.write_text(s[:i].rstrip("\n") + "\n\n" + text)
    return [p]
