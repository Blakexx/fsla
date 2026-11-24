# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The measured section of `docs/lab/mask_prepare_time.md`, generated from `measure/lab/results/mask_prepare_time/sm_90a.json`.

`write(root, instance)` reads that record under `root`, and the lab kernel's record
`measure/lab/results/mask_prep/sm_90a.json` it is set beside, and rewrites `docs/lab/mask_prepare_time.md` under `root`
from its `MEASURED (` line to its end: every figure a record's own or arithmetic on one. It REFUSES, naming what is
missing, where a record is absent, the record lacks one of the experiment's seven rows or holds a row whose check
failed, the record is not the one the predictions were written for (`PREDICTED_FOR`), or its misses are not the ones
the reading speaks of (`MISSED`). `instance` is the rented box's label, which only words the card's line. It returns
the files it wrote (`lab page mask_prepare_time --instance LABEL`, docs/internals/tools/lab.md).
"""

from __future__ import annotations

import json
import pathlib

from _record import Record, card

UNIT = "SM-cycles a token"
GROUPS = (1, 4, 16, 32, 64, 128, 256)
#: a level-side's digits and a window's mask words in the experiment, and a word's tokens
DIGITS, WORDS, TOKENS = 256, 32, 32
#: the mask's bytes a token the pass reads: a 4-byte word a digit a word of 32 tokens
READ = DIGITS * 4 / TOKENS
#: THE RECORD THE PREDICTIONS ARE OF, by its experiment's commit
PREDICTED_FOR = "0fb5eb86"
#: THE RANGES PREDICTED FOR THIS FORM OF THE KERNEL before any run of it: 16 words' loads held at one digit and 8 at
#: four (priced by the mask bytes a thread has in flight), the lab kernel's row at 16 digits, and the shuffle form at 32
#: digits and above (priced as the row at 16 digits)
PREDICTED = {1: (2.9, 3.9), 4: (2.6, 3.2), 16: (2.5, 2.9), 32: (2.4, 3.0), 64: (2.4, 3.0), 128: (2.4, 3.0), 256: (2.4, 3.0)}
#: THE ROWS THE RUN MISSED, which the reading is of
MISSED = {1, 4}
READING = """WHAT THE MISSES SAY. The ranges at groups of 1 and 4 digits priced the pass by the mask bytes a thread has in flight,
counting the mask's read traffic alone. What the table above fits is the multiprocessor's TOTAL memory traffic, the
stored counts and words included: every row's bytes a token over its cycles a token is one rate, the column on the
right. At one digit the stored counts are half as many bytes again as the mask read, and no lookahead removes them."""


def name(group: int) -> str:
    return f"groups of {group} digit{'s' if group > 1 else ''}, the mask once-read"


def stored(group: int) -> float:
    """The bytes a token the pass stores at `group`: a count a word and a total a group (2 bytes each), and a word a word (4) where the group is more than a digit."""
    groups = DIGITS // group
    return groups * (2 * WORDS + 2 + (4 * WORDS if group > 1 else 0)) / (WORDS * TOKENS)


def _load(root: pathlib.Path, experiment: str) -> dict:
    path = root / "measure/lab/results" / experiment / "sm_90a.json"
    if not path.exists():
        raise SystemExit(f"lab page: no record at measure/lab/results/{experiment}/sm_90a.json")
    d = json.loads(path.read_text())
    failed = [c["variant"] for c in d["checks"] if c["lab"] == "check" and not c["ok"]]
    if failed:
        raise SystemExit(f"lab page: the record of {experiment} holds failed checks, {failed}: its figures are not the page's to state")
    return d


def write(root: pathlib.Path, instance: str) -> list[pathlib.Path]:
    d = _load(root, "mask_prepare_time")
    lab = _load(root, "mask_prep")
    if not d["git"]["sha"].startswith(PREDICTED_FOR) or d["git"].get("dirty"):
        raise SystemExit(
            f"lab page: the predictions here were written for the run of the experiment at {PREDICTED_FOR}, and the "
            f"record is of {d['git']['sha'][:8]}{' with uncommitted changes' if d['git'].get('dirty') else ''}: write "
            "that run's predictions and the reading of its misses (PREDICTED_FOR, PREDICTED, MISSED, READING), then generate"
        )
    R = Record({r["name"]: r["median"] for r in d["results"]})
    L = {r["name"]: r["median"] for r in lab["results"]}
    #: the lab kernel's once-read rows, at the group sizes it measured
    lab_row = {
        1: "4 to 5 live digits a token, groups of 1 digit, the counts alone, the mask once-read",
        4: "4 to 5 live digits a token, groups of 4 digits, the groups' words and counts, the mask once-read",
        16: "4 to 5 live digits a token, groups of 16 digits, the groups' words and counts, the mask once-read",
    }
    checked = sum(1 for c in d["checks"] if c["lab"] == "check" and c["ok"])
    rows, verdicts, missed, rates = [], [], set(), []
    for group in GROUPS:
        got = R[name(group)]
        beside = f"{L[lab_row[group]]:.2f}" if group in lab_row else "not measured"
        total = READ + stored(group)
        rates.append(total / got)
        rows.append(f"| {group} | {got:.2f} | {beside} | {READ:.0f} + {stored(group):.2f} | {total / got:.1f} |")
        low, high = PREDICTED[group]
        held = low <= got <= high
        if not held:
            missed.add(group)
        verdicts.append(f"- a group of {group}: predicted {low:g} to {high:g}, measured {got:.2f} -- {'HELD' if held else 'MISSED'}.")
    if missed != MISSED:
        raise SystemExit(
            f"lab page: the record misses {sorted(missed)} and the reading here is of {sorted(MISSED)}: the reading "
            "of a run's misses (MISSED, READING) is written for that run"
        )
    table = "\n".join(
        [
            "| a group, digits | the library's pass | the lab's kernel | bytes a token, read + stored | bytes a cycle an SM |",
            "|---:|---:|---:|---:|---:|",
        ]
        + rows
    )
    text = f"""MEASURED ({card(d['device'], instance)}, driver {d['driver']}; {d['date'][:10]}, the experiment at
{d['git']['sha'][:8]}; {d['reps']} repetitions; the {checked} checked rows pass, every word, count and total of each group size's
first round exact; the median). {UNIT[0].upper()}{UNIT[1:]}, the mask read once; the lab's kernel is [`mask_prep`](mask_prep.md)'s row at
the same group size:

{table}

WHAT IT SHOWS.
- THE COST BY GROUP SIZE: from {max(R[name(g)] for g in GROUPS):.2f} at a group of one digit to {min(R[name(g)] for g in GROUPS):.2f} at the largest.
- THE SHUFFLE FORM (32 digits and above): {min(R[name(g)] for g in GROUPS[3:]):.2f} to {max(R[name(g)] for g in GROUPS[3:]):.2f}, beside {R[name(16)]:.2f} at 16 digits, where a thread is a group.
- ONE RATE: the bytes a token a row moves, read and stored, over its cycles a token is {min(rates):.1f} to {max(rates):.1f} bytes a cycle an SM
  at every group size.

THE PREDICTIONS WRITTEN FOR THIS FORM OF THE KERNEL, AND THEIR BASIS. The lab's kernel had measured {L[lab_row[1]]:.2f}, {L[lab_row[4]]:.2f} and
{L[lab_row[16]]:.2f} at groups of 1, 4 and 16 digits, with 8, 32 and 128 bytes of the mask in flight a thread. This kernel holds 16
words' loads at one digit a thread (64 bytes in flight) and 8 at four digits (128): each was predicted between the
lab's rows at the bytes in flight around it. A group of 16 digits was predicted as the lab's row; 32 digits and above,
which no kernel had measured, as the row at 16 digits, a thread's mask traffic being the same.
{chr(10).join(verdicts)}

{READING}

Record: `measure/lab/results/mask_prepare_time/sm_90a.json`; the lab kernel's: `measure/lab/results/mask_prep/sm_90a.json`.
"""
    p = root / "docs/lab/mask_prepare_time.md"
    s = p.read_text()
    i = s.index("MEASURED (") if "MEASURED (" in s else len(s.rstrip("\n")) + 2
    p.write_text(s[:i].rstrip("\n") + "\n\n" + text)
    return [p]
