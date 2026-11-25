# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The measured section of `docs/lab/mask_prepare_time.md`, generated from `measure/lab/results/mask_prepare_time/sm_90a.json`.

`write(root, instance)` reads that record under `root`, and the lab kernel's record
`measure/lab/results/mask_prep/sm_90a.json` it is set beside, and rewrites `docs/lab/mask_prepare_time.md` under `root`
from its `MEASURED (` line to its end: every figure a record's own or arithmetic on one. It REFUSES, naming what is
missing, where a record is absent, the record lacks one of the experiment's rows or holds a row whose check
failed, the record is not the one the predictions were written for (`PREDICTED_FOR`), or its misses are not the ones
the reading speaks of (`MISSED`). `instance` is the rented box's label, which only words the card's line. It returns
the files it wrote (`lab page mask_prepare_time --instance LABEL`, docs/internals/tools/lab.md).
"""

from __future__ import annotations

import json
import pathlib

from _record import Record, card

UNIT = "SM-cycles a token"
GROUPS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
#: the sizes whose lanes OR by warp shuffle
SHUFFLED = tuple(g for g in GROUPS if g > 16)
#: a level-side's digits and a window's mask words in the experiment, and a word's tokens
DIGITS, WORDS, TOKENS = 256, 32, 32
#: the mask's bytes a token the pass reads: a 4-byte word a digit a word of 32 tokens
READ = DIGITS * 4 / TOKENS
#: THE RECORD THE PREDICTIONS ARE OF, by its experiment's commit
PREDICTED_FOR = "9c2a0f30"
#: THE RANGES PREDICTED FOR THIS FORM OF THE KERNEL before any run of it: the bytes a token it reads and stores over the
#: one rate the pass with a count a word had fit (11.8 to 12.9 bytes a cycle an SM, the experiment at 0fb5eb86), the row
#: at one digit widened upward for its held loads, 8 words' where they were 16
#: groups of 2 and of 8 are forms the run before did not have (a pair a thread; two lines a thread, 4 words' loads held): the same rate over their bytes
PREDICTED = {1: (2.5, 3.1), 2: (3.6, 4.2), 4: (3.0, 3.5), 8: (2.7, 3.15), 16: (2.6, 2.9), 32: (2.6, 2.9), 64: (2.5, 2.85), 128: (2.5, 2.8), 256: (2.4, 2.7)}
#: THE ROWS THE RUN MISSED, which the reading is of
MISSED: set[int] = set()
READING = """WHAT THE RUN SAYS, AND WHAT IS NOT SHOWN. Every range held. The ranges came from one rate over the bytes a token read
and stored, fitted on the pass when it stored a count a word; this form stores fewer bytes and its rows fall where that
rate puts them. It is two forms on one rate and no more: no counter was read, so that the multiprocessor's memory
traffic is what bounds the pass is LIKELY, NOT SHOWN."""


def name(group: int) -> str:
    return f"groups of {group} digit{'s' if group > 1 else ''}, the mask once-read"


def stored(group: int) -> float:
    """The bytes a token the pass stores at `group`: a total a group (2 bytes), and a word a mask word (4) where the group is more than a digit."""
    groups = DIGITS // group
    return groups * (2 + (4 * WORDS if group > 1 else 0)) / (WORDS * TOKENS)


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
    high = max(GROUPS, key=lambda g: R[name(g)])
    low = min(GROUPS, key=lambda g: R[name(g)])
    table = "\n".join(
        [
            "| a group, digits | the library's pass | the lab's kernel | bytes a token, read + stored | bytes a cycle an SM |",
            "|---:|---:|---:|---:|---:|",
        ]
        + rows
    )
    text = f"""MEASURED ({card(d['device'], instance)}, driver {d['driver']}; {d['date'][:10]}, the experiment at
{d['git']['sha'][:8]}; {d['reps']} repetitions; the {checked} checked rows pass, every word and total of each group size's
first round exact; the median). {UNIT[0].upper()}{UNIT[1:]}, the mask read once; the lab's kernel is [`mask_prep`](mask_prep.md)'s row at
the same group size:

{table}

WHAT IT SHOWS.
- THE COST BY GROUP SIZE: {R[name(1)]:.2f} at a group of one digit; the highest {R[name(high)]:.2f}, at a group of {high}; the lowest {R[name(low)]:.2f}, at a group of {low}.
- THE SHUFFLE FORM (32 digits and above): {min(R[name(g)] for g in SHUFFLED):.2f} to {max(R[name(g)] for g in SHUFFLED):.2f}, beside {R[name(16)]:.2f} at 16 digits, where a thread is a group.
- ONE RATE: the bytes a token a row moves, read and stored, over its cycles a token is {min(rates):.1f} to {max(rates):.1f} bytes a cycle an SM
  at every group size.

THE PREDICTIONS WRITTEN FOR THIS FORM OF THE KERNEL, AND THEIR BASIS. The pass when it stored a count a word, measured
at the experiment's 0fb5eb86, fit one rate over the bytes a token it read and stored: 11.8 to 12.9 bytes a cycle an SM.
Each row here was predicted as the bytes this form moves a token over that rate; the row at one digit was widened
upward, its held loads being 8 words' where they were 16.
{chr(10).join(verdicts)}

{READING}

Record: `measure/lab/results/mask_prepare_time/sm_90a.json`; the lab kernel's: `measure/lab/results/mask_prep/sm_90a.json`.
"""
    p = root / "docs/lab/mask_prepare_time.md"
    s = p.read_text()
    i = s.index("MEASURED (") if "MEASURED (" in s else len(s.rstrip("\n")) + 2
    p.write_text(s[:i].rstrip("\n") + "\n\n" + text)
    return [p]
