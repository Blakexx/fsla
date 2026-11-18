# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The measured section of `docs/lab/stamp_cost.md`, generated from `measure/lab/results/stamp_cost/sm_90a.json`.

`write(root, instance)` reads that record under `root` and rewrites `docs/lab/stamp_cost.md` under `root` from its
`MEASURED (` line (the first run: from its `NOT YET MEASURED` line) through its `Record:` line, every figure the record's
own. `instance` is the rented box's label (`nb...` Nebius, `hs...` Hyperstack, digits vast.ai), which only words the
card's line. It returns the files it wrote (`lab page stamp_cost --instance LABEL`, docs/internals/tools/lab.md).
"""

from __future__ import annotations

import json
import pathlib

from _record import Record


def write(root: pathlib.Path, instance: str) -> list[pathlib.Path]:
    inst = instance
    d = json.loads((root / "measure/lab/results/stamp_cost/sm_90a.json").read_text())
    R = Record({r["name"]: r["median"] for r in d["results"]})

    STEPS = ("a wgmma step", "a chained step", "an independent step")
    LEAN = "a ring, every lane to one entry, no wrap"
    HELD = "a ring, one lane, its predicate held"
    FOLD = "a ring, one lane, its predicate in the offset"
    TRACE = "the lab's trace"
    STORE = "the ring store alone"
    FORMS = (
        ("the lean stamp: every lane to one entry, no predicate, no wrap", LEAN),
        ("one lane, its predicate held", HELD),
        ("one lane, its predicate in the offset", FOLD),
        ("the lab's trace", TRACE),
        ("the store alone, no clock read", STORE),
    )
    ONE, TWO = ", one stamp a step", ", two stamps a step"

    def plain(step):
        return R[f"{step}, plain"]

    def added(step, form, per):
        """cycles a stamp adds to the step: the form's median less the plain step's, over its stamps a step"""
        return (R[f"{step}, {form}{ONE if per == 1 else TWO}"] - plain(step)) / per

    head = "| the form |" + "".join(
        f" {s}, plain {plain(s):.1f}: one stamp a step | two stamps a step |" for s in STEPS
    )
    tab = [head, "|---|" + "---|---|" * 3]
    for label, form in FORMS:
        row = f"| {label} |"
        for s in STEPS:
            if f"{s}, {form}{ONE}" in R:
                row += f" {added(s, form, 1):.1f} | {added(s, form, 2):.1f} |"
            else:
                row += " -- | -- |"
        tab.append(row)

    CARD = (
        f"{d['device']}, 132 SMs, Nebius preemptible VM {inst[2:]}"
        if inst.startswith("nb")
        else f"{d['device']}, Hyperstack spot VM {inst[2:]}"
        if inst.startswith("hs")
        else f"{d['device']}, vast.ai instance {inst}"
    )
    assert all(c["ok"] for c in d["checks"]), "lab page: the record holds a check that failed"
    W, C, I = STEPS
    lean_c1, lean_c2 = added(C, LEAN, 1), added(C, LEAN, 2)
    lean_i1, lean_i2 = added(I, LEAN, 1), added(I, LEAN, 2)
    lean_w2 = added(W, LEAN, 2)
    wait_lo = min(added(W, f, 1) for f in (LEAN, HELD))
    wait_hi = max(added(W, f, 1) for f in (LEAN, HELD))

    text = f"""MEASURED ({CARD}, driver {d["driver"]}; {d["date"][:10]}, the experiment at
{d["git"]["sha"][:8]}; {d["reps"]} repetitions; every check passed: every kernel's work equal to the host's, every ring
form's stamps read back in order; the median): THE CYCLES A STAMP ADDS TO ITS STEP, one stamp a step and two, each the
form's cycles a step less the plain step's over its stamps a step (the plain step's cycles in the column head; a dash is
a form the step does not run):

{chr(10).join(tab)}

WHAT IT DECIDES.
- ONE STAMP A STEP COSTS A WAITING STEP NOTHING MEASURABLE ({wait_lo:.1f} to {wait_hi:.1f} cycles on {plain(W):.1f}, the lean stamp and the held
  predicate) AND AN INTEGER STEP {lean_c1:.1f} TO {added(C, HELD, 1):.1f} CYCLES on the chained step in the two good forms (the lean stamp
  {lean_c1:.1f}, the held predicate {added(C, HELD, 1):.1f}); on the independent step the lean stamp is {lean_i1:.1f} and the held predicate {added(I, HELD, 1):.1f}.
- NO FORM IS CHEAPEST EVERYWHERE. The lean stamp is where stamps are a step apart (one a step: {added(W, LEAN, 1):.1f}, {lean_c1:.1f} and {lean_i1:.1f}
  against the held predicate's {added(W, HELD, 1):.1f}, {added(C, HELD, 1):.1f} and {added(I, HELD, 1):.1f}); the held predicate is where two fall close together (two a
  step: {added(W, HELD, 2):.1f}, {added(C, HELD, 2):.1f} and {added(I, HELD, 2):.1f} against the lean stamp's {lean_w2:.1f}, {lean_c2:.1f} and {lean_i2:.1f}). The lean form's second stamp
  costs about twice its first -- its stamp at two a step is {lean_c2 / lean_c1:.1f} times its stamp at one on the chained step and
  {lean_i2 / lean_i1:.1f} times on the independent step. NOT ISOLATED: read as two 32-lane stores a trip.
- THE FOLDED PREDICATE AND THE LAB'S TRACE ARE NEVER THE CHEAPEST of the forms that read the clock: the folded
  predicate is {added(W, FOLD, 1):.1f}, {added(C, FOLD, 1):.1f} and {added(I, FOLD, 1):.1f} at one a step, the trace {added(W, TRACE, 1):.1f}, {added(C, TRACE, 1):.1f} and {added(I, TRACE, 1):.1f}.
- AS COMPILED, a stamp's cost on a step bound by issue is its added instructions at about two cycles each for a warp
  alone: on the independent step ({plain(I):.1f} cycles plain, eight operations) the lean stamp adds {lean_i1:.1f} cycles and the
  store alone {added(I, STORE, 1):.1f}.

Record: `measure/lab/results/stamp_cost/sm_90a.json`.
"""
    p = root / "docs/lab/stamp_cost.md"
    s = p.read_text()
    i = s.index("MEASURED (") if "MEASURED (" in s else s.index("NOT YET MEASURED")
    rec = "Record: `measure/lab/results/stamp_cost/sm_90a.json`.\n"
    j = s.index(rec, i) + len(rec) if rec in s[i:] else len(s)
    p.write_text(s[:i] + text + s[j:])
    return [p]
