# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The measured section of `docs/lab/gather_pass.md`, generated from `data:lab/gather_pass/sm_90a.json`.

`write(root, data, instance)` reads that record in the pinned data `data` and rewrites `docs/lab/gather_pass.md` under `root` from its first
`MEASURED (H100` to the end of the page (the page's last line is the `Record:` line), every figure the record's own.
`instance` is the rented box's label (`hs...` Hyperstack, digits vast.ai), which only words the card's line. It returns
the files it wrote (`lab page gather_pass --instance LABEL`, docs/internals/tools/lab.md).
"""

from __future__ import annotations

import json
import pathlib

from _record import Record


def write(root: pathlib.Path, data: pathlib.Path, instance: str) -> list[pathlib.Path]:
    inst = instance
    d = json.loads((data / "lab/gather_pass/sm_90a.json").read_text())
    R = Record({r["name"]: r["median"] for r in d["results"]})

    def g(n, f="{:,.0f}"):
        return f.format(R[n])

    C = ["4 to 5 live digits a token", "32 live digits a token"]
    FL = "a flat stream of its copies"
    MV = "its lines moved"
    FR = ", a fresh window a round"
    SM = ", its rounds the same windows"

    def nm(c, plane, t, mv, extra="", tail=FR):
        return f"{c}, {plane}, {t} threads, {mv}{extra}{tail}"

    BS = "4 to 5 live digits a token, the block stream's tables, values and sparse bytes, its lists made from the mask, 512 threads"
    o = []
    CARD = (
        ("H100 PCIe 80GB, 114 SMs, Hyperstack spot VM " + inst[2:])
        if inst.startswith("hs")
        else ("H100 80GB HBM3, 132 SMs, vast.ai instance " + inst)
    )
    o.append(
        f"MEASURED ({CARD}, driver {d['driver']}; {d['date'][:10]}, the experiment at\n{d['git']['sha'][:8]}; {d['reps']} repetitions; every arranged line and pad of every moved form, and every word, byte, line,\npad, step and plan of the block stream, exact on every multiprocessor; the median, SM-cycles a token for one\nside's sparse level; every figure below is this one run's):\n"
    )
    o.append("THE BLOCK STREAM, its lists the pass's own (4 to 5 live digits a token, 512 threads, a fresh window a round):\n")
    o.append("| every multiprocessor | a quarter |\n|---|---|")
    o.append(f"| {g(BS + FR)} | {g(BS + ', a quarter of the multiprocessors' + FR)} |\n")
    CF = "4 to 5 live digits a token, the block stream's coefficients in the step's format, every dense block's, its lists made from the mask, 512 threads"
    o.append(
        "THE COEFFICIENT RUNS in the step's format, every dense block's (4 to 5 live digits a token, 512 threads, a fresh window a round):\n"
    )
    o.append("| every multiprocessor | a quarter |\n|---|---|")
    SCV = BS + ", an entry's values multiplied by its sparse byte"
    C4 = CF + ", four slots a trip"
    W4 = CF + ", the dense digits widened alone, four slots a trip"
    o.append(f"| {g(CF + FR)} | {g(CF + ', a quarter of the multiprocessors' + FR)} |\n")
    o.append("THE SCALAR ON THE VALUES and the runs' units (every multiprocessor, a fresh window a round):\n")
    o.append("| the row | SM-cycles a token |\n|---|---|")
    o.append(f"| the shared part, an entry's values multiplied by its sparse byte | {g(SCV + FR)} |")
    o.append(f"| the coefficient runs, the product formed, four slots a trip | {g(C4 + FR)} |")
    o.append(f"| the coefficient runs, the dense digits widened alone, four slots a trip | {g(W4 + FR)} |")
    RG = ", its plane a ring of 164 KB a CTA"
    RC = ", its plane a ring of 256 KB a CTA"
    o.append(f"| the shared part into a ring of 164 KB a CTA | {g(BS + RG + FR)} |")
    o.append(
        f"| the coefficient runs, the product formed, four slots a trip, into a ring of 256 KB a CTA | {g(C4 + RC + FR)} |"
    )
    o.append(f"| those runs made and no copy out | {g(C4 + ', its runs made and no copy out' + FR)} |")
    o.append(f"| their copies out and no run made | {g(C4 + ', its copies out and no run made' + FR)} |\n")

    o.append("THE ARRANGED STEPS, by digit:\n")
    o.append(
        "| the cell | the plane | threads | a flat stream, a fresh window a round | its lines moved, a fresh window | moved, the same windows |\n|---|---|---|---|---|---|"
    )
    for c in C:
        for pl, pn in (("the values plane", "values"), ("the dense plane", "dense")):
            for t in (256, 512, 1024):
                o.append(
                    f"| {c} | {pn} | {t} | {g(nm(c, pl, t, FL))} | {g(nm(c, pl, t, MV))} | {g(nm(c, pl, t, MV, tail=SM))} |"
                )
    o.append("\nBOTH FORMS at 512 threads, a fresh window a round:\n")
    o.append(
        "| the cell | the plane | flat, every multiprocessor | flat, a quarter | moved, every multiprocessor | moved, a quarter |\n|---|---|---|---|---|---|"
    )
    Q = ", a quarter of the multiprocessors"
    for c in C:
        for pl, lab_, ex in (
            ("the values plane", "values plane", ""),
            ("the dense plane", "dense plane", ""),
            ("the dense plane", "dense plane, six buffers out", ", six buffers out"),
            ("the stream's values and sparse lines", "stream's values and sparse lines", ""),
            ("the stream's dense lines", "stream's dense lines", ""),
        ):
            o.append(
                f"| {c} | {lab_} | {g(nm(c, pl, 512, FL, ex))} | {g(nm(c, pl, 512, FL, ex + Q))} | {g(nm(c, pl, 512, MV, ex))} | {g(nm(c, pl, 512, MV, ex + Q))} |"
            )
    o.append("\nTHE STREAM ON A FEW MULTIPROCESSORS, 512 threads, a fresh window a round:\n")
    o.append(
        "| the cell | the plane | CTAs | flat, a ring | flat, a new region a round | moved, a ring | moved, a new region a round |\n|---|---|---|---|---|---|---|"
    )
    for c in C:
        for pl in ("the stream's values and sparse lines", "the stream's dense lines"):
            for n in (16, 8):
                ring = f", {n} multiprocessors, its plane a ring"
                new = f", {n} multiprocessors, a new region a round"
                f1 = "{:,.1f}"
                o.append(
                    f"| {c} | {pl[4:]} | {n} | {g(nm(c, pl, 512, FL, ring), f1)} | {g(nm(c, pl, 512, FL, new), f1)} | {g(nm(c, pl, 512, MV, ring), f1)} | {g(nm(c, pl, 512, MV, new), f1)} |"
                )
    SV, SD = "the stream's values and sparse lines", "the stream's dense lines"

    def st(c, mv=MV, ex=""):
        return R[nm(c, SV, 512, mv, ex)] + R[nm(c, SD, 512, mv, ex)]

    def ar(c):
        return R[nm(c, "the values plane", 512, MV)] + R[nm(c, "the dense plane", 512, MV)]

    def few(n, mv):
        return (
            R[nm(C[0], SV, 512, mv, f", {n} multiprocessors, its plane a ring")]
            + R[nm(C[0], SD, 512, mv, f", {n} multiprocessors, its plane a ring")]
        )

    bs = R[BS + FR]
    tv = R[nm(C[0], "the values plane", 512, MV)]
    rv = R[nm(C[0], SV, 512, MV)]
    o.append(f"""
WHAT IT SHOWS.
- MAKING ITS OWN LISTS, THE BLOCK STREAM'S SHARED PART COSTS {bs:,.0f} SM-CYCLES A TOKEN at 4 to 5 live digits a token
  with every multiprocessor at work ({g(BS + Q + FR)} on a quarter of them): the mask turned around in the kernel, the tables,
  the value runs and the sparse bytes, one copy out a slot. The arranged values plane, its lists the host's, is
  {tv:,.0f} -- the same entries' lines moved, a digit's steps padded: THE PASS'S OWN LISTS AND TABLES ARE {bs - tv:+,.0f} ON IT
  (the token-ordered stream's values and sparse lines, {rv:,.0f}, move a token's lines once a sparse block, not
  once a digit). It falls little with the card unloaded: its own CTA binds it. NOT IN IT: the dense lines (the arranged dense plane is
  {g(nm(C[0], "the dense plane", 512, MV))}, the stream's {g(nm(C[0], SD, 512, MV))}); a cell past its bounds -- a list over 32 entries, a slot over 150;
  32 live digits a token.
- THE COEFFICIENTS IN THE STEP'S FORMAT COST {R[CF + FR]:,.0f} SM-CYCLES A TOKEN, every dense block's ({g(CF + Q + FR)} on a quarter of
  the multiprocessors: the kernel's own work binds it), where the arranged dense plane -- the same entries' bytes
  moved, no product -- is {g(nm(C[0], "the dense plane", 512, MV))}: THE BLOCK STREAM'S WRITE SIDE IS {bs + R[CF + FR]:,.0f} with its shared part, more than
  the carry's window a token. A unit here is one slot's runs for one dense block, 512 trips a window, each with
  its fence, its barrier and its copy out; the dense plane's unit is 192 steps. NOT ISOLATED: the trips from the
  products (the rows below part them). NOT IN IT: the scales; the steps' operands' fields; the read side.
- THE ENTRY'S SCALAR ON ITS VALUES COSTS {R[SCV + FR] - bs:+,.0f}: the shared part is {g(SCV + FR)} with an entry's 64 values multiplied
  by its sparse byte, {bs:,.0f} without. THE COEFFICIENT RUNS ARE NEITHER THEIR PRODUCT NOR THEIR TRIPS: four slots a
  trip are {g(C4 + FR)} where one is {g(CF + FR)}, and with no product at all -- the dense digits widened alone
  -- {g(W4 + FR)}. THE WRITE SIDE, the product formed, IS {bs + R[C4 + FR]:,.0f} SM-CYCLES A TOKEN. NOT ISOLATED: what the runs' kernel is
  bound by -- its lines are twice the arranged dense plane's bytes, read through a list's entries.
- A RING THE CACHE HOLDS DOES NOT MOVE THE PASS, every multiprocessor at work: the shared part is {g(BS + RG + FR)} against
  {bs:,.0f} and the coefficient runs {g(C4 + RC + FR)} against {g(C4 + FR)} (that row unstable in this record). The pass is not
  bound by the bytes it sends to the card's memory.
- THE RUNS' KERNEL, PARTED: its runs made and no copy out {g(C4 + ", its runs made and no copy out" + FR)}, its copies out and no run made
  {g(C4 + ", its copies out and no run made" + FR)}, the whole {g(C4 + FR)}.
- THE TOKEN-ORDERED STREAM COSTS {st(C[0]):,.0f} SM-CYCLES A TOKEN a side at 4 to 5 live digits a token with every multiprocessor at work
  ({rv:,.0f} its values and sparse lines, {g(nm(C[0], SD, 512, MV))} its dense lines) and {st(C[1]):,.0f} at 32; the arranged steps {ar(C[0]):,.0f} and {ar(C[1]):,.0f}.
  At 32 live digits it is {st(C[1]) / ar(C[1]):.2f} of the arranged steps: a token is in at most 16 sparse blocks however many of
  its digits are live.
- WITH THE CARD UNLOADED IT IS {few(16, MV):,.0f}, AND THERE IT IS ITS OWN CTA THAT BINDS. On 16 or on 8 multiprocessors the stream is
  {few(16, MV):,.0f} and {few(8, MV):,.0f} at the sparse cell ({st(C[0], MV, Q):,.0f} on a quarter of them), its flat form {few(16, FL):,.0f}: the cost stops falling with
  the load, and the rest of it is the moving of lines.
- A RING THE CACHE HOLDS DOES NOT MOVE THE PASS'S WRITES ON AN UNLOADED CARD, at the sparse cell: there 16 CTAs'
  rings fit the cache and their new regions do not, and the moved forms cost the same (the table's sparse rows).
  NOT ASKED: a ring on a loaded card, where the card's memory binds; a reader of the ring; a discard.
- SIX BUFFERS OUT AGAINST THREE (the arranged dense plane): {g(nm(C[0], "the dense plane", 512, MV, ", six buffers out"))} against {g(nm(C[0], "the dense plane", 512, MV))} moved at the sparse cell,
  {g(nm(C[1], "the dense plane", 512, MV, ", six buffers out"))} against {g(nm(C[1], "the dense plane", 512, MV))} at 32 live digits.
- WHAT IT DOES NOT SAY: the block stream's read side -- no values -- is not built; its head's operands' fields are not written; the
  token-ordered stream's and the arranged steps' lists are the host's; a dense level and a clustered one are
  unmeasured; the pass beside a running carry is unmeasured (the carry filling from a stream the host laid is
  [`carry_window`](carry_window.md)'s); the same kernels on another rented card are not compared.

Record: `data:lab/gather_pass/sm_90a.json`.
""")
    p = root / "docs/lab/gather_pass.md"
    s = p.read_text()
    i = s.index("MEASURED (H100")
    p.write_text(s[:i] + "\n".join(o).lstrip("\n"))
    return [p]
