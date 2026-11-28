# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The measured section of `docs/lab/decode_ring.md`, generated from the newest sample of `decode_ring`'s counters record.

`write(root, data, instance)` reads that record in the pinned data `data` (`-- ncu --metric dram__bytes_read.sum --metric
dram__bytes_write.sum`: each kernel's median over its launches in one repetition, a launch a row) and rewrites
`docs/lab/decode_ring.md` under `root` from its `MEASURED` line to its end (where the page has none, it appends the
section after the page's text), every figure the record's own and every comparative word chosen by the figures it
compares. It REFUSES, naming what is missing, where the record is absent or lacks a row's kernel. `instance` is the
rented box's label (`nb...` Nebius, `hs...` Hyperstack, digits vast.ai), which only words the card's line. The timed
rows are owed: their record is the experiment's next recorded run. It returns the files it wrote (`lab page
decode_ring --instance LABEL`, docs/internals/tools/lab.md).
"""

from __future__ import annotations

import pathlib
import re

from _record import Record, card, read

from fsla.ops.paging import page_layout

#: a launch's grid: a CTA an SM of the record's card (the H100 SXM's 132)
SMS = 132
FORMS = ("REG", "BULK", "PIECES", "PRODUCER")
PATTERNS = ("WRITE_ALL", "READ_ALL", "WRITE_ONE", "READ_ONE", "RECORD")
#: A ROW'S DECLARED WORK, the experiment's own constants (measure/lab/experiments/decode_ring.cu, `Work<PAT, DV>` and
#: `Geo<DV>`), stated here as that source states them: a CTA's positions a launch by pattern (`kPositions`), and a
#: RECORD's stride by `d_v` (`kRecord`: the hi line, the exponents and the mass, padded to whole sectors)
POSITIONS = (1024, 2048, 2048, 3072, 3072)
RECORD = {64: 160, 32: 96}
#: the rows' configurations, (d_v, G, stamped), and the forms each pattern runs in: REG holds its ALL patterns at G 1
#: alone, RECORD runs in REG and PIECES
CONFIGS = ((64, 1, 0), (64, 8, 0), (32, 8, 0), (64, 8, 1))
#: the prediction's band: every row's DRAM bytes within this share of the bytes its pattern declares
BAND = 0.10
#: the grain the scattered rows' reads are priced at: a miss moves 64-byte blocks (the page's section THE PREDICTION'S
#: PRICE NAMED, from the counters record this page reads)
BLOCK = 64


def forms(pat: int, g: int, stamped: int = 0) -> tuple[int, ...]:
    """The forms a pattern's rows run in: RECORD in REG and PIECES; the ALL patterns without REG past G 1; PRODUCER's
    rows unstamped alone (its one warp's stamps overflow its timeline row)."""
    if pat == 4:
        return (0, 2)
    every = (1, 2, 3) if pat <= 1 and g != 1 else (0, 1, 2, 3)
    return tuple(f for f in every if not (stamped and f == 3))


def declared(pat: int, dv: int) -> tuple[int, int, int]:
    """A position's declared bytes read and written and its leaves: the experiment's `Work<PAT, DV>::kRead` (its 32-byte
    sectors) on the state's page (`fsla.ops.paging.page_layout`, its tail the exponents and masses), `Geo<DV>`'s atoms a
    ONE position (a lane group's eight channels a lane)."""
    page = page_layout(dv)
    full, tail = page.bytes, page.bytes - page.exp
    atoms = 32 // (dv // 8)
    rd = (full, 32 * dv + tail, atoms * (3 * dv + 64), atoms * (2 * dv + 64), atoms * RECORD[dv])[pat]
    return rd, rd if pat in (0, 2) else 0, 16 if pat <= 1 else atoms


def blocks(pat: int, dv: int) -> int:
    """A scattered leaf's bytes read at the 64-byte block's grain: on the ruled page its hi line (and lo line where
    written), a block of exponents and one of masses; a record, the blocks a `kRecord`-byte stride's record spans."""
    if pat == 4:
        return BLOCK * max(-(-(o + RECORD[dv]) // BLOCK) - o // BLOCK for o in range(0, BLOCK, 32))
    lines = 2 * dv + (dv if pat == 2 else 0)
    return -(-lines // BLOCK) * BLOCK + 2 * BLOCK


def write(root: pathlib.Path, data: pathlib.Path, instance: str) -> list[pathlib.Path]:
    rec = read(data, "decode_ring", kind="counters")
    d = rec.body()
    K = Record(d["kernels"])
    rows = []
    for dv, g, stamped in CONFIGS:
        for pat in range(5):
            for form in forms(pat, g, stamped):
                m = K[f"void walk<{form}, {pat}, {dv}, {g}, {stamped}>(Args)"]
                rd, wr, leaves = declared(pat, dv)
                n = SMS * POSITIONS[pat]
                rows.append(dict(dv=dv, g=g, stamped=stamped, pat=pat, form=form,
                                 read=m["dram__bytes_read.sum"] / (n * rd),
                                 written=m["dram__bytes_write.sum"] / (n * wr) if wr else None,
                                 extra=m["dram__bytes_write.sum"] / 1e6 if not wr else None,
                                 leaf=m["dram__bytes_read.sum"] / (n * leaves), declared_leaf=rd / leaves))

    def cfg(r):
        return f"{r['dv']}, {r['g']}" + (", stamped" if r["stamped"] else "")

    def wcol(r):
        return f"{r['written']:.2f}" if r["written"] is not None else f"({r['extra']:,.0f} MB)"

    table = "\n".join(
        ["| pattern | form | `d_v`, G | read, of declared | written, of declared | read a leaf, bytes (declared) |",
         "|---|---|---|---:|---:|---:|"]
        + [f"| {PATTERNS[r['pat']]} | {FORMS[r['form']]} | {cfg(r)} | {r['read']:.2f} | {wcol(r)} | "
           f"{r['leaf']:,.0f} ({r['declared_leaf']:,.0f}) |" for r in rows]
    )

    def whole_line(dv: int) -> str:
        sel = [r for r in rows if r["pat"] <= 1 and r["dv"] == dv]
        spread = max(max(abs(r["read"] - 1), abs((r["written"] or 1) - 1)) for r in sel)
        worst = max(sel, key=lambda r: max(abs(r["read"] - 1), abs((r["written"] or 1) - 1)))
        word = "HELD" if spread <= BAND else "MISSED"
        return (f"- THE WHOLE ATOMS (WRITE_ALL, READ_ALL), `d_v` {dv}: every row within {100 * spread:.1f}% of its "
                f"declared bytes, read and written (the widest {PATTERNS[worst['pat']]} by {FORMS[worst['form']]}, "
                f"G {worst['g']}): {word}.")

    def leaf_line(pat: int, dv: int) -> str:
        sel = [r for r in rows if r["pat"] == pat and r["dv"] == dv and not r["stamped"]]
        lo, hi = min(r["leaf"] for r in sel), max(r["leaf"] for r in sel)
        rng = f"{lo:,.0f}" if hi - lo < 4 else f"{lo:,.0f} to {hi:,.0f}"
        dec = sel[0]["declared_leaf"]
        times = f"{lo / dec:.2f}x" if hi / dec - lo / dec < 0.005 else f"{lo / dec:.2f}x to {hi / dec:.2f}x"
        word = "HELD" if max(abs(r["read"] - 1) for r in sel) <= BAND else "MISSED"
        blk = blocks(pat, dv)
        gap = max(abs(r["leaf"] - blk) for r in sel)
        at = (f"the measured within {gap:,.0f} of it" if gap < 4 else
              f"the measured {lo - blk:,.0f} to {hi - blk:,.0f} bytes over it")
        return (f"- {PATTERNS[pat]}, `d_v` {dv}: {rng} bytes read a leaf against the {dec:,.0f} declared, "
                f"{times}: {word}. At the 64-byte block's grain a leaf reads {blk:,.0f}, {at}.")

    def form_leaf(pat: int, dv: int, form: int) -> float:
        return next(r["leaf"] for r in rows if (r["pat"], r["dv"], r["g"], r["stamped"], r["form"]) == (pat, dv, 8 if dv == 32 else 1, 0, form))

    ones = "\n".join(leaf_line(p, dv) for dv in (64, 32) for p in (3, 4))
    reg, pieces = form_leaf(2, 64, 0), form_leaf(2, 64, 2)
    reg32, pieces32 = form_leaf(2, 32, 0), form_leaf(2, 32, 2)
    stamped = [r for r in rows if r["stamped"]]
    twin_gap = max(abs(r["read"] - next(t["read"] for t in rows if not t["stamped"] and (t["dv"], t["g"], t["pat"], t["form"])
                                        == (r["dv"], r["g"], r["pat"], r["form"]))) for r in stamped)
    record_gap = max(abs(r["leaf"] - blocks(4, r["dv"])) for r in rows if r["pat"] == 4)
    text = f"""MEASURED, THE SECTORS ({card(d['arch'], instance)}; {d['date'][:10]}, the experiment at {d['git']['sha'][:8]};
`-- ncu --metric dram__bytes_read.sum --metric dram__bytes_write.sum`, each kernel's median over its launches in one
repetition, a launch {SMS} CTAs). A row's DRAM bytes against the bytes its pattern declares (its 32-byte sectors); a
read-only row's written bytes, which it declares none of, in MB (the sums, and a stamped row's timeline):

{table}

WHAT IT SHOWS.
{whole_line(64)}
{whole_line(32)}
{ones}
- WRITE_ONE reads more than it declares in every form, and the rings more than REG: {pieces:,.0f} bytes a leaf by
  PIECES against REG's {reg:,.0f} at `d_v` 64, G 1 ({pieces32:,.0f} against {reg32:,.0f} at `d_v` 32, G 8); its written
  bytes are its declared ones within {100 * max(abs(r['written'] - 1) for r in rows if r['pat'] == 2 and r['dv'] == 64):.0f}% at `d_v` 64.
- A stamped row reads within {100 * twin_gap:.1f}% of its twin's bytes.

THE PREDICTION'S PRICE NAMED. The L2 does not fetch a scattered miss at the sector's 32 bytes: the record's leaves read
the 64-byte blocks their records span, within {record_gap:.1f} bytes, and the ruled page's scattered leaf reads its blocks and a residual
beside them. Every scattered pattern's floor is priced on 64-byte blocks from here; the timed rows' bands that rest on
the 32-byte grain (every ONE pattern and RECORD) are re-based on it with the next recorded run.

Record: `{rec.cited}`.
"""
    p = root / "docs/lab/decode_ring.md"
    s = p.read_text()
    m = re.search(r"^MEASURED\b", s, re.M)
    i = m.start() if m else len(s.rstrip("\n")) + 2
    p.write_text(s[:i].rstrip("\n") + "\n\n" + text)
    return [p]
