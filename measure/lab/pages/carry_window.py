# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The measured section of `docs/lab/carry_window.md`, generated from `measure/lab/results/carry_window/sm_90a.json`.

`write(root, instance)` reads that record under `root` and rewrites, in the docs under `root`:
- `docs/lab/carry_window.md`: everything from its first `MEASURED (H100 80GB` up to the line starting
  ``Record: `measure/lab/results``, every figure the record's own;
- `docs/ARCHITECTURE.md`: the five figures it restates of this record, each substituted where its pattern matches once;
- `docs/lab/intra_window.md`: the carry's figure it quotes.
`instance` is the rented box's label (`nb...` Nebius, `hs...` Hyperstack, digits vast.ai), which only words the card's
line. It returns the files it wrote (`lab page carry_window --instance LABEL`, docs/internals/tools/lab.md).

THE LAUNCH'S EDGES, a section at the end of that span, is generated from the one-CTA record
`sm_90a.blocks1.json` and its trace `sm_90a.blocks1.trace.json`: for each row of the record whose name carries
`, the launch's two edges stamped`, the plain row's window, the edges row's and the coarse row's (the name with the
suffix removed, and with it replaced by `, every warp's phases stamped`: rows are matched by name, never by position),
and the edges row's and the coarse row's launch from the trace (`tools/lab_trace.py`'s `Cta.launch`).
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

from _record import Record

#: two windows within this share of each other (of the lesser) are said to agree
SAME = 0.005


def edges_section(root: pathlib.Path) -> str:
    """THE LAUNCH'S EDGES, from the one-CTA record and its trace; empty where the record holds no edges row."""
    sys.path.insert(0, str(root / "tools"))
    import lab_trace

    folder = root / "measure/lab/results/carry_window"
    rec = json.loads((folder / "sm_90a.blocks1.json").read_text())
    windows = {r["name"]: r["median"] for r in rec["results"] if r.get("unit") == "cycles a window a CTA"}
    traces = {r["variant"]: r for r in lab_trace.load(folder / "sm_90a.blocks1.trace.json")}

    def launch(name):
        row = traces[name]
        cta = lab_trace.Cta(row, row["ctas"][0])
        out = cta.launch()
        assert out is not None, f"lab page: the trace's row {name!r} holds no launch"
        return out, row["ctas"][0]["cta"]

    def against(a, b):
        share = (a - b) / b
        return share, "the same as" if abs(share) <= SAME else ("over" if share > 0 else "under")

    parts = []
    for edged in (n for n in windows if lab_trace.EDGES_SUFFIX in n):
        plain, coarse = lab_trace.untraced_name(edged), edged.replace(lab_trace.EDGES_SUFFIX, lab_trace.COARSE_SUFFIX)
        for n in (plain, coarse):
            assert n in windows, f"lab page: the edges row {edged!r} has no twin {n!r} in the record"
        (le, ce), (lc, _) = launch(edged), launch(coarse)
        se, we = against(windows[edged], windows[plain])
        sc, wc = against(windows[coarse], windows[plain])

        def cost(share, word):
            return "the same as the plain row's" if word == "the same as" else f"{abs(share) * 100:.1f}% {word} the plain row's"

        parts.append(
            f"""THE LAUNCH'S EDGES (one CTA, `sm_90a.blocks1.json` and `sm_90a.blocks1.trace.json`, the experiment at
{rec["git"]["sha"][:8]}{" (a dirty tree)" if rec["git"].get("dirty") else ""}, {rec["reps"]} repetitions; the row whose name with `{lab_trace.EDGES_SUFFIX}` exists, its plain
and coarse twins by its name with that suffix removed and replaced): each warp marks its entry at the kernel's top and its exit at its role's end, on both clocks.
- THE WINDOW, cycles a window a CTA: the plain row's {windows[plain]:,.0f}; the edges row's {windows[edged]:,.0f}, {cost(se, we)}
  ({windows[edged] - windows[plain]:+,.0f} cycles); the coarse row's (its phases' starts and the edges) {windows[coarse]:,.0f}, {cost(sc, wc)}
  ({windows[coarse] - windows[plain]:+,.0f} cycles).
- THE LAUNCH, CTA {ce}'s, from the trace (the first warp's entry to the last warp's exit): the edges row's
  {le["cycles"]:,.0f} cycles, {le["ns"]:,.0f} ns, {le["rate"]:,.1f} cycles a microsecond; the coarse row's {lc["cycles"]:,.0f} cycles,
  {lc["ns"]:,.0f} ns, {lc["rate"]:,.1f} cycles a microsecond. The warps enter within {le["entry_spread"]:,.0f} (the coarse row's: {lc["entry_spread"]:,.0f}) cycles
  and exit within {le["exit_spread"]:,.0f} ({lc["exit_spread"]:,.0f}).
"""
        )
    return "\n".join(parts)


def write(root: pathlib.Path, instance: str) -> list[pathlib.Path]:
    j = json.loads((root / "measure/lab/results/carry_window/sm_90a.json").read_text())
    R = Record({r["name"]: r["median"] for r in j["results"]})
    # a role's steps as the page reads them: their issue and their wait together (the record holds the two apart),
    # a warpgroup's first warp's the same; the wait alone is kept under the site's name with " (wait)"
    _FW = ", its warpgroup's first warp"
    for _k in list(R):
        for _site, _issue in (
            (": a readout warp in its steps", ": a readout warp issuing its steps and its next bytes' loads"),
            (": a fold warp in its steps and issuing its copies", ": a fold warp issuing its steps, to their commit"),
        ):
            for _tail in ("", _FW):
                if _k.endswith(_site + _tail) and _k[: -len(_site + _tail)] + _issue + _tail in R:
                    R[_k + " (wait)"] = R[_k]
                    R[_k] = R[_k] + R[_k[: -len(_site + _tail)] + _issue + _tail]
    # the card's multiprocessors, by the record's device: a counter a CTA a window is over them
    _SMS = 114 if "PCIe" in str(j.get("device")) else 132

    def g(n):
        return f"{R[n]:,.0f}"

    # a kernel's counters of the record, a CTA a window: its first launch is the checked one, `kOutWindows` = 4 windows a CTA
    _CN = j.get("counters") or {}

    def cw(roles, key):
        ks = [k for k in _CN if str(roles) in k]
        assert len(ks) == 1, (roles, ks)
        v = _CN[ks[0]]
        return v["stalls"][key[1:]] if key.startswith("%") else v[key] / (_SMS * 4.0)

    def q(side, has, no=None, yes=None, wait=False):
        pre = "the fold's side alone" if side == "fold" else "the carry: both sides"
        c = [
            k
            for k in R
            if k.startswith(pre)
            and "coefficients' bytes" in k
            and has in k
            and (no is None or no not in k)
            and (yes is None or yes in k)
            and (k.endswith("waits on its producer") == wait)
        ]
        c = [
            k
            for k in c
            if (("four slots" in k) == ("four slots" in has))
            and (("packed by digit" in k) == ("four slots" in has or "packed" in has))
            and ("each box its own" not in k)
            and ("never read before" not in k)
            and ("expanding their own slot" not in k)
            and ("a block a list" not in k)
        ]
        assert len(c) == 1, (side, has, no, yes, wait, c)
        return f"{R[c[0]]:,.0f}"

    def t(n):
        return f"{R[n] / 8:,.0f}"

    F = "the fold's side alone: producers and steps"
    D = "the read side alone: the image, the readout and the drain"
    B = "the carry: both sides"
    SELF = ", the readout draining its own rows and four producers"
    LEAN = ", the readout's A in two sets and two batches a producer"
    FIVE = ", five warpgroups: four producers and four drain warps"
    SLEEP = ", the long waits asleep at named barriers"
    NL = ", the producers' global loads taken out"
    NA = ", the drains' global adds taken out"
    NI = ", the image's stores taken out"
    NF = ", no fold warp running"
    SUS = ", every wait suspended"
    WIDE = ", the readout's bytes one wide load a warp"
    NX = ", the fold's fence behind its image taken out"
    PARK = ", the fold warps parked at a named barrier"
    SPIN = ", the fold warps spinning at a polled barrier"
    DUMP = ", the fold's state stored out in every call"
    SL = ", every wait suspended, the readout's A in two sets and two batches a producer"
    SW = ", every wait suspended, the readout's bytes one wide load a warp"
    SLW = ", every wait suspended, the readout's A in two sets and two batches a producer, the readout's bytes one wide load a warp"
    LSW = ", the readout's A in two sets and two batches a producer, the long waits asleep at named barriers, the readout's bytes one wide load a warp"
    SUS = ", every wait suspended"
    BIG = ", the tables 16 replicas the cache cannot hold"
    WALL = ": the long call by wall time, scaled to 65,536 tokens"
    NM = ", the producers' coefficient arithmetic taken out"
    NB = ", the readout's byte loads taken out"
    WD = ": a readout warp waits on the drain"
    DW = ": a drain warp waits on the panel"
    WP = ": a fold warp waits on its producer"
    WR = ": a fold warp waits on the readout's read"
    ST = ": a readout warp in its steps"
    FA = ": a readout warp forming its steps' A"
    DR = ": a readout warp draining its own rows"
    DEN = ", the den: two more steps a slot, the readout's steps 72 columns and an fp32 a token"
    AL = ", the sides' carves equal: an image tile a list, its lines four threads' words"
    NC = ", no clock inside a role"
    FE = ", the fold warps expanding their own slot"
    FI = FE + " and issuing its copies"
    D4 = ", four drain warps"
    TS = ", the readout a sum a box"
    MR = ", the readout's coefficients in shared memory and its tokens on the step's columns"
    C8 = ", its chunk 8 digits"
    RP = ", its sums rounded by the drain warps"
    BK = ", the fold from a block a list, its lists uneven and its steps one block"
    P1 = NC + ", the den's parts alone: its fold steps"
    P2 = P1 + ", its image brick"
    P3 = P2 + ", its register split"
    P4 = P3 + ", its 72-column steps"
    TV = ", the values tile-major in global memory"
    LEAN1 = ", every wait suspended, the readout's A in two sets"
    AH = ", a window's rows asked for a window ahead"
    EX = ", the values landed from a column's expanded lists"
    WH = ", the coefficients' bytes landed with them"
    JN = ", its two slots laid before one rendezvous and its 32 steps one block"
    CST = ", the readout's sums to the panel by stmatrix"
    CAH = ", the steps' operands loaded two steps ahead"
    CNR = ", the readout's steps taken out"
    CSV = ", the drain warps expanding the fold's slots a slot ahead"
    CSH = ", the steps' operands one load a slot and shuffled"
    CSK = ", only the steps a list reaches"
    CRDY = ", its slots landed with their operands made, the readout's coefficient tiles landed made"
    TKS = ", its slots laid by the fold from views in token order landed whole"
    TKT = ", its slots laid by the fold from views in token order, their value lines through registers"
    ONV = ", its window laid by the fold in one pass from one view in token order through registers, its 32 steps one block"
    LS = ", the fold's entries from lists, a window's own tokens"
    LH = ", the fold's entries from lists, every window the same tokens"
    P155 = NC + ", the image's tiles 155 lines apart"
    P154 = ", the image's tiles 154 lines apart"
    P161 = ", the image's tiles 161 lines apart"
    rows = [
        (F, F),
        (D, D),
        (B, B),
        ("both sides, the producers' global loads taken out", B + NL),
        ("both sides, the drains' global adds taken out", B + NA),
        ("both sides, the image's stores taken out", B + NI),
        ("the read side alone, no fold warp running", D + NF),
        ("the read side alone, every wait suspended", D + SUS),
        ("both sides, every wait suspended", B + SUS),
        ("THE READOUT DRAINING ITS OWN ROWS, FOUR PRODUCERS: the fold's side alone", F + SELF),
        ("the same: the read side alone", D + SELF),
        ("the same: both sides", B + SELF),
        ("THE READOUT LEAN, TWO BATCHES A PRODUCER: the fold's side alone", F + LEAN),
        ("the same: the read side alone", D + LEAN),
        ("the same: both sides", B + LEAN),
        ("FIVE WARPGROUPS, FOUR PRODUCERS AND FOUR DRAIN WARPS AT 40: the fold's side alone", F + FIVE),
        ("the same: the read side alone", D + FIVE),
        ("the same: both sides", B + FIVE),
        ("THE LONG WAITS ASLEEP AT NAMED BARRIERS: the fold's side alone", F + SLEEP),
        ("the same: the read side alone", D + SLEEP),
        ("the same: both sides", B + SLEEP),
        ("the fold's side alone, the producers' global loads taken out", F + NL),
        ("the fold's side alone, their coefficient arithmetic taken out", F + NM),
        ("the fold's side alone, both taken out", F + NL + NM),
        ("the read side alone, the readout's byte loads taken out", D + NB),
        ("the read side alone, no fold warp running, the byte loads taken out", D + NF + NB),
        ("both sides, the readout's byte loads taken out", B + NB),
        ("both sides, the drains' adds and the readout's byte loads taken out", B + NA + NB),
        ("THE READOUT'S BYTES ONE WIDE LOAD A WARP: the read side alone", D + WIDE),
        ("the same: the read side alone, no fold warp running", D + NF + WIDE),
        ("the same: both sides", B + WIDE),
        ("the same: both sides, the long waits asleep", B + SLEEP + WIDE),
        ("the read side alone, the image's stores taken out", D + NI),
        ("the read side alone, the fold's fence behind its image taken out", D + NX),
        ("the read side alone, both taken out", D + NI + NX),
        ("both sides, that fence taken out", B + NX),
        ("the read side alone, no fold warp running, the fold warps parked asleep", D + NF + PARK),
        ("the same, spinning", D + NF + SPIN),
        ("THE DUMP RESTORED: both sides, the fold's state stored out in every call", B + DUMP),
        ("THE WINNERS COMBINED: both sides, suspended waits and the lean readout", B + SL),
        ("the same, suspended waits and the wide byte load", B + SW),
        ("the same, suspended waits, the lean readout and the wide byte load (the fourth warpgroup at 104)", B + SLW),
        ("the same, sleeping waits, the lean readout and the wide byte load", B + LSW),
        ("THE TABLES ONCE-READ, 16 replicas the cache cannot hold: both sides", B + BIG),
        ("the same, the winner's form", B + SL + BIG),
        ("the same: the fold's side alone", F + BIG),
        ("THE DEN, folded, published, read and drained: the fold's side alone", F + DEN),
        ("the same: both sides, the winner's form", B + SL + DEN),
        ("the same, the tables once-read", B + SL + BIG + DEN),
        ("THE IMAGE'S TILES MOVED APART, nothing else changed: the winner's form at 154 lines (137 as it is)", B + SL + P154),
        ("the winner's form with the den at 161 lines (155 as it is)", B + SL + P161 + DEN),
        ("THE ALIGNED CELL, an image tile a list: the winner's form", B + SL + AL),
        ("the same with the den", B + SL + AL + DEN),
        ("NO CLOCK INSIDE A ROLE: the winner's form", B + SL + NC),
        ("the same with the den", B + SL + NC + DEN),
        ("the same at the aligned cell", B + SL + AL + NC),
        ("the same at the aligned cell with the den", B + SL + AL + NC + DEN),
        ("THE DEN'S PARTS, a chain with no clock inside a role: the fold's steps", B + SL + P1),
        ("then the image's ninth brick and its store", B + SL + P2),
        ("then the register split", B + SL + P3),
        ("then the 72-column steps", B + SL + P4),
        ("the winner's form with its tiles at the den's pitch, 155 lines", B + SL + P155),
        ("THE VALUES TILE-MAJOR in global memory, no clock inside a role: the winner's form", B + SL + NC + TV),
        ("the tables once-read, the values token-major", B + SL + BIG + NC),
        ("the tables once-read, the values tile-major", B + SL + BIG + NC + TV),
        ("THE FOLD'S ENTRIES FROM LISTS, a window's own tokens: the fold's side alone", F + LS),
        ("the same: both sides, no clock inside a role", B + LEAN1 + NC + LS),
        ("the same with every window the same tokens: the fold's side alone", F + LH),
        ("the same: both sides", B + LEAN1 + NC + LH),
        ("a window's own tokens, their rows asked for a window ahead: both sides", B + LEAN1 + NC + AH + LS),
        ("THE VALUES LANDED BY BULK COPY from expanded lists, a window's own tokens: the fold's side alone", F + EX + LS),
        ("the same: both sides", B + LEAN1 + NC + EX + LS),
        ("THE WHOLE ENTRY LANDED, values and coefficient bytes: the fold's side alone", F + EX + WH + LS),
        ("the same: both sides", B + LEAN1 + NC + EX + WH + LS),
        (
            "the whole entry, EACH BOX ITS OWN expanded windows, past the cache: the fold's side alone",
            F + EX + WH + ", each box its own expanded windows" + LS,
        ),
        ("the same: both sides", B + LEAN1 + NC + EX + WH + ", each box its own expanded windows" + LS),
        (
            "the whole entry, its 16 columns' windows NEVER READ BEFORE, 16 boxes a window's readers: the fold's side alone",
            F + EX + WH + ", a column's windows never read before" + LS,
        ),
        ("the same: both sides", B + LEAN1 + NC + EX + WH + ", a column's windows never read before" + LS),
        (
            "THE PACKED STREAM, packed by digit, its lists uneven (8 to 24 entries), last steps truncated: the fold's side alone",
            F + EX + WH + ", a column's stream packed by digit, its lists uneven and their last steps truncated" + LS,
        ),
        (
            "the same: both sides",
            B
            + LEAN1
            + NC
            + EX
            + WH
            + ", a column's stream packed by digit, its lists uneven and their last steps truncated"
            + LS,
        ),
        (
            "the packed stream, FOUR SLOTS OF 4 TILES a box: the fold's side alone",
            F + EX + WH + ", a column's stream packed by digit, its lists uneven and their last steps truncated"
            ", four slots of 4 tiles a box" + LS,
        ),
        (
            "the same: both sides",
            B + LEAN1 + NC + EX + WH + ", a column's stream packed by digit, its lists uneven and their last steps truncated"
            ", four slots of 4 tiles a box" + LS,
        ),
        (
            "the whole entry, THE FOLD WARPS EXPANDING their own slot: the fold's side alone",
            F + EX + WH + ", the fold warps expanding their own slot" + LS,
        ),
        ("the same: both sides", B + LEAN1 + NC + EX + WH + ", the fold warps expanding their own slot" + LS),
    ]
    tab = "| the row | cycles a window a CTA | SM-cycles a token |\n|---|---|---|\n" + "".join(
        f"| {a} | {g(n)} | {t(n)} |\n" for a, n in rows
    )
    sites = [
        ("a fold warp waits on the readout's read of a tile group", "a fold warp waits on the readout's read"),
        ("a fold warp waits on its producer",) * 2,
        ("a readout warp waits on a tile group's publish", "a readout warp waits on the image"),
        ("a readout warp waits on the panel's drain", "a readout warp waits on the drain"),
        ("a readout warp draining its own rows",) * 2,
        ("a readout warp forming its steps' A",) * 2,
        ("a readout warp in its steps (a lean readout's hold the forming)", "a readout warp in its steps"),
        ("a drain warp waits on the panel",) * 2,
    ]
    cols = [B, D, D + NB, D + NF, D + NF + NB, B + SELF, B + FIVE]

    def cell(c, s):
        v = R.get(c + ": " + s)
        return "" if v is None else f"{v:,.0f}"

    tab2 = (
        "| the role's count | both sides | the read side alone | alone, byte loads out | alone, no fold warp | alone, no fold warp, byte loads out | both, readout draining | both, five warpgroups |\n|---|---|---|---|---|---|---|---|\n"
        + "".join("| " + a + " | " + " | ".join(cell(c, s) for c in cols) + " |\n" for a, s in sites)
    )
    _BLK = sum(1 << i for i in (0, 1, 5, 8, 22, 27, 28, 30, 33, 35, 36, 38))
    _RDY = _BLK + (1 << 43)
    _RR = _RDY + (1 << 44)

    def cn(roles, key):
        return f"{cw(roles, key):,.0f}"

    COUNTERS = (
        f"""- THE COUNTERS (the profiler's, each kernel's checked launch of 4 windows a CTA, which also stores the fold's state
  out -- 128 stores a thread, each over 32 lines, which the timed windows leave out; a CTA a window, its 16 warps
  together) OF THE BLOCK FORM, THE FORM WITH THE FOLD'S OPERANDS MADE, AND WITH BOTH SIDES': {cn(_BLK, "inst")}, {cn(_RDY, "inst")} and
  {cn(_RR, "inst")} warp-instructions -- over four schedulers {cw(_BLK, "inst") / 4:,.0f}, {cw(_RDY, "inst") / 4:,.0f} and {cw(_RR, "inst") / 4:,.0f} cycles of issue, about half of each window;
  {cn(_BLK, "hmma")} wgmma issues (96 steps, a warpgroup's four warps each); {cn(_BLK, "wavefronts")}, {cn(_RDY, "wavefronts")} and {cn(_RR, "wavefronts")} shared-memory wavefronts
  ({cn(_BLK, "load_wavefronts")} loads' and {cn(_BLK, "store_wavefronts")} stores' in the block form, {cn(_BLK, "conflicts")} of them bank conflicts) -- NOT PRICED on this
  architecture, which declares no rate: at the one a cycle measured on the RTX 3080 Ti ([`smem_rates`](smem_rates.md);
  ampere's, not this card's) they would be {100 * cw(_BLK, "wavefronts") / R[B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + LS]:.0f}% of the block
  form's window ({g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + LS)}) and {100 * cw(_RR, "wavefronts") / R[B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + ", its slots landed with their operands made" + ", the readout's coefficient tiles landed made" + LS]:.0f}% of the both-sides form's ({g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + ", its slots landed with their operands made" + ", the readout's coefficient tiles landed made" + LS)}); {cn(_BLK, "xu")}, {cn(_RDY, "xu")} and {cn(_RR, "xu")}
  conversion-unit instructions, the expansions' alone; {cn(_BLK, "red_requests")} add requests, a token's line each; {cw(_BLK, "l2_read") * 32 / 1024:,.0f} KB read from the cache.
  STALLED, of a warp's waiting cycles: the block form {cw(_BLK, "%lg_throttle"):.0f}% on the global-memory queue -- the checked launch's state stores
  among them -- {cw(_BLK, "%long_scoreboard"):.0f}% on global data and {cw(_BLK, "%mio_throttle"):.0f}% on the shared-memory queue; with both sides' operands made {cw(_RR, "%mio_throttle"):.0f}% on the
  shared-memory queue, {cw(_RR, "%lg_throttle"):.0f}% on the global-memory queue and {cw(_RR, "%long_scoreboard"):.0f}% on global data. What a warp asleep at a barrier's wait is
  counted as is NOT KNOWN.
  SO, BY COUNTS: issue some half of each window, the tensor pipe 96 steps of 20 cycles ([`wgmma_width`](wgmma_width.md)),
  the adds 256 lines of some 9 by four warps ([`add_rate`](add_rate.md)); THE SHARED-MEMORY PIPE, unpriced here, is the one the
  counts put nearest the whole window, and a wavefront's price on this card is what decides it."""
        if _CN
        else "- THE COUNTERS: this record's run read none; the phase ledger below holds the windows' counts."
    )
    sha = j["git"]["sha"][:8]
    inst = instance
    both_steps = f"{R[B + ST] + R[B + FA]:,.0f}"
    CARD = (
        f"H100 80GB HBM3, 132 SMs, Nebius preemptible VM {inst[2:]}"
        if inst.startswith("nb")
        else f"H100 PCIe 80GB, 114 SMs, Hyperstack spot VM {inst[2:]}; a role's steps are its issue and its wait, which the record holds apart"
        if inst.startswith("hs")
        else f"H100 80GB HBM3, 132 SMs, vast.ai instance {inst}"
    )
    text = f"""MEASURED ({CARD}, driver {j["driver"]}; {j["date"][:10]}, the experiment at
{sha}, one session; 7 repetitions, every checked row's state and output values exact;
the median; SM-cycles a token are the window's cycles over 8 -- a CTA of two boxes is 8 tokens a window. THE FOLD'S STATE
DUMP STAYS OUT OF THE TIMED CALLS, and one row restores it to price it in this session):

{tab}
THE WALL TIME, measured on this box and immune to its clock ({min(v for n, v in R.items() if "the card's clock" in n):.2f} to {max(v for n, v in R.items() if "the card's clock" in n):.2f} GHz, its rows' medians): the long call
scaled to 65,536 tokens is {R[B + WALL]:.3f} ms for the baseline, {R[B + SL + WALL]:.3f} ms for the winner's form, and {R[B + SL + BIG + WALL]:.3f} ms at the
once-read bound ({R[B + BIG + WALL]:.3f} without the winner's changes). THE FLASH REFERENCES, from the session of instance 60450651 ON ANOTHER BOX (a different offer and driver)
(`rola_bench.measure.attention`, causal, 1 head, dv 64, bf16; the records in rola-results at `flash/reference`):
FlashAttention-3, built on the box at Dao's pinned commit, 1.775 ms at 65,536 tokens (0.300 at 16,384; 0.121 at
1,024); torch's FlashAttention-2 3.072 ms (0.323; 0.040). THESE ARE ADJACENT MEASUREMENTS, NOT A COMPARISON: the
carry's wall time is this lab's and the flash times the bench's attention entry's, two processes on a box with no
clock lock, and the carry is one part of the op (no intra, facts pass, den or join). Flash is a LAYER entry, so
the comparison is a layer comparison in the declared suite ([`measurement.md`](../measurement.md)), owed when a
layer of this op is a session member beside it. THE PRICE (R22.2) is the once-read wall time -- a cache that may
not hold the working set is priced as if it does not -- and the cached-table time is the upside when the cache
holds; whether a streamed carry's arrivals keep it there is unmeasured.

A warp's clocked cycles a window (the sleeping row's are not tabled: a clock read is not held to its side of a named
barrier, and its waits read as its neighbours' time):

{tab2}
WHAT IT SHOWS.
- THE CARRY COMPOSES, EXACTLY: two boxes' folds, the image a window, one readout warpgroup over both -- its
  coefficients expanded in its registers, its steps' A operand -- and one f16 line a token, in one CTA of 512
  threads at the SM's whole register file.
- A TOKEN IS PRICED AT {t(B + SL + BIG)} SM-CYCLES COMPOSED, its tables read once (R22.2), in the winner's form; {t(B + SL)} when the
  cache holds them ({t(B)} in the first form) at this cell (its sides sparse on opposite levels, 256 x 256 leaves;
  [`carry_floor`](../internals/tools/carry_floor.md) prices a dense side higher).
- THE FILL WANTS WARPS THAT EACH HOLD A BATCH: two producers a box, a batch of 32 entries in registers each, take
  the fold's side alone to {g(F + SELF)} cycles a window where one a box is {g(F)}; one a box holding TWO batches is
  {g(F + LEAN)}; and two a box at 40 registers a thread, each holding one value instruction's four entries, are
  {g(F + FIVE)} -- slower than one a box that holds a batch.
- SUSPENDED WAITS WITH THE LEAN READOUT ARE THE FASTEST COMPOSED FORM: {g(B + SL)} cycles a window against the
  baseline's {g(B)}, with the suspended waits beside the wide byte load a near tie at {g(B + SW)}. The parts: every wait
  suspended alone {g(B + SUS)}, the long waits asleep at named barriers {g(B + SLEEP)}, the lean readout alone {abs(R[B + LEAN] / R[B] - 1) * 100:.1f}%
  under the baseline, the wide load alone {g(B + WIDE)}. ALL THREE AT ONCE LOSE the pairing's gain ({g(B + SLW)}: the lean
  readout beside the wide load's scratch needs 88 registers a thread and the fourth warpgroup drops to 104), and
  the sleeping triple is {g(B + LSW)}.
- THE DEN COSTS {(R[B + SL + NC + DEN] / R[B + SL + NC] - 1) * 100:.1f}% OF THE COMPOSED WINDOW AT THIS CELL AND {(R[B + SL + AL + NC + DEN] / R[B + SL + AL + NC] - 1) * 100:.1f}% AT THE ALIGNED ONE, the tables cached and no
  clock inside a role: {g(B + SL + NC + DEN)} cycles against {g(B + SL + NC)}, and {g(B + SL + AL + NC + DEN)} against {g(B + SL + AL + NC)}. THE ROLES' CLOCKS ARE NOT FREE AND NOT EQUAL: they are
  {R[B + SL] - R[B + SL + NC]:,.0f} cycles of the winner's clocked window and {R[B + SL + DEN] - R[B + SL + NC + DEN]:,.0f} of the den's, and which is larger has changed between
  records, so a clocked row does not price the den (clocked: {g(B + SL + DEN)} against {g(B + SL)}; why the clocks differ is
  not isolated); with the tables once-read {g(B + SL + BIG + DEN)} against {g(B + SL + BIG)}; the fold's side alone {g(F + DEN)} against {g(F)}. One
  cell (its sides sparse on opposite levels); a dense write side, where the fold's side binds, is unmeasured. ITS
  PARTS, each a row less the row before, no clock inside a role: the fold's four steps a window {R[B + SL + P1] - R[B + SL + NC]:,.0f} cycles;
  the image's ninth brick and its store {R[B + SL + P2] - R[B + SL + P1]:,.0f} (the pitch itself nothing: the winner's form at 155 lines is {g(B + SL + P155)}
  against {g(B + SL + NC)}); the register split, readout 88 and fourth warpgroup 104, {R[B + SL + P3] - R[B + SL + P2]:,.0f}; the steps 72 columns wide {R[B + SL + P4] - R[B + SL + P3]:,.0f};
  the panel's word and the drain's add {R[B + SL + NC + DEN] - R[B + SL + P4]:,.0f}. No part is a third of it. The split cannot be avoided in this
  form: the 72-column readout serializes its steps at 80 registers with no clock read too.
- THE FOLD'S FORM WAS A CONTEST. A step a list with ones in one row of A is 16 steps a window for 1 row in 64 and
  a 17th accumulator set of 8, which the toolkit will not hold at 160 registers: at a 168-register fold with
  one-batch producers it cost an eighth of the composed window in its session (the experiment at ac342446, whose
  record this one replaced). The coefficients as A are four steps into 4 registers.
- AN IMAGE TILE'S PITCH MUST BE AN ODD COUNT OF LINES: a warp's lanes store to four tiles two apart, and at an
  even pitch two of the four land on the same banks. The winner's form at 154 lines is {g(B + SL + P154)} against {g(B + SL)} at 137;
  the den's at 161 is {g(B + SL + P161 + DEN)} against {g(B + SL + DEN)} at 155.
- THE IMAGE FORM COSTS {(R[B + SL + AL + NC] / R[B + SL + NC] - 1) * 100:.1f}% MORE WHERE THE CARVES ARE EQUAL, no clock inside a role ({g(B + SL + AL + NC)} cycles a window against
  {g(B + SL + NC)}); clocked, the aligned cell's winner's form is {g(B + SL + AL)} cycles a
  window against {g(B + SL)} at the cell whose sides list opposite levels -- 16 word stores a thread a tile group where
  that cell's are 4 lines -- and with the den {g(B + SL + AL + DEN)} against {g(B + SL + DEN)}, the den's lines there 2-byte
  stores of 8 threads. A form that reads the fold's own registers has nothing of the window to win back but the
  image's stores; it is not built.
- THE CARRY'S PRODUCERS WANT A TOKEN'S VALUES AS ONE RUN: with the values tile-major in global memory -- a token's 8
  lines a kilobyte apart -- the winner's form is {g(B + SL + NC + TV)} cycles a window against {g(B + SL + NC)}, and with the tables once-read
  {g(B + SL + BIG + NC + TV)} against {g(B + SL + BIG + NC)}. The intra's window moves 5% with the values' order ([`intra_window`](intra_window.md)), so
  the values stay token-major and the levels, which the carry reads 16 bytes at a time, go tile-major.
- FROM LISTS, WITH A WINDOW'S OWN TOKENS, THE FOLD'S SIDE ALONE IS {g(F + LS)} CYCLES A WINDOW where the synthetic entries'
  is {g(F)} cached and {g(F + BIG)} once-read; both sides {g(B + LEAN1 + NC + LS)} (the readout's side still on synthetic, cached readers) against
  {g(B + SL + NC)}. Two things are in it, and the rows with every window the same tokens part them: THE LISTS' CODE -- a
  batch's places loaded a batch ahead, a shuffle a value instruction, the sparse word, tile-major addresses, and one
  batch held where the winner holds two -- is {g(F + LH)} against {g(F)} alone and {g(B + LEAN1 + NC + LH)} against {g(B + SL + NC)} composed; and ROWS THE
  CACHE HAS NOT SEEN, a window's own being new each window as a call's are, are the rest, {R[F + LS] - R[F + LH]:,.0f} alone and
  {R[B + LEAN1 + NC + LS] - R[B + LEAN1 + NC + LH]:,.0f} composed. A window's rows are loaded one batch ahead of their use. ASKING FOR THEM A WINDOW AHEAD BY
  PREFETCH LOSES: {g(B + LEAN1 + NC + AH + LS)} composed against {g(B + LEAN1 + NC + LS)} -- six prefetches an entry-batch are more requests of the same
  memory, not less waiting. Whether loads issued two batches ahead hide the rows' arrival is not measured (the
  producers' registers hold one batch from lists). The check of these rows is weak: every table's byte is its token's parity.
- LANDING THE VALUES ALONE FROM EXPANDED LISTS is {g(B + LEAN1 + NC + EX + LS)} cycles a window composed against {g(B + LEAN1 + NC + LS)} with every entry's
  values gathered ({g(B + LEAN1 + NC + LH)} for the gather with rows the cache holds), and the fold's side alone {g(F + EX + LS)} against {g(F + LS)}: a
  list's values are one bulk copy of 2 KB from a buffer laid contiguous a window (the host's here; 16 sources
  share a buffer), new each window, and its producers still gather an entry's 16 dense bytes and its sparse byte,
  a line of each level's rows an entry. What that leaves is the next finding's.
- WITH THE WHOLE ENTRY LANDED -- its coefficient bytes beside its values, raw, a list one bulk copy and no global load
  in a producer's trip -- the window is {g(B + LEAN1 + NC + EX + WH + LS)} cycles composed and the fold's side alone {g(F + EX + WH + LS)}, against {g(B + SL + NC)} and
  {g(F)} for the synthetic entries whose rows the cache holds.
- WHAT THAT ROW NEEDS IS ITS BYTES SHARED, NOT HELD BEFOREHAND: with a window's bytes never read before and its
  16 boxes its readers, the window is {g(B + LEAN1 + NC + EX + WH + ", a column's windows never read before" + LS)} cycles composed against {g(B + LEAN1 + NC + EX + WH + LS)} with the cache holding them, and
  the fold's side alone {g(F + EX + WH + ", a column's windows never read before" + LS)} against {g(F + EX + WH + LS)}: one box's miss serves its 15 mates. With EACH BOX ITS OWN it is
  {g(B + LEAN1 + NC + EX + WH + ", each box its own expanded windows" + LS)} composed, {R[B + LEAN1 + NC + EX + WH + ", each box its own expanded windows" + LS] / R[B + LEAN1 + NC + EX + WH + LS]:.2f} times, and {g(F + EX + WH + ", each box its own expanded windows" + LS)} alone, {R[F + EX + WH + ", each box its own expanded windows" + LS] / R[F + EX + WH + LS]:.2f} times -- 16 times the bytes from global memory, on
  every multiprocessor at once.
- THE PACKED STREAM IS EXACT, AND ITS UNEVEN LISTS COST THE FOLD {R[F + EX + WH + ", a column's stream packed by digit, its lists uneven and their last steps truncated" + LS] / R[F + EX + WH + LS]:.2f} TIMES: {g(F + EX + WH + ", a column's stream packed by digit, its lists uneven and their last steps truncated" + LS)} cycles a window the
  fold's side alone against {g(F + EX + WH + LS)}, {g(B + LEAN1 + NC + EX + WH + ", a column's stream packed by digit, its lists uneven and their last steps truncated" + LS)} composed against {g(B + LEAN1 + NC + EX + WH + LS)}. The same 256 entries a box a window, as
  lists of 8 to 24 where the row above has 16 each, are 24 steps where it has 16 -- three slots of 8 tiles where it
  has two -- and a truncated step, used where it landed with its bricks as far apart as it has entries, folds
  exactly. In four slots of 4 tiles, a slot's copies issued two slots ahead, the same stream is {q("fold", "four slots", no="draining")} alone
  and {q("both", "four slots")} composed: more slots cost more, so a slot's price is not its bytes' landing.
- THE FOLD'S SIDE WAITS ON ITS PRODUCER. With even lists a fold warp waits {q("fold", "", wait=True)} of the fold's side's {g(F + EX + WH + LS)} cycles a
  window on its producer: its own steps are what is left. One producer warp a box expands every entry. On the
  packed stream in four slots the wait is {q("fold", "four slots", no="draining", wait=True)} of {q("fold", "four slots", no="draining")}, and with TWO PRODUCERS A BOX {q("fold", "four slots", yes="draining", wait=True)} of {q("fold", "four slots", yes="draining")}: the wait
  falls and the fold's own time is then the larger part. That fold runs a list's steps as a loop of its own, a
  vote a list a slot, where the even rows' steps are one block; the block rows below run the uneven lists' steps so.
- WITH THE FOLD WARPS EXPANDING THEIR OWN SLOT (even lists; a producer only issues copies) the fold's side alone
  is {g(F + EX + WH + FE + LS)} cycles a window against {g(F + EX + WH + LS)}, a fold warp's wait -- now on its slot's bytes -- {g(F + EX + WH + FE + LS + ": a fold warp waits on its producer")} against {q("fold", "", wait=True)}.
  THE COMPOSED WINDOW DOES NOT MOVE: {g(B + LEAN1 + NC + EX + WH + FE + LS)} against {g(B + LEAN1 + NC + EX + WH + LS)}; with the fold issuing its own copies too, no
  producer running, {g(B + LEAN1 + NC + EX + WH + FI + LS)}.
- THE WINDOW IS THE READOUT'S. That form with every role's waits clocked ({g(B + LEAN1 + EX + WH + FI + LS)} cycles a window): a readout warp
  waits {g(B + LEAN1 + EX + WH + FI + LS + ": a readout warp waits on the image")} on the image and {g(B + LEAN1 + EX + WH + FI + LS + ": a readout warp waits on the drain")} on the drain and is {g(B + LEAN1 + EX + WH + FI + LS + ": a readout warp in its steps")} in its coefficients and steps; a fold
  warp waits {g(B + LEAN1 + EX + WH + FI + LS + ": a fold warp waits on the readout's read")} on the readout's read and {g(B + LEAN1 + EX + WH + FI + LS + ": a fold warp waits on its producer")} on its bytes; a drain warp waits {g(B + LEAN1 + EX + WH + FI + LS + ": a drain warp waits on the panel")} on the
  panel. The one role that does not wait is the readout's warpgroup.
- FOUR DRAIN WARPS BUY NOTHING: {g(B + LEAN1 + NC + EX + WH + FI + D4 + LS)} against {g(B + LEAN1 + NC + EX + WH + FI + LS)}, each drain warp then waiting {g(B + LEAN1 + EX + WH + FI + D4 + LS + ": a drain warp waits on the panel")} of a window on the
  panel, and a readout warp is {g(B + LEAN1 + EX + WH + FI + D4 + LS + ": a readout warp in its steps")} in its coefficients and steps against {g(B + LEAN1 + EX + WH + FI + LS + ": a readout warp in its steps")}. The drain is not the
  window's bound.
- THE MIRRORED READOUT (coefficients in shared memory, tokens on the step's columns; the fold expanding, a producer
  issuing): the window is {g(B + LEAN1 + NC + EX + WH + FE + MR + LS)} cycles against {g(B + LEAN1 + NC + EX + WH + FE + LS)}. Clocked ({g(B + LEAN1 + EX + WH + FE + MR + LS)} against {g(B + LEAN1 + EX + WH + FE + LS)}): a readout warp is
  {g(B + LEAN1 + EX + WH + FE + MR + LS + ": a readout warp forming its steps' A")} forming its coefficients and {g(B + LEAN1 + EX + WH + FE + MR + LS + ": a readout warp in its steps")} in its steps, against {g(B + LEAN1 + EX + WH + FE + LS + ": a readout warp in its steps")} for both in the register form; it
  waits {g(B + LEAN1 + EX + WH + FE + MR + LS + ": a readout warp waits on the image")} on the image and {g(B + LEAN1 + EX + WH + FE + MR + LS + ": a readout warp waits on the drain")} on the drain; a fold warp waits {g(B + LEAN1 + EX + WH + FE + MR + LS + ": a fold warp waits on the readout's read")} on the readout's read against {g(B + LEAN1 + EX + WH + FE + LS + ": a fold warp waits on the readout's read")};
  a drain warp {g(B + LEAN1 + EX + WH + FE + MR + LS + ": a drain warp waits on the panel")} on the panel against {g(B + LEAN1 + EX + WH + FE + LS + ": a drain warp waits on the panel")}.
  What its clocks do not name is {R[B + LEAN1 + EX + WH + FE + MR + LS] - sum(R[B + LEAN1 + EX + WH + FE + MR + LS + ": " + x] for x in ("a readout warp waits on the image", "a readout warp waits on the drain", "a readout warp forming its steps' A", "a readout warp in its steps")):,.0f} cycles a window: its rows' rounding, their stores and its arrivals. Its 32
  steps are a quarter the register form's multiply-adds and take {g(B + LEAN1 + EX + WH + FE + MR + LS + ": a readout warp in its steps")}: a step's price here is not its size.
- THE BLOCK ROWS -- uneven lists, a block a list, the stream saying what to do: the window is {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + LS)} cycles
  composed ({g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS)} clocked), against {g(B + LEAN1 + NC + EX + WH + FI + D4 + MR + C8 + LS)} for even lists under the same readout and drains (whole-entry
  tiles, the fold issuing) -- {R[B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + LS] / R[B + LEAN1 + NC + EX + WH + FI + D4 + MR + C8 + LS]:.2f} times it -- and
  {g(B + LEAN1 + NC + EX + WH + ", a column's stream packed by digit, its lists uneven and their last steps truncated" + LS)} for the uneven lists' truncated steps behind a producer and a loop a list.
  The fold's side alone is {g(F + EX + WH + BK + LS)},
  a fold warp waiting {g(F + EX + WH + BK + LS + ": a fold warp waits on its producer")} of it on its bytes, against {g(F + EX + WH + FE + LS)} for the even lists' fold expanding its own slot. Clocked: a fold warp waits {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a fold warp waits on the readout's read")} on the
  readout's read and {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a fold warp waits on its producer")} on its bytes; a readout warp {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a readout warp waits on the image")} on the image ({g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS + ": a readout warp waits on the image")} with even lists) and {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a readout warp waits on the drain")} on the drain; a drain warp
  {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a drain warp waits on the panel")} on the panel.
  THE FOLD'S PHASES, clocked, the block form against the even lists' (a clocked fold holds nine more
  registers; the windows are {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS)} and {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS)} with them): publishing the image {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a fold warp publishing the image, its wait on the readout's read in it")} against {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS + ": a fold warp publishing the image, its wait on the readout's read in it")},
  its wait on the readout's read in both ({g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a fold warp waits on the readout's read")} and {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS + ": a fold warp waits on the readout's read")}); expanding its entries' lines {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a fold warp expanding its entries' lines")} against {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS + ": a fold warp expanding its entries' lines")}; its pads and its
  steps' operands {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a fold warp at its pads and its steps' operands")}, the block form's alone; its slots' fences {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a fold warp at its first slot's fence")} and {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a fold warp at its second slot's fence")} against {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS + ": a fold warp at its first slot's fence")}
  and {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS + ": a fold warp at its second slot's fence")}; AT ITS SLOTS' RENDEZVOUS -- between a reading before the warpgroup's sync and one behind it --
  {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a fold warp at its first slot's rendezvous")} and {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a fold warp at its second slot's rendezvous")} against {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS + ": a fold warp at its first slot's rendezvous")} and {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS + ": a fold warp at its second slot's rendezvous")}; its steps and its next copy's issue {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a fold warp in its steps and issuing its copies")} against {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS + ": a fold warp in its steps and issuing its copies")}.
  A FENCE IS SOME 60 CYCLES, AND A FOLD WARP IS LONGER AT A SLOT'S RENDEZVOUS THAN EXPANDING THE SLOT. With no
  arithmetic in the fold (the control: windows {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + ", the fold's coefficient lines its landed bytes as they are" + BK + LS)} against {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + LS)} and {g(B + LEAN1 + NC + EX + WH + FI + D4 + MR + C8 + ", the fold's coefficient lines its landed bytes as they are" + LS)} against
  {g(B + LEAN1 + NC + EX + WH + FI + D4 + MR + C8 + LS)}; its entries' lines {g(B + LEAN1 + EX + WH + D4 + MR + C8 + ", the fold's coefficient lines its landed bytes as they are" + BK + LS + ": a fold warp expanding its entries' lines")} and {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + ", the fold's coefficient lines its landed bytes as they are" + LS + ": a fold warp expanding its entries' lines")}) its rendezvous are {g(B + LEAN1 + EX + WH + D4 + MR + C8 + ", the fold's coefficient lines its landed bytes as they are" + BK + LS + ": a fold warp at its first slot's rendezvous")}
  and {g(B + LEAN1 + EX + WH + D4 + MR + C8 + ", the fold's coefficient lines its landed bytes as they are" + BK + LS + ": a fold warp at its second slot's rendezvous")}, and {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + ", the fold's coefficient lines its landed bytes as they are" + LS + ": a fold warp at its first slot's rendezvous")} and {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + ", the fold's coefficient lines its landed bytes as they are" + LS + ": a fold warp at its second slot's rendezvous")}: they do not follow the lines' work. THE
  FOLD'S ARITHMETIC IS NOT WHAT THE WINDOW WAITS ON.
  WITH BOTH COPIES ISSUED AT THE WINDOW'S END (even lists) the window is {g(B + LEAN1 + NC + EX + WH + FI + D4 + MR + C8 + ", both its copies issued at the window's end" + LS)} against {g(B + LEAN1 + NC + EX + WH + FI + D4 + MR + C8 + LS)},
  its fences {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + ", both its copies issued at the window's end" + LS + ": a fold warp at its first slot's fence")} and {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + ", both its copies issued at the window's end" + LS + ": a fold warp at its second slot's fence")}: a fence does not wait on a copy in flight, and the later
  copies cost the window.
  ITS RENDEZVOUS MOVE WITH THE ISSUE: {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + ", both its copies issued at the window's end" + LS + ": a fold warp at its first slot's rendezvous")} at its first slot and {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + ", both its copies issued at the window's end" + LS + ": a fold warp at its second slot's rendezvous")} at its second, where the form
  that issues a slot's copies as it leaves the slot is {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS + ": a fold warp at its first slot's rendezvous")} and {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS + ": a fold warp at its second slot's rendezvous")} -- the rendezvous behind an issue
  is the long one.
  A WARPGROUP WAITS AT ITS RENDEZVOUS FOR ITS FIRST WARP, whose lanes issue the slot's 8 copies: the first warp
  alone is {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS + ": a fold warp at its first slot's rendezvous, its warpgroup's first warp")} and {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS + ": a fold warp at its second slot's rendezvous, its warpgroup's first warp")} at its two rendezvous where the warps' mean is {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS + ": a fold warp at its first slot's rendezvous")} and {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS + ": a fold warp at its second slot's rendezvous")}, and
  {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS + ": a fold warp in its steps and issuing its copies, its warpgroup's first warp")} in its steps and its issue where the mean is {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS + ": a fold warp in its steps and issuing its copies")}.
  WITH THE COPIES ISSUED TWO A WARP every warp is {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + ", its copies issued two a warp" + LS + ": a fold warp at its first slot's rendezvous")} and {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + ", its copies issued two a warp" + LS + ": a fold warp at its second slot's rendezvous")} at its rendezvous and {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + ", its copies issued two a warp" + LS + ": a fold warp in its steps and issuing its copies")} in its
  steps and issue; the window is {g(B + LEAN1 + NC + EX + WH + FI + D4 + MR + C8 + ", its copies issued two a warp" + LS)} against {g(B + LEAN1 + NC + EX + WH + FI + D4 + MR + C8 + LS)}, and a fold warp then waits {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + ", its copies issued two a warp" + LS + ": a fold warp waits on the readout's read")} on the
  readout's read against {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS + ": a fold warp waits on the readout's read")}: THE FOLD'S TIME SAVED IS ITS WAIT ON THE READOUT, WHICH IS THE WINDOW.
  THE BLOCK FORM'S FIRST WARP IS LAST TOO: {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a fold warp at its first slot's rendezvous, its warpgroup's first warp")} and {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a fold warp at its second slot's rendezvous, its warpgroup's first warp")} at its rendezvous against {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a fold warp at its first slot's rendezvous")} and
  {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a fold warp at its second slot's rendezvous")}; its plans, its one copy a slot, 16 threads' operands and the first pads are its own -- {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a fold warp at its pads and its steps' operands, its warpgroup's first warp")} at
  its pads and operands against {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a fold warp at its pads and its steps' operands")}, {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a fold warp expanding its entries' lines, its warpgroup's first warp")} expanding against {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a fold warp expanding its entries' lines")}, {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a fold warp in its steps and issuing its copies, its warpgroup's first warp")} in its steps and
  issue against {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a fold warp in its steps and issuing its copies")}. WITH THOSE JOBS SHARED AMONG ITS WARPS THE WINDOW DOES NOT MOVE: {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + ", its warpgroup's lone jobs shared among its warps" + BK + LS)} against
  {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + LS)}. Its rendezvous fall to {g(B + LEAN1 + EX + WH + D4 + MR + C8 + ", its warpgroup's lone jobs shared among its warps" + BK + LS + ": a fold warp at its first slot's rendezvous")} and {g(B + LEAN1 + EX + WH + D4 + MR + C8 + ", its warpgroup's lone jobs shared among its warps" + BK + LS + ": a fold warp at its second slot's rendezvous")} and every warp's pads and operands rise to
  {g(B + LEAN1 + EX + WH + D4 + MR + C8 + ", its warpgroup's lone jobs shared among its warps" + BK + LS + ": a fold warp at its pads and its steps' operands")}: the phase is some 53 instructions a thread, and a warp is as long in it with a pad and four operands
  as the first warp was with all of them. NOT ISOLATED: what a warp waits on in that phase.
  THE READY PAIR -- the block form's slots landed with their operands made, the fold only landing and stepping:
  the window is {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + ", its slots landed with their operands made" + LS)} against {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + LS)} for the block form and {g(B + LEAN1 + NC + EX + WH + FI + D4 + MR + C8 + LS)} for even
  lists; clocked ({g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + ", its slots landed with their operands made" + LS)}), a fold warp waits {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + ", its slots landed with their operands made" + LS + ": a fold warp waits on its producer")} on its bytes and
  {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + ", its slots landed with their operands made" + LS + ": a fold warp waits on the readout's read")} on the readout's read, is {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + ", its slots landed with their operands made" + LS + ": a fold warp issuing its steps, to their commit")} issuing its 32 steps and
  {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + ", its slots landed with their operands made" + LS + ": a fold warp in its steps and issuing its copies (wait)")} in their wait and its copy's issue, and A READOUT WARP WAITS {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + ", its slots landed with their operands made" + LS + ": a readout warp waits on the image")} ON THE
  IMAGE against {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a readout warp waits on the image")} with the fold expanding and {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS + ": a readout warp waits on the image")} with even lists: with its operands made the
  uneven lists' fold is no longer what the window waits on. What the window is over the even lists' by, no row
  parts: the fold issues twice the steps beside the same readout.
  THE READ-READY ROWS -- the readout's coefficient tiles landed made as well: the window is {g(B + LEAN1 + NC + EX + WH + FI + D4 + MR + C8 + ", the readout's coefficient tiles landed made" + LS)} against
  {g(B + LEAN1 + NC + EX + WH + FI + D4 + MR + C8 + LS)} with even lists, and {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + ", its slots landed with their operands made" + ", the readout's coefficient tiles landed made" + LS)} against {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + ", its slots landed with their operands made" + LS)} for the ready pair's form. Clocked
  ({g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + ", the readout's coefficient tiles landed made" + LS)} and {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + ", its slots landed with their operands made" + ", the readout's coefficient tiles landed made" + LS)}), a readout warp is {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + ", the readout's coefficient tiles landed made" + LS + ": a readout warp forming its steps' A")} and {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + ", its slots landed with their operands made" + ", the readout's coefficient tiles landed made" + LS + ": a readout warp forming its steps' A")} from its
  image's wait to its tiles landed, where forming them itself with its rendezvous is {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS + ": a readout warp forming its steps' A")} and
  {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + ", its slots landed with their operands made" + LS + ": a readout warp forming its steps' A")}; it waits {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + ", the readout's coefficient tiles landed made" + LS + ": a readout warp waits on the image")} and {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + ", its slots landed with their operands made" + ", the readout's coefficient tiles landed made" + LS + ": a readout warp waits on the image")} on the image, and a fold warp
  {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + ", the readout's coefficient tiles landed made" + LS + ": a fold warp waits on the readout's read")} and {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + ", its slots landed with their operands made" + ", the readout's coefficient tiles landed made" + LS + ": a fold warp waits on the readout's read")} on the readout's read.
  THE OWN ROWS -- every box its own stream, where 16 share a column's -- ARE THREE TIMES THE WINDOW: {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + ", its slots landed with their operands made" + ", the readout's coefficient tiles landed made" + ", every box its own stream" + LS)}
  against {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + ", its slots landed with their operands made" + ", the readout's coefficient tiles landed made" + LS)}. Their streams are some 680 MB, past the card's cache, where the form's are under it; a
  window's bytes a CTA are the same. Clocked ({g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + ", its slots landed with their operands made" + ", the readout's coefficient tiles landed made" + ", every box its own stream" + LS)}): A READOUT WARP WAITS {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + ", its slots landed with their operands made" + ", the readout's coefficient tiles landed made" + ", every box its own stream" + LS + ": a readout warp forming its steps' A")} ON ITS
  TILES' BYTES against {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + ", its slots landed with their operands made" + ", the readout's coefficient tiles landed made" + LS + ": a readout warp forming its steps' A")} -- its copy is issued one panel's stores ahead of its use -- and
  a fold warp, whose copies are issued a window ahead, {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + ", its slots landed with their operands made" + ", the readout's coefficient tiles landed made" + ", every box its own stream" + LS + ": a fold warp waits on its producer")} on its own against {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + ", its slots landed with their operands made" + ", the readout's coefficient tiles landed made" + LS + ": a fold warp waits on its producer")}; the
  fold then waits {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + ", its slots landed with their operands made" + ", the readout's coefficient tiles landed made" + ", every box its own stream" + LS + ": a fold warp waits on the readout's read")} on the readout's read. WITH ITS OPERANDS FROM THE CARD'S MEMORY THE CARRY IS
  THAT MEMORY'S: the ready stream is a window's ~123 KB a CTA, and past the cache it is this price. NOT ASKED: the
  readout's copy issued further ahead; the design's mix, a box's own coefficients and its column's values.
  THE TOKEN ROWS -- the fold laying its own operands from a view in token order -- ARE A QUARTER OVER THE BLOCK FORM, HOWEVER THE VIEW
  ARRIVES: the window is {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + TKS + LS)} with a half-window view staged by one copy and its value lines moved in shared memory,
  {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + ONV + LS)} with ONE VIEW A WINDOW loaded through registers and its 32 steps one block, and {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + TKT + LS)} with the half-window
  view through registers, against {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + LS)} for the block form, whose value runs land in place, and {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + ", its slots landed with their operands made" + LS)}
  with its operands made. Clocked ({g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + TKS + LS)} and {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + ONV + LS)}; the block form {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS)}):
  THE MOVE IS THE FOLD'S FILL -- a fold warp is {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + TKS + LS + ": a fold warp expanding its entries' lines")} expanding and moving its entries where the block form's is
  {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a fold warp expanding its entries' lines")} expanding them, 16 value lines a thread a window through the shared-memory pipe twice; and THROUGH REGISTERS
  THE FILL IS {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + ONV + LS + ": a fold warp expanding its entries' lines")}, two entries a thread of 9 lines each from global memory, the stream in the cache: a warp's
  loads of 16 bytes are a fifth of a bulk copy's bytes a cycle (KERNEL_STANDARDS 33). THE ONE-VIEW ROW'S STEPS AND RENDEZVOUS ARE SHORT, AND NOT BY ITS STRUCTURE (the joined row, below): its 32 steps
  issue in {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + ONV + LS + ": a fold warp issuing its steps, to their commit")} where the block form's two slots' are {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a fold warp issuing its steps, to their commit")}, its one rendezvous is
  {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + ONV + LS + ": a fold warp at its first slot's rendezvous")} where two are {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a fold warp at its first slot's rendezvous")} and {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a fold warp at its second slot's rendezvous")}, and its pads and operands
  {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + ONV + LS + ": a fold warp at its pads and its steps' operands")} against {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a fold warp at its pads and its steps' operands")}. A readout warp waits {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + TKS + LS + ": a readout warp waits on the image")} and {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + ONV + LS + ": a readout warp waits on the image")}
  on the image against {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a readout warp waits on the image")}: the fold is the window in both. NOT ASKED: one view a window staged by
  small bulk copies; a rank counted from the mask, where these rows' places are the host's.
  THE JOINED ROW -- the block form's two slots laid before ONE RENDEZVOUS, its 32 steps one block -- DOES NOT MOVE THE WINDOW:
  {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + JN + LS)} against {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS)}, both clocked. Its one rendezvous is {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + JN + LS + ": a fold warp at its first slot's rendezvous")} where the two are {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a fold warp at its first slot's rendezvous")} and
  {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a fold warp at its second slot's rendezvous")}, its steps' issue {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + JN + LS + ": a fold warp issuing its steps, to their commit")} against {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a fold warp issuing its steps, to their commit")}, with their wait {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + JN + LS + ": a fold warp in its steps and issuing its copies")} against
  {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a fold warp in its steps and issuing its copies")}; its expansion is {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + JN + LS + ": a fold warp expanding its entries' lines")} against {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a fold warp expanding its entries' lines")} and its pads and operands {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + JN + LS + ": a fold warp at its pads and its steps' operands")} against
  {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + LS + ": a fold warp at its pads and its steps' operands")}. The row also issues both copies at the window's end, which the late-issue rows price apart; the two are
  not parted here. SO THE ONE-VIEW ROWS' SHORT ISSUE AND RENDEZVOUS ARE NOT THEIR STRUCTURE'S. READ AS, not isolated: their
  fold is alone at its steps while the readout waits on the image, where this row's steps run beside the readout's.
  THE CUT ROWS: OF THE PHASE LEDGER'S THREE SHARED-MEMORY COSTS ONLY THE PANEL'S CUT IS WORTH TIME. The block form's
  window is {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + CST + LS)} with the panel by stmatrix against {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + LS)}, {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + CSH + LS)} with the steps' operands shuffled,
  {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + CSK + LS)} issuing only the steps a list reaches, and {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + CST + CSH + CSK + LS)} with all three; the read-ready ready pair's
  {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + CST + CRDY + LS)}, {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + CSH + CRDY + LS)}, {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + CSK + CRDY + LS)} and {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + CST + CSH + CSK + CRDY + LS)} against {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + CRDY + LS)}. All three cut the
  windows' shared-memory wavefronts by some 1,020 a CTA-window (the phase ledger: 4,238 to 3,216 and 3,758 to 2,735),
  and two of them lengthen the window: A WINDOW WITH A QUARTER LESS SHARED-MEMORY TRAFFIC IS NOT SHORTER, SO THAT PIPE
  IS NOT WHAT HOLDS IT. The shuffles put four instructions before each step's issue and the skip a branch around it,
  both on the fold's steps' issue; the panel's stores are the readout's, beside its steps. READ AS, not isolated: the
  window is held by the fold's and the readout's chain through the image, its waits and issue, not by a pipe's bytes.
  THE AHEAD ROWS -- the fold's steps' operand lines loaded two steps ahead -- DO NOT MOVE THE WINDOW: {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + CST + CAH + LS)}
  against {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + CST + LS)} for the block form with the panel by stmatrix, and {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + CST + CAH + CRDY + LS)} against {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + CST + CRDY + LS)} for the
  read-ready ready pair's. Alone, a fold step's issue IS its operands' load
  (the held-step row below: some 15 cycles a step held against 48); in the composed rows the loads ahead -- several in
  flight in their binary -- leave the issue where it was (the clocked ahead row: some 58 cycles a step either way).
  THE NO-READ-STEPS ROW -- the readout issuing no step -- SHOWS THE FOLD'S STEPS ARE HELD BY THEIR OWN: clocked, the window
  is {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + CNR + CRDY + LS)} against {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + CRDY + LS)} with the readout's steps, and a fold warp's steps' issue {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + CNR + CRDY + LS + ": a fold warp issuing its steps, to their commit")} against
  {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + CRDY + LS + ": a fold warp issuing its steps, to their commit")}: some 62 cycles a step with no readout's step beside it. The two fold warpgroups' 64 steps
  a window are 16 columns each, reading their 2.5 KB of operands at 20 cycles ([`wgmma_width`](wgmma_width.md)): 1,280
  cycles of the shared memory's reads between them. READ AS, not isolated: the fold's issue is its two warpgroups'
  narrow steps' operand reads, and a step's own issue beside them.
  THE SERVE ROWS -- the drain warps expanding the fold's slots a slot ahead -- LOSE: the window is {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + CST + CSV + LS)} against
  {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + CST + LS)} for the same form with the fold expanding its own, where some 5,700 was predicted. Clocked ({g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + CST + CSV + LS)}), the
  path moves to the drain warps: a drain warp waits {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + CST + CSV + LS + ": a drain warp waits on the panel")} of it (`lab path`: busy 82%), a fold warp
  {g(B + LEAN1 + EX + WH + D4 + MR + C8 + BK + CST + CSV + LS + ": a fold warp waits on its producer")} on its slots' expansion -- the fold left the path and the serve warps took it. The expansion
  cost them some 2,600 cycles a window where some 1,000 was predicted: a slot is expanded a box at a time, an entry a
  thread, its load, products, stores, fence and arrival in turn, with nothing in flight across boxes or slots.
  THE MEETING ROWS -- the rendezvous an mbarrier's -- DO NOT MOVE IT: the window is {g(B + LEAN1 + NC + EX + WH + FI + D4 + MR + C8 + ", a warpgroup's rendezvous an mbarrier's" + LS)} against
  {g(B + LEAN1 + NC + EX + WH + FI + D4 + MR + C8 + LS)} with even lists and {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + ", a warpgroup's rendezvous an mbarrier's" + BK + LS)} against {g(B + LEAN1 + NC + EX + WH + D4 + MR + C8 + BK + LS)} in the block form; clocked, a fold warp
  is {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + ", a warpgroup's rendezvous an mbarrier's" + LS + ": a fold warp at its first slot's rendezvous")} and {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + ", a warpgroup's rendezvous an mbarrier's" + LS + ": a fold warp at its second slot's rendezvous")} at its rendezvous (that row's folds at 168 registers), and
  {g(B + LEAN1 + EX + WH + D4 + MR + C8 + ", a warpgroup's rendezvous an mbarrier's" + BK + LS + ": a fold warp at its first slot's rendezvous")} and {g(B + LEAN1 + EX + WH + D4 + MR + C8 + ", a warpgroup's rendezvous an mbarrier's" + BK + LS + ": a fold warp at its second slot's rendezvous")} in the block form. The time is not the named barrier's own.
  WITH EVERY WAIT POLLED, not suspended (even lists): the window is {g(B + ", the readout's A in two sets" + NC + EX + WH + FI + D4 + MR + C8 + LS)} against {g(B + LEAN1 + NC + EX + WH + FI + D4 + MR + C8 + LS)};
  clocked ({g(B + ", the readout's A in two sets" + EX + WH + FI + D4 + MR + C8 + LS)}), a fold warp's rendezvous are {g(B + ", the readout's A in two sets" + EX + WH + FI + D4 + MR + C8 + LS + ": a fold warp at its first slot's rendezvous")} and {g(B + ", the readout's A in two sets" + EX + WH + FI + D4 + MR + C8 + LS + ": a fold warp at its second slot's rendezvous")}, its waits {g(B + ", the readout's A in two sets" + EX + WH + FI + D4 + MR + C8 + LS + ": a fold warp waits on the readout's read")} on the readout's
  read and {g(B + ", the readout's A in two sets" + EX + WH + FI + D4 + MR + C8 + LS + ": a fold warp waits on its producer")} on its bytes.
  A slot's steps are a
  fixed two rounds: ISSUING ONLY THE STEPS ITS LISTS HAVE, a second-round step a trip of one or none, WAS MEASURED
  AND LOST (the experiment at 758f0d43, its record that commit's: 7,022 against 6,822 composed, the fold's side
  alone 4,385 against 3,741), and its rows are not kept. NOT ASKED: the drain warps at 88 registers under this
  fold (they are at 72 here); lists past two rounds.
- THE MIRRORED READOUT'S SUMS TO THE PANEL ARE ITS LARGEST PHASE (the even lists' form, clocked): a readout warp is
  {R[B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS + ": a readout warp's sums to the panel, its wait on the drain in it"] - R[B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS + ": a readout warp waits on the drain"]:,.0f} cycles a window rounding its sums, storing them a half each and arriving -- its
  {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS + ": a readout warp's sums to the panel, its wait on the drain in it")} less its {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS + ": a readout warp waits on the drain")} waiting on the drain -- against {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS + ": a readout warp in its steps")} in its steps and
  {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS + ": a readout warp forming its steps' A")} forming its coefficients. NOT ASKED: the panel laid so a thread's two sums are one word's store.
- THE OUTPUT PATH ON FOUR WARPS (the chunk-of-8 form, the fold issuing its own copies, four drain warps): the window
  is {g(B + LEAN1 + NC + EX + WH + FI + D4 + MR + C8 + LS)} cycles against {g(B + LEAN1 + NC + EX + WH + FE + MR + C8 + LS)} with two drain warps and a producer; clocked {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS)} against {g(B + LEAN1 + EX + WH + FE + MR + C8 + LS)}: a readout
  warp waits {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS + ": a readout warp waits on the drain")} on the drain against {g(B + LEAN1 + EX + WH + FE + MR + C8 + LS + ": a readout warp waits on the drain")}, a drain warp {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS + ": a drain warp waits on the panel")} on the panel against {g(B + LEAN1 + EX + WH + FE + MR + C8 + LS + ": a drain warp waits on the panel")} -- and A FOLD WARP WAITS
  {R[B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS + ": a fold warp waits on the readout's read"] + R[B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS + ": a fold warp waits on its producer"]:,.0f} of the window, on the readout's read and on its bytes, against {R[B + LEAN1 + EX + WH + FE + MR + C8 + LS + ": a fold warp waits on the readout's read"] + R[B + LEAN1 + EX + WH + FE + MR + C8 + LS + ": a fold warp waits on its producer"]:,.0f}: the folds, which waited a third
  of a window, now wait a tenth; no clock says in what.
- WITH THE DRAIN WARPS ROUNDING the window is {g(B + LEAN1 + NC + EX + WH + FI + D4 + MR + C8 + RP + LS)} against {g(B + LEAN1 + NC + EX + WH + FI + D4 + MR + C8 + LS)}: nothing. Clocked ({g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + RP + LS)}), a readout
  warp's unclocked remainder is {R[B + LEAN1 + EX + WH + FI + D4 + MR + C8 + RP + LS] - sum(R[B + LEAN1 + EX + WH + FI + D4 + MR + C8 + RP + LS + ": " + x] for x in ("a readout warp waits on the image", "a readout warp waits on the drain", "a readout warp forming its steps' A", "a readout warp in its steps")):,.0f} against {R[B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS] - sum(R[B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS + ": " + x] for x in ("a readout warp waits on the image", "a readout warp waits on the drain", "a readout warp forming its steps' A", "a readout warp in its steps")):,.0f} and it waits {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + RP + LS + ": a readout warp waits on the image")} on the image against {g(B + LEAN1 + EX + WH + FI + D4 + MR + C8 + LS + ": a readout warp waits on the image")}: what the
  readout sheds it waits out on the folds' image.
- THE MIRRORED READOUT WITH A CHUNK OF 8 DIGITS: the window is {g(B + LEAN1 + NC + EX + WH + FE + MR + C8 + LS)} cycles against {g(B + LEAN1 + NC + EX + WH + FE + MR + LS)} at 4. Clocked ({g(B + LEAN1 + EX + WH + FE + MR + C8 + LS)}
  against {g(B + LEAN1 + EX + WH + FE + MR + LS)}): a readout warp is {g(B + LEAN1 + EX + WH + FE + MR + C8 + LS + ": a readout warp forming its steps' A")} forming and {g(B + LEAN1 + EX + WH + FE + MR + C8 + LS + ": a readout warp in its steps")} in its steps against {g(B + LEAN1 + EX + WH + FE + MR + LS + ": a readout warp forming its steps' A")} and {g(B + LEAN1 + EX + WH + FE + MR + LS + ": a readout warp in its steps")} -- half the
  passes, three quarters of the time -- and what its clocks do not name is {R[B + LEAN1 + EX + WH + FE + MR + C8 + LS] - sum(R[B + LEAN1 + EX + WH + FE + MR + C8 + LS + ": " + x] for x in ("a readout warp waits on the image", "a readout warp waits on the drain", "a readout warp forming its steps' A", "a readout warp in its steps")):,.0f} against {R[B + LEAN1 + EX + WH + FE + MR + LS] - sum(R[B + LEAN1 + EX + WH + FE + MR + LS + ": " + x] for x in ("a readout warp waits on the image", "a readout warp waits on the drain", "a readout warp forming its steps' A", "a readout warp in its steps")):,.0f}: its
  rows' rounding and stores are a value's, not a pass's. It waits {g(B + LEAN1 + EX + WH + FE + MR + C8 + LS + ": a readout warp waits on the drain")} on the drain against {g(B + LEAN1 + EX + WH + FE + MR + LS + ": a readout warp waits on the drain")}, and a drain
  warp waits {g(B + LEAN1 + EX + WH + FE + MR + C8 + LS + ": a drain warp waits on the panel")} on the panel against {g(B + LEAN1 + EX + WH + FE + MR + LS + ": a drain warp waits on the panel")}: the two drain warps are near the window's whole length at
  work (their warpgroup at 88 registers here, 96 there; no row parts that from the two panels handed at once).
- WITH THE READOUT A SUM A BOX (the fold expanding, a producer issuing) THE WINDOW IS LONGER: {g(B + LEAN1 + NC + EX + WH + FE + TS + LS)} cycles against
  {g(B + LEAN1 + NC + EX + WH + FE + LS)}. Clocked ({g(B + LEAN1 + EX + WH + FE + TS + LS)} against {g(B + LEAN1 + EX + WH + FE + LS)}): a readout warp is {g(B + LEAN1 + EX + WH + FE + TS + LS + ": a readout warp in its steps")} in its coefficients and steps against {g(B + LEAN1 + EX + WH + FE + LS + ": a readout warp in its steps")} --
  two chains of 4 steps are a tenth under one of 8, so the chain was not what its steps cost -- and it waits
  {g(B + LEAN1 + EX + WH + FE + TS + LS + ": a readout warp waits on the drain")} on the drain against {g(B + LEAN1 + EX + WH + FE + LS + ": a readout warp waits on the drain")}, a drain warp waiting {g(B + LEAN1 + EX + WH + FE + TS + LS + ": a drain warp waits on the panel")} on the panel against {g(B + LEAN1 + EX + WH + FE + LS + ": a drain warp waits on the panel")}. The row's serve
  warps are at 80 registers where the other's are at 112, and its drain is what the window then waits on; no row
  parts the drain's registers from the readout's second sum. NOT ASKED: the readout's steps by what they
  multiply -- a step is 64 rows of which one digit's 16 are not zero; its coefficients formed by warps that
  wait; the fold-expands form on the uneven stream; a truncated step's pads left out of the expanding.
- THE TABLES ONCE-READ COST {min(R[B + BIG] / R[B], R[B + SL + BIG] / R[B + SL], R[F + BIG] / R[F]):.1f} TO {max(R[B + BIG] / R[B], R[B + SL + BIG] / R[B + SL], R[F + BIG] / R[F]):.1f} TIMES THE WINDOW ON THIS BOX (in cycles, which for a row bound by global
  memory follow the card's clock): {g(B + BIG)} cycles composed against {g(B)}, the winner's form
  {g(B + SL + BIG)} against {g(B + SL)}, the fold's side alone {g(F + BIG)} against {g(F)} -- every access a fresh row of 128 MB of
  values, the model of a large table whose rows are each read once.
- THE DUMP IS PRICED IN THIS SESSION: the fold's state stored out in every call, as every record before 79ec1b08
  ran, puts the composed row at {g(B + DUMP)} against {g(B)} -- one binary, one box, one session.
- FIVE WARPGROUPS LOSE COMPOSED, {g(B + FIVE)} against {g(B)}: 640 threads leave four producers and four drain warps 40
  registers a thread, the fold waits on its producers {g(B + FIVE + WP)} cycles a window, and the lean readout the count
  forces is {g(B + FIVE + ST)} in its steps and their forming where one commit group's are {both_steps}. Four drain warps do
  help the read side alone, {g(D + FIVE)} against {g(D)}.
- THE LEAN READOUT WITH TWO-BATCH PRODUCERS IS {g(B + LEAN)} COMPOSED against {g(B)}: a readout warp is {g(B + LEAN + ST)} cycles
  a window in its steps and their forming. The two changes are not separated.
- THE READOUT DRAINING ITS OWN ROWS LOSES COMPOSED, {g(B + SELF)} against {g(B)}: a readout warp is {g(B + SELF + DR)} cycles a
  window in its 64 adds, and the window becomes the readout's -- the fold waits on its read {g(B + SELF + WR)}. A warp that
  issues an add is held until the SM passes it: adds issued by the readout hold the readout.
- COMPOSED, THE PRODUCERS' LOADS ARE NOW THE LARGEST SINGLE ACCESS: with them out the window is {g(B + NL)} against {g(B)}, and
  those loads are three fifths of the fold's side alone -- {g(F + NL)} cycles a window without them against {g(F)}, some
  {(R[F] - R[F + NL]) / 16:.0f} cycles a batch of a producer's 9 load instructions; their coefficient arithmetic is {R[F] - R[F + NM]:,.0f} of it
  ({g(F + NM)} without). The drains' adds out, the window is {g(B + NA)}; the image's stores out, {g(B + NI)}.
- A PARKED FOLD WARP COSTS THE READ SIDE {R[D + NF + PARK] - R[D + NF]:,.0f} CYCLES, A PUBLISHING ONE {R[D] - R[D + NF]:,.0f}: the read side alone is {g(D)}
  with the fold warps publishing and {g(D + NF)} with no fold warp running, while one parked asleep beside it costs
  {R[D + NF + PARK] - R[D + NF]:,.0f} cycles and one spinning {R[D + NF + SPIN] - R[D + NF]:,.0f}. What of the publishing fold's hand-off holds it is NOT ISOLATED
  (its stores are {R[D] - R[D + NI]:,.0f} of it; its fence nothing).
- THE READOUT'S OWN BYTE LOADS ARE A SIXTH OF ITS STEPS' SPAN: 8 instructions a chunk, each lane's 4 bytes of a
  different row; without them a readout warp's steps are {g(D + NB + ST)} cycles a window against {g(D + ST)}. The read side alone
  is then the drain's -- {g(D + NB)} against {g(D)}, the readout waiting on the drain {g(D + NB + WD)} -- at some
  {(R[D + NB] - R[D + NB + DW]) / 128:.0f} cycles an add a drain warp, where with no fold warp running an add is {(R[D + NF + NB] - R[D + NF + NB + DW]) / 128:.0f}.
- ONE WIDE LOAD A WARP FOR THE READOUT'S BYTES is the read side alone without those loads -- {g(D + WIDE)} cycles a window
  against {g(D + NB)} with them out and {g(D)} as they were -- and little composed: {g(B + WIDE)} against {g(B)}, where the
  byte loads out are {g(B + NB)} and the drains' adds out with them {g(B + NA + NB)}.
- THE READOUT FORMS ITS STEPS' A IN {g(D + FA)} TO {g(B + FA)} CYCLES A WINDOW, {R[D + FA] / 4:.0f} to {R[B + FA] / 4:.0f} a chunk: the coefficients' rounds
  and the 32 words held.
- THE STEPS THEMSELVES ARE OVER THEIR OWN TIME with nothing beside them but the drains: {R[D + NF + ST] / 4:.0f} cycles a chunk (the
  fence, 8 steps, the commit, 8 loads issued, the wait) where 8 steps of 64 columns into one accumulator from
  shared memory are 319 ([`wgmma_width`](wgmma_width.md)). Not isolated here.
- EVERY ROLE ISSUES AN INSTRUCTION EVERY 3.5 TO 7 CYCLES of its busy time (the static ledger's counts a trip: a
  producer 168 a batch, a drain warp 339 a chunk, the readout 284 a chunk, a fold warp 379 a window): no role is
  bound by the instructions it issues, and what holds each is not isolated.
{COUNTERS}

{edges_section(root)}
"""
    p = root / "docs/lab/carry_window.md"
    s = p.read_text()
    a = s.index("MEASURED (H100 80GB")
    b = s.index("Record: `measure/lab/results")
    p.write_text(s[:a] + text + s[b:])

    # the architecture's figures of this record, kept the record's
    ap = root / "docs/ARCHITECTURE.md"
    a = ap.read_text()
    clocks = [v for n, v in R.items() if "the card's clock" in n]
    subs = [
        (
            r"PRICED at [\d,]+ SM-cycles a token, its tables read once \(R22\.2\), and [\d,]+ when the cache holds them \([\d,]+ with no clock inside a role\)",
            f"PRICED at {t(B + SL + BIG)} SM-cycles a token, its tables read once (R22.2), and {t(B + SL)} when the cache holds them ({t(B + SL + NC)} with no clock inside a role)",
        ),
        (
            r"on a card clocked at [\d.]+ to [\d.]+ GHz over the session",
            f"on a card clocked at {min(clocks):.2f} to {max(clocks):.2f} GHz over the session",
        ),
        (
            r"[\d.]+% of the composed window with the tables cached and no clock inside a role \([\d.]+% at an aligned cell\) -- the fold's steps [\d,]+ cycles a window, the image's brick and store [\d,]+, the register split [\d,]+, the 72 columns [\d,]+, the panel's word and the drain's add [\d,]+",
            f"{(R[B + SL + NC + DEN] / R[B + SL + NC] - 1) * 100:.1f}% of the composed window with the tables cached and no clock inside a role ({(R[B + SL + AL + NC + DEN] / R[B + SL + AL + NC] - 1) * 100:.1f}% at an aligned cell) -- the fold's steps {R[B + SL + P1] - R[B + SL + NC]:,.0f} cycles a window, the image's brick and store {R[B + SL + P2] - R[B + SL + P1]:,.0f}, the register split {R[B + SL + P3] - R[B + SL + P2]:,.0f}, the 72 columns {R[B + SL + P4] - R[B + SL + P3]:,.0f}, the panel's word and the drain's add {R[B + SL + NC + DEN] - R[B + SL + P4]:,.0f}",
        ),
        (
            r"MEASURED [\d.]+% over the divergent cell's window",
            f"MEASURED {(R[B + SL + AL + NC] / R[B + SL + NC] - 1) * 100:.1f}% over the divergent cell's window",
        ),
        (
            r"\([\d,]+ cycles a window against [\d,]+, and [\d,]+ against [\d,]+ once-read:",
            f"({g(B + SL + NC + TV)} cycles a window against {g(B + SL + NC)}, and {g(B + SL + BIG + NC + TV)} against {g(B + SL + BIG + NC)} once-read:",
        ),
    ]
    for pat, new in subs:
        assert len(re.findall(pat, a)) == 1, pat[:50]
        a = re.sub(pat, new, a)
    ap.write_text(a)
    # the intra page's figure of the carry
    ip = root / "docs/lab/intra_window.md"
    i = ip.read_text()
    i2 = re.sub(
        r"against its (?:[\d,]+ )?SM-cycles a token with no clock inside a role",
        f"against its {t(B + SL + NC)} SM-cycles a token with no clock inside a role",
        i,
    )
    assert i2 != i or f"against its {t(B + SL + NC)} SM-cycles" in i
    ip.write_text(i2)
    return [p, ap, ip]
