# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The measured section of `docs/lab/carry_token.md`, generated from `measure/lab/results/carry_token/`.

`write(root, instance)` reads `sm_90a.blocks1.json` and `sm_90a.blocks132.json` (the clock locked, one CTA and the
full grid) and the warpline records beside them, `sm_90a.blocks1.trace/` and `sm_90a.blocks132.trace/` (a row each),
and rewrites `docs/lab/carry_token.md` under `root` from its `MEASURED (` line to its end, every figure the records' own
and every comparative word chosen by the figures it compares. Rows are matched by NAME -- a traced row's name with its
trace suffix removed (`lab_trace.untraced_name`) is its plain twin's -- never by position. A row the experiment has and
the record does not yet hold (the fold's fill by phases, the waits on the packets landed, the plain row's long call by
wall time) is named as not yet recorded, and its figures are written when a record holds it.

`instance` is the rented box's label: `nb...` Nebius, `hs...` Hyperstack, digits vast.ai. When the two records ran on two
boxes it is both labels, joined by a comma (`nb26cmmpjn,nb8sredw87`); the records hold no box's id, so the page says
which boxes the labels name and not which record ran on which. It returns the files it wrote (`lab page carry_token
--instance LABEL`, docs/internals/tools/lab.md).
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

from _record import Record

#: two windows (or two figures) within this share of each other (of the lesser) are said to agree
SAME = 0.02
#: the first windows left out of a traced row's steady span (`lab trace`'s default)
SKIP = 3
#: the experiment's tokens a window a CTA as `carry_window`'s wall-clock rows count them: 65,536 tokens are
#: 65,536 / (TOKENS_A_WINDOW SMs) windows a CTA
TOKENS_A_WINDOW = 8
TOKENS = 65536
PREFIX = "the token-stream carry: both sides"
UNIT = "cycles a window a CTA"
SHARED = ", lists overlapping on shared tokens"
#: the grouping rows' layouts, by the suffix the experiment's row name carries after its prefix (the order the name
#: writes them: layout, the per-column choice, the isolation, the registers, the trace)
GROUP4 = ", groups of four digits"
TOKEN4 = ", a token on all four digits of its group, groups of four"
TOKEN4_LIST = ", a token on all four digits of its group, a list a digit"
TOKEN4_TILES = ", a token on all four digits of its group, groups of four in a group's two tile places"
TOKEN2 = ", a token on two digits of its group, groups of four"
CLEARED = ", its tiles cleared"
COUNTED = ", a group's steps counted at run time"
TOKEN8 = ", groups of four taking four, two, one and one steps"
LOOPED = ", a group's steps a voted loop on its count"
TABLED = ", an entry's dense line from the tokens' table"
SPREAD = ", an entry's dense line from a 2 GB table of the tokens"
AHEAD = ", loaded a window ahead"
LISTED = ", the readout's passes its packet's"
READY = ", its tiles and lines offsets ready to add"
PADDED = ", a half in two padded passes"
ONE_BLOCK = ", every CTA on one block's reader packets"
FAR = ", the drain's lines loaded a pass ahead"
SPLIT_READ = ", the readout's pass by phases stamped"
SCAN = ", its drain warps also listing the window's readers from the mask"
ROOM = ", registers 176 / 104 / 56"
EARLY = ", the mask loaded a window ahead"
COUNTS = ", the live tokens before each word handed with it"
TOKEN16 = ", a token on all 16 digits of its box, groups of four on one run of value lines"
LAYOUTS = (
    (SHARED, "lists overlapping on shared tokens"),
    (TOKEN4_LIST, "a token on all four digits of its group, a list a digit"),
    (TOKEN4_TILES, "a token on all four digits of its group, groups of four in a group's two tile places"),
    (TOKEN4, "a token on all four digits of its group, groups of four"),
    (TOKEN2, "a token on two digits of its group, groups of four"),
    (TOKEN8, "groups of four taking four, two, one and one steps"),
    (TOKEN16, "a token on all 16 digits of its box, groups of four on one run of value lines"),
    (GROUP4, "groups of four digits"),
)
COLUMN = ", the box's width word choosing its fold"
SPLIT = ", registers 176 / 112 / 48"
IDLE = ", the read side idle"
FILL_ROW = ", the fold's fill by phases stamped"
PACKET_ROW = ", the waits on the packets landed stamped"
#: a detail row's words in the table, by its suffix: the group its coarse row keeps more
DETAILS = (
    (", the waits on the tile groups read stamped", "the waits on the tile groups read"),
    (", the waits on the tile groups published stamped", "the waits on the tile groups published"),
    (", the waits on the panels filled stamped", "the waits on the panels filled"),
    (", the waits on the panels drained stamped", "the waits on the panels drained"),
    (", the fold's steps by eighths stamped", "the fold's steps by eighths"),
    (FILL_ROW, "the fold's fill by phases"),
    (PACKET_ROW, "the waits on the packets landed"),
)
EDGES = ", the launch's two edges stamped"
COARSE_SUF = ", every warp's phases stamped"
FINE = ", every warp's timeline stamped"


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def against(a: float, b: float) -> tuple[float, str]:
    """a against b: its share over (positive) or under (negative) the lesser, and the word the share chooses"""
    share = (a - b) / min(a, b)
    return share, "the same as" if abs(share) <= SAME else ("over" if share > 0 else "under")


def vs(a, b, obj=""):
    """a against b: 'the same' (within SAME of the lesser) or its share of b over or under it, then `obj`"""
    _, word = against(a, b)
    if word == "the same as":
        return "the same" + (f" as {obj}" if obj else "")
    return f"{pct(abs(a - b) / b)} {word}" + (f" {obj}" if obj else "")


def card(inst: str, device: str, sms: int) -> str:
    def one(label):
        return (
            ("Nebius preemptible VM", label[2:])
            if label.startswith("nb")
            else ("Hyperstack spot VM", label[2:])
            if label.startswith("hs")
            else ("vast.ai instance", label)
        )

    labels = [x.strip() for x in inst.split(",") if x.strip()]
    assert 1 <= len(labels) <= 2, f"lab page: --instance names one box or two, comma-joined, not {inst!r}"
    kinds = [one(x) for x in labels]
    if len(kinds) == 1:
        text = f"{kinds[0][0]} {kinds[0][1]}"
    elif kinds[0][0] == kinds[1][0]:
        text = f"{kinds[0][0]}s {kinds[0][1]} and {kinds[1][1]}"
    else:
        text = f"{kinds[0][0]} {kinds[0][1]} and {kinds[1][0]} {kinds[1][1]}"
    return f"{device}, {sms} SMs, {text}"


def short(name: str) -> str:
    """A row's name as the table words it: the name after its prefix, the layout and the probe, and the trace's suffix."""
    rest = name.removeprefix(PREFIX)
    layout, probes, kind = "", [], "plain"
    for suffix, words in LAYOUTS:
        if suffix in rest:
            layout, rest = words, rest.replace(suffix, "")
            break
    for suffix, words in ((COLUMN, "the box's width word choosing its fold"), (CLEARED, "its tiles cleared"),
                          (COUNTED, "a group's steps counted at run time"), (LOOPED, "a group's steps a voted loop on its count"),
                          (SPREAD, "an entry's dense line from a 2 GB table of the tokens"), (TABLED, "an entry's dense line from the tokens' table"),
                          (AHEAD, "loaded a window ahead"),
                          (LISTED, "the readout's passes its packet's"), (READY, "its tiles and lines offsets ready to add"),
                          (PADDED, "a half in two padded passes"), (FAR, "the drain's lines loaded a pass ahead"),
                          (SCAN, "its drain warps also listing the window's readers from the mask"),
                          (ROOM, "registers 176 / 104 / 56"), (EARLY, "the mask loaded a window ahead"),
                          (COUNTS, "the live tokens before each word handed with it"),
                          (ONE_BLOCK, "every CTA on one block's reader packets (not checked)"), (SPLIT, "registers 176 / 112 / 48"),
                          (IDLE, "the read side idle (not checked)")):
        if suffix in rest:
            probes.append(words)
            rest = rest.replace(suffix, "")
    probe = ", ".join(probes)
    if SPLIT_READ in rest:
        kind = "coarse, the waits on the tile groups published and the panels drained, the readout's pass by phases"
        rest = rest.replace(COARSE_SUF, "").replace(SPLIT_READ, "")
        for suffix, _ in DETAILS[1:4:2]:
            rest = rest.replace(suffix, "")
    for suffix, words in DETAILS:
        if suffix in rest:
            kind, rest = f"coarse, {words}", rest.replace(COARSE_SUF, "").replace(suffix, "")
    for suffix, words in ((EDGES, "the launch's two edges stamped"), (COARSE_SUF, "coarse"), (FINE, "every warp's timeline stamped")):
        if suffix in rest:
            kind, rest = words, rest.replace(suffix, "")
    assert not rest, f"lab page: the row {name!r} has a part the generator does not name: {rest!r}"
    parts = [x for x in (layout, probe if probe and kind == "plain" else "", kind, probe if probe and kind != "plain" else "") if x]
    return ", ".join(parts)


def write(root: pathlib.Path, instance: str) -> list[pathlib.Path]:
    sys.path.insert(0, str(root / "tools"))
    import lab_trace

    folder = root / "measure/lab/results/carry_token"
    recs = {n: json.loads((folder / f"sm_90a.blocks{n}.json").read_text()) for n in (1, 132)}
    one, full = recs[1], recs[132]
    for n, r in recs.items():
        assert r["blocks"] == n, f"lab page: the record named for {n} CTAs is of {r['blocks']}"
        assert all(c["ok"] for c in r["checks"] if "ok" in c), f"lab page: the {n}-CTA record holds a check that failed"
    assert one["git"] == full["git"], "lab page: the two records are of different commits"
    assert one["toolkit"] == full["toolkit"], "lab page: the two records are of different toolkits"
    assert one["reps"] == full["reps"] and one["driver"] == full["driver"] and one["device"] == full["device"], (
        "lab page: the two records differ in repetitions, driver or device"
    )
    sms = full["blocks"]
    rows = {n: Record({r["name"]: r["median"] for r in rec["results"] if r["unit"] == UNIT}) for n, rec in recs.items()}
    other = {
        n: Record({r["name"]: r["median"] for r in rec["results"] if r["unit"] != UNIT}) for n, rec in recs.items()
    }
    names = list(rows[132])
    assert list(rows[1]) == names, "lab page: the two records hold different rows"
    traces = {n: dict(lab_trace.load(folder / f"sm_90a.blocks{n}.trace")) for n in (1, 132)}

    def plain_of(name):
        return lab_trace.untraced_name(name)

    def twin(name, n):
        """a traced row's plain twin, by name"""
        t = plain_of(name)
        assert t in rows[n], f"lab page: the traced row {name!r} has no plain twin {t!r} in the record"
        return rows[n][t]

    def w(n, name):
        return rows[n][name]

    def g(x):
        return f"{x:,.0f}"

    PLAIN = PREFIX
    COARSE = PREFIX + lab_trace.COARSE_SUFFIX
    for n in (1, 132):
        assert PLAIN in rows[n] and COARSE in rows[n], "lab page: the record holds no plain or no coarse row"

    # ---- the table: every row of the record, at one CTA and at the full grid, in the record's order
    table = "\n".join(
        ["| row | one CTA | 132 CTAs |", "|---|---:|---:|"]
        + [f"| {short(nm)} | {g(w(1, nm))} | {g(w(132, nm))} |" for nm in names]
    )
    checked = sum(1 for c in full["checks"] if "ok" in c)
    isolation = [c["variant"] for c in full["checks"] if "ok" not in c and "isolation" in c.get("reason", "")]
    one_block = [c["variant"] for c in full["checks"] if "ok" not in c and "reads block 0's reader packets" in c.get("reason", "")]
    assert checked == sum(1 for c in one["checks"] if "ok" in c)
    assert isolation == [c["variant"] for c in one["checks"] if "ok" not in c and "isolation" in c.get("reason", "")]
    assert one_block == [c["variant"] for c in one["checks"] if "ok" not in c and "reads block 0's reader packets" in c.get("reason", "")]
    assert all(v.endswith(ONE_BLOCK) for v in one_block), "lab page: an unchecked one-block row does not end with its suffix"
    # a rule: every row of the record is checked or named unchecked, and the record's checks are its own
    check_names = {c["variant"] for c in full["checks"]}
    assert all(nm in check_names for nm in names), "lab page: a row of the record has neither a check nor an unchecked mark"

    def exact(name):
        found = [c for c in full["checks"] if c["variant"] == name and "ok" in c]
        return bool(found) and found[0]["ok"] and all(
            c["ok"] for c in one["checks"] if c["variant"] == name and "ok" in c
        )

    # ---- one CTA against the full grid
    s1, word1 = against(w(1, PLAIN), w(132, PLAIN))
    if word1 == "the same as":
        grid = (
            f"THE WINDOW AT ONE CTA IS THE WINDOW AT THE FULL GRID ({g(w(1, PLAIN))} and {g(w(132, PLAIN))}, {pct(abs(s1))} apart; "
            f"called the same within {pct(SAME)}): {sms} CTAs each landing two packets a window do not bind on global memory at this cell."
        )
    else:
        grid = (
            f"THE WINDOW AT ONE CTA IS {pct(abs(s1))} {word1} THE WINDOW AT THE FULL GRID ({g(w(1, PLAIN))} and {g(w(132, PLAIN))}; "
            f"called the same within {pct(SAME)}): {sms} CTAs each landing two packets a window {'do not bind' if s1 < 0 else 'bind'} "
            "on global memory at this cell by that much."
        )

    # ---- the layout: the overlapping rows against the plain and the coarse rows, by name
    lay_plain, lay_coarse = PLAIN + SHARED, COARSE.replace(PREFIX, PREFIX + SHARED)
    for n in (1, 132):
        assert lay_plain in rows[n] and lay_coarse in rows[n]
        assert plain_of(lay_coarse) == lay_plain
    lay_parts = []
    for n, head in ((1, "one CTA"), (132, "132 CTAs")):
        s, word = against(w(n, lay_plain), w(n, PLAIN))
        lay_parts.append(
            f"at {head} {g(w(n, lay_plain))} against {g(w(n, PLAIN))}"
            + (": the same" if word == "the same as" else f", {pct(abs(s))} {word}")
        )
    lay_same = all(against(w(n, lay_plain), w(n, PLAIN))[1] == "the same as" for n in (1, 132))
    lay_exact = exact(lay_plain) and exact(lay_coarse)
    layout = (
        f"THE LAYOUT IS THE GATHER'S: lists overlapping on tokens written once {'cost the carry nothing' if lay_same else 'cost the carry something'}"
        f" ({'; '.join(lay_parts)}; called the same within {pct(SAME)}) and {'are exact' if lay_exact else 'are NOT exact'}"
        " (the checks of both overlapping rows, at both grids)."
    )

    # ---- the traced rows, at the full grid: phases (warpline's windows), the launch, a wait's length
    def cta0(n, name):
        rec = traces[n][name]
        scope = rec.scopes()[0]
        return rec, scope, lab_trace.Cta(rec, scope)

    def fold_phases(n, name):
        """the fold's phases, by name: duration, the mean of its two warpgroups' in the first CTA's own window; the
        window's cycles; the warpgroups counted"""
        rec, scope, cta = cta0(n, name)
        wins = lab_trace.wl_window.role_windows(rec, scope, lab_trace.WINDOW, SKIP)
        folds = [rw for rw in wins if rw.role.startswith("fold")]
        assert folds, f"lab page: the record of {name!r} holds no fold's windows"
        sums: dict[str, float] = {}
        order: list[str] = []
        for rw in folds:
            for r in rw.rows:
                key = r.name.removeprefix("phase.")
                assert not any(x.name == r.name and x is not r for x in rw.rows), f"a phase of {name!r} repeats in the fold"
                if key not in sums:
                    order.append(key)
                sums[key] = sums.get(key, 0.0) + r.dur
        return {k: sums[k] / len(folds) for k in order}, sum(rw.period for rw in folds) / len(folds), len(folds), scope

    def side_phases(n, name, role):
        """a role's phases in the first CTA's own window: (start, duration, name) each, in order"""
        rec, scope, cta = cta0(n, name)
        wins = lab_trace.wl_window.role_windows(rec, scope, lab_trace.WINDOW, SKIP)
        out = [rw for rw in wins if rw.role == role]
        assert out, f"lab page: the record of {name!r} holds no {role}'s windows"
        return [(r.start, r.dur, r.name.removeprefix("phase.")) for r in out[0].rows], out[0].period

    def launch(n, name):
        rec, scope, cta = cta0(n, name)
        out = cta.launch()
        assert out is not None, f"lab page: the traced row {name!r} holds no launch"
        return out, scope.removeprefix("cta")

    ph, period, folds_n, scope132 = fold_phases(132, COARSE)
    scope132 = scope132.removeprefix("cta")
    assert set(ph) == {"window", "expanding", "issuing steps"}, f"lab page: the coarse row's fold phases are {sorted(ph)}"
    image, fill, steps = ph["window"], ph["expanding"], ph["issuing steps"]
    total = image + fill + steps
    shares = {"image": image / total, "fill": fill / total, "steps": steps / total}
    longest = max(shares, key=shares.get)
    shortest = min(shares, key=shares.get)
    thirds = (
        "the three are near thirds of it"
        if max(shares.values()) <= 0.42 and min(shares.values()) >= 0.25
        else f"the {longest} phase is the longest and the {shortest} phase the shortest"
    )
    fold_text = (
        f"THE FOLD'S WINDOW, from the coarse row's record at 132 CTAs (`bazel run //tools:lab -- trace carry_token`; CTA {scope132}, "
        f"the mean of its {folds_n} fold warpgroups, a window of {g(period)} cycles in their own windows): its image {g(image)} cycles "
        f"({pct(shares['image'])}), its fill {g(fill)} ({pct(shares['fill'])}), its 32 steps {g(steps)} ({pct(shares['steps'])}); {thirds}"
        f" (the longest phase is {max(shares.values()) / min(shares.values()):.2f} times the shortest)."
    )

    # the read side's phases, in its own windows
    read_ph, read_period = side_phases(132, COARSE, "readout")
    drain_ph, drain_period = side_phases(132, COARSE, "drain")

    def role_line(role, phases_, per):
        body = "; ".join(f"{nm} {g(d)} from {g(s)}" for s, d, nm in phases_)
        return f"{role}, its phases in its own window of {g(per)} cycles (duration from start): {body}"

    # the read side's cost to the fold: the coarse row against the same row with the read side idle, by name
    idle = COARSE.replace(PREFIX, PREFIX + IDLE)
    assert idle in rows[132] and idle.replace(IDLE, "") == COARSE
    idle_ph, _, _, _ = fold_phases(132, idle)
    cost = (w(132, COARSE) - w(132, idle)) / w(132, COARSE)
    cost1 = (w(1, COARSE) - w(1, idle)) / w(1, COARSE)
    steps_idle = idle_ph["issuing steps"]
    read_side = (
        f"THE READ SIDE COSTS THE FOLDS {pct(cost)} OF THE WINDOW at {sms} CTAs ({pct(cost1)} at one CTA): the coarse row's window is"
        f" {g(w(132, COARSE))} cycles and the same kernel with the read side idle {g(w(132, idle))} (the isolation row, not checked: its"
        f" kernel is not the carry's); the fold's 32 steps are {g(steps)} cycles and, with it idle, {g(steps_idle)} ({vs(steps_idle, steps, 'the coarse row')})."
    )

    # the launch's own clocks: the rate between the cycle counter and the nanosecond timer, and a window's time
    lc, ctal = launch(132, COARSE)
    lc1, ctal1 = launch(1, COARSE)
    rate = lc["rate"]
    window_us = w(132, PLAIN) / rate
    windows_a_cta = TOKENS / (TOKENS_A_WINDOW * sms)
    ms = windows_a_cta * w(132, PLAIN) / rate / 1000.0
    clock_text = (
        f"BY THE LAUNCH'S OWN CLOCKS the coarse row's launch at {sms} CTAs (CTA {ctal}) is {g(lc['cycles'])} cycles in {g(lc['ns'])} ns:"
        f" the cycle counter runs {rate:,.0f} cycles a microsecond ({lc1['rate']:,.0f} at one CTA, CTA {ctal1}); the plain row's window"
        f" of {g(w(132, PLAIN))} cycles is {window_us:.2f} microseconds, and 65,536 tokens ({windows_a_cta:.0f} windows a CTA, a window"
        f" counted as {TOKENS_A_WINDOW} tokens a CTA as `carry_window`'s wall-clock rows count it) some {ms:.2f} ms."
    )

    # the trace's cost over the plain row, by name (the coarse row's twin is its name with the suffix removed)
    coarse_over = (w(132, COARSE) - twin(COARSE, 132)) / twin(COARSE, 132)
    marks_text = (
        f"THE MARKS' COST: the coarse row's window is {g(w(132, COARSE))} against the plain row's {g(twin(COARSE, 132))} at {sms} CTAs"
        f" ({pct(coarse_over)} over, its phases' starts and the launch's edges on both clocks); the figures above are the coarse"
        " row's."
    )

    # ---- the grouping rows, by name: what a layout and a step's shape decide
    def cycles_of(n, layout, idle=False, column=False):
        nm = PREFIX + layout + (COLUMN if column else "") + ((IDLE + COARSE_SUF) if idle else "")
        return rows[n][nm]

    list4, group4, wide16 = PREFIX + TOKEN4_LIST, PREFIX + TOKEN4, PREFIX + TOKEN16
    col_list = cycles_of(132, TOKEN4_LIST, column=True)
    col_group = cycles_of(132, TOKEN4, column=True)
    exact_group = all(
        exact(PREFIX + lay + (COLUMN if col else ""))
        for lay, col in ((TOKEN4_LIST, True), (TOKEN4, True), (TOKEN4_LIST, False), (TOKEN4, False), (TOKEN16, False), (GROUP4, False))
    ) and exact(PREFIX + SPLIT)
    col_free = col_list <= w(132, list4) * (1 + SAME) and col_group <= w(132, group4) * (1 + SAME)
    col_text = (
        f"(a) A STEP SHAPE CHOSEN BEFORE THE WINDOW LOOP {'IS FREE' if col_free else 'IS NOT FREE'}: the per-column kernel (one kernel "
        "holding both folds, the box's width word read before its window loop) is, at 132 CTAs, "
        f"{g(col_list)} on per-list packets against the per-list kernel's {g(w(132, list4))} ({vs(col_list, w(132, list4))}; at one CTA "
        f"{g(cycles_of(1, TOKEN4_LIST, column=True))} against {g(w(1, list4))}) and {g(col_group)} on group-of-four packets against the "
        f"group-of-four kernel's {g(w(132, group4))} ({vs(col_group, w(132, group4))}; at one CTA {g(cycles_of(1, TOKEN4, column=True))} against "
        f"{g(w(1, group4))}), {'exact' if exact_group else 'NOT exact'}."
    )
    sixteen = (
        f"(b) A GROUP WIDER THAN FOUR DIGITS NEEDS NO STEP SHAPE OF ITS OWN: a token on all 16 digits of its box, executed as groups of four on "
        f"one run of value lines, is {g(w(132, wide16))} cycles against the four-digit row's {g(w(132, group4))} ({vs(w(132, wide16), w(132, group4))}; at one CTA "
        f"{g(w(1, wide16))} against {g(w(1, group4))}), {'exact' if exact(wide16) else 'NOT exact'}."
    )
    g4, g4idle = w(132, PREFIX + GROUP4), cycles_of(132, GROUP4, idle=True)
    plain_idle = rows[132][PREFIX + IDLE + COARSE_SUF]
    t4g, t4gi = w(132, group4), cycles_of(132, TOKEN4, idle=True)
    t4l, t4li = w(132, list4), cycles_of(132, TOKEN4_LIST, idle=True)
    split = w(132, PREFIX + SPLIT)
    waste = (
        f"(c) WHERE A TOKEN IS ON ONE DIGIT OF ITS GROUP THE 64-COLUMN STEP IS WASTE, AND ON FOUR IT IS THE FOLD'S GAIN: groups of four digits "
        f"(a token on one digit) are {g(g4)} cycles against the per-list row's {g(w(132, PLAIN))} ({vs(g4, w(132, PLAIN))}), their folds alone {g(g4idle)} against "
        f"the per-list folds' {g(plain_idle)} ({vs(g4idle, plain_idle)}), and the per-list fold at the wide kernel's registers (176 / 112 / 48) "
        f"{g(split)} ({vs(split, w(132, PLAIN), 'the per-list row')}: the registers are not the excess). A token on all four digits of its group: the folds "
        f"alone are {g(t4gi)} against {g(t4li)} a list a digit ({vs(t4gi, t4li)}), and the composed window {g(t4g)} against {g(t4l)} ({vs(t4g, t4l)}): "
        "the fold's gain is paid in the composed window."
    )
    run_time = "(d) THE PADDED FORM IS NO ROW HERE: a fold that always issues a group's two steps with the absent one on zeros is not measured (see (f))."

    # ---- a fold's issue work: the cleared four-digit fold against the same fold without the clear
    cleared = TOKEN4 + CLEARED
    clr = {n: (cycles_of(n, cleared), cycles_of(n, cleared, idle=True), w(n, group4), cycles_of(n, TOKEN4, idle=True)) for n in (1, 132)}
    diff = {n: (c[0] - c[2], c[1] - c[3]) for n, c in clr.items()}
    pair_text = (
        f"(e) A FOLD'S ISSUE WORK, A PAIR OF ROWS: the four-digit grouped fold with its tiles' room cleared and the clear's barrier ({g(clr[132][0])} composed, "
        f"{g(clr[132][1])} with the read side idle) against the same fold without them ({g(clr[132][2])} and {g(clr[132][3])}) is "
        f"{g(diff[132][0])} cycles more composed and {g(diff[132][1])} more with the read side idle, {diff[132][0] / diff[132][1]:.2f} times; at one CTA "
        f"{g(diff[1][0])} and {g(diff[1][1])}, {diff[1][0] / diff[1][1]:.2f} times. The clear is shared-memory stores and a warpgroup barrier, which every "
        "entry of that fold's tiles makes needless: they cost more composed than alone. The record names no cause (it holds no counter)."
    )
    step1, step2 = TOKEN4_TILES + CLEARED + COUNTED, TOKEN2 + CLEARED + COUNTED
    cnt = {n: (cycles_of(n, step1), cycles_of(n, step1, idle=True), cycles_of(n, step2), cycles_of(n, step2, idle=True)) for n in (1, 132)}
    choice = {n: (c[0] - clr[n][0], c[1] - clr[n][1]) for n, c in cnt.items()}
    second = {n: (c[2] - c[0], c[3] - c[1]) for n, c in cnt.items()}
    exact_counted = exact(PREFIX + step1) and exact(PREFIX + step2) and exact(PREFIX + cleared)
    counted_text = (
        "(f) A GROUP'S STEPS COUNTED AT RUN TIME: the count is a byte a group in the packet's header, and a group's steps are a choice between two "
        "straight blocks (one step or two, each its own fence and commit group, the group's accumulators held across the choice) made outside them, "
        "in the group's own two places of the tiles' room and of the header's starts; any other count traps. (A loop on the loaded count, each thread deciding "
        "for itself, was serialized by ptxas 13.5, C7519 \"serialized due to program dependence on compiler-inserted WG.AR in divergent path\", which the "
        f"lab's checks refuse: a compile fact of this kernel; (g)'s loop, whose test is a warp vote, is not.) At one step a group the counted fold is {g(cnt[132][0])} composed and {g(cnt[132][1])} with the read side idle "
        f"against the cleared fixed-block fold's {g(clr[132][0])} and {g(clr[132][1])}: {g(choice[132][0])} more composed and {g(choice[132][1])} more alone, "
        f"{choice[132][0] / choice[132][1]:.2f} times (at one CTA {g(choice[1][0])} and {g(choice[1][1])}, {choice[1][0] / choice[1][1]:.2f} times): "
        f"not the clear's {diff[132][0] / diff[132][1]:.2f}, so the two kinds of issue work are not one price. At two steps a group (a token on two digits of its group) "
        f"it is {g(cnt[132][2])} and {g(cnt[132][3])}, {g(second[132][0])} more composed and {g(second[132][1])} more alone than the one-step rows (at one CTA "
        f"{g(second[1][0])} and {g(second[1][1])}); the rows are {'exact' if exact_counted else 'NOT exact'}. What follows from those two differences alone: "
        "for a count of one or two, the choice costs about what a second step does. NOT MEASURED: a fold that always issues two steps a group with the "
        "absent one on zeros (counts over two a group are (g)'s)."
    )

    # ---- a group's steps as a voted loop on its count (any count)
    loop4, loop2, loopu = TOKEN4 + CLEARED + LOOPED, TOKEN2 + CLEARED + LOOPED, TOKEN8 + CLEARED + LOOPED
    lp = {n: (cycles_of(n, loop4), cycles_of(n, loop4, idle=True), cycles_of(n, loop2), cycles_of(n, loopu), cycles_of(n, loopu, idle=True)) for n in (1, 132)}
    vs_fixed = {n: (lp[n][0] - clr[n][0], lp[n][1] - clr[n][1]) for n in lp}
    vs_choice = {n: (lp[n][0] - cnt[n][0], lp[n][1] - cnt[n][1]) for n in lp}
    vs_two = {n: (lp[n][2] - cnt[n][2]) for n in lp}
    exact_loop = all(exact(PREFIX + x) for x in (loop4, loop2, loopu))
    loop_text = (
        "(g) A GROUP'S STEPS AS A VOTED LOOP ON ITS COUNT: a loop a group on the packet's count whose test is a warp vote, a step's tile and start the "
        "window's next in the order issued: any count. On the four-digit packets (one step a group) it is "
        f"{g(lp[132][0])} composed and {g(lp[132][1])} with the read side idle against the cleared fixed block's {g(clr[132][0])} and {g(clr[132][1])}: "
        f"{g(vs_fixed[132][0])} more composed and {g(vs_fixed[132][1])} more alone (at one CTA {g(vs_fixed[1][0])} and {g(vs_fixed[1][1])}); against the "
        f"choice of straight blocks on the same four-digit tokens (the counted one-step row, {g(cnt[132][0])} and {g(cnt[132][1])}) it is "
        f"{g(vs_choice[132][0])} more composed and {g(vs_choice[132][1])} more alone (at one CTA {g(vs_choice[1][0])} and {g(vs_choice[1][1])}). "
        f"On a token on two digits of its group it is {g(lp[132][2])} against the counted two-step row's {g(cnt[132][2])} "
        f"({g(vs_two[132])} more; at one CTA {g(lp[1][2])} against {g(cnt[1][2])}, {g(vs_two[1])} more). "
        f"On the packet whose groups take four, two, one and one steps (a token on one digit of the first group, on two of the second, on all four of the "
        f"third and the fourth) it is {g(lp[132][3])} composed and {g(lp[132][4])} with the read side idle (at one CTA {g(lp[1][3])} and {g(lp[1][4])}). "
        f"Every row is {'exact, the uneven one included' if exact_loop else 'NOT exact'}. A compile fact of this kernel: ptxas 13.5 does not serialize the voted loop "
        "(no C7519), and its steps are rolled, which the lab's `step_rolled` rule names and the rows declare ([`step_rolled`](step_rolled.md) prices a rolled "
        "16-column step alone). NOT MEASURED: a window that needs more steps than the tiles' room holds in one pass (eight a box), a second pass, an outer "
        "loop on the count of passes; what carries a rolled step's cost (the vote, the branch, the run-time operands, the overlap between a group's steps)."
    )

    # ---- who reads a membership's dense row: the packet's packed line, or the fold's own load from the tokens' table
    def tab(n, layout, table, idle=False, ahead=False):
        return rows[n][PREFIX + layout + table + (AHEAD if ahead else "") + ((IDLE + COARSE_SUF) if idle else "")]

    packed_g = {n: (w(n, group4), cycles_of(n, TOKEN4, idle=True)) for n in (1, 132)}
    packed_l = {n: w(n, PLAIN) for n in (1, 132)}
    tg = {(n, t): (tab(n, TOKEN4, t), tab(n, TOKEN4, t, idle=True)) for n in (1, 132) for t in (TABLED, SPREAD)}
    tl = {(n, t): tab(n, "", t) for n in (1, 132) for t in (TABLED, SPREAD)}
    exact_tab = all(exact(PREFIX + lay + t) for lay in (TOKEN4, "") for t in (TABLED, SPREAD))

    def over(n, t):
        return tg[(n, t)][0] - packed_g[n][0], tg[(n, t)][1] - packed_g[n][1], tl[(n, t)] - packed_l[n]

    ah = {(n, t): (tab(n, TOKEN4, t, ahead=True), tab(n, TOKEN4, t, idle=True, ahead=True)) for n in (1, 132) for t in (TABLED, SPREAD)}
    al = {(n, t): tab(n, "", t, ahead=True) for n in (1, 132) for t in (TABLED, SPREAD)}
    exact_ahead = all(exact(PREFIX + lay + t + AHEAD) for lay in (TOKEN4, "") for t in (TABLED, SPREAD))
    grow = {n: (tg[(n, SPREAD)][0] - tg[(n, TABLED)][0], tl[(n, SPREAD)] - tl[(n, TABLED)]) for n in (1, 132)}
    table_text = (
        "(h) WHO READS A MEMBERSHIP'S DENSE ROW: the packet's packed 16-byte line, or the fold's own load from the tokens' table by a two-byte index "
        "in the packet (two chained global loads an entry at the window's start; the rows 16 bytes; the table 1 MB, or the same rows 32 KB apart, 2 GB). "
        f"The four-digit grouped fold on the table is {g(tg[(132, TABLED)][0])} composed and {g(tg[(132, TABLED)][1])} with the read side idle against the packed "
        f"line's {g(packed_g[132][0])} and {g(packed_g[132][1])}: {g(over(132, TABLED)[0])} and {g(over(132, TABLED)[1])} more (at one CTA {g(over(1, TABLED)[0])} and "
        f"{g(over(1, TABLED)[1])}); on the 2 GB table {g(tg[(132, SPREAD)][0])} and {g(tg[(132, SPREAD)][1])}, {g(over(132, SPREAD)[0])} and {g(over(132, SPREAD)[1])} "
        f"more (at one CTA {g(over(1, SPREAD)[0])} and {g(over(1, SPREAD)[1])}). The per-list fold on the table is {g(tl[(132, TABLED)])} against the packed "
        f"line's {g(packed_l[132])}, {g(over(132, TABLED)[2])} more (at one CTA {g(over(1, TABLED)[2])}); on the 2 GB table {g(tl[(132, SPREAD)])}, "
        f"{g(over(132, SPREAD)[2])} more (at one CTA {g(over(1, SPREAD)[2])}). Every row is {'exact' if exact_tab and exact_ahead else 'NOT exact'}. Two facts, and nothing beyond them: "
        "the table's own load is not free even from a cached table (the 1 MB rows above); and it grows with the table's size by the DISTINCT rows a packet names "
        f"(the four-digit packet's 256 entries a box name 64 tokens, the per-list packet's name 256): the 2 GB table over the 1 MB one costs the grouped fold "
        f"{g(grow[132][0])} cycles and the per-list fold {g(grow[132][1])}, {grow[132][1] / grow[132][0]:.1f} times (at one CTA {g(grow[1][0])} and {g(grow[1][1])}, {grow[1][1] / grow[1][0]:.1f} times). "
        f"THE ROWS LOADED A WINDOW AHEAD (behind the entries of the window before; the fold then holds the next window's two lines through its steps) "
        f"against the same rows loaded at the window's start: the grouped fold on the table {g(ah[(132, TABLED)][0])} composed and {g(ah[(132, TABLED)][1])} with the read "
        f"side idle against {g(tg[(132, TABLED)][0])} and {g(tg[(132, TABLED)][1])} (at one CTA {g(ah[(1, TABLED)][0])} and {g(ah[(1, TABLED)][1])} against "
        f"{g(tg[(1, TABLED)][0])} and {g(tg[(1, TABLED)][1])}); on the 2 GB table {g(ah[(132, SPREAD)][0])} and {g(ah[(132, SPREAD)][1])} against "
        f"{g(tg[(132, SPREAD)][0])} and {g(tg[(132, SPREAD)][1])} (at one CTA {g(ah[(1, SPREAD)][0])} and {g(ah[(1, SPREAD)][1])} against "
        f"{g(tg[(1, SPREAD)][0])} and {g(tg[(1, SPREAD)][1])}). The per-list fold on the table {g(al[(132, TABLED)])} against {g(tl[(132, TABLED)])} "
        f"(at one CTA {g(al[(1, TABLED)])} against {g(tl[(1, TABLED)])}); on the 2 GB table {g(al[(132, SPREAD)])} against {g(tl[(132, SPREAD)])} "
        f"(at one CTA {g(al[(1, SPREAD)])} against {g(tl[(1, SPREAD)])}). The window-ahead kernels run at 176 / 112 / 48 registers a thread and the "
        f"tabled per-list rows at 168 / 120 / 56: the per-list fold at 176 / 112 / 48 with its packed lines is {g(w(132, PREFIX + SPLIT))} "
        f"(at one CTA {g(w(1, PREFIX + SPLIT))}), so (a compile fact of this kernel, observed when the rows were built: the lab's checks refused the 168-register build of the per-list window-ahead rows for register spills) the per-list window-ahead rows are {g(al[(132, TABLED)] - w(132, PREFIX + SPLIT))} and "
        f"{g(al[(132, SPREAD)] - w(132, PREFIX + SPLIT))} over that row, and {g(al[(132, TABLED)] - tl[(132, TABLED)])} over and "
        f"{g(abs(al[(132, SPREAD)] - tl[(132, SPREAD)]))} {'over' if al[(132, SPREAD)] >= tl[(132, SPREAD)] else 'under'} the tabled rows at the window's start. What the figures show and nothing beyond: loaded a window ahead, the rows took nothing off the window in these rows, and are a little more where "
        "they differ. WHY is OPEN: the form changes the load's timing, the next window's two lines held live through the steps and, for the per-list rows, the "
        "register split, and the record holds no counter, so what the table load's cost is paid on is not named; it is owed ([`open-work.md`](../open-work.md)). "
        "NOT MEASURED: what the gather pays to write a packed 16-byte line a membership in its current form ([`gather_group`](gather_group.md)'s row is values "
        "only, and writes the entries' token indices, no dense line), so the two forms are not compared end to end here."
    )

    # ---- (i) the readout's passes its packet's: the listed readout, the ready packet and the padded packets, against the fixed readout
    fixed_n = PREFIX + TOKEN4
    listed_n, ready_n = fixed_n + LISTED, fixed_n + LISTED + READY
    padded_n, ready_padded_n = listed_n + PADDED, ready_n + PADDED
    forms = {"fixed": fixed_n, "listed": listed_n, "ready": ready_n, "listed padded": padded_n, "ready padded": ready_padded_n}
    fm = {k: {n: w(n, nm) for n in (1, 132)} for k, nm in forms.items()}
    exact_forms = all(exact(nm) for nm in forms.values())
    gap = {k: (v[132] - v[1]) / v[1] for k, v in fm.items()}

    def more(a, b):
        return f"{g(abs(a - b))} cycles, {pct(abs(a - b) / b)} {'over' if a > b else 'under'}"

    assert exact_forms, "lab page: a row of the listed readout is not exact"
    assert abs(gap["fixed"]) <= SAME < gap["listed"] < gap["ready"] < gap["ready padded"], (
        "lab page: the gaps between the grids are not the paragraph (i)'s order: " + str(gap))
    assert against(fm["ready"][1], fm["listed"][1])[0] < -SAME and against(fm["ready"][132], fm["listed"][132])[1] == "the same as", (
        "lab page: the ready offsets are not a gain at one CTA and none at 132")
    listed_text = (
        "(i) THE READOUT'S PASSES ITS PACKET'S: the readout's slots take their read digit, their readers, the readers' bytes and their output lines from a "
        "packet in global memory, and a half's passes are a loop on the packet's count (the four-digit grouped fold's window, the host playing the gather). "
        f"At one CTA and at {sms} CTAs: the fixed readout {g(fm['fixed'][1])} and {g(fm['fixed'][132])}; the listed readout on the fixed readout's own readers "
        f"{g(fm['listed'][1])} and {g(fm['listed'][132])}; the same with the packet's offsets ready to add (a slot's tile of the image in 16-byte lines, a token's "
        f"output line in bytes) {g(fm['ready'][1])} and {g(fm['ready'][132])}; the listed readout with a half in two passes of four slots and four pads "
        f"{g(fm['listed padded'][1])} and {g(fm['listed padded'][132])}, and the ready form on those padded packets {g(fm['ready padded'][1])} and "
        f"{g(fm['ready padded'][132])}; every row {'exact' if exact_forms else 'NOT exact'}. THE LISTED FORM OVER THE FIXED READOUT: at {sms} CTAs "
        f"{more(fm['listed'][132], fm['fixed'][132])}, at one CTA {more(fm['listed'][1], fm['fixed'][1])}. THE READY FORM AGAINST THE LISTED: at {sms} CTAs "
        f"{more(fm['ready'][132], fm['listed'][132])} (called {against(fm['ready'][132], fm['listed'][132])[1].removesuffix(' as')} within {pct(SAME)}), at one CTA "
        f"{more(fm['ready'][1], fm['listed'][1])}; on the padded packets, at {sms} CTAs {more(fm['ready padded'][132], fm['listed padded'][132])}, at one CTA "
        f"{more(fm['ready padded'][1], fm['listed padded'][1])}. THE PADDED PACKETS: the listed form with a half in two passes is {more(fm['listed padded'][132], fm['listed'][132])} "
        f"the one-pass listed form at {sms} CTAs ({more(fm['listed padded'][1], fm['listed'][1])} at one CTA), the ready form {more(fm['ready padded'][132], fm['ready'][132])} "
        f"at {sms} CTAs ({more(fm['ready padded'][1], fm['ready'][1])} at one CTA). THE GAP BETWEEN THE GRIDS, one CTA to {sms} (a window at {sms} CTAs over a window at one CTA): "
        f"the fixed readout {pct(gap['fixed'])}, the listed {pct(gap['listed'])}, the ready {pct(gap['ready'])}, the listed on padded packets {pct(gap['listed padded'])}, "
        f"the ready on padded packets {pct(gap['ready padded'])}. WHAT THE FIGURES SHOW AND NOTHING BEYOND: a readout driven by its packet is exact on both packets; "
        f"ready offsets are a gain at one CTA ({pct(abs(against(fm['ready'][1], fm['listed'][1])[0]))} under the listed form) and none at {sms} CTAs "
        f"({against(fm['ready'][132], fm['listed'][132])[1].removesuffix(' as')} the listed); the forms that load their packet from global memory in the read side's roles slow with the card's CTAs "
        f"(the gaps above) and the fixed readout does not ({pct(gap['fixed'])}, called the same within {pct(SAME)}). OPEN: WHY. The candidate: at one CTA a block's packets "
        f"stay cached between launches and at {sms} the card's buffers do not fit the cache; (j)'s one-block rows test it, and (j) states what they show. "
        "NOT BUILT: the read packet landed in shared memory by the bulk copy, as the fold's packet is ([`open-work.md`](../open-work.md) C-o)."
    )

    # ---- the two readouts' timelines: CTA 0 of each every-stamp row, each role's cycles a window a warp by state
    def role_states(n, name):
        rec, scope, cta = cta0(n, name)
        span = cta.steady(SKIP)
        assert span is not None, f"lab page: the record of {name!r} holds too few windows"
        t0, t1, nw = span
        by_role: dict[str, dict[str, float]] = {}
        members: dict[str, int] = {}
        for wp in cta.warps:
            members[wp.role] = members.get(wp.role, 0) + 1
            cta.clip(wp.index, t0, t1, by_role.setdefault(wp.role, defaultdict(float)))
        out = {}
        for role in ("readout", "drain"):
            out[role] = {lab.split(": ", 1)[1]: v / (members[role] * nw) for lab, v in by_role[role].items()}
        return (t1 - t0) / nw, out, scope.removeprefix("cta")

    far_n = listed_n + FAR
    timeline_rows = (
        (fixed_n + FINE, "the fixed readout"),
        (listed_n + FINE, "the listed readout"),
        (ready_n + FINE, "the ready packet's"),
        (far_n + FINE, "the listed readout with the drain's lines a pass ahead"),
    )
    timeline_parts = []
    drain_waits = []
    for nm, label in timeline_rows:
        assert nm in rows[132], f"lab page: the record holds no row {nm!r}"
        win, states, scope_ = role_states(132, nm)

        def line(role, keys):
            return ", ".join(f"{k} {g(states[role].get(k, 0.0))}" for k in keys)

        timeline_parts.append(
            f"{label}: the row's window {g(w(132, nm))} cycles ({g(win)} in CTA {scope_}'s steady span); the readout, in cycles a window a warp, "
            f"{line('readout', ('issuing steps', 'storing the panel', 'waits on panel # drained', 'waits on tile group # published', 'waits on its readout steps'))}; "
            f"the drain {line('drain', ('adding', 'waits on panel # filled'))}"
        )
        drain_waits.append(states["drain"].get("waits on panel # filled", 0.0))
    assert all(x > 0 for x in drain_waits)
    timelines_text = (
        "THE TWO READOUTS' TIMELINES (the four every-stamp rows of the four-digit grouped fold at "
        f"{sms} CTAs, CTA {scope132}; `bazel run //tools:lab -- trace carry_token --record measure/lab/results/carry_token/sm_90a.blocks{sms}.json --row \"<row>\"` "
        "with the rows' endings `groups of four, every warp's timeline stamped$`, `packet's, every warp's timeline stamped$`, `ready to add, every warp's timeline stamped$` and `a pass ahead, every warp's timeline stamped$`): "
        + "; ".join(timeline_parts) + ". What they show: in all four the drain waits on a panel the readout fills "
        f"({', '.join(g(x) for x in drain_waits)} cycles a window a warp, the fixed, the listed, the ready and the listed with its lines a pass ahead) and the readout waits on the panels the drain has drained "
        "(its own wait, above); how much of the window each waits is the figures above."
    )

    # ---- (j) the packets cached at the full grid; the drain's lines and the ready offsets a pass ahead
    cu_text = (root / "measure/lab/experiments/carry_token.cu").read_text()
    assert "(PROBE & (kReady | kFar)) == (kReady | kFar) || (PROBE & kRoom) != 0 ? 104" in cu_text and "(PROBE & (kReady | kFar)) == (kReady | kFar) || (PROBE & kRoom) != 0 ? 56" in cu_text, (
        "lab page: the ready-and-far kernel's registers are no longer 176 / 104 / 56")
    one_names = {k: nm + ONE_BLOCK for k, nm in forms.items() if k != "fixed"}
    far_names = {k: nm + FAR for k, nm in (("listed", listed_n), ("listed padded", padded_n), ("ready", ready_n), ("ready padded", ready_padded_n))}
    for nm in (*one_names.values(), *far_names.values()):
        assert nm in rows[132] and nm in rows[1], f"lab page: the record holds no row {nm!r}"
    ob = {k: {n: w(n, nm) for n in (1, 132)} for k, nm in one_names.items()}
    fr = {k: {n: w(n, nm) for n in (1, 132)} for k, nm in far_names.items()}
    assert all(exact(nm) for nm in far_names.values()), "lab page: a pass-ahead row is not exact"
    assert not any(one_names[k] not in one_block for k in ob), "lab page: a one-block row is checked"
    cached_same = all(against(ob[k][132], fm[k][132])[1] == "the same as" for k in ("listed", "listed padded"))
    assert cached_same, "lab page: a listed one-block row moved at the full grid"
    ob_text = (
        f"(j) THE PACKETS CACHED AT THE FULL GRID. The one-block rows: every CTA reads one block's reader packets, which the cache then holds at any grid; their outputs "
        f"are that block's readers' and the rows are not checked. At {sms} CTAs each against its own-packets row: the listed readout {g(ob['listed'][132])} against {g(fm['listed'][132])} "
        f"({more(ob['listed'][132], fm['listed'][132])}); the ready form {g(ob['ready'][132])} against {g(fm['ready'][132])} ({more(ob['ready'][132], fm['ready'][132])}); "
        f"the listed form on the padded packets {g(ob['listed padded'][132])} against {g(fm['listed padded'][132])} ({more(ob['listed padded'][132], fm['listed padded'][132])}); "
        f"the ready form on the padded packets {g(ob['ready padded'][132])} against {g(fm['ready padded'][132])} ({more(ob['ready padded'][132], fm['ready padded'][132])}). "
        f"WHAT THEY SHOW AND NOTHING BEYOND: with the packets cached at the full grid the listed row does not move ({g(ob['listed'][132])} against {g(fm['listed'][132])}, "
        f"{against(ob['listed'][132], fm['listed'][132])[1].removesuffix(' as')} within {pct(SAME)}; on the padded packets {g(ob['listed padded'][132])} against {g(fm['listed padded'][132])}, "
        f"{against(ob['listed padded'][132], fm['listed padded'][132])[1].removesuffix(' as')}), so the listed forms' growth from one CTA to {sms} ({pct(gap['listed'])}, "
        f"{pct(gap['listed padded'])} on the padded packets) is not their packets' loads being cold; the ready form's rows move by the figures stated "
        f"({more(ob['ready'][132], fm['ready'][132])}, {more(ob['ready padded'][132], fm['ready padded'][132])} on the padded packets)."
    )

    def adding(n, nm):
        win, states, _ = role_states(n, nm)
        return states["drain"]["adding"], states["drain"].get("waits on panel # filled", 0.0), win

    dl132, dl132w, _ = adding(132, listed_n + FINE)
    df132, df132w, _ = adding(132, far_n + FINE)
    dl1, dl1w, _ = adding(1, listed_n + FINE)
    assert df132 < dl132, "lab page: the pass-ahead drain is not less busy than the listed drain at the full grid"
    far_text = (
        f"THE DRAIN'S LINES A PASS AHEAD (the drain loads a group's lines a pass ahead, a set of lines a group of the pass, where the listed form loads them a group ahead): "
        f"the listed row against its pass-ahead row, at {sms} CTAs {g(fm['listed'][132])} and {g(fr['listed'][132])} ({more(fr['listed'][132], fm['listed'][132])}), at one CTA "
        f"{g(fm['listed'][1])} and {g(fr['listed'][1])} ({more(fr['listed'][1], fm['listed'][1])}); on the padded packets, at {sms} CTAs {g(fm['listed padded'][132])} and "
        f"{g(fr['listed padded'][132])} ({more(fr['listed padded'][132], fm['listed padded'][132])}), at one CTA {g(fm['listed padded'][1])} and {g(fr['listed padded'][1])} "
        f"({more(fr['listed padded'][1], fm['listed padded'][1])}); every row exact. From the every-stamp rows (CTA {scope132}, cycles a window a warp, the drain's \"adding\"): "
        f"the listed row at {sms} CTAs {g(dl132)}, the pass-ahead row at {sms} CTAs {g(df132)}, the listed row at one CTA {g(dl1)}: the pass-ahead drain's busy time at {sms} CTAs "
        f"is {g(abs(df132 - dl1))} cycles {'over' if df132 > dl1 else 'under'} the listed drain's at one CTA, {g(dl132 - df132)} under the listed drain's at {sms} CTAs "
        f"(its wait on a panel filled {g(df132w)} against {g(dl132w)} and, at one CTA, {g(dl1w)}), while the window moves by {more(fr['listed'][132], fm['listed'][132])} (the plain rows): the drain "
        "was stalled on its lines at the full grid and was not what bound the window."
    )
    ready_far_text = (
        f"READY OFFSETS A PASS AHEAD (registers 176 / 104 / 56 a thread, the folds, the readout, the drain: the readout's 8 registers a thread moved to the drain; no traced row, its stamped kernels spill): "
        f"at {sms} CTAs {g(fr['ready'][132])} against the listed row's {g(fm['listed'][132])} ({more(fr['ready'][132], fm['listed'][132])}), the ready form's {g(fm['ready'][132])} "
        f"({more(fr['ready'][132], fm['ready'][132])}) and the listed pass-ahead row's {g(fr['listed'][132])} ({more(fr['ready'][132], fr['listed'][132])}); the fixed readout's {g(fm['fixed'][132])} "
        f"({more(fr['ready'][132], fm['fixed'][132])}). At one CTA {g(fr['ready'][1])} against the listed row's "
        f"{g(fm['listed'][1])} ({more(fr['ready'][1], fm['listed'][1])}), the ready form's {g(fm['ready'][1])} ({more(fr['ready'][1], fm['ready'][1])}: slower than the ready form alone, "
        f"as measured, with 8 registers a thread moved from the readout to the drain) and the fixed readout's {g(fm['fixed'][1])} ({more(fr['ready'][1], fm['fixed'][1])}). "
        f"On the padded packets {g(fr['ready padded'][132])} and {g(fr['ready padded'][1])} (at {sms} CTAs and at one CTA) against the ready form's {g(fm['ready padded'][132])} and {g(fm['ready padded'][1])}. "
        f"OPEN: why the listed forms slow with the grid beyond the drain's stall (the one-block rows leave the listed form's {pct(gap['listed'])} from one CTA to {sms} where it was, and the pass-ahead row, which takes the drain's busy time at {sms} CTAs "
        f"to about its one-CTA figure, moves the window by {more(fr['listed'][132], fm['listed'][132])}); where the listed readout's instructions are (the static counts, above, are "
        "the compiled loops' sizes and no row prices them)."
    )
    scan_names = {k: far_names[k] + SCAN for k in ("ready", "ready padded")}
    for nm in scan_names.values():
        assert nm in rows[132] and nm in rows[1], f"lab page: the record holds no row {nm!r}"
        assert exact(nm), "lab page: a listing row is not exact"
        dets = [c["detail"] for c in full["checks"] + one["checks"] if c["variant"] == nm and "ok" in c]
        assert len(dets) == 2 and all(d.split("; ")[-1].startswith("0 of ") and d.endswith(" listed places differ") for d in dets), (
            f"lab page: a listing row's lists are not exact: {dets}")
    sc = {k: {n: w(n, nm) for n in (1, 132)} for k, nm in scan_names.items()}
    list_detail = {n: [c["detail"] for c in (full if n == 132 else one)["checks"] if c["variant"] == scan_names["ready"]][0].split("; ")[-1] for n in (1, 132)}
    assert all(sc[k][132] > sc[k][1] for k in sc), "lab page: a listing row does not cost more at the full grid than at one CTA"
    scan_text = (
        f"(k) THE CTA LISTING ITS OWN READERS (the ready offsets a pass ahead, whose drain warps also list the window's readers from the read mask: each drain warp its four read digits, "
        f"a lane its word of 32 tokens a digit, a scan of the lanes' counts, the live tokens' places compacted in order to shared memory and fenced). Each listing row against the same row "
        f"without the listing, cycles a window a CTA: at {sms} CTAs {g(sc['ready'][132])} against {g(fr['ready'][132])} ({more(sc['ready'][132], fr['ready'][132])}), at one CTA "
        f"{g(sc['ready'][1])} against {g(fr['ready'][1])} ({more(sc['ready'][1], fr['ready'][1])}); on the padded packets at {sms} CTAs {g(sc['ready padded'][132])} against "
        f"{g(fr['ready padded'][132])} ({more(sc['ready padded'][132], fr['ready padded'][132])}), at one CTA {g(sc['ready padded'][1])} against {g(fr['ready padded'][1])} "
        f"({more(sc['ready padded'][1], fr['ready padded'][1])}). Every row is exact, and the lists are exact: the lists are checked against the host's (the last window's, copied out at the launch's end) "
        f"and the checks' detail says \"{list_detail[132]}\" at {sms} CTAs and \"{list_detail[1]}\" at one CTA, on both listing rows. WHAT THIS IS AND IS NOT: it is the listing's cost on the drain warps of that kernel, at a mask of one reader a digit in every 64 tokens "
        "(the sparsest end; a denser mask is NOT MEASURED, and this lab's list area holds 16 places a read digit with the index wrapping, so a digit with more than 16 readers in a window would overwrite its list: the row is valid only at the lab's 16 a digit); the lists are not consumed, the readout still takes its packet's, so the readout reading them from shared memory is not in the row; "
        "a reader's byte on its digit is not loaded (the lab computes it). NOT PARTED: in this row the mask's words are loaded and used at once at the window's start (the tuned row below loads them a window ahead), so the row's cost is not parted between the mask's load "
        f"and the scan and the emission; and the row costs more at {sms} CTAs than at one CTA ({more(sc['ready'][132], sc['ready'][1])} on the packets, {more(sc['ready padded'][132], sc['ready padded'][1])} on the padded packets), "
        "a difference no row here explains."
    )
    # ---- (k) tuned: the ready kernel at the pass-ahead kernel's registers, and its drain warps listing from a mask loaded ahead, the counts handed
    twin_n = ready_n + ROOM
    tuned_n = ready_n + SCAN + ROOM + EARLY + COUNTS
    for nm in (twin_n, tuned_n):
        assert nm in rows[132] and nm in rows[1], f"lab page: the record holds no row {nm!r}"
        assert exact(nm), "lab page: a tuned-listing row is not exact"
    tdets = [c["detail"] for c in full["checks"] + one["checks"] if c["variant"] == tuned_n and "ok" in c]
    assert len(tdets) == 2 and all(d.split("; ")[-1].startswith("0 of ") and d.endswith(" listed places differ") for d in tdets), (
        f"lab page: the tuned listing's lists are not exact: {tdets}")
    tuned_detail = {n: [c["detail"] for c in (full if n == 132 else one)["checks"] if c["variant"] == tuned_n][0].split("; ")[-1] for n in (1, 132)}
    tw = {n: w(n, twin_n) for n in (1, 132)}
    tu = {n: w(n, tuned_n) for n in (1, 132)}
    assert tw[132] > tw[1] and tu[132] > tu[1], "lab page: the twin or the tuned row does not cost more at the full grid than at one CTA"
    tuned_text = (
        f"(k, tuned) THE LISTING TUNED (`kScan | kEarly | kCounts | kRoom`: the same listing at registers 176 / 104 / 56, a window's words of the read mask loaded a window ahead, and each word's count of its digit's live tokens "
        f"before it handed with the mask, a byte a word in a second plane the host lays, so no scan of the lanes' counts; the emission's rounds unchanged; the lists checked and not consumed). "
        f"Its twin is the ready-offsets kernel at the same registers with no listing (`kRoom`), cycles a window a CTA: at {sms} CTAs the tuned listing {g(tu[132])} against the twin's {g(tw[132])} "
        f"({more(tu[132], tw[132])}), at one CTA {g(tu[1])} against {g(tw[1])} ({more(tu[1], tw[1])}); beside the first listing's cost on its own kernel (above): "
        f"{more(sc['ready'][132], fr['ready'][132])} at {sms} CTAs and {more(sc['ready'][1], fr['ready'][1])} at one CTA. Every row is exact and the lists are exact: the checks' detail says "
        f"\"{tuned_detail[132]}\" at {sms} CTAs and \"{tuned_detail[1]}\" at one CTA. THE BYTE'S LIMIT is a limit of this lab row: a byte holds at most 255 live tokens of a digit before a word in a window, "
        f"and the lab's mask has 16 a digit a window. THE STATIC COUNTS (`ledger carry_token --static --arch sm_90a`, kernel probes 3981392 the tuned listing, 442448 the first listing, 2146384 the twin): "
        "the drain's window loop holds 60 instructions with the tuned listing (the drain's add loop of 163, 32 global add, inside it, and four loops of 19, 1 shared store each, beside it; "
        "instructions outside every loop 297), 112 with the first listing (20 shuffle; add loop 160; 279 outside) and 21 without a listing (the twin, add loop 163; 260 outside). "
        f"NOT PARTED: the tuned row costs more at {sms} CTAs than at one CTA ({more(tu[132], tu[1])}), on a twin that itself costs more at {sms} CTAs than at one ({more(tw[132], tw[1])}); "
        "the twin is the ready-offsets kernel without a pass of lead for its drain's offsets, and no row here parts the twin's own cost at the full grid from the listing's. "
        "NOT BUILT: the tuned listing on the pass-ahead kernel, and the mask a window ahead with the lanes' scan (a drain's registers spill in each). "
        "A DENSER MASK is not measured and the row is valid only at the lab's 16 readers a digit in a window: this lab's list area holds 16 places a read digit and the index wraps, "
        "so a digit with more readers than 16 in a window would overwrite its list; denser does not only run more rounds of the emission."
    )
    far_all = f"{ob_text} {far_text} {ready_far_text}\n- {scan_text}\n- {tuned_text}"

    # ---- the readout's pass by phases: the two kSplitRead rows at the full grid, CTA 0
    split_fixed = fixed_n + COARSE_SUF + DETAILS[1][0] + DETAILS[3][0] + SPLIT_READ
    split_far = far_n + COARSE_SUF + DETAILS[1][0] + DETAILS[3][0] + SPLIT_READ
    for nm in (split_fixed, split_far):
        assert nm in rows[132], f"lab page: the record holds no row {nm!r}"
        assert plain_of(nm) in rows[132], f"lab page: the row {nm!r} has no plain twin"
    phase_keys = ("expanding", "at its coefficients' barrier", "issuing steps", "after its steps' issue", "storing the panel",
                  "waits on tile group # published", "waits on panel # drained")
    st_f_win, st_f, sc_f = role_states(132, split_fixed)
    st_l_win, st_l, _ = role_states(132, split_far)
    for st in (st_f, st_l):
        assert all(k in st["readout"] for k in phase_keys), f"lab page: a readout state is missing in {sorted(st['readout'])}"

    def states_line(st):
        return ", ".join(f"{k} {g(st['readout'][k])}" for k in phase_keys)

    diffs = ", ".join(f"{k} {st_l['readout'][k] - st_f['readout'][k]:+,.0f}" for k in phase_keys)
    split_text = (
        f"THE READOUT'S PASS BY PHASES (the two rows with the readout's pass stamped by phases and the waits on the tile groups published and the panels drained, at {sms} CTAs, CTA {sc_f}; "
        f"`bazel run //tools:lab -- trace carry_token --record measure/lab/results/carry_token/sm_90a.blocks{sms}.json --row \"the readout's pass by phases stamped$\"`): "
        f"the fixed readout, the row's window {g(w(132, split_fixed))} cycles ({g(st_f_win)} in CTA {sc_f}'s steady span; its unstamped row {g(twin(split_fixed, 132))}, "
        f"{more(w(132, split_fixed), twin(split_fixed, 132))}), the readout in cycles a window a warp: {states_line(st_f)}; "
        f"the listed readout with the drain's lines a pass ahead, the row's window {g(w(132, split_far))} cycles ({g(st_l_win)} in the steady span; its unstamped row {g(twin(split_far, 132))}, "
        f"{more(w(132, split_far), twin(split_far, 132))}), the readout: {states_line(st_l)}. "
        f"THE DIFFERENCE, the listed readout minus the fixed readout, in cycles a window a warp: {diffs}. "
        "These rows' stamps cost several hundred cycles a window (the windows above against their unstamped rows'), so the differences are what read, not the states' own figures."
    )

    # ---- the bytes a layout holds: arithmetic from the packet's layout, checked against the experiment's constants
    src = (root / "measure/lab/experiments/carry_token.cu").read_text()
    for needle in (
        "kLists = 16, kEntries = 16, kWindow = kLists * kEntries;",
        "kGroup = 128,",
        "kValueLines = 16 * kGroup, kValueBytes = kValueLines * 16,",
    ):
        assert needle in src, f"lab page: carry_token.cu no longer holds {needle!r}, the bytes' arithmetic reads it"
    window, group_ = 16 * 16, 128
    value_lines, value_bytes = 16 * group_, 16 * group_ * 16
    line_bytes = 16
    lines_a_token = value_lines // window
    token_bytes = lines_a_token * line_bytes
    assert value_bytes == window * token_bytes
    layouts_bytes = [(1, "per list at one digit a token"), (4, "grouped at four digits a token"), (16, "grouped at 16 digits a token")]
    held = "; ".join(f"{label}, {window // k} tokens, {g((window // k) * token_bytes)} bytes of value lines" for k, label in layouts_bytes)
    bytes_text = (
        f"THE BYTES A LAYOUT HOLDS, by arithmetic from the packet's layout (a token's value lines are {lines_a_token} lines, {token_bytes} bytes; a window is "
        f"{window} memberships a box; `kValueBytes` is {g(value_bytes)}): {held}. Grouping's larger purpose is the stream's bytes when dense: a token on k "
        "digits of a group is laid once, with its coefficients for those digits, so its value lines are k times fewer than a list a digit's."
    )
    not_here = (
        "NOT MEASURED HERE: this kernel lands the whole `kValueBytes` of every packet whatever the layout (`issue_one`), and the cell is not "
        f"bound on global memory (one CTA against {sms}), so no row shows the traffic a grouped layout saves; the {pct((t4l - t4g) / t4l)} at four digits a token "
        "is the tensor steps' side alone. OWED: a row whose packet lands only its tokens' value lines; and the gather's cost of writing a token once. "
        "ALSO OWED, as an observation: every membership holds its own dense line (16 bytes) though a token's dense line is the same on all its digits; "
        "a grouped layout could hold it once a token."
    )
    grouping = (
        "THE GROUPED FOLD (a group's tokens once, a step 16 tokens' value lines against a tile of 16 tokens by four digits' 64 columns), cycles a window "
        f"a CTA at {sms} CTAs unless said, {'every row exact' if exact_group else 'NOT every row exact'}:\n- {col_text}\n- {sixteen}\n- {waste}\n- {run_time}\n- {pair_text}\n- {counted_text}\n- {loop_text}\n- {table_text}\n- {listed_text}\n- {far_all}"
    )

    # ---- rows of the experiment the record does not hold yet
    pending = []
    fill_name, packet_name = COARSE + FILL_ROW, COARSE + PACKET_ROW
    extra = []
    if fill_name in rows[132]:
        fph, _, _, fscope = fold_phases(132, fill_name)
        body = "; ".join(f"{k} {g(v)}" for k, v in fph.items() if k not in ("window", "issuing steps"))
        extra.append(
            f"THE FILL'S PHASES, from the fill row's record at {sms} CTAs (the mean of the fold warpgroups of CTA {fscope.removeprefix('cta')}): {body}"
            f" cycles; the fill's own work is the phases before the barrier and its wait is `at the fill's barrier`."
        )
    else:
        pending.append("the fold's fill by phases (the fill's thread work against its wait at the barrier)")
    if packet_name in rows[132]:
        rec, scope, cta = cta0(132, packet_name)
        span = cta.steady(SKIP)
        assert span is not None, "lab page: the packets row's record holds too few windows"
        t0, t1, n = span
        waits = [x for wp in cta.warps if wp.role.startswith("fold") for x in wp.waits if "packet landed" in x.barrier and t0 <= x.t0 and x.t1 <= t1]
        folders = sum(1 for wp in cta.warps if wp.role.startswith("fold"))
        per = sum(x.t1 - x.t0 for x in waits) / (folders * n)
        extra.append(
            f"THE FOLD'S WAIT ON ITS PACKET, from the packets row's record at {sms} CTAs (CTA {scope.removeprefix('cta')}, the mean of"
            f" its {folders} fold warps): {g(per)} cycles a window ({pct(per / ((t1 - t0) / n))} of the window)."
        )
    else:
        pending.append("the fold's wait on its packet's landing (the waits and the copies' issues)")
    wall_name = PLAIN + ": the long call by wall time, scaled to 65,536 tokens"
    clock_name = PLAIN + ": the card's clock, the long call's span over its wall time"
    if wall_name in other[132] and clock_name in other[132]:
        wall = other[132][wall_name]
        extra.append(
            f"THE LONG CALL BY WALL TIME at {sms} CTAs: {wall:.3f} ms for 65,536 tokens ({TOKENS_A_WINDOW} tokens a window a CTA), the"
            f" call's span over its wall time {other[132][clock_name]:.3f} GHz (the card's clock), against the launch-derived {ms:.3f} ms above."
        )
    else:
        pending.append("the plain row's long call by wall time (the card's clock, and the call scaled to 65,536 tokens)")
    not_measured = "NOT MEASURED: the gather that writes the packets; a cell where the packets' bytes grow with duplication"
    if pending:
        not_measured += "; and, in the experiment and not yet in a record: " + "; ".join(pending)
    not_measured += "."

    # ---- the section
    toolkit = full["toolkit"].split("cuda_")[1].split(".r")[0]
    sha = full["git"]["sha"][:8] + (" (a dirty tree)" if full["git"].get("dirty") else "")
    dates = f"{one['date'][:10]} {one['date'][11:16]} and {full['date'][11:16]} UTC" if one["date"][:10] == full["date"][:10] else f"{one['date'][:10]} and {full['date'][:10]} UTC"
    text = f"""MEASURED ({card(instance, full["device"], sms)}, driver {full["driver"]}, toolkit {toolkit}, the clock locked; {dates} (the one-CTA and the full-grid records), the
experiment at {sha}, {full["reps"]} repetitions; the labels name the boxes of the two records, which hold no box's id; the {checked} checked rows pass at one CTA
and at {sms}{f", the {len(isolation)} isolation rows are unchecked (their kernels are not the carry's)" if isolation else ""}{f"{' and' if isolation else ','} the {len(one_block)} one-block rows are unchecked for their own reason (every CTA reads block 0's reader packets, so its outputs are that block's readers' and the state is not the grid's: they are not isolation rows)" if one_block else ""}; `sm_90a.blocks1.json` and `sm_90a.blocks{sms}.json`, each with its traced rows' records beside it). {UNIT[0].upper() + UNIT[1:]}:

{table}

WHAT IT DECIDES. {grid} {layout} {fold_text} {read_side} {marks_text}
{clock_text} {not_measured}

{grouping}

{bytes_text}

{not_here}

{timelines_text}

{split_text}

THE PHASES, in each role's own window of the coarse row's record at {sms} CTAs (CTA {scope132}), cycles:
- the fold: image {g(image)}, fill {g(fill)}, steps {g(steps)}.
- {role_line("the readout", read_ph, read_period)}.
- {role_line("the drain", drain_ph, drain_period)}.
"""
    if extra:
        text += "\n" + "\n".join(f"- {e}" for e in extra) + "\n"
    text += (
        "\nRecord: `measure/lab/results/carry_token/sm_90a.blocks1.json` and `sm_90a.blocks132.json`, each with its traced rows' records in"
        " `sm_90a.blocks1.trace/` and `sm_90a.blocks132.trace/`.\n"
    )
    p = root / "docs/lab/carry_token.md"
    s = p.read_text()
    i = s.index("MEASURED (")
    p.write_text(s[:i] + text)
    return [p]
