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
                          (SPREAD, "an entry's dense line from a 2 GB table of the tokens"), (TABLED, "an entry's dense line from the tokens' table"), (SPLIT, "registers 176 / 112 / 48"),
                          (IDLE, "the read side idle (not checked)")):
        if suffix in rest:
            probes.append(words)
            rest = rest.replace(suffix, "")
    probe = ", ".join(probes)
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
    assert checked == sum(1 for c in one["checks"] if "ok" in c)
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
    def tab(n, layout, table, idle=False):
        return rows[n][PREFIX + layout + table + ((IDLE + COARSE_SUF) if idle else "")]

    packed_g = {n: (w(n, group4), cycles_of(n, TOKEN4, idle=True)) for n in (1, 132)}
    packed_l = {n: w(n, PLAIN) for n in (1, 132)}
    tg = {(n, t): (tab(n, TOKEN4, t), tab(n, TOKEN4, t, idle=True)) for n in (1, 132) for t in (TABLED, SPREAD)}
    tl = {(n, t): tab(n, "", t) for n in (1, 132) for t in (TABLED, SPREAD)}
    exact_tab = all(exact(PREFIX + lay + t) for lay in (TOKEN4, "") for t in (TABLED, SPREAD))

    def over(n, t):
        return tg[(n, t)][0] - packed_g[n][0], tg[(n, t)][1] - packed_g[n][1], tl[(n, t)] - packed_l[n]

    grow = {n: (tg[(n, SPREAD)][0] - tg[(n, TABLED)][0], tl[(n, SPREAD)] - tl[(n, TABLED)]) for n in (1, 132)}
    table_text = (
        "(h) WHO READS A MEMBERSHIP'S DENSE ROW: the packet's packed 16-byte line, or the fold's own load from the tokens' table by a two-byte index "
        "in the packet (two chained global loads an entry at the window's start; the rows 16 bytes; the table 1 MB, or the same rows 32 KB apart, 2 GB). "
        f"The four-digit grouped fold on the table is {g(tg[(132, TABLED)][0])} composed and {g(tg[(132, TABLED)][1])} with the read side idle against the packed "
        f"line's {g(packed_g[132][0])} and {g(packed_g[132][1])}: {g(over(132, TABLED)[0])} and {g(over(132, TABLED)[1])} more (at one CTA {g(over(1, TABLED)[0])} and "
        f"{g(over(1, TABLED)[1])}); on the 2 GB table {g(tg[(132, SPREAD)][0])} and {g(tg[(132, SPREAD)][1])}, {g(over(132, SPREAD)[0])} and {g(over(132, SPREAD)[1])} "
        f"more (at one CTA {g(over(1, SPREAD)[0])} and {g(over(1, SPREAD)[1])}). The per-list fold on the table is {g(tl[(132, TABLED)])} against the packed "
        f"line's {g(packed_l[132])}, {g(over(132, TABLED)[2])} more (at one CTA {g(over(1, TABLED)[2])}); on the 2 GB table {g(tl[(132, SPREAD)])}, "
        f"{g(over(132, SPREAD)[2])} more (at one CTA {g(over(1, SPREAD)[2])}). Every row is {'exact' if exact_tab else 'NOT exact'}. Two facts, and nothing beyond them: "
        "the table's own load is not free even from a cached table (the 1 MB rows above); and it grows with the table's size by the DISTINCT rows a packet names "
        f"(the four-digit packet's 256 entries a box name 64 tokens, the per-list packet's name 256): the 2 GB table over the 1 MB one costs the grouped fold "
        f"{g(grow[132][0])} cycles and the per-list fold {g(grow[132][1])}, {grow[132][1] / grow[132][0]:.1f} times (at one CTA {g(grow[1][0])} and {g(grow[1][1])}, {grow[1][1] / grow[1][0]:.1f} times). "
        "NOT MEASURED: what the gather pays to write a packed 16-byte line a membership in its current form ([`gather_group`](gather_group.md)'s row is values "
        "only, and writes the entries' token indices, no dense line), so the two forms are not compared end to end here; the rows loaded a window ahead."
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
        f"a CTA at {sms} CTAs unless said, {'every row exact' if exact_group else 'NOT every row exact'}:\n- {col_text}\n- {sixteen}\n- {waste}\n- {run_time}\n- {pair_text}\n- {counted_text}\n- {loop_text}\n- {table_text}"
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
and at {sms}{f", the {len(isolation)} isolation rows are unchecked (their kernels are not the carry's)" if isolation else ""}; `sm_90a.blocks1.json` and `sm_90a.blocks{sms}.json`, each with its traced rows' records beside it). {UNIT[0].upper() + UNIT[1:]}:

{table}

WHAT IT DECIDES. {grid} {layout} {fold_text} {read_side} {marks_text}
{clock_text} {not_measured}

{grouping}

{bytes_text}

{not_here}

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
