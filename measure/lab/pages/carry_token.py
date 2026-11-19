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
    layout, probe, kind = "", "", "plain"
    if SHARED in rest:
        layout, rest = "lists overlapping on shared tokens", rest.replace(SHARED, "")
    if IDLE in rest:
        probe, rest = "the read side idle (not checked)", rest.replace(IDLE, "")
    for suffix, words in DETAILS:
        if suffix in rest:
            kind, rest = f"coarse, {words}", rest.replace(COARSE_SUF, "").replace(suffix, "")
    for suffix, words in ((EDGES, "the launch's two edges stamped"), (COARSE_SUF, "coarse"), (FINE, "every warp's timeline stamped")):
        if suffix in rest:
            kind, rest = words, rest.replace(suffix, "")
    assert not rest, f"lab page: the row {name!r} has a part the generator does not name: {rest!r}"
    if layout:
        return f"{layout}, {kind}"
    return f"{kind}, {probe}" if probe and kind != "plain" else kind


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
    unchecked = [c["variant"] for c in full["checks"] if "ok" not in c]
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
    steps_cost = (steps - steps_idle) / steps
    read_side = (
        f"THE READ SIDE COSTS THE FOLDS {pct(cost)} OF THE WINDOW at {sms} CTAs ({pct(cost1)} at one CTA): the coarse row's window is"
        f" {g(w(132, COARSE))} cycles and the same kernel with the read side idle {g(w(132, idle))} (the isolation row, not checked: its"
        f" kernel is not the carry's); the fold's 32 steps are {g(steps)} cycles and, with it idle, {g(steps_idle)} ({pct(steps_cost)}"
        f" under the coarse row's)."
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
    sha = full["git"]["sha"][:8] + (" (a dirty tree)" if full["git"].get("dirty") else "")
    dates = f"{one['date'][:10]} {one['date'][11:16]} and {full['date'][11:16]} UTC" if one["date"][:10] == full["date"][:10] else f"{one['date'][:10]} and {full['date'][:10]} UTC"
    text = f"""MEASURED ({card(instance, full["device"], sms)}, driver {full["driver"]}, the clock locked; {dates} (the one-CTA and the full-grid records), the
experiment at {sha}, {full["reps"]} repetitions; the labels name the boxes of the two records, which hold no box's id; the {checked} checked rows pass at one CTA
and at {sms}{f", {len(unchecked)} row is unchecked (the isolation row)" if len(unchecked) == 1 else ""}; `sm_90a.blocks1.json` and `sm_90a.blocks{sms}.json`, each with its traced rows' records beside it). {UNIT[0].upper() + UNIT[1:]}:

{table}

WHAT IT DECIDES. {grid} {layout} {fold_text} {read_side} {marks_text}
{clock_text} {not_measured}

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
