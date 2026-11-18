# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE TIMELINE OF A TRACED ROW (`lab trace`), read from warpline's native record (`lab.cuh`'s `trace_record` writes it;
a directory of records beside the lab's record, one a traced row): the steady span, each role's phases with their
durations and rates, and the launch (entry to exit in cycles and nanoseconds, the rate between them) are warpline's
library's -- `warpline.reader.steady`, `.window.role_windows`, `.rates.analyze`, `.launch.launch_of`, `.check` -- and what
warpline has no notion of is the lab's, read from the record's stamps: each role's time by state, which warp's arrival
released each wait, and the window's critical path walked back through those releases.

A fine row holds each warp's waits' starts and ends, its arrivals, its copies' issues and its phases' starts; a coarse
row holds its phases' starts alone, and has no waits to walk a path through; a coarse row's and an EDGES row's warps also
mark the launch's two edges (`launch.begin` at the kernel's top and `launch.end` at its role's end), on both clocks, and
an edges row holds nothing else.

A wait is BLOCKING when it lasted more than `BLOCKING` cycles; a shorter one found its barrier passed, and is its
warp's own time. A blocking wait's RELEASER is the event that completed its barrier's phase: the latest arrival at the
barrier at or before the wait's end, a named barrier's (one the record declares with no stamped arrival and no copy) the
latest warp to reach it (its wait's begin), and a copy's barrier the copy's issue (its flight the release's latency). A
step wait (a barrier declared with a count of 0) is its warp's own steps and releases nothing.

THE PATH: from the end of the steady span (its last window's start on the fold's first warp), that warp's own time
back to its latest blocking wait, then the releaser's from its release, and so on to the span's start. Every cycle of
the span is then one warp's state, a copy's flight or a release's latency: the window is shortened only by shortening
a piece of it. See docs/internals/tools/lab.md#trace.
"""
from __future__ import annotations

import bisect
import html
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import dev_config

if dev_config.warpline_python() is None:
    raise SystemExit("the lab's traces are read by warpline's library, which the graph hands it: bazel run //tools:lab -- ...")
sys.path.insert(0, dev_config.warpline_python())
from warpline import check, rates, reader  # noqa: E402
from warpline import launch as wl_launch
from warpline import record as wl_record
from warpline import window as wl_window

PHASE, WAITS, WAITED, ARRIVES, ISSUES = 1, 2, 3, 4, 5
BLOCKING = reader.BLOCKING
#: the window's anchor: the fold's phase the lab's windows are counted by
WINDOW = "phase.window"
#: the suffixes a traced row's name carries over its untraced twin's: a fine timeline, coarse phases, the launch's edges
FINE_SUFFIX, COARSE_SUFFIX, EDGES_SUFFIX = (", every warp's timeline stamped", ", every warp's phases stamped",
                                            ", the launch's two edges stamped")
TRACE_SUFFIXES = (FINE_SUFFIX, COARSE_SUFFIX, EDGES_SUFFIX)


@dataclass
class Wait:
    t0: int
    t1: int
    barrier: str
    warp: int


@dataclass
class Warp:
    index: int
    lane: str
    role: str
    #: how many of the warp's stamps its row had no room for
    dropped: int
    #: (time, kind, name): a phase's start, a wait's start and end, an arrival, a copy's issue
    events: list[tuple[int, int, str]]
    #: (start, end, state): a state "work:<phase>", "wait:<barrier>" or "steps:<barrier>"
    spans: list[tuple[int, int, str]] = field(default_factory=list)
    waits: list[Wait] = field(default_factory=list)


def group(name: str) -> str:
    """A site's name with its indices taken out: the waits of a kind read together."""
    return re.sub(r"\d+", "#", name)


class Cta:
    """One CTA of a traced row: a scope of the record, a warp a lane."""

    def __init__(self, rec: wl_record.Record, scope: str):
        self.rec, self.scope = rec, scope
        barriers = rec.facts.get("barriers", {})
        #: a barrier's kind, by the record's facts: its warp's own steps (a count of 0), a copy's (an `issue`), a
        #: named barrier's (no stamped arrival: the warp's reaching it is its wait's begin), else arrivals stamped
        self.steps = {b for b, f in barriers.items() if f["count"] == 0}
        self.copies = {f["issue"]: b for b, f in barriers.items() if f.get("issue")}
        self.named = {b for b, f in barriers.items()
                      if f["count"] > 0 and f.get("stamped", f["count"]) == 0 and not f.get("issue")}
        self.warps: list[Warp] = []
        self.arrivals: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
        for i, lane in enumerate(ln for ln in rec.lanes if wl_record.scope_of(ln) == scope):
            events = []
            for _idx, name, t, _payload in rec.lane_stamps(lane):
                kind, key = wl_record.classify(name)
                if kind == "region_begin" and key.startswith("phase."):
                    events.append((t, PHASE, key[len("phase."):]))
                elif kind == "wait_begin":
                    events.append((t, WAITS, key))
                elif kind == "wait_end":
                    events.append((t, WAITED, key))
                elif kind == "arrive":
                    events.append((t, ARRIVES, key))
                elif kind == "op_issue":
                    events.append((t, ISSUES, self.copies.get(key, key)))
            self.warps.append(Warp(i, lane, rec.lanes[lane].role, rec.dropped.get(lane, 0), events))
        for w in self.warps:
            self._spans(w)
        for sites in self.arrivals.values():
            sites.sort()
        regions = reader.regions(rec)[0]
        carrier = next((w.lane for w in self.warps if any(r.name == WINDOW for r in regions[w.lane])), None)
        self.windows = [r.begin for r in regions[carrier] if r.name == WINDOW] if carrier else []

    def _spans(self, w: Warp) -> None:
        state, phase, opened, prev = "work:start", "start", None, None
        for t, k, name in w.events:
            if prev is not None and t > prev:
                w.spans.append((prev, t, state))
            if k == PHASE:
                phase = name
                state = f"work:{phase}"
            elif k == WAITS:
                opened = (t, name)
                state = ("steps:" if name in self.steps else "wait:") + name
            elif k == WAITED:
                if opened and opened[1] == name:
                    w.waits.append(Wait(opened[0], t, name, w.index))
                    # a named barrier's arrival is its warp's reaching it: the release of the group's last
                    if name in self.named:
                        self.arrivals[name].append((opened[0], w.index, ARRIVES))
                opened = None
                state = f"work:{phase}"
            elif k in (ARRIVES, ISSUES):
                self.arrivals[name].append((t, w.index, k))
            prev = t

    def label(self, warp: int, state: str) -> str:
        kind, what = state.split(":", 1)
        role = self.warps[warp].role
        return f"{role}: " + {"work": what, "wait": f"waits on {group(what)}", "steps": f"waits on its {what}"}[kind]

    def releaser(self, w: Wait) -> tuple[int, int, int] | None:
        """The event that completed the wait's barrier phase: the latest arrival or issue at or before its end, by
        another warp (or the waiter's own copy)."""
        if w.barrier in self.steps:
            return None
        found = self.arrivals.get(w.barrier, [])
        i = bisect.bisect_right(found, (w.t1, 1 << 30, 9)) - 1
        while i >= 0:
            t, warp, kind = found[i]
            if warp != w.warp or kind == ISSUES:
                return t, warp, kind
            i -= 1
        return None

    def state_at(self, warp: int, t: int) -> str:
        spans = self.warps[warp].spans
        i = bisect.bisect_right(spans, (t, 1 << 62, "")) - 1
        return spans[i][2] if 0 <= i < len(spans) else "work:start"

    def clip(self, warp: int, a: int, b: int, into: dict[str, float]) -> None:
        for s0, s1, state in self.warps[warp].spans:
            lo, hi = max(s0, a), min(s1, b)
            if hi > lo:
                into[self.label(warp, state)] += hi - lo

    def launch(self) -> dict[str, float] | None:
        """The CTA's launch, from warpline's (`launch.launch_of`): the first warp's entry to the last warp's exit, in
        cycles (the SM's clock, one for the CTA) and in nanoseconds (the timer's), the rate between them, and the spread
        of the warps' entries and of their exits in cycles. None where any warp lacks an edge (a row that marks none)."""
        edges = {w.lane: {n: t for _i, n, t, _p in self.rec.lane_stamps(w.lane) if n in (wl_record.LAUNCH_BEGIN,
                                                                                   wl_record.LAUNCH_END)}
                 for w in self.warps}
        if not edges or any(len(e) != 2 for e in edges.values()):
            return None
        enters = [e[wl_record.LAUNCH_BEGIN] for e in edges.values()]
        exits = [e[wl_record.LAUNCH_END] for e in edges.values()]
        both = wl_launch.launch_of(self.rec, self.scope, need_windows=False)
        return {"cycles": both.cycles, "ns": both.length_ns, "rate": 1000.0 * both.cycles / both.length_ns,
                "entry_spread": max(enters) - min(enters), "exit_spread": max(exits) - min(exits)}

    def steady(self, skip: int) -> tuple[int, int, int] | None:
        """The span's start, end and windows (warpline's `reader.steady`): past the first `skip` windows, to the last
        window every warp's stamps cover."""
        found = reader.steady(self.rec, WINDOW, skip, self.scope)
        return None if found is None else found[1:]

    def path(self, t0: int, t1: int) -> tuple[dict[str, float], dict[str, list[int]]]:
        """The path's pieces, and for each release it follows the MARGINS it binds by: each wait's length, the cycles
        its warp's own work would have to grow by before that work, not the release, held it -- a small margin is a
        release that only just binds, beside a chain of the warp's own nearly as long."""
        pieces: dict[str, float] = defaultdict(float)
        margins: dict[str, list[int]] = defaultdict(list)
        warp, t = 0, t1
        while t > t0:
            blocking = [w for w in self.warps[warp].waits if w.t1 <= t and w.t1 - w.t0 > BLOCKING
                        and self.releaser(w) is not None]
            last = max(blocking, key=lambda w: w.t1, default=None)
            if last is None or last.t1 <= t0:
                self.clip(warp, t0, t, pieces)
                break
            self.clip(warp, last.t1, t, pieces)
            at, by, kind = self.releaser(last)
            at = max(at, t0)
            what = group(last.barrier)
            label = (f"copy in flight: {what}" if kind == ISSUES
                     else f"release of {self.warps[last.warp].role} at {what}")
            pieces[label] += last.t1 - at
            margins[label].append(last.t1 - last.t0)
            if at >= t:
                break
            warp, t = by, at
        return pieces, margins


def untraced_name(name: str) -> str:
    """A traced row's untraced twin's name: the row's name with its trace's suffix removed."""
    for suffix in TRACE_SUFFIXES:
        name = name.replace(suffix, "")
    return name


def launch_line(launch: dict[str, float]) -> str:
    return (f"launch {launch['cycles']:,.0f} cycles, {launch['ns']:,.0f} ns, {launch['rate']:,.1f} cycles a microsecond; "
            f"the warps enter within {launch['entry_spread']:,.0f} cycles and exit within {launch['exit_spread']:,.0f}")


def phases(rec: wl_record.Record, scope: str, skip: int) -> list[str]:
    """Each role's phases in the role's own window and their rates (warpline's `window.role_windows` and
    `rates.analyze`): a phase's start in the window, its duration in cycles, and its cycles a microsecond."""
    out = []
    try:
        windows = wl_window.role_windows(rec, scope, WINDOW, skip)
    except wl_window.WindowError:
        return out
    by_phase = {}
    if rec.ties:
        scoped = rates.analyze(rec, WINDOW, skip, [scope]).scopes.get(scope)
        by_phase = {(r.role, r.name, r.occ): r for r in scoped.by_role} if scoped else {}
    for rw in windows:
        out.append(f"    {rw.role} ({len(rw.lanes)} warps), its phases in its own window of {rw.period:,.0f} cycles "
                   f"({rw.windows} windows): start, duration" + (", rate" if by_phase else ""))
        for r in rw.rows:
            rate = by_phase.get((r.role, r.name, r.occ))
            out.append(f"      {r.start:8,.0f} {r.dur:8,.0f}  {r.name.removeprefix('phase.')}"
                       + (f"  {rate.rate:,.0f} cycles a microsecond" if rate else ""))
    return out


def report(rows: list[tuple[str, wl_record.Record]], pattern: str | None, skip: int, windows: dict[str, float],
           out: Path | None) -> int:
    shown = 0
    pages = []
    for name, rec in rows:
        if pattern and not re.search(pattern, name):
            continue
        shown += 1
        timed = windows.get(name)
        untraced = windows.get(untraced_name(name))
        print(f"\n{name}")
        if timed and untraced:
            print(f"  window: {timed:,.0f} cycles traced, {untraced:,.0f} untraced ({timed / untraced - 1:+.1%} the trace's cost)")

        for scope in rec.scopes():
            cta = Cta(rec, scope)
            number = scope.removeprefix("cta")
            span = cta.steady(skip)
            dropped = [w.index for w in cta.warps if w.dropped]
            launch = cta.launch()
            if span is None and launch is not None and not cta.windows:
                # an EDGES row holds the launch's two edges and nothing else: its launch is all it has to read
                print(f"  CTA {number}: {launch_line(launch)} (an edges row: no windows stamped)")
                continue
            if span is None:
                print(f"  CTA {number}: too few windows stamped")
                if launch is not None:
                    print(f"    {launch_line(launch)}")
                continue
            t0, t1, n = span
            print(f"  CTA {number}: windows {skip} to {skip + n}, {(t1 - t0) / n:,.0f} cycles a window"
                  + (f"; warps {dropped} past their stamps' room" if dropped else ""))
            if launch is not None:
                print(f"    {launch_line(launch)}")
            for line in phases(rec, scope, skip):
                print(line)

            # each role's time by state
            by_role: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
            members: dict[str, int] = defaultdict(int)
            for w in cta.warps:
                members[w.role] += 1
                cta.clip(w.index, t0, t1, by_role[w.role])
            for role, states in by_role.items():
                scale = members[role] * n
                print(f"    {role} ({members[role]} warps), cycles a window a warp:")
                for label, v in sorted(states.items(), key=lambda kv: -kv[1])[:8]:
                    print(f"      {v / scale:8,.0f}  {label.split(': ', 1)[1]}")

            # who released the blocking waits
            held: dict[tuple[str, str, str], list[int]] = defaultdict(list)
            for w in cta.warps:
                for wt in w.waits:
                    if not (t0 <= wt.t1 <= t1) or wt.t1 - wt.t0 <= BLOCKING:
                        continue
                    r = cta.releaser(wt)
                    if r is None:
                        continue
                    at, by, kind = r
                    source = ("its copy" if kind == ISSUES else
                              cta.label(by, cta.state_at(by, at - 1)).replace(": ", ", from "))
                    held[(w.role, group(wt.barrier), source)].append(wt.t1 - wt.t0)
            print("    blocking waits (over 64 cycles), a window a warp: held, by whom")
            for (role, site, source), v in sorted(held.items(), key=lambda kv: -sum(kv[1])):
                print(f"      {sum(v) / (members[role] * n):8,.0f}  {role} at {site} <- {source}")

            pieces, margins = cta.path(t0, t1)
            print("    THE PATH, cycles a window (a release: the mean margin it binds by, its waits' length):")
            for label, v in sorted(pieces.items(), key=lambda kv: -kv[1]):
                if v / n >= 20:
                    m = margins.get(label)
                    print(f"      {v / n:8,.0f}  {label}" + (f"  (binding by {sum(m) / len(m):,.0f})" if m else ""))
            if out is not None and scope == rec.scopes()[0]:
                pages.append(svg(name, cta, t0, cta.windows[cta.windows.index(t0) + 3] if len(cta.windows) > cta.windows.index(t0) + 3 else t1))

    if out is not None and pages:
        out.write_text(page(pages))
        print(f"\nlab: the timelines at {out}")
    return 0 if shown else 1


def svg(name: str, cta: Cta, t0: int, t1: int) -> str:
    """Three windows of a CTA, a lane a warp: its work by phase, its waits pale, a blocking wait's release a line."""
    width, lane, left = 1400, 16, 130
    scale = (width - left - 10) / (t1 - t0)
    phases = sorted({s.split(":", 1)[1] for w in cta.warps for _, _, s in w.spans if s.startswith("work:")})
    hues = ["#2f6fb3", "#c26b19", "#3d8f4f", "#9a4fb0", "#b8433a", "#2a8f8f", "#8a7a1e", "#5b5fc7", "#b0487a"]
    color = {p: hues[i % len(hues)] for i, p in enumerate(phases)}
    x = lambda t: left + (t - t0) * scale  # noqa: E731
    parts = []
    for w in cta.warps:
        y = 20 + w.index * (lane + 4)
        parts.append(f'<text x="4" y="{y + 12}" class="lbl">{html.escape(w.role)} {w.index}</text>')
        for s0, s1, state in w.spans:
            if s1 < t0 or s0 > t1:
                continue
            a, b = x(max(s0, t0)), x(min(s1, t1))
            kind, what = state.split(":", 1)
            fill = color.get(what, "#888") if kind == "work" else "var(--steps)" if kind == "steps" else "var(--wait)"
            tip = html.escape(f"{cta.label(w.index, state)}: {s1 - s0:,} cycles")
            parts.append(f'<rect x="{a:.1f}" y="{y}" width="{max(b - a, 0.5):.1f}" height="{lane}" fill="{fill}">'
                         f'<title>{tip}</title></rect>')
        for wt in w.waits:
            if t0 <= wt.t1 <= t1 and wt.t1 - wt.t0 > BLOCKING:
                r = cta.releaser(wt)
                if r and r[2] != ISSUES and r[0] >= t0:
                    y2 = 20 + r[1] * (lane + 4) + lane / 2
                    parts.append(f'<line x1="{x(r[0]):.1f}" y1="{y2:.1f}" x2="{x(wt.t1):.1f}" y2="{y + lane / 2:.1f}" '
                                 f'class="rel"/>')
    for t in cta.windows:
        if t0 <= t <= t1:
            parts.append(f'<line x1="{x(t):.1f}" y1="10" x2="{x(t):.1f}" y2="{20 + len(cta.warps) * (lane + 4)}" class="win"/>')
    height = 30 + len(cta.warps) * (lane + 4)
    keys = "".join(f'<span><i style="background:{c}"></i>{html.escape(p)}</span>' for p, c in color.items())
    keys += '<span><i style="background:var(--wait)"></i>waiting</span><span><i style="background:var(--steps)"></i>waiting on its own steps</span>'
    return (f'<section><h2>{html.escape(name)}</h2><p class="keys">{keys}</p><div class="scroll">'
            f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(name)}: three windows, a lane a warp">'
            f'{"".join(parts)}</svg></div></section>')


def page(sections: list[str]) -> str:
    return ("<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
            "<title>Carry Window Timelines</title><style>"
            ":root{--ink:#1d2330;--ground:#f7f8fa;--wait:#d9dde4;--steps:#a9b0bd;--rel:#c0392b;--win:#1d2330}"
            "@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--ink:#e4e7ee;--ground:#151922;"
            "--wait:#353b47;--steps:#596273;--rel:#ff7a6b;--win:#e4e7ee}}"
            ":root[data-theme=dark]{--ink:#e4e7ee;--ground:#151922;--wait:#353b47;--steps:#596273;--rel:#ff7a6b;--win:#e4e7ee}"
            "body{background:var(--ground);color:var(--ink);font:14px/1.5 system-ui,sans-serif;padding:16px}"
            "h2{font-size:15px;font-weight:600}.scroll{overflow-x:auto}svg{width:1400px;max-width:none}"
            ".lbl{font:11px system-ui;fill:var(--ink)}.rel{stroke:var(--rel);stroke-width:1;opacity:.8}"
            ".win{stroke:var(--win);stroke-dasharray:3 3;opacity:.5}.keys span{margin-right:14px;white-space:nowrap}"
            ".keys i{display:inline-block;width:10px;height:10px;margin-right:4px;vertical-align:middle}"
            "</style></head><body>" + "".join(sections) + "</body></html>")


def findings(rec: wl_record.Record) -> list[str]:
    """What warpline's `check` finds in a traced row's record, but the row's own: a row with no launch exit (a fine row's
    kernels are at their registers' limit and hold no entry beside their timeline) leaves its last phase open, which the
    reader names `region_unclosed`."""
    exitless = {ln for ln in rec.lanes if wl_record.LAUNCH_END not in {n for _i, n, _t, _p in rec.lane_stamps(ln)}}
    return [str(f) for f in check.all_findings(rec) if not (f.code == "region_unclosed" and f.lane in exitless)]


def unsound(rows: list[dict]) -> list[str]:
    """The findings in the records a run's traced rows wrote (`{"variant", "record"}` each), a row a line."""
    out = []
    for item in rows:
        found = findings(wl_record.from_dict(item["record"]))
        if found:
            out.append(f"{item['variant'][:80]}: {len(found)} finding(s); first: {found[0]}")
    return out


def load(folder: Path) -> list[tuple[str, wl_record.Record]]:
    """A record's traced rows, `(variant, record)` each, in the order they were run: the directory `<record>.trace`."""
    files = sorted(folder.glob("row*.json"), key=lambda p: int(p.stem.removeprefix("row")))
    recs = [wl_record.load(f) for f in files]
    return [(rec.source["variant"], rec) for rec in recs]
