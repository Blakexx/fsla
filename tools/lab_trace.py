# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE TIMELINE OF A TRACED ROW (`lab trace`): each warp's stamps -- its waits' starts and ends, its arrivals, its
copies' issues and its phases' starts (measure/lab/lab.cuh, `lab::Trace`) -- read as what each warp did when, who
released each wait it was held at, and the window's critical path walked back through those releases.

A wait is BLOCKING when it lasted more than `BLOCKING` cycles; a shorter one found its barrier passed, and is its
warp's own time. A blocking wait's RELEASER is the event that completed its barrier's phase: the latest arrival at the
barrier at or before the wait's end, a named barrier's the latest warp to reach it, and a copy's barrier the copy's
issue (its flight the release's latency). A step wait is its warp's own steps and releases nothing.

THE PATH: from the end of the steady span (its last window's start on the fold's first warp), that warp's own time
back to its latest blocking wait, then the releaser's from its release, and so on to the span's start. Every cycle of
the span is then one warp's state, a copy's flight or a release's latency: the window is shortened only by shortening
a piece of it. See docs/internals/tools/lab.md#trace.
"""
from __future__ import annotations

import bisect
import html
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

PHASE, WAITS, WAITED, ARRIVES, ISSUES = 1, 2, 3, 4, 5
#: a site's range: a barrier's offset over 8 below NAMED, a named barrier's below MARKS, the experiment's own above
NAMED, MARKS = 0x8000, 0xC000
BLOCKING = 64


@dataclass
class Wait:
    t0: int
    t1: int
    site: int
    warp: int


@dataclass
class Warp:
    index: int
    role: str
    count: int
    events: list[tuple[int, int, int]]
    #: (start, end, state): a state "work:<phase>", "wait:<site>" or "steps:<site>"
    spans: list[tuple[int, int, str]] = field(default_factory=list)
    waits: list[Wait] = field(default_factory=list)


def group(name: str) -> str:
    """A site's name with its indices taken out: the waits of a kind read together."""
    return re.sub(r"\d+", "#", name)


class Cta:
    def __init__(self, row: dict, cta: dict):
        self.legend = {int(k): v for k, v in row["legend"].items()}
        self.warps: list[Warp] = []
        self.arrivals: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        for i, w in enumerate(cta["warps"]):
            s = w["stamps"]
            ev = [(s[k], s[k + 1] >> 16 & 15, s[k + 1] & 0xFFFF) for k in range(0, len(s), 2)]
            self.warps.append(Warp(i, row["roles"][i], w["count"], ev))
        for w in self.warps:
            self._spans(w)
        for sites in self.arrivals.values():
            sites.sort()
        self.windows = [t for t, k, site in self.warps[0].events if k == PHASE and self.name(site) == "window"]

    def name(self, site: int) -> str:
        return self.legend.get(site, f"site {site:#x}")

    def _spans(self, w: Warp) -> None:
        state, phase, opened, prev = "work:start", "start", None, None
        for t, k, site in w.events:
            if prev is not None and t > prev:
                w.spans.append((prev, t, state))
            if k == PHASE:
                phase = self.name(site)
                state = f"work:{phase}"
            elif k == WAITS:
                opened = (t, site)
                state = ("steps:" if self.name(site).endswith(" steps") else "wait:") + self.name(site)
            elif k == WAITED:
                if opened and opened[1] == site:
                    w.waits.append(Wait(opened[0], t, site, w.index))
                    # a named barrier's arrival is its warp's reaching it: the release of the group's last
                    if NAMED <= site < MARKS:
                        self.arrivals[site].append((opened[0], w.index, ARRIVES))
                opened = None
                state = f"work:{phase}"
            elif k in (ARRIVES, ISSUES):
                self.arrivals[site].append((t, w.index, k))
            prev = t

    def label(self, warp: int, state: str) -> str:
        kind, what = state.split(":", 1)
        role = self.warps[warp].role
        return f"{role}: " + {"work": what, "wait": f"waits on {group(what)}", "steps": f"waits on its {what}"}[kind]

    def releaser(self, w: Wait) -> tuple[int, int, int] | None:
        """The event that completed the wait's barrier phase: the latest arrival or issue at or before its end, by
        another warp (or the waiter's own copy)."""
        if self.name(w.site).endswith(" steps"):
            return None
        found = self.arrivals.get(w.site, [])
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

    def steady(self, skip: int) -> tuple[int, int, int] | None:
        """The span's start, end and windows: past the first `skip` windows, to the last window every warp's stamps
        cover."""
        last = min(w.events[-1][0] for w in self.warps if w.events)
        marks = [t for t in self.windows if t <= last]
        if len(marks) < skip + 3:
            return None
        return marks[skip], marks[-2], len(marks) - 2 - skip

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
            what = group(self.name(last.site))
            label = (f"copy in flight: {what}" if kind == ISSUES
                     else f"release of {self.warps[last.warp].role} at {what}")
            pieces[label] += last.t1 - at
            margins[label].append(last.t1 - last.t0)
            if at >= t:
                break
            warp, t = by, at
        return pieces, margins


def report(rows: list[dict], pattern: str | None, skip: int, record: dict[str, float], out: Path | None) -> int:
    shown = 0
    pages = []
    for row in rows:
        if pattern and not re.search(pattern, row["variant"]):
            continue
        shown += 1
        name = row["variant"]
        timed = record.get(name)
        untraced = record.get(name.replace(", every warp's timeline stamped", "").replace(", every warp's phases stamped", ""))
        print(f"\n{name}")
        if timed and untraced:
            print(f"  window: {timed:,.0f} cycles traced, {untraced:,.0f} untraced ({timed / untraced - 1:+.1%} the trace's cost)")

        for cta_row in row["ctas"]:
            cta = Cta(row, cta_row)
            span = cta.steady(skip)
            dropped = [w.index for w in cta.warps if w.count > len(w.events)]
            if span is None:
                print(f"  CTA {cta_row['cta']}: too few windows stamped")
                continue
            t0, t1, n = span
            print(f"  CTA {cta_row['cta']}: windows {skip} to {skip + n}, {(t1 - t0) / n:,.0f} cycles a window"
                  + (f"; warps {dropped} past their stamps' room" if dropped else ""))

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
                    held[(w.role, group(cta.name(wt.site)), source)].append(wt.t1 - wt.t0)
            print("    blocking waits (over 64 cycles), a window a warp: held, by whom")
            for (role, site, source), v in sorted(held.items(), key=lambda kv: -sum(kv[1])):
                print(f"      {sum(v) / (members[role] * n):8,.0f}  {role} at {site} <- {source}")

            pieces, margins = cta.path(t0, t1)
            print("    THE PATH, cycles a window (a release: the mean margin it binds by, its waits' length):")
            for label, v in sorted(pieces.items(), key=lambda kv: -kv[1]):
                if v / n >= 20:
                    m = margins.get(label)
                    print(f"      {v / n:8,.0f}  {label}" + (f"  (binding by {sum(m) / len(m):,.0f})" if m else ""))
            if out is not None and cta_row is row["ctas"][0]:
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


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text())["rows"]
