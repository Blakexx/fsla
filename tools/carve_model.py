#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CARVE MODEL: the carry's mechanics for any carve on any card, over a real cell's lists -- each phase's MMAs,
shared-memory bytes and instructions a warp, its registers and shared-memory ledger, its floors, and the window's
predicted cost. It ranks no carve until it reproduces the measured forms (KERNEL_STANDARDS §32.2; the validation rows
in docs/internals/tools/carve_model.md#validation).

    python tools/carve_model.py stats nl64k-alt-k4 [nl64k-dense ...]   # the cells' statistics
    python tools/carve_model.py events nl64k-alt-k4 [...]              # today's carve's events a CTA-window
    python tools/carve_model.py measured <SourceCounters csv> --cta-windows N

Docs: docs/internals/tools/carve_model.md
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: the box every carve in this model partitions: lists (the fold's list level, the readout's dense level and K), digits
#: (the fold's dense level, the readout's list level), channels; a window of tokens
BOX_LISTS = 32
WINDOW = 512


#: the lists' boundaries fall every this many union slots (the facts pass's prefix rows' columns)
SEGMENT = 64


@dataclass(frozen=True)
class Side:
    """One side of one box-window: its union's slots and each of its 32 lists' entries, by the union's 64-slot segments
    (a dense side: every slot in every list, `listed` False)."""
    listed: bool
    union: int
    segments: np.ndarray  #: [32, S] a list's entries in each 64-slot segment of the union

    @property
    def entries(self) -> np.ndarray:
        return self.segments.sum(1)

    def in_chunk(self, first: int, slots: int) -> np.ndarray:
        """[32] each list's entries among union slots `first .. first + slots - 1` (`first` a multiple of 64)."""
        return self.segments[:, first // SEGMENT: -(-(first + slots) // SEGMENT)].sum(1)

    @property
    def total(self) -> int:
        return int(self.entries.sum())

    @property
    def refetch(self) -> float:
        """Rows a list-at-a-time stream fetches per row the union stages once."""
        return self.total / self.union if self.union else 0.0


@dataclass(frozen=True)
class CellStats:
    """A cell's box-windows: `write[i]`, `read[i]` the two sides of box-window `i` and `words[i]` its plan word on this
    card (the plan pass's rule), `group` query heads a write head."""
    name: str
    group: int
    write: tuple[Side, ...]
    read: tuple[Side, ...]
    words: tuple[int, ...]


def _sides(lists, plan, routes, bh: int, g: int, windows: int) -> list[Side]:
    """A side's box-windows at box group `g` of batch-head `bh`: each list's entries from the prefix rows (`pre[l][-1]`
    the boundaries of lists < l over the whole union, modulo 2^16; a list's entries the difference)."""
    out = []
    for w in range(windows):
        tokens = min(WINDOW, routes.tokens - w * WINDOW)
        if not plan.lists:
            seg = np.zeros((BOX_LISTS, -(-tokens // SEGMENT)), dtype=np.int64)
            for j in range(seg.shape[1]):
                seg[:, j] = min(SEGMENT, tokens - j * SEGMENT)
            out.append(Side(False, tokens, seg))
            continue
        pre = lists.pre[bh, w, g].long().cpu().numpy() & 0xFFFF
        before = (pre[1:BOX_LISTS + 1] - pre[:BOX_LISTS]) & 0xFFFF  #: [32, columns] a list's entries before each boundary
        seg = np.diff(np.concatenate([np.zeros((BOX_LISTS, 1), dtype=np.int64), before], 1), axis=1) & 0xFFFF
        out.append(Side(True, int(lists.count[bh, w, g]), seg))
    return out


def cell_stats(name: str) -> CellStats:
    """Every box-window's two sides on a registry cell, from the facts pass's lists as the carry reads them."""
    from measure.cells import by_name
    from measure.subjects import carry_operands
    from rola.ops import facts as facts_ops

    routes, _values, facts, carve, _unit = carry_operands(by_name(name))
    windows = -(-routes.tokens // WINDOW)
    write = facts_ops.side_lists(facts, "write") if carve.fold.lists else None
    read = facts_ops.side_lists(facts, "read") if carve.readout.lists else None
    groups = lambda plan: max(1, carve.widths[plan.lists[0]] // BOX_LISTS) if plan.lists else 1  # noqa: E731
    from tests.oracle import carry_plan as cp

    ws, rs = [], []
    for bh in range(routes.batch * routes.write_heads):
        for gr in range(groups(carve.readout)):
            for gw in range(groups(carve.fold)):
                rs += _sides(read, carve.readout, routes, bh, gr, windows)
                ws += _sides(write, carve.fold, routes, bh, gw, windows)
    words = tuple(rec.word for box in cp.plans(routes, facts, carve) for rec in box)
    assert len(words) == len(ws), "the plan records' order is the sides'"
    return CellStats(name, routes.group, tuple(ws), tuple(rs), words)


@dataclass
class Region:
    """A measured region of one launch, per CTA-window: its executed warp instructions, MMAs by SASS shape and shared
    wavefronts."""
    instructions: float = 0.0
    wavefronts: float = 0.0
    mmas: dict = None

    def add(self, n: float, op: str, wf: float) -> None:
        self.instructions += n
        self.wavefronts += wf
        if op.startswith(("HMMA", "IMMA")):
            self.mmas = self.mmas or {}
            self.mmas[op] = self.mmas.get(op, 0.0) + n


def measured(csv: Path, so: Path, cta_windows: int) -> dict[tuple, Region]:
    """THE MEASURED SIDE: one launch's `ncu --section SourceCounters --page source --print-source sass --csv` dump of the
    carry, each SASS line's executed instructions and shared wavefronts attributed to (the innermost MMA loop holding it,
    or None; its innermost carry function), a CTA-window's worth. The arm is the dump's kernel."""
    import re

    sys.path.insert(0, str(ROOT / "tools"))
    import region_ledger
    import sass
    import sass_gate

    text = csv.read_text()
    kernel = re.search(r'"Kernel Name","([^"]+)"', text).group(1)
    arm = int(re.search(r"Arm<\(int\)(\d+)", kernel).group(1))
    member = f"carry_arm_{arm}"
    arch = sass.device_arch()
    cubin = sass.cubin(so, member, arch)
    listing = sass.section(sass.disassemble(cubin), "carry_kernel", cubin.name)
    lines, labels = [], {}
    for line in listing.splitlines():
        m = re.match(r"^(\$?[\w.$]+):\s*$", line)
        if m:
            labels[m.group(1)] = len(lines)
            continue
        m = sass.INSTRUCTION.match(line)
        if m:
            w = sass_gate.WHOLE.match(line)
            pred = (m.group(2) or "").strip()
            lines.append((int(w.group(1), 16), w.group(3), w.group(4).strip(), pred) if w
                         else (int(m.group(1), 16), m.group(3), "", pred))
    loop_of = {}
    for i, (t, k) in enumerate(sass_gate.mma_loops(lines, labels)):
        for a, _, _, _ in lines[t:k + 1]:
            loop_of.setdefault(a, i)
    frames = region_ledger.frames_map(so, member, arch, "carry_kernel")
    from region_ledger import fn_of, function_map
    fmaps = {p.name: function_map(p.read_text().splitlines()) for p in (ROOT / "csrc/rola/src/carry").glob("*.cuh")}
    rows = region_ledger.read_source_counters(csv)
    detail = region_ledger.read_source_detail(csv)
    base = rows[0][0]
    out: dict[tuple, Region] = {}
    for addr, n, _samples, _stalls in rows:
        off = addr - base
        chain = [(f, ln) for f, ln in frames.get(off, []) if f in fmaps]
        fn = fn_of(fmaps[chain[0][0]], chain[0][1]) if chain else "?"
        op, wf, _ideal = detail.get(addr, ("?", 0, 0))
        out.setdefault((loop_of.get(off), fn), Region()).add(n / cta_windows, op, wf / cta_windows)
    return out


def _steps(n: int) -> tuple[int, int]:
    """A list's fold steps over `n` entries (fold.cuh's `steps`): 16-token steps while more than 8 remain, then an 8-token
    tail where any remain."""
    s16 = max(0, -(-(n - 8) // 16)) if n > 8 else 0
    return s16, int(n - 16 * s16 > 0)


@dataclass
class Events:
    """Today's carve's work a CTA-window, from the plan words and the lists (each box-window summed, then averaged)."""
    read_tiles: float = 0.0     #: readout tile-steps a warp issues (a pair's or a digit's), summed over warps
    read_mmas: float = 0.0
    read_digits: float = 0.0    #: digits stepped (state fragments loaded), summed over warps
    fold_steps16: float = 0.0
    fold_steps8: float = 0.0
    fold_mmas16: float = 0.0
    fold_mmas8: float = 0.0
    fold_lists: float = 0.0     #: (list, fold chunk) pairs
    images: float = 0.0         #: whole images stored (a half counts a half)
    read_chunks: float = 0.0
    fold_chunks: float = 0.0
    read_rows: float = 0.0      #: union slots filled, read side
    fold_rows: float = 0.0
    read_entries: float = 0.0
    fold_entries: float = 0.0
    drained: float = 0.0        #: panel slots drained
    steps: float = 0.0          #: the window's steps (entries + 1)


def events_today(s: CellStats) -> Events:
    """THE EVENTS of today's carve (8 warps, a warp 4 fold lists and the read digits `w + 8 i`, two at a time; the rule's
    chunks), a CTA-window's mean."""
    from tests.oracle import carry_plan as cp

    ev = Events()
    for rd, wr, word in zip(s.read, s.write, s.words):
        mode, rc, fc, nr, nf = cp.chunks(word)
        ev.read_chunks += nr
        ev.fold_chunks += nf
        ev.steps += nr + nf + 1
        ev.images += (1 if mode == cp.WHOLE else nr) if nr else 0
        for first in range(0, rd.union, rc or 1):
            if not nr:
                break
            slots = min(rc, rd.union - first)
            n = rd.in_chunk(first, slots)
            ev.read_rows += slots
            ev.read_entries += n.sum() if rd.listed else 0
            ev.drained += slots
            for w in range(8):
                for pair in ((w, w + 8), (w + 16, w + 24)):
                    ev.read_digits += 2
                    if rd.listed:
                        t = max(-(-int(n[d]) // 8) for d in pair)
                        ev.read_tiles += t
                        ev.read_mmas += 8 * t
                    else:
                        t = -(-slots // 8)
                        ev.read_tiles += 2 * t
                        ev.read_mmas += 8 * t
        for first in range(0, wr.union, fc or 1):
            if not nf:
                break
            slots = min(fc, wr.union - first)
            n = wr.in_chunk(first, slots)
            ev.fold_rows += slots
            ev.fold_entries += n.sum() if wr.listed else 0
            for l in range(32):
                a, b = _steps(int(n[l]))
                ev.fold_lists += 1
                ev.fold_steps16 += a
                ev.fold_steps8 += b
                ev.fold_mmas16 += 8 * a
                ev.fold_mmas8 += 8 * b
    k = len(s.words)
    for f in ev.__dataclass_fields__:
        setattr(ev, f, getattr(ev, f) / k)
    return ev


#: THE DRIVERS: each carry function's phase and the event its executed instructions and wavefronts scale with (a
#: function inside an MMA loop keyed `name@L`); a function not listed is the window's bookkeeping, per step
DRIVERS = {
    "readout": {"read_tiles": "tile_in tile_mma tile_add s16_pair panel_red masked_entry tile read_chunk@L ?@L chunk_addr@L "
                              "Entry::slots@L chunk_addr",
                "read_digits": "state_in image_lane",
                "read_chunks": "read_chunk"},
    "fold": {"fold_steps16": "step16 step16@L", "fold_steps8": "step8",
             "fold_steps": "steps steps@L entry_of entry_of@L",
             "fold_lists": "fold_chunk", "one": "fold_units next_unit pow2_half2 list_units"},
    "handoff": {"images": "store_batches", "one": "post_maxima digit_exp ceil_exp pow2"},
    "fills": {"rows": "fill", "entries": "entries_fill", "chunks": "read_fill fold_fill pre_at chunk_runs chunk_cols "
                                                                  "list_row Flow::slot_landed stage_bytes"},
    "drain": {"drained": "drain red_nonzero"},
}
_DRIVER_OF = {fn: (phase, event) for phase, evs in DRIVERS.items() for event, fns in evs.items() for fn in fns.split()}


def driver_of(loop, fn: str) -> tuple[str, str]:
    """A measured region's (phase, event)."""
    keyed = f"{fn}@L" if loop is not None else fn
    return _DRIVER_OF.get(keyed) or _DRIVER_OF.get(fn) or ("glue", "steps")


def event_count(ev: Events, event: str) -> float:
    """An event's count a CTA-window, the drivers' names."""
    return {"one": 1.0, "read_tiles": ev.read_tiles, "read_mmas": ev.read_mmas, "read_digits": ev.read_digits,
            "read_chunks": ev.read_chunks, "fold_steps16": ev.fold_steps16, "fold_steps8": ev.fold_steps8,
            "fold_steps": ev.fold_steps16 + ev.fold_steps8, "fold_mmas16": ev.fold_mmas16,
            "fold_mmas8": ev.fold_mmas8, "fold_lists": ev.fold_lists, "images": ev.images,
            "rows": ev.read_rows + ev.fold_rows, "entries": ev.read_entries + ev.fold_entries,
            "chunks": ev.read_chunks + ev.fold_chunks, "drained": ev.drained, "steps": ev.steps}[event]


def calibrate(regions: dict, ev: Events) -> dict[tuple, tuple[float, float]]:
    """Each (phase, event)'s instructions and wavefronts per event on a calibration cell."""
    tot: dict[tuple, list] = {}
    for (loop, fn), r in regions.items():
        key = driver_of(loop, fn)
        t = tot.setdefault(key, [0.0, 0.0])
        t[0] += r.instructions
        t[1] += r.wavefronts
    return {k: (v[0] / max(event_count(ev, k[1]), 1e-9), v[1] / max(event_count(ev, k[1]), 1e-9))
            for k, v in tot.items()}


def predict(prices: dict, ev: Events) -> dict[str, list]:
    """Each phase's [instructions, wavefronts] a CTA-window from the prices and a cell's events."""
    out: dict[str, list] = {}
    for (phase, event), (i, w) in prices.items():
        n = event_count(ev, event)
        o = out.setdefault(phase, [0.0, 0.0])
        o[0] += i * n
        o[1] += w * n
    return out


def by_phase(regions: dict) -> dict[str, list]:
    out: dict[str, list] = {}
    for (loop, fn), r in regions.items():
        o = out.setdefault(driver_of(loop, fn)[0], [0.0, 0.0])
        o[0] += r.instructions
        o[1] += r.wavefronts
    return out


#: THE VALIDATION CELLS: the prices are a listed side's on the first, a dense side's on the second; the rest are held out
CALIBRATION = ("nl64k-alt-k4", "nl64k-dense")
HELD_OUT = ("nl64k-alt-k4-g4", "flagship-cohort-k4", "flagship-alt-k4", "flagship-alt-k16", "flagship-dense")


def profile(cell: str) -> Path:
    """One carry launch's SourceCounters dump on `cell` (cached under build/carve_model by the binary's mtime)."""
    from rola_devtools import process

    so = ROOT / "rola_cu13" / "_C.abi3.so"
    out = ROOT / "build" / "carve_model" / f"{cell}.{int(so.stat().st_mtime)}.csv"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    launch = ("import sys, torch; sys.path.insert(0, '.'); from measure.cells import by_name; "
              "from measure.subjects import carry_operands; from rola.ops import carry; "
              f"ops = carry_operands(by_name({cell!r})); [carry.carry(*ops) for _ in range(3)]; "
              "torch.cuda.synchronize()")
    import dev_config

    done = process.run([dev_config.get("toolchain.ncu"), "--kernel-name", "regex:carry_kernel", "--launch-skip", "2",
                        "--launch-count", "1", "--section", "SourceCounters", "--page", "source", "--print-source",
                        "sass", "--csv", sys.executable, "-c", launch], cwd=ROOT, timeout=3600)
    out.write_text(done.stdout)
    return out


def validate(cells: tuple[str, ...] = CALIBRATION + HELD_OUT) -> list[str]:
    """The gate's table: each cell's instructions and wavefronts a CTA-window predicted from the calibration cells'
    prices against its measured counts, by phase."""
    sys.path.insert(0, str(ROOT / "tools"))
    import phase_ledger

    so = ROOT / "rola_cu13" / "_C.abi3.so"
    reg = {c: measured(profile(c), so, phase_ledger.cta_windows(c)) for c in cells}
    ev = {c: events_today(cell_stats(c)) for c in cells}
    prices = {"listed": calibrate(reg[CALIBRATION[0]], ev[CALIBRATION[0]]),
              "dense": calibrate(reg[CALIBRATION[1]], ev[CALIBRATION[1]])}
    rows = []
    for c in cells:
        side = "dense" if not cell_stats(c).read[0].listed else "listed"
        pred, meas = predict(prices[side], ev[c]), by_phase(reg[c])
        for k, what in ((0, "instructions"), (1, "wavefronts")):
            total_m = sum(v[k] for v in meas.values())
            total_p = sum(v[k] for v in pred.values())
            phases = " ".join(f"{ph} {100 * (pred.get(ph, [0, 0])[k] / meas[ph][k] - 1):+.0f}%"
                              for ph in sorted(meas) if meas[ph][k] > 0.02 * total_m)
            rows.append(f"{c:20s} {what:12s} {total_m:9.0f} predicted {total_p:9.0f} "
                        f"({100 * (total_p / total_m - 1):+5.1f}%)  {phases}")
    return rows


def describe(s: CellStats) -> str:
    def side(name, sides):
        u = np.array([x.union for x in sides])
        e = np.stack([x.entries for x in sides])
        per = e.flatten()
        if not sides[0].listed:
            return f"{name} dense: union {u.mean():.0f}, every slot in all {BOX_LISTS} lists"
        return (f"{name}: union {u.mean():.0f} (max {u.max()}), entries {e.sum(1).mean():.0f}, a list mean "
                f"{per.mean():.1f} p50 {np.percentile(per, 50):.0f} p90 {np.percentile(per, 90):.0f} max {per.max()}, "
                f"refetch {np.mean([x.refetch for x in sides]):.2f}x")
    return (f"{s.name} (g {s.group}, {len(s.write)} box-windows)\n  {side('write', s.write)}\n"
            f"  {side('read', s.read)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stats", help="the cells' box-window statistics")
    p.add_argument("cells", nargs="+")
    p = sub.add_parser("events", help="today's carve's events a CTA-window on cells")
    p.add_argument("cells", nargs="+")
    sub.add_parser("validate", help="the validation table: calibration cells' prices against the held-out cells")
    p = sub.add_parser("measured", help="a launch's per-region counts a CTA-window, from its SourceCounters dump")
    p.add_argument("csv", type=Path)
    p.add_argument("--so", type=Path, default=ROOT / "rola_cu13" / "_C.abi3.so")
    p.add_argument("--cta-windows", type=int, required=True)
    p.add_argument("--top", type=int, default=30)
    a = ap.parse_args()
    if a.cmd == "stats":
        for name in a.cells:
            print(describe(cell_stats(name)))
    elif a.cmd == "events":
        for name in a.cells:
            ev = events_today(cell_stats(name))
            print(name, " ".join(f"{k} {v:.1f}" for k, v in ev.__dict__.items()))
    elif a.cmd == "validate":
        print("\n".join(validate()))
    elif a.cmd == "measured":
        regions = measured(a.csv, a.so, a.cta_windows)
        total = sum(r.instructions for r in regions.values())
        print(f"{total:.0f} warp instructions a CTA-window, "
              f"{sum(r.wavefronts for r in regions.values()):.0f} shared wavefronts")
        for (loop, fn), r in sorted(regions.items(), key=lambda kv: -kv[1].instructions)[:a.top]:
            mm = " ".join(f"{k} {v:.0f}" for k, v in sorted((r.mmas or {}).items()))
            print(f"  loop {'-' if loop is None else loop:>3} {fn:22s} instr {r.instructions:8.0f} "
                  f"wf {r.wavefronts:7.0f} {mm}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
