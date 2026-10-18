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
    area: int = 0        #: the plan's area on this card, bytes
    read_row: int = 0    #: a read row's bytes in the pool
    write_row: int = 0
    window: int = WINDOW
    ring: int = 0  #: rows streamed a list (a digit) at a time into each warp's ring of this many stages; 0 resident


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


def cell_stats(name: str, forced: str | None = None) -> CellStats:
    """Every box-window's two sides on a registry cell, from the facts pass's lists as the carry reads them; the plan
    words the rule's, or a forced form's (`rola.ops.carry.forced_word`) where it fits."""
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
    words = tuple(rec.word for box in cp.plans(routes, facts, carve, forced) for rec in box)
    assert len(words) == len(ws), "the plan records' order is the sides'"
    from rola.ops import carry as carry_ops

    key = carry_ops.arm_key(carve)
    row = next(r for r in carry_ops.census() if tuple(r[1:7]) == tuple(key))
    return CellStats(name, routes.group, tuple(ws), tuple(rs), words, cp.area_of(row[9], row[12], routes.group),
                     row[13], row[14])


def _sums(side: Side) -> tuple:
    """A side's segment sums as the plan reads them: the lists' entries before each segment's end."""
    return tuple(int(x) for x in np.cumsum(side.segments.sum(0)))


def merged(s: CellStats, k: int) -> CellStats:
    """THE WINDOW AT `k` x 512 TOKENS: each `k` consecutive windows of a box merged -- their unions added (their tokens
    are disjoint), each list's segments concatenated -- and each merged window planned by the same rule in the area
    its larger tables leave (the window inputs and the read tables are two windows' each: union tokens, prefix rows,
    units)."""
    from tests.oracle import carry_plan as cp

    if k == 1:
        return s
    w = WINDOW * k
    grow_fixed = 2 * (w - WINDOW) * 2 + 2 * 33 * (w - WINDOW) // SEGMENT * 2
    grow_tables = (2 * s.group * (w - WINDOW) * 2 + 2 * 33 * (w - WINDOW) // SEGMENT * s.group * 2
                   + 2 * s.group * (w - WINDOW))
    area = s.area - grow_fixed - grow_tables
    per_box = len(s.words)  # windows run in order inside each box's records; boxes follow one another
    reads, writes, words = [], [], []
    i = 0
    boxes_windows = _box_windows(s)
    for n in boxes_windows:
        for j in range(0, n, k):
            idx = list(range(i + j, min(i + j + k, i + n)))
            rd = _merge([s.read[t] for t in idx])
            wr = _merge([s.write[t] for t in idx])
            reads.append(rd)
            writes.append(wr)
            c = cp.Counts(rd.union, _sums(rd) if rd.listed else (0,) * rd.segments.shape[1],
                          wr.union, _sums(wr) if wr.listed else (0,) * wr.segments.shape[1])
            rows = tuple(0 if (s.ring and side.listed) else row
                         for row, side in ((s.read_row, rd), (s.write_row, wr)))
            words.append(cp.win_of(c, -1, *rows, area).word)
        i += n
    assert i == per_box
    return CellStats(f"{s.name}@W{w}", s.group, tuple(writes), tuple(reads), tuple(words), area, s.read_row,
                     s.write_row, w, s.ring)


#: A STREAMED SIDE'S COST beyond its MMA phase on sm_86 (labs `row_stream`, `read_stream`): cycles a CTA-window, a fixed
#: part and a part a stage of the window's busiest warp (8 rows a stage, counted x 8) -- the producer's records, cursor
#: and waits and the copies' exposed share. The fold's from three cells (fit at nl64k-alt-k4 and flagship-alt-k16,
#: flagship-cohort-k16 predicted within 4.5%); the readout's at nl64k-alt-k4, split as the fold's
STREAM = {"fold": (693.0, 14.85), "readout": (499.0, 10.68)}
#: THE READ RING: its stages' bytes (8 rows of 64 bytes and their entry words) and each depth's addition to the
#: readout's fixed part (4: the first tiles' block holds the ring, nothing ahead); the fold's ring (4 stages of 8 rows of
#: 128 bytes, a 16-token step's two and two ahead) takes the image's place, a window's read chunks all run before its
#: fold
READ_STAGE = 8 * 64 + 64
READ_RING = {4: 367.0, 5: 10.0, 6: 0.0}


def streamed(s: CellStats, stages: int) -> CellStats:
    """THE FORM WITH ROWS STREAMED: no chunk of a listed side holds rows or entries -- a read chunk is its panel -- and
    every entry fetches its row into its warp's ring (a token in several lists fetched once a list), each warp's entries
    arriving as its window's stage record. The read ring, `stages` deep, takes the area's share beside the image; the
    fold's takes the image's place. A dense side keeps its rows resident (every token is in all 32 lists). Re-planned
    by the same rule."""
    from tests.oracle import carry_plan as cp

    listed = (s.read[0].listed, s.write[0].listed)
    area = s.area - (8 * stages * READ_STAGE if listed[0] else 0)
    rows = tuple(0 if lst else row for row, lst in zip((s.read_row, s.write_row), listed))
    words = []
    for rd, wr in zip(s.read, s.write):
        c = cp.Counts(rd.union, _sums(rd) if rd.listed else (0,) * rd.segments.shape[1],
                      wr.union, _sums(wr) if wr.listed else (0,) * wr.segments.shape[1])
        words.append(cp.win_of(c, -1, *rows, area).word)
    return CellStats(f"{s.name}+ring{stages}", s.group, s.write, s.read, tuple(words), area, s.read_row, s.write_row,
                     s.window, stages)


#: THE FOLD'S RING: 8 warps x 4 stages x 8 rows of 128 bytes, in the image's place once the read chunks have run
FOLD_RING = 8 * 4 * 8 * 128


def ring_beside_panel(s: CellStats, area: int) -> float:
    """The share of a streamed form's box-windows whose fold ring fits in `area` (the plan's whole area: the image and
    the read ring are dead once the read chunks have run) beside its last read chunk's panel (64 bytes a slot), so
    that panel can drain while the fold runs; elsewhere the drain ends before the fold begins."""
    from tests.oracle import carry_plan as cp

    fit = 0
    for rd, word in zip(s.read, s.words):
        _mode, rc, _fc, nr, _nf = cp.chunks(word)
        last = rd.union - rc * (nr - 1) if nr else 0
        fit += -(-last * cp.PANEL_ROW // 128) * 128 + FOLD_RING <= area
    return fit / len(s.words)


def _merge(sides: list[Side]) -> Side:
    """Consecutive windows' sides as one: the later window's union slots follow the earlier's (its tokens rank after
    them), each list's entries spread evenly over its segment's slots and re-binned at the merged union's 64-slot
    boundaries (a segment's entries' places within it are not in the facts' prefix rows)."""
    per_slot = []
    for x in sides:
        dens = np.repeat(x.segments / SEGMENT, SEGMENT, axis=1)[:, :x.union]
        if x.union and x.segments[:, -1].any() and x.union % SEGMENT:
            last = x.union // SEGMENT  # the last segment holds fewer than 64 slots
            dens[:, last * SEGMENT:] = (x.segments[:, last] / (x.union - last * SEGMENT))[:, None]
        per_slot.append(dens)
    allslots = np.concatenate(per_slot, axis=1)
    union = allslots.shape[1]
    pad = -(-union // SEGMENT) * SEGMENT - union
    binned = np.pad(allslots, ((0, 0), (0, pad))).reshape(BOX_LISTS, -1, SEGMENT).sum(2)
    return Side(sides[0].listed, union, binned)


def _box_windows(s: CellStats) -> list[int]:
    """Each box's windows (the records run box by box, every box the call's windows)."""
    from measure.cells import by_name

    tokens = by_name(s.name.split("@")[0].split("+")[0]).tokens
    n = -(-tokens // WINDOW)
    return [n] * (len(s.words) // n)


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


def _steps(n: float) -> tuple[int, int]:
    """A list's fold steps over `n` entries (fold.cuh's `steps`): 16-token steps while more than 8 remain, then an 8-token
    tail where any remain (a merged window's `n` rounded to the nearest entry)."""
    n = int(round(n))
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
    read_rows: float = 0.0      #: union slots filled, read side
    fold_rows: float = 0.0
    read_stages: float = 0.0    #: a streamed listed side's stages of its window's busiest warp, x 8 (a digit at least one)
    fold_stages: float = 0.0    #: the same for the fold (a list's run its own)
    staged_entries: float = 0.0  #: entries staged into chunks (a resident listed side's)
    staged_chunks: float = 0.0   #: chunks whose rows or entries are staged (every side's but a streamed listed one's)
    drained: float = 0.0        #: panel slots drained
    steps: float = 0.0          #: the window's steps (entries + 1)
    read_imbalance: float = 1.0  #: the slowest warp's readout tile-steps over the mean warp's, each window
    fold_imbalance: float = 1.0  #: the same for the fold's MMA pipe time


def events_today(s: CellStats) -> Events:
    """THE EVENTS of today's carve (8 warps, a warp 4 fold lists and the read digits `w + 8 i`, two at a time; the rule's
    chunks), a CTA-window's mean."""
    from tests.oracle import carry_plan as cp

    ev = Events()
    read_max = read_mean = fold_max = fold_mean = 0.0
    for rd, wr, word in zip(s.read, s.write, s.words):
        rwarp, fwarp, rstage, fstage = np.zeros(8), np.zeros(8), np.zeros(8), np.zeros(8)
        mode, rc, fc, nr, nf = cp.chunks(word)
        ev.read_chunks += nr
        ev.staged_chunks += (0 if (s.ring and rd.listed) else nr) + (0 if (s.ring and wr.listed) else nf)
        ev.steps += nr + nf + 1
        ev.images += (1 if mode == cp.WHOLE else nr) if nr else 0
        for first in range(0, rd.union, rc or 1):
            if not nr:
                break
            slots = min(rc, rd.union - first)
            n = rd.in_chunk(first, slots)
            ev.read_rows += 0 if (s.ring and rd.listed) else slots
            if s.ring and rd.listed:
                for d in range(32):
                    rstage[d % 8] += max(1, -(-int(round(n[d])) // 8))
            ev.staged_entries += n.sum() if (rd.listed and not s.ring) else 0
            ev.drained += slots
            for w in range(8):
                for pair in ((w, w + 8), (w + 16, w + 24)):
                    ev.read_digits += 2
                    if rd.listed:
                        t = max(-(-int(round(n[d])) // 8) for d in pair)
                        ev.read_tiles += t
                        ev.read_mmas += 8 * t
                        rwarp[w] += t
                    else:
                        t = -(-slots // 8)
                        ev.read_tiles += 2 * t
                        ev.read_mmas += 8 * t
                        rwarp[w] += 2 * t
        for first in range(0, wr.union, fc or 1):
            if not nf:
                break
            slots = min(fc, wr.union - first)
            n = wr.in_chunk(first, slots)
            ev.fold_rows += 0 if (s.ring and wr.listed) else slots
            if s.ring and wr.listed:
                for l in range(32):
                    fstage[l % 8] += -(-int(round(n[l])) // 8)
            ev.staged_entries += n.sum() if (wr.listed and not s.ring) else 0
            for l in range(32):
                a, b = _steps(int(n[l]))
                ev.fold_lists += 1
                ev.fold_steps16 += a
                ev.fold_steps8 += b
                ev.fold_mmas16 += 8 * a
                ev.fold_mmas8 += 8 * b
                fwarp[l % 8] += 8 * a * PIPE["HMMA.16816.F32"] + 8 * b * PIPE["HMMA.1688.F32"]
        ev.read_stages += 8 * rstage.max()
        ev.fold_stages += 8 * fstage.max()
        read_max += rwarp.max()
        read_mean += rwarp.mean()
        fold_max += fwarp.max()
        fold_mean += fwarp.mean()
    k = len(s.words)
    for f in ev.__dataclass_fields__:
        setattr(ev, f, getattr(ev, f) / k)
    ev.read_imbalance = read_max / read_mean if read_mean else 1.0
    ev.fold_imbalance = fold_max / fold_mean if fold_mean else 1.0
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
            "rows": ev.read_rows + ev.fold_rows, "entries": ev.staged_entries, "chunks": ev.staged_chunks,
            "drained": ev.drained, "steps": ev.steps}[event]


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


#: a scheduler's cycles an MMA by SASS shape on the calibration card (`common/hardware.json`'s product rates; an
#: m16n8k8 half its k16, lab `mma_k8`), and the schedulers an SM
PIPE = {"HMMA.16816.F32": 32.132, "HMMA.16816.F16": 16.131, "HMMA.1688.F32": 16.066, "HMMA.1688.F16": 8.066}
SCHEDULERS = 4
#: each model phase's phase-ledger columns (`fills` waits on its copies in `landed`; the bookkeeping runs in `wait`)
LEDGER = {"readout": ("readout",), "fold": ("fold",), "handoff": ("maxima", "store"), "fills": ("fill", "landed"),
          "drain": ("drain",), "glue": ("wait",)}


def floors(instructions: float, wavefronts: float, pipe: float) -> dict[str, float]:
    """A phase's floors in cycles a CTA-window: its instructions over the schedulers at one a clock, its shared
    wavefronts at one a clock, its MMAs' pipe time over the schedulers."""
    return {"issue": instructions / SCHEDULERS, "shared": wavefronts, "tensor": pipe / SCHEDULERS}


def phase_counts(regions: dict) -> dict[str, list]:
    """Each phase's [instructions, wavefronts, MMA pipe cycles] from measured regions."""
    out: dict[str, list] = {}
    for (loop, fn), r in regions.items():
        o = out.setdefault(driver_of(loop, fn)[0], [0.0, 0.0, 0.0])
        o[0] += r.instructions
        o[1] += r.wavefronts
        o[2] += sum(n * PIPE[k] for k, n in (r.mmas or {}).items())
    return out


def imbalance(ev: Events) -> dict[str, float]:
    """Each phase's slowest warp over its mean warp (a phase ends at a rendezvous, so its slowest warp sets its time)."""
    return {"readout": ev.read_imbalance, "fold": ev.fold_imbalance}


#: the events whose work differs between a phase's warps (its per-digit and per-list work does not)
VARYING = {"readout": ("read_tiles",), "fold": ("fold_steps16", "fold_steps8", "fold_steps")}


def varying_share(prices: dict, ev: Events) -> dict[str, float]:
    """Each MMA phase's instructions that vary between its warps, a share of the phase's."""
    out = {}
    for phase, events in VARYING.items():
        part = sum(i * event_count(ev, e) for (p, e), (i, _w) in prices.items() if p == phase and e in events)
        whole = sum(i * event_count(ev, e) for (p, e), (i, _w) in prices.items() if p == phase)
        out[phase] = part / whole if whole else 0.0
    return out


def idle(own: dict[str, float], share: dict[str, float], ev: Events) -> float:
    """THE RENDEZVOUS' IDLE: a phase ends when its slowest warp does, so the others wait the slowest warp's extra work --
    the phase's varying share times its imbalance less one (the ledger books it in `wait`)."""
    imb = imbalance(ev)
    return sum(own.get(p, 0.0) * share.get(p, 0.0) * (imb[p] - 1.0) for p in imb)


def efficiencies(counts: dict, ledger: dict, ev: Events, prices: dict) -> dict[str, float]:
    """Each phase's measured cycles over its binding floor on a calibration cell; the bookkeeping's after the
    rendezvous' idle is taken out of the wait it is booked in."""
    own = {p: sum(ledger[x] for x in LEDGER[p]) for p in counts}
    own["glue"] = own.get("glue", 0.0) - idle(own, varying_share(prices, ev), ev)
    return {p: own[p] / max(floors(*c).values()) for p, c in counts.items()}


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


def mma_pipe(ev: Events) -> dict[str, float]:
    """Each MMA phase's pipe cycles a CTA-window from today's carve's events (the readout's fp16-accumulate on this
    card, the fold's fp32)."""
    return {"readout": ev.read_mmas * PIPE["HMMA.16816.F16"],
            "fold": ev.fold_mmas16 * PIPE["HMMA.16816.F32"] + ev.fold_mmas8 * PIPE["HMMA.1688.F32"]}


class Model:
    """THE CALIBRATED MODEL: prices and efficiencies from the calibration cells (their SourceCounters dumps and phase
    ledgers), predicting a cell's or a form's window."""

    def __init__(self, ledgers: Path):
        import json

        sys.path.insert(0, str(ROOT / "tools"))
        import phase_ledger

        so = ROOT / "rola_cu13" / "_C.abi3.so"
        self.prices, self.effs = {}, {}
        for side, cal in (("listed", CALIBRATION[0]), ("dense", CALIBRATION[1])):
            reg = measured(profile(cal), so, phase_ledger.cta_windows(cal))
            ev = events_today(cell_stats(cal))
            led = json.loads((ledgers / f"ledger_{cal}.json").read_text())["per_phase"]
            self.prices[side] = calibrate(reg, ev)
            self.effs[side] = efficiencies(phase_counts(reg), led, ev, self.prices[side])

    def window(self, stats: CellStats) -> dict[str, float]:
        """A cell's (or a form's) predicted cycles a CTA-window, by phase; a streamed side's cost beyond its MMA phase
        (`STREAM`, the read ring's depth) booked in `fills` -- its part a stage already the busiest warp's, so the
        rendezvous' idle does not take it again."""
        side = "listed" if stats.read[0].listed else "dense"
        ev = events_today(stats)
        pred = predict(self.prices[side], ev)
        pipe = mma_pipe(ev)
        cyc = {p: max(floors(i, w, pipe.get(p, 0.0)).values()) * self.effs[side][p] for p, (i, w) in pred.items()}
        for phase, stages in (("readout", ev.read_stages), ("fold", ev.fold_stages)):
            if stages:
                fixed, per = STREAM[phase]
                cyc["fills"] = (cyc.get("fills", 0.0) + fixed + per * stages
                                + (READ_RING[stats.ring] if phase == "readout" else 0.0))
        cyc["glue"] = cyc.get("glue", 0.0) + idle(cyc, varying_share(self.prices[side], ev), ev)
        return cyc


def rank(model: Model, base: CellStats) -> list[str]:
    """THE FORMS RANKED on this card at a cell: each -- resident, or streamed at each read ring -- at W 512 and 1024,
    best first, its cycles a 512 tokens against today's at the calibration cells' efficiencies, and with every phase at
    its floor (the phases still in turn)."""
    side = "listed" if base.read[0].listed else "dense"
    ref = sum(model.window(base).values())
    rows = []
    for form, stats in (("resident", base), *((f"streamed, read ring {d}", streamed(base, d)) for d in READ_RING)):
        for k in (1, 2):
            form_k = merged(stats, k)
            eff = model.effs[side]
            at_eff = sum(model.window(form_k).values()) / k
            model.effs[side] = dict.fromkeys(eff, 1.0)
            at_floor = sum(model.window(form_k).values()) / k
            model.effs[side] = eff
            rows.append((at_eff, f"{base.name:20s} {form:24s} W {WINDOW * k:5d}: {at_eff / ref:5.2f}x today, "
                                 f"phases at their floors {at_floor / ref:5.2f}x"))
    return [line for _, line in sorted(rows)]


def validate(cells: tuple[str, ...] = CALIBRATION + HELD_OUT, ledgers: Path | None = None) -> list[str]:
    """The gate's table: each cell's instructions and wavefronts a CTA-window predicted from the calibration cells'
    prices against its measured counts, by phase; with `ledgers` (a directory of `phase_ledger.py --json` readings,
    `ledger_<cell>.json`, the clock locked), each window's cycles from the phases' floors at the calibration cells'
    efficiencies against the ledger's."""
    sys.path.insert(0, str(ROOT / "tools"))
    import phase_ledger

    so = ROOT / "rola_cu13" / "_C.abi3.so"
    reg = {c: measured(profile(c), so, phase_ledger.cta_windows(c)) for c in cells}
    ev = {c: events_today(cell_stats(c)) for c in cells}
    prices = {"listed": calibrate(reg[CALIBRATION[0]], ev[CALIBRATION[0]]),
              "dense": calibrate(reg[CALIBRATION[1]], ev[CALIBRATION[1]])}
    import json

    led = {c: json.loads((ledgers / f"ledger_{c}.json").read_text())["per_phase"] for c in cells} if ledgers else {}
    effs = ({side: efficiencies(phase_counts(reg[cal]), led[cal], ev[cal], prices[side])
             for side, cal in (("listed", CALIBRATION[0]), ("dense", CALIBRATION[1]))} if ledgers else {})
    rows = []
    for c in cells:
        side = "dense" if not cell_stats(c).read[0].listed else "listed"
        pred, meas = predict(prices[side], ev[c]), by_phase(reg[c])
        if ledgers:
            pipe = mma_pipe(ev[c])
            cyc = {p: max(floors(i, w, pipe.get(p, 0.0)).values()) * effs[side][p] for p, (i, w) in pred.items()}
            cyc["glue"] = cyc.get("glue", 0.0) + idle(cyc, varying_share(prices[side], ev[c]), ev[c])
            got = {p: sum(led[c][x] for x in LEDGER[p]) for p in LEDGER}
            tot_p, tot_m = sum(cyc.values()), sum(led[c].values())
            rows.append(f"{c:20s} {'cycles':12s} {tot_m:9.0f} predicted {tot_p:9.0f} ({100 * (tot_p / tot_m - 1):+5.1f}%)  "
                        + " ".join(f"{p} {100 * (cyc.get(p, 0) / got[p] - 1):+.0f}%" for p in sorted(got) if got[p]))
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
    p = sub.add_parser("validate", help="the validation table: calibration cells' prices against the held-out cells")
    p.add_argument("--ledgers", type=Path, default=None,
                   help="a directory of ledger_<cell>.json phase-ledger readings (clock locked): adds cycles")

    p = sub.add_parser("forms", help="predicted window cycles of forced forms against the rule's, on a cell")
    p.add_argument("cell")
    p.add_argument("forms", nargs="+")
    p.add_argument("--ledgers", type=Path, required=True)
    p = sub.add_parser("window", help="predicted cycles a 512 tokens at windows of k x 512 tokens, on cells")
    p.add_argument("cells", nargs="+")
    p.add_argument("--k", type=int, nargs="+", default=[1, 2])
    p.add_argument("--ledgers", type=Path, required=True)
    p = sub.add_parser("rank", help="every form on cells, best first: its cycles a 512 tokens at the calibration cells' "
                                    "efficiencies and with every phase at its floor")
    p.add_argument("cells", nargs="+")
    p.add_argument("--ledgers", type=Path, required=True)
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
        print("\n".join(validate(ledgers=a.ledgers)))
    elif a.cmd == "forms":
        model = Model(a.ledgers)
        base = sum(model.window(cell_stats(a.cell)).values())
        print(f"{a.cell} rule: {base:.0f} cycles a CTA-window")
        for form in a.forms:
            cyc = model.window(cell_stats(a.cell, form))
            print(f"  {form:18s} {sum(cyc.values()):8.0f} ({sum(cyc.values()) / base:.3f}x)  "
                  + " ".join(f"{k} {v:.0f}" for k, v in sorted(cyc.items())))
    elif a.cmd == "window":
        model = Model(a.ledgers)
        for name in a.cells:
            base = cell_stats(name)
            ref = sum(model.window(base).values())
            for form, stats0 in (("resident", base), *((f"streamed, read ring {d}", streamed(base, d)) for d in READ_RING)):
                for k in a.k:
                    form_k = merged(stats0, k)
                    cyc = model.window(form_k)
                    tot = sum(cyc.values()) / k
                    beside = (f"  fold ring beside the last panel {100 * ring_beside_panel(form_k, base.area):.0f}%"
                              if form_k.ring else "")
                    print(f"{name:20s} {form:24s} W {WINDOW * k:5d}: {tot:8.0f} a 512 tokens ({tot / ref:.2f}x)  "
                          + " ".join(f"{p} {v / k:.0f}" for p, v in sorted(cyc.items())) + beside)
    elif a.cmd == "rank":
        model = Model(a.ledgers)
        for name in a.cells:
            for line in rank(model, cell_stats(name)):
                print(line)
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
