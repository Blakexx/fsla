# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE FOLD PART'S INPUTS: one head's write lists as the facts pass hands them to the carry, for `fold_part.cu`
(docs/internals/tools/lab.md#fold-part).

The file is a header of four int32 -- the magic word 0x444C4F46, windows, the list level's width, the 32-digit groups --
then the records `[windows][width][512]` (8 bytes each: slot, 0, c_v, c_m), the pass starts `[windows][width][8]` u16
and the unions' counts `[windows][groups]` u16. `ordered.bin` is the same lists with each fold unit's entries reordered
so a step's 8-entry halves hold distinct slot residues mod 8 where the list allows (a residue is a row's bank group in
the pool), then each unit's heads `[windows][width][units][16]` (8-byte records): a unit of at most 16 entries dealt
across both halves of its step by residue, the pads between them on residues their half leaves free (c = 0).
`packed.bin` is the same heads for 8-token steps: a unit's entries first in that residue order, then the pads, each on
a residue its 8-entry group leaves free.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

#: the probe cell, its first 32 windows
CELL, TOKENS = "nl64k-alt-k4", 16384
#: the even box arm's fold unit, in passes (`Plan::kFoldUnit`; `fold_part.cu` checks it), and a window's units
UNIT_PASSES, UNITS = 2, 2


def _halves(res: list[int]) -> list[int]:
    """An order of entries with residues `res`: each 8-entry half takes one entry of each residue still held, the
    fullest residues first, and only then repeats one."""
    buckets = [[i for i, r in enumerate(res) if r == k] for k in range(8)]
    out: list[int] = []
    while len(out) < len(res):
        half: list[int] = []
        for k in sorted(range(8), key=lambda k: -len(buckets[k])):
            if buckets[k] and len(half) < min(8, len(res) - len(out)):
                half.append(buckets[k].pop())
        while len(half) < min(8, len(res) - len(out)):
            half.append(buckets[max(range(8), key=lambda k: len(buckets[k]))].pop())
        out += half
    return out


def _dealt(span):
    """A first step of at most 16 entries: residue by residue, fullest first, entries dealt to the emptier half, so a
    residue's copies split across the halves; each half's pads on the residues it leaves free, c = 0."""
    import numpy as np

    halves: list[list[int]] = [[], []]
    res = [int(x) & 7 for x in span[:, 0]]
    for k in sorted(range(8), key=lambda k: -res.count(k)):
        for i in (i for i, r in enumerate(res) if r == k):
            h = min((0, 1), key=lambda h: (k in {res[x] for x in halves[h]}, len(halves[h])))
            h = h if len(halves[h]) < 8 else 1 - h
            halves[h].append(i)
    head = np.zeros((16, 4), dtype=np.uint16)
    for h in range(2):
        free = [k for k in range(8) if k not in {res[x] for x in halves[h]}]
        for e in range(8):
            head[8 * h + e] = span[halves[h][e]] if e < len(halves[h]) else (free.pop(0), 0, 0, 0)
    return head


def _packed(span):
    """A unit's heads for 8-token steps: its first 16 entries in residue order, then pads on their group's free
    residues."""
    import numpy as np

    head = np.zeros((16, 4), dtype=np.uint16)
    n = min(len(span), 16)
    head[:n] = span[:n]
    for g in range(2):
        free = [k for k in range(8) if k not in {int(x) & 7 for x in head[8 * g:min(n, 8 * g + 8), 0]}]
        for e in range(max(n, 8 * g), 8 * g + 8):
            head[e] = (free.pop(0), 0, 0, 0)
    return head


def _ordered(records, at):
    """Each unit's entries reordered by `_halves`, and its heads: a unit of at most 16 entries `_dealt`, a longer unit's
    first 16 in that order."""
    import numpy as np

    records = records.copy()
    windows, width = at.shape[:2]
    heads = np.zeros((windows, width, UNITS, 16, 4), dtype=np.uint16)
    packed = np.zeros_like(heads)
    for w in range(windows):
        for j in range(width):
            for u in range(UNITS):
                begin, end = int(at[w, j, UNIT_PASSES * u]), int(at[w, j, UNIT_PASSES * (u + 1)])
                span = records[w, j, begin:end]
                order = _halves([int(x) & 7 for x in span[:, 0]])
                records[w, j, begin:end] = span[order]
                heads[w, j, u] = _dealt(span) if end - begin <= 16 else records[w, j, begin:begin + 16]
                packed[w, j, u] = _packed(records[w, j, begin:end])
    return records, heads, packed


def write(directory: Path) -> list[str]:
    import torch

    from measure.cells import by_name, realize
    from measure.subjects import bundle_of, facts_form
    from rola.ops import facts as facts_ops

    spec = by_name(CELL)
    routes, values, layout, statics = bundle_of(spec, realize(spec), slice(0, TOKENS))
    facts = facts_ops.facts(routes, values, layout, *statics, facts_form(spec))
    lists = facts_ops.side_lists(facts, "write")
    records = lists.entries[0].contiguous()
    windows, width = records.shape[0], records.shape[1]
    groups = lists.count.shape[-1]
    header = struct.pack("<4i", 0x444C4F46, windows, width, groups)
    body = [records.cpu().numpy().tobytes(), lists.at[0].contiguous().cpu().numpy().tobytes(),
            lists.count[0].to(torch.int16).contiguous().cpu().numpy().tobytes()]
    (directory / "lists.bin").write_bytes(header + b"".join(body))
    host = records.cpu().numpy().view("uint16").reshape(windows, width, -1, 4)
    ordered, heads, packed = _ordered(host, lists.at[0].cpu().numpy().view("uint16"))
    header = struct.pack("<5i", 0x4F4C4F46, windows, width, groups, UNIT_PASSES)
    (directory / "ordered.bin").write_bytes(header + ordered.tobytes() + body[1] + body[2] + heads.tobytes())
    (directory / "packed.bin").write_bytes(packed.tobytes())
    torch.cuda.empty_cache()
    return ["lists.bin", "ordered.bin", "packed.bin"]
