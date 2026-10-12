# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE READOUT LAB'S INPUTS: one head's read lists as the facts pass hands them to the carry, for `read_image.cu`
(docs/internals/tools/lab.md#read-image), in the final form a readout step consumes.

`read.bin` is a header of four int32 -- the magic word 0x44414552, windows, the list level's width, the 32-digit groups
-- then the slots `[windows][width][512]` u16, the facts pass's pass starts `[windows][width][8]` u16 and the unions'
counts `[windows][groups]` u16: the floors and the check read it. For each step form (`m`, tokens on M; `n`, on N),
`entries_<form>.bin` is each digit's list `[windows][width][512]` of 8-byte entries, its slots dealt round-robin by
residue mod 8 (a slot's bank group in the pool and the panel) so a step's 8-entry groups fall on distinct banks where
the residues allow, an entry `(pool, word)`: the pool row's byte offset from the CTA's shared base with its swizzle
folded in (its high half the digit's steps, of 16 tokens on M and tiles of 8 on N, and bit 24 when they run past the
HEAD staged entries; zero past the count), and the coefficient times the unit (bf16, `256 x (1 + (slot ^ digit) & 1)`) low with the panel field high --
the panel row's offset past the pads' words less the lane offset of the entry's token in its step (16 x (e % 8) on M,
32 x (e % 8 / 2) on N), so a lane's pad base plus the field is the token's row and a zero field is the lane's own pad
word. `heads_<form>.bin` is each warp's window block `[windows][boxes][8][4 x HEAD x 8]`: its four digits' first HEAD
entries, a pad past the count on a residue its 8-entry group leaves free, with the digit's steps and a zero word.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

#: the probe cell, its first 32 windows
CELL, TOKENS = "nl64k-alt-k4", 16384
#: `read_image.cu`'s layout at a 256-slot pass: 16 image planes of 2 KB before the pool; the pads' words before the panel
PLANE_BYTES, PADS_BYTES, PANEL_PITCH = 2048, 192, 80
#: the coefficients' bf16 bits by `(slot ^ digit) & 1`: 256 and 512
COEFFICIENT = (0x4380, 0x4400)


def lists(windows: int) -> tuple:
    """nl64k-alt-k4's read lists over `windows` windows of 512 tokens: (slots, at, count) as int64 arrays."""
    import numpy as np

    from measure.cells import by_name, realize
    from measure.subjects import bundle_of
    from rola.ops import facts as facts_ops

    spec = by_name(CELL)
    routes, values, layout, statics = bundle_of(spec, realize(spec), slice(0, 512 * windows))
    facts = facts_ops.facts(routes, values, layout, *statics)
    side = facts_ops.side_lists(facts, "read")
    return tuple(x[0].cpu().numpy().astype(np.int64) for x in (side.slot, side.at, side.count))


def _row(slot):
    return slot * 64 + 16 * ((slot >> 1) & 3)


def emit(directory: Path, slots, at, count, pass_slots: int, planes: int, head: int) -> list[str]:
    """Write the files for a `pass_slots`-slot readout pass over `planes` image planes with HEAD entries a digit staged,
    from read lists whose every box's union fits one pass (a digit's entries `slots[w, d, at[w, d, 0]:at[w, d, -1]]`)."""
    import numpy as np

    windows, width, cap = slots.shape
    groups = count.shape[-1]
    assert (count <= pass_slots).all(), "a box's union exceeds one readout pass"
    header = struct.pack("<4i", 0x44414552, windows, width, groups)
    body = slots.astype(np.uint16).tobytes() + at.astype(np.uint16).tobytes() + count.astype(np.uint16).tobytes()
    (directory / "read.bin").write_bytes(header + body)
    pool = planes * PLANE_BYTES
    fold = {"m": lambda e: 16 * (e % 16 % 8), "n": lambda e: 32 * (e % 8 // 2)}
    per = {"m": 16, "n": 8}
    boxes = width // 32
    for form in ("m", "n"):
        entries = np.zeros((windows, width, cap, 2), np.uint32)
        heads = np.zeros((windows, boxes, 8, 4 * head * 2), np.uint32)
        for w in range(windows):
            for d in range(width):
                span = [int(x) for x in slots[w, d, at[w, d, 0]:at[w, d, -1]]]
                buckets = [[x for x in span if x & 7 == r] for r in range(8)]
                dealt = [b[j] for j in range(max(map(len, buckets), default=0)) for b in buckets if len(b) > j]
                n = len(dealt)
                steps = -(-n // per[form])
                tag = (steps | (steps * per[form] > head) << 8) << 16
                for e, s in enumerate(dealt):
                    field = PADS_BYTES + PANEL_PITCH * s - fold[form](e)
                    entries[w, d, e] = (pool + _row(s) | tag, COEFFICIENT[(s ^ d) & 1] | field << 16)
                box, warp, i = d // 32, d % 8, d % 32 // 8
                block = heads[w, box, warp]
                for e in range(head):
                    if e < n:
                        block[2 * (head * i + e):2 * (head * i + e) + 2] = entries[w, d, e]
                        continue
                    group = e // 8 * 8
                    used = {dealt[x] & 7 for x in range(group, min(n, group + 8))}
                    pads = [r for r in range(8) if r not in used]
                    block[2 * (head * i + e)] = pool + _row(pads[(e - max(n, group)) % len(pads)]) | tag
        (directory / f"entries_{form}.bin").write_bytes(entries.tobytes())
        (directory / f"heads_{form}.bin").write_bytes(heads.tobytes())
    return ["read.bin", "entries_m.bin", "entries_n.bin", "heads_m.bin", "heads_n.bin"]


def write(directory: Path) -> list[str]:
    import torch

    slots, at, count = lists(TOKENS // 512)
    out = emit(directory, slots, at, count, 256, 16, 16)
    torch.cuda.empty_cache()
    return out
