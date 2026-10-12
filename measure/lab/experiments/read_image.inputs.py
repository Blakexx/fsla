# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE READOUT LAB'S INPUTS: one head's read lists as the facts pass hands them to the carry, for `read_image.cu`
(docs/internals/tools/lab.md#read-image).

The file is a header of four int32 -- the magic word 0x44414552, windows, the list level's width, the 32-digit groups --
then the slots `[windows][width][512]` u16 (a read list holds slots only: a token's read coefficient is in its row),
the pass starts `[windows][width][8]` u16 and the unions' counts `[windows][groups]` u16. `ordered.bin` is the same
lists with each digit's entries in a 256-slot readout pass (two of the facts pass's) dealt round-robin by residue mod 8
(a slot's bank group in the pool and the panel), so its 8-entry groups fall on distinct banks where its residues
allow; the facts pass orders them the same way within its own 128-slot passes. `offsets.bin` is `ordered.bin`'s
entries in the form the facts pass would emit for the readout, u32 `[windows][width][512]`: the pool row's byte offset
with its swizzle folded in (`(slot % 256) * 64 + 16 * ((slot >> 1) & 3)`) low, the panel row's byte offset plus one
(zero marks no entry) high.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

#: the probe cell, its first 32 windows
CELL, TOKENS = "nl64k-alt-k4", 16384


def write(directory: Path) -> list[str]:
    import torch

    from measure.cells import by_name, realize
    from measure.subjects import bundle_of
    from rola.ops import facts as facts_ops

    spec = by_name(CELL)
    routes, values, layout, statics = bundle_of(spec, realize(spec), slice(0, TOKENS))
    facts = facts_ops.facts(routes, values, layout, *statics)
    lists = facts_ops.side_lists(facts, "read")
    records = lists.slot[0].contiguous()
    windows, width = records.shape[0], records.shape[1]
    groups = lists.count.shape[-1]
    header = struct.pack("<4i", 0x44414552, windows, width, groups)
    body = [records.cpu().numpy().tobytes(), lists.at[0].contiguous().cpu().numpy().tobytes(),
            lists.count[0].contiguous().cpu().numpy().tobytes()]
    (directory / "read.bin").write_bytes(header + b"".join(body))
    ordered = records.cpu().numpy().copy()
    at = lists.at[0].cpu().numpy()
    for w in range(windows):
        for d in range(width):
            for p in range(2):
                begin, end = int(at[w, d, 2 * p]), int(at[w, d, 2 * p + 2])
                span = [int(x) for x in ordered[w, d, begin:end]]
                buckets = [[x for x in span if x & 7 == r] for r in range(8)]
                dealt = [b[j] for j in range(max(map(len, buckets), default=0)) for b in buckets if len(b) > j]
                ordered[w, d, begin:end] = dealt
    (directory / "ordered.bin").write_bytes(header + ordered.tobytes() + body[1] + body[2])
    import numpy as np

    row = ordered.astype(np.uint32) & 255
    offsets = (row * 64 + 16 * ((row >> 1) & 3)) | ((row * 80 + 1) << 16)
    (directory / "offsets.bin").write_bytes(offsets.astype(np.uint32).tobytes())
    torch.cuda.empty_cache()
    return ["read.bin", "ordered.bin", "offsets.bin"]
