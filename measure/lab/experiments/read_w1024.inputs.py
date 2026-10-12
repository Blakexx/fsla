# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE W = 1024 READOUT LAB'S INPUTS: nl64k-alt-k4's read lists (one head, 32 windows of 512 tokens) merged in pairs
into 16 windows of 1024 tokens, for `read_w1024.cu` (docs/internals/tools/lab.md#read-w1024): a box's union ranks of the
second window follow the first's (a box's union a window holds at most 512 tokens, asserted), a list's entries are the
two windows' concatenated, and one 512-slot readout pass holds them (`at` 0 then the count). The files are
`read_image.inputs.py`'s three in its formats: `read.bin`, `ordered.bin` (each list dealt round-robin by residue over
the pass) and `offsets.bin` (the pool and panel byte offsets of a rank in 0..511).
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

#: the probe cell, its first 32 windows of 512 tokens
CELL, TOKENS = "nl64k-alt-k4", 16384


def write(directory: Path) -> list[str]:
    import numpy as np
    import torch

    from measure.cells import by_name, realize
    from measure.subjects import bundle_of
    from rola.ops import facts as facts_ops

    spec = by_name(CELL)
    routes, values, layout, statics = bundle_of(spec, realize(spec), slice(0, TOKENS))
    facts = facts_ops.facts(routes, values, layout, *statics)
    lists = facts_ops.side_lists(facts, "read")
    slots = lists.slot[0].cpu().numpy().astype(np.int64)
    at = lists.at[0].cpu().numpy().astype(np.int64)
    count = lists.count[0].cpu().numpy().astype(np.int64)
    windows, width, cap = slots.shape
    groups = count.shape[-1]
    merged, plain, ordered = windows // 2, np.zeros((windows // 2, width, cap), np.uint16), None
    at2 = np.zeros((merged, width, 8), np.uint16)
    count2 = np.zeros((merged, groups), np.uint16)
    ordered = np.zeros_like(plain)
    for v in range(merged):
        first, second = count[2 * v], count[2 * v + 1]
        assert (first + second <= 512).all(), "a box's union over two windows exceeds a 512-slot pass"
        count2[v] = first + second
        for d in range(width):
            g = d * groups // width
            a = slots[2 * v, d, at[2 * v, d, 0]:at[2 * v, d, 4]]
            b = slots[2 * v + 1, d, at[2 * v + 1, d, 0]:at[2 * v + 1, d, 4]] + first[g]
            span = np.concatenate([a, b])
            plain[v, d, :len(span)] = span
            buckets = [[x for x in span if x & 7 == r] for r in range(8)]
            dealt = [bk[j] for j in range(max(map(len, buckets), default=0)) for bk in buckets if len(bk) > j]
            ordered[v, d, :len(dealt)] = dealt
            at2[v, d, 1:] = len(span)
    header = struct.pack("<4i", 0x44414552, merged, width, groups)
    tail = at2.tobytes() + count2.tobytes()
    (directory / "read.bin").write_bytes(header + plain.tobytes() + tail)
    (directory / "ordered.bin").write_bytes(header + ordered.tobytes() + tail)
    row = ordered.astype(np.uint32) & 511
    offsets = (row * 64 + 16 * ((row >> 1) & 3)) | ((row * 80 + 1) << 16)
    (directory / "offsets.bin").write_bytes(offsets.astype(np.uint32).tobytes())
    torch.cuda.empty_cache()
    return ["read.bin", "ordered.bin", "offsets.bin"]
