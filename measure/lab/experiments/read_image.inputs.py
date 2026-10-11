# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE READOUT LAB'S INPUTS: one head's read lists as the facts pass hands them to the carry, for `read_image.cu`
(docs/internals/tools/lab.md#read-image).

The file is a header of four int32 -- the magic word 0x44414552, windows, the list level's width, the 32-digit groups --
then the slots `[windows][width][512]` u16 (a read list holds slots only: a token's read coefficient is in its row),
the pass starts `[windows][width][8]` u16 and the unions' counts `[windows][groups]` u16.
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
    torch.cuda.empty_cache()
    return ["read.bin"]
