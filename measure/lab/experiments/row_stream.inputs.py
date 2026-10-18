# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE ROW STREAM'S INPUTS: real write lists of nl64k-alt-k4 for `row_stream.cu`
(docs/internals/tools/lab.md#row-stream), the rows they read, shared between the boxes as the carry's are, the warps'
stage records, and the exact state the fold over them must reach.

CTA `b` is box `b` of one head (the carry's order, its channel tile fastest): channel tile `b % 2`, write group
`(b // 2) % 8` (its 32 lists), level-1 slice group `b // 16`; every CTA folds windows `0 .. WINDOWS - 1`, so a token's
slice is read by each box of its slice group and its values by each box of its channel tile, as the carry reads them.
Files (little-endian):
- `lists.bin`: int32 header (magic 0x4D525453, ctas, windows, groups); then per window, per write group: u16 `n[32]`
  (a list's entries), u16 `tok[32][512]` (an entry's token in its window), u16 `count` (the union's slots) and u16
  `union[512]` (a slot's token).
- `slices.bin`: f16 `[windows][512][8][16]`: a token's level-1 slices, 32 bytes a slice group (the carry's u8 slice
  bytes), 16 digits of 0 or 1 -- the fold reads a 64-byte slice slot whose last 32 bytes are zero-filled.
- `values.bin`: f16 `[windows][512][64]`: a token's 64 channels' values (0, 1 or 2) -- small integers, so every
  product and sum the MMAs form is exact.
- `records.bin`: int32 header (magic 0x4D525453, a record's bytes); then u16 records `[windows][groups][8 warps]`: a
  warp's stage plan for the window, as a pre-pass would emit it -- its stages `S` (in a 16-byte header), then each
  stage's 8 tokens (0xFFFF a pad row), its lists `w + 8 j` in order, each list's entries 8 a stage.
- `state.bin`: f32 `[ctas][32 lists][32 digits][32 channels]`: the sum over the windows and each list's entries of
  slice x values, digits 16 .. 31 zero -- the state every form must reach bit for bit.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

CELL, CTAS, WINDOWS, GROUPS, TILES, WARPS = "nl64k-alt-k4", 80, 64, 8, 2, 8
W = 512


def box(b: int) -> tuple[int, int, int]:
    """CTA `b`'s (channel tile, write group, slice group)."""
    return b % TILES, (b // TILES) % GROUPS, b // (TILES * GROUPS)


def write(directory: Path) -> list[str]:
    from measure.cells import by_name
    from measure.subjects import carry_operands
    from rola.ops import facts as facts_ops

    _routes, _values, facts, _carve, _unit = carry_operands(by_name(CELL))
    lists = facts_ops.side_lists(facts, "write")
    entries = lists.entries.cpu().numpy()
    pre = lists.pre.long().cpu().numpy() & 0xFFFF
    union = lists.union.cpu().numpy().astype(np.int64) & 0xFFFF
    count = lists.count.cpu().numpy()
    rng = np.random.default_rng(20261017)
    slices = rng.integers(0, 2, size=(WINDOWS, W, 8, 16)).astype(np.float16)
    values = rng.integers(0, 3, size=(WINDOWS, W, 64)).astype(np.float16)
    state = np.zeros((CTAS, 32, 32, 32), dtype=np.float64)
    boxes = [box(b) for b in range(CTAS)]
    plans: dict[tuple[int, int, int], list[np.ndarray]] = {}
    out = bytearray(struct.pack("<4i", 0x4D525453, CTAS, WINDOWS, GROUPS))
    for w in range(WINDOWS):
        for g in range(GROUPS):
            n = ((pre[0, w, g][1:33] - pre[0, w, g][:32]) & 0xFFFF)[:, -1].astype(np.int64)
            tok = np.zeros((32, W), dtype=np.uint16)
            for l in range(32):
                slots = entries[0, w, 32 * g + l, :n[l], 0].astype(np.int64) & 0xFFFF
                tok[l, :n[l]] = union[0, w, g, slots] & 0x1FF
                t = tok[l, :n[l]].astype(np.int64)
                prod = np.einsum("nsd,nc->sdc", slices[w, t].astype(np.float64), values[w, t].astype(np.float64))
                for b, (ct, gb, sg) in enumerate(boxes):
                    if gb == g:
                        state[b, l, :16] += prod[sg, :, 32 * ct:32 * ct + 32]
            for wp in range(WARPS):
                stages = [np.pad(tok[wp + 8 * j, k:min(k + 8, n[wp + 8 * j])], (0, max(0, k + 8 - n[wp + 8 * j])),
                                 constant_values=0xFFFF)
                          for j in range(4) for k in range(0, n[wp + 8 * j], 8)]
                plans[w, g, wp] = stages
            un = np.zeros(W, dtype=np.uint16)
            u = int(count[0, w, g])
            un[:u] = union[0, w, g, :u] & 0x1FF
            out += n.astype(np.uint16).tobytes() + tok.tobytes() + struct.pack("<H", u) + un.tobytes()
    (directory / "lists.bin").write_bytes(bytes(out))
    most = max(len(x) for x in plans.values())
    rec = np.zeros((WINDOWS, GROUPS, WARPS, 8 * (1 + most)), dtype=np.uint16)
    for (w, g, wp), stages in plans.items():
        rec[w, g, wp, 0] = len(stages)
        if stages:
            rec[w, g, wp, 8:8 * (1 + len(stages))] = np.concatenate(stages)
    (directory / "records.bin").write_bytes(struct.pack("<2i", 0x4D525453, rec.shape[-1] * 2) + rec.tobytes())
    (directory / "slices.bin").write_bytes(slices.tobytes())
    (directory / "values.bin").write_bytes(values.tobytes())
    assert state.max() < 2 ** 24, "the state must stay exact in fp32"
    (directory / "state.bin").write_bytes(state.astype(np.float32).tobytes())
    return ["lists.bin", "slices.bin", "values.bin", "records.bin", "state.bin"]


if __name__ == "__main__":
    write(Path(sys.argv[1]))
