# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE READ STREAM'S INPUTS: nl64k-alt-k4's read lists for `read_stream.cu` (docs/internals/tools/lab.md#read-stream)
in the readout's tokens-on-N form (`read_image.inputs.py`'s coloring and placement), each warp's stage records, the
read rows shared between the boxes as the carry's are, and each CTA's exact panel.

CTA `b` is box `b` of one head (the carry's order, its channel tile fastest): channel tile `b % 2`, level-0 slice group
`(b // 2) % 8` (its read rows' 32 lists), read group `b // 16` (its 32 read digits); every CTA reads windows
`0 .. WINDOWS - 1`, so a token's read row is fetched by each box of its slice group. Files (little-endian):
- `head.bin`: int32 (magic 0x4D525453, ctas, windows, a record's bytes).
- `tiles.bin`: u16 `[windows][256]`: a read digit's 8-token tiles, at least one (an empty digit's all pads, as the
  readout steps every digit's first tile).
- `records.bin`: u16 records `[windows][8 read groups][8 warps]`: a warp's stage plan for the window -- its 4 digits'
  tiles and its stages in a 16-byte header, then each stage's 8 entries of 8 bytes: the token (0xFFFF a pad) and the
  entry word (`read_image.inputs.py`'s: the coefficient low, the panel row's offset past the lane's base high); the
  stages in the readout's order, every digit's first tile, then each digit's rest.
- `rows.bin`: bf16 `[windows x 512][8 slice groups][32 lists]`: a token's read row, its 32 lists one value, 0.0625 or
  0.125 -- so every product the MMAs form is exact and a misplaced row changes the sum.
- `panel.bin`: u32 `[ctas][256 slots][16 words]`: each CTA's panel summed over the windows, the check's reference.
"""
from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

CTAS, WINDOWS, TILES, SLICES, LISTS = 80, 64, 2, 8, 32
#: the panel's slots and words, and the bf16 bits of a read row's two values
SLOTS, WORDS, VALUE = 256, 16, (0x3D80, 0x3E00)


def box(b: int) -> tuple[int, int]:
    """CTA `b`'s (slice group, read group)."""
    return (b // TILES) % SLICES, b // (TILES * SLICES)


def one_add(k: int) -> int:
    """A panel word's one add at the unit (`read_image.cuh`'s, tokens on N): channels lo = 16 (k / 8) + k % 8 and
    hi = lo + 8, each -/+ the lists' count, packed hi x 2^16 + lo mod 2^32."""
    lo = 16 * (k >> 3) + (k & 7)
    v = lambda ch: LISTS if ch & 1 else -LISTS  # noqa: E731
    return (v(lo + 8) * 65536 + v(lo)) & 0xFFFFFFFF


def write(directory: Path) -> list[str]:
    spec = importlib.util.spec_from_file_location("read_image_inputs", HERE / "read_image.inputs.py")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    slots, at, count, union = base.lists(WINDOWS, with_tokens=True)
    width = slots.shape[1]
    groups = width // 32
    rng = np.random.default_rng(20261018)
    high = rng.integers(0, 2, size=(WINDOWS * 512, SLICES))
    rows = np.where(high[..., None] == 1, VALUE[1], VALUE[0]).astype(np.uint16).repeat(LISTS, axis=-1)

    tiles = np.zeros((WINDOWS, width), np.uint16)
    plans: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    heads: dict[tuple[int, int, int], list[int]] = {}
    m = np.zeros((CTAS, SLOTS), np.int64)
    panel_rows = base.DUMMY_ROWS + 8 * base.PANEL_PITCH

    for w in range(WINDOWS):
        for g in range(groups):
            spans = [[int(x) for x in slots[w, 32 * g + d, at[w, 32 * g + d, 0]:at[w, 32 * g + d, -1]]]
                     for d in range(32)]
            new, _extent = base.color(spans, int(count[w, g]))
            token_of = {s_new: int(union[w, g, s_old]) for s_old, s_new in new.items()}
            live_slots, live_mult, live_tok = [], [], []
            entries: dict[int, list[tuple[int, int]]] = {}
            for d in range(32):
                placed = base.place([new[s] for s in spans[d]]) or [(res, True) for res in range(8)]
                tiles[w, 32 * g + d] = len(placed) // 8
                entries[d] = []
                for e, (s, pad) in enumerate(placed):
                    field = (base.DUMMY_ROWS if pad else panel_rows) + base.PANEL_PITCH * s - base.FOLD["n"](e)
                    word = field << 16 | (0 if pad else base.COEFFICIENT[(s ^ d) & 1])
                    entries[d].append((0xFFFF if pad else token_of[s], word))
                    if not pad:
                        live_slots.append(s)
                        live_mult.append(1 + ((s ^ d) & 1))
                        live_tok.append(512 * w + token_of[s])
            for warp in range(8):
                ds = [warp + 8 * i for i in range(4)]
                plans[w, g, warp] = [x for d in ds for x in entries[d][:8]] + [x for d in ds for x in entries[d][8:]]
                heads[w, g, warp] = [len(entries[d]) // 8 for d in ds]
            s_arr, mult, tok = np.array(live_slots), np.array(live_mult), np.array(live_tok)
            for b in range(CTAS):
                sg, rg = box(b)
                if rg == g and len(s_arr):
                    np.add.at(m[b], s_arr, mult * (1 + high[tok, sg]))

    stages = max(len(x) // 8 for x in plans.values())
    record = 16 + 64 * stages
    rec = np.zeros((WINDOWS, groups, 8, record // 4), np.uint32)
    for (w, g, warp), plan in plans.items():
        h = heads[w, g, warp]
        rec[w, g, warp, 0] = h[0] | h[1] << 16
        rec[w, g, warp, 1] = h[2] | h[3] << 16
        rec[w, g, warp, 2] = len(plan) // 8
        if plan:
            rec[w, g, warp, 4:4 + 2 * len(plan)] = np.array(plan, np.uint32).reshape(-1)

    words = np.array([one_add(k) for k in range(WORDS)], np.uint64)
    panel = ((m.astype(np.uint64)[..., None] * words) & 0xFFFFFFFF).astype(np.uint32)
    (directory / "head.bin").write_bytes(struct.pack("<4i", 0x4D525453, CTAS, WINDOWS, record))
    (directory / "tiles.bin").write_bytes(tiles.tobytes())
    (directory / "records.bin").write_bytes(rec.tobytes())
    (directory / "rows.bin").write_bytes(rows.tobytes())
    (directory / "panel.bin").write_bytes(panel.tobytes())
    print(f"read_stream: a record {record} bytes ({stages} stages at most), "
          f"{tiles.sum() / (WINDOWS * groups):.1f} tiles a box-window")
    return ["head.bin", "tiles.bin", "records.bin", "rows.bin", "panel.bin"]


if __name__ == "__main__":
    write(Path(sys.argv[1]))
