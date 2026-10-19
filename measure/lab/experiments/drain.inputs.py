# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE DRAIN'S INPUTS: each box-window's read union for `drain.cu` (docs/lab/drain.md) -- `read_stream.inputs.py`'s
three cells, a union's slots its tokens in rank order (the facts pass's rank slots) -- and the head's exact `num`.

CTA `b` is box `b` of one head (the carry's order): channel tile `b % 2`, slice group `(b // 2) % 8`, read group
`b // 16`. A panel word of box `b`, window `w`, slot `s`, word `j` holds `pattern(w, b, s, j)`; the drain adds it at its
token's row of `num` (`[L tokens][32 words]`, d_v 64: a token's 128-byte row, a channel tile's 16 words its half), so
the 8 slice groups' boxes and every read group whose union holds the token add into the same words. Files
(little-endian), each dataset's directory as `read_stream.inputs.py`'s:
- `head.bin`: int32 (magic 0x4E494152, ctas, windows).
- `counts.bin`: u16 `[windows][8 read groups]`: a box-window's union.
- `tokens.bin`: u16 `[windows][8 read groups][512]`: its tokens in rank order (a window's token index).
- `num.bin`: u32 `[windows x 512][32]`: the head's `num` after one launch.
"""
from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stream = _load("read_stream.inputs")
base = _load("read_image.inputs")
CTAS, WORDS = 80, 16


def pattern(w: int, b: int, s: np.ndarray) -> np.ndarray:
    """`drain.cu`'s panel words of box `b`, window `w`, slots `s`: `[len(s)][16]` as u32 in u64."""
    s = np.asarray(s, np.uint64)[:, None]
    j = np.arange(WORDS, dtype=np.uint64)[None, :]
    return (np.uint64(w) * np.uint64(2654435761) + np.uint64(b) * np.uint64(40503) + s * np.uint64(16) + j) & np.uint64(
        0xFFFFFFFF)


def write(directory: Path) -> list[str]:
    out = []
    for sub, cell, windows in stream.DATASETS:
        (directory / sub).mkdir(exist_ok=True)
        out += [str(Path(sub) / f) for f in dataset(directory / sub, cell, windows)]
    return out


def dataset(directory: Path, cell: str, windows: int) -> list[str]:
    _, _, count, union = base.lists(windows, with_tokens=True, cell=cell)
    groups = count.shape[1]
    num = np.zeros((windows * 512, 2 * WORDS), np.uint64)
    for w in range(windows):
        for b in range(CTAS):
            ct, g = b % 2, b // 16
            n = int(count[w, g])
            tokens = 512 * w + union[w, g, :n]
            num[tokens, WORDS * ct:WORDS * ct + WORDS] += pattern(w, b, np.arange(n))

    (directory / "head.bin").write_bytes(struct.pack("<3i", 0x4E494152, CTAS, windows))
    (directory / "counts.bin").write_bytes(count.astype(np.uint16).tobytes())
    (directory / "tokens.bin").write_bytes(union.astype(np.uint16).tobytes())
    (directory / "num.bin").write_bytes((num & np.uint64(0xFFFFFFFF)).astype(np.uint32).tobytes())
    lines = count.mean() * CTAS
    print(f"drain {cell}: a box-window's union {count.mean():.1f} tokens ({groups} read groups), "
          f"{lines:.0f} half-line requests a window over the head's {CTAS} boxes")
    return ["head.bin", "counts.bin", "tokens.bin", "num.bin"]


if __name__ == "__main__":
    write(Path(sys.argv[1]))
