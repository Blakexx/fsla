# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE 16-DIGIT BOX LAB'S INPUTS, for `digit_box.cu` (docs/internals/tools/lab.md#digit-box): nl64k-alt-k4's read
lists re-ranked per 16-digit half of each 32-digit box (a half's union its own slots, at most 256, asserted) in
`read_image.inputs.py`'s files and formats for 16-digit boxes; and in `fold/` the 32-digit box's files, whose lists the
fold stand-in folds (both digit boxes of a list group fold the same tokens)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent


def write(directory: Path) -> list[str]:
    import numpy as np
    import torch

    spec = importlib.util.spec_from_file_location("read_image_inputs", HERE / "read_image.inputs.py")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    slots, at, count = base.lists(base.TOKENS // 512)
    (directory / "fold").mkdir(exist_ok=True)
    fold = base.emit(directory / "fold", slots, at, count, 256, 16, 16)
    windows, width, _ = slots.shape
    half = slots.copy()
    count16 = np.zeros((windows, width // 16), np.int64)
    for w in range(windows):
        for b in range(width // 16):
            rows = range(16 * b, 16 * b + 16)
            union = sorted({int(x) for d in rows for x in slots[w, d, at[w, d, 0]:at[w, d, -1]]})
            rank = {s: i for i, s in enumerate(union)}
            count16[w, b] = len(union)
            for d in rows:
                half[w, d, at[w, d, 0]:at[w, d, -1]] = [rank[int(x)] for x in slots[w, d, at[w, d, 0]:at[w, d, -1]]]
    out = base.emit(directory, half, at, count16, 256, 16, 16, digits=16)
    torch.cuda.empty_cache()
    return out + [f"fold/{f}" for f in fold]
