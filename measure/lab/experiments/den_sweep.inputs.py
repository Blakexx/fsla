# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE DEN SWEEP'S INPUTS: each cell's routing as the facts pass hands it to the carry, one file a cell, for
`den_sweep.cu` (docs/internals/tools/lab.md#den-sweep).

A file is a header of six int32 -- the magic word 0x314E4544, tokens, the level-S index, whether level R's write side is sparse,
whether level S's read side is sparse, the window -- then level S's write plane and level R's write plane as f16 (a u8
plane's values exact), level S's read plane and level R's read plane as u8, and the write gain (`fm`'s m) as fp32, each
`[tokens][256]` but the gain. Level R is the list level of `rola.ops.facts`, level S the other.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

#: the probe cells at the flagship widths with 32 windows each, one of each draw
CELLS = ("flagship-alt-k4-L16384", "flagship-alt-k16-L16384", "flagship-cohort-k16-L16384", "flagship-dense-L16384")


def write(directory: Path) -> list[str]:
    import numpy as np
    import torch

    from measure.harness.bench_owner_skeleton import WINDOW, cell_bundle

    files = []
    for name in CELLS:
        spec, routes, _, facts, level_s, sparse_write, sparse_read = cell_bundle(name)
        level_r = 1 - level_s

        def plane(rows, dtype):
            return rows[0, :, 0, :].to(dtype).contiguous().cpu().numpy()

        header = struct.pack("<6i", 0x314E4544, spec.tokens, level_s, sparse_write, sparse_read, WINDOW)
        body = [plane(routes.amp_write[level_s], torch.float16), plane(routes.amp_write[level_r], torch.float16),
                plane(routes.amp_read[level_s], torch.uint8), plane(routes.amp_read[level_r], torch.uint8),
                facts.fm[0, :, 1].float().contiguous().cpu().numpy().astype(np.float32)]
        (directory / f"{name}.den").write_bytes(header + b"".join(a.tobytes() for a in body))
        files.append(f"{name}.den")
        torch.cuda.empty_cache()
    (directory / "cells.txt").write_text("\n".join(CELLS) + "\n")
    return files + ["cells.txt"]
