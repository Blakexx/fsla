# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE 32-DIGIT REFERENCE'S INPUTS, for `digit_box32.cu` (docs/internals/tools/lab.md#digit-box):
`read_image.inputs.py`'s 32-digit files, and the same in `fold/`, whose lists the fold stand-in folds."""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent


def write(directory: Path) -> list[str]:
    import torch

    spec = importlib.util.spec_from_file_location("read_image_inputs", HERE / "read_image.inputs.py")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    slots, at, count = base.lists(base.TOKENS // 512)
    (directory / "fold").mkdir(exist_ok=True)
    fold = base.emit(directory / "fold", slots, at, count, 256, 16, 16)
    out = base.emit(directory, slots, at, count, 256, 16, 16)
    torch.cuda.empty_cache()
    return out + [f"fold/{f}" for f in fold]
