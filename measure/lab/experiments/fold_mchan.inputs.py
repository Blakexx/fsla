# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE M = CHANNEL FOLD'S INPUTS: the same lists as `fold_part.inputs.py` (one head of nl64k-alt-k4, 32 windows), for
`fold_mchan.cu` (docs/internals/tools/lab.md#fold-mchan)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("fold_part_inputs", Path(__file__).with_name("fold_part.inputs.py"))
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def write(directory: Path) -> list[str]:
    return _MODULE.write(directory)
