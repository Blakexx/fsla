# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The predictions and the measured section of `docs/lab/tma_box.md`: `RUNS` states each run's bands once
(`_banded.py`'s form), the newest sample read against its own run's (`lab page tma_box --instance LABEL`,
docs/internals/tools/lab.md)."""

from __future__ import annotations

import pathlib

from _banded import PENDING, Claim, write_banded

CP_ASYNC = "cp.async, a warp, 16 bytes a lane: a 48 KB stage"
BOX128 = "a tensor copy, 128-byte lines swizzled: a 48 KB stage"
BOX16 = "a tensor copy, 16-byte lines (the unswizzled bricks): a 48 KB stage"

#: each run's predictions, in the order they ran
RUNS = (
    (
        PENDING,
        "2026-11-28",
        "a stage is the CTA's own x tile (16 KB of distinct lines, stream_rate's 30.8 bytes a cycle an SM: 532 cycles) and "
        "the 128 KB of weights every CTA reads (55.2 at that footprint: 594); the two paths adding give 1,126 cycles a "
        "stage, apart 594 -- the row decides which; the 16-byte-line box's 3,072 lines a stage each a request of the "
        "copy engine's",
        (
            Claim(CP_ASYNC, lambda R: R[CP_ASYNC], 560.0, 1200.0, "cycles a stage"),
            Claim(BOX128, lambda R: R[BOX128], 560.0, 1200.0, "cycles a stage"),
            Claim("the 16-byte-line box over the 128-byte box", lambda R: R[BOX16] / R[BOX128], 1.5, 4.0, "times"),
        ),
    ),
)


def write(root: pathlib.Path, data: pathlib.Path, instance: str) -> list[pathlib.Path]:
    return write_banded(root, data, instance, "tma_box", RUNS)
