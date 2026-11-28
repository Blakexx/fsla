# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The predictions and the measured section of `docs/lab/produce_skeleton.md`: `RUNS` states each run's bands once
(`_banded.py`'s form), the newest sample read against its own run's (`lab page produce_skeleton --instance LABEL`,
docs/internals/tools/lab.md)."""

from __future__ import annotations

import pathlib

from _banded import PENDING, Claim, write_banded

PLAIN = "the empty pipeline at the cell"
LO = PLAIN + ", the lo plane"
FLOOR = ": against the write floor"

#: each run's predictions, in the order they ran
RUNS = (
    (
        PENDING,
        "2026-11-28",
        "the outputs at the cell, rows 4 x 65,536 x 4 x 256 = 268.4 MB, scales 4 x 1.05 MB, mask words 4 x 8.39 MB, "
        "values 33.6 MB: 339.7 MB at HBM's write rate (stream_rate, 3,170 GB/s), 107.2 us; the lo plane adds the rows' "
        "268.4 MB, 608.2 MB and 191.9 us. A CTA's 165,888 bytes take 12,800 cycles at an SM's share of that rate, its "
        "20 ring stages ~2,000 and its ~18K warp instructions ~4,500 of issue (tools/budgets/produce.json): an SM's stores "
        "outrun its share, so the launch is write-bound; up to ~10% more where the scales' 4-byte entries, interleaved by "
        "write head and written by CTAs ~27 us apart, leave L2 as partial sectors (33.5 MB if each is a read-modify-write)",
        (
            Claim(PLAIN, lambda R: R[PLAIN], 107.0, 124.0, "us"),
            Claim(PLAIN + FLOOR, lambda R: R[PLAIN + FLOOR], 1.00, 1.16, "times"),
            Claim(LO, lambda R: R[LO], 192.0, 215.0, "us"),
            Claim(LO + FLOOR, lambda R: R[LO + FLOOR], 1.00, 1.12, "times"),
        ),
    ),
)


def write(root: pathlib.Path, data: pathlib.Path, instance: str) -> list[pathlib.Path]:
    return write_banded(root, data, instance, "produce_skeleton", RUNS)
