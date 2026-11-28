# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The predictions and the measured section of `docs/lab/dependent_launch.md`: `RUNS` states each run's bands once
(`_banded.py`'s form), the newest sample read against its own run's (`lab page dependent_launch --instance LABEL`,
docs/internals/tools/lab.md)."""

from __future__ import annotations

import pathlib

from _banded import Claim, write_banded

PLAIN = "two plain launches: a pair"
DEPENDENT = "the second launched dependent (griddepcontrol): a pair"

#: each run's predictions, in the order they ran
RUNS = (
    (
        "20261128T014912Z-160980-6ad3db89",
        "2026-11-27",
        "the pair's bytes (8.4 MB written, 4.2 MB of weights and 8.4 MB read) at ~3 TB/s, ~7 us, and two launch gaps; the "
        "dependent second saves its gap and its prologue (the weights, ~1.4 us)",
        (
            Claim(PLAIN, lambda R: R[PLAIN], 8.0, 14.0, "us a pair"),
            Claim("saved a pair by the dependent launch", lambda R: R[PLAIN] - R[DEPENDENT], 1.0, 3.5, "us"),
        ),
    ),
)


def write(root: pathlib.Path, data: pathlib.Path, instance: str) -> list[pathlib.Path]:
    return write_banded(root, data, instance, "dependent_launch", RUNS)
