# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The predictions and the measured section of `docs/lab/dependent_launch.md`: `RUNS` states each run's bands once
(`_banded.py`'s form), the newest sample read against its own run's (`lab page dependent_launch --instance LABEL`,
docs/internals/tools/lab.md)."""

from __future__ import annotations

import pathlib

from _banded import PENDING, Claim, write_banded

PLAIN = "two plain launches: a pair"
DEPENDENT = "the second launched dependent (griddepcontrol): a pair"
AFTER = "the second launched dependent, the first's trigger after its writes: a pair"

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
    (
        PENDING,
        "2026-11-28",
        "the first run's dependent pair (the trigger at the first's top) was 0.94 us slower than plain: the dependent's "
        "CTAs launched at once and shared the SMs and HBM with the first's writes; with the trigger after its writes they "
        "launch as the first's CTAs finish, so the pair keeps the launch's latency hidden (~0.3-1 us) and the prologue's "
        "overlap of the first's tail, without the contention; the other two forms repeat within 5%",
        (
            Claim(PLAIN, lambda R: R[PLAIN], 7.9, 8.7, "us a pair"),
            Claim(DEPENDENT, lambda R: R[DEPENDENT], 8.8, 9.7, "us a pair"),
            Claim("saved a pair by the trigger after the writes", lambda R: R[PLAIN] - R[AFTER], 0.1, 1.0, "us"),
        ),
    ),
)


def write(root: pathlib.Path, data: pathlib.Path, instance: str) -> list[pathlib.Path]:
    return write_banded(root, data, instance, "dependent_launch", RUNS)
