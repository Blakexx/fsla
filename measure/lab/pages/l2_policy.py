# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The predictions and the measured section of `docs/lab/l2_policy.md`: `RUNS` states each run's bands once
(`_banded.py`'s form), the newest sample read against its own run's (`lab page l2_policy --instance LABEL`,
docs/internals/tools/lab.md). The table's hit rate is a counters record of its own, which this page does not read."""

from __future__ import annotations

import pathlib

from _banded import Claim, write_banded


def _trip(mb: int, hint: bool) -> str:
    how = "the stream evict-first and the table evict-last" if hint else "no hint"
    return f"{mb} MB table beside a 1 GiB stream, {how}: a trip"


#: each run's predictions, in the order they ran
RUNS = (
    (
        "20261128T014645Z-403597-16c6d19e",
        "2026-11-27",
        "a table line is re-read after ~2x the table's bytes of distinct insertions, and LRU keeps it iff that is under "
        "the 50 MB L2; the trip HBM-bound on its misses",
        (
            Claim("16 MB: the plain trip over the hinted", lambda R: R[_trip(16, False)] / R[_trip(16, True)], 0.90, 1.10,
                  "times"),
            Claim("40 MB: the plain trip over the hinted", lambda R: R[_trip(40, False)] / R[_trip(40, True)], 1.30, 2.00,
                  "times"),
        ),
    ),
    (
        "20261128T050251Z-715035-6e09bfc4",
        "2026-11-28",
        "the first run's trips, a kernel now a table size (the profiler's hit rate then each table's): the 16 MB plain "
        "trip 1.31x the hinted and the 40 MB 1.10x, repeating within 5%",
        (
            Claim("16 MB: the plain trip over the hinted", lambda R: R[_trip(16, False)] / R[_trip(16, True)], 1.24, 1.38,
                  "times"),
            Claim("40 MB: the plain trip over the hinted", lambda R: R[_trip(40, False)] / R[_trip(40, True)], 1.05, 1.16,
                  "times"),
        ),
    ),
)


def write(root: pathlib.Path, data: pathlib.Path, instance: str) -> list[pathlib.Path]:
    return write_banded(root, data, instance, "l2_policy", RUNS)
