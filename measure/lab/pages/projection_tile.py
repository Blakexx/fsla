# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The predictions and the measured section of `docs/lab/projection_tile.md`: `RUNS` states each run's bands once
(`_banded.py`'s form), the newest sample read against its own run's (`lab page projection_tile --instance LABEL`,
docs/internals/tools/lab.md)."""

from __future__ import annotations

import pathlib

from _banded import PENDING, Claim, write_banded


def _floor(hidden: int) -> str:
    return f"hidden {hidden}, a job 128 tokens x 256 columns, three stages: against the tensor floor"


#: each run's predictions, in the order they ran
RUNS = (
    (
        PENDING,
        "2026-11-28",
        "a K-step's tensor floor 1,024 cycles (8 steps x 128, two warpgroups on one pipe); its x, 16 KB of the CTA's own "
        "lines, through L2 at stream_rate's distinct-line rate (30.8 bytes a cycle an SM, an aggregate limit at 132 SMs), "
        "its weights, the job's 128 KB (hidden 256), 256 KB (512) or 1 MB (2,048) every CTA reads, at the shared rate of "
        "that footprint (55.2 at 128 KB, 42.6 at 4 MB: ~53 and ~48 between); whether the two paths add on the SM or run "
        "apart is unmeasured, so a K-step is 1,024-1,126 cycles at hidden 256 (532 + 594 adding, or the tensor floor) and "
        "1,024-1,154 at 512; at 2,048 x comes from HBM (69 MB, re-read each job, overruns L2) at 12.95, 1,265 cycles; a "
        "job its K-steps plus ~900 (the ring's first fill's share, its end's wait, fold and clear)",
        (
            Claim("hidden 256: a job over its tensor floor", lambda R: R[_floor(256)], 1.18, 1.36, "times"),
            Claim("hidden 512: a job over its tensor floor", lambda R: R[_floor(512)], 1.08, 1.28, "times"),
            Claim("hidden 2,048: a job over its tensor floor", lambda R: R[_floor(2048)], 1.18, 1.36, "times"),
        ),
    ),
)


def write(root: pathlib.Path, data: pathlib.Path, instance: str) -> list[pathlib.Path]:
    return write_banded(root, data, instance, "projection_tile", RUNS)
