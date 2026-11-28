# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The predictions and the measured section of `docs/lab/projection_tile.md`: `RUNS` states each run's bands once
(`_banded.py`'s form), the newest sample read against its own run's (`lab page projection_tile --instance LABEL`,
docs/internals/tools/lab.md)."""

from __future__ import annotations

import pathlib

from _banded import Claim, write_banded


def _floor(hidden: int, loader: str = "") -> str:
    return f"hidden {hidden}, a job 128 tokens x 256 columns, three stages{loader}: against the tensor floor"


TENSOR = ", the tensor copy"


#: each run's predictions, in the order they ran
RUNS = (
    (
        "20261128T034916Z-771677-038e8f02",
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
    (
        "20261128T045918Z-121596-2eacf67b",
        "2026-11-28",
        "the first run's 5.3x at every hidden was the one loader warp's cp.async (~9 bytes a cycle an SM, lg_throttle); "
        "tma_box's 128-byte swizzled tensor copy staged the same 48 KB in 966 cycles (50.9 bytes a cycle an SM), under the "
        "tensor floor's 1,024; so a K-step is the floor's 1,024 plus what the ring fails to hide (~0-100) at hidden 256 "
        "and 512, and at 2,048 x's 16 KB from HBM (12.95 bytes a cycle an SM: 1,265) binds; a job its K-steps plus ~900 "
        "(the ring's first fill's share, its end's wait, fold and clear); the cp.async form repeats within 5%",
        (
            Claim("hidden 256: the cp.async job over its floor", lambda R: R[_floor(256)], 5.0, 5.6, "times"),
            Claim("hidden 256: the tensor-copy job over its floor", lambda R: R[_floor(256, TENSOR)], 1.12, 1.40, "times"),
            Claim("hidden 512: the tensor-copy job over its floor", lambda R: R[_floor(512, TENSOR)], 1.05, 1.30, "times"),
            Claim("hidden 2,048: the tensor-copy job over its floor", lambda R: R[_floor(2048, TENSOR)], 1.15, 1.40,
                  "times"),
        ),
    ),
)


def write(root: pathlib.Path, data: pathlib.Path, instance: str) -> list[pathlib.Path]:
    return write_banded(root, data, instance, "projection_tile", RUNS)
