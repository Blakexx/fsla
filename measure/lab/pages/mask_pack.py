# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The predictions and the measured section of `docs/lab/mask_pack.md`: `RUNS` states each run's bands once
(`_banded.py`'s form), the newest sample read against its own run's (`lab page mask_pack --instance LABEL`,
docs/internals/tools/lab.md)."""

from __future__ import annotations

import pathlib

from _banded import Claim, write_banded

UNIT = "cycles a block"
BLOCK = "256 columns, {}% of digits nonzero: a 64-token block"
DOUBLE = "256 columns, {}% of digits nonzero, the halves double-buffered: a 64-token block"

#: each run's predictions, in the order they ran
RUNS = (
    (
        "20261128T014617Z-889657-f2c62487",
        "2026-11-27",
        "a warp's 16 tokens x 256 columns at ~560 instructions (128 compares, 128 ballots, ~160 keeps, 8 columns x 2 "
        "gathers x 8, ~20 for the halves' store, the barrier and the words) at one a clock, a warpgroup's 4 warps on 4 "
        "schedulers; no branch on the data, so flat in density",
        tuple(Claim(BLOCK.format(p), lambda R, p=p: R[BLOCK.format(p)], 500.0, 900.0, UNIT) for p in (10, 50, 100))
        + (Claim("the densest block over the sparsest", lambda R: R[BLOCK.format(100)] / R[BLOCK.format(10)], 0.97, 1.03,
                 "times"),),
    ),
    (
        "20261128T034805Z-927513-666b225e",
        "2026-11-28",
        "the first run's 1,130 cycles a block are 500 instructions a warp at 0.44 a clock (the SASS: 352 of the ballots and "
        "their keeps, 120 of the gathers): latency, not count, so no form with fewer instructions is in the SASS; what it "
        "shows removable is the trip's second barrier (the halves double-buffered, a buffer the trip's parity's), one "
        "rendezvous of the pair's 64 threads, ~40-150 cycles",
        tuple(Claim(BLOCK.format(p), lambda R, p=p: R[BLOCK.format(p)], 1074.0, 1187.0, "cycles a block") for p in (10, 50, 100))
        + (Claim("the double-buffered block over the one-buffer, 50%", lambda R: R[DOUBLE.format(50)] / R[BLOCK.format(50)],
                 0.85, 0.97, "times"),
           Claim("the double-buffered densest over its sparsest", lambda R: R[DOUBLE.format(100)] / R[DOUBLE.format(10)],
                 0.97, 1.03, "times")),
    ),
)


def write(root: pathlib.Path, data: pathlib.Path, instance: str) -> list[pathlib.Path]:
    return write_banded(root, data, instance, "mask_pack", RUNS)
