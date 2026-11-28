# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The predictions and the measured section of `docs/lab/produce_skeleton.md`: `RUNS` states each run's bands once
(`_banded.py`'s form), the newest sample read against its own run's (`lab page produce_skeleton --instance LABEL`,
docs/internals/tools/lab.md)."""

from __future__ import annotations

import pathlib

from _banded import PENDING, Claim, write_banded

CELL = "the empty pipeline at the cell"
NO_STAGES = CELL + ", hidden 0"
TWO = CELL + ", two CTAs an SM"
LO = CELL + ", the lo plane"

#: THE FLOOR (measure/floors/produce.work.json at measure/floors/h100.rates.json, warpline's floor): the cell's
#: outputs at HBM's write rate, without and with the lo plane
FLOOR_US = 101.875
LO_FLOOR_US = 186.55

#: each run's predictions, in the order they ran
RUNS = (
    (
        PENDING,
        "2026-11-28",
        "warpline's floor at the cell is 101.9 us, its 323.0 MB at HBM's write rate (measure/floors/produce.work.json). "
        "THE HYPOTHESIS, NOT A FINDING until these rows name the price: at one CTA an SM a CTA's non-store time (launch, "
        "setup, its 20 ring stages' edges, the tickets, its exit) runs in series with its own writes, ~10K cycles a CTA "
        "(the CTA's ~25K warp instructions of tools/budgets/produce.json at ~2.5 a cycle) over 2,048 CTAs on 132 SMs, "
        "15.5 waves at ~1.8 GHz: ~85 us past the floor, ~187 us; at hidden 0 the 20 stages' edges go (~100 cycles each, "
        "~17 us a launch); at two CTAs an SM one CTA's serial time runs under the other's writes, back near the floor. "
        "The rival (the stub stores' instruction issue, their runtime division 29 of a trip's ~46 instructions) would "
        "leave hidden 0 and two CTAs an SM where the plain row is, its time inside the traced rows' epilogue phases; the "
        "traced twins' timelines (lab trace: each role's phases, the path) say which. The lo plane adds its 268.4 MB at "
        "the write rate",
        (
            Claim(CELL, lambda R: R[CELL], 170.0, 205.0, "us"),
            Claim(CELL + ": over its floor", lambda R: R[CELL] / FLOOR_US, 1.67, 2.01, "times"),
            Claim(NO_STAGES, lambda R: R[NO_STAGES], 152.0, 190.0, "us"),
            Claim(TWO, lambda R: R[TWO], 104.0, 128.0, "us"),
            Claim(LO, lambda R: R[LO], 255.0, 292.0, "us"),
            Claim(LO + ": over its floor", lambda R: R[LO] / LO_FLOOR_US, 1.37, 1.57, "times"),
        ),
    ),
)


def write(root: pathlib.Path, data: pathlib.Path, instance: str) -> list[pathlib.Path]:
    return write_banded(root, data, instance, "produce_skeleton", RUNS)
