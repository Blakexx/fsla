# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The predictions and the measured section of `docs/lab/newton_row.md`: `RUNS` states each run's bands once
(`_banded.py`'s form), the newest sample read against its own run's (`lab page newton_row --instance LABEL`,
docs/internals/tools/lab.md). The first run (2026-11-28, the experiment at 00123881) failed its exactness check and was
not recorded: its figures are in no record and on no page."""

from __future__ import annotations

import pathlib

from _banded import PENDING, Claim, write_banded

BLOCK = "a 64-token block"


def _row(alpha: str, kind: str, wg: int, interleaved: bool, what: str) -> str:
    name = f"alpha {alpha}, {kind}, {wg} warpgroup{'s' if wg > 1 else ''} a CTA"
    return name + (", both rows interleaved" if interleaved else "") + f": {what}"


def _ratio(alpha: str, kind: str, wg: int, lo: float, hi: float) -> Claim:
    return Claim(f"alpha {alpha}, {kind}, {wg} WG: the interleaved block over the serial",
                 lambda R: R[_row(alpha, kind, wg, True, BLOCK)] / R[_row(alpha, kind, wg, False, BLOCK)], lo, hi, "times")


#: the model fitted to the first run's blocks (cycles a block = c0 + S c, S the block's steps, its 4 warps' most, from
#: the host's float32 iteration on the experiment's own fixtures): at one warpgroup c0 ~420, c 1,157 (alpha 1.5) and
#: 1,100 (alpha 2) against 861 and 685 instructions a step (0.74 and 0.62 issued a clock: latency exposed); at two, c
#: 986 and 770 a block
RUNS = (
    (
        PENDING,
        "2026-11-28",
        "the first run's blocks fit cycles = 420 + S c within 4% at one warpgroup (c 1,157 alpha 1.5, 1,100 alpha 2, the "
        "steps S a block's from the host's float32 iteration: 6.0 and 4.8 random); a step issues 861 (alpha 1.5) or 685 "
        "(alpha 2) instructions, so ~300-400 cycles of latency are exposed; the interleaved form issues 872 and 690 with "
        "twelve chains of 32 (the chains' latency under the issue) and the phases' own latency (the shuffles, the root "
        "and divide, the vote: ~150-180) left, so c ~1,050 and ~840 at one warpgroup and issue-bound at two (872, 690 a "
        "block); the steps a row are now the row's (its 4 lanes' most), as the host's iteration counts them",
        (
            Claim("alpha 1.5, random, 1 WG, serial: a block", lambda R: R[_row("1.5", "random", 1, False, BLOCK)],
                  6990.0, 7720.0, "cycles"),
            _ratio("1.5", "random", 1, 0.82, 0.98),
            _ratio("1.5", "near the boundary", 1, 0.82, 0.98),
            _ratio("2", "random", 1, 0.68, 0.85),
            _ratio("2", "near the boundary", 1, 0.68, 0.85),
            _ratio("1.5", "random", 2, 0.80, 0.98),
            _ratio("2", "random", 2, 0.80, 0.98),
            Claim("alpha 1.5, random: steps a row, the mean",
                  lambda R: R[_row("1.5", "random", 1, False, "steps a row, the mean")], 3.89, 4.75, "steps"),
            Claim("alpha 1.5, random: steps a block, the mean",
                  lambda R: R[_row("1.5", "random", 1, False, "steps a block (its 4 warps' most), the mean")], 5.5, 6.5,
                  "steps"),
            Claim("alpha 2, random: steps a row, the mean",
                  lambda R: R[_row("2", "random", 1, False, "steps a row, the mean")], 2.53, 3.09, "steps"),
            Claim("alpha 1.5, near the boundary, 1 WG: supports differing within fp32's resolution",
                  lambda R: R[_row("1.5", "near the boundary", 1, False, "supports that differ within fp32's resolution")],
                  0.0, 12.0, "rows"),
        ),
    ),
)


def write(root: pathlib.Path, data: pathlib.Path, instance: str) -> list[pathlib.Path]:
    return write_banded(root, data, instance, "newton_row", RUNS)
