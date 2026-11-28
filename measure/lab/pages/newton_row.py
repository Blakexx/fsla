# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The predictions and the measured section of `docs/lab/newton_row.md`: `RUNS` states each run's bands once
(`_banded.py`'s form), the newest sample read against its own run's (`lab page newton_row --instance LABEL`,
docs/internals/tools/lab.md). The first run (2026-11-28, the experiment at 00123881) failed its exactness check and was
not recorded: its figures are in no record and on no page."""

from __future__ import annotations

import pathlib

from _banded import Claim, write_banded

BLOCK = "a 64-token block"


def _row(alpha: str, kind: str, wg: int, interleaved: bool, what: str) -> str:
    name = f"alpha {alpha}, {kind}, {wg} warpgroup{'s' if wg > 1 else ''} a CTA"
    return name + (", both rows interleaved" if interleaved else "") + f": {what}"


def _ratio(alpha: str, kind: str, wg: int, lo: float, hi: float) -> Claim:
    return Claim(f"alpha {alpha}, {kind}, {wg} WG: the interleaved block over the serial",
                 lambda R: R[_row(alpha, kind, wg, True, BLOCK)] / R[_row(alpha, kind, wg, False, BLOCK)], lo, hi, "times")


EXACT = ", both rows interleaved, the closed form IEEE"


def _digits(alpha: str, kind: str, wg: int, form: str, lo: float, hi: float) -> Claim:
    name = f"alpha {alpha}, {kind}, {wg} warpgroup{'s' if wg > 1 else ''} a CTA" + (form or ", both rows interleaved")
    return Claim(f"alpha {alpha}, {kind}, {wg} WG{' IEEE' if form else ''}: digits that differ from fp64's",
                 lambda R: R[name + ": digits that differ from fp64's"], lo, hi, "digits")


def _exact(alpha: str, kind: str, wg: int, lo: float, hi: float) -> Claim:
    def figure(R):
        base = f"alpha {alpha}, {kind}, {wg} warpgroup{'s' if wg > 1 else ''} a CTA"
        return R[base + EXACT + ": " + BLOCK] / R[_row(alpha, kind, wg, True, BLOCK)]
    return Claim(f"alpha {alpha}, {kind}, {wg} WG: the IEEE closed form's block over the approximate one's", figure, lo,
                 hi, "times")


#: the model fitted to the first run's blocks (cycles a block = c0 + S c, S the block's steps, its 4 warps' most, from
#: the host's float32 iteration on the experiment's own fixtures): at one warpgroup c0 ~420, c 1,157 (alpha 1.5) and
#: 1,100 (alpha 2) against 861 and 685 instructions a step (0.74 and 0.62 issued a clock: latency exposed); at two, c
#: 986 and 770 a block
RUNS = (
    (
        "unrecorded 2026-11-28T03:46 (its check failed on one near-boundary row past the 2-ulp clause)",
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
    (
        "20261128T045839Z-149866-f2c33cba",
        "2026-11-28",
        "the fourth run held every timing and step band; its interleaved blocks give a step of (6,766 - 420) / 6.0 = 1,058 "
        "cycles (alpha 1.5) and (4,622 - 473) / 4.39 = 944 (alpha 2) at one warpgroup. The IEEE closed form (`__fsqrt_rn`, "
        "`__fdiv_rn`) adds 36 instructions a step at alpha 1.5 (908 against 872) and 19 at alpha 2 (709 against 690), and "
        "puts the rounded root's and divide's refinements on the step's critical path (~30-60 cycles); its threshold within "
        "fp32's rounding of the sums. The check is the producer's outputs against fp64's (levels under the stored values' "
        "clause, digits, mask bits, density); a digit flips where its value lies within the level's error of a half-point: "
        "~20 live digits a row, the approximate threshold ~3e-7 off fp64 (255 x 0.5 x 3e-7 ~ 4e-5 of a digit) gives a "
        "flip in ~1e-4 of them, ~15 in 8,448 rows; the IEEE one ~5e-8 off, a few. The other forms repeat within 3%",
        (
            Claim("alpha 1.5, random, 1 WG, serial: a block", lambda R: R[_row("1.5", "random", 1, False, BLOCK)],
                  7130.0, 7570.0, "cycles"),
            _ratio("1.5", "random", 1, 0.89, 0.95),
            _ratio("2", "random", 1, 0.78, 0.84),
            _exact("1.5", "random", 1, 1.02, 1.09),
            _exact("1.5", "near the boundary", 1, 1.02, 1.09),
            _exact("2", "random", 1, 1.01, 1.07),
            _exact("1.5", "random", 2, 1.01, 1.09),
            _digits("1.5", "random", 1, "", 1.0, 150.0),
            _digits("1.5", "random", 1, EXACT, 0.0, 30.0),
            _digits("1.5", "near the boundary", 1, "", 1.0, 200.0),
            _digits("1.5", "near the boundary", 1, EXACT, 0.0, 40.0),
        ),
    ),
)


def write(root: pathlib.Path, data: pathlib.Path, instance: str) -> list[pathlib.Path]:
    return write_banded(root, data, instance, "newton_row", RUNS)
