# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The carry's value tile, by its banks (docs/internals/carry/carry_values.md#tile).

A round's instruction copies 4 consecutive tokens, 8 pieces of 16 bytes each, a lane a piece. Shared memory has 32
banks of 4 bytes. The header's layout -- a tile's bricks `kStepTokens + 1` lines apart -- must put exactly four of an
instruction's 32 lanes on every bank; a pitch of a whole number of 128 bytes must not.
"""

from __future__ import annotations

import pathlib
import re

HEADER = pathlib.Path(__file__).resolve().parents[2] / "csrc/fsla/src/carry/carry_values.cuh"
LANES, BANKS, BANK_BYTES, LINE = 32, 32, 4, 16
STEP_TOKENS, PIECES = 16, 8
TOKENS = LANES // PIECES


def _most_on_a_bank(pitch_lines: int) -> int:
    """The most lanes any bank takes, over a round's instructions, at a brick pitch of `pitch_lines` lines."""
    worst = 0
    for instruction in range(PIECES):
        on = [0] * BANKS
        for lane in range(LANES):
            piece, which = lane % PIECES, lane // PIECES
            place = TOKENS * instruction + which
            at = (piece * pitch_lines + place % STEP_TOKENS) * LINE
            for word in range(LINE // BANK_BYTES):
                on[(at // BANK_BYTES + word) % BANKS] += 1
        worst = max(worst, max(on))
    return worst


def test_the_header_states_the_pitch_the_test_is_of():
    text = HEADER.read_text()
    assert re.search(r"constexpr int kStepTokens = 16;", text)
    assert re.search(r"constexpr int kBrickPitch = kStepTokens \+ 1;", text)


def test_an_instruction_puts_four_lanes_on_every_bank():
    assert _most_on_a_bank(STEP_TOKENS + 1) == LANES * (LINE // BANK_BYTES) // BANKS == 4


def test_a_pitch_of_whole_128_bytes_does_not():
    assert _most_on_a_bank(STEP_TOKENS) > 4
