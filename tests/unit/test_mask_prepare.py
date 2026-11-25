# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE MASK-PREPARATION PASS'S HOST SIDE, no GPU (csrc/fsla/src/mask_prepare/mask_prepare.cu, docs/internals/mask_prepare/mask_prepare.md): the host
reference's group rule is the carve's (`plan_tool --cta-carve`, on every density byte and every width the carve holds), the reference's layout
arithmetic is the layout header's, on a hand-computed case at two windows (32 and 16 mask words), and the pass's registry (`hardware/kernels/mask_prepare/`) holds the call's facts and refuses a call outside them.

Symbols: ``byte`` a density byte (``d = byte + 1`` 256ths), ``width`` a level-side's digits (a power of two), ``g`` its group in digits."""
from __future__ import annotations

import json
import random
import re
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fsla.routing.masks import PREFILL, Masks  # noqa: E402
from hardware import registry  # noqa: E402
from tests.oracle import mask_prepare_reference as reference  # noqa: E402

TOOL = ROOT / "csrc" / "fsla" / "plan_tool"
HOPPER = registry.device_point()
LAYOUT = ROOT / "csrc" / "fsla" / "src" / "common" / "plan" / "mask_prepare_layout.h"


def _planner_sizes(width_bits: int) -> list[int]:
    """The size of one level of ``width_bits`` at each density 1..256 (read side), in digits, as the planner's tool prints it: the group the density gives, before a set's least raises it (the pass prepares at the raised group)."""
    densities = range(1, 257)
    run = subprocess.run([str(TOOL), "--cta-carve", ";".join(str(width_bits) for _ in densities), "--write-density", ";".join("256" for _ in densities),
                          "--read-density", ";".join(str(d) for d in densities)], capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr
    lines = [json.loads(line) for line in run.stdout.splitlines()]
    assert len(lines) == 256
    return [1 << line["read_size_bits"][0] for line in lines]


@pytest.mark.parametrize("width_bits", range(1, 9))
def test_the_references_size_is_the_planners_for_every_density_byte(width_bits):
    width = 1 << width_bits
    assert [reference.group_of(byte, width) for byte in range(256)] == _planner_sizes(width_bits)


@pytest.mark.parametrize("width", [2, 4, 8, 16, 32, 64, 128, 256])
def test_a_level_side_at_byte_255_is_whole_its_group_the_level_itself_at_every_width(width):
    assert reference.group_of(255, width) == width


def test_a_level_side_is_whole_from_just_over_three_quarters_density_at_every_width():
    #: the level's own size against the next one down, half of it: nearer from d * width / 256 over three quarters of the width, d > 192 (a tie to the smaller)
    for width in (2, 4, 8, 16, 32, 64, 128, 256):
        assert [reference.group_of(byte, width) == width for byte in range(256)] == [byte + 1 > 192 for byte in range(256)]


def test_every_allowed_size_no_larger_than_the_width_is_some_bytes_group_at_256_digits_and_a_narrow_width_takes_only_what_it_holds():
    assert {reference.group_of(byte, 256) for byte in range(255)} == set(reference.GROUP_SIZES)
    for width in (2, 4, 16, 32, 64, 128):
        assert {reference.group_of(byte, width) for byte in range(255)} == {s for s in reference.GROUP_SIZES if s <= width}


def test_a_tie_goes_to_the_smaller_size():
    #: width 256, byte 11 (d = 12): 12 digits live, 4 from 8 and 4 from 16 (1,024 256ths from each): a tie
    assert reference.group_of(11, 256) == 8


def _planner_groups(calls: list[tuple[list[int], list[int], list[int]]]) -> list[tuple[list[int], list[int]]]:
    """The planner's (read group bits, write group bits) of each call, a call its levels' (width bits, read densities, write densities)."""

    def join(rows: list[list[int]]) -> str:
        return ";".join(",".join(map(str, row)) for row in rows)

    run = subprocess.run([str(TOOL), "--cta-carve", join([c[0] for c in calls]), "--write-density", join([c[2] for c in calls]),
                          "--read-density", join([c[1] for c in calls])], capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr
    lines = [json.loads(line) for line in run.stdout.splitlines()]
    assert len(lines) == len(calls)
    return [(line["read_group_bits"][:len(call[0])], line["write_group_bits"][:len(call[0])]) for line, call in zip(lines, calls)]


@pytest.mark.parametrize("levels", [1, 2, 3, 4])
def test_the_references_carve_written_from_the_rule_gives_the_planners_groups_on_400_calls_a_depth(levels):
    rng = random.Random(levels)
    densities = [1, 2, 4, 7, 13, 25, 40, 49, 97, 193, 256]
    calls = [([rng.choice([1, 2, 4, 6, 8]) for _ in range(levels)], [rng.choice(densities) for _ in range(levels)], [rng.choice(densities) for _ in range(levels)])
             for _ in range(400)]
    for (bits, read, write), want in zip(calls, _planner_groups(calls)):
        assert reference.carve_groups(bits, read, write) == want, (bits, read, write)


@pytest.mark.parametrize("width", [2, 4, 8, 16, 32, 64, 128, 256])
def test_beside_a_whole_level_of_256_digits_a_group_is_the_level_sides_size_and_alone_a_narrow_level_is_raised_to_whole(width):
    for size in (g for g in reference.GROUP_SIZES if g < width):
        byte = next(b for b in range(255) if reference.group_of(b, width) == size)
        assert reference.groups_of(torch.tensor([[byte, 255], [255, 255]], dtype=torch.uint8), [width, 256])[0] == size
    #: alone, a level of at most 16 digits holds no set of 16 leaves but itself: a group of one digit is raised to the level
    if width <= 16:
        assert reference.groups_of(torch.tensor([[0], [255]], dtype=torch.uint8), [width])[0] == width


def test_one_level_of_64_digits_raises_the_write_sides_group_of_4_to_16_in_a_fold_of_32():
    assert reference.groups_of(torch.tensor([[48], [12]], dtype=torch.uint8), [64]) == {0: 16, 4: 16}
    assert (reference.group_of(48, 64), reference.group_of(12, 64)) == (16, 4)


def _tiny() -> Masks:
    """One level-side (read, level 0), 1 batch, 1 head, 33 tokens (two words), 4 digits."""
    mask = torch.tensor([[[[0x1, 0x2, 0x0, 0x2], [0x1, 0x0, 0x0, 0x0]]]], dtype=torch.int32)
    return Masks(PREFILL, (mask,), (None,))


@pytest.mark.parametrize(
    ("window_words", "block_bytes_at_width_2", "block_bytes_at_width_4", "offsets", "total"),
    [
        #: 32 words (a window of 1,024 tokens). Width 2: a total a digit, words_at(2) = (2 * 2 + 3) & ~3 = 4 bytes (its group of 2 is the level, whole: nothing is stored for it). Width 4: the larger of
        #: words_at(4) = (8 + 3) & ~3 = 8 and, two groups of 2 digits, words_at(2) + 4 * 32 * 2 = 4 + 256 = 260 bytes. Slot 0: 1 batch-head, 3 windows of 4 bytes = 12; slots 1 to 4 have no heads;
        #: slot 5: 2 batch-heads, 3 windows of 260 = 1,560 from offset 12; total 1,572
        (32, 4, 260, [0, 12, 12, 12, 12, 12, 1572, 1572], 1572),
        #: 16 words (a window of 512 tokens): width 2 is 4 bytes; width 4 is max(8, 4 + 4 * 16 * 2 = 132) = 132. Slot 0: 3 * 4 = 12; slot 5: 6 * 132 = 792 from 12; total 804
        (16, 4, 132, [0, 12, 12, 12, 12, 12, 804, 804], 804),
    ],
)
def test_the_layout_of_a_hand_computed_call_at_two_windows(window_words, block_bytes_at_width_2, block_bytes_at_width_4, offsets, total):
    #: widths 2 and 4 on slots 0 and 5, 1 and 2 batch-heads, 3 windows
    widths, heads = [2, 2, 2, 2, 2, 4, 2, 2], [1, 0, 0, 0, 0, 2, 0, 0]
    assert (reference.block_bytes(2, window_words), reference.block_bytes(4, window_words)) == (block_bytes_at_width_2, block_bytes_at_width_4)
    assert reference.layout(widths, heads, 3, window_words) == (offsets, total)
    #: 256 digits: the total a digit is words_at(256) = 512 bytes; groups of 2: 128 groups, words_at(128) = (256 + 3) & ~3 = 256, plus 4 * words * 128
    assert reference.block_bytes(256, window_words) == 256 + 512 * window_words
    #: the words start after the totals, rounded up to 4: (2 * 1 + 3) & ~3 = 4, (2 * 2 + 3) & ~3 = 4, (2 * 8 + 3) & ~3 = 16
    assert (reference.words_at(1), reference.words_at(2), reference.words_at(8)) == (4, 4, 16)
    assert reference.block_bytes(256, 32) == 16640


def test_the_reference_of_a_hand_computed_block():
    #: 4 digits in one group of 4. Digits' word 0: 0x1, 0x2, 0x0, 0x2 -> OR 0x3; word 1: 0x1 only on digit 0.
    masks = _tiny()
    block = reference.blocks_of(masks.read[0], 4, 32)
    assert block.group == 4
    assert block.totals.tolist() == [[3]]
    assert block.words[0, :3, 0].tolist() == [0x3, 0x1, 0x0]
    assert reference.bytes_of(masks, 32) == 1 * 1 * reference.block_bytes(4, 32)


def test_the_pass_writes_nothing_for_a_whole_level_side_and_a_block_for_one_that_is_not():
    #: 4 digits: at byte 254 (d = 255, 1020 of 1024 live) the group is the 4 digits, the level itself: whole. At byte 0 the group is one digit.
    #: a second level of 256 digits, whole and unmasked, holds a set at 16 leaves, so the carve raises nothing
    masks = Masks(PREFILL, (_tiny().read[0], None), (None, None))
    assert reference.expected(masks, torch.tensor([[254, 255], [255, 255]], dtype=torch.uint8), [4, 256], 32) == {}
    out = reference.expected(masks, torch.tensor([[0, 255], [255, 255]], dtype=torch.uint8), [4, 256], 32)
    assert list(out) == [0] and out[0].group == 1 and out[0].words is None
    #: a digit's own live tokens of the window: digit 0 two (a bit in each word), digits 1 and 3 one, digit 2 none
    assert out[0].totals.tolist() == [[2, 1, 0, 1]]


@pytest.mark.parametrize("window_words", [32, 16])
def test_the_reference_reads_the_bytes_the_layout_names(window_words):
    masks = _tiny()
    groups, width = 1, 4
    block = bytearray(reference.block_bytes(width, window_words))
    struct.pack_into("<H", block, 0, 3)
    for j, word in enumerate([0x3, 0x1] + [0] * (window_words - 2)):
        struct.pack_into("<I", block, reference.words_at(groups) + 4 * j, word)
    got = reference.read(torch.frombuffer(block, dtype=torch.uint8).clone(), masks, 0, 4, window_words)
    want = reference.blocks_of(masks.read[0], 4, window_words)
    assert got.totals.tolist() == want.totals.tolist() and got.words.tolist() == want.words.tolist()


def test_a_group_of_one_digit_keeps_no_words_and_its_totals_are_a_digits_own():
    masks = Masks(PREFILL, (torch.tensor([[[[0x5, 0x0]]]], dtype=torch.int32),), (None,))
    block = reference.blocks_of(masks.read[0], 1, 32)
    assert block.words is None and block.totals.tolist() == [[2, 0]]


def test_the_references_block_formulas_are_the_layout_headers_at_both_windows():
    text = LAYOUT.read_text()
    assert "kMaskWindowWords" not in text
    assert re.search(r"prepared_block_bytes\(int width_bits, int words\)", text)
    assert re.search(r"prepared_words_at\(int groups\)", text)
    assert "prepared_counts_at" not in text
    assert re.search(r"\(std::int64_t\{2\} \* groups \+ 3\) & ~std::int64_t\{3\}", text)
    assert re.search(r"prepared_words_at\(1 << width_bits\)", text)
    assert re.search(r"prepared_words_at\(groups\) \+ std::int64_t\{4\} \* words \* groups", text)
    for window_words in (32, 16):
        for bits in range(1, 9):
            width = 1 << bits
            ones = (2 * width + 3) & ~3
            twos = ((2 * (width // 2) + 3) & ~3) + 4 * window_words * (width // 2) if width >= 4 else 0
            assert reference.block_bytes(width, window_words) == max(ones, twos)
        assert reference.words_at(3) == (2 * 3 + 3) & ~3


# ---------------------------------------------------------------- the registry and the contract


def test_the_pass_has_its_own_registry_of_two_call_facts():
    assert "mask_prepare" in registry.kernels()
    assert tuple(registry.kernel_schema("mask_prepare")) == ("width", "levels")
    assert registry.contexts("mask_prepare") and not registry.owed("mask_prepare")


def _run(call: str):
    return subprocess.run([str(TOOL), "--kernel", "mask_prepare", "--device", ",".join(f"{k}={v}" for k, v in HOPPER.items()), "--call", call],
                          capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("width", [2, 3, 16, 100, 256])
@pytest.mark.parametrize("levels", [1, 4])
def test_a_call_inside_the_contract_has_a_descriptor(width, levels):
    run = _run(f"width={width},levels={levels}")
    assert run.returncode == 0, run.stderr
    assert run.stdout.startswith("descriptor: ")


@pytest.mark.parametrize(("call", "named"), [("width=1,levels=2", "width=1"), ("width=257,levels=2", "width=257"), ("width=16,levels=0", "levels=0"), ("width=16,levels=5", "levels=5")])
def test_a_call_outside_the_contract_is_refused_naming_the_fact(call, named):
    run = _run(call)
    assert run.returncode == 1
    assert "outside FSLA's mask_prepare contract" in run.stderr and named in run.stderr
