# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE MASK-PREPARATION PASS'S HOST SIDE, no GPU (csrc/fsla/src/mask_prepare/mask_prepare.cu, docs/internals/mask_prepare/mask_prepare.md): the host
reference's group rule is the carve's (`plan_tool --cta-carve`, on every density byte and every width the carve holds), the reference's layout
arithmetic is the layout header's, on a hand-computed case at two windows (32 and 16 mask words), and the pass's registry (`hardware/kernels/mask_prepare/`) holds the call's facts and refuses a call outside them.

Symbols: ``byte`` a density byte (``d = byte + 1`` 256ths), ``width`` a level-side's digits (a power of two), ``g`` its group in digits."""
from __future__ import annotations

import json
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


def _carved_groups(width_bits: int) -> list[int]:
    """The carve's group of one level of ``width_bits`` at each density 1..255 (read side), in digits."""
    densities = range(1, 256)
    run = subprocess.run([str(TOOL), "--cta-carve", ";".join(str(width_bits) for _ in densities), "--write-density", ";".join("256" for _ in densities),
                          "--read-density", ";".join(str(d) for d in densities)], capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr
    lines = [json.loads(line) for line in run.stdout.splitlines()]
    assert len(lines) == 255
    return [1 << line["read_group_bits"][0] for line in lines]


@pytest.mark.parametrize("width_bits", range(1, 9))
def test_the_references_group_is_the_carves_for_every_density_byte(width_bits):
    width = 1 << width_bits
    assert [reference.group_of(byte, width) for byte in range(255)] == _carved_groups(width_bits)


@pytest.mark.parametrize("width", [2, 4, 16, 32, 64, 128, 256])
def test_a_level_side_at_byte_255_is_dense_and_has_no_group(width):
    assert reference.group_of(255, width) is None


def test_every_allowed_size_no_larger_than_the_width_is_some_bytes_group_at_256_digits_and_a_narrow_width_takes_only_what_it_holds():
    assert {reference.group_of(byte, 256) for byte in range(255)} == set(reference.GROUP_SIZES)
    for width in (2, 4, 16, 32, 64, 128):
        assert {reference.group_of(byte, width) for byte in range(255)} == {s for s in reference.GROUP_SIZES if s <= width}


def test_a_tie_goes_to_the_smaller_size():
    #: width 256, byte 9 (d = 10): 10 digits live, 6 from 4 and 6 from 16 (1,536 256ths from each): a tie
    assert reference.group_of(9, 256) == 4


def _tiny() -> Masks:
    """One level-side (read, level 0), 1 batch, 1 head, 33 tokens (two words), 4 digits."""
    mask = torch.tensor([[[[0x1, 0x2, 0x0, 0x2], [0x1, 0x0, 0x0, 0x0]]]], dtype=torch.int32)
    return Masks(PREFILL, (mask,), (None,))


@pytest.mark.parametrize(
    ("window_words", "block_bytes_at_width_2", "block_bytes_at_width_4", "offsets", "total", "words_at"),
    [
        #: 32 words (a window of 1,024 tokens): a digit's block 66 bytes, slot 0's blocks 132 each, slot 5's 264; the words start at (66 * groups + 3) & ~3
        (32, 132, 264, [0, 396, 396, 396, 396, 396, 1980, 1980], 1980, (68, 132, 528)),
        #: 16 words (a window of 512 tokens): a digit's block 34 bytes, slot 0's blocks 68 each, slot 5's 136; the words start at (34 * groups + 3) & ~3
        (16, 68, 136, [0, 204, 204, 204, 204, 204, 1020, 1020], 1020, (36, 68, 272)),
    ],
)
def test_the_layout_of_a_hand_computed_call_at_two_windows(window_words, block_bytes_at_width_2, block_bytes_at_width_4, offsets, total, words_at):
    #: widths 2 and 4 on slots 0 and 5, 1 and 2 batch-heads, 3 windows
    widths, heads = [2, 2, 2, 2, 2, 4, 2, 2], [1, 0, 0, 0, 0, 2, 0, 0]
    assert (reference.block_bytes(2, window_words), reference.block_bytes(4, window_words)) == (block_bytes_at_width_2, block_bytes_at_width_4)
    assert reference.layout(widths, heads, 3, window_words) == (offsets, total)
    assert reference.block_bytes(256, window_words) == 2 * (window_words + 1) * 256
    assert (reference.counts_at(1), reference.counts_at(2), reference.counts_at(8)) == (2, 4, 16)
    assert (reference.words_at(1, window_words), reference.words_at(2, window_words), reference.words_at(8, window_words)) == words_at


def test_the_reference_of_a_hand_computed_block():
    #: 4 digits at byte 254 (d = 255, 1020 of 1024 live): the group is the 4 digits. Digits' word 0: 0x1, 0x2, 0x0, 0x2 -> OR 0x3; word 1: 0x1 only on digit 0.
    masks = _tiny()
    density = torch.tensor([[254, 255, 255, 255], [255, 255, 255, 255]], dtype=torch.uint8)
    out = reference.expected(masks, density, 32)
    assert list(out) == [0]
    block = out[0]
    assert block.group == 4
    assert block.totals.tolist() == [[3]]
    assert block.words[0, :3, 0].tolist() == [0x3, 0x1, 0x0]
    #: the live tokens before each word: none before word 0, the 2 of word 0 before word 1, all 3 from word 2 on (the window's words past the mask are zero)
    assert block.counts[0, :4, 0].tolist() == [0, 2, 3, 3] and block.counts[0, :, 0].max().item() == 3
    assert reference.bytes_of(masks, 32) == 1 * 1 * reference.block_bytes(4, 32)


@pytest.mark.parametrize("window_words", [32, 16])
def test_the_reference_reads_the_bytes_the_layout_names(window_words):
    masks = _tiny()
    groups, width = 1, 4
    block = bytearray(reference.block_bytes(width, window_words))
    struct.pack_into("<H", block, 0, 3)
    for j, count in enumerate([0, 2] + [3] * (window_words - 2)):
        struct.pack_into("<H", block, reference.counts_at(groups) + 2 * j, count)
    for j, word in enumerate([0x3, 0x1] + [0] * (window_words - 2)):
        struct.pack_into("<I", block, reference.words_at(groups, window_words) + 4 * j, word)
    got = reference.read(torch.frombuffer(block, dtype=torch.uint8).clone(), masks, 0, 4, window_words)
    want = reference.expected(masks, torch.tensor([[254, 255, 255, 255], [255, 255, 255, 255]], dtype=torch.uint8), window_words)[0]
    assert got.totals.tolist() == want.totals.tolist() and got.counts.tolist() == want.counts.tolist() and got.words.tolist() == want.words.tolist()


def test_a_group_of_one_digit_keeps_no_words_and_its_counts_are_a_digits_own():
    masks = Masks(PREFILL, (torch.tensor([[[[0x5, 0x0]]]], dtype=torch.int32),), (None,))
    block = reference.blocks_of(masks.read[0], 1, 32)
    assert block.words is None and block.totals.tolist() == [[2, 0]] and block.counts[0, 0].tolist() == [0, 0]


def test_the_references_block_formulas_are_the_layout_headers_at_both_windows():
    text = LAYOUT.read_text()
    assert "kMaskWindowWords" not in text
    assert re.search(r"prepared_block_bytes\(int width_bits, int words\)", text)
    assert re.search(r"std::int64_t\{2\} \* \(words \+ 1\) << width_bits", text)
    assert re.search(r"std::int64_t\{2\} \* groups", text)
    assert re.search(r"\(std::int64_t\{2\} \* \(words \+ 1\) \* groups \+ 3\) & ~std::int64_t\{3\}", text)
    for window_words in (32, 16):
        assert reference.block_bytes(1 << 3, window_words) == (2 * (window_words + 1)) << 3
        assert reference.words_at(3, window_words) == (2 * (window_words + 1) * 3 + 3) & ~3


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
