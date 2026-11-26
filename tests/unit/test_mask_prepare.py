# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE MASK-PREPARATION PASS'S HOST SIDE, no GPU (csrc/fsla/src/mask_prepare/mask_prepare.cu, docs/internals/mask_prepare/mask_prepare.md): the host
reference's group rule and leaf groups' order are the planner's (`plan_tool --cta-carve`, on every density byte and every width the carve holds, and on random calls of depth 1 to 4), the reference's layout
arithmetic is the layout header's (`leaf_line_bytes`, `leaf_lines_at`, `leaf_block_bytes`, `prepared_layout`), on hand-computed cases at two windows (32 and 16 mask words), the reference's fields are the rule's on a hand-worked call,
the op's own refusals that need no GPU, and the pass's registry (`hardware/kernels/mask_prepare/`) holds the call's facts and refuses a call outside them.

Symbols: ``byte`` a density byte (``d = byte + 1`` 256ths), ``width`` a level's digits (a power of two), ``g`` a side's group of it in digits, ``bits`` a side's leaf groups as log2, ``words`` a window's mask words."""
from __future__ import annotations

import functools
import json
import operator
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


def _planner(calls: list[tuple[list[int], list[int], list[int]]]) -> list[dict]:
    """The planner's JSON of each call, a call its levels' (width bits, read densities, write densities) in 256ths."""

    def join(rows: list[list[int]]) -> str:
        return ";".join(",".join(map(str, row)) for row in rows)

    run = subprocess.run([str(TOOL), "--cta-carve", join([c[0] for c in calls]), "--write-density", join([c[2] for c in calls]),
                          "--read-density", join([c[1] for c in calls])], capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr
    lines = [json.loads(line) for line in run.stdout.splitlines()]
    assert len(lines) == len(calls)
    return lines


def _calls(levels: int, count: int, seed: int) -> list[tuple[list[int], list[int], list[int]]]:
    rng = random.Random(seed)
    densities = [1, 2, 4, 7, 13, 25, 40, 49, 97, 193, 256]
    return [([rng.choice([1, 2, 4, 6, 8]) for _ in range(levels)], [rng.choice(densities) for _ in range(levels)], [rng.choice(densities) for _ in range(levels)]) for _ in range(count)]


@pytest.mark.parametrize("levels", [1, 2, 3, 4])
def test_the_references_carve_written_from_the_rule_gives_the_planners_groups_on_400_calls_a_depth(levels):
    calls = _calls(levels, 400, levels)
    for (bits, read, write), line in zip(calls, _planner(calls)):
        carve = reference.carve_groups(bits, read, write)
        assert (carve.read_groups, carve.write_groups) == (line["read_group_bits"][:levels], line["write_group_bits"][:levels]), (bits, read, write)
        assert (carve.carve, carve.fold) == (line["carve_bits"][:levels], line["fold_bits"][:levels]), (bits, read, write)


@pytest.mark.parametrize("levels", [1, 2, 3, 4])
def test_the_references_leaf_groups_order_is_the_planners_on_100_calls_a_depth(levels):
    """Each side's leaf group bits, inner and outer fields and their places (a place only where the field has bits), and most (a bit a level whose density is under 256ths: the planner's masked)."""
    calls = _calls(levels, 100, 10 + levels)
    for (bits, read, write), line in zip(calls, _planner(calls)):
        carve = reference.carve_groups(bits, read, write)
        for side, name, density in ((0, "read", read), (1, "write", write)):
            order = reference.order_of(bits, carve, side)
            assert order.bits == line[f"{name}_leaf_bits"], (name, bits, read, write)
            for field in ("inner", "outer"):
                assert getattr(order, field) == line[f"{name}_leaf_{field}"][:levels], (name, field, bits, read, write)
                at = getattr(order, f"{field}_at")
                assert [a for a, n in zip(at, getattr(order, field)) if n] == [a for a, n in zip(line[f"{name}_leaf_{field}_at"][:levels], getattr(order, field)) if n], (name, field, bits, read, write)
            masked = sum(1 << level for level in range(levels) if density[level] < 256)
            assert reference.most(bits, masked) == line[f"{name}_leaf_bits_most"], (name, bits, read, write)


@pytest.mark.parametrize("width", [2, 4, 8, 16, 32, 64, 128, 256])
def test_beside_a_whole_level_of_256_digits_a_group_is_the_level_sides_size_and_alone_a_narrow_level_is_raised_to_whole(width):
    for size in (g for g in reference.GROUP_SIZES if g < width):
        byte = next(b for b in range(255) if reference.group_of(b, width) == size)
        assert 1 << reference.groups_of(torch.tensor([[byte, 255], [255, 255]], dtype=torch.uint8), [width, 256]).read_groups[0] == size
    #: alone, a level of at most 16 digits holds no set of 16 leaves but itself: a group of one digit is raised to the level
    if width <= 16:
        assert 1 << reference.groups_of(torch.tensor([[0], [255]], dtype=torch.uint8), [width]).read_groups[0] == width


def test_one_level_of_64_digits_raises_the_write_sides_group_of_4_to_16_in_a_fold_of_32():
    carve = reference.groups_of(torch.tensor([[48], [12]], dtype=torch.uint8), [64])
    assert (1 << carve.read_groups[0], 1 << carve.write_groups[0]) == (16, 16)
    assert (reference.group_of(48, 64), reference.group_of(12, 64)) == (16, 4)


def test_the_cells_leaf_groups_hand_worked_64_a_side_and_a_box_holds_4_sets():
    """Two levels of 256 digits; the write side on 4 digits of level 0 and whole on level 1, the read side the reverse (density 4 and 256 256ths). A group is 4 digits of the sparse level by
    the whole level: 256 / 4 = 64 leaf groups a side, 6 bits. The loop order, sparsest outermost: the write side's is level 0 then 1, so its densest is level 1 and its inner bits (from bit 0) are level 1's -- it
    has none to give, the level being whole -- then level 0's: a fold holds 16 digits of level 0, 4 groups, 2 inner bits at place 0 and 4 outer bits above them."""
    carve = reference.carve_groups([8, 8], [256, 4], [4, 256])
    write = reference.order_of([8, 8], carve, 1)
    assert (write.bits, write.inner, write.inner_at[0], write.outer, write.outer_at[0]) == (6, [2, 0], 0, [4, 0], 2)
    read = reference.order_of([8, 8], carve, 0)
    assert (read.bits, read.inner, read.inner_at[1], read.outer, read.outer_at[1]) == (6, [0, 2], 0, [0, 4], 2)
    assert reference.most([8, 8], 0b11) == 12 and reference.most([8, 8], 0b01) == 8 and reference.most([8, 8], 0) == 0


def test_a_leaf_groups_index_and_its_groups_are_one_to_one():
    carve = reference.carve_groups([4, 3, 2], [3, 20, 256], [100, 8, 40])
    for side in (0, 1):
        order = reference.order_of([4, 3, 2], carve, side)
        assert sorted(order.leaf_of(order.groups_of(leaf)) for leaf in range(1 << order.bits)) == list(range(1 << order.bits))


def _tiny() -> Masks:
    """One level (read, level 0), 1 batch, 1 head, 33 tokens (two words), 4 digits; word 1 has bits past token 32 set (token 32 is its only own bit)."""
    mask = torch.tensor([[[[0x1, 0x2, 0x0, 0x2], [-1, 0x0, 0x0, 0x0]]]], dtype=torch.int32)
    return Masks(PREFILL, (mask,), (None,))


def test_the_references_fields_of_a_hand_worked_call():
    """4 digits, density byte 63 (64 256ths, 1 live digit of 4: groups of 1 digit, 4 leaf groups, alone on a level of 4 digits: the carve raises the group to the level -- so a second level of 256 digits, whole and
    unmasked, holds the set). Each leaf group is one digit; the call's tokens are 33 (word 0 whole, word 1 only bit 0), a window of 32 mask words. Digit 0: word 0 bit 0, word 1 all bits: its line is
    [0x1, 0x1], count 2. Digit 1: [0x2, 0], 1. Digit 2: nothing. Digit 3: [0x2, 0], 1."""
    masks = Masks(PREFILL, (_tiny().read[0], None), (None, None))
    density = torch.tensor([[0, 255], [255, 255]], dtype=torch.uint8)
    read, write = reference.expected(masks, density, [4, 256], [1, 1], 33, 32)
    assert read.bits == 2 and read.counts.shape == (1, 4) and read.lines.shape == (1, 4, 32)
    assert read.counts.tolist() == [[2, 1, 0, 1]]
    assert read.lines[0, :, :2].tolist() == [[0x1, 0x1], [0x2, 0x0], [0x0, 0x0], [0x2, 0x0]] and int(read.lines[0, :, 2:].count_nonzero()) == 0
    #: the write side lists nothing: one leaf group, the call's own tokens (word 0 whole, word 1 bit 0)
    assert write.bits == 0 and write.counts.tolist() == [[33]] and write.lines[0, 0, :3].tolist() == [0xFFFFFFFF, 0x1, 0x0]


def test_a_listed_level_with_no_mask_asks_nothing_and_a_second_listed_level_is_anded():
    """Two levels of 16 digits, density byte 0 on both read levels (the carve raises their groups until a set holds 16 leaves): a leaf group is one group on each level and its line is the call's own tokens
    ANDed with the OR of each level's group's digits; with level 1 absent (dense, byte 255), the line is level 0's group's alone, whatever level 1's group of the leaf group."""
    rng = torch.Generator().manual_seed(3)
    m0 = torch.randint(0, 16, (1, 1, 1, 16), generator=rng, dtype=torch.int32)
    m1 = torch.randint(0, 16, (1, 1, 1, 16), generator=rng, dtype=torch.int32)
    density = torch.tensor([[0, 0], [255, 255]], dtype=torch.uint8)
    carve = reference.groups_of(density, [16, 16])
    order = reference.order_of([4, 4], carve, 0)
    size = [1 << g for g in carve.read_groups]
    group_or = lambda m, g, l: functools.reduce(operator.or_, (int(m[0, 0, 0, d]) for d in range(g * size[l], (g + 1) * size[l])))  # noqa: E731
    both, _ = reference.expected(Masks(PREFILL, (m0, m1), (None, None)), density, [16, 16], [1, 1], 4, 32)
    assert both.bits == order.bits > 0
    for leaf in range(1 << order.bits):
        g0, g1 = order.groups_of(leaf)
        assert int(both.lines[0, leaf, 0]) == group_or(m0, g0, 0) & group_or(m1, g1, 1) & 0xF
    only0, _ = reference.expected(Masks(PREFILL, (m0, None), (None, None)), torch.tensor([[0, 255], [255, 255]], dtype=torch.uint8), [16, 16], [1, 1], 4, 32)
    order0 = reference.order_of([4, 4], reference.groups_of(torch.tensor([[0, 255], [255, 255]], dtype=torch.uint8), [16, 16]), 0)
    for leaf in range(1 << order0.bits):
        assert int(only0.lines[0, leaf, 0]) == group_or(m0, order0.groups_of(leaf)[0], 0) & 0xF


def test_a_level_side_with_no_mask_takes_the_whole_in_the_carve_whatever_its_byte():
    """A missing mask's byte is taken as 255 before the carve, on both sides: a read level 1 at byte 3 with no mask gives the groups of byte 255 there, and the write side's leaf groups with it."""
    m0 = torch.zeros(1, 1, 1, 256, dtype=torch.int32)
    masks = Masks(PREFILL, (m0, None), (m0, m0))
    raw = torch.tensor([[3, 3], [3, 3]], dtype=torch.uint8)
    whole = torch.tensor([[3, 255], [3, 3]], dtype=torch.uint8)
    assert torch.equal(reference.effective_density(masks, raw), whole)
    assert reference.groups_of(reference.effective_density(masks, raw), [256, 256]) == reference.groups_of(whole, [256, 256])
    assert reference.groups_of(raw, [256, 256]) != reference.groups_of(whole, [256, 256])


def test_the_leaf_group_count_never_exceeds_most_for_arbitrary_bytes_and_masks_on_an_arbitrary_subset_of_level_sides():
    rng = random.Random(2026)
    for _ in range(600):
        levels = rng.randint(1, 4)
        widths = [rng.choice([2, 4, 8, 16, 32, 64, 128, 256]) for _ in range(levels)]
        width_bits = [w.bit_length() - 1 for w in widths]
        have = [[rng.random() < 0.5 for _ in range(levels)] for _ in range(2)]
        mask = lambda side, l: torch.zeros(1, 1, 1, widths[l], dtype=torch.int32) if have[side][l] else None  # noqa: E731
        masks = Masks(PREFILL, tuple(mask(0, l) for l in range(levels)), tuple(mask(1, l) for l in range(levels)))
        density = torch.tensor([[rng.randrange(256) for _ in range(levels)] for _ in range(2)], dtype=torch.uint8)
        carve = reference.groups_of(reference.effective_density(masks, density), widths)
        for side, bits_masked in enumerate(reference.masked_of(masks)):
            order = reference.order_of(width_bits, carve, side)
            assert order.bits <= reference.most(width_bits, bits_masked), (widths, density.tolist(), have)
            assert all(have[side][l] for l in order.listed), (widths, density.tolist(), have)


def test_the_layout_header_names_these_functions_and_these_arithmetics():
    text = LAYOUT.read_text()
    assert "kMaskWindowWords" not in text and "prepared_words_at" not in text and "prepared_block_bytes" not in text
    assert re.search(r"leaf_line_bytes\(int words\)", text) and re.search(r"std::int64_t\{4\} \* words", text)
    assert re.search(r"leaf_lines_at\(int bits, int words\)", text)
    assert re.search(r"counts = std::int64_t\{2\} << bits", text) and re.search(r"\(counts \+ line - 1\) / line \* line", text)
    assert re.search(r"leaf_block_bytes\(int bits, int words\)", text)
    assert re.search(r"leaf_lines_at\(bits, words\) \+ \(leaf_line_bytes\(words\) << bits\)", text)
    assert re.search(r"prepared_layout\(int levels, const int\* width_bits, const int\* masked", text)
    assert re.search(r"leaf_group_bits_most\(levels, width_bits, masked\[side\]\)", text)
    assert re.search(r"heads\[side\] \* windows \* leaf_block_bytes\(most, words\)", text)


@pytest.mark.parametrize(
    ("words", "bits", "lines_at", "block"),
    [
        #: 32 words: a line is 128 bytes. 1 leaf group: counts 2 bytes -> one line, 128; block 128 + 128 = 256. 64 (bits 6): counts 128 -> 128; block 128 + 64 * 128 = 8,320. 4,096 (bits 12): counts 8,192 -> 8,192;
        #: block 8,192 + 4,096 * 128 = 532,480. 256 (bits 8): counts 512 -> 512; block 512 + 256 * 128 = 33,280
        (32, 0, 128, 256), (32, 6, 128, 8320), (32, 8, 512, 33280), (32, 12, 8192, 532480),
        #: 16 words: a line is 64 bytes. bits 0: counts 2 -> 64; block 128. bits 6: counts 128 -> 128; block 128 + 64 * 64 = 4,224. bits 3: counts 16 -> 64; block 64 + 8 * 64 = 576
        (16, 0, 64, 128), (16, 6, 128, 4224), (16, 3, 64, 576),
    ],
)
def test_the_references_block_arithmetic_at_two_windows(words, bits, lines_at, block):
    assert reference.line_bytes(words) == 4 * words
    assert (reference.lines_at(bits, words), reference.block_bytes(bits, words)) == (lines_at, block)


def test_the_layout_of_a_hand_computed_call():
    """Two levels of 256 digits; the read side has a mask on both (most 12), the write side on one (most 8); 2 and 3 batch-heads, 2 windows, 32 words. Read region: 2 * 2 blocks of 532,480 = 2,129,920 bytes;
    write region from there: 3 * 2 blocks of 33,280 = 199,680; total 2,329,600."""
    assert reference.layout([8, 8], [0b11, 0b01], [2, 3], 2, 32) == ([0, 2129920], 2329600)
    #: a side with no mask has one leaf group: a block of 256 bytes at 32 words; with 1 and 1 batch-heads over 3 windows, 768 bytes each, the second region from 768
    assert reference.layout([4, 4], [0, 0], [1, 1], 3, 32) == ([0, 768], 1536)
    #: a small call: two levels of 2 digits, 4 leaves: both masked is at most 2 bits (bounded by the leaves), 1 bit with the first level only
    assert (reference.most([1, 1], 0b11), reference.most([1, 1], 0b01)) == (2, 1)


def test_the_bytes_and_windows_of_a_call_come_from_its_tokens_alone():
    assert [reference.windows_of(t, 32) for t in (1, 1024, 1025, 2048, 2049)] == [1, 1, 2, 2, 3]
    assert [reference.windows_of(t, 16) for t in (1, 512, 513)] == [1, 1, 2]
    masks = Masks(PREFILL, (None,), (None,))
    assert reference.bytes_of(masks, [16], [2, 3], 1025, 32) == 2 * 2 * 256 + 3 * 2 * 256


@pytest.mark.parametrize("words", [32, 16])
def test_the_reference_reads_the_bytes_the_layout_names(words):
    """A write of a call with two leaf groups on the read side (one level of 2 digits at a group of 1, alone: raised to the level, so a second level of 256 whole beside it keeps it at 1) laid by hand."""
    mask = torch.tensor([[[[0x5, 0x6]]]], dtype=torch.int32)
    masks = Masks(PREFILL, (mask, None), (None, None))
    density = torch.tensor([[0, 255], [255, 255]], dtype=torch.uint8)
    widths, heads, tokens = [2, 256], [1, 1], 3
    bits = reference.order_of([1, 8], reference.groups_of(density, widths), 0).bits
    assert bits == 1
    total = reference.bytes_of(masks, widths, heads, tokens, words)
    workspace = bytearray(total)
    #: read side (offset 0): counts [2, 2] (uint16), lines at the line boundary: leaf group 0 = digit 0's word 0 = 0x5, leaf group 1 = 0x6; the write side's block follows its region
    struct.pack_into("<HH", workspace, 0, 2, 2)
    struct.pack_into("<I", workspace, reference.lines_at(bits, words), 0x5)
    struct.pack_into("<I", workspace, reference.lines_at(bits, words) + reference.line_bytes(words), 0x6)
    got = reference.read(torch.frombuffer(workspace, dtype=torch.uint8).clone(), masks, density, widths, heads, tokens, words, 0)
    want, _ = reference.expected(masks, density, widths, heads, tokens, words)
    assert got.counts.tolist() == want.counts.tolist() == [[2, 2]]
    assert got.lines[0, :, 0].tolist() == want.lines[0, :, 0].tolist() == [0x5, 0x6]


def test_the_face_refuses_what_needs_no_card():
    from fsla.ops.mask_prepare import prepare_masks
    from fsla.routing.masks import DECODE

    density = torch.zeros(2, 1, dtype=torch.uint8)
    with pytest.raises(ValueError, match="prefill's"):
        prepare_masks(Masks(DECODE, (torch.zeros(1, 1, 1, dtype=torch.int32),), (torch.zeros(1, 1, 1, dtype=torch.int32),)), density, [16], [1, 1], 1)
    with pytest.raises(ValueError, match="density is on the host"):
        prepare_masks(Masks(PREFILL, (None,), (None,)), density, [16], [1, 1], 1)


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
