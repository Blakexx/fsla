# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CTA'S CARVE (`csrc/fsla/src/common/plan/cta_carve.h`, docs/internals/common/plan/cta_carve.md): `plan_tool --cta-carve` prints the carve of a call's levels from their width bits
and each side's densities, with no GPU. The hand-worked cases are worked in their comments; the properties run over the calls of depth 1 to 4.

Run: ``bazel test //tests:cpu --test_filter=test_cta_carve``."""
from __future__ import annotations

import itertools
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import one_read_head

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TOOL = ROOT / "csrc" / "fsla" / "plan_tool"
CTA_LEAVES = 512
WHOLE = 256  # a side walks the level whole
SPARSE = 1  # a side walks the level by its lists


def _run(calls: list[tuple[list[int], list[int], list[int]]]) -> list[dict]:
    """The tool's answers to calls of (width bits, write densities, read densities) a level, one process for each thousand of them (an argument has a length limit)."""
    if len(calls) > 1000:
        return [out for at in range(0, len(calls), 1000) for out in _run(calls[at:at + 1000])]
    arg = lambda pick: ";".join(",".join(map(str, pick(c))) for c in calls)  # noqa: E731
    run = subprocess.run([str(TOOL), "--cta-carve", arg(lambda c: c[0]), "--write-density", arg(lambda c: c[1]), "--read-density", arg(lambda c: c[2])],
                         capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr
    out = [one_read_head(json.loads(line)) for line in run.stdout.splitlines()]
    assert len(out) == len(calls)
    return out


def _one(widths: list[int], write: list[int], read: list[int]) -> dict:
    return _run([(widths, write, read)])[0]


def _sizes(out: dict) -> list[int]:
    return [1 << b for b in out["carve_bits"]]


A = ([8], [SPARSE], [WHOLE])
B = ([8], [WHOLE], [SPARSE])
C = ([8], [WHOLE], [WHOLE])


def _join(*levels):
    return [sum((lv[i] for lv in levels), []) for i in range(3)]


def test_a_writes_sparse_and_reads_whole_b_writes_whole_and_reads_sparse_carve_32_x_16():
    # each doubling goes to the level that saves the most fetches. A is the read side's level (its size 256) and B the write side's, and each side has as much left, so the savings tie at every pair of equal
    # shares and the level the READ side needs (A) takes the tie: A, B, A, B ... and the ninth is A's: 32 x 16. The groups follow from the densities (write: A 1 digit, B 256; read: A 256, B 1): the fold warpgroups split along A
    # (its write group, 1, is under its width), 16 x 16 digits each; a token on the write side spans A's 1 x B's 16 = 16 leaves of a fold, so a fold holds 16 sets; the read side spans A's 32 x B's 1 = 32
    # leaves of the carve, 16 sets. 8 CTAs along A and 16 along B: 128 a head; B's whole write group spans 16 CTAs, so those 16 CTAs see one token schedule and are taken together (the schedule): 8 units.
    out = _one(*_join(A, B))
    assert _sizes(out) == [32, 16] and out["cta_leaves"] == 512
    assert (out["write_group_bits"], out["read_group_bits"]) == ([0, 8], [8, 0])
    assert (out["split_level"], out["folds"], out["fold_bits"], out["fold_leaves"]) == (0, 2, [4, 4], 256)
    assert (out["fold_set_leaves"], out["fold_sets"], out["read_set_leaves"], out["read_sets"]) == (16, 16, 32, 16)
    assert (out["ctas_per_head"], out["unit_ctas"], out["units_per_head"]) == (128, 16, 8)
    # the write side is sparse on level 0 and whole on level 1, so it loops 0 outside, 1 inside; the read side is the other way round; only level 1's write group (8 bits) exceeds its carve (4 bits)
    assert (out["write_order"], out["read_order"], out["unit_bits"]) == ([0, 1], [1, 0], [0, 4])


@pytest.mark.parametrize("groups, sizes", [
    (_join(B, A), [16, 32]),
    (_join(C, C), [32, 16]),
    (_join(A, B, C), [2, 1, 256]),
    (_join(A, B, A, B), [8, 4, 4, 4]),
    (_join(([8], [5], [WHOLE]), B), [32, 16]),
])
def test_the_coordinators_cases(groups, sizes):
    # A sparse / whole against B whole / sparse: each side needs one level and has as much left, so the level the read side needs takes each tie; B sparse / whole against A whole / sparse puts
    # that on B. Whole / whole twice ties in everything and the lower level takes the ninth doubling. With C, whole on both sides, both sides save on C and it takes its eight doublings; the ninth is A's,
    # the level the read side needs. A write size of 4 digits on a level the read side needs anyway changes no share here.
    assert _sizes(_one(*groups)) == sizes


def test_the_level_the_read_side_needs_takes_a_tie_and_the_lower_level_a_tie_of_both():
    assert _sizes(_one(*_join(([8], [WHOLE], [SPARSE]), ([8], [SPARSE], [WHOLE])))) == [16, 32]
    assert _sizes(_one(*_join(C, C))) == [32, 16]


@pytest.mark.parametrize("write, bits", [(1, [0, 0]), (5, [2, 2])])
def test_c_and_d_take_the_doublings_that_cover_their_write_groups_and_no_more(write, bits):
    # C and D are read at one digit. Written at one digit neither side needs them and they stay at one; written at 5 of 256 (a size of 4 digits) each takes the two doublings that cover its group.
    c = ([8], [write], [SPARSE])
    out = _one(*_join(A, B, c, c))
    assert out["carve_bits"][2:] == bits and out["cta_leaves"] == CTA_LEAVES


def test_a_level_whole_on_both_sides_takes_the_whole_cta_from_a_sparse_one():
    # A's sizes are one digit on both sides: neither side needs it, so B, which both need, takes its eight doublings and A the one that is left.
    assert _sizes(_one(*_join(([8], [SPARSE], [SPARSE]), C))) == [2, 256]


@pytest.mark.parametrize("widths, leaves", [([8], 256), ([4, 4], 256), ([4], 16), ([2, 3], 32)])
def test_a_call_of_fewer_leaves_than_a_cta_owns_takes_every_leaf(widths, leaves):
    out = _one(widths, [WHOLE] * len(widths), [SPARSE] * len(widths))
    assert out["cta_leaves"] == leaves and out["carve_bits"] == widths


def _left(sizes, carve) -> int:
    """A side's left: the product of size over share on the levels where its size is the larger (bits in, a count out)."""
    return 2 ** sum(max(0, size - share) for size, share in zip(sizes, carve))


@pytest.mark.parametrize("depth", [2, 3])
def test_the_carve_has_the_least_fetches_of_every_carve_over_every_call_of_sizes_1_4_32_and_256(depth):
    # the rule gives each doubling to the level that saves the most; over every call of these sizes on levels of 256 digits its carve has the least sum of the two sides' left of all carves of nine doublings
    sizes = (0, 2, 5, 8)
    calls = [(list(w), list(r)) for w in itertools.product(sizes, repeat=depth) for r in itertools.product(sizes, repeat=depth)]
    outs = _run([([8] * depth, [1 << b for b in w], [1 << b for b in r]) for w, r in calls])
    carves = [c for c in itertools.product(range(9), repeat=depth) if sum(c) == 9]
    for (w, r), out in zip(calls, outs):
        got = _left(w, out["carve_bits"]) + _left(r, out["carve_bits"])
        assert got == min(_left(w, c) + _left(r, c) for c in carves), (w, r, out["carve_bits"])


def _calls(depth: int, widths=(4, 6, 8)):
    for ws in itertools.product(widths, repeat=depth):
        for write in itertools.product((SPARSE, 5, WHOLE), repeat=depth):
            for read in itertools.product((SPARSE, 5, WHOLE), repeat=depth):
                yield list(ws), list(write), list(read)


def _check(call, out) -> int:
    widths, _, _ = call
    carve = out["carve_bits"]
    assert all(c <= w for c, w in zip(carve, widths)), call
    assert out["cta_leaves"] == min(CTA_LEAVES, 1 << sum(widths)), call
    assert out["cta_leaves"] == 1 << sum(carve), call
    assert out["fold_sets"] * out["fold_set_leaves"] == out["fold_leaves"], call
    assert out["read_sets"] * out["read_set_leaves"] == out["cta_leaves"], call
    assert out["units_per_head"] * out["unit_ctas"] == out["ctas_per_head"], call
    levels = range(len(widths))
    for order, density in ((out["write_order"], call[1]), (out["read_order"], call[2])):
        assert order == sorted(levels, key=lambda l: (density[l], l)), call
    assert out["unit_bits"] == [max(0, out["write_group_bits"][l] - carve[l]) for l in levels], call
    #: a set is at least 16 leaves where its box holds as many, and a group is never under its level's size
    assert out["fold_set_leaves"] >= min(16, out["fold_leaves"]) and out["read_set_leaves"] >= min(16, out["cta_leaves"]), call
    assert all(g >= s for g, s in zip(out["write_group_bits"], out["write_size_bits"], strict=True)), call
    assert all(g >= s for g, s in zip(out["read_group_bits"], out["read_size_bits"], strict=True)), call
    assert out["unit_ctas"] == 2 ** sum(out["unit_bits"]), call
    assert out["fold_leaves"] * out["folds"] == out["cta_leaves"] and out["fold_leaves"] == 1 << sum(out["fold_bits"]), call
    assert out["ctas_per_head"] == 1 << (sum(widths) - sum(carve)), call
    return out["cta_leaves"]


def test_the_properties_over_every_call_of_depth_1_to_3_and_a_fixed_sample_of_depth_4():
    calls = [c for depth in (1, 2, 3) for c in _calls(depth)]
    rng = random.Random(20261123)
    calls += rng.sample(list(_calls(4)), 20000)
    below = [c for c, out in zip(calls, _run(calls)) if _check(c, out) < CTA_LEAVES]
    print(f"calls checked {len(calls)}; calls ending below {CTA_LEAVES} leaves {len(below)}; five of them:")
    for c in below[:5]:
        print("  ", c)
    # a call ends below 512 leaves only where it has fewer than 512 leaves, every level at its width
    assert all(sum(c[0]) < 9 for c in below)


# --- several read heads (grouped queries): the write side and each read head are parties of the one rule ---------

def _run_heads(calls: list[tuple[list[int], list[int], list[list[int]]]]) -> list[dict]:
    """The tool's answers to calls of (width bits, write densities, each read head's densities), a read field a list of the heads'."""
    arg = lambda pick: ";".join(",".join(map(str, pick(c))) for c in calls)  # noqa: E731
    reads = ";".join("/".join(",".join(map(str, head)) for head in c[2]) for c in calls)
    run = subprocess.run([str(TOOL), "--cta-carve", arg(lambda c: c[0]), "--write-density", arg(lambda c: c[1]), "--read-density", reads],
                         capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr
    out = [json.loads(line) for line in run.stdout.splitlines()]
    assert len(out) == len(calls)
    return out


def _total_saving_carve(widths: list[int], parties: list[list[int]], read_parties: int) -> list[int]:
    """The rule, restated: nine doublings, each to the level of the largest saving -- half of the left of every party it
    halves, added -- then the smaller share, a level a read head needs, the lower level. `parties` are sizes as bits, the
    write side first, the last `read_parties` the read heads."""
    carve = [0] * len(widths)
    for _ in range(9):
        best, best_key = None, None
        for level, width in enumerate(widths):
            if carve[level] + 1 > width:
                continue
            halves = [party[level] > carve[level] for party in parties]
            saving = sum(_left(party, carve) // 2 for party, cut in zip(parties, halves) if cut)
            key = (saving, -carve[level], any(halves[-read_parties:]), -level)
            if best_key is None or key > best_key:
                best, best_key = level, key
        if best is not None:
            carve[best] += 1
    return carve


def test_several_read_heads_are_parties_of_the_one_rule_over_a_fixed_sample():
    rng = random.Random(20261127)
    calls = []
    for _ in range(4000):
        depth, heads = rng.choice((1, 2, 3, 4)), rng.choice((1, 2, 3, 4, 8))
        calls.append(([rng.choice((2, 4, 6, 8)) for _ in range(depth)], [rng.choice((1, 4, 32, 255, 256)) for _ in range(depth)],
                      [[rng.choice((1, 4, 32, 255, 256)) for _ in range(depth)] for _ in range(heads)]))
    for call, out in zip(calls, _run_heads(calls)):
        widths, _write, reads = call
        assert out["read_heads"] == len(reads), call
        assert out["carve_bits"] == _total_saving_carve(widths, [out["write_size_bits"], *out["read_size_bits"]], len(reads)), call
        #: the read side has ONE loop order, the levels by the heads' densities summed, the lower level outside a tie
        summed = [sum(head[level] for head in out["read_density"]) for level in range(len(widths))]
        assert out["read_order"] == sorted(range(len(widths)), key=lambda level: (summed[level], level)), call
        #: each head has its own groups and sets over the one carve
        for head in range(len(reads)):
            assert out["read_sets"][head] * out["read_set_leaves"][head] == out["cta_leaves"], call
            assert all(g >= s for g, s in zip(out["read_group_bits"][head], out["read_size_bits"][head], strict=True)), call
            assert out["read_set_leaves"][head] >= min(16, out["cta_leaves"]), call
            assert out["read_leaf_one_to_one"][head] is True, call


def test_read_heads_that_are_alike_pull_the_carve_toward_the_level_they_need():
    # the write side sparse on level 0 and whole on level 1, a head a group of 32 digits on level 0: one head carves
    # 8 x 64; more heads alike are more of the read side's fetches, and level 0 takes a doubling at two heads and
    # another at eight -- 16 x 32, then 32 x 16. A cell whose sides already tie does not move: the measured cell stays 32 x 16
    calls = [([8, 8], [1, 256], [[32, 1]] * heads) for heads in (1, 2, 4, 8)] + [([8, 8], [4, 256], [[256, 4]] * heads) for heads in (1, 4)]
    carves = [out["carve_bits"] for out in _run_heads(calls)]
    assert carves == [[3, 6], [4, 5], [4, 5], [5, 4], [5, 4], [5, 4]]


def test_the_tool_refuses_more_read_heads_than_the_carve_takes_and_a_head_of_another_depth():
    for reads in ("/".join(["256,4"] * 9), "256,4/256"):
        run = subprocess.run([str(TOOL), "--cta-carve", "8,8", "--write-density", "4,256", "--read-density", reads], capture_output=True, text=True, timeout=60)
        assert run.returncode != 0 and "--read-density" in run.stderr, reads
