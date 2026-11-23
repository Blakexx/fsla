# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""THE LEAF ORDER's own arithmetic: the lattice C++ solves (`csrc/fsla/src/common/plan/lattice.h`) and `fsla.ops.lattice`'s permutation by it, on the host, with no kernel.

`pi` maps a CANONICAL leaf index `ell` (MSB-first mixed radix, the oracle's forever) to a
LATTICE index `lambda`. Every plane the oracle and a kernel both touch crosses at that
map, so two properties of the map itself are load-bearing before any comparison means
anything: it must be a BIJECTION -- a map that dropped or doubled a leaf would silently
drop or double a row -- and `to_canonical . to_lattice` must be the identity, which is
what lets a test state its claim in whichever order is natural and convert.

The pins are the solver's answers where a flagship or a built arm depends on them: the two built carry arms must derive the
SAME box, because the flagship and the carry family write and read one plane. The solver is one function of the widths, reached by the
extension and by `plan_tool --lattice`; the two must agree, and the atom's leaves the law holds are `MMA_K_QUANTUM`.

Run: ``bazel test //tests:cpu --test_filter=test_lattice``
"""
from __future__ import annotations

import math
import subprocess
from pathlib import Path

import pytest
import torch

from fsla.engine.types import MMA_K_QUANTUM
from fsla.ops._ext import extension
from fsla.ops.decode import lattice_of
from fsla.ops.lattice import permutation, to_canonical, to_lattice
from fsla.ops.paging import bytes_equal

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "csrc" / "fsla" / "plan_tool"

#: Every shape of box the shipped topologies produce: `k = 1` (no admissible span),
#: `m = 1` (no admissible capacity), `D` from 1 to 4, and the flagship's `(4, 8)`.
_TOPOLOGIES = [
    (16,), (33,), (256,), (8, 16), (7, 9), (16, 16), (64, 64), (256, 256),
    (4, 4, 4), (3, 5, 7), (8, 8, 8), (2, 3, 4, 4), (4, 4, 4, 4),
]


def _box(widths, k, m):
    """``(bc, owners, spans, grids)`` of the ``(k, m)`` box, from the extension."""
    words = [int(x) for x in extension().fsla_decode_lattice_box(list(widths), k, m)]
    depth = len(widths)
    return words[2], words[3], words[4:4 + depth], words[4 + depth:4 + 2 * depth]


def _plan_tool_lattice(widths):
    run = subprocess.run([str(TOOL), "--lattice", ",".join(str(w) for w in widths)], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    lattice, constants = run.stdout.splitlines()
    return (dict(cell.split("=") for cell in lattice.split(": ", 1)[1].split()),
            dict(cell.split("=") for cell in constants.split(": ", 1)[1].split()))


def test_the_permutation_is_a_bijection():
    for widths in _TOPOLOGIES:
        k, m = lattice_of(widths)
        pi = permutation(widths, k, m, device="cpu")
        n = math.prod(widths)
        assert pi.shape == (n,) and pi.dtype == torch.int64
        assert torch.equal(pi.sort().values, torch.arange(n)), (
            f"pi is not a bijection at widths={widths}, (k, m)=({k}, {m})")


def test_the_two_conversions_are_inverse():
    for widths in _TOPOLOGIES:
        k, m = lattice_of(widths)
        plane = torch.randn(2, 3, math.prod(widths), 5)
        assert bytes_equal(to_canonical(to_lattice(plane, widths, k, m), widths, k, m),
                           plane), f"the seam is not round-trip exact at widths={widths}"


def test_the_box_is_consistent_with_the_widths():
    for widths in _TOPOLOGIES:
        k, m = lattice_of(widths)
        bc, owners, spans, grids = _box(widths, k, m)
        assert bc == k ** len(widths) * m
        assert bc * owners == math.prod(widths)
        for level, (width, span, extent) in enumerate(zip(widths, spans, grids, strict=True)):
            assert span * extent == width, f"level {level} of {widths} is not tiled by its span"


@pytest.mark.parametrize("widths, k, m, bc, spans", [
    ((16,), 4, 4, 16, [16]),
    ((32,), 4, 8, 32, [32]),
    ((256,), 4, 8, 32, [32]),
    ((64, 64), 4, 8, 128, [8, 16]),
    ((256, 256), 4, 8, 128, [8, 16]),
    ((16, 16, 16), 2, 16, 128, [2, 4, 16]),
    ((16, 16, 16, 16), 2, 8, 128, [2, 2, 2, 16]),
])
def test_the_solver_answers_the_shapes_the_built_arms_depend_on(widths, k, m, bc, spans):
    """`(64, 64)` and `(256, 256)` both give `(k, m) = (4, 8)`.

    This is what makes the flagship rows agree with the carry family's built arms: the
    two families write and read ONE plane, so a solver that answered differently for the
    two would put the same leaf in two places.
    """
    assert lattice_of(widths) == (k, m)
    assert _box(widths, k, m)[0::2] == (bc, spans)


def test_a_capacity_no_level_carries_is_refused_in_the_laws_words():
    with pytest.raises(RuntimeError, match=r"an owner sits inside a level"):
        extension().fsla_decode_lattice_box([31, 45], 4, 8)
    with pytest.raises(RuntimeError, match=r"does not fit widths \(16,\) at k = 4"):
        extension().fsla_decode_lattice_box([16], 4, 64)
    with pytest.raises(RuntimeError, match=r"capacity m must be a power of two, got 6"):
        extension().fsla_decode_lattice_box([16], 4, 6)


def test_the_extension_and_plan_tool_reach_one_solver():
    for widths in _TOPOLOGIES:
        solved, _ = _plan_tool_lattice(widths)
        words = [int(x) for x in extension().fsla_decode_lattice(list(widths))]
        assert words[:4] == [int(solved[key]) for key in ("k", "m", "bc", "owners")]
        assert ",".join(str(x) for x in words[4:4 + len(widths)]) == solved["spans"]
        assert ",".join(str(x) for x in words[4 + len(widths):]) == solved["grids"]


def test_the_atom_the_law_holds_is_the_page_granule_python_pages_by():
    assert int(_plan_tool_lattice((16,))[1]["atom_leaves"]) == MMA_K_QUANTUM
