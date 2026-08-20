# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""K31 R2 conformance: the intra kernel against the fp64 intra reference.

The reference is `tests/k31/intra_reference.py`, itself gated against the oracle
by `test_intra_reference.py`. The cells walk the per-level support law: dense on
both sides, ALT (one entmax side and one softmax side, alternating across
levels), and the union-routing SHARED_SPARSE case, which must pass on the same
machinery with no second code path.
"""

from __future__ import annotations

import pytest
import torch

from rola.ops.intra import WINDOW_SMALL, WINDOW_WIDE, build_stamp, intra_forward
from tests.k31.intra_reference import (
    DENSE_BOTH,
    READ_SPARSE,
    SHARED_SPARSE,
    WINDOW,
    WRITE_SPARSE,
    as_token_major,
    intra_reference,
    make_cell,
    slab_skip_rate,
)
from tests.oracle.tolerances import BF16_RTOL

HEADS = 2
LEVELS = 2

#: THE WINDOW IS THE THIRD AXIS OF THE CELL: it fixes the output block's tile
#: count, and `W = 64` is the arm whose window holds a single tile -- the block
#: collapses onto it and the CTA halves, so it is a shape of its own.
CELLS = [
    ("dense-one-window", (DENSE_BOTH, DENSE_BOTH), None, WINDOW, False, WINDOW),
    ("dense-eight-windows", (DENSE_BOTH, DENSE_BOTH), None, 8 * WINDOW, False, WINDOW),
    ("alt-k4", (WRITE_SPARSE, READ_SPARSE), 4, 8 * WINDOW, False, WINDOW),
    ("alt-k16", (WRITE_SPARSE, READ_SPARSE), 16, 8 * WINDOW, False, WINDOW),
    ("alt-k4-cohort-1", (WRITE_SPARSE, READ_SPARSE), 4, 8 * WINDOW, True, WINDOW),
    ("alt-k4-cohort-64", (WRITE_SPARSE, READ_SPARSE), 4, 8 * WINDOW, 64, WINDOW),
    ("shared-k16", (SHARED_SPARSE, SHARED_SPARSE), 16, 2 * WINDOW, False, WINDOW),
    ("shared-k4-cohort-64", (SHARED_SPARSE, SHARED_SPARSE), 4, 2 * WINDOW, 64, WINDOW),
    ("dense-W512", (DENSE_BOTH, DENSE_BOTH), None, 4 * WINDOW_WIDE, False, WINDOW_WIDE),
    ("alt-k4-W512", (WRITE_SPARSE, READ_SPARSE), 4, 4 * WINDOW_WIDE, False, WINDOW_WIDE),
    ("shared-k16-W512", (SHARED_SPARSE, SHARED_SPARSE), 16, 2 * WINDOW_WIDE, False,
     WINDOW_WIDE),
    ("dense-W64", (DENSE_BOTH, DENSE_BOTH), None, 16 * WINDOW_SMALL, False, WINDOW_SMALL),
    ("alt-k4-W64", (WRITE_SPARSE, READ_SPARSE), 4, 16 * WINDOW_SMALL, False, WINDOW_SMALL),
]


def _relative(got, want):
    got, want = got.double(), want.double()
    return ((got - want).abs().max() / want.abs().max()).item()


@pytest.mark.parametrize("name,modes,k_tok,length,clustered,window",
                         CELLS, ids=[c[0] for c in CELLS])
def test_intra_matches_the_fp64_reference(name, modes, k_tok, length, clustered, window):
    pread, pwrite, gwrite, v_bh = make_cell(2 * HEADS, length, LEVELS, modes,
                                            k_tok=k_tok or 4, seed=11, clustered=clustered)
    o, den = intra_forward(pread, pwrite, gwrite, as_token_major(v_bh, HEADS), modes,
                           window=window)
    o_ref, den_ref = intra_reference(pread, pwrite, gwrite, v_bh, LEVELS, window=window)
    e_o, e_den = _relative(o, o_ref), _relative(den, den_ref)
    assert e_o < BF16_RTOL, f"{name}: the output leaves the bf16 band at {e_o:.3e}"
    assert e_den < BF16_RTOL, f"{name}: the mass leaves the bf16 band at {e_den:.3e}"


def test_the_write_back_reduces_rather_than_stores():
    """`fan_in_add` semantics: a second launch doubles what one launch deposited."""
    modes = (DENSE_BOTH, DENSE_BOTH)
    pread, pwrite, gwrite, v_bh = make_cell(HEADS, WINDOW, LEVELS, modes, seed=3)
    v = as_token_major(v_bh, HEADS)
    once_o, once_den = intra_forward(pread, pwrite, gwrite, v, modes)
    twice_o, twice_den = intra_forward(pread, pwrite, gwrite, v, modes)
    intra_forward(pread, pwrite, gwrite, v, modes, o=twice_o, den=twice_den)
    assert torch.equal(twice_o, once_o * 2) and torch.equal(twice_den, once_den * 2)


def test_the_slab_gate_does_not_change_the_answer():
    """The mask is a zero certificate, so declaring a side sparse cannot move a
    number: the same operands run under DENSE_BOTH and under the sparse mode."""
    modes = (WRITE_SPARSE, READ_SPARSE)
    pread, pwrite, gwrite, v_bh = make_cell(HEADS, 2 * WINDOW, LEVELS, modes, k_tok=4, seed=5,
                                            clustered=64)
    skipped, total = slab_skip_rate(pread, pwrite, modes)
    assert skipped > total // 2, (
        f"the cohort cell gates only {skipped}/{total} slabs; the claim is untested")
    v = as_token_major(v_bh, HEADS)
    gated_o, gated_den = intra_forward(pread, pwrite, gwrite, v, modes)
    ungated_o, ungated_den = intra_forward(pread, pwrite, gwrite, v, (DENSE_BOTH, DENSE_BOTH))
    assert torch.equal(gated_o, ungated_o) and torch.equal(gated_den, ungated_den)


def test_the_binary_under_test_is_this_tree():
    """A device-side fact: the loaded fatbin's own view of its shared-memory
    footprint and CTA width. Path and hash checks cannot catch a stale extension.

    The footprint is past the 48 KB static ceiling and the arm runs on opted-in
    dynamic shared memory, so the band runs to sm_86's 99 KB opt-in limit."""
    stamp = build_stamp()
    assert stamp % 1000003 == 256 and 16384 < stamp // 1000003 <= 101376
