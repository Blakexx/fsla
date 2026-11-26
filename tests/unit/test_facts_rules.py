# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The rule layer, WITHOUT a device: every rule, fed values and asserted on.

A rule is a pure host fold from values to a decision, and it reads a value's shape,
dtype, device and PRESENCE but never its contents. That is what makes this battery
possible at all: it feeds plain tuples and ints, calls no kernel and
touches no device, and it must pass under `CUDA_VISIBLE_DEVICES=""`.

The run-table rows are the DERIVATION, not a spot check: every one of the eight
liveness bytes is asserted against every row of `engine/rules.md#run-table`, and the row
lines themselves are matched against that document so the doc and the rule cannot
drift apart.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from fsla.engine.rules import (
    envelope,
    run_table,
)

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# EnvelopeRule
# ---------------------------------------------------------------------------


def test_the_operand_envelope_names_the_device_and_no_activation():
    assert envelope(torch.device("cuda")) is None
    assert envelope(torch.device("cpu")) == (
        "the operands are on cpu; the arm is a CUDA kernel")


# ---------------------------------------------------------------------------
# RunTableRule -- the derivation, over every bit value and every documented row
# ---------------------------------------------------------------------------

_PRESENT = object()

#: `docs/internals/engine/rules.md#run-table`, transcribed: the row's line as it must
#: appear in that document, the two PRESENCE facts it is keyed by, and its rule.
_DOC_ROWS = (
    ("| stateless `y`-only | absent | absent | `R && W` |",
     (None, None),
     lambda R, W: R and W),
    ("| state-in `y`-only | present | absent | `R` |",
     (_PRESENT, None),
     lambda R, W: R),
    ("| continued state-out | present | present | `R \\| W` |",
     (_PRESENT, _PRESENT),
     lambda R, W: R or W),
    ("| fresh state-out | absent | present | `W` |",
     (None, _PRESENT),
     lambda R, W: W),
)


def test_the_documented_rows_are_the_rows_the_document_carries():
    """The row prose became an executable fixture, and this is the tie that
    keeps it one: a doc edit that changes a rule fails here."""
    doc = (ROOT / "docs" / "internals" / "engine" / "rules.md").read_text()
    missing = [row for row, _pair, _rule in _DOC_ROWS if row not in doc]
    assert not missing, f"rules.md#run-table no longer carries {missing}"


@pytest.mark.parametrize("row,pair,predicate", _DOC_ROWS,
                         ids=[r.split("|")[1].strip() for r, _t, _p in _DOC_ROWS])
def test_the_run_table_derives_every_bit_value_of_every_row(row, pair, predicate):
    """VACUITY GUARD: all eight liveness bytes, every row -- the derivation, never a
    spot check of the rows someone remembered. The STATE bit (4) is in no row's rule:
    a dense plane holds every atom, so it is asserted to change nothing."""
    state_in, state_out = pair
    table = run_table(state_in, state_out)
    for v in range(8):
        want = predicate(bool(v & 1), bool(v & 2))
        assert bool(table >> v & 1) is bool(want), (
            f"{row}: byte {v:#05b} derives {bool(table >> v & 1)}, the rule says "
            f"{bool(want)}")


def test_every_argument_pair_is_covered_by_a_row():
    """Four pairs, four rows."""
    covered = {pair for _row, pair, _rule in _DOC_ROWS}
    every = {(s_in, s_out) for s_in in (None, _PRESENT) for s_out in (None, _PRESENT)}
    assert every == covered


def test_the_run_table_reads_presence_and_never_a_tensor():
    """THE HARD INVARIANT, on the one rule whose inputs are tensors: it is handed
    planes whose contents would raise if read, and it derives the same table."""

    class _Unreadable:
        def __bool__(self):
            raise AssertionError("a rule read a fact's CONTENTS")

    assert run_table(_Unreadable(), _Unreadable()) == run_table(_PRESENT, _PRESENT)


@pytest.fixture(scope="module", autouse=True)
def _extension_calls_at_start():
    from fsla.ops._ext import extension

    info = extension.cache_info()
    return info.hits + info.misses


def test_no_rule_touched_the_extension(_extension_calls_at_start):
    """DEVICE-FREE IS THE CLAIM, and this is it measured rather than asserted: the
    battery must not have resolved the compiled extension at all. A DELTA from this
    module's start, because a shared pytest process may already have loaded the
    extension for another file's tests."""
    from fsla.ops._ext import extension

    info = extension.cache_info()
    assert info.hits + info.misses == _extension_calls_at_start, (
        f"the rule battery called into the extension ({info})")
