# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The rule layer, WITHOUT a device: every rule, fed values and asserted on.

A rule is a pure host fold from values to a decision, and it reads a value's shape,
dtype, device and PRESENCE but never its contents. That is what makes this battery
possible at all: it feeds plain tuples and ints, calls no kernel and
touches no device, and it must pass under `CUDA_VISIBLE_DEVICES=""`. Before P75 the
run table was reachable only through a launch (docs/internals/engine/rules.md).

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
    admission,
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
# AdmissionRule (the pure half)
# ---------------------------------------------------------------------------

def test_a_run_that_fits_the_remainder_stays_in_the_open_extent():
    assert admission(allocated=5, capacity=64, extent_atoms=8, count=3) == 5


def test_a_run_that_does_not_fit_opens_the_next_extent():
    """STRUCTURAL, not a threshold: the count is compared against the remainder, so a
    large admission lands contiguous rather than straddling."""
    assert admission(allocated=5, capacity=64, extent_atoms=8, count=4) == 8
    assert admission(allocated=8, capacity=64, extent_atoms=8, count=8) == 8


def test_an_empty_run_moves_nothing():
    assert admission(allocated=5, capacity=64, extent_atoms=8, count=0) == 5


def test_admission_is_all_or_nothing_at_the_ceiling():
    #: THE ALIGNMENT IS INSIDE THE CEILING CHECK: this run does not fit the open
    #: extent's remainder, so it starts at 64 and there is nothing left -- a rule
    #: that checked the demand against `capacity - allocated` would grant it.
    with pytest.raises(RuntimeError, match="5 new atoms demanded, 0 free"):
        admission(allocated=60, capacity=64, extent_atoms=8, count=5)
    with pytest.raises(RuntimeError, match="grants ALL or refuses"):
        admission(allocated=0, capacity=64, extent_atoms=8, count=65)
    #: and a run that DOES fit the remainder is served from it, at the same cursor.
    assert admission(allocated=60, capacity=64, extent_atoms=8, count=4) == 60


# ---------------------------------------------------------------------------
# RunTableRule -- the derivation, over every bit value and every documented row
# ---------------------------------------------------------------------------

_PRESENT = object()

#: `docs/internals/engine/rules.md#run-table`, transcribed: the row's line as it must
#: appear in that document, the three PRESENCE facts it is keyed by, and its rule.
_DOC_ROWS = (
    ("| stateless `y`-only | absent | absent | absent | `R && W` |",
     [(None, None, None)],
     lambda R, W, S: R and W),
    ("| state-in `y`-only, paged | present | absent | present | `R && (W \\| S)` |",
     [(_PRESENT, None, _PRESENT)],
     lambda R, W, S: R and (W or S)),
    ("| state-in `y`-only, dense | present | absent | absent | `R` |",
     [(_PRESENT, None, None)],
     lambda R, W, S: R),
    ("| continued state-out, paged | present | present | present | `(R && (W \\| S)) \\| W` |",
     [(_PRESENT, _PRESENT, _PRESENT)],
     lambda R, W, S: (R and (W or S)) or W),
    ("| fresh state-out, paged | absent | present | present | `W` |",
     [(None, _PRESENT, _PRESENT)],
     lambda R, W, S: W),
    ("| continued state-out, dense | present | present | absent | `R \\| W` |",
     [(_PRESENT, _PRESENT, None)],
     lambda R, W, S: R or W),
    ("| fresh state-out, dense | absent | present | absent | `W` |",
     [(None, _PRESENT, None)],
     lambda R, W, S: W),
)


def test_the_documented_rows_are_the_rows_the_document_carries():
    """The seven-row prose became an executable fixture, and this is the tie that
    keeps it one: a doc edit that changes a rule fails here."""
    doc = (ROOT / "docs" / "internals" / "engine" / "rules.md").read_text()
    missing = [row for row, _triples, _rule in _DOC_ROWS if row not in doc]
    assert not missing, f"rules.md#run-table no longer carries {missing}"


@pytest.mark.parametrize("row,triples,predicate", _DOC_ROWS,
                         ids=[r.split("|")[1].strip() for r, _t, _p in _DOC_ROWS])
def test_the_run_table_derives_every_bit_value_of_every_row(row, triples, predicate):
    """VACUITY GUARD: all eight liveness bytes, every row -- the derivation, never a
    spot check of the rows someone remembered."""
    for state_in, state_out, page_table in triples:
        table = run_table(state_in, state_out, page_table)
        for v in range(8):
            want = predicate(bool(v & 1), bool(v & 2), bool(v & 4))
            assert bool(table >> v & 1) is bool(want), (
                f"{row}: byte {v:#05b} derives {bool(table >> v & 1)}, the rule says "
                f"{bool(want)}")


def test_every_argument_triple_is_covered_by_a_row_or_refused_upstream():
    """Eight triples, seven rows: the one triple no row carries is a page table with no
    state, which the binding refuses before the derivation is reached -- and which
    derives the stateless row regardless, so nothing depends on the refusal."""
    covered = {t for _row, triples, _rule in _DOC_ROWS for t in triples}
    every = {(s_in, s_out, tbl)
             for s_in in (None, _PRESENT)
             for s_out in (None, _PRESENT)
             for tbl in (None, _PRESENT)}
    assert every - covered == {(None, None, _PRESENT)}
    assert run_table(None, None, _PRESENT) == run_table(None, None, None)


def test_the_run_table_reads_presence_and_never_a_tensor():
    """THE HARD INVARIANT, on the one rule whose inputs are tensors: it is handed
    planes whose contents would raise if read, and it derives the same table."""

    class _Unreadable:
        def __bool__(self):
            raise AssertionError("a rule read a fact's CONTENTS")

    assert run_table(_Unreadable(), _Unreadable(), _Unreadable()) == run_table(
        _PRESENT, _PRESENT, _PRESENT)


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
