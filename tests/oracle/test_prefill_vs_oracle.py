# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE OP AGAINST THE fp64 ORACLE: `fsla.prefill`'s output and its handed-back state, per slot.

THE REFERENCE IS `fsla.ops.naive.naive_fsla` DIRECTLY, on the bundle's own dequantized levels and values -- the op gate
compares on the same quantized inputs the kernels read (ruling 2026-09-26). The atom fold, the carry, the den pass,
the intra and the epilogue run as one call and the whole answer is compared against the recurrence itself, so a
window-grid disagreement, a double-counted diagonal or a dropped tile pair has nowhere to hide.

OWED BY STRICT XFAIL (docs/testing.md): no prefill kernel is built on this line, so each test that calls the op carries
`requires_kernel("prefill")` (`tests/conftest.py`): the refusal `fsla.ops._ext.KernelNotBuilt` is the expected failure,
a pass is a failure, and the marker comes off in the commit that builds the kernel. The rule's own teeth and the
decomposition's non-vacuity run no kernel.

Symbols: ``W`` the window, ``L`` tokens, ``y`` the op's output ``[B, T, H, d_v]``.
"""
from __future__ import annotations

import pytest
import torch

import fsla
from fsla.ops.window import window_of
from measure.cells import by_name, carry_cells, realize
from measure.subjects import bundle_of
from tests.oracle import reference
from tests.oracle.fixtures import assert_slots_close, oracle_run
from tests.oracle.tolerances import PREFILL_READOUT, PREFILL_STATE

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")

#: every oracle cell a fresh call can run: a carried cell's entry state reaches the op through a chained call
CELLS = tuple(c for c in carry_cells(tier="oracle") if c.state == "fresh")
IDS = [c.name for c in CELLS]


def operands(spec, tokens: slice = slice(None)):
    """``(routes, values, masks, density)``: the cell's draw over ``tokens`` in the format the kernels read."""
    return bundle_of(spec, realize(spec), tokens)


def call(operands_, state=None):
    """The op over one call's operands (``routes, values, masks, density``)."""
    routes, values, masks, density = operands_
    return fsla.prefill(routes, values, masks, density, state=state)


def oracle(routes, values, entry=None):
    """`tests.oracle.fixtures.oracle_run` on the bundle's own operands, chained from ``entry``."""
    read, write = routes.dequantized()
    return oracle_run(values.dequantized().double(), tuple(x.double() for x in read), tuple(x.double() for x in write), routes.topology, entry=entry)


@pytest.mark.parametrize("spec", CELLS, ids=IDS)
@pytest.mark.requires_kernel("prefill")
def test_the_op_matches_the_fp64_oracle(spec):
    """``y`` of a stateless call, every token and channel, against the recurrence's readout and its envelope."""
    routes, values, _masks, _density = operands_ = operands(spec)
    y, state = call(operands_)
    assert state is None
    want = oracle(routes, values)
    assert_slots_close(y, want.y, output=PREFILL_READOUT, envelope=want.y_envelope, what=f"{spec.name} y")


@pytest.mark.parametrize("spec", CELLS, ids=IDS)
@pytest.mark.requires_kernel("prefill")
def test_the_op_hands_back_the_fp64_state(spec):
    """A stateful call binds the state and hands back the recurrence's final state, leaf by leaf in canonical order
    (`FSLAState.materialize`), against the oracle's and its envelope."""
    routes, values, _masks, _density = operands_ = operands(spec)
    state = fsla.state()
    call(operands_, state=state)
    want = oracle(routes, values)
    assert_slots_close(state.materialize(), want.state, output=PREFILL_STATE, envelope=want.state_envelope,
                       what=f"{spec.name} state")


@pytest.mark.requires_kernel("prefill")
def test_a_chained_call_equals_one_long_call():
    """THE STATE IS THE WHOLE CONTRACT of a chunked prefill: two calls over the halves, one state between them, give
    the long call's readout and final state, each against the oracle's one long run."""
    spec = by_name("flagship-alt-k4-L4096")
    half = spec.tokens // 2
    state = fsla.state()
    ys = [call(operands(spec, part), state=state)[0] for part in (slice(0, half), slice(half, spec.tokens))]
    routes, values, _masks, _density = operands(spec)
    want = oracle(routes, values)
    assert_slots_close(torch.cat(ys, dim=1), want.y, output=PREFILL_READOUT, envelope=want.y_envelope,
                       what="chained y")
    assert_slots_close(state.materialize(), want.state, output=PREFILL_STATE, envelope=want.state_envelope,
                       what="chained state")


@pytest.mark.requires_kernel("prefill")
def test_the_per_slot_rule_fails_what_a_global_max_passes():
    """The rule the cells are graded in can fail (docs/testing.md): a small slot moved by a hundredth of the readout's
    largest magnitude passes a global max-relative test at 2% and fails per slot."""
    routes, values, _masks, _density = operands(by_name("flagship-alt-k4"))
    want = oracle(routes, values)
    planted = want.y.clone()
    i = int(torch.argmin(want.y.abs().flatten()))
    planted.view(-1)[i] += 0.01 * float(want.y.abs().max())
    assert float((planted - want.y).abs().max()) <= 0.02 * float(want.y.abs().max())
    with pytest.raises(AssertionError, match="off the oracle"):
        assert_slots_close(planted, want.y, output=PREFILL_READOUT, envelope=want.y_envelope, what="planted")


@pytest.mark.requires_kernel("prefill")
def test_the_intra_term_is_actually_present():
    """NON-VACUITY: the inter term alone fails the op's gate on a multi-window cell, so a call that dropped the intra
    (or the carry) could not pass. Runs no kernel."""
    routes, values, _masks, _density = operands(by_name("flagship-alt-k4"))
    read, write = routes.dequantized()
    num, den, _ = reference.inter_reference(tuple(x.double() for x in read), tuple(x.double() for x in write), values.dequantized().double(), routes.topology.widths, window_of())
    want = oracle(routes, values)
    with pytest.raises(AssertionError, match="off the oracle"):
        assert_slots_close(num / (den[..., None] + 1e-5), want.y, output=PREFILL_READOUT, envelope=want.y_envelope,
                           what="inter alone")


def test_a_prefill_with_a_decay_is_outside_the_contract():
    """A DECAY IS DECODE'S: no descriptor, guaranteed or owed, holds a prefill that decays, so its call is refused when its descriptor is
    made -- as a call outside the contract, before any kernel could run."""
    from tests.descriptors import prefill_descriptor

    routes, values, _masks, _density = operands(by_name("flagship-alt-k4"))
    with pytest.raises(RuntimeError, match="outside FSLA's prefill contract"):
        prefill_descriptor(routes.topology, d_v=values.d_v, decay=True, tokens=routes.tokens, batch_heads=routes.batch * routes.heads,
                           stateful=False, device=routes.device)
