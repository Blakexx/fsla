# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""THE GATED DECODE STEP: the deposit fact that forced it, and CUDA-graph capture.

S4b left the steady paged step host-wait-bound -- 0.093 ms of a 0.177 ms step was the
host waiting for a 4-byte growth flag to become visible, in front of a 0.007 ms kernel.
P67-a moves the wait behind the launches, which is only sound because the step can be
made a device-side NO-OP. This file gates the whole of that.

Vocabulary, defined here and used unqualified below: **atom** = 16 consecutive leaves in
canonical order = the page granule; **slot** = the physical page an atom maps to (`-1` =
ABSENT); **gate** = the `[1]` int32 the probe reduces into and the step reads; **steady
step** = one whose write atoms are all resident; **growth step** = one whose are not.

**THE FACT THAT DECIDED THE DESIGN (cell 1).** A deposit into an absent atom is DROPPED
by the kernel -- `decode.cu`'s walk takes `if (in_w && atom_present)`. So a host that
checked the growth flag LAZILY (every K steps, a pinned poll) would run steps whose
writes are silently lost: laziness is not an available design at any K > 1, and the only
sound shapes are per-step synchronization or a device-side conditional. The gate is that
conditional.

**WHAT THE GATE BUYS (cells 2-3).** A gated step touches nothing -- no `y`, no state, no
counter -- so it can be ENQUEUED before its verdict is host-visible and, if the verdict
says grow, discarded and replayed after the admission. The result is bit-identical to
S4b's admit-then-step order, which cell 3 states as `torch.equal` against the enumerated
reference's own admission.

**WHAT IT UNLOCKS (cells 4-5).** With no host read inside it, the step is CUDA-graph
capturable. A replay is SELF-CORRECTING: the probe re-runs inside the graph, so after an
admission the same captured graph performs the step it had skipped.

Run: ``pytest tests/integration/test_decode_graph_step.py``
"""
from __future__ import annotations

import pytest
import torch

from rola.engine.dags.decode_dag import admit_growth, decode_step_gated, growth_pending
from rola.ops.decode import _decode_step, _write_atom_bitmap
from rola.ops.paging import MMA_K_QUANTUM
from rola.routing.factors import RouteFactors
from tests.integration.test_decode_paging import (
    _COLS,
    _DV,
    _arena,
    _config,
    _f32,
    _fixture,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="decode is a CUDA kernel"),
]

_WIDTHS, _BC = (16, 16), 64


def _absent_one_written_atom(fixture, config, arena):
    """Admit the step's write set except ONE atom, and say which one. `(bitmap, atom)`."""
    _v, _read, write, _g = _f32(fixture)
    bitmap = _write_atom_bitmap(write, config)
    row, column = (bitmap.nonzero())[0].tolist()
    held_out = bitmap.clone()
    held_out[row, column] = False
    arena.plan_exact(held_out, fresh=True)
    arena.wait()
    return bitmap, (row, column)


def _step(fixture, config, arena, *, gate=None):
    v, read, write, g_write = _f32(fixture)
    return _decode_step(v, read, write, None, g_write, config, arena.state,
                        page_table=arena.page_table, gate=gate)[0]


def _atom_rows(arena, row, column):
    """The `[16, cols]` page an atom maps to. The caller must know it is resident."""
    slot = int(arena.page_table[row, column])
    assert slot >= 0, "the atom is absent; there are no rows to read"
    return arena.state[slot]


# ---------------------------------------------------------------------------
# 1. the deposit fact
# ---------------------------------------------------------------------------

def test_a_deposit_into_an_absent_atom_is_dropped_which_is_why_laziness_is_refused():
    """The step DROPS what it cannot address, so a late flag read LOSES WRITES.

    Run one step with exactly one written atom held out, then admit it and look: its
    rows are zero, not the deposit. The same step with the atom resident writes them.
    This is the whole argument against checking the flag every K steps.
    """
    fixture = _fixture(_WIDTHS, seed=11, support=0.5)
    config = _config(_WIDTHS, fixture)
    BH = fixture["B"] * fixture["H"]

    lossy = _arena(_WIDTHS, _BC, BH)
    bitmap, (row, column) = _absent_one_written_atom(fixture, config, lossy)
    _step(fixture, config, lossy)
    #: Admitting AFTER the step is exactly what a lazy host would do one token late.
    lossy.plan_exact(bitmap, fresh=False)
    lossy.wait()
    lost = _atom_rows(lossy, row, column)

    kept = _arena(_WIDTHS, _BC, BH)
    kept.plan_exact(bitmap, fresh=True)
    kept.wait()
    _step(fixture, config, kept)
    deposited = _atom_rows(kept, row, column)

    assert float(deposited.abs().max()) > 0.0, "the fixture's held-out atom is never written"
    assert float(lost.abs().max()) == 0.0, (
        "a deposit into an absent atom SURVIVED; if the kernel ever starts keeping them "
        "this file's whole design argument changes and the lazy-poll design reopens")


# ---------------------------------------------------------------------------
# 2. the gate is a no-op, exactly
# ---------------------------------------------------------------------------

def test_a_gated_step_touches_nothing():
    """Gate up: no state, no `y` semantics, no counter -- the step never happened."""
    fixture = _fixture(_WIDTHS, seed=12, support=0.5)
    config = _config(_WIDTHS, fixture)
    BH = fixture["B"] * fixture["H"]
    arena = _arena(_WIDTHS, _BC, BH)
    _v, _read, write, _g = _f32(fixture)
    arena.plan_exact(_write_atom_bitmap(write, config), fresh=True)
    arena.wait()

    before = arena.state.clone()
    gate = torch.ones(1, dtype=torch.int32, device="cuda")
    _step(fixture, config, arena, gate=gate)
    assert torch.equal(arena.state, before), "a gated step wrote state"

    #: And the gate DOWN is the ordinary step, on the same buffer -- the field is an
    #: input, never a mode.
    gate.zero_()
    _step(fixture, config, arena, gate=gate)
    assert not torch.equal(arena.state, before), "the gate down did not run the step"


# ---------------------------------------------------------------------------
# 3. the gated growth order is the admit-first order
# ---------------------------------------------------------------------------

def test_probe_then_gated_step_then_admit_then_replay_is_bit_identical_to_admitting_first():
    """P67-a's host order against S4b's, `torch.equal` on `y` and on the plane.

    The reference admits from `_write_atom_bitmap` -- the enumerated reference, whose
    only callers are gates -- so the two orders are compared through two independent
    derivations of the same write set.
    """
    fixture = _fixture(_WIDTHS, seed=13, support=0.5)
    config = _config(_WIDTHS, fixture)
    BH = fixture["B"] * fixture["H"]
    _v, _read, write, _g = _f32(fixture)
    bitmap = _write_atom_bitmap(write, config)

    reference = _arena(_WIDTHS, _BC, BH)
    reference.plan_exact(bitmap, fresh=True)
    reference.wait()
    y_reference = _step(fixture, config, reference)

    #: The gated order: a step that cannot address its writes runs as a no-op, the
    #: admission follows, and the replay is the step.
    gated = _arena(_WIDTHS, _BC, BH)
    _absent_one_written_atom(fixture, config, gated)
    gate = torch.ones(1, dtype=torch.int32, device="cuda")
    _step(fixture, config, gated, gate=gate)
    gated.plan_exact(bitmap, fresh=False)
    gated.wait()
    gate.zero_()
    y_gated = _step(fixture, config, gated, gate=gate)

    assert torch.equal(y_reference, y_gated)
    assert torch.equal(_gathered(reference, BH), _gathered(gated, BH))


def _gathered(arena, BH):
    """`[BH, N, cols]` from the arena's slots -- what the two backings can both hold."""
    table = arena.page_table
    N = table.shape[1] * MMA_K_QUANTUM
    out = torch.zeros(BH, table.shape[1], MMA_K_QUANTUM, _COLS, device="cuda")
    present = table >= 0
    out[present] = arena.state[table[present].long()]
    return out.reshape(BH, N, _COLS)


# ---------------------------------------------------------------------------
# 4-5. capture
# ---------------------------------------------------------------------------

_HIDDEN, _H, _T, _B = 64, 2, 128, 2


def _sequence():
    """A prefilled paged state, its producer and a stream of decode tokens."""
    import rola
    from rola.routing.producer import union_routing
    from tests.conftest import build_routes

    torch.manual_seed(7)
    producer = build_routes(hidden_size=_HIDDEN, num_heads=_H, widths=_WIDTHS,
                           routing=union_routing(1)).cuda()
    torch.manual_seed(507)
    x = torch.randn(_B, _T + 8, _HIDDEN, device="cuda")
    v = torch.randn(_B, _T + 8, _H, _DV, device="cuda")
    with torch.no_grad():
        state = rola.state()
        _y, state = rola.rola_op(producer(x[:, :_T]), v[:, :_T], state=state)
    return producer, x, v, state


def _static(routes, v):
    """A bundle over buffers that never move -- what capture requires of its inputs."""
    static = RouteFactors(
        topology=routes.topology, normalization=routes.normalization,
        read=tuple(t.clone() for t in routes.read), write=tuple(t.clone() for t in routes.write),
        g_read=None, g_write=routes.g_write.clone(),
        read_mass=None if routes.read_mass is None else routes.read_mass.clone())
    return static, v.clone()


def _fill(static, static_v, routes, v):
    for buffer, level in zip(static.read, routes.read):
        buffer.copy_(level)
    for buffer, level in zip(static.write, routes.write):
        buffer.copy_(level)
    static.g_write.copy_(routes.g_write)
    if static.read_mass is not None:
        static.read_mass.copy_(routes.read_mass)
    static_v.copy_(v)


def _capture(static, static_v, state):
    """Warm up on a side stream, then capture ONE gated step. Returns `(graph, y)`."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            decode_step_gated(static, static_v, state)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y, _ = decode_step_gated(static, static_v, state)
    return graph, y


@torch.no_grad()
def test_the_captured_step_is_the_eager_step_bit_for_bit():
    """Same kernels, same order, same bytes -- the capture owes exactly this."""
    producer, x, v, state = _sequence()
    token = slice(_T + 4, _T + 5)
    routes = producer(x[:, token])
    static, static_v = _static(routes, v[:, token])
    _fill(static, static_v, routes, v[:, token])
    graph, y_graph = _capture(static, static_v, state)

    #: Make the token's write set resident FIRST, so neither run is the vacuous no-op
    #: cell 5 is about.
    decode_step_gated(static, static_v, state)
    if growth_pending(state):
        admit_growth(state)
    plane = state._arena.state
    snapshot = plane.clone()

    graph.replay()
    torch.cuda.synchronize()
    captured_y, captured_plane = y_graph.clone(), plane.clone()
    assert not torch.equal(captured_plane, snapshot), "the compared step did nothing"

    plane.copy_(snapshot)
    eager_y, _ = decode_step_gated(static, static_v, state)
    torch.cuda.synchronize()
    assert torch.equal(captured_y, eager_y)
    assert torch.equal(captured_plane, plane)


@torch.no_grad()
def test_a_growth_step_lands_under_the_graph_by_replay():
    """The replay is SELF-CORRECTING: no-op, admit, replay, and it is the eager answer."""
    producer, x, v, state = _sequence()
    token = slice(_T + 5, _T + 6)
    routes = producer(x[:, token])
    static, static_v = _static(routes, v[:, token])
    _fill(static, static_v, routes, v[:, token])
    graph, y_graph = _capture(static, static_v, state)

    #: A 128-token prefill over 16 atoms leaves nothing absent, so the growth is MADE:
    #: one atom this token writes is dropped from the plan. Re-planning also reassigns
    #: slots, which is the stronger cell -- the captured graph holds the arena's BASE
    #: and its table, and neither moves.
    written = _write_atom_bitmap(static.write, _config(_WIDTHS, dict(B=_B, H=_H)))
    keep = (state._arena.page_table >= 0) & written
    row, column = (keep.nonzero())[0].tolist()
    keep[row, column] = False
    state._arena.plan_exact(keep, fresh=True)
    state._arena.wait()
    plane = state._arena.state
    snapshot = plane.clone()

    graph.replay()
    assert growth_pending(state), "the fixture's growth token is already resident"
    assert torch.equal(plane, snapshot), "a growth step under the graph wrote state"

    #: The admission ZERO-INITS what it commits, so the comparison's common starting
    #: point is the plane AFTER it -- restoring the pre-admission bytes would hand the
    #: eager run the evicted page's stale contents.
    admit_growth(state)
    admitted = plane.clone()
    graph.replay()
    torch.cuda.synchronize()
    assert not growth_pending(state), "the replay did not clear the verdict"
    landed_y, landed_plane = y_graph.clone(), plane.clone()

    plane.copy_(admitted)
    eager_y, _ = decode_step_gated(static, static_v, state)
    torch.cuda.synchronize()
    assert torch.equal(landed_y, eager_y)
    assert torch.equal(landed_plane, plane)
