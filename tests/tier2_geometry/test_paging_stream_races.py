"""H4 (coverage-closure): STREAM-DISCIPLINE spies on the paged consumer path.

The postmerge phase found and fixed two stream races in Track P's
PLANNED-EXACT-ASYNC arena (`PageArena.wait()` never called before the launch;
`atom_ids` crossing to the side stream with no lifetime contract) -- both found
the expensive way, as an 866-failure CUDA-context poisoning that only reproduced
with the per-launch `BC` cut-over's timing. Nothing gated the DISCIPLINE itself:
the fixes could regress and no test would say so until the timing happened to
expose them again. These spies make the ordering and lifetime contracts
DETERMINISTIC test failures instead of timing-dependent context poisonings.

**THE SPIES THEMSELVES LIVE IN `stream_discipline.py`.** This file INSTANTIATES the
fixture for the paged path; it owns the cell, not the instruments. The four checks
below are what any stream-crossing path owes, and copying them into each new path is
how three of the four end up missing. The fixture's own teeth -- each check run
against a planted violation, plus the CUDA-graph capture detector proved in both
directions -- are in `test_stream_discipline_fixture.py`.

Four cells:

(a) ORDERING -- on the paged path the observed sequence must be
    `plan_exact -> wait -> kernel launch`; the wait is §4.5's one event-join and
    it must sit between the side-stream publication and the consumer.
(b) LIFETIME -- `atom_ids` crossing to the side stream must be
    `record_stream`-ed on that stream (the documented cross-stream lifetime
    contract; its absence WAS the measured `index_copy_(): index out of bounds`).
(c) ALLOCATOR-REUSE PRESSURE -- an alloc/free burst on the caller's stream
    between `plan_exact` and the launch, the exact window a lifetime bug gives
    the caching allocator to hand the planned tensors' bytes to someone else.
    Run normally AND in a `CUDA_LAUNCH_BLOCKING=1` child (which serializes the
    streams), REQUIRING AGREEMENT: the state is atomic-free and deterministic,
    so both runs must be bit-identical to the unpaged dense state.
(d) THE `adopt_dense` SEAM -- audit finding: `adopt_dense` wrote the page table
    and value pool on the CALLER's stream with no join against side-stream work
    still in flight from a previous `plan_exact`/`reset` -- the same missing-join
    class the postmerge `wait()` fix closed at the launch seam. The spy asserts
    the join exists (deterministically, by observing `wait()` from inside
    `adopt_dense` while a prior publication is pending).

Symbols: `BC` leaves per owner; `cols = d_v + 1` (`normalization` is always
`'global'`); the fixture/plan/arena builders are `test_paging_equivalence`'s,
so these cells gate the same configurations that file proves bit-identical.
"""

from __future__ import annotations

import pytest
import torch

import rola.ops._ext as ext_mod
from rola.ops.paging import PageArena
from rola.planner.schedule import build_schedule
from tests.tier2_geometry.stream_discipline import (
    OrderingSpy,
    allocator_pressure,
    assert_agrees_under_launch_blocking,
    assert_recorded_on,
    record_stream_spy,
)
from tests.tier2_geometry.test_paging_equivalence import (
    _arena,
    _fixture,
    _plan,
    _reordering_bound,
    _run,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="V3 paging is a CUDA feature"),
]

_WIDTHS, _DV, _BC = (4, 8, 8), 32, 16


def _cell():
    plan = _plan(_WIDTHS, _DV, BC=_BC, decay=False)
    fixture = _fixture(_WIDTHS, _DV, p_nonzero=0.4, seed=61)
    B, _T, H, _ = fixture[4].shape
    return plan, fixture, B * H


# ---------------------------------------------------------------------------
# (a) plan_exact -> wait -> launch
# ---------------------------------------------------------------------------

def test_the_paged_path_orders_plan_exact_then_wait_then_launch(monkeypatch):
    """The publication must be JOINED before the consumer reads it.

    The spy is `stream_discipline.OrderingSpy`, not a private one: this is the check
    every stream-crossing path owes, and `reset()`'s own join is recorded as
    `wait_inner` so it cannot stand in for the pre-launch one.
    """
    plan, fixture, BH = _cell()
    spy = OrderingSpy()
    spy.watch(monkeypatch, PageArena, "plan_exact", label="plan_exact")
    spy.watch(monkeypatch, PageArena, "wait", label="wait", nested_label="wait_inner")
    spy.watch_extension(monkeypatch, ext_mod, "rola_consumer_forward", label="launch")

    arena = _arena(plan, BH=BH)
    _run(plan, fixture, arena=arena)

    spy.assert_between("plan_exact", "wait", "launch")


# ---------------------------------------------------------------------------
# (b) atom_ids records its stream
# ---------------------------------------------------------------------------

def test_atom_ids_crossing_to_the_side_stream_is_record_streamed(monkeypatch):
    """`atom_ids` crosses to the arena's side stream, so it must declare its lifetime
    there. Its absence WAS the measured `index_copy_(): index out of bounds`."""
    plan, fixture, BH = _cell()
    admitted: list[int] = []
    orig_admit = PageArena._admit_pages

    def spy_admit(self, atom_ids, first_slot):
        admitted.append(id(atom_ids))
        return orig_admit(self, atom_ids, first_slot)

    monkeypatch.setattr(PageArena, "_admit_pages", spy_admit)

    with record_stream_spy(monkeypatch) as recorded:
        arena = _arena(plan, BH=BH)
        _run(plan, fixture, arena=arena)

    assert admitted, "the fixture admitted no atoms; it gates nothing"
    assert_recorded_on(recorded, admitted, arena._stream, what="atom_ids")


# ---------------------------------------------------------------------------
# (c) allocator-reuse pressure between plan_exact and the launch
# ---------------------------------------------------------------------------

def _pressured_state_and_dense(seed_burst=True):
    plan, fixture, BH = _cell()
    read, write, g_read, g_write, v = fixture

    orig_plan = PageArena.plan

    def pressured_plan(self, schedule, *, fresh=True):
        result = orig_plan(self, schedule, fresh=fresh)
        if seed_burst:
            allocator_pressure()
        return result

    _dense_y, dense_state, _ = _run(plan, fixture)
    try:
        PageArena.plan = pressured_plan
        arena = _arena(plan, BH=BH)
        y_paged, returned, _ = _run(plan, fixture, arena=arena)
    finally:
        PageArena.plan = orig_plan
    cols = dense_state.shape[-1]
    B, _T, H, _ = v.shape
    return (dense_state.reshape(B * H, -1, cols), arena.materialize(),
            y_paged, _dense_y, plan)


def test_allocator_reuse_pressure_does_not_move_the_paged_state():
    dense, paged, y_paged, y_dense, plan = _pressured_state_and_dense()
    assert torch.equal(dense, paged), (
        "an alloc/free burst between plan_exact and the launch changed the paged "
        "state: a planned tensor's bytes were recycled before the side stream was "
        "done with them (stream-lifetime regression)")
    bound = _reordering_bound(y_dense, plan)
    assert (y_dense - y_paged).abs().max() <= bound


def test_allocator_reuse_pressure_agrees_with_a_launch_blocking_child():
    """The same cell under CUDA_LAUNCH_BLOCKING=1, REQUIRING AGREEMENT.

    Blocking serializes the streams, so a run that only passes blocked (or only
    passes unblocked) is a stream race by signature -- exactly how the postmerge
    defect was diagnosed. The state is deterministic, so agreement is bit-level.
    """
    assert_agrees_under_launch_blocking("""
        import torch
        torch.cuda.set_per_process_memory_fraction(0.65)
        from tests.tier2_geometry.test_paging_stream_races import _pressured_state_and_dense
        dense, paged, _y, _yd, _p = _pressured_state_and_dense()
        assert torch.equal(dense, paged), 'blocked run: paged state moved'
        print('OK')
    """)


# ---------------------------------------------------------------------------
# (d) the adopt_dense seam
# ---------------------------------------------------------------------------

def test_adopt_dense_joins_the_side_stream_before_writing(monkeypatch):
    """`adopt_dense` writes the page table and value pool on the CALLER's stream.

    If side-stream work from a previous `plan_exact` is still in flight (its
    zero-init of the same `_value_base`), those writes race it -- the identical
    missing-join class `_consumer_forward`'s `paging.wait()` fix closed at the
    launch seam. The join must be observed WHILE a publication is pending, so the
    cell plans first (recording `_ready` on the side stream) and then adopts.
    """
    plan, fixture, BH = _cell()
    read, write, g_read, g_write, v = fixture
    schedule = build_schedule(read, write, g_read, g_write, plan)

    arena = _arena(plan, BH=BH)
    arena.plan(schedule)  # side-stream publication now pending (_ready recorded)
    assert arena._ready is not None

    waited: list[bool] = []
    orig_wait = PageArena.wait

    def spy_wait(self):
        waited.append(True)
        return orig_wait(self)

    monkeypatch.setattr(PageArena, "wait", spy_wait)
    carried = torch.randn(BH, arena.N, arena.cols, device="cuda")
    arena.adopt_dense(carried)

    assert waited, (
        "adopt_dense wrote the page table and value pool on the caller's stream "
        "with NO join against the pending side-stream publication -- the same "
        "missing-wait class the postmerge fix closed before the consumer launch")
    assert torch.equal(arena.materialize(), carried)
