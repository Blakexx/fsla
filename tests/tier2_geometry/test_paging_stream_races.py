"""H4 (coverage-closure): STREAM-DISCIPLINE spies on the paged consumer path.

The postmerge phase found and fixed two stream races in the PLANNED-EXACT-ASYNC
arena (`PageArena.wait()` never called before the launch; `atom_ids` crossing to
the side stream with no lifetime contract) — both found the expensive way, as an
866-failure CUDA-context poisoning that only reproduced with a per-launch `BC`
cut-over's timing. Nothing gated the DISCIPLINE itself: the fixes could regress
and no test would say so until the timing happened to expose them again. These
spies make the ordering and lifetime contracts DETERMINISTIC test failures
instead of timing-dependent context poisonings.

**THE SPIES THEMSELVES LIVE IN `stream_discipline.py`.** This file INSTANTIATES
the fixture for the paged path; it owns the cell, not the instruments. The
checks below are what any stream-crossing path owes, and copying them into each
new path is how three of four end up missing. The fixture's own teeth — each
check run against a planted violation, plus the CUDA-graph capture detector
proved in both directions — are in `test_stream_discipline_fixture.py`.

**P67 D2-b re-anchored the cell onto the CHUNK arm, and onto the FACADE.** The
spies used to watch the retired tiled launch, driven through a private entry with
a hand-built launch geometry; the arena, the side stream, `plan_exact` and `wait`
are all unchanged, so the discipline claim survives its executor whole and only the
launch it brackets moves. It is now driven through `rola_op(state=rola.state())` — the production
sequence — rather than through a private entry, which makes the ordering claim
one about what SHIPS: the facade runs the stats pass, `RoLAState._kernel_entry`
plans the arena on its side stream, and `rola.ops.chunk.run` issues the single
`wait()` that puts the admission before the kernel.

Three cells:

(a) ORDERING — the observed sequence must be `plan_exact -> wait -> kernel
    launch`; the wait is the one event-join and it must sit between the
    side-stream publication and the consumer.
(b) LIFETIME — `atom_ids` crossing to the side stream must be `record_stream`-ed
    on that stream (the documented cross-stream lifetime contract; its absence
    WAS the measured `index_copy_(): index out of bounds`).
(c) ALLOCATOR-REUSE PRESSURE — an alloc/free burst on the caller's stream
    between `plan_exact` and the launch, the exact window a lifetime bug gives
    the caching allocator to hand the planned tensors' bytes to someone else.
    Run normally AND in a `CUDA_LAUNCH_BLOCKING=1` child (which serializes the
    streams), REQUIRING AGREEMENT: the state is atomic-free and deterministic,
    so both runs must be bit-identical to the unpaged dense state.

A fourth cell retired with its subject: `PageArena.adopt_dense`'s join against a
pending side-stream publication. `adopt_dense` commits the dense limit and has
had no production caller since the state facade landed (`RoLAState`'s
`adopt_dense_contents` says in prose why it deliberately does not use it), and
it is deleted with the tiled consumer. The missing-join CLASS it guarded is
still gated by cell (a), at the seam that does ship.

Symbols: `BC` leaves per owner (the built arm's, not a host choice);
`cols = d_v + 1` (`normalization` is always `'global'`); **atom** = the 16-leaf
page granule.
"""

from __future__ import annotations

import pytest
import torch

import rola
import rola.ops.chunk as chunk_mod
from rola.expert import PlanOverrides
from rola.ops.paging import PageArena
from tests.tier2_geometry.stream_discipline import (
    OrderingSpy,
    allocator_pressure,
    assert_agrees_under_launch_blocking,
    assert_recorded_on,
    record_stream_spy,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="paging is a CUDA feature"),
]

#: A built arm (`D = 2, B = 16`) at a chunk-multiple `T`, driven through the real
#: gateway. MEASURED: at this cell the realized entmax write side admits EVERY
#: atom — unstructured sparsity at any real `T` reaches the whole leaf space, and
#: only a structurally confined write support produces absent atoms. Nothing here
#: depends on partial residency: the ordering, lifetime and allocator-reuse claims
#: are about the side stream and the admission itself, both of which happen in
#: full. Partial-residency ADDRESSING is `test_chunk_paging_equivalence.py`'s
#: subject, and it confines its supports on purpose to get there.
_WIDTHS, _DV, _T, _H = (16, 16), 64, 64, 2


def _cell():
    """`(routes, v)` from the real gateway — `rola_op` refuses any other bundle."""
    from rola import sparse_routing
    from tests.conftest import build_routes

    torch.manual_seed(61)
    producer = build_routes(hidden_size=64, num_heads=_H, d_v=_DV, widths=_WIDTHS,
                            routing=sparse_routing(1.5), normalization="global").cuda()
    gen = torch.Generator(device="cuda").manual_seed(62)
    x = torch.randn(1, _T, 64, device="cuda", generator=gen)
    v = torch.randn(1, _T, _H, _DV, device="cuda", generator=gen)
    with torch.no_grad():
        routes = producer(x)
    return routes, v


def _run(routes, v, *, paging: bool):
    """One `rola_op` call carrying a state, at the requested backing."""
    st = rola.state()
    with torch.no_grad():
        _y, st = rola.rola_op(routes, v, state=st, _arm="kernel",
                              expert=PlanOverrides() if paging
                              else PlanOverrides(paging=False))
    return st


# ---------------------------------------------------------------------------
# (a) plan_exact -> wait -> launch
# ---------------------------------------------------------------------------

def test_the_paged_path_orders_plan_exact_then_wait_then_launch(monkeypatch):
    """The publication must be JOINED before the consumer reads it.

    The spy is `stream_discipline.OrderingSpy`, not a private one: this is the
    check every stream-crossing path owes, and any nested `wait` is recorded as
    `wait_inner` so it cannot stand in for the pre-launch one.
    """
    routes, v = _cell()
    spy = OrderingSpy()
    spy.watch(monkeypatch, PageArena, "plan_exact", label="plan_exact")
    spy.watch(monkeypatch, PageArena, "wait", label="wait", nested_label="wait_inner")
    #: THE MODULE THAT BINDS THE NAME, not `rola.ops._ext`. `rola/ops/chunk.py`
    #: does `from rola.ops._ext import extension`, so the callable it reaches is
    #: bound at ITS import; patching the source module only works when the
    #: importer has not been imported yet, which is a property of test ordering
    #: and not of the code under test.
    spy.watch_extension(monkeypatch, chunk_mod, "chunk_forward", label="launch")

    _run(routes, v, paging=True)

    spy.assert_between("plan_exact", "wait", "launch")


# ---------------------------------------------------------------------------
# (b) atom_ids records its stream
# ---------------------------------------------------------------------------

def test_atom_ids_crossing_to_the_side_stream_is_record_streamed(monkeypatch):
    """`atom_ids` crosses to the arena's side stream, so it must declare its
    lifetime there. Its absence WAS the measured `index_copy_(): index out of
    bounds`."""
    routes, v = _cell()
    admitted: list[int] = []
    arenas: list[PageArena] = []
    orig_admit = PageArena._admit_pages

    def spy_admit(self, atom_ids, first_slot):
        admitted.append(id(atom_ids))
        arenas.append(self)
        return orig_admit(self, atom_ids, first_slot)

    monkeypatch.setattr(PageArena, "_admit_pages", spy_admit)

    with record_stream_spy(monkeypatch) as recorded:
        _run(routes, v, paging=True)

    assert admitted, "the fixture admitted no atoms; it gates nothing"
    assert_recorded_on(recorded, admitted, arenas[0]._stream, what="atom_ids")


# ---------------------------------------------------------------------------
# (c) allocator-reuse pressure between plan_exact and the launch
# ---------------------------------------------------------------------------

def _pressured_state_and_dense(seed_burst=True):
    """`(dense state, paged state)` with an alloc/free burst wedged into the
    window between the admission and the launch."""
    routes, v = _cell()
    orig_plan_exact = PageArena.plan_exact

    def pressured_plan_exact(self, bitmap, *, fresh=True):
        result = orig_plan_exact(self, bitmap, fresh=fresh)
        if seed_burst:
            allocator_pressure()
        return result

    dense = _run(routes, v, paging=False).materialize()
    try:
        PageArena.plan_exact = pressured_plan_exact
        paged = _run(routes, v, paging=True).materialize()
    finally:
        PageArena.plan_exact = orig_plan_exact
    return dense, paged


def test_allocator_reuse_pressure_does_not_move_the_paged_state():
    dense, paged = _pressured_state_and_dense()
    assert torch.equal(dense, paged), (
        "an alloc/free burst between plan_exact and the launch changed the paged "
        "state: a planned tensor's bytes were recycled before the side stream was "
        "done with them (stream-lifetime regression)")


def test_allocator_reuse_pressure_agrees_with_a_launch_blocking_child():
    """The same cell under CUDA_LAUNCH_BLOCKING=1, REQUIRING AGREEMENT.

    Blocking serializes the streams, so a run that only passes blocked (or only
    passes unblocked) is a stream race by signature — exactly how the postmerge
    defect was diagnosed. The state is deterministic, so agreement is bit-level.
    """
    assert_agrees_under_launch_blocking("""
        import torch
        torch.cuda.set_per_process_memory_fraction(0.65)
        from tests.tier2_geometry.test_paging_stream_races import _pressured_state_and_dense
        dense, paged = _pressured_state_and_dense()
        assert torch.equal(dense, paged), 'blocked run: paged state moved'
        print('OK')
    """)
