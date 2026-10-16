"""Gate for THE PLAN PASS (csrc/rola/src/carry/schedule.cuh `window_plan_kernel`): every window's plan for every
(batch-head, box class) is the plans' model's (`tests/oracle/carry_plan.py`), head and items word for word, on every
form cell and under a spread of preferred candidates (docs/internals/carry/schedule.md#pass).

Symbols: ``W`` the window (512 tokens), a box class a pair of the two list levels' 32-digit groups.
"""
from __future__ import annotations

import pytest
import torch

from rola.engine.carve import plan_carve
from rola.ops import carry as carry_ops
from rola.ops import facts as facts_ops
from tests.oracle import carry_plan
from tests.oracle.carry_fixtures import FORM_CELLS, alternating_bundle

#: the preferred candidates gated: none, then each image mode, fold chunk and order at least once
PREFERRED = (None, "whole/union/pipelined", "ring8/128/pipelined-late", "ring16/64/serial", "ring8/union/serial-late",
             "ring16/192/pipelined", "whole/256/serial")


@pytest.mark.parametrize("prefer", PREFERRED)
@pytest.mark.parametrize(("widths", "live"), sorted({(cell[0], cell[1]) for cell in FORM_CELLS}))
def test_the_plans_are_the_model(widths, live, prefer):
    """Each window's head (its items, mode, chunks, unions, entries' offsets, cycles) and every item (kind, flags,
    chunk, dependence, buffer, hosted fills and their waits) as the model plans them -- so the forms the carry's gates
    assert by are the kernel's."""
    routes, values = alternating_bundle(widths, 2048, 0, live=live)
    layout = facts_ops.call_layout(routes)
    _assert_plans_are_the_model(routes, values, layout, facts_ops.call_statics(routes, layout), prefer)


def _assert_plans_are_the_model(routes, values, layout, statics, prefer=None):
    """The plan pass's words for every (batch-head, box class, window), head and items, against the model's; returns
    the model's plans."""
    carve = plan_carve(tuple(routes.topology.padded_widths), *routes.sparse, values.v.shape[-1])
    facts = facts_ops.facts(routes, values, layout, *statics, carry_ops.form(carry_ops.arm_key(carve)).read_row)
    plans = carry_ops.plans_out(routes, carve)
    unit = torch.zeros((routes.batch * routes.heads, routes.tokens), dtype=torch.int8, device="cuda")
    carry_ops.carry(routes, values, facts, carve, unit, prefer=prefer, plans=plans)
    torch.cuda.synchronize()
    words = plans.cpu().view(torch.int32).long() & 0xFFFFFFFF
    want = carry_plan.plans(routes, facts, carve, prefer)
    classes = plans.shape[2]
    for box, windows in enumerate(want):
        bh, cls = divmod(box, classes)
        for w, pl in enumerate(windows):
            got = words[bh, w, cls].tolist()
            n = len(pl.items)
            assert tuple(got[:4]) == pl.head, (box, w, "head")
            assert tuple(zip(got[4:4 + 2 * n:2], got[5:5 + 2 * n:2])) == pl.items, (box, w, "items")
    return want


def test_a_window_that_reads_nothing_hosts_the_next_read_past_the_store():
    """A window whose box reads nothing has no first-read region holding the next window's first read back, so its
    host is the store's bound: past the store, whose work stages the next window's inputs the fill reads (a cohort
    draw, whose groups go unread for whole windows; docs/internals/carry/schedule.md#planner). The plans are the
    model's, and every next-window fill sits past the store."""
    from measure.cells import by_name, realize
    from measure.subjects import bundle_of

    spec = by_name("flagship-cohort-k4-L4096")
    routes, values, layout, statics = bundle_of(spec, realize(spec))
    want = _assert_plans_are_the_model(routes, values, layout, statics)
    empty_then_read = 0
    for windows in want:
        for w, pl in enumerate(windows):
            hosts = [h for h, (x, y) in enumerate(pl.items)
                     if carry_plan.NEXT_ITEM in (carry_plan._bits(x, 25, 5), carry_plan._bits(y, 19, 5))]
            assert all(h > carry_plan.STORE_ITEM for h in hosts), (w, hosts)
            reads = carry_plan._bits(pl.head[1], 0, 10)
            empty_then_read += int(reads == 0 and bool(hosts))
    assert empty_then_read > 0, "no window reads nothing before one that reads: the cell is vacuous"
