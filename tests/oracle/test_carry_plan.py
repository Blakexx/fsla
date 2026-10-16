"""Gate for THE PLAN PASS (csrc/rola/src/carry/schedule.cuh `window_plan_kernel`): every window's record for every
(batch-head, box class) is the plans' model's (`tests/oracle/carry_plan.py`), plan word and step words word for word,
on every form cell and under a spread of forced forms; and the rule the model states never places a fill over anything
live (docs/internals/carry/schedule.md#pass).

Symbols: ``W`` the window (512 tokens), a box class a pair of the two list levels' 32-digit groups, ``B`` the area's
bytes, ``n`` a window's entries.
"""
from __future__ import annotations

import random

import pytest
import torch

from rola.engine.carve import plan_carve
from rola.ops import carry as carry_ops
from rola.ops import facts as facts_ops
from tests.oracle import carry_plan as cp
from tests.oracle.carry_fixtures import FORM_CELLS, alternating_bundle

#: the forced forms gated: none, then each image mode and each chunk choice at least once
FORCED = (None, "whole/union/union", "halves/64/128", "halves/union/64", "whole/128/192", "rule/64/rule",
          "halves/rule/union")


@pytest.mark.parametrize("forced", FORCED)
@pytest.mark.parametrize(("widths", "live"), sorted({(cell[0], cell[1]) for cell in FORM_CELLS}))
def test_the_plans_are_the_model(widths, live, forced):
    """Each window's plan word (its mode, chunks and their counts) and every step word (the entry's place, filled ahead,
    the next filled now, the drain's rendezvous) as the model plans them -- so the forms the carry's gates assert by
    are the kernel's."""
    routes, values = alternating_bundle(widths, 2048, 0, live=live)
    layout = facts_ops.call_layout(routes)
    _assert_plans_are_the_model(routes, values, layout, facts_ops.call_statics(routes, layout), forced)


def _assert_plans_are_the_model(routes, values, layout, statics, forced=None):
    """The plan pass's written words for every (batch-head, box class, window) against the model's; returns the model's
    records."""
    carve = plan_carve(tuple(routes.topology.padded_widths), *routes.sparse, values.v.shape[-1])
    facts = facts_ops.facts(routes, values, layout, *statics, carry_ops.form(carry_ops.arm_key(carve)).read_row)
    plans = carry_ops.plans_out(routes, carve)
    unit = torch.zeros((routes.batch * routes.heads, routes.tokens), dtype=torch.int8, device="cuda")
    carry_ops.carry(routes, values, facts, carve, unit, forced=forced, plans=plans)
    torch.cuda.synchronize()
    words = plans.cpu().view(torch.int32).long() & 0xFFFFFFFF
    want = cp.plans(routes, facts, carve, forced)
    classes = plans.shape[2]
    for box, windows in enumerate(want):
        bh, cls = divmod(box, classes)
        for w, rec in enumerate(windows):
            expect = rec.words()
            assert tuple(words[bh, w, cls, :len(expect)].tolist()) == expect, (box, w)
    return want


def test_a_window_that_reads_nothing_is_planned_as_the_model():
    """A window whose box reads nothing (a cohort draw, whose groups go unread for whole windows) before one that reads:
    the records are the model's. Its successor's first read is filled in one of its steps or at its own, all of which
    run past the store, whose work stages the inputs the fill reads (docs/internals/carry/schedule.md#steps); the
    successor's readout is held to the oracle by `tests/oracle/test_carry_vs_oracle.py`."""
    from measure.cells import by_name, realize
    from measure.subjects import bundle_of

    spec = by_name("flagship-cohort-k4-L4096")
    routes, values, layout, statics = bundle_of(spec, realize(spec))
    want = _assert_plans_are_the_model(routes, values, layout, statics)
    nothing_then_reads = sum(int(cp.chunks(windows[w - 1].word)[3] == 0 and cp.chunks(windows[w].word)[3] > 0)
                             for windows in want for w in range(1, len(windows)))
    assert nothing_then_reads > 0, "no window reading nothing precedes one that reads: the cell is vacuous"


def test_grouped_plans_are_the_model():
    """GROUPED QUERIES (four query heads a write head): the read union is the group's merged rows, so a window's read
    chunks, its records' size and the area are the group's (`tables_of`); the records are the model's word for word."""
    from measure.cells import by_name, realize
    from measure.subjects import bundle_of

    spec = by_name("flagship-alt-k4-g4")
    routes, values, layout, statics = bundle_of(spec, realize(spec))
    assert routes.group == 4
    want = _assert_plans_are_the_model(routes, values, layout, statics)
    assert max(cp.chunks(rec.word)[3] for windows in want for rec in windows) > 1, (
        "no window reads more than one chunk: the group's union never passes one head's buffer")


def _sides(rng: random.Random):
    """A random window's counts: each side's union (none, small or full) and segment sums of up to 32 lists a slot."""
    def side():
        u = rng.choice((0, rng.randrange(8, 128, 8), rng.randrange(128, 512, 8), 512))
        density = rng.choice((1, 4, 16, 32))
        totals, run = [], 0
        for j in range(cp.LIST_AT):
            slots = max(0, min(u, cp.LIST_BOUND * (j + 1)) - cp.LIST_BOUND * j)
            run += rng.randint(0, density * slots)
            totals.append(run)
        return u, tuple(totals)
    (ru, rs), (fu, fs) = side(), side()
    return cp.Counts(ru, rs, fu, fs)


@pytest.mark.parametrize("seed", range(4))
def test_the_rule_never_fills_over_what_is_live(seed):
    """THE RULE'S SAFETY on random windows of this card's arm: every entry inside the area and every read above its
    image; at each step the entry computing, the entry filled ahead beside it and the live image pairwise clear; a
    deferred fill clear of the image where it is live; the next window's first clear of the last entry and above its own
    image. A fill the rule cannot fit beside the live entry waits for its own step -- never a byte shared."""
    key = (32, 1, 32, 1, 32, 0)
    row = next(r for r in carry_ops.census() if tuple(r[1:7]) == key)
    area, read_row, write_row = row[12], row[13], row[14]
    rng = random.Random(seed)
    wins = [cp.win_of(_sides(rng), -1, read_row, write_row, area) for _ in range(64)]
    recs = cp.records(wins, read_row, write_row, area)
    for w, (x, rec) in enumerate(zip(wins, recs)):
        for t in range(x.n):
            at, ahead, nxt, _ = rec.steps[t]
            b = cp.entry_bytes(x, t, read_row, write_row)
            read = t < x.nr
            assert at >= 0 and at + b <= area, (w, t)
            if read:
                assert at >= x.image, (w, t, "a read under the image")
            if nxt:
                if t + 1 < x.n:
                    nat, nb = rec.steps[t + 1][0], cp.entry_bytes(x, t + 1, read_row, write_row)
                else:
                    nx = wins[w + 1]
                    nat, nb = recs[w + 1].steps[0][0], cp.entry_bytes(nx, 0, read_row, write_row)
                    assert recs[w + 1].steps[0][1] == 1
                    assert nat >= nx.image, (w, t, "the next window's first under its image")
                assert cp.disjoint(at, b, nat, nb), (w, t, "a fill over the computing entry")
                if read:
                    assert nat >= x.image, (w, t, "a fill over the live image")
            if ahead and t > 0:
                assert rec.steps[t - 1][2] == 1, (w, t, "filled ahead by a step that did not fill it")
