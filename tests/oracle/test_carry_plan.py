"""Gate for THE PLAN PASS (csrc/rola/src/carry/schedule.cuh `window_plan_kernel`): every window's plan for every
(batch-head, box class) is the plans' model's (`tests/oracle/carry_plan.py`), field for field, on every form cell and
every preferred candidate (docs/internals/carry/schedule.md#pass).

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


def _fields(plans: torch.Tensor) -> dict[str, torch.Tensor]:
    """The plan pass's plans ``[BH, windows, classes, PLAN_BYTES]`` u8 as each field's ``[BH, windows, classes]``."""
    words = plans.cpu().view(torch.int16).long() & 0xFFFF
    return {name: words[..., k] for k, name in enumerate(carry_ops.PLAN_FIELDS)}


@pytest.mark.parametrize("prefer", (None,) + carry_ops.READ_CANDIDATES)
@pytest.mark.parametrize(("widths", "live"), sorted({(cell[0], cell[1]) for cell in FORM_CELLS}))
def test_the_plans_are_the_model(widths, live, prefer):
    """Each window's readout form, buffers, unions, chunks, the fold beside or not, the zones and the first read
    buffer's end, as the model plans them -- the forms the carry's gates assert by are the kernel's."""
    routes, values = alternating_bundle(widths, 2048, 0, live=live)
    carve = plan_carve(tuple(routes.topology.padded_widths), *routes.sparse, values.v.shape[-1])
    layout = facts_ops.call_layout(routes)
    facts = facts_ops.facts(routes, values, layout, *facts_ops.call_statics(routes, layout),
                            carry_ops.form(carry_ops.arm_key(carve)).read_row)
    plans = carry_ops.plans_out(routes, carve)
    unit = torch.zeros((routes.batch * routes.heads, routes.tokens), dtype=torch.int8, device="cuda")
    carry_ops.carry(routes, values, facts, carve, unit, prefer=prefer, plans=plans)
    torch.cuda.synchronize()
    got = _fields(plans)
    want = carry_plan.plans(routes, facts, carve, prefer)
    classes = plans.shape[2]
    for box, windows in enumerate(want):
        bh, cls = divmod(box, classes)
        for w, pl in enumerate(windows):
            g = {name: int(v[bh, w, cls]) for name, v in got.items()}
            expect = {"form": (0, 1, 0, 2, 2)[pl.candidate], "rbufs": 1 if pl.candidate == 4 else 2,
                      "runion": pl.runion, "rchunk": pl.rchunk, "funion": pl.funion, "fchunk": pl.fchunk,
                      "beside": int(pl.beside), "rzone": pl.rzone // 16, "fzone": pl.fzone // 16,
                      "rbuf1": pl.rbuf_end // 16}
            assert {k: g[k] for k in expect} == expect, (box, w)
