# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The carry's cells: alternating bundles on the even box, shared by its fold and readout gates.

Symbols: ``L`` tokens, ``B_l`` level ``l``'s width, ``d_v`` the value width.
"""
from __future__ import annotations

import torch

from rola.routing.factors import RouteFactors
from rola.routing.types import OpaqueRouting, Topology
from rola.values import Values


def levels(widths, L, live, *, B=1, H=2, seed=0):
    """Each level's rows on the simplex: ``live[l]`` digits a token at random (the rest exact zeros), positive."""
    gen = torch.Generator().manual_seed(seed)
    out = []
    for width, k in zip(widths, live):
        x = torch.rand((B, L, H, width), generator=gen) + 0.05
        if k < width:
            keep = torch.rand((B, L, H, width), generator=gen).argsort(-1) < k
            x = x * keep
        out.append((x / x.sum(-1, keepdim=True)).cuda())
    return out


def alternating_bundle(widths, L, write_sparse, *, live=4, d_v=64, seed=0):
    """An alternating bundle: the write side sparse at ``write_sparse`` (``live`` digits a token), the read side sparse
    at the other level, each side dense elsewhere."""
    D = len(widths)
    write_flags = tuple(level == write_sparse for level in range(D))
    read_flags = tuple(not flag for flag in write_flags)
    live_w = [live if flag else width for flag, width in zip(write_flags, widths)]
    live_r = [live if flag else width for flag, width in zip(read_flags, widths)]
    gen = torch.Generator().manual_seed(seed + 1)
    topology = Topology(levels=tuple(OpaqueRouting(width=w) for w in widths))
    routes = RouteFactors.from_levels(topology, levels(widths, L, live_r, seed=seed),
                                      levels(widths, L, live_w, seed=seed + 7),
                                      (torch.rand((1, L, 2), generator=gen) + 0.5).cuda(),
                                      sparse=(read_flags, write_flags))
    values = Values.from_projection(torch.randn((1, L, 2, d_v), generator=gen).cuda())
    return routes, values


#: THE PLAN'S FORMS, each exercised by a cell -- its widths, live digits a token, the form it forces (None: the rule's)
#: -- and named as `tests/oracle/carry_plan.forms` names them: each image mode, alone and over read chunks, the fold in
#: chunks, an entry filled ahead and one filled at its own step, a window's first filled by the window before, and a
#: fill that waits on the retired read's drain (docs/internals/carry/schedule.md#planner).
FORM_CELLS = (
    ((64, 64), 4, None, {"halves", "halves-chunks", "ahead", "deferred", "reuse"}),
    ((64, 64), 16, None, {"fold-chunks", "next"}),
    ((256, 256), 1, None, {"whole", "next"}),
    ((256, 256), 4, "whole/64/rule", {"whole-chunks"}),
    ((256, 256), 2, "halves/128/128", {"halves-chunks", "fold-chunks"}),
)
