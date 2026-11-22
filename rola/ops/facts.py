# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE FACTS PASS, as a caller reaches it: no facts kernel is built on this line, so :func:`facts` raises
:class:`rola.ops._ext.KernelNotBuilt`. What stands is the shape law the engine reads: the layout and the two sides' static
operands a bundle's topology implies (:func:`call_layout`, :func:`call_statics`).
"""
from __future__ import annotations

from rola.engine.facts.liveness import LivenessLayout, SideStatics, side_statics
from rola.ops._ext import not_built

__all__ = ["call_layout", "call_statics", "facts"]


def call_layout(routes) -> LivenessLayout:
    """The layout a bundle's words are read in: every level at its padded width, the call's tokens."""
    widths = routes.topology.padded_widths
    return LivenessLayout(D=len(widths), B=widths, L=routes.tokens)


def call_statics(routes, layout: LivenessLayout) -> tuple[SideStatics, SideStatics]:
    """``(read, write)``: each side's static operands as the bundle declares them -- a level not flagged ``sparse`` is
    DENSE (neither read nor written: its words are the width mask), and the rows' own widths are the logical ones."""
    return tuple(side_statics(layout, tuple(not flag for flag in flags), routes.topology.widths)
                 for flags in routes.sparse)


def facts(routes, values, layout: LivenessLayout, read_statics: SideStatics, write_statics: SideStatics) -> None:
    """THE PASS: not built on this line (:func:`rola.ops._ext.not_built`). The words, unions, columns and lists of a
    bundle are what the Hopper design reads in its gather pass; the layout and the pure-torch model of the words they
    are graded against stand (:mod:`rola.engine.facts.liveness`)."""
    not_built("facts")
