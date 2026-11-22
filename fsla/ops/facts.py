# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE FACTS PASS, as a caller reaches it: no facts kernel is built on this line, so :func:`facts` raises
:class:`fsla.ops._ext.KernelNotBuilt`. What stands is the shape law the engine reads: the layout a bundle's topology implies (:func:`call_layout`).
"""
from __future__ import annotations

from fsla.engine.facts.liveness import LivenessLayout
from fsla.ops._ext import not_built

__all__ = ["call_layout", "facts"]


def call_layout(routes) -> LivenessLayout:
    """The layout a bundle's words are read in: every level at its padded width, the call's tokens."""
    widths = routes.topology.padded_widths
    return LivenessLayout(D=len(widths), B=widths, L=routes.tokens)


def facts(masks, layout: LivenessLayout) -> None:
    """THE PASS: not built on this line (:func:`fsla.ops._ext.not_built`). The words, unions, columns and lists of a
    call's masks are what the Hopper design reads in its gather pass; the layout and the pure-torch model of the words they
    are graded against stand (:mod:`fsla.engine.facts.liveness`)."""
    not_built("facts")
