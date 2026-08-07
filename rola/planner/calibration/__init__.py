"""Planner MEASUREMENT instruments -- exact statistics over a realized routing.

Two modules, one job each:

``census``
    The EXACT per-``(owner, tile)`` bit-word census.
``union_stats``
    ``TileUnionStats`` and the pass that produces it: the gathered extent, the
    candidate fraction and the k-quantum histogram of one realization.

Nothing in ``rola/`` imports this package at module scope, and nothing in the dispatch
path calls it: ``BC`` is DERIVED from launch-static facts
(:mod:`rola.planner.derive`), so no runtime rule reads a routing statistic. The
instruments exist for the bench cells, which report these numbers beside a measured
time. Keeping the edge out of the shipped import graph is what keeps ``import rola``
free of ``census``'s ``torch.compile``/inductor dependency.
"""
