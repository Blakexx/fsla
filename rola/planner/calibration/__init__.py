"""Planner CALIBRATION and AUDIT (``THUNDER_RUN_SPEC.md`` §6 P4).

Three modules, one job each:

``census``
    The EXACT per-``(owner, tile)`` bit-word census. Relocated verbatim -- it was
    written dependency-free of the planner precisely so it could move.
``cost_model``
    ``plan_cost_per_token``, the four measured constants, ``TileUnionStats`` and the
    ``BT`` argmin. The FIT whose compiled output is
    ``rola.planner.select``'s rule table.
``auditor``
    Runs BOTH deciders on recorded cells and asserts agreement, or writes the
    disagreement ledger with modelled regret.

Nothing in ``rola/`` imports this package at module scope. ``rola.planner.schedule``
holds lazy call proxies for the names that moved, so a shipped install without the
bench fails at the call and not at ``import rola``.
"""
