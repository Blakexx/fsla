"""The measurement RECORD and the judgement over it.

Three modules, because measuring has three jobs and only one of them is a fixture:

* :mod:`bench.ledger` -- the record. `perf/ledger.jsonl`: one row per
  ``(cell, arm)`` per commit, the RAW samples kept with the instrument that took
  them and the provenance that makes a row replayable.
* :mod:`bench.stats` -- the statistics. The paired signed-rank verdict and its
  committed manifest, so a claim of "faster" is a test and not a pair of medians.
* :mod:`bench.regression` -- the judgement. Per-cell thresholds derived from that
  cell's own measured IQR, and the clock-settling precondition a comparable number
  requires.

WHAT is measured is not here: the fixtures and the timing driver belong to the arm
being measured, and the chunk arm's harness is its own card.
"""
