"""The bench library: what fsla's kernels are timed as. The method, the statistics and the record are docs/measurement.md.

* :mod:`measure.subjects` -- the roster: one lean callable per kernel this line carries, over a registry cell.
* :mod:`measure.provider` -- the roster as timed arms for fsla-devtools' timing system, and the executors this tree's
  timing entries name (`measure/entries.bzl`).

The cells are fsla-devtools' central registry, read through `measure/cells`.
"""
