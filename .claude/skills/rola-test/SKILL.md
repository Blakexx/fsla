---
name: rola-test
description: Load before running or adding tests, or changing the oracle or a reference (rola/ops/naive.py, tests/oracle/*reference*.py, tolerances.py): scoped runs, the tiers, the dual-run protocol, tolerance discipline, the sanitizer lane.
---
# Testing

1. Run by tier, `bazel test //tests:<tier>`, a few of its tests by `--test_filter` (pytest's `-k`); GPU entry points
   lock themselves and run bare (`docs/testing.md#running-it`). New device code's first run is under `timeout -s KILL`
   (`docs/KERNEL_STANDARDS.md#first-run`).
2. The oracle is the spec (`docs/testing.md#the-oracle-tier--semantics`). A change to it or to a reference is proven by
   running both versions (`docs/testing.md#changing-the-oracle-the-dual-run-protocol`), never by reading the diff.
3. A kernel is graded per output slot under its kind's tolerance (`docs/testing.md#tolerance-discipline`,
   `tests/oracle/tolerances.py`); its test is the `oracle` pointer in `tools/coverage.json`
   (`docs/internals/tools/coverage.md#declaration`).
4. The sanitizer lane: `bazel test //tests:sanitizer` (`docs/testing.md#the-sanitizer-lane`).
