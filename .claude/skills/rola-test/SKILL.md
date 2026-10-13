---
name: rola-test
description: Load before running or adding tests: scoped runs only, the oracle as the spec, tolerance discipline.
---
# Testing

1. Never a bare `pytest tests/`: name the files or nodes (`docs/testing.md#running-it`). GPU entry points take the GPU
   lock themselves; run them bare (`CLAUDE.md`).
2. The oracle is the spec (`docs/testing.md#the-oracle-tier--semantics`); changing it follows the dual-run protocol
   (`docs/testing.md#changing-the-oracle-the-dual-run-protocol`).
3. Tolerances: `docs/testing.md#tolerance-discipline`.
