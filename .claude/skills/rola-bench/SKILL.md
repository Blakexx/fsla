---
name: rola-bench
description: Load before measuring the shipped kernels' performance or reading the rola-results store: the declared measurement, its preconditions, its record and its verdict.
---
# Measuring the shipped kernels

1. The method and the preconditions the targets acquire themselves (GPU lock, clock lock, binary and extension
   identity): `docs/measurement.md#the-method`, `docs/measurement.md#preconditions-which-the-targets-acquire`; the
   protocol every comparison obeys: `docs/KERNEL_STANDARDS.md#s21`; the suite: `docs/KERNEL_STANDARDS.md#s22`.
2. Run the declared measurement, pruned by label with `--only` and `--skip` for the dev loop
   (`docs/measurement.md#the-build-everything-declared-a-run-pruned-by-label`).
3. Two builds (a change against its baseline): the baseline as a reference checkout, one interleaved session, never
   two runs and never a scratch timing loop (`docs/measurement.md#comparing-two-builds`, `docs/KERNEL_STANDARDS.md#s32`).
4. Read it from the store, never from a terminal: `python -m rola_results show|history|latest|instruments`
   (`docs/measurement.md#the-measurements-record-rola_results`); a regression is `python -m rola_results verdict`
   under the three gates (`docs/measurement.md#the-flagging-rule-three-gates-all-must-fire`).
