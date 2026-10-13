---
name: rola-bench
description: Load before measuring the shipped kernels' performance: the declared measurement, its protocol, its record.
---
# Measuring the shipped kernels

1. The method and its preconditions: `docs/measurement.md#the-method` and
   `docs/measurement.md#preconditions-which-the-targets-acquire`; the protocol `docs/KERNEL_STANDARDS.md#s21`; the
   suite `docs/KERNEL_STANDARDS.md#s22`.
2. Run the declared measurement (`docs/measurement.md#the-build-everything-declared-a-run-pruned-by-label`).
3. Read a verdict by the flagging rule (`docs/measurement.md#the-flagging-rule-three-gates-all-must-fire`); the
   records: `docs/measurement.md#the-measurements-record-rola_results`.
