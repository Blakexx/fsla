# `docs/architecture/` — the permanent decisions

This tree holds the decisions that outlive any particular kernel: the things a
contributor must know before changing code, and the reasons they were decided,
written for a reader who was not here. Nothing here is a status report. If a
statement is true only of the current implementation, it belongs in
[`docs/internals/`](../internals/README.md) — the per-source-file mirror, which
carries the reasons attached to a *line*. If it is the record of who measured
what on which day, it belongs in the campaign journal, which is deliberately out
of this tree (`rola-scratch/development/plans/PERF_CAMPAIGN_DOCKET.md`, an
append-only session log). These documents are the distillation of that journal:
the decision and its argument, with citations back to the journal entry, the
workflow report, the test, or the source line, so that a future reader can
**re-derive** rather than trust.

Everything here is falsifiable. Where the record is genuinely uncertain, the
document says so and names the measurement that would settle it.

| Document | What it settles |
|---|---|
| [self-masking-theorem.md](self-masking-theorem.md) | Dead routes are exact zeros and zeros are inert. The licence for over-fetch, identity-store, and trustless capability flags — and the conditions under which it would fail. |
| [no-delta-rule.md](no-delta-rule.md) | The state update is affine in the input and never in the state. Commutativity, chunkability, schedulability. Where expressiveness goes instead. |
| [factored-representation.md](factored-representation.md) | Store the factors; expand transiently where they are consumed. Never materialize the expansion. |
| [fp32-accumulators.md](fp32-accumulators.md) | Accumulation precision is closed at fp32. Storage precision is a live axis. |
| [closed-world-codegen.md](closed-world-codegen.md) | Only measured binaries run: pinned assembler, AOT-only, ratified per-arch manifest, runtime refusal. Why the JIT fork was rejected. |
| [planner.md](planner.md) | Dispatch on what the data *does*, not on a threshold over what it *is*; model with measured coefficients, never a table and never a tuned constant. |
| [guarantees-choices-preferences.md](guarantees-choices-preferences.md) | The taxonomy every proposed change is sorted into, populated with real entries, and what each category demands of a contributor. |

## Reading order

A contributor changing the kernel should read
[guarantees-choices-preferences.md](guarantees-choices-preferences.md) first: it
tells you which of the other documents your change is arguing against, and what
kind of argument would be admissible.
