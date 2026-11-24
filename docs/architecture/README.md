# `docs/architecture/` — the permanent decisions

This tree holds the decisions that outlive any particular kernel: the things a
contributor must know before changing code, and the reasons behind them, written
for a reader who was not here. Nothing here is a status report. If a statement is
true only of the current implementation, it belongs in
[`docs/internals/`](../internals/README.md) — the per-source-file mirror, which
carries the reasons attached to a *line*. If it is the record of who measured
what on which day, it belongs in the measurement record
([`docs/measurement.md`](../measurement.md)) or the lab's results. Each document here
states a decision and its argument, with citations back to the test, the lab record,
or the source line, so that a future reader can **re-derive** rather than trust.

Everything here is falsifiable. Where the record is genuinely uncertain, the
document says so and names the measurement that would settle it.

| Document | What it settles |
|---|---|
| [self-masking-theorem.md](self-masking-theorem.md) | Dead routes are exact zeros and zeros are inert. The licence for over-fetch, identity-store, and trustless capability flags — and the conditions under which it would fail. |
| [no-delta-rule.md](no-delta-rule.md) | The state update is affine in the input and never in the state. Commutativity, chunkability, schedulability. Where expressiveness goes instead. |
| [factored-representation.md](factored-representation.md) | Store the factors; expand transiently where they are consumed. Never materialize the expansion. |
| [hopper-carry.md](hopper-carry.md) | The design the hopper carry's kernel is built from: the call's shape, the packets it is handed, its CTA and a window, entry and exit, the scales and the den, the hand-off from the gather (not designed), the order of the build, and what is answered in the kernel. |
| [hopper-gather.md](hopper-gather.md) | The hopper gather's design before it is built -- PROPOSED, not locked: the mask-preparation pass that prepares both sides' group words and counts for it and for the carry (RULED), what it is handed, the packets it writes for the carry, its CTA's rules from the carry's labs, what is measured and what is to be ruled. |
| [precision.md](precision.md) | Every accumulator is fp32 or exact integer but the output's sum across CTAs (fp32 or f16, the plan's); storage and operands are one family on every card: u8 rows, bf16 values and operands. |
| [closed-world-codegen.md](closed-world-codegen.md) | Only measured binaries run: pinned assembler, AOT-only, the declared targets and the per-build gates, runtime refusal. Why JIT compilation is rejected. |
| [families.md](families.md) | One design, a schedule per family: what differs by structure (a family's schedule), by instruction (an atom) and by number (a plan); the capability rows hold only what is read; one entry a kernel, its hardware facts in the C++ path. |
| [guarantees-choices-preferences.md](guarantees-choices-preferences.md) | The taxonomy every proposed change is sorted into, populated with real entries, and what each category demands of a contributor. |

## Reading order

A contributor changing the kernel should read
[guarantees-choices-preferences.md](guarantees-choices-preferences.md) first: it
tells you which of the other documents your change is arguing against, and what
kind of argument would be admissible.
