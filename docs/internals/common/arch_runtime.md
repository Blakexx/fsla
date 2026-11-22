# `common/arch_runtime.cu` — the closed-world arch rule's runtime half

Mirrors `csrc/fsla/src/common/arch_runtime.{cu,cuh}`.

`arch_caps.cuh`'s `static_assert(tabulated(kArch))` fixes the compiling target: a
binary refuses to BUILD for a compute capability the table has no row for. That
static assertion cannot see the device it eventually runs on — the compiling
machine and the running machine are not always the same one, and even on one box a
binary is portable to a different card. `check_arch_table()` is the other half: it
runs on the first launch, reads the device the process is actually holding, and
refuses if that architecture was never tabulated or if the driver's own numbers
disagree with the row a compiled derivation was closed over.

<a id="segment-tokens"></a>
## `segment_tokens(budget, state, per_token, batch, length, device)`

THE LAYER'S SEGMENT LENGTH on `device`'s card: the most whole windows whose working set (`per_token` bytes a
batch-token at `batch`) fits in `budget` beside `state`, at most the call's `length`; 0 where not one window fits,
which the layer refuses by name. Segments are whole windows because the window grid is global: a boundary on a window
boundary moves no pair. The window is the card's, so the length is the binary's answer rather than a Python
computation over a hardware fact; the working-set model and the budget are the layer's (`docs/api.md` section 4.2).

<a id="check-arch-table"></a>
## `check_arch_table()`

The current device's card admitted: `(void)current_card()` ([`select.md#card-of`](select.md#card-of)). The admission runs once a device a
process (the card is kept) and checks two things, in order:

1. The planner's read of the device (`plan::device_facts`, [`plan/environment.md`](plan/environment.md)) holds the contract: the driver's
   own numbers (`cc`, `regs_per_sm`, the two shared-memory sizes, `max_threads_per_sm`, the SM count) are in a context of the planner's contract
   ([`plan/planner.md`](plan/planner.md)), and a device outside it is refused there, by name, with the model key beside the facts.
   There is no second table of the driver's numbers to drift from the contract's.
2. The binary carries the card's build (its architecture's): a tabulated architecture this
   binary was not built for is refused by name, never run on another architecture's image.

## Where it is called

Every entry point that reaches a kernel launch — the carry forward pass, the SM clock
read, the liveness pass, the standalone intra forward pass, the entmax and softmax
routing forward/backward families, and the T=1 decode step —
calls `check_arch_table()` as the first statement of its body.
`tests/unit/test_closed_world_arch_gate.py` scans the registration translation unit's own
operator list and asserts this for every entry that is not named as launch-less (an arm
census, a build stamp, a host-only derivation over the descriptor, a host query of the card it is asked about -- whose
`card_of` admits that card -- a ledger binding, or a VMM driver operation).

## Why its own translation unit

`arch_caps.cuh`'s facts are `constexpr` and paid for at compile time; the refusal
that reads the device is unavoidably a runtime cost, however small, and giving it
one translation unit means every calling family links one definition rather than
each carrying its own copy that could drift from the others — the same reason the
gate insists there is exactly one `void check_arch_table() {` in the tree.
