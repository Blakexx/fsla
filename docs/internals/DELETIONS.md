# `DELETIONS.md` — the ledger

Every removal of code from this tree gets a row here: what went, the commit that
removed it, and one line on why it was safe. Deletions land in **deletion-only
commits** — never mixed with a move, a rename or a rewrite — so that this ledger
and `git log` say the same thing, and so a reviewer can read a deletion without
reading a refactor.

The ledger exists because a codebase that cannot delete accumulates plausible
code nobody dares touch. Recording the deletion is what makes deleting cheap.

## What must be recorded

1. Code removed from `csrc/`, `rola/`, `tools/`, `benchmarks/` or `tests/`.
2. A **capability** removed — an axis, a mode, a kernel arm — even if the code
   was already unreachable.

## What must NOT be deleted, and is recorded here instead

Implemented, tested, oracle-gated machinery that the planner does not currently
select is **dormant by plan**, not dead. Deleting it deletes a measured-and-kept
capability, and it is exactly what a stranger reading the tree will mistake for
dead code. The [dormant register](#dormant) below is the answer to that mistake.

---

## Deletions

| Date | Commit | What | Why it was safe |
|---|---|---|---|
| 2026-08-04 | `DELETION-1` | `rola::mma::mma_m16n8k8_tf32x2` (`csrc/rola/src/mma.cuh`) — the two-term tf32 split | No caller anywhere in the tree (`.cu`, `.cuh`, `.cpp`, `.py`). The engine's exactness arm is `tf32x3`, the three-term split, which is what `mma_precision = "tf32x3"` selects and what the fp64-oracle gate runs; the two-term variant is a rung that was measured and not taken. It is a `__device__ __forceinline__` function with no callers, so it emitted no code — the SASS gate is unmoved by its removal, which is the evidence that nothing depended on it. |
| 2026-08-04 | `DELETION-1` | A commented-out profiler line at `benchmarks/bench.py:252` | A debug variant of the line immediately below it (`sort_by="self_cuda_time_total"`). Commented-out code is a claim that someone might want it back, made in a place where nothing checks it; the live line is one edit away from carrying the sort. |

Historical removals that predate this ledger are recorded where they happened —
the retired intra-kernel tracer at
[`consumer.md`](consumer.md#tracer-retired), the deleted register-resident
consumer and its `intra_lowering` axis at
[`consumer.md`](consumer.md#one-consumer), and the deleted `gram` intra lowering
at [`intra_panel.md`](intra_panel.md). The ledger begins here; it does not
reconstruct what the source repository's history already holds
(`docs/provenance.md`).

---

## <a id="dormant"></a>Dormant by plan — implemented, kept, NOT selected

| What | Where | Why it is kept | What would activate it |
|---|---|---|---|
| **The three-pass §6.4 sequence-parallel split** (`P > 1`): pass 1's sparse per-segment summary, `scan_kernel`'s ordered fold, pass 3's entry-state stash, the fourteen optional split arguments | `csrc/rola/src/consumer.cu`, `consumer.cuh`; [`consumer.md`](consumer.md#three-pass) | Fully implemented, oracle-gated, covered by `tests/tier2_geometry/`. `P = 1` is the primary benchmarked configuration and the planner chooses it everywhere today, so the machinery is reachable but unexercised in production. Its correctness structures (the stash) were paid for by a real failure, not a hypothetical one. | A measured cell where a split beats `P = 1`. The planner already takes `P` from the plan; nothing needs to be written to turn it on. |
| **The `panel` intra lowering** | `csrc/rola/src/intra_panel.cuh`; [`intra_panel.md`](intra_panel.md#measured-status) | Implemented, oracle-gated, explicitly reachable, and measured SLOWER than the token-sequential walk at 16 of 16 cells (+12% to +106%). The cause is named and is a property of what is missing (an unfused intra epilogue), not of the lowering. | Fusing the intra term into the readout epilogue, then re-measuring. That is the standing follow-up for both lowerings. |
| **The `.def`-plus-shard-TU dispatch split** | not present; the mechanism is described at [`consumer.md`](consumer.md#instantiation-matrix) | Not code in this tree — a recorded ANSWER to a cost that has not yet been incurred. Recorded so that a future contributor facing an unacceptable build does not re-derive it. | A per-TU `nvcc` wall-clock measurement showing `consumer.cu` dominating the build under the shipped `MAX_JOBS`. |

A row leaves this register in exactly two ways: it becomes plan-selected (and the
row goes), or its activating measurement is taken and comes back negative for
good (and it moves to Deletions with that measurement cited).
