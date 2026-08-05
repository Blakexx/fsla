# `DELETIONS.md` — the ledger

Every removal of code from this tree gets a row here: what went, the commit that
removed it, and one line on why it was safe. Deletions land in **deletion-only
commits** — never mixed with a move, a rename or a rewrite — so that this ledger
and `git log` say the same thing, and so a reviewer can read a deletion without
reading a refactor. The SHA is stamped in the immediately following commit, since
a commit cannot name itself.

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
| 2026-08-04 | `bcd9a16` | `rola::mma::mma_m16n8k8_tf32x2` (`csrc/rola/src/mma.cuh`) — the two-term tf32 split | No caller anywhere in the tree (`.cu`, `.cuh`, `.cpp`, `.py`). The engine's exactness arm is `tf32x3`, the three-term split, which is what `mma_precision = "tf32x3"` selects and what the fp64-oracle gate runs; the two-term variant is a rung that was measured and not taken. It is a `__device__ __forceinline__` function with no callers, so it emitted no code — the SASS gate is unmoved by its removal, which is the evidence that nothing depended on it. |
| 2026-08-04 | `bcd9a16` | A commented-out profiler line at `benchmarks/bench.py:252` | A debug variant of the line immediately below it (`sort_by="self_cuda_time_total"`). Commented-out code is a claim that someone might want it back, made in a place where nothing checks it; the live line is one edit away from carrying the sort. |

---

## Pre-extraction removals

This repository's history begins at the extraction, so a bare commit hash cited
anywhere in this tree resolves in the source repository named by
[`docs/provenance.md`](../provenance.md) — `flash-linear-attention-rola`, branch
`rola-v3` — and **not** here. The rows below carry that repository's hashes.

| Commit (`flash-linear-attention-rola` @ `rola-v3`) | What | Invariant it leaves behind |
|---|---|---|
| `c3c2f845` (2026-08-01) | The register-resident consumer: its kernel header (also named `consumer_kernel.cuh`, a DIFFERENT kernel from this tree's file of that name), `intra_gram.cuh` and the `gram` intra lowering, `consumer_matrix.def` and its eight shard translation units, the `intra_lowering` template axis, and the analytic SMEM mirror | One consumer kernel family, so what the planner prices and what the driver launches are the same object: [`consumer.md`](consumer.md#one-consumer) |
| `c3c2f845` (2026-08-01) | The intra-kernel tracer: `v3_trace.cuh`, `v3_trace_host.{cu,cuh}` and the capture scripts, whose 27 `V3T*` emission sites were all in that consumer | No trace-emission site exists in the shipped consumer: [`consumer.md`](consumer.md#tracer-retired) |
| pre-extraction | The V2 CUDA page-plan producer and its eight pybind entries | The page table is published by the torch arena in `rola/ops/paging.py`: [`rola_api.md`](rola_api.md#deleted-page-plan) |

The ledger does not reconstruct what that repository's history already holds.

---

## <a id="dormant"></a>Dormant by plan — implemented, kept, NOT selected

| What | Where | Why it is kept | What would activate it |
|---|---|---|---|
| **The three-pass §6.4 sequence-parallel split** (`P > 1`): pass 1's sparse per-segment summary, `scan_kernel`'s ordered fold, pass 3's entry-state stash, the fourteen optional split arguments | `csrc/rola/src/consumer.cu`, `consumer.cuh`; [`consumer.md`](consumer.md#three-pass) | Fully implemented, oracle-gated, covered by `tests/tier2_geometry/`. `P = 1` is the primary benchmarked configuration and the planner chooses it everywhere today, so the machinery is reachable but unexercised in production. Its correctness structures (the stash) were paid for by a real failure, not a hypothetical one. | A measured cell where a split beats `P = 1`. The planner already takes `P` from the plan; nothing needs to be written to turn it on. |

A row leaves this register in exactly two ways: it becomes plan-selected (and the
row goes), or its activating measurement is taken and comes back negative for
good (and it moves to Deletions with that measurement cited).
