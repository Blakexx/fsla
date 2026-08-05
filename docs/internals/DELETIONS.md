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
| 2026-08-05 | `589b383` | **The COMPACT executor arm, whole.** `csrc/`: the consumer's support-driven gather (`c_idx`, `c_slot`, `c_count` and the two-pass ballot in `stage_tile`), every extent it made runtime (`K`, `KI`, `Kpad`, `KIpad`, `KpadM`, `wt_rows`, `n_mt`, `qmax`), `mma_tile_fragA`'s `B_GATHER` indirection and its `b_row` operand, `consumer_smem_bytes_at`'s gathered-list term, the `COMPACT` template axis of the kernel / `launch_consumer_arm` / `query_consumer_occupancy_arm` / `consumer_smem_bytes` / `consumer_ring_depth` / `ShardRow`, and the `compact` argument of `rola_consumer_forward`, `rola_consumer_occupancy` and `rola_consumer_smem_bytes` — the matrix falls from 512 arms to 256. `tools/`: the axis in `gen_shards.py`'s matrix definition. `rola/`: `Plan.compact`, `ConfigChoice.packing`, the `compact` argument of the `OccupancyOracle` contract, `select_packing`, `pack_breakeven`, `derived_pack_breakeven`, `C_PACK_BREAKEVEN`, `compact_is_structurally_inert` (rung P) and `FeederFlags.any_side_dense`, the gather coefficients `RULE_GATHER_FIXED_WORK`, `RULE_GATHER_QUANTUM_WORK`, `GATHER_CANDIDATE_FIXED_WORK` and `GATHER_CANDIDATE_QUANTUM_WORK`, the packed branch of `cost_per_token`, and the compact plumbing in `ops/consumer.py`, `ops/decode.py`, `layers/rola.py` and `routing/sequence_parallel.py`. `tests/`: `tier2_geometry/test_compact_equivalence.py` (a whole file — the tree's dense-degeneracy bit-identity gate, whose subject was the two compact arms' equality at `K == BC`), the `COMPACT` axis declaration and geometry pair in `tier2_geometry/matrix.py`, `test_a_saturated_dense_cell_never_argmins_onto_a_compacted_arm`, and every compact cell of every surviving parametrization. `benchmarks/`: the compact arms of `bench_consumer.py`, `bench_regression.py`, `bench_scaling.py` and `bench_sequence_parallel.py`, and their rows in `cells/latency_baseline.json` | A CAPABILITY REMOVAL, ruled by the user on 2026-08-05: the arm doubled every gate's cost and the planner selected it nowhere (`C_PACK_BREAKEVEN` shipped EMPTY, so `select_packing` declined every cell, and rung P excluded the arm structurally on the dense-flagged side). The surviving dense arm is what every shipped plan ran. The evidence that this is a specialization and not a rewrite is the binary: over the 256 surviving arms the SASS bodies are unchanged, and the only absences are the 256 COMPACT entries themselves. The design survives on the P14 big-BC card and the code in this commit's parent; a measured cell that wants a gather brings it back with `git revert`, not from memory. ONE GUARD WENT WITH THE ARM AND IS REPLACED BY A MEASUREMENT, not by an assertion: `test_a_saturated_dense_cell_never_argmins_onto_a_compacted_arm` made the cost model's known `BT` inversion unreachable by construction (rung P excluded packing on a dense-flagged side), and with no compacted arm it cannot be restated. What bounds the limitation now is that the shipped closed form reproduces the MEASURED argmin `BT = 32` on all four saturated-dense rows of the calibration table (`topo-D2-w64x64-BC64` and `topo-D2-w64x16-BC16`, decay off and on) — the two decay rows being exactly the cells the inversion was measured on — so the inversion moves no decision at any cell it was measured at. `RULE_TABLE_PROVENANCE` and `cost_model.py`'s provenance string carry that sentence, and `tests/unit/test_planner.py`'s calibration gates assert it. |
| 2026-08-05 | `589b383` | `rola::panel_clock_prefix()` (`csrc/rola/src/intra_panel.cuh`) and `rola::readout_fragment()` (`csrc/rola/src/readout_mma.cuh`) | No caller anywhere in the tree. The kernel computes the clock prefix inline in `stage_tile` and reads the state through `mma_tile_fragA`, so both were earlier lowerings kept beside the ones that ship. Found while removing the `COMPACT` axis they both took as a template parameter — deleting an unused parameter of an unused function is how a dead symbol survives an audit, so the function goes instead. Neither emitted code (`__device__ __forceinline__`, uninstantiated), so the SASS of every surviving arm is unmoved. |
| 2026-08-04 | `bcd9a16` | `rola::mma::mma_m16n8k8_tf32x2` (`csrc/rola/src/mma.cuh`) — the two-term tf32 split | No caller anywhere in the tree (`.cu`, `.cuh`, `.cpp`, `.py`). The engine's exactness arm is `tf32x3`, the three-term split, which is what `mma_precision = "tf32x3"` selects and what the fp64-oracle gate runs; the two-term variant is a rung that was measured and not taken. It is a `__device__ __forceinline__` function with no callers, so it emitted no code — the SASS gate is unmoved by its removal, which is the evidence that nothing depended on it. |
| 2026-08-05 | `5ee4cf0` | `rola::consumer::kFoldPrecDefault` and `kReadPrecDefault` (`csrc/rola/src/consumer_kernel.cuh`) — two `MmaPrec` constants naming the production point of the precision axis | No caller anywhere in the tree. Every arm takes `READ_PREC`/`FOLD_PREC` as template arguments supplied by the generated instantiation matrix from the plan's `mma_precision` field, so a DEFAULT for an argument that is always passed is a second spelling of the production point, and a second spelling is what goes stale. The axis and its contract are unchanged and stated where the choice is made: [`consumer_kernel.md#mma-precision`](consumer_kernel.md#mma-precision). Two `constexpr` values with no reader emit no code — the SASS gate is unmoved by their removal, which is the evidence that nothing read them. |
| 2026-08-05 | `5ee4cf0` | `rola::consumer::staging_slot_bytes()` (`csrc/rola/src/smem_layout.cuh`) — the per-ring-slot footprint of the staging set | No caller anywhere in the tree, here or on master. `consumer_smem_bytes_at` sizes the ring from the two carves it actually rings (`vw_union_bytes` and the two row scales) and `consumer_ring_depth` prices residency off those same carves; this function was a THIRD description of the same bytes, read by nothing and checked by nothing. What the staging set IS remains stated at [`smem_layout.md#staging-set`](smem_layout.md#staging-set), which is about the ring's membership rather than this arithmetic. A `constexpr` function template with no instantiation emits no code — the SASS gate is unmoved. |
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
