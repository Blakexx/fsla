# Open work

Carried in from the extraction. These become GitHub issues when the repository is
published (execution checklist E12); until then this file is the list.

## Blocking the release

| # | Item |
|---|---|
| **W1** | **The AOT build and the re-ratification, as ONE unit.** C++ namespace `fla_rola::v3`/`::p3` → `rola`; the `rola_api.cpp` pybind split; `arch_table.cuh` split out; `setup.py` per design §3.3/§3.5/§3.6 (gencode with the rule-2 no-PTX assertion, R1/R3 pre-build gates, post-build manifest check, `_build_config.py`); then `tools/ratify.py --write` and the §0.2 **key-set/value-set diff** — identical `{regs, spill_stores, spill_loads}` multisets, 512 entries per arch, only symbol strings differing. **Any value change means codegen moved and blocks the extraction.** The rename must not land without the re-measurement: it invalidates all 1024 manifest keys, and a manifest asserting spills for symbols that no longer exist is the exact failure the gate exists to prevent. |
| **W2** | **`rola/interface.py` and the O-3 rename.** The three-layer op stack (private `_forward` → `torch.autograd.Function` → public `rola_func`/`rola_forward`), `torch.library` registration + `_fake` meta kernels, and the dropping of `v3` from every public python name (`naive_rola_v3`, `select_v3_plan`, `plan_and_schedule_v3`, `v3_consumer_forward`). One commit, so that no function has two public spellings. A single autograd boundary is also the structural fix for the layer-dispatch autograd gap. |
| **W3** | **Tier 2 as data**: `tests/tier2_geometry/matrix.py` plus the three meta-tests — M1 census (a new geometry cannot enter the product undeclared), M2 non-vacuity, M3 class drift. M3 must be verified *by construction*: force `P>1` to degrade to `P=1` and confirm M3 fails. A meta-test that cannot fail is not a meta-test. |
| **W4** | **Per-arch manifest split.** `tools/manifests/_all_archs.json` is still the un-split original. Split into `sm_80.json` / `sm_86.json` so "does arch X have a ratified manifest?" is a file-existence question, which is what the setup.py rule-3 gate needs to ask cheaply. Generalize `CUDA_BIN` resolution off the hard-coded `/usr/local/cuda/bin`; add `--arch`. |
| **W5** | **Bench harness restructure** (§6.9): `benchmarks/bench_consumer_new.py` is 1385 lines of reusable core plus 11 accreted per-question drivers. Split into `bench/cells.py`, `bench/pairing.py` (the interleaved paired-arm engine — this is THE methodology and must be documented as such), `bench/instruments.py` (ncu counters, gatherstats), and thin drivers. Add the pytest-benchmark regression layer with thresholds **derived from each cell's measured IQR**, never invented percentages. Apply the O-11 provenance normalization at the same time: GPU model + driver + toolchain only, host identifiers stripped. |
| **W6** | **Docs**: `docs/spec.md` with REQUIREMENT-ID anchors per the superseded O-10 — every normative clause gets an immutable `[SPEC-…]` ID, tests cite IDs only, and CI checks that every cited ID exists exactly once and that citing tests pin a content hash of their clause, so a silent normative edit fails on purpose. Plus `evidence.md`, `planner.md`, `bringup.md`, `api.md`, the README (§7), and the generated supported-configurations table. |

## Measurement, resuming in this repository

| # | Item |
|---|---|
| **M-1** | **`STAGE_COLUMN_WORK` — the missing term. First measurement task here.** `rewrite.md` §2.37's Ruling-A flip table identifies the gap; isolate the per-tile width-scaled quantum by the §2.19 method and rename the constant with its provenance. A cost constant whose name does not say what it prices is how a cost model drifts. |
| **M-2** | Deferred kernel-performance queue, resuming after the campaign starts: single-copy `W`; windowed-metadata / async pipeline; stage 3 planner batching + the dense-saturation closed form; stage 4 closed-world rules 2–4; stage 5 headline benches including the latency-vs-capacity frontier cells. |
| **M-3** | Level-split implementation — the campaign's third ablation arm. |
