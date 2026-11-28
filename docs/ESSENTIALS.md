# ESSENTIALS — injected after every compaction / resume. Read before acting.

Everything here is IN THIS REPO. A site's own notes (scratch, memory, the queue) are appended by
the hook only when the environment names them (`FSLA_SCRATCH`); the repo never references out.

## The constitution and the laws
`docs/CONSTITUTION.md` — the owner's standing rulings; `docs/KERNEL_STANDARDS.md` — the LAWS. Before touching a
kernel:
- `#s19` the component budget gate: skeleton first, then one component at a time against its budget line.
- `#s20` the ptxas signatures the library's SASS gate (warpline's check, `bazel/defs.bzl`'s `_sass_gate`) refuses, on every build.
- `#s21` the measurement protocol: quiet GPU, the baseline's own schedule, counts beside time.
- `#s26` a rule about the machine cites a lab experiment ([`docs/lab/`](lab/README.md)).
- `#s27` a step is one block; `#s28` no phase idles the binding stream.
- `#s29` every instruction budgeted by class before code; `#s30` shared memory conflict-free by construction.
- `#first-run` new device code runs first under a kill timeout; `#r22` degrade smoothly, independent units.
- `#stepped` a composed kernel is read by its TIMELINE, measured and modelled: trace it with warpline (stamps through its
  tracer, `Bench::trace`, `lab trace`; `lab.py` refuses an untraced composed row) and model it from `warpline.floor`; a floor or prediction comes from the
  model that reproduces measured timelines, never from a unit's rate alone. `#s32` item 2: PREDICT, THEN MEASURE, AND A
  MISS STOPS -- a missed prediction stops work on the kernel until the wrong price or model is named and corrected.
  `#s23` addendum 2: a model reproduces the probe's rows before it reads the kernel; one point is not validation.
- Every rule has a row in `docs/rules.json` naming what enforces it; a new rule lands with its mechanism, never as `owed`, and the owed list only shrinks (`docs/KERNEL_STANDARDS.md#every-rule-has-a-row`).
- Efficiency is correctness; build the whole design or stop and name the gap; no fallbacks, no dead code;
  comments are decl blocks, prose goes to `docs/internals/`; the oracle is the truth (`tests/oracle/`).

## What exists, by action
- Before asserting a kernel fact, grep the source or contract doc in the current worktree; if unchecked say
  `unverified`; cite a searchable string, never a line number.
- Prove it: correctness `docs/testing.md`; a lab row `docs/internals/tools/lab.md#the-loop`; a shipped kernel's cost
  `docs/internals/tools/README.md`; a declared measurement `docs/measurement.md`; a commit `docs/build-graph.md#checks`.
- A build or tool that will not start: `bazel run //:check` (`docs/internals/tools/dev.md`).
- A tool call is refused by `.claude/hooks/` (a polling wait, `--no-verify`, a toolchain or install command typed by hand: `.claude/hooks/graph_only.py`, override `GRAPH_OVERRIDE="<why>"`) through the dispatcher `bazel run //:init` installs; the refusal names the rule and its override (`docs/internals/tools/lint.md#action-hooks`).
- A push to the default branch waits until every review of its commits has landed and been read; nothing refuses it
  yet (`docs/internals/tools/review_gate.md#pipelined`).

## The instruments
- Perf: `bazel run //measure:session_entmax` (an interleaved timing session, its clock claim `unlocked`, over this tree's timing entries,
  stored; `docs/measurement.md`), `bazel run //measure:store_oracle_tier` (the oracle tier on the device, recorded) and
  `bazel run //measure:store_static_manifest` (the static manifests, recorded); attribution:
  `bazel run //tools:region_ledger -- --csv <ncu SourceCounters> --so <.so> --member <cubin member>
  --source <kernel.cuh> --budget tools/budgets/<k>.json` (no kernel has a budget file yet: owed).
- Correctness: `tests/oracle/` (scoped runs only; never a bare `tests/`).
- Guidance on a checker rule: its entry in the `checker` section of `docs/rules.json` (`guidance`, `triggers`, `rules`; no standard, no row) is printed after the checker's fix in a lab refusal and injected by the edit hook `.claude/hooks/guidance.py` when new kernel source matches its triggers (`docs/internals/tools/lint.md#rules-table`).
- Build: `bazel run //:library` with `--//:archs`; the machine is `bazel run //:check`
  (`docs/build.md`); the device stamp proves the binary is the source.
- The map of the documents: `docs/README.md`; of the mechanisms: `docs/ARCHITECTURE.md` §10.
