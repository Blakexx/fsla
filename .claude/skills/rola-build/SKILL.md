---
name: rola-build
description: Iteration vs gate CUDA builds for rola-standalone — arm subsets, tools/build/build_lock.sh, background builds with an exit marker, the compile ledger
---
Read `docs/build.md` and KERNEL_STANDARDS §7, §14, §15, §16 addendum 3 first. This skill is the
mechanics of running a build, not deciding whether one is owed (that's §16: stage gates are
iteration builds, only milestones ratify).

## Iteration vs gate
- **ONE lock mechanism for both** (K39, Blake ruling 2026-08-28 — supersedes the old two-mechanism
  split of a build-slot script's parallel slots and a separate exclusive `/tmp/rola_ram.lock`):
  `tools/build/build_lock.sh [--gate] -- <cmd>`, IN the repo (§17). An iteration build takes ONE
  free slot; a gate build takes EVERY slot (`--gate`) before running, so the two can never overlap
  and OOM the host — read the script's own header for the exact mechanism.
- **Iteration build**: single-arch, arm-subset. `ROLA_CUDA_ARCHS=<arch> ROLA_CARRY_ARMS=<csv|all>
  pip install -e . --no-build-isolation`, run through `tools/build/build_lock.sh -- <cmd>` (no
  `--gate`) — read the script's own header table for the current `MAX_JOBS`/`ROLA_NVCC_THREADS` it
  uses, don't hand-restate them here.
- **Gate build**: both arches, default (shipped) arm set, no `ROLA_CARRY_ARMS` override, run through
  `tools/build/build_lock.sh --gate -- <cmd>`. One gate build per stage; ratify only at milestones
  (§16). A stage's own gate build does NOT set `ROLA_STRICT_MANIFEST` — the committed manifest is
  expected to be stale between milestones (K43), so the post-build check REPORTS, not fails. The
  build that immediately precedes `ratify.py --write` at a milestone DOES set
  `ROLA_STRICT_MANIFEST=1` (docs/build.md) — that is the one build that must fail loudly if the
  tree is not actually shippable yet.
- `ROLA_CARRY_ARMS=all` builds every row (SHIPPED + TEST) — needed for the battery/oracle across
  arms outside the default shipped set (`gen_shards.CARRY_TEST_ARMS`).
- Don't hand-restate `MAX_JOBS`: `tools/build_flags.py:default_max_jobs` derives it from free RAM /
  a per-thread footprint constant in that same file (re-measured whenever a stage shows the old
  value was OOM-risky). Only override when the derived value is measured wrong for the host.

## ROLA_CARRY_ARMS is a FILE subset (K36b), not a `-D`
An arm-set change no longer touches `nvcc_flags` on every TU. `setup.py` (build) or
`tools/ratify.py` (ratify) writes `build/generated/carry_selection.inc` — a build-scope macro plus
one per-selected-row macro (read the generator, `tools/gen_shards.py`, for the exact names) —
UNCONDITIONALLY, before any TU that includes it compiles. Only `carry_arms.inc` includes it (the
per-arm TUs and `carry/carry.cu` reach it through that). So an arm-set change rebuilds only the
requested arm object files plus the dispatch object — check `setup.py`'s TU list for which other
objects exist and confirm they're untouched.
Never hand-edit or reuse a stale `carry_selection.inc` between runs — a scoped build/ratify writing
its own subset and a later `--write` inheriting the wrong one is the exact K36 trap (a manifest
carrying a partial arm set was written and reported success — see FINDINGS `## K36b` for the
reproduction).

## Background builds — never poll (KERNEL_STANDARDS §14)
Launch the build as a background command whose LAST step appends an exit marker, then end your
turn:
```bash
( <build-env-vars> pip install -e . --no-build-isolation > /tmp/build.log 2>&1 ; \
  echo "BUILD_EXIT=$?" >> /tmp/build.log ) &
```
On re-invocation for ANY reason, read the marker first — never trust the notification alone. Wait
with a BOUNDED timer, not a loop: `timeout <N> tail --pid=$PID -f /dev/null`, then inspect state and
repeat if still building. NEVER `until pgrep -f ... ; do sleep ...; done` — it self-matches its own
command line (the documented multi-hour deadlock: two waiters each saw the other's `pgrep -f`
invocation as a live `pip install`), it matches other tracks' builds in the same venv, and it
re-wakes stopped agents into worktrees they no longer own. Never edit a source file while its build
is in flight.

## The torch-header floor
`#include <torch/extension.h>` costs a large fixed compile wall + RSS per `cicc`, paid once by
`carry/carry.cu` (the dispatch TU that reaches it) and by every other torch-bearing TU — see
FINDINGS `## K36` for the measured cost at that stage, cited as illustration only, not a reference
value (re-measure via the compile ledger below if you need a current number). Every
`carry_arm_<i>.cu` is torch-FREE (raw-pointer ABI, `carry_arm_abi.cuh`) and pays only a small
per-TU cost — so arm count barely moves iteration-build wall time; what moves it is which
torch-bearing TUs got touched. Grep `setup.py`'s extension sources list for the current set of
torch-bearing TUs rather than trusting a remembered list — it changes as mechanisms move in or out.

## The compile ledger (KERNEL_STANDARDS §16 addendum 3)
Record per built arm TU, every stage gate: compile wall + `cicc` peak RSS, beside the PREVIOUS
stage's numbers. A jump above the threshold KERNEL_STANDARDS §16 addendum 3 sets is RED — report
it, don't rationalize it; diagnose against §7 (one-instantiation-per-TU, arm-subset) first. Measure
RSS with e.g. `/usr/bin/time -v` around the single-arm compile, or read it from a
`--census`/`ratify.py` run (it doesn't gate wall/RSS itself, but the compile ledger is a manual
add-on next to the census, filed in the stage's FINDINGS entry).
Arm TUs never include torch headers — if a compile ledger row for a `carry_arm_<i>.cu` shows a
wall time near the torch-header floor above, that's a broken ABI seam (a torch-typed function crept
into `carry_arm_abi.cuh`), not noise.

## Standards lint (LINT, 2026-08-28)
Before committing, run the standards lint: the ast-grep rule pack under tools/lint/rules (sgconfig.yml at the repo root) and tools/lint/lint_standards.py; clang-tidy on host TUs via tools/lint/run_clang_tidy.sh (skips cleanly without a build). Each finding names its standards section and the incident behind it. Wired into pre-commit; docs/build.md "Standards lint" has the invocations.

## Addenda (F1b skills feedback, 2026-08-29)
- ARCH FORM: `ROLA_CUDA_ARCHS` takes the bare number (`86`), never `sm_86` (the failure reads `sm_sm_86`).
- THE THING YOU BACKGROUND IS THE BUILD: never wrap the build in a second backgrounding layer; if a launcher
  exits first, the exit MARKER is the fact, not the harness notification.
- SINGLE-TU COMPILE = THE REGISTER CENSUS: compile one arm's instantiation TU alone (`nvcc -c` of
  `csrc/rola/src/instantiations/carry_arm_<i>.cu` with the include dirs `csrc/rola/src` + the generated
  header dir, `-std=c++17 -arch=sm_<N> --expt-relaxed-constexpr -DROLA_CARRY_BUILD_<i>`; without
  `--expt-relaxed-constexpr` it fails with unrelated-looking `constexpr __host__` errors) and read
  `cuobjdump -res-usage` on the `.o`: it reproduces the full build's numbers exactly (verified F1b, ten
  kernels, two trees). ~4 s per arm. A register-gated stage answers its gate here, before any full build.
- A BUILD LAUNCHED AS A SHELL BACKGROUND JOB SENDS NO NOTIFICATION (G2, 2026-08-29): ending your turn to
  "resume on its notification" is a dead stop — only harness-tracked tasks wake you. Wait on the exit
  marker yourself (Monitor with an until-loop on the marker file, or `until [ -f marker ]; do sleep 30;
  done` with a long timeout) and continue in the same turn. Never end a turn on an untracked build.
- A MERGE THAT TOUCHES csrc/ IS FOLLOWED BY A TIP REBUILD BEFORE ANY GATE IS QUOTED (13th trap form,
  2026-08-29): the tip's .so stayed pre-merge while the Python moved on; assert_fresh_binary did not
  fire. Python-only stages: the worktree still needs the built .so (copy the tip's, then
  assert_fresh_binary); a pointer venv without it = rola._C ImportError = CPU-only battery = not a gate.
- EVERY COMPILE RUNS UNDER THE SLOT BUDGET (incident 2026-08-29: an agent pegged the CPU for 10 min and
  the host had to be restarted): single-TU census compiles, clang-tidy device runs and iteration builds
  take tools/build/build_lock.py's slots exactly like setup.py's gate build; never `xargs -P`/`&` a fan-out
  of nvcc/clang/cicc; never more than the budget's slots in flight from one agent.
