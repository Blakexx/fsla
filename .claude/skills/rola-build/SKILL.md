---
name: rola-build
description: Iteration vs gate CUDA builds for rola-standalone — arm subsets, build_slot.sh, background builds with an exit marker, the compile ledger
---
Read `docs/build.md` and KERNEL_STANDARDS §7, §14, §15, §16 addendum 3 first. This skill is the
mechanics of running a build, not deciding whether one is owed (that's §16: stage gates are
iteration builds, only milestones ratify).

## Iteration vs gate
- **Iteration build**: single-arch, arm-subset. `ROLA_CUDA_ARCHS=<arch> ROLA_CARRY_ARMS=<csv|all>
  pip install -e . --no-build-isolation`. Runs in one of two parallel slots via
  `bash /home/blake/rola-scratch/tools/build_slot.sh <cmd>` — read that script's own header table
  for the current `MAX_JOBS`/`ROLA_NVCC_THREADS` it uses, don't hand-restate them here.
- **Gate build**: both arches, default (shipped) arm set, no `ROLA_CARRY_ARMS` override, under the
  EXCLUSIVE `/tmp/rola_ram.lock` (KERNEL_STANDARDS §15 addendum). One gate build per stage; ratify
  only at milestones (§16).
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
