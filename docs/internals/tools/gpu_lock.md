# `tools/gpu_lock.py` — the GPU lock also lowers its own priority

`gpu_lock()` is the one primitive every GPU-touching entry point (pytest,
`tools/probe_cells.py`, `tools/sanitize_oracle.py`,
`benchmarks/bench/discipline.py`) takes itself, reentrant by construction
(this file's own module docstring covers the mechanism). The NICE stage
(Blake, 2026-08-29) adds the same treatment `tools/build_lock.py` gives a
compile slot: the process that actually acquires the lock file (never a
nested no-op acquirer — those inherit the ancestor's already-lowered state)
lowers its own CPU niceness (`os.nice(10)`) and, if `ionice` is on `PATH`,
its I/O class to idle (`ionice -c 3 -p <pid>`) before the `with` block's body
— a test, a probe launch, a sanitizer run, a bench — runs. A missing
`ionice` binary is a logged skip, not a failure.

`ROLA_NO_NICE=1` skips this (the coordinator's interactive gate runs;
`docs/build.md`'s env-var table). Default is off. This changes nothing about
what a GPU-locked run measures: the probe and bench harnesses' numbers are
GPU-timed, and niceness only affects host CPU/IO scheduling around the GPU
work, not the work itself.

`tests/unit/test_gpu_lock_lowers_child_priority` (in
`tests/unit/test_lock_priority.py`, shared with the build-lock proof) is the
gate: a child process launched from inside `gpu_lock()`'s `with` block
reports nice 10 in `/proc/<pid>/stat` (field 19), and 0 under
`ROLA_NO_NICE=1` — each condition its own fresh driver subprocess, for the
same reason `build_lock.md` states (niceness cannot be lowered back down by
an unprivileged process). The lock path used is a throwaway `tmp_path` file,
never the real `/tmp/rola_gpu.lock`, so the test never contends with actual
GPU work.
