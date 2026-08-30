# `tools/build_lock.py` — the compile budget also lowers its own priority

`build_lock.acquire()` is the in-process context manager `setup.py`'s
`RoLABuildExtension.run()` wraps every real compile in (the machine-wide
`GLOBAL_CICC_SLOTS` pool this file's own module docstring covers). The NICE
stage (Blake, 2026-08-29: "I would like it so all our things don't completely
freeze my system so during future incidents I can continue to do things")
adds one more thing that acquisition does before yielding control back to the
compile: it lowers the ACQUIRING process's own CPU niceness (`os.nice(10)`)
and, if the `ionice` binary is on `PATH`, its I/O scheduling class to idle
(`ionice -c 3 -p <pid>`) — no `ionice` binary is a logged skip, never a
failure. Both a compiled child process (`cicc`, `ptxas`, `nvcc`) and any
Python subprocess this same interpreter later launches inherit the lowered
niceness through ordinary `fork`+`exec`, so the effect reaches the actual CPU
load without touching the compiler invocation itself.

`ROLA_NO_NICE=1` skips this entirely — the coordinator's own interactive gate
runs opt out (`docs/build.md`'s env-var table). Default is off (nice
applied); nothing about the compile's behaviour, output, or elapsed wall time
changes — this is a scheduling hint, not a resource cap, and the probe/bench
harnesses' own numbers are GPU-timed, not host-scheduler-timed, so they are
unaffected by the host running its compiles niced.

`tests/unit/test_lock_priority.py` is the proof: a child process launched
from inside `acquire()`'s `with` block reports nice 10 in its own
`/proc/<pid>/stat` (field 19), and the same child reports 0 under
`ROLA_NO_NICE=1` — run in a fresh driver subprocess per condition, since an
unprivileged process's own niceness can only go up, never back down, so the
two conditions cannot share one process across both assertions.
