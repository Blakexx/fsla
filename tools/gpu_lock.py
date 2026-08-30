#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""ONE gpu_lock() every GPU-touching tool takes ITSELF (K46: docs/KERNEL_STANDARDS.md
§14/§18, "why even have agents manage locks").

**BEFORE THIS FILE.** The GPU lock was half-and-half: `tools/probe_cells.py` and
`tools/sanitize_oracle.py`'s design took the GPU lock file internally, but pytest
and the `benchmarks/bench` harness relied on the CALLER remembering to wrap the
command with an EXTERNAL `flock` on that same lock path. Two incidents came out of
that split: K38 wrapped a self-locking tool in such an external `flock` and
deadlocked 17 CPU-idle minutes (`flock` is held per open file description, so the
wrapper's own lock blocks the wrapped process's identical `fcntl.flock` call
forever); and every brief had to restate the wrapping correctly by hand, which is a
rule a human has to remember rather than a rule a tool enforces.

**THE FIX.** Every GPU entry point calls `gpu_lock()` itself -- pytest (a session
fixture, `tests/conftest.py`), `tools/probe_cells.py`, `tools/sanitize_oracle.py`,
`benchmarks/bench/discipline.py`. Nothing is invoked wrapped in an external `flock`
again; the word leaves the command line (a lint rule in `tools/lint/lint_standards.py`
finds it if it comes back).

**REENTRANT BY CONSTRUCTION.** Acquiring the lock sets `ROLA_GPU_LOCK_HELD` in the
process environment; a NESTED acquire that finds the marker already set is a no-op.
Because `os.environ` is inherited by every subprocess a locked process spawns (`fork`
+ `exec` copies the parent's environment unless told otherwise), this covers both
same-process re-entry (a test fixture calling a helper that also locks) and the
subprocess case (`tools/sanitize_oracle.py` locking once, then launching
`compute-sanitizer python -m pytest ...`, whose own conftest fixture sees the marker
and proceeds without re-locking) -- so a double acquisition cannot deadlock by
construction, closing the OTHER half of the K38 shape (an outer tool that itself
locks, calling an inner tool that also locks, rather than an outer bare `flock`).

**THE LOCK FILE.** `/tmp/rola_gpu.lock` by default -- the name every doc, skill, and
script in this tree already cites (`docs/measurement.md`, `docs/testing.md`,
`docs/setup.md`); `ROLA_GPU_LOCK` overrides it, the same override name
`benchmarks/bench/discipline.py` already read before this file existed. `/tmp` is not
a home path (KERNEL_STANDARDS §17) and is the shared directory KERNEL_STANDARDS §14
addendum already requires every GPU-touching container to bind-mount from the host,
so a container and the bare host serialize against the same lock without a new
directory convention to keep in sync across both.
"""
from __future__ import annotations

import contextlib
import fcntl
import os
import shutil
import subprocess

GPU_LOCK_DEFAULT = "/tmp/rola_gpu.lock"

#: docs/internals/tools/gpu_lock.md -- the NICE stage (Blake, 2026-08-29).
_NO_NICE_ENV = "ROLA_NO_NICE"


def _lower_own_priority() -> None:
    if os.environ.get(_NO_NICE_ENV) == "1":
        return
    with contextlib.suppress(OSError):
        os.nice(10)
    ionice = shutil.which("ionice")
    if ionice is None:
        print("[gpu_lock] ionice not on PATH; CPU niceness only", flush=True)
        return
    subprocess.run(
        [ionice, "-c", "3", "-p", str(os.getpid())],
        check=False, capture_output=True,
    )


#: Set to this process's pid on acquire, cleared on release. Presence alone (not an
#: exact pid match) is what a nested acquire checks -- a marker inherited from an
#: ancestor process through `exec` still proves "an ancestor already holds this",
#: which is the case that matters (see module docstring's subprocess case).
_HELD_MARKER = "ROLA_GPU_LOCK_HELD"


@contextlib.contextmanager
def gpu_lock(path: str | None = None):
    """Exclusive GPU access, held for the `with` block's duration.

    Reentrant: if `ROLA_GPU_LOCK_HELD` is already set (this process or an ancestor
    already holds a `gpu_lock()`), this is a no-op context -- it neither opens nor
    waits on the lock file a second time.
    """
    if _HELD_MARKER in os.environ:
        yield
        return
    lock_path = path or os.environ.get("ROLA_GPU_LOCK", GPU_LOCK_DEFAULT)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.environ[_HELD_MARKER] = str(os.getpid())
        _lower_own_priority()
        try:
            yield
        finally:
            os.environ.pop(_HELD_MARKER, None)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
