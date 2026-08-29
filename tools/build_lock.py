# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE MACHINE-WIDE CICC BUDGET, TOOL-ENFORCED (KERNEL_STANDARDS §14 addendum 2,
Blake ruling 2026-08-28-night: "three agents' MAX_JOBS=4 each -> 12+ cicc" sat
this host at 100% CPU for 10+ minutes and forced a WSL shutdown mid-session).

**WHY THIS CANNOT BE A BRIEF INSTRUCTION.** `tools/build/build_lock.sh` (K39,
earlier the same day) already unified the iteration-slot and gate-lock
mechanisms into one -- but it is an EXTERNAL wrapper a caller has to remember
to invoke, and three concurrent agents each forgetting it (or each reasonably
believing their OWN `MAX_JOBS=4` was the whole host's budget, not a share of
it) is exactly what happened. `RoLABuildExtension.run()` (`setup.py`) now
acquires this budget ITSELF, unconditionally, before the actual compile --
`pip install -e .` cannot start `cicc` without going through it, from any
worktree, any agent, any invocation path, brief or no brief.

**THE MECHANISM.** `GLOBAL_CICC_SLOTS` (default 4, `ROLA_GLOBAL_CICC_SLOTS`
overrides) lock files, `/tmp/rola_cicc_slot_<i>.lock` -- ONE POOL, shared by
every worktree on this host (the lock files are NOT per-worktree; a build in
`k31-k39` and a build in `k31-k41` contend for the SAME four slots, which is
the whole point: the budget is the HOST's, not a checkout's). A GATE build
(the shipped arm set) takes EVERY slot, blocking until the whole budget is
free, mirroring `build_lock.sh --gate`'s exclusivity. An ITERATION build takes
however many slots are free RIGHT NOW, from 1 up to its own desired
`MAX_JOBS`, BLOCKING (never proceeding with zero) until at least one is free --
so three iteration builds started at once each still get SOME progress rather
than one winning everything or all three deadlocking. `MAX_JOBS` is then set
to the COUNT ACTUALLY HELD, never to what the caller originally asked for --
the whole enforcement is worthless if a build keeps its self-chosen `MAX_JOBS`
after being handed fewer slots than that.
"""
from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path

GLOBAL_CICC_SLOTS = int(os.environ.get("ROLA_GLOBAL_CICC_SLOTS", "4"))


def _slot_path(i: int) -> Path:
    return Path(f"/tmp/rola_cicc_slot_{i}.lock")


def _try_acquire(i: int):
    #: The file object OUTLIVES this function by design -- it is the lock's own
    #: handle, released by the caller's `finally` block once the compile it
    #: guards is done -- so a `with` block here (which would close it on this
    #: function's own return) is the wrong shape, not a missed one (ruff SIM115).
    fd = open(_slot_path(i), "w")  # noqa: SIM115
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fd.close()
        return None
    return fd


def _blocking_acquire(i: int, label: str):
    fd = open(_slot_path(i), "w")  # noqa: SIM115 -- see _try_acquire
    print(f"[build_lock] waiting for slot {i} ({label}) ...", flush=True)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


@contextmanager
def acquire(*, gate: bool, desired: int):
    """Hold `held` slots (yielded) for the wrapped compile; releases on exit.

    `gate=True`: waits for and holds ALL `GLOBAL_CICC_SLOTS` slots -- the
    machine-wide budget, exclusively, the same discipline a milestone ratify
    build already required against a single lock file (KERNEL_STANDARDS §15
    addendum), now against the shared pool every build (gate or iteration)
    draws from.
    `gate=False`: waits for AT LEAST ONE free slot (never proceeds with zero),
    then opportunistically takes up to `desired` MORE slots that are free
    RIGHT NOW without waiting further for them -- an iteration build gets
    *some* parallelism promptly rather than either the whole budget or an
    unbounded wait for it.
    """
    desired = max(1, desired)
    held: list = []
    try:
        if gate:
            print(f"[build_lock] GATE build: waiting for all "
                  f"{GLOBAL_CICC_SLOTS} slot(s) of the machine-wide cicc "
                  f"budget ...", flush=True)
            for i in range(GLOBAL_CICC_SLOTS):
                held.append(_blocking_acquire(i, "gate"))
            print(f"[build_lock] GATE build: holding all "
                  f"{GLOBAL_CICC_SLOTS} slot(s)", flush=True)
        else:
            # First pass: grab whatever is free without waiting.
            for i in range(GLOBAL_CICC_SLOTS):
                if len(held) >= desired:
                    break
                fd = _try_acquire(i)
                if fd is not None:
                    held.append(fd)
            # Never proceed with zero: block on slot 0 if the first pass got
            # nothing, then retry a free sweep once more (best-effort only --
            # this does not re-loop indefinitely trying to reach `desired`).
            if not held:
                held.append(_blocking_acquire(0, "iteration, at least one slot"))
                for i in range(1, GLOBAL_CICC_SLOTS):
                    if len(held) >= desired:
                        break
                    fd = _try_acquire(i)
                    if fd is not None:
                        held.append(fd)
            print(f"[build_lock] iteration build: holding {len(held)}/"
                  f"{GLOBAL_CICC_SLOTS} slot(s) of the machine-wide cicc "
                  f"budget (wanted {desired})", flush=True)
        yield len(held)
    finally:
        for fd in held:
            fcntl.flock(fd, fcntl.LOCK_UN)
            fd.close()
