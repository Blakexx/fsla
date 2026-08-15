"""The measurement preconditions, ENFORCED rather than documented.

Every one of these steps was already written down somewhere -- in eight docstrings,
in `docs/testing.md`, in the usage line of every benchmark script -- and was absent
from most of the throwaway probe scripts that produced the banked numbers. Advisory
discipline is discipline that gets skipped under time pressure, which is exactly the
condition a probe runs under. So it is acquired by the harness:

* the GPU lock is taken by this context manager, and there is no flag that skips it;
* the process memory fraction is set before any allocation the probe makes;
* the clock is settled and an UNSETTLED box ABORTS the run rather than measuring;
* the tier, the tree's sha, its dirtiness and the kernel identity are captured and
  stamped into every row the run writes.

The one thing this module must never do is invent a stopwatch. Timing goes through
:mod:`bench.pairing`; here the instrument is only recorded.
"""
from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import torch

from bench.instruments import settle_clocks

#: THE IN-TREE LOCK. Every benchmark and test recipe in this repository names this
#: path (`docs/testing.md`, every `benchmarks/*.py` usage line), so the harness takes
#: the same one rather than a private file that would exclude nobody. `ROLA_GPU_LOCK`
#: exists for a caller that must compose with an OUTER lock on a shared box; it
#: cannot be used to disable locking, only to move it.
LOCK_PATH = os.environ.get("ROLA_GPU_LOCK", "/tmp/rola_gpu.lock")

#: MEASURED REMEDY, not a guess: the fraction the two device-heavy test modules
#: already pin (`tests/tier2_geometry/test_paging_stream_races.py`,
#: the tiled consumer's visit-barrier gate, retired with it at P67 D2-b) after two
#: whole-machine VRAM freezes.
MEMORY_FRACTION = 0.65

#: The two tiers, as data. `light` is the quick answer during a batch and is stamped
#: as such, so a light-tier number can never be quoted as a landing one; `landing`
#: collects enough rounds for the paired test to be able to reach its alpha
#: (`bench.stats.MIN_ROUNDS`).
TIERS = ("light", "landing")


def _ancestor_holding(path: str) -> int | None:
    """The pid of an ancestor process that already has `path` open, if any.

    `flock` locks are held per OPEN FILE DESCRIPTION, so a parent that wrapped this
    command in `flock <path>` blocks this process's own acquisition of the same file
    — permanently, and silently, because the parent is waiting for the child it is
    blocking. That is not a contention case to wait out; it is a caller error with
    exactly one remedy, so it is DETECTED and named rather than hung on.
    """
    target = os.path.realpath(path)
    pid = os.getppid()
    while pid > 1:
        try:
            for entry in os.scandir(f"/proc/{pid}/fd"):
                try:
                    if os.path.realpath(entry.path) == target:
                        return pid
                except OSError:
                    continue
            with open(f"/proc/{pid}/status", encoding="utf-8") as fh:
                pid = next(int(line.split()[1]) for line in fh
                           if line.startswith("PPid:"))
        except (OSError, StopIteration):
            return None
    return None


@contextmanager
def gpu_lock(path: str = LOCK_PATH):
    """Exclusive use of the device, for the duration of the block.

    Non-blocking first, so a waiting run SAYS it is waiting instead of appearing
    hung; then blocking. The lock is released by closing the descriptor, which
    happens even if the probe raises. A CALLER-HELD lock on the same path is refused
    outright — see :func:`_ancestor_holding`.
    """
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            holder = _ancestor_holding(path)
            if holder is not None:
                raise SystemExit(
                    f"DEADLOCK REFUSED: this process's ancestor (pid {holder}) is "
                    f"already holding {path}, and the harness takes that lock "
                    f"itself. Waiting would block forever, because the holder is "
                    f"waiting for this command. Invoke rola-probe BARE -- every "
                    f"other device line is wrapped in `flock {path}`, this one is "
                    f"not (docs/testing.md, 'Running it'). ROLA_GPU_LOCK moves the "
                    f"harness's lock for a genuinely DIFFERENT outer lock; it must "
                    f"never name the path the caller already holds.") from None
            print(f"waiting for the GPU lock at {path} ...", file=sys.stderr)
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield path
    finally:
        os.close(fd)


def _csrc_sha256() -> str:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    import ratify

    return ratify.csrc_digest()


def _git_state():
    from bench.ledger import git_state

    return git_state()


def _extension_in_this_checkout() -> str:
    """REFUSE to measure a binary that belongs to a DIFFERENT checkout.

    Every other identity this module stamps -- the sha, the dirty flag, the csrc
    digest -- describes THIS tree. None of them describes the ``rola`` package
    Python actually imported, and the two come apart SILENTLY: ``rola`` is
    installed EDITABLE into the shared venv, so a probe run from a worktree
    imports the MAIN tree's extension unless the worktree sits ahead of it on
    ``sys.path``, and the row is then written with this tree's csrc digest
    against another tree's kernel.

    That is not hypothetical. It is a RECURRING defect with a recorded cost:
    P11's first cp.async A/B was discarded for exactly this reason ("the first
    worktree A/B and the wall-clock A/B were INVALID -- script-dir sys.path
    imported the main tree on both sides"). The failure mode is invisible in the
    output, because both arms measure the same binary and therefore simply
    AGREE -- a flat A/B is what it looks like, which is also what a true
    negative looks like.

    So it is a REFUSAL at the identity step rather than a line in a docstring,
    and it returns the resolved path so every row carries the binary it actually
    measured alongside the tree it claims to describe.
    """
    import rola

    root = Path(__file__).resolve().parents[2]
    pkg = Path(rola.__file__).resolve()
    try:
        pkg.relative_to(root)
    except ValueError:
        raise SystemExit(
            f"REFUSING TO MEASURE: `import rola` resolved to {pkg}, which is NOT "
            f"inside this checkout ({root}). The row would stamp this tree's sha "
            f"and csrc digest onto another tree's binary -- the invalid-A/B shape "
            f"P11 discarded a whole measurement cycle to. Put this checkout ahead "
            f"of the editable install: PYTHONPATH={root}") from None
    return str(pkg)


def _gpu_temp_c() -> float | None:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits",
         f"--id={torch.cuda.current_device()}"],
        capture_output=True, text=True)
    if out.returncode != 0:
        return None
    return float(out.stdout.strip().splitlines()[0])


class Run:
    """One harness invocation: its identity, its environment, and its row factory."""

    def __init__(self, *, tier: str, cmd: str):
        if tier not in TIERS:
            raise SystemExit(f"tier must be one of {TIERS}, not {tier!r}")
        from bench.regression import provenance

        self.tier = tier
        self.cmd = cmd
        self.run_id = uuid.uuid4().hex[:16]
        self.started = time.time()
        self.sha, self.dirty = _git_state()
        self.csrc_sha256 = _csrc_sha256()
        #: BEFORE the clock settles and before any row exists: an identity that
        #: does not describe the measured binary is worse than no identity.
        self.extension = _extension_in_this_checkout()
        self.temp_start = _gpu_temp_c()

        clocks = settle_clocks()
        if not clocks["settled"]:
            raise SystemExit(
                f"REFUSING TO MEASURE: the SM clock did not settle ({clocks}). "
                f"A number taken while the clock is ramping is a number about the "
                f"clock. Re-run once the box is idle; a genuine timeout means the "
                f"box is throttled or busy, which is a fact about the measurement, "
                f"not a condition to work around.")
        self.clocks = clocks
        self.env = dict(provenance(), settled=True, sm_mhz=clocks["sm_mhz"],
                        plateau_mhz=clocks["plateau_mhz"],
                        settle_elapsed_s=clocks["elapsed_s"],
                        extension=self.extension,
                        gpu_temp_c_start=self.temp_start)

    def row(self, **kwargs) -> dict:
        """A ledger row with this run's identity already filled in, so a probe cannot
        write one that omits it."""
        from bench.ledger import make_row

        env = dict(self.env, gpu_temp_c_end=_gpu_temp_c())
        return make_row(env=env, tier=self.tier, run_id=self.run_id,
                        csrc_sha256=self.csrc_sha256, sha=self.sha,
                        dirty=self.dirty, cmd=self.cmd, **kwargs)


@contextmanager
def disciplined(*, tier: str, cmd: str):
    """Lock, memory fraction, settled clock, stamped identity -- in that order.

    The memory fraction is applied INSIDE the lock and before the fixture is built,
    because it bounds this process's share of a device another process may still be
    releasing.
    """
    with gpu_lock():
        if not torch.cuda.is_available():
            raise SystemExit("no CUDA device; this harness measures a CUDA kernel")
        torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION)
        yield Run(tier=tier, cmd=cmd)
