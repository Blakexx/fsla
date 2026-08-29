"""K38 -- the measurements record: the writer epilogue every measuring tool shares.

**THE RECORD IS COMMITTED JSONL, THE DATABASE IS DERIVED.** One small file per run,
under `benchmarks/results/<hw-profile>/<yyyy-mm>/<run_id>.jsonl` (text, mergeable,
provenance in every row). `.rola/measurements.sqlite` is a gitignored cache rebuilt
FROM that JSONL by `tools/rola_db.py rebuild`; it is never itself the record
(`docs/measurement.md`, "a committed SQLite database... a binary blob has no diff").

**ONE EPILOGUE, TWO CALLERS.** `tools/probe_cells.py` (the carry-family perf probe)
and `benchmarks/bench/driver.py` (every `benchmarks/unit`/`benchmarks/integration`
driver) both end their measurement by calling :func:`write_run` -- the schema, the
hardware-profile keying and the provenance refusal live in exactly one place so the
two writers cannot drift into two shapes the way two throwaway ncu scripts once
drifted on their metric list.

**A ROW WITHOUT PROVENANCE IS REFUSED, NOT WRITTEN THIN.** `git_sha`, the device
stamp (`manifest_sha256` -- the binary's own account of itself, K38 KERNEL_STANDARDS
"the extension trap") and `harness_sha` (the identity of the MEASURING CODE, distinct
from the kernel it measures) are required on every run record; :func:`build_run`
raises rather than writing an unusable row, mirroring `bench.ledger.make_row`'s own
`REQUIRED` check.

**NOTHING IDENTIFIABLE IS CHECKED IN (RULED, 2026-08-28).** The path key is the
hardware profile (`bench.hw_profile.hw_profile`: GPU model + SM arch), never a
hostname, username or machine id. `machine_tag` is `None` unless an operator has
opted a salted hash in (`bench.hw_profile.machine_tag`).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

from bench.hw_profile import hw_profile, machine_tag

ROOT = Path(__file__).resolve().parents[2]

#: Bump only for a change that makes OLD LINES UNREADABLE (adding a field does not
#: qualify -- readers ignore what they do not know). Mirrors `bench.ledger.SCHEMA`'s
#: own rule, kept as a SEPARATE number because the two records are separate schemas.
SCHEMA = 1

#: The kinds `tools/probe_cells.py` and the unit/integration drivers can write. The
#: sketch in K38's card names the full roster this table is meant to grow into
#: (ncu sections, the census, the compile ledger, ratify, oracle) even though only
#: `probe` has a live writer as of this stage.
KINDS = ("probe", "ncu", "census", "compile", "oracle", "ratify")

#: A run record without every one of these is not a weaker record, it is an
#: unusable one: nobody could later tell what binary, what tree, or what measuring
#: code produced it. The writer refuses rather than the reader guessing.
REQUIRED_RUN_FIELDS = ("run_id", "ts", "git_sha", "stamp_magic", "harness_sha")


def results_dir() -> Path:
    """`ROLA_RESULTS_DIR`, repo-relative default `benchmarks/results` (§17: no
    ad-hoc or home-rooted default -- an override is the operator's choice, not a
    convenience path this module invents)."""
    override = os.environ.get("ROLA_RESULTS_DIR")
    return Path(override).resolve() if override else ROOT / "benchmarks" / "results"


def git_state() -> tuple[str, bool]:
    """`(sha, dirty)` of the tree being measured. Shared with `bench.ledger` by
    re-deriving the same two `git` calls rather than importing across the two
    record modules, which would otherwise couple K38's schema to the retired one."""
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    porcelain = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout
    return sha, bool(porcelain.strip())


def branch_name() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def harness_sha256(*paths: str | Path) -> str:
    """The identity of the MEASURING CODE, as opposed to `csrc_sha256`'s identity of
    the kernel being measured. Hashing the harness file(s) means a rewrite of the
    probe itself starts a new comparable identity rather than silently reusing the
    old one's rows under an unchanged label."""
    h = hashlib.sha256()
    for p in sorted(str(Path(p).resolve()) for p in paths):
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def device_stamp() -> dict:
    """The binary's OWN account of itself (`rola._build_config.BUILD_CONFIG`), read
    device-side rather than off a path -- a stale `.so` at the right path is the
    extension trap a path check cannot catch (`extension-trap-device-side-check`)."""
    from rola._build_config import BUILD_CONFIG

    return {"manifest_sha256": BUILD_CONFIG["manifest_sha256"], "version": BUILD_CONFIG["version"]}


def env_stamp() -> dict:
    """GPU model, SM, driver/CUDA runtime, torch, assembler -- normalized the same
    way `bench.regression.provenance` is (O-11): no hostname, no username, no path."""
    import torch

    from rola._build_config import BUILD_CONFIG

    props = torch.cuda.get_device_properties(0)
    return {
        "gpu": props.name,
        "sm": f"sm_{props.major}{props.minor}",
        "driver_cuda": torch.version.cuda,
        "torch": torch.__version__,
        "ptxas": BUILD_CONFIG["ptxas"],
    }


def build_run(*, harness_paths, stage: str | None = None, agent: str | None = None,
              purpose: str | None = None, arm_census_hash: str | None = None,
              env: dict | None = None, git_sha: str | None = None, dirty: bool | None = None,
              branch: str | None = None, worktree_name: str | None = None,
              stamp_magic: str | None = None) -> dict:
    """One `runs` record. Raises if a required provenance field cannot be filled --
    a row missing `git_sha`, the device stamp, or `harness_sha` is refused at the
    source, not silently written thin and caught later by a reader.

    Every identity field is OVERRIDABLE (`git_sha`, `dirty`, `branch`,
    `worktree_name`, `stamp_magic`, `env`) because `tools/probe_cells.py` measures
    OTHER worktrees through their own venv pythons in a subprocess: the caller's
    `import rola` describes the caller's tree, not the binary actually launched, so
    a cross-tip A/B must pass each binary's own identity in rather than have this
    function re-derive the wrong one from the current process."""
    if git_sha is None or dirty is None:
        sha, dirty = git_state()
    else:
        sha = git_sha
    if branch is None:
        branch = branch_name()
    if worktree_name is None:
        worktree_name = ROOT.name
    stamp_magic = stamp_magic if stamp_magic is not None else device_stamp()["manifest_sha256"]
    env = env_stamp() if env is None else env
    run = {
        "schema": SCHEMA,
        "kind": "run",
        "run_id": uuid.uuid4().hex[:16],
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hw_profile": hw_profile(env["gpu"], env["sm"]),
        "machine_tag": machine_tag(),
        "gpu": env["gpu"],
        "driver": env["driver_cuda"],
        "cuda": env["driver_cuda"],
        "image_digest": os.environ.get("ROLA_IMAGE_DIGEST", ""),
        "git_sha": sha,
        "branch": branch,
        "worktree": worktree_name,
        "dirty": dirty,
        "stamp_magic": stamp_magic,
        "arm_census_hash": arm_census_hash,
        "harness_sha": harness_sha256(*harness_paths),
        "stage": stage,
        "agent": agent,
        "purpose": purpose,
        "env": env,
    }
    missing = [k for k in REQUIRED_RUN_FIELDS if not run.get(k)]
    if missing:
        raise ValueError(f"a K38 run record is missing required field(s) {missing}; "
                          f"refusing to write an unusable row (docs/measurement.md)")
    return run


def build_result(run: dict, *, kind: str, cell: str | None, arm: str | None,
                 metric: str, value: float, unit: str, extra: dict | None = None) -> dict:
    if kind not in KINDS:
        raise ValueError(f"result kind {kind!r} is not one of {KINDS}")
    return {
        "schema": SCHEMA,
        "kind": kind,
        "run_id": run["run_id"],
        "cell": cell,
        "arm": arm,
        "metric": metric,
        "value": float(value),
        "unit": unit,
        "extra_json": json.dumps(extra or {}, sort_keys=True),
    }


def write_run(run: dict, results: list) -> Path:
    """Append the run record and its results to
    `benchmarks/results/<hw-profile>/<yyyy-mm>/<run_id>.jsonl`. One file per run, so
    two branches that both measured never conflict on the same line the way one
    shared ledger file would; `.gitattributes`' `merge=union` is not needed here
    because nothing ever shares a path."""
    month = run["ts"][:7]
    out_dir = results_dir() / run["hw_profile"] / month
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run['run_id']}.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(run, sort_keys=True) + "\n")
        for r in results:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    return out_path
