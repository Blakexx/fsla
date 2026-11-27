# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""FSLA'S EXECUTORS: what fsla's declared targets (`declare.py`) run, in this checkout's environment.

The build system's worker runs them from this checkout's directory with `.`, `measure` and `tools` on the path.
Each takes the target's context (`fsla_devtools.build.context.Context`) and returns its JSON output, keeping raw files
in its workspace:

    compile_kernel      the extension, by the build graph (`bazel run //:library`: the gated build, copied into the
                        checkout; its front door takes the host budget itself); its output is the binary's path in the
                        checkout and its sha256
    probe_environment   the machine facts a number depends on (GPU and driver, assembler, torch, Nsight Compute, the
                        clock the host locks), read from the machine and the dev config, never from a path
    timed               a timing entry: this checkout's runner (`measure.provider`) arm on a cell, as a Timed with its reset
    read_clock          the SM clock as this binary reads it, for the timing system's proof
"""
from __future__ import annotations

import subprocess
from importlib import metadata
from pathlib import Path

CHECKOUT = Path(__file__).resolve().parents[1]
#: seconds a build may take
BUILD_TIMEOUT_S = 6 * 3600


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    from fsla_devtools.process import run

    return run(cmd, cwd=CHECKOUT, timeout=timeout)


def _portable(text: str) -> str:
    return text.replace(str(CHECKOUT) + "/", "").replace(str(Path.home()) + "/", "~/")


def binary() -> Path:
    sos = sorted((CHECKOUT / ".bazel" / "bin").glob("fsla_*/_C*.so"))
    if not sos:
        raise RuntimeError(f"no built extension in {CHECKOUT.name} (.bazel/bin/fsla_<toolchain>/_C*.so)")
    return sos[0]


def compile_kernel(ctx) -> dict:
    from fsla_devtools.build.identity import digest

    done = _run(["bazel", "run", "--noshow_progress", "//:library"], BUILD_TIMEOUT_S)
    (ctx.workspace / "build.log").write_text(_portable(done.stdout[-50000:] + done.stderr[-50000:]))
    if done.returncode:
        raise RuntimeError(f"the build exited {done.returncode}:\n{_portable((done.stdout + done.stderr)[-3000:])}")
    so = binary()
    return {"binary": so.relative_to(CHECKOUT).as_posix(), "sha256": digest(so)}


def _version(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def probe_environment(ctx) -> dict:
    import dev_config

    smi = _version(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"]).strip()
    ptxas = _version([dev_config.cuda_bin("ptxas"), "--version"]).split()
    #: provenance only: a host without the pinned Nsight Compute records none
    try:
        ncu = _version([dev_config.ncu(), "--version"])
    except SystemExit:
        ncu = ""
    return {"gpu": smi, "ptxas": next((w for w in ptxas if w.startswith("V")), " ".join(ptxas[-2:])),
            "torch": metadata.version("torch"),
            "ncu": next((line.split("Version ")[1].split()[0] for line in ncu.splitlines() if "Version " in line), ""),
            "clock_ghz": dev_config.get("clock.ghz")}


def timed(cell, params):
    from measure.provider import arms

    offered = arms(cell)
    if params["arm"] not in offered:
        raise LookupError(f"{cell.name}: this checkout runs no arm {params['arm']} there; it runs {sorted(offered)}")
    return offered[params["arm"]]()


def read_clock() -> float | None:
    from fsla.ops import carry

    return carry.sm_clock_ghz()
