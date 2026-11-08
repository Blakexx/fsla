#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE GRAPH'S FRONT DOOR: bazelisk runs `bazel/bazel` -- this file, the wrapper `.bazeliskrc` names -- in place of
the Bazel it downloaded (`BAZEL_REAL`), so every Bazel command in a checkout goes through here, typed or run by a tool.

It hands Bazel the machine's settings from the dev config (the shared disk and repository caches under
`host.build_cache`, as an rc file read after the workspace's), and it runs every BUILD under the host budget
(`rola_devtools.locks.host`): a command's build phase holds the slots free when it starts, at least one, and runs one
compile job a slot pair; its test or run phase holds none, so a test or a measurement takes its own locks as it does
outside the graph -- a GPU test under a held budget is the deadlock the host's lock order exists to prevent.

Runs under the machine's `python3` before any venv exists: the reader and the budget come from the rola-devtools
checkout (`tools/bootstrap.py`). Docs: docs/build-graph.md#front-door.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import bootstrap  # noqa: E402 -- the path insert above must precede it

sys.path.insert(1, str(bootstrap.devtools()))
import dev_config  # noqa: E402
from rola_devtools.locks import host  # noqa: E402

#: the commands whose whole run is a build, and those whose build comes before a phase that holds no budget
BUILDS = ("build",)
TESTS = ("test", "coverage")
#: the test command's options the build command refuses: they shape only the test summary
TEST_SUMMARY_ONLY = ("--test_verbose_timeout_warnings", "--notest_verbose_timeout_warnings",
                     "--print_relative_test_log_paths", "--noprint_relative_test_log_paths")


def _split(argv: list[str]) -> tuple[list[str], str | None, list[str]]:
    """(startup options, the command, its arguments): the command is the first word that is not an option, so a
    startup option is written `--name=value`."""
    for i, word in enumerate(argv):
        if not word.startswith("-"):
            return argv[:i], word, argv[i + 1:]
    return argv, None, []


def machine_rc() -> Path:
    """The machine's settings as an rc file, regenerated from the dev config on every run."""
    cache = Path(dev_config.get("host.build_cache")) / "bazel"
    text = (f"# written by bazel/front_door.py from the dev config's host.build_cache; regenerated every run\n"
            f"common --disk_cache={cache / 'disk'}\n"
            f"common --repository_cache={cache / 'repository'}\n")
    path = cache / "machine.bazelrc"
    if not path.is_file() or path.read_text() != text:
        cache.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=cache, prefix=".machine.bazelrc.")
        with os.fdopen(fd, "w") as out:
            out.write(text)
        os.replace(tmp, path)
    return path


def _wait(argv: list[str]) -> int:
    """Run Bazel's client to its end: an interrupt reaches it too (one process group), and it stops its server's work."""
    proc = subprocess.Popen(argv)
    while True:
        try:
            return proc.wait()
        except KeyboardInterrupt:
            continue


def under_budget(head: list[str], command: str, rest: list[str]) -> int:
    """The command under the host budget: the slots free now (at least one), one compile job a slot pair -- nvcc runs
    `host.nvcc_threads` cicc at once, one an architecture."""
    with host.acquire(host.budget_slots(), label=f"bazel {command}") as held:
        jobs = max(1, held // dev_config.get("host.nvcc_threads"))
        return _wait([*head, command, f"--jobs={jobs}", *rest])


def main(argv: list[str]) -> int:
    real = os.environ.get("BAZEL_REAL")
    if not real:
        raise SystemExit("bazel/bazel is the graph's front door, which bazelisk runs in Bazel's place: run `bazel`, the "
                         "pinned bazelisk (toolchain.bazelisk; python tools/dev.py init)")
    startup, command, rest = _split(argv)
    head = [real, *startup, f"--bazelrc={machine_rc()}"]
    if command in BUILDS:
        return under_budget(head, command, rest)
    if command in TESTS:
        built = under_budget(head, "build", [w for w in rest if w.split("=")[0] not in TEST_SUMMARY_ONLY])
        if built:
            return built
    elif command == "run" and not any(w.startswith("--script_path") for w in rest):
        #: the run's build under the budget writes the launch script, which then runs holding nothing
        scratch = Path(dev_config.get("host.scratch"))
        scratch.mkdir(parents=True, exist_ok=True)
        fd, script = tempfile.mkstemp(prefix="bazel_run_", suffix=".sh", dir=scratch)
        os.close(fd)
        built = under_budget(head, command, [f"--script_path={script}", *rest])
        text = Path(script).read_text() if not built else ""
        os.unlink(script)
        if built:
            return built
        os.execv("/bin/bash", ["/bin/bash", "-c", text])
    elif command == "run":
        return under_budget(head, command, rest)
    os.execv(real, [*head, *([command] if command else []), *rest])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
