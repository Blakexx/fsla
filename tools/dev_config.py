# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE DEV CONFIG (KERNEL_STANDARDS §23): every value that depends on the machine, read from one directory.

The directory is `~/.config/fsla/`, or the directory `FSLA_DEV_CONFIG` names (a test). It holds one JSON
file a section; every file and every key is declared in `SCHEMA` below with its type, default and meaning, and
anything undeclared is refused, so a typo fails loudly instead of silently taking a default. A missing file or key
takes the declared default. `bazel run //:init` writes the files for this machine, `bazel run //:show`
prints every resolved value with the file it came from.

The reader and the machine's own sections -- `host` (the locks, the compute budget, scratch) and `clock` -- are
fsla-devtools' (`fsla_devtools.config`), shared with the locks (`fsla_devtools.locks`) and every repository on the
host; the sections below them are fsla's. fsla-devtools is a development dependency, pinned in MODULE.bazel: the
graph's environment installs it from the module's source, and `bazel run //:init` links the checkout at
`workspace.devtools` into every venv here.

A setting is never read from the process environment. The environment variables the tree still uses are listed in
`HANDOFFS`, each a value one of the tree's own processes passes to a child it starts (a build's parameters are the
graph's settings, docs/build-graph.md#compile); `tools/lint/lint_standards.py` refuses a read of any other. Docs:
docs/internals/tools/dev_config.md.
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import cache
from pathlib import Path
from typing import Any

try:
    from fsla_devtools.config import (
        MACHINE,
        POINTER,
        Key,
        directory,  # noqa: F401 -- callers read dev_config.directory()
        read,
    )
    from fsla_devtools.config import reload as reload_machine
except ImportError as missing:
    raise SystemExit("tools/dev_config.py: fsla_devtools is not importable -- run from the checkout's venv "
                     "(`bazel run //:dev_env`), or put a fsla-devtools checkout on the path (docs/setup.md)") from missing

from bootstrap import DEVTOOLS, SIBLINGS  # noqa: E402 -- the defaults a pre-reader tool shares

ROOT = Path(__file__).resolve().parents[1]

SCHEMA: dict[str, dict[str, Key]] = {
    "toolchain": {
        "cuda_home": Key("path?", None,
                         "a CUDA toolkit the machine's toolchain file may name, for the checkouts on this machine that "
                         "read it; this tree uses the pinned one (`cuda_home()`). A loader refuses a key it does not "
                         "declare, so it stays declared while machine files carry it"),
        "ncu": Key("path?", None,
                   "a Nsight Compute CLI the machine's toolchain file may name, for the checkouts on this machine that "
                   "read it; this tree finds the pinned one by its pin (`ncu()`). A loader refuses a key it does not "
                   "declare, so it stays declared while machine files carry it"),
        "sccache": Key("path?", None,
                       "an sccache the machine's toolchain file may name, for the checkouts on this machine that read "
                       "it; this tree reads none. A loader refuses a key it does not declare, so it stays declared "
                       "while machine files carry it"),
        "sccache_enabled": Key("bool", True,
                               "the sccache switch the machine's toolchain file may hold, for the checkouts on this "
                               "machine that read it; this tree reads none. A loader refuses a key it does not declare, "
                               "so it stays declared while machine files carry it"),
        "mold": Key("path?", None,
                    "a mold the machine's toolchain file may name, for the checkouts on this machine that read it; this "
                    "tree links with the graph's (@mold). A loader refuses a key it does not declare, so it stays "
                    "declared while machine files carry it"),
        "clang_tidy": Key("path?", None, "a clang-tidy the machine's toolchain file may name, for the checkouts on this "
                          "machine whose commit hooks run it; this tree's unit lints run the graph's (@clang_tidy). A "
                          "loader refuses a key it does not declare, so it stays declared while machine files carry it"),
        "bazel": Key("path", "~/.local/bin/bazel", "where the bootstrap links `bazel`, the command the docs name, to the pinned "
                     "bazelisk (under host.tools_dir by its pin, never a machine key: every checkout loads this "
                     "machine's files, and an older loader refuses a key it does not declare): a path on PATH"),
    },
    **MACHINE,
    "workspace": {
        "suite": Key("path", str(SIBLINGS / "fsla-bench"),
                     "the fsla-bench checkout whose measurement suite (its root declare.py) runs this tree's declarations"),
        "devtools": Key("path", str(DEVTOOLS),
                        "the fsla-devtools checkout (the content checks, the declared build system), linked into "
                        "every venv that runs here"),
        "worktrees": Key("path", str(ROOT.parent / "worktrees") if ROOT.parent.name != "worktrees" else str(ROOT.parent),
                         "the one folder every worktree and its venv live in"),
        "hook_roots": Key("argv?", None,
                          "the folders under which the action-hook dispatcher (`.claude/hooks/repo_hooks.py`) runs a "
                          "checkout's tracked hooks; `init` writes them to the user's `~/.claude/repo_hooks_roots.json`, "
                          "which is derived and rewritten on every `init`. Null: this repository's checkouts and "
                          "workspace.worktrees"),
        "base_venv": Key("path?", None,
                         "a venv the machine's workspace file may name, for the checkouts on this machine whose hooks "
                         "run from it; this tree reads none. A loader refuses a key it does not declare, so it stays "
                         "declared while machine files carry it"),
    },
}

#: where WSL puts the driver's Linux libraries, which a container this tree starts sees at the same path (the devbox
#: image's loader searches it first)
WSL_LIB = "/usr/lib/wsl/lib"

#: Environment variables one of the tree's processes sets for a child it starts. Never a setting: the parent resolved
#: the value (from this config or from its own state) and the child must use exactly that value.
HANDOFFS = {
    POINTER: "which config directory a child reads (set for the graph's tests by the front door, or by a test; never by a setting)",
    "PYTEST_XDIST_WORKER": "pytest-xdist's worker id",
    "USER": "the login name printed into the sudoers rule `bazel run //:clock` suggests",
    "FSLA_SCRATCH": "the agent harness's campaign directory (.claude/settings.json), outside the repository",
    "FSLA_CUDA_HOME": "the graph's copy of the pinned CUDA toolkit, which a test tier hands its tests (fsla-devtools' pytest_tier)",
    "FSLA_DEVTOOLS": "the root of fsla-devtools' module in a test tier's runfiles, whose `pins/` a test reads (the tier's `repos`)",
    "FSLA_WARPLINE_PYTHON": "the directory that holds the `warpline` package (the reader of the lab's traced rows), "
                            "which the graph's actions that read compiled code hand it (bazel/defs.bzl's `lab_run` and gate action, "
                            "bazel/lab.bzl's validation action)",
    "PATH": "the search path bazelisk hands what it runs, the front door's directory first (`check` finds the user's "
            "`bazel` past it)",
    "FSLA_NVCC_THREADS": "host.nvcc_threads, which the front door hands the graph's compile flags (bazel/flags.py)",
    "FSLA_DATA": "the pinned data's root (@fsla_data, MODULE.bazel's `environment.data`), which the graph hands what reads "
                 "a stored record (bazel/defs.bzl's `lab_run`, a check's or a tier's `repos`; measure/data.py)",
}


@cache
def load() -> dict[str, dict[str, tuple[Any, str]]]:
    """Every section's every key as (value, source): the file it was read from, or "default"."""
    return read(SCHEMA, owns_directory=True)


def reload() -> None:
    """Forget what was read, here and in the locks' reader: the next read takes the directory `FSLA_DEV_CONFIG` names."""
    load.cache_clear()
    reload_machine()


def get(dotted: str) -> Any:
    section, name = dotted.split(".", 1)
    return load()[section][name][0]


@cache
def _graph_toolkit() -> str:
    """The graph's own copy of the pinned toolkit (`@cuda_tools`, fsla-devtools' pins/cuda_toolkit_pin.json): the graph fetches its
    repository and says where its nvcc is (`bazel query --output=location`)."""
    import shutil
    import subprocess

    bazel = shutil.which("bazel")
    if bazel is None:
        raise SystemExit("`bazel` is not on PATH: the toolkit is the graph's (@cuda_tools), which bazel fetches")
    found = subprocess.run([bazel, "query", "--output=location", "@cuda_tools//:bin/nvcc"], cwd=ROOT, capture_output=True,
                           text=True)
    if found.returncode != 0 or not found.stdout.strip():
        raise SystemExit(f"the graph's toolkit (@cuda_tools) is not there:\n{found.stderr}")
    nvcc = Path(found.stdout.split(":", 1)[0])
    return str(nvcc.parents[1])


def cuda_home() -> str:
    """The toolkit every compile, disassembly and assembler check outside the graph's own rules uses: the graph's
    (`@cuda_tools`, pinned by fsla-devtools' pins/cuda_toolkit_pin.json) -- handed in as `FSLA_CUDA_HOME` by a test tier, else read
    from the graph's repository."""
    if os.environ.get("FSLA_CUDA_HOME"):
        return os.environ["FSLA_CUDA_HOME"]
    return _graph_toolkit()


def warpline_python() -> str | None:
    """The directory that holds the `warpline` package, which reads the lab's traced rows: the pinned module's
    (`@warpline//:warpline`), which `bazel run //tools:lab` hands in (`FSLA_WARPLINE_PYTHON`) and a test tier carries as
    data, where it is the module's directory of the tier's runfiles. None where the process has none (a box that runs a
    pack outside Bazel: its binary writes the records, and nothing there reads them)."""
    found = os.environ.get("FSLA_WARPLINE_PYTHON")
    if not found and os.environ.get("TEST_SRCDIR"):
        found = str(Path(os.environ["TEST_SRCDIR"]) / "warpline+")
    return found


def cuda_bin(tool: str) -> str:
    return str(Path(cuda_home()) / "bin" / tool)


def scratch(name: str) -> Path:
    """A tool's scratch directory, this checkout's own: `host.scratch` / a short digest of the checkout's root path /
    `name`, so two checkouts on one machine never write the same intermediate (docs/internals/tools/dev_config.md)."""
    path = Path(get("host.scratch")) / hashlib.sha256(str(ROOT).encode()).hexdigest()[:12] / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_pin(name: str) -> dict:
    """A tool's pin, from fsla-devtools' `pins/<name>_pin.json` (`@fsla_devtools//pins:<name>_pin.json`) in the checkout
    `workspace.devtools` names, the very checkout the front door hands the graph as `@fsla_devtools`
    (`docs/build-graph.md#front-door`), so the tools and the graph read one file."""
    return json.loads((Path(get("workspace.devtools")) / "pins" / f"{name}_pin.json").read_text())


def pinned(name: str) -> Path:
    """Where a pin's binary is installed (`read_pin(name)`): its folder under `host.tools_dir` (`archive_root`,
    which carries the version), the binary in it (`binary_path`)."""
    pin = read_pin(name)
    return Path(get("host.tools_dir")) / pin["archive_root"] / pin["binary_path"]


def ncu() -> str:
    """The Nsight Compute CLI the measurement tools run: the pinned one, refused until init installed it."""
    path = pinned("nsight_compute")
    if not path.is_file():
        raise SystemExit(f"the pinned Nsight Compute is not installed at {path}: bazel run //:init")
    return str(path)


def windows_tool(rel: str) -> Path | None:
    system32 = get("host.windows_system32")
    return Path(system32) / rel if system32 else None


def detect() -> dict[str, dict[str, Any]]:
    """This machine's values where they differ from the declared defaults: what `bazel run //:init` writes."""
    found: dict[str, dict[str, Any]] = {"toolchain": {}, "host": {}, "store": {}, "workspace": {}}
    #: the measurements store (fsla-devtools' `store.root`, which names no default): the data repository's checkout beside
    #: the checkouts
    found["store"]["root"] = str(SIBLINGS / "fsla-results")
    system32 = Path("/mnt/c/Windows/System32")
    if system32.exists():
        found["host"]["windows_system32"] = str(system32)
    wsl_lib = Path(WSL_LIB)
    if (wsl_lib / "libdxcore.so").exists():
        found["host"]["wsl_lib"] = str(wsl_lib)
    return {k: v for k, v in found.items() if v}
