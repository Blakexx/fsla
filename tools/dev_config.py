# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE DEV CONFIG (KERNEL_STANDARDS §23): every value that depends on the machine, read from one directory.

The directory is `~/.config/rola/`, or the directory `ROLA_DEV_CONFIG` names (a container, a test). It holds one JSON
file a section; every file and every key is declared in `SCHEMA` below with its type, default and meaning, and
anything undeclared is refused, so a typo fails loudly instead of silently taking a default. A missing file or key
takes the declared default. `python tools/dev.py init` writes the files for this machine, `python tools/dev.py show`
prints every resolved value with the file it came from.

A setting is never read from the process environment. The environment variables the tree still uses are listed in
`HANDOFFS` (a value one of the tree's own processes passes to a child it starts) and `BUILD_PARAMETERS` (what one build
is, set per invocation); `tools/lint/lint_standards.py` refuses a read of any other. Docs:
docs/internals/tools/dev_config.md.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POINTER = "ROLA_DEV_CONFIG"


@dataclass(frozen=True)
class Key:
    kind: str
    default: Any
    doc: str


def _repo_default(rel: str) -> str:
    return str(ROOT / rel)


SCHEMA: dict[str, dict[str, Key]] = {
    "toolchain": {
        "cuda_home": Key("path?", None, "the CUDA toolkit the extension builds with (nvcc, ptxas, nvdisasm, cuobjdump "
                         "under bin/); null takes torch.utils.cpp_extension's CUDA_HOME"),
        "ncu": Key("path", "/usr/local/cuda/bin/ncu",
                   "the Nsight Compute CLI; 2025.3 or later for PM sampling (tools/pipe_timeline.py)"),
        "sccache": Key("path?", None, "the pinned sccache (tools/sccache_pin.json); null means none installed"),
        "sccache_enabled": Key("bool", True, "false builds uncached, the one deliberate opt-out of the sccache gate"),
        "mold": Key("path?", None, "the pinned mold (tools/mold_pin.json), the only linker; null means none installed"),
    },
    "host": {
        "nvcc_threads": Key("int", 2, "nvcc -t for each compile job"),
        "build_jobs": Key("int?", None, "ninja jobs a build asks for; null derives it from nvcc_threads and memory"),
        "budget_slots": Key("int?", None, "the machine-wide compute slots every build and census compile shares; "
                            "null derives cores - 2 bounded by memory (tools/host_budget.py)"),
        "lock_dir": Key("path", "/tmp", "where the host budget's slot locks live; host and containers must share it"),
        "gpu_lock": Key("path", "/tmp/rola_gpu.lock", "the GPU lock file; host and containers must share it"),
        "gpu_shared_slots": Key("int", 2, "correctness runs that may hold the GPU lock shared at once"),
        "nice": Key("bool", True, "locked processes lower their own CPU and IO priority"),
        "locks_trace": Key("bool", False, "print every lock acquire and release to stderr"),
        "scratch": Key("path", "/tmp/rola", "the tools' scratch root; never read back as a record"),
        "tools_dir": Key("path", "~/.local/share/rola/tools", "where `tools/dev.py init` installs the pinned tools"),
        "windows_system32": Key("path?", None, "WSL only: the Windows System32 directory as mounted in Linux"),
    },
    "clock": {
        "ghz": Key("float?", None, "the SM clock the harness locks, in GHz; null means this host runs unlocked"),
        "lock": Key("argv?", None, "the command that locks the clock"),
        "unlock": Key("argv?", None, "the command that releases it"),
    },
    "store": {
        "results": Key("path", _repo_default("benchmarks/results"), "the measurements record every writer appends to"),
        "db": Key("path", _repo_default(".rola/measurements.sqlite"), "the derived index tools/rola_db.py rebuilds"),
    },
    "workspace": {
        "worktrees": Key("path", str(ROOT.parent / "worktrees") if ROOT.parent.name != "worktrees" else str(ROOT.parent),
                         "the one folder every worktree and its venv live in"),
        "base_venv": Key("path?", None, "the shared venv every worktree's pointer venv borrows site-packages from"),
    },
    "environment": {
        "image_digest": Key("str", "", "inside the dev container: sha256 of the Dockerfile the image was built from"),
        "fla_crosscheck": Key("bool", False, "flash-linear-attention is installed: run the oracle's one-time FLA check"),
        "machine_salt": Key("str?", None, "an operator's secret salt; set, rows carry a salted hash telling two "
                            "same-profile machines apart (benchmarks/bench/hw_profile.py)"),
    },
}

#: Environment variables one of the tree's processes sets for a child it starts. Never a setting: the parent resolved
#: the value (from this config or from its own state) and the child must use exactly that value.
HANDOFFS = {
    POINTER: "which config directory a child reads (set by a test or the container, never by a setting)",
    "ROLA_HOST_BUDGET_HELD": "a parent holds host budget slots; a nested acquire is a no-op (tools/host_budget.py)",
    "ROLA_GPU_LOCK_HELD": "a parent holds the GPU lock; a nested acquire is a no-op (tools/gpu_lock.py)",
    "ROLA_FILE_LOCK_HELD_": "prefix: a parent holds the named file lock (tools/host_budget.py file_lock)",
    "ROLA_REAL_NVCC": "the gated nvcc setup.py hands the sccache wrapper (tools/sccache_nvcc.py)",
    "ROLA_SCCACHE_BIN": "the gated sccache setup.py hands the sccache wrapper",
    "CUDA_HOME": "torch.utils.cpp_extension's toolkit, set by setup.py from toolchain.cuda_home before torch is imported",
    "MAX_JOBS": "torch's ninja job count, set by setup.py from the slots it holds",
    "PYTORCH_NVCC": "torch's compiler, set by setup.py to the sccache wrapper",
    "PYTEST_XDIST_WORKER": "pytest-xdist's worker id",
    "USER": "the login name printed into the sudoers rule `tools/dev.py clock` suggests",
    "ROLA_SCRATCH": "the agent harness's campaign directory (.claude/settings.json), outside the repository",
}

#: Environment variables that say what one build is. Set per invocation (pip passes nothing else to setup.py).
BUILD_PARAMETERS = {
    "ROLA_CUDA_ARCHS": "the architectures to compile; unset reads the ratified manifests",
    "ROLA_CARRY_ARMS": "the carry arm subset of an iteration build",
    "ROLA_DECODE_ARMS": "the decode arm subset of an iteration build",
    "ROLA_CARRY_PARTS": "the carry parts a composition build makes real (tools/gen_shards.py CARRY_PARTS)",
    "ROLA_BUILD_PARTS": "the source families a build compiles",
    "ROLA_NO_EXTENSION": "install the Python package without building the extension",
    "ROLA_STRICT_MANIFEST": "refuse a build whose manifest does not match exactly (§16 tiering)",
    "ROLA_SKIP_POST_BUILD_RATIFY": "skip the post-build ratification recompile",
}


def directory() -> Path:
    pointer = os.environ.get(POINTER)
    return Path(pointer).expanduser() if pointer else Path.home() / ".config" / "rola"


def _check(section: str, name: str, key: Key, value: Any) -> Any:
    kind, optional = key.kind.rstrip("?"), key.kind.endswith("?")
    where = f"{directory() / (section + '.json')}: {name}"
    if value is None:
        if optional:
            return None
        raise SystemExit(f"{where} may not be null ({key.doc})")
    ok = {"path": isinstance(value, str), "str": isinstance(value, str), "int": isinstance(value, int)
          and not isinstance(value, bool), "float": isinstance(value, (int, float)) and not isinstance(value, bool),
          "bool": isinstance(value, bool), "argv": isinstance(value, list) and all(isinstance(x, str) for x in value)}
    if not ok[kind]:
        raise SystemExit(f"{where} must be {kind}, got {value!r} ({key.doc})")
    return str(Path(value).expanduser()) if kind == "path" else value


@cache
def load() -> dict[str, dict[str, tuple[Any, str]]]:
    """Every section's every key as (value, source): the file it was read from, or "default"."""
    root = directory()
    if root.exists():
        stray = sorted(p.name for p in root.glob("*.json") if p.stem not in SCHEMA)
        if stray:
            raise SystemExit(f"{root}: undeclared config file(s) {stray}; declared: {sorted(SCHEMA)}")
    out: dict[str, dict[str, tuple[Any, str]]] = {}
    for section, keys in SCHEMA.items():
        path = root / f"{section}.json"
        doc = json.loads(path.read_text()) if path.exists() else {}
        unknown = sorted(set(doc) - set(keys))
        if unknown:
            raise SystemExit(f"{path}: undeclared key(s) {unknown}; declared: {sorted(keys)}")
        out[section] = {name: (_check(section, name, key, doc[name]), str(path)) if name in doc
                        else (_check(section, name, key, key.default), "default") for name, key in keys.items()}
    return out


def get(dotted: str) -> Any:
    section, name = dotted.split(".", 1)
    return load()[section][name][0]


def cuda_home() -> str:
    """The toolkit every compile, disassembly and assembler check uses: the configured one, else torch's."""
    home = get("toolchain.cuda_home")
    if home is None:
        try:
            from torch.utils.cpp_extension import CUDA_HOME as home
        except ImportError:
            home = None
    if not home:
        raise SystemExit("no CUDA toolkit: set toolchain.cuda_home (python tools/dev.py init)")
    return home


def cuda_bin(tool: str) -> str:
    return str(Path(cuda_home()) / "bin" / tool)


def scratch(name: str) -> Path:
    path = Path(get("host.scratch")) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def windows_tool(rel: str) -> Path | None:
    system32 = get("host.windows_system32")
    return Path(system32) / rel if system32 else None


def detect() -> dict[str, dict[str, Any]]:
    """This machine's values where they differ from the declared defaults: what `tools/dev.py init` writes."""
    found: dict[str, dict[str, Any]] = {"toolchain": {}, "host": {}, "workspace": {}, "environment": {}}
    nvcc = shutil.which("nvcc")
    if nvcc:
        found["toolchain"]["cuda_home"] = str(Path(nvcc).parents[1])
    ncus = sorted(Path("/usr/local").glob("cuda-*/bin/ncu"),
                  key=lambda p: tuple(int(x) for x in p.parts[3].removeprefix("cuda-").split(".") if x.isdigit()))
    if ncus:
        found["toolchain"]["ncu"] = str(ncus[-1])
    for tool in ("sccache", "mold"):
        if shutil.which(tool):
            found["toolchain"][tool] = shutil.which(tool)
    system32 = Path("/mnt/c/Windows/System32")
    if system32.exists():
        found["host"]["windows_system32"] = str(system32)
    return {k: v for k, v in found.items() if v}
