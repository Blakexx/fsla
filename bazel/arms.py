#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE ARM FAMILIES AS THE BUILD'S GATES READ THEM: each family's table (the Python file `bazel/arms.bzl` generates from
the kernel's declared list), a build's scope over it, and the checks of what a build carries against it -- the selection
header the units compiled against, and the binary asked what it carries. Docs: docs/build-graph.md#arms.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path


class Table:
    """A family's generated table: `key` (a row's fields) and `arms` (the rows, arm `i` everywhere)."""

    def __init__(self, path: Path):
        spec = importlib.util.spec_from_file_location(f"_arms_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.key: tuple[str, ...] = module.KEY
        self.arms: tuple = module.ARMS


def by_family(values: list[str], knob: str) -> dict[str, str]:
    """`family=value` arguments (`--table decode=<path>`), by family."""
    out: dict[str, str] = {}
    for value in values:
        family, sep, rest = value.partition("=")
        if not sep or not family:
            raise SystemExit(f"{knob}: {value!r} is not <family>=<value>")
        out[family] = rest
    return out


def is_iteration(contract: str, archs: list[str], built: list[str]) -> bool:
    """A build whose contract is a narrowed one (`--//:contract=<name>`) or whose architectures are not the built set is an ITERATION build:
    never a wheel, and self-identifying (docs/build.md#iteration). The arms it carries are those its contract reaches."""
    return contract != "full" or {f"sm_{a}" for a in archs} != set(built)


def check_built_arm_table(header: str | None, want: list[int], table: Table, family: str) -> list[str]:
    """The built arm table against the declared arms: the selection header the build compiled against names exactly the
    arms the build selected, and its counts and members are those rows. `header` is that header's text, `family` the
    family's name (`decode`). The findings, empty when both hold."""
    up = family.upper()
    if header is None:
        return [f"the build has no {family} selection header: the arm units compiled against something else"]
    named = sorted(int(m) for m in re.findall(rf"^#define FSLA_{up}_BUILD_(\d+) 1$", header, re.MULTILINE))
    if named != list(want):
        return [f"the selection header names {family} arms {named}, the build selected {list(want)}: the binary carries "
                "the header's set and not the build's"]
    count = re.search(rf"^#define FSLA_{up}_ARM_COUNT (\d+)$", header, re.MULTILINE)
    declared = re.search(rf"^#define FSLA_{up}_ARM_DECLARED (\d+)$", header, re.MULTILINE)
    if count is None or declared is None:
        return [f"the selection header carries no FSLA_{up}_ARM_COUNT/FSLA_{up}_ARM_DECLARED: `arm_switch` has no "
                "set to dispatch over and no way to tell an unbuilt arm from an undeclared one"]
    if int(count.group(1)) != len(want) or int(declared.group(1)) != len(table.arms):
        return [f"the selection header declares {count.group(1)} built / {declared.group(1)} declared arms; the build "
                f"selected {len(want)} of {len(table.arms)} declared"]
    members = sorted(tuple(int(x) for x in m.split(", ")) for m in re.findall(r"^  F\(([\d, ]+)\)", header,
                                                                                re.MULTILINE))
    if members != sorted((i,) + tuple(table.arms[i]) for i in want):
        return [f"the arm set names {members}, which is not the build's rows of the declaration"]
    return []


#: THE COVERAGE SWEEP'S PROBE, run in its own interpreter against a built extension LOADED BY PATH: the fact wanted is
#: a property of that artifact, and `import fsla` would resolve through whatever the environment has installed
_ARM_SWEEP_PROBE = """
import importlib.util, json, sys
from pathlib import Path
import torch  # noqa: F401 -- the extension links against it and must load first
name = Path(sys.argv[1]).name.split(".")[0]  # the module's init symbol is PyInit_<name>
spec = importlib.util.spec_from_file_location(name, sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)  # loading the library registers torch.ops.fsla
print(json.dumps([list(row) for row in getattr(torch.ops.fsla, f"fsla_{sys.argv[2]}_arms")()]))
"""


def sweep_binary(python: str, so: Path, family: str, want: list[int], table: Table, cwd: Path,
                 env: dict | None = None) -> int:
    """R9's fourth layer, host side: the binary is asked what it carries (`python`, an interpreter with torch, loads it
    by path in `cwd`; the family's own op, `fsla_<family>_arms`), and the answer must be exactly the rows the build selected -- the table, the selection header and
    the binary are three statements of one set. The count swept; raises on any difference."""
    import subprocess

    proc = subprocess.run([python, "-c", _ARM_SWEEP_PROBE, str(so), family], capture_output=True, text=True, cwd=str(cwd),
                          env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"the arm coverage sweep could not load {so}:\n{proc.stderr}")
    got = sorted(tuple(row) for row in json.loads(proc.stdout))
    expect = sorted(tuple(table.arms[i]) for i in want)
    if got != expect:
        raise RuntimeError(f"{so.name} reports the {family} arms {[list(r) for r in got]}, the build selected "
                           f"{[list(r) for r in expect]}: a binary that cannot name what it carries cannot refuse "
                           "what it does not")
    return len(got)
