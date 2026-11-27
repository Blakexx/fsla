"""The pinned CUDA toolkit (fsla-devtools' pins/cuda_toolkit_pin.json, FSLA_DEVTOOLS the module's root) is the graph's one: the graph's CUDA compile (bazel/cuda.bzl) takes nvcc,
its headers and the runtime from it (`@cuda_tools`), so the compile reads only the pin's own unpacked tree, and its compiler is
a declared toolchain. The nvcc and ptxas outside the graph's rules are the graph's own. No CUDA, no GPU."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import toolchains  # noqa: E402


def pin() -> dict:
    """The pin, read when a test runs: the CPU tier hands `FSLA_DEVTOOLS`; a tier that only collects this file need not."""
    return json.loads((Path(os.environ["FSLA_DEVTOOLS"]) / "pins" / "cuda_toolkit_pin.json").read_text())


def test_the_manifest_is_the_pins():
    p = pin()
    assert p["manifest_url"].endswith(f"/redistrib_{p['version']}.json")


def test_the_pinned_compiler_is_a_declared_toolchain():
    assert any(pin()["version_string"] in toolchain.ptxas for toolchain in toolchains.records().values())
