"""The pinned CUDA toolkit (tools/cuda_toolkit_pin.json) is the graph's one: the graph's CUDA compile (bazel/cuda.bzl) takes nvcc,
its headers and the runtime from it (`@cuda_tools`), so every component the compile reads is one of the pin's, and its compiler is
a declared toolchain. The nvcc and ptxas outside the graph's rules are the graph's own. No CUDA, no GPU."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import toolchains  # noqa: E402

PIN = json.loads((ROOT / "tools" / "cuda_toolkit_pin.json").read_text())

#: what the compile and the link read: nvcc, ptxas, fatbinary, cudafe++ (cuda_nvcc), cicc and libdevice (libnvvm), the runtime's
#: headers and libraries (cuda_cudart, cuda_crt, cuda_cccl, cuda_culibos)
_COMPILE_COMPONENTS = ["cuda_nvcc", "libnvvm", "cuda_cudart", "cuda_crt", "cuda_cccl", "cuda_culibos"]


def test_the_manifest_is_the_pins():
    assert PIN["manifest_url"].endswith(f"/redistrib_{PIN['version']}.json")


def test_every_component_the_graph_compiles_with_is_pinned():
    missing = [c for c in _COMPILE_COMPONENTS if c not in PIN["components"]]
    assert not missing, missing


def test_the_pinned_compiler_is_a_declared_toolchain():
    assert any(PIN["version_string"] in toolchain.ptxas for toolchain in toolchains.records().values())
