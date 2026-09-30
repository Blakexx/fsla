"""The SASS gate (`tools/sass_gate.py`) on two compiled probes: one clean kernel, and one whose runtime-indexed local
array must go to local memory -- the gate's reds are what it reads, proven on a cubin that has them.

See KERNEL_STANDARDS §20.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import dev_config  # noqa: E402
import sass_gate  # noqa: E402

PROBE = r"""
__global__ void clean_probe(float* y, const float* x) { y[threadIdx.x] = 2.0f * x[threadIdx.x]; }

__global__ void local_probe(float* y, const int* i) {
  float a[512];
  for (int k = 0; k < 512; ++k) a[k] = k * y[k];
  a[i[threadIdx.x + 32]] = y[threadIdx.x];
  y[threadIdx.x] = a[i[threadIdx.x]];
}
"""


@pytest.fixture(scope="module")
def cubin(tmp_path_factory):
    nvcc = dev_config.cuda_bin("nvcc")
    if nvcc is None or not Path(nvcc).exists():
        pytest.skip("no nvcc in this environment's toolkit")
    root = tmp_path_factory.mktemp("sass_gate")
    (root / "probe.cu").write_text(PROBE)
    subprocess.run([nvcc, "-cubin", "-arch=sm_86", "-O3", "-o", str(root / "probe.cubin"), str(root / "probe.cu")],
                   check=True, capture_output=True)
    return root / "probe.cubin"


def test_the_gate_passes_the_clean_kernel_and_reds_local_memory(cubin):
    ok, report = sass_gate.gate(cubin, None, None)
    clean = next(v for k, v in sass_gate.STATS.items() if "clean_probe" in k)
    local = next(v for k, v in sass_gate.STATS.items() if "local_probe" in k)
    assert not clean["reds"], report
    assert local["stack"] > 0 and local["ldl"] > 0, report
    assert any("stack frame" in r for r in local["reds"]) and not ok


def test_a_narrowing_that_matches_nothing_is_red(cubin, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sass_gate.py", str(cubin), "--kernel", "no_such_kernel"])
    assert sass_gate.main() == 1
    assert "no function matched" in capsys.readouterr().out
