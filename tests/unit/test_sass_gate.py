"""The SASS gate (`tools/sass_gate.py`) on compiled probes: a clean kernel, one whose runtime-indexed local array must
go to local memory, one whose shuffle sits in a loop ptxas cannot prove warp-uniform (a convergence subroutine), and
one whose division calls its slow path (a call that is not one) -- the gate's reds are what it reads, proven on a cubin
that has them.

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

__global__ void converge_probe(float* y, const int* n) {
  float v = y[threadIdx.x];
  for (int i = 0; i < n[threadIdx.x]; ++i) v += __shfl_xor_sync(0xffffffffu, v, 1);
  y[threadIdx.x] = v;
}

__global__ void divide_probe(float* y, const float* x) { y[threadIdx.x] = x[threadIdx.x] / y[threadIdx.x + 32]; }
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


def test_the_gate_reds_a_convergence_subroutine_and_passes_a_slow_path_call(cubin):
    ok, report = sass_gate.gate(cubin, None, None)
    converge = next(v for k, v in sass_gate.STATS.items() if "converge_probe" in k)
    divide = next(v for k, v in sass_gate.STATS.items() if "divide_probe" in k)
    assert converge["calls"] == 1 and any("convergence" in r for r in converge["reds"]), report
    assert divide["calls"] == 0 and not divide["reds"], report


def test_the_backlog_holds_a_function_at_its_recorded_count_and_asks_to_shrink_below_it(cubin, capsys, monkeypatch):
    sass_gate.gate(cubin, None, None)
    name = next(k for k in sass_gate.STATS if "converge_probe" in k)
    assert sass_gate.gate(cubin, "converge_probe", None, {name: 1})[0], "a function at its recorded count is not red"
    monkeypatch.setattr(sass_gate, "backlog", lambda: {name: 2})
    monkeypatch.setattr(sys, "argv", ["sass_gate.py", str(cubin), "--kernel", "converge_probe"])
    assert sass_gate.main() == 1
    assert "paid down" in capsys.readouterr().out, "a function below its recorded count asks for the record to shrink"
