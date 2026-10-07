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


def test_a_guarded_convergence_fallback_and_a_slow_path_call_are_not_red(cubin):
    ok, report = sass_gate.gate(cubin, None, None)
    converge = next(v for k, v in sass_gate.STATS.items() if "converge_probe" in k)
    divide = next(v for k, v in sass_gate.STATS.items() if "divide_probe" in k)
    assert converge["calls"] == 0 and converge["guarded"] == 1 and not converge["reds"], report
    assert divide["calls"] == 0 and divide["guarded"] == 0 and not divide["reds"], report


#: a convergence subroutine and a call into it, as a disassembly lists them: `(address, opcode, operands)`
_SUBROUTINE = [(0x100, "WARPSYNC", "0xffffffff", ""), (0x110, "SHFL.BFLY", "PT, R0, R1, 0x1, 0x1f", ""),
               (0x120, "RET.REL.NODEC", "R2 0x0", "")]


def test_the_classifier_reds_an_unguarded_call_and_counts_a_guarded_one():
    """This toolchain's ptxas guards every call it emits (the probe above), so the red is proven on the lines."""
    hot = [(0x0, "CALL.REL.NOINC", "0x100", ""), (0x10, "EXIT", "", "")] + _SUBROUTINE
    via_div = [(0x0, "BRA.DIV", "~URZ, 0x40", ""), (0x10, "SHFL.BFLY", "PT, R0, R1, 0x1, 0x1f", ""),
               (0x20, "EXIT", "", ""), (0x40, "CALL.REL.NOINC", "0x100", ""), (0x50, "BRA", "0x20", "")] + _SUBROUTINE
    past_conv = [(0x0, "BRA.CONV", "~URZ, 0x20", ""), (0x10, "CALL.REL.NOINC", "0x100", ""),
                 (0x20, "EXIT", "", "")] + _SUBROUTINE
    assert sass_gate.convergence_calls(hot, {}) == (1, 0)
    assert sass_gate.convergence_calls(via_div, {}) == (0, 1)
    assert sass_gate.convergence_calls(past_conv, {}) == (0, 1)
    never = [(0x0, "BRA", "0x20", "@!PT"), (0x10, "BRA", "0x30", ""), (0x20, "CALL.REL.NOINC", "0x100", ""),
             (0x30, "EXIT", "", "")] + _SUBROUTINE
    assert sass_gate.convergence_calls(never, {}) == (0, 1)
    reached = [(0x0, "BRA", "0x20", "@!PT"), (0x10, "MOV", "R0, R1", ""), (0x20, "CALL.REL.NOINC", "0x100", ""),
               (0x30, "EXIT", "", "")] + _SUBROUTINE
    assert sass_gate.convergence_calls(reached, {}) == (1, 0)


def test_a_branch_inside_an_mma_step_is_counted_and_one_outside_it_is_not():
    """KERNEL_STANDARDS 27 on the lines: a loop (a backward branch) holding an HMMA counts its convergence barriers and
    forward conditional branches; an outer loop's branches around an inner MMA loop do not count, nor a loop with no
    MMA."""
    step = [(0x0, "LDS", "R4, [R2]", ""), (0x10, "HMMA.16816.F16", "R8, R4, R6, R8", ""),
            (0x20, "BSSY", "0x50", ""), (0x30, "BRA", "0x50", "@!P0"), (0x40, "ATOMS.ADD", "RZ, [R3], R8", ""),
            (0x50, "BSYNC", "", ""), (0x60, "BRA", "0x0", "@P1"), (0x70, "EXIT", "", "")]
    assert sass_gate.mma_loop_branches(step, {}) == 2
    clean = [(0x0, "LDS", "R4, [R2]", ""), (0x10, "HMMA.16816.F16", "R8, R4, R6, R8", ""),
             (0x20, "ATOMS.ADD", "RZ, [R3], R8", ""), (0x30, "BRA", "0x0", "@P1"), (0x40, "EXIT", "", "")]
    assert sass_gate.mma_loop_branches(clean, {}) == 0
    nested = [(0x0, "BRA", "0x40", "@P2"), (0x10, "HMMA.16816.F16", "R8, R4, R6, R8", ""),
              (0x20, "BRA", "0x10", "@P1"), (0x30, "BAR.SYNC", "0x0", ""), (0x40, "BRA", "0x0", "@P3"),
              (0x50, "EXIT", "", "")]
    assert sass_gate.mma_loop_branches(nested, {}) == 0
    no_mma = [(0x0, "BSSY", "0x30", ""), (0x10, "BRA", "0x30", "@!P0"), (0x20, "LDS", "R4, [R2]", ""),
              (0x30, "BRA", "0x0", "@P1"), (0x40, "EXIT", "", "")]
    assert sass_gate.mma_loop_branches(no_mma, {}) == 0
