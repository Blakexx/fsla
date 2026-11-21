"""The SASS gate (`tools/sass_gate.py`) on compiled probes: a clean kernel, one whose runtime-indexed local array must
go to local memory, one whose shuffle sits in a loop ptxas cannot prove warp-uniform (a convergence subroutine), and
one whose division calls its slow path (a call that is not one), and two whose MMA loops convert on the conversion unit
or read global memory (§29's step lints) beside one that converts on the FMA pipe and stages by `cp.async` -- the
gate's reds are what it reads, proven on a cubin that has them. The step lints are the checker's rules (`mma.sync` read as
steps of kind `sync`), and their cases on the lines -- the loops that hold steps, the branches in a step, the exit laid out as
its latch, the convergence call's hot and guarded forms -- are the checker's tests.

See KERNEL_STANDARDS §20, §29.
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

__device__ __forceinline__ void mma(float (&d)[4], unsigned a, unsigned b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%4,%4,%4}, {%5,%5}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a), "r"(b));
}

//: a step loop: an MMA, its outputs converted to s16 and stored; `XU` converts on the conversion unit (F2I), else on
//: the FMA pipe; `GLOBAL` reads its operand from global memory inside the step, else from shared memory
template <bool XU, bool GLOBAL>
__device__ void step_loop(int* y, const unsigned* x, int n) {
  __shared__ unsigned s[256];
  s[threadIdx.x] = x[threadIdx.x];
  __syncthreads();
  float d[4] = {};
  int out = 0;
  for (int i = 0; i < n; ++i) {
    const unsigned a = GLOBAL ? x[i * 32 + threadIdx.x] : s[(i + threadIdx.x) & 255];
    mma(d, a, a ^ i);
    out += XU ? __float2int_rn(d[0]) : static_cast<int>(__float_as_uint(fmaf(d[0], 1.0f, 12582912.0f)) & 0xffff);
  }
  y[threadIdx.x] = out;
}

__global__ void step_xu_probe(int* y, const unsigned* x, int n) { step_loop<true, false>(y, x, n); }
__global__ void step_global_probe(int* y, const unsigned* x, int n) { step_loop<false, true>(y, x, n); }
__global__ void step_clean_probe(int* y, const unsigned* x, int n) { step_loop<false, false>(y, x, n); }
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
    #: the bare probe by its mangled name: `clean_probe` alone also matches the step probe `step_clean_probe`
    clean = sass_gate.STATS["_Z11clean_probePfPKf"]
    local = next(v for k, v in sass_gate.STATS.items() if "local_probe" in k)
    assert not clean["reds"], report
    assert local["stack"] > 0 and local["ldl"] > 0, report
    assert any("stack frame" in r for r in local["reds"]) and not ok


def test_the_step_lints_red_the_conversion_unit_and_global_memory_in_an_mma_loop(cubin):
    """§29: F2I inside an MMA loop and a global load inside one are red; the FMA-pipe conversion over shared memory is
    clean of both. The probes' step cover is §32's reading, not this test's subject: one dependent MMA a step leaves
    the clean probe under the line."""
    ok, report = sass_gate.gate(cubin, None, None)
    xu = next(v for k, v in sass_gate.STATS.items() if "step_xu_probe" in k)
    memory = next(v for k, v in sass_gate.STATS.items() if "step_global_probe" in k)
    clean = next(v for k, v in sass_gate.STATS.items() if "step_clean_probe" in k)
    assert xu["step_xu"] > 0 and any("conversion unit" in r for r in xu["reds"]), report
    assert memory["step_global"] > 0 and any("global memory" in r for r in memory["reds"]), report
    assert clean["step_xu"] == 0 and clean["step_global"] == 0, report
    assert not any("conversion unit" in r or "global memory" in r for r in clean["reds"]), report
    assert not ok


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


def test_an_instruments_declaration_permits_the_step_signature_it_names(cubin, monkeypatch):
    """A probe keeps a step signature on purpose where its declarations name it (docs/internals/tools/sass_gate.md#instrument):
    the gate matches the instrument's declarations to the function's name (`_declares`) and hands the checker them as its declarations,
    so the conversion probe is clean and records what it keeps, and the global-memory probe, undeclared, stays red."""
    monkeypatch.setattr(sass_gate, "INSTRUMENT", {"step_xu_probe": {"step_xu": "the cost the probe measures"}})
    ok, report = sass_gate.gate(cubin, None, None)
    xu = next(v for k, v in sass_gate.STATS.items() if "step_xu_probe" in k)
    memory = next(v for k, v in sass_gate.STATS.items() if "step_global_probe" in k)
    assert not xu["reds"] and xu["kept"] == [["step_xu_probe", "step_xu"]], report
    assert memory["reds"] and not ok


def test_the_instructions_per_step_ceiling_is_the_checkers_density_bound(cubin):
    """`--max-per-hmma` is the checker's `step_density` bound, declared for every function: a kernel over it is red, one under it is
    not."""
    ok, report = sass_gate.gate(cubin, "step_clean_probe", 1)
    assert not ok and "static instructions per HMMA" in report
    ok, report = sass_gate.gate(cubin, "step_clean_probe", 10**6)
    assert ok, report


def test_an_instrument_declaration_may_name_its_architecture():
    """`sm_80 <pattern>` keeps a signature on that architecture's SASS alone, a bare pattern on every one
    (docs/internals/tools/sass_gate.md#instrument)."""
    name = "_ZN4rola5carry5calib12calib_kernelILi4ELi12ELi36EEEvNS1_11CalibParamsE"
    assert sass_gate._declares(r"calib_kernelILi\d+ELi12ELi", name, "sm_86")
    assert sass_gate._declares(r"sm_80 calib_kernelILi\d+ELi12ELi", name, "sm_80")
    assert not sass_gate._declares(r"sm_80 calib_kernelILi\d+ELi12ELi", name, "sm_86")
    assert not sass_gate._declares(r"calib_kernelILi\d+ELi13ELi", name, "sm_80")
