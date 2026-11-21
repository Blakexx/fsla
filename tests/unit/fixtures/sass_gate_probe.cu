// tests/unit/fixtures/sass_gate_probe.cu -- the probe kernels of tests/unit/test_sass_gate.py, compiled by the graph's CUDA rule
// (tests/BUILD.bazel); not part of the extension.
__global__ void clean_probe(float* y, const float* x) { y[threadIdx.x] = 2.0f * x[threadIdx.x]; }

__global__ void local_probe(float* y, const int* i) {
  float a[512];
  for (int k = 0; k < 512; ++k) a[k] = k * y[k];
  a[i[threadIdx.x + 32]] = y[threadIdx.x];
  y[threadIdx.x] = a[i[threadIdx.x]];
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
