// tests/unit/fixtures/life_ranges_probe.cu -- the probe kernels of tests/unit/test_life_ranges.py, compiled by the graph's CUDA rule
// (tests/BUILD.bazel); not part of the extension.
//: 32 values held across an IEEE division, whose slow path is a call into a subroutine (Hopper's only such call: its shuffle
//: is inline after a `WARPSYNC`, the convergence subroutine being an Ampere shape)
__global__ void division_probe(float* y, const float* d) {
  float keep[32];
#pragma unroll
  for (int k = 0; k < 32; ++k) keep[k] = y[k * 256 + threadIdx.x];
  y[threadIdx.x] = 0.0f;
  float v = keep[0] / d[threadIdx.x];
#pragma unroll
  for (int k = 0; k < 32; ++k) v += keep[k] * (k + 1);
  y[threadIdx.x + 256] = v;
}
