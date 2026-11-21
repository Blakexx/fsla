// tests/unit/fixtures/life_ranges_probe.cu -- the probe kernels of tests/unit/test_life_ranges.py, compiled by the graph's CUDA rule
// (tests/BUILD.bazel); not part of the extension.
//: 32 values held across a loop whose trip count differs by lane, so its shuffle's slow path is a subroutine call
__global__ void shuffle_probe(float* y, const int* n) {
  float keep[32];
#pragma unroll
  for (int k = 0; k < 32; ++k) keep[k] = y[k * 256 + threadIdx.x];
  y[threadIdx.x] = 0.0f;
  float v = keep[0];
  for (int i = 0; i < n[threadIdx.x]; ++i) v += __shfl_xor_sync(0xffffffffu, v, 1);
#pragma unroll
  for (int k = 0; k < 32; ++k) v += keep[k] * (k + 1);
  y[threadIdx.x + 256] = v;
}
