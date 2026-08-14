// P14 step 1 -- the scattered-emission microbench (SUPPORT_GRANULARITY_DESIGN.md
// section 11, probe P1). Standalone, no rola-standalone build dependency: this
// reproduces the CONSUMER KERNEL'S EXACT EMISSION ADDRESS PATTERN in isolation
// (csrc/rola/src/consumer_kernel.cuh:951-961 at rola-standalone c4bbba9) so the
// scattered-write question can be priced without touching or building the real
// kernel. No production source is edited by this file.
//
// Real pattern being reproduced, verbatim from the tree:
//   float* num_t0 = p.num + (bh*T + t0) * DV;
//   auto emit = [&](int t, int ch, float val) {
//     if (t0 + t >= T) return;
//     const float out = sr[t] * val;
//     if (out != 0.0f) atomicAdd(&num_t0[t * DV + ch], out);
//   };
//   emit(_g, c0, y[0]); emit(_g, c1, y[1]); emit(_g+8, c0, y[2]); emit(_g+8, c1, y[3]);
// with warp = threadIdx/32 in [0,8) (kWarps=8, DV=64, NT_DV=DV/8=8), lane in
// [0,32), _r = lane%4, _g = lane/4, n0 = warp*8, c0 = n0+2*_r, c1 = c0+1.
//
// The only thing this probe varies is WHERE t0 (equivalently, each token's
// row) lands in `num`: arm (a) every tile's 16 tokens are the tile's own
// contiguous 16; arm (b) [+ the concentration sweep] gathers 16 tokens whose
// spacing is `gap` tiles apart, matching the design doc's "token gather"
// write pattern once tokens are pulled from scattered live positions. The
// `out != 0.0f` skip is dropped (val is always nonzero here) -- MATCHED WORK:
// every arm issues the identical 1024 atomics/block (16 tokens x 64 chan),
// only the destination addresses differ.
//
// Standalone: no torch, no rola-probe registry entry (there is no shipped
// fixture this pattern belongs to -- it is deliberately isolated from the
// real kernel per the probe spec, "no kernel change"). Build and run:
//   nvcc -O3 -arch=sm_86 -o /tmp/p14_probe p14_emission_probe.cu
//   flock /tmp/rola_gpu.lock /tmp/p14_probe [n_tiles=8192] [reps=30]
// Prints one CSV row per concentration-sweep arm (cuda_events median/IQR).
// ncu counters (dram__bytes.sum, l1tex__t_sectors.sum) are taken the same
// way, wrapped in the same lock, per docs/measurement.md's instrument
// discipline; the development journal holds the captured raw rows and the
// verdict.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <algorithm>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
  exit(1); } } while (0)

constexpr int DV = 64;       // matches consumer_kernel.cuh DV
constexpr int BT = 16;       // tile width (tokens/tile), matches BT
constexpr int NT_DV = DV / 8; // = 8 = kWarps
constexpr int THREADS = NT_DV * 32; // 256 = 8 warps

// One block = one tile/visit. token_idx[block*BT + t] gives the destination
// token row for local index t in [0,16). num has T_alloc*DV floats.
__global__ void emission_kernel(float* __restrict__ num,
                                 const int32_t* __restrict__ token_idx) {
  const int32_t* tile_tokens = token_idx + blockIdx.x * BT;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int n0 = warp * 8;
  const int _r = lane % 4;
  const int _g = lane / 4;
  const int c0 = n0 + 2 * _r, c1 = c0 + 1;
  const float val = 1.0f; // matched-work: always emitted, no zero-skip

  auto emit = [&](int t, int ch) {
    const int32_t token = tile_tokens[t];
    atomicAdd(&num[static_cast<int64_t>(token) * DV + ch], val);
  };
  emit(_g, c0);
  emit(_g, c1);
  emit(_g + 8, c0);
  emit(_g + 8, c1);
}

// Builds the token_idx table for one arm.
//   gap_tiles == 0  -> CONTENTION control: every token in every tile is the
//                      SAME single row (max amplification positive control).
//   gap_tiles == 1  -> arm (a), contiguous: tile i covers tokens [i*16,i*16+16)
//                      (stride 1 token between the tile's own 16 rows -- this
//                      IS today's shipped emission pattern, unchanged).
//   gap_tiles  > 1  -> arm (b)-family: tile i's 16 "gathered" tokens are each
//                      gap_tiles*BT tokens apart (mean gap ~gap_tiles tiles,
//                      per the design doc's measured "mean gap ~14 tiles"
//                      figure), based at a distinct region per tile so tiles
//                      do not collide with each other.
static std::vector<int32_t> build_token_idx(int n_tiles, int gap_tiles, int64_t T_alloc) {
  std::vector<int32_t> idx(static_cast<size_t>(n_tiles) * BT);
  if (gap_tiles == 0) {
    std::fill(idx.begin(), idx.end(), 0);
    return idx;
  }
  // stride between a tile's own 16 gathered tokens: 1 token (contiguous, arm a)
  // or gap_tiles*BT tokens (scattered, arm b-family).
  const int64_t stride = (gap_tiles == 1) ? 1 : static_cast<int64_t>(gap_tiles) * BT;
  const int64_t tile_footprint = stride * BT; // token-space span one tile occupies
  const int64_t region = std::max<int64_t>(tile_footprint + BT, 1);
  for (int i = 0; i < n_tiles; ++i) {
    const int64_t base = (static_cast<int64_t>(i) * region) % (T_alloc - region);
    for (int t = 0; t < BT; ++t) {
      idx[static_cast<size_t>(i) * BT + t] = static_cast<int32_t>(base + t * stride);
    }
  }
  return idx;
}

struct Result { double median_ms; double p25, p75; };

static Result time_arm(float* d_num, const int32_t* d_idx, int n_tiles, int reps) {
  std::vector<double> samples;
  samples.reserve(reps);
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  // warm-up
  emission_kernel<<<n_tiles, THREADS>>>(d_num, d_idx);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  for (int r = 0; r < reps; ++r) {
    CUDA_CHECK(cudaEventRecord(start));
    emission_kernel<<<n_tiles, THREADS>>>(d_num, d_idx);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    samples.push_back(ms);
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  std::sort(samples.begin(), samples.end());
  auto pct = [&](double p) {
    double idx = p * (samples.size() - 1);
    size_t lo = static_cast<size_t>(idx);
    double frac = idx - lo;
    if (lo + 1 < samples.size()) return samples[lo] * (1 - frac) + samples[lo + 1] * frac;
    return samples[lo];
  };
  return Result{pct(0.5), pct(0.25), pct(0.75)};
}

int main(int argc, char** argv) {
  int n_tiles = 8192;
  int reps = 30;
  if (argc > 1) n_tiles = atoi(argv[1]);
  if (argc > 2) reps = atoi(argv[2]);

  // Concentration sweep: gap_tiles in tiles. 1 = contiguous control (arm a).
  // 14 = the design doc's measured mean gap (arm b). 0 = single-address
  // contention positive control (must show maximal amplification).
  std::vector<int> gaps = {1, 2, 4, 8, 14, 32, 64, 0};

  // Big enough T_alloc so every arm's addresses stay in range; sized off the
  // largest gap used (64) plus the contiguous arm's own footprint.
  const int64_t T_alloc = static_cast<int64_t>(n_tiles) * 64 * BT + BT + 1024;
  float* d_num;
  CUDA_CHECK(cudaMalloc(&d_num, static_cast<size_t>(T_alloc) * DV * sizeof(float)));

  printf("gap_tiles,n_tiles,reps,median_ms,p25_ms,p75_ms,ratio_vs_contiguous\n");
  double contiguous_median = -1.0;
  for (int gap : gaps) {
    std::vector<int32_t> h_idx = build_token_idx(n_tiles, gap, T_alloc);
    int32_t* d_idx;
    CUDA_CHECK(cudaMalloc(&d_idx, h_idx.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_idx, h_idx.data(), h_idx.size() * sizeof(int32_t),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_num, 0, static_cast<size_t>(T_alloc) * DV * sizeof(float)));
    Result r = time_arm(d_num, d_idx, n_tiles, reps);
    if (gap == 1) contiguous_median = r.median_ms;
    double ratio = (contiguous_median > 0) ? r.median_ms / contiguous_median : 0.0;
    printf("%d,%d,%d,%.6f,%.6f,%.6f,%.4f\n", gap, n_tiles, reps, r.median_ms, r.p25, r.p75, ratio);
    CUDA_CHECK(cudaFree(d_idx));
  }
  CUDA_CHECK(cudaFree(d_num));
  return 0;
}
