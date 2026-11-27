// measure/lab/experiments/newton_row.cu -- QUESTION: the producer's entmax threshold is solved on the projection's
// accumulators by Newton from the left: on a `wgmma` m64n256 accumulator's rows, how many steps does a row take, and a
// warp (its 16 rows' vote), what does a 64-token block cost in cycles, and is the support EXACT against the fp64 sorted
// closed form? THE MATHEMATICS: with a row shifted by its max (`x = z - max`), alpha 1.5's level is `((x - tau) /
// 2)_+^2` and alpha 2's `(x - tau)_+`; `f(tau) = sum level - 1` is convex and decreasing and `f(-2) >= 0` (alpha 2:
// `f(-1) >= 0`, the max alone), so Newton from there never passes the root. At a bound `lo`, with `r = max(x - lo, 0)`,
// `n` the count of `r > 0`, `s1 = sum r`, `s2 = sum r^2`: alpha 1.5's step is `lo + (s2 - 4) / (2 s1)`, alpha 2's `lo +
// (s1 - 1) / n`; the closed form on the support `{x > lo}` is `lo + u`, `u` the smaller root of `n u^2 - 2 s1 u + s2 -
// 4` (alpha 2: `u = (s1 - 1) / n`), and it is the root exactly when no member lies below it, which the count of `x >=
// lo + u` against `n` decides. METHOD: a warpgroup holds a 64-token block of 256 logits a token in the m64n256
// accumulator's own layout (a thread's `d[4 q + 2 j + i]` is row `16 w + g + 8 j`, column `8 q + 2 t + i`:
// `wgmma_regs`), loaded before the clock; one or two warpgroups a CTA, a CTA an SM. Each step is a pass over a thread's
// 128 values (a subtract, a max, an add, a multiply-add and a compare-add a value, then the settled count's
// compare-add), two shuffles a sum, and the warp's vote over its 16 rows ends the loop, cap 16. The inputs are
// `tests/integration/test_production_levels.py`'s stress kinds at 256 columns (random, near the boundary, a collapsed
// support, five times the scale); the oracle the sorted closed form in fp64 (the entmax package's `entmax15` and
// `sparsemax` algorithm). Every row's support, threshold and step count is written.
// TIMELINE: no load in the clock: the block is in registers before it. A trip is one solve: the max and shift, then the
// steps, each a pass over the 128 values a thread, two shuffles a sum and one vote. Live: the 128 accumulators, the two
// rows' bounds and sums.
// see docs/internals/tools/lab.md#newton-row
#include <algorithm>
#include <cmath>

#include "common/atoms/math.cuh"
#include "lab.cuh"

namespace newton_row {

constexpr int kWidth = 256, kBlockRows = 64, kValues = 128, kCap = 16;
enum Kind : int { kRandom = 0, kNear, kCollapsed, kLarge, kKinds };
const char* kKindName[kKinds] = {"random", "near the boundary", "a collapsed support", "five times the scale"};

//: a row's sum over the 4 lanes that hold it (the lanes of one `g`)
__device__ __forceinline__ float row_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

__device__ __forceinline__ float row_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}

//: a thread's value `v` of its row `j`: accumulator `4 q + 2 j + i`, `v = 2 q + i`
__device__ __forceinline__ int at(int j, int v) { return 4 * (v >> 1) + 2 * j + (v & 1); }

template <int A15, int WG>
__global__ void __launch_bounds__(128 * WG, 1)
    solve(long long* times, const float* logits, float* taus, int* steps, uint8_t* support, int trips) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const long long block = static_cast<long long>(blockIdx.x) * WG + (warp >> 2);
  const long long row0 = block * kBlockRows + 16 * (warp & 3) + g;
  float d[kValues];
#pragma unroll
  for (int j = 0; j < 2; ++j)
#pragma unroll
    for (int q = 0; q < 32; ++q) {
      const float2 v = *reinterpret_cast<const float2*>(logits + (row0 + 8 * j) * kWidth + 8 * q + 2 * t);
      d[4 * q + 2 * j] = v.x;
      d[4 * q + 2 * j + 1] = v.y;
    }
  float tau[2] = {0.0f, 0.0f};
  int first[2] = {0, 0};
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int trip = 0; trip < trips; ++trip) {
    float lo[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      float m = d[at(j, 0)];
#pragma unroll
      for (int v = 1; v < 64; ++v) m = fmaxf(m, d[at(j, v)]);
      m = row_max(m);
#pragma unroll
      for (int v = 0; v < 64; ++v) d[at(j, v)] -= m;
      lo[j] = A15 ? -2.0f : -1.0f;
      first[j] = 0;
    }
#pragma unroll 1
    for (int step = 0; step < kCap; ++step) {
      bool settled = true;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        //: the row's sums past its bound: `r = max(x - lo, 0)`, its count, sum and sum of squares
        float n = 0.0f, s1 = 0.0f, s2 = 0.0f;
#pragma unroll
        for (int v = 0; v < 64; ++v) {
          const float r = fmaxf(d[at(j, v)] - lo[j], 0.0f);
          n += r > 0.0f ? 1.0f : 0.0f;
          s1 += r;
          s2 = fmaf(r, r, s2);
        }
        const float here = n;
        n = row_sum(n);
        s1 = row_sum(s1);
        s2 = row_sum(s2);
        //: the closed form on the support `{x > lo}`, and Newton's step (alpha 2's is the closed form itself)
        const float u = A15 ? __fdividef(s1 - ::denseref::ops::sqrt_approx(fmaxf(s1 * s1 - n * (s2 - 4.0f), 0.0f)), n)
                            : __fdividef(s1 - 1.0f, n);
        tau[j] = lo[j] + u;
        float past = 0.0f;
#pragma unroll
        for (int v = 0; v < 64; ++v) past += d[at(j, v)] >= tau[j] ? 1.0f : 0.0f;
        //: settled: no member of the support below the closed form, so it is the root
        const bool done = (u <= 0.0f) | (past == here);
        settled = settled & done;
        first[j] = (first[j] == 0 && done) ? step + 1 : first[j];
        lo[j] = A15 ? lo[j] + __fdividef(s2 - 4.0f, 2.0f * s1) : tau[j];
      }
      const bool all = __all_sync(0xffffffffu, settled);
      if (all) break;
    }
  }
#pragma unroll
  for (int i = 0; i < kValues; ++i) lab::wg_hold(d[i]);
  clock.stop(times);
  //: every row's threshold and steps (its quad's first lane), and every value's support: the level `relu(h (z - tau))`
  const float h = A15 ? 0.5f : 1.0f;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const long long row = row0 + 8 * j;
    if (t == 0) {
      taus[row] = tau[j];
      steps[row] = first[j] == 0 ? kCap + 1 : first[j];
    }
#pragma unroll
    for (int v = 0; v < 64; ++v)
      support[row * kWidth + 8 * (v >> 1) + 2 * t + (v & 1)] = fmaxf(fmaf(h, d[at(j, v)], -h * tau[j]), 0.0f) > 0.0f;
  }
}

//: the stress kinds of `test_production_levels.py`'s `_stress_logits`, at 256 columns, from one seeded generator
std::vector<float> fixture(int kind, long long rows) {
  uint64_t s = 0x9e3779b97f4a7c15ull * (kind + 17);
  const auto next = [&] {
    s += 0x9e3779b97f4a7c15ull;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return ((z ^ (z >> 31)) >> 11) * (1.0 / 9007199254740992.0);
  };
  const auto normal = [&] { return std::sqrt(-2.0 * std::log(1.0 - next())) * std::cos(6.283185307179586 * next()); };
  std::vector<float> z(static_cast<size_t>(rows) * kWidth);
  for (long long r = 0; r < rows; ++r) {
    const int top = static_cast<int>(next() * kWidth) % kWidth;
    for (int c = 0; c < kWidth; ++c) {
      const double base = normal();
      double v = base;
      if (kind == kNear) v = std::round(base * 2.0) / 2.0 + 1.0e-4 * normal();
      if (kind == kCollapsed) v = c == top ? 20.0 : -20.0;
      if (kind == kLarge) v = 5.0 * base;
      z[r * kWidth + c] = static_cast<float>(v);
    }
  }
  return z;
}

//: THE ORACLE: the sorted closed form in fp64 on the shifted row (`x = z - max`, in z's units): the threshold, and the
//: support `x > tau`
double oracle(const float* z, bool a15, std::vector<double>& x) {
  double m = z[0];
  for (int c = 1; c < kWidth; ++c) m = std::max(m, static_cast<double>(z[c]));
  for (int c = 0; c < kWidth; ++c) x[c] = static_cast<double>(z[c]) - m;
  std::vector<double> s(x);
  std::sort(s.begin(), s.end(), std::greater<double>());
  double sum = 0, sq = 0, tau = 0;
  for (int k = 1; k <= kWidth; ++k) {
    if (a15) {
      //: entmax15 in its own units, X = x / 2
      const double X = s[k - 1] / 2;
      sum += X;
      sq += X * X;
      const double mean = sum / k, delta = std::max((1.0 - k * (sq / k - mean * mean)) / k, 0.0);
      const double cand = mean - std::sqrt(delta);
      if (cand <= X) tau = 2 * cand;
    } else {
      sum += s[k - 1];
      if (k * s[k - 1] > sum - 1.0) tau = (sum - 1.0) / k;
    }
  }
  return tau;
}

}  // namespace newton_row

template <int A15, int WG>
void run(lab::Bench& b, int kind, float* logits, float* taus, int* steps, uint8_t* support) {
  using namespace newton_row;
  const long long blocks = static_cast<long long>(b.sms) * WG, rows = blocks * kBlockRows;
  const std::vector<float> z = fixture(kind, rows);
  LAB_CHECK(cudaMemcpy(logits, z.data(), z.size() * 4, cudaMemcpyHostToDevice));
  const std::string name = std::string("alpha ") + (A15 ? "1.5" : "2") + ", " + kKindName[kind] + ", "
                           + std::to_string(WG) + (WG == 1 ? " warpgroup" : " warpgroups") + " a CTA";
  const auto kernel = solve<A15, WG>;
  kernel<<<b.sms, 128 * WG>>>(b.times, logits, taus, steps, support, 1);
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<float> tau(rows);
  std::vector<int> st(rows);
  std::vector<uint8_t> sup(z.size());
  LAB_CHECK(cudaMemcpy(tau.data(), taus, rows * 4, cudaMemcpyDeviceToHost));
  LAB_CHECK(cudaMemcpy(st.data(), steps, rows * 4, cudaMemcpyDeviceToHost));
  LAB_CHECK(cudaMemcpy(sup.data(), support, sup.size(), cudaMemcpyDeviceToHost));
  //: THE CHECK: every row's support against the fp64 oracle's; the threshold's distance from it; the steps
  std::vector<double> x(kWidth);
  long long differ = 0, capped = 0, row_steps = 0, most = 0, warp_steps = 0, warps = rows / 16;
  double worst = 0, nearest = 1e30;
  for (long long r = 0; r < rows; ++r) {
    const double t64 = oracle(z.data() + r * kWidth, A15, x);
    bool same = true;
    for (int c = 0; c < kWidth; ++c) {
      same = same && (sup[r * kWidth + c] != 0) == (x[c] > t64);
      nearest = std::min(nearest, std::fabs(x[c] - t64));
    }
    differ += !same;
    worst = std::max(worst, std::fabs(tau[r] - t64));
    capped += st[r] > kCap;
    row_steps += st[r];
    most = std::max<long long>(most, st[r]);
  }
  for (long long w = 0; w < warps; ++w) {
    int m = 0;
    for (int r = 0; r < 16; ++r) m = std::max(m, st[w * 16 + r]);
    warp_steps += m;
  }
  char detail[256];
  std::snprintf(detail, sizeof detail,
                "%lld of %lld rows' supports differ from the fp64 oracle's; %lld capped; the threshold within %.2e of "
                "fp64; the nearest logit to an fp64 threshold %.2e from it",
                differ, rows, capped, worst, nearest);
  b.check(name, differ == 0 && capped == 0, detail);
  constexpr int kTrips = 64;
  const lab::Samples cycles = b.variant([&] { kernel<<<b.sms, 128 * WG>>>(b.times, logits, taus, steps, support, kTrips); },
                                        [&] { return b.span(4 * WG) / kTrips / WG; });
  b.record(name + ": a 64-token block", "cycles a block (the CTA's span over its blocks)", cycles);
  b.record(name + ": steps a row, the mean", "steps", lab::Samples{{static_cast<double>(row_steps) / rows}});
  b.record(name + ": steps a row, the most", "steps", lab::Samples{{static_cast<double>(most)}});
  b.record(name + ": steps a warp (its 16 rows' vote), the mean", "steps",
           lab::Samples{{static_cast<double>(warp_steps) / warps}});
}

int main(int argc, char** argv) {
  using namespace newton_row;
  lab::Bench b(argc, argv);
  for (int a15 = 0; a15 < 2; ++a15)
    for (int wg : {1, 2}) {
      const std::string k = lab::kernel("solve", {a15, wg});
      b.expect(k, "SHFL", ">=", 8);
      b.expect(k, "VOTE", ">=", 1);
    }
  if (b.expect_only) return 0;
  float *logits = nullptr, *taus = nullptr;
  int* steps = nullptr;
  uint8_t* support = nullptr;
  const size_t rows = static_cast<size_t>(b.sms) * 2 * kBlockRows;
  LAB_CHECK(cudaMalloc(&logits, rows * kWidth * 4));
  LAB_CHECK(cudaMalloc(&taus, rows * 4));
  LAB_CHECK(cudaMalloc(&steps, rows * 4));
  LAB_CHECK(cudaMalloc(&support, rows * kWidth));
  for (int kind = 0; kind < kKinds; ++kind) {
    run<1, 1>(b, kind, logits, taus, steps, support);
    run<1, 2>(b, kind, logits, taus, steps, support);
    run<0, 1>(b, kind, logits, taus, steps, support);
    run<0, 2>(b, kind, logits, taus, steps, support);
  }
  return 0;
}
