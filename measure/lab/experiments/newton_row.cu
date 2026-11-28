// measure/lab/experiments/newton_row.cu -- QUESTION: the producer's entmax threshold is solved on the projection's
// accumulators by Newton from the left: on a `wgmma` m64n256 accumulator's rows, how many steps does a row take, and a
// warp (its 16 rows' vote), what does a 64-token block cost in cycles, and are its OUTPUTS (levels, u8 digits, mask
// bits, density) fp64's closed form? THE MATHEMATICS: with a row shifted by its max (`x = z - max`), alpha 1.5's level
// is `((x - tau) / 2)_+^2` and alpha 2's `(x - tau)_+`; `f(tau) = sum level - 1` is convex and decreasing and `f(-2) >=
// 0` (alpha 2: `f(-1) >= 0`, the max alone), so Newton from there never passes the root. At a bound `lo`, with `n`, `P`
// and `Q` the count, sum and sum of squares of the support `{x > lo}`: alpha 1.5's `f(lo) = (Q - 2 P lo + n lo^2) / 4 -
// 1` and `f'(lo) = -(P - n lo) / 2`, so its step is `lo + (Q - 2 P lo + n lo^2 - 4) / (2 (P - n lo))`; the closed form
// on that support is alpha 1.5's smaller root of `n tau^2 - 2 P tau + Q - 4`, alpha 2's `(P - 1) / n` (alpha 2's step
// itself), and it is the root exactly when no member of the support lies below it, which the count of `x >= tau`
// against `n` decides. METHOD: a warpgroup holds a 64-token block of 256 logits a token in the m64n256 accumulator's
// own layout (a thread's `d[4 q + 2 j + i]` is row `16 w + g + 8 j`, column `8 q + 2 t + i`: `wgmma_regs`), loaded
// before the clock; one or two warpgroups a CTA, a CTA an SM. Each step is a pass over a thread's 128 values (one
// compare and three predicated adds a value, then the settled count's compare and predicated add), two shuffles a sum,
// and the warp's vote over its 16 rows ends the loop, cap 16. The inputs are
// `tests/integration/test_production_levels.py`'s stress kinds at 256 columns (random, near the boundary, a collapsed
// support, five times the scale); the oracle the sorted closed form in fp64 (the entmax package's `entmax15` and
// `sparsemax` algorithm). Every row's levels, threshold and steps are written.
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
//: THE STEP'S FORMS: each row's pass, sums and closed form in turn (SERIAL), or both rows' passes in one loop with
//: every sum split into two chains, the six sums' shuffles issued together and both counts in one loop (INTERLEAVED)
//: and the interleaved form with its closed form in IEEE arithmetic, the square root and the divide rounded to nearest
//: (EXACT: the threshold within fp32's rounding of the sums; Newton's own step stays approximate, a bound)
enum Form : int { kSerial = 0, kInterleaved, kExact, kForms };
const char* kFormName[kForms] = {"each row in turn", "both rows interleaved", "both rows interleaved, the closed form IEEE"};
const char* kKindName[kKinds] = {"random", "near the boundary", "a collapsed support", "five times the scale"};

//: ACCUMULATED PAST `lo`: where `x > lo`, one more member, its value and its square -- the compare and three adds under
//: its predicate, so a value costs four instructions and no select
__device__ __forceinline__ void accumulate_past(float x, float lo, float& n, float& P, float& Q) {
  asm("{\n .reg .pred in;\n setp.gt.f32 in, %3, %4;\n @in add.rn.f32 %0, %0, 0f3F800000;\n"
      " @in add.rn.f32 %1, %1, %3;\n @in fma.rn.f32 %2, %3, %3, %2;\n}"
      : "+f"(n), "+f"(P), "+f"(Q)
      : "f"(x), "f"(lo));
}

//: COUNTED FROM `tau`: one more where `x >= tau`, the compare and one predicated add
__device__ __forceinline__ void count_from(float x, float tau, float& n) {
  asm("{\n .reg .pred in;\n setp.ge.f32 in, %1, %2;\n @in add.rn.f32 %0, %0, 0f3F800000;\n}" : "+f"(n) : "f"(x), "f"(tau));
}

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

template <int A15, int WG, int FORM>
__global__ void __launch_bounds__(128 * WG, 1)
    solve(long long* times, const float* logits, float* taus, int* steps, float* levels, int trips) {
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
      if constexpr (FORM == kSerial) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        //: the row's support past its bound, `{x > lo}`: its count, sum and sum of squares, one compare and three
        //: predicated adds a value
        float n = 0.0f, P = 0.0f, Q = 0.0f;
#pragma unroll
        for (int v = 0; v < 64; ++v) accumulate_past(d[at(j, v)], lo[j], n, P, Q);
        const float here = n;
        n = row_sum(n);
        P = row_sum(P);
        Q = row_sum(Q);
        //: the closed form on that support: alpha 1.5's smaller root of `n tau^2 - 2 P tau + Q - 4`, alpha 2's
        //: `(P - 1) / n`
        tau[j] = A15 ? __fdividef(P - ::denseref::ops::sqrt_approx(fmaxf(P * P - n * (Q - 4.0f), 0.0f)), n)
                     : __fdividef(P - 1.0f, n);
        //: settled: every member at or past the closed form (one compare and one predicated add a value)
        float past = 0.0f;
#pragma unroll
        for (int v = 0; v < 64; ++v) count_from(d[at(j, v)], tau[j], past);
        const bool done = (tau[j] <= lo[j]) | (past == here);
        settled = settled & done;
        first[j] = (first[j] == 0 && done) ? step + 1 : first[j];
        //: Newton's step from the left: alpha 1.5's `lo - f / f'` with `f = (Q - 2 P lo + n lo^2) / 4 - 1` and
        //: `f' = -(P - n lo) / 2`; alpha 2's step is the closed form
        lo[j] = A15 ? lo[j] + __fdividef(Q - 2.0f * P * lo[j] + n * lo[j] * lo[j] - 4.0f, 2.0f * (P - n * lo[j]))
                    : tau[j];
      }
      } else {
        //: BOTH ROWS' SUMS IN ONE PASS, each sum two chains (even and odd values), so twelve chains interleave
        float n[2][2] = {}, P[2][2] = {}, Q[2][2] = {};
#pragma unroll
        for (int v = 0; v < 64; ++v)
#pragma unroll
          for (int j = 0; j < 2; ++j) accumulate_past(d[at(j, v)], lo[j], n[j][v & 1], P[j][v & 1], Q[j][v & 1]);
        //: the six sums' shuffles issued together: one round of six, then the other
        float r[6], here[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          here[j] = n[j][0] + n[j][1];
          r[3 * j] = here[j];
          r[3 * j + 1] = P[j][0] + P[j][1];
          r[3 * j + 2] = Q[j][0] + Q[j][1];
        }
#pragma unroll
        for (int k = 0; k < 6; ++k) r[k] += __shfl_xor_sync(0xffffffffu, r[k], 1);
#pragma unroll
        for (int k = 0; k < 6; ++k) r[k] += __shfl_xor_sync(0xffffffffu, r[k], 2);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const float nn = r[3 * j], PP = r[3 * j + 1], QQ = r[3 * j + 2];
          if constexpr (FORM == kExact)
            tau[j] = A15 ? __fdiv_rn(PP - __fsqrt_rn(fmaxf(PP * PP - nn * (QQ - 4.0f), 0.0f)), nn)
                         : __fdiv_rn(PP - 1.0f, nn);
          else
            tau[j] = A15 ? __fdividef(PP - ::denseref::ops::sqrt_approx(fmaxf(PP * PP - nn * (QQ - 4.0f), 0.0f)), nn)
                         : __fdividef(PP - 1.0f, nn);
        }
        //: BOTH ROWS' SETTLED COUNTS IN ONE PASS, two chains a row
        float past[2][2] = {};
#pragma unroll
        for (int v = 0; v < 64; ++v)
#pragma unroll
          for (int j = 0; j < 2; ++j) count_from(d[at(j, v)], tau[j], past[j][v & 1]);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const float nn = r[3 * j], PP = r[3 * j + 1], QQ = r[3 * j + 2];
          const bool done = (tau[j] <= lo[j]) | (past[j][0] + past[j][1] == here[j]);
          settled = settled & done;
          first[j] = (first[j] == 0 && done) ? step + 1 : first[j];
          lo[j] = A15 ? lo[j] + __fdividef(QQ - 2.0f * PP * lo[j] + nn * lo[j] * lo[j] - 4.0f, 2.0f * (PP - nn * lo[j]))
                      : tau[j];
        }
      }
      const bool all = __all_sync(0xffffffffu, settled);
      if (all) break;
    }
  }
#pragma unroll
  for (int i = 0; i < kValues; ++i) lab::wg_hold(d[i]);
  clock.stop(times);
  //: a row settles at the step every lane holding it settled: the most of its 4 lanes' (a lane's own test turns true
  //: once its share holds no member below the closed form, and stays so)
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int a = first[j] == 0 ? kCap + 1 : first[j];
    const int b = max(a, __shfl_xor_sync(0xffffffffu, a, 1));
    first[j] = max(b, __shfl_xor_sync(0xffffffffu, b, 2));
  }
  //: every row's threshold and steps (its quad's first lane), and every value's level: `relu(h (x - tau))`, squared at
  //: alpha 1.5
  const float h = A15 ? 0.5f : 1.0f;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const long long row = row0 + 8 * j;
    if (t == 0) {
      taus[row] = tau[j];
      steps[row] = first[j];
    }
#pragma unroll
    for (int v = 0; v < 64; ++v)
    {
      const float base = fmaxf(fmaf(h, d[at(j, v)], -h * tau[j]), 0.0f);
      levels[row * kWidth + 8 * (v >> 1) + 2 * t + (v & 1)] = A15 ? base * base : base;
    }
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

template <int A15, int WG, int FORM>
void run(lab::Bench& b, int kind, float* logits, float* taus, int* steps, float* levels) {
  using namespace newton_row;
  const long long blocks = static_cast<long long>(b.sms) * WG, rows = blocks * kBlockRows;
  const std::vector<float> z = fixture(kind, rows);
  LAB_CHECK(cudaMemcpy(logits, z.data(), z.size() * 4, cudaMemcpyHostToDevice));
  const std::string name = std::string("alpha ") + (A15 ? "1.5" : "2") + ", " + kKindName[kind] + ", "
                           + std::to_string(WG) + (WG == 1 ? " warpgroup" : " warpgroups") + " a CTA"
                           + (FORM == kSerial ? "" : std::string(", ") + kFormName[FORM]);
  const auto kernel = solve<A15, WG, FORM>;
  kernel<<<b.sms, 128 * WG>>>(b.times, logits, taus, steps, levels, 1);
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<float> tau(rows);
  std::vector<int> st(rows);
  std::vector<float> lev(z.size());
  LAB_CHECK(cudaMemcpy(tau.data(), taus, rows * 4, cudaMemcpyDeviceToHost));
  LAB_CHECK(cudaMemcpy(st.data(), steps, rows * 4, cudaMemcpyDeviceToHost));
  LAB_CHECK(cudaMemcpy(lev.data(), levels, lev.size() * 4, cudaMemcpyDeviceToHost));
  //: THE CHECK: WHAT THE PRODUCER OUTPUTS, against fp64's -- every level under the stored entmax values' clause
  //: (`tests/oracle/tolerances.py`'s ENTMAX_VALUES: off by at most max(2e-5 |p|, 2e-6)), every u8 digit
  //: (`rint(255 p / max)`) equal to fp64's or one off where fp64's value lies within that allowance of the digits'
  //: half-point, and the mask bits (a digit nonzero) and the density (the nonzero digits' count) the digits' own. A
  //: support that differs at a logit within fp32's resolution of the threshold has a level within the allowance of 0
  //: on both sides, so its digit is 0 on both: no failure
  std::vector<double> x(kWidth), p64(kWidth);
  long long off = 0, digit_flips = 0, digit_wrong = 0, mask_flips = 0, live_dev = 0, live64 = 0, capped = 0,
            row_steps = 0, most = 0, warp_steps = 0, block_steps = 0, warps = rows / 16;
  double worst = 0;
  const auto allowance = [](double v) { return std::max(2e-5 * std::fabs(v), 2e-6); };
  for (long long r = 0; r < rows; ++r) {
    const double t64 = oracle(z.data() + r * kWidth, A15, x);
    double top64 = 0, top = 0;
    for (int c = 0; c < kWidth; ++c) {
      const double base = std::max(A15 ? (x[c] - t64) / 2 : x[c] - t64, 0.0);
      p64[c] = A15 ? base * base : base;
      top64 = std::max(top64, p64[c]);
      top = std::max(top, static_cast<double>(lev[r * kWidth + c]));
    }
    for (int c = 0; c < kWidth; ++c) {
      const double p = lev[r * kWidth + c];
      off += std::fabs(p - p64[c]) > allowance(p64[c]);
      const double v = 255.0 * p64[c] / top64;
      const long long d = std::llrint(255.0 * p / top), d64 = std::llrint(v);
      if (d != d64) {
        ++digit_flips;
        //: within the allowance of the half-point between them (the value's and the row maximum's allowances)
        const double half = 0.5 * static_cast<double>(d + d64);
        const double slack = 255.0 * allowance(p64[c]) / top64 + v * allowance(top64) / top64;
        digit_wrong += std::llabs(d - d64) > 1 || std::fabs(v - half) > slack;
      }
      mask_flips += (d != 0) != (d64 != 0);
      live_dev += d != 0;
      live64 += d64 != 0;
    }
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
  for (long long blk = 0; blk < rows / kBlockRows; ++blk) {
    int m = 0;
    for (int r = 0; r < kBlockRows; ++r) m = std::max(m, st[blk * kBlockRows + r]);
    block_steps += m;
  }
  char detail[320];
  std::snprintf(detail, sizeof detail,
                "%lld of %lld levels off fp64 past the stored values' clause; %lld digits differ from fp64's, %lld past "
                "its allowance; %lld mask bits differ, the density %lld nonzero digits against fp64's %lld; %lld capped; "
                "the threshold within %.2e of fp64",
                off, rows * kWidth, digit_flips, digit_wrong, mask_flips, live_dev, live64, capped, worst);
  b.check(name, off == 0 && digit_wrong == 0 && capped == 0, detail);
  constexpr int kTrips = 64;
  const lab::Samples cycles = b.variant([&] { kernel<<<b.sms, 128 * WG>>>(b.times, logits, taus, steps, levels, kTrips); },
                                        [&] { return b.span(4 * WG) / kTrips / WG; });
  b.record(name + ": a 64-token block", "cycles a block (the CTA's span over its blocks)", cycles);
  b.record(name + ": steps a row, the mean", "steps", lab::Samples{{static_cast<double>(row_steps) / rows}});
  b.record(name + ": steps a row, the most", "steps", lab::Samples{{static_cast<double>(most)}});
  b.record(name + ": steps a warp (its 16 rows' vote), the mean", "steps",
           lab::Samples{{static_cast<double>(warp_steps) / warps}});
  b.record(name + ": steps a block (its 4 warps' most), the mean", "steps",
           lab::Samples{{static_cast<double>(block_steps) / (rows / kBlockRows)}});
  b.record(name + ": digits that differ from fp64's", "digits", lab::Samples{{static_cast<double>(digit_flips)}});
  b.record(name + ": mask bits that differ from fp64's", "bits", lab::Samples{{static_cast<double>(mask_flips)}});
}

int main(int argc, char** argv) {
  using namespace newton_row;
  lab::Bench b(argc, argv);
  for (int a15 = 0; a15 < 2; ++a15)
    for (int wg : {1, 2})
      for (int form = 0; form < kForms; ++form) {
      const std::string k = lab::kernel("solve", {a15, wg, form});
      //: the IEEE root's and divide's slow paths (a denormal or special operand: never taken here) are branches in
      //: the step, two a root or divide a row
      if (form == kExact) b.expect(k, "LOOP_CONVERGENCE", "<=", a15 ? 8 : 4);
      b.expect(k, "SHFL", ">=", 8);
      b.expect(k, "VOTE", ">=", 1);
    }
  if (b.expect_only) return 0;
  float *logits = nullptr, *taus = nullptr;
  int* steps = nullptr;
  float* levels = nullptr;
  const size_t rows = static_cast<size_t>(b.sms) * 2 * kBlockRows;
  LAB_CHECK(cudaMalloc(&logits, rows * kWidth * 4));
  LAB_CHECK(cudaMalloc(&taus, rows * 4));
  LAB_CHECK(cudaMalloc(&steps, rows * 4));
  LAB_CHECK(cudaMalloc(&levels, rows * kWidth * 4));
  for (int kind = 0; kind < kKinds; ++kind) {
    run<1, 1, kSerial>(b, kind, logits, taus, steps, levels);
    run<1, 2, kSerial>(b, kind, logits, taus, steps, levels);
    run<0, 1, kSerial>(b, kind, logits, taus, steps, levels);
    run<0, 2, kSerial>(b, kind, logits, taus, steps, levels);
    run<1, 1, kInterleaved>(b, kind, logits, taus, steps, levels);
    run<1, 2, kInterleaved>(b, kind, logits, taus, steps, levels);
    run<0, 1, kInterleaved>(b, kind, logits, taus, steps, levels);
    run<0, 2, kInterleaved>(b, kind, logits, taus, steps, levels);
    run<1, 1, kExact>(b, kind, logits, taus, steps, levels);
    run<1, 2, kExact>(b, kind, logits, taus, steps, levels);
    run<0, 1, kExact>(b, kind, logits, taus, steps, levels);
    run<0, 2, kExact>(b, kind, logits, taus, steps, levels);
  }
  return 0;
}
