// measure/lab/experiments/list_fold.cu -- QUESTION: the fold's step forms for the carve re-price. Per list (a warp
// steps each of its lists over that list's own tokens, gathered from the resident rows) against merged (a warp steps
// its lists' union, every list's MMAs on every union token, the operands loaded once): what does a per-list step cost
// beside its tensor floor at a 32 x 32 list tile (4 lists x 32 level-S digits x 32 channels a warp, the even carve)
// and at today's 16 x 32 (8 lists x 16 x 32), full and half full, and what does reloading every list's operands cost
// at dense against loading them once for all lists? METHOD: 8 warps, two a scheduler, fp32 accumulate, the kernel's
// 128 accumulator registers live; 256 rows resident (160 bytes: level S's 32 f16 digits, 32 f16 values, level R's 32
// u8 digits; the halves of each 32-byte pair swapped on rows with bit 2 set) and their (F, m); a list step's rows from
// its entries (gathered: hashed rows; contiguous: the pass's own), an entry past the list's live count reading row 0
// with c = 0; the step's operands by `ldmatrix.trans`, its c from the row's digit and (F, m), two steps' operands
// loaded ahead of their MMAs; the block's span a pass against the pass's tensor floor (the busiest scheduler's two
// warps' MMAs at 32 cycles). see docs/internals/tools/lab.md#list-fold
#include "lab.cuh"

#include <utility>

namespace list_fold {

template <class F, int... I>
__device__ __forceinline__ void static_for_impl(F&& f, std::integer_sequence<int, I...>) {
  (f(std::integral_constant<int, I>{}), ...);
}

//: `f(std::integral_constant<int, i>)` for i in [0, N), unrolled
template <int N, class F>
__device__ __forceinline__ void static_for(F&& f) {
  static_for_impl(f, std::make_integer_sequence<int, N>{});
}

constexpr int kThreads = 256, kWarps = 8, kRows = 256, kPitch = 160, kMaxSteps = 8, kStep = 16;
constexpr uint16_t kPad = 0xffffu;

//: FORM 0 merged at today's tile (8 lists x 16 x 32: one A, 32 + 1 MMAs a step); 1 per list at the even tile (4 lists
//: x 32 x 32: two A, 8 + 2 MMAs a list step); 2 merged at the even tile (4 lists, one load a step for all: 32 + 2); 3
//: per list at today's tile (4 + 1 a list step); 4 as 1 with the loads interleaved among the previous chunk's MMAs
//: (each step's operands loaded under the MMAs of the step before)
constexpr int kForms = 5;
const char* kFormName[kForms] = {"merged, 8 lists x 16 x 32", "per list, 4 lists x 32 x 32",
                                 "merged, 4 lists x 32 x 32", "per list, 8 lists x 16 x 32",
                                 "per list, 4 lists x 32 x 32, staged"};

__host__ __device__ constexpr int lists_of(int form) {
  return form == 1 || form == 2 || form == 4 ? 4 : 8;
}

__host__ __device__ constexpr int mtiles_of(int form) {
  return form == 1 || form == 2 || form == 4 ? 2 : 1;
}

__host__ __device__ constexpr bool merged(int form) { return form == 0 || form == 2; }

//: a pass's MMAs a warp: merged, STEPS steps of every list's atoms and the mass's; per list, STEPS steps a list
__host__ __device__ constexpr int mmas_of(int form, int steps) {
  return merged(form) ? steps * (lists_of(form) * mtiles_of(form) * 4 + mtiles_of(form))
                      : lists_of(form) * steps * (mtiles_of(form) * 4 + mtiles_of(form));
}

__device__ __forceinline__ int at(int r, int c) { return r * kPitch + 16 * (c ^ ((r >> 2) & 1)); }

struct alignas(16) Shared {
  uint8_t rows[kRows * kPitch];
  float2 fm[kRows];
  uint16_t entries[kWarps][8][kMaxSteps][kStep];
  float mass[kWarps][2][4][32];
};

template <int FORM>
struct Op {
  uint32_t a[mtiles_of(FORM)][4], b[4][2], cv[2], cm[2];
};

//: A STEP'S ROWS: entry `e` of the warp's list `j` at step `s` (merged: list 0's entries, the step's rows), past the
//: live count row 0 and not in
__device__ __forceinline__ int entry_row(const Shared& sh, int w, int j, int s, int e, bool& in) {
  const uint16_t x = sh.entries[w][j][s][e];
  in = x != kPad;
  return in ? x : 0;
}

//: THE STEP'S OPERANDS: A (level S's digits, one or two m-tiles), B (four n-tiles of values), and c at the lane's four
//: tokens -- per list the step's list's digit; merged the lane group's list (`g` or `g & 3`)
template <int FORM>
__device__ __forceinline__ void load(Op<FORM>& o, const Shared& sh, int w, int j, int s, int lane,
                                     float pu, float pv) {
  const int r = lane & 7, q = lane >> 3, g = lane >> 2, tq = lane & 3;
  bool unused;
  const int ra = entry_row(sh, w, j, s, 8 * (q >> 1) + r, unused);
  const int rb = entry_row(sh, w, j, s, 8 * (q & 1) + r, unused);
#pragma unroll
  for (int mt = 0; mt < mtiles_of(FORM); ++mt)
    lab::ldsm_t(o.a[mt], lab::smem(sh.rows + at(ra, 2 * mt + (q & 1))));
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    uint32_t x[4];
    lab::ldsm_t(x, lab::smem(sh.rows + at(rb, 4 + 2 * h + (q >> 1))));
    o.b[2 * h][0] = x[0];
    o.b[2 * h][1] = x[1];
    o.b[2 * h + 1][0] = x[2];
    o.b[2 * h + 1][1] = x[3];
  }
  const int list = merged(FORM) ? (lists_of(FORM) == 8 ? g : (g & 3)) : j;
  const int digit = (lists_of(FORM) * w + list) & 31;
  float v[4], m[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    bool in;
    const int row = entry_row(sh, w, j, s, 2 * tq + (k & 1) + 8 * (k >> 1), in);
    const float b = static_cast<float>(sh.rows[at(row, 8 + (digit >> 4)) + (digit & 15)]);
    const float2 f = sh.fm[row];
    v[k] = in ? __fmul_rn(__fmul_rn(b, f.x), pu) : 0.0f;
    m[k] = in ? __fmul_rn(__fmul_rn(b, f.y), pv) : 0.0f;
  }
  o.cv[0] = lab::pack_half2(v[0], v[1]);
  o.cv[1] = lab::pack_half2(v[2], v[3]);
  o.cm[0] = lab::pack_half2(m[0], m[1]);
  o.cm[1] = lab::pack_half2(m[2], m[3]);
}

template <int FORM>
struct Acc {
  float acc[lists_of(FORM)][mtiles_of(FORM)][4][4];
};

//: A LIST'S ATOMS on a step: A times its c (the lane's own tokens' c, or the list's lanes' by two shuffles), four
//: n-tiles an m-tile
template <int FORM, int J>
__device__ __forceinline__ void list_atoms(Acc<FORM>& st, const Op<FORM>& o, uint32_t c01,
                                           uint32_t c89) {
#pragma unroll
  for (int mt = 0; mt < mtiles_of(FORM); ++mt) {
    const uint32_t aj[4] = {lab::hmul2(o.a[mt][0], c01), lab::hmul2(o.a[mt][1], c01),
                            lab::hmul2(o.a[mt][2], c89), lab::hmul2(o.a[mt][3], c89)};
#pragma unroll
    for (int n = 0; n < 4; ++n) lab::mma_f16_f32(st.acc[J][mt][n], aj, o.b[n][0], o.b[n][1]);
  }
}

//: THE MASS on a step: A (the raw digits) by B = c_m, a fresh fragment an m-tile added into the sums
template <int FORM>
__device__ __forceinline__ void mass_atoms(float (&m)[2][4], const Op<FORM>& o) {
#pragma unroll
  for (int mt = 0; mt < mtiles_of(FORM); ++mt) {
    float d[4];
    lab::mma_f16_f32_fresh(d, o.a[mt], o.cm[0], o.cm[1]);
#pragma unroll
    for (int i = 0; i < 4; ++i) m[mt][i] += d[i];
  }
}

template <int FORM>
__device__ __forceinline__ void merged_step(Acc<FORM>& st, float (&m)[2][4], const Op<FORM>& o,
                                            int lane) {
  const int tq = lane & 3;
  mass_atoms<FORM>(m, o);
  static_for<lists_of(FORM)>([&](auto J) {
    constexpr int j = decltype(J)::value;
    list_atoms<FORM, j>(st, o, __shfl_sync(0xffffffffu, o.cv[0], 4 * j + tq),
                        __shfl_sync(0xffffffffu, o.cv[1], 4 * j + tq));
  });
}

//: a list's pass mass into the warp's shared sums (one buffer for the warp's lists: the same loads, adds and stores)
template <int FORM>
__device__ __forceinline__ void add_mass(Shared& sh, float (&m)[2][4], int w, int lane) {
#pragma unroll
  for (int mt = 0; mt < mtiles_of(FORM); ++mt)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      sh.mass[w][mt][i][lane] += m[mt][i];
      m[mt][i] = 0.0f;
    }
}

template <int FORM, int STEPS>
__device__ __forceinline__ void list_pass(Acc<FORM>& st, Shared& sh, int w, int lane, float pu,
                                          float pv) {
  static_for<lists_of(FORM)>([&](auto J) {
    constexpr int j = decltype(J)::value;
    float m[2][4] = {};
    if constexpr (FORM == 4) {
      //: staged: step s + 1's operands loaded before step s's MMAs, two register sets alternating
      Op<FORM> cur, nxt;
      load<FORM>(cur, sh, w, j, 0, lane, pu, pv);
#pragma unroll 1
      for (int s = 0; s < STEPS; s += 2) {
        if (s + 1 < STEPS) load<FORM>(nxt, sh, w, j, s + 1, lane, pu, pv);
        mass_atoms<FORM>(m, cur);
        list_atoms<FORM, j>(st, cur, cur.cv[0], cur.cv[1]);
        if (s + 1 < STEPS) {
          if (s + 2 < STEPS) load<FORM>(cur, sh, w, j, s + 2, lane, pu, pv);
          mass_atoms<FORM>(m, nxt);
          list_atoms<FORM, j>(st, nxt, nxt.cv[0], nxt.cv[1]);
        }
      }
    } else {
#pragma unroll 1
      for (int s = 0; s < STEPS; s += 2) {
        Op<FORM> o0, o1;
        load<FORM>(o0, sh, w, j, s, lane, pu, pv);
        if (s + 1 < STEPS) load<FORM>(o1, sh, w, j, s + 1, lane, pu, pv);
        mass_atoms<FORM>(m, o0);
        list_atoms<FORM, j>(st, o0, o0.cv[0], o0.cv[1]);
        if (s + 1 < STEPS) {
          mass_atoms<FORM>(m, o1);
          list_atoms<FORM, j>(st, o1, o1.cv[0], o1.cv[1]);
        }
      }
    }
    add_mass<FORM>(sh, m, w, lane);
  });
}

template <int FORM, int STEPS>
__device__ __forceinline__ void merged_pass(Acc<FORM>& st, Shared& sh, int w, int lane, float pu,
                                            float pv) {
  float m[2][4] = {};
#pragma unroll 1
  for (int s = 0; s < STEPS; s += 2) {
    Op<FORM> o0, o1;
    load<FORM>(o0, sh, w, 0, s, lane, pu, pv);
    load<FORM>(o1, sh, w, 0, s + 1, lane, pu, pv);
    merged_step<FORM>(st, m, o0, lane);
    merged_step<FORM>(st, m, o1, lane);
  }
  add_mass<FORM>(sh, m, w, lane);
}

uint32_t hash3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA77u ^ (c + 0x165667B1u) * 0xC2B2AE3Du;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  return h ^ (h >> 12);
}

template <int FORM, int STEPS, int LIVE, int GATHER>
__global__ void __launch_bounds__(kThreads, 1) fold(int passes, const uint16_t* entries,
                                                    long long* t, int* out) {
  extern __shared__ __align__(128) unsigned char raw[];
  Shared& sh = *reinterpret_cast<Shared*>(raw);
  const int lane = lab::lane_id(), w = lab::warp_id(), tid = static_cast<int>(threadIdx.x);
#pragma unroll 1
  for (int i = tid; i < kRows * kPitch / 2; i += kThreads)
    reinterpret_cast<uint16_t*>(sh.rows)[i] = static_cast<uint16_t>(0x2c00 + ((i * 7) & 255));
#pragma unroll 1
  for (int i = tid; i < kRows; i += kThreads)
    sh.fm[i] = make_float2(1e-3f * (1 + (i & 7)), 1e-3f * (1 + (i & 3)));
#pragma unroll 1
  for (int i = tid; i < kWarps * 8 * kMaxSteps * kStep; i += kThreads)
    (&sh.entries[0][0][0][0])[i] = entries[i];
#pragma unroll 1
  for (int i = tid; i < kWarps * 2 * 4 * 32; i += kThreads) (&sh.mass[0][0][0][0])[i] = 0.0f;
  __syncthreads();
  const float pu = 0.5f + 1e-3f * lane, pv = 0.25f + 1e-3f * lane;
  Acc<FORM> st = {};
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int x = 0; x < passes; ++x) {
    if constexpr (merged(FORM))
      merged_pass<FORM, STEPS>(st, sh, w, lane, pu, pv);
    else
      list_pass<FORM, STEPS>(st, sh, w, lane, pu, pv);
  }
  clock.stop(t);
  uint32_t fold = lab::bits(sh.mass[w][0][0][lane]);
  for (int j = 0; j < lists_of(FORM); ++j)
    for (int mt = 0; mt < mtiles_of(FORM); ++mt)
      for (int n = 0; n < 4; ++n)
        for (int i = 0; i < 4; ++i) fold += lab::bits(st.acc[j][mt][n][i]);
  lab::sink(out, fold);
}

//: THE ENTRIES: per warp, list and step, `LIVE` rows then pads -- gathered rows hashed over the 256, or the step's own
//: contiguous 16 (merged forms always contiguous: the union's rows)
std::vector<uint16_t> make_entries(int live, bool gather) {
  std::vector<uint16_t> e(kWarps * 8 * kMaxSteps * kStep);
  for (int w = 0; w < kWarps; ++w)
    for (int j = 0; j < 8; ++j)
      for (int s = 0; s < kMaxSteps; ++s)
        for (int i = 0; i < kStep; ++i) {
          const size_t at = ((static_cast<size_t>(w) * 8 + j) * kMaxSteps + s) * kStep + i;
          if (i >= live) {
            e[at] = kPad;
          } else if (gather) {
            e[at] = static_cast<uint16_t>(hash3(w, j * kMaxSteps + s, i) % kRows);
          } else {
            e[at] = static_cast<uint16_t>((s * kStep + i) % kRows);
          }
        }
  return e;
}

constexpr int kPasses = 400;

template <int FORM, int STEPS, int LIVE, int GATHER>
void variant(const lab::Bench& b) {
  const std::string name = lab::kernel("fold", {FORM, STEPS, LIVE, GATHER});
  b.expect(name, "HMMA.16816.F32", ">=", lists_of(FORM) * mtiles_of(FORM) * 4);
  b.expect(name, "LDSM", ">=", mtiles_of(FORM) + 2);
  b.expect(name, "HMUL2", ">=", lists_of(FORM) * mtiles_of(FORM) * 4);
  b.expect(name, "BSSY", ">=", 1);
  if (b.expect_only) return;
  const std::vector<uint16_t> h = make_entries(LIVE, GATHER != 0);
  uint16_t* d = nullptr;
  LAB_CHECK(cudaMalloc(&d, h.size() * sizeof(uint16_t)));
  LAB_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
  const int bytes = static_cast<int>(sizeof(Shared));
  LAB_CHECK(cudaFuncSetAttribute(fold<FORM, STEPS, LIVE, GATHER>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  const lab::Samples s = b.variant(
      [&] {
        fold<FORM, STEPS, LIVE, GATHER><<<b.sms, kThreads, bytes>>>(kPasses, d, b.times, b.sinks);
      },
      [&] { return b.span(kWarps) / kPasses; });
  //: the busiest scheduler's two warps' MMAs at 32 cycles: every warp the same work
  const double floor = 2.0 * mmas_of(FORM, STEPS) * 32;
  const std::string label = std::string(kFormName[FORM]) + " / " + std::to_string(STEPS)
                            + (merged(FORM) ? " steps" : " steps a list") + " / "
                            + std::to_string(LIVE) + " live"
                            + (GATHER ? " / gathered" : " / contiguous");
  b.record(label, "cycles a pass", s);
  lab::Samples over;
  for (double c : s.v) over.v.push_back(c / floor);
  b.record(label + " / floor", "x", over);
  LAB_CHECK(cudaFree(d));
}

}  // namespace list_fold

int main(int argc, char** argv) {
  using namespace list_fold;
  lab::Bench b(argc, argv);
  //: dense: merged at both tiles against per list reloading every list's operands (the same 128 rows each list)
  variant<0, 8, 16, 0>(b);
  variant<2, 8, 16, 0>(b);
  variant<1, 8, 16, 0>(b);
  variant<4, 8, 16, 0>(b);
  variant<3, 8, 16, 0>(b);
  //: sparse per list, gathered: full steps, and half full (alt-k4's ~8 tokens a list a window)
  variant<1, 2, 16, 1>(b);
  variant<4, 2, 16, 1>(b);
  variant<1, 2, 8, 1>(b);
  variant<1, 1, 8, 1>(b);
  variant<3, 2, 16, 1>(b);
  variant<3, 1, 8, 1>(b);
}
