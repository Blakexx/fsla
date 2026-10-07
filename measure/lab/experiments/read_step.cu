// measure/lab/experiments/read_step.cu -- QUESTION: the readout's step on the even carve (a warp owns 4 level-S read
// digits x 32 level-R digits x 32 channels): per digit (each digit over its own read tokens, gathered) or merged (the
// warp's digits summed over the union in registers), what does a step cost beside its tensor floor, what do its output
// adds into the shared panel add, and does draining the panel to global memory cost anything once it is double
// buffered and spread through the next pass, against draining it in a burst at the pass's end? METHOD: 8 warps, two a
// scheduler, the fold's 128 fp32 accumulators live beside the published slice (B fragments: K = level R, permuted so a
// lane's four k rows are one token's four consecutive digits, N = channels); a step's A = 16 tokens' level-R bytes by
// one `ldmatrix.x4` (rows = tokens: 96-byte read rows, level R's 32 bytes then level S's, the 16-byte halves swapped on
// rows with bit 2 set) and a PRMT + HFMA2 a register to f16 with the token's level-S coefficient; 8
// fp16-accumulate MMAs and 2 for den (the mass as a fifth n-tile); the outputs to s16 pairs (two channels a word) added
// by `red.shared.add` into a 128-slot panel (17 words a slot); the drain two slots a warp request
// (`red.global.add.u32`, scattered over 16 MB), each word cleared as it goes. The block's span a pass against its
// tensor floor (the busiest scheduler's two warps' MMAs at 16 cycles). see docs/internals/tools/lab.md#read-step
#include "lab.cuh"

#include <utility>

namespace read_step {

template <class F, int... I>
__device__ __forceinline__ void static_for_impl(F&& f, std::integer_sequence<int, I...>) {
  (f(std::integral_constant<int, I>{}), ...);
}

//: `f(std::integral_constant<int, i>)` for i in [0, N), unrolled
template <int N, class F>
__device__ __forceinline__ void static_for(F&& f) {
  static_for_impl(f, std::make_integer_sequence<int, N>{});
}

constexpr int kThreads = 256, kWarps = 8, kSlots = 128, kRowBytes = 96, kPanelPitch = 17;
constexpr int kDigits = 4, kMaxSteps = 8, kStep = 16, kTargets = 1 << 18;
constexpr uint16_t kPad = 0xffffu;
constexpr uint32_t kHalfMagic = 0x64646464u;
//: a warp's drain requests a pass: its 16 slots, two a request
constexpr int kShare = kSlots / kWarps / 2;

//: MODE 0 the steps alone (outputs folded into the sink); 1 + the panel adds; 2 + the drain in a burst at the pass's end
//: (one panel, a CTA barrier either side); 3 + the drain spread: two panels, a warp draining its share of the last
//: pass's panel a request after each step, under mbarriers (`complete` when every warp's adds are in, `drained` when
//: every warp's share is out). Modes 2 and 3 drain 17-word rows (a slot's 16 channel words and den) to a 17-word target
//: pitch, den a request of its own; 4 and 5 are 2 and 3 with the channel words alone to 64-byte rows (den carried
//: elsewhere), so a request is a token's half line each; 6 no den (the den pass carries it), each step's adds issued
//: after the next step's MMAs; 7 is 6 with the aligned burst drain; 8 (per digit) is 6 with each step's operands loaded
//: before the step ahead's MMAs, across digits
constexpr int kModes = 9;
const char* kModeName[kModes] = {"steps alone",
                                 "+ panel adds",
                                 "+ drain, burst at the pass end",
                                 "+ drain, double panel, spread",
                                 "+ drain, burst, aligned rows, no den",
                                 "+ drain, spread, aligned rows, no den",
                                 "no den, adds a step behind",
                                 "no den, adds a step behind, + aligned burst drain",
                                 "no den, operands a step ahead, adds a step behind"};

//: FORM 0 per digit (a step = one digit's 16 tokens, its outputs added); 1 merged (a step = 16 union tokens, every
//: digit's MMAs into one accumulator, one add)
const char* kFormName[2] = {"per digit", "merged"};

struct alignas(16) Shared {
  uint8_t rows[kSlots * kRowBytes];
  uint32_t panel[2][kSlots * kPanelPitch];
  uint16_t entries[kWarps][kDigits][kMaxSteps][kStep];
  uint64_t complete[2], drained[2];
};

__device__ __forceinline__ int at(int r, int c) {
  return r * kRowBytes + 16 * (c ^ ((r >> 2) & 1));
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

__device__ __forceinline__ uint32_t hfma2(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

__device__ __forceinline__ void red_shared(uint32_t a, uint32_t v) {
  asm volatile("red.shared.add.u32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}

__device__ __forceinline__ void red_global(uint32_t* p, uint32_t v) {
  asm volatile("red.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

//: an f16 pair at the panel's unit to two s16 in a word
__device__ __forceinline__ uint32_t to_s16x2(uint32_t h) {
  const __half2 x = __hmul2(*reinterpret_cast<const __half2*>(&h), __float2half2_rn(64.0f));
  const int lo = __half2short_rn(__low2half(x)), hi = __half2short_rn(__high2half(x));
  return (static_cast<uint32_t>(lo) & 0xffffu) | (static_cast<uint32_t>(hi) << 16);
}

struct State {
  uint32_t b[kDigits][2][4][2];
  uint32_t mass[kDigits][2][2];
};

//: A STEP'S A for 16 tokens (slots `ra` of the lane's ldmatrix row) at a digit: the level-R bytes to f16 at the
//: token's level-S coefficient (tokens g and g + 8 for the lane's rows)
__device__ __forceinline__ void step_a(uint32_t (&a)[2][4], const Shared& sh, int ra, int khalf,
                                       int rg, int rg8, int digit, bool in_g, bool in_g8) {
  uint32_t x[4];
  lab::ldsm(x, lab::smem(sh.rows + at(ra, khalf)));
  const float sg = in_g ? (1.0f + sh.rows[at(rg, 2) + digit]) * (1.0f / 65536.0f) : 0.0f;
  const float s8 = in_g8 ? (1.0f + sh.rows[at(rg8, 2) + digit]) * (1.0f / 65536.0f) : 0.0f;
  const uint32_t scale_g = lab::pack_half2(sg, sg), scale_g8 = lab::pack_half2(s8, s8);
  const uint32_t bias_g = lab::pack_half2(-1024.0f * sg, -1024.0f * sg);
  const uint32_t bias_g8 = lab::pack_half2(-1024.0f * s8, -1024.0f * s8);
#pragma unroll
  for (int kk = 0; kk < 2; ++kk) {
    a[kk][0] = hfma2(prmt(x[2 * kk], kHalfMagic, 0x4140u), scale_g, bias_g);
    a[kk][1] = hfma2(prmt(x[2 * kk + 1], kHalfMagic, 0x4140u), scale_g8, bias_g8);
    a[kk][2] = hfma2(prmt(x[2 * kk], kHalfMagic, 0x4342u), scale_g, bias_g);
    a[kk][3] = hfma2(prmt(x[2 * kk + 1], kHalfMagic, 0x4342u), scale_g8, bias_g8);
  }
}

//: the step's rows: the lane's ldmatrix row (`8 (q & 1) + r` of the step's 16, k half `q >> 1`) and its two output tokens
__device__ __forceinline__ int slot_of(const Shared& sh, int w, int d, int s, int e, bool& in) {
  const uint16_t x = sh.entries[w][d][s][e];
  in = x != kPad;
  return in ? x : 0;
}

template <int D0, int DIGITS, bool kDen>
__device__ __forceinline__ void mmas(uint32_t (&out)[4][2], uint32_t (&den)[2], const State& st,
                                     const uint32_t (&a)[2][4]) {
  constexpr int d0 = D0;
#pragma unroll
  for (int dd = 0; dd < DIGITS; ++dd)
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
#pragma unroll
      for (int n = 0; n < 4; ++n)
        lab::mma_f16_f16(out[n], a[kk], st.b[d0 + dd][kk][n][0], st.b[d0 + dd][kk][n][1]);
      if constexpr (kDen)
        lab::mma_f16_f16(den, a[kk], st.mass[d0 + dd][kk][0], st.mass[d0 + dd][kk][1]);
    }
}

//: THE STEP'S OUTPUTS into panel `p`: lane (g, tq) holds tokens g and g + 8, channels 8 n + 2 tq, + 1 -- a word each;
//: den by the tq = 0 lanes
template <bool kDen>
__device__ __forceinline__ void panel_add(Shared& sh, int p, const uint32_t (&out)[4][2],
                                          const uint32_t (&den)[2], int slot_g, int slot_g8,
                                          bool in_g, bool in_g8, int lane) {
  const int tq = lane & 3;
  const uint32_t base = lab::smem(sh.panel[p]);
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    if (in_g) red_shared(base + 4 * (slot_g * kPanelPitch + 4 * n + tq), to_s16x2(out[n][0]));
    if (in_g8) red_shared(base + 4 * (slot_g8 * kPanelPitch + 4 * n + tq), to_s16x2(out[n][1]));
  }
  if (kDen && tq == 0) {
    if (in_g) red_shared(base + 4 * (slot_g * kPanelPitch + 16), to_s16x2(den[0]));
    if (in_g8) red_shared(base + 4 * (slot_g8 * kPanelPitch + 16), to_s16x2(den[1]));
  }
}

//: ONE DRAIN REQUEST: slots `2 i` and `2 i + 1` of the warp's share of panel `p`, a lane a word (lanes 0-15 the first,
//: 16-31 the second; den by lanes 0 and 16 after), to the slots' scattered targets, each word cleared
template <bool kAligned>
__device__ __forceinline__ void drain_request(Shared& sh, int p, uint32_t* target, int pass, int i,
                                              int lane) {
  const int w = lab::warp_id(), slot = (kSlots / kWarps) * w + 2 * i + (lane >> 4),
            word = lane & 15;
  const uint32_t token =
      (static_cast<uint32_t>(blockIdx.x) * 7919u + pass * 131u + slot * 37u) % kTargets;
  uint32_t* const at = &sh.panel[p][slot * kPanelPitch + word];
  red_global(target + token * (kAligned ? 16u : 17u) + word, *at);
  *at = 0u;
  if (!kAligned && word == 0) {
    uint32_t* const d = &sh.panel[p][slot * kPanelPitch + 16];
    red_global(target + token * 17u + 16, *d);
    *d = 0u;
  }
}

uint32_t hash3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA77u ^ (c + 0x165667B1u) * 0xC2B2AE3Du;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  return h ^ (h >> 12);
}

template <int MODE, int FORM, int STEPS, int LIVE>
__global__ void __launch_bounds__(kThreads, 1) read(int passes, const uint16_t* entries,
                                                    uint32_t* target, const uint32_t* init,
                                                    long long* t, int* out) {
  extern __shared__ __align__(128) unsigned char raw[];
  Shared& sh = *reinterpret_cast<Shared*>(raw);
  const int lane = lab::lane_id(), w = lab::warp_id(), tid = static_cast<int>(threadIdx.x);
  const int r = lane & 7, q = lane >> 3, g = lane >> 2;
#pragma unroll 1
  for (int i = tid; i < kSlots * kRowBytes / 4; i += kThreads)
    reinterpret_cast<uint32_t*>(sh.rows)[i] = 0x01010101u * static_cast<uint32_t>(1 + (i & 63));
#pragma unroll 1
  for (int i = tid; i < 2 * kSlots * kPanelPitch; i += kThreads) (&sh.panel[0][0])[i] = 0u;
#pragma unroll 1
  for (int i = tid; i < kWarps * kDigits * kMaxSteps * kStep; i += kThreads)
    (&sh.entries[0][0][0][0])[i] = entries[i];
  if (tid < 2) {
    lab::mbar_init(lab::smem(&sh.complete[tid]), kThreads);
    lab::mbar_init(lab::smem(&sh.drained[tid]), kThreads);
  }
  __syncthreads();
  //: the fold's accumulators, live across the readout as in the owner kernel, and the published slice -- both loaded
  //: from memory, so neither is rematerialized inside the loop
  const uint32_t* const mine = init + (static_cast<size_t>(blockIdx.x) * kThreads + tid) * 208;
  float fold[128];
#pragma unroll
  for (int i = 0; i < 128; ++i) fold[i] = __uint_as_float(mine[i]);
  State st;
#pragma unroll
  for (int d = 0; d < kDigits; ++d)
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        st.b[d][kk][n][0] = mine[128 + ((d * 2 + kk) * 4 + n) * 2];
        st.b[d][kk][n][1] = mine[128 + ((d * 2 + kk) * 4 + n) * 2 + 1];
      }
      st.mass[d][kk][0] = mine[192 + (d * 2 + kk) * 2];
      st.mass[d][kk][1] = mine[192 + (d * 2 + kk) * 2 + 1];
    }
  uint32_t fold_out = 0u;
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int x = 0; x < passes; ++x) {
    constexpr bool kSpread = MODE == 3 || MODE == 5, kBurst = MODE == 2 || MODE == 4 || MODE == 7,
                   kAligned = MODE >= 4, kDen = MODE < 6, kBehind = MODE >= 6;
    const int p = kSpread ? (x & 1) : 0, other = p ^ 1;
    if constexpr (kSpread) {
      if (x >= 2)
        lab::mbar_wait(lab::smem(&sh.drained[p]), static_cast<uint32_t>((x - 2) >> 1) & 1u);
    }
    int drained_requests = 0;
    [[maybe_unused]] uint32_t held[4][2] = {};
    [[maybe_unused]] int held_g = 0, held_g8 = 0;
    [[maybe_unused]] bool held_in_g = false, held_in_g8 = false, holding = false;
    const auto spread = [&]() {
      if constexpr (kSpread) {
        if (x >= 1 && drained_requests < kShare) {
          if (drained_requests == 0)
            lab::mbar_wait(lab::smem(&sh.complete[other]),
                           static_cast<uint32_t>((x - 1) >> 1) & 1u);
          drain_request<kAligned>(sh, other, target, x - 1, drained_requests, lane);
          ++drained_requests;
        }
      }
    };
    const auto one_step = [&](auto digit0, int s, auto digits) {
      constexpr int d0 = decltype(digit0)::value, kDigitsHere = decltype(digits)::value;
      bool in_a, in_g, in_g8;
      const int ra = slot_of(sh, w, d0, s, 8 * (q & 1) + r, in_a);
      const int rg = slot_of(sh, w, d0, s, g, in_g), rg8 = slot_of(sh, w, d0, s, g + 8, in_g8);
      uint32_t a[2][4];
      step_a(a, sh, ra, q >> 1, rg, rg8, d0, in_g, in_g8);
      uint32_t o[4][2] = {}, den[2] = {};
      mmas<d0, kDigitsHere, kDen>(o, den, st, a);
      if constexpr (MODE == 0) {
#pragma unroll
        for (int n = 0; n < 4; ++n) fold_out += o[n][0] ^ o[n][1];
        fold_out += den[0] ^ den[1];
      } else if constexpr (kBehind) {
        //: the last step's adds after this step's MMAs, this step's held for the next
        if (holding)
          panel_add<false>(sh, p, held, den, held_g, held_g8, held_in_g, held_in_g8, lane);
#pragma unroll
        for (int n = 0; n < 4; ++n) {
          held[n][0] = o[n][0];
          held[n][1] = o[n][1];
        }
        held_g = rg;
        held_g8 = rg8;
        held_in_g = in_g;
        held_in_g8 = in_g8;
        holding = true;
      } else {
        panel_add<kDen>(sh, p, o, den, rg, rg8, in_g, in_g8, lane);
      }
      spread();
    };
    if constexpr (MODE == 8) {
      //: the warp's digit steps as one sequence: the next step's operands loaded before this step's MMAs
      const auto load = [&](int item, uint32_t(&a)[2][4], int& rg, int& rg8, bool& ig, bool& ig8) {
        const int dd = item / STEPS, ss = item % STEPS;
        bool ia;
        const int ra = slot_of(sh, w, dd, ss, 8 * (q & 1) + r, ia);
        rg = slot_of(sh, w, dd, ss, g, ig);
        rg8 = slot_of(sh, w, dd, ss, g + 8, ig8);
        step_a(a, sh, ra, q >> 1, rg, rg8, dd, ig, ig8);
      };
      uint32_t a[2][4], an[2][4];
      int rg = 0, rg8 = 0, rgn = 0, rg8n = 0;
      bool ig = false, ig8 = false, ign = false, ig8n = false;
      load(0, a, rg, rg8, ig, ig8);
#pragma unroll 1
      for (int item = 0; item < kDigits * STEPS; ++item) {
        if (item + 1 < kDigits * STEPS) load(item + 1, an, rgn, rg8n, ign, ig8n);
        uint32_t o[4][2] = {}, den[2] = {};
        switch (item / STEPS) {
          case 0:
            mmas<0, 1, false>(o, den, st, a);
            break;
          case 1:
            mmas<1, 1, false>(o, den, st, a);
            break;
          case 2:
            mmas<2, 1, false>(o, den, st, a);
            break;
          default:
            mmas<3, 1, false>(o, den, st, a);
            break;
        }
        if (holding)
          panel_add<false>(sh, p, held, den, held_g, held_g8, held_in_g, held_in_g8, lane);
#pragma unroll
        for (int n = 0; n < 4; ++n) {
          held[n][0] = o[n][0];
          held[n][1] = o[n][1];
        }
        held_g = rg;
        held_g8 = rg8;
        held_in_g = ig;
        held_in_g8 = ig8;
        holding = true;
#pragma unroll
        for (int kk = 0; kk < 2; ++kk)
#pragma unroll
          for (int i = 0; i < 4; ++i) a[kk][i] = an[kk][i];
        rg = rgn;
        rg8 = rg8n;
        ig = ign;
        ig8 = ig8n;
      }
    } else if constexpr (FORM == 0) {
      static_for<kDigits>([&](auto D) {
#pragma unroll 1
        for (int s = 0; s < STEPS; ++s) one_step(D, s, std::integral_constant<int, 1>{});
      });
    } else {
#pragma unroll 1
      for (int s = 0; s < STEPS; ++s)
        one_step(std::integral_constant<int, 0>{}, s, std::integral_constant<int, kDigits>{});
    }
    if constexpr (kBehind) {
      const uint32_t none[2] = {};
      if (holding)
        panel_add<false>(sh, p, held, none, held_g, held_g8, held_in_g, held_in_g8, lane);
    }
    if constexpr (kSpread) {
#pragma unroll 1
      while (x >= 1 && drained_requests < kShare) spread();
      if (x >= 1) lab::mbar_arrive(lab::smem(&sh.drained[other]));
      lab::mbar_arrive(lab::smem(&sh.complete[p]));
    }
    if constexpr (kBurst) {
      __syncthreads();
#pragma unroll 1
      for (int i = 0; i < kShare; ++i) drain_request<kAligned>(sh, 0, target, x, i, lane);
      __syncthreads();
    }
  }
  if constexpr (MODE == 3 || MODE == 5) {
    //: the last pass's panel, exposed
    lab::mbar_wait(lab::smem(&sh.complete[(passes - 1) & 1]),
                   static_cast<uint32_t>((passes - 1) >> 1) & 1u);
#pragma unroll 1
    for (int i = 0; i < kShare; ++i)
      drain_request<MODE >= 4>(sh, (passes - 1) & 1, target, passes - 1, i, lane);
  }
  clock.stop(t);
  uint32_t f = fold_out;
#pragma unroll
  for (int i = 0; i < 128; ++i) f += lab::bits(fold[i]);
  lab::sink(out, f);
}

//: THE ENTRIES: per warp, digit and step, `LIVE` slots then pads -- per digit distinct gathered slots (a hashed start,
//: then 37 apart over the 128: a list never holds a token twice); merged
//: the step's own 16 (digit 0's entries, the union)
std::vector<uint16_t> make_entries(int form, int live) {
  std::vector<uint16_t> e(kWarps * kDigits * kMaxSteps * kStep);
  for (int w = 0; w < kWarps; ++w)
    for (int d = 0; d < kDigits; ++d)
      for (int s = 0; s < kMaxSteps; ++s)
        for (int i = 0; i < kStep; ++i) {
          const size_t at = ((static_cast<size_t>(w) * kDigits + d) * kMaxSteps + s) * kStep + i;
          if (i >= live)
            e[at] = kPad;
          else if (form == 0)
            e[at] = static_cast<uint16_t>((hash3(w, d * kMaxSteps + s, 0) + 37u * i) % kSlots);
          else
            e[at] = static_cast<uint16_t>((s * kStep + i) % kSlots);
        }
  return e;
}

constexpr int kPasses = 400;

template <int MODE, int FORM, int STEPS, int LIVE>
void variant(const lab::Bench& b, uint32_t* target, const uint32_t* init) {
  const std::string name = lab::kernel("read", {MODE, FORM, STEPS, LIVE});
  b.expect(name, "HMMA.16816.F16", ">=", MODE >= 6 ? 8 : 10);
  b.expect(name, "LDSM", ">=", 1);
  if (MODE != 0) b.expect(name, "ATOMS", ">=", 8);
  if (MODE >= 2 && MODE != 6 && MODE != 8) b.expect(name, "RED", ">=", 1);
  if (b.expect_only) return;
  const std::vector<uint16_t> h = make_entries(FORM, LIVE);
  uint16_t* d = nullptr;
  LAB_CHECK(cudaMalloc(&d, h.size() * sizeof(uint16_t)));
  LAB_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
  const int bytes = static_cast<int>(sizeof(Shared));
  LAB_CHECK(cudaFuncSetAttribute(read<MODE, FORM, STEPS, LIVE>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  const lab::Samples s = b.variant(
      [&] {
        read<MODE, FORM, STEPS, LIVE>
            <<<b.sms, kThreads, bytes>>>(kPasses, d, target, init, b.times, b.sinks);
      },
      [&] { return b.span(kWarps) / kPasses; });
  //: a warp's MMAs a pass at 16 cycles, two warps a scheduler: per digit 4 digits x STEPS steps x 10 (8 without den);
  //: merged STEPS x 40 (32)
  const double per = MODE >= 6 ? 8.0 : 10.0;
  const double mmas = FORM == 0 ? kDigits * STEPS * per : STEPS * 4 * per;
  const double floor = 2.0 * mmas * 16;
  const std::string label = std::string(kFormName[FORM]) + " / " + std::to_string(STEPS)
                            + (FORM == 0 ? " steps a digit" : " steps") + " / "
                            + std::to_string(LIVE) + " live / " + kModeName[MODE];
  b.record(label, "cycles a pass", s);
  lab::Samples over;
  for (double c : s.v) over.v.push_back(c / floor);
  b.record(label + " / floor", "x", over);
  LAB_CHECK(cudaFree(d));
}

template <int FORM, int STEPS, int LIVE>
void modes(const lab::Bench& b, uint32_t* target, const uint32_t* init) {
  variant<0, FORM, STEPS, LIVE>(b, target, init);
  variant<1, FORM, STEPS, LIVE>(b, target, init);
  variant<2, FORM, STEPS, LIVE>(b, target, init);
  variant<3, FORM, STEPS, LIVE>(b, target, init);
  variant<4, FORM, STEPS, LIVE>(b, target, init);
  variant<5, FORM, STEPS, LIVE>(b, target, init);
  variant<6, FORM, STEPS, LIVE>(b, target, init);
  variant<7, FORM, STEPS, LIVE>(b, target, init);
  if constexpr (FORM == 0) variant<8, FORM, STEPS, LIVE>(b, target, init);
}

}  // namespace read_step

int main(int argc, char** argv) {
  using namespace read_step;
  lab::Bench b(argc, argv);
  uint32_t *target = nullptr, *init = nullptr;
  if (!b.expect_only) {
    LAB_CHECK(cudaMalloc(&target, static_cast<size_t>(kTargets) * 17 * sizeof(uint32_t)));
    LAB_CHECK(cudaMemset(target, 0, static_cast<size_t>(kTargets) * 17 * sizeof(uint32_t)));
    const size_t words = static_cast<size_t>(b.sms) * kThreads * 208;
    std::vector<uint32_t> h(words);
    for (size_t i = 0; i < words; ++i)
      h[i] = (i % 208) < 128 ? 0x3a800000u + (i & 1023) : 0x2c002c00u ^ (i & 255);
    LAB_CHECK(cudaMalloc(&init, words * sizeof(uint32_t)));
    LAB_CHECK(cudaMemcpy(init, h.data(), words * sizeof(uint32_t), cudaMemcpyHostToDevice));
  }
  //: dense: merged (one add a step for 4 digits) against per digit (an add a digit step)
  modes<1, 8, 16>(b, target, init);
  modes<0, 8, 16>(b, target, init);
  //: sparse per digit, gathered: full steps, and half full (alt-k4's ~8 tokens a digit a window)
  modes<0, 2, 16>(b, target, init);
  modes<0, 1, 8>(b, target, init);
}
