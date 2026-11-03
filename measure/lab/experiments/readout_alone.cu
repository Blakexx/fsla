// measure/lab/experiments/readout_alone.cu -- QUESTION: what does the carry's readout phase cost a warp-window ALONE
// -- production's `read_chunk` (fp16 accumulate, digits in pairs) over read lists drawn as nl64k-alt-k4's are, the image
// in halves as alt-k4 reads it, nothing else in the window -- against its floors and the composed kernel's readout?
// METHOD: one CTA an SM, the arm-0 plan's tenants; each window a box's 32 read digits of Binomial(16, 1/2) entries over a
// 216-slot union; a warp writes its own digits' entries, then each half (batches 0-1, 2-3) behind a rendezvous, as the
// carry runs it, each half's readout timed. see docs/lab/readout_alone.md
#include <cstdio>
#include <vector>

#include "carry/carry_arm.cuh"
#include "lab.cuh"

namespace readout_alone {

namespace rc = rola::carry;
namespace ops = ::denseref::ops;
using Arm = rola::Arm<0, 32, 1, 32, 1, 32, 0>;
using P = rc::Plan<Arm>;
constexpr int kWindows = 64, kSlots = 216;

//: the draw's hash (the kernel its lists, the host the floor)
__host__ __device__ __forceinline__ uint32_t mix(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

__host__ __device__ __forceinline__ uint32_t draw(int block, int w, int digit) {
  return mix(static_cast<uint32_t>(block) * 7919u + static_cast<uint32_t>(w) * 104729u
             + static_cast<uint32_t>(digit) + 0x9e3779b9u);
}

//: MODE 0 as the carry runs it (a rendezvous before each half); 1 the windows and halves back to back
template <int MODE>
__global__ __launch_bounds__(rc::kThreads, 1) void readout(
    const __grid_constant__ rc::CarryParams p, long long* t, int* out) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = static_cast<int>(threadIdx.x), warp = tid / 32, lane = tid % 32;
  //: the area: the image's half (16 planes), then the read buffer (rows, dummy panel rows and panel, entries)
  const ops::SmemAddr image = ops::smem_addr(smem + P::kAreaAt);
  const ops::SmemAddr buf = image + static_cast<uint32_t>(16 * rc::kPlaneBytes);
  const ops::SmemAddr pads = buf + static_cast<uint32_t>(rc::read_pads<P>(kSlots));
  const ops::SmemAddr ents = buf + static_cast<uint32_t>(rc::read_ents<P>(kSlots));
  uint32_t* const words = reinterpret_cast<uint32_t*>(smem + P::kAreaAt);
  //: the image and the rows: small f16 values; the panel zero; the units 0; each digit's exponent at the image's top
  for (int i = tid; i < (16 * rc::kPlaneBytes + rc::read_pads<P>(kSlots)) / 4; i += rc::kThreads)
    words[i] = 0x24002400u ^ (mix(static_cast<uint32_t>(i)) & 0x03ff03ffu);
  for (int i = tid; i < (rc::read_ents<P>(kSlots) - rc::read_pads<P>(kSlots)) / 4;
       i += rc::kThreads)
    reinterpret_cast<uint32_t*>(smem + (pads - ops::smem_addr(smem)))[i] = 0u;
  for (int i = tid; i < p.tables.heads * P::kWindow; i += rc::kThreads)
    reinterpret_cast<int8_t*>(smem + p.tables.units_at)[i] = 0;
  if (tid < rc::kBoxDigits)
    reinterpret_cast<int*>(smem + P::kExps)[tid] = rc::kImageTop + rc::kCoefTop;
  //: the window at the CTA's own tokens of `den` (the den's adds land on no other CTA's lines)
  const rc::Window rw{P::kWindow, static_cast<int>(blockIdx.x) * P::kWindow, 0, 0, rc::kBoxLists,
                      true};
  __syncthreads();

  //: the fold's state, live across the readout as the carry holds it (empty asm uses: registers, no instructions)
  rc::State<P> st;
#pragma unroll
  for (int k = 0; k < P::kStateRegs; ++k)
    (&st.acc[0][0][0][0])[k] = 1e-3f * static_cast<float>(k + lane);
  long long busy = 0;
  float sink = 0.0f;
#pragma unroll 1
  for (int w = 0; w < kWindows; ++w) {
    //: lane `l` digit `l`'s count and place (the counts before it), as `chunk_runs` gives them
    const int n = __popc(draw(blockIdx.x, w, lane) & 0xffffu);
    int to = n;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, to, o);
      if (lane >= o) to += v;
    }
    const rc::Run run{0, n, to - n};
    //: the warp's own digits' entries: a slot's row (16-byte units, its swizzle folded) and units index, c ~ 1 and the
    //: slot's panel row
    rola::static_for<rc::kBatches>([&](auto I) {
      const int d = warp + rc::kWarps * decltype(I)::value;
      const int nd = __shfl_sync(0xffffffffu, run.n, d), td = __shfl_sync(0xffffffffu, run.to, d);
      for (int e = lane; e < nd; e += 32) {
        const uint32_t h = mix(draw(blockIdx.x, w, d) + 977u * static_cast<uint32_t>(e));
        const uint32_t s = h % kSlots;
        const uint32_t row = s * (P::kReadRow / 16) + rc::chunk_at<P::kReadRow>(s, 0);
        const uint32_t panel = static_cast<uint32_t>(rc::kPadBytes) + rc::panel_row(s);
        reinterpret_cast<uint2*>(smem + (ents - ops::smem_addr(smem)))[td + e] =
            make_uint2(row | (s << 16), (0x3c00u ^ ((h >> 20) & 0xffu)) | (panel << 16));
      }
    });
    __syncwarp();
#pragma unroll 1
    for (int half = 0; half < 2; ++half) {
      if constexpr (MODE == 0) ops::rendezvous();
      const long long t0 = clock64();
      rc::read_chunk<P, true, true>(p, smem, rw, run, 0, kSlots, 2 * half, 2 * half + 2, image,
                                    2 * rc::kBatchPlanes - 1, buf, pads, ents, warp, lane);
      busy += clock64() - t0;
    }
    sink += __uint_as_float(
        reinterpret_cast<const uint32_t*>(smem + (pads - ops::smem_addr(smem)))[lane]);
#pragma unroll
    for (int k = 0; k < P::kStateRegs; ++k) asm volatile("" : "+f"((&st.acc[0][0][0][0])[k]));
  }
#pragma unroll
  for (int k = 0; k < P::kStateRegs; ++k) sink += (&st.acc[0][0][0][0])[k];
  if (lane == 0) t[blockIdx.x * rc::kWarps + warp] = busy;
  lab::sink(out, lab::bits(sink));
}

//: THE TENSOR FLOOR a warp-window: each scheduler's two warps' MMA cycles -- a digit pair's tile 8 HMMA.16816.F16 at
//: 16 cycles, its tiles the pair's longer run's in 8-token tiles -- the mean over the schedulers and the busiest's
struct Floor {
  double mean = 0, busiest = 0;
};

Floor floor_of(int blocks) {
  Floor f;
  for (int b = 0; b < blocks; ++b)
    for (int w = 0; w < kWindows; ++w) {
      double top = 0;
      for (int s = 0; s < 4; ++s) {
        double c = 0;
        for (int warp : {s, s + 4})
          for (int i = 0; i < rc::kBatches; i += 2) {
            const int n0 = __builtin_popcount(draw(b, w, warp + rc::kWarps * i) & 0xffffu);
            const int n1 = __builtin_popcount(draw(b, w, warp + rc::kWarps * (i + 1)) & 0xffffu);
            const int tiles = ((n0 > n1 ? n0 : n1) + 7) / 8;
            c += tiles * 8 * 16.0;
          }
        f.mean += c / 4;
        top = c > top ? c : top;
      }
      f.busiest += top;
    }
  f.mean /= double(blocks) * kWindows;
  f.busiest /= double(blocks) * kWindows;
  return f;
}

template <int MODE>
void variant(const lab::Bench& b, const rc::CarryParams& p, const char* label) {
  const std::string name = lab::kernel("readout", {MODE});
  b.expect(name, "HMMA.16816.F16", ">=", 8);
  b.exempt(name,
           "an isolation of the production readout; its budget is tools/budgets/carry.json's");
  if (b.expect_only) return;
  LAB_CHECK(cudaFuncSetAttribute(readout<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 P::kSmemBytes));
  const lab::Samples s = b.variant(
      [&] { readout<MODE><<<b.sms, rc::kThreads, P::kSmemBytes>>>(p, b.times, b.sinks); },
      [&] { return lab::Bench::mean(b.cycles(rc::kWarps), rc::kWarps, 0, rc::kWarps) / kWindows; });
  b.unchecked(label,
              "a timing isolation; the readout's arithmetic is the oracle's (test_carry_readout)");
  b.record(label, "cycles a warp-window", s);
}

}  // namespace readout_alone

int main(int argc, char** argv) {
  using namespace readout_alone;
  lab::Bench b(argc, argv);
  rc::CarryParams p{};
  p.tables = rc::tables_of<P::kWindow>(1, P::kSmemBytes);
  //: the den's out: a window of tokens a CTA
  p.L = b.sms * P::kWindow;
  LAB_CHECK(cudaMalloc(&p.den, sizeof(float) * p.L));
  LAB_CHECK(cudaMemset(p.den, 0, sizeof(float) * p.L));
  readout_alone::variant<0>(b, p, "composed: a rendezvous a half");
  readout_alone::variant<1>(b, p, "back to back");
  if (!b.expect_only) {
    const Floor f = floor_of(b.sms);
    b.unchecked("tensor floor (mean scheduler)", "computed on the host from the same draw");
    b.unchecked("tensor floor (busiest scheduler)", "computed on the host from the same draw");
    std::printf(
        "{\"lab\": \"result\", \"name\": \"tensor floor (mean scheduler)\", \"unit\": \"cycles a warp-window\", "
        "\"samples\": [%.1f]}\n",
        f.mean);
    std::printf(
        "{\"lab\": \"result\", \"name\": \"tensor floor (busiest scheduler)\", \"unit\": \"cycles a "
        "warp-window\", \"samples\": [%.1f]}\n",
        f.busiest);
  }
  return 0;
}
