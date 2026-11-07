// measure/lab/experiments/read_alone.cu -- QUESTION: what does the value-carve readout cost a warp-window ALONE --
// production's `read_chunk` over a read chunk drawn as nl64k-alt-k4's is, nothing else in the window -- against its
// tensor and shared-memory floors and against the composed kernel's readout phase? METHOD: one CTA an SM, the arm-0
// plan's tenants; each window a 216-slot union whose 32 digits hold ~8 entries each (slot `s` in digit `d`'s list with
// probability 10 / 256; at alt-k16's density 18 / 256 -- 2.2 reads a token -- and a DENSE 40 / 256 past it), the image in
// halves as sm_86 runs it, both halves read with a rendezvous between.
// see docs/lab/read_alone.md
#include <cstdio>
#include <vector>

#include "carry/carry_arm.cuh"
#include "lab.cuh"

namespace read_alone {

namespace rc = rola::carry;
namespace ops = ::denseref::ops;
using Arm = rola::Arm<0, 32, 1, 32, 1, 32, 0>;
using P = rc::Plan<Arm>;
constexpr int kWindows = 64, kSlots = 216, kLiveOf256 = 10, kK16Of256 = 18, kDenseOf256 = 40;

__host__ __device__ __forceinline__ uint32_t mix(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

//: whether slot `s` is in digit `d`'s list in window `w` of block `b` (both sides draw it: the kernel its runs, the
//: host the floors): RANDOM slots, or (`colored`) the coloring's best case -- `n` = popcount of 16 drawn bits entries,
//: entry `k` on slot `8 (27 k / n) + k mod 8`, so any 8 consecutive entries meet 8 residues
__host__ __device__ __forceinline__ uint32_t key(int b, int w, int d, int s) {
  return mix(static_cast<uint32_t>(b) * 7919u + static_cast<uint32_t>(w) * 104729u
             + static_cast<uint32_t>(d) * 31u + static_cast<uint32_t>(s) * 977u);
}

__host__ __device__ __forceinline__ int popc16(uint32_t x) {
#ifdef __CUDA_ARCH__
  return __popc(x & 0xffffu);
#else
  return __builtin_popcount(x & 0xffffu);
#endif
}

__host__ __device__ __forceinline__ bool live(int b, int w, int d, int s, bool colored, int of256) {
  if (!colored) return (key(b, w, d, s) & 0xffu) < static_cast<uint32_t>(of256);
  const int n = popc16(key(b, w, d, -1)), u = s / 8, k0 = s % 8;
  for (int k = k0; k < n; k += 8)
    if (27 * k / n == u) return true;
  return false;
}

//: MODE 0: an f16 panel, fp16 accumulate (sm_86's production form); 1 an f16 panel, fp32 accumulate (the A100's);
//: 2 an fp32 panel, fp32 accumulate; 3-5 the same on colored slots
template <int MODE>
__global__ __launch_bounds__(rc::kThreads, 1) void read(long long* t, int* out) {
  //: MODE 10 and 11 the sections, in sm_86's form on random slots; 12 and 13 sm_86's form on dense lists and at
  //: alt-k16's density
  constexpr bool kF32 = MODE < 10 && MODE % 3 == 2, kF16 = MODE >= 10 || MODE % 3 == 0,
                 kColored = MODE < 10 && MODE % 6 >= 3, kEmpty = MODE >= 6 && MODE < 10,
                 kFirst = MODE != 10, kSecond = MODE != 10 && MODE != 11;
  constexpr int kOf256 = MODE == 12 ? kDenseOf256 : MODE == 13 ? kK16Of256 : kLiveOf256;
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = static_cast<int>(threadIdx.x), warp = tid / 32, lane = tid % 32;
  rc::CarryParams p{};
  p.tables = rc::tables_of<P::kWindow>(1, P::kSmemBytes);
  p.reduce = kF32 ? rc::kReduceF32 : rc::kReduceF16;
  const int split = rc::group_split(kSlots);
  const ops::SmemAddr image = ops::smem_addr(smem + P::kAreaAt);
  const ops::SmemAddr rows = image + static_cast<uint32_t>(P::kImageRing);
  const ops::SmemAddr panel = rows + static_cast<uint32_t>(rc::read_panel<P>(kSlots)),
                      dens = rows + static_cast<uint32_t>(rc::read_dens<P>(kSlots, p.reduce)),
                      ents = rows + static_cast<uint32_t>(rc::read_ents<P>(kSlots, p.reduce));
  //: the image half, the union's rows and the den image small f16 values; each digit's exponent and den factor; every
  //: token's scale byte
  for (int i = tid; i < (P::kImageRing + kSlots * P::kReadRow) / 4; i += rc::kThreads)
    reinterpret_cast<uint32_t*>(smem + P::kAreaAt)[i] =
        0x24002400u ^ (mix(static_cast<uint32_t>(i)) & 0x03ff03ffu);
  for (int i = tid; i < rc::kMaxExtent * rc::kMaxExtent / 2; i += rc::kThreads)
    reinterpret_cast<uint32_t*>(smem + P::kDenImage)[i] =
        0x24002400u ^ (mix(~static_cast<uint32_t>(i)) & 0x03ff03ffu);
  for (int i = tid; i < rc::kWarps * rc::kMaxExtent; i += rc::kThreads)
    reinterpret_cast<int*>(smem + P::kExps)[i] = 4;
  if (tid < rc::kMaxExtent) reinterpret_cast<float*>(smem + P::kDenFactor)[tid] = 1.0f;
  for (int i = tid; i < 2 * P::kWindow; i += rc::kThreads) (smem + p.tables.scale_at)[i] = 2;
  __syncthreads();

  long long busy = 0;
  float fold = 0.0f;
#pragma unroll 1
  for (int w = 0; w < kWindows; ++w) {
    //: lane `d` digit `d`'s run: its entries in slot order, its count below the token groups' split, its place (the
    //: counts before it), as `chunk_runs` gives them; warp 0 writes every run's entries
    int n = 0, n0 = 0;
    for (int s = 0; s < kSlots; ++s)
      if (live(blockIdx.x, w, lane, s, kColored, kOf256)) {
        ++n;
        n0 += s < split;
      }
    int to = n;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, to, o);
      if (lane >= o) to += v;
    }
    //: the draw through shared memory, stored before the barrier and loaded inside the span: register arithmetic
    //: alone would sink past the barrier and the clock into the timed span
    int* const drawn =
        reinterpret_cast<int*>(smem + P::kUnionW);  // a tenant the readout never reads
    if (warp == 0) {
      drawn[lane] = n;
      drawn[32 + lane] = n0;
      drawn[64 + lane] = to - n;
    }
    if (warp == 0) {
      int k = to - n;
      for (int s = 0; s < kSlots; ++s)
        if (live(blockIdx.x, w, lane, s, kColored, kOf256)) {
          const uint32_t sl = static_cast<uint32_t>(s);
          reinterpret_cast<uint2*>(smem + (ents - ops::smem_addr(smem)))[k++] =
              make_uint2((sl * (P::kReadRow / 16) + rc::pool_chunk(P::kReadRow, sl, 0u)) | sl << 16,
                         0x3800u ^ (mix(sl ^ static_cast<uint32_t>(lane)) & 0x3ffu));
        }
    }
    //: the panel and the den words cleared, as the chunk's fill clears them
    for (int i = tid; i < (rc::read_ents<P>(kSlots, p.reduce) - rc::read_panel<P>(kSlots)) / 4;
         i += rc::kThreads)
      reinterpret_cast<uint32_t*>(smem + (panel - ops::smem_addr(smem)))[i] = 0u;
    //: two barriers before the clock: after one, the empty span read 19.4K cycles a window (the warps not lined up
    //: by it, measured); after two, 38
    __syncthreads();
    __syncthreads();
    const long long t0 = clock64();
    const rc::Run run{0, drawn[lane], drawn[32 + lane], drawn[64 + lane]};
    if constexpr (!kEmpty) rc::read_factors<P>(p, smem, run, w, 0, ents, warp, lane);
    ops::rendezvous();
    if constexpr (!kEmpty && kFirst)
      rc::read_chunk<P, kF32, kF16>(smem, rc::read_run<P>(run, warp, lane), 0, 16, image, 15u, rows,
                                    panel, dens, ents, warp, lane);
    ops::rendezvous();
    if constexpr (!kEmpty && kSecond)
      rc::read_chunk<P, kF32, kF16>(smem, rc::read_run<P>(run, warp, lane), 16, 32, image, 15u,
                                    rows, panel, dens, ents, warp, lane);
    const long long t1 = clock64();
    busy += t1 - t0;
    //: the sink read between barriers: every warp's den words written before it, none cleared until after it
    __syncthreads();
    fold += reinterpret_cast<const float*>(smem + (dens - ops::smem_addr(smem)))[tid];
    __syncthreads();
  }
  if (lane == 0) t[blockIdx.x * rc::kWarps + warp] = busy;
  lab::sink(out, lab::bits(fold));
}

//: THE FLOORS a window, from the same draw: TENSOR, a scheduler's two warps' MMAs (4 a tile: the num's 2 and the den's
//: 2, `mma` cycles each); SHARED, the CTA's wavefront-cycles (a 128-byte load one, a store two): a digit a warp its two
//: slices (8), its exponent and den factor (2); a tile a warp its entries (2), its rows (8), and a token pair's panel
//: and den words (a load and a store each, the f16 panel's scale byte a load: 14 f16, 18 fp32)
struct Floor {
  double tensor = 0, shared = 0;
};

Floor floor_of(int blocks, double mma, double tile_shared, bool colored, int of256) {
  Floor f;
  const int split = rc::group_split(kSlots);
  for (int b = 0; b < blocks; ++b)
    for (int w = 0; w < kWindows; ++w) {
      double tiles = 0, digits = 0;
      for (int d = 0; d < P::kDigits; ++d) {
        int n0 = 0, n1 = 0;
        for (int s = 0; s < kSlots; ++s)
          if (live(b, w, d, s, colored, of256)) (s < split ? n0 : n1)++;
        tiles += (n0 + 15) / 16 + (n1 + 15) / 16;
        digits += 2;
      }
      f.tensor += tiles * 4 * mma;
      f.shared += 4 * (digits * 10 + tiles * tile_shared);
    }
  f.tensor /= double(blocks) * kWindows;
  f.shared /= double(blocks) * kWindows;
  return f;
}

template <int MODE>
void variant(const lab::Bench& b, const char* label) {
  const std::string name = lab::kernel("read", {MODE});
  if constexpr (MODE < 6 || MODE >= 11)
    b.expect(name, MODE % 3 == 0 || MODE >= 11 ? "HMMA.16816.F16" : "HMMA.16816.F32", ">=", 4);
  b.exempt(name,
           "an isolation of the production readout; its budget is tools/budgets/carry.json's");
  if (b.expect_only) return;
  LAB_CHECK(
      cudaFuncSetAttribute(read<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, P::kSmemBytes));
  const lab::Samples s = b.variant(
      [&] { read<MODE><<<b.sms, rc::kThreads, P::kSmemBytes>>>(b.times, b.sinks); },
      [&] { return lab::Bench::mean(b.cycles(rc::kWarps), rc::kWarps, 0, rc::kWarps) / kWindows; });
  b.unchecked(label,
              "a timing isolation; the readout's arithmetic is the oracle's (test_carry_readout)");
  b.record(label, "cycles a warp-window", s);
  //: the floors are the whole readout's: the controls and the sections have none
  if constexpr (MODE >= 6 && MODE < 12) return;
  //: sm_86's m16n8k16 at fp32 accumulate is half rate (32 cycles a scheduler), at fp16 16; sm_80's 8 either way
  cudaDeviceProp prop{};
  LAB_CHECK(cudaGetDeviceProperties(&prop, 0));
  const bool geforce = prop.major == 8 && prop.minor == 6;
  //: the dtypes as `read`'s: MODE 12 and 13 sm_86's form
  constexpr bool kF16 = MODE >= 10 || MODE % 3 == 0, kF32 = MODE < 10 && MODE % 3 == 2;
  const Floor f = floor_of(b.sms, geforce ? (kF16 ? 16.0 : 32.0) : 8.0, kF32 ? 26.0 : 22.0,
                           MODE >= 3 && MODE < 6,
                           MODE == 12   ? kDenseOf256
                           : MODE == 13 ? kK16Of256
                                        : kLiveOf256);
  for (const auto& [what, v] :
       {std::pair<const char*, double>{"tensor floor", f.tensor}, {"shared floor", f.shared}}) {
    const std::string nm = std::string(label) + ": " + what;
    b.unchecked(nm, "computed on the host from the same draw");
    std::printf(
        "{\"lab\": \"result\", \"name\": \"%s\", \"unit\": \"cycles a warp-window\", \"samples\": [%.1f]}\n",
        nm.c_str(), v);
  }
}

}  // namespace read_alone

int main(int argc, char** argv) {
  lab::Bench b(argc, argv);
  read_alone::variant<0>(b, "f16 panel, fp16 accumulate (sm_86's form)");
  read_alone::variant<1>(b, "f16 panel, fp32 accumulate (the A100's form)");
  read_alone::variant<2>(b, "fp32 panel, fp32 accumulate");
  read_alone::variant<3>(b, "colored slots: f16 panel, fp16 accumulate");
  read_alone::variant<4>(b, "colored slots: f16 panel, fp32 accumulate");
  read_alone::variant<5>(b, "colored slots: fp32 panel, fp32 accumulate");
  read_alone::variant<6>(b, "control: the timed span without the readout");
  read_alone::variant<9>(b, "control: the timed span without the readout, colored slots");
  read_alone::variant<10>(b, "section: the factors and their rendezvous alone (sm_86's form)");
  read_alone::variant<11>(b, "section: the factors and the first half (sm_86's form)");
  read_alone::variant<13>(b, "alt-k16's density (18 / 256): f16 panel, fp16 accumulate");
  read_alone::variant<12>(b, "dense lists (40 / 256): f16 panel, fp16 accumulate");
  return 0;
}
