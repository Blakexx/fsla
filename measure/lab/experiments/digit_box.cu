// measure/lab/experiments/digit_box.cu -- QUESTION: on sm_86, does a box of 32 lists x 16 digits x 32 channels (its
// image 32 KB, so the whole image and the two-barrier hand-off fit) cost a token less than the 32-digit box's? Its
// readout keeps 32-channel tiles and gathers per entry, its read union shrinks; its fold loads each write token's
// values in both digit boxes. METHOD: `read_image.cu`'s tokens-on-N readout on the read lists re-ranked per 16-digit
// half (`digit_box.inputs.py`: each half's union colored and placed), 2 digits a warp; a 32 KB image of 16 digit planes
// ([list][32 channels], 16-byte units swizzled `(4 list + (c ^ s(list))) ^ (digit & 7)`, so the fold's stores and the
// readout's `ldmatrix.trans` are both conflict-free, KERNEL_STANDARDS 30); a fold stand-in with the fold's shared
// traffic on the box's 32 lists (the 32-digit box's write lists, so both digit boxes fold the same tokens): a warp's 4
// lists, each step 8 tokens -- an entry, their write rows over the box's digits by `ldmatrix.trans`, their 32 channels'
// values, the coefficients, m16n8k8 MMAs into the state. The hand-off: the readout, a named barrier, the fold storing
// each list's rows into the image as its last step lands, a named barrier; the values zero there, so the state and the
// check stay exact. `digit_box32.cu` builds this file for the 32-digit box's reference (a static image only: its whole
// image does not fit beside the rest). Modes: 0 a static image, free-running; 1 the hand-off; 2 the readout alone; 3
// the fold alone. see docs/internals/tools/lab.md#digit-box
#ifndef READ_IMAGE_DIGITS
#define READ_IMAGE_DIGITS 16
#endif
#include "read_image.cuh"

namespace digit_box {

using namespace read_image;

//: the fold's values (a slot's 32 channels, 64 bytes, swizzled as the pool's rows): zero in their own region in the
//: 16-digit box; the 32-digit box's static reference reads the pool's rows (finite, its state never published), so its
//: layout fits sm_86. The fold's heads (a warp's 4 lists' first entries, 512 bytes, double-buffered) after them
constexpr int kValues = kDigits == 16 ? kBytes : kPool;
constexpr int kFoldHeadBytes = 4 * kHead * 8;
constexpr int kFoldHeads = kDigits == 16 ? kValues + kSlots * 64 : kBytes;
constexpr int kAll = kFoldHeads + 2 * kWarps * kFoldHeadBytes;
constexpr int kReadDigits = kDigits / kWarps, kMTiles = kDigits / 16;

//: a 16-byte unit of digit `d`'s plane for list `l`, chunk `c`
__host__ __device__ constexpr int unit_of(int l, int c, int d) {
  return ((4 * l + (c ^ ((l >> 1) & 3))) ^ (d & 7)) & 127;
}

__device__ __forceinline__ uint32_t plane_of(uint32_t sm, int d) {
  return sm + kImage + static_cast<uint32_t>(d % kPlanes) * kPlaneBytes;
}

__device__ __forceinline__ void ldsm_t2(uint32_t (&r)[2], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}

__device__ __forceinline__ void cp_async8(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(dst), "l"(src));
}

__device__ __forceinline__ void mma_k8(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t b) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(b));
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
  return d;
}

//: A DIGIT'S STATE FRAGMENTS (32 channels x 32 lists, the A operand, 16 registers) from its plane: `u` the lane's 4
//: units' bytes (m-tile, kb) at digit 0, XORed with the digit's
__device__ __forceinline__ void state_in(uint32_t (&f)[16], uint32_t sm, const uint32_t (&u)[4], int d) {
  const uint32_t plane = plane_of(sm, d), x = 16u * (d & 7);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    uint32_t a[4];
    lab::ldsm_t(a, plane + (u[k] ^ x));
#pragma unroll
    for (int i = 0; i < 4; ++i) f[k * 4 + i] = a[i];
  }
}

//: THE FOLD STAND-IN'S STEP, 8 tokens of a list: their write rows over the box's digits (A, `ldmatrix.trans`) times
//: their coefficients, their 32 channels' values (B), m16n8k8 MMAs into the list's state
__device__ __forceinline__ void fold_step(uint32_t sm, uint2 e, const Lane& c, uint32_t half,
                                          float (&acc)[kMTiles][4][4]) {
  const uint32_t ea = __shfl_sync(0xffffffffu, e.x, c.src_a) & 0xffffu;
  const uint32_t e0 = __shfl_sync(0xffffffffu, e.y, c.src_g), e1 = __shfl_sync(0xffffffffu, e.y, c.src_h);
  uint32_t a[2 * kMTiles];
  if constexpr (kMTiles == 1) {
    ldsm_t2(*reinterpret_cast<uint32_t(*)[2]>(a), (sm + ea) ^ half);
  } else {
    lab::ldsm_t(*reinterpret_cast<uint32_t(*)[4]>(a), (sm + ea) ^ c.chunk);
  }
  const uint32_t cp = prmt(e0, e1, 0x5410u);
#pragma unroll
  for (int k = 0; k < 2 * kMTiles; ++k) a[k] = lab::hmul2(a[k], cp);
  const uint32_t s = (ea - kPool) >> 6;
  uint32_t b[4];
  lab::ldsm_t(b, sm + kValues + 64u * s + (c.chunk ^ (16u * ((s >> 1) & 3u))));
#pragma unroll
  for (int m = 0; m < kMTiles; ++m)
#pragma unroll
    for (int n = 0; n < 4; ++n) mma_k8(acc[m][n], a[2 * m], a[2 * m + 1], b[n]);
}

template <int MODE>
__global__ __launch_bounds__(kThreads, 1) void box(Lists rl, Lists fl, const uint32_t* zeros, long long* times,
                                                    int* sinks) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr bool kRead = MODE != 3, kFold = MODE != 2, kHand = MODE == 1;
  static_assert(!kHand || kDigits == 16, "the hand-off's whole image fits the 16-digit box only");
  const int warp = lab::warp_id(), lane = lab::lane_id();
  const int boxes = rl.width / kDigits, bx = static_cast<int>(blockIdx.x) % boxes, fbox = bx * kDigits / 32;
  const uint32_t sm = lab::smem(smem);
  uint4* chunks = reinterpret_cast<uint4*>(smem);
  for (int k = static_cast<int>(threadIdx.x); k < kAll / 16; k += kThreads) {
    const uint32_t v = k < kPool / 16 ? 0x3d80bd80u : k < kPads / 16 ? 0x3d803d80u : 0u;
    chunks[k] = make_uint4(v, v, v, v);
  }
  __syncthreads();
  const Lane c = lane_of<1>(sm, warp, lane);
  const uint32_t half = lab::held(16u * ((lane >> 3) & 1));
  //: the lane's state-load units (m-tile, kb) at digit 0
  uint32_t u[4];
#pragma unroll
  for (int k = 0; k < 4; ++k)
    u[k] = lab::held(16u * unit_of(8 * (lane >> 4) + (lane & 7) + 16 * (k & 1), 2 * (k >> 1) + ((lane >> 3) & 1), 0));
  //: THE STATE, a warp's lists w + 8 J by their digits' m-tiles and the channels' n-tiles: -0.0625 even, +0.0625 odd
  const uint32_t zero = zeros[0];
  float acc[4][kMTiles][4][4];
#pragma unroll
  for (int k = 0; k < 64 * kMTiles; ++k)
    (&acc[0][0][0][0])[k] = __uint_as_float(zero | (k & 1 ? 0x3d800000u : 0xbd800000u));
  //: a lane's store address for list J (n-tile 0, its row g of digit block 0): its g's plane and unit, its t's word
  const int g = lane >> 2, t = lane & 3;
  uint32_t put[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) put[j] = lab::held(plane_of(sm, g) + 16u * unit_of(warp + 8 * j, 0, g) + 4u * t);
  float sink = 0.0f;
  //: both heads a window ahead: the readout's (8 bytes a lane), the fold's (16)
  const uint8_t* rsrc = rl.heads + (static_cast<int64_t>(bx) * kWarps + warp) * kHeadBytes + kHeadBytes / 32 * lane;
  const uint8_t* fsrc = fl.heads + (static_cast<int64_t>(fbox) * kWarps + warp) * kFoldHeadBytes + 16 * lane;
  const int64_t rstep = static_cast<int64_t>(boxes) * kWarps * kHeadBytes;
  const int64_t fstep = static_cast<int64_t>(fl.width / 32) * kWarps * kFoldHeadBytes;
  const uint32_t rto = lab::held(sm + kHeads + warp * kHeadBytes + (kHeadBytes / 32) * lane);
  const uint32_t fto = lab::held(sm + kFoldHeads + warp * kFoldHeadBytes + 16 * lane);
  const uint32_t rhe = lab::held(sm + kHeads + warp * kHeadBytes + c.entry);
  const uint32_t fhe = lab::held(sm + kFoldHeads + warp * kFoldHeadBytes + c.entry);
  const auto heads_in = [&](uint32_t rflip, uint32_t fflip) {
    if constexpr (kHeadBytes == 256) {
      cp_async8(rto + rflip, rsrc);
    } else {
      lab::cp_async16(rto + rflip, rsrc);
    }
    lab::cp_async16(fto + fflip, fsrc);
  };
  heads_in(0, 0);
  lab::cp_commit();
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int w = 0; w < rl.windows; ++w) {
    const uint32_t rflip = (w & 1) * kWarps * kHeadBytes, fflip = (w & 1) * kWarps * kFoldHeadBytes;
    __syncwarp();
    rsrc += rstep;
    fsrc += fstep;
    heads_in(kWarps * kHeadBytes - rflip, kWarps * kFoldHeadBytes - fflip);
    lab::cp_commit();
    asm volatile("cp.async.wait_group 1;" ::: "memory");
    __syncwarp();
    //: an entry past the staged ones, from its list in global memory (zero past its count): 0.3% of lists at alt-k4
    const auto listed = [&](const Lists& ls, int b, int digits, int d, int e) {
      return ls.entries[(static_cast<int64_t>(w) * ls.width + b * digits + d) * kListCap + e];
    };
    //: THE READOUT, the warp's digits w + 8 i
    if constexpr (kRead) {
#pragma unroll
      for (int i = 0; i < kReadDigits; ++i) {
        const uint2 e = lds64(rhe + rflip + kHead * 8 * i);
        const int steps = static_cast<int>((e.x >> 16) & 0xffu), d = warp + 8 * i;
        uint32_t f[16];
        state_in(f, sm, u, d);
        tile_n<1>(sm, e, c, f, sink);
        for (int k = __any_sync(0xffffffffu, steps > 1) ? 1 : 0; k > 0; --k)
          tile_n<1>(sm, lds64(rhe + rflip + kHead * 8 * i + 64), c, f, sink);
        for (int k = __any_sync(0xffffffffu, steps > kHead / 8) ? 1 : 0, s = kHead / 8; k > 0; --k)
          do tile_n<1>(sm, listed(rl, bx, kDigits, d, 8 * s + (lane & 7)), c, f, sink);
          while (__any_sync(0xffffffffu, ++s < steps));
      }
    }
    if constexpr (kHand) lab::bar_sync(1, kThreads);
    //: THE FOLD, the warp's 4 lists, each list's rows stored into the image as its last step lands
    if constexpr (kFold) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint2 e = lds64(fhe + fflip + kHead * 8 * j);
        const int steps = static_cast<int>((e.x >> 16) & 0xffu);
        fold_step(sm, e, c, half, acc[j]);
        for (int k = __any_sync(0xffffffffu, steps > 1) ? 1 : 0; k > 0; --k)
          fold_step(sm, lds64(fhe + fflip + kHead * 8 * j + 64), c, half, acc[j]);
        for (int k = __any_sync(0xffffffffu, steps > kHead / 8) ? 1 : 0, s = kHead / 8; k > 0; --k)
          do fold_step(sm, listed(fl, fbox, 32, warp + 8 * j, 8 * s + (lane & 7)), c, half, acc[j]);
          while (__any_sync(0xffffffffu, ++s < steps));
        if constexpr (kHand) {
#pragma unroll
          for (int n = 0; n < 4; ++n)
#pragma unroll
            for (int hi = 0; hi < 2; ++hi)
              asm volatile("st.shared.u32 [%0], %1;" ::"r"((put[j] ^ (16u * n)) + 8u * kPlaneBytes * hi),
                           "r"(pack_bf16(acc[j][0][n][2 * hi], acc[j][0][n][2 * hi + 1]))
                           : "memory");
        }
      }
    }
    if constexpr (kHand) lab::bar_sync(1, kThreads);
  }
  clock.stop(times);
  lab::cp_wait_all();
#pragma unroll
  for (int k = 0; k < 64 * kMTiles; ++k) sink += (&acc[0][0][0][0])[k];
  lab::sink(sinks, lab::bits(sink));
  __syncthreads();
  constexpr int kRowChunks = kWords / 4;
  for (int k = static_cast<int>(threadIdx.x); k < kSlots * kRowChunks; k += kThreads)
    reinterpret_cast<uint4*>(rl.panel_out)[static_cast<int64_t>(blockIdx.x) * kSlots * kRowChunks + k] =
        chunks[(kPanel + (k / kRowChunks) * kPanelPitch * 4) / 16 + k % kRowChunks];
}

//: THE BUDGET (KERNEL_STANDARDS sections 29-30) of the hand-off (mode 1), per MMA at alt-k4: a warp-window's ~2.8
//: readout tiles (4 MMAs each) and ~5.6 fold steps (4 m16n8k8 each), 33.8 MMAs. A tile: an entry and a read-row
//: load, 3 shuffles, 8 FFMAs, ~11 integer (a mask-XOR, 2 shifts, 2 bases, 4 packs, ~1.3 fillers), 4 adds. A step: an
//: entry, the write rows' (x2) and values' (x4) loads, 3 shuffles, a PRMT, 2 HMUL2s, ~7 integer (the rows' XOR, the
//: values' address). A digit: 4 state loads, its plane (~3), its tag, 2 votes and their branches (~4 select, ~3
//: control); a list the same, and its 8 packs and 8 stores. A window: 2 heads' copies and their 6 never-executed
//: `LDS`, ~20 integer, 2 barriers, ~12 control. Wavefronts: a tile 10 (entry 2, rows 4, adds 4), a step 8 (entry 2,
//: rows 2, values 4), a digit's state 16, a list's stores 8, the heads' copies 12, ~0.6 a shuffle: 165 a
//: warp-window. The state's 64 FADDs into the sink after the loop count too (0.06 an MMA over the 32 windows).
struct Budget {
  double shared_load, shuffle, float_, pack, integer, select, control, shared_store, atomic, global, wavefronts;
};

constexpr double kMmas = 33.8;
constexpr Budget kBudget = {37 / kMmas, 26 / kMmas, 36 / kMmas, 38 / kMmas, 125 / kMmas, 30 / kMmas,
                            32 / kMmas, 32 / kMmas, 11.5 / kMmas, 2.5 / kMmas, 170 / kMmas};

template <int MODE>
void variant(const lab::Bench& b, const Lists& rl, const Lists& fl, const uint32_t* zeros, const Reference& ref) {
  const std::string name = lab::kernel("digit_box::box", {MODE});
  b.expect(name, "HMMA", ">=", 4);
  b.expect(name, "STEP_GLOBAL", ">=", 0);
  if (MODE == 1) {
    b.budget(name, "mma", 1.0);
    b.budget(name, "shared load", kBudget.shared_load);
    b.budget(name, "shared store", kBudget.shared_store);
    b.budget(name, "shared atomic", kBudget.atomic);
    b.budget(name, "shuffle", kBudget.shuffle);
    b.budget(name, "pack", kBudget.pack);
    b.budget(name, "conversion unit", 0.0);
    b.budget(name, "float", kBudget.float_);
    b.budget(name, "select/compare", kBudget.select);
    b.budget(name, "integer/address", kBudget.integer);
    b.budget(name, "control", kBudget.control);
    b.budget(name, "global", kBudget.global);
    b.budget(name, "other", 0.0);
    b.budget(name, "wavefronts", kBudget.wavefronts);
  } else {
    b.exempt(name, MODE == 0   ? "a baseline: the readout and fold free-running on a static image"
                   : MODE == 2 ? "an isolation: the readout alone"
                               : "an isolation: the fold alone");
  }
  if (b.expect_only) return;
  LAB_CHECK(cudaFuncSetAttribute(box<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, kAll));
  const auto launch = [&] { box<MODE><<<b.sms, kThreads, kAll>>>(rl, fl, zeros, b.times, b.sinks); };
  const lab::Samples s = b.variant(launch, [&] { return b.span(kWarps) / rl.windows; });
  const char* labels[4] = {"a static image, free-running", "the hand-off: readout, barrier, fold storing, barrier",
                           "the readout alone, free-running", "the fold alone"};
  const std::string label = std::string(labels[MODE]) + (kDigits == 16 ? " / 16-digit box" : " / 32-digit box");
  b.record(label, "cycles a CTA-window", s);
  if constexpr (MODE == 3) {
    b.unchecked(label, "an isolation: the fold's state is not read out");
  } else {
    LAB_CHECK(cudaMemset(rl.panel_out, 0, static_cast<size_t>(b.sms) * kSlots * kWords * 4));
    launch();
    LAB_CHECK(cudaDeviceSynchronize());
    const Verdict v = verify<1, 1, 0>(ref, rl);
    b.check(label, v.bad == 0 && v.words > 0, v.detail);
  }
}

}  // namespace digit_box

int main(int argc, char** argv) {
  using namespace digit_box;
  lab::Bench b(argc, argv);
  std::string dir;
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], "--inputs")) dir = argv[i + 1];
  Lists rl{}, fl{}, unused{};
  HostLists host;
  uint32_t* zeros = nullptr;
  if (!b.expect_only) {
    if (dir.empty()) {
      std::fprintf(stderr, "digit_box: --inputs <directory> (tools/lab.py writes it)\n");
      return 1;
    }
    load(dir, b.sms, host, unused, rl);
    const HostLists fh = read_lists(dir + "/fold/read.bin");
    const size_t entries = static_cast<size_t>(fh.windows) * fh.width * kListCap * 8;
    const size_t heads = static_cast<size_t>(fh.windows) * (fh.width / 32) * kWarps * kFoldHeadBytes;
    fl = rl;
    fl.width = fh.width;
    fl.entries = upload<uint2>(dir + "/fold/entries_n.bin", entries);
    fl.heads = upload<uint8_t>(dir + "/fold/heads_n.bin", heads, heads / fh.windows);
    LAB_CHECK(cudaMalloc(&zeros, 64));
    LAB_CHECK(cudaMemset(zeros, 0, 64));
    std::printf("{\"lab\": \"note\", \"free_floor_n\": %.1f, \"bytes\": %d}\n", floor_of(host, 1, false, true), kAll);
  }
  const Reference ref{&host, b.sms};
  variant<0>(b, rl, fl, zeros, ref);
#if READ_IMAGE_DIGITS == 16
  variant<1>(b, rl, fl, zeros, ref);
#endif
  variant<2>(b, rl, fl, zeros, ref);
  variant<3>(b, rl, fl, zeros, ref);
}
