// measure/lab/experiments/read_image.cu -- QUESTION: the carry's readout with the state from a bf16 image in shared
// memory: tokens on the MMA's M (16-token steps, the state the B operand) or on its N (8-token tiles, the state the A
// operand), what does each cost against its tensor floor at nl64k-alt-k4's read lists and at a dense cell, and what
// do the panel's adds and the pass's drain add? METHOD: 8 warps, one CTA an SM a box of 32 lists x 32 read digits x
// 32 channels; a warp reads digits w, w + 8, w + 16, w + 24, holding a digit's state fragments (16 registers, from the
// image by `ldmatrix.trans`) while its tokens stream; read rows (32 lists of bf16, 64 bytes, swizzled) gathered from a
// 256-slot pool by slot; bf16 MMAs with fp32 accumulate; outputs times each token's coefficient to s16 pairs added by
// `red.shared` into the token's panel row; the drain each pass's end (`red.global.add` two rows a request, each word
// cleared). The image holds 16 planes (digit d on plane d % 16: a plane's access pattern is all that matters). Modes:
// form 0 tokens on M, 1 tokens on N; out 0 the MMAs alone (outputs summed into a sink), 1 + panel adds, 2 + the drain;
// outputs converted by F2I or on the FMA pipe (one FFMA a value, the coefficient and unit folded in, a PRMT a pair);
// the state a digit at a time, or a ring of two digits (the next one's fragments loading under this one's steps); a
// warp's heads (its digits' pass starts and first 16 slots) staged a window ahead by `cp.async`, as the fill would.
// see docs/internals/tools/lab.md#read-image
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "lab.cuh"

namespace read_image {

constexpr int kWarps = 8, kThreads = 32 * kWarps;
constexpr int kLists = 32, kDigits = 32, kChannels = 32;
constexpr int kListCap = 512, kListAt = 8, kSlots = 256, kWords = 16, kPanelPitch = 20;
constexpr int kPlanes = 16, kPlaneBytes = kLists * kChannels * 2, kRowBytes = kLists * 2;
constexpr int kImage = 0, kPool = kImage + kPlanes * kPlaneBytes, kPanel = kPool + kSlots * kRowBytes;
//: the pad tokens' adds land on a lane's own word, consecutive across lanes (no bank conflict at a common offset), in a
//: region that holds a step's largest offset past the base (60 bytes) for the last lane
constexpr int kDummy = kPanel + kSlots * kPanelPitch * 4, kHeads = kDummy + 256;
//: a warp's heads for a window, double-buffered: its 4 digits' pass starts (16 bytes each), then their first 16 slots
constexpr int kHeadBytes = 4 * 16 + 4 * 64, kBytes = kHeads + 2 * kWarps * kHeadBytes;
constexpr float kUnit = 256.0f;

struct Lists {
  const uint16_t* slots;
  const uint32_t* offs;
  const uint16_t* at;
  const uint16_t* count;
  int windows, width, groups;
  uint32_t* out;
  uint32_t* panel_out;
};

//: a 16-byte chunk `c` of a 64-byte row `row` (an image plane's list or a pool slot), swizzled by the row's bits 1-2
__device__ __forceinline__ uint32_t chunk(uint32_t base, int row, int c) {
  return base + row * kRowBytes + 16 * (c ^ ((row >> 1) & 3));
}

//: entry `e` of a digit's pass as (slot | c << 16), c a bf16 (1 here: the coefficient comes with the token's row and
//: moves no cycles), zero past its count: the first 16 of pass 0 from the warp's heads in shared memory (`head`), the
//: rest from global memory; a dense cell's every token
template <int DENSE>
__device__ __forceinline__ uint32_t entry(const Lists& ls, const uint16_t* head, int64_t row, int begin, int n,
                                          int e) {
  if constexpr (DENSE) {
    return e < n ? static_cast<uint32_t>(e) | (0x3f80u << 16) : 0u;
  } else {
    const uint32_t slot = head ? head[e & 15] : ls.slots[row * kListCap + begin + min(e, max(n - 1, 0))];
    return e < n ? slot | (0x3f80u << 16) : 0u;
  }
}

__device__ __forceinline__ float coef(uint32_t packed) { return __uint_as_float(packed & 0xffff0000u) * kUnit; }

//: CONV 1: two outputs times (coefficient x unit) to an s16 pair on the FMA pipe -- one FFMA each adds 1.5 x 2^23, so
//: the rounded integer lands in the low 16 bits (|x| < 2^15), and one PRMT packs the pair
__device__ __forceinline__ uint32_t s16_fma(float lo, float hi, float cu) {
  const float magic = 12582912.0f;
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(d) : "r"(__float_as_uint(fmaf(lo, cu, magic))),
      "r"(__float_as_uint(fmaf(hi, cu, magic))));
  return d;
}

__device__ __forceinline__ uint32_t s16_pair(float lo, float hi) {
  int a, b;
  asm("cvt.rni.sat.s16.f32 %0, %1;" : "=r"(a) : "f"(lo));
  asm("cvt.rni.sat.s16.f32 %0, %1;" : "=r"(b) : "f"(hi));
  return (static_cast<uint32_t>(a) & 0xffffu) | (static_cast<uint32_t>(b) << 16);
}

//: a token's add into its panel row, branch-free: a pad (entry 0) adds nothing to the lane's own dummy word
__device__ __forceinline__ void panel_add(uint32_t sm, uint32_t packed, int word, uint32_t v, int lane) {
  const uint32_t at = packed != 0u ? sm + kPanel + ((packed & 0xffu) * kPanelPitch + word) * 4 : sm + kDummy + lane * 4;
  asm volatile("red.shared.add.u32 [%0], %1;" ::"r"(at), "r"(v) : "memory");
}

//: A DIGIT'S STATE FRAGMENTS from its image plane, 16 registers: form 0's B (K = lists, N = channels) at
//: `(kb * 4 + nt) * 2 + h`, form 1's A (M = channels, K = lists) at `(m * 2 + kb) * 4 + i`
template <int FORM>
__device__ __forceinline__ void state_in(uint32_t (&f)[16], uint32_t plane, int lane) {
  const int r = lane & 7, q = lane >> 3;
#pragma unroll
  for (int u = 0; u < 2; ++u)
#pragma unroll
    for (int v = 0; v < 2; ++v) {
      uint32_t x[4];
      if constexpr (FORM == 0) {
        lab::ldsm_t(x, chunk(plane, 16 * u + 8 * (q & 1) + r, 2 * v + (q >> 1)));
        f[(u * 4 + 2 * v) * 2] = x[0];
        f[(u * 4 + 2 * v) * 2 + 1] = x[1];
        f[(u * 4 + 2 * v + 1) * 2] = x[2];
        f[(u * 4 + 2 * v + 1) * 2 + 1] = x[3];
      } else {
        lab::ldsm_t(x, chunk(plane, 16 * v + 8 * (q >> 1) + r, 2 * u + (q & 1)));
#pragma unroll
        for (int i = 0; i < 4; ++i) f[(u * 2 + v) * 4 + i] = x[i];
      }
    }
}

//: FORM 0's HALF STEP, a digit's last 8 tokens or fewer: their 8 read rows by one `ldmatrix.x4` (tokens 8-15 zero
//: in A), the same 8 MMAs (M is 16 tokens), outputs and adds for tokens 0-7 only -- no padding row loaded or added
template <int OUT, int DENSE, int CONV>
__device__ __forceinline__ void half_m(uint32_t sm, const Lists& ls, const uint16_t* head, int64_t row, int begin,
                                       int n, const uint32_t (&f)[16], int lane, float& sink, int s) {
  const int r = lane & 7, q = lane >> 3, g = lane >> 2, t = lane & 3;
  const uint32_t mine = entry<DENSE>(ls, s < 1 ? head : nullptr, row, begin, n, 16 * s + (lane & 7));
  const uint32_t ea = __shfl_sync(0xffffffffu, mine, r), eg = __shfl_sync(0xffffffffu, mine, g);
  uint32_t x[4];
  lab::ldsm(x, chunk(sm + kPool, ea & 0xffu, q));
  float acc[4][4] = {};
#pragma unroll
  for (int kb = 0; kb < 2; ++kb) {
    const uint32_t a[4] = {x[2 * kb], 0u, x[2 * kb + 1], 0u};
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) lab::mma_bf16_f32(acc[nt], a, f[(kb * 4 + nt) * 2], f[(kb * 4 + nt) * 2 + 1]);
  }
#pragma unroll
  for (int nt = 0; nt < 4; ++nt) {
    if constexpr (OUT == 0) {
      sink += acc[nt][0] + acc[nt][1];
    } else {
      const float cg = coef(eg);
      panel_add(sm, eg, 4 * nt + t,
                CONV ? s16_fma(acc[nt][0], acc[nt][1], cg) : s16_pair(acc[nt][0] * cg, acc[nt][1] * cg), lane);
    }
  }
}

//: a token's panel row for lane word `t`, or the lane's own dummy word for a pad: once a token a step (CONV 2), each
//: add then a constant offset from it
__device__ __forceinline__ uint32_t panel_base(uint32_t sm, uint32_t packed, int t, int lane) {
  return packed != 0u ? sm + kPanel + ((packed & 0xffu) * kPanelPitch + t) * 4 : sm + kDummy + lane * 4;
}

__device__ __forceinline__ void panel_red(uint32_t at, uint32_t v) {
  asm volatile("red.shared.add.u32 [%0], %1;" ::"r"(at), "r"(v) : "memory");
}

//: CONV 3, THE FACTS PASS'S FORM: an entry is its pool row's byte offset (swizzle folded in) low and its panel row's
//: byte offset plus one high (zero: none), zero past the count. HEAD 1, a pass's first step: from the warp's heads at
//: shared address `head` (a 32-bit address, no generic pointer to test); HEAD 0, a later step: global memory
template <int HEAD>
__device__ __forceinline__ uint32_t entry3(const Lists& ls, uint32_t head, int64_t row, int begin, int n, int e) {
  uint32_t v;
  if constexpr (HEAD == 1) {
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(head + 4 * (e & 15)));
  } else {
    v = ls.offs[row * kListCap + begin + min(e, max(n - 1, 0))];
  }
  return e < n ? v : 0u;
}

//: CONV 3's step: a read row's address one add and one XOR with the lane's chunk; a token's panel base once
template <int OUT, int HEAD>
__device__ __forceinline__ void step_m3(uint32_t sm, const Lists& ls, uint32_t head, int64_t row, int begin,
                                        int n, const uint32_t (&f)[16], int lane, float& sink, int s) {
  const int r = lane & 7, q = lane >> 3, g = lane >> 2, t = lane & 3;
  const uint32_t mine = entry3<HEAD>(ls, head, row, begin, n, 16 * s + (lane & 15));
  const uint32_t ea = __shfl_sync(0xffffffffu, mine, 8 * (q & 1) + r);
  const uint32_t eg = __shfl_sync(0xffffffffu, mine, g), eg8 = __shfl_sync(0xffffffffu, mine, g + 8);
  const uint32_t at = sm + kPool + (ea & 0xffffu);
  float acc[4][4] = {};
#pragma unroll
  for (int kb = 0; kb < 2; ++kb) {
    uint32_t a[4];
    lab::ldsm(a, at ^ (32u * kb + 16u * static_cast<uint32_t>(q >> 1)));
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) lab::mma_bf16_f32(acc[nt], a, f[(kb * 4 + nt) * 2], f[(kb * 4 + nt) * 2 + 1]);
  }
  const float cg = eg ? kUnit : 0.0f, cg8 = eg8 ? kUnit : 0.0f;
  const uint32_t bg = (eg >> 16) ? sm + kPanel - 1 + (eg >> 16) + 4 * t : sm + kDummy + 4 * lane;
  const uint32_t bg8 = (eg8 >> 16) ? sm + kPanel - 1 + (eg8 >> 16) + 4 * t : sm + kDummy + 4 * lane;
#pragma unroll
  for (int nt = 0; nt < 4; ++nt) {
    if constexpr (OUT == 0) {
      sink += acc[nt][0] + acc[nt][1] + acc[nt][2] + acc[nt][3];
    } else {
      panel_red(bg + 16 * nt, s16_fma(acc[nt][0], acc[nt][1], cg));
      panel_red(bg8 + 16 * nt, s16_fma(acc[nt][2], acc[nt][3], cg8));
    }
  }
}

//: THE EXACT PAIR: two outputs times (coefficient x unit) to the integer hi x 2^16 + lo, so u32 adds sum both channels
//: exactly while each final fits s16 -- lo's FFMA adds 1.5 x 2^23 (its bits 0x4B400000 + lo), hi's adds 0x4B40 less
//: (its low half hi - 0x4B40), and one LEA (hi bits << 16, plus lo bits) cancels lo's exponent bits
__device__ __forceinline__ uint32_t s16_exact(float lo, float hi, float cu) {
  const uint32_t l = __float_as_uint(fmaf(lo, cu, 12582912.0f));
  const uint32_t h = __float_as_uint(fmaf(hi, cu, 12563648.0f));
  return (h << 16) + l;
}

//: FORM 1 AND CONV 3, TOKENS ON N IN THE FACTS PASS'S FORM: an 8-token tile's read rows (B: K = lists, N = tokens) by
//: one `ldmatrix.x4`, an address an add and an XOR; 4 MMAs against the digit's state (A, 16 registers); each lane's two
//: tokens' panel bases once, each output pair exact. HEAD 1: the tile's entries from the warp's heads (tiles 0-1)
template <int OUT, int HEAD>
__device__ __forceinline__ void tile_n3(uint32_t sm, const Lists& ls, uint32_t head, int64_t row, int begin, int n,
                                        const uint32_t (&f)[16], int lane, float& sink, int s) {
  const int r = lane & 7, q = lane >> 3, g = lane >> 2, t = lane & 3;
  const uint32_t mine = entry3<HEAD>(ls, head, row, begin, n, 8 * s + (lane & 7));
  const uint32_t eb = __shfl_sync(0xffffffffu, mine, r);
  const uint32_t e0 = __shfl_sync(0xffffffffu, mine, 2 * t), e1 = __shfl_sync(0xffffffffu, mine, 2 * t + 1);
  uint32_t b[4];
  lab::ldsm(b, (sm + kPool + (eb & 0xffffu)) ^ (16u * static_cast<uint32_t>(q)));
  float acc[2][4] = {};
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int kb = 0; kb < 2; ++kb) {
      const uint32_t a[4] = {f[(m * 2 + kb) * 4], f[(m * 2 + kb) * 4 + 1], f[(m * 2 + kb) * 4 + 2],
                             f[(m * 2 + kb) * 4 + 3]};
      lab::mma_bf16_f32(acc[m], a, b[2 * kb], b[2 * kb + 1]);
    }
  const float c0 = e0 ? kUnit : 0.0f, c1 = e1 ? kUnit : 0.0f;
  const uint32_t b0 = (e0 >> 16) ? sm + kPanel - 1 + (e0 >> 16) + 4 * g : sm + kDummy + 4 * lane;
  const uint32_t b1 = (e1 >> 16) ? sm + kPanel - 1 + (e1 >> 16) + 4 * g : sm + kDummy + 4 * lane;
#pragma unroll
  for (int m = 0; m < 2; ++m) {
    if constexpr (OUT == 0) {
      sink += acc[m][0] + acc[m][1] + acc[m][2] + acc[m][3];
    } else {
      panel_red(b0 + 32 * m, s16_exact(acc[m][0], acc[m][2], c0));
      panel_red(b1 + 32 * m, s16_exact(acc[m][1], acc[m][3], c1));
    }
  }
}

//: FORM 0, TOKENS ON M: the digit's state as B, a 16-token step's read rows as A
template <int OUT, int DENSE, int CONV = 0>
__device__ __forceinline__ void step_m(uint32_t sm, const Lists& ls, const uint16_t* head, int64_t row, int begin,
                                       int n, const uint32_t (&f)[16], int lane, float& sink, int s) {
  const int r = lane & 7, q = lane >> 3, g = lane >> 2, t = lane & 3;
  {
    const uint32_t mine = entry<DENSE>(ls, s < 1 ? head : nullptr, row, begin, n, 16 * s + (lane & 15));
    const uint32_t ea = __shfl_sync(0xffffffffu, mine, 8 * (q & 1) + r);
    const uint32_t eg = __shfl_sync(0xffffffffu, mine, g), eg8 = __shfl_sync(0xffffffffu, mine, g + 8);
    float acc[4][4] = {};
#pragma unroll
    for (int kb = 0; kb < 2; ++kb) {
      uint32_t a[4];
      lab::ldsm(a, chunk(sm + kPool, ea & 0xffu, 2 * kb + (q >> 1)));
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) lab::mma_bf16_f32(acc[nt], a, f[(kb * 4 + nt) * 2], f[(kb * 4 + nt) * 2 + 1]);
    }
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      if constexpr (OUT == 0) {
        sink += acc[nt][0] + acc[nt][1] + acc[nt][2] + acc[nt][3];
      } else if constexpr (CONV == 2) {
        const float cg = coef(eg), cg8 = coef(eg8);
        const uint32_t bg = panel_base(sm, eg, t, lane), bg8 = panel_base(sm, eg8, t, lane);
        panel_red(bg + 16 * nt, s16_fma(acc[nt][0], acc[nt][1], cg));
        panel_red(bg8 + 16 * nt, s16_fma(acc[nt][2], acc[nt][3], cg8));
      } else {
        const float cg = coef(eg), cg8 = coef(eg8);
        panel_add(sm, eg, 4 * nt + t,
                  CONV ? s16_fma(acc[nt][0], acc[nt][1], cg) : s16_pair(acc[nt][0] * cg, acc[nt][1] * cg), lane);
        panel_add(sm, eg8, 4 * nt + t,
                  CONV ? s16_fma(acc[nt][2], acc[nt][3], cg8) : s16_pair(acc[nt][2] * cg8, acc[nt][3] * cg8), lane);
      }
    }
  }
}

template <int OUT, int DENSE>
__device__ __forceinline__ void digit_m(uint32_t sm, const Lists& ls, const uint16_t* head, int64_t row, int begin,
                                        int n, const uint32_t (&f)[16], int lane, float& sink) {
#pragma unroll 1
  for (int s = 0; s < (n + 15) / 16; ++s) step_m<OUT, DENSE>(sm, ls, head, row, begin, n, f, lane, sink, s);
}

//: FORM 1, TOKENS ON N: the digit's state as A, an 8-token tile's read rows as B
template <int OUT, int DENSE>
__device__ __forceinline__ void digit_n(uint32_t sm, const Lists& ls, const uint16_t* head, int64_t row, int begin,
                                        int n, const uint32_t (&f)[16], int lane, float& sink) {
  const int r = lane & 7, q = lane >> 3, g = lane >> 2, t = lane & 3;
#pragma unroll 1
  for (int s = 0; s < (n + 7) / 8; ++s) {
    const uint32_t mine = entry<DENSE>(ls, s < 2 ? head : nullptr, row, begin, n, 8 * s + (lane & 7));
    const uint32_t eb = __shfl_sync(0xffffffffu, mine, r);
    const uint32_t e0 = __shfl_sync(0xffffffffu, mine, 2 * t), e1 = __shfl_sync(0xffffffffu, mine, 2 * t + 1);
    uint32_t b[4];
    lab::ldsm(b, chunk(sm + kPool, eb & 0xffu, q));
    float acc[2][4] = {};
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int kb = 0; kb < 2; ++kb) {
        const uint32_t a[4] = {f[(m * 2 + kb) * 4], f[(m * 2 + kb) * 4 + 1], f[(m * 2 + kb) * 4 + 2],
                               f[(m * 2 + kb) * 4 + 3]};
        lab::mma_bf16_f32(acc[m], a, b[2 * kb], b[2 * kb + 1]);
      }
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      if constexpr (OUT == 0) {
        sink += acc[m][0] + acc[m][1] + acc[m][2] + acc[m][3];
      } else {
        const float c0 = coef(e0), c1 = coef(e1);
        panel_add(sm, e0, 8 * m + g, s16_pair(acc[m][0] * c0, acc[m][2] * c0), lane);
        panel_add(sm, e1, 8 * m + g, s16_pair(acc[m][1] * c1, acc[m][3] * c1), lane);
      }
    }
  }
}

template <int FORM, int OUT, int DENSE, int RING, int NOPAD, int BAR, int CONV>
__global__ __launch_bounds__(kThreads, 1) void read(Lists ls, long long* times, int* sinks) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int warp = lab::warp_id(), lane = lab::lane_id();
  const int boxes = ls.width / kDigits, box = static_cast<int>(blockIdx.x) % boxes;
  const uint32_t sm = lab::smem(smem);
  uint32_t* words = reinterpret_cast<uint32_t*>(smem);
  for (int k = static_cast<int>(threadIdx.x); k < kPanel / 4; k += kThreads) words[k] = 0x3d803d80u;
  for (int k = kPanel / 4 + static_cast<int>(threadIdx.x); k < kBytes / 4; k += kThreads) words[k] = 0u;
  __syncthreads();
  float sink = 0.0f;
  //: the warp's heads for window `w` into buffer `w & 1` by `cp.async` (lanes 0-11: four pass-start rows, eight
  //: 16-byte halves of the first 16 slots), a window ahead of their use
  const auto heads_in = [&](int w) {
    const int i = lane & 3, part = lane >> 2;
    const int64_t row = static_cast<int64_t>(w) * ls.width + box * kDigits + warp + 8 * i;
    const uint32_t dst = sm + kHeads + ((w & 1) * kWarps + warp) * kHeadBytes;
    if (!DENSE && part == 0) lab::cp_async16(dst + 16 * i, ls.at + row * kListAt);
    if (!DENSE && CONV != 3 && (part == 1 || part == 2))
      lab::cp_async16(dst + 64 + 32 * i + 16 * (part - 1), ls.slots + row * kListCap + 8 * (part - 1));
    if (!DENSE && CONV == 3 && part >= 1 && part <= 4)
      lab::cp_async16(dst + 64 + 64 * i + 16 * (part - 1), ls.offs + row * kListCap + 4 * (part - 1));
    lab::cp_commit();
  };
  //: CONV 3, form 1: the lane's two A-fragment offsets in a plane (m = 0, 1; kb = 1 is 1024 bytes on), once a kernel
  uint32_t sn[2];
#pragma unroll
  for (int m = 0; m < 2; ++m) {
    const int srow = 8 * (lane >> 4) + (lane & 7), c = 2 * m + ((lane >> 3) & 1);
    sn[m] = srow * kRowBytes + 16 * (c ^ ((srow >> 1) & 3));
  }
  //: CONV 3: the lane's two state-load offsets in a plane (u = 0; u = 1 is 1024 bytes on), once a kernel
  uint32_t so[2];
#pragma unroll
  for (int v = 0; v < 2; ++v) {
    const int srow = 8 * ((lane >> 3) & 1) + (lane & 7), c = 2 * v + (lane >> 4);
    so[v] = srow * kRowBytes + 16 * (c ^ ((srow >> 1) & 3));
  }
  heads_in(0);
  int next = DENSE ? 2 * kSlots : ls.count[box];
  const long long t0 = clock64();
#pragma unroll 1
  for (int w = 0; w < ls.windows; ++w) {
    const int count = next;
    for (int k = w + 1 < ls.windows ? 1 : 0; k > 0; --k) {
      heads_in(w + 1);
      next = DENSE ? 2 * kSlots : ls.count[(w + 1) * ls.groups + box];
    }
    asm volatile("cp.async.wait_group 1;" ::: "memory");
    __syncwarp();
    const uint8_t* mine = smem + kHeads + ((w & 1) * kWarps + warp) * kHeadBytes;
    const int passes = (count + kSlots - 1) / kSlots;
#pragma unroll 1
    for (int p = 0; p < passes; ++p) {
      const auto digit = [&](int i, const uint32_t(&f)[16]) {
        const int64_t row = static_cast<int64_t>(w) * ls.width + box * kDigits + warp + 8 * i;
        const uint16_t* at = reinterpret_cast<const uint16_t*>(mine) + 8 * i;
        const uint16_t* head = p == 0 ? reinterpret_cast<const uint16_t*>(mine + 64) + 16 * i : nullptr;
        const int begin = DENSE ? 0 : at[2 * p], n = DENSE ? kSlots : at[2 * p + 2] - begin;
        if constexpr (FORM == 0) digit_m<OUT, DENSE>(sm, ls, head, row, begin, n, f, lane, sink);
        if constexpr (FORM == 1) digit_n<OUT, DENSE>(sm, ls, head, row, begin, n, f, lane, sink);
      };
      const auto plane = [&](int i) { return sm + kImage + ((warp + 8 * i) % kPlanes) * kPlaneBytes; };
      //: RING 2 (form 0): every digit's first step in one straight-line block, as the fold's, so the scheduler loads a
      //: digit's operands under the MMAs before them; the rare digit past 16 tokens continues after
      if constexpr (RING == 2) {
        const auto args = [&](int i, int64_t& row, const uint16_t*& head, int& begin, int& n) {
          row = static_cast<int64_t>(w) * ls.width + box * kDigits + warp + 8 * i;
          const uint16_t* at = reinterpret_cast<const uint16_t*>(mine) + 8 * i;
          head = p == 0 ? reinterpret_cast<const uint16_t*>(mine + 64) + 16 * i : nullptr;
          begin = DENSE ? 0 : at[2 * p];
          n = DENSE ? kSlots : at[2 * p + 2] - begin;
        };
        uint32_t f[4][16];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          int64_t row;
          const uint16_t* head;
          int begin, n;
          args(i, row, head, begin, n);
          if constexpr (CONV == 3 && FORM == 1) {
#pragma unroll
            for (int m = 0; m < 2; ++m) {
              lab::ldsm_t(*reinterpret_cast<uint32_t(*)[4]>(&f[i][(m * 2) * 4]), plane(i) + sn[m]);
              lab::ldsm_t(*reinterpret_cast<uint32_t(*)[4]>(&f[i][(m * 2 + 1) * 4]), plane(i) + sn[m] + 1024u);
            }
          } else if constexpr (CONV == 3) {
#pragma unroll
            for (int v = 0; v < 2; ++v) {
              uint32_t x[4];
              lab::ldsm_t(x, plane(i) + so[v]);
              f[i][(2 * v) * 2] = x[0];
              f[i][(2 * v) * 2 + 1] = x[1];
              f[i][(2 * v + 1) * 2] = x[2];
              f[i][(2 * v + 1) * 2 + 1] = x[3];
              lab::ldsm_t(x, plane(i) + so[v] + 1024u);
              f[i][(4 + 2 * v) * 2] = x[0];
              f[i][(4 + 2 * v) * 2 + 1] = x[1];
              f[i][(4 + 2 * v + 1) * 2] = x[2];
              f[i][(4 + 2 * v + 1) * 2 + 1] = x[3];
            }
          } else {
            state_in<FORM>(f[i], plane(i), lane);
          }
          if constexpr (CONV == 3 && FORM == 1) {
            const uint32_t h32 = sm + kHeads + ((w & 1) * kWarps + warp) * kHeadBytes + 64 + 64 * i;
            for (int k = p == 0 ? 1 : 0; k > 0; --k) tile_n3<OUT, 1>(sm, ls, h32, row, begin, n, f[i], lane, sink, 0);
            for (int k = p == 0 ? 0 : 1; k > 0; --k) tile_n3<OUT, 0>(sm, ls, 0u, row, begin, n, f[i], lane, sink, 0);
          } else if constexpr (CONV == 3) {
            const uint32_t h32 = sm + kHeads + ((w & 1) * kWarps + warp) * kHeadBytes + 64 + 64 * i;
            for (int k = p == 0 ? 1 : 0; k > 0; --k) step_m3<OUT, 1>(sm, ls, h32, row, begin, n, f[i], lane, sink, 0);
            for (int k = p == 0 ? 0 : 1; k > 0; --k) step_m3<OUT, 0>(sm, ls, 0u, row, begin, n, f[i], lane, sink, 0);
          } else if constexpr (NOPAD == 1) {
            for (int k = n > 8 ? 1 : 0; k > 0; --k) step_m<OUT, DENSE, CONV>(sm, ls, head, row, begin, n, f[i], lane, sink, 0);
            for (int k = n > 0 && n <= 8 ? 1 : 0; k > 0; --k)
              half_m<OUT, DENSE, CONV>(sm, ls, head, row, begin, n, f[i], lane, sink, 0);
          } else {
            step_m<OUT, DENSE, CONV>(sm, ls, head, row, begin, n, f[i], lane, sink, 0);
          }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          int64_t row;
          const uint16_t* head;
          int begin, n;
          args(i, row, head, begin, n);
          const int full = FORM == 1 ? (n + 7) / 8 : NOPAD ? (n + 7) / 16 : (n + 15) / 16;
          if constexpr (CONV == 3 && FORM == 1) {
            const uint32_t h32 = sm + kHeads + ((w & 1) * kWarps + warp) * kHeadBytes + 64 + 64 * i;
            for (int k = p == 0 && full > 1 ? 1 : 0; k > 0; --k)
              tile_n3<OUT, 1>(sm, ls, h32, row, begin, n, f[i], lane, sink, 1);
#pragma unroll 1
            for (int s = p == 0 ? 2 : 1; s < full; ++s) tile_n3<OUT, 0>(sm, ls, 0u, row, begin, n, f[i], lane, sink, s);
            continue;
          }
#pragma unroll 1
          for (int s = 1; s < full; ++s) {
            if constexpr (CONV == 3) {
              step_m3<OUT, 0>(sm, ls, 0u, row, begin, n, f[i], lane, sink, s);
            } else {
              step_m<OUT, DENSE, CONV>(sm, ls, head, row, begin, n, f[i], lane, sink, s);
            }
          }
          for (int k = NOPAD && full > 0 && n - 16 * full > 0 ? 1 : 0; k > 0; --k)
            half_m<OUT, DENSE, CONV>(sm, ls, head, row, begin, n, f[i], lane, sink, full);
        }
      } else if constexpr (RING == 1) {
        uint32_t f0[16], f1[16];
        state_in<FORM>(f0, plane(0), lane);
        state_in<FORM>(f1, plane(1), lane);
        digit(0, f0);
        state_in<FORM>(f0, plane(2), lane);
        digit(1, f1);
        state_in<FORM>(f1, plane(3), lane);
        digit(2, f0);
        digit(3, f1);
      } else {
#pragma unroll 1
        for (int i = 0; i < 4; ++i) {
          uint32_t f[16];
          state_in<FORM>(f, plane(i), lane);
          digit(i, f);
        }
      }
      if constexpr (BAR == 1) __syncthreads();
      if constexpr (OUT == 2) {
        const int rows = min(kSlots, count - kSlots * p);
#pragma unroll 1
        for (int k = 0; k < (min(rows - kSlots / kWarps * warp, kSlots / kWarps) + 1) / 2; ++k) {
          const int slot = kSlots / kWarps * warp + 2 * k + (lane >> 4), word = lane & 15;
          uint32_t* cell = words + kPanel / 4 + slot * kPanelPitch + word;
          uint32_t* target = ls.out + ((static_cast<int64_t>(w) * 2 + p) * kSlots + slot) * kWords + word;
          asm volatile("red.global.add.u32 [%0], %1;" ::"l"(target), "r"(*cell) : "memory");
          *cell = 0u;
        }
        __syncthreads();
      }
    }
  }
  const long long spent = clock64() - t0;
  lab::sink(sinks, lab::bits(sink));
  //: the check's read of the panel, after the timed loop: every word a CTA's adds left in it
  if constexpr (OUT >= 1) {
    __syncthreads();
    for (int k = static_cast<int>(threadIdx.x); k < kSlots * kWords; k += kThreads)
      ls.panel_out[static_cast<int64_t>(blockIdx.x) * kSlots * kWords + k] =
          words[kPanel / 4 + (k / kWords) * kPanelPitch + k % kWords];
  }
  if (lane == 0) times[blockIdx.x * kWarps + warp] = spent;
}

struct HostLists {
  int windows = 0, width = 0, groups = 0;
  std::vector<uint8_t> records;
  std::vector<uint16_t> at, count;
};

HostLists read_lists(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  int head[4];
  f.read(reinterpret_cast<char*>(head), sizeof(head));
  if (!f || head[0] != 0x44414552) {
    std::fprintf(stderr, "read_image: %s is not a read lists file\n", path.c_str());
    std::exit(1);
  }
  HostLists h;
  h.windows = head[1];
  h.width = head[2];
  h.groups = head[3];
  h.records.resize(static_cast<size_t>(h.windows) * h.width * kListCap * 2);
  h.at.resize(static_cast<size_t>(h.windows) * h.width * kListAt);
  h.count.resize(static_cast<size_t>(h.windows) * h.groups);
  f.read(reinterpret_cast<char*>(h.records.data()), static_cast<std::streamsize>(h.records.size()));
  f.read(reinterpret_cast<char*>(h.at.data()), static_cast<std::streamsize>(h.at.size() * 2));
  f.read(reinterpret_cast<char*>(h.count.data()), static_cast<std::streamsize>(h.count.size() * 2));
  return h;
}

//: THE TENSOR FLOOR a CTA-window: the busiest scheduler's two warps' MMAs (form 0: 8 a 16-token step; form 1: 4 an
//: 8-token tile) at 32 cycles (bf16, fp32 accumulate, sm_86), averaged over the boxes and windows
double floor_of(const HostLists& h, int form, bool dense) {
  const int boxes = h.width / kDigits;
  double total = 0;
  for (int w = 0; w < h.windows; ++w)
    for (int box = 0; box < boxes; ++box) {
      const int passes = dense ? 2 : (h.count[w * h.groups + box] + kSlots - 1) / kSlots;
      long long mmas[kWarps] = {};
      for (int p = 0; p < passes; ++p)
        for (int d = 0; d < kDigits; ++d) {
          const uint16_t* a = &h.at[(static_cast<size_t>(w) * h.width + box * kDigits + d) * kListAt];
          const int n = dense ? kSlots : a[2 * p + 2] - a[2 * p];
          mmas[d % kWarps] += form == 0 ? (n + 15) / 16 * 8 : (n + 7) / 8 * 4;
        }
      long long busiest = 0;
      for (int s = 0; s < 4; ++s) busiest = std::max(busiest, mmas[s] + mmas[s + 4]);
      total += busiest * 32.0;
    }
  return total / (static_cast<double>(h.windows) * boxes);
}

//: THE CHECK's reference: each live (token, digit) add is 32 lists x 0.0625 x 0.0625 x the unit 256 = 32 a channel, a
//: word 32 x 65537 (either pair form, exact under u32 adds mod 2^32); a slot's row is that times its live entries --
//: OUT 1, a CTA's panel summed over its windows; OUT 2, a window-pass's drain summed over the CTAs
struct Reference {
  const HostLists* h;
  int sms;
};

struct Verdict {
  long long bad, words;
  std::string detail;
};

template <int OUT, int DENSE>
Verdict verify(const Reference& ref, const Lists& ls) {
  const HostLists& h = *ref.h;
  const int boxes = h.width / kDigits, windows = h.windows;
  const uint32_t word = 32u * 65537u;
  const auto live = [&](int w, int box, int p, std::vector<uint32_t>& count) {
    for (int d = 0; d < kDigits; ++d) {
      if (DENSE) {
        for (int s = 0; s < kSlots; ++s) count[s] += 1;
        continue;
      }
      const size_t row = static_cast<size_t>(w) * h.width + box * kDigits + d;
      const uint16_t* at = &h.at[row * kListAt];
      const uint16_t* slots = reinterpret_cast<const uint16_t*>(h.records.data()) + row * kListCap;
      for (int e = at[2 * p]; e < at[2 * p + 2]; ++e) count[slots[e] & 255] += 1;
    }
  };
  std::vector<uint32_t> got;
  long long bad = 0, words = 0;
  if constexpr (OUT == 1) {
    got.resize(static_cast<size_t>(ref.sms) * kSlots * kWords);
    LAB_CHECK(cudaMemcpy(got.data(), ls.panel_out, got.size() * 4, cudaMemcpyDeviceToHost));
    for (int c = 0; c < ref.sms; ++c) {
      std::vector<uint32_t> count(kSlots, 0);
      for (int w = 0; w < windows; ++w)
        for (int p = 0; p < (DENSE ? 2 : 1); ++p) live(w, c % boxes, p, count);
      for (int s = 0; s < kSlots; ++s)
        for (int k = 0; k < kWords; ++k, ++words)
          bad += got[(static_cast<size_t>(c) * kSlots + s) * kWords + k] != count[s] * word;
    }
  } else {
    got.resize(static_cast<size_t>(windows) * 2 * kSlots * kWords);
    LAB_CHECK(cudaMemcpy(got.data(), ls.out, got.size() * 4, cudaMemcpyDeviceToHost));
    for (int w = 0; w < windows; ++w)
      for (int p = 0; p < 2; ++p) {
        std::vector<uint32_t> count(kSlots, 0);
        if (DENSE || p == 0)
          for (int c = 0; c < ref.sms; ++c) live(w, c % boxes, p, count);
        for (int s = 0; s < kSlots; ++s)
          for (int k = 0; k < kWords; ++k, ++words)
            bad += got[((static_cast<size_t>(w) * 2 + p) * kSlots + s) * kWords + k] != count[s] * word;
      }
  }
  return {bad, words,
          std::to_string(words - bad) + " of " + std::to_string(words) + " panel words exact" +
              (OUT == 1 ? " (each CTA's panel)" : " (each window-pass's drain)")};
}

template <int FORM, int OUT, int DENSE, int RING, int NOPAD = 0, int BAR = 1, int CONV = 0>
void variant(const lab::Bench& b, const Lists& ls, double floor, double floor_m, const char* lists = "",
             const Reference* ref = nullptr) {
  static_assert(BAR == 1 || OUT < 2, "the drain needs the pass's barrier");
  const std::string name = lab::kernel("read_image::read", {FORM, OUT, DENSE, RING, NOPAD, BAR, CONV});
  b.expect(name, "HMMA.16816.F32.BF16", ">=", FORM == 0 ? 8 : 4);
  //: a half-step form chooses a digit's full or half step by its count, a warp-uniform branch among the first steps
  if (NOPAD) b.expect(name, "BSSY", ">=", 0);
  //: the F2I forms measure the conversion unit on purpose; a digit past 16 tokens (0.2% at alt-k4) reads its later
  //: entries from global memory inside its steps
  if (CONV == 0) b.expect(name, "STEP_XU", ">=", 0);
  if (!DENSE) b.expect(name, "STEP_GLOBAL", ">=", 0);
  //: THE BUDGET (KERNEL_STANDARDS section 29), the facts-form readout's operation count per MMA at alt-k4, where a
  //: read digit is one 16-token step a window (8 MMAs) and a warp four digits a window. A step: 2 read-row loads, a
  //: head load, 3 shuffles, 16 FFMA conversions, 8 packs, 8 panel adds, its 2 tokens' panel bases (a shift, a compare,
  //: a select, an add each), an add and 2 XORs for the read rows. A digit: 4 state loads (2 adds), its pass start and
  //: count (2 loads), its list's row, the step loop's bound and branch. A window: the heads' copies (5), the count's
  //: load, the loop. Per MMA: shared load (7 + 2/1)/8, integer ~14/8, select ~3/8, control ~4/8, global ~6/32.
  //: tokens on N, an 8-token tile 4 MMAs: a read-row load, a head load, 3 shuffles, 8 FFMAs, 4 LEA packs, 4 panel
  //: adds, 2 panel bases; a digit (~1.45 tiles at alt-k4) its 4 state loads and its pass starts
  //: the variants without a budget, each with why: an isolation, a drain over a budgeted step, or a baseline form
  const bool budgeted = OUT == 1 && ((CONV == 3 && FORM == 1) || (CONV >= 2 && FORM == 0));
  if (!budgeted)
    b.exempt(name, OUT == 0                            ? "an isolation: the MMAs alone"
                   : OUT == 2 && CONV >= 2             ? "a budgeted step (its OUT 1 variant) plus the pass's drain"
                   : DENSE                             ? "the dense cell, priced by its floor"
                                                       : "a baseline form the facts form replaced, its counts recorded");
  if (CONV == 3 && FORM == 1 && OUT == 1) {
    b.mma(name, "hmma_bf16_f32", 16);
    b.budget(name, "mma", 1.0);
    b.budget(name, "shared load", 1.6);
    b.budget(name, "shared atomic", 1.0);
    b.budget(name, "shuffle", 0.8);
    b.budget(name, "pack", 0.1);
    b.budget(name, "float", 2.1);
    b.budget(name, "conversion unit", 0.0);
    b.budget(name, "select/compare", 0.8);
    b.budget(name, "integer/address", 3.2);
    b.budget(name, "control", 0.8);
    b.budget(name, "global", 0.3);
  }
  if (CONV >= 2 && FORM == 0 && OUT == 1) {
    b.mma(name, "hmma_bf16_f32", 16);
    b.budget(name, "mma", 1.0);
    b.budget(name, "shared load", 1.15);
    b.budget(name, "shared atomic", 1.0);
    b.budget(name, "shuffle", 0.4);
    b.budget(name, "pack", 1.0);
    b.budget(name, "float", 2.3);
    b.budget(name, "conversion unit", 0.0);
    b.budget(name, "select/compare", 0.4);
    b.budget(name, "integer/address", 1.8);
    b.budget(name, "control", 0.55);
    b.budget(name, "global", 0.25);
  }
  if (b.expect_only) return;
  LAB_CHECK(cudaFuncSetAttribute(read<FORM, OUT, DENSE, RING, NOPAD, BAR, CONV>, cudaFuncAttributeMaxDynamicSharedMemorySize, kBytes));
  const lab::Samples s = b.variant([&] { read<FORM, OUT, DENSE, RING, NOPAD, BAR, CONV><<<b.sms, kThreads, kBytes>>>(ls, b.times, b.sinks); },
                                   [&] { return lab::Bench::mean(b.cycles(kWarps), kWarps, 0, kWarps) / ls.windows; });
  const char* forms[2] = {"tokens on M", "tokens on N"};
  const char* outs[3] = {"the MMAs alone", "+ panel adds", "+ panel adds + drain"};
  const std::string label = std::string(DENSE ? "dense / " : "alt-k4 / ") + forms[FORM] + " / " + outs[OUT]
                            + (RING == 2 ? " / first steps straight-line" : RING ? " / state ring" : " / state a digit at a time")
                            + lists + (NOPAD ? " / half steps" : "") + (BAR ? "" : " / no pass barrier")
                            + (CONV == 3   ? " / FMA conversion / facts-form entries"
                               : CONV == 2 ? " / FMA conversion / a token's panel base once"
                               : CONV      ? " / FMA conversion"
                                           : "");
  b.record(label, "cycles a CTA-window", s);
  lab::Samples own, m;
  for (double v : s.v) {
    own.v.push_back(v / floor);
    m.v.push_back(v / floor_m);
  }
  b.record(label + " / its floor", "x", own);
  b.record(label + " / tokens-on-M floor", "x", m);
  if constexpr (OUT == 0) {
    b.unchecked(label, "an isolation: the MMAs alone, their outputs folded into a sink");
  } else {
    LAB_CHECK(cudaMemset(ls.out, 0, static_cast<size_t>(ls.windows) * 2 * kSlots * kWords * 4));
    LAB_CHECK(cudaMemset(ls.panel_out, 0, static_cast<size_t>(b.sms) * kSlots * kWords * 4));
    read<FORM, OUT, DENSE, RING, NOPAD, BAR, CONV><<<b.sms, kThreads, kBytes>>>(ls, b.times, b.sinks);
    LAB_CHECK(cudaDeviceSynchronize());
    const Verdict v = verify<OUT, DENSE>(*ref, ls);
    b.check(label, v.bad == 0 && v.words > 0, v.detail);
  }
}

}  // namespace read_image

int main(int argc, char** argv) {
  using namespace read_image;
  lab::Bench b(argc, argv);
  std::string dir;
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], "--inputs")) dir = argv[i + 1];
  Lists ls{}, lo{};
  HostLists host;
  double fm = 1, fn = 1, dm = 1, dn = 1;
  if (!b.expect_only) {
    if (dir.empty()) {
      std::fprintf(stderr, "read_image: --inputs <directory> (tools/lab.py writes it)\n");
      return 1;
    }
    const HostLists h = read_lists(dir + "/read.bin");
    uint8_t* records = nullptr;
    uint16_t *at = nullptr, *count = nullptr;
    uint32_t* out = nullptr;
    LAB_CHECK(cudaMalloc(&records, h.records.size()));
    LAB_CHECK(cudaMalloc(&at, h.at.size() * 2));
    LAB_CHECK(cudaMalloc(&count, h.count.size() * 2));
    LAB_CHECK(cudaMalloc(&out, static_cast<size_t>(h.windows) * 2 * kSlots * kWords * 4));
    uint32_t* panel_out = nullptr;
    LAB_CHECK(cudaMalloc(&panel_out, static_cast<size_t>(b.sms) * kSlots * kWords * 4));
    LAB_CHECK(cudaMemcpy(records, h.records.data(), h.records.size(), cudaMemcpyHostToDevice));
    LAB_CHECK(cudaMemcpy(at, h.at.data(), h.at.size() * 2, cudaMemcpyHostToDevice));
    LAB_CHECK(cudaMemcpy(count, h.count.data(), h.count.size() * 2, cudaMemcpyHostToDevice));
    ls = Lists{reinterpret_cast<const uint16_t*>(records), nullptr, at, count, h.windows, h.width, h.groups, out,
               panel_out};
    host = h;
    const HostLists o = read_lists(dir + "/ordered.bin");
    uint8_t* ordered = nullptr;
    LAB_CHECK(cudaMalloc(&ordered, o.records.size()));
    LAB_CHECK(cudaMemcpy(ordered, o.records.data(), o.records.size(), cudaMemcpyHostToDevice));
    lo = ls;
    lo.slots = reinterpret_cast<const uint16_t*>(ordered);
    std::vector<uint32_t> host_offs(o.records.size() / 2);
    std::ifstream fo(dir + "/offsets.bin", std::ios::binary);
    fo.read(reinterpret_cast<char*>(host_offs.data()), static_cast<std::streamsize>(host_offs.size() * 4));
    if (!fo) {
      std::fprintf(stderr, "read_image: %s/offsets.bin is short\n", dir.c_str());
      return 1;
    }
    uint32_t* offs = nullptr;
    LAB_CHECK(cudaMalloc(&offs, host_offs.size() * 4));
    LAB_CHECK(cudaMemcpy(offs, host_offs.data(), host_offs.size() * 4, cudaMemcpyHostToDevice));
    lo.offs = offs;
    fm = floor_of(h, 0, false);
    fn = floor_of(h, 1, false);
    dm = floor_of(h, 0, true);
    dn = floor_of(h, 1, true);
    std::printf("{\"lab\": \"note\", \"floor_m\": %.1f, \"floor_n\": %.1f, \"dense_m\": %.1f, \"dense_n\": %.1f}\n", fm,
                fn, dm, dn);
  }
  const Reference ref{&host, b.sms};
  variant<0, 1, 0, 0>(b, ls, fm, fm, "", &ref);
  variant<1, 1, 0, 0>(b, ls, fn, fm, "", &ref);
  variant<0, 0, 0, 1>(b, ls, fm, fm, "", &ref);
  variant<0, 1, 0, 1>(b, ls, fm, fm, "", &ref);
  variant<0, 2, 0, 1>(b, ls, fm, fm, "", &ref);
  variant<1, 0, 0, 1>(b, ls, fn, fm, "", &ref);
  variant<1, 1, 0, 1>(b, ls, fn, fm, "", &ref);
  variant<1, 2, 0, 1>(b, ls, fn, fm, "", &ref);
  variant<0, 0, 0, 2>(b, ls, fm, fm, "", &ref);
  variant<0, 1, 0, 2>(b, ls, fm, fm, "", &ref);
  variant<0, 2, 0, 2>(b, ls, fm, fm, "", &ref);
  variant<0, 0, 0, 2>(b, lo, fm, fm, " / ordered", &ref);
  variant<0, 1, 0, 2>(b, lo, fm, fm, " / ordered", &ref);
  variant<0, 0, 0, 2, 1>(b, lo, fm, fm, " / ordered", &ref);
  variant<0, 1, 0, 2, 1>(b, lo, fm, fm, " / ordered", &ref);
  variant<0, 2, 0, 2, 1>(b, lo, fm, fm, " / ordered", &ref);
  variant<0, 0, 0, 2, 1, 0>(b, lo, fm, fm, " / ordered", &ref);
  variant<0, 1, 0, 2, 1, 0>(b, lo, fm, fm, " / ordered", &ref);
  variant<0, 1, 0, 2, 0, 1, 1>(b, lo, fm, fm, " / ordered", &ref);
  variant<0, 2, 0, 2, 0, 1, 1>(b, lo, fm, fm, " / ordered", &ref);
  variant<0, 1, 0, 2, 1, 1, 1>(b, lo, fm, fm, " / ordered", &ref);
  variant<0, 2, 0, 2, 1, 1, 1>(b, lo, fm, fm, " / ordered", &ref);
  variant<0, 1, 0, 2, 0, 0, 1>(b, lo, fm, fm, " / ordered", &ref);
  variant<0, 2, 1, 2, 0, 1, 1>(b, ls, dm, dm, "", &ref);
  variant<0, 1, 0, 2, 0, 1, 2>(b, lo, fm, fm, " / ordered", &ref);
  variant<0, 2, 0, 2, 0, 1, 2>(b, lo, fm, fm, " / ordered", &ref);
  variant<0, 1, 0, 2, 0, 0, 2>(b, lo, fm, fm, " / ordered", &ref);
  variant<0, 2, 1, 2, 0, 1, 2>(b, ls, dm, dm, "", &ref);
  variant<0, 1, 0, 2, 0, 1, 3>(b, lo, fm, fm, " / ordered", &ref);
  variant<0, 2, 0, 2, 0, 1, 3>(b, lo, fm, fm, " / ordered", &ref);
  variant<0, 1, 0, 2, 0, 0, 3>(b, lo, fm, fm, " / ordered", &ref);
  variant<1, 1, 0, 2, 0, 1, 3>(b, lo, fn, fm, " / ordered", &ref);
  variant<1, 1, 0, 2, 0, 0, 3>(b, lo, fn, fm, " / ordered", &ref);
  variant<1, 0, 0, 2, 0, 0, 3>(b, lo, fn, fm, " / ordered", &ref);
  variant<1, 2, 0, 2, 0, 1, 3>(b, lo, fn, fm, " / ordered", &ref);
  variant<0, 2, 1, 2, 1>(b, ls, dm, dm, "", &ref);
  variant<0, 2, 1, 1>(b, ls, dm, dm, "", &ref);
  variant<1, 2, 1, 1>(b, ls, dn, dm, "", &ref);
}
