// measure/lab/experiments/read_image.cu -- QUESTION: the carry's readout step in its ruled final form, the state from
// a bf16 image in shared memory: tokens on the MMA's M (16-token steps, the state the B operand) or on its N (8-token
// tiles, the state the A operand) -- what does each cost against its tensor floor and its instruction budget at
// nl64k-alt-k4's read lists and at a dense cell, and what do the panel's adds and the pass's drain add? METHOD: 8
// warps, one CTA an SM a box of 32 lists x 32 read digits x 32 channels; a warp reads digits w, w + 8, w + 16, w + 24,
// every digit's first step in one straight-line block, its state fragments (16 registers) from the image by
// `ldmatrix.trans`; read rows (32 lists of bf16, 64 bytes, swizzled) gathered from the pass's pool by the entries the
// facts pass emits in their final form (`read_image.inputs.py`): the pool row's offset, the coefficient, the panel
// row's offset with the lane's token folded in, so a pad lands on its residue's dummy row with no test -- the slots
// colored and the tokens placed so every gather and add is free of bank conflicts (KERNEL_STANDARDS 30); bf16 MMAs with
// fp32 accumulate; each output pair times its coefficient to the exact integer hi x 2^16 + lo on the FMA pipe, added
// by `red.shared` into its token's panel row; the drain at the pass's end (`red.global.add`, each word cleared). A
// warp's heads (its digits' first entries) staged a window ahead by `cp.async`. Outputs checked exactly, the
// even channels negative (a low half below zero carries into the high half) and two coefficients.
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
//: a readout pass's slots: 256 (sm_86's), or 512 (sm_80's) where `read_w1024.cu` builds this file for W = 1024
#ifndef READ_IMAGE_SLOTS
#define READ_IMAGE_SLOTS 256
#endif
constexpr int kListCap = 512, kListAt = 8, kSlots = READ_IMAGE_SLOTS, kWords = 16, kPanelPitch = 20;
//: the image's planes (a plane's access pattern is all a digit's reads see) and a digit's entries staged: 16 and 16, or
//: 4 and 32 beside a 512-slot pass (so the counts fit sm_86)
constexpr int kPlanes = kSlots == 512 ? 4 : 16, kHead = kSlots == 512 ? 32 : 16;
constexpr int kPlaneBytes = kLists * kChannels * 2, kRowBytes = kLists * 2;
//: THE LAYOUT the entries are emitted against (`read_image.inputs.py` mirrors it): image, pool, the lanes' pad bases
//: (128 bytes, never themselves touched), 8 dummy panel rows (a pad's add lands on its residue's, the banks a real row
//: of that residue would use), panel, heads
constexpr int kImage = 0, kPool = kImage + kPlanes * kPlaneBytes, kPads = kPool + kSlots * kRowBytes;
constexpr int kPanel = kPads + 128 + 8 * kPanelPitch * 4, kHeads = kPanel + kSlots * kPanelPitch * 4;
//: a warp's heads for a window, double-buffered: its 4 digits' first entries, a 16-byte chunk (or two) a lane
constexpr int kHeadBytes = 4 * kHead * 8, kBytes = kHeads + 2 * kWarps * kHeadBytes;
//: the dense cell's coefficient times the unit
constexpr float kUnit = 256.0f;

struct Lists {
  const uint2* entries;
  const uint8_t* heads;
  const uint16_t* count;
  int windows, width, groups;
  uint32_t* out;
  uint32_t* panel_out;
};

//: A LANE'S CONSTANTS, once a kernel: its shuffle sources (the read row's token, the outputs' two tokens), the read
//: rows' chunk XOR, its pad word, its state loads' two addresses in its plane (each + 1024 bytes for lists 16-31), its
//: entry's offset in a head block
struct Lane {
  int src_a, src_g, src_h;
  uint32_t chunk, pad, st0, st1, entry;
};

__device__ __forceinline__ uint32_t in_plane(int row, int c) {
  return static_cast<uint32_t>(row * kRowBytes + 16 * (c ^ ((row >> 1) & 3)));
}

template <int FORM>
__device__ __forceinline__ Lane lane_of(uint32_t sm, int warp, int lane) {
  const int r = lane & 7, q = lane >> 3, g = lane >> 2, t = lane & 3;
  const uint32_t plane = sm + kImage + (warp % kPlanes) * kPlaneBytes;
  const int srow = FORM == 0 ? 8 * (q & 1) + r : 8 * (lane >> 4) + r, sc = FORM == 0 ? q >> 1 : q & 1;
  const auto h = [](auto v) { return static_cast<decltype(v)>(lab::held(static_cast<uint32_t>(v))); };
  return {h(FORM == 0 ? 8 * (q & 1) + r : r), h(FORM == 0 ? g : 2 * t), h(FORM == 0 ? g + 8 : 2 * t + 1),
          h(FORM == 0 ? 16u * (q >> 1) : 16u * q), h(sm + kPads + (FORM == 0 ? 4u * lane : 4u * g + 32u * t)),
          h(plane + in_plane(srow, sc)), h(plane + in_plane(srow, 2 + sc)),
          h(8u * (FORM == 0 ? lane & 15 : r))};
}

//: digit `i`'s plane, bytes past the warp's first: a compile-time offset for an unrolled `i`
__device__ __forceinline__ constexpr uint32_t plane_off(int i) { return (8 * i % kPlanes) * kPlaneBytes; }

//: A DIGIT'S STATE FRAGMENTS, 16 registers: form 0's B (K = lists, N = channels) at `(kb * 4 + nt) * 2 + h`, form 1's
//: A (M = channels, K = lists) at `(m * 2 + kb) * 4 + i`
template <int FORM>
__device__ __forceinline__ void state_in(uint32_t (&f)[16], const Lane& c, uint32_t off) {
#pragma unroll
  for (int v = 0; v < 2; ++v) {
    uint32_t x[4], y[4];
    lab::ldsm_t(x, (v ? c.st1 : c.st0) + off);
    lab::ldsm_t(y, (v ? c.st1 : c.st0) + off + 1024u);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      f[(FORM == 0 ? 4 : 8) * v + k] = x[k];
      f[(FORM == 0 ? 8 + 4 * v : 8 * v + 4) + k] = y[k];
    }
  }
}

__device__ __forceinline__ uint2 lds64(uint32_t a) {
  uint2 v;
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a));
  return v;
}

__device__ __forceinline__ void panel_red(uint32_t at, uint32_t v) {
  asm volatile("red.shared.add.u32 [%0], %1;" ::"r"(at), "r"(v) : "memory");
}

//: THE EXACT PAIR: two outputs times (coefficient x unit) to the integer hi x 2^16 + lo, so u32 adds sum both channels
//: exactly while each final fits s16 -- lo's FFMA adds 1.5 x 2^23 (its bits 0x4B400000 + lo), hi's adds 0x4B40 less
//: (its low half hi - 0x4B40), and one LEA (hi bits << 16, plus lo bits) cancels lo's exponent bits
__device__ __forceinline__ uint32_t s16_exact(float lo, float hi, float cu) {
  const uint32_t l = __float_as_uint(fmaf(lo, cu, 12582912.0f));
  const uint32_t h = __float_as_uint(fmaf(hi, cu, 12563648.0f));
  return (h << 16) + l;
}

//: an entry word's coefficient times the unit (bf16, its low half) as fp32
__device__ __forceinline__ float coef(uint32_t word) { return __uint_as_float(word << 16); }

//: FORM 0'S STEP, 16 tokens on M: their read rows by two `ldmatrix.x4` (an address one add and an XOR), 8 MMAs against
//: the digit's state, each output pair exact and added at its token's panel base (one add a token)
template <int OUT>
__device__ __forceinline__ void step_m(uint32_t sm, uint2 e, const Lane& c, const uint32_t (&f)[16], float& sink) {
  const uint32_t ea = __shfl_sync(0xffffffffu, e.x, c.src_a) & 0xffffu;
  const uint32_t eg = __shfl_sync(0xffffffffu, e.y, c.src_g), eh = __shfl_sync(0xffffffffu, e.y, c.src_h);
  const uint32_t at = (sm + ea) ^ c.chunk;
  float acc[4][4] = {};
#pragma unroll
  for (int kb = 0; kb < 2; ++kb) {
    uint32_t a[4];
    lab::ldsm(a, at ^ (32u * kb));
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) lab::mma_bf16_f32(acc[nt], a, f[(kb * 4 + nt) * 2], f[(kb * 4 + nt) * 2 + 1]);
  }
  const uint32_t bg = c.pad + (eg >> 16), bh = c.pad + (eh >> 16);
#pragma unroll
  for (int nt = 0; nt < 4; ++nt) {
    if constexpr (OUT == 0) {
      sink += acc[nt][0] + acc[nt][1] + acc[nt][2] + acc[nt][3];
    } else {
      panel_red(bg + 16 * nt, s16_exact(acc[nt][0], acc[nt][1], coef(eg)));
      panel_red(bh + 16 * nt, s16_exact(acc[nt][2], acc[nt][3], coef(eh)));
    }
  }
}

//: FORM 1'S TILE, 8 tokens on N: their read rows (B: K = lists, N = tokens) by one `ldmatrix.x4`, 4 MMAs against the
//: digit's state (A), each lane's two tokens' outputs exact and added at their panel bases
template <int OUT>
__device__ __forceinline__ void tile_n(uint32_t sm, uint2 e, const Lane& c, const uint32_t (&f)[16], float& sink) {
  const uint32_t eb = __shfl_sync(0xffffffffu, e.x, c.src_a) & 0xffffu;
  const uint32_t e0 = __shfl_sync(0xffffffffu, e.y, c.src_g), e1 = __shfl_sync(0xffffffffu, e.y, c.src_h);
  uint32_t b[4];
  lab::ldsm(b, (sm + eb) ^ c.chunk);
  float acc[2][4] = {};
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int kb = 0; kb < 2; ++kb) {
      const uint32_t a[4] = {f[(m * 2 + kb) * 4], f[(m * 2 + kb) * 4 + 1], f[(m * 2 + kb) * 4 + 2],
                             f[(m * 2 + kb) * 4 + 3]};
      lab::mma_bf16_f32(acc[m], a, b[2 * kb], b[2 * kb + 1]);
    }
  const uint32_t b0 = c.pad + (e0 >> 16), b1 = c.pad + (e1 >> 16);
#pragma unroll
  for (int m = 0; m < 2; ++m) {
    if constexpr (OUT == 0) {
      sink += acc[m][0] + acc[m][1] + acc[m][2] + acc[m][3];
    } else {
      panel_red(b0 + 32 * m, s16_exact(acc[m][0], acc[m][2], coef(e0)));
      panel_red(b1 + 32 * m, s16_exact(acc[m][1], acc[m][3], coef(e1)));
    }
  }
}

//: THE DENSE CELL'S STEP `s` of a pass (every slot a token): a lane's read row and panel row its bases plus
//: compile-time offsets; the coefficient the unit
template <int OUT>
__device__ __forceinline__ void step_dense(uint32_t a0, uint32_t p0, const uint32_t (&f)[16], float& sink, int s) {
  float acc[4][4] = {};
#pragma unroll
  for (int kb = 0; kb < 2; ++kb) {
    uint32_t a[4];
    lab::ldsm(a, (a0 ^ (32u * kb)) + 1024u * s);
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) lab::mma_bf16_f32(acc[nt], a, f[(kb * 4 + nt) * 2], f[(kb * 4 + nt) * 2 + 1]);
  }
#pragma unroll
  for (int nt = 0; nt < 4; ++nt) {
    if constexpr (OUT == 0) {
      sink += acc[nt][0] + acc[nt][1] + acc[nt][2] + acc[nt][3];
    } else {
      panel_red(p0 + 1280u * s + 16 * nt, s16_exact(acc[nt][0], acc[nt][1], kUnit));
      panel_red(p0 + 1280u * s + 640 + 16 * nt, s16_exact(acc[nt][2], acc[nt][3], kUnit));
    }
  }
}

//: THE DENSE CELL'S TILE `s` (tokens on N): 8 consecutive slots' read rows by one `ldmatrix.x4`, a lane's read row
//: and panel rows its bases plus compile-time offsets; the coefficient the unit
template <int OUT>
__device__ __forceinline__ void tile_dense(uint32_t b0, uint32_t p0, const uint32_t (&f)[16], float& sink, int s) {
  uint32_t b[4];
  lab::ldsm(b, b0 + 512u * s);
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
      panel_red(p0 + 640u * s + 32 * m, s16_exact(acc[m][0], acc[m][2], kUnit));
      panel_red(p0 + 640u * s + 80 + 32 * m, s16_exact(acc[m][1], acc[m][3], kUnit));
    }
  }
}

template <int FORM, int OUT, int DENSE, int BAR>
__global__ __launch_bounds__(kThreads, 1) void read(Lists ls, long long* times, int* sinks) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int warp = lab::warp_id(), lane = lab::lane_id();
  const int boxes = ls.width / kDigits, box = static_cast<int>(blockIdx.x) % boxes;
  const uint32_t sm = lab::smem(smem);
  //: the image's even channels -0.0625 and odd +0.0625, the pool's lists +0.0625 (bf16), the rest zero
  uint4* chunks = reinterpret_cast<uint4*>(smem);
  for (int k = static_cast<int>(threadIdx.x); k < kBytes / 16; k += kThreads) {
    const uint32_t v = k < kPool / 16 ? 0x3d80bd80u : k < kPads / 16 ? 0x3d803d80u : 0u;
    chunks[k] = make_uint4(v, v, v, v);
  }
  __syncthreads();
  const Lane c = lane_of<FORM>(sm, warp, lane);
  float sink = 0.0f;
  //: the warp's heads, a window ahead: a lane's 16-byte chunks of the entries (its first at `src`, `to` in the window's
  //: buffer)
  const uint8_t* src = ls.heads + (static_cast<int64_t>(box) * kWarps + warp) * kHeadBytes + 16 * lane;
  const int64_t src_step = static_cast<int64_t>(boxes) * kWarps * kHeadBytes;
  const uint32_t hb0 = sm + kHeads + warp * kHeadBytes, hb_step = kWarps * kHeadBytes;
  const uint32_t to0 = lab::held(hb0 + 16 * lane), he0 = lab::held(hb0 + c.entry);
  const auto heads_in = [&](uint32_t to) {
#pragma unroll
    for (int k = 0; k < kHead / 16; ++k) lab::cp_async16(to + 512 * k, src + 512 * k);
  };
  const uint16_t* count_at = ls.count + box;
  if constexpr (!DENSE) heads_in(to0);
  lab::cp_commit();
  int next = DENSE ? 2 * kSlots : *count_at;
  //: the drain of pass `p` (`rows` slots): each warp's slots, two a request, each word cleared
  const auto drain = [&](int w, int p, int rows) {
    uint32_t* words = reinterpret_cast<uint32_t*>(smem);
#pragma unroll 1
    for (int k = 0; k < (min(rows - kSlots / kWarps * warp, kSlots / kWarps) + 1) / 2; ++k) {
      const int slot = kSlots / kWarps * warp + 2 * k + (lane >> 4), word = lane & 15;
      uint32_t* cell = words + kPanel / 4 + slot * kPanelPitch + word;
      uint32_t* target = ls.out + ((static_cast<int64_t>(w) * 2 + p) * kSlots + slot) * kWords + word;
      asm volatile("red.global.add.u32 [%0], %1;" ::"l"(target), "r"(*cell) : "memory");
      *cell = 0u;
    }
    __syncthreads();
  };
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int w = 0; w < ls.windows; ++w) {
    const int count = next;
    if constexpr (DENSE) {
      const int r = lane & 7, q = lane >> 3, g = lane >> 2, t = lane & 3;
      const uint32_t a0 = (sm + kPool + in_plane(FORM == 0 ? 8 * (q & 1) + r : r, 0)) ^ c.chunk;
      const uint32_t p0 = sm + kPanel + (FORM == 0 ? 80u * g + 4u * t : 160u * t + 4u * g);
#pragma unroll 1
      for (int p = 0; p < 2; ++p) {
#pragma unroll 1
        for (int i = 0; i < 4; ++i) {
          uint32_t f[16];
          state_in<FORM>(f, c, plane_off(i));
#pragma unroll
          for (int s = 0; s < kSlots / (FORM == 0 ? 16 : 8); ++s) {
            if constexpr (FORM == 0) step_dense<OUT>(a0, p0, f, sink, s);
            if constexpr (FORM == 1) tile_dense<OUT>(a0, p0, f, sink, s);
          }
        }
        if constexpr (BAR == 1) __syncthreads();
        if constexpr (OUT == 2) drain(w, p, kSlots);
      }
      continue;
    }
    const uint32_t flip = (w & 1) * hb_step;
    //: the next window's heads and count, unconditionally: the last window's next is the inputs' zero pad. The buffer
    //: the copies overwrite was read last window by every lane
    __syncwarp();
    src += src_step;
    count_at += ls.groups;
    heads_in(to0 + (hb_step - flip));
    next = *count_at;
    lab::cp_commit();
    asm volatile("cp.async.wait_group 1;" ::: "memory");
    __syncwarp();
    const uint32_t he = he0 + flip;
    //: a digit's entry `e` past the staged ones, from its list in global memory (zero past its count, in its row since
    //: its steps end by entry 512): 0.3% of digits at alt-k4, the facts pass's past-16 debt
    const auto listed = [&](int i, int e) {
      const int64_t row = static_cast<int64_t>(w) * ls.width + box * kDigits + warp + 8 * i;
      return ls.entries[row * kListCap + e];
    };
    //: a digit's steps (or tiles) and whether any of the warp's run past the staged entries, from the high half of the
    //: lane's first entry's pool word: each branch on them a vote, so ptxas sees it warp-uniform (no convergence
    //: barrier around the collectives it guards)
    uint32_t tag[4] = {};
    const auto steps = [&](int i) { return static_cast<int>((tag[i] >> 16) & 0xffu); };
    const auto past = [&] { return __any_sync(0xffffffffu, ((tag[0] | tag[1] | tag[2] | tag[3]) >> 24) & 1u); };
    constexpr int kStaged = kHead / (FORM == 0 ? 16 : 8);
    if constexpr (FORM == 0) {
      uint32_t f[4][16];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        state_in<0>(f[i], c, plane_off(i));
        const uint2 e = lds64(he + kHead * 8 * i);
        tag[i] = e.x;
        step_m<OUT>(sm, e, c, f[i], sink);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int s = 1; s < kStaged; ++s)
          for (int k = __any_sync(0xffffffffu, steps(i) > s) ? 1 : 0; k > 0; --k)
            step_m<OUT>(sm, lds64(he + kHead * 8 * i + 128 * s), c, f[i], sink);
      for (int rare = past() ? 1 : 0; rare > 0; --rare)
#pragma unroll
        for (int i = 0; i < 4; ++i)
          for (int k = __any_sync(0xffffffffu, steps(i) > kStaged) ? 1 : 0, s = kStaged; k > 0; --k)
            do step_m<OUT>(sm, listed(i, 16 * s + (lane & 15)), c, f[i], sink);
            while (__any_sync(0xffffffffu, ++s < steps(i)));
    } else {
      uint32_t f[4][16];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        state_in<1>(f[i], c, plane_off(i));
        const uint2 e = lds64(he + kHead * 8 * i);
        tag[i] = e.x;
        tile_n<OUT>(sm, e, c, f[i], sink);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int s = 1; s < kStaged; ++s)
          for (int k = __any_sync(0xffffffffu, steps(i) > s) ? 1 : 0; k > 0; --k)
            tile_n<OUT>(sm, lds64(he + kHead * 8 * i + 64 * s), c, f[i], sink);
      for (int rare = past() ? 1 : 0; rare > 0; --rare)
#pragma unroll
        for (int i = 0; i < 4; ++i)
          for (int k = __any_sync(0xffffffffu, steps(i) > kStaged) ? 1 : 0, s = kStaged; k > 0; --k)
            do tile_n<OUT>(sm, listed(i, 8 * s + (lane & 7)), c, f[i], sink);
            while (__any_sync(0xffffffffu, ++s < steps(i)));
    }
    if constexpr (BAR == 1) __syncthreads();
    if constexpr (OUT == 2) drain(w, 0, count);
  }
  clock.stop(times);
  lab::cp_wait_all();
  lab::sink(sinks, lab::bits(sink));
  //: the check's read of the panel, after the timed loop: every word a CTA's adds left in it
  if constexpr (OUT >= 1) {
    __syncthreads();
    for (int k = static_cast<int>(threadIdx.x); k < kSlots * kWords / 4; k += kThreads)
      reinterpret_cast<uint4*>(ls.panel_out)[static_cast<int64_t>(blockIdx.x) * kSlots * kWords / 4 + k] =
          chunks[(kPanel + (k >> 2) * kPanelPitch * 4) / 16 + (k & 3)];
  }
}

struct HostLists {
  int windows = 0, width = 0, groups = 0;
  std::vector<uint16_t> slots, at, count;
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
  h.slots.resize(static_cast<size_t>(h.windows) * h.width * kListCap);
  h.at.resize(static_cast<size_t>(h.windows) * h.width * kListAt);
  h.count.resize(static_cast<size_t>(h.windows) * h.groups);
  f.read(reinterpret_cast<char*>(h.slots.data()), static_cast<std::streamsize>(h.slots.size() * 2));
  f.read(reinterpret_cast<char*>(h.at.data()), static_cast<std::streamsize>(h.at.size() * 2));
  f.read(reinterpret_cast<char*>(h.count.data()), static_cast<std::streamsize>(h.count.size() * 2));
  return h;
}

//: a file's first `bytes` on the device, then `pad` zero bytes (a window past the last, for the prefetch)
template <typename T>
T* upload(const std::string& path, size_t bytes, size_t pad = 0) {
  std::vector<uint8_t> host(bytes + pad, 0);
  std::ifstream f(path, std::ios::binary);
  f.read(reinterpret_cast<char*>(host.data()), static_cast<std::streamsize>(bytes));
  if (!f) {
    std::fprintf(stderr, "read_image: %s is short\n", path.c_str());
    std::exit(1);
  }
  T* dev = nullptr;
  LAB_CHECK(cudaMalloc(&dev, bytes + pad));
  LAB_CHECK(cudaMemcpy(dev, host.data(), bytes + pad, cudaMemcpyHostToDevice));
  return dev;
}

//: a digit's entries in its window's one pass (the inputs assert a box's union fits it)
int entries_of(const HostLists& h, int w, int d) {
  const uint16_t* a = &h.at[(static_cast<size_t>(w) * h.width + d) * kListAt];
  return a[kListAt - 1] - a[0];
}

//: THE TENSOR FLOOR a CTA-window: the busiest scheduler's two warps' MMAs (form 0: 8 a 16-token step; form 1: 4 an
//: 8-token tile) at 32 cycles (bf16, fp32 accumulate, sm_86), averaged over the boxes and windows -- each window's
//: busiest when a barrier ends every window, the busiest over all windows when warps free-run (a scheduler idle in one
//: window takes another's work)
double floor_of(const HostLists& h, int form, bool dense, bool free) {
  const int boxes = h.width / kDigits;
  double total = 0;
  for (int box = 0; box < boxes; ++box) {
    long long all[4] = {};
    for (int w = 0; w < h.windows; ++w) {
      long long mmas[kWarps] = {};
      for (int p = 0; p < (dense ? 2 : 1); ++p)
        for (int d = 0; d < kDigits; ++d) {
          const int n = dense ? kSlots : entries_of(h, w, box * kDigits + d);
          mmas[d % kWarps] += form == 0 ? (n + 15) / 16 * 8 : (n + 7) / 8 * 4;
        }
      long long busiest = 0;
      for (int s = 0; s < 4; ++s) {
        busiest = std::max(busiest, mmas[s] + mmas[s + 4]);
        all[s] += mmas[s] + mmas[s + 4];
      }
      total += free ? 0.0 : busiest * 32.0;
    }
    total += free ? *std::max_element(all, all + 4) * 32.0 : 0.0;
  }
  return total / (static_cast<double>(h.windows) * boxes);
}

//: THE CHECK's reference: a live (token, digit) add is 32 lists x 0.0625 x (-/+ 0.0625) x the coefficient, -/+ 32 a
//: channel at the unit 256 (even channels negative), times m = 1 + ((slot ^ digit) & 1) for the lists' coefficients
//: (1 dense); a panel word holds channels (2k, 2k + 1) on M and (16 (k / 8) + k % 8, + 8) on N as hi x 2^16 + lo, so a
//: slot's word is its sum of m times one add's word, mod 2^32 -- OUT 1, a CTA's panel summed over its windows; OUT 2,
//: a window-pass's drain summed over the CTAs
struct Reference {
  const HostLists* h;
  int sms;
};

struct Verdict {
  long long bad, words;
  std::string detail;
};

uint32_t one_add(int form, int k) {
  const int lo = form == 0 ? 2 * k : 16 * (k >> 3) + (k & 7), hi = form == 0 ? lo + 1 : lo + 8;
  const auto v = [](int ch) { return static_cast<uint32_t>(ch & 1 ? 32 : -32); };
  return v(hi) * 65536u + v(lo);
}

template <int FORM, int OUT, int DENSE>
Verdict verify(const Reference& ref, const Lists& ls) {
  const HostLists& h = *ref.h;
  const int boxes = h.width / kDigits, windows = h.windows;
  const auto live = [&](int w, int box, std::vector<uint32_t>& m) {
    for (int d = 0; d < kDigits; ++d) {
      if (DENSE) {
        for (int s = 0; s < kSlots; ++s) m[s] += 1;
        continue;
      }
      const size_t row = static_cast<size_t>(w) * h.width + box * kDigits + d;
      const uint16_t* at = &h.at[row * kListAt];
      for (int e = at[0]; e < at[kListAt - 1]; ++e) {
        const int slot = h.slots[row * kListCap + e];
        m[slot] += 1 + ((slot ^ d) & 1);
      }
    }
  };
  std::vector<uint32_t> got;
  long long bad = 0, words = 0;
  const auto compare = [&](const std::vector<uint32_t>& m, const uint32_t* panel) {
    for (int s = 0; s < kSlots; ++s)
      for (int k = 0; k < kWords; ++k, ++words) bad += panel[s * kWords + k] != m[s] * one_add(FORM, k);
  };
  if constexpr (OUT == 1) {
    got.resize(static_cast<size_t>(ref.sms) * kSlots * kWords);
    LAB_CHECK(cudaMemcpy(got.data(), ls.panel_out, got.size() * 4, cudaMemcpyDeviceToHost));
    for (int c = 0; c < ref.sms; ++c) {
      std::vector<uint32_t> m(kSlots, 0);
      for (int w = 0; w < windows; ++w)
        for (int p = 0; p < (DENSE ? 2 : 1); ++p) live(w, c % boxes, m);
      compare(m, &got[static_cast<size_t>(c) * kSlots * kWords]);
    }
  } else {
    got.resize(static_cast<size_t>(windows) * 2 * kSlots * kWords);
    LAB_CHECK(cudaMemcpy(got.data(), ls.out, got.size() * 4, cudaMemcpyDeviceToHost));
    for (int w = 0; w < windows; ++w)
      for (int p = 0; p < 2; ++p) {
        std::vector<uint32_t> m(kSlots, 0);
        if (DENSE || p == 0)
          for (int c = 0; c < ref.sms; ++c) live(w, c % boxes, m);
        compare(m, &got[(static_cast<size_t>(w) * 2 + p) * kSlots * kWords]);
      }
  }
  return {bad, words,
          std::to_string(words - bad) + " of " + std::to_string(words) + " panel words exact" +
              (OUT == 1 ? " (each CTA's panel)" : " (each window-pass's drain)")};
}

//: THE BUDGETS (KERNEL_STANDARDS sections 29-30), each class's instructions per MMA and the shared-memory wavefronts
//: per MMA, from the operation count at alt-k4: a read digit ~8 tokens a window (one 16-position step, 8 MMAs; ~1.41
//: 8-token tiles, ~5.6 MMAs), a warp 4 digits. TOKENS ON M -- a step: 2 read-row loads and an entry load, 3 shuffles,
//: 16 FFMAs, 8 adds; integer 14 (the read rows' mask-XOR and XOR, 2 coefficients' shifts, 2 panel bases, 8 packs). A
//: digit: 4 state loads (immediates). A window (32 MMAs a warp): the heads' copy (global) and the 3 never-executed
//: `LDS` ptxas sets before an `LDGSTS`; integer ~14 (the source and count pointers, the buffer's parity, the tags' OR
//: and test, the loop counter); select ~6 (the loop's test, the past-staged vote and its predicate); control ~8 (2
//: alignment NOPs, commit, wait, the loop's exit CALL and branch, the past-staged branch; the pass barrier). Per MMA:
//: shared load 0.97, shuffle 0.38, float 2.0, integer 2.2, select 0.19, control 0.25 (0.28 with the barrier), global
//: 0.04. TOKENS ON N -- a tile (4 MMAs): a read-row load and an entry load, 3 shuffles, 8 FFMAs, 4 adds, integer 9 (a
//: mask-XOR, 2 shifts, 2 bases, 4 packs) and ~1.3 of ptxas's never-executed `UIADD3` fillers between its dependent
//: MMAs (two accumulators, each two MMAs deep); a digit (~5.6 MMAs): 4 state loads, its tag's extraction (2), the
//: second tile's vote and branch, after every digit's first tile in one straight-line block (as M's first steps); a
//: window (~22.6 MMAs): as on M. Per MMA: shared load 1.36, shuffle 0.75, float 2.0,
//: integer 3.7, select 0.45, control 0.55 (0.6), global 0.05. DENSE (a digit 16 steps or 32 tiles a pass): a step 2
//: read-row loads, 16 FFMAs, 8 packs, 8 adds (a tile 1, 8, 4, 4), its addresses immediates; a digit 4 state loads.
//: Per MMA: shared load 0.28, float 2.0, integer 1.05 (N: its fillers too), control 0.05. AT W = 1024 (`read_w1024.cu`:
//: ~16 tokens a digit, 32 entries staged, two 16-byte copies a lane): M's digit adds a vote and its branch for its
//: second step (~12 MMAs a digit), N's three for its tiles past the first (~9.7): select 0.35 and 0.95, global 0.07.
//: Each adds the kernel's setup and the check's copy-out over its ~1000 MMAs a warp (~0.1 an MMA, mostly integer);
//: "other" (an opcode in no class) is budgeted zero.
//: WAVEFRONTS, conflict-free by the entries' coloring and placement: a step's 2 read-row loads 8, its 8 adds 8, its
//: entry load 2 (an 8-byte load serves a half-warp a wavefront); a tile's 4, 4 and 2; a digit's state 16; a window's
//: heads copy 8 a 512 bytes (`LDGSTS` counts twice a copy's 16 bytes); the init and copy-out ~0.1 an MMA; and the
//: counter's remainder no operation is named for, ~0.6 a shuffle. Per MMA at alt-k4: M 2.25 + 1.99 + 0.25 + 0.1 +
//: 0.22 = 4.8, N 2.5 + 2.84 + 0.36 + 0.1 + 0.45 = 6.25; dense 2.0 + 0.13; at W = 1024 M 2.25 + 1.4 + 0.35 + 0.1 + 0.22
//: = 4.3, N 2.5 + 1.65 + 0.42 + 0.1 + 0.45 = 5.1.
struct Budget {
  double shared_load, shuffle, float_, integer, select, control, global, wavefronts;
};

constexpr bool kWide = kSlots == 512;
constexpr Budget kBudget[4] = {
    {1.0, 0.4, 2.0, 2.5, kWide ? 0.35 : 0.2, 0.3, kWide ? 0.07 : 0.05, kWide ? 4.45 : 4.95},
    {1.4, 0.8, 2.0, 3.75, kWide ? 0.95 : 0.5, 0.6, kWide ? 0.08 : 0.06, kWide ? 5.25 : 6.4},
    {0.3, 0.0, 2.0, 1.15, 0.03, 0.08, 0.01, 2.25},
    {0.3, 0.0, 2.0, 1.5, 0.03, 0.08, 0.01, 2.25}};

template <int FORM, int OUT, int DENSE, int BAR = 1>
void variant(const lab::Bench& b, const Lists& ls, const char* lists, const Reference& ref) {
  static_assert(BAR == 1 || OUT < 2, "the drain needs the pass's barrier");
  const std::string name = lab::kernel("read_image::read", {FORM, OUT, DENSE, BAR});
  b.expect(name, "HMMA.16816.F32.BF16", ">=", FORM == 0 ? 8 : 4);
  //: a digit past its staged entries reads them from global memory inside its steps (the facts pass's past-16 debt)
  if (!DENSE) b.expect(name, "STEP_GLOBAL", ">=", 0);
  if (OUT == 0) b.exempt(name, "an isolation: the MMAs alone");
  if (OUT == 2) b.exempt(name, "a budgeted step (its OUT 1 variant) plus the pass's drain");
  if (OUT == 1) {
    const Budget& u = kBudget[DENSE ? 2 + FORM : FORM];
    b.mma(name, "hmma_bf16_f32", 16);
    b.budget(name, "mma", 1.0);
    b.budget(name, "shared load", u.shared_load);
    b.budget(name, "shared store", 0.03);
    b.budget(name, "shared atomic", 1.0);
    b.budget(name, "shuffle", u.shuffle);
    b.budget(name, "pack", 0.0);
    b.budget(name, "conversion unit", 0.0);
    b.budget(name, "float", u.float_);
    b.budget(name, "select/compare", u.select);
    b.budget(name, "integer/address", u.integer);
    b.budget(name, "control", u.control);
    b.budget(name, "global", u.global);
    b.budget(name, "other", 0.0);
    b.budget(name, "wavefronts", u.wavefronts);
  }
  if (b.expect_only) return;
  LAB_CHECK(cudaFuncSetAttribute(read<FORM, OUT, DENSE, BAR>, cudaFuncAttributeMaxDynamicSharedMemorySize, kBytes));
  const auto launch = [&] { read<FORM, OUT, DENSE, BAR><<<b.sms, kThreads, kBytes>>>(ls, b.times, b.sinks); };
  const double floor = floor_of(*ref.h, FORM, DENSE, BAR == 0), floor_m = floor_of(*ref.h, 0, DENSE, BAR == 0);
  const lab::Samples s = b.variant(launch,
                                   [&] { return b.span(kWarps) / ls.windows; });
  const char* forms[2] = {"tokens on M", "tokens on N"};
  const char* outs[3] = {"the MMAs alone", "+ panel adds", "+ panel adds + drain"};
  const std::string label = std::string(DENSE ? "dense / " : "alt-k4 / ") + forms[FORM] + " / " + outs[OUT] + lists +
                            (BAR ? "" : " / no pass barrier");
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
    launch();
    LAB_CHECK(cudaDeviceSynchronize());
    const Verdict v = verify<FORM, OUT, DENSE>(ref, ls);
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
  Lists lm{}, ln{};
  HostLists host;
  if (!b.expect_only) {
    if (dir.empty()) {
      std::fprintf(stderr, "read_image: --inputs <directory> (tools/lab.py writes it)\n");
      return 1;
    }
    host = read_lists(dir + "/read.bin");
    const size_t entries = static_cast<size_t>(host.windows) * host.width * kListCap * 8;
    const size_t heads = static_cast<size_t>(host.windows) * (host.width / kDigits) * kWarps * kHeadBytes;
    uint16_t* count = upload<uint16_t>(dir + "/read.bin", 16 + host.slots.size() * 2 + host.at.size() * 2 +
                                                              host.count.size() * 2, host.groups * 2);
    uint32_t *out = nullptr, *panel_out = nullptr;
    LAB_CHECK(cudaMalloc(&out, static_cast<size_t>(host.windows) * 2 * kSlots * kWords * 4));
    LAB_CHECK(cudaMalloc(&panel_out, static_cast<size_t>(b.sms) * kSlots * kWords * 4));
    const uint16_t* counts = count + 8 + host.slots.size() + host.at.size();
    const size_t pad = heads / host.windows;
    lm = Lists{upload<uint2>(dir + "/entries_m.bin", entries), upload<uint8_t>(dir + "/heads_m.bin", heads, pad),
               counts, host.windows, host.width, host.groups, out, panel_out};
    ln = lm;
    ln.entries = upload<uint2>(dir + "/entries_n.bin", entries);
    ln.heads = upload<uint8_t>(dir + "/heads_n.bin", heads, pad);
    std::printf("{\"lab\": \"note\", \"floor_m\": %.1f, \"floor_n\": %.1f, \"free_m\": %.1f, \"free_n\": %.1f, "
                "\"dense_m\": %.1f}\n", floor_of(host, 0, false, false), floor_of(host, 1, false, false),
                floor_of(host, 0, false, true), floor_of(host, 1, false, true), floor_of(host, 0, true, false));
  }
  const Reference ref{&host, b.sms};
  const char* lists = kSlots == 512 ? " / W 1024, a 512-slot pass" : "";
  variant<0, 0, 0>(b, lm, lists, ref);
  variant<0, 1, 0>(b, lm, lists, ref);
  variant<0, 1, 0, 0>(b, lm, lists, ref);
  variant<0, 2, 0>(b, lm, lists, ref);
  variant<1, 0, 0>(b, ln, lists, ref);
  variant<1, 1, 0>(b, ln, lists, ref);
  variant<1, 1, 0, 0>(b, ln, lists, ref);
  variant<1, 2, 0>(b, ln, lists, ref);
#if READ_IMAGE_SLOTS == 256
  variant<0, 1, 1>(b, lm, "", ref);
  variant<0, 2, 1>(b, lm, "", ref);
  variant<1, 1, 1>(b, ln, "", ref);
  variant<1, 2, 1>(b, ln, "", ref);
#endif
}
