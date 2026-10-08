// measure/lab/experiments/facts_pipeline.cu -- QUESTION: the two-stage facts pass end to end. A warp streams an item
// (the vote; on the write list level also the lists' exponents and the packed stash of live bytes) keeping its bits
// in registers, issues its next item's first block, then builds its four 32-digit groups' lists from them -- does the
// row stream keep its rate, and what does the build cost? METHOD: rows [level-side, T, H, 256] u8, each token's head
// live in k random digits; level-side 0 the write list level (exponents, stash, 8-byte records with c), 1 the read
// list level (2-byte slot lists), 2-3 voted only. Seven warps an SM (a 2 x 4 KB ring and ~4.8 KB of build scratch
// each), each taking (window, level-side, head, 128-digit span) items in order from a counter, the span innermost;
// an item's 16 blocks of 32 tokens x 128 bytes fully unrolled so each lane holds its four digits' 16 words. The build
// a group: the union and each token's slot, each digit's entries dealt evenly to the lanes (one find a lane), a walk
// totalling each (digit, pass) by residue, a segmented scan of ranks, a walk placing each entry round-robin -- a write
// entry's byte from the stash (its place from a prefix of the group's per-block counts) times the token's fp32 gains at
// the list's exponent. Bytes a cycle against STAGE1 (the build skipped) and VOTE (every level-side voted only); the
// lists' total length checked against the live count. see docs/internals/tools/lab.md#facts-pipeline
#include "lab.cuh"

#include <cuda_fp16.h>

namespace facts_pipeline {

constexpr int kWarps = 7, kThreads = kWarps * 32, kStages = 2;
constexpr int kSpan = 128, kGroups = kSpan / 32, kBlockBytes = 32 * kSpan, kWindow = 512, kBlocks = kWindow / 32;
constexpr int kTokens = 65536, kHeads = 4, kWidth = 256, kLevelSides = 4, kTokenWords = kTokens / 32;
constexpr long long kLevelBytes = static_cast<long long>(kTokens) * kHeads * kWidth;
constexpr long long kBytes = kLevelBytes * kLevelSides;
constexpr int kSpans = kWidth / kSpan, kWindows = kTokens / kWindow;
constexpr int kItems = kWindows * kLevelSides * kHeads * kSpans, kRoleItems = kWindows * kHeads * kSpans;
constexpr int kPassSlots = 128, kMaxPasses = 4, kListCap = kWindow, kListAt = 8, kCoefTop = 4;
constexpr int kUnitMin = kCoefTop - 127, kKeys = 32 * kMaxPasses, kOwned = kKeys / 32;
constexpr int kStashSegment = 32 * kWindow, kTileStride = 17;
constexpr float kListFloor = 1.17549435e-38f;

struct Params {
  const uint8_t* rows;
  const float2* fm;          // [T, H] a token's (m, F)
  uint32_t* bits;            // [level-side, H, 256 rows, token words]
  uint8_t* stash;            // [role item, group, segment]
  uint32_t* unions;          // [item, group, 16]
  uint16_t* at;              // [item, group, digit, kListAt]
  uint2* records;            // [role item, group, digit, kListCap]
  uint16_t* slots;           // [role item, group, digit, kListCap]
  int* counter;
};

//: a level-side's role: 0 the write list level, 1 the read list level, 2 voted only
__device__ __forceinline__ int role_of(int ls) { return ls < 2 ? ls : 2; }

//: A WARP'S BUILD SCRATCH: its group's bits (a row a lane, the padding word the row's scales), the union, the cells'
//: mask, the table and starts, the stash's per-block prefixes
struct alignas(16) Scratch {
  uint32_t tile[32 * kTileStride];
  uint32_t live[16];
  uint16_t bases[16];
  uint16_t cells[32];
  uint2 totals[kKeys];
  uint16_t starts[kKeys];
  uint16_t pref[kBlocks * 32];
};

__device__ __forceinline__ uint32_t mix(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA77u ^ (c + 0x165667B1u) * 0xC2B2AE3Du;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  return h ^ (h >> 12);
}

__global__ void fill(uint8_t* rows, float2* fm, int k) {
  const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < static_cast<long long>(kTokens) * kHeads)
    fm[i] = make_float2(1e-3f * (1 + mix(static_cast<uint32_t>(i), 1u, 2u) % 1000u),
                        1e-3f * (1 + mix(static_cast<uint32_t>(i), 3u, 4u) % 1000u));
  if (i >= static_cast<long long>(kLevelSides) * kTokens * kHeads) return;
  for (int j = 0; j < k; ++j) {
    const uint32_t h = mix(static_cast<uint32_t>(i), static_cast<uint32_t>(j), 0x5eedu);
    rows[i * kWidth + (h & 255u)] = static_cast<uint8_t>(1u + (h >> 8) % 255u);
  }
}

__global__ void count_live(const uint8_t* rows, unsigned long long* live) {
  unsigned long long n = 0;
  const long long total = 2 * kLevelBytes;
  for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += static_cast<long long>(gridDim.x) * blockDim.x)
    n += rows[i] != 0;
  atomicAdd(live, n);
}

__global__ void count_lists(const uint16_t* at, int lists, unsigned long long* total) {
  unsigned long long n = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < lists; i += gridDim.x * blockDim.x)
    n += at[static_cast<long long>(i) * kListAt + kMaxPasses];
  atomicAdd(total, n);
}

__device__ __forceinline__ int ring_row(int t) { return 4 * (t & 7) + (t >> 3); }

template <uint32_t Sel>
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "n"(Sel));
  return d;
}

__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }

template <int N>
__device__ __forceinline__ void wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

__device__ __forceinline__ void issue(const uint8_t* block, uint32_t stage, int lane) {
#pragma unroll
  for (int q = 0; q < kSpan / 16; ++q) {
    const int token = 4 * q + (lane >> 3), part = lane & 7;
    const uint32_t dst = stage + (part >> 1) * 1024u + ring_row(token) * 32u + 16u * (part & 1);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst),
                 "l"(block + static_cast<long long>(token) * kHeads * kWidth + 16 * part));
  }
}

__device__ __forceinline__ int ceil_exponent(float x) {
  const uint32_t bits = __float_as_uint(fmaxf(x, kListFloor));
  return static_cast<int>(bits >> 23) - 127 + ((bits & 0x7fffffu) != 0u);
}

__device__ __forceinline__ float pow2(int e) { return __uint_as_float(static_cast<uint32_t>(127 + e) << 23); }

__device__ __forceinline__ int round_robin(uint32_t lo, uint32_t hi, int j, int r) {
  const uint32_t jj = 0x01010101u * static_cast<uint32_t>(j);
  const int before = static_cast<int>(__dp4a(__vminu4(lo, jj), 0x01010101u, __dp4a(__vminu4(hi, jj), 0x01010101u, 0u)));
  const uint32_t gl = __vcmpgtu4(lo, jj) & 0x01010101u, gh = __vcmpgtu4(hi, jj) & 0x01010101u;
  const uint32_t below_lo = r >= 4 ? 0xffffffffu : (1u << (8 * r)) - 1u;
  const uint32_t below_hi = r >= 4 ? (1u << (8 * (r - 4))) - 1u : 0u;
  return before + __popc(gl & below_lo) + __popc(gh & below_hi);
}

__device__ __forceinline__ int nth_bit(uint32_t w, int m) {
  int pos = 0;
#pragma unroll
  for (int s = 16; s > 0; s >>= 1) {
    const int c = __popc(w & ((1u << s) - 1u));
    const bool up = m >= c;
    m = up ? m - c : m;
    pos = up ? pos + s : pos;
    w = up ? w >> s : w;
  }
  return pos;
}

//: a row's digit within its group: lane `(tk, dq)` voted digit `4 dq + tk`
__device__ __forceinline__ int digit_of(int row) { return 4 * (row & 7) + (row >> 3); }

//: ONE GROUP'S LISTS from the tile (row `L` lane L's digit's 16 words), its entries dealt evenly to the lanes
template <bool kWrite>
__device__ __forceinline__ void build(const Params& p, Scratch& sc, int lane, int group, int item, int ridx, int window,
                                   int head) {
  uint32_t un = 0u;
#pragma unroll
  for (int j = 0; j < kBlocks; ++j) {
    const uint32_t u = __reduce_or_sync(0xffffffffu, sc.tile[lane * kTileStride + j]);
    un = lane == j ? u : un;
  }
  uint32_t ub = __popc(un);
#pragma unroll
  for (int off = 1; off < kBlocks; off <<= 1) {
    const uint32_t x = __shfl_up_sync(0xffffffffu, ub, off);
    ub += lane >= off ? x : 0u;
  }
  ub -= __popc(un);
  if (lane < kBlocks) {
    sc.live[lane] = un;
    sc.bases[lane] = static_cast<uint16_t>(ub);
    p.unions[(static_cast<long long>(item) * kGroups + group) * kBlocks + lane] = un;
  }
  if constexpr (kWrite) {
    //: the stash's places: block j's lanes in order, after the blocks before
    int block_base = 0;
#pragma unroll
    for (int j = 0; j < kBlocks; ++j) {
      const int n = __popc(sc.tile[lane * kTileStride + j]);
      int x = n;
#pragma unroll
      for (int off = 1; off < 32; off <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, off);
        x += lane >= off ? y : 0;
      }
      sc.pref[j * 32 + lane] = static_cast<uint16_t>(block_base + x - n);
      block_base += __shfl_sync(0xffffffffu, x, 31);
    }
  }
  uint32_t pre[4];
  int gstart[4], n = 0;
  uint32_t nonempty = 0u;
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    gstart[g] = n;
    pre[g] = 0u;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      pre[g] |= static_cast<uint32_t>(n - gstart[g]) << (8 * i);
      const int c = __popc(sc.tile[lane * kTileStride + 4 * g + i]);
      n += c;
      nonempty |= c != 0 ? 1u << (4 * g + i) : 0u;
    }
  }
  sc.cells[lane] = static_cast<uint16_t>(nonempty);
  int first = n;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const int x = __shfl_up_sync(0xffffffffu, first, off);
    first += lane >= off ? x : 0;
  }
  const int total = __shfl_sync(0xffffffffu, first, 31);
  first -= n;
#pragma unroll
  for (int h = 0; h < kOwned; ++h) sc.totals[32 * h + lane] = make_uint2(0u, 0u);
  __syncwarp();

  const int share = lane * total >> 5, count = ((lane + 1) * total >> 5) - share;
  int c0 = 0;
  uint32_t bits0 = 0u, slot0 = 0u;
  if (total > 0) {
    const int g = min(share, total - 1);
    int o = 0;
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
      const int x = __shfl_sync(0xffffffffu, first, o + s);
      o = x <= g ? o + s : o;
    }
    int m = g - __shfl_sync(0xffffffffu, first, o), k = 0, at = 0;
    uint32_t sums = __shfl_sync(0xffffffffu, pre[0], o);
#pragma unroll
    for (int gg = 1; gg < 4; ++gg) {
      const int gs = __shfl_sync(0xffffffffu, gstart[gg], o);
      const uint32_t ps = __shfl_sync(0xffffffffu, pre[gg], o);
      sums = m >= gs ? ps : sums;
      k = m >= gs ? 4 * gg : k;
      at = m >= gs ? gs : at;
    }
    m -= at;
    const int w = __popc((((static_cast<uint32_t>(m) * 0x01010101u) | 0x80808080u) - sums) & 0x80808080u) - 1;
    m -= static_cast<int>((sums >> (8 * w)) & 0xffu);
    c0 = o * kBlocks + k + w;
    const uint32_t word = sc.tile[(c0 >> 4) * kTileStride + (c0 & 15)];
    const int t0 = nth_bit(word, m);
    bits0 = word & (0xffffffffu << t0);
    slot0 = sc.bases[c0 & 15] + __popc(sc.live[c0 & 15] & ((1u << t0) - 1u));
  }
  __syncwarp();
  const int key0 = (c0 >> 4) * kMaxPasses + static_cast<int>(slot0) / kPassSlots;
  const uint32_t* const cell_mask = reinterpret_cast<const uint32_t*>(sc.cells);
  const auto walk = [&](auto&& visit) {
    int c = c0;
    uint32_t bits = bits0, word = sc.tile[(c0 >> 4) * kTileStride + (c0 & 15)], u = sc.live[c0 & 15];
    uint32_t b = sc.bases[c0 & 15];
#pragma unroll 1
    for (int e = 0; e < count; ++e) {
      if (bits == 0u) {
        int x = c + 1;
        uint32_t mw = cell_mask[x >> 5] & (0xffffffffu << (x & 31));
        while (mw == 0u) {
          x = (x | 31) + 1;
          mw = cell_mask[x >> 5];
        }
        c = (x & ~31) + __ffs(mw) - 1;
        bits = word = sc.tile[(c >> 4) * kTileStride + (c & 15)];
        u = sc.live[c & 15];
        b = sc.bases[c & 15];
      }
      const int t = __ffs(bits) - 1;
      bits &= bits - 1u;
      visit(c, t, b + __popc(u & ((1u << t) - 1u)), word);
    }
  };

  int cur = key0;
  bool many = false;
  uint32_t run_lo = 0u, run_hi = 0u;
  walk([&](int c, int, uint32_t slot, uint32_t) {
    const int key = (c >> 4) * kMaxPasses + static_cast<int>(slot) / kPassSlots, r = static_cast<int>(slot & 7u);
    const uint32_t inc = 1u << (8 * (r & 3));
    atomicAdd(reinterpret_cast<uint32_t*>(sc.totals + key) + (r >> 2), inc);
    const bool same = key == cur;
    many |= !same;
    cur = key;
    run_lo = (same ? run_lo : 0u) + (r < 4 ? inc : 0u);
    run_hi = (same ? run_hi : 0u) + (r < 4 ? 0u : inc);
  });
  const int cur_before = __shfl_up_sync(0xffffffffu, cur, 1);
  uint32_t v_lo = run_lo, v_hi = run_hi;
  int head_flag = many || lane == 0 || cur_before != cur;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const uint32_t xl = __shfl_up_sync(0xffffffffu, v_lo, off), xh = __shfl_up_sync(0xffffffffu, v_hi, off);
    const int xf = __shfl_up_sync(0xffffffffu, head_flag, off);
    if (lane >= off) {
      v_lo += head_flag ? 0u : xl;
      v_hi += head_flag ? 0u : xh;
      head_flag |= xf;
    }
  }
  const bool carried = lane > 0 && cur_before == key0;
  const uint32_t before_lo = __shfl_up_sync(0xffffffffu, v_lo, 1), before_hi = __shfl_up_sync(0xffffffffu, v_hi, 1);
  __syncwarp();
  const long long lists = (static_cast<long long>(item) * kGroups + group) * 32;
#pragma unroll
  for (int h = 0; h < kOwned; ++h) {
    const int key = 32 * h + lane, pass = key % kMaxPasses;
    const uint2 c = sc.totals[key];
    const int s = static_cast<int>(__dp4a(c.x, 0x01010101u, __dp4a(c.y, 0x01010101u, 0u)));
    int incl = s;
#pragma unroll
    for (int off = 1; off < kMaxPasses; off <<= 1) {
      const int x = __shfl_up_sync(0xffffffffu, incl, off);
      incl += pass >= off ? x : 0;
    }
    const int end = __shfl_sync(0xffffffffu, incl, lane | (kMaxPasses - 1));
    sc.starts[key] = static_cast<uint16_t>(incl - s);
    uint16_t* const at = p.at + (lists + digit_of(key / kMaxPasses)) * kListAt;
    at[pass] = static_cast<uint16_t>(incl - s);
    at[kMaxPasses + pass] = static_cast<uint16_t>(end);
  }
  __syncwarp();

  cur = key0;
  run_lo = carried ? before_lo : 0u;
  run_hi = carried ? before_hi : 0u;
  const long long role_lists = (static_cast<long long>(ridx) * kGroups + group) * 32;
  const uint8_t* const stash = p.stash + (static_cast<long long>(ridx) * kGroups + group) * kStashSegment;
  walk([&](int c, int t, uint32_t slot, uint32_t word) {
    const int row = c >> 4, j = c & 15;
    const int key = row * kMaxPasses + static_cast<int>(slot) / kPassSlots, r = static_cast<int>(slot & 7u);
    const uint2 tc = sc.totals[key];
    const bool same = key == cur;
    cur = key;
    run_lo = same ? run_lo : 0u;
    run_hi = same ? run_hi : 0u;
    const int rank = static_cast<int>(((r < 4 ? run_lo : run_hi) >> (8 * (r & 3))) & 0xffu);
    const uint32_t inc = 1u << (8 * (r & 3));
    run_lo += r < 4 ? inc : 0u;
    run_hi += r < 4 ? 0u : inc;
    const long long at = (role_lists + digit_of(row)) * kListCap + sc.starts[key] + round_robin(tc.x, tc.y, rank, r);
    if constexpr (kWrite) {
      const float byte = static_cast<float>(stash[sc.pref[j * 32 + row] + __popc(word & ((1u << t) - 1u))]);
      const float2 g = p.fm[static_cast<long long>(window * kWindow + 32 * j + t) * kHeads + head];
      const uint32_t sc_word = sc.tile[row * kTileStride + 16];
      const float sm = pow2(static_cast<int>(sc_word & 0xffu) - 128), sv = pow2(static_cast<int>(sc_word >> 8) - 128);
      p.records[at] = make_uint2(slot, __half_as_ushort(__float2half_rn(byte * g.y * sv))
                                           | static_cast<uint32_t>(__half_as_ushort(__float2half_rn(byte * g.x * sm)))
                                                 << 16);
    } else {
      p.slots[at] = static_cast<uint16_t>(slot);
    }
  });
}

//: ONE ITEM: its 16 blocks streamed (block b + 1, or the next item's first, issued before block b is read), each
//: block's four sub-words voted into the lane's registers; the write list level also maxes each digit's b x gains and
//: appends its live bytes to the group's stash; then the bits out and, on a list level, each group's lists
template <int ROLE, bool kBuild>
__device__ __forceinline__ void run_item(const Params& p, int item, int next, int& seq, const uint8_t* ring,
                                         uint32_t ring_at, Scratch& sc, int lane) {
  const int tk = lane >> 3, dq = lane & 7;
  const auto decode = [](int it, int& span, int& head, int& ls, int& window) {
    span = it % kSpans;
    head = (it / kSpans) % kHeads;
    ls = (it / kSpans / kHeads) % kLevelSides;
    window = it / kSpans / kHeads / kLevelSides;
  };
  const auto source = [&](int it, int b) {
    int span, head, ls, window;
    decode(it, span, head, ls, window);
    return p.rows + ls * kLevelBytes + (static_cast<long long>(window * kWindow + 32 * b) * kHeads + head) * kWidth
           + span * kSpan;
  };
  int span, head, ls, window;
  decode(item, span, head, ls, window);
  const int ridx = (window * kHeads + head) * kSpans + span;
  uint32_t words[kGroups][kBlocks];
  float mx[kGroups][4], fx[kGroups][4];
  int written[kGroups];
#pragma unroll
  for (int s = 0; s < kGroups; ++s) {
    written[s] = 0;
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) mx[s][jj] = fx[s][jj] = 0.0f;
  }
#pragma unroll
  for (int b = 0; b < kBlocks; ++b) {
    const uint32_t ahead = ring_at + ((seq + 1) % kStages) * kBlockBytes;
    if (b + 1 < kBlocks)
      issue(source(item, b + 1), ahead, lane);
    else if (next < kItems)
      issue(source(next, 0), ahead, lane);
    commit();
    wait<1>();
    __syncwarp();
    const uint8_t* const stage = ring + (seq % kStages) * kBlockBytes;
    float2 g[8];
    if constexpr (ROLE == 0) {
#pragma unroll
      for (int i = 0; i < 8; ++i)
        g[i] = __ldg(p.fm + static_cast<long long>(window * kWindow + 32 * b + 8 * tk + i) * kHeads + head);
    }
#pragma unroll
    for (int s = 0; s < kGroups; ++s) {
      const uint8_t* const slot = stage + s * 1024;
      uint32_t flags = 0u;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const uint32_t x = *reinterpret_cast<const uint32_t*>(slot + (4 * i + tk) * 32 + 4 * dq);
        flags |= ((((x & 0x7f7f7f7fu) + 0x7f7f7f7fu) | x) & 0x80808080u) >> (7 - i);
        if constexpr (ROLE == 0) {
#pragma unroll
          for (int jj = 0; jj < 4; ++jj) {
            const float bf = static_cast<float>((x >> (8 * jj)) & 0xffu);
            mx[s][jj] = fmaxf(mx[s][jj], bf * g[i].x);
            fx[s][jj] = fmaxf(fx[s][jj], bf * g[i].y);
          }
        }
      }
      uint32_t y = __shfl_xor_sync(0xffffffffu, flags, 16);
      flags = (tk & 2) ? prmt<0x3276u>(flags, y) : prmt<0x5410u>(flags, y);
      y = __shfl_xor_sync(0xffffffffu, flags, 8);
      flags = (tk & 1) ? prmt<0x3715u>(flags, y) : prmt<0x6240u>(flags, y);
      words[s][b] = flags;
      if constexpr (ROLE == 0) {
        const int digit = 4 * dq + tk, n = __popc(flags);
        int first = n;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
          const int x = __shfl_up_sync(0xffffffffu, first, off);
          first += lane >= off ? x : 0;
        }
        const int total = __shfl_sync(0xffffffffu, first, 31);
        uint8_t* out = p.stash + (static_cast<long long>(ridx) * kGroups + s) * kStashSegment + written[s] + first - n;
        uint32_t live = flags;
#pragma unroll 1
        while (live != 0u) {
          const int tok = __ffs(live) - 1;
          live &= live - 1u;
          *out++ = slot[ring_row(tok) * 32 + digit];
        }
        written[s] += total;
      }
    }
    __syncwarp();
    ++seq;
  }
  //: the bits out, a row's window of words one 64-byte run
#pragma unroll
  for (int s = 0; s < kGroups; ++s) {
    const long long row = (static_cast<long long>(ls) * kHeads + head) * kWidth + span * kSpan + 32 * s + 4 * dq + tk;
    uint4* const out = reinterpret_cast<uint4*>(p.bits + row * kTokenWords + window * kBlocks);
#pragma unroll
    for (int q = 0; q < 4; ++q)
      out[q] = make_uint4(words[s][4 * q], words[s][4 * q + 1], words[s][4 * q + 2], words[s][4 * q + 3]);
  }
  if constexpr (ROLE <= 1 && kBuild) {
#pragma unroll
    for (int s = 0; s < kGroups; ++s) {
      uint32_t scales = 0u;
      if constexpr (ROLE == 0) {
        //: each digit's maxima over the four token blocks' lanes, the lane taking its own digit's
        float m = 0.0f, f = 0.0f;
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
          float a = mx[s][jj], c = fx[s][jj];
          a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 8));
          a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 16));
          c = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 8));
          c = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 16));
          m = jj == tk ? a : m;
          f = jj == tk ? c : f;
        }
        const int em = kCoefTop - max(ceil_exponent(m), kUnitMin), ev = kCoefTop - max(ceil_exponent(f), kUnitMin);
        scales = static_cast<uint32_t>(em + 128) | static_cast<uint32_t>(ev + 128) << 8;
      }
#pragma unroll
      for (int j = 0; j < kBlocks; ++j) sc.tile[lane * kTileStride + j] = words[s][j];
      sc.tile[lane * kTileStride + 16] = scales;
      __syncwarp();
      build<ROLE == 0>(p, sc, lane, s, item, ridx, window, head);
      __syncwarp();
    }
  }
}

template <int MODE>
__global__ void __launch_bounds__(kThreads, 1) pipeline(Params p, long long* t) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int lane = lab::lane_id(), warp = lab::warp_id();
  uint8_t* const ring = smem + warp * (kStages * kBlockBytes + static_cast<int>(sizeof(Scratch)));
  Scratch& sc = *reinterpret_cast<Scratch*>(ring + kStages * kBlockBytes);
  const uint32_t ring_at = lab::smem(ring);
  const auto take = [&] {
    int item = 0;
    if (lane == 0) item = atomicAdd(p.counter, 1);
    return __shfl_sync(0xffffffffu, item, 0);
  };
  lab::WarpSpan clock;
  clock.start();
  int item = take(), next = take(), seq = 0;
  if (item < kItems) {
    const int span = item % kSpans, head = (item / kSpans) % kHeads, ls = (item / kSpans / kHeads) % kLevelSides;
    const int window = item / kSpans / kHeads / kLevelSides;
    issue(p.rows + ls * kLevelBytes + (static_cast<long long>(window * kWindow) * kHeads + head) * kWidth
              + span * kSpan,
          ring_at, lane);
    commit();
  }
#pragma unroll 1
  for (; item < kItems; item = next, next = take()) {
    const int role = MODE == 2 ? 2 : role_of((item / kSpans / kHeads) % kLevelSides);
    if (role == 0)
      run_item<0, MODE == 0>(p, item, next, seq, ring, ring_at, sc, lane);
    else if (role == 1)
      run_item<1, MODE == 0>(p, item, next, seq, ring, ring_at, sc, lane);
    else
      run_item<2, false>(p, item, next, seq, ring, ring_at, sc, lane);
  }
  clock.stop(t);
}

const char* kModeName[3] = {"two stages", "stage 1 alone", "voted only"};

template <int MODE, int K>
void variant(const lab::Bench& b, Params p, unsigned long long* scratch) {
  const std::string name = lab::kernel("pipeline", {MODE});
  b.expect(name, "LDGSTS", ">=", 8);
  if (b.expect_only) return;
  LAB_CHECK(cudaMemset(const_cast<uint8_t*>(p.rows), 0, kBytes));
  const long long heads = static_cast<long long>(kLevelSides) * kTokens * kHeads;
  fill<<<static_cast<int>((heads + 255) / 256), 256>>>(const_cast<uint8_t*>(p.rows), const_cast<float2*>(p.fm), K);
  LAB_CHECK(cudaDeviceSynchronize());
  const int bytes = kWarps * (kStages * kBlockBytes + static_cast<int>(sizeof(Scratch)));
  LAB_CHECK(cudaFuncSetAttribute(pipeline<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  const lab::Samples s = b.variant(
      [&] {
        LAB_CHECK(cudaMemset(p.counter, 0, sizeof(int)));
        pipeline<MODE><<<b.sms, kThreads, bytes>>>(p, b.times);
      },
      [&] { return static_cast<double>(kBytes) / b.span(kWarps); });
  b.record(std::string(kModeName[MODE]) + " / k " + std::to_string(K), "bytes a cycle", s);
  if (MODE == 0) {
    //: every live (token, digit) of the two list levels is one entry of its list
    LAB_CHECK(cudaMemset(scratch, 0, 2 * sizeof(unsigned long long)));
    count_live<<<4 * b.sms, 256>>>(p.rows, scratch);
    count_lists<<<4 * b.sms, 256>>>(p.at, kItems * kGroups * 32, scratch + 1);
    unsigned long long h[2];
    LAB_CHECK(cudaMemcpy(h, scratch, sizeof(h), cudaMemcpyDeviceToHost));
    if (h[0] != h[1]) {
      std::fprintf(stderr, "facts_pipeline: %llu live entries on the list levels, %llu listed\n", h[0], h[1]);
      std::exit(1);
    }
  }
}

}  // namespace facts_pipeline

int main(int argc, char** argv) {
  using namespace facts_pipeline;
  lab::Bench b(argc, argv);
  Params p{};
  unsigned long long* scratch = nullptr;
  if (!b.expect_only) {
    uint8_t* rows = nullptr;
    float2* fm = nullptr;
    LAB_CHECK(cudaMalloc(&rows, kBytes));
    LAB_CHECK(cudaMalloc(&fm, sizeof(float2) * kTokens * kHeads));
    p.rows = rows;
    p.fm = fm;
    LAB_CHECK(cudaMalloc(&p.bits, kBytes / 8));
    LAB_CHECK(cudaMalloc(&p.stash, static_cast<size_t>(kRoleItems) * kGroups * kStashSegment));
    LAB_CHECK(cudaMalloc(&p.unions, sizeof(uint32_t) * kItems * kGroups * kBlocks));
    LAB_CHECK(cudaMalloc(&p.at, sizeof(uint16_t) * kItems * kGroups * 32 * kListAt));
    LAB_CHECK(cudaMemset(p.at, 0, sizeof(uint16_t) * kItems * kGroups * 32 * kListAt));
    LAB_CHECK(cudaMalloc(&p.records, sizeof(uint2) * kRoleItems * kGroups * 32 * kListCap));
    LAB_CHECK(cudaMalloc(&p.slots, sizeof(uint16_t) * kRoleItems * kGroups * 32 * kListCap));
    LAB_CHECK(cudaMalloc(&p.counter, sizeof(int)));
    LAB_CHECK(cudaMalloc(&scratch, 2 * sizeof(unsigned long long)));
  }
  variant<2, 4>(b, p, scratch);
  variant<1, 4>(b, p, scratch);
  variant<0, 4>(b, p, scratch);
  variant<2, 16>(b, p, scratch);
  variant<1, 16>(b, p, scratch);
  variant<0, 16>(b, p, scratch);
}
