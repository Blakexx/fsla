// measure/lab/experiments/decode_ring.cu -- QUESTION: what form of decode's walk holds the card's HBM rate on the
// state's pages? A step's bytes are the touched pages' alone, scattered over a plane far larger than the cache, so the
// walk's form is decided by how many bytes it holds in flight and what it costs a byte: (REG) a warp loads its
// position's pieces into registers and computes from them (the walk as it is built today); (BULK) a ring of slots a
// warp in shared memory, filled by bulk copies (`cp.async.bulk`, a slot's mbarrier counting its bytes) a position
// ahead of the ring's depth; (PIECES) the same ring filled by 16-byte `cp.async` copies a lane, a warp's copy groups
// its completion; (PRODUCER) one warp filling fifteen consumer warps' rings by bulk copies, a full and an empty
// barrier a slot. Each walks five PATTERNS of a step's touched leaves: WRITE_ALL (an atom written whole: its page read,
// every leaf deposited, re-rounded and stored, then read), READ_ALL (an atom's sixteen leaves read only: its hi plane,
// exponents and masses), WRITE_ONE and READ_ONE (one leaf an atom, scattered, written or read only), and RECORD (READ_ONE
// on a page laid out with a leaf's exponents and mass beside its hi line, the layout a scattered read would want), at
// G = 1 and 8 read heads a leaf's accumulators and d_v 64 and 32.
// METHOD: a CTA an SM of 512 threads, the plane 2^21 pages at d_v 64 (2^22 at 32: over 6.6 GB). A POSITION is one atom
// (the ALL patterns) or one atom a lane group of d_v / 8 lanes (the ONE patterns); position `q` of CTA `b` is the
// CTA's items `(b Q + q) L + k`, each item's atom a bijection of its index over the plane (an odd multiplier), so a
// launch touches each atom at most once and every form walks the same atoms. The plane holds small integers (an s16 hi
// at exponent -8, a zero lo, a mass of 1 to 8) and the weights are 1 or 2, so every sum is exact and order-free: the
// check sums every thread's accumulators against the host's, and compares every page CTA 0 wrote byte for byte. A row
// is GB/s the device, the bytes the pattern moves (its sectors) over the launch's time by events.
// TIMELINE: a ring warp's trip issues position `p`'s copies into slot `p % K`, waits position `p - K + 1`'s slot (its
// mbarrier phase, or its copy group with `K - 1` groups after it), computes it from shared memory (every live leaf's G
// heads' reads, and for a written leaf its deposit, its re-rounding and its stores from registers) and syncs the warp,
// the slot's issue edge, before the next trip's copy into it: K - 1 positions in flight a warp. REG's trip loads its
// position into registers and computes it: one position in flight a warp. PRODUCER's warp 0 waits a slot's empty phase,
// states its bytes and issues its copies; a consumer waits its full phase, computes and arrives on its empty barrier.
// Live: G x 9 accumulators a lane, a leaf's hi (and lo) words, REG a position's pieces.
// see docs/internals/tools/lab.md#decode-ring
#include "lab.cuh"
#include "common/state_page.cuh"

#include <cmath>

namespace decode_ring {

constexpr int kWarps = 16, kThreads = 32 * kWarps;
//: the dynamic shared memory a CTA reserves on the H100 (227 KB), and what the walk's tables take of it
constexpr uint32_t kSmem = 232448, kTables = 4096;
enum Form { kReg, kBulk, kPieces, kProducer };
enum Pattern { kWriteAll, kReadAll, kWriteOne, kReadOne, kRecord };
//: a phase's site in a warp's timeline: above every barrier's
constexpr uint32_t kPhases = 0xc000u, kEntry = 0, kExit = 1, kCompute = 2, kLoad = 3;

//: A LEAF'S GEOMETRY at value width DV: its lanes, a warp's leaves a trip, its pieces' bytes and a RECORD's stride
template <int DV>
struct Geo {
  using Page = fsla::state_page::Blocks<DV>;
  static constexpr int kLanes = DV / 8, kGroups = 32 / kLanes, kHiLine = 2 * DV, kLoLine = DV, kExpLeaf = DV / 32;
  //: a record: the hi line, the leaf's exponents at `kHiLine`, its mass at `kHiLine + 4`, padded to whole sectors
  static constexpr int kRecord = DV == 64 ? 160 : 96;
};

//: A PATTERN'S WORK: the atoms a position, the trips a position (a trip a leaf a lane group), the slot's bytes, the bytes
//: a position moves from HBM and to it, and the positions a CTA walks
template <int PAT, int DV>
struct Work {
  using G = Geo<DV>;
  using Page = typename G::Page;
  static constexpr bool kOne = PAT >= kWriteOne, kWrite = PAT == kWriteAll || PAT == kWriteOne;
  static constexpr int kAtoms = kOne ? G::kGroups : 1, kTrips = kOne ? 1 : 16 / G::kGroups;
  //: a lane group's piece of a slot (a ONE pattern): its bytes, and its stride, 64 past a multiple of 128 so a
  //: quarter-warp's two 64-byte hi lines (d_v 32) and a half-warp's two lo lines (d_v 64) share no bank
  static constexpr uint32_t kPieceBytes = PAT == kWriteOne ? 3 * DV + 32 : PAT == kReadOne ? 2 * DV + 32 : 2 * DV + 16;
  static constexpr uint32_t kPiece = (kPieceBytes + 63) / 128 * 128 + 64;
  //: the tail of a page: its exponents and masses, `[kExp, kBytes)`
  static constexpr uint32_t kTail = Page::kBytes - Page::kExp;
  static constexpr uint32_t kSlot = PAT == kWriteAll ? Page::kBytes : PAT == kReadAll ? 32 * DV + kTail : kAtoms * kPiece;
  //: the bytes a position's copies land: its slot but the pieces' pads
  static constexpr uint32_t kCopied = kOne ? kAtoms * kPieceBytes : kSlot;
  //: the bytes a position reads and writes: its sectors
  static constexpr long long kRead = PAT == kWriteAll ? Page::kBytes
                                     : PAT == kReadAll ? 32 * DV + kTail
                                     : PAT == kWriteOne ? kAtoms * (3 * DV + 64)
                                     : PAT == kReadOne ? kAtoms * (2 * DV + 64)
                                                       : kAtoms * G::kRecord;
  static constexpr long long kWritten = kWrite ? kRead : 0;
  static constexpr int kPositions = PAT == kWriteAll ? 64 * kWarps : PAT == kReadAll ? 128 * kWarps
                                    : PAT == kWriteOne ? 128 * kWarps : 192 * kWarps;
};

//: a ring's depth: the slots a ring warp's share of the shared memory holds, at most 8
template <int PAT, int DV, int RINGS>
constexpr int depth() {
  const uint32_t share = (kSmem - kTables) / RINGS / Work<PAT, DV>::kSlot;
  return share < 8 ? static_cast<int>(share) : 8;
}

//: the plane's contents: leaf `r` of atom `a`, channel `ch`, an s16 at exponent -8 (the value itself), its mass
__host__ __device__ constexpr int hi_of(uint32_t a, int r, int ch) { return static_cast<int>((a * 7u + r * 3u + ch) % 61u) - 30; }
__host__ __device__ constexpr float mass_of(uint32_t a, int r) { return static_cast<float>(1 + ((a + r) & 7)); }
//: the weights: head `g`'s read of leaf `r`, the write side's, and the token's value at channel `ch`
__host__ __device__ constexpr float read_w(int g, int r) { return static_cast<float>(1 + ((r + g) & 1)); }
__host__ __device__ constexpr float write_w(int r) { return static_cast<float>(1 + (r & 1)); }
__host__ __device__ constexpr float value_of(int ch) { return static_cast<float>(ch % 7 - 3); }
//: an item's atom over a plane of `mask + 1` atoms, and its leaf (a ONE pattern)
__host__ __device__ constexpr uint32_t atom_of(uint32_t item, uint32_t mask) { return (item * 2654435761u) & mask; }
__host__ __device__ constexpr int row_of(uint32_t item) { return static_cast<int>((item * 2246822519u) >> 11) & 15; }

template <int DV>
__global__ void fill(uint8_t* plane, long long atoms, int record) {
  using G = Geo<DV>;
  using Page = typename G::Page;
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < atoms * 16; i += stride) {
    const uint32_t a = static_cast<uint32_t>(i >> 4);
    const int r = static_cast<int>(i & 15);
    if (record) {
      uint8_t* const rec = plane + i * G::kRecord;
      for (int ch = 0; ch < DV; ++ch) reinterpret_cast<int16_t*>(rec)[ch] = static_cast<int16_t>(hi_of(a, r, ch));
      for (int b = G::kHiLine; b < G::kRecord; ++b) rec[b] = 0;
      for (int gi = 0; gi < G::kExpLeaf; ++gi) rec[G::kHiLine + gi] = static_cast<uint8_t>(-8);
      *reinterpret_cast<float*>(rec + G::kHiLine + 4) = mass_of(a, r);
    } else {
      uint8_t* const page = plane + static_cast<long long>(a) * Page::kBytes;
      for (int ch = 0; ch < DV; ++ch) reinterpret_cast<int16_t*>(page + Page::kHi)[r * DV + ch] = static_cast<int16_t>(hi_of(a, r, ch));
      for (int ch = 0; ch < DV; ++ch) page[Page::kLo + r * DV + ch] = 0;
      for (int gi = 0; gi < G::kExpLeaf; ++gi) page[Page::kExp + r * G::kExpLeaf + gi] = static_cast<uint8_t>(-8);
      if (r == 0)
        for (uint32_t b = Page::kExp + 16 * G::kExpLeaf; b < Page::kMass; ++b) page[b] = 0;
      reinterpret_cast<float*>(page + Page::kMass)[r] = mass_of(a, r);
    }
  }
}

//: A LEAF AS A LANE HOLDS IT: its eight channels' hi and lo words, its group's exponent, its mass
struct Leaf {
  uint4 hi;
  uint2 lo;
  int e;
  float mass;
};

//: THE WALK'S ARGUMENTS: the clocks, the sums out, the timelines, the plane and its atoms' mask, the positions a CTA
struct Args {
  long long* times;
  float* sinks;
  unsigned long long* trace;
  uint8_t* plane;
  uint32_t mask;
};

//: THE TABLES a CTA stages once: head g's weight of leaf r, the write side's, the token's value (fp32 a channel)
template <int DV, int G>
struct Tables {
  float read[G][16];
  float write[16];
  float value[DV];
};

//: THE ACCUMULATORS a lane: G heads' eight channels and masses
template <int G>
struct Acc {
  float ch[G][8];
  float m[G];
};

//: A LEAF READ by every head: the hi plane at `2^(e + 8)`, into the lane's partials (the mass by the group's first lane)
template <int DV, int G>
__device__ __forceinline__ void read_leaf(const Leaf& s, const Tables<DV, G>& t, int row, bool lead, Acc<G>& acc) {
  float f[8];
#pragma unroll
  for (int c = 0; c < 8; ++c) f[c] = static_cast<float>(fsla::state_page::hi8(s.hi, c));
  const float scale = fsla::state_page::pow2(s.e + 8);
#pragma unroll
  for (int g = 0; g < G; ++g) {
    const float w = t.read[g][row];
    const float ws = w * scale;
#pragma unroll
    for (int c = 0; c < 8; ++c) acc.ch[g][c] = fmaf(f[c], ws, acc.ch[g][c]);
    acc.m[g] = fmaf(lead ? w : 0.0f, s.mass, acc.m[g]);
  }
}

//: A LEAF WRITTEN, THEN READ: the deposit `S + aw [v, 1]` re-rounded to its group's exponent (four lanes a group), stored
//: from registers into its page, then read by every head (update first, then read)
template <int DV, int G>
__device__ __forceinline__ void write_leaf(const Leaf& s, uint8_t* hi, uint8_t* lo, uint8_t* ex, float* mass,
                                           const Tables<DV, G>& t, int row, int j, Acc<G>& acc) {
  namespace sp = fsla::state_page;
  const float aw = t.write[row];
  const float kept = sp::pow2(s.e);
  float sv[8];
  float amax = 0.0f;
#pragma unroll
  for (int c = 0; c < 8; ++c) {
    sv[c] = fmaf(static_cast<float>(sp::mantissa8(s.hi, s.lo, c)), kept, aw * t.value[8 * j + c]);
    amax = fmaxf(amax, fabsf(sv[c]));
  }
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  const int en = sp::group_exponent(amax);
  int m[8];
#pragma unroll
  for (int c = 0; c < 8; ++c) m[c] = sp::mantissa(sv[c], en);
  *reinterpret_cast<uint4*>(hi) = sp::pack_hi8(m);
  *reinterpret_cast<uint2*>(lo) = sp::pack_lo8(m);
  if ((j & 3) == 0) *reinterpret_cast<int8_t*>(ex) = static_cast<int8_t>(en);
  const float mnew = s.mass + aw;
  if (j == 0) *mass = mnew;
  const float scale = sp::pow2(en);
#pragma unroll
  for (int g = 0; g < G; ++g) {
    const float w = t.read[g][row];
    const float ws = w * scale;
#pragma unroll
    for (int c = 0; c < 8; ++c) acc.ch[g][c] = fmaf(static_cast<float>(m[c]), ws, acc.ch[g][c]);
    acc.m[g] = fmaf(j == 0 ? w : 0.0f, mnew, acc.m[g]);
  }
}

//: A LEAF'S ADDRESSES in the plane: atom `a`, leaf `r`, the lane's eighth `j` of the channels
template <int PAT, int DV>
struct At {
  using G = Geo<DV>;
  using Page = typename G::Page;
  uint8_t *hi, *lo, *ex, *mass;
  __device__ __forceinline__ At(uint8_t* plane, uint32_t a, int r, int j) {
    if constexpr (PAT == kRecord) {
      uint8_t* const rec = plane + (static_cast<long long>(a) * 16 + r) * G::kRecord;
      hi = rec + 16 * j;
      lo = rec;
      ex = rec + G::kHiLine + (j >> 2);
      mass = rec + G::kHiLine + 4;
    } else {
      uint8_t* const page = plane + static_cast<long long>(a) * Page::kBytes;
      hi = page + Page::kHi + r * G::kHiLine + 16 * j;
      lo = page + Page::kLo + r * G::kLoLine + 8 * j;
      ex = page + Page::kExp + r * G::kExpLeaf + (j >> 2);
      mass = page + Page::kMass + 4 * r;
    }
  }
};

//: A LEAF'S PLACE in a slot: the pieces of position-local leaf (trip `tr`, group `k`) as the pattern lays them
template <int PAT, int DV>
struct InSlot {
  using W = Work<PAT, DV>;
  using G = Geo<DV>;
  using Page = typename G::Page;
  uint32_t hi, lo, ex, mass;
  __device__ __forceinline__ InSlot(uint32_t slot, int r, int k, int j) {
    if constexpr (PAT == kWriteAll) {
      hi = slot + Page::kHi + r * G::kHiLine + 16 * j;
      lo = slot + Page::kLo + r * G::kLoLine + 8 * j;
      ex = slot + Page::kExp + r * G::kExpLeaf + (j >> 2);
      mass = slot + Page::kMass + 4 * r;
    } else if constexpr (PAT == kReadAll) {
      hi = slot + r * G::kHiLine + 16 * j;
      lo = hi;
      ex = slot + 32 * DV + r * G::kExpLeaf + (j >> 2);
      mass = slot + 32 * DV + (Page::kMass - Page::kExp) + 4 * r;
    } else {
      const uint32_t piece = slot + k * W::kPiece;
      hi = piece + 16 * j;
      if constexpr (PAT == kWriteOne) {
        lo = piece + G::kHiLine + 8 * j;
        ex = piece + 3 * DV + ((r * G::kExpLeaf) & 15) + (j >> 2);
        mass = piece + 3 * DV + 16 + ((4 * r) & 15);
      } else if constexpr (PAT == kReadOne) {
        lo = hi;
        ex = piece + G::kHiLine + ((r * G::kExpLeaf) & 15) + (j >> 2);
        mass = piece + G::kHiLine + 16 + ((4 * r) & 15);
      } else {
        lo = hi;
        ex = piece + G::kHiLine + (j >> 2);
        mass = piece + G::kHiLine + 4;
      }
    }
  }
};

__device__ __forceinline__ uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
  return v;
}
__device__ __forceinline__ uint2 lds64(uint32_t a) {
  uint2 v;
  asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a));
  return v;
}
__device__ __forceinline__ int lds_s8(uint32_t a) {
  int v;
  asm volatile("ld.shared.s8 %0, [%1];" : "=r"(v) : "r"(a));
  return v;
}
__device__ __forceinline__ float lds_f32(uint32_t a) {
  float v;
  asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(a));
  return v;
}

//: A POSITION: its items' first index, and a leaf's atom and row (trip `tr`, group `k`)
template <int PAT, int DV>
struct Position {
  using W = Work<PAT, DV>;
  uint32_t first;
  __device__ __forceinline__ uint32_t item(int k) const { return first + (W::kOne ? k : 0); }
  __device__ __forceinline__ uint32_t atom(int k, uint32_t mask) const { return atom_of(item(k), mask); }
  __device__ __forceinline__ int row(int tr, int k) const {
    return W::kOne ? row_of(item(k)) : tr * Geo<DV>::kGroups + k;
  }
};

template <int PAT, int DV>
__device__ __forceinline__ Position<PAT, DV> position(int q) {
  using W = Work<PAT, DV>;
  return {(blockIdx.x * static_cast<uint32_t>(W::kPositions) + q) * W::kAtoms};
}

//: one leaf of a position computed: read by every head, or written first
template <int PAT, int DV, int G>
__device__ __forceinline__ void leaf_step(const Position<PAT, DV>& pos, const Args& a, const Tables<DV, G>& t, int k,
                                          int j, int row, const Leaf& s, Acc<G>& acc) {
  if constexpr (Work<PAT, DV>::kWrite) {
    const At<PAT, DV> g(a.plane, pos.atom(k, a.mask), row, j);
    write_leaf<DV, G>(s, g.hi, g.lo, g.ex, reinterpret_cast<float*>(g.mass), t, row, j, acc);
  } else {
    read_leaf<DV, G>(s, t, row, j == 0, acc);
  }
}

//: A POSITION COMPUTED: every leaf a lane group, a trip at a time, its pieces from `in` (registers or the slot). REG's
//: trips are unrolled (its pieces are registers a trip); a ring's are not, so no trip's shared loads are hoisted into
//: another's registers beside G heads' accumulators
template <int PAT, int DV, int G, bool UNROLLED, class In>
__device__ __forceinline__ void compute(const Position<PAT, DV>& pos, const Args& a, const Tables<DV, G>& t, int k,
                                        int j, In in, Acc<G>& acc) {
  using W = Work<PAT, DV>;
  if constexpr (UNROLLED) {
#pragma unroll
    for (int tr = 0; tr < W::kTrips; ++tr) leaf_step<PAT, DV, G>(pos, a, t, k, j, pos.row(tr, k), in(tr, pos.row(tr, k)), acc);
  } else {
#pragma unroll 1
    for (int tr = 0; tr < W::kTrips; ++tr) leaf_step<PAT, DV, G>(pos, a, t, k, j, pos.row(tr, k), in(tr, pos.row(tr, k)), acc);
  }
}

//: A POSITION'S COPIES into a slot: a ring warp's (or the producer's) lanes. BULK: lane 0's, one bulk copy a page, the
//: hi plane and the tail, or every lane group's leaf's pieces, each byte completing `bar` (lane 0 states them first);
//: PIECES: 16-byte copies, a lane its chunks, one copy group
template <int PAT, int DV, bool BULK>
__device__ __forceinline__ void issue(const Position<PAT, DV>& pos, uint32_t mask, uint8_t* plane, uint32_t slot,
                                      uint32_t bar, int lane) {
  using W = Work<PAT, DV>;
  using Gm = Geo<DV>;
  using Page = typename Gm::Page;
  const int k = lane / Gm::kLanes, j = lane % Gm::kLanes;
  if constexpr (BULK) {
    if (lane == 0) lab::mbar_expect(bar, W::kCopied);
    if constexpr (PAT == kWriteAll) {
      if (lane == 0) lab::bulk_load(slot, plane + static_cast<long long>(pos.atom(0, mask)) * Page::kBytes, Page::kBytes, bar);
    } else if constexpr (PAT == kReadAll) {
      const uint8_t* const page = plane + static_cast<long long>(pos.atom(0, mask)) * Page::kBytes;
      if (lane == 0) {
        lab::bulk_load(slot, page, 32 * DV, bar);
        lab::bulk_load(slot + 32 * DV, page + Page::kExp, W::kTail, bar);
      }
    } else {
      //: a ONE pattern's copies, every lane group's by lane 0: a bulk copy's addresses are the warp's (uniform), so one
      //: lane issues them all rather than the toolkit looping over the lanes that differ
      if (lane == 0) {
#pragma unroll
        for (int kk = 0; kk < W::kAtoms; ++kk) {
          const int r = pos.row(0, kk);
          const At<PAT, DV> g(plane, pos.atom(kk, mask), r, 0);
          const uint32_t piece = slot + kk * W::kPiece;
          lab::bulk_load(piece, g.hi, Gm::kHiLine, bar);
          if constexpr (PAT == kWriteOne) {
            lab::bulk_load(piece + Gm::kHiLine, g.lo, Gm::kLoLine, bar);
            lab::bulk_load(piece + 3 * DV, g.ex - ((r * Gm::kExpLeaf) & 15), 16, bar);
            lab::bulk_load(piece + 3 * DV + 16, g.mass - ((4 * r) & 15), 16, bar);
          } else if constexpr (PAT == kReadOne) {
            lab::bulk_load(piece + Gm::kHiLine, g.ex - ((r * Gm::kExpLeaf) & 15), 16, bar);
            lab::bulk_load(piece + Gm::kHiLine + 16, g.mass - ((4 * r) & 15), 16, bar);
          } else {
            lab::bulk_load(piece + Gm::kHiLine, g.hi + Gm::kHiLine, 16, bar);
          }
        }
      }
    }
  } else {
    if constexpr (PAT == kWriteAll || PAT == kReadAll) {
      const uint8_t* const page = plane + static_cast<long long>(pos.atom(0, mask)) * Page::kBytes;
      constexpr int kChunks = W::kSlot / 16, kHiChunks = PAT == kReadAll ? 2 * DV : kChunks;
#pragma unroll
      for (int i = 0; i < (kChunks + 31) / 32; ++i) {
        const int c = i * 32 + lane;
        const uint8_t* const src = page + (c < kHiChunks ? 16 * c : Page::kExp + 16 * (c - kHiChunks));
        if (c < kChunks) lab::cp_async16(slot + 16 * c, src);
      }
    } else {
      const int r = pos.row(0, k);
      const At<PAT, DV> g(plane, pos.atom(k, mask), r, j);
      const uint32_t piece = slot + k * W::kPiece;
      lab::cp_async16(piece + 16 * j, g.hi);
      if constexpr (PAT == kWriteOne) {
        const At<PAT, DV> g0(plane, pos.atom(k, mask), r, 0);
        if (j < DV / 16) lab::cp_async16(piece + Gm::kHiLine + 16 * j, g0.lo + 16 * j);
        if (j == DV / 16) lab::cp_async16(piece + 3 * DV, g0.ex - ((r * Gm::kExpLeaf) & 15));
        if (j == DV / 16 + 1) lab::cp_async16(piece + 3 * DV + 16, g0.mass - ((4 * r) & 15));
      } else if constexpr (PAT == kReadOne) {
        const At<PAT, DV> g0(plane, pos.atom(k, mask), r, 0);
        if (j == 0) lab::cp_async16(piece + Gm::kHiLine, g0.ex - ((r * Gm::kExpLeaf) & 15));
        if (j == 1) lab::cp_async16(piece + Gm::kHiLine + 16, g0.mass - ((4 * r) & 15));
      } else {
        if (j == 0) lab::cp_async16(piece + Gm::kHiLine, g.hi - 16 * j + Gm::kHiLine);
      }
    }
    lab::cp_commit();
  }
}

//: a leaf read from a slot
template <int PAT, int DV>
__device__ __forceinline__ Leaf from_slot(uint32_t slot, int row, int k, int j) {
  const InSlot<PAT, DV> p(slot, row, k, j);
  Leaf s;
  s.hi = lds128(p.hi);
  s.lo = Work<PAT, DV>::kWrite ? lds64(p.lo) : make_uint2(0u, 0u);
  s.e = lds_s8(p.ex);
  s.mass = lds_f32(p.mass);
  return s;
}

//: THE WALK: one row's kernel; the tables staged, the clock started, the form's loop, the sums out
template <int FORM, int PAT, int DV, int G, int TRACE>
__global__ void __launch_bounds__(kThreads, 1) walk(Args a) {
  using W = Work<PAT, DV>;
  using Gm = Geo<DV>;
  constexpr int kRings = FORM == kProducer ? kWarps - 1 : kWarps;
  constexpr int K = FORM == kReg ? 1 : depth<PAT, DV, kRings>();
  extern __shared__ __align__(128) uint8_t smem[];
  auto& t = *reinterpret_cast<Tables<DV, G>*>(smem);
  //: a slot's two barriers: its full phase (its bytes landed) and, the producer's, its empty phase (its reads done)
  uint64_t* const full = reinterpret_cast<uint64_t*>(smem + sizeof(Tables<DV, G>));
  uint64_t* const empty = full + kRings * K;
  const uint32_t base = lab::smem(smem), rings = base + kTables;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int k = lane / Gm::kLanes, j = lane % Gm::kLanes;
  lab::Trace<TRACE != 0> tr;
  tr.bind(a.trace);
  tr.mark(warpline::kPhaseBegin, kPhases + kEntry);
  for (int i = tid; i < G * 16; i += kThreads) t.read[i / 16][i % 16] = read_w(i / 16, i % 16);
  if (tid < 16) t.write[tid] = write_w(tid);
  if (tid < DV) t.value[tid] = value_of(tid);
  if (tid < kRings * K) {
    lab::mbar_init(lab::smem(full + tid), 1);
    lab::mbar_init(lab::smem(empty + tid), 1);
  }
  lab::bulk_fence();
  __syncthreads();
  Acc<G> acc;
#pragma unroll
  for (int g = 0; g < G; ++g) {
#pragma unroll
    for (int c = 0; c < 8; ++c) acc.ch[g][c] = 0.0f;
    acc.m[g] = 0.0f;
  }
  lab::WarpSpan clock;
  clock.start();
  if constexpr (FORM == kReg) {
    constexpr int kMine = W::kPositions / kWarps;
#pragma unroll 1
    for (int p = 0; p < kMine; ++p) {
      const Position<PAT, DV> pos = position<PAT, DV>(p * kWarps + warp);
      tr.mark(warpline::kPhaseBegin, kPhases + kLoad);
      Leaf held[W::kTrips];
#pragma unroll
      for (int i = 0; i < W::kTrips; ++i) {
        const At<PAT, DV> g(a.plane, pos.atom(k, a.mask), pos.row(i, k), j);
        held[i].hi = *reinterpret_cast<const uint4*>(g.hi);
        held[i].lo = W::kWrite ? *reinterpret_cast<const uint2*>(g.lo) : make_uint2(0u, 0u);
        held[i].e = *reinterpret_cast<const int8_t*>(g.ex);
        held[i].mass = *reinterpret_cast<const float*>(g.mass);
      }
      tr.mark(warpline::kPhaseBegin, kPhases + kCompute);
      compute<PAT, DV, G, true>(pos, a, t, k, j, [&](int i, int) { return held[i]; }, acc);
    }
  } else if constexpr (FORM == kBulk || FORM == kPieces) {
    constexpr int kMine = W::kPositions / kWarps;
    const uint32_t ring = rings + warp * K * W::kSlot;
    const auto slot = [&](int p) { return ring + (p % K) * W::kSlot; };
    const auto bar = [&](int p) { return lab::smem(full + warp * K + p % K); };
    const auto consume = [&](int q) {
      const uint32_t s = slot(q);
      tr.at(warpline::kWaitBegin, bar(q));
      if constexpr (FORM == kBulk) lab::mbar_wait(bar(q), static_cast<uint32_t>((q / K) & 1));
      else lab::cp_wait<K - 1>();
      __syncwarp();
      tr.at(warpline::kWaitEnd, bar(q));
      tr.mark(warpline::kPhaseBegin, kPhases + kCompute);
      compute<PAT, DV, G, false>(position<PAT, DV>(q * kWarps + warp), a, t, k, j,
                          [&](int, int row) { return from_slot<PAT, DV>(s, row, k, j); }, acc);
      __syncwarp();
    };
#pragma unroll 1
    for (int p = 0; p < K - 1; ++p) {
      tr.at(warpline::kIssue, bar(p));
      issue<PAT, DV, FORM == kBulk>(position<PAT, DV>(p * kWarps + warp), a.mask, a.plane, slot(p), bar(p), lane);
    }
#pragma unroll 1
    for (int p = K - 1; p < kMine; ++p) {
      tr.at(warpline::kIssue, bar(p));
      issue<PAT, DV, FORM == kBulk>(position<PAT, DV>(p * kWarps + warp), a.mask, a.plane, slot(p), bar(p), lane);
      consume(p - (K - 1));
    }
#pragma unroll 1
    for (int q = kMine - (K - 1); q < kMine; ++q) {
      if constexpr (FORM == kPieces) lab::cp_commit();
      consume(q);
    }
  } else {
    //: THE PRODUCER: warp 0 fills consumer c's (warp c + 1's) slot `n % K` with the CTA's position `n (W - 1) + c`
    constexpr int kConsumers = kWarps - 1;
    const auto slot = [&](int c, int n) { return rings + (c * K + n % K) * W::kSlot; };
    const auto full_of = [&](int c, int n) { return lab::smem(full + c * K + n % K); };
    const auto empty_of = [&](int c, int n) { return lab::smem(empty + c * K + n % K); };
    if (warp == 0) {
#pragma unroll 1
      for (int q = 0; q < W::kPositions; ++q) {
        const int c = q % kConsumers, n = q / kConsumers;
        tr.at(warpline::kWaitBegin, empty_of(c, n));
        lab::mbar_wait(empty_of(c, n), static_cast<uint32_t>(((n / K) & 1) ^ 1));
        tr.at(warpline::kWaitEnd, empty_of(c, n));
        tr.at(warpline::kIssue, full_of(c, n));
        issue<PAT, DV, true>(position<PAT, DV>(q), a.mask, a.plane, slot(c, n), full_of(c, n), lane);
      }
    } else {
      const int c = warp - 1;
      const int mine = (W::kPositions - c + kConsumers - 1) / kConsumers;
#pragma unroll 1
      for (int n = 0; n < mine; ++n) {
        const uint32_t s = slot(c, n);
        tr.at(warpline::kWaitBegin, full_of(c, n));
        lab::mbar_wait(full_of(c, n), static_cast<uint32_t>((n / K) & 1));
        tr.at(warpline::kWaitEnd, full_of(c, n));
        tr.mark(warpline::kPhaseBegin, kPhases + kCompute);
        compute<PAT, DV, G, false>(position<PAT, DV>(n * kConsumers + c), a, t, k, j,
                            [&](int, int row) { return from_slot<PAT, DV>(s, row, k, j); }, acc);
        __syncwarp();
        tr.at(warpline::kArrive, empty_of(c, n));
        if (lane == 0) lab::mbar_arrive(empty_of(c, n));
      }
    }
  }
  clock.stop(a.times);
  tr.mark(warpline::kPhaseBegin, kPhases + kExit);
  tr.close(base);
  float* const out = a.sinks + (static_cast<long long>(blockIdx.x) * kThreads + tid) * G * 9;
#pragma unroll
  for (int g = 0; g < G; ++g) {
#pragma unroll
    for (int c = 0; c < 8; ++c) out[g * 9 + c] = acc.ch[g][c];
    out[g * 9 + 8] = acc.m[g];
  }
}

//: the shared memory a row reserves: the tables, then the rings (at a 1 KB alignment)
template <int FORM, int PAT, int DV>
constexpr uint32_t shared_bytes() {
  constexpr int kRings = FORM == kProducer ? kWarps - 1 : kWarps;
  return FORM == kReg ? kTables : kTables + kRings * depth<PAT, DV, kRings>() * Work<PAT, DV>::kSlot;
}

}  // namespace decode_ring

namespace host {
using namespace decode_ring;

//: the host's s24 codec, as `common/state_page.cuh` stores a deposit: the group's exponent and a mantissa
inline int exponent_of(double amax) {
  if (amax == 0.0) return 0;
  int e = 0;
  std::frexp(amax, &e);
  return std::max(e - 1 - 22, -126);
}

//: THE HOST'S REFERENCE of a row: the sums every thread's accumulators add to (head, channel; then the masses), and a
//: written leaf's expected bytes
template <int PAT, int DV, int G>
struct Reference {
  using W = Work<PAT, DV>;
  using Gm = Geo<DV>;
  std::vector<long long> sums = std::vector<long long>(G * (DV + 1), 0);

  //: the value a leaf holds after the step at channel `ch` (written: the deposit), and its mass
  static long long value(uint32_t a, int r, int ch) {
    return hi_of(a, r, ch) + (W::kWrite ? static_cast<long long>(write_w(r) * value_of(ch)) : 0);
  }
  static long long mass(uint32_t a, int r) {
    return static_cast<long long>(mass_of(a, r) + (W::kWrite ? write_w(r) : 0.0f));
  }

  //: the sums by the leaf's row parity (a head's weight is its parity's), then by head
  void add(int sms, uint32_t mask) {
    std::vector<long long> by_parity(2 * (DV + 1), 0);
    for (long long b = 0; b < sms; ++b)
      for (int q = 0; q < W::kPositions; ++q)
        for (int k = 0; k < (W::kOne ? Gm::kGroups : 1); ++k) {
          const uint32_t item = static_cast<uint32_t>((b * W::kPositions + q) * W::kAtoms + (W::kOne ? k : 0));
          const uint32_t a = atom_of(item, mask);
          for (int i = 0; i < (W::kOne ? 1 : 16); ++i) {
            const int r = W::kOne ? row_of(item) : i;
            long long* const into = by_parity.data() + (r & 1) * (DV + 1);
            for (int ch = 0; ch < DV; ++ch) into[ch] += value(a, r, ch);
            into[DV] += mass(a, r);
          }
        }
    for (int g = 0; g < G; ++g)
      for (int p = 0; p < 2; ++p)
        for (int x = 0; x <= DV; ++x)
          sums[g * (DV + 1) + x] += static_cast<long long>(read_w(g, p)) * by_parity[p * (DV + 1) + x];
  }

  //: leaf `r` of atom `a` as the step stores it: its hi and lo planes' bytes, its exponents, its mass's bits
  static void leaf(uint32_t a, int r, std::vector<uint8_t>& hi, std::vector<uint8_t>& lo, std::vector<int8_t>& ex,
                   float& m) {
    hi.assign(2 * DV, 0);
    lo.assign(DV, 0);
    ex.assign(Gm::kExpLeaf, 0);
    for (int gi = 0; gi < Gm::kExpLeaf; ++gi) {
      long long amax = 0;
      for (int ch = 32 * gi; ch < 32 * gi + 32; ++ch) amax = std::max(amax, std::llabs(value(a, r, ch)));
      const int e = exponent_of(static_cast<double>(amax));
      ex[gi] = static_cast<int8_t>(e);
      for (int ch = 32 * gi; ch < 32 * gi + 32; ++ch) {
        const long long mant = static_cast<long long>(std::ldexp(static_cast<double>(value(a, r, ch)), -e));
        const uint32_t u = static_cast<uint32_t>(mant);
        hi[2 * ch] = static_cast<uint8_t>(u >> 8);
        hi[2 * ch + 1] = static_cast<uint8_t>(u >> 16);
        lo[ch] = static_cast<uint8_t>(u);
      }
    }
    m = static_cast<float>(mass(a, r));
  }
};

}  // namespace host

template <int FORM, int PAT, int DV, int G, int TRACE>
void run(lab::Bench& b, uint8_t* plane, float* sinks, unsigned long long* rows, const char* what) {
  using namespace decode_ring;
  using W = Work<PAT, DV>;
  using Gm = Geo<DV>;
  using Page = typename Gm::Page;
  constexpr int kRings = FORM == kProducer ? kWarps - 1 : kWarps;
  constexpr int K = FORM == kReg ? 1 : depth<PAT, DV, kRings>();
  static_assert(FORM == kReg || K >= 2, "a ring holds two slots at least");
  static_assert(sizeof(Tables<DV, G>) + 2 * kRings * K * 8 <= kTables, "the tables and barriers fit their share");
  static const char* const kForms[] = {"REG, a position in registers", "BULK, a ring a warp by bulk copies",
                                       "PIECES, a ring a warp by 16-byte cp.async", "PRODUCER, one warp filling 15 rings"};
  static const char* const kPatterns[] = {"WRITE_ALL", "READ_ALL", "WRITE_ONE", "READ_ONE", "RECORD"};
  const std::string name = std::string(kPatterns[PAT]) + ", d_v " + std::to_string(DV) + ", G " + std::to_string(G) + ": " +
                           kForms[FORM] + (FORM == kReg ? "" : " (" + std::to_string(K) + " slots)") + what;
  const uint32_t pages = DV == 64 ? (1u << 21) : (1u << 22);
  const uint32_t mask = pages - 1;
  static_assert(static_cast<long long>(Work<PAT, DV>::kPositions) * Work<PAT, DV>::kAtoms * 132 <= (DV == 64 ? (1 << 21) : (1 << 22)),
                "a launch touches each atom at most once");
  const auto kernel = walk<FORM, PAT, DV, G, TRACE>;
  const uint32_t shared = shared_bytes<FORM, PAT, DV>();
  LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared)));
  const Args args{b.times, sinks, rows, plane, mask};
  fill<DV><<<4096, 256>>>(plane, pages, PAT == kRecord);
  LAB_CHECK(cudaDeviceSynchronize());
  kernel<<<b.sms, kThreads, shared>>>(args);
  LAB_CHECK(cudaDeviceSynchronize());
  //: THE CHECK: every thread's sums, summed by (head, channel), against the host's; every leaf CTA 0 wrote, byte for byte
  std::vector<float> got(static_cast<size_t>(b.sms) * kThreads * G * 9);
  LAB_CHECK(cudaMemcpy(got.data(), sinks, got.size() * 4, cudaMemcpyDeviceToHost));
  std::vector<long long> device(G * (DV + 1), 0);
  for (size_t th = 0; th < static_cast<size_t>(b.sms) * kThreads; ++th) {
    const int j = static_cast<int>(th % 32) % Gm::kLanes;
    for (int g = 0; g < G; ++g) {
      for (int c = 0; c < 8; ++c) device[g * (DV + 1) + 8 * j + c] += static_cast<long long>(got[(th * G + g) * 9 + c]);
      device[g * (DV + 1) + DV] += static_cast<long long>(got[(th * G + g) * 9 + 8]);
    }
  }
  host::Reference<PAT, DV, G> ref;
  ref.add(b.sms, mask);
  long long wrong = 0;
  for (size_t i = 0; i < device.size(); ++i) wrong += device[i] != ref.sums[i];
  long long bytes_wrong = 0;
  if constexpr (W::kWrite) {
    std::vector<uint8_t> page(Page::kBytes), hi, lo;
    std::vector<int8_t> ex;
    for (int q = 0; q < W::kPositions; ++q)
      for (int kk = 0; kk < W::kAtoms; ++kk) {
        const uint32_t item = static_cast<uint32_t>(q * W::kAtoms + kk);
        const uint32_t at = atom_of(item, mask);
        LAB_CHECK(cudaMemcpy(page.data(), plane + static_cast<long long>(at) * Page::kBytes, Page::kBytes, cudaMemcpyDeviceToHost));
        for (int i = 0; i < (W::kOne ? 1 : 16); ++i) {
          const int r = W::kOne ? row_of(item) : i;
          float m = 0;
          host::Reference<PAT, DV, G>::leaf(at, r, hi, lo, ex, m);
          for (int x = 0; x < 2 * DV; ++x) bytes_wrong += page[Page::kHi + r * 2 * DV + x] != hi[x];
          for (int x = 0; x < DV; ++x) bytes_wrong += page[Page::kLo + r * DV + x] != lo[x];
          for (int x = 0; x < Gm::kExpLeaf; ++x) bytes_wrong += static_cast<int8_t>(page[Page::kExp + r * Gm::kExpLeaf + x]) != ex[x];
          float stored = 0;
          std::memcpy(&stored, page.data() + Page::kMass + 4 * r, 4);
          bytes_wrong += stored != m;
        }
      }
  }
  b.check(name, wrong == 0 && bytes_wrong == 0,
          std::to_string(wrong) + " of " + std::to_string(device.size()) + " sums and " + std::to_string(bytes_wrong) +
              " written bytes differ");
  //: THE TIME: events around each launch; the bytes the pattern moves over it, GB/s the device
  const double bytes = static_cast<double>(b.sms) * W::kPositions * (W::kRead + W::kWritten);
  cudaEvent_t e0, e1;
  LAB_CHECK(cudaEventCreate(&e0));
  LAB_CHECK(cudaEventCreate(&e1));
  const lab::Samples rate = b.variant(
      [&] {
        LAB_CHECK(cudaEventRecord(e0));
        kernel<<<b.sms, kThreads, shared>>>(args);
        LAB_CHECK(cudaEventRecord(e1));
      },
      [&] {
        float ms = 0;
        LAB_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        return bytes / (static_cast<double>(ms) * 1e6);
      });
  b.record(name, "GB/s the device", rate);
  LAB_CHECK(cudaEventDestroy(e0));
  LAB_CHECK(cudaEventDestroy(e1));
  if constexpr (TRACE != 0) {
    std::vector<std::string> roles;
    for (int w = 0; w < kWarps; ++w) roles.push_back(FORM == kProducer ? (w == 0 ? "producer" : "consumer") : "walker");
    std::vector<std::pair<uint32_t, std::string>> names = {{kPhases + kEntry, "entry"}, {kPhases + kExit, "exit"},
                                                           {kPhases + kCompute, "compute"}, {kPhases + kLoad, "load"}};
    //: a slot's barriers' sites: their shared address over 8, read against the CTA's base
    const uint32_t full0 = static_cast<uint32_t>(sizeof(Tables<DV, G>));
    for (int r = 0; r < (FORM == kReg ? 0 : kRings); ++r)
      for (int s = 0; s < K; ++s) {
        const std::string owner = (FORM == kProducer ? "consumer " : "warp ") + std::to_string(r) + " slot " + std::to_string(s);
        names.push_back({(full0 + 8 * (r * K + s)) >> 3, owner + " full"});
        names.push_back({(full0 + 8 * (kRings * K + r * K + s)) >> 3, owner + " empty"});
      }
    if (b.sms > 1) b.trace(name, rows, kWarps, {0, b.sms / 2}, roles, names);
    else b.trace(name, rows, kWarps, {0}, roles, names);
  }
}

//: a configuration's rows: every pattern in every form, RECORD in the two forms that read a scattered leaf alike. REG
//: holds a whole atom's pieces in registers, which beside G = 8 heads' accumulators the 128 registers of a thread do
//: not hold (ptxas spills both ALL patterns at both widths: docs/lab/decode_ring.md), so REG walks them at G = 1 alone
template <int DV, int G, int TRACE>
void rows_of(lab::Bench& b, uint8_t* plane, float* sinks, unsigned long long* rows) {
  using namespace decode_ring;
  const char* const what = TRACE != 0 ? ", every warp's timeline stamped" : "";
  if constexpr (G == 1) run<kReg, kWriteAll, DV, G, TRACE>(b, plane, sinks, rows, what);
  run<kBulk, kWriteAll, DV, G, TRACE>(b, plane, sinks, rows, what);
  run<kPieces, kWriteAll, DV, G, TRACE>(b, plane, sinks, rows, what);
  run<kProducer, kWriteAll, DV, G, TRACE>(b, plane, sinks, rows, what);
  if constexpr (G == 1) run<kReg, kReadAll, DV, G, TRACE>(b, plane, sinks, rows, what);
  run<kBulk, kReadAll, DV, G, TRACE>(b, plane, sinks, rows, what);
  run<kPieces, kReadAll, DV, G, TRACE>(b, plane, sinks, rows, what);
  run<kProducer, kReadAll, DV, G, TRACE>(b, plane, sinks, rows, what);
  run<kReg, kWriteOne, DV, G, TRACE>(b, plane, sinks, rows, what);
  run<kBulk, kWriteOne, DV, G, TRACE>(b, plane, sinks, rows, what);
  run<kPieces, kWriteOne, DV, G, TRACE>(b, plane, sinks, rows, what);
  run<kProducer, kWriteOne, DV, G, TRACE>(b, plane, sinks, rows, what);
  run<kReg, kReadOne, DV, G, TRACE>(b, plane, sinks, rows, what);
  run<kBulk, kReadOne, DV, G, TRACE>(b, plane, sinks, rows, what);
  run<kPieces, kReadOne, DV, G, TRACE>(b, plane, sinks, rows, what);
  run<kProducer, kReadOne, DV, G, TRACE>(b, plane, sinks, rows, what);
  run<kReg, kRecord, DV, G, TRACE>(b, plane, sinks, rows, what);
  run<kPieces, kRecord, DV, G, TRACE>(b, plane, sinks, rows, what);
}

int main(int argc, char** argv) {
  using namespace decode_ring;
  lab::Bench b(argc, argv);
  //: what each form must hold: REG its global loads, BULK and PRODUCER their bulk copies, PIECES its cp.async copies
  b.expect(lab::kernel_family("decode_ring::walk", {kReg}), "LDG", ">=", 1);
  b.expect(lab::kernel_family("decode_ring::walk", {kBulk}), "UBLKCP", ">=", 1);
  b.expect(lab::kernel_family("decode_ring::walk", {kPieces}), "LDGSTS", ">=", 1);
  b.expect(lab::kernel_family("decode_ring::walk", {kProducer}), "UBLKCP", ">=", 1);
  //: REG IS ITS LOADS USED IN THE TRIP THAT ISSUED THEM: a position's pieces loaded into registers and computed in the
  //: same trip, one position in flight a warp -- the form the row measures against the rings
  b.expect(lab::kernel_family("decode_ring::walk", {kReg}), "LOAD_IN_TRIP", "<=", 64);
  //: THE RINGS' JOINS: a slot's wait polls its mbarrier (BULK, PRODUCER) in a loop of its own inside the walk, and one
  //: lane issues a position's bulk copies, PIECES' lanes their own chunks: each a join in the walk's loop the forms are
  for (int form : {kBulk, kPieces, kProducer})
    b.expect(lab::kernel_family("decode_ring::walk", {form}), "LOOP_CONVERGENCE", "<=", 64);
  if (b.expect_only) return 0;
  uint8_t* plane = nullptr;
  float* sinks = nullptr;
  unsigned long long* rows = nullptr;
  //: the plane at its widest: 2^22 pages at d_v 32, the largest of the four layouts
  const size_t plane_bytes = std::max({static_cast<size_t>(1u << 21) * Geo<64>::Page::kBytes,
                                       static_cast<size_t>(1u << 22) * Geo<32>::Page::kBytes,
                                       static_cast<size_t>(1u << 21) * 16 * Geo<64>::kRecord,
                                       static_cast<size_t>(1u << 22) * 16 * Geo<32>::kRecord});
  LAB_CHECK(cudaMalloc(&plane, plane_bytes));
  LAB_CHECK(cudaMalloc(&sinks, static_cast<size_t>(b.sms) * kThreads * 8 * 9 * 4));
  const size_t trace_words = static_cast<size_t>(b.sms) * kWarps * lab::Trace<true>::kRow;
  LAB_CHECK(cudaMalloc(&rows, trace_words * 8));
  LAB_CHECK(cudaMemset(rows, 0, trace_words * 8));
  rows_of<64, 1, 0>(b, plane, sinks, rows);
  rows_of<64, 8, 0>(b, plane, sinks, rows);
  rows_of<32, 8, 0>(b, plane, sinks, rows);
  rows_of<64, 8, 1>(b, plane, sinks, rows);
  return 0;
}
