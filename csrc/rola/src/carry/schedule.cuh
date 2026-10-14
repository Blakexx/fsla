// csrc/rola/src/carry/schedule.cuh -- THE WINDOWS' PLANS: from each window's counts (the facts pass's unions and
// boundaries), its readout's form and chunks, its fold's chunks, and each buffer's place in the area -- one plan a
// (batch-head, box class, window), made by the plan pass before the carry runs. see docs/internals/carry/schedule.md
#pragma once

#include "carry/parts.cuh"

namespace rola::carry {

//: THE READOUT'S FORMS: the image whole and the union in token chunks (one when it fits); the union resident and the
//: image a ring of batches; both in chunks, the image stored again for each chunk. -- see schedule.md#forms
enum ReadForm : int { kFormImage, kFormUnion, kFormTiled, kReadForms };

//: A WINDOW'S PLAN: its readout's form, each side's union and chunk, and its buffers' places in the area (bytes / 16).
//: -- see docs/internals/carry/schedule.md#plan
struct WindowPlan {
  uint16_t form, beside, rbufs;
  uint16_t runion, rchunk, rchunks, funion, fchunk, fchunks;
  uint16_t rbuf[2], fbuf[2];
  uint16_t rents, fents;
  uint16_t rzone, fzone;
};

static_assert(sizeof(WindowPlan) <= kPlanBytes, "a window's plan fits its slot");

//: A READ BUFFER: its rows, the dummy panel rows, the panel, its entries; a FOLD BUFFER: its rows and entries.
template <class P>
__host__ __device__ constexpr int read_bytes(int slots, int entries) {
  return align_of(slots * P::kReadRow, 128) + kPadBytes + align_of(slots * kPanelRow, 128)
         + align_of(entries * 8, 128);
}

template <class P>
__host__ __device__ constexpr int fold_bytes(int slots, int entries) {
  return align_of(slots * P::kWriteRow, 128) + align_of(entries * 8, 128);
}

//: THE LEAST FORM fits every window: the image's ring and one 64-slot read chunk of lists full on every token, two
//: such fold chunks.
template <class P>
constexpr bool least_form_fits() {
  return kImageRing + read_bytes<P>(kListBound, kListBound * kBoxDigits) <= P::kArea
         && 2 * fold_bytes<P>(kListBound, kListBound * kBoxLists) <= P::kArea;
}

//: A SIDE'S COUNTS in a window: its union's slots and `S[j]`, its box lists' entries before union slot `64 (j + 1)`
//: (the prefix rows' totals row); every window token and no entries where the side has no list level.
//: -- see docs/internals/carry/schedule.md#totals
__device__ __forceinline__ int side_counts(const SideLists& l, bool listed, int tokens,
                                           int64_t group, int (&S)[kListAt]) {
  uint4 v = make_uint4(0u, 0u, 0u, 0u);
  if (listed) v = reinterpret_cast<const uint4*>(l.pre)[(group + 1) * kPreRows - 1];
  const uint32_t q[4] = {v.x, v.y, v.z, v.w};
  static_for<4>([&](auto K) {
    S[2 * K] = static_cast<int>(q[K] & 0xffffu);
    S[2 * K + 1] = static_cast<int>(q[K] >> 16);
  });
  return listed ? l.union_count[group] : tokens;
}

//: THE MOST ENTRIES any chunk of `64 M` slots holds, from the segment sums (a chunk past the union holds none).
template <int M>
__device__ __forceinline__ int chunk_most(const int (&S)[kListAt]) {
  int most = S[M - 1 < kListAt - 1 ? M - 1 : kListAt - 1];
  static_for<(kListAt + M - 1) / M - 1>([&](auto Q) {
    constexpr int q = Q + 1, hi = q * M + M - 1 < kListAt - 1 ? q * M + M - 1 : kListAt - 1;
    most = max(most, S[hi] - S[q * M - 1]);
  });
  return most;
}

//: THE CTA'S PLAN SLOT for window `w`, by parity, staged a window ahead: read at each use, so no plan field holds a
//: register.
template <class P>
__device__ __forceinline__ volatile WindowPlan& plan_of(uint8_t* smem, int w) {
  return *reinterpret_cast<volatile WindowPlan*>(smem + P::kPlans + (w & 1) * kPlanBytes);
}

//: A SIDE'S CHUNKING: the largest chunk of `64 m` slots, fewer than the union, `Bufs` of whose buffers fit `room`; 0
//: for none. Every candidate is priced from the sums in registers.
template <class P, bool Read, int Bufs = 2>
__device__ __forceinline__ int chunk_of(const int (&S)[kListAt], int slots, int room, int& cap) {
  int best = 0;
  static_for<kListAt - 1>([&](auto K) {
    constexpr int m = K + 1;
    const int most = chunk_most<m>(S);
    const int bytes =
        Read ? read_bytes<P>(kListBound * m, most) : fold_bytes<P>(kListBound * m, most);
    if (kListBound * m < slots && Bufs * bytes <= room) {
      best = m;
      cap = bytes;
    }
  });
  return best;
}

//: A WINDOW'S PLAN from its sides' unions and segment sums: the first readout form whose buffers the area holds -- the
//: image whole and the union resident, the union resident and the image a ring, the image whole and the union in
//: chunks, both in chunks, both in chunks on one read buffer (`prefer`, when one of them, first) -- then the fold: the
//: whole union beside the readout, else after it, else in chunks. -- see docs/internals/carry/schedule.md#planner
template <class P>
__device__ __forceinline__ WindowPlan make_plan(int ru, const int (&rs)[kListAt], int fu,
                                                const int (&fs)[kListAt], int prefer) {
  static_assert(least_form_fits<P>(), "the least form fits the area");
  const int r_all = rs[kListAt - 1], f_all = fs[kListAt - 1];

  //: THE READOUT: each form's buffers against the area, the first that fits
  const int resident = read_bytes<P>(ru, r_all);
  int whole_cap = 0, ring_cap = 0, lone_cap = 0;
  const int whole_m = chunk_of<P, true>(rs, ru, P::kArea - kImageWhole, whole_cap);
  const int ring_m = chunk_of<P, true>(rs, ru, P::kArea - kImageRing, ring_cap);
  const int lone_m = chunk_of<P, true, 1>(rs, ru, P::kArea - kImageRing, lone_cap);
  const bool f0 = kImageWhole + resident <= P::kArea, f1 = kImageRing + resident <= P::kArea;
  const bool f2 = whole_m > 0, f3 = ring_m > 0, f4 = lone_m > 0;
  const int pick = prefer == 0 && f0   ? 0
                   : prefer == 1 && f1 ? 1
                   : prefer == 2 && f2 ? 2
                   : prefer == 3 && f3 ? 3
                   : prefer == 4 && f4 ? 4
                   : f0                ? 0
                   : f1                ? 1
                   : f2                ? 2
                   : f3                ? 3
                                       : 4;
  const int form = pick == 0 || pick == 2 ? kFormImage : pick == 1 ? kFormUnion : kFormTiled;
  const bool one = pick < 2;
  const int bufs = pick == 4 ? 1 : 2;
  const int image = form == kFormImage ? kImageWhole : kImageRing;
  const int rchunk = one ? ru : kListBound * (pick == 2 ? whole_m : pick == 3 ? ring_m : lone_m);
  const int rcap = one ? resident : (pick == 2 ? whole_cap : pick == 3 ? ring_cap : lone_cap);
  const int rzone = image + (one ? 1 : bufs) * rcap;

  //: THE FOLD: its whole union beside the readout, else after it, else in chunks
  const int fres = fold_bytes<P>(fu, f_all);
  int fcap = fres, fchunk = fu;
  if (fres > P::kArea) fchunk = kListBound * chunk_of<P, false>(fs, fu, P::kArea, fcap);
  const int fbuf0 = P::kArea - fcap, fbuf1 = fchunk < fu ? fbuf0 - fcap : fbuf0;

  WindowPlan pl{};
  pl.form = static_cast<uint16_t>(form);
  pl.beside = static_cast<uint16_t>(fbuf0 >= rzone);
  pl.rbufs = static_cast<uint16_t>(bufs);
  pl.runion = static_cast<uint16_t>(ru);
  pl.rchunk = static_cast<uint16_t>(rchunk);
  pl.rchunks = static_cast<uint16_t>(ru == 0 ? 0 : (ru + rchunk - 1) / rchunk);
  pl.funion = static_cast<uint16_t>(fu);
  pl.fchunk = static_cast<uint16_t>(fchunk);
  pl.fchunks = static_cast<uint16_t>(fu == 0 ? 0 : (fu + fchunk - 1) / fchunk);
  pl.rbuf[0] = static_cast<uint16_t>(image / 16);
  pl.rbuf[1] = static_cast<uint16_t>((image + rcap) / 16);
  pl.fbuf[0] = static_cast<uint16_t>(fbuf0 / 16);
  pl.fbuf[1] = static_cast<uint16_t>(fbuf1 / 16);
  pl.rents = static_cast<uint16_t>(
      (align_of(rchunk * P::kReadRow, 128) + kPadBytes + align_of(rchunk * kPanelRow, 128)) / 16);
  pl.fents = static_cast<uint16_t>(align_of(fchunk * P::kWriteRow, 128) / 16);
  pl.rzone = static_cast<uint16_t>(rzone / 16);
  pl.fzone = static_cast<uint16_t>(fbuf1 / 16);
  return pl;
}

//: A CTA'S BOX CLASS: the pair of its read and write list levels' groups, which fixes its counts.
__device__ __forceinline__ int box_classes(const CarryParams& p) {
  return max(1, p.read_lists.groups) * max(1, p.write_lists.groups);
}

//: THE PLAN PASS, a thread a (batch-head, box class, window): its sides' counts from the facts pass, its plan out to
//: `p.plans` (`[BH][windows][classes]`, a plan's bytes each). -- see docs/internals/carry/schedule.md#pass
template <class Arm>
__global__ __launch_bounds__(kThreads) void window_plan_kernel(
    const __grid_constant__ CarryParams p) {
  using P = Plan<Arm>;
  const int classes = box_classes(p), groups_w = max(1, p.write_lists.groups);
  const int k = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (k >= classes * p.windows) return;
  const int w = k / classes, cls = k - w * classes, bh = static_cast<int>(blockIdx.y);
  const int tokens = min(kWindow, p.L - w * kWindow);
  const int64_t win = static_cast<int64_t>(bh) * p.windows + w;
  int rs[kListAt], fs[kListAt];
  const int ru = side_counts(p.read_lists, p.read.list >= 0, tokens,
                             win * max(1, p.read_lists.groups) + cls / groups_w, rs);
  const int fu =
      side_counts(p.write_lists, p.write.list >= 0, tokens, win * groups_w + cls % groups_w, fs);
  const WindowPlan pl = make_plan<P>(ru, rs, fu, fs, p.prefer);
  static_assert(sizeof(WindowPlan) <= kPlanBytes && kPlanBytes % 16 == 0,
                "a plan whole 16-byte quanta");
  uint4 q[kPlanBytes / 16] = {};
  memcpy(q, &pl, sizeof(pl));
  uint4* const out = reinterpret_cast<uint4*>(p.plans + (win * classes + cls) * kPlanBytes);
  static_for<kPlanBytes / 16>([&](auto K) { out[K] = q[K]; });
}

}  // namespace rola::carry
