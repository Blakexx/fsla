// csrc/rola/src/carry/schedule.cuh -- THE WINDOW'S PLAN: from a window's staged counts, its readout's form and chunks,
// its fold's chunks, and each buffer's place in the area -- every warp computing the same plan from the same tenants,
// so none waits on another for it. see docs/internals/carry/schedule.md
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

static_assert(sizeof(WindowPlan) <= kPlanBytes, "a window's plan fits a warp's slot");

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

//: A SIDE'S SEGMENT SUMS in every lane: `S[j]` the box lists' entries before union slot `64 (j + 1)` -- a lane one
//: list's boundaries (one 16-byte load, four u16 pairs), the pairs summed over the warp. -- see schedule.md#totals
template <class P>
__device__ __forceinline__ void segment_sums(int (&S)[kListAt], const uint8_t* smem,
                                             const Window& win, int side, int w, int lane) {
  static_assert(kListAt == 8 && Tenants::kAt % 16 == 0,
                "a list's boundaries one aligned 16-byte load");
  uint4 v = make_uint4(0u, 0u, 0u, 0u);
  if (win.listed && lane < win.lists)
    v = *reinterpret_cast<const uint4*>(at_of<P>(smem, side, w, lane));
  uint32_t q[4] = {v.x, v.y, v.z, v.w};
  static_for<4>([&](auto K) {
    static_for<5>([&](auto X) { q[K] += __shfl_xor_sync(0xffffffffu, q[K], 16 >> X); });
    S[2 * K] = static_cast<int>(q[K] & 0xffffu);
    S[2 * K + 1] = static_cast<int>(q[K] >> 16);
  });
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

//: A WARP'S PLAN SLOT for window `w`, by parity: read at each use, so no plan field holds a register.
template <class P>
__device__ __forceinline__ volatile WindowPlan& plan_of(uint8_t* smem, int warp, int w) {
  return *reinterpret_cast<volatile WindowPlan*>(smem + P::kPlans
                                                 + (2 * warp + (w & 1)) * kPlanBytes);
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

//: THE PLAN of window `w` into the warp's slot: the first readout form whose buffers the area holds -- the image whole
//: and the union resident, the union resident and the image a ring, the image whole and the union in chunks, both in
//: chunks, both in chunks on one read buffer (`prefer`, when one of them, first) -- then the fold: the whole union beside
//: the readout, else after it, else in chunks. -- see docs/internals/carry/schedule.md#planner
template <class P>
__device__ __forceinline__ void plan_window(const CarryParams& p, uint8_t* smem, int w, int warp,
                                            int lane) {
  static_assert(least_form_fits<P>(), "the least form fits the area");
  const Window rw = window_of<P>(p, smem, kRead, w), ww = window_of<P>(p, smem, kWrite, w);
  const int ru = slots_of<P>(smem, rw, kRead, w), fu = slots_of<P>(smem, ww, kWrite, w);
  int rs[kListAt], fs[kListAt];
  segment_sums<P>(rs, smem, rw, kRead, w, lane);
  segment_sums<P>(fs, smem, ww, kWrite, w, lane);
  const int r_all = rs[kListAt - 1], f_all = fs[kListAt - 1];

  //: THE READOUT: each form's buffers against the area, the first that fits
  const int resident = read_bytes<P>(ru, r_all);
  int whole_cap = 0, ring_cap = 0, lone_cap = 0;
  const int whole_m = chunk_of<P, true>(rs, ru, P::kArea - kImageWhole, whole_cap);
  const int ring_m = chunk_of<P, true>(rs, ru, P::kArea - kImageRing, ring_cap);
  const int lone_m = chunk_of<P, true, 1>(rs, ru, P::kArea - kImageRing, lone_cap);
  //: the candidates in order: the image whole and the union resident, the union resident, the image whole and the
  //: union in chunks, both in chunks, both in chunks on one buffer (selects, so nothing indexes at run time)
  const bool f0 = kImageWhole + resident <= P::kArea, f1 = kImageRing + resident <= P::kArea;
  const bool f2 = whole_m > 0, f3 = ring_m > 0, f4 = lone_m > 0;
  const int pick = p.prefer == 0 && f0   ? 0
                   : p.prefer == 1 && f1 ? 1
                   : p.prefer == 2 && f2 ? 2
                   : p.prefer == 3 && f3 ? 3
                   : p.prefer == 4 && f4 ? 4
                   : f0                  ? 0
                   : f1                  ? 1
                   : f2                  ? 2
                   : f3                  ? 3
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

  if (lane == 0) {
    volatile WindowPlan& pl = plan_of<P>(smem, warp, w);
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
  }
  __syncwarp();
}

}  // namespace rola::carry
