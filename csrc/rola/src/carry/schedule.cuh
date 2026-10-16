// csrc/rola/src/carry/schedule.cuh -- THE WINDOWS' PLANS AND THE RULE THE CARRY RUNS THEM BY: a window's plan is one
// word -- its image mode, its read chunk and its fold chunk -- made by the plan pass from the facts pass's counts
// before the carry runs; the carry derives the window's ENTRIES (its read chunks, then its fold chunks) and places them
// in two alternating slots, filling entry t + 1 while entry t computes wherever the two fit side by side.
// see docs/internals/carry/schedule.md
#pragma once

#include "carry/parts.cuh"

namespace rola::carry {

//: A WINDOW'S IMAGE MODE: whole, or stored and read 16 digits at a time by each read chunk.
enum ImageMode : uint32_t { kModeWhole, kModeHalves };

__host__ __device__ __forceinline__ constexpr int image_bytes(uint32_t mode) {
  return mode == kModeWhole ? kImageWhole : kImageRing;
}

//: A WINDOW'S PLAN, a word: mode | read chunk / 64 << 1 | fold chunk / 64 << 9 | read chunks << 13 | fold chunks << 21,
//: a chunk 0 where its side has no slots and past its union the union. -- see docs/internals/carry/schedule.md#plan
struct PlanWord {
  uint32_t v;

  __host__ __device__ __forceinline__ uint32_t mode() const { return v & 1u; }

  __host__ __device__ __forceinline__ int read_chunk() const {
    return kListBound * static_cast<int>((v >> 1) & 0xffu);
  }

  __host__ __device__ __forceinline__ int fold_chunk() const {
    return kListBound * static_cast<int>((v >> 9) & 0xfu);
  }

  __host__ __device__ __forceinline__ int reads() const {
    return static_cast<int>((v >> 13) & 0xffu);
  }

  __host__ __device__ __forceinline__ int folds() const {
    return static_cast<int>((v >> 21) & 0xfu);
  }
};

__host__ __device__ __forceinline__ constexpr int chunks_of(int u, int chunk) {
  return chunk > 0 ? (u + chunk - 1) / chunk : 0;
}

__host__ __device__ __forceinline__ constexpr uint32_t plan_word(uint32_t mode, int rc, int fc,
                                                                 int ru, int fu) {
  return mode | static_cast<uint32_t>((rc + kListBound - 1) / kListBound) << 1
         | static_cast<uint32_t>((fc + kListBound - 1) / kListBound) << 9
         | static_cast<uint32_t>(chunks_of(ru, rc)) << 13
         | static_cast<uint32_t>(chunks_of(fu, fc)) << 21;
}

//: A READ BUFFER'S BYTES and a FOLD BUFFER'S, at a chunk's slots and entries (`read_ents`, `fold_ents`).
template <class P>
__host__ __device__ constexpr int read_bytes(int slots, int entries) {
  return read_ents<P>(slots) + align_of(entries * 8, 128);
}

template <class P>
__host__ __device__ constexpr int fold_bytes(int slots, int entries) {
  return fold_ents<P>(slots) + align_of(entries * 8, 128);
}

//: THE LEAST PLAN fits every window: the halves and one 64-slot read buffer of lists full on every token; one such
//: fold buffer.
template <class P>
constexpr bool least_plan_fits() {
  return kImageRing + read_bytes<P>(kListBound, kListBound * kBoxDigits) <= P::kArea
         && fold_bytes<P>(kListBound, kListBound * kBoxLists) <= P::kArea;
}

//: A SIDE'S COUNTS in a window: its union's slots and `S[j]`, its box lists' entries before union slot `64 (j + 1)`
//: (the group's exact totals); every window token and no entries where the side has no list level.
//: -- see docs/internals/carry/schedule.md#totals
__device__ __forceinline__ int side_counts(const SideLists& l, bool listed, int tokens,
                                           int64_t group, int (&S)[kListAt]) {
  int4 lo = make_int4(0, 0, 0, 0), hi = lo;
  if (listed) {
    lo = reinterpret_cast<const int4*>(l.totals)[2 * group];
    hi = reinterpret_cast<const int4*>(l.totals)[2 * group + 1];
  }
  static_assert(kListAt == 8, "a group's totals are two 16-byte words");
  const int q[kListAt] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
  static_for<kListAt>([&](auto K) { S[K] = q[K]; });
  return listed ? l.union_count[group] : tokens;
}

//: THE ENTRIES of union slots `first .. first + slots - 1` (`first` a multiple of 64, the end one or the union's end
//: `total`), from the segment sums; the most any chunk of `64 m` slots holds.
__device__ __forceinline__ int sum_at(const int (&S)[kListAt], int j) {
  int v = 0;
  static_for<kListAt>([&](auto J) { v = J == j ? S[J] : v; });
  return v;
}

__device__ __forceinline__ int entries_of(const int (&S)[kListAt], int first, int slots,
                                          int total) {
  const int hi = first + slots >= total ? kListAt - 1 : (first + slots) / kListBound - 1;
  return sum_at(S, hi) - sum_at(S, first / kListBound - 1);
}

__device__ __forceinline__ int chunk_max(const int (&S)[kListAt], int m, int total) {
  int most = 0;
#pragma unroll 1
  for (int first = 0; first < total; first += kListBound * m)
    most = max(most, entries_of(S, first, min(kListBound * m, total - first), total));
  return most;
}

//: THE LARGEST CHUNK of a side: the whole union where one buffer of it fits `room`; else the largest `64 m` slots
//: below the union two of whose largest buffers fit (one lands while the other computes), else one.
//: -- see docs/internals/carry/schedule.md#chunks
template <class Bytes>
__device__ __forceinline__ int largest_chunk(int u, const int (&S)[kListAt], int room,
                                             Bytes bytes) {
  if (bytes(u, S[kListAt - 1]) <= room) return u;
  int two = 0, one = 1;
#pragma unroll 1
  for (int m = 1; kListBound * m < u; ++m) {
    const int b = bytes(kListBound * m, chunk_max(S, m, u));
    if (2 * b <= room) two = m;
    if (b <= room) one = m;
  }
  return kListBound * (two > 0 ? two : one);
}

//: A WINDOW'S COUNTS: each side's union and segment sums.
struct Counts {
  int ru, fu;
  int rs[kListAt], fs[kListAt];
};

//: THE RULE'S PLAN: the read chunk sized beside the halves, the image whole where it fits beside the read buffers in
//: flight (one, or with read chunks two, of the largest), else in halves; the fold's largest chunk in the area.
//: -- see docs/internals/carry/schedule.md#planner
template <class P>
__device__ __forceinline__ uint32_t rule_plan(const Counts& c) {
  const int rc = c.ru > 0 ? largest_chunk(c.ru, c.rs, P::kArea - kImageRing,
                                          [](int s, int e) { return read_bytes<P>(s, e); })
                          : 0;
  const int nr = c.ru > 0 ? (c.ru + rc - 1) / rc : 0;
  const int most =
      nr > 0 ? read_bytes<P>(min(rc, c.ru),
                             rc < c.ru ? chunk_max(c.rs, rc / kListBound, c.ru) : c.rs[kListAt - 1])
             : 0;
  const bool whole = nr > 0 && kImageWhole + (nr > 1 ? 2 : 1) * most <= P::kArea;
  const int fc = c.fu > 0 ? largest_chunk(c.fu, c.fs, P::kArea,
                                          [](int s, int e) { return fold_bytes<P>(s, e); })
                          : 0;
  return plan_word(whole ? kModeWhole : kModeHalves, rc, fc, c.ru, c.fu);
}

//: A FORCED FORM'S PLAN (`CarryParams::forced`, -1 none: mode 0 the rule's, 1 whole, 2 halves; each chunk in 64-slot
//: steps, 0 the rule's and the field's largest the union), where every chunk fits alone beside its image and a whole
//: image has reads to serve; else the rule's.
template <class P>
__device__ __forceinline__ uint32_t forced_plan(int forced, const Counts& c, uint32_t rule) {
  if (forced < 0) return rule;
  const PlanWord r{rule};
  const int fm = forced & 3, fr = (forced >> 2) & 0xff, ff = (forced >> 10) & 0xf;
  const uint32_t mode = fm == 0 ? r.mode() : static_cast<uint32_t>(fm - 1);
  const int rc = c.ru == 0 ? 0 : fr == 0 ? r.read_chunk() : fr == 0xff ? c.ru : kListBound * fr;
  const int fc = c.fu == 0 ? 0 : ff == 0 ? r.fold_chunk() : ff == 0xf ? c.fu : kListBound * ff;
  bool fits = mode == kModeHalves || c.ru > 0;
#pragma unroll 1
  for (int first = 0; first < c.ru; first += rc) {
    const int s = min(rc, c.ru - first);
    fits =
        fits && image_bytes(mode) + read_bytes<P>(s, entries_of(c.rs, first, s, c.ru)) <= P::kArea;
  }
#pragma unroll 1
  for (int first = 0; first < c.fu; first += fc) {
    const int s = min(fc, c.fu - first);
    fits = fits && fold_bytes<P>(s, entries_of(c.fs, first, s, c.fu)) <= P::kArea;
  }
  return fits ? plan_word(mode, rc, fc, c.ru, c.fu) : rule;
}

//: A CTA'S BOX CLASS: the pair of its read and write list levels' groups, which fixes its counts.
__device__ __forceinline__ int box_classes(const CarryParams& p) {
  return max(1, p.read_lists.groups) * max(1, p.write_lists.groups);
}

//: A WINDOW'S COUNTS for a box class, the read and the write side's.
__device__ __forceinline__ void window_counts(const CarryParams& p, int bh, int w, int cls,
                                              Counts& c) {
  const int groups_w = max(1, p.write_lists.groups);
  const int tokens = min(kWindow, p.L - w * kWindow);
  const int64_t win = static_cast<int64_t>(bh) * p.windows + w;
  c.ru = side_counts(p.read_lists, p.read.list >= 0, tokens,
                     win * max(1, p.read_lists.groups) + cls / groups_w, c.rs);
  c.fu =
      side_counts(p.write_lists, p.write.list >= 0, tokens, win * groups_w + cls % groups_w, c.fs);
}

//: THE TWO SLOTS, alternating down the CTA's stream across windows: the top slot's entry ends at the area's top, the
//: low slot's sits above its window's image where the image is live at its fill (a read; the first fold filled while
//: the last read computes), else at the bottom. -- see docs/internals/carry/schedule.md#rule
enum SlotIndex : int { kSlotTop, kSlotLow };

__device__ __forceinline__ int slot_place(int slot, int bytes, int base, int area) {
  return slot == kSlotTop ? area - bytes : base;
}

//: AN ENTRY BESIDE ANOTHER: `[at, at + bytes)` inside the area, clear of the live image's bytes and the live entry's.
__device__ __forceinline__ bool disjoint(int a, int a_bytes, int b, int b_bytes) {
  return a + a_bytes <= b || b + b_bytes <= a;
}

__device__ __forceinline__ bool fits_beside(int at, int bytes, int live_at, int live_bytes,
                                            int image, int area) {
  return at >= image && at + bytes <= area && disjoint(at, bytes, live_at, live_bytes);
}

//: A WINDOW AS THE PLAN PASS SEES IT: its counts, its plan word, its entries and read entries, its image's bytes; an
//: entry's buffer bytes from the exact totals.
struct Win {
  Counts c;
  uint32_t word;
  int n, nr, image;
};

template <class P>
__device__ __forceinline__ Win win_of(const CarryParams& p, int bh, int w, int cls) {
  Win x;
  window_counts(p, bh, w, cls, x.c);
  x.word = forced_plan<P>(p.forced, x.c, rule_plan<P>(x.c));
  const PlanWord pw{x.word};
  x.nr = pw.reads();
  x.n = x.nr + pw.folds();
  x.image = x.nr > 0 ? image_bytes(pw.mode()) : 0;
  return x;
}

template <class P>
__device__ __forceinline__ int entry_bytes_of(const Win& x, int t) {
  const PlanWord pw{x.word};
  const bool read = t < x.nr;
  const int chunk = read ? pw.read_chunk() : pw.fold_chunk(), total = read ? x.c.ru : x.c.fu;
  const int first = (read ? t : t - x.nr) * chunk, slots = min(chunk, total - first);
  return read ? read_bytes<P>(slots, entries_of(x.c.rs, first, slots, total))
              : fold_bytes<P>(slots, entries_of(x.c.fs, first, slots, total));
}

//: A STEP WORD, 16 bits a step of a window (its entries, then one that only retires the last): the entry's place / 128
//: | filled ahead << 11 | the next entry filled now << 12 | a rendezvous between the retired read's drain and a fill
//: that reuses its place << 13. -- see docs/internals/carry/schedule.md#steps
struct StepWord {
  uint32_t v;

  __device__ __forceinline__ int at() const { return 128 * static_cast<int>(v & 0x7ffu); }

  __device__ __forceinline__ bool ahead() const { return (v >> 11 & 1u) != 0u; }

  __device__ __forceinline__ bool next() const { return (v >> 12 & 1u) != 0u; }

  __device__ __forceinline__ bool reuse() const { return (v >> 13 & 1u) != 0u; }
};

//: THE PLAN PASS, a block a (box class, batch-head) and a thread a window of each 128-window tile: the forced form's
//: plan where it fits, else the rule's; the stream's slot parity at each window (an exclusive scan of the windows'
//: entry counts); then each window's steps by the rule, its first entry filled ahead where the window before's last
//: leaves it room. -- see docs/internals/carry/schedule.md#pass
constexpr int kPlanThreads = 128, kPlanWarps = kPlanThreads / 32;

template <class Arm>
__global__ __launch_bounds__(kPlanThreads) void window_plan_kernel(
    const __grid_constant__ CarryParams p) {
  using P = Plan<Arm>;
  static_assert(least_plan_fits<P>(), "the least plan fits the area");
  static_assert(2 * kListAt + 1 <= kMaxSteps, "a window's steps fit its record");
  constexpr int B = P::kArea;
  __shared__ uint32_t parities[kPlanWarps];
  const int cls = static_cast<int>(blockIdx.x), bh = static_cast<int>(blockIdx.y);
  const int lane = static_cast<int>(threadIdx.x) % 32, warp = static_cast<int>(threadIdx.x) / 32;
  uint32_t carry = 0u;
#pragma unroll 1
  for (int w0 = 0; w0 < p.windows; w0 += kPlanThreads) {
    const int w = w0 + static_cast<int>(threadIdx.x);
    const bool live = w < p.windows;
    const Win x = win_of<P>(p, bh, live ? w : 0, cls);
    //: THE PARITY: an exclusive scan of the entry counts' parities over the tile, the tiles before carried
    const uint32_t odd = live ? static_cast<uint32_t>(x.n) & 1u : 0u;
    uint32_t scan = odd;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const uint32_t below = __shfl_up_sync(0xffffffffu, scan, o);
      scan ^= lane >= o ? below : 0u;
    }
    if (lane == 31) parities[warp] = scan;
    __syncthreads();
    uint32_t before = carry;
#pragma unroll
    for (int k = 0; k < kPlanWarps; ++k) before ^= k < warp ? parities[k] : 0u;
#pragma unroll
    for (int k = 0; k < kPlanWarps; ++k) carry ^= parities[k];
    __syncthreads();
    if (!live) continue;
    const int par = static_cast<int>((before ^ scan ^ odd) & 1u);

    //: THE FIRST ENTRY'S PLACE where the window before filled it ahead: beside that window's last entry and its
    //: live image, or with nothing beside it
    bool ahead = false;
    int at = 0;
    if (x.n > 0) {
      const int b0 = entry_bytes_of<P>(x, 0);
      at = slot_place(par, b0, x.image, B);
      ahead = true;
      if (w > 0) {
        const Win pv = win_of<P>(p, bh, w - 1, cls);
        if (pv.n > 0) {
          const int ppar = par ^ (pv.n & 1), lt = pv.n - 1;
          const bool lread = lt < pv.nr;
          const int lb = entry_bytes_of<P>(pv, lt);
          int lbase = lread ? pv.image : 0;
          if (!lread && lt == pv.nr && pv.nr > 0) {
            const int rb = entry_bytes_of<P>(pv, pv.nr - 1);
            const int rat = slot_place(ppar ^ ((pv.nr - 1) & 1), rb, pv.image, B);
            lbase =
                fits_beside(slot_place(ppar ^ (lt & 1), lb, pv.image, B), lb, rat, rb, pv.image, B)
                    ? pv.image
                    : 0;
          }
          ahead = fits_beside(at, b0, slot_place(ppar ^ (lt & 1), lb, lbase, B), lb,
                              lread ? pv.image : 0, B);
        }
      }
    }
    //: THE STEPS: entry `t`'s place (where filled ahead, the place it took then; else above the image where a read and
    //: at the bottom where a fold); the next entry, the window's or at its last the next window's first, in the other slot, filled
    //: now where it fits beside entry `t` and the live image; the rendezvous between the retired read's drain and a
    //: fill that reuses its place
    const Win nx = win_of<P>(p, bh, w + 1 < p.windows ? w + 1 : w, cls);
    uint32_t* const out =
        p.plans + ((static_cast<int64_t>(bh) * p.windows + w) * box_classes(p) + cls) * kPlanWords;
    uint32_t pair = 0u;
    int gone = 0, gone_b = 0;
#pragma unroll 1
    for (int t = 0; t <= x.n; ++t) {
      const bool work = t < x.n, read = t < x.nr, cross = t + 1 >= x.n && (work || t == 0);
      const int slot = par ^ (t & 1);
      const int b = work ? entry_bytes_of<P>(x, t) : 0;
      if (work && !ahead) at = slot_place(slot, b, read ? x.image : 0, B);
      bool next = false;
      int nat = 0, nb = 0;
      if (!cross && work) {
        nb = entry_bytes_of<P>(x, t + 1);
        nat = slot_place(slot ^ 1, nb, t + 1 < x.nr || read ? x.image : 0, B);
        next = fits_beside(nat, nb, at, b, read ? x.image : 0, B);
      } else if (cross && w + 1 < p.windows && nx.n > 0) {
        nb = entry_bytes_of<P>(nx, 0);
        nat = slot_place(work ? slot ^ 1 : slot, nb, nx.image, B);
        next = fits_beside(nat, nb, work ? at : 0, b, work && read ? x.image : 0, B);
      }
      const bool drained = t > 0 && t - 1 < x.nr;
      const bool reuse = drained
                         && ((work && !ahead && !disjoint(at, b, gone, gone_b))
                             || (next && !disjoint(nat, nb, gone, gone_b)));
      const uint32_t word = static_cast<uint32_t>(work ? at / 128 : 0) | (ahead ? 1u : 0u) << 11
                            | (next ? 1u : 0u) << 12 | (reuse ? 1u : 0u) << 13;
      if (t & 1) {
        out[1 + t / 2] = pair | word << 16;
      } else {
        pair = word;
        if (t == x.n) out[1 + t / 2] = pair;
      }
      gone = at;
      gone_b = b;
      ahead = next;
      at = nat;
    }
    out[0] = x.word;
  }
}

//: A WINDOW'S STREAM in the CTA, from its staged plan word and counts, read at each use: its mode, each side's chunk
//: and slots, its read entries and all its entries, its image's bytes (0 where it reads nothing).
//: -- see docs/internals/carry/schedule.md#rule
struct Stream {
  uint32_t mode;
  int rc, fc, ru, fu, nr, n, image;
};

template <class P>
__device__ __forceinline__ Stream stream_of(const CarryParams& p, const uint8_t* smem, int w) {
  const int parity = parity_of(w);
  const PlanWord word{
      *reinterpret_cast<const volatile uint32_t*>(smem + P::kPlans + 4 * kPlanWords * parity)};
  const volatile int* const count = reinterpret_cast<const volatile int*>(smem + P::kCount);
  const int tokens = min(kWindow, p.L - w * kWindow);
  const int ru = count[2 * kRead + parity], fu = count[2 * kWrite + parity];
  Stream s;
  s.mode = word.mode();
  s.rc = word.read_chunk();
  s.fc = word.fold_chunk();
  s.ru = p.read.list >= 0 ? ru : tokens;
  s.fu = p.write.list >= 0 ? fu : tokens;
  s.nr = word.reads();
  s.n = s.nr + word.folds();
  s.image = s.nr > 0 ? image_bytes(s.mode) : 0;
  return s;
}

//: STEP `t`'s word of window `w`'s record, staged by parity
template <class P>
__device__ __forceinline__ StepWord step_of(const uint8_t* smem, int w, int t) {
  return StepWord{*reinterpret_cast<const volatile uint16_t*>(
      smem + P::kPlans + 4 * kPlanWords * parity_of(w) + 4 + 2 * t)};
}

//: ENTRY `t` of a window's stream, two words as a consumer re-extracts them: x = side | first / 64 << 1 | slots << 9,
//: y = its union's slots -- the read chunks first.
struct Entry {
  uint32_t x, y;

  __device__ __forceinline__ int side() const { return static_cast<int>(x & 1u); }

  __device__ __forceinline__ int first() const {
    return kListBound * static_cast<int>((x >> 1) & 0xffu);
  }

  __device__ __forceinline__ int slots() const { return static_cast<int>(x >> 9); }

  __device__ __forceinline__ int total() const { return static_cast<int>(y); }
};

__device__ __forceinline__ Entry entry_at(const Stream& s, int t) {
  const bool read = t < s.nr;
  const int chunk = read ? s.rc : s.fc, total = read ? s.ru : s.fu;
  const int first = (read ? t : t - s.nr) * chunk;
  return Entry{static_cast<uint32_t>(read ? kRead : kWrite)
                   | static_cast<uint32_t>(first / kListBound) << 1
                   | static_cast<uint32_t>(min(chunk, total - first)) << 9,
               static_cast<uint32_t>(total)};
}

}  // namespace rola::carry
