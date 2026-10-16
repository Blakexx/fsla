// csrc/rola/src/carry/schedule.cuh -- THE WINDOWS' PLANS: each window a sequence of ITEMS, (phase, chunk) pairs -- the
// maxima, the store, a chunk's readout, its drain, a chunk's fold, the boundary -- each buffer placed where its region
// frees earliest and each fill issued once its region is free, its chunks the largest that fit; made by the plan pass
// before the carry runs. see docs/internals/carry/schedule.md
#pragma once

#include "carry/parts.cuh"

namespace rola::carry {

//: AN ITEM'S KIND and FLAGS (its buffer filled by a fill; the window's units step), and a window's IMAGE MODE: whole, a ring of two 8-digit batches, one 16-digit half.
//: -- see docs/internals/carry/schedule.md#items
enum ItemKind : uint32_t {
  kItemMaxima,
  kItemStore,
  kItemRead,
  kItemDrain,
  kItemFold,
  kItemBoundary
};

enum ItemFlag : uint32_t { kFlagFilled = 1u, kFlagUnits = 2u };

enum ImageMode : uint32_t { kModeWhole, kModeRing8, kModeRing16, kImageModes };

//: an index naming no item; a fill's target the next window's first read, which is its item `kFirstRead`; the store,
//: whose work stages the next window's inputs, item `kStoreItem`
constexpr uint32_t kNoItem = 31u, kNextItem = 30u;
constexpr int kStoreItem = 1, kFirstRead = 2;
static_assert(kPlanItems < static_cast<int>(kNextItem),
              "an item index fits five bits, beside its two names");

__host__ __device__ __forceinline__ constexpr uint32_t bits_of(uint32_t v, int at, int n) {
  return (v >> at) & ((1u << n) - 1u);
}

//: AN ITEM, a uint2: x = kind | flags << 3 | first / 64 << 7 | slots << 10 | dep << 20 | fill0 << 25 | fills << 30;
//: y = its buffer (bytes / 16) | fwait0 << 14 | fill1 << 19 | fwait1 << 24 -- a dep or a fill's wait an item whose
//: done it waits for, a fill an item whose buffer it issues, `fills` how many it hosts. A WINDOW'S HEAD, a uint4: x = items | mode << 5 | rchunk << 7 | fchunk << 17;
//: y = runion | funion << 10; z = serial | late << 1 (its order); w = 0, the head a 16-byte quantum.
struct Item {
  uint32_t x, y;

  __device__ __forceinline__ uint32_t kind() const { return bits_of(x, 0, 3); }

  __device__ __forceinline__ uint32_t flags() const { return bits_of(x, 3, 4); }

  __device__ __forceinline__ int first() const {
    return kListBound * static_cast<int>(bits_of(x, 7, 3));
  }

  __device__ __forceinline__ int slots() const { return static_cast<int>(bits_of(x, 10, 10)); }

  __device__ __forceinline__ uint32_t dep() const { return bits_of(x, 20, 5); }

  __device__ __forceinline__ uint32_t buf() const { return bits_of(y, 0, 14); }

  __device__ __forceinline__ int fills() const { return static_cast<int>(bits_of(x, 30, 2)); }

  __device__ __forceinline__ uint32_t fill(int f) const {
    return f == 0 ? bits_of(x, 25, 5) : bits_of(y, 19, 5);
  }

  __device__ __forceinline__ uint32_t fwait(int f) const {
    return f == 0 ? bits_of(y, 14, 5) : bits_of(y, 24, 5);
  }
};

struct Head {
  uint4 v;

  __device__ __forceinline__ int items() const { return static_cast<int>(bits_of(v.x, 0, 5)); }

  __device__ __forceinline__ uint32_t mode() const { return bits_of(v.x, 5, 2); }

  __device__ __forceinline__ int rchunk() const { return static_cast<int>(bits_of(v.x, 7, 10)); }

  __device__ __forceinline__ int runion() const { return static_cast<int>(bits_of(v.y, 0, 10)); }

  __device__ __forceinline__ int funion() const { return static_cast<int>(bits_of(v.y, 10, 10)); }
};

//: A READ BUFFER'S BYTES and a FOLD BUFFER'S, at a chunk's slots and entries (`read_ents`, `fold_ents`).
template <class P>
__host__ __device__ constexpr int read_bytes(int slots, int entries) {
  return read_ents<P>(slots) + align_of(entries * 8, 128);
}

template <class P>
__host__ __device__ constexpr int fold_bytes(int slots, int entries) {
  return fold_ents<P>(slots) + align_of(entries * 8, 128);
}

//: THE LEAST PLAN fits every window: the ring and one 64-slot read buffer of lists full on every token; one such fold
//: buffer.
template <class P>
constexpr bool least_plan_fits() {
  return kImageRing + read_bytes<P>(kListBound, kListBound * kBoxDigits) <= P::kArea
         && fold_bytes<P>(kListBound, kListBound * kBoxLists) <= P::kArea;
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

//: A WINDOW'S READ CHUNK, from its own counts alone (so the window before agrees on its first buffer), beside the
//: ring; its first buffer's bytes.
template <class P>
__device__ __forceinline__ int read_chunk_of(int ru, const int (&rs)[kListAt]) {
  return largest_chunk(ru, rs, P::kArea - kImageRing,
                       [](int s, int e) { return read_bytes<P>(s, e); });
}

template <class P>
__device__ __forceinline__ int first_read_bytes(int ru, const int (&rs)[kListAt]) {
  const int first = min(read_chunk_of<P>(ru, rs), ru);
  return read_bytes<P>(first, entries_of(rs, 0, first, ru));
}

//: A WINDOW'S COUNTS: each side's union and segment sums, and the next window's first read buffer's bytes.
struct Counts {
  int ru, fu, rnext;
  int rs[kListAt], fs[kListAt];
};

//: A WINDOW'S FORM: its image mode, its fold chunk, its order -- read chunks each drained before the next is read
//: (`serial`), the last drain after the fold's first chunk (`late`).
struct Form {
  uint32_t mode;
  int fc;
  bool serial, late;
};

//: THE RULE'S FORM: the image whole where it fits beside the read buffers in flight, else in 16-digit halves; the fold's
//: largest chunk in the area; the read chunks pipelined (`build` falls back to serial where they do not place).
template <class P>
__device__ __forceinline__ Form rule_form(const Counts& c) {
  const int rc = c.ru > 0 ? read_chunk_of<P>(c.ru, c.rs) : 0;
  const int nr = c.ru > 0 ? (c.ru + rc - 1) / rc : 0;
  const int most =
      nr > 0 ? read_bytes<P>(min(rc, c.ru),
                             rc < c.ru ? chunk_max(c.rs, rc / kListBound, c.ru) : c.rs[kListAt - 1])
             : 0;
  const bool whole = nr > 0 && kImageWhole + (nr > 1 ? 2 : 1) * most <= P::kArea;
  const int fc = c.fu > 0 ? largest_chunk(c.fu, c.fs, P::kArea,
                                          [](int s, int e) { return fold_bytes<P>(s, e); })
                          : 0;
  return Form{whole ? kModeWhole : kModeRing16, fc, false, false};
}

//: A NAMED CANDIDATE's form (`prefer`: an image mode by a fold chunk -- the union, 256, 192, 128, 64 slots -- by an
//: order), false where it names no form of the window distinct from another.
constexpr int kFoldOptions = 5, kOrders = 4;
static_assert(kCandidates == kImageModes * kFoldOptions * kOrders, "the named candidates");

__device__ __forceinline__ bool named_form(int cand, const Counts& c, Form& f) {
  const int fopt = cand / kOrders % kFoldOptions, order = cand % kOrders;
  f = Form{static_cast<uint32_t>(cand / (kFoldOptions * kOrders)),
           fopt == 0 ? c.fu : kListBound * (kFoldOptions - fopt), order >= 2, (order & 1) != 0};
  return cand >= 0 && cand < kCandidates && (fopt == 0 || f.fc < c.fu);
}

//: A THREAD'S PLAN in shared memory while it is built: its items, the live regions (bytes / 16 and last user), each
//: chunk's items.
constexpr int kRegions = 2 + 2 * kListAt;

struct Sim {
  uint2 items[kPlanItems];
  int16_t lo[kRegions], hi[kRegions];
  int8_t last[kRegions];
  uint8_t read_at[kListAt], drain_at[kListAt], fold_at[kListAt];
};

//: A WINDOW'S PLAN in a form: its items in order, each buffer placed (`place`), each fill hosted (`host`); false where
//: the form does not fit. -- see docs/internals/carry/schedule.md#planner
template <class P>
__device__ __forceinline__ bool build(Sim& s, const Form& f, const Counts& c, uint4& head) {
  constexpr int B = P::kArea;
  const int rc = c.ru > 0 ? read_chunk_of<P>(c.ru, c.rs) : 0;
  const int nr = c.ru > 0 ? (c.ru + rc - 1) / rc : 0;
  const int fc = f.fc, nf = c.fu > 0 ? (c.fu + fc - 1) / fc : 0;
  const uint32_t mode = f.mode;
  const bool serial = f.serial, late = f.late;
  const int ibytes = mode == kModeWhole ? kImageWhole : kImageRing;
  //: a chunk's buffer's bytes, at its own slots and entries
  const auto bytes_of = [&](uint32_t kind, int first, int slots) {
    return kind == kItemRead ? read_bytes<P>(slots, entries_of(c.rs, first, slots, c.ru))
                             : fold_bytes<P>(slots, entries_of(c.fs, first, slots, c.fu));
  };
  const int r0 = nr > 0 ? bytes_of(kItemRead, 0, min(rc, c.ru)) : 0;
  if ((mode == kModeWhole && nr == 0) || (serial && nr <= 1) || (late && (nr == 0 || nf == 0))
      || (nr > 0 && ibytes + r0 > B))
    return false;

  //: THE ITEMS: the maxima, the store, each read chunk with a drain, each fold chunk, the boundary
  int n = 0;
  const auto put = [&](uint32_t kind, uint32_t flags, int first, int slots, uint32_t dep) {
    s.items[n] = make_uint2(kind | flags << 3 | static_cast<uint32_t>(first / kListBound) << 7
                                | static_cast<uint32_t>(slots) << 10 | dep << 20 | kNoItem << 25,
                            kNoItem << 14 | kNoItem << 19 | kNoItem << 24);
    return static_cast<uint8_t>(n++);
  };
  const auto drain = [&](int q) {
    s.drain_at[q] = put(kItemDrain, 0u, q * rc, min(rc, c.ru - q * rc), s.read_at[q]);
  };
  put(kItemMaxima, 0u, 0, 0, kNoItem);
  put(kItemStore, 0u, 0, 0, kNoItem);
#pragma unroll 1
  for (int q = 0; q < nr; ++q) {
    s.read_at[q] = put(kItemRead, kFlagFilled, q * rc, min(rc, c.ru - q * rc),
                       mode == kModeWhole ? 1u : kNoItem);
    if (serial && !(late && q == nr - 1)) drain(q);
    if (!serial && q >= 1) drain(q - 1);
  }
  if (!serial && nr > 0 && !late) drain(nr - 1);
#pragma unroll 1
  for (int j = 0; j < nf; ++j) {
    s.fold_at[j] = put(kItemFold, kFlagFilled | (j == 0 ? kFlagUnits : 0u), j * fc,
                       min(fc, c.fu - j * fc), kNoItem);
    if (j == 0 && late) drain(nr - 1);
  }
  const int boundary = put(kItemBoundary, nf == 0 ? kFlagUnits : 0u, 0, 0, kNoItem);

  //: THE REGIONS: the image to the last read, the first read buffer (at the top, filled by the window before) to its
  //: drain, then each buffer placed
  int nreg = 0;
  const auto region = [&](int lo, int bytes, int last) {
    s.lo[nreg] = static_cast<int16_t>(lo / 16);
    s.hi[nreg] = static_cast<int16_t>((lo + bytes) / 16);
    s.last[nreg] = static_cast<int8_t>(last);
    ++nreg;
  };
  const auto set_buf = [&](int k, int lo) {
    s.items[k].y = (s.items[k].y & ~0x3fffu) | static_cast<uint32_t>(lo / 16);
  };
  if (nr > 0) {
    region(0, ibytes, s.read_at[nr - 1]);
    region(B - r0, r0, s.drain_at[0]);
    set_buf(s.read_at[0], B - r0);
    set_buf(s.drain_at[0], B - r0);
  }
  //: A BUFFER'S PLACE: of the area's ends and the regions' edges, the one whose overlapping regions' last user is
  //: earliest (none in use by its consumer), the lowest among equals; `wait` that user, -1 for none
  const auto place = [&](int bytes, int consumer, int& wait) {
    int best = -1, best_wait = 1 << 30;
    const auto consider = [&](int lo) {
      if (lo < 0 || lo + bytes > B) return;
      int w = -1;
#pragma unroll 1
      for (int r = 0; r < nreg; ++r) {
        if (16 * s.lo[r] < lo + bytes && lo < 16 * s.hi[r]) {
          if (s.last[r] >= consumer) return;
          w = max(w, static_cast<int>(s.last[r]));
        }
      }
      if (w < best_wait || (w == best_wait && lo < best)) {
        best = lo;
        best_wait = w;
      }
    };
    consider(0);
    consider(B - bytes);
#pragma unroll 1
    for (int r = 0; r < nreg; ++r) {
      consider(16 * s.hi[r]);
      consider(16 * s.lo[r] - bytes);
    }
    wait = best_wait;
    return best;
  };
  //: A FILL'S HOST: the first item past its region's last user and at `from` or later, at most two fills an item; that
  //: user flagged done
  const auto host = [&](uint32_t target, int wait, int consumer, int from) {
    const uint32_t fw = wait < 0 ? kNoItem : static_cast<uint32_t>(wait);
#pragma unroll 1
    for (int h = max(from, wait + 1); h <= consumer; ++h) {
      uint2& it = s.items[h];
      if (bits_of(it.x, 25, 5) == kNoItem) {
        it.x = (it.x & ~(31u << 25)) | target << 25 | 1u << 30;
        it.y = (it.y & ~(31u << 14)) | fw << 14;
      } else if (bits_of(it.y, 19, 5) == kNoItem) {
        it.y = (it.y & ~(0x3ffu << 19)) | target << 19 | fw << 24;
        it.x = (it.x & ~(3u << 30)) | 2u << 30;
      } else {
        continue;
      }
      return true;
    }
    return false;
  };
#pragma unroll 1
  for (int q = 1; q < nr; ++q) {
    int wait = 0;
    const int bytes = bytes_of(kItemRead, q * rc, min(rc, c.ru - q * rc));
    const int at = place(bytes, s.read_at[q], wait);
    if (at < 0) return false;
    region(at, bytes, s.drain_at[q]);
    set_buf(s.read_at[q], at);
    set_buf(s.drain_at[q], at);
    if (!host(s.read_at[q], wait, s.read_at[q], kStoreItem)) return false;
  }
#pragma unroll 1
  for (int j = 0; j < nf; ++j) {
    int wait = 0;
    const int bytes = bytes_of(kItemFold, j * fc, min(fc, c.fu - j * fc));
    const int at = place(bytes, s.fold_at[j], wait);
    if (at < 0) return false;
    region(at, bytes, s.fold_at[j]);
    set_buf(s.fold_at[j], at);
    if (!host(s.fold_at[j], wait, s.fold_at[j], kStoreItem)) return false;
  }
  //: THE BOUNDARY: the next window's first read buffer at the top, once its region's last user here is done and past
  //: the store (a host's fills run before its work, and the store's stages the inputs the fill reads)
  if (c.rnext > 0) {
    int wait = -1;
#pragma unroll 1
    for (int r = 0; r < nreg; ++r)
      if (16 * s.hi[r] > B - c.rnext) wait = max(wait, static_cast<int>(s.last[r]));
    if (!host(kNextItem, wait, boundary, kStoreItem + 1)) return false;
  }

  head = make_uint4(static_cast<uint32_t>(n) | mode << 5 | static_cast<uint32_t>(rc) << 7
                        | static_cast<uint32_t>(fc) << 17,
                    static_cast<uint32_t>(c.ru) | static_cast<uint32_t>(c.fu) << 10,
                    static_cast<uint32_t>(serial) | static_cast<uint32_t>(late) << 1, 0u);
  return true;
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

//: THE PLAN PASS, a thread a (batch-head, box class, window): the preferred form where it builds, else the rule's
//: (its reads pipelined, else serial), its head then its items written to `p.plans`.
//: -- see docs/internals/carry/schedule.md#pass
constexpr int kPlanThreads = 64;

template <class Arm>
__global__ __launch_bounds__(kPlanThreads) void window_plan_kernel(
    const __grid_constant__ CarryParams p) {
  using P = Plan<Arm>;
  static_assert(least_plan_fits<P>(), "the least plan fits the area");
  __shared__ Sim sims[kPlanThreads];
  const int k = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (k >= box_classes(p) * p.windows) return;
  const int w = k % p.windows, cls = k / p.windows, bh = static_cast<int>(blockIdx.y);
  Sim& s = sims[threadIdx.x];
  Counts c;
  window_counts(p, bh, w, cls, c);
  c.rnext = 0;
  if (w + 1 < p.windows) {
    Counts nx;
    window_counts(p, bh, w + 1, cls, nx);
    if (nx.ru > 0) c.rnext = first_read_bytes<P>(nx.ru, nx.rs);
  }
  uint4 head;
  Form f;
  bool built = named_form(p.prefer, c, f) && build<P>(s, f, c, head);
  if (!built) {
    f = rule_form<P>(c);
    built = build<P>(s, f, c, head);
  }
  if (!built) {
    f.serial = true;
    build<P>(s, f, c, head);
  }
  uint4* const out = reinterpret_cast<uint4*>(
      p.plans + ((static_cast<int64_t>(bh) * p.windows + w) * box_classes(p) + cls) * kPlanBytes);
  //: the whole record, past its items zero, as the carry stages it whole
  out[0] = head;
  uint2* const items = reinterpret_cast<uint2*>(out + 1);
#pragma unroll 1
  for (int i = 0; i < (kPlanBytes - 16) / 8; ++i)
    items[i] = i < static_cast<int>(bits_of(head.x, 0, 5)) ? s.items[i] : make_uint2(0u, 0u);
}

//: THE CTA'S PLAN for window `w`, by parity, staged a window ahead: its head, its item count and its item `k`, read
//: at each use.
template <class P>
__device__ __forceinline__ Head plan_head(const uint8_t* smem, int w) {
  const volatile uint4* at =
      reinterpret_cast<const volatile uint4*>(smem + P::kPlans + (w & 1) * kPlanBytes);
  return Head{make_uint4(at->x, at->y, at->z, at->w)};
}

template <class P>
__device__ __forceinline__ int plan_items(const uint8_t* smem, int w) {
  return static_cast<int>(bits_of(
      *reinterpret_cast<const volatile uint32_t*>(smem + P::kPlans + (w & 1) * kPlanBytes), 0, 5));
}

template <class P>
__device__ __forceinline__ Item plan_item(const uint8_t* smem, int w, int k) {
  const volatile uint2* at =
      reinterpret_cast<const volatile uint2*>(smem + P::kPlans + (w & 1) * kPlanBytes + 16) + k;
  return Item{at->x, at->y};
}

}  // namespace rola::carry
