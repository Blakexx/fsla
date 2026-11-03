// measure/lab/experiments/carry_window.cu -- QUESTION: what does the carry's window cost a CTA on sm_86 in each form
// the critical-path design admits, phase by phase in cycles and registers, against the production kernel on the same
// inputs? METHOD: a real call captured (`carry_window.inputs.py`: nl64k-alt-k4's first 32 windows, and the same with
// its lists thinned to 1/16), its buffers re-hosted and its operand block's pointers moved to them; each form launched
// over the call's grid with the phase ledger bound, its readout's output checked bit for bit against the call's.
// Forms: 0 the production kernel (its plan pass, then the carry). see docs/lab/carry_window.md
#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "carry/carry_arm.cuh"
#include "lab.cuh"

namespace carry_window {

namespace rc = rola::carry;
using Arm = rola::Arm<0, 32, 1, 32, 1, 32, 0>;
using P = rc::Plan<Arm>;
//: the phase trace's laps a window and its start's phase (`lap`)
constexpr int kLapsAWindow = 32, kStart = 15, kEnd = 14;

//: A CAPTURED CALL, re-hosted: its operand block with every pointer moved to this process's copy of its buffer, the
//: grid, and the readout's output buffer with the call's result beside it.
struct Call {
  rc::CarryCapture cap;
  uint32_t* num = nullptr;
  float* check = nullptr;
  std::vector<uint32_t> want;
  size_t num_bytes = 0;
  int moved = 0;
};

inline std::vector<char> slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "carry_window: cannot read %s\n", path.c_str());
    std::exit(1);
  }
  return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

inline Call load(const std::string& dir) {
  Call c;
  const std::vector<char> cap = slurp(dir + "/capture.bin");
  if (cap.size() != sizeof(rc::CarryCapture)
      || reinterpret_cast<const rc::CarryCapture*>(cap.data())->bytes != sizeof(rc::CarryCapture)) {
    std::fprintf(stderr, "carry_window: the capture is %zu bytes, this build's is %zu\n",
                 cap.size(), sizeof(rc::CarryCapture));
    std::exit(1);
  }
  std::memcpy(&c.cap, cap.data(), sizeof c.cap);
  std::istringstream arm(std::string(slurp(dir + "/arm.txt").data()));
  int key[6];
  for (int& k : key) arm >> k;
  if (key[0] != Arm::fold_rows || key[1] != Arm::fold_factors || key[2] != Arm::read_rows
      || key[3] != Arm::read_factors || key[4] != Arm::channels || key[5] != Arm::publish_regs) {
    std::fprintf(stderr, "carry_window: the call's arm is not this build's\n");
    std::exit(1);
  }
  std::istringstream lines(std::string(slurp(dir + "/buffers.txt").data()));
  uint64_t* words = reinterpret_cast<uint64_t*>(&c.cap.p);
  //: each word matched against the capture's addresses once: a moved pointer may land in a later buffer's old range
  constexpr size_t kWords = sizeof(rc::CarryParams) / 8;
  std::vector<uint64_t> was(words, words + kWords);
  std::vector<bool> moved(kWords, false);
  int k;
  std::string role;
  uint64_t base, bytes;
  while (lines >> k >> role >> base >> bytes) {
    const std::vector<char> data = slurp(dir + "/buffer_" + std::to_string(k) + ".bin");
    char* dev = nullptr;
    LAB_CHECK(cudaMalloc(&dev, bytes));
    LAB_CHECK(cudaMemcpy(dev, data.data(), bytes, cudaMemcpyHostToDevice));
    for (size_t i = 0; i < kWords; ++i)
      if (!moved[i] && was[i] >= base && was[i] < base + bytes) {
        words[i] = reinterpret_cast<uint64_t>(dev) + (was[i] - base);
        moved[i] = true;
        ++c.moved;
      }
    if (role == "num") {
      c.want.resize(bytes / 4);
      std::memcpy(c.want.data(), data.data(), bytes);
      c.num_bytes = bytes;
      c.num = reinterpret_cast<uint32_t*>(dev);
    } else if (role == "check") {
      c.check = reinterpret_cast<float*>(dev);
    }
  }
  if (c.num != c.cap.p.num || c.check != c.cap.p.check) {
    std::fprintf(stderr, "carry_window: the readout's output was not moved to its copy\n");
    std::exit(1);
  }
  //: the read tables on this card, as the production launch lays them out (the capture precedes that)
  c.cap.p.tables = rc::tables_of<P::kWindow>(c.cap.p.tables.heads, P::kSmemBytes);
  return c;
}

//: THE LEDGER'S READING: the mean over CTAs of the busiest warp's cycles (a trace's: the CTA's span), a window -- and each phase's, the mean warp's
struct Reading {
  double window = 0.0;
  double phase[rc::kCarryPhases] = {};
  double busiest[rc::kCarryPhases] = {};
};

//: A TRACE'S READING (the lab's forms): each phase the stamps' differences, a warp's laps in order from its start
inline Reading read_trace(const long long* trace, int ctas, int windows) {
  const size_t per = static_cast<size_t>(windows) * kLapsAWindow + 2;
  std::vector<long long> h(static_cast<size_t>(ctas) * rc::kWarps * per);
  LAB_CHECK(cudaMemcpy(h.data(), trace, h.size() * 8, cudaMemcpyDeviceToHost));
  Reading r;
  for (int c = 0; c < ctas; ++c) {
    double most[rc::kCarryPhases] = {};
    long long first = 0, last = 0;
    for (int w = 0; w < rc::kWarps; ++w) {
      const long long* e = h.data() + (static_cast<size_t>(c) * rc::kWarps + w) * per;
      double mine[rc::kCarryPhases] = {};
      size_t k = 1;
      for (; k < per && e[k] != 0; ++k)
        if ((e[k] & 15) < rc::kCarryPhases)
          mine[e[k] & 15] += double((e[k] >> 4) - (e[k - 1] >> 4));
      first = w == 0 ? e[0] >> 4 : std::min(first, e[0] >> 4);
      last = std::max(last, e[k - 1] >> 4);
      for (int ph = 0; ph < rc::kCarryPhases; ++ph) {
        r.phase[ph] += mine[ph];
        most[ph] = std::max(most[ph], mine[ph]);
      }
    }
    r.window += double(last - first);
    for (int ph = 0; ph < rc::kCarryPhases; ++ph) r.busiest[ph] += most[ph];
  }
  r.window /= double(ctas) * windows;
  for (double& x : r.phase) x /= double(ctas) * rc::kWarps * windows;
  for (double& x : r.busiest) x /= double(ctas) * windows;
  return r;
}

inline Reading read_ledger(const long long* ledger, int ctas, int windows) {
  std::vector<long long> h(static_cast<size_t>(ctas) * rc::kWarps * rc::kCarryPhases);
  LAB_CHECK(cudaMemcpy(h.data(), ledger, h.size() * 8, cudaMemcpyDeviceToHost));
  Reading r;
  for (int c = 0; c < ctas; ++c) {
    double busiest = 0.0, most[rc::kCarryPhases] = {};
    for (int w = 0; w < rc::kWarps; ++w) {
      double sum = 0.0;
      for (int ph = 0; ph < rc::kCarryPhases; ++ph) {
        const double x =
            double(h[(static_cast<size_t>(c) * rc::kWarps + w) * rc::kCarryPhases + ph]);
        sum += x;
        r.phase[ph] += x;
        most[ph] = std::max(most[ph], x);
      }
      busiest = std::max(busiest, sum);
    }
    r.window += busiest;
    for (int ph = 0; ph < rc::kCarryPhases; ++ph) r.busiest[ph] += most[ph];
  }
  r.window /= double(ctas) * windows;
  for (double& x : r.phase) x /= double(ctas) * rc::kWarps * windows;
  for (double& x : r.busiest) x /= double(ctas) * windows;
  return r;
}

const char* const kPhases[rc::kCarryPhases] = {"landed", "maxima", "store", "readout", "drain",
                                               "fill",   "fold",   "wait",  "units"};

namespace ops = ::denseref::ops;

//: THE THREAD'S INDEX, opaque, read at each phase so nothing derived from it is held across phases (the register line)
__device__ __forceinline__ int tid() {
  int t;
  asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t));
  return t;
}

//: THE PHASE TRACE: at each lap a warp's lane 0 stores the clock, the phase in its low 4 bits, at the warp's next slot
//: of `p.ledger` (`kLapsAWindow` a window), one predicated store with nothing to wait on; the host takes each phase's
//: cycles from consecutive stamps. Its one register is the warp's lap count. `kStart` marks the start.

__device__ __forceinline__ long long* trace_of(const rc::CarryParams& p, int warp) {
  return p.ledger
         + (static_cast<long long>(blockIdx.y * gridDim.x + blockIdx.x) * rc::kWarps + warp)
               * (static_cast<long long>(p.windows) * kLapsAWindow + 2);
}

template <bool On = true>
__device__ __forceinline__ void lap(const rc::CarryParams& p, int& laps, int ph) {
  if constexpr (!On) return;
  const int t = tid();
  const long long now = clock64();
  if (p.ledger != nullptr && (t & 31) == 0) trace_of(p, t >> 5)[laps] = (now << 4) | ph;
  ++laps;
}

//: A SIDE'S CHUNK, the whole union: its slots and its buffer's bytes (`read_bytes` / `fold_bytes`)
struct Sized {
  int slots, bytes;
};

template <int Side>
__device__ __forceinline__ Sized sized(const rc::CarryParams& p, const uint8_t* smem, int w) {
  const rc::Window win = rc::window_of<P>(p, smem, Side, w);
  const int slots = rc::slots_of<P>(smem, win, Side, w);
  const int entries = win.listed
                          ? static_cast<int>(rc::pre_of<P>(
                                p, smem, Side, w)[rc::kBoxLists * rc::kListAt + rc::kListAt - 1])
                          : 0;
  return Sized{slots, Side == rc::kRead ? rc::read_bytes<P>(slots, entries, true)
                                        : rc::fold_bytes<P>(slots, entries, win.listed)};
}

//: a read buffer's place, at the area's top
__device__ __forceinline__ uint32_t top_of(int bytes) {
  return static_cast<uint32_t>((P::kArea - bytes) / 128 * 128);
}

//: THE NEXT WINDOW'S INPUTS, a window ahead: both sides' lists, the write exponents and the token units
__device__ __forceinline__ void stage(const rc::CarryParams& p, int w, uint8_t* smem) {
  const rc::Place c{static_cast<int>(blockIdx.y), 0};
  const rc::Window rw = rc::window_of<P>(p, smem, rc::kRead, w),
                   ww = rc::window_of<P>(p, smem, rc::kWrite, w);
  rc::lists_in<P>(p, p.read_lists, rw, c, w, smem, rc::kRead);
  rc::lists_in<P>(p, p.write_lists, ww, c, w, smem, rc::kWrite);
  rc::facts_in<P>(p, ww, c, w, smem);
}

__device__ __forceinline__ void read_fill(const rc::CarryParams& p, uint8_t* smem, int w,
                                          uint32_t area) {
  const Sized r = sized<rc::kRead>(p, smem, w);
  const int t = tid();
  const rc::Window rw = rc::window_of<P>(p, smem, rc::kRead, w);
  rc::read_fill<P>(p, smem, rw, rc::Place{static_cast<int>(blockIdx.y), 0}, w,
                   rc::chunk_runs<P>(p, smem, rw, rc::kRead, w, 0, r.slots, r.slots, t & 31), 0,
                   r.slots, area + top_of(r.bytes), t >> 5, t & 31);
}

//: A CHUNK'S ENTRIES IN ONE PASS (parts.cuh's `entries_fill`): a warp's four lists' runs as one, each lane an entry --
//: its list by the runs' running counts, its source and place by its list's -- 8 bytes a lane
template <class P_>
__device__ __forceinline__ void entries_ahead(const rc::CarryParams& p, const rc::SideLists& l,
                                              const rc::Window& win, int w, const rc::Run& run,
                                              uint32_t dst, int warp, int lane) {
  const uint2* const lists =
      l.entries
      + rc::list_row(l, rc::Place{static_cast<int>(blockIdx.y), 0}, p.windows, w, win.digit0)
            * rc::kListCap;
  int at[4], to[4], end[4];
  int total = 0;
  rola::static_for<4>([&](auto J) {
    const int list = warp + rc::kWarps * J;
    at[J] = __shfl_sync(0xffffffffu, run.at, list);
    to[J] = __shfl_sync(0xffffffffu, run.to, list);
    total += __shfl_sync(0xffffffffu, run.n, list);
    end[J] = total;
  });
#pragma unroll 1
  for (int e = lane; e < total; e += 32) {
    const int j = (e >= end[0]) + (e >= end[1]) + (e >= end[2]);
    const int k = e - (j == 0 ? 0 : j == 1 ? end[0] : j == 2 ? end[1] : end[2]);
    const int a = j == 0 ? at[0] : j == 1 ? at[1] : j == 2 ? at[2] : at[3];
    const int o = j == 0 ? to[0] : j == 1 ? to[1] : j == 2 ? to[2] : to[3];
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(dst + 8u * static_cast<uint32_t>(o + k)),
        "l"(lists + (warp + rc::kWarps * j) * rc::kListCap + a + k));
  }
}

//: A READ BUFFER'S PANEL AND DUMMY ROWS cleared by stores, and its entries copied (a warp its own digits' runs);
//: `Ahead` the entries in one pass
template <bool Ahead = false>
__device__ __forceinline__ void panel_entries(const rc::CarryParams& p, uint8_t* smem, int w,
                                              uint32_t buf, int slots) {
  const int t = tid();
  const uint32_t pads = buf + rc::read_pads<P>(slots);
  const int quanta = (rc::read_ents<P>(slots, true) - rc::read_pads<P>(slots)) / 16;
#pragma unroll 4
  for (int k = t; k < quanta; k += rc::kThreads)
    asm volatile("st.shared.v4.u32 [%0], {%1, %1, %1, %1};" ::"r"(pads + 16u * k),
                 "r"(rc::kPanelClear)
                 : "memory");
  const rc::Window rw = rc::window_of<P>(p, smem, rc::kRead, w);
  if (!rw.listed) return;
  const rc::Run run = rc::chunk_runs<P>(p, smem, rw, rc::kRead, w, 0, slots, slots, t & 31);
  if constexpr (Ahead) {
    entries_ahead<P>(p, p.read_lists, rw, w, run, buf + rc::read_ents<P>(slots, true), t >> 5,
                     t & 31);
  } else {
    rc::entries_fill<P>(p, p.read_lists, rw, rc::Place{static_cast<int>(blockIdx.y), 0}, w, run,
                        buf + rc::read_ents<P>(slots, true), t >> 5, t & 31);
  }
}

//: AN ASYNC 16-BYTE COPY issued where `issue`, zero-filled where not `live`
__device__ __forceinline__ void copy16(uint32_t dst, const void* src, bool issue, bool live) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %2, 0;\n@p cp.async.cg.shared.global [%0], [%1], 16, %3;\n}\n" ::
          "r"(dst),
      "l"(src), "r"(static_cast<int>(issue)), "r"(live ? 16 : 0));
}

//: A CHUNK'S ROWS WITH ITS LOADS AHEAD (parts.cuh's `fill`, slices of 16 bytes or more): a thread's slots four at a
//: time, their four tokens loaded before any copy, so a slot's copies wait on no other slot's load
template <int Factors, bool Values>
__device__ __forceinline__ void fill_ahead(const volatile rc::Slices& sl, const uint8_t* values,
                                           int values_st, const uint16_t* tok,
                                           const rc::Window& win, int first, int slots,
                                           uint32_t pool) {
  constexpr int kRow = rc::kSliceBytes * Factors + (Values ? rc::kBoxChannels * 2 : 0);
  constexpr int kQuads = rc::kThreads / 4, kChunks = rc::kSliceBytes / ops::kStageBytes, kAhead = 4;
  const int q = tid() >> 2, r = tid() & 3;
  const uint8_t* base[Factors];
  int st[Factors];
  bool active[Factors];
  rola::static_for<Factors>([&](auto i) {
    if (sl.bytes[i] < ops::kStageBytes) __trap();
    st[i] = sl.st[i];
    base[i] = sl.at[i] + static_cast<int64_t>(win.t0) * st[i] + ops::kStageBytes * r;
    active[i] = ops::kStageBytes * r < sl.bytes[i];
  });
  const uint8_t* const vbase =
      Values ? values + static_cast<int64_t>(win.t0) * values_st + ops::kStageBytes * r : nullptr;
#pragma unroll 1
  for (int s0 = q; s0 < slots; s0 += kAhead * kQuads) {
    int u[kAhead];
    rola::static_for<kAhead>([&](auto j) {
      const int s = s0 + kQuads * j;
      u[j] = s < slots ? (win.listed ? tok[first + s] : first + s) : rc::kGap;
    });
    rola::static_for<kAhead>([&](auto j) {
      const int s = s0 + kQuads * j, t = u[j] & (P::kWindow - 1);
      const bool issue = s < slots, live = u[j] == t;
      const uint32_t row = pool + static_cast<uint32_t>(s * kRow);
      rola::static_for<Factors>([&](auto i) {
        copy16(row
                   + 16u
                         * rc::chunk_at<kRow>(static_cast<uint32_t>(s),
                                              static_cast<uint32_t>(kChunks * i + r)),
               base[i] + t * st[i], issue, live && active[i]);
      });
      if constexpr (Values)
        copy16(row
                   + 16u
                         * rc::chunk_at<kRow>(static_cast<uint32_t>(s),
                                              static_cast<uint32_t>(kChunks * Factors + r)),
               vbase + t * values_st, issue, live);
    });
  }
}

//: A READ BUFFER'S ROWS `s0 .. s1 - 1` (`s0` a multiple of 8, the rows' swizzle period); `Ahead` by `fill_ahead`
template <bool Ahead = false>
__device__ __forceinline__ void read_rows(const rc::CarryParams& p, uint8_t* smem, int w,
                                          uint32_t buf, int s0, int s1) {
  const volatile rc::Addr& addr = *reinterpret_cast<const volatile rc::Addr*>(smem + P::kAddr);
  const rc::Window rw = rc::window_of<P>(p, smem, rc::kRead, w);
  const uint32_t at = buf + static_cast<uint32_t>(s0 * P::kReadRow);
  if constexpr (Ahead) {
    fill_ahead<P::kReadFactors, false>(addr.slices[rc::kRead], nullptr, 0,
                                       rc::union_of<P>(p, smem, rc::kRead, w), rw, s0, s1 - s0, at);
  } else {
    rc::fill<P::kReadFactors, false>(addr.slices[rc::kRead], nullptr, 0,
                                     rc::union_of<P>(p, smem, rc::kRead, w), rw, s0, s1 - s0, at);
  }
}

//: A FOLD BUFFER'S FILL with its loads ahead: its rows (`fill_ahead`), then its entries (fold.cuh's `fold_fill`)
__device__ __forceinline__ void fold_fill_ahead(const rc::CarryParams& p, uint8_t* smem, int w,
                                                uint32_t buf, int slots) {
  const volatile rc::Addr& addr = *reinterpret_cast<const volatile rc::Addr*>(smem + P::kAddr);
  const rc::Window ww = rc::window_of<P>(p, smem, rc::kWrite, w);
  fill_ahead<P::kFoldFactors, true>(addr.slices[rc::kWrite], addr.values,
                                    static_cast<int>(p.values.st),
                                    rc::union_of<P>(p, smem, rc::kWrite, w), ww, 0, slots, buf);
  const int t = tid();
  if (ww.listed)
    entries_ahead<P>(p, p.write_lists, ww, w,
                     rc::chunk_runs<P>(p, smem, ww, rc::kWrite, w, 0, slots, slots, t & 31),
                     buf + rc::fold_ents<P>(slots), t >> 5, t & 31);
}

//: A HALF'S STORES (parts.cuh's `store_batches` for batches `2 HALF` and `2 HALF + 1`, the image's 16 planes): a lane's
//: four lists' units and two digits' exponents read first, its eight scales formed, then its 32 pairs scaled, packed
//: and stored as independent chains
template <int HALF>
__device__ __forceinline__ void store_half(const rc::State<P>& st, const uint8_t* smem,
                                           uint32_t image, int dexp) {
  const int t = tid(), warp = t >> 5, g = (t & 31) >> 2, q = t & 3;
  const volatile rc::UnitBook* const book =
      reinterpret_cast<const volatile rc::UnitBook*>(smem + P::kBook);
  int u[P::kFoldLists], e[2];
  rola::static_for<P::kFoldLists>([&](auto J) { u[J] = book[warp + rc::kWarps * J].unit; });
  rola::static_for<2>(
      [&](auto H) { e[H] = __shfl_sync(0xffffffffu, dexp, rc::exp_slot(16 * HALF + 8 * H + g)); });
  uint32_t base;
  asm volatile("mov.u32 %0, %1;" : "=r"(base) : "r"(rc::unit_of(warp, 0, g)));
  float sc[P::kFoldLists][2];
  uint32_t at[2][4];
  rola::static_for<2>([&](auto H) {
    rola::static_for<P::kFoldLists>(
        [&](auto J) { sc[J][H] = rc::pow2(min(max(u[J] + rc::kImageTop - e[H], -126), 127)); });
    rola::static_for<4>([&](auto N) {
      at[H][N] = image + 4u * q + 16u * (base ^ static_cast<uint32_t>(N))
                 + static_cast<uint32_t>((8 * H + g) * rc::kPlaneBytes);
    });
  });
  rola::static_for<P::kFoldLists>([&](auto J) {
    rola::static_for<2>([&](auto H) {
      rola::static_for<4>([&](auto N) {
        const __half2 h = __floats2half2_rn(st.acc[J][HALF][N][2 * H] * sc[J][H],
                                            st.acc[J][HALF][N][2 * H + 1] * sc[J][H]);
        asm volatile("st.shared.u32 [%0], %1;" ::"r"(at[H][N] + 512u * J),
                     "r"(*reinterpret_cast<const uint32_t*>(&h))
                     : "memory");
      });
    });
  });
}

//: A FOLD over the whole union of a listed side, `hook(J)` run before list `J`'s steps (fold.cuh's `fold_chunk`, its
//: rescale applied)
template <class Hook>
__device__ __forceinline__ void fold_lists(const rc::CarryParams& p, rc::State<P>& st,
                                           const uint8_t* smem, const rc::Window& ww, int w,
                                           int slots, uint32_t rows, int warp, int lane,
                                           Hook hook) {
  const rc::Run run = rc::chunk_runs<P>(p, smem, ww, rc::kWrite, w, 0, slots, slots, lane);
  const uint32_t ents = rows + rc::fold_ents<P>(slots);
  rola::static_for<P::kFoldLists>([&](auto J) {
    hook(J);
    const int list = rc::fold_list(warp, J);
    const rc::ListScale sc = reinterpret_cast<const rc::ListScale*>(smem + P::kScale)[list];
    rola::static_for<P::kFoldMTiles>([&](auto M) {
      rola::static_for<4>(
          [&](auto N) { rola::static_for<4>([&](auto I) { st.acc[J][M][N][I] *= sc.f; }); });
    });
    const int n = __shfl_sync(0xffffffffu, run.n, list),
              to = __shfl_sync(0xffffffffu, run.to, list);
    const auto fetch = [&](int k) {
      uint2 e;
      asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];"
                   : "=r"(e.x), "=r"(e.y)
                   : "r"(ents + 8u * static_cast<uint32_t>(to + k)));
      return e;
    };
    uint32_t off, cpair, dpair = 0u;
    const int e = lane & 15;
    rc::entry_of<P, false>(fetch(max(min(e, n - 1), 0)), e < n, 0, sc.cf, off, cpair, dpair);
    rc::DenColumn<P> none;
    rc::steps<P, J, false>(st, sc.cf, rows, 0, n, lane, off, cpair, 0u, none, fetch);
  });
}

//: FORM 1, OPTION A by hand: a window's maxima, then each image half stored and read out, the fold's rows over the
//: image once every warp has read it, the drain, the next window's read rows at the area's top, the fold. One chunk a
//: side (a window that does not fit traps). FORM 3: the next window's panel zeroed by stores, its entries copied before
//: the fold, its rows in four slices, one before each of the warp's lists' steps. FORM 4: the window's read buffer
//: issued at its top (rows, panel by stores, entries), the drain a warp's share inside the fold before its list
//: `(warp >> 1) & 3` (a scheduler's two warps at different lists). FORM 5: form 4, its fills' loads ahead. CLOCK: the
//: phase trace compiled in (the attribution) or out (the timing). FORM 6 (form 5 with each half's stores in
//: independent chains, `store_half`) is withdrawn from the runs: its stores predate the 16-byte image, which
//: production's `store_batches` now writes in the lean form (2026-10-18).
template <int FORM, bool CLOCK>
__global__ __launch_bounds__(rc::kThreads, 1) void window(
    const __grid_constant__ rc::CarryParams p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t area = ops::smem_addr(smem + P::kAreaAt);
  {
    const int t = tid();
    if (t < rc::kBoxLists)
      reinterpret_cast<rc::UnitBook*>(smem + P::kBook)[t] = rc::UnitBook{rc::kUnitNone, 0.0f};
    if (t < 2 * rc::kBoxDigits) reinterpret_cast<int*>(smem + P::kKeys)[t] = INT_MIN;
    //: the bounds' counters and the units, as the production kernel's prologue; the scales' bits (`list_scale`)
    for (int i = t; i < p.tables.heads * P::kWindow; i += rc::kThreads) {
      reinterpret_cast<uint32_t*>(smem + p.tables.bounds_at)[i] = 0u;
      reinterpret_cast<int8_t*>(smem + p.tables.units_at)[i] =
          static_cast<int8_t>(rc::kUnitByteNone);
    }
    if (t == 0) *reinterpret_cast<uint2*>(smem + P::kBits) = make_uint2(1u << 23, 127u << 23);
    if (t == 0) rc::addr_in(*reinterpret_cast<rc::Addr*>(smem + P::kAddr), p, rc::place_of(p));
  }
  ops::rendezvous();
  int laps = 0;
  lap<true>(p, laps, kStart);
  rc::State<P> st;
#pragma unroll
  for (int k = 0; k < P::kStateRegs; ++k) (&st.acc[0][0][0][0])[k] = 0.0f;
  int dexp = 0;

  stage(p, 0, smem);
  ops::stage_commit();
  ops::stage_wait<0>();
  ops::rendezvous();
  if constexpr (FORM < 4) {
    read_fill(p, smem, 0, area);
    ops::stage_commit();
  }

#pragma unroll 1
  for (int w = 0; w < p.windows; ++w) {
    //: THE WINDOW'S INPUTS AND READ ROWS landed, the next window's staged
    ops::stage_wait<0>();
    ops::rendezvous();
    lap<CLOCK>(p, laps, rc::kPhaseLanded);
    if constexpr (FORM >= 4) {
      //: THE WINDOW'S READ BUFFER, under the maxima and the first half's stores
      const Sized r = sized<rc::kRead>(p, smem, w);
      read_rows<FORM >= 5>(p, smem, w, area + top_of(r.bytes), 0, r.slots);
      panel_entries<FORM >= 5>(p, smem, w, area + top_of(r.bytes), r.slots);
      ops::stage_commit();
      lap<CLOCK>(p, laps, rc::kPhaseFill);
    }
    if (w + 1 < p.windows) stage(p, w + 1, smem);
    ops::stage_commit();
    //: THE DIGIT MAXIMA of the state before the window's writes
    {
      const int t = tid();
      const int top = rc::post_keys<P>(st, smem, w, t >> 5, t & 31);
      lap<CLOCK>(p, laps, rc::kPhaseMaxima);
      ops::rendezvous();
      lap<CLOCK>(p, laps, rc::kPhaseWait);
      rc::post_exps<P>(smem, w, top, t & 31);
      rc::leaf_scales<P>(smem);
      rc::bounds_clear<P>(p, smem);
      dexp = reinterpret_cast<const int*>(smem + P::kExps)[t & 31];
    }
    lap<CLOCK>(p, laps, rc::kPhaseMaxima);
    //: THE HALVES: each stored into the image, then read out
    const Sized r = sized<rc::kRead>(p, smem, w);
    if (r.bytes + rc::kImageRing > P::kArea) __trap();
#pragma unroll 1
    for (int half = 0; half < 2; ++half) {
      if (half > 0) {
        ops::rendezvous();
        lap<CLOCK>(p, laps, rc::kPhaseWait);
      }
      {
        const int t = tid();
        if constexpr (FORM >= 6) {
          if (half == 0) {
            store_half<0>(st, smem, area, dexp);
          } else {
            store_half<1>(st, smem, area, dexp);
          }
        } else {
          rc::store_batches<P, 2 * rc::kBatchPlanes - 1>(st, smem, area, 2 * half, 2 * half + 2,
                                                         t >> 5, t & 31);
        }
      }
      if constexpr (FORM >= 4) {
        if (half == 0) ops::stage_wait<1>();
      }
      lap<CLOCK>(p, laps, rc::kPhaseStore);
      ops::rendezvous();
      lap<CLOCK>(p, laps, rc::kPhaseWait);
      {
        const int t = tid();
        const uint32_t rows = area + top_of(r.bytes), pads = rows + rc::read_pads<P>(r.slots),
                       ents = rows + rc::read_ents<P>(r.slots, true);
        const rc::Window rw = rc::window_of<P>(p, smem, rc::kRead, w);
        const rc::Run run =
            rc::chunk_runs<P>(p, smem, rw, rc::kRead, w, 0, r.slots, r.slots, t & 31);
        //: THE CHUNK'S UNITS, the CTA's own, before its first add (parts.cuh's, as the production window)
        if (half == 0) {
          if constexpr (P::kReadFactors == 1) {
            rc::units_listed_add<P>(p, smem, run, 0, rows, ents, t >> 5, t & 31);
          } else {
            rc::units_dense<P>(p, smem, 0, r.slots, rows, t >> 5, t & 31);
          }
          ops::rendezvous();
          rc::units_close<P>(p, smem, w, 0, r.slots);
          ops::rendezvous();
          lap<CLOCK>(p, laps, rc::kPhaseUnits);
        }
        if (p.readout_f16) {
          rc::read_chunk<P, true, true>(p, smem, rw, run, 0, r.slots, 2 * half, 2 * half + 2, area,
                                        2 * rc::kBatchPlanes - 1, rows, pads, ents, t >> 5, t & 31);
        } else {
          rc::read_chunk<P, false, true>(p, smem, rw, run, 0, r.slots, 2 * half, 2 * half + 2, area,
                                         2 * rc::kBatchPlanes - 1, rows, pads, ents, t >> 5,
                                         t & 31);
        }
      }
      lap<CLOCK>(p, laps, rc::kPhaseReadout);
    }
    ops::rendezvous();
    lap<CLOCK>(p, laps, rc::kPhaseWait);
    //: THE FOLD'S ROWS, over the image every warp has read
    const Sized f = sized<rc::kWrite>(p, smem, w);
    if (f.bytes > static_cast<int>(top_of(r.bytes))) __trap();
    {
      const int t = tid();
      if constexpr (FORM >= 5) {
        fold_fill_ahead(p, smem, w, area, f.slots);
      } else {
        const rc::Window ww = rc::window_of<P>(p, smem, rc::kWrite, w);
        rc::fold_fill<P>(p, smem, ww, rc::Place{static_cast<int>(blockIdx.y), 0}, w,
                         rc::chunk_runs<P>(p, smem, ww, rc::kWrite, w, 0, f.slots, f.slots, t & 31),
                         0, f.slots, area, t >> 5, t & 31);
      }
    }
    ops::stage_commit();
    lap<CLOCK>(p, laps, rc::kPhaseFill);
    if constexpr (FORM >= 4) {
      ops::stage_wait<0>();
      ops::rendezvous();
      lap<CLOCK>(p, laps, rc::kPhaseWait);
      //: THE FOLD, each warp's share of the drain before its list `warp & 3`
      {
        const int t = tid();
        const rc::Window ww = rc::window_of<P>(p, smem, rc::kWrite, w);
        if (!ww.listed) __trap();
        rc::fold_units<P>(smem, ww, w, t >> 5, t & 31);
        fold_lists(p, st, smem, ww, w, f.slots, area, t >> 5, t & 31, [&](auto J) {
          if (((tid() >> 6) & (P::kFoldLists - 1)) == decltype(J)::value)
            rc::drain<P>(p, smem, rc::window_of<P>(p, smem, rc::kRead, w),
                         rc::Place{static_cast<int>(blockIdx.y), 0}, w, 0, r.slots,
                         area + top_of(r.bytes) + rc::read_pads<P>(r.slots) + rc::kPadBytes);
        });
      }
      lap<CLOCK>(p, laps, rc::kPhaseFold);
    } else {
      //: THE DRAIN, the panel to `num`
      rc::drain<P>(p, smem, rc::window_of<P>(p, smem, rc::kRead, w),
                   rc::Place{static_cast<int>(blockIdx.y), 0}, w, 0, r.slots,
                   area + top_of(r.bytes) + rc::read_pads<P>(r.slots) + rc::kPadBytes);
      lap<CLOCK>(p, laps, rc::kPhaseDrain);
      //: THE NEXT WINDOW'S READ ROWS, at the top once every warp has drained (its inputs and the fold's rows landed)
      ops::stage_wait<0>();
      ops::rendezvous();
      lap<CLOCK>(p, laps, rc::kPhaseWait);
      const bool next = w + 1 < p.windows;
      const Sized n = next ? sized<rc::kRead>(p, smem, w + 1) : Sized{0, 0};
      if (next && f.bytes > static_cast<int>(top_of(n.bytes))) __trap();
      const uint32_t nbuf = area + top_of(n.bytes);
      if constexpr (FORM == 1) {
        if (next) read_fill(p, smem, w + 1, area);
        ops::stage_commit();
        lap<CLOCK>(p, laps, rc::kPhaseFill);
        //: THE FOLD, the window's units stepped first
        {
          const int t = tid();
          const rc::Window ww = rc::window_of<P>(p, smem, rc::kWrite, w);
          rc::fold_units<P>(smem, ww, w, t >> 5, t & 31);
          rc::fold_chunk<P>(
              st, true, smem, ww, w,
              rc::chunk_runs<P>(p, smem, ww, rc::kWrite, w, 0, f.slots, f.slots, t & 31), 0,
              f.slots, area, area + rc::fold_ents<P>(f.slots), t >> 5, t & 31,
              rc::den_lists(rc::place_of(p).channel0, p.dv), rc::den_state_of(p));
        }
      } else {
        //: THE NEXT WINDOW'S PANEL, zeroed by stores, and its entries
        if (next) panel_entries(p, smem, w + 1, nbuf, n.slots);
        lap<CLOCK>(p, laps, rc::kPhaseFill);
        //: THE FOLD, the window's units stepped first, the next window's rows a slice before each list
        {
          const int t = tid();
          const rc::Window ww = rc::window_of<P>(p, smem, rc::kWrite, w);
          if (!ww.listed) __trap();
          rc::fold_units<P>(smem, ww, w, t >> 5, t & 31);
          fold_lists(p, st, smem, ww, w, f.slots, area, t >> 5, t & 31, [&](auto J) {
            if (next) {
              constexpr int j = decltype(J)::value;
              const int s0 = (n.slots * j / P::kFoldLists) & ~7,
                        s1 = j + 1 == P::kFoldLists ? n.slots
                                                    : (n.slots * (j + 1) / P::kFoldLists) & ~7;
              read_rows(p, smem, w + 1, nbuf, s0, s1);
            }
          });
        }
        ops::stage_commit();
      }
      lap<CLOCK>(p, laps, rc::kPhaseFold);
    }
  }

  lap<true>(p, laps, kEnd);
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < P::kStateRegs; ++k) sum += (&st.acc[0][0][0][0])[k];
  //: the check a warp's: its lanes' sums joined by shuffles, one add a warp (`carry_kernel.md#check`)
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  if ((tid() & 31) == 0) atomicAdd(p.check, sum);
}

template <int FORM, bool CLOCK>
void launch_form(const Call& c, const rc::CarryParams& p) {
  static const bool sized =
      cudaFuncSetAttribute(window<FORM, CLOCK>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           P::kSmemBytes)
      == cudaSuccess;
  if (!sized) std::exit(2);
  window<FORM, CLOCK><<<dim3(static_cast<unsigned>(c.cap.boxes), static_cast<unsigned>(c.cap.bh)),
                        rc::kThreads, P::kSmemBytes>>>(p);
  LAB_CHECK(cudaGetLastError());
}

//: a form's launch, its clock compiled in or out
template <int FORM>
void launch_either(const Call& c, const rc::CarryParams& p, bool clock) {
  if (clock) {
    launch_form<FORM, true>(c, p);
  } else {
    launch_form<FORM, false>(c, p);
  }
}

//: A FORM'S READINGS: its launch timed with every clock off (the result), then its phases with its clock on (the
//: attribution, from its trace or the production kernel's ledger); its readout output checked after each against the
//: call's
template <class Launch>
void form(const lab::Bench& b, Call& c, const char* name, const char* cell, Launch launch,
          bool traced = true) {
  const int ctas = static_cast<int>(c.cap.boxes * c.cap.bh), windows = c.cap.p.windows;
  const size_t words =
      static_cast<size_t>(ctas) * rc::kWarps
      * (traced ? static_cast<size_t>(windows) * kLapsAWindow + 2 : rc::kCarryPhases);
  long long* ledger = nullptr;
  LAB_CHECK(cudaMalloc(&ledger, sizeof(long long) * words));
  const auto reset = [&] {
    LAB_CHECK(cudaMemset(ledger, 0, sizeof(long long) * words));
    LAB_CHECK(cudaMemset(c.num, 0, c.num_bytes));
    LAB_CHECK(cudaMemset(c.check, 0, 4));
  };
  //: the readout's output against the call's: the words that differ, and by most x 2^-16 of the token's largest
  const auto compare = [&] {
    std::vector<uint32_t> got(c.want.size());
    LAB_CHECK(cudaMemcpy(got.data(), c.num, c.num_bytes, cudaMemcpyDeviceToHost));
    //: the call's num is a floating sum across CTAs (f16 pairs, or fp32), its order the hardware's, so a form matches
    //: it to the sum's own rounding: each value within 2^-6 of the larger magnitude, or under 2^-12 absolute
    const auto half_to_float = [](uint32_t h) {
      const uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 31u, man = h & 1023u;
      float v;
      if (exp == 0) {
        v = std::ldexp(static_cast<float>(man), -24);
      } else if (exp == 31) {
        v = man ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
      } else {
        v = std::ldexp(static_cast<float>(man + 1024u), static_cast<int>(exp) - 25);
      }
      return sign ? -v : v;
    };
    //: a token's 32 channels are 16 words (f16 pairs) or 32 (fp32): each value within 2^-5 of the token's largest
    //: (an f16 sum of up to 64 CTAs' partials rounds each add by 2^-11), or under 2^-12 absolute
    const auto close = [](float a, float b, float tok) {
      const float d = std::fabs(a - b);
      return d <= tok * (1.0f / 32.0f) || d <= 1.0f / 4096.0f;
    };
    size_t bad = got.empty() ? 1 : 0;
    int shown = 0;
    double worst = 0.0;
    const bool f32 = c.cap.p.reduce == rc::kReduceF32;
    const size_t per_tok = f32 ? 32 : 16;
    std::vector<float> tok_max(got.size() / per_tok + 1, 0.0f);
    for (size_t i = 0; i < got.size(); ++i) {
      float w0, w1 = 0.0f;
      if (f32) {
        std::memcpy(&w0, &c.want[i], 4);
      } else {
        w0 = half_to_float(c.want[i] & 0xffffu);
        w1 = half_to_float(c.want[i] >> 16);
      }
      float& m = tok_max[i / per_tok];
      m = std::max(m, std::max(std::fabs(w0), std::fabs(w1)));
    }
    for (size_t i = 0; i < got.size(); ++i) {
      float a[2], w[2];
      int n = f32 ? 1 : 2;
      if (f32) {
        std::memcpy(&a[0], &got[i], 4);
        std::memcpy(&w[0], &c.want[i], 4);
      } else {
        a[0] = half_to_float(got[i] & 0xffffu);
        a[1] = half_to_float(got[i] >> 16);
        w[0] = half_to_float(c.want[i] & 0xffffu);
        w[1] = half_to_float(c.want[i] >> 16);
      }
      bool ok = true;
      for (int k = 0; k < n; ++k) {
        ok = ok && close(a[k], w[k], tok_max[i / per_tok]);
        const float m = tok_max[i / per_tok];
        if (m > 0.0f) worst = std::max(worst, static_cast<double>(std::fabs(a[k] - w[k]) / m));
      }
      if (!ok && shown < 6) {
        std::fprintf(stderr, "  differs word %zu: got %08x (%g %g) want %08x (%g %g)\n", i, got[i],
                     a[0], f32 ? 0.0f : a[1], c.want[i], w[0], f32 ? 0.0f : w[1]);
        ++shown;
      }
      bad += !ok;
    }
    return std::make_pair(bad, static_cast<int>(worst * 65536.0));
  };
  cudaEvent_t e0, e1;
  LAB_CHECK(cudaEventCreate(&e0));
  LAB_CHECK(cudaEventCreate(&e1));
  //: timed with every clock off: a lab form's start and end stamps alone (its span), the production kernel unbound
  c.cap.p.ledger = traced ? ledger : nullptr;
  lab::Samples span;
  const lab::Samples us = b.variant(
      [&] {
        reset();
        LAB_CHECK(cudaEventRecord(e0));
        launch(c.cap.p, false);
        LAB_CHECK(cudaEventRecord(e1));
      },
      [&] {
        float ms = 0.0f;
        LAB_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        if (traced) span.v.push_back(read_trace(ledger, ctas, windows).window);
        return 1000.0 * ms;
      });
  const auto off = compare();
  c.cap.p.ledger = ledger;
  Reading last;
  lab::Samples clocked_us;
  const lab::Samples s = b.variant(
      [&] {
        reset();
        LAB_CHECK(cudaEventRecord(e0));
        launch(c.cap.p, true);
        LAB_CHECK(cudaEventRecord(e1));
      },
      [&] {
        float ms = 0.0f;
        LAB_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        clocked_us.v.push_back(1000.0 * ms);
        last = traced ? read_trace(ledger, ctas, windows) : read_ledger(ledger, ctas, windows);
        return last.window;
      });
  const auto on = compare();
  const std::string label = std::string(name) + " / " + cell;
  b.record(label, "us a launch, every clock off", us);
  if (traced) b.record(label + " / span", "cycles a CTA-window, its span, every clock off", span);
  b.record(label + " / clocked", "cycles a CTA-window, its clock on", s);
  b.record(label + " / clocked launch", "us a launch, its clock on", clocked_us);
  for (int ph = 0; ph < rc::kCarryPhases; ++ph) {
    b.record(label + " / " + kPhases[ph], "cycles a warp-window", lab::Samples{{last.phase[ph]}});
    b.record(label + " / " + kPhases[ph] + " busiest", "cycles a warp-window",
             lab::Samples{{last.busiest[ph]}});
  }
  b.check(label, off.first == 0 && on.first == 0,
          "clocks off " + std::to_string(off.first) + ", on " + std::to_string(on.first) + " of "
              + std::to_string(c.want.size()) + " num words differ from the call's, by at most "
              + std::to_string(std::max(off.second, on.second))
              + " x 2^-16 of the token's largest");
  c.cap.p.ledger = nullptr;
  LAB_CHECK(cudaEventDestroy(e0));
  LAB_CHECK(cudaEventDestroy(e1));
  LAB_CHECK(cudaFree(ledger));
}

//: THE FIXED COST A CALL: a form's launch timed with every clock off over the call's first 1, 2, 4 ... windows, and the
//: line through them -- its intercept the call's fixed cost (plan pass, prologue, epilogue, the last wave's tail), its
//: slope a window's
template <class Launch>
void sweep(const lab::Bench& b, Call& c, const char* name, const char* cell, Launch launch) {
  const int all = c.cap.p.windows;
  const std::string label = std::string(name) + " / " + cell + " / by windows";
  cudaEvent_t e0, e1;
  LAB_CHECK(cudaEventCreate(&e0));
  LAB_CHECK(cudaEventCreate(&e1));
  c.cap.p.ledger = nullptr;
  double sn = 0, st = 0, snn = 0, snt = 0;
  int k = 0;
  for (int n = 1; n <= all; n *= 2, ++k) {
    c.cap.p.windows = n;
    const lab::Samples us = b.variant(
        [&] {
          LAB_CHECK(cudaMemset(c.num, 0, c.num_bytes));
          LAB_CHECK(cudaEventRecord(e0));
          launch(c.cap.p, false);
          LAB_CHECK(cudaEventRecord(e1));
        },
        [&] {
          float ms = 0.0f;
          LAB_CHECK(cudaEventElapsedTime(&ms, e0, e1));
          return 1000.0 * ms;
        });
    b.record(label + " / " + std::to_string(n), "us a launch, every clock off", us);
    const double t = us.median();
    sn += n;
    st += t;
    snn += double(n) * n;
    snt += n * t;
  }
  c.cap.p.windows = all;
  const double slope = (k * snt - sn * st) / (k * snn - sn * sn), fixed = (st - slope * sn) / k;
  b.record(label + " / fixed", "us a call, the line's intercept", lab::Samples{{fixed}});
  b.record(label + " / a window", "us a window, the line's slope", lab::Samples{{slope}});
  b.unchecked(label,
              "a shorter call's readout is a prefix of the full call's, checked with the form");
  LAB_CHECK(cudaEventDestroy(e0));
  LAB_CHECK(cudaEventDestroy(e1));
}

}  // namespace carry_window

int main(int argc, char** argv) {
  using namespace carry_window;
  lab::Bench b(argc, argv);
  const std::string production = "rola::carry::carry_kernel<";
  b.expect(production, "HMMA", ">=", 1);
  b.exempt(production,
           "the production kernel, the baseline: its budget is tools/budgets/carry.json");
  //: each form's kernels, its clock in and out; the steps themselves are one block each, the spans the gate reads as
  //: steps besides them run from a shuffle's cold fallback, laid out past the epilogue, back into the window loop
  for (int f : {1, 3, 4, 5}) {
    const std::string k = "carry_window::window<\\(int\\)" + std::to_string(f) + ",";
    b.expect(k, "HMMA", ">=", 1);
    b.expect(k, "STEP_GLOBAL", ">=", 0);
    b.expect(k, "BSSY", ">=", 0);
    b.exempt(k, "a window form against the production kernel: its counts set the window's lines");
  }
  if (b.expect_only) return 0;
  std::string dir;
  int only = -1;
  for (int i = 1; i + 1 < argc; ++i) {
    if (!std::strcmp(argv[i], "--inputs")) dir = argv[i + 1];
    if (!std::strcmp(argv[i], "--form")) only = std::atoi(argv[i + 1]);
  }
  if (dir.empty()) {
    std::fprintf(stderr, "carry_window: --inputs <directory> (tools/lab.py writes it)\n");
    return 1;
  }
  const char* cells[2][2] = {{"alt", "alt-k4"}, {"thin", "thinned 1/16"}};
  for (const auto& cell : cells) {
    Call c = load(dir + "/" + cell[0]);
    std::printf(
        "{\"lab\": \"note\", \"cell\": \"%s\", \"moved\": %d, \"ctas\": %lld, \"windows\": %d}\n",
        cell[1], c.moved, static_cast<long long>(c.cap.boxes * c.cap.bh), c.cap.p.windows);
    if (only < 0 || only == 0)
      sweep(b, c, "the production kernel", cell[1], [&](const rc::CarryParams& p, bool) {
        LAB_CHECK(rc::carry_launch<Arm>(p, static_cast<int>(c.cap.boxes),
                                        static_cast<int>(c.cap.bh), nullptr));
      });
    if (only < 0 || only == 5)
      sweep(b, c, "form 5, form 4 with the fills' loads ahead", cell[1],
            [&](const rc::CarryParams& p, bool on) { launch_either<5>(c, p, on); });
    if (only < 0 || only == 0)
      form(
          b, c, "the production kernel", cell[1],
          [&](const rc::CarryParams& p, bool) {
            LAB_CHECK(rc::carry_launch<Arm>(p, static_cast<int>(c.cap.boxes),
                                            static_cast<int>(c.cap.bh), nullptr));
          },
          false);
    if (only < 0 || only == 1)
      form(b, c, "form 1, option A by hand", cell[1],
           [&](const rc::CarryParams& p, bool on) { launch_either<1>(c, p, on); });
    if (only < 0 || only == 3)
      form(b, c, "form 3, A with the next rows under the fold", cell[1],
           [&](const rc::CarryParams& p, bool on) { launch_either<3>(c, p, on); });
    if (only < 0 || only == 4)
      form(b, c, "form 4, A with the read buffer at the top, the drain in the fold", cell[1],
           [&](const rc::CarryParams& p, bool on) { launch_either<4>(c, p, on); });
    if (only < 0 || only == 5)
      form(b, c, "form 5, form 4 with the fills' loads ahead", cell[1],
           [&](const rc::CarryParams& p, bool on) { launch_either<5>(c, p, on); });
  }
}
