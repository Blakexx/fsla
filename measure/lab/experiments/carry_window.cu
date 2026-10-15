// measure/lab/experiments/carry_window.cu -- QUESTION: what does the carry's window cost a CTA on sm_86 in each form
// the critical-path design admits, phase by phase in cycles and registers, against the production kernel on the same
// inputs? METHOD: a real call captured (`carry_window.inputs.py`: nl64k-alt-k4's first 32 windows, and the same with
// its lists thinned to 1/16), its buffers re-hosted and its operand block's pointers moved to them; each form launched
// over the call's grid with the phase ledger bound, its readout's output checked bit for bit against the call's.
// Forms: 0 the production kernel (its plan pass, then the carry). see docs/lab/carry_window.md
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "carry/carry_arm.cuh"
#include "lab.cuh"

namespace carry_window {

namespace rc = rola::carry;
using Arm = rola::Arm<0, 32, 1, 32, 1, 32, 0>;
using P = rc::Plan<Arm>;

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
  int k;
  std::string role;
  uint64_t base, bytes;
  while (lines >> k >> role >> base >> bytes) {
    const std::vector<char> data = slurp(dir + "/buffer_" + std::to_string(k) + ".bin");
    char* dev = nullptr;
    LAB_CHECK(cudaMalloc(&dev, bytes));
    LAB_CHECK(cudaMemcpy(dev, data.data(), bytes, cudaMemcpyHostToDevice));
    for (size_t i = 0; i < sizeof(rc::CarryParams) / 8; ++i)
      if (words[i] >= base && words[i] < base + bytes) {
        words[i] = reinterpret_cast<uint64_t>(dev) + (words[i] - base);
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
  return c;
}

//: THE LEDGER'S READING: the mean over CTAs of the busiest warp's cycles, a window -- and each phase's, the mean warp's
struct Reading {
  double window = 0.0;
  double phase[rc::kCarryPhases] = {};
  double busiest[rc::kCarryPhases] = {};
};

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

const char* const kPhases[rc::kCarryPhases] = {"landed", "maxima", "store", "readout",
                                               "drain",  "fill",   "fold",  "wait"};

namespace ops = ::denseref::ops;

//: THE THREAD'S INDEX, opaque, read at each phase so nothing derived from it is held across phases (the register line)
__device__ __forceinline__ int tid() {
  int t;
  asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t));
  return t;
}

//: THE PHASE CLOCK, holding no register: a warp's lane 0 adds the cycles since its row's mark to phase `ph`, its row
//: (the `kClock` tenant) addressed from the thread index at each lap; bound only with the ledger
__device__ __forceinline__ void lap(const rc::CarryParams& p, uint8_t* smem, int ph) {
  const int t = tid();
  if (p.ledger != nullptr && (t & 31) == 0) {
    long long* const row =
        reinterpret_cast<long long*>(smem + P::kClock) + (t >> 5) * (rc::kCarryPhases + 1);
    const long long now = clock64();
    row[ph] += now - row[rc::kCarryPhases];
    row[rc::kCarryPhases] = now;
  }
}

//: A SIDE'S CHUNK, the whole union: its slots and its buffer's bytes (`read_bytes` / `fold_bytes`)
struct Sized {
  int slots, bytes;
};

template <int Side>
__device__ __forceinline__ Sized sized(const rc::CarryParams& p, const uint8_t* smem, int w) {
  const rc::Window win = rc::window_of<P>(p, smem, Side, w);
  const int slots = rc::slots_of<P>(smem, win, Side, w);
  const int entries =
      win.listed ? rc::pre_at(rc::pre_of<P>(smem, Side, w), rc::kBoxLists, rc::kListAt - 1) : 0;
  return Sized{slots, Side == rc::kRead ? rc::read_bytes<P>(slots, entries)
                                        : rc::fold_bytes<P>(slots, entries)};
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
  rc::read_fill<P>(p, smem, rc::window_of<P>(p, smem, rc::kRead, w),
                   rc::Place{static_cast<int>(blockIdx.y), 0}, w, 0, r.slots, r.slots,
                   area + top_of(r.bytes), t >> 5, t & 31);
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

//: A READ BUFFER'S PANEL AND DUMMY ROWS zeroed by stores, and its entries copied (a warp its own digits' runs);
//: `Ahead` the entries in one pass
template <bool Ahead = false>
__device__ __forceinline__ void panel_entries(const rc::CarryParams& p, uint8_t* smem, int w,
                                              uint32_t buf, int slots) {
  const int t = tid();
  const uint32_t pads = buf + rc::read_pads<P>(slots);
  const int quanta = (rc::read_ents<P>(slots) - rc::read_pads<P>(slots)) / 16;
#pragma unroll 4
  for (int k = t; k < quanta; k += rc::kThreads)
    asm volatile("st.shared.v4.u32 [%0], {%1, %1, %1, %1};" ::"r"(pads + 16u * k), "r"(0)
                 : "memory");
  const rc::Window rw = rc::window_of<P>(p, smem, rc::kRead, w);
  if (!rw.listed) return;
  const rc::Run run = rc::chunk_runs<P>(smem, rw, rc::kRead, w, 0, slots, slots, t & 31);
  if constexpr (Ahead) {
    entries_ahead<P>(p, p.read_lists, rw, w, run, buf + rc::read_ents<P>(slots), t >> 5, t & 31);
  } else {
    rc::entries_fill<P>(p, p.read_lists, rw, rc::Place{static_cast<int>(blockIdx.y), 0}, w, run,
                        buf + rc::read_ents<P>(slots), t >> 5, t & 31);
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
      const int s = s0 + kQuads * j, t = u[j] & (rc::kWindow - 1);
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
                                       rc::union_of<P>(smem, rc::kRead, w), rw, s0, s1 - s0, at);
  } else {
    rc::fill<P::kReadFactors, false>(addr.slices[rc::kRead], nullptr, 0,
                                     rc::union_of<P>(smem, rc::kRead, w), rw, s0, s1 - s0, at);
  }
}

//: A FOLD BUFFER'S FILL with its loads ahead: its rows (`fill_ahead`), then its entries (fold.cuh's `fold_fill`)
__device__ __forceinline__ void fold_fill_ahead(const rc::CarryParams& p, uint8_t* smem, int w,
                                                uint32_t buf, int slots) {
  const volatile rc::Addr& addr = *reinterpret_cast<const volatile rc::Addr*>(smem + P::kAddr);
  const rc::Window ww = rc::window_of<P>(p, smem, rc::kWrite, w);
  fill_ahead<P::kFoldFactors, true>(addr.slices[rc::kWrite], addr.values,
                                    static_cast<int>(p.values.st),
                                    rc::union_of<P>(smem, rc::kWrite, w), ww, 0, slots, buf);
  const int t = tid();
  if (ww.listed)
    entries_ahead<P>(p, p.write_lists, ww, w,
                     rc::chunk_runs<P>(smem, ww, rc::kWrite, w, 0, slots, slots, t & 31),
                     buf + rc::fold_ents<P>(slots), t >> 5, t & 31);
}

//: A FOLD over the whole union of a listed side, `hook(J)` run before list `J`'s steps (fold.cuh's `fold_chunk`, its
//: rescale applied)
template <class Hook>
__device__ __forceinline__ void fold_lists(rc::State<P>& st, const uint8_t* smem,
                                           const rc::Window& ww, int w, int slots, uint32_t rows,
                                           int warp, int lane, Hook hook) {
  const rc::Run run = rc::chunk_runs<P>(smem, ww, rc::kWrite, w, 0, slots, slots, lane);
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
    rc::steps<P, J>(st, sc.cf, rows, 0, 0, n, lane, [&](int k) {
      uint2 e;
      asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];"
                   : "=r"(e.x), "=r"(e.y)
                   : "r"(ents + 8u * static_cast<uint32_t>(to + k)));
      return e;
    });
  });
}

//: FORM 1, OPTION A by hand: a window's maxima, then each image half stored and read out, the fold's rows over the
//: image once every warp has read it, the drain, the next window's read rows at the area's top, the fold. One chunk a
//: side (a window that does not fit traps). FORM 3: the next window's panel zeroed by stores, its entries copied before
//: the fold, its rows in four slices, one before each of the warp's lists' steps. FORM 4: the window's read buffer
//: issued at its top (rows, panel by stores, entries), the drain a warp's share inside the fold before its list
//: `(warp >> 1) & 3` (a scheduler's two warps at different lists). FORM 5: form 4, its fills' loads ahead.
template <int FORM>
__global__ __launch_bounds__(rc::kThreads, 1) void window(
    const __grid_constant__ rc::CarryParams p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t area = ops::smem_addr(smem + P::kEnd);
  {
    const int t = tid();
    if (t < rc::kBoxLists)
      reinterpret_cast<rc::UnitBook*>(smem + P::kBook)[t] = rc::UnitBook{rc::kUnitNone, 0.0f};
    if (t == 0) rc::addr_in(*reinterpret_cast<rc::Addr*>(smem + P::kAddr), p, rc::place_of(p));
    if ((t & 31) == 0) {
      long long* const row =
          reinterpret_cast<long long*>(smem + P::kClock) + (t >> 5) * (rc::kCarryPhases + 1);
      for (int ph = 0; ph < rc::kCarryPhases; ++ph) row[ph] = 0;
      row[rc::kCarryPhases] = clock64();
    }
  }
  ops::rendezvous();
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
    lap(p, smem, rc::kPhaseLanded);
    if constexpr (FORM >= 4) {
      //: THE WINDOW'S READ BUFFER, under the maxima and the first half's stores
      const Sized r = sized<rc::kRead>(p, smem, w);
      read_rows<FORM >= 5>(p, smem, w, area + top_of(r.bytes), 0, r.slots);
      panel_entries<FORM >= 5>(p, smem, w, area + top_of(r.bytes), r.slots);
      ops::stage_commit();
      lap(p, smem, rc::kPhaseFill);
    }
    if (w + 1 < p.windows) stage(p, w + 1, smem);
    ops::stage_commit();
    //: THE DIGIT MAXIMA of the state before the window's writes
    {
      const int t = tid();
      rc::post_maxima<P>(st, smem, w, t >> 5, t & 31);
      ops::rendezvous();
      dexp = rc::digit_exp<P>(smem, w, tid() & 31);
      if (tid() < 32) reinterpret_cast<int*>(smem + P::kExps)[tid()] = dexp;
    }
    lap(p, smem, rc::kPhaseMaxima);
    //: THE HALVES: each stored into the image, then read out
    const Sized r = sized<rc::kRead>(p, smem, w);
    if (r.bytes + rc::kImageRing > P::kArea) __trap();
#pragma unroll 1
    for (int half = 0; half < 2; ++half) {
      if (half > 0) ops::rendezvous();
      {
        const int t = tid();
        rc::store_batches<P>(st, smem, area, 2 * half, 2 * half + 2, 2 * rc::kBatchPlanes - 1, dexp,
                             t >> 5, t & 31);
      }
      if constexpr (FORM >= 4) {
        if (half == 0) ops::stage_wait<1>();
      }
      ops::rendezvous();
      lap(p, smem, rc::kPhaseStore);
      {
        const int t = tid();
        const uint32_t rows = area + top_of(r.bytes), pads = rows + rc::read_pads<P>(r.slots),
                       ents = rows + rc::read_ents<P>(r.slots);
        const rc::Window rw = rc::window_of<P>(p, smem, rc::kRead, w);
        if (p.readout_f16) {
          rc::read_chunk<P, true, true>(smem, rw, w, 0, r.slots, r.slots, 2 * half, 2 * half + 2,
                                        area, 2 * rc::kBatchPlanes - 1, rows, pads, ents, t >> 5,
                                        t & 31);
        } else {
          rc::read_chunk<P, false, true>(smem, rw, w, 0, r.slots, r.slots, 2 * half, 2 * half + 2,
                                         area, 2 * rc::kBatchPlanes - 1, rows, pads, ents, t >> 5,
                                         t & 31);
        }
      }
      lap(p, smem, rc::kPhaseReadout);
    }
    ops::rendezvous();
    //: THE FOLD'S ROWS, over the image every warp has read
    const Sized f = sized<rc::kWrite>(p, smem, w);
    if (f.bytes > static_cast<int>(top_of(r.bytes))) __trap();
    {
      const int t = tid();
      if constexpr (FORM >= 5) {
        fold_fill_ahead(p, smem, w, area, f.slots);
      } else {
        rc::fold_fill<P>(p, smem, rc::window_of<P>(p, smem, rc::kWrite, w),
                         rc::Place{static_cast<int>(blockIdx.y), 0}, w, 0, f.slots, f.slots, area,
                         t >> 5, t & 31);
      }
    }
    ops::stage_commit();
    lap(p, smem, rc::kPhaseFill);
    if constexpr (FORM >= 4) {
      ops::stage_wait<0>();
      ops::rendezvous();
      lap(p, smem, rc::kPhaseWait);
      //: THE FOLD, each warp's share of the drain before its list `warp & 3`
      {
        const int t = tid();
        const rc::Window ww = rc::window_of<P>(p, smem, rc::kWrite, w);
        if (!ww.listed) __trap();
        rc::fold_units<P>(smem, ww, w, t >> 5, t & 31);
        fold_lists(st, smem, ww, w, f.slots, area, t >> 5, t & 31, [&](auto J) {
          if (((tid() >> 6) & (P::kFoldLists - 1)) == decltype(J)::value)
            rc::drain<P>(p, smem, rc::window_of<P>(p, smem, rc::kRead, w),
                         rc::Place{static_cast<int>(blockIdx.y), 0}, w, 0, r.slots,
                         area + top_of(r.bytes) + rc::read_pads<P>(r.slots) + rc::kPadBytes);
        });
      }
      lap(p, smem, rc::kPhaseFold);
    } else {
      //: THE DRAIN, the panel to `num`
      rc::drain<P>(p, smem, rc::window_of<P>(p, smem, rc::kRead, w),
                   rc::Place{static_cast<int>(blockIdx.y), 0}, w, 0, r.slots,
                   area + top_of(r.bytes) + rc::read_pads<P>(r.slots) + rc::kPadBytes);
      lap(p, smem, rc::kPhaseDrain);
      //: THE NEXT WINDOW'S READ ROWS, at the top once every warp has drained (its inputs and the fold's rows landed)
      ops::stage_wait<0>();
      ops::rendezvous();
      lap(p, smem, rc::kPhaseWait);
      const bool next = w + 1 < p.windows;
      const Sized n = next ? sized<rc::kRead>(p, smem, w + 1) : Sized{0, 0};
      if (next && f.bytes > static_cast<int>(top_of(n.bytes))) __trap();
      const uint32_t nbuf = area + top_of(n.bytes);
      if constexpr (FORM == 1) {
        if (next) read_fill(p, smem, w + 1, area);
        ops::stage_commit();
        lap(p, smem, rc::kPhaseFill);
        //: THE FOLD, the window's units stepped first
        {
          const int t = tid();
          const rc::Window ww = rc::window_of<P>(p, smem, rc::kWrite, w);
          rc::fold_units<P>(smem, ww, w, t >> 5, t & 31);
          rc::fold_chunk<P>(st, true, smem, ww, w, 0, f.slots, f.slots, area,
                            area + rc::fold_ents<P>(f.slots), t >> 5, t & 31);
        }
      } else {
        //: THE NEXT WINDOW'S PANEL, zeroed by stores, and its entries
        if (next) panel_entries(p, smem, w + 1, nbuf, n.slots);
        lap(p, smem, rc::kPhaseFill);
        //: THE FOLD, the window's units stepped first, the next window's rows a slice before each list
        {
          const int t = tid();
          const rc::Window ww = rc::window_of<P>(p, smem, rc::kWrite, w);
          if (!ww.listed) __trap();
          rc::fold_units<P>(smem, ww, w, t >> 5, t & 31);
          fold_lists(st, smem, ww, w, f.slots, area, t >> 5, t & 31, [&](auto J) {
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
      lap(p, smem, rc::kPhaseFold);
    }
  }

  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < P::kStateRegs; ++k) sum += (&st.acc[0][0][0][0])[k];
  atomicAdd(p.check, sum);
  const int t = tid();
  if (p.ledger != nullptr && (t & 31) == 0) {
    const long long* const row =
        reinterpret_cast<const long long*>(smem + P::kClock) + (t >> 5) * (rc::kCarryPhases + 1);
    long long* const out =
        p.ledger
        + ((static_cast<long long>(blockIdx.y) * gridDim.x + blockIdx.x) * rc::kWarps + (t >> 5))
              * rc::kCarryPhases;
    for (int ph = 0; ph < rc::kCarryPhases; ++ph)
      atomicAdd(reinterpret_cast<unsigned long long*>(out + ph), row[ph]);
  }
}

template <int FORM>
void launch_form(const Call& c, const rc::CarryParams& p) {
  window<FORM><<<dim3(static_cast<unsigned>(c.cap.boxes), static_cast<unsigned>(c.cap.bh)),
                 rc::kThreads, P::kSmemBytes>>>(p);
  LAB_CHECK(cudaGetLastError());
}

template <class Launch>
void form(const lab::Bench& b, Call& c, const char* name, const char* cell, Launch launch) {
  const int ctas = static_cast<int>(c.cap.boxes * c.cap.bh), windows = c.cap.p.windows;
  long long* ledger = nullptr;
  LAB_CHECK(cudaMalloc(&ledger, sizeof(long long) * ctas * rc::kWarps * rc::kCarryPhases));
  c.cap.p.ledger = ledger;
  Reading last;
  const auto run = [&] {
    LAB_CHECK(cudaMemset(ledger, 0, sizeof(long long) * ctas * rc::kWarps * rc::kCarryPhases));
    LAB_CHECK(cudaMemset(c.num, 0, c.num_bytes));
    LAB_CHECK(cudaMemset(c.check, 0, 4));
    launch(c.cap.p);
  };
  const lab::Samples s = b.variant(run, [&] {
    last = read_ledger(ledger, ctas, windows);
    return last.window;
  });
  const std::string label = std::string(name) + " / " + cell;
  b.record(label, "cycles a CTA-window (busiest warp)", s);
  for (int ph = 0; ph < rc::kCarryPhases; ++ph) {
    b.record(label + " / " + kPhases[ph], "cycles a warp-window", lab::Samples{{last.phase[ph]}});
    b.record(label + " / " + kPhases[ph] + " busiest", "cycles a warp-window",
             lab::Samples{{last.busiest[ph]}});
  }
  std::vector<uint32_t> got(c.want.size());
  LAB_CHECK(cudaMemcpy(got.data(), c.num, c.num_bytes, cudaMemcpyDeviceToHost));
  size_t bad = 0;
  int worst = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    bad += got[i] != c.want[i];
    const int lo =
        static_cast<int16_t>(got[i] & 0xffffu) - static_cast<int16_t>(c.want[i] & 0xffffu);
    const int hi = static_cast<int16_t>(got[i] >> 16) - static_cast<int16_t>(c.want[i] >> 16);
    worst = std::max(worst, std::max(std::abs(lo), std::abs(hi)));
  }
  b.check(label, bad == 0 && !got.empty(),
          std::to_string(bad) + " of " + std::to_string(got.size())
              + " num words differ from the call's, by at most " + std::to_string(worst)
              + " in an s16 half");
  c.cap.p.ledger = nullptr;
  LAB_CHECK(cudaFree(ledger));
}

}  // namespace carry_window

int main(int argc, char** argv) {
  using namespace carry_window;
  lab::Bench b(argc, argv);
  const std::string production = "rola::carry::carry_kernel<";
  b.expect(production, "HMMA", ">=", 1);
  b.exempt(production,
           "the production kernel, the baseline: its budget is tools/budgets/carry.json");
  const std::string form1 = lab::kernel("carry_window::window", {1}),
                    form3 = lab::kernel("carry_window::window", {3}),
                    form4 = lab::kernel("carry_window::window", {4}),
                    form5 = lab::kernel("carry_window::window", {5});
  //: the steps themselves are one block each; the spans the gate reads as steps besides them run from a shuffle's cold
  //: fallback, laid out past the epilogue, back into the window loop, over the epilogue's ledger and check adds
  for (const std::string& k : {form1, form3, form4, form5}) {
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
      form(b, c, "the production kernel", cell[1], [&](const rc::CarryParams& p) {
        LAB_CHECK(rc::carry_launch<Arm>(p, static_cast<int>(c.cap.boxes),
                                        static_cast<int>(c.cap.bh), nullptr));
      });
    LAB_CHECK(cudaFuncSetAttribute(window<1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   P::kSmemBytes));
    LAB_CHECK(cudaFuncSetAttribute(window<3>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   P::kSmemBytes));
    if (only < 0 || only == 1)
      form(b, c, "form 1, option A by hand", cell[1],
           [&](const rc::CarryParams& p) { launch_form<1>(c, p); });
    LAB_CHECK(cudaFuncSetAttribute(window<4>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   P::kSmemBytes));
    LAB_CHECK(cudaFuncSetAttribute(window<5>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   P::kSmemBytes));
    if (only < 0 || only == 5)
      form(b, c, "form 5, form 4 with the fills' loads ahead", cell[1],
           [&](const rc::CarryParams& p) { launch_form<5>(c, p); });
    if (only < 0 || only == 4)
      form(b, c, "form 4, A with the read buffer at the top, the drain in the fold", cell[1],
           [&](const rc::CarryParams& p) { launch_form<4>(c, p); });
    if (only < 0 || only == 3)
      form(b, c, "form 3, A with the next rows under the fold", cell[1],
           [&](const rc::CarryParams& p) { launch_form<3>(c, p); });
  }
}
