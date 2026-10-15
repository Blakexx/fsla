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
};

inline Reading read_ledger(const long long* ledger, int ctas, int windows) {
  std::vector<long long> h(static_cast<size_t>(ctas) * rc::kWarps * rc::kCarryPhases);
  LAB_CHECK(cudaMemcpy(h.data(), ledger, h.size() * 8, cudaMemcpyDeviceToHost));
  Reading r;
  for (int c = 0; c < ctas; ++c) {
    double busiest = 0.0;
    for (int w = 0; w < rc::kWarps; ++w) {
      double sum = 0.0;
      for (int ph = 0; ph < rc::kCarryPhases; ++ph) {
        const double x =
            double(h[(static_cast<size_t>(c) * rc::kWarps + w) * rc::kCarryPhases + ph]);
        sum += x;
        r.phase[ph] += x;
      }
      busiest = std::max(busiest, sum);
    }
    r.window += busiest;
  }
  r.window /= double(ctas) * windows;
  for (double& x : r.phase) x /= double(ctas) * rc::kWarps * windows;
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

//: FORM 1, OPTION A by hand: a window's maxima, then each image half stored and read out, the fold's rows over the
//: image once every warp has read it, the drain, the next window's read rows at the area's top, the fold. One chunk a
//: side (a window that does not fit traps).
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
  read_fill(p, smem, 0, area);
  ops::stage_commit();

#pragma unroll 1
  for (int w = 0; w < p.windows; ++w) {
    //: THE WINDOW'S INPUTS AND READ ROWS landed, the next window's staged
    ops::stage_wait<0>();
    ops::rendezvous();
    lap(p, smem, rc::kPhaseLanded);
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
      rc::fold_fill<P>(p, smem, rc::window_of<P>(p, smem, rc::kWrite, w),
                       rc::Place{static_cast<int>(blockIdx.y), 0}, w, 0, f.slots, f.slots, area,
                       t >> 5, t & 31);
    }
    ops::stage_commit();
    lap(p, smem, rc::kPhaseFill);
    //: THE DRAIN, the panel to `num`
    rc::drain<P>(p, smem, rc::window_of<P>(p, smem, rc::kRead, w),
                 rc::Place{static_cast<int>(blockIdx.y), 0}, w, 0, r.slots,
                 area + top_of(r.bytes) + rc::read_pads<P>(r.slots) + rc::kPadBytes);
    lap(p, smem, rc::kPhaseDrain);
    //: THE NEXT WINDOW'S READ ROWS, at the top once every warp has drained (its inputs and the fold's rows landed)
    ops::stage_wait<0>();
    ops::rendezvous();
    lap(p, smem, rc::kPhaseWait);
    if (w + 1 < p.windows) {
      if (f.bytes > static_cast<int>(top_of(sized<rc::kRead>(p, smem, w + 1).bytes))) __trap();
      read_fill(p, smem, w + 1, area);
    }
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
    lap(p, smem, rc::kPhaseFold);
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
  for (int ph = 0; ph < rc::kCarryPhases; ++ph)
    b.record(label + " / " + kPhases[ph], "cycles a warp-window", lab::Samples{{last.phase[ph]}});
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
  const std::string form1 = lab::kernel("carry_window::window", {1});
  b.expect(form1, "HMMA", ">=", 1);
  //: the steps themselves are one block each; the spans the gate reads as steps besides them run from a shuffle's cold
  //: fallback, laid out past the epilogue, back into the window loop, over the epilogue's ledger and check adds
  b.expect(form1, "STEP_GLOBAL", ">=", 0);
  b.expect(form1, "BSSY", ">=", 0);
  b.exempt(
      form1,
      "a window form compared against the production kernel: its counts set the window's class lines");
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
    if (only < 0 || only == 1)
      form(b, c, "form 1, option A by hand", cell[1], [&](const rc::CarryParams& p) {
        window<1><<<dim3(static_cast<unsigned>(c.cap.boxes), static_cast<unsigned>(c.cap.bh)),
                    rc::kThreads, P::kSmemBytes>>>(p);
        LAB_CHECK(cudaGetLastError());
      });
  }
}
