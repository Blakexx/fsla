// measure/lab/experiments/drain.cu -- QUESTION: what does the readout's drain -- a pass's panel words added into the
// head's `num` by `red.global.add` at each slot's token, the panel cleared for the next pass's adds -- cost a
// CTA-window, in today's form and in forms that pair conflict-free slots, clear by exchange or by a 16-byte sweep,
// and beside a stream's copies? METHOD: 80 CTAs, box b of one head (`drain.inputs.py`: `read_stream`'s three cells,
// each box-window's union in rank order); each pass every thread fills its slot's 16 panel words (four 16-byte
// stores) and its token, a CTA barrier, the drain, a barrier. A warp drains its 32 slots, two a `red.global` (a
// lane a word: lanes 0-15 one slot's 16 words, 16-31 another's), each request a token's half line of `num` (d_v 64:
// two channel tiles a 128-byte row). The forms differ in which two slots a request pairs and how a word is cleared:
// FORM 0 slots (s, s + 1) at the panel's 20-word pitch (4 banks shared: 2 wavefronts an access), a load and a 4-byte
// store; 1 slots (s, s + 4) (16 banks apart: 1 wavefront); 2 as 1 with one `atom.shared.exch` reading and clearing; 3
// as 1 with no clear in the drain, a 16-byte sweep after the barrier; 4 as 1 with a warp's `cp.async` of 512 bytes
// every 4 adds (a stream beside it); 7 the same every add; 6 as 1 with no clear (a bound); 5 the fill alone (a
// bound); 8 as 3 with the warp's groups of 4 requests past its slots in the union not issued (a warp-uniform count).
// Each draining form's `num` is checked word for word against the mirror's; a row prices a line request from form 8.
// A sample is the CTA's span a window. see docs/lab/drain.md
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "lab.cuh"

namespace drain_lab {

constexpr int kWarps = 8, kThreads = 32 * kWarps, kSlots = 256, kPitch = 20, kWords = 16, kGroups = 8;
//: THE LAYOUT: the panel (256 slots of 80 bytes), each slot's token, each warp's 4 KB copy ring
constexpr int kPanel = 0, kTok = kSlots * kPitch * 4, kRing = kTok + 4 * kSlots, kBytes = kRing + kWarps * 4096;
//: the copies' table: 1 MB, L2-resident beside the head's `num`
constexpr uint32_t kTable = 1u << 20;

struct In {
  const uint16_t* counts;
  const uint16_t* tokens;
  uint32_t* num;
  const uint8_t* table;
  int windows;
};

__device__ __forceinline__ uint32_t pattern(int w, int b, int s, int j) {
  return static_cast<uint32_t>(w) * 2654435761u + static_cast<uint32_t>(b) * 40503u + static_cast<uint32_t>(s) * 16u +
         static_cast<uint32_t>(j);
}

__device__ __forceinline__ uint32_t lds32(uint32_t a) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a));
  return v;
}

__device__ __forceinline__ void sts32(uint32_t a, uint32_t v) {
  asm volatile("st.shared.u32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}

__device__ __forceinline__ void sts128(uint32_t a, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(a), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}

__device__ __forceinline__ uint32_t exch0(uint32_t a) {
  uint32_t v;
  asm volatile("atom.shared.exch.b32 %0, [%1], 0;" : "=r"(v) : "r"(a) : "memory");
  return v;
}

//: an add into `num`: unpredicated -- ptxas puts a predicated global add in a branch and a convergence region (BSSY,
//: BRA, BSYNC around it), so a slot past the pass's union adds zero at its CTA's own dummy row past the head's tokens
//: (a shared dummy row would be one line every CTA's dead slots serialize on)
__device__ __forceinline__ void red(uint32_t* at, uint32_t v) {
  asm volatile("red.global.add.u32 [%0], %1;" ::"l"(at), "r"(v) : "memory");
}

__device__ __forceinline__ void copy512(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}

//: THE DRAIN of a pass's `m` slots by FORM (header): a warp's 16 requests, each slot's word and token loaded four
//: requests ahead of their adds
template <int FORM>
__device__ __forceinline__ void drain_pass(const In& in, uint32_t sm, int m, int ct, uint32_t dummy, int warp, int lane,
                                           int& copies, const uint8_t* table_lane, uint32_t ring_lane) {
  const int j = lane & 15, h = lane >> 4;
  uint32_t* const row = in.num + ct * kWords + j;
  //: FORM 8 issues the groups holding a live slot (a group q its slots 32 w + 8 q .. + 7), the rest all of them
  const int groups = FORM == 8 ? min(4, max(0, (m - 32 * warp + 7) / 8)) : 4;
#pragma unroll 1
  for (int q = 0; q < groups; ++q) {
    uint32_t v[4], tok[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int k = 4 * q + i;
      const int s = FORM == 0 ? 32 * warp + h + 2 * k : 32 * warp + 8 * (k >> 2) + (k & 3) + 4 * h;
      const uint32_t at = sm + kPanel + static_cast<uint32_t>((s * kPitch + j) * 4);
      if constexpr (FORM == 2) {
        v[i] = exch0(at);
      } else {
        v[i] = lds32(at);
        if constexpr (FORM == 0 || FORM == 1 || FORM == 4 || FORM == 7) sts32(at, 0u);
      }
      const uint32_t tk = lds32(sm + kTok + 4 * s);
      tok[i] = s < m ? tk : dummy;
      v[i] = s < m ? v[i] : 0u;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      red(row + static_cast<size_t>(tok[i]) * (2 * kWords), v[i]);
      if (FORM == 7 || (FORM == 4 && i == 0)) {
        copy512(ring_lane + (copies & 7) * 512, table_lane + static_cast<size_t>(copies & 255) * 512);
        ++copies;
      }
    }
  }
  if constexpr (FORM == 4 || FORM == 7) asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
}

template <int FORM, int SET>
__global__ void __launch_bounds__(kThreads, 1) drain(In in, long long* t) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int warp = lab::warp_id(), lane = lab::lane_id(), tid = static_cast<int>(threadIdx.x);
  const int b = static_cast<int>(blockIdx.x), ct = b % 2, g = b / 16;
  const uint32_t sm = lab::smem(smem);
  const uint8_t* const table_lane = in.table + 16 * lane + static_cast<size_t>(warp) * (kTable / kWarps);
  const uint32_t ring_lane = sm + kRing + warp * 4096 + 16 * lane;
  int copies = 0;
  //: this CTA's dummy row, past the head's 512 x windows tokens
  const uint32_t dummy = 512u * static_cast<uint32_t>(in.windows) + static_cast<uint32_t>(b);
  const long long t0 = clock64();
#pragma unroll 1
  for (int w = 0; w < in.windows; ++w) {
    const int n = in.counts[w * kGroups + g];
    const uint16_t* const toks = in.tokens + (static_cast<size_t>(w) * kGroups + g) * 512;
#pragma unroll 1
    for (int p = 0; p * kSlots < n; ++p) {
      const int m = min(kSlots, n - p * kSlots);
      //: the pass's fill: a thread its slot's 16 words and its token
      if (tid < m) {
        const int s = p * kSlots + tid;
        const uint32_t at = sm + kPanel + static_cast<uint32_t>(tid * kPitch * 4);
#pragma unroll
        for (int q = 0; q < 4; ++q)
          sts128(at + 16 * q, pattern(w, b, s, 4 * q), pattern(w, b, s, 4 * q + 1), pattern(w, b, s, 4 * q + 2),
                 pattern(w, b, s, 4 * q + 3));
        sts32(sm + kTok + 4 * tid, 512u * static_cast<uint32_t>(w) + toks[s]);
      }
      __syncthreads();
      if constexpr (FORM != 5) drain_pass<FORM>(in, sm, m, ct, dummy, warp, lane, copies, table_lane, ring_lane);
      __syncthreads();
      //: FORM 3's clear: a thread its slot's 16 words by four 16-byte stores
      if constexpr (FORM == 3 || FORM == 8) {
        const uint32_t at = sm + kPanel + static_cast<uint32_t>(tid * kPitch * 4);
#pragma unroll
        for (int q = 0; q < 4; ++q) sts128(at + 16 * q, 0u, 0u, 0u, 0u);
      }
    }
  }
  const long long t1 = clock64();
  if (lane == 0) {
    t[2 * (blockIdx.x * kWarps + warp)] = t0;
    t[2 * (blockIdx.x * kWarps + warp) + 1] = t1;
  }
}

struct Host {
  std::vector<uint16_t> counts, tokens;
  std::vector<uint32_t> num;
  int ctas = 0, windows = 0;
};

std::vector<uint8_t> file(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    std::fprintf(stderr, "drain: %s is missing\n", path.c_str());
    std::exit(1);
  }
  std::vector<uint8_t> v(static_cast<size_t>(f.tellg()));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size()));
  return v;
}

template <class T>
std::vector<T> as(const std::vector<uint8_t>& b) {
  return std::vector<T>(reinterpret_cast<const T*>(b.data()), reinterpret_cast<const T*>(b.data() + b.size()));
}

template <class T>
T* up(const std::vector<T>& v) {
  T* d = nullptr;
  LAB_CHECK(cudaMalloc(&d, v.size() * sizeof(T)));
  LAB_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

template <int FORM, int SET>
double run(const lab::Bench& b, const std::string& variant, const In& in, const Host& h) {
  const std::string name = lab::kernel("drain_lab::drain", {FORM, SET});
  if (FORM != 5) b.expect(name, "RED.E.ADD", ">=", 4);
  if (FORM == 2) b.expect(name, "ATOMS.EXCH", ">=", 4);
  if (FORM == 4 || FORM == 7) b.expect(name, "LDGSTS", ">=", 1);
  //: fixed counts (the expect pass runs before any input is read): 80 boxes, the cell's 64 or 32 windows
  b.units(name, "a CTA-window", 80LL * (SET == 0 ? 64 : 32));
  b.exempt(name, "a rate probe of the drain's forms; no MMAs");
  if (b.expect_only) return 0.0;
  LAB_CHECK(cudaFuncSetAttribute(drain<FORM, SET>, cudaFuncAttributeMaxDynamicSharedMemorySize, kBytes));
  const auto launch = [&] { drain<FORM, SET><<<h.ctas, kThreads, kBytes>>>(in, b.times); };
  const lab::Samples s = b.variant(launch, [&] { return b.span(kWarps, h.ctas) / h.windows; });
  if (FORM == 5) {
    b.unchecked(variant, "a bound: the fill alone, nothing drained");
  } else {
    LAB_CHECK(cudaMemset(in.num, 0, h.num.size() * 4));
    launch();
    LAB_CHECK(cudaDeviceSynchronize());
    std::vector<uint32_t> got(h.num.size());
    LAB_CHECK(cudaMemcpy(got.data(), in.num, got.size() * 4, cudaMemcpyDeviceToHost));
    size_t bad = 0;
    for (size_t i = 0; i < got.size(); ++i) bad += got[i] != h.num[i];
    b.check(variant, bad == 0, std::to_string(got.size() - bad) + " of " + std::to_string(got.size()) +
                                   " num words exact");
  }
  b.record(variant, "cycles a window", s);
  return s.median();
}

//: the half-line requests a CTA-window FORM 8 issues (a warp's groups holding a live slot, 4 adds of two lines each),
//: the mean over the cell's CTA-windows
double requests_skipping(const Host& h) {
  double n = 0;
  for (int w = 0; w < h.windows; ++w)
    for (int b = 0; b < h.ctas; ++b) {
      const int u = h.counts[w * kGroups + b / 16];
      for (int p = 0; p * kSlots < u; ++p) {
        const int m = std::min(kSlots, u - p * kSlots);
        for (int warp = 0; warp < kWarps; ++warp) n += 8.0 * std::min(4, std::max(0, (m - 32 * warp + 7) / 8));
      }
    }
  return n / (static_cast<double>(h.windows) * h.ctas);
}

//: the global path's price of a line request with every SM draining (`common/hardware.json`'s `global.red_request`):
//: the skipping form's cycles less the fill's, over its requests a CTA-window
void red_request(const lab::Bench& b, const std::string& label, double skipping, double fill, const Host& h) {
  if (b.expect_only) return;
  const std::string row = label + "cycles a line request, every SM draining";
  b.unchecked(row, "the difference of two checked forms' medians over the requests the inputs fix");
  b.record(row, "cycles a request", lab::Samples{{(skipping - fill) / requests_skipping(h)}});
}

template <int SET>
void cell(const lab::Bench& b, const char* label, const In& in, const Host& h) {
  const std::string l = label;
  const double fill = run<5, SET>(b, l + "bound: the fill alone", in, h);
  run<0, SET>(b, l + "slots (s, s + 1), load and 4-byte clear", in, h);
  run<1, SET>(b, l + "slots (s, s + 4), load and 4-byte clear", in, h);
  run<2, SET>(b, l + "slots (s, s + 4), exchange", in, h);
  run<3, SET>(b, l + "slots (s, s + 4), load; 16-byte clear sweep", in, h);
  run<6, SET>(b, l + "bound: slots (s, s + 4), no clear", in, h);
  run<4, SET>(b, l + "slots (s, s + 4) beside copies, 512 bytes every 4 adds", in, h);
  run<7, SET>(b, l + "slots (s, s + 4) beside copies, 512 bytes every add", in, h);
  const double skipping =
      run<8, SET>(b, l + "slots (s, s + 4), load; 16-byte clear sweep; groups past the union skipped", in, h);
  red_request(b, l, skipping, fill, h);
}

}  // namespace drain_lab

int main(int argc, char** argv) {
  using namespace drain_lab;
  lab::Bench b(argc, argv);
  std::string dir;
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], "--inputs")) dir = argv[i + 1];
  Host h[3];
  In in[3] = {};
  const char* sub[3] = {"", "/k16", "/c16"};
  if (!b.expect_only) {
    if (dir.empty()) {
      std::fprintf(stderr, "drain: --inputs <directory> (tools/lab.py writes it)\n");
      return 1;
    }
    uint8_t* table = nullptr;
    LAB_CHECK(cudaMalloc(&table, kTable));
    LAB_CHECK(cudaMemset(table, 0x5a, kTable));
    for (int k = 0; k < 3; ++k) {
      const std::vector<int> head = as<int>(file(dir + sub[k] + "/head.bin"));
      h[k].ctas = head[1];
      h[k].windows = head[2];
      h[k].counts = as<uint16_t>(file(dir + sub[k] + "/counts.bin"));
      h[k].tokens = as<uint16_t>(file(dir + sub[k] + "/tokens.bin"));
      h[k].num = as<uint32_t>(file(dir + sub[k] + "/num.bin"));
      uint32_t* num = nullptr;
      const size_t words = h[k].num.size() + static_cast<size_t>(h[k].ctas) * 2 * kWords;
      LAB_CHECK(cudaMalloc(&num, words * 4));
      LAB_CHECK(cudaMemset(num, 0, words * 4));
      in[k] = In{up(h[k].counts), up(h[k].tokens), num, table, h[k].windows};
    }
  }
  cell<0>(b, "alt-k4: ", in[0], h[0]);
  cell<1>(b, "alt-k16: ", in[1], h[1]);
  cell<2>(b, "cohort-k16: ", in[2], h[2]);
}
