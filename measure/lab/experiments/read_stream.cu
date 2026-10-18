// measure/lab/experiments/read_stream.cu -- QUESTION: when the readout's rows are streamed a digit at a time into each
// warp's ring (merge width 1) instead of gathered from a staged pool, what does the stream cost beyond the readout,
// and at what ring depth? METHOD: the readout's tokens-on-N form (`read_image.cuh`: a tile 8 tokens, 4 bf16 MMAs
// against the digit's state from the image, each output pair exact and added into its token's panel row; each pass's
// every digit's first tile in one straight-line block, then each digit's rest; the panel drained a pass) over three
// cells' read lists (`read_stream.inputs.py`): 80 CTAs, box `b` of one head, a warp its digits w, w + 8, w + 16,
// w + 24; a stage a tile's 8 read rows (a token's 64 bytes of its slice group, shared between the boxes as the
// carry's are, so L2 serves what one box fetched to the others) and its 8 entry words, copied ahead across digits,
// passes and windows from the warp's window records (a pre-pass's plan): its tokens staged a window ahead, its
// words copied from global memory with its rows. Bounds: the readout with the producer and no row copies, the copies
// alone, the readout with no producer. A sample is the CTA's span a window; each forming variant's drained panel is
// checked word for word.
// see docs/internals/tools/lab.md#read-stream
#include <cstdlib>

#include "read_image.cuh"

namespace read_stream {

using read_image::kPanelPitch;
using read_image::kPlaneBytes;
using read_image::kPlanes;
using read_image::kSlots;
using read_image::kThreads;
using read_image::kWarps;
using read_image::kWords;

constexpr int kWindow = 512, kGroups = 8, kTiles = 2, kSliceGroups = 8, kRow = 64, kPasses = 2;
//: THE LAYOUT: the image's planes, each warp's ring of `D` stages (`T` tiles' 8 read rows each, then their entry
//: words), the lanes' pad bases (128 bytes), the 8 dummy panel rows and the panel, each warp's two token records
constexpr int kRing = kPlanes * kPlaneBytes;

template <int D, int T>
struct Layout {
  static constexpr int kStage = T * 8 * kRow + 64;
  static constexpr int kPads = kRing + kWarps * D * kStage, kPanel = kPads + 128 + 8 * kPanelPitch * 4;
  static constexpr int kRec = kPanel + kSlots * kPanelPitch * 4;
};

//: THE INPUTS on the device: a box-window's passes `[windows][8]`, a read digit's tiles in a pass `[windows][2][256]`,
//: the token records (`token_bytes` each) and word records (`word_bytes`), per window, read group and warp, a token's
//: read rows `[windows x 512][8 slice groups][64 B]`, and each CTA's drained panel `[ctas][2][256][16]`
struct In {
  const uint16_t* passes;
  const uint16_t* tiles;
  const uint8_t* rec;
  const uint8_t* words;
  const uint8_t* rows;
  uint32_t* panel_out;
  int windows, token_bytes, word_bytes;
};

//: THIS CTA'S BOX (the carry's order, its channel tile fastest): its slice group and read group
struct Box {
  int slice, read;
};

__device__ __forceinline__ Box box_of() {
  const int b = static_cast<int>(blockIdx.x);
  return Box{(b / kTiles) % kSliceGroups, b / (kTiles * kSliceGroups)};
}

//: A LANE'S CONSTANTS for tokens on N (`read_image::lane_of<1>`), its pad base in this layout
__device__ __forceinline__ read_image::Lane lane_of(uint32_t sm, int warp, int lane, int pads) {
  const int r = lane & 7, q = lane >> 3, g = lane >> 2, t = lane & 3;
  const uint32_t plane = sm + (warp % kPlanes) * kPlaneBytes;
  const int srow = 8 * (lane >> 4) + r, sc = q & 1;
  const auto h = [](auto v) { return static_cast<decltype(v)>(lab::held(static_cast<uint32_t>(v))); };
  return {h(r), h(2 * t), h(2 * t + 1), h(16u * q), h(sm + pads + 4u * g + 32u * t),
          h(plane + read_image::in_plane(srow, sc)), h(plane + read_image::in_plane(srow, 2 + sc)), h(8u * r)};
}

__device__ __forceinline__ void copy16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}

__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }

template <int N>
__device__ __forceinline__ void wait_groups() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

__device__ __forceinline__ uint32_t lds32(uint32_t a) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a));
  return v;
}

__device__ __forceinline__ uint32_t lds16(uint32_t a) {
  uint16_t v;
  asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a));
  return v;
}

//: THE READOUT STREAMED: each warp's ring of `D` stages (`D >= 4`: the first tiles' block holds four); the warp's
//: stages its window records' in order, each copied as soon as a slot no tile holds is free, across digits, passes and
//: windows -- the producer reading a stage's tokens from its token record (copied a window ahead) and copying its
//: rows and its entry words; the consumer its passes and its digits' tiles a window ahead into registers, the panel
//: drained after each pass. `PART` 1 the readout with the producer, no row copies (the ring's stale rows), 2 the copies
//: alone, 3 the readout with no producer (its entry words from global memory, a tile ahead), 4 the stream with no waits
//: on its stages (a bound: what waiting on the copies costs). `SET` names the dataset's
//: instantiation, so the profiler keeps each dataset's counters apart. A token record lands with the stage group it was
//: issued in, a window ahead: every warp-window holds at least 4 stages (a digit's first tile a pass), so at `D <= 5`
//: a window's entry waits only on the groups the ring's next stage needs. `WL`: the consumer loads its entry words from
//: global memory a tile ahead (as with no producer), the producer copying rows alone. `TA`: the producer loads the next
//: stage's token a stage ahead. `T`: tiles a stage (2: the records' paired stages, a pass's last perhaps one tile)
template <int D, int PART, int SET, int WL, int TA, int T>
__global__ void __launch_bounds__(kThreads, 1) stream(In in, long long* t) {
  static_assert(T == 1 || (!TA && !WL), "a stage of two tiles loads its two tokens at once and copies its words");
  extern __shared__ __align__(128) uint8_t smem[];
  using L = Layout<D, T>;
  constexpr int kStage = L::kStage;
  const int warp = lab::warp_id(), lane = lab::lane_id(), W = in.windows, C = in.token_bytes, CW = in.word_bytes,
            r = lane >> 2, c = lane & 3;
  const Box x = box_of();
  const uint32_t sm = lab::smem(smem);
  //: the image's even channels -0.0625 and odd +0.0625 (bf16), the rest zero
  uint4* chunks = reinterpret_cast<uint4*>(smem);
  for (int k = static_cast<int>(threadIdx.x); k < (L::kRec + kWarps * 2 * C) / 16; k += kThreads) {
    const uint32_t v = k < kRing / 16 ? 0x3d80bd80u : 0u;
    chunks[k] = make_uint4(v, v, v, v);
  }
  __syncthreads();
  const read_image::Lane lc = lane_of(sm, warp, lane, L::kPads);
  const uint32_t ring = sm + kRing + warp * D * kStage, ring_last = ring + (D - 1) * kStage;
  const uint32_t recs = sm + L::kRec + warp * 2 * C;
  const auto next_slot = [&](uint32_t slot) { return slot == ring_last ? ring : slot + kStage; };
  float sink = 0.0f;
  const auto warp_record = [&](int w) {
    return (static_cast<size_t>(w) * kGroups + x.read) * kWarps + warp;
  };
  //: window `w`'s token record into its buffer `w % 2` (fetched a window ahead), a lane a 16-byte chunk
  const auto fetch = [&](int w) {
#pragma unroll 1
    for (int q = lane; q < (w < W ? C / 16 : 0); q += 32)
      copy16(recs + (w & 1) * C + 16 * q, in.rec + warp_record(w) * C + 16 * q);
  };
  //: THE PRODUCER'S CURSOR: the window and stage it copies next and the window's stages; its running addresses -- the
  //: lane's token in the window's record, the stage's words in global memory, its row's base in the window's rows,
  //: the ring slot it fills next; the groups committed and the group carrying the next window's record
  int pw = 0, ps = 0, groups = 0, carrier = 0;
  fetch(0);
  commit();
  wait_groups<0>();
  __syncwarp();
  fetch(1);
  const uint32_t row_c = r * kRow + 16 * (c ^ ((r >> 1) & 3));
  const uint8_t* const rows_c = in.rows + x.slice * kRow + 16 * c;
  uint32_t pe = recs + 16 + 2 * T * r, pslot = ring;
  const uint8_t* pwd = in.words + warp_record(0) * CW + 16 * (lane & (2 * T - 1));
  const uint8_t* prow = rows_c;
  int S = static_cast<int>(lds16(recs));
  uint32_t tk_next = TA ? lds16(pe) : 0u;
  //: the cursor to the next window holding stages, its record landed first (a wait on every group but the newest D - 1,
  //: or every group where fewer have been committed since)
  const auto enter = [&] {
#pragma unroll 1
    do {
      ++pw;
      ps = 0;
      if (pw < W) {
        if (groups - 1 - carrier < D - 1) {
          commit();
          ++groups;
          wait_groups<0>();
        } else {
          wait_groups<D - 1>();
        }
        __syncwarp();
      }
      fetch(pw + 1);
      carrier = groups;
      const uint32_t rec = recs + (pw & 1) * C;
      pe = rec + 16 + 2 * T * r;
      pwd = in.words + warp_record(pw) * CW + 16 * (lane & (2 * T - 1));
      prow = rows_c + static_cast<size_t>(pw) * kWindow * kSliceGroups * kRow;
      S = pw < W ? static_cast<int>(lds16(rec)) : 0;
    } while ((pw < W) & (S == 0));
  };
  const auto issue = [&] {
    if (pw < W) {
      const uint32_t tk = TA ? tk_next : T == 1 ? lds16(pe) : lds32(pe);
      if constexpr (PART != 1) {
        copy16(pslot + row_c, prow + (tk & 0xffffu) * (kSliceGroups * kRow));
        if constexpr (T == 2) copy16(pslot + 8 * kRow + row_c, prow + (tk >> 16) * (kSliceGroups * kRow));
      }
      if constexpr (!WL) {
        if (lane < 2 * T) copy16(pslot + T * 8 * kRow + 16 * lane, pwd);
      }
      pe += 16 * T;
      pwd += 32 * T;
      if (++ps >= S) enter();
      if constexpr (TA) tk_next = pw < W ? lds16(pe) : 0u;
    }
    pslot = next_slot(pslot);
    commit();
    ++groups;
  };
  const long long t0 = clock64();
  if constexpr (PART != 3) {
#pragma unroll 1
    for (int q = 0; q < D - 1; ++q) issue();
  }
  //: THE CONSUMER: its next ring slot, its window's passes and its digits' tiles a pass (lane `i` digit `w + 8 i`'s, of
  //: pass 0 and 1), each held a window ahead; with no producer, its window's words at their use
  const auto passes_of = [&](int w) {
    return w < W ? static_cast<int>(in.passes[static_cast<size_t>(w) * kGroups + x.read]) : 0;
  };
  const auto tiles_of = [&](int w, int p) {
    return w < W && lane < 4 ? static_cast<int>(in.tiles[(static_cast<size_t>(w) * kPasses + p) * 256 + 32 * x.read +
                                                         warp + 8 * lane])
                             : 0;
  };
  int np = passes_of(0), np_next = passes_of(1), ct0 = tiles_of(0, 0), ct1 = tiles_of(0, 1);
  int ct0_next = tiles_of(1, 0), ct1_next = tiles_of(1, 1);
  uint32_t cslot = ring;
  const uint32_t row_r = static_cast<uint32_t>((lane & 7) * kRow + 16 * (((lane & 7) >> 1) & 3));
  const uint32_t* cwd = nullptr;
  uint32_t cw_next = 0;
  //: a tile, the `half`-th of the stage at ring slot `slot`: its rows and its entry words from the slot (from global
  //: memory where no producer runs, the next tile's loaded a tile ahead)
  const auto tile = [&](const uint32_t (&f)[16], uint32_t slot, int half) {
    uint32_t word;
    if constexpr (PART == 3 || WL) {
      word = cw_next;
      cwd += 8;
      cw_next = __ldg(cwd + (lane & 7));
    } else {
      word = lds32(slot + T * 8 * kRow + 32 * half + 4 * (lane & 7));
    }
    read_image::tile_n<1>(sm, make_uint2(slot + 8 * kRow * half + row_r - sm, word), lc, f, sink);
  };
  //: the drain of pass `p`: each warp's 32 slots, two a request, each word cleared
  const auto drain = [&](int p) {
    uint32_t* words = reinterpret_cast<uint32_t*>(smem);
#pragma unroll 1
    for (int k = 0; k < 16; ++k) {
      const int slot = 32 * warp + 2 * k + (lane >> 4), word = lane & 15;
      uint32_t* cell = words + L::kPanel / 4 + slot * kPanelPitch + word;
      uint32_t* target =
          in.panel_out + ((static_cast<int64_t>(blockIdx.x) * kPasses + p) * kSlots + slot) * kWords + word;
      asm volatile("red.global.add.u32 [%0], %1;" ::"l"(target), "r"(*cell) : "memory");
      *cell = 0u;
    }
  };
#pragma unroll 1
  for (int w = 0; w < W; ++w) {
    if constexpr (PART == 3 || WL) {
      cwd = reinterpret_cast<const uint32_t*>(in.words + warp_record(w) * CW);
      cw_next = __ldg(cwd + (lane & 7));
    }
#pragma unroll 1
    for (int p = 0; p < np; ++p) {
      const int ct = p == 0 ? ct0 : ct1;
      //: every digit's first tile, one block: its 4 / T slots held, the ring's other slots ahead
      if constexpr (PART != 3) {
        issue();
        if constexpr (PART != 4) wait_groups<D - 4 / T>();
        __syncwarp();
      }
      uint32_t f[4][16];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if constexpr (PART != 2) {
          read_image::state_in<1>(f[i], lc, read_image::plane_off(i));
          tile(f[i], cslot, i % T);
        }
        if (i % T == T - 1) cslot = next_slot(cslot);
      }
      if constexpr (PART != 3) {
#pragma unroll 1
        for (int q = 0; q < 4 / T - 1; ++q) issue();
      }
      //: each digit's rest, a stage's first tile waiting for its stage
      int k = 0;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int n = __shfl_sync(0xffffffffu, ct, i) - 1;
#pragma unroll 1
        for (int s = 0; s < n; ++s, ++k) {
          if constexpr (PART != 3) {
            if (k % T == 0) {
              issue();
              if constexpr (PART != 4) wait_groups<D - 1>();
              __syncwarp();
            }
          }
          if constexpr (PART != 2) tile(f[i], cslot, k % T);
          if (k % T == T - 1) cslot = next_slot(cslot);
        }
      }
      if (k % T != 0) cslot = next_slot(cslot);
      __syncthreads();
      if constexpr (PART != 2) drain(p);
      __syncthreads();
    }
    np = np_next;
    np_next = passes_of(w + 2);
    ct0 = ct0_next;
    ct1 = ct1_next;
    ct0_next = tiles_of(w + 2, 0);
    ct1_next = tiles_of(w + 2, 1);
  }
  wait_groups<0>();
  const long long t1 = clock64();
  if (lane == 0) {
    t[2 * (blockIdx.x * kWarps + warp)] = t0;
    t[2 * (blockIdx.x * kWarps + warp) + 1] = t1;
  }
  (void)sink;
}

//: the inputs from a dataset's directory
struct Host {
  std::vector<uint16_t> passes, tiles;
  std::vector<uint8_t> rec, words, rec2, words2, rows;
  std::vector<uint32_t> panel;
  int ctas = 0, windows = 0, token_bytes = 0, word_bytes = 0, token_bytes2 = 0, word_bytes2 = 0;
};

std::vector<uint8_t> file(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    std::fprintf(stderr, "read_stream: %s is missing\n", path.c_str());
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

Host load(const std::string& dir) {
  Host h;
  const std::vector<int> head = as<int>(file(dir + "/head.bin"));
  h.ctas = head[1];
  h.windows = head[2];
  h.token_bytes = head[3];
  h.token_bytes2 = head[5];
  h.word_bytes2 = head[6];
  h.word_bytes = head[4];
  h.passes = as<uint16_t>(file(dir + "/passes.bin"));
  h.tiles = as<uint16_t>(file(dir + "/tiles.bin"));
  h.panel = as<uint32_t>(file(dir + "/panel.bin"));
  h.rec = file(dir + "/records.bin");
  h.words = file(dir + "/words.bin");
  h.rec2 = file(dir + "/records2.bin");
  h.words2 = file(dir + "/words2.bin");
  h.rows = file(dir + "/rows.bin");
  return h;
}

template <class T>
T* up(const std::vector<T>& v) {
  T* d = nullptr;
  LAB_CHECK(cudaMalloc(&d, v.size() * sizeof(T)));
  LAB_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

template <int D, int PART, int SET, int WL = 0, int TA = 0, int T = 1>
void run(const lab::Bench& b, const std::string& variant, const In& in, const Host& h) {
  const std::string name = lab::kernel("read_stream::stream", {D, PART, SET, WL, TA, T});
  b.expect(name, "LDGSTS", ">=", 1);
  b.expect(name, "HMMA.16816.F32.BF16", PART == 2 ? "==" : ">=", PART == 2 ? 0 : 4);
  //: the producer's cursor runs inside the readout's tile loop, a stage a tile; with no producer, the entry words load
  //: from global memory inside it
  b.expect(name, "BSSY", ">=", 0);
  if (PART == 3 || WL) b.expect(name, "STEP_GLOBAL", ">=", 0);
  b.exempt(name, "a copy-overlap probe: the readout's tile over streamed rows, its cost the question");
  if (b.expect_only) return;
  const int bytes = Layout<D, T>::kRec + kWarps * 2 * (T == 1 ? h.token_bytes : h.token_bytes2);
  LAB_CHECK(cudaFuncSetAttribute(stream<D, PART, SET, WL, TA, T>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  LAB_CHECK(cudaMemset(in.panel_out, 0, h.panel.size() * 4));
  const lab::Samples s = b.variant([&] { stream<D, PART, SET, WL, TA, T><<<h.ctas, kThreads, bytes>>>(in, b.times); },
                                   [&] { return b.span(kWarps, h.ctas) / h.windows; });
  if (PART) {
    b.unchecked(variant, PART == 2   ? "a bound: copies and their waits alone, no panel formed"
                         : PART == 4 ? "a bound: the stream with no waits, its rows racing their copies"
                                     : "a bound: the readout over stale rows, no panel formed");
  } else {
    //: one launch from a cleared panel: the drains of the timed launches summed over their repetitions
    LAB_CHECK(cudaMemset(in.panel_out, 0, h.panel.size() * 4));
    stream<D, PART, SET, WL, TA, T><<<h.ctas, kThreads, bytes>>>(in, b.times);
    LAB_CHECK(cudaDeviceSynchronize());
    std::vector<uint32_t> got(h.panel.size());
    LAB_CHECK(cudaMemcpy(got.data(), in.panel_out, got.size() * 4, cudaMemcpyDeviceToHost));
    size_t bad = 0;
    for (size_t i = 0; i < got.size(); ++i) bad += got[i] != h.panel[i];
    b.check(variant, bad == 0, std::to_string(got.size() - bad) + " of " + std::to_string(got.size()) +
                                   " drained panel words exact (each CTA's, summed over the windows)");
  }
  b.record(variant, "cycles a window", s);
}

}  // namespace read_stream

int main(int argc, char** argv) {
  using namespace read_stream;
  lab::Bench b(argc, argv);
  std::string dir;
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], "--inputs")) dir = argv[i + 1];
  //: each dataset's inputs (`read_stream.inputs.py`'s `DATASETS`): nl64k-alt-k4, flagship-alt-k16 in `k16/` and
  //: flagship-cohort-k16 in `c16/`
  Host h[3];
  In in[3] = {}, in2[3] = {};
  const char* sub[3] = {"", "/k16", "/c16"};
  if (!b.expect_only) {
    if (dir.empty()) {
      std::fprintf(stderr, "read_stream: --inputs <directory> (tools/lab.py writes it)\n");
      return 1;
    }
    for (int k = 0; k < 3; ++k) {
      h[k] = load(dir + sub[k]);
      if (h[k].ctas > b.sms) {
        std::fprintf(stderr, "read_stream: %d boxes need as many SMs, the device has %d\n", h[k].ctas, b.sms);
        return 1;
      }
      uint32_t* out = nullptr;
      LAB_CHECK(cudaMalloc(&out, h[k].panel.size() * 4));
      in[k] = In{up(h[k].passes), up(h[k].tiles), up(h[k].rec), up(h[k].words), up(h[k].rows), out,
                 h[k].windows, h[k].token_bytes, h[k].word_bytes};
      in2[k] = in[k];
      in2[k].rec = up(h[k].rec2);
      in2[k].words = up(h[k].words2);
      in2[k].token_bytes = h[k].token_bytes2;
      in2[k].word_bytes = h[k].word_bytes2;
    }
  }
  run<4, 0, 0>(b, "streamed, ring 4 (the first tiles' block, nothing ahead)", in[0], h[0]);
  run<5, 0, 0>(b, "streamed, ring 5", in[0], h[0]);
  run<6, 0, 0>(b, "streamed, ring 6", in[0], h[0]);
  run<5, 1, 0>(b, "bound: the readout with the producer, no row copies (ring 5)", in[0], h[0]);
  run<5, 2, 0>(b, "bound: the copies alone (ring 5)", in[0], h[0]);
  run<5, 3, 0>(b, "bound: the readout alone, no producer", in[0], h[0]);
  run<5, 4, 0>(b, "bound: the stream with no waits on its stages (ring 5)", in[0], h[0]);
  run<5, 0, 0, 1>(b, "streamed, ring 5, the consumer's words from global memory a tile ahead", in[0], h[0]);
  run<5, 1, 0, 1>(b, "bound: the consumer's words a tile ahead, the producer and no row copies", in[0], h[0]);
  run<5, 0, 0, 0, 1>(b, "streamed, ring 5, the next stage's token a stage ahead", in[0], h[0]);
  run<3, 0, 0, 0, 0, 2>(b, "streamed, two tiles a stage, ring 3 (6 tiles)", in2[0], h[0]);
  run<3, 1, 0, 0, 0, 2>(b, "bound: two tiles a stage, the producer and no row copies", in2[0], h[0]);
  run<5, 1, 0, 0, 1>(b, "bound: the token a stage ahead, the producer and no row copies", in[0], h[0]);
  run<5, 0, 1>(b, "alt-k16: streamed, ring 5", in[1], h[1]);
  run<5, 1, 1>(b, "alt-k16: bound: the readout with the producer, no row copies", in[1], h[1]);
  run<5, 3, 1>(b, "alt-k16: bound: the readout alone, no producer", in[1], h[1]);
  run<5, 0, 1, 1>(b, "alt-k16: streamed, ring 5, the consumer's words a tile ahead", in[1], h[1]);
  run<5, 0, 1, 0, 1>(b, "alt-k16: streamed, ring 5, the next stage's token a stage ahead", in[1], h[1]);
  run<3, 0, 1, 0, 0, 2>(b, "alt-k16: streamed, two tiles a stage, ring 3", in2[1], h[1]);
  run<5, 0, 2>(b, "cohort-k16: streamed, ring 5", in[2], h[2]);
  run<5, 1, 2>(b, "cohort-k16: bound: the readout with the producer, no row copies", in[2], h[2]);
  run<5, 3, 2>(b, "cohort-k16: bound: the readout alone, no producer", in[2], h[2]);
  run<5, 0, 2, 1>(b, "cohort-k16: streamed, ring 5, the consumer's words a tile ahead", in[2], h[2]);
  run<5, 0, 2, 0, 1>(b, "cohort-k16: streamed, ring 5, the next stage's token a stage ahead", in[2], h[2]);
  run<3, 0, 2, 0, 0, 2>(b, "cohort-k16: streamed, two tiles a stage, ring 3", in2[2], h[2]);
}
