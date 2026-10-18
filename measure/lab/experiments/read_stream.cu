// measure/lab/experiments/read_stream.cu -- QUESTION: when the readout's rows are streamed a digit at a time into each
// warp's ring (merge width 1) instead of gathered from a staged pool, how much of their copies hides under the
// readout, and at what ring depth? METHOD: the readout's tokens-on-N form (`read_image.cuh`: a tile 8 tokens, 4 bf16
// MMAs against the digit's state from the image, each output pair exact and added into its token's panel row; every
// digit's first tile in one straight-line block, then each digit's rest) over real
// nl64k-alt-k4 read lists (`read_stream.inputs.py`): 80 CTAs, box `b` of one head over 64 windows, a warp its digits
// w, w + 8, w + 16, w + 24; a stage a tile's 8 read rows (a token's 64 bytes of its slice group, shared between the
// boxes as the carry's are, so L2 serves what one box fetched to the others) and its 8 entry words, copied ahead across
// digits and windows from the warp's window stage record (a pre-pass's plan) staged a window ahead. Bounds: the
// readout alone (no row copies) and the copies alone. A sample is the CTA's span a window; each forming variant's panel
// is checked word for word.
// see docs/internals/tools/lab.md#read-stream
#include "read_image.cuh"

namespace read_stream {

using read_image::kPanelPitch;
using read_image::kPlaneBytes;
using read_image::kPlanes;
using read_image::kSlots;
using read_image::kThreads;
using read_image::kWarps;
using read_image::kWords;

constexpr int kWindow = 512, kGroups = 8, kTiles = 2, kSliceGroups = 8, kRow = 64;
//: THE LAYOUT: the image's planes, each warp's ring of `D` stages (8 read rows, then the stage's 8 entry words), the
//: lanes' pad bases (128 bytes), the 8 dummy panel rows and the panel, each warp's two records
constexpr int kRing = kPlanes * kPlaneBytes, kStage = 8 * kRow + 64;

template <int D>
struct Layout {
  static constexpr int kPads = kRing + kWarps * D * kStage, kPanel = kPads + 128 + 8 * kPanelPitch * 4;
  static constexpr int kRec = kPanel + kSlots * kPanelPitch * 4;
};

//: THE INPUTS on the device: a read digit's tiles `[windows][256]`, the stage records (`record_bytes` each, per window,
//: read group and warp), a token's read rows `[windows x 512][8 slice groups][64 B]`, and each CTA's panel out
struct In {
  const uint16_t* tiles;
  const uint8_t* rec;
  const uint8_t* rows;
  uint32_t* panel_out;
  int windows, record_bytes;
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

__device__ __forceinline__ void copy16(uint32_t dst, const void* src, bool live) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(live ? 16 : 0) : "memory");
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

__device__ __forceinline__ void sts32(uint32_t a, uint32_t v) {
  asm volatile("st.shared.u32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}

//: THE READOUT STREAMED: each warp's ring of `D` stages (`D >= 4`: the first tiles' block holds four); the warp's
//: stages its window records' in order (every digit's first tile, then each digit's rest), each copied as soon as a
//: slot no tile holds is free, across digits and windows, the producer reading its tokens and entry words from the
//: record (copied a window ahead) and writing the words beside the rows; the consumer its digits' tiles a window ahead
//: into registers. `PART` 1 the readout alone (no row copies, the ring's stale rows), 2 the copies alone, 3 the readout
//: alone with no producer (its entry words from the record, staged a window ahead)
template <int D, int PART>
__global__ void __launch_bounds__(kThreads, 1) stream(In in, long long* t) {
  extern __shared__ __align__(128) uint8_t smem[];
  using L = Layout<D>;
  const int warp = lab::warp_id(), lane = lab::lane_id(), W = in.windows, C = in.record_bytes, r = lane >> 2,
            c = lane & 3;
  const Box x = box_of();
  const uint32_t sm = lab::smem(smem);
  //: the image's even channels -0.0625 and odd +0.0625 (bf16), the rest zero
  uint4* chunks = reinterpret_cast<uint4*>(smem);
  for (int k = static_cast<int>(threadIdx.x); k < (L::kRec + kWarps * 3 * C) / 16; k += kThreads) {
    const uint32_t v = k < kRing / 16 ? 0x3d80bd80u : 0u;
    chunks[k] = make_uint4(v, v, v, v);
  }
  __syncthreads();
  const read_image::Lane lc = lane_of(sm, warp, lane, L::kPads);
  const uint32_t ring = sm + kRing + warp * D * kStage, recs = sm + L::kRec + warp * 3 * C;
  float sink = 0.0f;
  const auto record = [&](int w) {
    return in.rec + ((static_cast<size_t>(w) * kGroups + x.read) * kWarps + warp) * C;
  };
  //: window `w`'s record into its buffer `w % 3` (fetched two windows ahead), a lane a 16-byte chunk
  const auto fetch = [&](int w) {
#pragma unroll 1
    for (int q = lane; q < (w < W ? C / 16 : 0); q += 32) copy16(recs + (w % 3) * C + 16 * q, record(w) + 16 * q, true);
  };
  //: THE PRODUCER'S CURSOR: the window and stage it copies next and the window's stages; its running addresses -- the
  //: lane's entry in the window's record, its row's base in the window's rows, the ring slot it fills next; the groups
  //: committed and the groups carrying the next two windows' records
  int pw = 0, ps = 0, groups = 0, carrier = 0, carrier_next = 0;
  fetch(0);
  commit();
  wait_groups<0>();
  __syncwarp();
  fetch(1);
  fetch(2);
  const uint32_t ring_last = ring + (D - 1) * kStage, row_c = r * kRow + 16 * (c ^ ((r >> 1) & 3));
  const auto next_slot = [&](uint32_t slot) { return slot == ring_last ? ring : slot + kStage; };
  const uint8_t* const rows_c = in.rows + x.slice * kRow + 16 * c;
  uint32_t pe = recs + 16 + 8 * r, pslot = ring;
  const uint8_t* prow = rows_c;
  int S = static_cast<int>(lds32(recs + 8));
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
      fetch(pw + 2);
      carrier = carrier_next;
      carrier_next = groups;
      const uint32_t rec = recs + (pw % 3) * C;
      pe = rec + 16 + 8 * r;
      prow = rows_c + static_cast<size_t>(pw) * kWindow * kSliceGroups * kRow;
      S = pw < W ? static_cast<int>(lds32(rec + 8)) : 0;
    } while ((pw < W) & (S == 0));
  };
  const auto issue = [&] {
    if (pw < W) {
      const uint2 e = read_image::lds64(pe);
      const bool live = e.x != 0xFFFFu;
      if constexpr (PART != 1) copy16(pslot + row_c, prow + (live ? e.x : 0u) * (kSliceGroups * kRow), live);
      if (c == 0) sts32(pslot + 8 * kRow + 4 * r, e.y);
      pe += 64;
      if (++ps >= S) enter();
    }
    pslot = next_slot(pslot);
    commit();
    ++groups;
  };
  const long long t0 = clock64();
#pragma unroll 1
  for (int q = 0; q < D - 1; ++q) issue();
  //: THE CONSUMER: its next ring slot, its digits' tiles in its window and the next (lane `i` digit `w + 8 i`'s), and
  //: (with no producer) the lane's entry word in the window's record
  const auto tiles_of = [&](int w) {
    return w < W && lane < 4 ? static_cast<int>(in.tiles[static_cast<size_t>(w) * 256 + 32 * x.read + warp + 8 * lane])
                             : 0;
  };
  int ct = tiles_of(0), ct_next = tiles_of(1);
  uint32_t cslot = ring, crec = 0;
  const uint32_t row_r = static_cast<uint32_t>((lane & 7) * kRow + 16 * (((lane & 7) >> 1) & 3));
  //: a tile at ring slot `slot`: its rows from the slot, its entry words from the slot (from the record where no
  //: producer runs)
  const auto tile = [&](const uint32_t (&f)[16], uint32_t slot) {
    uint32_t word;
    if constexpr (PART == 3) {
      word = lds32(crec);
      crec += 64;
    } else {
      word = lds32(slot + 8 * kRow + 4 * (lane & 7));
    }
    read_image::tile_n<1>(sm, make_uint2(slot + row_r - sm, word), lc, f, sink);
  };
#pragma unroll 1
  for (int w = 0; w < W; ++w) {
    //: every digit's first tile, one block: four slots held, the ring's other slots ahead
    if constexpr (PART == 3) {
      fetch(w + 2);
      commit();
      wait_groups<1>();
      crec = recs + (w % 3) * C + 16 + 8 * (lane & 7) + 4;
    } else {
      issue();
      wait_groups<D - 4>();
    }
    __syncwarp();
    uint32_t f[4][16];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if constexpr (PART != 2) {
        read_image::state_in<1>(f[i], lc, read_image::plane_off(i));
        tile(f[i], cslot);
      }
      cslot = next_slot(cslot);
    }
    if constexpr (PART != 3) {
#pragma unroll 1
      for (int q = 0; q < 3; ++q) issue();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int n = __shfl_sync(0xffffffffu, ct, i) - 1;
#pragma unroll 1
      for (int s = 0; s < n; ++s) {
        if constexpr (PART != 3) {
          issue();
          wait_groups<D - 1>();
          __syncwarp();
        }
        if constexpr (PART != 2) tile(f[i], cslot);
        cslot = next_slot(cslot);
      }
    }
    ct = ct_next;
    ct_next = tiles_of(w + 2);
  }
  wait_groups<0>();
  const long long t1 = clock64();
  if (lane == 0) {
    t[2 * (blockIdx.x * kWarps + warp)] = t0;
    t[2 * (blockIdx.x * kWarps + warp) + 1] = t1;
  }
  (void)sink;
  //: the check's read of the panel, after the timed loop
  __syncthreads();
  for (int k = static_cast<int>(threadIdx.x); k < kSlots * kWords / 4; k += kThreads)
    reinterpret_cast<uint4*>(in.panel_out)[static_cast<int64_t>(blockIdx.x) * kSlots * kWords / 4 + k] =
        chunks[(L::kPanel + (k >> 2) * kPanelPitch * 4) / 16 + (k & 3)];
}

//: the inputs from `--inputs`
struct Host {
  std::vector<uint16_t> tiles;
  std::vector<uint8_t> rec, rows;
  std::vector<uint32_t> panel;
  int ctas = 0, windows = 0, record_bytes = 0;
};

std::vector<uint8_t> file(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  std::vector<uint8_t> v(static_cast<size_t>(f.tellg()));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size()));
  return v;
}

Host load(const std::string& dir) {
  Host h;
  const std::vector<uint8_t> head = file(dir + "/head.bin");
  const int* hv = reinterpret_cast<const int*>(head.data());
  h.ctas = hv[1];
  h.windows = hv[2];
  h.record_bytes = hv[3];
  const std::vector<uint8_t> tiles = file(dir + "/tiles.bin"), panel = file(dir + "/panel.bin");
  h.tiles.assign(reinterpret_cast<const uint16_t*>(tiles.data()),
                 reinterpret_cast<const uint16_t*>(tiles.data() + tiles.size()));
  h.panel.assign(reinterpret_cast<const uint32_t*>(panel.data()),
                 reinterpret_cast<const uint32_t*>(panel.data() + panel.size()));
  h.rec = file(dir + "/records.bin");
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

template <int D, int PART>
void run(const lab::Bench& b, const char* variant, const In& in, const Host& h) {
  const std::string name = lab::kernel("read_stream::stream", {D, PART});
  b.expect(name, "LDGSTS", ">=", 1);
  b.expect(name, "HMMA.16816.F32.BF16", PART == 2 ? "==" : ">=", PART == 2 ? 0 : 4);
  //: the producer's cursor runs inside the readout's tile loop, a stage a tile
  b.expect(name, "BSSY", ">=", 0);
  b.exempt(name, "a copy-overlap probe: the readout's tile over streamed rows, its cost the question");
  if (b.expect_only) return;
  const int bytes = Layout<D>::kRec + kWarps * 3 * h.record_bytes;
  LAB_CHECK(cudaFuncSetAttribute(stream<D, PART>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  const lab::Samples s = b.variant([&] { stream<D, PART><<<h.ctas, kThreads, bytes>>>(in, b.times); },
                                   [&] { return b.span(kWarps, h.ctas) / h.windows; });
  if (PART) {
    b.unchecked(variant, PART == 2 ? "a bound: copies and their waits alone, no panel formed"
                                   : "a bound: the readout over stale rows, no panel formed");
  } else {
    std::vector<uint32_t> got(h.panel.size());
    LAB_CHECK(cudaMemcpy(got.data(), in.panel_out, got.size() * 4, cudaMemcpyDeviceToHost));
    size_t bad = 0;
    for (size_t i = 0; i < got.size(); ++i) bad += got[i] != h.panel[i];
    b.check(variant, bad == 0, std::to_string(got.size() - bad) + " of " + std::to_string(got.size()) +
                                   " panel words exact (each CTA's, summed over the windows)");
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
  Host h;
  In in{};
  if (!b.expect_only) {
    if (dir.empty()) {
      std::fprintf(stderr, "read_stream: --inputs <directory> (tools/lab.py writes it)\n");
      return 1;
    }
    h = load(dir);
    if (h.ctas > b.sms) {
      std::fprintf(stderr, "read_stream: %d boxes need as many SMs, the device has %d\n", h.ctas, b.sms);
      return 1;
    }
    uint32_t* out = nullptr;
    LAB_CHECK(cudaMalloc(&out, h.panel.size() * 4));
    in = In{up(h.tiles), up(h.rec), up(h.rows), out, h.windows, h.record_bytes};
  }
  run<4, 0>(b, "streamed, ring 4 (the first tiles' block, nothing ahead)", in, h);
  run<5, 0>(b, "streamed, ring 5", in, h);
  run<6, 0>(b, "streamed, ring 6", in, h);
  run<7, 0>(b, "streamed, ring 7", in, h);
  run<6, 1>(b, "bound: the readout alone (ring 6, no row copies)", in, h);
  run<6, 2>(b, "bound: the copies alone (ring 6)", in, h);
  run<6, 3>(b, "bound: the readout alone, no producer (entries from the record)", in, h);
}
