// measure/lab/experiments/intra_stream.cu -- QUESTION: the fp16 intra moves ~5 B a cycle an SM (its whole sweep with
// the Gram skipped ~6.8) where cp.async streams 18 (smem_rates): what does its stage ring deliver an SM beside a
// Gram, and which form of issuing the refills keeps the copies under the MMAs? METHOD: one CTA an SM (99 KB of dynamic
// shared memory), a sweep of 600 stages over L2-resident rows (4 MB: 4,096 tokens of 1 KB, the dense plane's), a
// stage the tile's 64 rows' 128-byte slice (the intra's 8 KB stage). Each compute warp waits for a
// stage, runs `SLABS` slabs of the dense Gram on it -- a slab two k-steps, an `ldmatrix.x4` of a static 16-reader A
// and two of the stage's B a k-step, 8 fp16-accumulate MMAs a slab (the 8-warp walk's 4 n-tiles a warp) -- and
// releases it. ISSUE 0: the intra's -- a release count, the last warp to release a stage refills its buffer with the
// stage three on, all 64 rows by its lanes; ISSUE 2: every warp, at a stage's start, awaits the previous stage's
// `empty` mbarrier (every warp's release) and copies its share of the stage two on into that buffer; ISSUE 4: a ninth
// warp copies every stage, awaiting each buffer's `empty`. A sample is the CTA's span over the bytes it moved and over
// the MMAs it issued, the mean over SMs; ISSUE 9 the Gram alone on resident stages, STRIDE 128 a stage's rows one
// contiguous run, and `kB` the ring's depth (3, 6, 10 buffers). see docs/internals/tools/lab.md#intra-stream
#include "lab.cuh"

namespace intra_stream {

constexpr int kRows = 64, kRowBytes = 1024, kTokens = 4096, kStages = 600, kSmem = 99 * 1024;
constexpr int kStage = kRows * 128, kMaxB = 10, kReaderAt = kMaxB * kStage, kBarAt = kReaderAt + 16 * 1024;

template <int WARPS, int ISSUE, int SLABS, int STRIDE, int kB>
__global__ void __launch_bounds__((WARPS + (ISSUE == 4)) * 32, 1)
    ring(const uint8_t* table, long long* t, int* out) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint64_t* const full = reinterpret_cast<uint64_t*>(smem + kBarAt);
  uint64_t* const empty = full + kB;
  int* const released = reinterpret_cast<int*>(empty + kB);
  const int lane = lab::lane_id(), warp = lab::warp_id(), tid = static_cast<int>(threadIdx.x);
  constexpr int kCopiers = ISSUE == 2 ? WARPS * 32 : 32;
  if (tid == 0) {
    for (int i = 0; i < kB; ++i) {
      lab::mbar_init(lab::smem(&full[i]), kCopiers);
      lab::mbar_init(lab::smem(&empty[i]), WARPS);
      released[i] = 0;
    }
  }
  for (int i = tid; i < 16 * 1024 / 4; i += blockDim.x) reinterpret_cast<uint32_t*>(smem + kReaderAt)[i] = i;
  __syncthreads();
  const int base_token = (blockIdx.x * 512) % kTokens;
  //: stage `i`'s 512 16-byte chunks by `n` threads from thread `k`, into its buffer, arriving when they land
  const auto copy = [&](int i, int k, int n) {
    const int tile = i / 8, slice = i % 8, slot = i % kB;
    //: a stage's rows `STRIDE` bytes apart: 1 KB the dense plane's token stride, 128 a stage's own contiguous run
    const size_t at = STRIDE == kRowBytes
                          ? static_cast<size_t>((base_token + tile * kRows) % kTokens) * kRowBytes + slice * 128
                          : (static_cast<size_t>(blockIdx.x) * 64 + i) * kStage % (static_cast<size_t>(kTokens) * kRowBytes);
    const uint8_t* const src = table + at;
    const uint32_t dst = lab::smem(smem + slot * kStage);
#pragma unroll 4
    for (int c = k; c < kRows * 8; c += n) {
      const int row = c >> 3, ch = c & 7;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst + row * 128 + ((ch ^ (row & 7)) << 4)),
                   "l"(src + row * STRIDE + ch * 16)
                   : "memory");
    }
    lab::mbar_arrive_when_landed(lab::smem(&full[slot]));
  };
  lab::WarpSpan clock;
  clock.start();
  uint32_t fold = 0;
  if (ISSUE == 4 && warp == WARPS) {
#pragma unroll 1
    for (int i = 0; i < kStages; ++i) {
      if (i >= kB) lab::mbar_wait(lab::smem(&empty[i % kB]), (i / kB - 1) & 1);
      copy(i, lane, 32);
    }
  } else {
    if (ISSUE == 0 && warp == 0)
      for (int i = 0; i < kB; ++i) copy(i, lane, 32);
    if (ISSUE == 2)
      for (int i = 0; i < kB - 1; ++i) copy(i, tid, WARPS * 32);
    uint32_t acc[4][2] = {};
    const uint32_t a_row = lab::smem(smem + kReaderAt) + (warp & 3) * 4096 + (lane & 15) * 256;
    const int brow = (lane & 7) + ((lane >> 4) << 3) + 32 * (warp & 1);
#pragma unroll 1
    for (int i = 0; i < kStages; ++i) {
      const int slot = i % kB, use = i / kB;
      if (ISSUE == 2 && i >= 1 && i - 1 + kB < kStages) {
        lab::mbar_wait(lab::smem(&empty[(i - 1) % kB]), ((i - 1) / kB) & 1);
        copy(i - 1 + kB, tid, WARPS * 32);
      }
      if (ISSUE == 2 && i == 0) copy(kB - 1, tid, WARPS * 32);
      if (ISSUE != 9) lab::mbar_wait(lab::smem(&full[slot]), use & 1);
      const uint32_t buf = lab::smem(smem + slot * kStage);
#pragma unroll
      for (int s = 0; s < SLABS; ++s)
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          uint32_t a[4], b0[4], b1[4];
          lab::ldsm(a, a_row + (((4 * s + 2 * k + (lane >> 4)) ^ (lane & 7)) << 4));
          const int c = (4 * s + 2 * k + ((lane >> 3) & 1)) & 7;
          lab::ldsm(b0, buf + brow * 128 + ((c ^ (brow & 7)) << 4));
          lab::ldsm(b1, buf + (brow + 16) * 128 + ((c ^ (brow & 7)) << 4));
          lab::mma_f16_f16(acc[0], a, b0[0], b0[1]);
          lab::mma_f16_f16(acc[1], a, b0[2], b0[3]);
          lab::mma_f16_f16(acc[2], a, b1[0], b1[1]);
          lab::mma_f16_f16(acc[3], a, b1[2], b1[3]);
        }
      __syncwarp();
      if (ISSUE == 9) {
      } else if (ISSUE == 0) {
        int last = 0;
        if (lane == 0) {
          __threadfence_block();
          last = atomicAdd(&released[slot], 1) == (use + 1) * WARPS - 1;
        }
        if (__shfl_sync(0xffffffffu, last, 0) && i + kB < kStages) {
          __threadfence_block();
          copy(i + kB, lane, 32);
        }
      } else if (lane == 0) {
        lab::mbar_arrive(lab::smem(&empty[slot]));
      }
    }
    for (int n = 0; n < 4; ++n) fold ^= acc[n][0] ^ acc[n][1];
  }
  clock.stop(t);
  lab::sink(out, fold);
}

template <int WARPS, int ISSUE, int SLABS, int STRIDE = kRowBytes, int kB = 3>
void variant(const lab::Bench& b, const uint8_t* table) {
  constexpr int kWarpsAll = WARPS + (ISSUE == 4);
  const std::string name = lab::kernel("ring", {WARPS, ISSUE, SLABS, STRIDE, kB});
  if (ISSUE != 9) b.expect(name, "LDGSTS", ">=", 1);
  if (SLABS > 0) {
    b.expect(name, "HMMA", ">=", 4);
    b.exempt(name, "a rate probe: its MMAs fold into a sink");
  }
  if (b.expect_only) return;
  LAB_CHECK(cudaFuncSetAttribute(ring<WARPS, ISSUE, SLABS, STRIDE, kB>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  lab::Samples mmas;
  const lab::Samples rate = b.variant(
      [&] { ring<WARPS, ISSUE, SLABS, STRIDE, kB><<<b.sms, kWarpsAll * 32, kSmem>>>(table, b.times, b.sinks); },
      [&] {
        std::vector<long long> h(static_cast<size_t>(b.sms) * kWarpsAll * 2);
        LAB_CHECK(cudaMemcpy(h.data(), b.times, h.size() * sizeof(long long), cudaMemcpyDeviceToHost));
        double r = 0, m = 0;
        for (int sm = 0; sm < b.sms; ++sm) {
          long long lo = h[2 * sm * kWarpsAll], hi = h[2 * sm * kWarpsAll + 1];
          for (int w = 1; w < kWarpsAll; ++w) {
            lo = std::min(lo, h[2 * (sm * kWarpsAll + w)]);
            hi = std::max(hi, h[2 * (sm * kWarpsAll + w) + 1]);
          }
          r += double(kStages) * kStage / double(hi - lo);
          m += SLABS > 0 ? double(hi - lo) / (double(kStages) * SLABS * 8 * WARPS / 4) : 0.0;
        }
        mmas.v.push_back(m / b.sms);
        return r / b.sms;
      });
  const char* form = ISSUE == 0   ? "last releaser refills"
                     : ISSUE == 2 ? "every warp a share two on"
                     : ISSUE == 4 ? "a producer warp"
                                  : "no copies (the Gram alone on resident stages)";
  const std::string row = std::string(form) + ", " + std::to_string(WARPS) + " compute warps, " + std::to_string(SLABS) +
                          " Gram slabs a stage, rows " + std::to_string(STRIDE) + " B apart, " +
                          std::to_string(kB) + " buffers";
  b.unchecked(row, "a rate probe: its copies and MMAs fold into a sink");
  b.record(row, "bytes a cycle an SM", rate);
  b.record(row + ": cycles an MMA a scheduler", "cycles", mmas);
}

}  // namespace intra_stream

int main(int argc, char** argv) {
  using namespace intra_stream;
  lab::Bench b(argc, argv);
  uint8_t* table = nullptr;
  if (!b.expect_only) {
    LAB_CHECK(cudaMalloc(&table, static_cast<size_t>(kTokens) * kRowBytes));
    LAB_CHECK(cudaMemset(table, 0x5a, static_cast<size_t>(kTokens) * kRowBytes));
  }
  //: the Gram alone on resident stages, then each issue form without and beside the dense (2 slabs) and u8 (4) Gram
  variant<8, 9, 2>(b, table);
  variant<8, 9, 4>(b, table);
  variant<8, 0, 0>(b, table);
  variant<8, 2, 0>(b, table);
  variant<8, 4, 0>(b, table);
  variant<8, 0, 2>(b, table);
  variant<8, 2, 2>(b, table);
  variant<8, 4, 2>(b, table);
  variant<8, 0, 4>(b, table);
  variant<8, 2, 4>(b, table);
  variant<8, 4, 4>(b, table);
  //: a stage's rows one contiguous run instead of the dense plane's 1 KB token stride
  variant<8, 4, 0, 128>(b, table);
  variant<8, 4, 2, 128>(b, table);
  //: a deeper ring: the bytes in flight
  variant<8, 4, 0, kRowBytes, 6>(b, table);
  variant<8, 4, 2, kRowBytes, 6>(b, table);
  variant<8, 4, 0, kRowBytes, 10>(b, table);
  variant<8, 4, 2, kRowBytes, 10>(b, table);
  variant<8, 4, 4, kRowBytes, 10>(b, table);
  variant<8, 0, 2, kRowBytes, 10>(b, table);
}
