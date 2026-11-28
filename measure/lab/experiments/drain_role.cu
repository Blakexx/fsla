// measure/lab/experiments/drain_role.cu -- QUESTION: a CTA's two warpgroups each hold a token's partial; what does
// a token cost when a DRAIN ROLE of warps pulls both from shared memory, adds them, and issues ONE add of f16
// pairs to global memory -- beside an f16 line added straight from registers -- and do value loads on the same
// path add to it? METHOD: a CTA an SM. `direct`: 4 warps add a token's 64 channels as 32 f16 pairs, one line, from
// registers. `role`: two source warpgroups store a chunk of 64 tokens x 64 channels fp32 each into their panels;
// 1, 2 or 4 drain warps, a chunk behind, load a token's two rows, add in fp32, round once to f16 pairs and add the
// line into the token's row -- the CTA's own rows, or rows drawn from 1,024 every CTA shares. `loads`: one thread
// bulk-loads 32 KB a chunk from global memory, a chunk ahead, nothing else running. `role and loads`: both, with
// two drain warps and with four.
// Every drained sum is checked exactly.
// TIMELINE: a source's trip is a chunk: the wait on the drain's barrier two chunks back, its stores, its arrival.
// A drain warp's trip is two batches of 8 of its tokens, each at its own registers: a batch's adds, rounds and
// global adds, then those registers' loads for the batch two on -- but a chunk's first two batches are loaded as
// it opens, its panels not whole before, so with four drain warps (two batches a chunk) no load runs ahead; a
// chunk on, its arrival frees the panels. The loader's trip is a chunk: the wait on the load two chunks back,
// the next load issued. Live in a drain warp: 4 fp32 a token of two batches, 64 registers.
// see docs/internals/tools/lab.md#drain-role
#include "lab.cuh"

namespace drain_role {

constexpr int kTokens = 64, kChannels = 64, kPanelBytes = kTokens * kChannels * 4, kRows = 1024, kBatch = 8;
constexpr int kSourceThreads = 256, kLoadBytes = 32768;
//: the shared memory: two sources' panels of two chunks each, the loads' two slots, six barriers
constexpr int kLoadAt = 4 * kPanelBytes, kBarsAt = kLoadAt + 2 * kLoadBytes, kSmemBytes = kBarsAt + 48;
enum Bits : int { kRole = 1, kLoads = 2, kSpread = 4 };

//: a source's value at a channel, a small integer: every f16 sum exact
constexpr int value_of(int source, int channel) { return channel % 5 + 1 + 2 * source; }
constexpr uint32_t row_of(uint32_t block, uint32_t chunk, uint32_t token) {
  return ((block * kTokens + token) * 2654435761u + chunk * 40503u) >> 22;
}

__device__ __forceinline__ uint32_t f16_of(int v) { return __half_as_ushort(__float2half(static_cast<float>(v))); }

//: THE DIRECT LINE: a token's 32 pairs from registers, a warp's lanes one 128-byte line
__global__ void __launch_bounds__(128, 1) direct(long long* times, uint32_t* out, int rounds) {
  constexpr int kMine = kTokens * (kChannels / 2) / 128;
  const int tid = threadIdx.x;
  uint32_t v[kMine];
#pragma unroll
  for (int i = 0; i < kMine; ++i) {
    const int c = 2 * ((i * 128 + tid) % (kChannels / 2));
    v[i] = f16_of(value_of(0, c)) | f16_of(value_of(0, c + 1)) << 16;
  }
  uint32_t* const own = out + static_cast<long long>(blockIdx.x) * kTokens * (kChannels / 2);
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int r = 0; r < rounds; ++r) {
#pragma unroll
    for (int i = 0; i < kMine; ++i) lab::red_add_f16x2(own + i * 128 + tid, v[i]);
  }
  clock.stop(times);
}

template <int DRAINS, int FORM>
__global__ void __launch_bounds__(kSourceThreads + 32 * DRAINS, 1)
    role(long long* times, int* sinks, const uint32_t* table, uint32_t* out, int chunks) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t base = lab::smem(smem);
  //: THE BARRIERS, a pair each by the chunk's parity: the sources' arrivals, the drain warps', the loads' bytes
  const uint32_t ready = base + kBarsAt, freed = ready + 16, landed = freed + 16;
  if (tid == 0) {
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      lab::mbar_init(ready + 8 * p, kSourceThreads / 32);
      lab::mbar_init(freed + 8 * p, DRAINS);
      lab::mbar_init(landed + 8 * p, 1);
    }
  }
  lab::bulk_fence();
  __syncthreads();

  uint32_t seen = 0;
  lab::WarpSpan clock;
  clock.start();
  if (warp < kSourceThreads / 32) {
    //: A SOURCE: its warpgroup's partial of the chunk, 16 fp32 a thread, into its panel of the chunk's parity
    const int source = warp >> 2, mine = tid & 127;
    float v[kTokens * kChannels / 128 / 2];
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = static_cast<float>(value_of(source, (i * 128 + mine) % kChannels));
#pragma unroll 1
    for (int c = 0; c < chunks; ++c) {
      if constexpr (FORM & kRole) {
        if (c >= 2) lab::mbar_wait(freed + 8 * (c & 1), ((c >> 1) - 1) & 1);
        float* const panel = reinterpret_cast<float*>(smem + (2 * source + (c & 1)) * kPanelBytes);
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int i = 0; i < 16; ++i) panel[(16 * h + i) * 128 + mine] = v[i];
        __syncwarp();
        if (lane == 0) lab::mbar_arrive(ready + 8 * (c & 1));
      }
      //: THE LOADER, one thread: a chunk's bytes into the slot of its parity, two in flight
      if constexpr (FORM & kLoads) {
        if (tid == 0) {
          if (c >= 2) lab::mbar_wait(landed + 8 * (c & 1), ((c >> 1) - 1) & 1);
          ::denseref::ops::mbar_expect_bytes(landed + 8 * (c & 1), kLoadBytes);
          lab::bulk_load(base + kLoadAt + (c & 1) * kLoadBytes, table + (c % 128) * (kLoadBytes / 4), kLoadBytes,
                         landed + 8 * (c & 1));
        }
      }
    }
    if constexpr (FORM & kLoads) {
      if (tid == 0) {
#pragma unroll
        for (int p = 0; p < 2; ++p)
          if (chunks > p) lab::mbar_wait(landed + 8 * ((chunks - 1 - p) & 1), ((chunks - 1 - p) >> 1) & 1);
        seen = reinterpret_cast<const uint32_t*>(smem + kLoadAt)[0];
      }
    }
  } else if constexpr (FORM & kRole) {
    //: A DRAIN WARP: the tokens of its residue; a lane holds a token's two channels of each source
    const int mine = warp - kSourceThreads / 32;
    constexpr int kBatches = kTokens / DRAINS / kBatch;
    static_assert(kBatches % 2 == 0);
    const float* const panels = reinterpret_cast<const float*>(smem);
    float a0[2][kBatch], a1[2][kBatch], b0[2][kBatch], b1[2][kBatch];
    //: a batch's rows, from the panels of the chunk it lies in (a batch past the last chunk reads the first)
    const auto load = [&](int h, int batch) {
      const int c = (batch / kBatches) % 2;
#pragma unroll
      for (int i = 0; i < kBatch; ++i) {
        const int t = mine + DRAINS * ((batch % kBatches) * kBatch + i);
        const float* const row = panels + c * (kPanelBytes / 4) + t * kChannels + 2 * lane;
        a0[h][i] = row[0];
        a1[h][i] = row[1];
        b0[h][i] = row[2 * (kPanelBytes / 4)];
        b1[h][i] = row[2 * (kPanelBytes / 4) + 1];
      }
    };
#pragma unroll 1
    for (int c = 0; c < chunks; ++c) {
      lab::mbar_wait(ready + 8 * (c & 1), (c >> 1) & 1);
      //: the chunk's first two batches are loaded as it opens: its panels were not whole a trip before
      load(0, c * kBatches);
      load(1, c * kBatches + 1);
#pragma unroll 1
      for (int b = 0; b < kBatches; b += 2) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
          for (int i = 0; i < kBatch; ++i) {
            const int t = mine + DRAINS * ((b + h) * kBatch + i);
            const uint32_t row = FORM & kSpread ? row_of(blockIdx.x, c, t) : blockIdx.x * kTokens + t;
            lab::red_add_f16x2(out + static_cast<long long>(row) * (kChannels / 2) + lane,
                               lab::pack_half2(a0[h][i] + b0[h][i], a1[h][i] + b1[h][i]));
          }
          //: no branch: past the chunk's last batch it loads that batch again, and nothing uses it
          load(h, c * kBatches + min(b + 2 + h, kBatches - 1));
        }
      }
      __syncwarp();
      if (lane == 0) lab::mbar_arrive(freed + 8 * (c & 1));
    }
  }
  clock.stop(times);
  lab::sink(sinks, seen);
}

}  // namespace drain_role

inline float half_to_float(uint16_t h) {
  const int e = (h >> 10) & 31, m = h & 1023;
  const float mag = e == 0 ? std::ldexp(static_cast<float>(m), -24) : std::ldexp(static_cast<float>(m | 1024), e - 25);
  return h & 0x8000 ? -mag : mag;
}

//: the drained rows against the exact sums: `adds[row]` lines of `per` a channel
long long differing(const std::vector<uint32_t>& got, const std::vector<long long>& adds, int sources) {
  using namespace drain_role;
  long long wrong = 0;
  for (size_t w = 0; w < got.size(); ++w)
    for (int half = 0; half < 2; ++half) {
      const int channel = 2 * static_cast<int>(w % (kChannels / 2)) + half;
      const int per = value_of(0, channel) + (sources == 2 ? value_of(1, channel) : 0);
      wrong += half_to_float(static_cast<uint16_t>(got[w] >> (16 * half)))
               != static_cast<float>(adds[w / (kChannels / 2)] * per);
    }
  return wrong;
}

void run_direct(lab::Bench& b, uint32_t* out) {
  using namespace drain_role;
  constexpr int kCheckRounds = 16, kRounds = 2048;
  const std::string name = "an f16 line from registers, 4 warps";
  const size_t words = static_cast<size_t>(b.sms) * kTokens * (kChannels / 2);
  LAB_CHECK(cudaMemset(out, 0, words * 4));
  direct<<<b.sms, 128>>>(b.times, out, kCheckRounds);
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<uint32_t> got(words);
  LAB_CHECK(cudaMemcpy(got.data(), out, words * 4, cudaMemcpyDeviceToHost));
  const long long wrong = differing(got, std::vector<long long>(words / (kChannels / 2), kCheckRounds), 1);
  b.check(name, wrong == 0,
          std::to_string(wrong) + " of " + std::to_string(2 * words) + " sums differ from the exact ones");
  LAB_CHECK(cudaMemset(out, 0, words * 4));
  const lab::Samples cycles = b.variant([&] { direct<<<b.sms, 128>>>(b.times, out, kRounds); },
                                        [&] { return b.span(4) / kRounds / kTokens; });
  b.record(name, "cycles a token a CTA", cycles);
}

template <int DRAINS, int FORM>
void run(lab::Bench& b, const uint32_t* table, uint32_t* out) {
  using namespace drain_role;
  constexpr int kCheckChunks = 8, kChunks = 1024, kThreads = kSourceThreads + 32 * DRAINS;
  const std::string name = FORM == kLoads ? "the loads alone, 32 KB a chunk"
                           : std::string("the drain role, ") + std::to_string(DRAINS)
                                 + (DRAINS == 1 ? " drain warp" : " drain warps")
                                 + (FORM & kSpread ? ", rows spread over 1,024" : ", the CTA's own rows")
                                 + (FORM & kLoads ? ", beside the loads" : "");
  const auto kernel = role<DRAINS, FORM>;
  LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  const size_t rows = FORM & kSpread ? static_cast<size_t>(kRows) : static_cast<size_t>(b.sms) * kTokens;
  const size_t words = rows * (kChannels / 2);
  if (FORM & kRole) {
    LAB_CHECK(cudaMemset(out, 0, words * 4));
    kernel<<<b.sms, kThreads, kSmemBytes>>>(b.times, b.sinks, table, out, kCheckChunks);
    LAB_CHECK(cudaDeviceSynchronize());
    std::vector<uint32_t> got(words);
    LAB_CHECK(cudaMemcpy(got.data(), out, words * 4, cudaMemcpyDeviceToHost));
    std::vector<long long> adds(rows, FORM & kSpread ? 0 : kCheckChunks);
    if (FORM & kSpread)
      for (int blk = 0; blk < b.sms; ++blk)
        for (int c = 0; c < kCheckChunks; ++c)
          for (int t = 0; t < kTokens; ++t) ++adds[row_of(blk, c, t)];
    const long long wrong = differing(got, adds, 2);
    b.check(name, wrong == 0,
            std::to_string(wrong) + " of " + std::to_string(2 * words) + " sums differ from the exact ones");
    LAB_CHECK(cudaMemset(out, 0, words * 4));
  } else {
    b.unchecked(name, "an isolation: the loads land and nothing reads them but one word");
  }
  const lab::Samples cycles = b.variant(
      [&] { kernel<<<b.sms, kThreads, kSmemBytes>>>(b.times, b.sinks, table, out, kChunks); },
      [&] { return b.span(kThreads >> 5) / kChunks / kTokens; });
  b.record(name, "cycles a token a CTA", cycles);
}

int main(int argc, char** argv) {
  using namespace drain_role;
  lab::Bench b(argc, argv);
  b.expect("direct", "RED", ">=", 16);
  for (int drains : {1, 2, 4})
    for (int form : {int(kRole), kRole | kSpread, kRole | kLoads}) {
      if (drains == 1 && form != kRole) continue;
      if (drains == 4 && form == (kRole | kSpread)) continue;
      b.expect(lab::kernel("role", {drains, form}), "RED", ">=", 2 * kBatch);
      //: once a chunk, not a token: a role's `__syncwarp` before its one-lane arrival is a convergence call, and
      //: the loader is one thread of its warp
      b.expect(lab::kernel("role", {drains, form}), "LOOP_CONVERGENCE", "<=", form & kLoads ? 19 : 7);
    }
  b.expect(lab::kernel("role", {2, kLoads}), "UBLKCP.S.G", "==", 1);
  b.expect(lab::kernel("role", {2, kLoads}), "LOOP_CONVERGENCE", "<=", 4);
  if (b.expect_only) return 0;
  std::vector<uint32_t> bytes(128 * (kLoadBytes / 4));
  for (size_t w = 0; w < bytes.size(); ++w) bytes[w] = static_cast<uint32_t>(w) * 2246822519u + 1u;
  uint32_t *table = nullptr, *out = nullptr;
  LAB_CHECK(cudaMalloc(&table, bytes.size() * 4));
  LAB_CHECK(cudaMemcpy(table, bytes.data(), bytes.size() * 4, cudaMemcpyHostToDevice));
  const size_t rows = std::max(static_cast<size_t>(b.sms) * kTokens, static_cast<size_t>(kRows));
  LAB_CHECK(cudaMalloc(&out, rows * (kChannels / 2) * 4));
  run_direct(b, out);
  run<1, kRole>(b, table, out);
  run<2, kRole>(b, table, out);
  run<4, kRole>(b, table, out);
  run<2, kRole | kSpread>(b, table, out);
  run<2, kLoads>(b, table, out);
  run<2, kRole | kLoads>(b, table, out);
  run<4, kRole | kLoads>(b, table, out);
  return 0;
}
