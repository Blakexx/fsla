// measure/lab/experiments/state_move.cu -- QUESTION: a CTA's state leaves its registers for other CTAs once a
// window; what does a kilobyte of it cost a CTA by bulk copies into its cluster-mates' shared memory, and by a bulk
// copy out to global memory and one back in? And what does a token's 128-byte value cost, loaded by itself?
// METHOD: clusters of 8 CTAs, 4 warps each. A round moves 8 blocks of BLOCK bytes a CTA (1 KB and 8 KB blocks).
// `stage`: the round's words stored into the CTA's own shared memory, the base every moving form carries. `mates`:
// the stage, then a bulk copy of one block to each CTA of the cluster, whose bytes complete the receiver's barrier.
// `out`: the stage, then one bulk copy of it to global memory, waited. `out and in`: that, then one bulk copy back
// into the image. `in`: one bulk copy from global memory into the image, no stage. `rows`: 64 requests of 128 bytes
// from scattered rows of an 8 MB table, issued by one thread or by 64. Every image is checked exactly.
// see docs/internals/tools/lab.md#state-move
#include "lab.cuh"

namespace state_move {

constexpr int kCtas = 8, kThreads = 128, kRowBytes = 128, kRows = 64, kTableRows = 65536;
enum Form : int { kStage = 0, kMates, kOut, kOutIn, kIn, kRowsOne, kRowsSpread, kForms };
const char* kFormName[kForms] = {"the stage's stores alone",
                                 "a bulk copy a cluster-mate",
                                 "one bulk copy out to global memory",
                                 "one bulk copy out to global memory and one back in",
                                 "one bulk copy in from global memory",
                                 "64 loads of 128 bytes, one thread issuing",
                                 "64 loads of 128 bytes, a thread each"};

template <int BLOCK>
constexpr int kBytes = kCtas * BLOCK;
template <int BLOCK>
constexpr int kWords = kBytes<BLOCK> / 4;
template <int FORM>
constexpr bool kStaged = FORM == kStage || FORM == kMates || FORM == kOut || FORM == kOutIn;
template <int FORM>
constexpr bool kRowed = FORM == kRowsOne || FORM == kRowsSpread;

//: the word CTA `rank` stages at `i` in round `r`; the table's word `w`; the row a (block, round, request) loads
constexpr uint32_t word_of(uint32_t rank, uint32_t i, uint32_t r) { return (rank << 28) ^ (i * 2654435761u + r); }
constexpr uint32_t table_of(uint32_t w) { return w * 2246822519u + 1u; }
constexpr uint32_t row_of(uint32_t block, uint32_t r, uint32_t j) {
  return ((block * kRows + j) * 2654435761u + r * 40503u) >> 16;
}

//: THE SHARED MEMORY: the image (what arrives), the stage (what leaves), two barriers
template <int FORM, int BLOCK>
__global__ void __cluster_dims__(kCtas, 1, 1)
    move(long long* times, int* sinks, uint32_t* global, uint32_t* out, int rounds) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int kBytes = state_move::kBytes<BLOCK>, kWords = state_move::kWords<BLOCK>;
  const int tid = threadIdx.x;
  const uint32_t rank = lab::cluster_rank();
  const uint32_t base = lab::smem(smem), stage = base + kBytes, full = stage + kBytes, done = full + 8;
  uint32_t* const image = reinterpret_cast<uint32_t*>(smem);
  uint32_t* const staged = image + kWords;
  uint32_t* const mine = global + static_cast<long long>(blockIdx.x) * kWords;

#pragma unroll 1
  for (int i = tid; i < kWords; i += kThreads) image[i] = 0u;
  if (tid == 0) {
    lab::mbar_init(full, 1);
    lab::mbar_init(done, kCtas);
    lab::cluster_fence_init();
  }
  lab::bulk_fence();
  __syncthreads();
  //: a receiver states its round's bytes before any mate can send them
  if constexpr (FORM == kMates) {
    if (tid == 0) ::denseref::ops::mbar_expect_bytes(full, kBytes);
  }
  lab::cluster_sync();
  //: THE MATES' ADDRESSES, mapped once: the image and barrier of the CTA this thread sends to, every CTA's `done`
  uint32_t dones[kCtas];
#pragma unroll
  for (int c = 0; c < kCtas; ++c) dones[c] = lab::cluster_addr(done, c);
  const uint32_t to = tid % kCtas;
  const uint32_t image_to = lab::cluster_addr(base, to), full_to = lab::cluster_addr(full, to);

  uint32_t seen = 0;
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int r = 0; r < rounds; ++r) {
    if constexpr (kStaged<FORM>) {
      //: THE STAGE: the round's words, a warp's 32 lanes on 32 banks; then the edge the bulk engine reads them behind
#pragma unroll 1
      for (int i = tid; i < kWords; i += kThreads) staged[i] = word_of(rank, i, r);
      if constexpr (FORM != kStage) lab::bulk_fence();
      __syncthreads();
    }
    if constexpr (FORM == kStage) seen ^= staged[(tid + 1) % kThreads] + r;
    if constexpr (FORM == kMates) {
      //: block `to` of the stage into slot `rank` of CTA `to`'s image
      if (tid < kCtas) lab::cluster_bulk_copy(image_to + rank * BLOCK, stage + to * BLOCK, BLOCK, full_to);
      lab::cluster_wait(full, r & 1);
      seen ^= image[tid] + r;
      lab::cluster_fence();
      lab::bulk_fence();
      __syncthreads();
      //: the image is read: the next round's bytes stated, then every mate released into it
      if (tid == 0) {
        ::denseref::ops::mbar_expect_bytes(full, kBytes);
#pragma unroll
        for (int c = 0; c < kCtas; ++c) lab::cluster_arrive(dones[c]);
      }
      lab::cluster_wait(done, r & 1);
    }
    if constexpr (FORM == kOut || FORM == kOutIn) {
      if (tid == 0) {
        lab::bulk_store(mine, stage, kBytes);
        lab::bulk_commit();
        lab::bulk_wait<0>();
      }
      __syncthreads();
    }
    if constexpr (FORM == kOutIn || FORM == kIn) {
      if (tid == 0) {
        ::denseref::ops::mbar_expect_bytes(full, kBytes);
        lab::bulk_load(base, mine, kBytes, full);
      }
      lab::mbar_wait(full, r & 1);
      seen ^= image[tid] + r;
    }
    if constexpr (kRowed<FORM>) {
      //: THE ROWS: request `j` lands at row `j` of the image
      if (tid == 0) ::denseref::ops::mbar_expect_bytes(full, kRows * kRowBytes);
      if constexpr (FORM == kRowsSpread) __syncthreads();
#pragma unroll 1
      for (int j = tid; j < (FORM == kRowsOne ? (tid == 0 ? kRows : 0) : (tid < kRows ? tid + 1 : 0)); ++j) {
        const uint32_t row = row_of(blockIdx.x, r, j);
        lab::bulk_load(base + j * kRowBytes, global + static_cast<long long>(row) * (kRowBytes / 4), kRowBytes, full);
      }
      lab::mbar_wait(full, r & 1);
      seen ^= image[tid] + r;
    }
  }
  clock.stop(times);
  lab::sink(sinks, seen);
  //: the last round's image, for the host
#pragma unroll 1
  for (int i = tid; i < kWords; i += kThreads) out[static_cast<long long>(blockIdx.x) * kWords + i] = image[i];
  //: no CTA leaves while a mate may still reach into it
  lab::cluster_sync();
}

}  // namespace state_move

template <int FORM, int BLOCK>
void run(lab::Bench& b, uint32_t* global, uint32_t* out, size_t global_words) {
  using namespace state_move;
  constexpr int kBytes = state_move::kBytes<BLOCK>, kWords = state_move::kWords<BLOCK>;
  constexpr int kCheckRounds = 4, kRounds = BLOCK >= 8192 ? 256 : 1024;
  const int blocks = b.sms / kCtas * kCtas;
  const std::string name = std::string(kFormName[FORM])
                           + (kRowed<FORM> ? "" : ", blocks of " + std::to_string(BLOCK) + " bytes");
  constexpr int bytes = 2 * kBytes + 16;
  const auto kernel = move<FORM, BLOCK>;
  LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  std::vector<uint32_t> table(global_words);
  for (size_t w = 0; w < global_words; ++w) table[w] = table_of(static_cast<uint32_t>(w));
  const auto fill = [&] {
    LAB_CHECK(cudaMemcpy(global, table.data(), global_words * sizeof(uint32_t), cudaMemcpyHostToDevice));
  };

  if (FORM == kStage) {
    b.unchecked(name, "the base: it writes shared memory alone");
  } else {
    fill();
    kernel<<<blocks, kThreads, bytes>>>(b.times, b.sinks, global, out, kCheckRounds);
    LAB_CHECK(cudaDeviceSynchronize());
    const size_t words = static_cast<size_t>(blocks) * kWords;
    std::vector<uint32_t> got(words);
    LAB_CHECK(cudaMemcpy(got.data(), FORM == kOut ? global : out, words * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    const uint32_t last = kCheckRounds - 1;
    long long wrong = 0;
    for (size_t at = 0; at < words; ++at) {
      const uint32_t block = static_cast<uint32_t>(at / kWords), i = static_cast<uint32_t>(at % kWords);
      const uint32_t rank = block % kCtas;
      uint32_t want = 0;
      if (FORM == kMates) {
        //: slot `i / block words` came from that rank's stage, its block `rank`
        const uint32_t per = BLOCK / 4;
        want = word_of(i / per, rank * per + i % per, last);
      } else if (FORM == kOut || FORM == kOutIn) {
        want = word_of(rank, i, last);
      } else if (FORM == kIn) {
        want = table[at];
      } else {
        const uint32_t per = kRowBytes / 4, row = row_of(block, last, i / per);
        want = i / per < static_cast<uint32_t>(kRows) ? table[static_cast<size_t>(row) * per + i % per] : 0u;
      }
      wrong += got[at] != want;
    }
    b.check(name, wrong == 0,
            std::to_string(wrong) + " of " + std::to_string(words) + " words differ from the exact ones after "
                + std::to_string(kCheckRounds) + " rounds");
  }
  fill();
  const lab::Samples cycles = b.variant(
      [&] { kernel<<<blocks, kThreads, bytes>>>(b.times, b.sinks, global, out, kRounds); }, [&] {
        return b.span(kThreads >> 5, blocks) / kRounds / (kRowed<FORM> ? double(kRows) : kBytes / 1024.0);
      });
  b.record(name, kRowed<FORM> ? "cycles a 128-byte load a CTA" : "cycles a kilobyte a CTA", cycles);
}

template <int BLOCK>
void expect(lab::Bench& b) {
  using namespace state_move;
  for (int f = 0; f < kForms; ++f) {
    if (kRowsOne <= f && BLOCK != 1024) continue;
    const std::string k = lab::kernel("move", {f, BLOCK});
    b.expect(k, "UBLKCP.S.S", "==", f == kMates ? 1 : 0);
    b.expect(k, "UBLKCP.G.S", "==", f == kOut || f == kOutIn ? 1 : 0);
    b.expect(k, "UBLKCP.S.G", "==", f == kOutIn || f == kIn || kRowsOne <= f ? 1 : 0);
  }
}

template <int BLOCK>
void blocks(lab::Bench& b, uint32_t* global, uint32_t* out, size_t global_words) {
  using namespace state_move;
  run<kStage, BLOCK>(b, global, out, global_words);
  run<kMates, BLOCK>(b, global, out, global_words);
  run<kOut, BLOCK>(b, global, out, global_words);
  run<kOutIn, BLOCK>(b, global, out, global_words);
  run<kIn, BLOCK>(b, global, out, global_words);
}

int main(int argc, char** argv) {
  using namespace state_move;
  lab::Bench b(argc, argv);
  expect<1024>(b);
  expect<8192>(b);
  if (b.expect_only) return 0;
  //: the global side: every CTA's largest image, and never less than the rows' table
  const size_t images = static_cast<size_t>(b.sms) * kWords<8192>;
  const size_t global_words = std::max(images, static_cast<size_t>(kTableRows) * (kRowBytes / 4));
  uint32_t *global = nullptr, *out = nullptr;
  LAB_CHECK(cudaMalloc(&global, global_words * sizeof(uint32_t)));
  LAB_CHECK(cudaMalloc(&out, images * sizeof(uint32_t)));
  blocks<1024>(b, global, out, global_words);
  blocks<8192>(b, global, out, global_words);
  run<kRowsOne, 1024>(b, global, out, global_words);
  run<kRowsSpread, 1024>(b, global, out, global_words);
  return 0;
}
