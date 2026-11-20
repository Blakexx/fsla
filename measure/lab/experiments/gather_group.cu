// measure/lab/experiments/gather_group.cu -- QUESTION: the gather of a sparse block's window in the grouped digit
// stream's plainest form -- every list a run of its own, values only -- what does a block's window cost a CTA, and a
// token a side? METHOD: a CTA an SM of 16 warps, a warp a digit of the block. A trip is one (block, window): THE
// LISTS -- a lane loads 32 tokens' bits of its warp's digit (the mask digit-major), counts them, the warp sums its
// lanes' counts before each and its digit's whole; behind the CTA's barrier a warp knows the lists before its own,
// and a lane writes each of its live tokens' indices at its place in the packet. THE VALUES -- behind the barrier
// thread `e` takes entry `e`: its token's eight value lines from the token-major table to the packet's eight planes
// (line `c` of entry `e` at `c (R + 1) + e`: the operand a fold step reads in place), and its sparse byte on its list's
// digit; an entry past the lists' end is the zero token's. THE STORE -- the packet (a head: R, the entries, a list's
// count; the entries' tokens; their sparse bytes; the planes) to the CTA's ring by one bulk copy, a packet's copy
// waited two trips on, three packets of shared memory. Every packet is checked byte for byte against the host's.
// TIMELINE: a trip is its packet's 8 loads and a byte issued; under them the NEXT packet's plan -- the mask's word
// (loaded a trip ahead), 5 shuffles, the barrier, a lane's live tokens' stores (a loop a lane, its trips its tokens),
// the barrier -- then this packet's 9 stores, the fence, the barrier, thread 0's copy, commit and wait on the copy
// before. Live: 8 lines a thread through the plan.
// see docs/internals/tools/lab.md#gather-group
#include <random>

#include "lab.cuh"

namespace gather_group {

constexpr int kWarps = 16, kThreads = 32 * kWarps, kDigits = 16, kWindow = 1024, kPackets = 3, kRing = 8;
constexpr int kTableTokens = 65536, kZeroToken = kTableTokens;
//: a form's bits: its phases clocked; its values moved by asynchronous copies, a row's 8 pieces 8 lanes of one instruction
//: or by loads and stores so laid: instruction `i` of a warp four rows, lane `8 r + c` piece `c` of row `r`
enum Form { kPlain = 0, kPhased = 1, kLaned = 4 };
enum Probe { kNoStore = 2, kNoTokens = 16 };
enum Phase { kLists, kEntries, kLoads, kStores, kOut, kPhases };
//: A PACKET: a head (R, the entries, a list's count: 48 bytes), R tokens (16 bits, in the window), R sparse bytes, 8
//: planes of R lines; R the entries and a step's slack, a multiple of 32
constexpr int kHead = 48, kMaxRun = 512;
//: A PLANE'S PITCH is its run and a line: 8 lanes storing one row's 8 pieces, a plane each, then fall on 8 bank groups
//: (a pitch a multiple of 8 lines puts them on one)
constexpr int pitch_of(int run) { return run + 1; }
constexpr int packet_bytes(int run) { return kHead + 3 * run + 128 * pitch_of(run); }
constexpr int kPacketRoom = (packet_bytes(kMaxRun) + 63) / 64 * 64, kCountsAt = kPackets * kPacketRoom,
              kDigitAt = kCountsAt + 64, kShared = kDigitAt + kMaxRun;
constexpr int run_of(int entries) { return (entries + 16 + 63) / 64 * 64; }

//: the window a (CTA, trip) takes of the table, and a token's value word, sparse byte on a digit
constexpr uint32_t window_of(uint32_t block, uint32_t trip) { return (block * 7u + trip) % (kTableTokens / kWindow); }
constexpr uint32_t value_of(uint32_t token, uint32_t word) { return (token * 32u + word) * 2246822519u + 1u; }
constexpr uint32_t sparse_of(uint32_t token, uint32_t digit) { return 1u + (((token * 16u + digit) * 2654435761u) >> 9) % 255u; }

template <int FORM>
__global__ void __launch_bounds__(kThreads, 1)
    gather(long long* times, const uint32_t* masks, const uint4* values, const uint8_t* sparse, uint8_t* out, long long* phases,
           int trips, int ring, int probe) {
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t base = lab::smem(smem);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  uint32_t* const counts = reinterpret_cast<uint32_t*>(smem + kCountsAt);
  //: a (CTA, trip)'s mask: a digit's 32 words one after another, one of `kRing` a CTA
  const auto word_of = [&](int trip) {
    return masks[(static_cast<size_t>(blockIdx.x) * kRing + trip % kRing) * (kDigits * 32) + warp * 32 + lane];
  };
  uint32_t ahead = word_of(0);
  //: THE PHASED FORM: a thread's cycles in each of a trip's five phases, read behind a barrier's own completion
  long long in_phase[kPhases] = {0, 0, 0, 0, 0}, lapped = 0;
  const auto lap = [&](int phase) {
    if constexpr ((FORM & kPhased) != 0) {
      asm volatile("membar.cta;" ::: "memory");
      const long long now = clock64();
      in_phase[phase] += now - lapped;
      lapped = now;
    }
  };
  //: A PACKET'S PLAN, a thread's: its run and entries, and entry `tid`'s token of the table and list's digit
  int run = 0;
  uint32_t entries = 0, token = 0, digit = 0;
  //: THE LISTS AND THE ENTRIES of packet `trip`, into its room: two barriers
  const auto plan = [&](int trip) {
    uint8_t* const packet = smem + (trip % kPackets) * kPacketRoom;
    const uint32_t word = ahead;
    ahead = word_of(trip + 1);
    const uint32_t own = static_cast<uint32_t>(__popc(word));
    uint32_t upto = own;
#pragma unroll
    for (int d = 1; d < 32; d *= 2) {
      const uint32_t left = __shfl_up_sync(0xffffffffu, upto, d);
      if (lane >= d) upto += left;
    }
    if (lane == 31) counts[warp] = upto;
    __syncthreads();
    lap(kLists);
    //: THE LISTS' PLACES: every warp sums the 16 counts across its lanes (a lane a digit, each half of the warp the
    //: same), and takes the lists before its own digit's and the whole
    const uint32_t count = counts[lane & 15];
    uint32_t through = count;
#pragma unroll
    for (int d = 1; d < 16; d *= 2) {
      const uint32_t before = __shfl_up_sync(0xffffffffu, through, d);
      if ((lane & 15) >= d) through += before;
    }
    const uint32_t first = __shfl_sync(0xffffffffu, through - count, warp);
    entries = __shfl_sync(0xffffffffu, through, 15);
    run = run_of(static_cast<int>(entries));
    uint8_t* const digits = smem + kDigitAt;
    uint16_t* const tokens = reinterpret_cast<uint16_t*>(packet + kHead);
    uint32_t left = word;
    uint32_t at = first + upto - own;
    if ((probe & kNoTokens) != 0) left = 0;
#pragma unroll 1
    while (left != 0) {
      digits[at] = static_cast<uint8_t>(warp);
      tokens[at++] = static_cast<uint16_t>(32 * lane + __ffs(left) - 1);
      left &= left - 1;
    }
    if (tid == 0) {
      reinterpret_cast<uint32_t*>(packet)[0] = static_cast<uint32_t>(run);
      reinterpret_cast<uint32_t*>(packet)[1] = entries;
      reinterpret_cast<uint32_t*>(packet)[2] = 0;
      reinterpret_cast<uint32_t*>(packet)[3] = 0;
    }
    if (tid < kDigits) reinterpret_cast<uint16_t*>(packet + 16)[tid] = static_cast<uint16_t>(counts[tid]);
    __syncthreads();
    lap(kEntries);
    const bool there = static_cast<uint32_t>(tid) < entries;
    digit = there ? digits[tid] : 0u;
    token = there ? window_of(blockIdx.x, trip) * kWindow + (tokens[tid] & (kWindow - 1)) : static_cast<uint32_t>(kZeroToken);
    if (!there && tid < run) tokens[tid] = 0;
  };
  lab::WarpSpan clock;
  clock.start();
  if constexpr ((FORM & kPhased) != 0) lapped = clock64();
  plan(0);
#pragma unroll 1
  for (int trip = 0; trip < trips; ++trip) {
    uint8_t* const packet = smem + (trip % kPackets) * kPacketRoom;
    //: THE VALUES' LOADS of this packet, then THE NEXT PACKET'S PLAN under them (a stage ahead: the loads' latency
    //: is the plan's two barriers)
    const int mine = run;
    const bool moves = tid < mine, there = static_cast<uint32_t>(tid) < entries;
    uint4 line[8];
    uint8_t byte = 0;
    if constexpr ((FORM & kLaned) != 0) {
      //: THE LANED FORM: instruction `i` of a warp loads entries `64 i + 4 warp` to that and 3, lane `8 r + c` piece
      //: `c` of entry `r`'s token's line -- 8 lanes a row an instruction, a register each
      const uint16_t* const tokens = reinterpret_cast<const uint16_t*>(packet + kHead);
#pragma unroll
      for (int i = 0; i < kMaxRun / 64; ++i) {
        if (64 * i < mine) {
          const uint32_t e = static_cast<uint32_t>(64 * i + 4 * warp + (lane >> 3));
          const uint32_t from = e < entries ? window_of(blockIdx.x, trip) * kWindow + (tokens[e] & (kWindow - 1)) : static_cast<uint32_t>(kZeroToken);
          line[i] = values[static_cast<size_t>(from) * 8 + (lane & 7)];
        }
      }
      if (there) byte = sparse[static_cast<size_t>(token) * kDigits + digit];
    } else if (moves) {
#pragma unroll
      for (int c = 0; c < 8; ++c) line[c] = values[static_cast<size_t>(token) * 8 + c];
      if (there) byte = sparse[static_cast<size_t>(token) * kDigits + digit];
    }
    plan(trip + 1);
    if constexpr ((FORM & kLaned) != 0) {
      if constexpr ((FORM & kPhased) != 0) {
        int landed = static_cast<int>(line[0].w) + byte;
        lab::wg_hold(landed);
        lap(kLoads);
      }
      uint4* const planes = reinterpret_cast<uint4*>(packet + kHead + 3 * mine);
#pragma unroll
      for (int i = 0; i < kMaxRun / 64; ++i)
        if (64 * i < mine) planes[(lane & 7) * pitch_of(mine) + 64 * i + 4 * warp + (lane >> 3)] = line[i];
      if (moves) packet[kHead + 2 * mine + tid] = byte;
    } else if (moves) {
      if constexpr ((FORM & kPhased) != 0) {
        int landed = static_cast<int>(line[7].w) + byte;
        lab::wg_hold(landed);
        lap(kLoads);
      }
      packet[kHead + 2 * mine + tid] = byte;
      uint4* const planes = reinterpret_cast<uint4*>(packet + kHead + 3 * mine);
#pragma unroll
      for (int c = 0; c < 8; ++c) planes[c * pitch_of(mine) + tid] = line[c];
    }
    //: a plane's one line past its run (the pitch's) is zero
    if (tid < 8) reinterpret_cast<uint4*>(packet + kHead + 3 * mine)[tid * pitch_of(mine) + mine] = make_uint4(0, 0, 0, 0);
    lab::bulk_fence();
    __syncthreads();
    lap(kStores);
    //: THE STORE: thread 0's one copy, and its wait on the copy before: a packet is planned into two trips after it left
    if (tid == 0 && (probe & kNoStore) == 0) {
      lab::bulk_store(out + (static_cast<size_t>(blockIdx.x) * kRing + trip % ring) * kPacketRoom,
                      base + (trip % kPackets) * kPacketRoom, static_cast<uint32_t>(packet_bytes(mine)));
      lab::bulk_commit();
      lab::bulk_wait<1>();
    }
    lap(kOut);
  }
  if (tid == 0) lab::bulk_wait<0>();
  clock.stop(times);
  if constexpr ((FORM & kPhased) != 0) {
#pragma unroll
    for (int p = 0; p < kPhases; ++p) phases[(static_cast<size_t>(blockIdx.x) * kThreads + tid) * kPhases + p] = in_phase[p];
  }
}

}  // namespace gather_group

int main(int argc, char** argv) {
  using namespace gather_group;
  lab::Bench b(argc, argv);
  for (int form : {0, 4, 5}) {
    const std::string k = lab::kernel("gather_group::gather", {form});
    b.expect(k, "LDG", ">=", 2);
    //: a lane's live tokens are a loop of its own trips, and thread 0 alone stores a packet
    b.expect(k, "LOOP_CONVERGENCE", "<=", 12);
  }
  if (b.expect_only) return 0;
  //: THE MEASURED RING IS TWO PACKETS A CTA, 10 MB the card's: 8 a CTA were 40 MB written and the loads waited on the
  //: memory (the values' loads landing 5,574 cycles of a 7,918-cycle trip)
  constexpr int kMeasured = 256, kLive = 4, kStored = 2;
  //: THE MASKS: a token live on `kLive` digits of 256, the block's 16 kept, digit-major
  std::vector<uint32_t> masks(static_cast<size_t>(b.sms) * kRing * kDigits * 32, 0);
  std::mt19937 rng(12345);
  for (size_t m = 0; m < static_cast<size_t>(b.sms) * kRing; ++m)
    for (int t = 0; t < kWindow; ++t) {
      uint32_t taken[kLive];
      for (int n = 0; n < kLive; ++n) {
        bool fresh = false;
        while (!fresh) {
          taken[n] = rng() % 256u;
          fresh = true;
          for (int i = 0; i < n; ++i) fresh = fresh && taken[i] != taken[n];
        }
        if (taken[n] < kDigits) masks[(m * kDigits + taken[n]) * 32 + t / 32] |= 1u << (t % 32);
      }
    }
  std::vector<uint32_t> table((static_cast<size_t>(kTableTokens) + 1) * 32, 0);
  for (size_t token = 0; token < kTableTokens; ++token)
    for (uint32_t w = 0; w < 32; ++w) table[token * 32 + w] = value_of(static_cast<uint32_t>(token), w);
  std::vector<uint8_t> bytes((static_cast<size_t>(kTableTokens) + 1) * kDigits, 0);
  for (size_t token = 0; token < kTableTokens; ++token)
    for (uint32_t d = 0; d < kDigits; ++d) bytes[token * kDigits + d] = static_cast<uint8_t>(sparse_of(static_cast<uint32_t>(token), d));
  uint32_t *dmasks = nullptr, *dtable = nullptr;
  uint8_t *dbytes = nullptr, *out = nullptr;
  long long* dphases = nullptr;
  LAB_CHECK(cudaMalloc(&dphases, static_cast<size_t>(b.sms) * kThreads * kPhases * 8));
  const size_t room = static_cast<size_t>(b.sms) * kRing * kPacketRoom;
  LAB_CHECK(cudaMalloc(&dmasks, masks.size() * 4));
  LAB_CHECK(cudaMalloc(&dtable, table.size() * 4));
  LAB_CHECK(cudaMalloc(&dbytes, bytes.size()));
  LAB_CHECK(cudaMalloc(&out, room));
  LAB_CHECK(cudaMemcpy(dmasks, masks.data(), masks.size() * 4, cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMemcpy(dtable, table.data(), table.size() * 4, cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMemcpy(dbytes, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
  const std::string cell = "a block's window gathered: every list a run, values only, " + std::to_string(kLive) + " live digits a token";
  const char* const phase_name[kPhases] = {"the lists and their barrier", "the entries' tokens and their barrier", "the values landing",
                                           "the values' stores and their barrier", "the packet's store"};
  //: one form's row: every packet of a ring's trips against the host's, then its window, a token and an entry
  const auto row = [&](auto kernel, const std::string& name, bool phased) {
    LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared));
    kernel<<<b.sms, kThreads, kShared>>>(b.times, dmasks, reinterpret_cast<const uint4*>(dtable), dbytes, out, dphases, kRing, kRing, 0);
    LAB_CHECK(cudaDeviceSynchronize());
    std::vector<uint8_t> got(room);
    LAB_CHECK(cudaMemcpy(got.data(), out, room, cudaMemcpyDeviceToHost));
    long long wrong = 0, total_entries = 0;
    std::string first;
    for (int blk = 0; blk < b.sms; ++blk)
      for (int trip = 0; trip < kRing; ++trip) {
        const size_t m = static_cast<size_t>(blk) * kRing + trip;
        std::vector<std::pair<int, int>> list;
        uint16_t counts[kDigits];
        for (int j = 0; j < kDigits; ++j) {
          const size_t before = list.size();
          for (int t = 0; t < kWindow; ++t)
            if ((masks[(m * kDigits + j) * 32 + t / 32] >> (t % 32)) & 1u) list.push_back({j, t});
          counts[j] = static_cast<uint16_t>(list.size() - before);
        }
        const int entries = static_cast<int>(list.size()), run = run_of(entries);
        total_entries += entries;
        if (run > kMaxRun) { std::fprintf(stderr, "lab: a packet of %d entries\n", entries); std::exit(3); }
        std::vector<uint8_t> want(static_cast<size_t>(packet_bytes(run)), 0);
        const uint32_t head[4] = {static_cast<uint32_t>(run), static_cast<uint32_t>(entries), 0, 0};
        std::memcpy(want.data(), head, 16);
        std::memcpy(want.data() + 16, counts, 32);
        for (int e = 0; e < run; ++e) {
          const bool there = e < entries;
          const uint32_t token = there ? window_of(blk, trip) * kWindow + list[e].second : static_cast<uint32_t>(kZeroToken);
          const uint16_t local = static_cast<uint16_t>(there ? list[e].second : 0);
          std::memcpy(want.data() + kHead + 2 * e, &local, 2);
          want[kHead + 2 * run + e] = static_cast<uint8_t>(there ? sparse_of(token, list[e].first) : 0);
          for (int c = 0; c < 8; ++c)
            std::memcpy(want.data() + kHead + 3 * run + (static_cast<size_t>(c) * pitch_of(run) + e) * 16, table.data() + static_cast<size_t>(token) * 32 + 4 * c, 16);
        }
        const uint8_t* const mine = got.data() + m * kPacketRoom;
        for (size_t i = 0; i < want.size(); ++i)
          if (mine[i] != want[i] && wrong++ < 4)
            first += " [CTA " + std::to_string(blk) + " trip " + std::to_string(trip) + " byte " + std::to_string(i) + " of run "
                     + std::to_string(run) + ": " + std::to_string(mine[i]) + " for " + std::to_string(want[i]) + "]";
      }
    b.check(name, wrong == 0, std::to_string(wrong) + " bytes differ" + first);
    const double mean_entries = static_cast<double>(total_entries) / (static_cast<double>(b.sms) * kRing);
    const lab::Samples cycles = b.variant(
        [&] { kernel<<<b.sms, kThreads, kShared>>>(b.times, dmasks, reinterpret_cast<const uint4*>(dtable), dbytes, out, dphases, kMeasured, kStored, 0); },
        [&] { return b.span(kWarps) / kMeasured; });
    b.record(name, "cycles a block's window a CTA", cycles);
    lab::Samples token = cycles, entry = cycles;
    for (double& v : token.v) v = v * 16.0 / kWindow;
    for (double& v : entry.v) v = v / mean_entries;
    b.record(name + ": a token a side", "SM-cycles a token", token);
    b.record(name + ": an entry", "cycles an entry a CTA", entry);
    if (!phased) return;
    //: thread 0's trip by phase (it alone stores the packet): the last launch's
    std::vector<long long> laps(static_cast<size_t>(b.sms) * kThreads * kPhases);
    LAB_CHECK(cudaMemcpy(laps.data(), dphases, laps.size() * 8, cudaMemcpyDeviceToHost));
    for (int p = 0; p < kPhases; ++p) {
      double first_thread = 0;
      for (int blk = 0; blk < b.sms; ++blk) first_thread += static_cast<double>(laps[static_cast<size_t>(blk) * kThreads * kPhases + p]) / kMeasured;
      lab::Samples one;
      one.v = {first_thread / b.sms};
      b.record(name + ": thread 0, " + phase_name[p], "cycles a trip", one);
    }
  };
  row(gather<kPlain>, cell + ", a thread an entry's eight loads", false);
  //: THE PROBES of the 8-lanes form, unchecked: what a trip is with no packet stored, and with no entry's token written
  for (int probe : {kNoStore, kNoTokens}) {
    const auto plain = gather<kLaned>;
    LAB_CHECK(cudaFuncSetAttribute(plain, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared));
    const std::string name = cell + ", loads and stores, 8 lanes a row, PROBE "
                             + (probe == kNoStore ? "no packet stored" : "no entry's token written");
    const lab::Samples cycles = b.variant(
        [&] { plain<<<b.sms, kThreads, kShared>>>(b.times, dmasks, reinterpret_cast<const uint4*>(dtable), dbytes, out, dphases, kMeasured, kStored, probe); },
        [&] { return b.span(kWarps) / kMeasured; });
    b.record(name, "cycles a block's window a CTA", cycles);
  }
  row(gather<kLaned>, cell + ", loads and stores, 8 lanes a row", false);
  row(gather<kLaned | kPhased>, cell + ", loads and stores, 8 lanes a row, its phases clocked", true);

  return 0;
}
