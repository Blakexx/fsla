// measure/harness/owner/owner_skeleton.cu -- THE OWNER CARRY'S SKELETON IN THE PART HARNESS: its components on a
// cell's real words and rows, the math stubbed, each part's products dumped for the harness's model check and its time
// for the carve's pricing (KERNEL_STANDARDS §19). A bench instrument, never shipped.
// See docs/internals/owner/owner.md#skeleton
#include <torch/csrc/stable/library.h>

#include "common/torch_seam.cuh"
#include "common/phase_clock.cuh"

#include <cstdint>
#include <optional>
#include <vector>

#include "owner/owner.cuh"
#include "owner/owner_drain.cuh"
#include "owner/owner_fill.cuh"
#include "owner/owner_fold.cuh"
#include "owner/owner_lists.cuh"
#include "owner/owner_readout.cuh"

namespace rola::owner::skeleton {

namespace ops = denseref::ops;
using detail::ListTenants;

//: A CTA-WINDOW'S DUMP: the unions, the row counts, the read forms and totals, each warp's read entries.
template <class C>
struct Dump {
  static constexpr int kUnions = 2 * kWindowWords;
  static constexpr int kCounts = C::kWordRows;
  static constexpr int kForms = 2 * kWarps;
  static constexpr int kEntries = kWarps * kListCapacity;
  static constexpr int kWords = kUnions + kCounts + kForms + kEntries;
};

template <class C>
__device__ void dump_lists(const ListTenants<C>& sm, const OwnerParams& p, uint32_t* out, int tid) {
  using Dp = Dump<C>;
  if (tid < Dp::kUnions) out[tid] = sm.uni[tid / kWindowWords][tid % kWindowWords];
  if (tid < Dp::kCounts) out[Dp::kUnions + tid] = sm.count[tid];
  if (tid < kWarps) {
    out[Dp::kUnions + Dp::kCounts + tid] = sm.read_form[tid];
    out[Dp::kUnions + Dp::kCounts + kWarps + tid] = sm.read_total[tid];
  }
  uint32_t* entries = out + Dp::kUnions + Dp::kCounts + Dp::kForms;
#pragma unroll 1
  for (int i = tid; i < Dp::kEntries; i += kThreads) {
    const int warp = i / kListCapacity, e = i % kListCapacity;
    entries[i] =
        p.sparse_read != 0 && e < sm.read_total[warp] ? sm.read_entries[warp][e] : 0xffffffffu;
  }
}

//: THE LISTS PART: the words and the lists over every window of the CTA's box, the next window's words staged before this window's
//: lists, and each CTA-window's lists dumped when asked.
template <class C>
__global__ __launch_bounds__(kThreads, 1) void lists_part(const __grid_constant__ OwnerParams p) {
  extern __shared__ __align__(128) unsigned char raw[];
  ListTenants<C>& sm = *reinterpret_cast<ListTenants<C>*>(raw);
  const int tid = static_cast<int>(threadIdx.x);
  const int s0 = static_cast<int>(blockIdx.x) * C::RS, r0 = static_cast<int>(blockIdx.y) * C::RD;
  const int bh = static_cast<int>(blockIdx.z);
  const int cta = (bh * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  detail::stage_words(sm, p, bh, s0, r0, 0, 0, tid);
#pragma unroll 1
  for (int win = 0; win < p.windows; ++win) {
    const int buf = win & 1;
    ops::stage_wait<0>();
    __syncthreads();
    if (win + 1 < p.windows) detail::stage_words(sm, p, bh, s0, r0, win + 1, buf ^ 1, tid);
    detail::window_lists(sm, p, buf, tid);
    if (p.dump != nullptr)
      dump_lists(sm, p, p.dump + (static_cast<int64_t>(cta) * p.windows + win) * Dump<C>::kWords,
                 tid);
    __syncthreads();
  }
}

//: THE FILL PART: each window's lists, then its write passes and read passes, each pass's pool rows dumped to their
//: slots when asked.
template <class W>
struct FillTenants {
  ListTenants<typename W::C> lists;
  detail::WriteSlot<W> write;
  detail::ReadPool<typename W::C> read;
};

//: A WRITE ROW AS DUMPED: its runs in order, the swizzle undone, then its `(F, m)`.
template <class W>
constexpr int kDumpRow = W::kRunsBytes + static_cast<int>(sizeof(float2));

template <class W>
__global__ __launch_bounds__(kThreads, 1) void fill_part(const __grid_constant__ OwnerParams p,
                                                         uint8_t* write_dump, uint8_t* read_dump) {
  using C = typename W::C;
  extern __shared__ __align__(128) unsigned char raw[];
  FillTenants<W>& sm = *reinterpret_cast<FillTenants<W>*>(raw);
  const int tid = static_cast<int>(threadIdx.x);
  const int s0 = static_cast<int>(blockIdx.x) * C::RS, r0 = static_cast<int>(blockIdx.y) * C::RD;
  const int bh = static_cast<int>(blockIdx.z) / kChannelGroups,
            cg = static_cast<int>(blockIdx.z) % kChannelGroups;
  const int b = bh / p.H, h = bh - b * p.H;
  const int64_t cta =
      (static_cast<int64_t>(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  const detail::WriteLanes<W> write_lanes(p, b, h, cg, s0, r0, tid);
  const detail::ReadLanes<C> read_lanes(p, b, h, s0, r0, tid);
  detail::stage_words(sm.lists, p, bh, s0, r0, 0, 0, tid);
#pragma unroll 1
  for (int win = 0; win < p.windows; ++win) {
    const int buf = win & 1;
    ops::stage_wait<0>();
    __syncthreads();
    if (win + 1 < p.windows) detail::stage_words(sm.lists, p, bh, s0, r0, win + 1, buf ^ 1, tid);
    detail::window_lists(sm.lists, p, buf, tid);
    const int64_t at = (cta * p.windows + win) * kWindow;
#pragma unroll
    for (int side = 0; side < 2; ++side) {
      const int live =
          sm.lists.base[side][kWindowWords - 1] + __popc(sm.lists.uni[side][kWindowWords - 1]);
      const int cap = side == 0 ? W::kPassRows : kReadPass;
#pragma unroll 1
      for (int first = 0; first < live; first += cap) {
        const int n = min(cap, live - first);
        if (side == 0) {
          detail::fill_write(sm.write, sm.lists, write_lanes, win, first, n, tid);
          ops::stage_commit();
        } else {
          detail::fill_read(sm.read, sm.lists, read_lanes, win, first, n, tid);
        }
        ops::stage_wait<0>();
        __syncthreads();
        if (write_dump != nullptr && side == 0) {
          constexpr int kPitch = kDumpRow<W>;
          const uint8_t* const fm = reinterpret_cast<const uint8_t*>(sm.write.fm);
          uint8_t* const dst = write_dump + (at + first) * kPitch;
#pragma unroll 1
          for (int i = tid; i < n * kPitch; i += kThreads) {
            const int r = i / kPitch, c = i - r * kPitch;
            dst[i] = c < W::kRunsBytes ? sm.write.rows[W::at(r, c >> 4) + (c & 15)]
                                       : fm[8 * r + c - W::kRunsBytes];
          }
        } else if (write_dump != nullptr) {
          constexpr int kPitch = detail::ReadRows<C>::kRow;
          uint8_t* const dst = read_dump + (at + first) * kPitch;
#pragma unroll 1
          for (int i = tid; i < n * kPitch; i += kThreads) dst[i] = (&sm.read.read[0][0])[i];
        }
        __syncthreads();
      }
    }
  }
}

//: THE FOLD PART: each window's lists and units, its first two passes filled, then its passes pipelined through the
//: three write slots and folded by every warp; the final state, mass and units written (`[cta][leaf][list][channel]`
//: fp32 in the list's unit, `[cta][leaf][list]`, `[cta][list][2]`).
template <class W>
struct FoldPartTenants {
  ListTenants<typename W::C> lists;
  detail::FoldPool<W> pool;
  detail::FoldTenants<typename W::C> fold;
  int b, h;
};

//: THE FOLD PART'S PHASES on its ledger (`kClock`): the window's lists and units (its barriers' waits included); its
//: first two passes' fills; then a pass's slot wait, rows, operand loads, MMAs, mid-pass fill and tail (the mass added).
enum FoldPartPhase : int {
  kPartLists,
  kPartPrologue,
  kPartWait,
  kPartRows,
  kPartLoads,
  kPartSteps,
  kPartMiddle,
  kPartTail,
  kPartPhases
};

template <class W, bool kClock>
__global__ __launch_bounds__(kThreads, 1) void fold_part(const __grid_constant__ OwnerParams p,
                                                         float* state, float* mass, int* units) {
  using C = typename W::C;
  extern __shared__ __align__(128) unsigned char raw[];
  using Clock = PhaseClock<kPartPhases>;
  __shared__ long long s_clock[kWarps][Clock::kRow];
  FoldPartTenants<W>& sm = *reinterpret_cast<FoldPartTenants<W>*>(raw);
  const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
  const int s0 = static_cast<int>(blockIdx.x) * C::RS, r0 = static_cast<int>(blockIdx.y) * C::RD;
  const int bh = static_cast<int>(blockIdx.z) / kChannelGroups;
  //: the batch and head a pass's fill reads, kept in shared memory: no register spans the window loop for them
  if (tid == 0) {
    sm.b = bh / p.H;
    sm.h = bh - sm.b * p.H;
  }
  detail::FoldAcc<C> st{};
  detail::ListUnits lu;
  if (lane < C::kFoldLists) sm.fold.units[warp][lane] = detail::UnitBook{};
  if (tid < kWriteSlots) {
    ops::mbar_init(ops::smem_addr(&sm.pool.landed[tid]), kThreads);
    ops::mbar_init(ops::smem_addr(&sm.pool.drained[tid]), kThreads);
  }
#pragma unroll 1
  for (int i = tid; i < C::RS * C::RD; i += kThreads) (&sm.fold.mass[0][0])[i] = 0.0f;
  Clock clock(&s_clock[warp][0], lane, kClock);
  const auto lap = [&](int ph) {
    if constexpr (kClock) clock.lap(ph);
  };
  detail::stage_words(sm.lists, p, bh, s0, r0, 0, 0, tid);
  int done = 0;
#pragma unroll 1
  for (int win = 0; win < p.windows; ++win) {
    const int buf = win & 1;
    ops::stage_wait<0>();
    __syncthreads();
    if (win + 1 < p.windows) detail::stage_words(sm.lists, p, bh, s0, r0, win + 1, buf ^ 1, tid);
    detail::window_lists(sm.lists, p, buf, tid);
    detail::fold_window(st, lu, sm.fold, p, bh, r0, win, warp, lane);
    const int live = sm.lists.base[0][kWindowWords - 1] + __popc(sm.lists.uni[0][kWindowWords - 1]);
    const int passes = (live + W::kPassRows - 1) / W::kPassRows;
    lap(kPartLists);
    //: pass `x` of the window into slot `(done + x) % 3`, every thread's copies arriving on its landing; the box's
    //: coordinates re-read, so no register spans the passes for them
    const auto fill = [&](int x) {
      const int first = x * W::kPassRows, slot = (done + x) % kWriteSlots;
      const detail::WriteLanes<W> lanes(
          p, sm.b, sm.h, static_cast<int>(blockIdx.z) % kChannelGroups,
          static_cast<int>(blockIdx.x) * C::RS, static_cast<int>(blockIdx.y) * C::RD, tid);
      detail::fill_write(sm.pool.slot[slot], sm.lists, lanes, win, first,
                         min(W::kPassRows, live - first), tid);
      ops::mbar_arrive_when_landed(ops::smem_addr(&sm.pool.landed[slot]));
    };
#pragma unroll 1
    for (int x = 0; x < min(2, passes); ++x) fill(x);
    lap(kPartPrologue);
#pragma unroll 1
    for (int x = 0; x < passes; ++x) {
      const int first = x * W::kPassRows;
      detail::fold_pass(
          st, sm.fold, sm.pool, sm.lists, lu, done + x, first, min(W::kPassRows, live - first),
          warp, lane,
          [&]() {
            if (x + 2 >= passes) return;
            if (x >= 1) {
              const int before = done + x - 1;
              ops::mbar_wait(ops::smem_addr(&sm.pool.drained[before % kWriteSlots]),
                             static_cast<uint32_t>(before / kWriteSlots) & 1u);
            }
            fill(x + 2);
          },
          [&](int k) { lap(kPartWait + k); });
    }
    done += passes;
  }
  if constexpr (kClock) clock.publish(p.ledger, 0, kWarps, warp);
  const int64_t cta =
      (static_cast<int64_t>(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  const int g = lane >> 2, tq = lane & 3;
#pragma unroll
  for (int j = 0; j < C::kFoldLists; ++j)
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int leaf = detail::acc_leaf<W>(g, i >> 1), list = warp * C::kFoldLists + j;
        state[((cta * C::RS + leaf) * C::RD + list) * kChannels + 8 * n + 2 * tq + (i & 1)] =
            st.acc[j][n][i];
      }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int leaf = g + 8 * (i >> 1), list = warp * C::kFoldLists + 2 * tq + (i & 1);
    mass[(cta * C::RS + leaf) * C::RD + list] = sm.fold.mass[leaf][list];
  }
  if (tq == 0) {
    units[(cta * C::RD + warp * C::kFoldLists + g) * 2] = sm.fold.units[warp][g].u;
    units[(cta * C::RD + warp * C::kFoldLists + g) * 2 + 1] = sm.fold.units[warp][g].v;
  }
}

template <class W, bool kClock>
std::vector<Tensor> run_fold(const OwnerParams& p, int64_t bh) {
  using C = typename W::C;
  const int64_t sb = p.width[p.level_s] / C::RS, rb = p.width[p.level_r] / C::RD;
  const int64_t ctas = sb * rb * bh * kChannelGroups;
  Tensor state = empty_cuda({ctas * C::RS * C::RD * kChannels}, Dtype::Float);
  Tensor mass = empty_cuda({ctas * C::RS * C::RD}, Dtype::Float);
  Tensor units = empty_cuda({ctas * C::RD * 2}, Dtype::Int);
  constexpr int kSmem = static_cast<int>(sizeof(FoldPartTenants<W>));
  ROLA_CUDA_CHECK(cudaFuncSetAttribute(fold_part<W, kClock>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  const dim3 grid(static_cast<unsigned>(sb), static_cast<unsigned>(rb),
                  static_cast<unsigned>(bh * kChannelGroups));
  fold_part<W, kClock><<<grid, kThreads, kSmem, current_stream()>>>(
      p, state.mutable_data_ptr<float>(), mass.mutable_data_ptr<float>(),
      units.mutable_data_ptr<int32_t>());
  ROLA_CUDA_LAUNCH_CHECK();
  return {state, mass, units};
}

//: THE DRAIN PART: each window's lists, then its read passes' panels -- ones, so the planes count each token's boxes --
//: drained into `num` and `den`.
template <class C>
struct DrainTenants {
  ListTenants<C> lists;
  detail::Panel panel;
};

template <class C>
__global__ __launch_bounds__(kThreads, 1) void drain_part(const __grid_constant__ OwnerParams p,
                                                          float* num, float* den) {
  extern __shared__ __align__(128) unsigned char raw[];
  DrainTenants<C>& sm = *reinterpret_cast<DrainTenants<C>*>(raw);
  const int tid = static_cast<int>(threadIdx.x);
  const int s0 = static_cast<int>(blockIdx.x) * C::RS, r0 = static_cast<int>(blockIdx.y) * C::RD;
  const int bh = static_cast<int>(blockIdx.z) / kChannelGroups,
            cg = static_cast<int>(blockIdx.z) % kChannelGroups;
#pragma unroll 1
  for (int i = tid; i < kReadPool * kChannels; i += kThreads) (&sm.panel.num[0][0])[i] = 1.0f;
  if (tid < kReadPool) sm.panel.den[tid] = 1.0f;
  detail::stage_words(sm.lists, p, bh, s0, r0, 0, 0, tid);
#pragma unroll 1
  for (int win = 0; win < p.windows; ++win) {
    const int buf = win & 1;
    ops::stage_wait<0>();
    __syncthreads();
    if (win + 1 < p.windows) detail::stage_words(sm.lists, p, bh, s0, r0, win + 1, buf ^ 1, tid);
    detail::window_lists(sm.lists, p, buf, tid);
    const int live = sm.lists.base[1][kWindowWords - 1] + __popc(sm.lists.uni[1][kWindowWords - 1]);
#pragma unroll 1
    for (int first = 0; first < live; first += kReadPool)
      detail::drain(sm.panel, sm.lists, num, den, bh, cg, p.L, win, first,
                    min(kReadPool, live - first), tid);
  }
}

template <class C>
void run_drain(const OwnerParams& p, int64_t bh, Tensor& num, Tensor& den) {
  const int64_t sb = p.width[p.level_s] / C::RS, rb = p.width[p.level_r] / C::RD;
  constexpr int kSmem = static_cast<int>(sizeof(DrainTenants<C>));
  ROLA_CUDA_CHECK(
      cudaFuncSetAttribute(drain_part<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  const dim3 grid(static_cast<unsigned>(sb), static_cast<unsigned>(rb),
                  static_cast<unsigned>(bh * kChannelGroups));
  drain_part<C><<<grid, kThreads, kSmem, current_stream()>>>(p, num.mutable_data_ptr<float>(),
                                                             den.mutable_data_ptr<float>());
  ROLA_CUDA_LAUNCH_CHECK();
}

//: THE READOUT PART: each window's lists, then its read passes, each filled and read out by every warp in its form into
//: the shared panel; the panel's checksum written so nothing is dead.
template <class C>
struct ReadTenants {
  ListTenants<C> lists;
  detail::ReadPool<C> read;
  detail::ReadPanel panel;
};

template <class C>
__global__ __launch_bounds__(kThreads, 1) void readout_part(const __grid_constant__ OwnerParams p,
                                                            uint32_t* checksum) {
  extern __shared__ __align__(128) unsigned char raw[];
  ReadTenants<C>& sm = *reinterpret_cast<ReadTenants<C>*>(raw);
  const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
  const int s0 = static_cast<int>(blockIdx.x) * C::RS, r0 = static_cast<int>(blockIdx.y) * C::RD;
  const int bh = static_cast<int>(blockIdx.z) / kChannelGroups,
            cg = static_cast<int>(blockIdx.z) % kChannelGroups;
  const int b = bh / p.H, h = bh - b * p.H;
  const detail::ReadLanes<C> lanes(p, b, h, s0, r0, tid);
  if (tid < detail::ReadRows<C>::kRow / 4)
    reinterpret_cast<uint32_t*>(sm.read.read[kReadPass])[tid] = 0u;
  detail::ReadState<C> rs;
  {
    uint32_t* a = &rs.a[0][0][0][0][0];
#pragma unroll
    for (int i = 0; i < static_cast<int>(sizeof(rs.a) / 4); ++i)
      a[i] = 0x01010101u * static_cast<uint32_t>(lane + i);
#pragma unroll
    for (int k = 0; k < detail::ReadState<C>::kSteps; ++k)
      rs.mass[k][0] = rs.mass[k][1] = 0x01010101u;
  }
  int cursor[C::kReadDigits];
  detail::stage_words(sm.lists, p, bh, s0, r0, 0, 0, tid);
#pragma unroll 1
  for (int win = 0; win < p.windows; ++win) {
    const int buf = win & 1;
    ops::stage_wait<0>();
    __syncthreads();
    if (win + 1 < p.windows) detail::stage_words(sm.lists, p, bh, s0, r0, win + 1, buf ^ 1, tid);
    detail::window_lists(sm.lists, p, buf, tid);
    {
      int off = 0;
#pragma unroll
      for (int i = 0; i < C::kReadDigits; ++i) {
        cursor[i] = sm.lists.read_form[warp] == kStepPerList ? off : 0;
        off += sm.lists.count[C::RD + warp * C::kReadDigits + i];
      }
    }
    const int live = sm.lists.base[1][kWindowWords - 1] + __popc(sm.lists.uni[1][kWindowWords - 1]);
#pragma unroll 1
    for (int first = 0; first < live; first += kReadPass) {
      const int n = min(kReadPass, live - first);
      detail::fill_read(sm.read, sm.lists, lanes, win, first, n, tid);
      ops::stage_wait<0>();
      __syncthreads();
      detail::readout_pass(sm.panel, sm.read, sm.lists, p, rs, cursor, first, n, warp, lane);
      __syncthreads();
    }
  }
  uint32_t sum = 0;
#pragma unroll 1
  for (int i = tid; i < kReadPool * kChannels; i += kThreads) sum += (&sm.panel.num[0][0])[i];
  atomicAdd(checksum, sum);
}

template <class C>
Tensor run_readout(const OwnerParams& p, int64_t bh) {
  const int64_t sb = p.width[p.level_s] / C::RS, rb = p.width[p.level_r] / C::RD;
  Tensor out = zeros_cuda({1}, Dtype::Int);
  constexpr int kSmem = static_cast<int>(sizeof(ReadTenants<C>));
  ROLA_CUDA_CHECK(
      cudaFuncSetAttribute(readout_part<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  const dim3 grid(static_cast<unsigned>(sb), static_cast<unsigned>(rb),
                  static_cast<unsigned>(bh * kChannelGroups));
  readout_part<C><<<grid, kThreads, kSmem, current_stream()>>>(
      p, reinterpret_cast<uint32_t*>(out.mutable_data_ptr<int32_t>()));
  ROLA_CUDA_LAUNCH_CHECK();
  return out;
}

template <class C>
Tensor run_lists(OwnerParams p, int64_t bh, bool dump, int device) {
  const int64_t ctas = (p.width[p.level_s] / C::RS) * (p.width[p.level_r] / C::RD) * bh;
  Tensor out = dump ? empty_cuda({ctas * p.windows * Dump<C>::kWords}, Dtype::Int)
                    : empty_cuda({1}, Dtype::Int);
  p.dump = dump ? reinterpret_cast<uint32_t*>(out.mutable_data_ptr<int32_t>()) : nullptr;
  constexpr int kSmem = static_cast<int>(sizeof(ListTenants<C>));
  ROLA_CUDA_CHECK(
      cudaFuncSetAttribute(lists_part<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  const dim3 grid(static_cast<unsigned>(p.width[p.level_s] / C::RS),
                  static_cast<unsigned>(p.width[p.level_r] / C::RD), static_cast<unsigned>(bh));
  lists_part<C><<<grid, kThreads, kSmem, current_stream()>>>(p);
  ROLA_CUDA_LAUNCH_CHECK();
  (void)device;
  return out;
}

//: -- see docs/internals/owner/owner.md#skeleton
Tensor lists_part_op(Tensor words, std::vector<int64_t> widths, int64_t level_s,
                     int64_t sparse_write, int64_t sparse_read, int64_t rs, int64_t rd,
                     int64_t step, int64_t dump) {
  STD_TORCH_CHECK(words.is_cuda() && words.scalar_type() == Dtype::Int && words.dim() == 4
                      && words.size(1) == 2 && words.is_contiguous(),
                  "words is the facts pass's contiguous int32 [BH, 2, rows, L / 32]");
  STD_TORCH_CHECK(widths.size() == 2 && (level_s == 0 || level_s == 1),
                  "two levels, one of them read-sparse");
  OwnerParams p{};
  p.words = reinterpret_cast<const uint32_t*>(words.const_data_ptr<int32_t>());
  p.row_words = words.size(3);
  p.side_words = words.size(2) * words.size(3);
  p.level_s = static_cast<int>(level_s);
  p.level_r = 1 - p.level_s;
  p.width[0] = static_cast<int>(widths[0]);
  p.width[1] = static_cast<int>(widths[1]);
  p.row_base[0] = 0;
  p.row_base[1] = static_cast<int>(widths[0]);
  p.sparse_write = static_cast<int>(sparse_write);
  p.sparse_read = static_cast<int>(sparse_read);
  p.L = static_cast<int>(words.size(3) * 32);
  STD_TORCH_CHECK(p.L % kWindow == 0, "whole windows");
  p.windows = p.L / kWindow;
  p.step = static_cast<int>(step);
  const int device = words.get_device_index();
  const DeviceGuard guard(device);
  if (rs == 16 && rd == 64) return run_lists<Carve<16, 64>>(p, words.size(0), dump != 0, device);
  STD_TORCH_CHECK(rs == 32 && rd == 32, "the carves are (16, 64) and (32, 32)");
  return run_lists<Carve<32, 32>>(p, words.size(0), dump != 0, device);
}

RowView row_view(const Tensor& t, int64_t width) {
  const int64_t e = static_cast<int64_t>(t.element_size());
  STD_TORCH_CHECK(t.is_cuda() && t.dim() == 4 && t.size(3) == width && t.stride(3) == 1
                      && reinterpret_cast<uintptr_t>(t.const_data_ptr()) % 16 == 0
                      && t.stride(1) * e % 16 == 0 && t.stride(2) * e % 16 == 0
                      && t.stride(0) * e % 16 == 0,
                  "a row view is [B, T, H, width], starting and striding on 16 bytes");
  return RowView{t.const_data_ptr(), t.stride(0) * e, t.stride(1) * e, t.stride(2) * e};
}

//: THE WRITE ROWS' FORMAT, from the write planes' types: `run(WriteRows<C, half S, half R>{})`; a dense level's rows
//: are f16 and a sparse one's u8, so level R's type is its sparsity, which the host holds to `sparse_write`.
template <class C, class Run>
auto with_write_rows(const OwnerParams& p, const std::vector<Tensor>& write, Run&& run) {
  const bool half_s = write[p.level_s].scalar_type() == Dtype::Half;
  const bool half_r = write[p.level_r].scalar_type() == Dtype::Half;
  STD_TORCH_CHECK(half_r == (p.sparse_write == 0),
                  "level R's write rows are f16 when it is write-dense and u8 when sparse");
  if (half_s)
    return half_r ? run(detail::WriteRows<C, true, true>{})
                  : run(detail::WriteRows<C, true, false>{});
  return half_r ? run(detail::WriteRows<C, false, true>{})
                : run(detail::WriteRows<C, false, false>{});
}

//: THE FACTS PASS'S `(F, m)` PLANE, `[BH, L, 2]` fp32.
const float2* fm_plane(const Tensor& fm, const OwnerParams& p, int64_t bh) {
  STD_TORCH_CHECK(fm.is_cuda() && fm.scalar_type() == Dtype::Float && fm.is_contiguous()
                      && fm.numel() == bh * p.L * 2,
                  "fm is the facts pass's contiguous fp32 [BH, L, 2]");
  return reinterpret_cast<const float2*>(fm.const_data_ptr<float>());
}

template <class W>
std::vector<Tensor> run_fill(OwnerParams p, int64_t bh, bool dump) {
  using C = typename W::C;
  const int64_t sb = p.width[p.level_s] / C::RS, rb = p.width[p.level_r] / C::RD;
  const int64_t slots = sb * rb * bh * kChannelGroups * p.windows * kWindow;
  Tensor w = dump ? zeros_cuda({slots * kDumpRow<W>}, Dtype::Byte) : empty_cuda({1}, Dtype::Byte);
  Tensor r = dump ? zeros_cuda({slots * detail::ReadRows<C>::kRow}, Dtype::Byte)
                  : empty_cuda({1}, Dtype::Byte);
  constexpr int kSmem = static_cast<int>(sizeof(FillTenants<W>));
  ROLA_CUDA_CHECK(
      cudaFuncSetAttribute(fill_part<W>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  const dim3 grid(static_cast<unsigned>(sb), static_cast<unsigned>(rb),
                  static_cast<unsigned>(bh * kChannelGroups));
  fill_part<W><<<grid, kThreads, kSmem, current_stream()>>>(
      p, dump ? w.mutable_data_ptr<uint8_t>() : nullptr,
      dump ? r.mutable_data_ptr<uint8_t>() : nullptr);
  ROLA_CUDA_LAUNCH_CHECK();
  return {w, r};
}

//: THE BUNDLE'S PARAMS for the fill and the readout.
OwnerParams bundle_params(const std::vector<Tensor>& read, const std::vector<Tensor>& write,
                          const Tensor& v, const Tensor& words, int64_t level_s,
                          int64_t sparse_write, int64_t sparse_read, int64_t step);

//: -- see docs/internals/owner/owner.md#skeleton
std::vector<Tensor> fill_part_op(std::vector<Tensor> read, std::vector<Tensor> write, Tensor v,
                                 Tensor fm, Tensor words, int64_t level_s, int64_t sparse_write,
                                 int64_t sparse_read, int64_t rs, int64_t rd, int64_t dump) {
  OwnerParams p =
      bundle_params(read, write, v, words, level_s, sparse_write, sparse_read, kStepChosen);
  const int64_t bh = read[0].size(0) * read[0].size(2);
  p.fm = fm_plane(fm, p, bh);
  const DeviceGuard guard(words.get_device_index());
  const auto fill = [&](auto w) { return run_fill<decltype(w)>(p, bh, dump != 0); };
  if (rs == 16 && rd == 64) return with_write_rows<Carve<16, 64>>(p, write, fill);
  STD_TORCH_CHECK(rs == 32 && rd == 32, "the carves are (16, 64) and (32, 32)");
  return with_write_rows<Carve<32, 32>>(p, write, fill);
}

//: -- see docs/internals/owner/owner.md#skeleton
Tensor readout_part_op(std::vector<Tensor> read, std::vector<Tensor> write, Tensor v, Tensor words,
                       int64_t level_s, int64_t sparse_write, int64_t sparse_read, int64_t rs,
                       int64_t rd, int64_t step) {
  const OwnerParams p =
      bundle_params(read, write, v, words, level_s, sparse_write, sparse_read, step);
  const int64_t bh = read[0].size(0) * read[0].size(2);
  const DeviceGuard guard(words.get_device_index());
  if (rs == 16 && rd == 64) return run_readout<Carve<16, 64>>(p, bh);
  STD_TORCH_CHECK(rs == 32 && rd == 32, "the carves are (16, 64) and (32, 32)");
  return run_readout<Carve<32, 32>>(p, bh);
}

OwnerParams bundle_params(const std::vector<Tensor>& read, const std::vector<Tensor>& write,
                          const Tensor& v, const Tensor& words, int64_t level_s,
                          int64_t sparse_write, int64_t sparse_read, int64_t step) {
  STD_TORCH_CHECK(read.size() == 2 && write.size() == 2, "two levels a side");
  OwnerParams p{};
  p.words = reinterpret_cast<const uint32_t*>(words.const_data_ptr<int32_t>());
  p.row_words = words.size(3);
  p.side_words = words.size(2) * words.size(3);
  p.level_s = static_cast<int>(level_s);
  p.level_r = 1 - p.level_s;
  for (int l = 0; l < 2; ++l) {
    p.width[l] = static_cast<int>(read[l].size(3));
    p.read[l] = row_view(read[l], p.width[l]);
    p.write[l] = row_view(write[l], p.width[l]);
  }
  p.row_base[0] = 0;
  p.row_base[1] = p.width[0];
  STD_TORCH_CHECK(v.scalar_type() == Dtype::Half, "the values are f16");
  p.v = row_view(v, 2 * kChannels);
  p.sparse_write = static_cast<int>(sparse_write);
  p.sparse_read = static_cast<int>(sparse_read);
  p.H = static_cast<int>(read[0].size(2));
  p.L = static_cast<int>(read[0].size(1));
  STD_TORCH_CHECK(p.L % kWindow == 0 && words.size(3) * 32 == p.L,
                  "whole windows, the words' tokens");
  p.windows = p.L / kWindow;
  p.step = static_cast<int>(step);
  return p;
}

//: -- see docs/internals/owner/owner.md#skeleton
std::vector<Tensor> fold_part_op(std::vector<Tensor> read, std::vector<Tensor> write, Tensor v,
                                 Tensor fm, Tensor words, Tensor list_exp, int64_t level_s,
                                 int64_t sparse_write, int64_t sparse_read, int64_t rs, int64_t rd,
                                 int64_t step, std::optional<Tensor> ledger) {
  OwnerParams p = bundle_params(read, write, v, words, level_s, sparse_write, sparse_read, step);
  STD_TORCH_CHECK(list_exp.is_cuda() && list_exp.scalar_type() == Dtype::Short
                      && list_exp.is_contiguous() && list_exp.dim() == 4
                      && list_exp.size(1) == p.windows && list_exp.size(3) == 2
                      && list_exp.size(2) >= p.width[p.level_r],
                  "list_exp is the facts pass's contiguous int16 [BH, windows, B_R, 2]");
  p.list_exp = reinterpret_cast<const uint32_t*>(list_exp.const_data_ptr<int16_t>());
  p.list_width = static_cast<int>(list_exp.size(2));
  if (ledger.has_value()) {
    STD_TORCH_CHECK(ledger->is_cuda() && ledger->scalar_type() == Dtype::Long
                        && ledger->is_contiguous() && ledger->numel() == kWarps * kPartPhases,
                    "the fold part's ledger is a contiguous CUDA int64 [8, 8]");
    p.ledger = reinterpret_cast<long long*>(ledger->mutable_data_ptr<int64_t>());
  }
  const int64_t bh = read[0].size(0) * read[0].size(2);
  p.fm = fm_plane(fm, p, bh);
  const DeviceGuard guard(words.get_device_index());
  STD_TORCH_CHECK(rs == 16 && rd == 64, "the fold is built for the ruled 16 x 64 carve");
  return with_write_rows<Carve<16, 64>>(p, write, [&](auto w) {
    using W = decltype(w);
    return p.ledger ? run_fold<W, true>(p, bh) : run_fold<W, false>(p, bh);
  });
}

//: -- see docs/internals/owner/owner.md#skeleton
void drain_part_op(Tensor words, std::vector<int64_t> widths, int64_t level_s, int64_t sparse_write,
                   int64_t sparse_read, int64_t heads, int64_t rs, int64_t rd, Tensor num,
                   Tensor den) {
  STD_TORCH_CHECK(words.is_cuda() && words.scalar_type() == Dtype::Int && words.dim() == 4
                      && words.is_contiguous(),
                  "words is the facts pass's contiguous int32 [BH, 2, rows, L / 32]");
  const int64_t bh = words.size(0), L = words.size(3) * 32;
  STD_TORCH_CHECK(num.is_contiguous() && num.numel() == bh * L * kChannelGroups * kChannels
                      && den.is_contiguous() && den.numel() == bh * L,
                  "num is fp32 [BH, L, 64] and den [BH, L]");
  OwnerParams p{};
  p.words = reinterpret_cast<const uint32_t*>(words.const_data_ptr<int32_t>());
  p.row_words = words.size(3);
  p.side_words = words.size(2) * words.size(3);
  p.level_s = static_cast<int>(level_s);
  p.level_r = 1 - p.level_s;
  p.width[0] = static_cast<int>(widths[0]);
  p.width[1] = static_cast<int>(widths[1]);
  p.row_base[0] = 0;
  p.row_base[1] = p.width[0];
  p.sparse_write = static_cast<int>(sparse_write);
  p.sparse_read = static_cast<int>(sparse_read);
  p.H = static_cast<int>(heads);
  p.L = static_cast<int>(L);
  p.windows = p.L / kWindow;
  const DeviceGuard guard(words.get_device_index());
  if (rs == 16 && rd == 64) return run_drain<Carve<16, 64>>(p, bh, num, den);
  STD_TORCH_CHECK(rs == 32 && rd == 32, "the carves are (16, 64) and (32, 32)");
  run_drain<Carve<32, 32>>(p, bh, num, den);
}

int64_t lists_dump_words(int64_t rs, int64_t rd) {
  return rs == 16 && rd == 64 ? Dump<Carve<16, 64>>::kWords : Dump<Carve<32, 32>>::kWords;
}

}  // namespace rola::owner::skeleton

STABLE_TORCH_LIBRARY(rola_owner, m) {
  m.def(
      "lists_part(Tensor words, int[] widths, int level_s, int sparse_write, int sparse_read, int rs, int rd, "
      "int step, int dump) -> Tensor");
  m.def("lists_dump_words(int rs, int rd) -> int");
  m.def(
      "fold_part(Tensor[] read, Tensor[] write, Tensor v, Tensor fm, Tensor words, Tensor list_exp, "
      "int level_s, int sparse_write, int sparse_read, int rs, int rd, int step, Tensor(a!)? ledger) "
      "-> Tensor[]");
  m.def(
      "drain_part(Tensor words, int[] widths, int level_s, int sparse_write, int sparse_read, int heads, int rs, "
      "int rd, Tensor(a!) num, Tensor(b!) den) -> ()");
  m.def(
      "readout_part(Tensor[] read, Tensor[] write, Tensor v, Tensor words, int level_s, int sparse_write, "
      "int sparse_read, int rs, int rd, int step) -> Tensor");
  m.def(
      "fill_part(Tensor[] read, Tensor[] write, Tensor v, Tensor fm, Tensor words, int level_s, "
      "int sparse_write, int sparse_read, int rs, int rd, int dump) -> Tensor[]");
}

STABLE_TORCH_LIBRARY_IMPL(rola_owner, CompositeExplicitAutograd, m) {
  m.impl("lists_part", TORCH_BOX(&rola::owner::skeleton::lists_part_op));
  m.impl("lists_dump_words", TORCH_BOX(&rola::owner::skeleton::lists_dump_words));
  m.impl("fill_part", TORCH_BOX(&rola::owner::skeleton::fill_part_op));
  m.impl("fold_part", TORCH_BOX(&rola::owner::skeleton::fold_part_op));
  m.impl("drain_part", TORCH_BOX(&rola::owner::skeleton::drain_part_op));
  m.impl("readout_part", TORCH_BOX(&rola::owner::skeleton::readout_part_op));
}
