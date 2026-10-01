// measure/harness/owner/owner_skeleton.cu -- THE OWNER CARRY'S SKELETON IN THE PART HARNESS: its components on a
// cell's real words and rows, the math stubbed, each part's products dumped for the harness's model check and its time
// for the carve's pricing (KERNEL_STANDARDS §19). A bench instrument, never shipped.
// See docs/internals/owner/owner.md#skeleton
#include <torch/csrc/stable/library.h>

#include "common/torch_seam.cuh"

#include <cstdint>
#include <optional>
#include <vector>

#include "owner/owner.cuh"
#include "owner/owner_fill.cuh"
#include "owner/owner_lists.cuh"

namespace rola::owner::skeleton {

namespace ops = denseref::ops;
using detail::ListTenants;

//: A CTA-WINDOW'S DUMP: the unions, the row counts, the forms and totals, each warp's entries a side.
template <class C>
struct Dump {
  static constexpr int kUnions = 2 * kWindowWords;
  static constexpr int kCounts = C::kWordRows;
  static constexpr int kForms = 4 * kWarps;
  static constexpr int kEntries = 2 * kWarps * kListCapacity;
  static constexpr int kWords = kUnions + kCounts + kForms + kEntries;
};

template <class C>
__device__ void dump_lists(const ListTenants<C>& sm, const OwnerParams& p, uint32_t* out, int tid) {
  using Dp = Dump<C>;
  if (tid < Dp::kUnions) out[tid] = sm.uni[tid / kWindowWords][tid % kWindowWords];
  if (tid < Dp::kCounts) out[Dp::kUnions + tid] = sm.count[tid];
  if (tid < 2 * kWarps) {
    out[Dp::kUnions + Dp::kCounts + tid] = sm.form[tid / kWarps][tid % kWarps];
    out[Dp::kUnions + Dp::kCounts + 2 * kWarps + tid] = sm.total[tid / kWarps][tid % kWarps];
  }
  uint32_t* entries = out + Dp::kUnions + Dp::kCounts + Dp::kForms;
#pragma unroll 1
  for (int i = tid; i < Dp::kEntries; i += kThreads) {
    const int side = i / (kWarps * kListCapacity), rest = i % (kWarps * kListCapacity);
    const int warp = rest / kListCapacity, e = rest % kListCapacity;
    const bool sparse = (side == 0 ? p.sparse_write : p.sparse_read) != 0;
    entries[i] = sparse && e < sm.total[side][warp] ? sm.entries[side][warp][e] : 0xffffffffu;
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
template <class C>
struct FillTenants {
  ListTenants<C> lists;
  detail::Pools<C> pools;
};

template <class C>
__global__ __launch_bounds__(kThreads, 1) void fill_part(const __grid_constant__ OwnerParams p,
                                                         uint8_t* write_dump, uint8_t* read_dump) {
  using P = detail::Pools<C>;
  extern __shared__ __align__(128) unsigned char raw[];
  FillTenants<C>& sm = *reinterpret_cast<FillTenants<C>*>(raw);
  const int tid = static_cast<int>(threadIdx.x);
  const int s0 = static_cast<int>(blockIdx.x) * C::RS, r0 = static_cast<int>(blockIdx.y) * C::RD;
  const int bh = static_cast<int>(blockIdx.z) / kChannelGroups,
            cg = static_cast<int>(blockIdx.z) % kChannelGroups;
  const int b = bh / p.H, h = bh - b * p.H;
  const int64_t cta =
      (static_cast<int64_t>(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  const detail::FillLanes<C> lanes(p, b, h, cg, s0, r0, tid);
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
      const int cap = side == 0 ? kWritePool : kReadPool;
#pragma unroll 1
      for (int first = 0; first < live; first += cap) {
        const int n = min(cap, live - first);
        if (side == 0)
          detail::fill_write(sm.pools, sm.lists, lanes, p, win, first, n, tid);
        else
          detail::fill_read(sm.pools, sm.lists, lanes, win, first, n, tid);
        ops::stage_wait<0>();
        __syncthreads();
        if (write_dump != nullptr) {
          const int row_bytes = side == 0 ? P::kWriteRow : P::kReadRow;
          const uint8_t* src = side == 0 ? &sm.pools.write[0][0] : &sm.pools.read[0][0];
          uint8_t* dst = (side == 0 ? write_dump : read_dump) + (at + first) * row_bytes;
#pragma unroll 1
          for (int i = tid; i < n * row_bytes; i += kThreads) dst[i] = src[i];
        }
        __syncthreads();
      }
    }
  }
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
  STD_TORCH_CHECK(t.is_cuda() && t.dim() == 4 && t.size(3) == width && t.stride(3) == 1
                      && reinterpret_cast<uintptr_t>(t.const_data_ptr()) % 16 == 0
                      && t.stride(1) % 16 == 0 && t.stride(2) % 16 == 0 && t.stride(0) % 16 == 0,
                  "a row view is [B, T, H, width] bytes, starting and striding on 16 bytes");
  return RowView{t.const_data_ptr(), t.stride(0), t.stride(1), t.stride(2)};
}

ColumnView column_view(const Tensor& t) {
  STD_TORCH_CHECK(t.is_cuda() && t.scalar_type() == Dtype::Float && t.dim() == 3,
                  "a column is fp32 [B, T, H]");
  return ColumnView{t.const_data_ptr<float>(), t.stride(0), t.stride(1), t.stride(2)};
}

template <class C>
std::vector<Tensor> run_fill(OwnerParams p, int64_t bh, bool dump) {
  using P = detail::Pools<C>;
  const int64_t sb = p.width[p.level_s] / C::RS, rb = p.width[p.level_r] / C::RD;
  const int64_t slots = sb * rb * bh * kChannelGroups * p.windows * kWindow;
  Tensor w = dump ? zeros_cuda({slots * P::kWriteRow}, Dtype::Byte) : empty_cuda({1}, Dtype::Byte);
  Tensor r = dump ? zeros_cuda({slots * P::kReadRow}, Dtype::Byte) : empty_cuda({1}, Dtype::Byte);
  constexpr int kSmem = static_cast<int>(sizeof(FillTenants<C>));
  ROLA_CUDA_CHECK(
      cudaFuncSetAttribute(fill_part<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  const dim3 grid(static_cast<unsigned>(sb), static_cast<unsigned>(rb),
                  static_cast<unsigned>(bh * kChannelGroups));
  fill_part<C><<<grid, kThreads, kSmem, current_stream()>>>(
      p, dump ? w.mutable_data_ptr<uint8_t>() : nullptr,
      dump ? r.mutable_data_ptr<uint8_t>() : nullptr);
  ROLA_CUDA_LAUNCH_CHECK();
  return {w, r};
}

//: -- see docs/internals/owner/owner.md#skeleton
std::vector<Tensor> fill_part_op(std::vector<Tensor> read, std::vector<Tensor> write,
                                 std::vector<Tensor> scale_write, Tensor g_write, Tensor v_hat,
                                 std::optional<Tensor> v_lo, Tensor v_scale, Tensor words,
                                 int64_t level_s, int64_t sparse_write, int64_t sparse_read,
                                 int64_t rs, int64_t rd, int64_t dump) {
  STD_TORCH_CHECK(read.size() == 2 && write.size() == 2 && scale_write.size() == 2,
                  "two levels a side");
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
  p.factor[0] = column_view(g_write);
  p.factor[2] = column_view(scale_write[0]);
  p.factor[3] = column_view(scale_write[1]);
  p.factor[1] = column_view(v_scale);
  p.v_hat = row_view(v_hat, 2 * kChannels);
  p.v_lo = v_lo.has_value() ? row_view(*v_lo, 2 * kChannels) : RowView{nullptr, 0, 0, 0};
  p.sparse_write = static_cast<int>(sparse_write);
  p.sparse_read = static_cast<int>(sparse_read);
  p.H = static_cast<int>(read[0].size(2));
  p.L = static_cast<int>(read[0].size(1));
  STD_TORCH_CHECK(p.L % kWindow == 0 && words.size(3) * 32 == p.L,
                  "whole windows, the words' tokens");
  p.windows = p.L / kWindow;
  const int64_t bh = read[0].size(0) * read[0].size(2);
  const DeviceGuard guard(words.get_device_index());
  if (rs == 16 && rd == 64) return run_fill<Carve<16, 64>>(p, bh, dump != 0);
  STD_TORCH_CHECK(rs == 32 && rd == 32, "the carves are (16, 64) and (32, 32)");
  return run_fill<Carve<32, 32>>(p, bh, dump != 0);
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
      "fill_part(Tensor[] read, Tensor[] write, Tensor[] scale_write, Tensor g_write, Tensor v_hat, Tensor? v_lo, "
      "Tensor v_scale, Tensor words, int level_s, int sparse_write, int sparse_read, int rs, int rd, int dump) "
      "-> Tensor[]");
}

STABLE_TORCH_LIBRARY_IMPL(rola_owner, CompositeExplicitAutograd, m) {
  m.impl("lists_part", TORCH_BOX(&rola::owner::skeleton::lists_part_op));
  m.impl("lists_dump_words", TORCH_BOX(&rola::owner::skeleton::lists_dump_words));
  m.impl("fill_part", TORCH_BOX(&rola::owner::skeleton::fill_part_op));
}
