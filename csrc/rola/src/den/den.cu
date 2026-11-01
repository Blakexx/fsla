// csrc/rola/src/den/den.cu -- THE DEN PASS'S TRANSLATION UNIT: the host entry that validates the bundle's planes and
// the facts pass's lists, resolves the form (lists or dense) and launches the kernel; the phase-ledger bind.
// see docs/internals/den/den.md
#include "den/den_api.cuh"

#include "common/torch_seam.cuh"

#include "common/arch_runtime.cuh"
#include "den/den_kernel.cuh"

namespace rola::den {

namespace detail {

long long* g_ledger = nullptr;

//: A LEVEL'S ROWS as the kernel reads them: `[B, T, H, width]` of `dtype`, digits contiguous, each row and stride on 16
//: bytes (the copies and quarters are 16-byte loads). -- see docs/internals/den/den_api.md#refusals
Plane plane_of(const Tensor& t, int64_t B, int64_t T, int64_t H, int64_t width, Dtype dtype,
               const char* name) {
  STD_TORCH_CHECK(t.is_cuda() && t.scalar_type() == dtype, name, " is a CUDA ", dtype,
                  " tensor, got ", t.scalar_type());
  STD_TORCH_CHECK(
      t.dim() == 4 && t.size(0) == B && t.size(1) == T && t.size(2) == H && t.size(3) == width,
      name, " is [B, T, H, width] = [", B, ", ", T, ", ", H, ", ", width, "], got ",
      shape(t.sizes()));
  const int64_t size = t.element_size();
  STD_TORCH_CHECK(t.stride(3) == 1 && reinterpret_cast<uintptr_t>(t.const_data_ptr()) % 16 == 0
                      && (t.stride(0) * size) % 16 == 0 && (t.stride(1) * size) % 16 == 0
                      && (t.stride(2) * size) % 16 == 0,
                  name, "'s rows start and stride on 16 bytes, digits contiguous");
  return Plane{static_cast<const uint8_t*>(t.const_data_ptr()), t.stride(0) * size,
               t.stride(1) * size, t.stride(2) * size};
}

//: ONE SIDE'S LISTS in the facts pass's final form (`carry.md#lists`), at `width` digits over `heads` query heads.
SideLists lists_of(const std::vector<Tensor>& t, int64_t BH, int64_t windows, int64_t width,
                   int64_t heads, int W, const char* name) {
  STD_TORCH_CHECK(t.size() == 5, name, " is (entries, pre, totals, union, count), got ", t.size(),
                  " tensors");
  const int64_t groups = (width + 31) / 32;
  const std::vector<std::vector<int64_t>> want = {
      {BH, windows, width, W * heads, 4},
      {BH, windows, groups, kPreRows, list_at(W) * heads},
      {BH, windows, groups, list_at(W) * heads},
      {BH, windows, groups, W * heads},
      {BH, windows, groups}};
  for (int k = 0; k < 5; ++k) {
    const bool wide = k == 2 || k == 4;
    STD_TORCH_CHECK(t[k].is_cuda() && t[k].is_contiguous()
                        && t[k].scalar_type() == (wide ? Dtype::Int : Dtype::Short)
                        && t[k].sizes() == Shape(want[k]),
                    name, " tensor ", k, " is a contiguous CUDA ", wide ? "int32 " : "int16 ",
                    shape(Shape(want[k])), ", got ", shape(t[k].sizes()));
  }
  return SideLists{reinterpret_cast<const uint16_t*>(t[3].const_data_ptr()),
                   t[4].const_data_ptr<int32_t>(),
                   static_cast<const uint2*>(t[0].const_data_ptr()),
                   reinterpret_cast<const uint16_t*>(t[1].const_data_ptr()),
                   t[2].const_data_ptr<int32_t>(),
                   static_cast<int>(width),
                   static_cast<int>(groups),
                   static_cast<int>(heads)};
}

//: the tenants' bytes at the device's window
int tenant_bytes(int W) { return W == 1024 ? TenantsAt<1024>::kBytes : TenantsAt<512>::kBytes; }

template <bool Dense>
void launch(const DenParams& p, int W, int64_t tiles, int64_t bh, cudaStream_t stream) {
  const int bytes = tenant_bytes(W);
  static bool configured = false;
  if (!configured) {
    const int optin = current_device_properties().sharedMemPerBlockOptin;
    STD_TORCH_CHECK(bytes <= optin, "the den pass's ", bytes, "-byte tenants exceed this device's ",
                    optin, "-byte opt-in");
    ROLA_CUDA_CHECK(cudaFuncSetAttribute(den_kernel<Dense>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    configured = true;
  }
  const dim3 grid(static_cast<unsigned>(tiles), static_cast<unsigned>(bh));
  den_kernel<Dense><<<grid, kThreads, static_cast<size_t>(bytes), stream>>>(p);
  ROLA_CUDA_LAUNCH_CHECK();
}

}  // namespace detail

//: -- see docs/internals/den/den_api.md#den-forward
void den_forward(std::vector<Tensor> read_rows, std::vector<Tensor> write_rows,
                 std::vector<Tensor> read_lists, std::vector<Tensor> write_lists, Tensor list_exp,
                 Tensor gains16, Tensor gains_exp, Tensor den, Tensor num,
                 std::vector<int64_t> widths, int64_t level_s, int64_t read_list,
                 int64_t write_list, int64_t heads) {
  check_arch_table();
  const int D = static_cast<int>(widths.size());
  STD_TORCH_CHECK(D == 2 && read_rows.size() == 2 && write_rows.size() == 2,
                  "the den pass is built for depth two: level S and level R; got ", D, " levels");
  STD_TORCH_CHECK(level_s == 0 || level_s == 1, "level S is level 0 or 1, got ", level_s);
  const int64_t ls = level_s, lr = 1 - level_s;
  STD_TORCH_CHECK(widths[ls] % kRows == 0 && widths[lr] % kSpan == 0, "level S is a multiple of ",
                  kRows, " digits and level R of ", kSpan, "; got ", widths[ls], " and ",
                  widths[lr]);
  const bool dense = read_list < 0 && write_list < 0;
  STD_TORCH_CHECK(dense || (read_list == ls && write_list == lr),
                  "the den pass takes a read side listing level S (", ls,
                  ") with a write side "
                  "listing level R (",
                  lr, "), or both sides dense; got read list ", read_list, ", write list ",
                  write_list);
  const Tensor& wr0 = write_rows[0];
  STD_TORCH_CHECK(wr0.dim() == 4, "a level's rows are [B, T, H, width]");
  const int64_t B = wr0.size(0), L = wr0.size(1), Hw = wr0.size(2), BH = B * Hw;
  STD_TORCH_CHECK(heads >= 1 && heads <= kMaxHeads, "1 to ", kMaxHeads,
                  " query heads a write head (the accumulator's rows), got ", heads);
  STD_TORCH_CHECK(L % kChunk == 0 && L > 0, "the call is a whole number of ", kChunk,
                  "-token chunks (a partial window is a window), got ", L);
  //: the device's window (`common/window.cuh`), never the host pass's
  int dev = 0;
  ROLA_CUDA_CHECK(cudaGetDevice(&dev));
  const int W = device_window(dev);
  STD_TORCH_CHECK(heads <= window_heads(W), "a write head has at most ", window_heads(W),
                  " query heads at W = ", W);
  const int64_t windows = (L + W - 1) / W;
  STD_TORCH_CHECK(
      BH <= 65535 && heads * L < (int64_t{1} << 31),
      "the grid is (tiles, batch-heads) within the launch limits, a head's rows in 31 bits");

  DenParams p{};
  p.read_r = detail::plane_of(read_rows[lr], B, L, Hw * heads, widths[lr], Dtype::Half,
                              "the read side's level R rows");
  p.write_s = detail::plane_of(write_rows[ls], B, L, Hw, widths[ls], Dtype::Half,
                               "the write side's level S rows");
  if (dense) {
    p.read_s = detail::plane_of(read_rows[ls], B, L, Hw * heads, widths[ls], Dtype::Half,
                                "the read side's level S rows");
    p.write_r = detail::plane_of(write_rows[lr], B, L, Hw, widths[lr], Dtype::Half,
                                 "the write side's level R rows");
    STD_TORCH_CHECK(gains16.is_cuda() && gains16.scalar_type() == Dtype::Half
                        && gains16.is_contiguous() && gains16.dim() == 3 && gains16.size(0) == BH
                        && gains16.size(1) == L && gains16.size(2) == kChannels,
                    "gains16 is the facts pass's contiguous f16 [BH, L, 2]");
    STD_TORCH_CHECK(gains_exp.is_cuda() && gains_exp.scalar_type() == Dtype::Short
                        && gains_exp.is_contiguous() && gains_exp.dim() == 2
                        && gains_exp.size(0) == BH && gains_exp.size(1) == windows * kChannels,
                    "gains_exp is the facts pass's contiguous int16 [BH, windows, 2] (flattened)");
    p.gains16 = static_cast<const __half*>(gains16.const_data_ptr());
    p.gains_exp = gains_exp.const_data_ptr<int16_t>();
  } else {
    p.read_lists =
        detail::lists_of(read_lists, BH, windows, widths[ls], heads, W, "the read lists");
    p.write_lists = detail::lists_of(write_lists, BH, windows, widths[lr], 1, W, "the write lists");
    STD_TORCH_CHECK(list_exp.is_cuda() && list_exp.scalar_type() == Dtype::Short
                        && list_exp.is_contiguous() && list_exp.dim() == 4 && list_exp.size(0) == BH
                        && list_exp.size(1) == windows && list_exp.size(2) == widths[lr]
                        && list_exp.size(3) == kChannels,
                    "list_exp is the facts pass's contiguous int16 [BH, windows, width_R, 2]");
    p.list_exp = list_exp.const_data_ptr<int16_t>();
  }
  //: a window's rows of a write head index in 32 bits (`den_kernel.md#rows`)
  for (const Plane* pl : {&p.read_r, &p.write_s, &p.read_s, &p.write_r}) {
    STD_TORCH_CHECK(pl->base == nullptr || W * pl->st + heads * pl->sh < (int64_t{1} << 32),
                    "a window's rows of a write head span under 2^32 bytes: token stride ", pl->st,
                    ", head stride ", pl->sh, " at W = ", W);
  }
  for (Tensor* out : {&den, &num}) {
    STD_TORCH_CHECK(out->is_cuda() && out->scalar_type() == Dtype::Float && out->is_contiguous()
                        && out->dim() == 2 && out->size(0) == BH * heads && out->size(1) == L,
                    "den and num are contiguous fp32 [BH heads, L] planes the caller zeroed");
  }
  p.den = den.mutable_data_ptr<float>();
  p.num = num.mutable_data_ptr<float>();
  p.ledger = detail::g_ledger;
  p.L = static_cast<int>(L);
  p.windows = static_cast<int>(windows);
  p.heads = static_cast<int>(heads);
  p.write_heads = static_cast<int>(Hw);
  p.width_s = static_cast<int>(widths[ls]);
  p.width_r = static_cast<int>(widths[lr]);
  p.tiles_s = static_cast<int>(widths[ls] / kRows);
  p.level_s = static_cast<int>(ls);
  const int64_t tiles = (widths[ls] / kRows) * (widths[lr] / kSpan);

  const DeviceGuard guard(wr0.get_device_index());
  const cudaStream_t stream = current_stream();
  if (dense) {
    detail::launch<true>(p, W, tiles, BH, stream);
  } else {
    detail::launch<false>(p, W, tiles, BH, stream);
  }
}

//: -- see docs/internals/den/den_api.md#den-ledger-bind
void den_ledger_bind(const std::optional<Tensor>& ledger) {
  if (!ledger.has_value()) {
    detail::g_ledger = nullptr;
    return;
  }
  STD_TORCH_CHECK(ledger->is_cuda() && ledger->scalar_type() == Dtype::Long && ledger->dim() == 3
                      && ledger->size(1) == kWarps && ledger->size(2) == kDenPhases
                      && ledger->is_contiguous(),
                  "the den pass's phase ledger is a contiguous CUDA int64 tensor [ctas][", kWarps,
                  "][", kDenPhases, "]");
  detail::g_ledger = reinterpret_cast<long long*>(ledger->mutable_data_ptr<int64_t>());
}

}  // namespace rola::den
