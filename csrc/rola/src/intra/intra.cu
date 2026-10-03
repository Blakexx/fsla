// THE INTRA'S TRANSLATION UNIT -- the call's geometry (its levels' widths, stages and footprint), the host entry that
// validates the bundle's views and launches the kernel, the phase-ledger bind and the build stamp.
// see docs/internals/intra/intra.md

#include "common/torch_seam.cuh"

#include <cstdint>
#include <optional>
#include <vector>

#include "common/arch_runtime.cuh"
#include "intra/intra_api.cuh"
#include "intra/intra_kernel.cuh"

namespace rola::intra {

//: NOT an anonymous namespace: a kernel's mangled name must not carry the build path.
namespace stampdet {

//: THE BUILT KERNEL's fixed tenants and the operand block's size, read back by a test as the device-side build fact.
__global__ void stamp_kernel(int64_t* out) {
  *out = static_cast<int64_t>(sizeof(detail::Smem)) * 1000003
         + static_cast<int64_t>(sizeof(IntraParams));
}

}  // namespace stampdet

namespace detail {

//: THE CALL'S GEOMETRY from its levels' widths: each level's slabs and stages, the reader row's pitch and the footprint.
//: -- see docs/internals/intra/intra.md#stages
struct Geometry {
  Stage stage[kMaxLevels][kLevelStages];
  int slab0[kMaxLevels], level_stages[kMaxLevels];
  uint32_t level_slabs[kMaxLevels];
  int stages, slabs, pitch;
  int64_t smem;
};

Geometry geometry(const std::vector<int64_t>& widths) {
  const int D = static_cast<int>(widths.size());
  STD_TORCH_CHECK(D >= 1 && D <= kMaxLevels,
                  "the intra reads u8 rows, which serve one or two levels; got ", D);
  Geometry g{};
  for (int l = 0; l < D; ++l) {
    const int64_t w = widths[l];
    STD_TORCH_CHECK(w >= kChunk && w <= kMaxLevelWidth && w % kChunk == 0, "level ", l, " is ", w,
                    " digits wide; the intra reads a level of ", kChunk, " to ", kMaxLevelWidth,
                    " digits in whole ", kChunk, "-digit copy granules");
    const int slabs = static_cast<int>((w + kSlabDigits - 1) / kSlabDigits);
    g.slab0[l] = g.slabs;
    g.level_slabs[l] = ((1u << slabs) - 1u) << g.slabs;
    for (int first = 0; first < slabs; first += kSlabsPerStage) {
      const int n = slabs - first < kSlabsPerStage ? slabs - first : kSlabsPerStage;
      g.stage[l][g.level_stages[l]++] = Stage{g.stages++, g.slabs + first, (1u << n) - 1u};
    }
    g.slabs += slabs;
  }
  g.pitch = (g.stages > kMinPitchStages ? g.stages : kMinPitchStages) * kStageDigits;
  g.smem = static_cast<int64_t>(sizeof(Smem)) + static_cast<int64_t>(kBlockRows) * g.pitch;
  return g;
}

RowView row_view(const Tensor& t, int64_t B, int64_t T, int64_t H, int64_t width, Dtype dtype,
                 const char* name) {
  STD_TORCH_CHECK(t.is_cuda() && t.scalar_type() == dtype, name, " is a CUDA ", dtype,
                  " tensor, got ", t.scalar_type());
  STD_TORCH_CHECK(
      t.dim() == 4 && t.size(0) == B && t.size(1) == T && t.size(2) == H && t.size(3) == width,
      name, " is [B, T, H, width] = [", B, ", ", T, ", ", H, ", ", width, "], got ",
      shape(t.sizes()));
  const int64_t bytes = t.element_size();
  STD_TORCH_CHECK(t.stride(3) == 1 && reinterpret_cast<uintptr_t>(t.const_data_ptr()) % kChunk == 0
                      && (t.stride(0) * bytes) % kChunk == 0 && (t.stride(1) * bytes) % kChunk == 0
                      && (t.stride(2) * bytes) % kChunk == 0,
                  name, "'s rows start and stride on 16 bytes, digits contiguous");
  STD_TORCH_CHECK(
      t.stride(1) > 0 && t.stride(1) <= INT32_MAX, name,
      "'s token stride fits 31 bits (a copy adds its tile's rows by one 32-bit stride)");
  return RowView{t.const_data_ptr(), t.stride(0), t.stride(1), t.stride(2)};
}

ColumnView column_view(const Tensor& t, int64_t B, int64_t T, int64_t H, const char* name) {
  STD_TORCH_CHECK(t.is_cuda() && t.scalar_type() == Dtype::Float && t.dim() == 3 && t.size(0) == B
                      && t.size(1) == T && t.size(2) == H,
                  name, " is a CUDA fp32 [B, T, H] = [", B, ", ", T, ", ", H, "], got ",
                  shape(t.sizes()));
  STD_TORCH_CHECK(t.stride(1) > 0 && t.stride(1) <= INT32_MAX, name,
                  "'s token stride fits 31 bits");
  return ColumnView{t.const_data_ptr<float>(), t.stride(0), t.stride(1), t.stride(2)};
}

template <int D>
void launch(const IntraParams& p, int64_t smem, int64_t bh, cudaStream_t stream) {
  //: the attribute call is host time on the launch path: raised only when a call's footprint passes the last one set
  static int64_t configured = 0;
  if (smem > configured) {
    const int optin = current_device_properties().sharedMemPerBlockOptin;
    STD_TORCH_CHECK(smem <= optin, "the intra's ", smem,
                    "-byte footprint at these widths exceeds this device's ", optin,
                    "-byte opt-in");
    ROLA_CUDA_CHECK(cudaFuncSetAttribute(
        intra_kernel<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
    configured = smem;
  }
  const dim3 grid(static_cast<unsigned>(p.windows * Pieces::count()), static_cast<unsigned>(bh));
  intra_kernel<D><<<grid, kThreads, static_cast<size_t>(smem), stream>>>(p);
  ROLA_CUDA_LAUNCH_CHECK();
}

//: THE BOUND PHASE LEDGER, or null.
long long* g_ledger = nullptr;

}  // namespace detail

//: -- see docs/internals/intra/intra_api.md#rola-intra-ledger-bind
void rola_intra_ledger_bind(std::optional<Tensor> ledger) {
  if (!ledger.has_value()) {
    detail::g_ledger = nullptr;
    return;
  }
  STD_TORCH_CHECK(ledger->is_cuda() && ledger->scalar_type() == Dtype::Long && ledger->dim() == 2
                      && ledger->size(0) == kWarps && ledger->size(1) == kIntraPhases
                      && ledger->is_contiguous(),
                  "the intra's phase ledger is a contiguous CUDA int64 tensor [", kWarps, "][",
                  kIntraPhases, "]");
  detail::g_ledger = reinterpret_cast<long long*>(ledger->mutable_data_ptr<int64_t>());
}

//: -- see docs/internals/intra/intra_api.md#rola-intra-forward
void rola_intra_forward(std::vector<Tensor> read, std::vector<Tensor> write,
                        std::vector<Tensor> scale_write, Tensor g_write, Tensor v_hat,
                        std::optional<Tensor> v_lo, Tensor v_scale, Tensor unions, Tensor num,
                        Tensor den, std::vector<int64_t> widths, int64_t dense_read,
                        int64_t dense_write, int64_t weight_digits, int64_t window) {
  check_arch_table();
  const int D = static_cast<int>(widths.size());
  STD_TORCH_CHECK(
      D >= 1 && static_cast<int>(read.size()) == D && static_cast<int>(write.size()) == D
          && static_cast<int>(scale_write.size()) == D,
      "each side one row set a level and the write side one scale a level, against the layout's ",
      D, " levels");
  STD_TORCH_CHECK(window == kWindow, "the intra's window is the carry's ", kWindow, " tokens, got ",
                  window);
  STD_TORCH_CHECK(weight_digits == 1 || weight_digits == 2,
                  "the value product's weights are one or two u8 digits, got ", weight_digits);
  STD_TORCH_CHECK(read[0].dim() == 4, "a level's rows are [B, T, H, width]");
  STD_TORCH_CHECK(read[0].scalar_type() == Dtype::Byte, "the bundle's rows are u8, got ",
                  read[0].scalar_type());
  const int64_t B = read[0].size(0), T = read[0].size(1), H = read[0].size(2);
  STD_TORCH_CHECK(T % kWindow == 0, "the call is a whole number of ", kWindow,
                  "-token windows, got ", T);
  std::vector<int64_t> row_widths(D);
  for (int l = 0; l < D; ++l) {
    row_widths[l] = read[l].dim() == 4 ? read[l].size(3) : 0;
    STD_TORCH_CHECK(row_widths[l] <= widths[l], "level ", l, "'s rows are ", row_widths[l],
                    " digits wide, past the layout's ", widths[l]);
  }
  const detail::Geometry g = detail::geometry(row_widths);

  IntraParams p{};
  facts::LivenessLayout layout{D, {}, static_cast<int>(T)};
  for (int l = 0; l < D; ++l) layout.B[l] = static_cast<int>(widths[l]);
  for (int l = 0; l < D; ++l) {
    p.read[l] = detail::row_view(read[l], B, T, H, row_widths[l], Dtype::Byte, "read[l]");
    p.write[l] = detail::row_view(write[l], B, T, H, row_widths[l], Dtype::Byte, "write[l]");
    p.scale_write[l] = detail::column_view(scale_write[l], B, T, H, "scale_write[l]");
    p.width[l] = static_cast<int>(row_widths[l]);
    p.slab0[l] = g.slab0[l];
    p.level_slabs[l] = g.level_slabs[l];
    p.level_stages[l] = g.level_stages[l];
    p.group0[l] = layout.group_base(l);
    for (int i = 0; i < g.level_stages[l]; ++i) p.stage[l][i] = g.stage[l][i];
  }
  p.stages = g.stages;
  p.slabs = g.slabs;
  p.groups = layout.groups();
  p.pitch = g.pitch;
  p.g_write = detail::column_view(g_write, B, T, H, "g_write");
  p.v_scale = detail::column_view(v_scale, B, T, H, "v_scale");
  p.v_hat = detail::row_view(v_hat, B, T, H, kValueWidth, Dtype::Char, "v_hat");
  p.v_lo = v_lo.has_value() ? detail::row_view(*v_lo, B, T, H, kValueWidth, Dtype::Byte, "v_lo")
                            : RowView{nullptr, 0, 0, 0};
  STD_TORCH_CHECK(!v_lo.has_value()
                      || (v_lo->stride(0) == v_hat.stride(0) && v_lo->stride(1) == v_hat.stride(1)
                          && v_lo->stride(2) == v_hat.stride(2)),
                  "v_lo is laid out as v_hat is (one copy address serves both planes)");
  p.H = static_cast<int>(H);
  p.L = static_cast<int>(T);
  p.windows = static_cast<int>(T / kWindow);
  p.weight_digits = static_cast<int>(weight_digits);
  STD_TORCH_CHECK(
      dense_read >= 0 && dense_read < (1 << D) && dense_write >= 0 && dense_write < (1 << D),
      "the dense-level bitmasks name levels inside the depth");
  p.dense[0] = static_cast<uint32_t>(dense_read);
  p.dense[1] = static_cast<uint32_t>(dense_write);
  STD_TORCH_CHECK(unions.is_cuda() && unions.scalar_type() == Dtype::Int && unions.is_contiguous()
                      && unions.dim() == 4 && unions.size(0) == B * H
                      && unions.size(1) == facts::kSides && unions.size(2) == layout.groups()
                      && unions.size(3) == layout.tiles(),
                  "unions is the facts pass's [", B * H, ", 2, ", layout.groups(), ", ",
                  layout.tiles(), "] int32 tile-union table");
  p.unions = reinterpret_cast<const uint32_t*>(unions.const_data_ptr<int32_t>());
  STD_TORCH_CHECK(num.is_cuda() && num.scalar_type() == Dtype::Float && num.is_contiguous()
                      && num.numel() == B * H * T * kValueWidth,
                  "num is the contiguous fp32 [BH, L, ", kValueWidth, "] output plane");
  STD_TORCH_CHECK(den.is_cuda() && den.scalar_type() == Dtype::Float && den.is_contiguous()
                      && den.numel() == B * H * T,
                  "den is the contiguous fp32 [BH, L] output plane");
  p.num = num.mutable_data_ptr<float>();
  p.den = den.mutable_data_ptr<float>();
  p.ledger = detail::g_ledger;

  const int device = read[0].get_device_index();
  const DeviceGuard guard(device);
  const cudaStream_t stream = current_stream();
  if (D == 1)
    detail::launch<1>(p, g.smem, B * H, stream);
  else
    detail::launch<2>(p, g.smem, B * H, stream);
}

//: -- see docs/internals/intra/intra_api.md#intra-footprint
int64_t intra_footprint(std::vector<int64_t> widths) { return detail::geometry(widths).smem; }

//: -- see docs/internals/intra/intra_api.md#intra-build-stamp
int64_t intra_build_stamp() {
  auto out = empty_cuda({1}, Dtype::Long);
  stampdet::stamp_kernel<<<1, 1, 0, current_stream()>>>(out.mutable_data_ptr<int64_t>());
  ROLA_CUDA_LAUNCH_CHECK();
  return item<int64_t>(out);
}

}  // namespace rola::intra
