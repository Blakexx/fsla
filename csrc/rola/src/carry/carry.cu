// csrc/rola/src/carry/carry.cu -- THE CARRY DISPATCH: the kernel boundary's refusals, the
// addressing block's one derivation, and the arm the launch resolves to.
//
// This translation unit holds no kernel body of its own beyond the build stamp: an arm's
// body lives in its generated translation unit, so an arm-subset build is a file subset.
// See docs/internals/carry/carry.md
#include "carry/carry_api.cuh"

#include <c10/cuda/CUDAStream.h>

#include "carry/box.cuh"
#include "carry/carry_arm_abi.cuh"
#include "carry/params.cuh"
#include "common/arch_runtime.cuh"
#include "common/arm_switch.cuh"
#include "common/design.cuh"
#include "common/geom.cuh"

namespace rola::carry {

ROLA_CARRY_ARM_SET_X(ROLA_CARRY_ARM_DECLARE)

namespace {

//: THE PAGE RECTANGLE the descriptor may name -- one law, shared with decode.
constexpr int64_t kPageBits = kAtomLeavesBits;

//: THE KERNEL BOUNDARY'S SHAPE LAW (KERNEL_STANDARDS.md §R13): the strict descriptor and
//: nothing else. The seam above pads; this refuses. -- see docs/internals/carry/carry.md#refusals
void refuse_shape(const std::vector<int64_t>& widths, int64_t dv, int64_t page_bits,
                  int64_t warps_per_cta) {
  TORCH_CHECK(page_bits == kPageBits, "the page rectangle is the trailing ", kPageBits,
              " canonical leaf bits; this call names ", page_bits);
  TORCH_CHECK(!widths.empty() && (int)widths.size() <= kGeomMaxLevels,
              "the addressing block is built to depth ", kGeomMaxLevels, "; this call names ",
              widths.size(), " levels");
  for (size_t l = 0; l < widths.size(); ++l)
    TORCH_CHECK(widths[l] >= kAtomLeaves && (widths[l] & (widths[l] - 1)) == 0, "level ", l, " is ",
                widths[l],
                " digits wide. THE KERNEL DOES NOT PAD: B_l is a power of two at or above ",
                kAtomLeaves, ".");
  TORCH_CHECK(dv > 0 && dv % 16 == 0 && (dv & (dv - 1)) == 0,
              "DV is a power of two at or above sixteen; this call names ", dv);
  TORCH_CHECK(is_warps_per_cta((int)warps_per_cta), "warps_per_cta=", warps_per_cta,
              " is not a launch shape this family builds");
}

void refuse_operand(const char* name, const at::Tensor& t, at::ScalarType dtype,
                    at::IntArrayRef want) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous(), name, " is a contiguous CUDA tensor");
  TORCH_CHECK(t.scalar_type() == dtype, name, " is ", dtype, ", got ", t.scalar_type());
  TORCH_CHECK(t.sizes() == want, name, " is ", want, ", got ", t.sizes());
}

}  // namespace

void carry_forward(const at::Tensor& read, const at::Tensor& write, const at::Tensor& gain,
                   const at::Tensor& v, at::Tensor& num, at::Tensor& den,
                   const std::vector<int64_t>& widths, int64_t dv, int64_t page_bits,
                   int64_t warps_per_cta, const at::Tensor& liveness, const at::Tensor& activity,
                   const c10::optional<at::Tensor>& state_in,
                   const c10::optional<at::Tensor>& state_out,
                   const c10::optional<at::Tensor>& page_table) {
  check_arch_table();
  refuse_shape(widths, dv, page_bits, warps_per_cta);

  const int D = (int)widths.size();
  int64_t leaves = 1, wtot = 0;
  for (int l = 0; l < D; ++l) {
    leaves *= widths[l];
    wtot += widths[l];
  }
  const int64_t BH = read.size(0), L = read.size(1);
  const int64_t pages = leaves / kAtomLeaves;
  const int64_t words = (L + 31) / 32;

  refuse_operand("read", read, at::kBFloat16, {BH, L, wtot});
  refuse_operand("write", write, at::kBFloat16, {BH, L, wtot});
  refuse_operand("gain", gain, at::kBFloat16, {BH, L});
  refuse_operand("v", v, at::kBFloat16, {BH, L, dv});
  refuse_operand("num", num, at::kFloat, {BH, L, dv});
  refuse_operand("den", den, at::kFloat, {BH, L});
  refuse_operand("liveness", liveness, at::kInt, {BH, 2, wtot, words});
  refuse_operand("activity", activity, at::kByte, {BH, pages});
  if (page_table) refuse_operand("page_table", *page_table, at::kInt, {BH, pages});
  const std::vector<int64_t> plane{BH, pages, (int64_t)kAtomLeaves, dv + 1};
  if (state_in) refuse_operand("state_in", *state_in, at::kFloat, plane);
  if (state_out) refuse_operand("state_out", *state_out, at::kFloat, plane);
  TORCH_CHECK(!state_in || !state_out || state_in->data_ptr() == state_out->data_ptr(),
              "a carried state is advanced IN PLACE: the exit sweep stores only the pages this "
              "call writes, so two planes would silently drop the rest");
  TORCH_CHECK(L <= 0x7FFFFFFF && BH * L * wtot < 0x40000000,
              "the packed routing plane's byte offsets are 32-bit displacements off one base");

  const int BC = box_leaves((int)dv, (int)warps_per_cta);
  CarryParams p{};
  int w[kGeomMaxLevels] = {0, 0, 0, 0};
  for (int l = 0; l < D; ++l) w[l] = (int)widths[l];
  //: THE CARVE ORDER is a runtime input; this surface carries no mode word, so every
  //: level is declared dense on both sides and the order is the canonical one.
  const char* err =
      derive_carry_geom(D, w, BC, (int)dv, 0u, (int)warps_per_cta, (int)warps_per_cta, p.g);
  TORCH_CHECK(err == nullptr, "the carry addressing block refuses this call: ", err);

  p.pread = reinterpret_cast<const __nv_bfloat16*>(read.data_ptr());
  p.pwrite = reinterpret_cast<const __nv_bfloat16*>(write.data_ptr());
  p.gwrite = reinterpret_cast<const __nv_bfloat16*>(gain.data_ptr());
  p.v = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr());
  p.num = num.data_ptr<float>();
  p.den = den.data_ptr<float>();
  p.state_in = state_in ? state_in->data_ptr<float>() : nullptr;
  p.state_out = state_out ? state_out->data_ptr<float>() : nullptr;
  p.page_table = page_table ? page_table->data_ptr<int32_t>() : nullptr;
  p.liveness = liveness.data_ptr<int32_t>();
  p.activity = activity.data_ptr<uint8_t>();
  p.L = (int)L;
  p.liveness_words = (int)words;
  p.readout_form = (int)Form::kUnion;
  p.fold_form = (int)Form::kUnion;

  const auto stream = at::cuda::getCurrentCUDAStream();
  cudaError_t status = cudaSuccess;
  bool launched = false;
  arm_switch<CarryArmSet>(D, (int)dv, (int)warps_per_cta, [&](auto A) {
    using Arm = decltype(A);
#define ROLA_CARRY_ARM_CALL(I_, D_, DV_, W_)                                          \
  if (Arm::index == (I_)) {                                                            \
    launched = true;                                                                   \
    status = carry_arm_forward_##I_(p, p.g.owners, (int)BH, stream);                   \
  }
    ROLA_CARRY_ARM_SET_X(ROLA_CARRY_ARM_CALL)
#undef ROLA_CARRY_ARM_CALL
  });
  TORCH_CHECK(launched, "the carry arm set resolved a member that carries no launch");
  TORCH_CHECK(status == cudaSuccess, "the carry launch failed: ", cudaGetErrorString(status));
}

namespace stamp {

//: THE DEVICE-SIDE FACT: the only answer a STALE binary cannot produce.
__global__ void carry_stamp_kernel(int* out) {
  out[0] = kAtomLeaves * 1000 + warps_per_sm;
  out[1] = denseref::design::kCapsDigest;
  out[2] = state_elems_per_sm(denseref::arch::kCaps);
  out[3] = kClusterCtas;
}

}  // namespace stamp

std::vector<int64_t> carry_build_stamp() {
  check_arch_table();
  auto buf = at::empty({4}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
  stamp::carry_stamp_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(buf.data_ptr<int32_t>());
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "the carry build stamp did not run");
  const auto host = buf.to(at::kCPU);
  std::vector<int64_t> out;
  for (int i = 0; i < 4; ++i) out.push_back((int64_t)host.data_ptr<int32_t>()[i]);
#define ROLA_CARRY_ARM_STAMP(I_, D_, DV_, W_)                                          \
  {                                                                                     \
    const ArmCensus c = carry_arm_census_##I_();                                        \
    for (int x : {(int)(I_), (int)(D_), (int)(DV_), (int)(W_), c.regs, c.local_bytes,   \
                  c.smem_bytes, c.max_threads, c.binary_version, c.threads,             \
                  c.box_leaves, c.atoms, c.acc_regs, c.state_elems,                     \
                  c.state_regs_per_lane, c.segment_slots, c.members})                   \
      out.push_back((int64_t)x);                                                        \
  }
  ROLA_CARRY_ARM_SET_X(ROLA_CARRY_ARM_STAMP)
#undef ROLA_CARRY_ARM_STAMP
  return out;
}

std::vector<std::vector<int64_t>> carry_arms() {
  std::vector<std::vector<int64_t>> out;
#define ROLA_CARRY_ARM_ROW(I_, D_, DV_, W_) out.push_back({(int64_t)(D_), (int64_t)(DV_), (int64_t)(W_)});
  ROLA_CARRY_ARM_SET_X(ROLA_CARRY_ARM_ROW)
#undef ROLA_CARRY_ARM_ROW
  return out;
}

}  // namespace rola::carry
