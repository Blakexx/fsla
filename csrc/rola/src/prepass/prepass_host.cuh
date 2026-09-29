// csrc/rola/src/prepass/prepass_host.cuh -- THE CALL'S DERIVATION: the pre-pass boundary's refusals and
// its parameter block, one derivation shared by the launch and by the part harness.
// See docs/internals/prepass/prepass.md#call
#pragma once

#include "common/torch_seam.cuh"

#include <cstdint>
#include <optional>
#include <vector>

#include "common/arch_runtime.cuh"
#include "common/geom.cuh"
#include "common/geom_api.cuh"
#include "prepass/params.cuh"
#include "prepass/plan.cuh"

namespace rola::prepass {

namespace host {

inline void refuse_operand(const char* name, const Tensor& t, Dtype dtype, Shape want) {
  STD_TORCH_CHECK(t.is_cuda() && t.is_contiguous(), name, " is a contiguous CUDA tensor");
  STD_TORCH_CHECK(t.scalar_type() == dtype, name, " is ", dtype, ", got ", t.scalar_type());
  STD_TORCH_CHECK(t.sizes() == want, name, " is ", shape(want), ", got ", shape(t.sizes()));
}

//: THE MODE WORD's two bits a level, as the seam spells them: bit 0 read-sparse, bit 1 write-sparse.
inline uint32_t sparse_mask_of_modes(int D, uint32_t modes, int side) {
  uint32_t m = 0u;
  for (int l = 0; l < D; ++l)
    if (carry::mode_declares(modes, l, side == 0 ? carry::kSideRead : carry::kSideWrite))
      m |= 1u << l;
  return m;
}

}  // namespace host

struct PrepassCall {
  PrepassParams p;
  int64_t BH;
};

//: THE CALL DERIVED: every refusal applied, the block filled but for the ledger.
inline PrepassCall derive_prepass_call(const Tensor& rows_read, const Tensor& rows_write,
                                       const Tensor& scale_read, const Tensor& scale_write,
                                       const Tensor& gain, const Tensor& vscale, Tensor& den,
                                       Tensor& scale, const std::vector<int64_t>& widths,
                                       int64_t level_modes, const Tensor& liveness,
                                       const std::optional<Tensor>& fold_dump) {
  check_arch_table();
  const int D = (int)widths.size();
  STD_TORCH_CHECK(D >= 1 && D <= carry::kGeomMaxLevels, "the addressing block is built to depth ",
                  carry::kGeomMaxLevels, "; this call names ", D, " levels");
  int64_t wtot = 0;
  for (int l = 0; l < D; ++l) {
    STD_TORCH_CHECK(widths[l] >= carry::kAtomLeaves && (widths[l] & (widths[l] - 1)) == 0, "level ",
                    l, " is ", widths[l], " digits wide; B_l is a power of two at or above ",
                    carry::kAtomLeaves);
    wtot += widths[l];
  }
  const int64_t BH = rows_read.size(0), L = rows_read.size(1);
  const int64_t words = (L + 127) / 128 * 4;  //: whole 16-byte copies a staged row
  host::refuse_operand("rows_read", rows_read, Dtype::Byte, {BH, L, wtot});
  host::refuse_operand("rows_write", rows_write, Dtype::Byte, {BH, L, wtot});
  host::refuse_operand("scale_read", scale_read, Dtype::Float, {BH, L, (int64_t)D});
  host::refuse_operand("scale_write", scale_write, Dtype::Float, {BH, L, (int64_t)D});
  host::refuse_operand("gain", gain, Dtype::Float, {BH, L});
  host::refuse_operand("vscale", vscale, Dtype::Float, {BH, L});
  host::refuse_operand("den", den, Dtype::Float, {BH, L});
  host::refuse_operand("scale", scale, Dtype::Float, {BH, L});
  host::refuse_operand("liveness", liveness, Dtype::Int, {BH, 2, wtot, words});
  STD_TORCH_CHECK(L <= 0x7FFFFFFF && BH * L * wtot < 0x40000000,
                  "the packed row planes' byte offsets are 32-bit displacements off one base");

  PrepassParams p{};
  p.g = carry::tall_block(widths, level_modes);
  p.sparse_mask[0] = host::sparse_mask_of_modes(D, (uint32_t)level_modes, 0);
  p.sparse_mask[1] = host::sparse_mask_of_modes(D, (uint32_t)level_modes, 1);
  p.rows_read = reinterpret_cast<const uint8_t*>(rows_read.data_ptr());
  p.rows_write = reinterpret_cast<const uint8_t*>(rows_write.data_ptr());
  p.scale_read = scale_read.const_data_ptr<float>();
  p.scale_write = scale_write.const_data_ptr<float>();
  p.gain = gain.const_data_ptr<float>();
  p.vscale = vscale.const_data_ptr<float>();
  p.den = den.mutable_data_ptr<float>();
  p.scale = scale.mutable_data_ptr<float>();
  p.liveness = liveness.const_data_ptr<int32_t>();
  p.fold_dump = nullptr;
  if (fold_dump) {
    int64_t N = 1;
    for (const int64_t w : widths) N *= w;
    host::refuse_operand("fold_dump", *fold_dump, Dtype::Float, {BH, N, 2});
    p.fold_dump = fold_dump->mutable_data_ptr<float>();
  }
  p.L = (int)L;
  p.liveness_words = (int)words;
  return PrepassCall{p, BH};
}

}  // namespace rola::prepass
