// THE PREFILL'S MASK-PREPARATION PASS -- hand CUDA, global loads and stores alone.
// Design of record: docs/internals/mask_prepare/mask_prepare.md.
// Gate: `tests/oracle/test_mask_prepare.py` (the host's words and totals, exactly).
//
// The op's host half: its operands checked, its grid and its place in the workspace, the launch.
// The device half -- the kernel, the walk and the switch on the group -- is mask_prepare_kernel.cuh.

#include <algorithm>

#include "common/torch_seam.cuh"

#include "common/arch_runtime.cuh"
#include "common/select.cuh"
#include "mask_prepare.cuh"
#include "mask_prepare_kernel.cuh"
#include "plan/environment.h"
#include "plan/mask_prepare/contract.h"
#include "plan/mask_prepare_layout.h"
#include "plan/prefill_launch.h"

namespace fsla {
namespace mask_prepare {
namespace detail {

//: the card `device`'s window in mask words (`plan::window_of_cc`); a card with none is refused by name
int window_words_of(int device) {
  const int words = plan::window_of_cc(card_of(device).cc) / 32;
  STD_TORCH_CHECK(words > 0, "mask_prepare has no window for this card");
  return words;
}

//: an operand the kernel reads as a contiguous array of `dtype` on the call's device
void check_operand(const Tensor& t, Dtype dtype, int device, const char* name) {
  STD_TORCH_CHECK(t.is_cuda() && t.get_device_index() == device, name, " is on the density's card");
  STD_TORCH_CHECK(t.scalar_type() == dtype, name, " has the wrong dtype, got ", t.scalar_type());
  STD_TORCH_CHECK(t.is_contiguous(), name, " is contiguous");
}

template <class Bt>
void launch_at(const Args& args, const Tensor& density, Tensor& workspace) {
  auto stream = current_stream();
  mask_prepare_kernel<Bt><<<args.first_cta[kSides], kThreads, 0, stream>>>(
      args, static_cast<const std::uint8_t*>(density.data_ptr()),
      static_cast<std::uint8_t*>(workspace.data_ptr()));
  FSLA_CUDA_LAUNCH_CHECK();
}

}  // namespace detail

void mask_prepare(const std::vector<Tensor>& masks, const std::vector<int64_t>& slots,
                  const Tensor& density, Tensor& workspace) {
  check_arch_table();
  using namespace detail;
  STD_TORCH_CHECK(masks.size() == slots.size() && !masks.empty() && masks.size() <= kSides,
                  "mask_prepare takes one slot (side * 4 + level) a mask, 1 to 8 masks");
  STD_TORCH_CHECK(
      density.dim() == 2 && density.size(0) == 2 && density.size(1) <= plan::kCarveMaxLevels,
      "mask_prepare's density is uint8 [2, D], D at most 4");
  STD_TORCH_CHECK(density.is_cuda(), "mask_prepare's density is on a card");
  const int device = density.get_device_index();
  check_operand(density, Dtype::Byte, device, "mask_prepare's density");
  check_operand(workspace, Dtype::Byte, device, "mask_prepare's workspace");

  Args args{};
  args.levels = static_cast<int>(density.size(1));
  args.words = static_cast<int>(masks[0].size(2));
  const int window_words = window_words_of(device);
  args.windows = (args.words + window_words - 1) / window_words;
  std::int64_t heads[kSides] = {};
  std::int64_t widest = 0;
  for (std::size_t i = 0; i < masks.size(); ++i) {
    const Tensor& mask = masks[i];
    const int s = static_cast<int>(slots[i]);
    check_operand(mask, Dtype::Int, device, "a mask");
    STD_TORCH_CHECK(
        s >= 0 && s < kSides && s % plan::kCarveMaxLevels < args.levels && args.mask[s] == nullptr,
        "mask_prepare's slots are distinct level-sides of the density's depth");
    STD_TORCH_CHECK(
        mask.dim() == 4 && mask.size(2) == args.words && mask.size(3) >= 2
            && (mask.size(3) & (mask.size(3) - 1)) == 0 && mask.size(3) <= 256,
        "a mask is int32 [B, H, ceil(T / 32), width], the width a power of two from 2 to 256");
    //: a group's digits are loaded 16 bytes at once
    STD_TORCH_CHECK(reinterpret_cast<std::uintptr_t>(mask.data_ptr()) % 16 == 0 || mask.size(3) < 4,
                    "a mask's memory is 16-byte aligned");
    args.mask[s] = static_cast<const std::uint32_t*>(mask.data_ptr());
    while ((std::int64_t{1} << args.width_bits[s]) < mask.size(3)) ++args.width_bits[s];
    widest = std::max<std::int64_t>(widest, mask.size(3));
    heads[s] = mask.size(0) * mask.size(1);
    args.heads[s] = static_cast<int>(heads[s]);
  }

  const plan::PreparedLayout layout =
      plan::prepared_layout(args.width_bits, heads, args.windows, window_words);
  STD_TORCH_CHECK(
      workspace.numel() * static_cast<std::int64_t>(workspace.element_size()) >= layout.bytes,
      "mask_prepare's workspace holds the prepared masks' layout");
  std::int64_t ctas = 0;
  for (int s = 0; s < kSides; ++s) {
    args.out[s] = layout.offset[s];
    args.first_cta[s] = static_cast<int>(ctas);
    ctas += ctas_of(heads[s], args.windows, args.width_bits[s]);
  }
  STD_TORCH_CHECK(ctas <= INT32_MAX, "mask_prepare's grid is at most 2^31 - 1 CTAs");
  args.first_cta[kSides] = static_cast<int>(ctas);

  const plan::Named call = {{"width", widest}, {"levels", args.levels}};
  plan::descriptor_of<plan::mask_prepare::Kernel>(device, call);
  const DeviceGuard guard(device);
  build_switch(card_of(device), [&](auto b) {
    constexpr ::denseref::arch::Build kB = decltype(b)::build;
    if constexpr (in_image(kB)) {
      launch_at<::denseref::arch::BuildTag<kB>>(args, density, workspace);
    }
  });
}

//: the bytes `mask_prepare` writes for a call's level-sides: `width_bits[s]` and `heads[s]` of each of the 8, 0 heads where it has no mask, of masks `words` long, on the card `device`
int64_t mask_prepare_bytes(const std::vector<int64_t>& width_bits,
                           const std::vector<int64_t>& heads, int64_t words, int64_t device) {
  STD_TORCH_CHECK(width_bits.size() == detail::kSides && heads.size() == detail::kSides,
                  "mask_prepare_bytes takes the 8 level-sides' width bits and batch-heads");
  const int window_words = detail::window_words_of(static_cast<int>(device));
  int bits[detail::kSides];
  std::int64_t each[detail::kSides];
  for (int s = 0; s < detail::kSides; ++s) {
    bits[s] = static_cast<int>(width_bits[s]);
    each[s] = heads[s];
  }
  const std::int64_t windows = (words + window_words - 1) / window_words;
  return plan::prepared_layout(bits, each, windows, window_words).bytes;
}

}  // namespace mask_prepare
}  // namespace fsla
