// THE PREFILL'S MASK-PREPARATION PASS -- hand CUDA: global loads, the levels' group words in shared memory behind CTA barriers, and global stores.
// Design of record: docs/internals/mask_prepare/mask_prepare.md.
// Gate: `tests/oracle/test_mask_prepare.py` (the host's leaf groups' words and counts, exactly).
//
// The op's host half: its operands checked, its grid and its place in the workspace, the launch.
// The device half -- the kernel, the walk, the switch on a level's group and the leaf groups' lines -- is mask_prepare_kernel.cuh.

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
  mask_prepare_kernel<Bt><<<args.first_cta[kSides], kThreads, Smem<Bt>::bytes, stream>>>(
      args, static_cast<const std::uint8_t*>(density.data_ptr()),
      static_cast<std::uint8_t*>(workspace.data_ptr()));
  FSLA_CUDA_LAUNCH_CHECK();
}

//: a level's digits as log2: a power of two from 2 to 256
int width_bits_of(std::int64_t width) {
  STD_TORCH_CHECK(width >= 2 && (width & (width - 1)) == 0 && width <= 256,
                  "a level's width is a power of two from 2 to 256");
  int bits = 0;
  while ((std::int64_t{1} << bits) < width) ++bits;
  return bits;
}

}  // namespace detail

void mask_prepare(const std::vector<Tensor>& masks, const std::vector<int64_t>& slots,
                  const std::vector<int64_t>& widths, const std::vector<int64_t>& heads,
                  int64_t tokens, const Tensor& density, Tensor& workspace) {
  check_arch_table();
  using namespace detail;
  STD_TORCH_CHECK(masks.size() == slots.size() && masks.size() <= kLevelSides,
                  "mask_prepare takes one slot (side * 4 + level) a mask, at most 8 masks");
  STD_TORCH_CHECK(
      density.dim() == 2 && density.size(0) == 2 && density.size(1) <= plan::kCarveMaxLevels,
      "mask_prepare's density is uint8 [2, D], D at most 4");
  STD_TORCH_CHECK(density.is_cuda(), "mask_prepare's density is on a card");
  const int device = density.get_device_index();
  check_operand(density, Dtype::Byte, device, "mask_prepare's density");
  check_operand(workspace, Dtype::Byte, device, "mask_prepare's workspace");
  //: a leaf group's line is stored 16 bytes at once
  STD_TORCH_CHECK(reinterpret_cast<std::uintptr_t>(workspace.data_ptr()) % 16 == 0,
                  "mask_prepare's workspace is 16-byte aligned");

  Args args{};
  args.levels = static_cast<int>(density.size(1));
  STD_TORCH_CHECK(static_cast<std::int64_t>(widths.size()) == density.size(1),
                  "mask_prepare takes the width of each level of the density's depth");
  std::int64_t widest = 0;
  for (int l = 0; l < args.levels; ++l) {
    args.width_bits[l] = width_bits_of(widths[static_cast<std::size_t>(l)]);
    widest = std::max<std::int64_t>(widest, widths[static_cast<std::size_t>(l)]);
  }
  STD_TORCH_CHECK(heads.size() == kSides && heads[0] >= 1 && heads[1] >= 1,
                  "mask_prepare takes each side's batch-heads, the read side's first");
  STD_TORCH_CHECK(tokens >= 1, "mask_prepare takes a call of a token or more");
  args.tokens = tokens;
  args.words = static_cast<int>((tokens + 31) / 32);
  const int window_words = window_words_of(device);
  args.windows = (args.words + window_words - 1) / window_words;
  int masked[kSides] = {};
  std::int64_t each[kSides] = {heads[0], heads[1]};
  for (std::size_t i = 0; i < masks.size(); ++i) {
    const Tensor& mask = masks[i];
    const int s = static_cast<int>(slots[i]);
    check_operand(mask, Dtype::Int, device, "a mask");
    STD_TORCH_CHECK(
        s >= 0 && s < kLevelSides && s % kLevels < args.levels && args.mask[s] == nullptr,
        "mask_prepare's slots are distinct level-sides of the density's depth");
    const int side = s / kLevels, level = s % kLevels;
    STD_TORCH_CHECK(mask.dim() == 4 && mask.size(2) == args.words
                        && mask.size(3) == widths[static_cast<std::size_t>(level)],
                    "a mask is int32 [B, H, ceil(T / 32), width], the width its level's");
    STD_TORCH_CHECK(mask.size(0) * mask.size(1) == each[side],
                    "a mask's batch-heads are its side's");
    //: a group's digits are loaded 16 bytes at once
    STD_TORCH_CHECK(reinterpret_cast<std::uintptr_t>(mask.data_ptr()) % 16 == 0 || mask.size(3) < 4,
                    "a mask's memory is 16-byte aligned");
    args.mask[s] = static_cast<const std::uint32_t*>(mask.data_ptr());
    masked[side] |= 1 << level;
  }
  args.masked[0] = masked[0];
  args.masked[1] = masked[1];

  const plan::PreparedLayout layout =
      plan::prepared_layout(args.levels, args.width_bits, masked, each, args.windows, window_words);
  STD_TORCH_CHECK(
      workspace.numel() * static_cast<std::int64_t>(workspace.element_size()) >= layout.bytes,
      "mask_prepare's workspace holds the leaf groups' masks' layout");
  std::int64_t ctas = 0;
  for (int side = 0; side < kSides; ++side) {
    args.out[side] = layout.offset[side];
    args.heads[side] = static_cast<int>(each[side]);
    args.first_cta[side] = static_cast<int>(ctas);
    ctas += ctas_of(each[side], args.windows);
    STD_TORCH_CHECK(ctas <= INT32_MAX, "mask_prepare's grid is at most 2^31 - 1 CTAs");
  }
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

//: the bytes `mask_prepare` needs for a call: its levels' digits, each side's masked levels (a bit a level) and batch-heads, its tokens a head, on the card `device`
int64_t mask_prepare_bytes(const std::vector<int64_t>& widths, const std::vector<int64_t>& masked,
                           const std::vector<int64_t>& heads, int64_t tokens, int64_t device) {
  using namespace detail;
  STD_TORCH_CHECK(!widths.empty() && widths.size() <= plan::kCarveMaxLevels,
                  "mask_prepare_bytes takes the digits of 1 to 4 levels");
  STD_TORCH_CHECK(masked.size() == kSides && heads.size() == kSides,
                  "mask_prepare_bytes takes each side's masked levels and batch-heads");
  STD_TORCH_CHECK(tokens >= 1, "mask_prepare_bytes takes a call of a token or more");
  const int window_words = window_words_of(static_cast<int>(device));
  int bits[plan::kCarveMaxLevels] = {};
  for (std::size_t l = 0; l < widths.size(); ++l) bits[l] = width_bits_of(widths[l]);
  const int levels = static_cast<int>(widths.size());
  const int listed[kSides] = {static_cast<int>(masked[0]), static_cast<int>(masked[1])};
  const std::int64_t each[kSides] = {heads[0], heads[1]};
  const std::int64_t words = (tokens + 31) / 32;
  const std::int64_t windows = (words + window_words - 1) / window_words;
  return plan::prepared_layout(levels, bits, listed, each, windows, window_words).bytes;
}

}  // namespace mask_prepare
}  // namespace fsla
