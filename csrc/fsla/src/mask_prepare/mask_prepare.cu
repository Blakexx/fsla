// THE PREFILL'S MASK-PREPARATION PASS -- hand CUDA, global loads and stores alone.
// Design of record: docs/internals/mask_prepare/mask_prepare.md.
// Gate: `tests/oracle/test_mask_prepare.py` (the host's words, counts and totals, exactly).
//
// One launch over every level-side that has a mask. A CTA finds its level-side, derives the
// side's group from its density byte with the carve's one rule, and switches once to the body
// compiled for that group: each group's word (its digits' words ORed), the count of the
// group's live tokens of the window before each word, and the window's total.

#include <algorithm>
#include <iterator>

#include "common/torch_seam.cuh"

#include "common/arch_runtime.cuh"
#include "common/select.cuh"
#include "common/structure_switch.cuh"
#include "mask_prepare.cuh"
#include "plan/environment.h"
#include "plan/mask_prepare/contract.h"
#include "plan/mask_prepare_layout.h"
#include "plan/prefill_launch.h"

namespace fsla {
namespace mask_prepare {
namespace detail {

constexpr int kThreads = 1024;
constexpr int kWords = plan::kMaskWindowWords;
constexpr int kSides = plan::kLevelSides;
constexpr int kGroupSizes = plan::kGroupSizes;
static_assert(plan::kHopperWindow == kWords * 32, "a window is 32 mask words of 32 tokens");

//: THE LAUNCH'S FACTS of each level-side `s`: its mask (null where it has none), where its blocks start in the workspace, its first CTA (`first_cta[kSides]` the grid), its width and its batch-heads
struct Args {
  const std::uint32_t* mask[kSides];
  std::int64_t out[kSides];
  int first_cta[kSides + 1];
  int width_bits[kSides];
  int heads[kSides];
  int words;
  int windows;
  int levels;
};

//: the index of a group's bits among `plan::kGroupSizeBits`
__host__ __device__ constexpr int group_index(int bits) { return bits < 4 ? bits / 2 : bits - 2; }

constexpr bool group_index_inverts_the_sizes() {
  for (int i = 0; i < kGroupSizes; ++i) {
    if (group_index(plan::group_size_bits(i)) != i) return false;
  }
  return true;
}

static_assert(group_index_inverts_the_sizes(), "group_index inverts group_size_bits");

//: A THREAD'S DIGITS OF ONE WORD AS LOADED: 16 bytes a load where it holds 4 digits or more
template <int U>
struct Raw {
  uint4 line[U / 4];
};

template <>
struct Raw<1> {
  std::uint32_t word;
};

template <int U>
__device__ __forceinline__ void load(Raw<U>& raw, const std::uint32_t* at) {
  if constexpr (U == 1) {
    raw.word = at[0];
  } else {
    constexpr int kLines = U / 4;
    const uint4* const lines = reinterpret_cast<const uint4*>(at);
#pragma unroll
    for (int i = 0; i < kLines; ++i) raw.line[i] = lines[i];
  }
}

//: a token's bit set where it is live on any of the thread's digits
template <int U>
__device__ __forceinline__ std::uint32_t word_of(const Raw<U>& raw) {
  if constexpr (U == 1) {
    return raw.word;
  } else {
    constexpr int kLines = U / 4;
    std::uint32_t w = 0;
#pragma unroll
    for (int i = 0; i < kLines; ++i) {
      w |= raw.line[i].x | raw.line[i].y | raw.line[i].z | raw.line[i].w;
    }
    return w;
  }
}

//: A WORD FINISHED -- see docs/internals/mask_prepare/mask_prepare.md#finish
template <int U, int LANES>
__device__ __forceinline__ void finish(const Raw<U>& raw, std::uint32_t keep, std::uint32_t* word,
                                       std::uint16_t* count, std::uint32_t& before) {
  std::uint32_t w = word_of<U>(raw) & keep;
#pragma unroll
  for (int d = 1; d < LANES; d <<= 1) w |= __shfl_xor_sync(0xffffffffu, w, d);
  if constexpr (U * LANES > 1) word[0] = w;
  count[0] = static_cast<std::uint16_t>(before);
  before += static_cast<std::uint32_t>(__popc(w));
}

//: A LEVEL-SIDE AT A GROUP OF `1 << GB` DIGITS -- see docs/internals/mask_prepare/mask_prepare.md#the-walk
template <int GB>
__device__ __forceinline__ void prepare(const Args& a, std::uint8_t* workspace, int s, int cta) {
  constexpr int kUnitBits = GB < 4 ? GB : 4, kUnit = 1 << kUnitBits, kLanes = 1 << (GB - kUnitBits);
  const int wb = a.width_bits[s], ub = wb - kUnitBits, gb = wb - GB;
  const std::int64_t windows = std::int64_t{a.heads[s]} * a.windows;
  const std::int64_t unit = std::int64_t{cta} * kThreads + threadIdx.x;
  if ((std::int64_t{cta} * kThreads) >> ub >= windows) return;

  //: a thread past the level-side's last window walks its first: it stores what that window's own thread stores
  const std::int64_t at_window = (unit >> ub) < windows ? unit >> ub : 0;
  const int u = static_cast<int>(unit) & ((1 << ub) - 1), group = u >> (GB - kUnitBits);
  const int window = static_cast<int>(at_window % a.windows);
  const std::int64_t first_word = (at_window / a.windows) * a.words + std::int64_t{window} * kWords;
  const int have = a.words - window * kWords < kWords ? a.words - window * kWords : kWords;
  const std::int64_t stride = std::int64_t{1} << wb;
  const std::uint32_t* const at = a.mask[s] + (first_word << wb) + (u << kUnitBits);
  std::uint8_t* const block = workspace + a.out[s] + at_window * plan::prepared_block_bytes(wb);
  std::uint16_t* const totals = reinterpret_cast<std::uint16_t*>(block);

  const int groups = 1 << gb;
  std::uint8_t* const counts = block + plan::prepared_counts_at(groups);
  std::uint8_t* const words = block + plan::prepared_words_at(groups);
  std::uint16_t* count = reinterpret_cast<std::uint16_t*>(counts) + group;
  std::uint32_t* word = reinterpret_cast<std::uint32_t*>(words) + group;

  //: THE WALK'S LOADS HELD, issued before any is read: 4 words' at one digit a thread, 2 words' at more
  constexpr int kHeld = kUnit == 1 ? 4 : 2;
  Raw<kUnit> held[kHeld];
  const std::uint32_t* row = at;
#pragma unroll
  for (int i = 0; i < kHeld; ++i) load<kUnit>(held[i], i < have ? row + i * stride : at);
  std::uint32_t before = 0;
#pragma unroll 1
  for (int j = 0; j < kWords; j += kHeld) {
#pragma unroll
    for (int i = 0; i < kHeld; ++i) {
      //: a word finished, and the load `kHeld` words on issued into the registers it left; a word past the mask's last reads as zero
      const std::uint32_t keep = j + i < have ? ~0u : 0u;
      finish<kUnit, kLanes>(held[i], keep, word + i * groups, count + i * groups, before);
      load<kUnit>(held[i], j + i + kHeld < have ? row + (i + kHeld) * stride : at);
    }
    row += kHeld * stride;
    word += kHeld * groups;
    count += kHeld * groups;
  }
  totals[group] = static_cast<std::uint16_t>(before);
}

//: ONE GRID OVER THE CALL'S MASKED LEVEL-SIDES -- see docs/internals/mask_prepare/mask_prepare.md#the-kernel
template <class Bt>
__global__ __launch_bounds__(kThreads, 1) void mask_prepare_kernel(const __grid_constant__ Args a,
                                                                   const std::uint8_t* density,
                                                                   std::uint8_t* workspace) {
  const int cta = static_cast<int>(blockIdx.x);
  int s = 0;
#pragma unroll
  for (int i = 1; i < kSides; ++i) s += cta >= a.first_cta[i] ? 1 : 0;
  const int side = s / plan::kCarveMaxLevels, level = s % plan::kCarveMaxLevels;
  //: a byte `b` is `b + 1` 256ths; at 256 the level-side is dense and its mask is never read
  const int dense = static_cast<int>(density[side * a.levels + level]) + 1;
  if (dense == plan::kWholeDensity) return;
  const int bits = plan::prefill_group_bits(dense, a.width_bits[s]);
  uniform_switch<kGroupSizes>(group_index(bits), [&](auto member) {
    prepare<plan::kGroupSizeBits[decltype(member)::value]>(a, workspace, s, cta - a.first_cta[s]);
  });
}

//: the grid a level-side needs at a group of one digit, which is what the host can know: a thread a (window, digit)
std::int64_t ctas_of(std::int64_t heads, std::int64_t windows, int width_bits) {
  return ((heads * windows << width_bits) + kThreads - 1) / kThreads;
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

  Args args{};
  args.levels = static_cast<int>(density.size(1));
  args.words = static_cast<int>(masks[0].size(2));
  args.windows = (args.words + kWords - 1) / kWords;
  std::int64_t heads[kSides] = {};
  std::int64_t widest = 0;
  for (std::size_t i = 0; i < masks.size(); ++i) {
    const Tensor& mask = masks[i];
    const int s = static_cast<int>(slots[i]);
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

  const plan::PreparedLayout layout = plan::prepared_layout(args.width_bits, heads, args.windows);
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

  const int device = density.get_device_index();
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

//: the bytes `mask_prepare` writes for a call's level-sides: `width_bits[s]` and `heads[s]` of each of the 8, 0 heads where it has no mask, of masks `words` long
int64_t mask_prepare_bytes(const std::vector<int64_t>& width_bits,
                           const std::vector<int64_t>& heads, int64_t words) {
  STD_TORCH_CHECK(width_bits.size() == detail::kSides && heads.size() == detail::kSides,
                  "mask_prepare_bytes takes the 8 level-sides' width bits and batch-heads");
  int bits[detail::kSides];
  std::int64_t each[detail::kSides];
  for (int s = 0; s < detail::kSides; ++s) {
    bits[s] = static_cast<int>(width_bits[s]);
    each[s] = heads[s];
  }
  return plan::prepared_layout(bits, each, (words + detail::kWords - 1) / detail::kWords).bytes;
}

}  // namespace mask_prepare
}  // namespace fsla
