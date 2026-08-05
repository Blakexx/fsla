// THE PRODUCER-EPILOGUE ROUTING SUMMARY PASS -- the kernel, its shared state, and the
// pybind entry that validates and launches it. `summaries.cuh` carries the parameter
// block, the declared bounds and what each output means.
//
// The kernel is FIVE STEPS over one `(bh, tile)` row, grid-strided: support words (with
// occupancy for free), per-level and per-slice union popcounts, the owner loop
// (`K_hat`, candidacy, histogram), §4.1's canonical per-leaf write bitmap with §4.2's
// owner condensation riding the same `atomicOr`, and the remaining global folds.
//
// Two facts a reader must not miss: CANDIDACY IS THE UNION OF THE TWO SIDES' PRODUCT
// SETS, never the candidacy of the union word; and `K_hat <= BC` STRUCTURALLY, which is
// why `bins = QPO + 1` is exhaustive and no clamp exists anywhere in this pass.
//
// Step by step, with the ballot's warp-uniformity requirement and the shared-byte
// aliasing between CUB's temp storage and the histogram: docs/internals/summaries.md

#include <cub/cub.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <vector>

#include "summaries.cuh"

namespace rola::summaries {
namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarps = kThreads / kWarpSize;

//: Bounded grid: `(bh, tile)` rows are walked with a GRID-STRIDE loop, so per-CTA
//: constant work is paid once per CTA and the grid does not grow with `L`. A launch
//: SHAPE, not a threshold on any data. docs/internals/summaries.md#launch-shape
constexpr int kMaxBlocks = 4096;

//: Ceilings the static shared-memory budget is sized from. `n_slices <= b_j`, and a
//: level wider than this has already failed `kMaxTotalDigits`.
constexpr int kMaxSlices = 512;
//: `QPO + 1` at `BC = 512`, and it is the ARRAY BOUND ONLY: every expression that uses
//: the bin count uses `p.bins = BC / kMmaKQuantum + 1` (invariant I1).
constexpr int kMaxBins = 33;

using BlockReduceU64 = cub::BlockReduce<unsigned long long, kThreads>;

//: THE SHARED-MEMORY LAYOUT, as one struct so the budget is a `sizeof` and not a sum
//: maintained by hand. `fold` (CUB's temp storage) and `hist` SHARE THE SAME BYTES --
//: disjoint phases separated by a barrier, said as a union rather than left to be
//: inferred from barrier placement: docs/internals/summaries.md#scratch-union
struct SharedState {
  int read_words[kMaxTotalDigits];
  int write_words[kMaxTotalDigits];
  int width[kMaxLevels];
  int row_offset[kMaxLevels];
  int leaf_stride[kMaxLevels];
  int fixed_radix[kMaxLevels];
  int level_pc[kMaxLevels];  //: popcount of the level's UNION digit set, this tile
  int slice_pc[kMaxSlices];  //: per boundary slice, the union digit count
  unsigned char slice_any_r[kMaxSlices];
  unsigned char slice_any_w[kMaxSlices];
  union {
    typename BlockReduceU64::TempStorage fold;
    unsigned long long hist[kMaxBins];
  } scratch;
};

// A CUB version bump that grew `TempStorage` past the histogram it shares bytes with,
// or past the SMEM a CTA may hold, must fail the COMPILE and not the launch.
static_assert(sizeof(SharedState) <= 48 * 1024,
              "the summary pass's shared state no longer fits a CTA's static shared "
              "memory; shrink kMaxTotalDigits or stage the digit rows in chunks");

//: Fold one per-thread partial into a global counter with ONE atomic per block.
//: BLOCK-WIDE: every thread must reach it, which is why it is only ever called from
//: block-uniform control flow.
__device__ __forceinline__ void fold_u64(SharedState& s, unsigned long long value,
                                         unsigned long long* out) {
  const unsigned long long total = BlockReduceU64(s.scratch.fold).Sum(value);
  if (threadIdx.x == 0 && total != 0ull) {
    atomicAdd(out, total);
  }
  //: HAZARD scratch-union -- docs/internals/summaries.md#scratch-union
  __syncthreads();
}

//: Is this owner WRITE-CANDIDATE in the staged tile? A §4 product-set membership test on
//: data the CTA already holds -- invariant I3's own criterion for an in-kernel decision
//: -- and EXACT rather than conservative.
__device__ __forceinline__ bool write_candidate(const SharedState& s, const SummaryParams& p,
                                                int owner) {
  const int slice = owner % p.n_slices;
  if (s.slice_any_w[slice] == 0) return false;
  const int rest = owner / p.n_slices;
  for (int l = 0; l < p.first_spanned_level; ++l) {
    const int digit = (rest / s.fixed_radix[l]) % s.width[l];
    if (s.write_words[s.row_offset[l] + digit] == 0) return false;
  }
  return true;
}

__global__ __launch_bounds__(kThreads) void summaries_kernel(SummaryParams p) {
  __shared__ SharedState s;

  for (int i = threadIdx.x; i < p.level_count; i += kThreads) {
    s.width[i] = p.level_width[i];
    s.row_offset[i] = p.level_row_offset[i];
    s.leaf_stride[i] = p.level_leaf_stride[i];
    s.fixed_radix[i] = p.level_fixed_radix[i];
  }
  __syncthreads();

  const int rows = p.batch_heads * p.n_tiles;
  const int j = p.first_spanned_level;
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;

  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const int bh = row / p.n_tiles;
    const int t0 = (row - bh * p.n_tiles) * p.token_tile;

    for (int r = threadIdx.x; r < p.total_digits; r += kThreads) {
      s.read_words[r] = 0;
      s.write_words[r] = 0;
    }
    __syncthreads();

    // ---- 1. THE SUPPORT WORDS, and occupancy for free --------------------
    // One `int32` per `(side, level, digit)`; bit `t` is "this digit is live at token
    // `t0 + t`". Threads cover consecutive DIGITS with the token loop inside, so each
    // read is coalesced; the padded tail (`t0 + t >= T`) contributes no bit, which is
    // the whole of what a ragged `L` needs. §4.2's `occupancy` counts live PAIRS, so it
    // is exactly `__popc` of the word -- free here.
    // docs/internals/summaries.md#support-words
    for (int l = 0; l < p.level_count; ++l) {
      const int width = s.width[l];
      const int base = s.row_offset[l];
      const float* read_level = p.read_amps[l];
      const float* write_level = p.write_amps[l];
      unsigned long long occ_r = 0ull;
      unsigned long long occ_w = 0ull;
      for (int d = threadIdx.x; d < width; d += kThreads) {
        int rw = 0;
        int ww = 0;
        for (int t = 0; t < p.token_tile; ++t) {
          const int token = t0 + t;
          if (token >= p.token_count) break;
          const long long index = (static_cast<long long>(bh) * p.token_count + token) * width + d;
          rw |= (read_level[index] != 0.0f) ? (1 << t) : 0;
          ww |= (write_level[index] != 0.0f) ? (1 << t) : 0;
        }
        s.read_words[base + d] = rw;
        s.write_words[base + d] = ww;
        occ_r += static_cast<unsigned long long>(__popc(rw));
        occ_w += static_cast<unsigned long long>(__popc(ww));
      }
      __syncthreads();  // the words this level's readers below will consult
      fold_u64(s, occ_r, &p.occupancy[l]);
      fold_u64(s, occ_w, &p.occupancy[p.level_count + l]);
    }

    // ---- 2. PER-LEVEL AND PER-SLICE UNION POPCOUNTS ----------------------
    // The union WORDS are `(read | write) != 0` -- a bitmap over DIGITS, which is the
    // form the owner-slice restriction needs: a popcount cannot be restricted to a
    // sub-range after the fact. docs/internals/summaries.md#union-popcounts
    for (int l = threadIdx.x; l < p.level_count; l += kThreads) {
      int count = 0;
      const int base = s.row_offset[l];
      for (int d = 0; d < s.width[l]; ++d) {
        count += ((s.read_words[base + d] | s.write_words[base + d]) != 0) ? 1 : 0;
      }
      s.level_pc[l] = count;
    }
    for (int sl = threadIdx.x; sl < p.n_slices; sl += kThreads) {
      const int base = s.row_offset[j] + sl * p.slice_digits;
      int count = 0;
      int any_r = 0;
      int any_w = 0;
      for (int k = 0; k < p.slice_digits; ++k) {
        const int rw = s.read_words[base + k];
        const int ww = s.write_words[base + k];
        count += ((rw | ww) != 0) ? 1 : 0;
        any_r |= rw;
        any_w |= ww;
      }
      s.slice_pc[sl] = count;
      s.slice_any_r[sl] = any_r != 0 ? 1 : 0;
      s.slice_any_w[sl] = any_w != 0 ? 1 : 0;
    }
    for (int b = threadIdx.x; b < p.bins; b += kThreads) {
      s.scratch.hist[b] = 0ull;
    }
    __syncthreads();

    // Levels strictly BELOW the boundary are spanned whole by every owner, so their
    // restricted popcount is the level's own and is owner-INVARIANT.
    int inner = 1;
    for (int l = j + 1; l < p.level_count; ++l) inner *= s.level_pc[l];

    // ---- 3. THE OWNER LOOP: K_hat, candidacy, the histogram --------------
    unsigned long long candidates = 0ull;
    for (int owner = threadIdx.x; owner < p.owners_total; owner += kThreads) {
      const int slice = owner % p.n_slices;
      const int rest = owner / p.n_slices;

      // FIXED levels: the owner holds ONE digit of each, so each contributes a 0/1
      // token-validity factor to `K_hat` and an AND term to candidacy.
      int fixed_u = 1;
      int fixed_r = 1;
      int fixed_w = 1;
      for (int l = 0; l < j; ++l) {
        const int digit = (rest / s.fixed_radix[l]) % s.width[l];
        const int rw = s.read_words[s.row_offset[l] + digit];
        const int ww = s.write_words[s.row_offset[l] + digit];
        fixed_u &= ((rw | ww) != 0) ? 1 : 0;
        fixed_r &= (rw != 0) ? 1 : 0;
        fixed_w &= (ww != 0) ? 1 : 0;
      }

      // §4 CANDIDACY is the union of the two SIDES' product sets, never the candidacy of
      // the union word: docs/internals/summaries.md#owner-loop
      const bool cand_r = fixed_r != 0 && s.slice_any_r[slice] != 0;
      const bool cand_w = fixed_w != 0 && s.slice_any_w[slice] != 0;
      if (!(cand_r || cand_w)) continue;
      ++candidates;

      // `K_hat <= BC` STRUCTURALLY by the owner rectangle, so `bins = QPO + 1` is
      // exhaustive and no clamp exists anywhere in this pass.
      const int k_hat = fixed_u * s.slice_pc[slice] * inner;
      atomicAdd(&s.scratch.hist[(k_hat + kMmaKQuantum - 1) / kMmaKQuantum], 1ull);
    }
    __syncthreads();

    for (int b = threadIdx.x; b < p.bins; b += kThreads) {
      const unsigned long long count = s.scratch.hist[b];
      if (count != 0ull) atomicAdd(&p.histogram[b], count);
    }
    __syncthreads();

    // ---- 4. §4.1's CANONICAL PER-LEAF WRITE BITMAP -----------------------
    // A leaf is written iff at SOME token of this tile EVERY level's digit for it is
    // live on the write side: an OR of ANDs, which is why this term is `O(N)` and cannot
    // be condensed from anything computed above. Threads take CONSECUTIVE leaves within
    // a warp, so `__ballot_sync` of the per-leaf bit IS the packed word: ONE `atomicOr`
    // per 32 leaves. docs/internals/summaries.md#leaf-bitmap
    int* leaf_bitmap = p.written_leaf_bitmap + static_cast<long long>(bh) * p.leaf_words;
    int* owner_bitmap = p.written_owner_bitmap + static_cast<long long>(bh) * p.owner_words;
    unsigned long long new_owners = 0ull;
    const int leaf_chunks = (p.leaf_count + kWarpSize - 1) / kWarpSize;

    //: HAZARD ballot-warp-uniform -- docs/internals/summaries.md#ballot-warp-uniform
    for (int chunk = warp; chunk < leaf_chunks; chunk += kWarps) {
      const int leaf = chunk * kWarpSize + lane;
      bool written = false;
      if (leaf < p.leaf_count && write_candidate(s, p, leaf / p.state_block)) {
        int word = ~0;
        for (int l = 0; l < p.level_count && word != 0; ++l) {
          const int digit = (leaf / s.leaf_stride[l]) % s.width[l];
          word &= s.write_words[s.row_offset[l] + digit];
        }
        written = word != 0;
      }
      const unsigned mask = __ballot_sync(0xffffffffu, written);
      if (lane == 0 && mask != 0u) {
        atomicOr(&leaf_bitmap[chunk], static_cast<int>(mask));

        // §4.1's DERIVED owner condensation, folded on the SAME data rather than
        // recomputed. The `atomicOr` RETURNS the old word, so `__popc` of the newly set
        // bits is the exact increment for the active-owner counter -- which is what
        // keeps the pass to ONE launch: docs/internals/summaries.md#owner-condensation
        const int owner_lo = (chunk * kWarpSize) / p.state_block;
        const int owner_hi = (chunk * kWarpSize + kWarpSize - 1) / p.state_block;
        for (int owner = owner_lo; owner <= owner_hi && owner < p.owners_total; ++owner) {
          const int first = owner * p.state_block - chunk * kWarpSize;
          const int lo = first > 0 ? first : 0;
          const int hi = first + p.state_block < kWarpSize ? first + p.state_block : kWarpSize;
          if (lo >= hi) continue;
          const unsigned span =
              (hi - lo == kWarpSize) ? 0xffffffffu : (((1u << (hi - lo)) - 1u) << lo);
          if ((mask & span) == 0u) continue;
          const int bit = 1 << (owner & (kWarpSize - 1));
          const int old = atomicOr(&owner_bitmap[owner >> 5], bit);
          if ((old & bit) == 0) ++new_owners;
        }
      }
    }
    __syncthreads();

    // ---- 5. THE REMAINING GLOBAL FOLDS -----------------------------------
    fold_u64(s, candidates, p.candidate_count);
    fold_u64(s, new_owners, p.written_owner_count);
  }
}

}  // namespace

void launch_summaries(const SummaryParams& params, cudaStream_t stream) {
  const int rows = params.batch_heads * params.n_tiles;
  if (rows <= 0) return;
  const int blocks = rows < kMaxBlocks ? rows : kMaxBlocks;
  summaries_kernel<<<blocks, kThreads, 0, stream>>>(params);
}

void rola_routing_summaries_cuda(
    const std::vector<torch::Tensor>& read_amps, const std::vector<torch::Tensor>& write_amps,
    const std::vector<int64_t>& level_widths, int64_t token_tile, int64_t state_block,
    int64_t first_spanned_level, int64_t slice_digits, int64_t n_slices, torch::Tensor occupancy,
    torch::Tensor histogram, torch::Tensor candidate_count, torch::Tensor written_owner_count,
    torch::Tensor written_leaf_bitmap, torch::Tensor written_owner_bitmap) {
  const int64_t D = static_cast<int64_t>(level_widths.size());
  TORCH_CHECK(D >= 1 && D <= kMaxLevels, "the summary pass carries at most ", kMaxLevels,
              " levels (the same bound FeederFlags' packed word has); got D=", D);
  TORCH_CHECK(
      static_cast<int64_t>(read_amps.size()) == D && static_cast<int64_t>(write_amps.size()) == D,
      "one folded [BH, T, b_l] allocation per level per side is required");
  TORCH_CHECK(token_tile >= 1 && token_tile <= kTokenBitsPerWord, "BT must lie in [1, ",
              kTokenBitsPerWord,
              "] so one tile's tokens are the BITS of one word; got BT=", token_tile,
              ". A wider BT is a representation change that must widen the word, not a "
              "runtime condition to branch on.");
  TORCH_CHECK(state_block % kMmaKQuantum == 0,
              "BC must be a whole number of MMA_K_QUANTUM=", kMmaKQuantum,
              " atoms (spec §1b); got BC=", state_block);

  const int64_t BH = read_amps[0].size(0);
  const int64_t T = read_amps[0].size(1);
  int64_t total_digits = 0;
  int64_t leaf_count = 1;
  for (int64_t l = 0; l < D; ++l) {
    TORCH_CHECK(read_amps[l].is_cuda() && write_amps[l].is_cuda(),
                "the summary pass is a CUDA kernel; level ", l, " is not on a device");
    TORCH_CHECK(read_amps[l].scalar_type() == torch::kFloat32
                    && write_amps[l].scalar_type() == torch::kFloat32,
                "the summary pass reads fp32 routing allocations");
    TORCH_CHECK(read_amps[l].is_contiguous() && write_amps[l].is_contiguous(), "level ", l,
                " must be contiguous [BH, T, b_l]");
    TORCH_CHECK(read_amps[l].dim() == 3 && read_amps[l].size(0) == BH && read_amps[l].size(1) == T
                    && read_amps[l].size(2) == level_widths[l],
                "level ", l, " read allocation must be [", BH, ", ", T, ", ", level_widths[l], "]");
    TORCH_CHECK(write_amps[l].sizes() == read_amps[l].sizes(), "level ", l,
                "'s two sides disagree in shape");
    total_digits += level_widths[l];
    leaf_count *= level_widths[l];
  }
  TORCH_CHECK(total_digits <= kMaxTotalDigits, "sum_l b_l = ", total_digits, " exceeds the ",
              kMaxTotalDigits,
              "-digit shared-memory budget one CTA stages. That is a BUDGET change "
              "(two int32 arrays of this length), not a runtime condition to branch on.");
  TORCH_CHECK(n_slices >= 1 && n_slices <= kMaxSlices, "n_slices=", n_slices,
              " exceeds the staged budget ", kMaxSlices);
  TORCH_CHECK(leaf_count % state_block == 0, "BC must divide N");
  TORCH_CHECK(first_spanned_level >= 0 && first_spanned_level < D,
              "first_spanned_level must index a level");
  TORCH_CHECK(slice_digits >= 1 && n_slices * slice_digits == level_widths[first_spanned_level],
              "the boundary level's slices must partition it exactly");

  const int64_t owners_total = leaf_count / state_block;
  const int64_t n_tiles = (T + token_tile - 1) / token_tile;
  const int64_t bins = state_block / kMmaKQuantum + 1;
  const int64_t leaf_words = (leaf_count + 31) / 32;
  const int64_t owner_words = (owners_total + 31) / 32;

  TORCH_CHECK(occupancy.numel() == 2 * D && occupancy.scalar_type() == torch::kLong,
              "occupancy must be an int64 [2, D]");
  TORCH_CHECK(histogram.numel() == bins && histogram.scalar_type() == torch::kLong,
              "histogram must be an int64 [QPO + 1] = [", bins, "]");
  TORCH_CHECK(candidate_count.numel() == 1 && written_owner_count.numel() == 1,
              "the two counters are int64 scalars");
  TORCH_CHECK(written_leaf_bitmap.numel() == BH * leaf_words
                  && written_leaf_bitmap.scalar_type() == torch::kInt,
              "written_leaf_bitmap must be an int32 [BH, ceil(N/32)]");
  TORCH_CHECK(written_owner_bitmap.numel() == BH * owner_words
                  && written_owner_bitmap.scalar_type() == torch::kInt,
              "written_owner_bitmap must be an int32 [BH, ceil(owners/32)]");

  SummaryParams params{};
  int running = 0;
  for (int64_t l = 0; l < D; ++l) {
    params.read_amps[l] = read_amps[l].data_ptr<float>();
    params.write_amps[l] = write_amps[l].data_ptr<float>();
    params.level_width[l] = static_cast<int>(level_widths[l]);
    params.level_row_offset[l] = running;
    running += params.level_width[l];
    int64_t stride = 1;
    for (int64_t k = l + 1; k < D; ++k) stride *= level_widths[k];
    params.level_leaf_stride[l] = static_cast<int>(stride);
    // `prod(b_{l+1..j-1})` -- the mixed radix of the owner id's FIXED part, level 0 most
    // significant. 1 on the spanned levels, which never index an owner.
    int64_t radix = 1;
    for (int64_t k = l + 1; k < first_spanned_level; ++k) radix *= level_widths[k];
    params.level_fixed_radix[l] = static_cast<int>(radix);
  }
  params.level_count = static_cast<int>(D);
  params.batch_heads = static_cast<int>(BH);
  params.token_count = static_cast<int>(T);
  params.token_tile = static_cast<int>(token_tile);
  params.n_tiles = static_cast<int>(n_tiles);
  params.total_digits = static_cast<int>(total_digits);
  params.leaf_count = static_cast<int>(leaf_count);
  params.state_block = static_cast<int>(state_block);
  params.owners_total = static_cast<int>(owners_total);
  params.n_slices = static_cast<int>(n_slices);
  params.slice_digits = static_cast<int>(slice_digits);
  params.first_spanned_level = static_cast<int>(first_spanned_level);
  params.bins = static_cast<int>(bins);
  params.leaf_words = static_cast<int>(leaf_words);
  params.owner_words = static_cast<int>(owner_words);
  params.occupancy = reinterpret_cast<unsigned long long*>(occupancy.data_ptr<int64_t>());
  params.histogram = reinterpret_cast<unsigned long long*>(histogram.data_ptr<int64_t>());
  params.candidate_count =
      reinterpret_cast<unsigned long long*>(candidate_count.data_ptr<int64_t>());
  params.written_owner_count =
      reinterpret_cast<unsigned long long*>(written_owner_count.data_ptr<int64_t>());
  params.written_leaf_bitmap = written_leaf_bitmap.data_ptr<int>();
  params.written_owner_bitmap = written_owner_bitmap.data_ptr<int>();

  launch_summaries(params, at::cuda::getCurrentCUDAStream());
}

}  // namespace rola::summaries
