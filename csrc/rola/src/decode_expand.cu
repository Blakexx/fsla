// THE T=1 SUPPORT EXPANSION -- `supp(R)` and `supp(W)` as EXACT leaf bitmaps.
//
// One CTA per batch-head. It builds two `N`-bit maps by Kronecker broadcast and
// emits the coarse popcount index the GEMV partitions and skips with.
//
// **WHY A BITMAP AND NOT A CANDIDATE LIST.** An earlier revision of this design
// enumerated a mixed-radix odometer over the per-level UNION lists and skipped
// candidates outside `supp(R) UNION supp(W)` before the state load. That was correct
// and cost no bandwidth, but it BOUNDED an object it could have BUILT: the union is
// not a Cartesian product, which is why the odometer reached for the containing
// product `prod_l U_l`, whereas the exact set is an `N`-bit map that is 8 KiB at
// `N = 65,536` against a state stream of 130 MiB. The bound was more expensive to
// reason about than the truth is to materialize. What that deletes, and it is the
// bulk of the older design: the per-row `in_R`/`in_W` running-AND chains, the skip
// test, the compacted list with digit ids and membership flags, and the
// enumeration-factor statistic (with the bench observable that would have had to
// report it). None of those objects exists here.
//
// **THE COST, STATED HONESTLY BECAUSE THE DESIGN'S FIRST ESTIMATE WAS WRONG.** The
// design doc priced a sparse at-scale expansion at ~544 B per bh by counting only the
// broadcast COPIES. That undercounts: a bitmap's DEAD ranges must be written as zeros
// too, so the irreducible cost of publishing an `N`-bit map is `ceil(N/32)` words per
// side regardless of sparsity -- 8 KiB per side per bh at `N = 65,536`, 16 KiB for
// both. That is a floor, not an inefficiency; what the SMEM arm below buys is that
// the D-1 INTERMEDIATE levels cost SMEM traffic rather than HBM traffic, so the HBM
// figure IS the floor. The global arm pays the intermediates to HBM and is `D` times
// the floor. Both are noise against the dense-read state stream (0.012 % at at-scale)
// and both are visible against a very sparse high-batch step; see the findings doc.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cub/block/block_scan.cuh>

#include "decode.cuh"

namespace rola {
namespace decode {
namespace {

using BlockScanT = cub::BlockScan<int, kDecodeThreads>;

//: The expansion, for ONE side. `amp` is the side's per-level amplitude row set.
//:
//: THE RECURSION IS OVER LEVELS, INNERMOST FIRST. Level `D-1` seeds one inner block
//: of `b_{D-1}` bits from its own amplitudes; each outer level `l` then broadcasts the
//: `S_{l+1}`-bit block it has into every LIVE digit's leaf range and leaves every dead
//: digit's range zero. After level 0 the buffer holds `S_0 = N` bits, which is the
//: exact support.
//:
//: DESTINATION-WORD OWNERSHIP is what makes this race-free without atomics: one
//: thread owns one destination word and PULLS every bit it needs, rather than each
//: source digit PUSHING into ranges two threads could share at a boundary word. The
//: pull needs at most `ceil(32 / S_{l+1}) + 1` source digits per destination word, so
//: the integer division that finds the digit is paid once or twice per word and never
//: once per bit.
//:
//: PING-PONG, not in place: destination word 0 is read by the thread writing
//: destination word 100 (every digit broadcasts the SAME source block), so writing
//: into the source would be a read-write race no barrier placement can fix. The two
//: buffers are the fix and they are why the SMEM arm costs `2 * n_words` words.
__device__ __forceinline__ void expand_side(const DecodeParams& p, int D, int bh,
                                            const float* const* amp, uint32_t* buf0, uint32_t* buf1,
                                            uint32_t* out) {
  const int tid = threadIdx.x;
  const int nw = p.n_words;
  uint32_t* buf[2] = {buf0, buf1};
  int cur = 0;

  // ---- the seed: level D-1's own indicator, one inner block ------------------
  {
    const int last = D - 1;
    const int width = p.level_width[last];
    const float* row = amp[last] + static_cast<int64_t>(bh) * width;
    const int seed_words = ceil_div_int(width, 32);
    for (int w = tid; w < nw; w += kDecodeThreads) {
      uint32_t m = 0u;
      if (w < seed_words) {
        const int g0 = w * 32;
        for (int b = 0; b < 32; ++b) {
          const int g = g0 + b;
          if (g >= width) break;
          //: MEMBERSHIP IS THE EXACT-NONZERO TEST, and that is not a threshold: it is
          //: the DEFINING EXPRESSION of `schedule.py`'s support bitmap
          //: (`read_mask = tuple(level != 0 ...)`, and §2386's "exact nonzero-ness of
          //: the amplitude, never a threshold"). Evaluating it here rather than
          //: reading a materialized bitmap is the same predicate one step earlier.
          if (row[g] != 0.0f) m |= 1u << b;
        }
      }
      buf[cur][w] = m;
    }
  }
  __syncthreads();

  // ---- the broadcast levels, outward ----------------------------------------
  for (int l = D - 2; l >= 0; --l) {
    const int span_in = p.level_span[l + 1];
    const int span_out = p.level_span[l];
    const int in_words = ceil_div_int(span_in, 32);
    const int out_words = ceil_div_int(span_out, 32);
    const int width = p.level_width[l];
    const float* row = amp[l] + static_cast<int64_t>(bh) * width;
    const uint32_t* src = buf[cur];
    uint32_t* dst = buf[cur ^ 1];
    for (int dw = tid; dw < nw; dw += kDecodeThreads) {
      uint32_t m = 0u;
      if (dw < out_words) {
        const int base = dw * 32;
        int q = 0;
        while (q < 32 && (base + q) < span_out) {
          const int pbit = base + q;
          const int g = pbit / span_in;
          const int i = pbit - g * span_in;
          int take = span_in - i;
          if (take > 32 - q) take = 32 - q;
          if (pbit + take > span_out) take = span_out - pbit;
          if (row[g] != 0.0f) m |= extract_bits(src, in_words, i, take) << q;
          q += take;
        }
      }
      dst[dw] = m;
    }
    __syncthreads();
    cur ^= 1;
  }

  // ---- publish ---------------------------------------------------------------
  const uint32_t* fin = buf[cur];
  for (int w = tid; w < nw; w += kDecodeThreads) out[w] = fin[w];
  __syncthreads();
}

//: `IN_SMEM` is a STRUCTURE-DRIVEN arm, not a threshold: the ping-pong pair is
//: `2 * n_words` words, `n_words` is a topology fact (`ceil(N/32)`) and the SMEM
//: budget is a hardware fact, and spec I2 admits both as branch inputs. The two arms
//: compute the identical function -- only where the intermediate levels live moves --
//: which is why the G3 gate runs both arms against the same expected bitmaps.
template <bool IN_SMEM>
__global__ __launch_bounds__(kDecodeThreads) void decode_expand_kernel(DecodeParams p) {
  extern __shared__ char smem_raw[];
  const int bh = blockIdx.x;
  const int tid = threadIdx.x;
  const int nw = p.n_words;
  const int ng = p.n_groups;

  uint32_t* smem = reinterpret_cast<uint32_t*>(smem_raw);
  uint32_t* bits_r = p.bits_r + static_cast<int64_t>(bh) * nw;
  uint32_t* bits_w = p.bits_w + static_cast<int64_t>(bh) * nw;
  uint32_t* scratch = p.scratch + static_cast<int64_t>(bh) * 2 * nw;

  uint32_t* buf0 = IN_SMEM ? smem : scratch;
  uint32_t* buf1 = IN_SMEM ? (smem + nw) : (scratch + nw);

  expand_side(p, p.D, bh, p.amp_read, buf0, buf1, bits_r);
  expand_side(p, p.D, bh, p.amp_write, buf0, buf1, bits_w);

  // ---- the coarse popcount index --------------------------------------------
  // `pc_coarse[g]` = number of set bits of `bits_r | bits_w` in groups strictly
  // before `g`; `pc_coarse[n_groups]` = `M`, the exact touched-row count. The GEMV
  // binary-searches it to resolve a ROW range into a WORD range, and consults it
  // again DURING the walk to skip a wholly empty group without reading its 32 words.
  //
  // The count is taken from the PUBLISHED words rather than accumulated per side
  // during publication, and the reason is arithmetic rather than tidiness: the two
  // sides' popcounts SUM to something that is not `popcount(r | w)` -- a leaf in both
  // supports would be counted twice, and `M` has to be the row count the GEMV will
  // actually visit. This is the one place the union is formed outside a register, and
  // it is formed as a COUNT; no third bitmap is ever stored.
  __shared__ typename BlockScanT::TempStorage scan_temp;
  __shared__ int running;
  if (tid == 0) running = 0;
  __syncthreads();
  for (int base = 0; base < ng; base += kDecodeThreads) {
    const int g = base + tid;
    int count = 0;
    if (g < ng) {
      const int w0 = g * kCoarseGroupWords;
      const int w1 = min(w0 + kCoarseGroupWords, nw);
      for (int w = w0; w < w1; ++w) count += __popc(bits_r[w] | bits_w[w]);
    }
    int excl = 0;
    int aggregate = 0;
    BlockScanT(scan_temp).ExclusiveSum(count, excl, aggregate);
    if (g < ng) p.pc_coarse[static_cast<int64_t>(bh) * (ng + 1) + g] = running + excl;
    __syncthreads();
    if (tid == 0) running += aggregate;
    __syncthreads();
  }
  if (tid == 0) p.pc_coarse[static_cast<int64_t>(bh) * (ng + 1) + ng] = running;
}

}  // namespace

//: The dynamic SMEM the SMEM arm needs: the ping-pong pair and nothing else (the
//: coarse-index scan's storage is static `__shared__`, sized by the block, not by the
//: topology). Host-visible so the launcher picks the arm from the device's real
//: budget rather than from a literal.
size_t decode_expand_smem_bytes(int n_words) {
  return static_cast<size_t>(2) * n_words * sizeof(uint32_t);
}

bool decode_expand_uses_smem(int n_words, int64_t smem_budget) {
  return static_cast<int64_t>(decode_expand_smem_bytes(n_words)) <= smem_budget;
}

void launch_decode_expand(const DecodeParams& p, int64_t smem_budget) {
  const bool in_smem = decode_expand_uses_smem(p.n_words, smem_budget);
  const size_t smem = in_smem ? decode_expand_smem_bytes(p.n_words) : 0;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (in_smem) {
    decode_expand_kernel<true><<<p.BH, kDecodeThreads, smem, stream>>>(p);
  } else {
    decode_expand_kernel<false><<<p.BH, kDecodeThreads, smem, stream>>>(p);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace decode
}  // namespace rola
