// THE T=1 SUPPORT EXPANSION -- `supp(R)` and `supp(W)` as EXACT leaf bitmaps.
//
// One CTA per batch-head. It builds two `N`-bit maps by Kronecker broadcast and emits
// the coarse popcount index the GEMV partitions and skips with.
//
// Three facts a reader must not miss:
//   1. A BITMAP, NOT A CANDIDATE LIST. The earlier mixed-radix odometer BOUNDED an
//      object it could have BUILT. Gone with it: the per-row in_R/in_W running-AND
//      chains, the skip test, the compacted list with digit ids and membership flags,
//      and the enumeration-factor statistic. None of those objects exists here.
//   2. Publishing an N-bit map costs ceil(N/32) words per side REGARDLESS OF SPARSITY,
//      because dead ranges must be written as zeros too. That is a FLOOR, not an
//      inefficiency, and the design's first estimate (~544 B per bh) undercounted it.
//   3. Membership is the EXACT-NONZERO test on the amplitude, never a threshold.
//
// The cost ledger, destination-word ownership, the ping-pong, and the SMEM arm:
// docs/internals/decode/decode_expand.md

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
//: THE RECURSION IS OVER LEVELS, INNERMOST FIRST: level `D-1` seeds one inner block,
//: each outer level broadcasts the block it has into every LIVE digit's leaf range and
//: leaves every dead digit's range zero. After level 0 the buffer holds `S_0 = N` bits.
//: DESTINATION-WORD OWNERSHIP -- one thread owns one destination word and PULLS -- is
//: what makes this race-free without atomics.
//: docs/internals/decode/decode_expand.md#the-recursion
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
          //: MEMBERSHIP IS THE EXACT-NONZERO TEST -- the DEFINING EXPRESSION of
          //: `schedule.py`'s support bitmap, never a threshold:
          //: docs/internals/decode/decode_expand.md#the-recursion
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
    //: HAZARD ping-pong -- docs/internals/decode/decode_expand.md#ping-pong
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

//: `IN_SMEM` is a STRUCTURE-DRIVEN arm, not a threshold: `n_words` is a topology fact
//: and the SMEM budget a hardware fact, both admitted as branch inputs by spec I2. The
//: two arms compute the IDENTICAL function: docs/internals/decode/decode_expand.md#smem-arm
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
  // `pc_coarse[g]` = set bits of `bits_r | bits_w` in groups strictly before `g`;
  // `pc_coarse[n_groups]` = `M`, the exact touched-row count. Taken from the PUBLISHED
  // words rather than accumulated per side, because the two sides' popcounts do not sum
  // to `popcount(r | w)`. This is the ONE place the union is formed outside a register,
  // and it is formed as a COUNT: docs/internals/decode/decode_expand.md#coarse-index
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

//: THE GROWTH PROBE. `expand_side` on the WRITE side alone, condensed 16-deep. The
//: condensation IS PUBLISHED as `atom_bits` -- the step's exact write-atom set, which is
//: what a growth step's `plan_exact` admits from -- and is ANDed against residency into
//: `growth[bh] = 1` iff this step deposits into an atom the page table does not map. It
//: shares the expansion with `decode_expand_kernel` -- the same device function over the
//: same `amp_write` pointers -- so "the probe and the step's admission agree about the
//: write support" is a property of the call graph rather than of an audit, and the set is
//: derived ONCE per growth step rather than once here and again on the host.
//: docs/internals/decode/decode_expand.md#growth-probe
template <bool IN_SMEM>
__global__ __launch_bounds__(kDecodeThreads) void decode_growth_probe_kernel(DecodeParams p) {
  extern __shared__ char smem_raw[];
  const int bh = blockIdx.x;
  const int tid = threadIdx.x;
  const int nw = p.n_words;

  uint32_t* smem = reinterpret_cast<uint32_t*>(smem_raw);
  uint32_t* scratch = p.scratch + static_cast<int64_t>(bh) * 2 * nw;
  uint32_t* buf0 = IN_SMEM ? smem : scratch;
  uint32_t* buf1 = IN_SMEM ? (smem + nw) : (scratch + nw);
  uint32_t* bits_w = p.bits_w + static_cast<int64_t>(bh) * nw;

  expand_side(p, p.D, bh, p.amp_write, buf0, buf1, bits_w);

  //: A bitmap WORD is exactly TWO atoms, because the atom is 16 leaves and leaf order is
  //: the bitmap's bit order. So the condensation is a pair of half-word tests and needs
  //: no loop over leaves. Every atom's byte is written UNCONDITIONALLY -- a dead atom's
  //: `false` is as much of the published set as a live one's `true`, and writing only the
  //: live ones would leave the buffer carrying the previous step's answer:
  //: docs/internals/decode/decode_expand.md#growth-probe
  const int32_t* page_row = p.page_tbl + static_cast<int64_t>(bh) * p.atoms_per_bh;
  bool* atom_row = p.atom_bits + static_cast<int64_t>(bh) * p.atoms_per_bh;
  int found = 0;
  for (int w = tid; w < nw; w += kDecodeThreads) {
    const uint32_t word = bits_w[w];
    const int atom = w << 1;
    const bool live[2] = {(word & 0x0000ffffu) != 0u, (word & 0xffff0000u) != 0u};
    for (int half = 0; half < 2; ++half) {
      const int a = atom + half;
      if (a >= p.atoms_per_bh) break;
      atom_row[a] = live[half];
      if (live[half] && page_row[a] < 0) found = 1;
    }
  }
  __shared__ int any_new;
  if (tid == 0) any_new = 0;
  __syncthreads();
  if (found) any_new = 1;
  __syncthreads();
  if (tid == 0) p.growth[bh] = any_new;
}

}  // namespace

//: The dynamic SMEM the SMEM arm needs: the ping-pong pair and nothing else. Host-
//: visible so the launcher picks the arm from the device's real budget, not a literal.
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

void launch_decode_growth_probe(const DecodeParams& p, int64_t smem_budget) {
  const bool in_smem = decode_expand_uses_smem(p.n_words, smem_budget);
  const size_t smem = in_smem ? decode_expand_smem_bytes(p.n_words) : 0;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (in_smem) {
    decode_growth_probe_kernel<true><<<p.BH, kDecodeThreads, smem, stream>>>(p);
  } else {
    decode_growth_probe_kernel<false><<<p.BH, kDecodeThreads, smem, stream>>>(p);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace decode
}  // namespace rola
