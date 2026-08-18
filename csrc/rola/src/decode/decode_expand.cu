// THE GROWTH PROBE -- does this step deposit into an atom the page table does not map?
//
// One CTA per batch-head. It folds its OWN write side, expands it to the exact leaf
// bitmap, condenses that 16-deep into the step's write-atom set, and ANDs the set against
// residency. It shares the fold and the expansion with the step -- the same device
// functions over the same raw operands -- so "the probe and the step's admission agree
// about the write support" is a property of the call graph rather than of an audit.
//
// The verdict's asynchrony, the condensation and the all-or-nothing reduction:
// docs/internals/decode/decode_expand.md

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include "decode.cuh"
#include "decode_expand.cuh"
#include "decode_fold.cuh"

namespace rola {
namespace decode {
namespace {

//: `IN_SMEM` is a STRUCTURE-DRIVEN arm, not a threshold: `n_words` is a topology fact
//: and the SMEM budget a hardware fact, both admitted as branch inputs by spec I2. The
//: two arms compute the IDENTICAL function: docs/internals/decode/decode_expand.md#smem-arm
template <bool IN_SMEM>
__global__ __launch_bounds__(kDecodeThreads) void decode_growth_probe_kernel(DecodeParams p) {
  extern __shared__ char smem_raw[];
  const int bh = blockIdx.x;
  const int b = bh / p.H;
  const int h = bh - b * p.H;
  const int tid = threadIdx.x;
  const int nw = p.n_words;

  float* sa_w = reinterpret_cast<float*>(smem_raw);
  uint32_t* smem_ping = reinterpret_cast<uint32_t*>(sa_w + p.total_rows);
  uint32_t* global_ping = p.work + static_cast<int64_t>(bh) * p.n_split * kExpandBufWords * nw;
  uint32_t* ping = IN_SMEM ? smem_ping : global_ping;
  uint32_t* bits_w = p.probe_bits + static_cast<int64_t>(bh) * nw;

  if (p.levels_bf16) {
    fold_rows<__nv_bfloat16>(p, b, h, p.write, nullptr, nullptr, sa_w);
  } else {
    fold_rows<float>(p, b, h, p.write, nullptr, nullptr, sa_w);
  }
  __syncthreads();

  const uint32_t* expanded = ping + expand_sides<1>(p, sa_w, nullptr, ping) * nw;
  //: THE PUBLISH, and the probe's whole output: the step's two maps die in its shared
  //: memory, this one is the object the residency answer was reached from:
  //: docs/internals/decode/decode_expand.md#published-map
  for (int w = tid; w < nw; w += kDecodeThreads) bits_w[w] = expanded[w];
  __syncthreads();

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

  //: THE REDUCTION THE GATE READS, and it is all-or-nothing across batch-heads on
  //: purpose: a per-`bh` skip would need a per-`bh` replay, and a replay that re-ran an
  //: already-deposited `bh` would double its write. The last-arriving CTA ORs the vector
  //: and resets the counter, so no `memset` stands between two steps:
  //: docs/internals/decode/decode_expand.md#growth-reduction
  __shared__ int s_last, s_any;
  __threadfence();
  __syncthreads();
  if (tid == 0) s_last = (atomicAdd(p.probe_ctr, 1) == gridDim.x - 1) ? 1 : 0;
  __syncthreads();
  if (s_last == 0) return;
  if (tid == 0) s_any = 0;
  __syncthreads();
  int seen = 0;
  for (int i = tid; i < gridDim.x; i += blockDim.x) seen |= __ldcg(p.growth + i);
  if (seen) s_any = 1;
  __syncthreads();
  if (tid == 0) {
    *p.growth_any = s_any;
    *p.probe_ctr = 0;
  }
}

}  // namespace

//: The probe's staged write row set plus, on the SMEM arm, the recursion's ping-pong
//: pair. Its ONE published map is its OUTPUT and lands in `probe_bits`, so it never holds
//: a third buffer.
size_t decode_probe_smem_bytes(int total_rows, int n_words, bool in_smem) {
  const size_t words = static_cast<size_t>(total_rows)
                       + (in_smem ? static_cast<size_t>(kProbeBufWords) * n_words : 0);
  return words * sizeof(uint32_t);
}

void launch_decode_growth_probe(const DecodeParams& p, int64_t smem_budget) {
  const bool in_smem =
      static_cast<int64_t>(decode_probe_smem_bytes(p.total_rows, p.n_words, true)) <= smem_budget;
  const size_t smem = decode_probe_smem_bytes(p.total_rows, p.n_words, in_smem);
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
