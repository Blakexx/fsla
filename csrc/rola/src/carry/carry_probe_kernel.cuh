// csrc/rola/src/carry/carry_probe_kernel.cuh -- THE SEGMENT LAYOUT'S ROUND TRIP.
//
// A NAMED SMEM LAYOUT IS GATED BY A ROUND TRIP (KERNEL_STANDARDS section 3), and the
// two access patterns that derive it are here side by side: the FETCHER's stores --
// the same base-plus-displacement arithmetic `fetch_values` gives `cp.async` and
// `expand` gives `st.shared` -- and the CONSUMER's `ldmatrix.trans` pulls, at the same
// `myV`/`myC` bases the fold uses.  The probe writes a pattern whose sixteen bits NAME
// the (row, column) it was stored at and hands back what each fragment recovered, at
// the fragment's own thread-value map; a stride, a base or a fragment map that
// disagrees between the two legs cannot survive that comparison.
//
// The pattern is a RAW bf16 BIT PATTERN and never a number: this exercises movement,
// so the only property asked of the value is that it be distinct per cell.
#pragma once

#include "carry_kernel.cuh"

namespace rola::carry {

template <int D, int B, int K, int M, int W, int C, int W0, int W1, int W2, int W3, int R0, int R1,
          int R2, int R3>
__global__ void carry_layout_probe_kernel(const __nv_bfloat16* __restrict__ pw,
                                          int* __restrict__ outV, int* __restrict__ outC,
                                          int* __restrict__ outS) {
  using G = CarryGeom<D, B, K, M, W, C, R0, R1, R2, R3>;
  constexpr int kStreams = G::kStreams, kSegRows = G::kSegRows;
  constexpr int LdV = G::LdV, LdS = G::LdS, kLeafW = G::kLeafW;
  constexpr int MCH = kDv / kMmaM;
  constexpr int NL = kLeafW / kMmaN, kNJ = NL / 2;
  constexpr int kChunkV = kDv / 8;
  constexpr int kWordsRow = kLeafW / 2;

  extern __shared__ __align__(16) char probe_smem[];
  __nv_bfloat16* const segV = reinterpret_cast<__nv_bfloat16*>(probe_smem);
  __nv_bfloat16* const segC = segV + kStreams * kSegRows * LdV;
  __nv_bfloat16* const wstg = segC + kStreams * kSegRows * LdS;
  const uint32_t segVB = smem_addr(segV), segCB = smem_addr(segC), stgB = smem_addr(wstg);
  const int tid = (int)threadIdx.x, warp = tid >> 5, lane = tid & 31;
  constexpr int kThreads = G::kThreads;
  using BP = typename G::BP;

  //: THE WRITER'S LEG.  `(stream, row)` addresses a segment row and the displacement
  //: inside it is the fetcher's: `c * 8` amplitudes for the value quantum, `jw` packed
  //: pairs for the coefficient word.
#pragma unroll 1
  for (int i = tid; i < kStreams * kSegRows * kChunkV; i += kThreads) {
    const int c = i % kChunkV, r = (i / kChunkV) % kSegRows, s = i / (kChunkV * kSegRows);
    uint4 v;
    unsigned short* const h = reinterpret_cast<unsigned short*>(&v);
#pragma unroll
    for (int e = 0; e < 8; ++e) h[e] = (unsigned short)(r * 128 + c * 8 + e);
    store_shared_vec16(segVB + (uint32_t)(((s * kSegRows + r) * LdV + c * 8) * 2), v);
  }
#pragma unroll 1
  for (int i = tid; i < kStreams * kSegRows * kWordsRow; i += kThreads) {
    const int jw = i % kWordsRow, r = (i / kWordsRow) % kSegRows, s = i / (kWordsRow * kSegRows);
    const uint32_t w = ((uint32_t)(r * 128 + 2 * jw + 1) << 16) | (uint32_t)(r * 128 + 2 * jw);
    store_shared_u32(segCB + (uint32_t)(((s * kSegRows + r) * LdS) * 2 + jw * 4), w);
  }
  //: THE STAGED FACTOR ROW'S LEG.  `stage_factors`' arithmetic verbatim, at owner
  //: zero (`runCol(l) == col_base(l)`), against a `[kSegRows][kWtot]` plane: what it
  //: gates is that each level's run lands at its own offset AND that both the global
  //: and the shared address of every asynchronous quantum are naturally aligned -- a
  //: misalignment is a fault, not a wrong answer, so it has to be executed to be seen.
  constexpr int kStgCh = ceil_pow2c(BP::kStageChunks);
  constexpr int kStgStep = kThreads / kStgCh;
  constexpr int kStgPass = (kSegRows + kStgStep - 1) / kStgStep;
  {
    const int c = tid & (kStgCh - 1);
    static_for<kStgPass>([&](auto Pc) {
      const int r = (tid >> ilog2c(kStgCh)) + decltype(Pc)::value * kStgStep;
      if (r < kSegRows) {
        const uint32_t tokB = (uint32_t)r * (uint32_t)(BP::kWtot * 2);
        static_for<D>([&](auto Lc) {
          constexpr int l = decltype(Lc)::value;
          if (c >= BP::stage_chunk_base(l) && c < BP::stage_chunk_base(l) + BP::stage_chunks(l)) {
            const int sub = (c - BP::stage_chunk_base(l)) * BP::stage_el(l);
            stage_run<BP::stage_el(l) * 2>(
                stgB + (uint32_t)((r * BP::kStageW + BP::stage_off(l) + sub) * 2),
                gather_ptr(pw, tokB + (uint32_t)((BP::col_base(l) + sub) * 2)));
          }
        });
      }
    });
    stage_commit();
    stage_wait<0>();
  }
  rendezvous();
#pragma unroll 1
  for (int i = tid; i < kSegRows * BP::kStageW; i += kThreads)
    outS[i] = (int)load_shared_u16(stgB + (uint32_t)(i * 2));

  //: THE CONSUMER'S LEG, at the fold's own bases, and the fragment maps the fold's
  //: MMAs read those fragments under.
  const int lr = lane & 7, tl = lane >> 3;
  const int rF = (tl & 1) * 8 + lr, cF = tl >> 1;
  const int rT = (tl >> 1) * 8 + lr, cT = tl & 1;
  const int s = warp;
  const uint32_t myV = segVB + (uint32_t)(((s * kSegRows + rT) * LdV + cT * 8) * 2);
  const uint32_t myC = segCB + (uint32_t)(((s * kSegRows + rF) * LdS + cF * 8) * 2);
  static_for<MCH>([&](auto Mc) {
    constexpr int m = decltype(Mc)::value;
    uint32_t a[4];
    load_frag_t(a, myV + (uint32_t)(m * kMmaM * 2));
    static_for<4>([&](auto Rc) {
      constexpr int r = decltype(Rc)::value;
      const int chan = m * kMmaM + 8 * (r & 1) + (lane >> 2);
      static_for<2>([&](auto Ec) {
        constexpr int e = decltype(Ec)::value;
        const int tok = 8 * (r >> 1) + 2 * (lane & 3) + e;
        outV[(s * kSegRows + tok) * kDv + chan] = (int)((a[r] >> (16 * e)) & 0xFFFFu);
      });
    });
  });
  static_for<kNJ>([&](auto Nc) {
    constexpr int nj = decltype(Nc)::value;
    uint32_t b[4];
    load_frag_t(b, myC + (uint32_t)(2 * nj * 8 * 2));
    static_for<4>([&](auto Rc) {
      constexpr int r = decltype(Rc)::value;
      const int slot = 16 * nj + 8 * (r >> 1) + (lane >> 2);
      static_for<2>([&](auto Ec) {
        constexpr int e = decltype(Ec)::value;
        const int tok = 8 * (r & 1) + 2 * (lane & 3) + e;
        outC[(s * kSegRows + tok) * kLeafW + slot] = (int)((b[r] >> (16 * e)) & 0xFFFFu);
      });
    });
  });
}

}  // namespace rola::carry
