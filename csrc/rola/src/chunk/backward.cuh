// csrc/rola/src/chunk/backward.cuh -- SCAN B, the reverse adjoint of the chunk
// consumer (P65).  It carries the two adjoint accumulators, the descending walk,
// both operand panels, the slot Gram with its staircase, dv COMPLETE, and the one
// new arithmetic primitive (a contraction whose k axis is d_v).  It does NOT yet
// carry the d_v Gram, the per-slot dW or the digit reduction, so d gw is the
// inter-chunk part alone and d pr / d pw are not emitted.
// What it computes, what it omits, why the walk needs no reversed table, why two
// Grams do not fit at once, and the register arithmetic behind the mass column:
// docs/internals/chunk/backward.md
#pragma once

#include "chunk_kernel.cuh"

namespace rola::chunk {

//: Scan B's launch arguments.  `dnum`/`dden` are the readout seeds, already bf16
//: because both are mma operands; `dnum` is `[BH, L, d_v]` and its row stride is
//: therefore d_v, unlike V's token-major one.
struct ScanBParams {
  const __nv_bfloat16 *pr, *pw, *gw, *v, *dnum, *dden;
  const uint32_t* tbl;
  float *dv, *dwsum, *dgw, *dpw;
  int BH, L;
  int v_row_bytes;
};

//: SCAN B'S GEOMETRY.  It is NOT `Geom`: the ring carries TWO value-shaped
//: panels (`dnum` gathered by the read list, V by the write list), there is no
//: Gram, and there is a cross-unit window the forward has no term for.
template <int C, int BC>
struct GeomB {
  static constexpr int GRP = groups_for(BC);
  static constexpr int BCg = BC / GRP;
  static constexpr int LdV = kD + kPadE;
  static constexpr int LdS = BCg + kPadE;
  static constexpr int LdG = C + kPadE;
  static constexpr int LdO = kD + kPadF;
  static constexpr int hdr = 16;
  static constexpr int idx_slot = hdr + ((5 * C + 15) / 16) * 16;
  static constexpr int slot = 2 * C * LdV * (int)sizeof(__nv_bfloat16) + idx_slot;
  static constexpr int form_bytes =
      2 * kFormWords * (int)sizeof(uint32_t) + kCookSlots * (int)sizeof(int);
  //: THE CHANNEL CONTRACTION'S k AXIS IS d_v, WHICH IS THE CARVE'S OWN AXIS, so
  //: every unit holds a PARTIAL and the sum across units is structural.  One row
  //: per unit, one column per token, is the whole of it.
  //: `grad` is the units' rendezvous for `d gw`; `dden_bytes` is the chunk's own
  //: readout seed for the mass column, staged once per chunk instead of gathered
  //: per k-tile per lane.
  static constexpr int grad_bytes = kUnits * C * (int)sizeof(float);
  static constexpr int dden_bytes = C * (int)sizeof(float);
  //: the chunk's write GAINS, staged per write entry: the panel is produced
  //: ungained so `d gw` can read the level product alone, and the gain is applied
  //: once, on the readout's token axis -- docs/internals/chunk/backward.md#dw-slotsum
  static constexpr int gain_bytes = C * (int)sizeof(float);
  //: the OUTER level's amplitudes for this block's two digits, per write entry --
  //: the leave-one-out factor the inner level's digit gradient is weighted by
  static constexpr int pw0_bytes = 2 * C * (int)sizeof(float);
  //: THE READOUT WINDOW IS TWO THINGS IN SEQUENCE, and it is sized for the
  //: larger: `dv`'s fp32 staging until the drain frees it, then `dz` published
  //: per slot for `d gw`'s token-major reduction.  They never overlap, so the
  //: mass column costs nothing.  The phase that comes next -- the per-slot dW's
  //: transposed adjoint state -- is carved to fit here as well:
  //: docs/internals/chunk/backward.md#readout-window
  static constexpr int dz_bytes = BCg * (int)sizeof(float);
  //: THE TRANSPOSED ADJOINT STATE, a QUARTER of the slots at a time.  (A.18)'s k
  //: axis is `d_v`, which is the carve's own axis, so no unit can finish the
  //: contraction from its own registers; each writes its 16 channels here, the
  //: writes are DISJOINT, and every unit then reads all `d_v` for its own slots.
  //: The row count carries `kMmaN` of pad because `ldmatrix` reads sixteen rows
  //: from an eight-row tile's base -- docs/internals/chunk/backward.md#readout-window
  static constexpr int LdDs = kD + kPadE;
  static constexpr int DsRows = BCg / 4 + kMmaN;
  static constexpr int ds_bytes = DsRows * LdDs * (int)sizeof(__nv_bfloat16);
  static constexpr int out_bytes = C * LdO * (int)sizeof(float);
  static constexpr int late_bytes = dz_bytes + ds_bytes;
  static constexpr int win_bytes = out_bytes > late_bytes ? out_bytes : late_bytes;
  //: ONE GRAM BUFFER, AND SCAN B NEEDS TWO GRAMS.  `dv`'s intra term contracts
  //: over SLOTS (the forward's own `R.W^T`) and `dW`'s contracts over `d_v`
  //: (`dnum.v^T`); two resident buffers reach `cta_target = 1` and the arm's
  //: occupancy is not negotiable, so the two share this one across disjoint
  //: phases -- docs/internals/chunk/backward.md#two-grams
  static constexpr int gram_bytes = C * LdG * (int)sizeof(__nv_bfloat16);
  static constexpr int fixed = GRP * (2 * C * LdS) * (int)sizeof(__nv_bfloat16) + GRP * win_bytes
                               + grad_bytes + dden_bytes + gain_bytes + pw0_bytes + gram_bytes
                               + (1 + kAheadExtra) * idx_slot + form_bytes;
  static constexpr int stages = design::stages_for(slot, fixed, threads_for(BC), reg_floor(C, BC));
  static constexpr size_t bytes = (size_t)stages * slot + fixed;
};

//: THE CTA TARGET SCAN B IS BOUND AT, and it is checked against the forward's at
//: the instantiation site: a backward that resident-fewer CTAs than its forward
//: would move the arm's occupancy without anybody deciding to.
template <int C, int BC>
constexpr int kScanBCtaTarget =
    design::cta_target((int)GeomB<C, BC>::bytes, threads_for(BC), reg_floor(C, BC));

//: THE FORM, DESCENDING.  `form_impl`'s body with the token axis walked the
//: other way: `formLow` is the LOWEST word formed, so the ring holds
//: `[formLow, formLow + kFormWords)` and refilling means lowering it.
template <int kClauses, int kFormWordsT, int kThreadsT>
__device__ __forceinline__ void form_desc_impl(int& formLow, int limit, int tid,
                                               const uint32_t* rowR, const uint32_t* rowW,
                                               uint32_t cA, uint32_t cB, uint32_t cC, uint32_t cD,
                                               uint32_t tailMask, int lastWord, uint32_t* sfR,
                                               uint32_t* sfW) {
  for (int w = formLow - 1 - tid; w >= limit; w -= kThreadsT) {
    const uint32_t off = (uint32_t)w * (uint32_t)sizeof(uint32_t);
    uint32_t rm, wm;
    if constexpr (kClauses == 0) {
      rm = 0xFFFFFFFFu;
      wm = 0xFFFFFFFFu;
    } else if constexpr (kClauses == 1) {
      rm = *reinterpret_cast<const uint32_t*>(gather_ptr(rowR, cA + off));
      wm = *reinterpret_cast<const uint32_t*>(gather_ptr(rowW, cA + off));
    } else if constexpr (kClauses == 2) {
      rm = *reinterpret_cast<const uint32_t*>(gather_ptr(rowR, cA + off))
           & *reinterpret_cast<const uint32_t*>(gather_ptr(rowR, cB + off));
      wm = *reinterpret_cast<const uint32_t*>(gather_ptr(rowW, cA + off))
           & *reinterpret_cast<const uint32_t*>(gather_ptr(rowW, cB + off));
    } else {
      rm = *reinterpret_cast<const uint32_t*>(gather_ptr(rowR, cA + off))
           & *reinterpret_cast<const uint32_t*>(gather_ptr(rowR, cB + off))
           & *reinterpret_cast<const uint32_t*>(gather_ptr(rowR, cC + off));
      wm = *reinterpret_cast<const uint32_t*>(gather_ptr(rowW, cA + off))
           & *reinterpret_cast<const uint32_t*>(gather_ptr(rowW, cB + off))
           & *reinterpret_cast<const uint32_t*>(gather_ptr(rowW, cC + off));
      if constexpr (kClauses > 3) {
        rm &= *reinterpret_cast<const uint32_t*>(gather_ptr(rowR, cD + off));
        wm &= *reinterpret_cast<const uint32_t*>(gather_ptr(rowW, cD + off));
      }
    }
    if (w == lastWord) {
      rm &= tailMask;
      wm &= tailMask;
    }
    const int slot = w & (kFormWordsT - 1);
    sfR[slot] = rm;
    sfW[slot] = wm;
  }
  formLow = limit;
}

//: THE DESCENDING SCHEDULE, as a component rather than as a closure.  Scan B and
//: the partition probe must run the SAME walk -- a probe that re-implemented the
//: closing rule would gate its own copy -- so the ring, the cook's carry and the
//: index slots live here and both kernels drive them.
//: The lists it publishes are ASCENDING; only the CHUNK ORDER is reversed.
//: docs/internals/chunk/backward.md
template <int C, int kClauses, int kThreadsT, int kIdxSlotT, int kHdrT>
struct Descent {
  char* sidx;
  uint32_t* sfR;
  uint32_t* sfW;
  int* scook;
  const uint32_t* rowR;
  const uint32_t* rowW;
  uint32_t cA, cB, cC, cD;
  uint32_t tailMask;
  int lastWord;
  int tokWords;

  int formLow;
  int cookWord, cookNr, cookNw, chunkWin, winBaseW;
  bool cookOpen;

  static constexpr int kWinWords = kWindow / kTokBits;
  static constexpr int kListPasses = (C + kTokBits - 1) / kTokBits;

  __device__ __forceinline__ int* ihdr(int d) const {
    return reinterpret_cast<int*>(sidx + d * kIdxSlotT);
  }
  __device__ __forceinline__ uint16_t* iread(int d) const {
    return reinterpret_cast<uint16_t*>(sidx + d * kIdxSlotT + kHdrT);
  }
  __device__ __forceinline__ uint16_t* iwrite(int d) const { return iread(d) + C; }
  __device__ __forceinline__ uint8_t* ithr(int d) const {
    return reinterpret_cast<uint8_t*>(sidx + d * kIdxSlotT + kHdrT + 4 * C);
  }

  //: Called by EVERY thread of the CTA before the first rendezvous.
  __device__ __forceinline__ void init(int tid) {
    formLow = tokWords;
    cookWord = tokWords - 1;
    cookNr = 0;
    cookNw = 0;
    cookOpen = false;
    chunkWin = 0;
    winBaseW = 0;
    if (tid == 0) {
      scook[kCookWord] = tokWords - 1;
      scook[kCookOpen] = 0;
    }
  }

  __device__ __forceinline__ void form(int tid) {
    const int tailW = scook[kCookWord] - kFormWords + 1;
    const int limit = tailW > 0 ? tailW : 0;
    form_desc_impl<kClauses, kFormWords, kThreadsT>(formLow, limit, tid, rowR, rowW, cA, cB, cC, cD,
                                                    tailMask, lastWord, sfR, sfW);
  }

  __device__ __forceinline__ void form_if_low(int tid) {
    if (formLow > 0 && scook[kCookWord] - formLow < kFormWords / 2) form(tid);
  }

  //: WARP ZERO ONLY.  One super-chunk's gather, or as much of it as the formed
  //: ring holds -- `cookOpen` says which, and `walk_wait` is what closes it.
  __device__ __forceinline__ void walk(int d, int lane) {
    uint16_t* lrd = iread(d);
    uint16_t* lwr = iwrite(d);
    uint8_t* ljt = ithr(d);
    const bool exists = cookOpen || cookWord >= 0;
    int nr = cookOpen ? cookNr : 0;
    int nw = cookOpen ? cookNw : 0;
    if (!cookOpen && cookWord >= 0) {
      chunkWin = cookWord / kWinWords;
      winBaseW = chunkWin * kWinWords;
    }
    bool closed = cookWord < 0;
    const uint32_t gt = lane == 31 ? 0u : ~((1u << (lane + 1)) - 1u);
#pragma unroll 1
    while (cookWord >= formLow) {
      if (cookWord < winBaseW) {
        if (nr != 0 || nw != 0) {
          closed = true;
          break;
        }
        --chunkWin;
        winBaseW -= kWinWords;
      }
      const int slot = cookWord & (kFormWords - 1);
      const uint32_t rm = sfR[slot];
      const uint32_t wm = sfW[slot];
      const int prx = bit_count(rm & gt);
      const int pwx = bit_count(wm & gt);
      const bool lvR = ((rm >> lane) & 1u) != 0u;
      const bool lvW = ((wm >> lane) & 1u) != 0u;
      const int cut = last_set(lane_ballot((lvR && nr + prx >= C) || (lvW && nw + pwx >= C)));
      const uint32_t keep = cut < 0 ? 0xFFFFFFFFu : (cut == 31 ? 0u : ~((1u << (cut + 1)) - 1u));
      const int t = (cookWord - winBaseW) * kTokBits + lane;
      if (lane > cut) {
        if (lvR) {
          lrd[C - 1 - (nr + prx)] = (uint16_t)t;
          ljt[C - 1 - (nr + prx)] = (uint8_t)(nw + pwx);
        }
        if (lvW) lwr[C - 1 - (nw + pwx)] = (uint16_t)t;
      }
      nr += bit_count(rm & keep);
      nw += bit_count(wm & keep);
      if (cut >= 0) {
        if (lane == 0) {
          sfR[slot] = rm & ~keep;
          sfW[slot] = wm & ~keep;
        }
        closed = true;
        break;
      }
      --cookWord;
      if (cookWord < 0) {
        closed = true;
        break;
      }
    }
    if (!closed) {
      cookNr = nr;
      cookNw = nw;
      cookOpen = true;
      if (lane == 0) {
        scook[kCookWord] = cookWord;
        scook[kCookOpen] = 1;
      }
      return;
    }
    //: THE DESCENDING FILL LEAVES THE LISTS AT THE BUFFER'S TOP, and the
    //: threshold it recorded is the count of write entries ABOVE the read
    //: entry -- both are resolved here, where the chunk's totals are known.
    uint16_t rv[kListPasses], wv[kListPasses];
    uint8_t jv[kListPasses];
#pragma unroll
    for (int k = 0; k < kListPasses; ++k) {
      const int i = lane + k * kTokBits;
      rv[k] = (i < nr) ? lrd[i + C - nr] : (uint16_t)0;
      wv[k] = (i < nw) ? lwr[i + C - nw] : (uint16_t)0;
      jv[k] = (i < nr) ? (uint8_t)(nw - ljt[i + C - nr]) : (uint8_t)0;
    }
    __syncwarp();
#pragma unroll
    for (int k = 0; k < kListPasses; ++k) {
      const int i = lane + k * kTokBits;
      if (i < C) {
        lrd[i] = rv[k];
        lwr[i] = wv[k];
        ljt[i] = jv[k];
      }
    }
    const bool wasOpen = cookOpen;
    cookOpen = false;
    if (lane == 0) {
      int* h = ihdr(d);
      h[0] = nr;
      h[1] = nw;
      h[2] = exists ? 1 : 0;
      h[3] = chunkWin;
      scook[kCookWord] = cookWord;
      if (wasOpen) scook[kCookOpen] = 0;
    }
  }

  //: EVERY THREAD.  A super-chunk that ran out of formed words is finished by
  //: re-forming and re-entering the walk; the barriers are CTA-uniform because
  //: the loop's condition is read out of shared.
  __device__ __forceinline__ void walk_wait(int d, int tid, int warp, int lane) {
#pragma unroll 1
    while (scook[kCookOpen]) {
      form(tid);
      rendezvous();
      if (warp == 0) walk(d, lane);
      rendezvous();
    }
  }
};

//: THE PARTITION PROBE.  Scan B's schedule is the one thing in P65 with no
//: forward counterpart to inherit, and its correctness is a SCHEDULE fact rather
//: than a numeric one: an arithmetic gate can only reach it through a reference
//: that replays the partition, which stops being writable the moment two blocks
//: of one head are live on different tokens.  So the walk is run alone and what
//: it published is emitted.  Per (bh, block, token): the ordinal of the chunk
//: that gathered the token and its position in that chunk's list, one plane per
//: side, plus the causal threshold the read entry carries.  `-1` is "not
//: gathered", which is what makes "exactly once" a checkable claim rather than a
//: covering one.  docs/internals/chunk/backward.md#partition-probe
template <int C, int D, int B, int BC>
__global__ __launch_bounds__(threads_for(BC)) void walk_probe_kernel(
    const uint32_t* __restrict__ tbl, int* __restrict__ stampR, int* __restrict__ stampW,
    int* __restrict__ jout, int* __restrict__ nchunk, int L) {
  using LP = LevelPlan<D, B, BC>;
  using G = GeomB<C, BC>;
  constexpr int kThreads = threads_for(BC);

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int blk = (int)blockIdx.x;
  const int tokWords = tok_words(L);

  extern __shared__ __align__(16) char smem_raw[];
  char* sidx = smem_raw;
  uint32_t* sfR = reinterpret_cast<uint32_t*>(sidx + G::idx_slot);
  uint32_t* sfW = sfR + kFormWords;
  int* scook = reinterpret_cast<int*>(sfW + kFormWords);

  Descent<C, LP::clauses, kThreads, G::idx_slot, G::hdr> dw{};
  dw.sidx = sidx;
  dw.sfR = sfR;
  dw.sfW = sfW;
  dw.scook = scook;
  dw.rowR = pin_address(tbl + ((long)(2 * blockIdx.y + 0) * LP::entries) * tokWords);
  dw.rowW = pin_address(tbl + ((long)(2 * blockIdx.y + 1) * LP::entries) * tokWords);
  dw.tokWords = tokWords;
  const uint32_t clauseStride = (uint32_t)tokWords * (uint32_t)sizeof(uint32_t);
#define ROLA_CLAUSE(CL_, C_)                                         \
  if constexpr ((CL_) < LP::clauses) {                               \
    constexpr int kDiv = LP::suffix_groups((CL_) + 1);               \
    constexpr int kGrp = LP::groups_at(CL_);                         \
    constexpr int kPre = LP::prefix_entries(CL_);                    \
    C_ = (uint32_t)(kPre + (blk / kDiv) % kGrp) * clauseStride;      \
  }
  ROLA_CLAUSE(0, dw.cA)
  ROLA_CLAUSE(1, dw.cB)
  ROLA_CLAUSE(2, dw.cC)
  ROLA_CLAUSE(3, dw.cD)
#undef ROLA_CLAUSE
  dw.tailMask = (L & (kTokBits - 1)) ? ((1u << (L & (kTokBits - 1))) - 1u) : 0xFFFFFFFFu;
  dw.lastWord = tokWords - 1;

  //: THE PLANE BASE IS FORMED ONCE AND PINNED, and every store inside the walk is
  //: a 32-bit byte displacement off it -- the same reason the forward's gathers
  //: are: a 64-bit index in a loop body is an IMAD.WIDE per iteration.
  const long row = (long)blockIdx.y * LP::blocks + blk;
  int* const sR = pin_address(stampR + row * L);
  int* const sW = pin_address(stampW + row * L);
  int* const sJ = pin_address(jout + row * L);
  int* const sN = pin_address(nchunk + row);

  dw.init(tid);
  rendezvous();
  dw.form(tid);
  rendezvous();

  int chunk = 0;
#pragma unroll 1
  for (;;) {
    if (warp == 0) dw.walk(0, lane);
    rendezvous();
    dw.walk_wait(0, tid, warp, lane);
    dw.form_if_low(tid);
    rendezvous();
    const int* h = dw.ihdr(0);
    if (!h[2]) break;
    const int nr = h[0], nw = h[1];
    const int tok0 = h[3] * kWindow;
    if (tid < C) {
      if (tid < nr) {
        const uint32_t off = ((uint32_t)tok0 + (uint32_t)dw.iread(0)[tid]) * 4u;
        *reinterpret_cast<int*>(gather_ptr(sR, off)) = chunk * C + tid;
        *reinterpret_cast<int*>(gather_ptr(sJ, off)) = (int)dw.ithr(0)[tid];
      }
      if (tid < nw)
        *reinterpret_cast<int*>(
            gather_ptr(sW, ((uint32_t)tok0 + (uint32_t)dw.iwrite(0)[tid]) * 4u)) = chunk * C + tid;
    }
    ++chunk;
    rendezvous();
  }
  if (tid == 0) *sN = chunk;
}

//: SCAN B.  One CTA per (bh, block), full grid: the sparse grid and the carried
//: state are not axes of the skeleton.
template <int C, int D, int B, int BC>
__global__ __launch_bounds__(threads_for(BC), (kScanBCtaTarget<C, BC>)) void scan_b_kernel(
    const __nv_bfloat16* __restrict__ pread, const __nv_bfloat16* __restrict__ pwrite,
    const __nv_bfloat16* __restrict__ gwrite, const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ dnum, const __nv_bfloat16* __restrict__ dden,
    const uint32_t* __restrict__ tbl, float* __restrict__ dv, float* __restrict__ dwsum,
    float* __restrict__ dgw, float* __restrict__ dpw, int L, int v_row_bytes) {
  using LP = LevelPlan<D, B, BC>;
  using G = GeomB<C, BC>;
  constexpr int GROUPS = groups_for(BC);
  constexpr int kThreads = threads_for(BC);
  constexpr int kGroupWarps = kGroupThreads / 32;
  constexpr int BCg = BC / GROUPS;
  static_assert(GROUPS == 1, "the S1 skeleton is written for the one-group carve");
  static_assert(C % kMmaM == 0, "chunk must be a multiple of the mma m quantum");
  static_assert(BCg % kMmaK == 0, "a group's slice must be a whole number of mma k-tiles");
  static_assert(C <= 255, "a causal threshold is one byte, chunk-relative");
  static_assert(design::smem_fits(G::bytes),
                "this arm does not fit this architecture's per-CTA shared-memory maximum");
  static_assert(kScanBCtaTarget<C, BC> == kCtaTarget<C, BC>,
                "SCAN B MOVED THE ARM'S OCCUPANCY: its shared-memory carve reaches a "
                "different CTA target than the forward's, so the two halves of one "
                "training step would run at different residencies. Re-carve rather "
                "than accept it.");
  constexpr int STAGES = G::stages;
  constexpr int AHEAD = STAGES - 1 + kAheadExtra;
  constexpr int IRING = AHEAD + 2;
  constexpr int LdV = G::LdV, LdS = G::LdS, LdO = G::LdO;
  constexpr int MTOK = C / kMmaM;
  constexpr int NTOK = C / kMmaN;
  constexpr int KD = BCg / kMmaK;
  constexpr int NDV = BCg / kMmaN;
  constexpr int kChunkV = kD / 8;
  //: the CHANNEL axis as an mma k: the d_v Gram and the per-slot dW both run it.
  constexpr int KV = kD / kMmaK;

  //: THE GRAM'S WARP SPLIT, the forward's: the C x C tile is cut over both token
  //: axes so every warp owns a rectangle of it.
  constexpr int MSPLIT = MTOK >= kGroupWarps ? kGroupWarps : MTOK;
  constexpr int NSPLIT_RAW = kGroupWarps / MSPLIT;
  constexpr int NSPLIT = NSPLIT_RAW < NTOK / 2 ? NSPLIT_RAW : NTOK / 2;
  constexpr int MPW = MTOK / MSPLIT;
  constexpr int NPW = NTOK / NSPLIT;
  constexpr int GWARPS = MSPLIT * NSPLIT;
  static_assert(NPW % 2 == 0, "n-tiles are consumed in ldmatrix pairs");
  //: THE WARP'S WHOLE n-RUN IS ONE PASS: the Gram's n axis is the chunk, which
  //: is small, so there is no second tile group to pipeline against.
  constexpr int GPAIR = NPW / 2;

  // ---- coefficient production geometry, the forward's -----------------------
  constexpr int kWtot = LP::col_base(LP::depth);
  constexpr int kDepth = LP::depth;
  constexpr int kInnerSpan = LP::span_at(kDepth - 1);
  constexpr int kOuterSpan = BC / kInnerSpan;
  constexpr int kLeadSpanG = kOuterSpan / GROUPS;
  constexpr int kRowThr = kGroupThreads / C;
  constexpr int kThrPerLead = kRowThr >= kLeadSpanG ? kRowThr / kLeadSpanG : 1;
  constexpr int kLeadPerThr = kLeadSpanG >= kRowThr ? kLeadSpanG / kRowThr : 1;
  constexpr int kTailPerThr = kInnerSpan / kThrPerLead;
  constexpr int kTailVec = (kTailPerThr * 2) / 16;
  constexpr int kFastSpan = kDepth >= 2 ? LP::span_at(kDepth - 2) : 1;
  static_assert(kLeadPerThr * kTailPerThr * kRowThr == BCg,
                "the row's threads must cover the slice exactly once");
  static_assert(kTailPerThr % 8 == 0,
                "a thread's trailing run is published in 16-byte vectors, so it "
                "must be a whole number of eight coefficients");
  static_assert(kLeadPerThr == 1 || kFastSpan % kLeadPerThr == 0,
                "a thread's outer run crosses a fast-level boundary");
  //: THE DIGIT REDUCTION IS WRITTEN FOR THE TWO-LEVEL ARM, and says so rather
  //: than mis-reducing a topology it was not derived for.
  static_assert(kDepth == 2, "the write-side digit reduction is the two-level form");
  static_assert(kOuterSpan * kInnerSpan == BCg, "the block is a rectangle in digit space");
  static_assert(kInnerSpan == 2 * (BCg / 4),
                "a quarter of the slots must be half an inner-level span, so a quarter "
                "fixes the outer digit and two quarters cover the inner one");

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int gwarp = warp % kGroupWarps;
  const int gtid = tid % kGroupThreads;
  const int blk = (int)blockIdx.x;

  extern __shared__ __align__(16) char smem_raw[];

  const long base = (long)blockIdx.y * L * kD;
  const long pbase = (long)blockIdx.y * L * kWtot;
  const int tokWords = tok_words(L);
  const int vRowElems = v_row_bytes / (int)sizeof(__nv_bfloat16);
  const int heads = vRowElems / kD;
  const __nv_bfloat16* const vPlane =
      pin_address(v + (long)(blockIdx.y / heads) * L * vRowElems + (long)(blockIdx.y % heads) * kD);
  const __nv_bfloat16* const dnumPlane = pin_address(dnum + base);
  const __nv_bfloat16* const ddenPlane = pin_address(dden + (long)blockIdx.y * L);
  float* const dvPlane = pin_address(dv + base);
  float* const dwsPlane = pin_address(dwsum + (long)blockIdx.y * L);
  float* const dgwPlane = pin_address(dgw + (long)blockIdx.y * L);
  float* const dpwPlane = pin_address(dpw + (long)blockIdx.y * L * kWtot);
  const __nv_bfloat16* const pRead = pin_address(pread + pbase);
  const __nv_bfloat16* const pWrite = pin_address(pwrite + pbase);
  const __nv_bfloat16* const gWrite = pin_address(gwrite + (long)blockIdx.y * L);
  const uint32_t* const rowR =
      pin_address(tbl + ((long)(2 * blockIdx.y + 0) * LP::entries) * tokWords);
  const uint32_t* const rowW =
      pin_address(tbl + ((long)(2 * blockIdx.y + 1) * LP::entries) * tokWords);
  const uint32_t clauseStride = (uint32_t)tokWords * (uint32_t)sizeof(uint32_t);

  __nv_bfloat16* sv = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* sdn = sv + STAGES * C * LdV;
  __nv_bfloat16* sr = sdn + STAGES * C * LdV;
  __nv_bfloat16* sw = sr + C * LdS;
  __nv_bfloat16* sg = sw + C * LdS;
  float* so = reinterpret_cast<float*>(sg + C * G::LdG);
  float* sgrad = reinterpret_cast<float*>(reinterpret_cast<char*>(so) + G::win_bytes);
  float* sdd = sgrad + kUnits * C;
  float* sgw = sdd + C;
  float* spw0 = sgw + C;
  char* sidx = reinterpret_cast<char*>(spw0 + 2 * C);
  //: THE READOUT WINDOW, RE-READ.  `so` stages `dv` until the drain frees it;
  //: `sdz` then publishes the entry mass column for `d gw` in the same bytes.
  //: One region, two phases, a barrier between --
  //: docs/internals/chunk/backward.md#readout-window
  float* const sdz = so;
  __nv_bfloat16* const sds = reinterpret_cast<__nv_bfloat16*>(sdz + BCg);
  uint32_t* sfR = reinterpret_cast<uint32_t*>(sidx + IRING * G::idx_slot);
  uint32_t* sfW = sfR + kFormWords;
  int* scook = reinterpret_cast<int*>(sfW + kFormWords);
  static_assert((2 * C * LdS + C * G::LdG) % 8 == 0, "the fp32 region must start 16-byte aligned");
  static_assert((G::win_bytes / (int)sizeof(float) + kUnits * C + C + C + 2 * C) % 4 == 0,
                "the index ring must start 16-byte aligned");
  static_assert(G::win_bytes >= G::late_bytes,
                "the readout window must cover every phase that overlays it");
  static_assert(G::idx_slot % 16 == 0,
                "the formed schedule follows the index ring and must stay aligned");

  //: THE SCHEDULE IS A COMPONENT, and the index slots are read through it, so
  //: the probe that gates the partition drives the same object this does.
  Descent<C, LP::clauses, kThreads, G::idx_slot, G::hdr> dw{};
  dw.sidx = sidx;
  dw.sfR = sfR;
  dw.sfW = sfW;
  dw.scook = scook;
  dw.rowR = rowR;
  dw.rowW = rowW;
  dw.tokWords = tokWords;
  auto ihdr = [&](int d) { return dw.ihdr(d); };
  auto iread = [&](int d) { return dw.iread(d); };
  auto iwrite = [&](int d) { return dw.iwrite(d); };
  auto ithr = [&](int d) { return dw.ithr(d); };

  // ---- lane-invariant operand addressing, the forward's ---------------------
  const int lr = lane & 7, tl = lane >> 3;
  const int rF = (tl & 1) * 8 + lr, cF = tl >> 1;
  const int rT = (tl >> 1) * 8 + lr, cT = tl & 1;
  const uint32_t rowFs = (uint32_t)((rF * LdS + cF * 8) * 2);
  const uint32_t rowTs = (uint32_t)((rT * LdS + cT * 8) * 2);
  const uint32_t rowFv = (uint32_t)((rF * LdV + cF * 8) * 2);
  const uint32_t rowTv = (uint32_t)((rT * LdV + cT * 8) * 2);
  //: A PANEL ADDRESS IS A BASE PLUS A DISPLACEMENT, and which of the two
  //: fragment maps it serves is carried by the BASE (`rowFs`/`rowTs`), never by
  //: the displacement -- so one displacement form covers both.
#define ADDR_S(b_, R_, g_) ((b_) + (uint32_t)(((R_) * LdS + (g_) * 8) * 2))

  const int dv0 = gwarp * 16;
#define ADDR_V(b_, R_) ((b_) + (uint32_t)(((R_) * LdV + dv0) * 2))
  //: the value-shaped panels read with a k-GROUP index rather than this unit's
  //: fixed channel slice -- the d_v Gram contracts the whole of `d_v`.
#define ADDR_VG(b_, R_, g_) ((b_) + (uint32_t)(((R_) * LdV + (g_) * 8) * 2))
#define ADDR_DS(b_, R_, g_) ((b_) + (uint32_t)(((R_) * (int)G::LdDs + (g_) * 8) * 2))
  //: THE GRAM'S ROWS ARE READ TOKENS AND ITS COLUMNS WRITE TOKENS, so the two
  //: fragment maps land on `LdG` exactly as they land on `LdS`.
#define ADDR_G(b_, R_, g_) ((b_) + (uint32_t)(((R_) * (int)G::LdG + (g_) * 8) * 2))
  const int gm0 = (gwarp / NSPLIT) * MPW;
  const int gn0 = (gwarp % NSPLIT) * NPW;

  //: THE ADJOINT STATE, transposed exactly as the forward's state is:
  //: m = d_v (this unit's 16 rows), n = this block's slots.
  float dst[NDV][4];
#pragma unroll
  for (int j = 0; j < NDV; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) dst[j][e] = 0.f;

  //: THE ADJOINT MASS COLUMN, ONE SLOT PER THREAD.  The forward rides its mass
  //: channel on the value mma, which spends a 16-row m quantum to carry ONE real
  //: row -- sixteen registers per lane for one slot's worth of information. Scan
  //: B cannot afford that beside the adjoint state, and it does not need to: `dz`
  //: is a per-SLOT quantity and there are exactly as many slots as threads, so it
  //: is a scalar and its fold is a dot product over the chunk's read entries.
  //: The digit reduction wants it in this layout too.
  //: docs/internals/chunk/backward.md#mass-column
  static_assert(BCg == kGroupThreads,
                "the adjoint mass column is one slot per thread, so a group's slice "
                "must be exactly its thread count");
  float dz_s = 0.f;

  // ---- global -> shared staging: V by the WRITE list, dnum by the READ list --
  constexpr int kRowsPerPass = kThreads / kChunkV;
  constexpr int kPasses = C / kRowsPerPass;
  static_assert(kPasses >= 1, "a chunk must be at least kThreads/8 rows");
  const int ldRow = tid / kChunkV, ldCol = (tid % kChunkV) * 8;
  const uint32_t ldColB = (uint32_t)(ldCol * 2);
  const uint32_t kVRowB = (uint32_t)v_row_bytes;
  constexpr uint32_t kDnRowB = (uint32_t)(kD * (int)sizeof(__nv_bfloat16));
  const uint32_t stgOff = (uint32_t)((ldRow * LdV + ldCol) * 2);

  constexpr uint32_t kSlot = (uint32_t)(C * LdV) * 2;
  constexpr uint32_t kRing = STAGES * kSlot;
  const uint32_t svL = smem_addr(sv) + stgOff;
  const uint32_t sdnL = smem_addr(sdn) + stgOff;
  const uint32_t svB = smem_addr(sv);
  const uint32_t sdnB = smem_addr(sdn);
  uint32_t stgSlot = 0, cmpSlot = 0;

  //: ONE BASE, TWO ROLES: the R panel is the Gram's A operand and the fold's B
  //: operand, and both want the same lane map, so there is one address to keep.
  const uint32_t brF = smem_addr(sr) + rowFs;
  const uint32_t bwT = smem_addr(sw) + rowTs;
  const uint32_t bgF = smem_addr(sg) + (uint32_t)((rF * (int)G::LdG + cF * 8) * 2);
  const uint32_t sgT = smem_addr(sg) + (uint32_t)((rT * (int)G::LdG + cT * 8) * 2);
  const uint32_t bdsT = smem_addr(sds) + (uint32_t)((rT * (int)G::LdDs + cT * 8) * 2);

  // ---- the fan-in's addressing, the forward's ------------------------------
  constexpr int kFanLanes = 32;
  constexpr int kFanPerThr = kD / kFanLanes;
  constexpr int kFanRows = kGroupThreads / kFanLanes;
  constexpr int kFanPasses = C / kFanRows;
  static_assert(kFanPasses * kFanRows == C && kFanPasses >= 1,
                "the fan-in's passes must cover the chunk exactly");
  const int fanRow = gtid / kFanLanes, fanCol = gtid % kFanLanes;
  const uint32_t fanColB = (uint32_t)(fanCol * (int)sizeof(float));
  constexpr uint32_t kORowB = (uint32_t)(kD * (int)sizeof(float));
  const int fanE = fanRow * LdO + fanCol;

  const int oq = (lane & 3) * 2, om = lane >> 2;
  //: ONE STAGING OFFSET, NOT FOUR.  The accumulator's other three entries sit at
  //: compile-time displacements of the first (`+LdO` down the token axis, `+8`
  //: across the channel axis), and holding them as separate live values is three
  //: registers this kernel does not have -- docs/internals/chunk/backward.md#residency
  const int oE0 = oq * LdO + dv0 + om;

  //: THE STAGE IS WHAT IS SKIPPED; THE RING SLOT IS NOT.  Both panels ride one
  //: slot cursor, so the wait/commit accounting stays the forward's.
  auto stage_chunk = [&](int d, bool goW, bool goR) {
    const uint32_t win = (uint32_t)ihdr(d)[3];
    if (goW) {
      const uint16_t* lw = iwrite(d);
      const uint32_t winCol = ldColB + win * (kWindow * kVRowB);
#pragma unroll
      for (int p = 0; p < kPasses; ++p) {
        const uint32_t off = (uint32_t)lw[ldRow + p * kRowsPerPass] * kVRowB + winCol;
        stage_tile(svL + stgSlot + (uint32_t)(p * kRowsPerPass * LdV * 2), gather_ptr(vPlane, off));
      }
    }
    if (goR) {
      const uint16_t* lrd = iread(d);
      const uint32_t winCol = ldColB + win * (kWindow * kDnRowB);
#pragma unroll
      for (int p = 0; p < kPasses; ++p) {
        const uint32_t off = (uint32_t)lrd[ldRow + p * kRowsPerPass] * kDnRowB + winCol;
        stage_tile(sdnL + stgSlot + (uint32_t)(p * kRowsPerPass * LdV * 2),
                   gather_ptr(dnumPlane, off));
      }
    }
    stgSlot += kSlot;
    if (stgSlot == kRing) stgSlot = 0;
  };

  //: dv's FAN-IN, gathered by the WRITE list: the partial staged for write entry
  //: `j` belongs to token `w[j]`, and entries past `nw` have no token.
  auto drain_dv = [&](int d, int nw) {
    const uint16_t* lw = iwrite(d);
    const uint32_t winFan = fanColB + (uint32_t)ihdr(d)[3] * (uint32_t)(kWindow * kORowB);
#pragma unroll
    for (int p = 0; p < kFanPasses; ++p) {
      const int row = fanRow + p * kFanRows;
      if (row < nw) {
        float* dp =
            reinterpret_cast<float*>(gather_ptr(dvPlane, (uint32_t)lw[row] * kORowB + winFan));
        const int e0 = fanE + p * kFanRows * LdO;
#pragma unroll
        for (int e = 0; e < kFanPerThr; ++e) fan_in_add(dp + e * kFanLanes, so[e0 + e * kFanLanes]);
      }
    }
  };

  //: d gw's FAN-IN.  Same shape as the one below and the same reason: every
  //: contributor to a token's gain gradient is a different unit or a different
  //: block, and none of them holds the whole sum.
  auto drain_dgw = [&](int d, int nw) {
    if (gtid < C && gtid < nw) {
      float s = 0.f;
#pragma unroll
      for (int u = 0; u < kUnits; ++u) s += sgrad[u * C + gtid];
      fan_in_add(
          reinterpret_cast<float*>(gather_ptr(
              dgwPlane, (uint32_t)iwrite(d)[gtid] * 4u
                            + (uint32_t)ihdr(d)[3] * (uint32_t)(kWindow * (int)sizeof(float)))),
          s);
    }
  };

  //: THE UNITS' PARTIAL CONTRACTIONS MEET HERE, which is the same shape the
  //: forward's denominator drain has and for the same reason: the contraction's
  //: k axis is the carve's axis, so no unit holds the whole sum.
  //: WHAT DRAINS HERE IS `sum_s dW[j,s]`, NOT `d gw` -- see
  //: docs/internals/chunk/backward.md#dw-slotsum
  auto drain_dwsum = [&](int d, int nw) {
    if (gtid < C && gtid < nw) {
      float s = 0.f;
#pragma unroll
      for (int u = 0; u < kUnits; ++u) s += sgrad[u * C + gtid];
      fan_in_add(
          reinterpret_cast<float*>(gather_ptr(
              dwsPlane, (uint32_t)iwrite(d)[gtid] * 4u
                            + (uint32_t)ihdr(d)[3] * (uint32_t)(kWindow * (int)sizeof(float)))),
          s);
    }
  };

  // ---- routed Kronecker coefficient production, the forward's --------------
  const int prow = gtid / kRowThr;
  const int rIn = gtid % kRowThr;
  const int leadLocal = (rIn / kThrPerLead) * kLeadPerThr;
  const int tailLocal = (rIn % kThrPerLead) * kTailPerThr;
  constexpr uint32_t kPRowB = (uint32_t)(kWtot * (int)sizeof(__nv_bfloat16));
  //: the digit gradients are fp32, so their row stride is not the amplitudes'.
  constexpr uint32_t kPRowW = (uint32_t)(kWtot * (int)sizeof(float));
  constexpr uint32_t kOneSplat = 0x3F803F80u;
  const int oBase = leadLocal;
  [[maybe_unused]] uint32_t colFast = 0;
  if constexpr (kDepth >= 2)
    colFast =
        (uint32_t)((LP::col_base(kDepth - 2) + LP::off_at(kDepth - 2, blk) + oBase % kFastSpan)
                   * 2);
  const uint32_t colInner =
      (uint32_t)((LP::col_base(kDepth - 1) + LP::off_at(kDepth - 1, blk) + tailLocal) * 2);
  constexpr int kUppers = kDepth - 2;
  [[maybe_unused]] uint32_t colUpA = 0, colUpB = 0;
  if constexpr (kUppers >= 1) {
    constexpr int lA = kDepth - 3;
    colUpA =
        (uint32_t)((LP::col_base(lA) + LP::off_at(lA, blk) + (oBase / kFastSpan) % LP::span_at(lA))
                   * 2);
  }
  if constexpr (kUppers >= 2) {
    constexpr int lB = kDepth - 4;
    colUpB = (uint32_t)((LP::col_base(lB) + LP::off_at(lB, blk)
                         + (oBase / (kFastSpan * LP::span_at(kDepth - 3))) % LP::span_at(lB))
                        * 2);
  }
  const int slot0 = leadLocal * kInnerSpan + tailLocal;
  char* dstR = reinterpret_cast<char*>(sr) + prow * (int)(LdS * 2) + slot0 * 2;
  char* dstW = reinterpret_cast<char*>(sw) + prow * (int)(LdS * 2) + slot0 * 2;
  //: THE PRODUCTION IS A COMPONENT -- chunk_kernel.cuh's `Producer`, which the
  //: forward drives too.  The column offsets and slot displacements above are this
  //: carve's; the production itself is everyone's.
  Producer<kDepth, kLeadPerThr, kTailVec, kUppers> prod{colFast, colInner, colUpA, colUpB, {}};
#pragma unroll
  for (int j = 0; j < kLeadPerThr; ++j)
#pragma unroll
    for (int i = 0; i < kTailVec; ++i) prod.pOff[j][i] = (j * kInnerSpan + i * 8) * 2;
  auto produce = [&](const __nv_bfloat16* bs, char* dst, uint32_t tokOff, bool have, uint32_t gsp) {
    prod.emit(bs, dst, tokOff, have, gsp);
  };

  // ---- the membership bits, the forward's clause rows ----------------------
#define ROLA_CLAUSE(CL_, C_)                                         \
  if constexpr ((CL_) < LP::clauses) {                               \
    constexpr int kDiv = LP::suffix_groups((CL_) + 1);               \
    constexpr int kGrp = LP::groups_at(CL_);                         \
    constexpr int kPre = LP::prefix_entries(CL_);                    \
    C_ = (uint32_t)(kPre + (blk / kDiv) % kGrp) * clauseStride;      \
  }
  ROLA_CLAUSE(0, dw.cA)
  ROLA_CLAUSE(1, dw.cB)
  ROLA_CLAUSE(2, dw.cC)
  ROLA_CLAUSE(3, dw.cD)
#undef ROLA_CLAUSE
  dw.tailMask = (L & (kTokBits - 1)) ? ((1u << (L & (kTokBits - 1))) - 1u) : 0xFFFFFFFFu;
  dw.lastWord = tokWords - 1;

  dw.init(tid);
  rendezvous();
  dw.form(tid);
  rendezvous();
#pragma unroll 1
  for (int s = 0; s < AHEAD; ++s) {
    if (warp == 0) dw.walk(s, lane);
    rendezvous();
    dw.walk_wait(s, tid, warp, lane);
    dw.form_if_low(tid);
    rendezvous();
  }
#pragma unroll 1
  for (int s = 0; s < STAGES - 1; ++s) {
    stage_chunk(s, ihdr(s)[1] != 0, ihdr(s)[0] != 0);
    stage_commit();
  }

  int dc = 0, ds = STAGES - 1, dn = AHEAD;
#pragma unroll 1
  for (;;) {
    const int* h = ihdr(dc);
    const int nr = h[0], nw = h[1];
    if (!h[2]) break;
    const bool liveR = nr != 0, liveW = nw != 0;
    const bool live = liveR || liveW;
    const uint32_t winPc = (uint32_t)h[3] * (uint32_t)(kWindow * kWtot * 2);
    const uint32_t winPw = (uint32_t)h[3] * (uint32_t)(kWindow * kWtot * 4);

    if (warp == 0) dw.walk(dn, lane);

    if (live) {
      const uint32_t winP = (uint32_t)ihdr(dc)[3] * (uint32_t)(kWindow * kWtot * 2);
      const uint32_t rOff = (uint32_t)iread(dc)[prow] * kPRowB + winP;
      const uint32_t wOff = (uint32_t)iwrite(dc)[prow] * kPRowB + winP;
      const uint32_t winG = (uint32_t)ihdr(dc)[3] * (uint32_t)(kWindow * 2);
      const uint32_t winP0 = winP;
      //: BOTH PANELS ARE PRODUCED UNGAINED.  The write gain is a per-TOKEN scale
      //: and the readout's `n` axis is the token, so applying it once at the `so`
      //: store is the same number -- and it leaves the level product alone, which
      //: is the operand (A.11) needs and the one a decay clock will tap.
      produce(pRead, dstR, rOff, prow < nr, kOneSplat);
      produce(pWrite, dstW, wOff, prow < nw, kOneSplat);
      //: THE PER-CHUNK ROWS, STAGED ONCE: the mass seed per READ entry and the
      //: gain per WRITE entry, both exactly zero past their counts so the
      //: reductions below can run the whole chunk unbounded.
      if (tid < C) {
        sdd[tid] = tid < nr
                       ? __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(gather_ptr(
                             ddenPlane, (uint32_t)iread(dc)[tid] * 2u
                                            + (uint32_t)ihdr(dc)[3] * (uint32_t)(kWindow * 2))))
                       : 0.f;
        sgw[tid] = tid < nw ? __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
                                  gather_ptr(gWrite, (uint32_t)iwrite(dc)[tid] * 2u + winG)))
                            : 0.f;
#pragma unroll
        for (int o = 0; o < kOuterSpan; ++o) {
          const uint32_t col = (uint32_t)((LP::col_base(0) + LP::off_at(0, blk) + o) * 2) + winP0;
          spw0[o * C + tid] =
              tid < nw ? __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
                             gather_ptr(pWrite, (uint32_t)iwrite(dc)[tid] * kPRowB + col)))
                       : 0.f;
        }
      }
    }

    stage_wait<STAGES - 2>();
    rendezvous();  // (1) panels and both staged tiles visible; the previous
                   //     chunk's `so`/`sgrad` are dead; the walk has published

    dw.walk_wait(dn, tid, warp, lane);
    dw.form_if_low(tid);
    stage_chunk(ds, ihdr(ds)[1] != 0, ihdr(ds)[0] != 0);
    stage_commit();

    const uint32_t atv = svB + cmpSlot + rowFv;
    const uint32_t adn = sdnB + cmpSlot + rowTv;
    //: the SAME V panel, addressed scalar-wise: `d gw`'s value half reads it one
    //: element at a time against the readout accumulator, not as an mma operand.
    const __nv_bfloat16* const vpan = sv + (cmpSlot >> 1);
    //: the d_v GRAM's operands: `dnum` as A (m = read token, k = channel) and V as
    //: B (k = channel, n = write token) -- the same pair the fold and the readout
    //: already stage, read through the other lane map.
    const uint32_t adnF = sdnB + cmpSlot + rowFv;
    const uint32_t avT = svB + cmpSlot + rowTv;
    cmpSlot += kSlot;
    if (cmpSlot == kRing) cmpSlot = 0;

    // ---- THE SLOT GRAM: G = staircase(R . W^T), the forward's, verbatim ------
    //: `dv[j]` reads the adjoint state the WRITE token j leaves the chunk with,
    //: which is the entry state plus every read token ABOVE j.  Factoring that
    //: sum over slots is this Gram -- the forward's own, contracted over slots --
    //: and the mask is the forward's staircase unchanged: read entry `tr` keeps
    //: write entries `tc < J[tr]`.
    if (liveR && liveW && gwarp < GWARPS) {
      const uint8_t* ljt = ithr(dc);
      //: THE OPERAND TILES ARE NOT HOISTED, and that is a REGISTER decision the
      //: forward does not have to make: Scan B carries the adjoint state through
      //: this block, so holding both panels' whole k-runs live (2 * KD * 4
      //: registers) puts sm_80's allocator over its bound and spills. One k-tile
      //: of each at a time costs nothing at this arm -- `MPW` is one, so nothing
      //: is re-read -- docs/internals/chunk/backward.md#two-grams
#pragma unroll
      for (int m = 0; m < MPW; ++m) {
        float g[GPAIR][2][4];
#pragma unroll
        for (int u = 0; u < GPAIR; ++u)
#pragma unroll
          for (int h2 = 0; h2 < 2; ++h2)
#pragma unroll
            for (int e = 0; e < 4; ++e) g[u][h2][e] = 0.f;
#pragma unroll
        for (int kk = 0; kk < KD; ++kk) {
          uint32_t qa[4];
          load_frag(qa, ADDR_S(brF, (gm0 + m) * 16, 2 * kk));
#pragma unroll
          for (int u = 0; u < GPAIR; ++u) {
            uint32_t bf[4];
            load_frag(bf, ADDR_S(bwT, (gn0 + u * 2) * 8, 2 * kk));
            mma(g[u][0], qa, &bf[0]);
            mma(g[u][1], qa, &bf[2]);
          }
        }
        const int tr = (gm0 + m) * 16 + (lane >> 2);
        //: THE THRESHOLDS ARE READ AT THE MASK, not before the contraction: two
        //: values held across the whole k-run are two registers at a ceiling.
        const int j0 = ljt[tr], j1 = ljt[tr + 8];
#pragma unroll
        for (int u = 0; u < GPAIR; ++u)
#pragma unroll
          for (int h2 = 0; h2 < 2; ++h2) {
            const int tc = (gn0 + u * 2 + h2) * 8 + (lane & 3) * 2;
            store_bf16x2(&sg[tr * G::LdG + tc], pack_bf16x2(tc + 0 < j0 ? g[u][h2][0] : 0.f,
                                                            tc + 1 < j0 ? g[u][h2][1] : 0.f));
            store_bf16x2(&sg[(tr + 8) * G::LdG + tc], pack_bf16x2(tc + 0 < j1 ? g[u][h2][2] : 0.f,
                                                                  tc + 1 < j1 ? g[u][h2][3] : 0.f));
          }
      }
    }

    rendezvous();  // (1b) the slot Gram is visible to every unit

    //: BOTH WRITE-SIDE TERMS READ THE STATE THIS CHUNK ENTERS WITH -- the
    //: adjoint arriving from the FUTURE -- so both run before the fold.
    //: d gw's value half outlives the readout's accumulator: one partial per
    //: (token pair, n-tile), summed over this unit's channels, carried from the
    //: readout to the fan-in phase two barriers later.
    float pa[NTOK][2];
#pragma unroll
    for (int j = 0; j < NTOK; ++j) pa[j][0] = pa[j][1] = 0.f;

    if (liveW) {
      //: dv's WHOLE TERM IS ONE SCOPE, so the accumulator and both operand
      //: fragments are dead before the channel contraction opens.  At this
      //: register ceiling a live range that merely OUTLIVES its use is a spill.
      {
        // ---- (A.17) dv^T = dS^T . W^T, the forward's readout with the write panel
        float acc[NTOK][4];
#pragma unroll
        for (int j = 0; j < NTOK; ++j)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[j][e] = 0.f;
        //: NEITHER OPERAND IS HOISTED, and that is a REGISTER decision the forward
        //: does not have to make.  The whole packed adjoint (`2 * KD * 4` registers)
        //: plus the whole W k-run put sm_80's allocator over its bound and spilled;
        //: the pack is a register op repeated `MTOK` times and the panel tile is an
        //: `ldmatrix` the loop was going to issue anyway, so both are re-formed per
        //: k-tile -- docs/internals/chunk/backward.md#residency
#pragma unroll
        for (int m = 0; m < MTOK; ++m)
#pragma unroll
          for (int kk = 0; kk < KD; ++kk) {
            uint32_t da[4];
            da[0] = pack_bf16x2(dst[2 * kk][0], dst[2 * kk][1]);
            da[1] = pack_bf16x2(dst[2 * kk][2], dst[2 * kk][3]);
            da[2] = pack_bf16x2(dst[2 * kk + 1][0], dst[2 * kk + 1][1]);
            da[3] = pack_bf16x2(dst[2 * kk + 1][2], dst[2 * kk + 1][3]);
            uint32_t wb[4];
            load_frag(wb, ADDR_S(bwT, m * 16, 2 * kk));
#pragma unroll
            for (int p = 0; p < 2; ++p) mma(acc[2 * m + p], da, &wb[2 * p]);
          }
        // ---- dv's INTRA term: dv^T += dnum^T . G, the same accumulator --------
        //: The staircase already lives in `G`, so this is the fold's own mma with
        //: the Gram in the W panel's place and the READ token axis as its k.
        if (liveR) {
#pragma unroll
          for (int kk = 0; kk < MTOK; ++kk) {
            uint32_t a[4];
            load_frag_t(a, ADDR_V(adn, kk * 16));
#pragma unroll
            for (int nj = 0; nj < NTOK / 2; ++nj) {
              uint32_t b[4];
              load_frag_t(b, ADDR_G(bgF, kk * 16, 2 * nj));
              mma(acc[2 * nj], a, &b[0]);
              mma(acc[2 * nj + 1], a, &b[2]);
            }
          }
        }

        //: d gw's VALUE HALF, taken here because `acc` IS the ungained readout
        //: `sum_s U[j,s] dS_after[j,s,:]` and (A.11) contracts exactly that
        //: against `v[j,:]`.  The contraction runs over THIS unit's channels, so
        //: what leaves the shuffle is a partial and the four units meet at the
        //: fan-in -- docs/internals/chunk/backward.md#dw-slotsum
#pragma unroll
        for (int j = 0; j < NTOK; ++j) {
          const int t0 = oq + j * 8;
          const int c0 = dv0 + om;
          pa[j][0] = acc[j][0] * __bfloat162float(vpan[t0 * LdV + c0])
                     + acc[j][2] * __bfloat162float(vpan[t0 * LdV + c0 + 8]);
          pa[j][1] = acc[j][1] * __bfloat162float(vpan[(t0 + 1) * LdV + c0])
                     + acc[j][3] * __bfloat162float(vpan[(t0 + 1) * LdV + c0 + 8]);
        }
        //: THE GAIN LANDS HERE AND NOWHERE ELSE.  `gw[j]` is uniform along the
        //: accumulator's token axis, so one multiply per stored entry turns the
        //: ungained readout into `dv`.
#pragma unroll
        for (int j = 0; j < NTOK; ++j) {
          const int jo = j * 8 * LdO;
          const float g0 = sgw[oq + j * 8], g1 = sgw[oq + j * 8 + 1];
          store_f32(&so[oE0 + jo], acc[j][0] * g0);
          store_f32(&so[oE0 + jo + LdO], acc[j][1] * g1);
          store_f32(&so[oE0 + jo + 8], acc[j][2] * g0);
          store_f32(&so[oE0 + jo + LdO + 8], acc[j][3] * g1);
        }
      }

      // ---- (A.18) the CHANNEL CONTRACTION: dW[j,s] = <dS[s,:], v[j,:]> + dz[s],
      //      reduced over this block's slots, which is d gw's own shape.
      uint32_t va[MTOK][4];
#pragma unroll
      for (int m = 0; m < MTOK; ++m) load_frag(va[m], ADDR_V(atv, m * 16));
      float tok[MTOK][2];
#pragma unroll
      for (int m = 0; m < MTOK; ++m) {
        tok[m][0] = 0.f;
        tok[m][1] = 0.f;
      }
      //: TILE BY TILE, REDUCED IN PLACE.  Holding the whole C x BCg gradient
      //: accumulator is the measured alternative and it does not fit -- neither
      //: in the register budget nor beside the ring (docs/internals/chunk/backward.md).
      //: THIS IS THE INTER-CHUNK PART ONLY.  Its intra half rides a SECOND Gram
      //: whose write side carries a ones column, so the staircase covers the mass
      //: channel as well as the value channels -- the design's (A.19) drops that
      //: term -- docs/internals/chunk/backward.md#mass-gram
#pragma unroll
      for (int n = 0; n < NDV; ++n) {
        uint32_t db[2];
        db[0] = transpose_frag_b16(pack_bf16x2(dst[n][0], dst[n][1]));
        db[1] = transpose_frag_b16(pack_bf16x2(dst[n][2], dst[n][3]));
        float g[MTOK][4];
#pragma unroll
        for (int m = 0; m < MTOK; ++m) {
#pragma unroll
          for (int e = 0; e < 4; ++e) g[m][e] = 0.f;
          mma(g[m], va[m], db);
          tok[m][0] += g[m][0] + g[m][1];
          tok[m][1] += g[m][2] + g[m][3];
        }
      }
      //: The mass column's share is slot-only, so it is ONE number per unit and
      //: it lands on every token of the chunk exactly once through the sum below.
      //: Its slots are spread over the whole unit -- the padding rows are
      //: identically zero, so the reduction needs no lane predicate, only every
      //: lane.
      float dzs = dz_s;
#pragma unroll
      for (int b2 = 1; b2 < 32; b2 <<= 1) dzs += __shfl_xor_sync(0xFFFFFFFFu, dzs, b2);
#pragma unroll
      for (int m = 0; m < MTOK; ++m) {
#pragma unroll
        for (int b2 = 1; b2 < 4; b2 <<= 1) {
          tok[m][0] += __shfl_xor_sync(0xFFFFFFFFu, tok[m][0], b2);
          tok[m][1] += __shfl_xor_sync(0xFFFFFFFFu, tok[m][1], b2);
        }
      }
      if ((lane & 3) == 0) {
#pragma unroll
        for (int m = 0; m < MTOK; ++m) {
          store_f32(&sgrad[gwarp * C + m * 16 + (lane >> 2)], tok[m][0] + dzs);
          store_f32(&sgrad[gwarp * C + m * 16 + (lane >> 2) + 8], tok[m][1] + dzs);
        }
      }
    }

    rendezvous();  // (2) `so` and `sgrad` are visible to every unit

    if (liveW) {
      drain_dv(dc, nw);
      drain_dwsum(dc, nw);
    }

    rendezvous();  // (2b) `so` and `sgrad` are drained; the window is free

    //: `dz` AS THIS CHUNK ENTERS IT -- the adjoint arriving from the future, which
    //: is what every write-side term reads.  The fold updates it at the end of the
    //: iteration, so this is the entry value, and it goes into the window the
    //: readout has just given up.
    sdz[gtid] = dz_s;

    rendezvous();  // (2b') the entry mass column is visible

    //: (A.11) d gw[j] = sum_s dW[j,s] * U[j,s], assembled from three partials that
    //: all land in `sgrad` and all meet at the same fan-in:
    //:   A  <acc[j,:], v[j,:]>            this unit's channels   (taken above)
    //:   B  sum_s U[j,s] * dz_a[s]        this quarter's slots
    //:   H  sum_i [j<J(i)] dden[i] Gu[i,j]  this quarter's read entries
    //: A is per-UNIT and B/H are per-QUARTER, and the drain sums four of each --
    //: which is why A must be stored once per unit and B/H added once.
    if (liveW) {
      sgrad[gwarp * C + lane] = 0.f;
      __syncwarp();
      //: only the lane that owns channel row zero publishes; every other lane in
      //: its column holds the same shuffled total.
#pragma unroll
      for (int j = 0; j < NTOK; ++j) {
        float p0 = pa[j][0], p1 = pa[j][1];
#pragma unroll
        for (int b2 = 4; b2 < 32; b2 <<= 1) {
          p0 += __shfl_xor_sync(0xFFFFFFFFu, p0, b2);
          p1 += __shfl_xor_sync(0xFFFFFFFFu, p1, b2);
        }
        if (om == 0) {
          sgrad[gwarp * C + oq + j * 8] = p0;
          sgrad[gwarp * C + oq + j * 8 + 1] = p1;
        }
      }
      __syncwarp();
      //: FOUR THREADS PER TOKEN, and the token is this warp's own eighth of the
      //: chunk, so the store below is the only writer of its cell.
      {
        const int j = gtid / 4, part = gtid % 4;
        constexpr int kSlotRun = BCg / 4;
        constexpr int kReadRun = C / 4;
        float pb = 0.f;
#pragma unroll
        for (int s = 0; s < kSlotRun; ++s) {
          const int sl = part * kSlotRun + s;
          pb += __bfloat162float(sw[j * LdS + sl]) * sdz[sl];
        }
        if (liveR) {
          //: `Gu` is stored ALREADY MASKED, so the staircase is not applied twice.
#pragma unroll
          for (int i = 0; i < kReadRun; ++i) {
            const int ir = part * kReadRun + i;
            pb += sdd[ir] * __bfloat162float(sg[ir * G::LdG + j]);
          }
        }
#pragma unroll
        for (int b2 = 1; b2 < 4; b2 <<= 1) pb += __shfl_xor_sync(0xFFFFFFFFu, pb, b2);
        if (part == 0) sgrad[gwarp * C + j] += pb;
      }
    }

    rendezvous();  // (2c) every unit's d gw partial is visible

    if (liveW) drain_dgw(dc, nw);

    rendezvous();  // (2d) `sgrad` is drained and the slot Gram is dead

    // ---- THE d_v GRAM, with the R-b ONES COLUMN ------------------------------
    //: `Gd'[i,j] = <dnum[i,:], v[j,:]> + dden[i]`.  The `+ dden[i]` is the design's
    //: (A.19) correction: `dW` differentiates BOTH uses of `W`, so the adjoint
    //: entering write entry `j` carries the in-chunk mass as well as the state,
    //: and a ones column on the write side puts both under ONE staircase.  It is
    //: a row broadcast of the `dden` row already staged, not a second contraction
    //: -- docs/internals/chunk/backward.md#mass-gram
    if (liveR && liveW && gwarp < GWARPS) {
      const uint8_t* ljt = ithr(dc);
#pragma unroll
      for (int m = 0; m < MPW; ++m) {
        float g[GPAIR][2][4];
#pragma unroll
        for (int u = 0; u < GPAIR; ++u)
#pragma unroll
          for (int h2 = 0; h2 < 2; ++h2)
#pragma unroll
            for (int e = 0; e < 4; ++e) g[u][h2][e] = 0.f;
#pragma unroll
        for (int kk = 0; kk < KV; ++kk) {
          uint32_t qa[4];
          load_frag(qa, ADDR_VG(adnF, (gm0 + m) * 16, 2 * kk));
#pragma unroll
          for (int u = 0; u < GPAIR; ++u) {
            uint32_t bf[4];
            load_frag(bf, ADDR_VG(avT, (gn0 + u * 2) * 8, 2 * kk));
            mma(g[u][0], qa, &bf[0]);
            mma(g[u][1], qa, &bf[2]);
          }
        }
        const int tr = (gm0 + m) * 16 + (lane >> 2);
        const int j0 = ljt[tr], j1 = ljt[tr + 8];
        const float m0 = sdd[tr], m1 = sdd[tr + 8];
#pragma unroll
        for (int u = 0; u < GPAIR; ++u)
#pragma unroll
          for (int h2 = 0; h2 < 2; ++h2) {
            const int tc = (gn0 + u * 2 + h2) * 8 + (lane & 3) * 2;
            store_bf16x2(&sg[tr * G::LdG + tc], pack_bf16x2(tc + 0 < j0 ? g[u][h2][0] + m0 : 0.f,
                                                            tc + 1 < j0 ? g[u][h2][1] + m0 : 0.f));
            store_bf16x2(&sg[(tr + 8) * G::LdG + tc],
                         pack_bf16x2(tc + 0 < j1 ? g[u][h2][2] + m1 : 0.f,
                                     tc + 1 < j1 ? g[u][h2][3] + m1 : 0.f));
          }
      }
    }

    rendezvous();  // (2e) the d_v Gram is visible

    // ---- THE PER-SLOT dW, A QUARTER OF THE SLOTS AT A TIME -------------------
    //: `u1` is the INNER level's digit gradient and needs no reduction at all:
    //: quarter `p` fixes the outer digit, and each `(token, inner digit)` cell is
    //: held by exactly one lane.  `u0` is the OUTER level's and reduces over the
    //: lane's slot run and then over the units.
    float u1[2][MTOK][4];
    float u0[kOuterSpan][MTOK][2];
#pragma unroll
    for (int c = 0; c < 2; ++c)
#pragma unroll
      for (int m = 0; m < MTOK; ++m)
#pragma unroll
        for (int e = 0; e < 4; ++e) u1[c][m][e] = 0.f;
#pragma unroll
    for (int o = 0; o < kOuterSpan; ++o)
#pragma unroll
      for (int m = 0; m < MTOK; ++m)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) u0[o][m][hh] = 0.f;

    if (liveW) {
      //: THE QUARTER INDEX IS COMPILE-TIME, and that is not a scheduling
      //: preference: `u0`/`u1` are indexed by the quarter's own digit class, and a
      //: runtime index into a register array is a stack frame -- the same rule the
      //: forward's clause scalars are written to.
#pragma unroll
      for (int pq = 0; pq < 4; ++pq) {
        //: EACH UNIT WRITES ITS OWN CHANNELS, so the writes are disjoint and the
        //: cross-unit sum (A.18)'s k axis would otherwise force does not exist.
#pragma unroll
        for (int q = 0; q < BCg / 4 / kMmaN; ++q) {
          const int nq = pq * (BCg / 4 / kMmaN) + q;
          const int row = q * kMmaN + (lane & 3) * 2;
          const int col = dv0 + (lane >> 2);
          sds[row * G::LdDs + col] = __float2bfloat16_rn(dst[nq][0]);
          sds[(row + 1) * G::LdDs + col] = __float2bfloat16_rn(dst[nq][1]);
          sds[row * G::LdDs + col + 8] = __float2bfloat16_rn(dst[nq][2]);
          sds[(row + 1) * G::LdDs + col + 8] = __float2bfloat16_rn(dst[nq][3]);
        }
        rendezvous();

        float acc[MTOK][4];
#pragma unroll
        for (int m = 0; m < MTOK; ++m)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[m][e] = 0.f;
        //: the INTER half: `<dS_a[s,:], v[j,:]>` with the WHOLE `d_v` contracted,
        //: because the state arrived through shared instead of through registers.
#pragma unroll
        for (int kk = 0; kk < KV; ++kk) {
          uint32_t bs[4];
          load_frag(bs, ADDR_DS(bdsT, gwarp * 8, 2 * kk));
#pragma unroll
          for (int m = 0; m < MTOK; ++m) {
            uint32_t av[4];
            load_frag(av, ADDR_VG(atv, m * 16, 2 * kk));
            mma(acc[m], av, &bs[0]);
          }
        }
        //: the INTRA half: the transposed staircase, `A = Gd'^T` off the Gram and
        //: `B` = the R panel, contracted over the chunk's READ tokens.
        if (liveR) {
#pragma unroll
          for (int kk = 0; kk < MTOK; ++kk) {
            uint32_t br[4];
            load_frag_t(br, ADDR_S(brF, kk * 16, (pq * (BCg / 4) + gwarp * 8) / 8));
#pragma unroll
            for (int m = 0; m < MTOK; ++m) {
              uint32_t ag[4];
              load_frag_t(ag, ADDR_G(sgT, kk * 16, m * 2));
              mma(acc[m], ag, &br[0]);
            }
          }
        }
        //: the ENTRY mass column, one value per slot, added where `dW` wants it.
        const int s0 = pq * (BCg / 4) + gwarp * 8 + (lane & 3) * 2;
#pragma unroll
        for (int m = 0; m < MTOK; ++m) {
          acc[m][0] += sdz[s0];
          acc[m][1] += sdz[s0 + 1];
          acc[m][2] += sdz[s0];
          acc[m][3] += sdz[s0 + 1];
        }

        // ---- the digit reduction, in lane --------------------------------------
        const int cls = pq & 1, od = pq >> 1;
        const int i1 = cls * (kInnerSpan / 2) + gwarp * 8 + (lane & 3) * 2;
#pragma unroll
        for (int m = 0; m < MTOK; ++m) {
          const int t0 = m * 16 + (lane >> 2);
          const float w0 = spw0[od * C + t0], w1 = spw0[od * C + t0 + 8];
          u1[cls][m][0] += acc[m][0] * w0;
          u1[cls][m][1] += acc[m][1] * w0;
          u1[cls][m][2] += acc[m][2] * w1;
          u1[cls][m][3] += acc[m][3] * w1;
          const uint32_t pcol = (uint32_t)((LP::col_base(kDepth - 1) + i1) * 2) + winPc;
          const uint32_t p0 = *reinterpret_cast<const uint32_t*>(
              gather_ptr(pWrite, (uint32_t)iwrite(dc)[t0] * kPRowB + pcol));
          const uint32_t p1 = *reinterpret_cast<const uint32_t*>(
              gather_ptr(pWrite, (uint32_t)iwrite(dc)[t0 + 8] * kPRowB + pcol));
          u0[od][m][0] +=
              acc[m][0] * __bfloat162float(__ushort_as_bfloat16((unsigned short)p0))
              + acc[m][1] * __bfloat162float(__ushort_as_bfloat16((unsigned short)(p0 >> 16)));
          u0[od][m][1] +=
              acc[m][2] * __bfloat162float(__ushort_as_bfloat16((unsigned short)p1))
              + acc[m][3] * __bfloat162float(__ushort_as_bfloat16((unsigned short)(p1 >> 16)));
        }
        rendezvous();
      }

      // ---- the write side's digit gradients leave here ------------------------
      //: THE INNER LEVEL NEEDS NO REDUCTION: every `(token, digit)` cell is held
      //: by exactly one lane, so the fan-in is `C * B` adds per block and that is
      //: the minimum the destination admits.
#pragma unroll
      for (int cls = 0; cls < 2; ++cls)
#pragma unroll
        for (int m = 0; m < MTOK; ++m) {
          const int i1 = cls * (BCg / 4) + gwarp * 8 + (lane & 3) * 2;
          const uint32_t col = (uint32_t)((LP::col_base(kDepth - 1) + i1) * 4) + winPw;
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int tk = m * 16 + (lane >> 2) + (e >> 1) * 8;
            if (tk < nw)
              fan_in_add(
                  reinterpret_cast<float*>(gather_ptr(
                      dpwPlane, (uint32_t)iwrite(dc)[tk] * kPRowW + col + (uint32_t)((e & 1) * 4))),
                  sgw[tk] * u1[cls][m][e]);
          }
        }
      //: THE OUTER LEVEL sums over this lane's slot run first -- two shuffles --
      //: and then over the units, which the fan-in does.
#pragma unroll
      for (int o = 0; o < kOuterSpan; ++o)
#pragma unroll
        for (int m = 0; m < MTOK; ++m)
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
            for (int b2 = 1; b2 < 4; b2 <<= 1)
              u0[o][m][hh] += __shfl_xor_sync(0xFFFFFFFFu, u0[o][m][hh], b2);
            const int tk = m * 16 + (lane >> 2) + hh * 8;
            if ((lane & 3) == 0 && tk < nw) {
              const uint32_t col =
                  (uint32_t)((LP::col_base(0) + LP::off_at(0, blk) + o) * 4) + winPw;
              fan_in_add(reinterpret_cast<float*>(
                             gather_ptr(dpwPlane, (uint32_t)iwrite(dc)[tk] * kPRowW + col)),
                         sgw[tk] * u0[o][m][hh]);
            }
          }
    }

    if (liveR) {
      // ---- (A.16) THE FOLD: dS^T += dnum^T . R, and dz^T += dden^T . R -------
#pragma unroll
      for (int kk = 0; kk < MTOK; ++kk) {
        uint32_t a[4];
        load_frag_t(a, ADDR_V(adn, kk * 16));
        //: THE R PANEL'S k-RUN IS NOT HOISTED.  The fold runs with the whole
        //: adjoint state live, so `KD * 4` more registers here is a peak this
        //: kernel cannot afford; one tile at a time is the same `ldmatrix` count.
#pragma unroll
        for (int nj = 0; nj < KD; ++nj) {
          uint32_t b[4];
          load_frag_t(b, ADDR_S(brF, kk * 16, 2 * nj));
          mma(dst[2 * nj], a, &b[0]);
          mma(dst[2 * nj + 1], a, &b[2]);
        }
      }
      //: dz's OWN FOLD: this thread's slot against the chunk's read entries.  The
      //: R panel's rows past `nr` are identically zero and so is `sdd` there, so
      //: the run is the whole chunk and needs no bound.
      const __nv_bfloat16* rcol = sr + gtid;
#pragma unroll
      for (int i = 0; i < C; ++i) dz_s += sdd[i] * __bfloat162float(rcol[i * LdS]);
    }

    rendezvous();  // (3) `so`/`sgrad` are free and the panels are dead

    if (++dc == IRING) dc = 0;
    if (++ds == IRING) ds = 0;
    if (++dn == IRING) dn = 0;
  }
#undef ADDR_S
#undef ADDR_G
#undef ADDR_V
}

}  // namespace rola::chunk
