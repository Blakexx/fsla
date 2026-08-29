// csrc/rola/src/facts/facts_kernel.cuh -- THE FACTS KERNELS: the union-table
// pass (`chunk_facts_tables`), the atom-grain stats reduction
// (`atom_bitmap_kernel`) and the block-grain liveness epilogue
// (`block_bitmap_kernel`), with `LevelPlan` -- the geometry law -- and the
// constants they share.  Extracted from the chunk consumer's header (K31
// deletion batch; baseline = tag `baseline/pre-k31`): decode, the paging
// layer and the engine facts layer consume these facts and outlive the
// consumer.  The namespace stays `rola::chunk` so the kernels' mangled names
// -- the ratification manifest's keys -- do not move with the file.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "facts_ops.cuh"

namespace rola::chunk {

using namespace denseref::ops;

//: THE INVARIANT QUANTUM: the leaves one page granule holds, and the rectangle -- see docs/internals/facts/facts_kernel.md#katomleaves
static constexpr int kAtomLeaves = 16;

// --------------------------------------------------------------- block plan

//: THE GEOMETRY LAW over D levels of one uniform power-of-two width B, stated -- see docs/internals/facts/facts_kernel.md#note-l30
constexpr long ipowl(int b, int e) { return e == 0 ? 1 : b * ipowl(b, e - 1); }

//: THE LEX POLICY, as free functions so `LevelPlan` can be an alias template.  A -- see docs/internals/facts/facts_kernel.md#lexwhole
constexpr int lex_whole(int D, int B, int BC) {
  return (long)BC >= ipowl(B, D)   ? D
         : (long)BC >= ipowl(B, 3) ? 3
         : (long)BC >= ipowl(B, 2) ? 2
         : BC >= B                 ? 1
                                   : 0;
}
constexpr int lex_partial(int D, int B, int BC) {
  return (int)(BC / ipowl(B, lex_whole(D, B, BC)));
}
constexpr int lex_singles(int D, int B, int BC) {
  return D - lex_whole(D, B, BC) - (lex_partial(D, B, BC) > 1 ? 1 : 0);
}
constexpr int lex_span(int D, int B, int BC, int l) {
  return l >= D                                                      ? B
         : l < lex_singles(D, B, BC)                                 ? 1
         : (lex_partial(D, B, BC) > 1 && l == lex_singles(D, B, BC)) ? lex_partial(D, B, BC)
                                                                     : B;
}

template <int D, int B, int S0, int S1, int S2, int S3>
struct SpanPlan {
  static_assert(D >= 1 && D <= 4, "the level plan is built to depth four");
  //: THE WIDTH FLOOR IS THE ATOM'S, not a routing preference: the atom plan -- see docs/internals/facts/facts_kernel.md#static-assert
  static_assert((B & (B - 1)) == 0 && B >= 4 && B <= 256 && ipowl(B, D) >= kAtomLeaves,
                "one uniform power-of-two width per topology, wide enough that the "
                "topology holds a whole page atom");
  static constexpr int kWidth = B;
  static constexpr long slots = ipowl(B, D);
  static constexpr int width(int) { return B; }  // the uniform widths policy
  //: per-level spans, most-significant level first -- GIVEN.
  static constexpr int span_at(int l) { return l == 0 ? S0 : l == 1 ? S1 : l == 2 ? S2 : S3; }
  static_assert(S0 >= 1 && S0 <= B && B % S0 == 0 && S1 >= 1 && S1 <= B && B % S1 == 0 && S2 >= 1
                    && S2 <= B && B % S2 == 0 && S3 >= 1 && S3 <= B && B % S3 == 0,
                "BLOCK GEOMETRY: every span is an aligned run that divides its level");
  static constexpr int prod_span(int l) { return l >= D ? 1 : span_at(l) * prod_span(l + 1); }
  //: `BC` is a CONSEQUENCE of the spans rather than the input they were derived -- see docs/internals/facts/facts_kernel.md#kbc
  static constexpr int kBC = prod_span(0);
  static constexpr long blocks = slots / kBC;
  //: a level's table-entry count: its group count when tested, zero when whole
  static constexpr int entries_at(int l) { return span_at(l) < B ? B / span_at(l) : 0; }
  static constexpr int groups_at(int l) { return span_at(l) < B ? B / span_at(l) : 1; }
  //: how many clause words the form ANDs per side -- the levels that HAVE rows.
  static constexpr int count_clauses(int l) {
    return l >= D ? 0 : (entries_at(l) > 0 ? 1 : 0) + count_clauses(l + 1);
  }
  static constexpr int clauses = count_clauses(0);
  //: THE CLAUSE LEVELS ARE A PREFIX -- a property of the TWO BITMAP REDUCTIONS -- see docs/internals/facts/facts_kernel.md#clauses-prefix
  static constexpr bool clauses_prefix(int l) {
    return l >= D || (((entries_at(l) > 0) == (l < clauses)) && clauses_prefix(l + 1));
  }
  //: all-INT arithmetic: the suffix products are bounded by the block count, -- see docs/internals/facts/facts_kernel.md#suffixgroups
  static constexpr int suffix_groups(int l) {
    return l >= D ? 1 : groups_at(l) * suffix_groups(l + 1);
  }
  static_assert(blocks == (long)(int)blocks, "the block lattice fits int");
  static constexpr int prefix_entries(int l) {
    return l == 0 ? 0 : prefix_entries(l - 1) + entries_at(l - 1);
  }
  //: the block's run index at level l, and its table row -- compile-time -- see docs/internals/facts/facts_kernel.md#entryat
  static __device__ __forceinline__ int entry_at(int l, int blk) {
    return prefix_entries(l) + (blk / suffix_groups(l + 1)) % groups_at(l);
  }
  static constexpr int depth = D;
  //: a level's first amplitude column in the packed [W0+..+W_{D-1}] token row -- -- see docs/internals/facts/facts_kernel.md#colbase
  static constexpr int col_base(int l) { return l == 0 ? 0 : col_base(l - 1) + width(l - 1); }
  //: the block's first digit at level l: entry_at's map scaled to digits.  A -- see docs/internals/facts/facts_kernel.md#offat
  static __device__ __forceinline__ int off_at(int l, int blk) {
    return ((blk / suffix_groups(l + 1)) % groups_at(l)) * span_at(l);
  }
  //: THE COMPOSITE OUTER INDEX'S RADIX. A block's slot splits into an innermost -- see docs/internals/facts/facts_kernel.md#outerplace
  static constexpr int outer_place(int l) {
    return l >= D - 2 ? 1 : span_at(l + 1) * outer_place(l + 1);
  }
  //: THE LEVEL IS A TEMPLATE PARAMETER for the same reason `base_term`'s is: a -- see docs/internals/facts/facts_kernel.md#near-line-124
  template <int Lv>
  static __device__ __forceinline__ int outer_digit(int o) {
    constexpr int kPlace = outer_place(Lv);
    constexpr int kSpan = span_at(Lv);
    if constexpr (kSpan == 1)
      return 0;
    else if constexpr (Lv == 0)
      return o / kPlace;
    else
      return (o / kPlace) % kSpan;
  }
  //: table entries per side: a level's group count when it is tested, NOTHING -- see docs/internals/facts/facts_kernel.md#sumentries
  static constexpr int sum_entries(int l) {
    return l >= D ? 0 : entries_at(l) + sum_entries(l + 1);
  }
  static constexpr int entries = sum_entries(0);
  //: A block's leaves are a CONTIGUOUS, BC-ALIGNED run in canonical leaf order, -- see docs/internals/facts/facts_kernel.md#base-term
  template <int Lv>
  static __device__ __forceinline__ int base_term(int blk) {
    constexpr int kDiv = suffix_groups(Lv + 1);
    constexpr int kGrp = groups_at(Lv);
    constexpr int kPlace = (int)(span_at(Lv) * ipowl(B, D - 1 - Lv));
    return ((blk / kDiv) % kGrp) * kPlace;
  }
  static __device__ __forceinline__ int block_base(int blk) {
    int s = 0;
    if constexpr (0 < D) s += base_term<0>(blk);
    if constexpr (1 < D) s += base_term<1>(blk);
    if constexpr (2 < D) s += base_term<2>(blk);
    if constexpr (3 < D) s += base_term<3>(blk);
    return s;
  }
  //: the law's STATIC piece, per level: a contributing level's offsets step by -- see docs/internals/facts/facts_kernel.md#basealigned
  static constexpr bool base_aligned(int l) {
    return l >= D
           || ((groups_at(l) == 1 || (ipowl(B, D - 1 - l) * span_at(l)) % kBC == 0)
               && base_aligned(l + 1));
  }
};

//: THE LEX PLAN, an ALIAS. -- see docs/internals/facts/facts_kernel.md#levelplan
template <int D, int B, int BC>
using LevelPlan = SpanPlan<D, B, lex_span(D, B, BC, 0), lex_span(D, B, BC, 1),
                           lex_span(D, B, BC, 2), lex_span(D, B, BC, 3)>;

//: (G4) The D = 2 compatibility view -- BlockPlan -- is RETIRED. -- see docs/internals/facts/facts_kernel.md#ktokbits

//: THE UNION TABLE'S ONE STRUCTURAL CONSTANT. Tokens are the table's minor axis -- see docs/internals/facts/facts_kernel.md#ktokbits-2
constexpr int kTokBits = 32;
__host__ __device__ constexpr int tok_words(int L) { return (L + kTokBits - 1) / kTokBits; }

// ------------------------------------------------------- the facts tables
//: one warp per token word, four warps per CTA.
constexpr int kFactsThreads = 128;

//: THE CALL CLASSES THIS PASS IS SPECIALIZED OVER -- the prefill half of the four -- see docs/internals/facts/facts_kernel.md#kprefillstateless
enum ChunkMask : int { kPrefillStateless = 1, kPrefillStateful = 2 };

//: THE TABLES ARE BUILT ONCE FOR THE WHOLE GRID, BOTH ROW SETS IN ONE PASS. -- see docs/internals/facts/facts_kernel.md#chunk-facts-tables
template <class PLAN_R, class PLAN_W, int MASK>
__global__ void chunk_facts_tables(const __nv_bfloat16* __restrict__ pread,
                                   const __nv_bfloat16* __restrict__ pwrite,
                                   uint32_t* __restrict__ tbl, uint32_t* __restrict__ atom_tbl,
                                   int L) {
  using LPR = PLAN_R;
  using LPW = PLAN_W;
  static_assert(LPR::depth == LPW::depth && LPR::kWidth == LPW::kWidth,
                "the two sides partition the SAME topology; only their runs differ");
  using LP = PLAN_W;
  constexpr int D = LP::depth;
  constexpr int B = LP::kWidth;
  using AP = LevelPlan<D, B, kAtomLeaves>;
  constexpr int kWtot = LP::col_base(LP::depth);
  //: the side's row block base, in ENTRIES: the read plan's rows and then the write -- see docs/internals/facts/facts_kernel.md#ksideentries
  constexpr long kSideEntries = (long)LPR::entries + (long)LPW::entries;
  static_assert(kWtot == AP::col_base(AP::depth),
                "both grains read the SAME amplitude row: the packed column layout is "
                "the widths policy's and carries no block geometry");
  const int tokWords = tok_words(L);
  const int w = blockIdx.x * (blockDim.x / kBallotLanes) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (w >= tokWords) return;  // a whole lockstep unit leaves together
  //: A TOKEN PAST THE END READS TOKEN ZERO AND VOTES DEAD.  Its bit has no -- see docs/internals/facts/facts_kernel.md#t
  const int t = w * kTokBits + lane;
  const bool real = t < L;
  const long off = (long)blockIdx.y * L * kWtot + (long)(real ? t : 0) * kWtot;

  //: ADDRESSING DISCIPLINE, THE SAME ONE THE CHUNK KERNEL KEEPS.  Every 64-bit -- see docs/internals/facts/facts_kernel.md#entrystride
  const long entryStride = tokWords;
  const __nv_bfloat16* const ampRead = pread + off;
  const __nv_bfloat16* const ampWrite = pwrite + off;

#define ROLA_FACTS_VOTE(P_, SPAN_) \
  lane_ballot(real && (or_run<SPAN_>(P_) & kBf16MagnitudeMask) != 0u)

  //: one stanza per level, spans from the plan -- a whole level emits nothing, -- see docs/internals/facts/facts_kernel.md#rolafactslevel
#define ROLA_FACTS_LEVEL(PLAN_, BASE_, OUT_, L_)                                \
  if constexpr (L_ < D && PLAN_::entries_at(L_) > 0) {                          \
    const __nv_bfloat16* p = (BASE_) + PLAN_::col_base(L_);                     \
    constexpr int kSpan = PLAN_::span_at(L_);                                   \
    _Pragma("unroll") for (int e = 0; e < PLAN_::entries_at(L_); ++e) {         \
      const uint32_t vote = ROLA_FACTS_VOTE(p, kSpan);                          \
      if (lane == 0) *(OUT_) = vote;                                            \
      p += kSpan;                                                               \
      (OUT_) += entryStride;                                                    \
    }                                                                           \
  }

  //: the write side at BOTH grains.  An owner entry's digit run is a whole number -- see docs/internals/facts/facts_kernel.md#rolafactslevelboth
#define ROLA_FACTS_LEVEL_BOTH(L_)                                               \
  if constexpr (L_ < D && AP::entries_at(L_) > 0) {                             \
    constexpr int kSpanA = AP::span_at(L_);                                     \
    constexpr int kEntA = AP::entries_at(L_);                                   \
    constexpr int kEntU = LP::entries_at(L_);                                   \
    constexpr int kOwners = kEntU > 0 ? kEntU : 1;                              \
    constexpr int kGroup = kEntA / kOwners;                                     \
    static_assert(kEntU == 0                                                    \
                      || (kGroup * kOwners == kEntA                             \
                          && LP::span_at(L_) == kSpanA * kGroup),               \
                  "the atom rectangle must REFINE the owner rectangle at every "  \
                  "level, or an owner row is not the OR of atom rows");          \
    const __nv_bfloat16* p = ampWrite + AP::col_base(L_);                       \
    _Pragma("unroll") for (int u = 0; u < kOwners; ++u) {                      \
      uint32_t owner = 0;                                                       \
      _Pragma("unroll") for (int g = 0; g < kGroup; ++g) {                      \
        const uint32_t vote = ROLA_FACTS_VOTE(p, kSpanA);                       \
        if (lane == 0) *aout = vote;                                            \
        owner |= vote;                                                          \
        p += kSpanA;                                                            \
        aout += entryStride;                                                    \
      }                                                                         \
      if constexpr (kEntU > 0) {                                                \
        if (lane == 0) *out = owner;                                            \
        out += entryStride;                                                     \
      }                                                                         \
    }                                                                           \
  } else if constexpr (L_ < D) {                                                \
    static_assert(LP::entries_at(L_) == 0,                                      \
                  "a level the ATOM grain spans whole cannot have owner rows: "  \
                  "the owner rectangle is the coarser partition");               \
  }

  {
    uint32_t* out = tbl + ((long)blockIdx.y * kSideEntries) * entryStride + w;
    ROLA_FACTS_LEVEL(LPR, ampRead, out, 0)
    ROLA_FACTS_LEVEL(LPR, ampRead, out, 1)
    ROLA_FACTS_LEVEL(LPR, ampRead, out, 2)
    ROLA_FACTS_LEVEL(LPR, ampRead, out, 3)
  }

  //: THE ATOM ROWS ARE BOTH SIDES' (P80b): the write side's say which atoms this -- see docs/internals/facts/facts_kernel.md#note-l339
  if constexpr (MASK == kPrefillStateful) {
    if (atom_tbl != nullptr) {
      uint32_t* out =
          tbl + ((long)blockIdx.y * kSideEntries + (long)LPR::entries) * entryStride + w;
      uint32_t* aoutR = atom_tbl + ((long)(2 * blockIdx.y) * AP::entries) * entryStride + w;
      uint32_t* aout = atom_tbl + ((long)(2 * blockIdx.y + 1) * AP::entries) * entryStride + w;
      ROLA_FACTS_LEVEL(AP, ampRead, aoutR, 0)
      ROLA_FACTS_LEVEL(AP, ampRead, aoutR, 1)
      ROLA_FACTS_LEVEL(AP, ampRead, aoutR, 2)
      ROLA_FACTS_LEVEL(AP, ampRead, aoutR, 3)
      ROLA_FACTS_LEVEL_BOTH(0)
      ROLA_FACTS_LEVEL_BOTH(1)
      ROLA_FACTS_LEVEL_BOTH(2)
      ROLA_FACTS_LEVEL_BOTH(3)
      return;
    }
  }
  {
    uint32_t* out = tbl + ((long)blockIdx.y * kSideEntries + (long)LPR::entries) * entryStride + w;
    ROLA_FACTS_LEVEL(LPW, ampWrite, out, 0)
    ROLA_FACTS_LEVEL(LPW, ampWrite, out, 1)
    ROLA_FACTS_LEVEL(LPW, ampWrite, out, 2)
    ROLA_FACTS_LEVEL(LPW, ampWrite, out, 3)
  }
#undef ROLA_FACTS_LEVEL_BOTH
#undef ROLA_FACTS_LEVEL
#undef ROLA_FACTS_VOTE
}

// ------------------------------------------------------- the atom-grain stats
//: THE STATS PASS' REDUCTION. -- see docs/internals/facts/facts_kernel.md#katomthreads
constexpr int kAtomThreads = 128;
constexpr int kAtomTileWords = 256;

//: THE REDUCTION, one CTA per (bh, upper prefix), OVER BOTH SIDES (P80b). A -- see docs/internals/facts/facts_kernel.md#atom-bitmap-kernel
template <int D, int B>
__global__ void atom_bitmap_kernel(const uint32_t* __restrict__ tbl,
                                   uint8_t* __restrict__ atom_bits, int L) {
  using LP = LevelPlan<D, B, kAtomLeaves>;
  static_assert(LP::base_aligned(0),
                "BLOCK GEOMETRY: a block's first leaf is not BC-aligned, so its leaves "
                "are not one aligned run and the state's atom addressing does not hold");
  static_assert(LP::clauses_prefix(0),
                "BLOCK GEOMETRY: this reduction addresses the innermost clause row as a "
                "prefix count, so the levels carrying rows must be a PREFIX of the levels");
  constexpr int kClauses = LP::clauses;
  constexpr int kAtoms = (int)LP::blocks;
  //: the innermost clause level's group count: the atoms of one prefix, and by -- see docs/internals/facts/facts_kernel.md#kinner
  constexpr int kInner = kClauses > 0 ? LP::groups_at(kClauses - 1) : 1;
  //: §6 unroll discipline: a bare `k`-prefixed bound, so the two-sided runs read -- see docs/internals/facts/facts_kernel.md#kslive-atom
  constexpr int kSLive = 2 * kInner;
  constexpr int kWarps = kAtomThreads / kBallotLanes;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int blkBase = blockIdx.x * kInner;
  //: THE BITMAP IS ONE BYTE PER ATOM, so `BH * atoms` is a 32-bit displacement -- see docs/internals/facts/facts_kernel.md#outoff
  const uint32_t outOff = (uint32_t)blockIdx.y * (uint32_t)kAtoms + (uint32_t)blkBase;

  //: NO CLAUSE AT ALL is the whole slot space in one atom: read and written -- see docs/internals/facts/facts_kernel.md#constexpr
  if constexpr (kClauses == 0) {
    if (tid == 0) *reinterpret_cast<uint8_t*>(gather_ptr(atom_bits, outOff)) = kAtomActive;
    return;
  } else {
    const int tokWords = tok_words(L);
    //: ADDRESSING DISCIPLINE, THE FORM'S: ONE 64-bit address per (bh, side), formed -- see docs/internals/facts/facts_kernel.md#pin-address
    const uint32_t* const baseR =
        pin_address(tbl + ((long)(2 * blockIdx.y + 0) * (long)LP::entries) * (long)tokWords);
    const uint32_t* const baseW =
        pin_address(tbl + ((long)(2 * blockIdx.y + 1) * (long)LP::entries) * (long)tokWords);
    const uint32_t rowBytes = (uint32_t)tokWords * (uint32_t)sizeof(uint32_t);
    //: NAMED SCALARS, never an array -- the consumer's rule, for the same -- see docs/internals/facts/facts_kernel.md#offua
    uint32_t offUa = 0u, offUb = 0u, offUc = 0u;
#define ROLA_ATOM_UPPER(CL_, O_)                                               \
  if constexpr ((CL_) + 1 < kClauses) {                                        \
    constexpr int kDiv = LP::suffix_groups((CL_) + 1);                         \
    constexpr int kGrp = LP::groups_at(CL_);                                   \
    constexpr int kPre = LP::prefix_entries(CL_);                              \
    O_ = (uint32_t)(kPre + (blkBase / kDiv) % kGrp) * rowBytes;                \
  }
    ROLA_ATOM_UPPER(0, offUa)
    ROLA_ATOM_UPPER(1, offUb)
    ROLA_ATOM_UPPER(2, offUc)
#undef ROLA_ATOM_UPPER
    //: the innermost rows are the CONTIGUOUS entry run at the prefix's level.
    const uint32_t offInner = (uint32_t)LP::prefix_entries(kClauses - 1) * rowBytes;

    __shared__ uint32_t sUp[2][kAtomTileWords];
    __shared__ uint8_t sLive[kSLive];
    __shared__ int sAllLive;
    for (int e = tid; e < kSLive; e += kAtomThreads) sLive[e] = 0;
    if (tid == 0) sAllLive = 0;
    rendezvous();

#pragma unroll 1
    for (int w0 = 0; w0 < tokWords; w0 += kAtomTileWords) {
      const int n = min(kAtomTileWords, tokWords - w0);
#pragma unroll 1
      for (int i = tid; i < n; i += kAtomThreads) {
        const uint32_t off = (uint32_t)(w0 + i) * (uint32_t)sizeof(uint32_t);
        uint32_t mr = 0xFFFFFFFFu, mw = 0xFFFFFFFFu;
#define ROLA_ATOM_AND(CL_, O_)                                               \
  if constexpr ((CL_) < kClauses) {                                          \
    mr &= *reinterpret_cast<const uint32_t*>(gather_ptr(baseR, (O_) + off)); \
    mw &= *reinterpret_cast<const uint32_t*>(gather_ptr(baseW, (O_) + off)); \
  }
        ROLA_ATOM_AND(1, offUa)
        ROLA_ATOM_AND(2, offUb)
        ROLA_ATOM_AND(3, offUc)
#undef ROLA_ATOM_AND
        sUp[0][i] = mr;
        sUp[1][i] = mw;
      }
      rendezvous();
      for (int e = tid >> 5; e < kSLive; e += kWarps) {
        if (sLive[e]) continue;
        const int s = e / kInner;
        const uint32_t* const base = s == 0 ? baseR : baseW;
        const uint32_t rowOff = offInner + (uint32_t)(e - s * kInner) * rowBytes;
        bool hit = false;
#pragma unroll 1
        for (int i = lane; i < n; i += kBallotLanes) {
          const uint32_t off = (uint32_t)(w0 + i) * (uint32_t)sizeof(uint32_t);
          hit |= (sUp[s][i] & *reinterpret_cast<const uint32_t*>(gather_ptr(base, rowOff + off)))
                 != 0u;
        }
        //: the vote is the WHOLE unit's -- every lane issues it, lane 0 files it.
        const uint32_t any = lane_ballot(hit);
        __syncwarp();
        if (lane == 0 && any != 0u) sLive[e] = 1;
      }
      rendezvous();
      //: THE EXIT IS CTA-UNIFORM: one warp reads every flag behind the barrier -- see docs/internals/facts/facts_kernel.md#near-line-428
      if (tid < kBallotLanes) {
        bool all = true;
        for (int e = tid; e < kSLive; e += kBallotLanes) all &= sLive[e] != 0;
        const uint32_t every = lane_ballot(all);
        if (tid == 0 && every == 0xFFFFFFFFu) sAllLive = 1;
      }
      rendezvous();
      if (sAllLive) break;
    }
    for (int a = tid; a < kInner; a += kAtomThreads)
      *reinterpret_cast<uint8_t*>(gather_ptr(atom_bits, outOff + (uint32_t)a)) =
          (uint8_t)((sLive[kInner + a] ? kAtomWritten : 0u) | (sLive[a] ? kAtomRead : 0u));
  }
}

// ------------------------------------------------------ the block-grain liveness
//: THE REDUCTION'S ROW GROUP: at most one innermost row per warp per side. -- see docs/internals/facts/facts_kernel.md#block-bitmap-kernel
constexpr int block_bitmap_group(int inner) {
  return inner < (kAtomThreads / kBallotLanes) / 2 ? inner : (kAtomThreads / kBallotLanes) / 2;
}

//: THE LIVENESS EPILOGUE, at the LAUNCH's block grain: `atom_bitmap_kernel`'s -- see docs/internals/facts/facts_kernel.md#block-bitmap-kernel-2
template <int D, int B, int BC>
__global__ void block_bitmap_kernel(const uint32_t* __restrict__ tbl,
                                    const int32_t* __restrict__ page_tbl,
                                    uint8_t* __restrict__ block_bits, int L) {
  using LP = LevelPlan<D, B, BC>;
  static_assert(LP::base_aligned(0),
                "BLOCK GEOMETRY: a block's first leaf is not BC-aligned, so its leaves "
                "are not one aligned run and the state's atom addressing does not hold");
  static_assert(LP::clauses_prefix(0),
                "BLOCK GEOMETRY: this reduction addresses the innermost clause row as a "
                "prefix count, so the levels carrying rows must be a PREFIX of the levels");
  constexpr int kClauses = LP::clauses;
  constexpr int kBlocks = (int)LP::blocks;
  constexpr int kInner = kClauses > 0 ? LP::groups_at(kClauses - 1) : 1;
  constexpr int kWarps = kAtomThreads / kBallotLanes;
  constexpr int kGroup = block_bitmap_group(kInner);
  static_assert(kInner % kGroup == 0, "a row group must divide the prefix's inner run");
  constexpr int kAtomsPer = BC / kAtomLeaves;
  static_assert(BC % kAtomLeaves == 0, "a block is a whole number of page atoms");
  //: §6 unroll discipline: the heuristic reads only a bare identifier/literal as -- see docs/internals/facts/facts_kernel.md#kslive
  constexpr int kSLive = 2 * kGroup;
  constexpr int kGroupAtoms = kGroup * kAtomsPer;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int blkBase = blockIdx.x * kGroup;
  const int innerBase = blkBase % kInner;
  const uint32_t outOff = (uint32_t)blockIdx.y * (uint32_t)kBlocks + (uint32_t)blkBase;

  //: the two sides share one token tile, so the read row's stream and the write -- see docs/internals/facts/facts_kernel.md#near-line-485
  __shared__ uint32_t sUp[2][kAtomTileWords];
  __shared__ uint8_t sLive[2 * kGroup];
  __shared__ uint8_t sState[kGroup];
  __shared__ int sAllLive;
  for (int e = tid; e < kSLive; e += kAtomThreads) sLive[e] = 0;
  for (int a = tid; a < kGroup; a += kAtomThreads) sState[a] = 0;
  if (tid == 0) sAllLive = 0;
  rendezvous();

  //: AN UNMAPPED ATOM'S ENTRY STATE IS PROVABLY ZERO, so residency is the exact -- see docs/internals/facts/facts_kernel.md#near-line-496
  if (page_tbl != nullptr) {
    const int32_t* const pt = pin_address(page_tbl);
    const uint32_t bhAtoms = (uint32_t)blockIdx.y * (uint32_t)(LP::slots / kAtomLeaves);
    for (int e = tid; e < kGroupAtoms; e += kAtomThreads) {
      const int a = e / kAtomsPer;
      const uint32_t atom = bhAtoms + (uint32_t)(LP::block_base(blkBase + a) / kAtomLeaves)
                            + (uint32_t)(e - a * kAtomsPer);
      if (*reinterpret_cast<const int32_t*>(gather_ptr(pt, atom * (uint32_t)sizeof(int32_t))) >= 0)
        sState[a] = 1;
    }
  }

  //: NO CLAUSE AT ALL is the whole slot space in one block: live on both sides -- see docs/internals/facts/facts_kernel.md#constexpr-2
  if constexpr (kClauses == 0) {
    for (int e = tid; e < kSLive; e += kAtomThreads) sLive[e] = 1;
  } else {
    const int tokWords = tok_words(L);
    //: ADDRESSING DISCIPLINE, THE FORM'S: one 64-bit base per (bh, side), every -- see docs/internals/facts/facts_kernel.md#baser
    const uint32_t* const baseR =
        pin_address(tbl + ((long)(2 * blockIdx.y + 0) * (long)LP::entries) * (long)tokWords);
    const uint32_t* const baseW =
        pin_address(tbl + ((long)(2 * blockIdx.y + 1) * (long)LP::entries) * (long)tokWords);
    const uint32_t rowBytes = (uint32_t)tokWords * (uint32_t)sizeof(uint32_t);
    //: NAMED SCALARS, never an array -- a runtime-indexed offset array is -- see docs/internals/facts/facts_kernel.md#offua-2
    uint32_t offUa = 0u, offUb = 0u, offUc = 0u;
#define ROLA_BLOCK_UPPER(CL_, O_)                              \
  if constexpr ((CL_) + 1 < kClauses) {                        \
    constexpr int kDiv = LP::suffix_groups((CL_) + 1);         \
    constexpr int kGrp = LP::groups_at(CL_);                   \
    constexpr int kPre = LP::prefix_entries(CL_);              \
    O_ = (uint32_t)(kPre + (blkBase / kDiv) % kGrp) * rowBytes; \
  }
    ROLA_BLOCK_UPPER(0, offUa)
    ROLA_BLOCK_UPPER(1, offUb)
    ROLA_BLOCK_UPPER(2, offUc)
#undef ROLA_BLOCK_UPPER
    const uint32_t offInner = (uint32_t)LP::prefix_entries(kClauses - 1) * rowBytes;

#pragma unroll 1
    for (int w0 = 0; w0 < tokWords; w0 += kAtomTileWords) {
      const int n = min(kAtomTileWords, tokWords - w0);
#pragma unroll 1
      for (int i = tid; i < n; i += kAtomThreads) {
        const uint32_t off = (uint32_t)(w0 + i) * (uint32_t)sizeof(uint32_t);
        uint32_t mr = 0xFFFFFFFFu, mw = 0xFFFFFFFFu;
#define ROLA_BLOCK_AND(CL_, O_)                                                  \
  if constexpr ((CL_) < kClauses) {                                              \
    mr &= *reinterpret_cast<const uint32_t*>(gather_ptr(baseR, (O_) + off));     \
    mw &= *reinterpret_cast<const uint32_t*>(gather_ptr(baseW, (O_) + off));     \
  }
        ROLA_BLOCK_AND(1, offUa)
        ROLA_BLOCK_AND(2, offUb)
        ROLA_BLOCK_AND(3, offUc)
#undef ROLA_BLOCK_AND
        sUp[0][i] = mr;
        sUp[1][i] = mw;
      }
      rendezvous();
      for (int e = tid >> 5; e < kSLive; e += kWarps) {
        if (sLive[e]) continue;
        const int s = e / kGroup;
        const uint32_t* const base = s == 0 ? baseR : baseW;
        const uint32_t rowOff = offInner + (uint32_t)(innerBase + e - s * kGroup) * rowBytes;
        bool hit = false;
#pragma unroll 1
        for (int i = lane; i < n; i += kBallotLanes) {
          const uint32_t off = (uint32_t)(w0 + i) * (uint32_t)sizeof(uint32_t);
          hit |= (sUp[s][i] & *reinterpret_cast<const uint32_t*>(gather_ptr(base, rowOff + off)))
                 != 0u;
        }
        //: the vote is the WHOLE unit's -- every lane issues it, lane 0 files it.
        const uint32_t any = lane_ballot(hit);
        __syncwarp();
        if (lane == 0 && any != 0u) sLive[e] = 1;
      }
      rendezvous();
      //: THE EXIT IS CTA-UNIFORM: one warp reads every flag behind the barrier -- see docs/internals/facts/facts_kernel.md#near-line-578
      if (tid < kBallotLanes) {
        bool all = true;
        for (int e = tid; e < kSLive; e += kBallotLanes) all &= sLive[e] != 0;
        const uint32_t every = lane_ballot(all);
        if (tid == 0 && every == 0xFFFFFFFFu) sAllLive = 1;
      }
      rendezvous();
      if (sAllLive) break;
    }
  }
  rendezvous();
  for (int a = tid; a < kGroup; a += kAtomThreads) {
    const uint8_t v =
        (uint8_t)(sLive[a] | (uint8_t)(sLive[kGroup + a] << 1) | (uint8_t)(sState[a] << 2));
    *reinterpret_cast<uint8_t*>(gather_ptr(block_bits, outOff + (uint32_t)a)) = v;
  }
}

// --------------------------------------------------------------- host side

// THE ARM SET, GENERATED: tools/gen_shards.py owns the one list, and this include
// is the whole of the dispatch's knowledge of it.
#include "../instantiations/facts_arms.inc"

}  // namespace rola::chunk
