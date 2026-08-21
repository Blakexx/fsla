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

//: THE INVARIANT QUANTUM: the leaves one page granule holds, and the rectangle
//: the stats pass keys residency to.  It is the MMA k-extent the state's own
//: tiling was derived at (the host mirror is `rola.ops.paging.MMA_K_QUANTUM`,
//: and decode's `kAtomLeaves` states the same 16 independently;
//: docs/internals/paging/stats.md).
static constexpr int kAtomLeaves = 16;

// --------------------------------------------------------------- block plan

//: THE GEOMETRY LAW over D levels of one uniform power-of-two width B, stated on
//: the quantity it is actually about: THE PER-LEVEL SPANS `s_l`.  Every clause
//: below is the law's, and a (widths, spans) tuple breaking one does not compile.
//: Widths are read through `width(l)` so a jagged policy can answer from a table
//: without touching a loop nest -- docs/internals/facts/facts_kernel.md#geometry-law
//:
//: THE SPANS ARE GIVEN, NOT DERIVED (K31-i-d).  The lex policy below -- `singles`
//: levels at one digit, at most one partial run, whole levels beneath -- is ONE way
//: of choosing them, and it admits at most one span that is neither `1` nor `B`.
//: The box lattice's owner runs are `D` spans (the flagship's `(8, 16)`),
//: which that shape forbids arithmetically rather than accidentally.  So the plan
//: TAKES the spans and the lex policy becomes `LevelPlan`, an alias that computes
//: them: one plan type, two ways of arriving at it, and every consumer below reads
//: `span_at` / `entries_at` / `col_base` exactly as it did.
constexpr long ipowl(int b, int e) { return e == 0 ? 1 : b * ipowl(b, e - 1); }

//: THE LEX POLICY, as free functions so `LevelPlan` can be an alias template.  A
//: level past `D` answers `B`, which is the identity for every quantity derived
//: from a span (`entries_at` 0, `groups_at` 1, `prod_span` unaffected).
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
  static_assert((B & (B - 1)) == 0 && B >= 8 && B <= 256,
                "one uniform power-of-two width per topology");
  static constexpr int kWidth = B;
  static constexpr long slots = ipowl(B, D);
  static constexpr int width(int) { return B; }  // the uniform widths policy
  //: per-level spans, most-significant level first -- GIVEN.
  static constexpr int span_at(int l) { return l == 0 ? S0 : l == 1 ? S1 : l == 2 ? S2 : S3; }
  static_assert(S0 >= 1 && S0 <= B && B % S0 == 0 && S1 >= 1 && S1 <= B && B % S1 == 0 && S2 >= 1
                    && S2 <= B && B % S2 == 0 && S3 >= 1 && S3 <= B && B % S3 == 0,
                "BLOCK GEOMETRY: every span is an aligned run that divides its level");
  static constexpr int prod_span(int l) { return l >= D ? 1 : span_at(l) * prod_span(l + 1); }
  //: `BC` is a CONSEQUENCE of the spans rather than the input they were derived
  //: from, which makes "BC is a rectangle in digit space" true by construction.
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
  //: THE CLAUSE LEVELS ARE A PREFIX, and the two bitmap reductions require it:
  //: they address the innermost clause row as `prefix_entries(clauses - 1)` and
  //: carve their grid on `suffix_groups`.  Under the lex policy it holds by
  //: construction (whole levels are the trailing ones); under a GIVEN-span plan it
  //: is a clause of the law, checked rather than assumed.
  static constexpr bool clauses_prefix(int l) {
    return l >= D || (((entries_at(l) > 0) == (l < clauses)) && clauses_prefix(l + 1));
  }
  static_assert(clauses_prefix(0),
                "BLOCK GEOMETRY: the levels carrying table rows must be a PREFIX of "
                "the levels; a whole level above a tested one has no clause index");
  //: all-INT arithmetic: the suffix products are bounded by the block count,
  //: and int-by-constexpr-pow2 is shifts -- the long form dragged 64-bit ops
  //: into two knife-edge register floors and spilled them.
  static constexpr int suffix_groups(int l) {
    return l >= D ? 1 : groups_at(l) * suffix_groups(l + 1);
  }
  static_assert(blocks == (long)(int)blocks, "the block lattice fits int");
  static constexpr int prefix_entries(int l) {
    return l == 0 ? 0 : prefix_entries(l - 1) + entries_at(l - 1);
  }
  //: the block's run index at level l, and its table row -- compile-time
  //: divisors throughout (pow2 groups), so the map is shifts and masks.
  static __device__ __forceinline__ int entry_at(int l, int blk) {
    return prefix_entries(l) + (blk / suffix_groups(l + 1)) % groups_at(l);
  }
  static constexpr int depth = D;
  //: a level's first amplitude column in the packed [W0+..+W_{D-1}] token row --
  //: the widths policy summed over the preceding levels.
  static constexpr int col_base(int l) { return l == 0 ? 0 : col_base(l - 1) + width(l - 1); }
  //: the block's first digit at level l: entry_at's map scaled to digits.  A
  //: whole level has one group, so its offset is zero without a branch.
  static __device__ __forceinline__ int off_at(int l, int blk) {
    return ((blk / suffix_groups(l + 1)) % groups_at(l)) * span_at(l);
  }
  //: THE COMPOSITE OUTER INDEX'S RADIX. A block's slot splits into an innermost
  //: digit and a composite outer index whose place values are the outer spans,
  //: level D-2 fastest -- docs/internals/facts/facts_kernel.md#outer-radix
  static constexpr int outer_place(int l) {
    return l >= D - 2 ? 1 : span_at(l + 1) * outer_place(l + 1);
  }
  //: THE LEVEL IS A TEMPLATE PARAMETER for the same reason `base_term`'s is: a
  //: runtime divisor does not fold.  A single-digit level has one digit and the
  //: MOST significant outer level cannot wrap, so neither pays the residue.
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
  //: table entries per side: a level's group count when it is tested, NOTHING
  //: when it is spanned whole -- a SUM over levels, never a product, which is why
  //: the table does not grow with N.
  static constexpr int sum_entries(int l) {
    return l >= D ? 0 : entries_at(l) + sum_entries(l + 1);
  }
  static constexpr int entries = sum_entries(0);
  //: A block's leaves are a CONTIGUOUS, BC-ALIGNED run in canonical leaf order, so
  //: the block is named by its first leaf -- docs/internals/facts/facts_kernel.md#leaf-order-law
  //: THE LEVEL IS A TEMPLATE PARAMETER, not an argument: `off_at`'s recursive
  //: constexpr divisor does not fold otherwise -- docs/internals/facts/facts_kernel.md#template-level
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
  //: the law's STATIC piece, per level: a contributing level's offsets step by
  //: `span_at(l)` digits at place value `B^(D-1-l)`, and every such step must be
  //: a whole number of blocks.  A whole level contributes nothing (one group).
  static constexpr bool base_aligned(int l) {
    return l >= D
           || ((groups_at(l) == 1 || (ipowl(B, D - 1 - l) * span_at(l)) % kBC == 0)
               && base_aligned(l + 1));
  }
};

//: THE LEX PLAN, an ALIAS.  `LevelPlan<D, B, BC>` is the span plan whose spans the
//: lex policy derives from `BC`, so every existing consumer keeps its spelling and
//: every existing instantiation keeps its geometry -- what moved is that the spans
//: are now the plan's input rather than its private derivation.
template <int D, int B, int BC>
using LevelPlan = SpanPlan<D, B, lex_span(D, B, BC, 0), lex_span(D, B, BC, 1),
                           lex_span(D, B, BC, 2), lex_span(D, B, BC, 3)>;

//: (G4) The D = 2 compatibility view -- BlockPlan -- is RETIRED.  Its geometry
//: law and its union-table entry accounting live in LevelPlan above, where
//: they are stated once for every depth; nothing below this line speaks
//: lead/tail.

//: THE UNION TABLE'S ONE STRUCTURAL CONSTANT.  Tokens are the table's minor axis
//: and they are packed thirty-two to a word, which is the lockstep unit's width,
//: so the cook reads one word per unit per side and needs no re-packing step.  A
//: sequence that does not fill its last word leaves the tail bits ZERO.
constexpr int kTokBits = 32;
__host__ __device__ constexpr int tok_words(int L) { return (L + kTokBits - 1) / kTokBits; }

// ------------------------------------------------------- the facts tables
//: one warp per token word, four warps per CTA.
constexpr int kFactsThreads = 128;

//: THE CALL CLASSES THIS PASS IS SPECIALIZED OVER -- the prefill half of the four
//: (`rola/engine/types.py`'s `Mask`).  Decode's two never reach this kernel and are
//: not instantiated for it, so an unmatched value is the int_switch's refusal.
enum ChunkMask : int { kPrefillStateless = 1, kPrefillStateful = 2 };

//: THE TABLES ARE BUILT ONCE FOR THE WHOLE GRID, BOTH ROW SETS IN ONE PASS.  A
//: lockstep unit takes thirty-two consecutive tokens, one per lane, and votes once
//: per entry; the vote IS the word, so nothing is packed afterwards and nothing is
//: zeroed first.
//:
//: The two grains are one partition refining another -- the atom rectangle is
//: `kAtomLeaves` leaves and the owner rectangle is `BC` of them -- so the write
//: side is voted at the ATOM grain and each owner row is the OR of the atom rows
//: inside it.  The digits are read once and voted once; nothing is evaluated twice.
//: Its cost does not depend on `BC`: a lane reads its token's amplitude row once
//: per level that has entries, whatever the spans partition it into.
//: docs/internals/facts/facts_kernel.md#the-facts-pass
template <class PLAN, int MASK>
__global__ void chunk_facts_tables(const __nv_bfloat16* __restrict__ pread,
                                   const __nv_bfloat16* __restrict__ pwrite,
                                   uint32_t* __restrict__ tbl, uint32_t* __restrict__ atom_tbl,
                                   int L) {
  using LP = PLAN;
  constexpr int D = LP::depth;
  constexpr int B = LP::kWidth;
  using AP = LevelPlan<D, B, kAtomLeaves>;
  constexpr int kWtot = LP::col_base(LP::depth);
  static_assert(kWtot == AP::col_base(AP::depth),
                "both grains read the SAME amplitude row: the packed column layout is "
                "the widths policy's and carries no block geometry");
  const int tokWords = tok_words(L);
  const int w = blockIdx.x * (blockDim.x / kBallotLanes) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (w >= tokWords) return;  // a whole lockstep unit leaves together
  //: A TOKEN PAST THE END READS TOKEN ZERO AND VOTES DEAD.  Its bit has no
  //: consumer and reading in range is what keeps the lockstep unit converged.
  const int t = w * kTokBits + lane;
  const bool real = t < L;
  const long off = (long)blockIdx.y * L * kWtot + (long)(real ? t : 0) * kWtot;

  //: ADDRESSING DISCIPLINE, THE SAME ONE THE CHUNK KERNEL KEEPS.  Every 64-bit
  //: address is formed ONCE and afterwards only advanced, by a stride that is
  //: itself 64-bit -- so no `pointer + int` widening multiply reaches a body.
  const long entryStride = tokWords;
  const __nv_bfloat16* const ampRead = pread + off;
  const __nv_bfloat16* const ampWrite = pwrite + off;

#define ROLA_FACTS_VOTE(P_, SPAN_) \
  lane_ballot(real && (or_run<SPAN_>(P_) & kBf16MagnitudeMask) != 0u)

  //: one stanza per level, spans from the plan -- a whole level emits nothing,
  //: exactly as its clause is not evaluated.  Amp row layout is
  //: [level 0 | level 1 | ...], each `width(l)` wide.
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

  //: the write side at BOTH grains.  An owner entry's digit run is a whole number
  //: of atom runs, so its ballot is the OR of theirs -- bit for bit, because the
  //: magnitude test distributes over the run's OR.
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
    uint32_t* out = tbl + ((long)(2 * blockIdx.y + 0) * LP::entries) * entryStride + w;
    ROLA_FACTS_LEVEL(LP, ampRead, out, 0)
    ROLA_FACTS_LEVEL(LP, ampRead, out, 1)
    ROLA_FACTS_LEVEL(LP, ampRead, out, 2)
    ROLA_FACTS_LEVEL(LP, ampRead, out, 3)
  }

  //: THE ATOM ROWS ARE THE WRITE SIDE'S: which atoms this routing writes is what
  //: keys residency, and reads of never-written atoms are inert.  A stateless call
  //: compiles none of this, so it pays a stateless call's price; a stateful call
  //: with no arena to admit into is the runtime-null OUTPUT skip, and takes the
  //: owner-grain emission alone -- which is the cheaper one when nothing needs the
  //: finer rows.
  if constexpr (MASK == kPrefillStateful) {
    if (atom_tbl != nullptr) {
      uint32_t* out = tbl + ((long)(2 * blockIdx.y + 1) * LP::entries) * entryStride + w;
      uint32_t* aout = atom_tbl + ((long)blockIdx.y * AP::entries) * entryStride + w;
      ROLA_FACTS_LEVEL_BOTH(0)
      ROLA_FACTS_LEVEL_BOTH(1)
      ROLA_FACTS_LEVEL_BOTH(2)
      ROLA_FACTS_LEVEL_BOTH(3)
      return;
    }
  }
  {
    uint32_t* out = tbl + ((long)(2 * blockIdx.y + 1) * LP::entries) * entryStride + w;
    ROLA_FACTS_LEVEL(LP, ampWrite, out, 0)
    ROLA_FACTS_LEVEL(LP, ampWrite, out, 1)
    ROLA_FACTS_LEVEL(LP, ampWrite, out, 2)
    ROLA_FACTS_LEVEL(LP, ampWrite, out, 3)
  }
#undef ROLA_FACTS_LEVEL_BOTH
#undef ROLA_FACTS_LEVEL
#undef ROLA_FACTS_VOTE
}

// --------------------------------------------------------- the support plane
//: THE SAME EXACT-NONZERO TEST THE UNION VOTE APPLIES, AT DIGIT GRAIN.  The union
//: table answers "does this token's support meet this owner's digit RUN"; the
//: support plane answers "which digits", which is what lets a consumer synthesize
//: only the live coefficients.  One warp votes one 32-digit word, so the map from
//: amplitudes to bits is a single ballot and nothing is packed afterwards.
//: docs/internals/facts/facts_kernel.md#the-support-plane
constexpr int kSupportThreads = 128;

//: the 32-bit words one level's bitmap occupies.  A level narrower than a word
//: still takes one, so a consumer's per-level stride is a shift at every width.
constexpr int support_words(int B) { return (B + kTokBits - 1) / kTokBits; }

//: `[BH, 2, L, D, SW]` uint32, side 0 read.  The minor axes are (level, word) so a
//: consumer's whole slice for one (token, level) is ONE aligned load.
template <int D, int B>
__global__ void support_plane_kernel(const __nv_bfloat16* __restrict__ pread,
                                     const __nv_bfloat16* __restrict__ pwrite,
                                     uint32_t* __restrict__ sup, int L) {
  constexpr int SW = support_words(B);
  constexpr int kRow = D * SW;
  constexpr int kWtot = D * B;
  const int lane = threadIdx.x & 31;
  const long unit = (long)blockIdx.x * (kSupportThreads / kBallotLanes) + (long)(threadIdx.x >> 5);
  if (unit >= (long)L * kRow) return;
  const int tok = (int)(unit / kRow);
  const int rem = (int)(unit - (long)tok * kRow);
  const int l = rem / SW, dig = (rem - l * SW) * kTokBits + lane;
  const int side = (int)blockIdx.z;
  const __nv_bfloat16* const amp =
      (side == 0 ? pread : pwrite) + (long)blockIdx.y * L * kWtot + (long)tok * kWtot + (long)l * B;
  const bool live =
      dig < B
      && ((uint32_t)*reinterpret_cast<const unsigned short*>(amp + dig) & kBf16MagnitudeMask) != 0u;
  const uint32_t word = lane_ballot(live);
  if (lane == 0) sup[(((long)blockIdx.y * 2 + side) * L + tok) * kRow + rem] = word;
}

// ------------------------------------------------------- the atom-grain stats
//: THE STATS PASS' REDUCTION.  Which atoms does this call's routing WRITE?  The
//: answer keys the state's residency, so it is taken at the invariant quantum
//: (`kAtomLeaves`) and never at `BC` -- the derivation, the exactness argument and
//: the rejected coarser reductions are in docs/internals/paging/stats.md.
//:
//: how many threads and how much of the token axis one CTA of the reduction
//: holds resident.  The tile is a RESIDENCY quantity, not an axis: it bounds the
//: shared word stream so the CTA's footprint is independent of `L`.
constexpr int kAtomThreads = 128;
constexpr int kAtomTileWords = 256;

//: THE REDUCTION, one CTA per (bh, upper prefix).  A prefix's atoms share every
//: clause row but the innermost, so the shared rows are ANDed into a word stream
//: ONCE and each innermost row is then ANDed against that stream and OR-reduced
//: over the token axis.  An atom is live as soon as one word survives, so the CTA
//: leaves the moment all of its atoms are live -- which makes the DENSE case the
//: cheap one.
template <int D, int B>
__global__ void atom_bitmap_kernel(const uint32_t* __restrict__ tbl,
                                   uint8_t* __restrict__ atom_bits, int L) {
  using LP = LevelPlan<D, B, kAtomLeaves>;
  static_assert(LP::base_aligned(0),
                "BLOCK GEOMETRY: a block's first leaf is not BC-aligned, so its leaves "
                "are not one aligned run and the state's atom addressing does not hold");
  constexpr int kClauses = LP::clauses;
  constexpr int kAtoms = (int)LP::blocks;
  //: the innermost clause level's group count: the atoms of one prefix, and by
  //: the plan's radix the LEAST significant part of the block index.
  constexpr int kInner = kClauses > 0 ? LP::groups_at(kClauses - 1) : 1;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int blkBase = blockIdx.x * kInner;
  //: THE BITMAP IS ONE BYTE PER ATOM, so `BH * atoms` is a 32-bit displacement
  //: wherever the state those atoms address exists at all (that state is 4160
  //: bytes an atom).  One pinned base, byte offsets after it.
  const uint32_t outOff = (uint32_t)blockIdx.y * (uint32_t)kAtoms + (uint32_t)blkBase;

  //: NO CLAUSE AT ALL is the whole slot space in one atom: written whenever the
  //: sequence has a token, which the caller's own topology guarantees.
  if constexpr (kClauses == 0) {
    if (tid == 0) *reinterpret_cast<uint8_t*>(gather_ptr(atom_bits, outOff)) = 1;
    return;
  } else {
    const int tokWords = tok_words(L);
    //: ADDRESSING DISCIPLINE, THE FORM'S: ONE 64-bit address per (bh), formed
    //: here and never re-derived; every row and every word after it is a 32-bit
    //: BYTE displacement that zero-extends, so no `pointer + int` widening
    //: multiply reaches a loop body.  The displacement's bound is one `bh`'s
    //: slice of the table, `entries * tok_words(L) * 4` bytes.
    const uint32_t* const base =
        pin_address(tbl + (long)blockIdx.y * ((long)LP::entries * (long)tokWords));
    const uint32_t rowBytes = (uint32_t)tokWords * (uint32_t)sizeof(uint32_t);
    //: NAMED SCALARS, never an array -- the consumer's rule, for the same
    //: reason: a runtime-indexed offset array is demoted to a stack frame.
    //: Unused upper clauses read row zero and are never consulted.
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

    __shared__ uint32_t sUp[kAtomTileWords];
    __shared__ uint8_t sLive[kInner];
    __shared__ int sAllLive;
    for (int a = tid; a < kInner; a += kAtomThreads) sLive[a] = 0;
    if (tid == 0) sAllLive = 0;
    rendezvous();

    for (int w0 = 0; w0 < tokWords; w0 += kAtomTileWords) {
      const int n = min(kAtomTileWords, tokWords - w0);
      for (int i = tid; i < n; i += kAtomThreads) {
        const uint32_t off = (uint32_t)(w0 + i) * (uint32_t)sizeof(uint32_t);
        uint32_t m = 0xFFFFFFFFu;
        if constexpr (1 < kClauses)
          m &= *reinterpret_cast<const uint32_t*>(gather_ptr(base, offUa + off));
        if constexpr (2 < kClauses)
          m &= *reinterpret_cast<const uint32_t*>(gather_ptr(base, offUb + off));
        if constexpr (3 < kClauses)
          m &= *reinterpret_cast<const uint32_t*>(gather_ptr(base, offUc + off));
        sUp[i] = m;
      }
      rendezvous();
      for (int a = tid >> 5; a < kInner; a += kAtomThreads / kBallotLanes) {
        if (sLive[a]) continue;
        const uint32_t rowOff = offInner + (uint32_t)a * rowBytes;
        bool hit = false;
        for (int i = lane; i < n; i += kBallotLanes) {
          const uint32_t off = (uint32_t)(w0 + i) * (uint32_t)sizeof(uint32_t);
          hit |=
              (sUp[i] & *reinterpret_cast<const uint32_t*>(gather_ptr(base, rowOff + off))) != 0u;
        }
        //: the vote is the WHOLE unit's -- every lane issues it, lane 0 files it.
        const uint32_t any = lane_ballot(hit);
        if (lane == 0 && any != 0u) sLive[a] = 1;
      }
      rendezvous();
      //: THE EXIT IS CTA-UNIFORM: one warp reads every flag behind the barrier
      //: and publishes a single answer, so no lane leaves a `rendezvous` alone.
      if (tid < kBallotLanes) {
        bool all = true;
        for (int a = tid; a < kInner; a += kBallotLanes) all &= sLive[a] != 0;
        const uint32_t every = lane_ballot(all);
        if (tid == 0 && every == 0xFFFFFFFFu) sAllLive = 1;
      }
      rendezvous();
      if (sAllLive) break;
    }
    for (int a = tid; a < kInner; a += kAtomThreads)
      *reinterpret_cast<uint8_t*>(gather_ptr(atom_bits, outOff + (uint32_t)a)) = sLive[a];
  }
}

// ------------------------------------------------------ the block-grain liveness
//: THE REDUCTION'S ROW GROUP: at most one innermost row per warp per side.  The
//: prefix carve amortizes the UPPER-clause stream, so a block with no upper
//: clause has nothing to amortize and one CTA per prefix would be one CTA for
//: the whole reduction; re-reading the upper stream per group buys the grid
//: back (docs/internals/facts/facts_kernel.md#block-bitmap).
constexpr int block_bitmap_group(int inner) {
  return inner < (kAtomThreads / kBallotLanes) / 2 ? inner : (kAtomThreads / kBallotLanes) / 2;
}

//: THE LIVENESS EPILOGUE, at the LAUNCH's block grain: `atom_bitmap_kernel`'s
//: reduction re-instantiated at `BC`, over BOTH sides, with the page table's
//: residency condensed beside them.  The three facts are emitted SEPARATELY --
//: one byte per (bh, block), bits {READ, WRITE, STATE} -- because what a skipped
//: block would have contributed differs by call class and by backing, so the
//: predicate over them is the host's (docs/internals/facts/facts_kernel.md#block-bitmap).
template <int D, int B, int BC>
__global__ void block_bitmap_kernel(const uint32_t* __restrict__ tbl,
                                    const int32_t* __restrict__ page_tbl,
                                    uint8_t* __restrict__ block_bits, int L) {
  using LP = LevelPlan<D, B, BC>;
  static_assert(LP::base_aligned(0),
                "BLOCK GEOMETRY: a block's first leaf is not BC-aligned, so its leaves "
                "are not one aligned run and the state's atom addressing does not hold");
  constexpr int kClauses = LP::clauses;
  constexpr int kBlocks = (int)LP::blocks;
  constexpr int kInner = kClauses > 0 ? LP::groups_at(kClauses - 1) : 1;
  constexpr int kWarps = kAtomThreads / kBallotLanes;
  constexpr int kGroup = block_bitmap_group(kInner);
  static_assert(kInner % kGroup == 0, "a row group must divide the prefix's inner run");
  constexpr int kAtomsPer = BC / kAtomLeaves;
  static_assert(BC % kAtomLeaves == 0, "a block is a whole number of page atoms");
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int blkBase = blockIdx.x * kGroup;
  const int innerBase = blkBase % kInner;
  const uint32_t outOff = (uint32_t)blockIdx.y * (uint32_t)kBlocks + (uint32_t)blkBase;

  //: the two sides share one token tile, so the read row's stream and the write
  //: row's are formed together and the tile is read once.
  __shared__ uint32_t sUp[2][kAtomTileWords];
  __shared__ uint8_t sLive[2 * kGroup];
  __shared__ uint8_t sState[kGroup];
  __shared__ int sAllLive;
  for (int e = tid; e < 2 * kGroup; e += kAtomThreads) sLive[e] = 0;
  for (int a = tid; a < kGroup; a += kAtomThreads) sState[a] = 0;
  if (tid == 0) sAllLive = 0;
  rendezvous();

  //: AN UNMAPPED ATOM'S ENTRY STATE IS PROVABLY ZERO, so residency is the exact
  //: witness for "this block may carry state in"; the dense backing has no such
  //: map and the host degrades the predicate rather than guessing one.
  if (page_tbl != nullptr) {
    const int32_t* const pt = pin_address(page_tbl);
    const uint32_t bhAtoms = (uint32_t)blockIdx.y * (uint32_t)(LP::slots / kAtomLeaves);
    for (int e = tid; e < kGroup * kAtomsPer; e += kAtomThreads) {
      const int a = e / kAtomsPer;
      const uint32_t atom = bhAtoms + (uint32_t)(LP::block_base(blkBase + a) / kAtomLeaves)
                            + (uint32_t)(e - a * kAtomsPer);
      if (*reinterpret_cast<const int32_t*>(gather_ptr(pt, atom * (uint32_t)sizeof(int32_t))) >= 0)
        sState[a] = 1;
    }
  }

  //: NO CLAUSE AT ALL is the whole slot space in one block: live on both sides
  //: whenever the sequence has a token, which the caller's topology guarantees.
  if constexpr (kClauses == 0) {
    for (int e = tid; e < 2 * kGroup; e += kAtomThreads) sLive[e] = 1;
  } else {
    const int tokWords = tok_words(L);
    //: ADDRESSING DISCIPLINE, THE FORM'S: one 64-bit base per (bh, side), every
    //: row and word after it a 32-bit byte displacement that zero-extends.
    const uint32_t* const baseR =
        pin_address(tbl + ((long)(2 * blockIdx.y + 0) * (long)LP::entries) * (long)tokWords);
    const uint32_t* const baseW =
        pin_address(tbl + ((long)(2 * blockIdx.y + 1) * (long)LP::entries) * (long)tokWords);
    const uint32_t rowBytes = (uint32_t)tokWords * (uint32_t)sizeof(uint32_t);
    //: NAMED SCALARS, never an array -- a runtime-indexed offset array is
    //: demoted to a stack frame.  The entry rows are the block's, so the two
    //: sides share every displacement and differ only in their base.
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

    for (int w0 = 0; w0 < tokWords; w0 += kAtomTileWords) {
      const int n = min(kAtomTileWords, tokWords - w0);
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
      for (int e = tid >> 5; e < 2 * kGroup; e += kWarps) {
        if (sLive[e]) continue;
        const int s = e / kGroup;
        const uint32_t* const base = s == 0 ? baseR : baseW;
        const uint32_t rowOff = offInner + (uint32_t)(innerBase + e - s * kGroup) * rowBytes;
        bool hit = false;
        for (int i = lane; i < n; i += kBallotLanes) {
          const uint32_t off = (uint32_t)(w0 + i) * (uint32_t)sizeof(uint32_t);
          hit |= (sUp[s][i] & *reinterpret_cast<const uint32_t*>(gather_ptr(base, rowOff + off)))
                 != 0u;
        }
        //: the vote is the WHOLE unit's -- every lane issues it, lane 0 files it.
        const uint32_t any = lane_ballot(hit);
        if (lane == 0 && any != 0u) sLive[e] = 1;
      }
      rendezvous();
      //: THE EXIT IS CTA-UNIFORM: one warp reads every flag behind the barrier
      //: and publishes a single answer, so no lane leaves a `rendezvous` alone.
      if (tid < kBallotLanes) {
        bool all = true;
        for (int e = tid; e < 2 * kGroup; e += kBallotLanes) all &= sLive[e] != 0;
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
