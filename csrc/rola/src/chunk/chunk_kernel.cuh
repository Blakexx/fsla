// csrc/rola/src/chunk/chunk_kernel.cuh -- THE CHUNK CONSUMER (adopted from
// the ladder's rung 6G; the ladder repo (rola-wt-ladder) holds the
// rung-by-rung
// derivation and its measurement receipts).  Device world only: constants,
// LevelPlan, Geom, the compacted kernel and the union-table kernel.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <type_traits>

#include "arch_caps.cuh"
#include "design.cuh"
#include "ops.cuh"

namespace rola::chunk {
//: the raw launch arguments shared by the dispatch TU and the arm surface.
struct ChunkParams {
  const __nv_bfloat16 *pr, *pw, *gw, *v;
  const uint32_t* tbl;
  float *o, *den;
  //: THE STATE PLANE, in and out.  Null means "no state I/O": a fresh entry
  //: state on the way in, no handle asked for on the way out.  Whether a launch
  //: carries state at all is a COMPILE-TIME axis of the kernel; these nulls say
  //: which END of it this call uses (docs/internals/chunk/chunk_kernel.md).
  const float* s_in;
  float* s_out;
  //: the atom -> slot map, `[BH, N/16]` int32 with `-1` for an absent atom; null
  //: is the DENSE plane, where the slot IS the atom.
  const int32_t* page_tbl;
  //: THE BLOCK-GRAIN LIVENESS BITMAP, one byte per (bh, block).  A dead block's
  //: CTA exits on it before it touches anything; a null bitmap runs every block.
  //: `run_table` is the truth table over the byte's three bits and it is the
  //: HOST's (docs/internals/chunk/chunk.md#run-table).
  const uint8_t* block_bits;
  uint32_t run_table;
  int BH, L;
  //: V's per-token row stride in BYTES. The value stream arrives token-major
  //: (`[B, T, H, d_v]`), so a head's rows are `H * d_v` apart; the head count is
  //: this over `d_v` and is not carried twice.
  int v_row_bytes;
};
}  // namespace rola::chunk

namespace rola::chunk {

using namespace denseref::ops;
namespace design = denseref::design;

static_assert(design::kStateResidency == design::StateResidency::REGISTERS && kMmaAcceptsRegisterA,
              "SELECTION UNSATISFIABLE GIVEN AVAILABLE OPERATIONS: this kernel's "
              "fold accumulator IS the readout's A operand, so it needs a state "
              "that lives in registers and a built `mma` that reads A out of "
              "them; otherwise the state takes a memory round trip and this is a "
              "different algorithm.");

static constexpr int kBuildMagic = 6700101;

//: THE SKIP PREDICATE IS A TRUTH TABLE over the liveness byte's three bits, so
//: every call class is ONE host constant and the kernel carries no branching
//: taxonomy (docs/internals/chunk/chunk.md#run-table).
__device__ __forceinline__ bool live_block(uint8_t v, uint32_t run_table) {
  return ((run_table >> (v & 7u)) & 1u) != 0u;
}

//: THE INVARIANT QUANTUM: the leaves one page granule holds, and the rectangle
//: the stats pass keys residency to.  It is the MMA's k-extent because the
//: state's own tiling is (the host mirror is `rola.ops.paging.MMA_K_QUANTUM`;
//: docs/internals/paging/stats.md).
static constexpr int kAtomLeaves = kMmaK;

static constexpr int kD = 64;    // d_v
static constexpr int kPadE = 8;  // bf16 row padding, elements (16 B)
static constexpr int kPadF = 4;  // fp32 row padding, elements (16 B)

//: An index-list entry is 16 bits, so a super-chunk must lie inside one window;
//: the walk forces a cut at every window line -- docs/internals/chunk/chunk_kernel.md#k-window
static constexpr int kWindow = 65536;
static_assert((kWindow & (kWindow - 1)) == 0,
              "the token -> (window, local) split is a shift and a mask");
//: THE BOUND ON L is the 32-bit gather displacement: the widest plane row is the
//: fp32 output at kD*4 bytes -- docs/internals/chunk/chunk_kernel.md#k-window
static constexpr long kMaxGatherL = (1L << 32) / (kD * (long)sizeof(float));

//: Schedule-ahead distance ABOVE the pipeline minimum. Zero by measurement, not by
//: default -- docs/internals/chunk/chunk_kernel.md#cook-distance
static constexpr int kAheadExtra = 0;

// The carve is derived: one MMA unit owns a private 16-row slice of the
// transposed state's m axis -- docs/internals/chunk/chunk_kernel.md#the-carve
static constexpr int kUnits = kD / kMmaM;
static constexpr int kGroupThreads = kUnits * kMmaUnitThreads;

//: A second slot group is added exactly where one group's k-run would exceed the
//: measured KD = 8 shape -- docs/internals/chunk/chunk_kernel.md#the-carve
constexpr int groups_for(int BC) { return BC >= 256 ? 2 : 1; }
constexpr int threads_for(int BC) { return kGroupThreads * groups_for(BC); }

// --------------------------------------------------------------- block plan

//: THE GEOMETRY LAW over D levels of one uniform power-of-two width B. Every
//: clause below is the law's, and a (widths, BC) pair breaking one does not
//: compile. Widths are read through `width(l)` so a jagged policy can answer from
//: a table without touching a loop nest -- docs/internals/chunk/chunk_kernel.md#geometry-law
constexpr long ipowl(int b, int e) { return e == 0 ? 1 : b * ipowl(b, e - 1); }

template <int D, int B, int BC>
struct LevelPlan {
  static_assert(D >= 1 && D <= 4, "the level plan is built to depth four");
  static_assert((B & (B - 1)) == 0 && B >= 8 && B <= 256,
                "one uniform power-of-two width per topology");
  static constexpr long slots = ipowl(B, D);
  //: how many trailing levels the block spans ENTIRELY
  static constexpr int whole = BC >= ipowl(B, D)         ? D
                               : (long)BC >= ipowl(B, 3) ? 3
                               : (long)BC >= ipowl(B, 2) ? 2
                               : BC >= B                 ? 1
                                                         : 0;
  //: the one partially spanned level's run (1 = no partial level)
  static constexpr int partial_span = (int)(BC / ipowl(B, whole));
  static_assert((long)partial_span * ipowl(B, whole) == BC,
                "BLOCK GEOMETRY: BC is not a rectangle in digit space");
  static_assert(partial_span == 1 || B % partial_span == 0,
                "BLOCK GEOMETRY: the partial run must divide its level");
  static constexpr int singles = D - whole - (partial_span > 1 ? 1 : 0);
  static_assert(singles >= 0, "BLOCK GEOMETRY: more block than slot space");
  static constexpr int width(int) { return B; }  // the uniform widths policy
  static constexpr long blocks = slots / BC;
  //: per-level spans, most-significant level first: `singles` levels at one
  //: digit, then the partial run (if any), then the whole levels.
  static constexpr int span_at(int l) {
    return l < singles ? 1 : (partial_span > 1 && l == singles) ? partial_span : B;
  }
  //: a level's table-entry count: its group count when tested, zero when whole
  static constexpr int entries_at(int l) { return span_at(l) < B ? B / span_at(l) : 0; }
  static constexpr int groups_at(int l) { return span_at(l) < B ? B / span_at(l) : 1; }
  //: how many clause words the form ANDs per side
  static constexpr int clauses = singles + (partial_span > 1 ? 1 : 0);
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
  //: table entries per side: full width per single-digit level, width/run for
  //: the partial one, NOTHING for a whole level -- a SUM over levels, never a
  //: product, which is why the table does not grow with N.
  static constexpr int entries = singles * B + (partial_span > 1 ? B / partial_span : 0);
  //: A block's leaves are a CONTIGUOUS, BC-ALIGNED run in canonical leaf order, so
  //: the block is named by its first leaf -- docs/internals/chunk/chunk_kernel.md#leaf-order-law
  //: THE LEVEL IS A TEMPLATE PARAMETER, not an argument: `off_at`'s recursive
  //: constexpr divisor does not fold otherwise -- docs/internals/chunk/chunk_kernel.md#template-level
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
           || ((groups_at(l) == 1 || (ipowl(B, D - 1 - l) * span_at(l)) % BC == 0)
               && base_aligned(l + 1));
  }
  static_assert(base_aligned(0),
                "BLOCK GEOMETRY: a block's first leaf is not BC-aligned, so its "
                "leaves are not one aligned run and the state's atom addressing "
                "does not hold");
};

//: (G4) The D = 2 compatibility view -- BlockPlan -- is RETIRED.  Its geometry
//: law and its union-table entry accounting live in LevelPlan above, where
//: they are stated once for every depth; nothing below this line speaks
//: lead/tail.

//: THE FORM'S BODY AS A REAL __forceinline__ FUNCTION: ptxas outlined the
//: grown two-clause form into a CALL with an ABI frame (stack stores the
//: spill gate refuses), and the always_inline attribute is not honored on a
//: lambda -- it is on a function.
template <int kClauses, int kFormWordsT, int kThreadsT>
__device__ __forceinline__ void form_impl(int& formEnd, int limit, int tid, const uint32_t* rowR,
                                          const uint32_t* rowW, uint32_t cA, uint32_t cB,
                                          uint32_t cC, uint32_t cD, uint32_t tailMask, int lastWord,
                                          uint32_t* sfR, uint32_t* sfW) {
  for (int w = formEnd + tid; w < limit; w += kThreadsT) {
    const uint32_t off = (uint32_t)w * (uint32_t)sizeof(uint32_t);
    //: the low clause counts keep rung 6's exact expression shapes (no seeded
    //: AND) -- codegen parity first, generality above two via the seed.
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
  formEnd = limit;
}

//: THE UNION TABLE'S ONE STRUCTURAL CONSTANT.  Tokens are the table's minor axis
//: and they are packed thirty-two to a word, which is the lockstep unit's width,
//: so the cook reads one word per unit per side and needs no re-packing step.  A
//: sequence that does not fill its last word leaves the tail bits ZERO.
constexpr int kTokBits = 32;
__host__ __device__ constexpr int tok_words(int L) { return (L + kTokBits - 1) / kTokBits; }

//: THE FORMED RING'S DEPTH, in words per side. A RESIDENCY quantity, not an axis:
//: `2 * kFormWords * 4` bytes must leave every arm's staging depth and CTA target
//: unmoved, and `arms()` reports both so it is checked. Build-time overridable so
//: a gate can make the wait fire -- docs/internals/chunk/chunk_kernel.md#form-ring
#ifndef CHUNK_FORM_WORDS
#define CHUNK_FORM_WORDS 128
#endif
constexpr int kFormWords = CHUNK_FORM_WORDS;
static_assert((kFormWords & (kFormWords - 1)) == 0,
              "the formed ring's index is a mask, so its depth is a power of two");

//: The walk publishes only what another warp needs -- the cursor and the ran-out
//: flag; four slots, not two, to keep the buffer behind it 16-byte aligned
//: -- docs/internals/chunk/chunk_kernel.md#the-walk
constexpr int kCookWord = 0, kCookOpen = 1;
constexpr int kCookSlots = 4;

//: THE ASCENDING SCHEDULE, as a component rather than as three closures.  The
//: closing rule is THE load-bearing law of the compaction and it must have ONE
//: implementation: the forward drives this, and so does Scan A, whose partition
//: is its own but whose rule is the same one.  Its mirror is `Descent` in
//: backward.cuh -- docs/internals/chunk/chunk_kernel.md#the-walk
template <int C, int kClauses, int kThreadsT, int kIdxSlotT, int kHdrT>
struct Ascent {
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

  int formEnd;
  int cookWord, cookNr, cookNw;
  bool cookOpen;
  int chunkWin, nextLineW, winBaseW;

  static constexpr int kWinWords = kWindow / kTokBits;

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

  //: THE STATE IS INITIALISED WHERE THE DRIVER DECLARED IT, not in one block:
  //: `formEnd` before the form, the cook's cursor before the walk.  The two live
  //: ranges are what they were as locals, which is why the extraction leaves the
  //: forward's codegen untouched.
  __device__ __forceinline__ void begin_form() { formEnd = 0; }
  __device__ __forceinline__ void begin_walk() {
    cookWord = 0;
    cookNr = 0;
    cookNw = 0;
    cookOpen = false;
    chunkWin = 0;
    nextLineW = kWinWords;
    winBaseW = 0;
  }

  __device__ __forceinline__ void form(int tid) {
    const int head = scook[kCookWord] + kFormWords;
    const int limit = head < tokWords ? head : tokWords;
    form_impl<kClauses, kFormWords, kThreadsT>(formEnd, limit, tid, rowR, rowW, cA, cB, cC, cD,
                                               tailMask, lastWord, sfR, sfW);
  }

  __device__ __forceinline__ void form_if_low(int tid) {
    if (formEnd < tokWords && formEnd - scook[kCookWord] < kFormWords / 2) form(tid);
  }

  __device__ __forceinline__ void walk(int d, int lane) {
    uint16_t* lrd = iread(d);
    uint16_t* lwr = iwrite(d);
    uint8_t* ljt = ithr(d);
    const bool exists = cookOpen || cookWord < tokWords;
    int nr = cookOpen ? cookNr : 0;
    int nw = cookOpen ? cookNw : 0;
    if (!cookOpen) {
      chunkWin = cookWord / kWinWords;
      winBaseW = chunkWin * kWinWords;
      nextLineW = winBaseW + kWinWords;
    }
    bool closed = cookWord >= tokWords;
    const uint32_t lt = (1u << lane) - 1u;
#pragma unroll 1
    while (cookWord < formEnd) {
      //: THE WINDOW LINE, as an EQUALITY against a maintained register --
      //: the walk advances a word at a time, so a line is never jumped.  A
      //: super-chunk holding entries closes at the line; an empty one slides
      //: its window forward, exactly as it crosses words.
      if (cookWord == nextLineW) {
        if (nr != 0 || nw != 0) {
          closed = true;
          break;
        }
        ++chunkWin;
        winBaseW = nextLineW;
        nextLineW += kWinWords;
      }
      const int slot = cookWord & (kFormWords - 1);
      const uint32_t rm = sfR[slot];
      const uint32_t wm = sfW[slot];
      const int prx = bit_count(rm & lt);
      const int pwx = bit_count(wm & lt);
      const bool lvR = ((rm >> lane) & 1u) != 0u;
      const bool lvW = ((wm >> lane) & 1u) != 0u;
      const int cut = first_set(lane_ballot((lvR && nr + prx >= C) || (lvW && nw + pwx >= C)));
      const uint32_t keep = cut >= kTokBits ? 0xFFFFFFFFu : ((1u << cut) - 1u);
      const int t = (cookWord - winBaseW) * kTokBits + lane;
      if (lane < cut) {
        if (lvR) {
          lrd[nr + prx] = (uint16_t)t;
          ljt[nr + prx] = (uint8_t)(nw + pwx + (lvW ? 1 : 0));
        }
        if (lvW) lwr[nw + pwx] = (uint16_t)t;
      }
      nr += bit_count(rm & keep);
      nw += bit_count(wm & keep);
      if (cut < kTokBits) {
        if (lane == 0) {
          sfR[slot] = rm & ~keep;
          sfW[slot] = wm & ~keep;
        }
        closed = true;
        break;
      }
      ++cookWord;
      if (cookWord == tokWords) {
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
    //: THE BUFFER'S TAIL.  A slot past the gathered count names token zero, so
    //: every gather stays in range without a predicate, and carries a threshold
    //: of zero, so its Gram row is empty.  Its panel row is zeroed at production
    //: and its output row is never drained.
#pragma unroll
    for (int k = 0; k < (C + kTokBits - 1) / kTokBits; ++k) {
      const int i = lane + k * kTokBits;
      if (i < C) {
        if (i >= nr) {
          lrd[i] = 0;
          ljt[i] = 0;
        }
        if (i >= nw) lwr[i] = 0;
      }
    }
    const bool wasOpen = cookOpen;
    cookOpen = false;
    if (lane == 0) {
      int* h = ihdr(d);
      h[0] = nr;
      h[1] = nw;
      h[2] = exists ? 1 : 0;
      h[3] = chunkWin;  // whose window the LOCAL indices live in
      scook[kCookWord] = cookWord;
      if (wasOpen) scook[kCookOpen] = 0;
    }
  }

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

//: THE ROUTED KRONECKER PRODUCTION, as a component.  A thread owns one token row
//: and a contiguous run of slots; the coefficient it publishes is the product of
//: one amplitude per level at that slot's digits, and the digits are LOOP INDICES
//: rather than arithmetic -- which is why this is the same code for every level
//: count and why it must be the same code for every scan.  The forward and Scan B
//: drove byte-identical copies of it before this component existed; Scan A would
//: have made a third -- docs/internals/chunk/chunk_kernel.md#the-production
//:
//: The per-thread COLUMN OFFSETS and the slot displacements are the caller's:
//: they are plan arithmetic, and the multi-group carve computes them differently
//: from the one-group one.  What is shared is the production itself.
template <int kDepth, int kLeadPerThr, int kTailVec, int kUppers>
struct Producer {
  uint32_t colFast, colInner, colUpA, colUpB;
  int pOff[kLeadPerThr][kTailVec];

  //: `have` is the row's existence, and a row past the gathered count must be
  //: EXACTLY zero -- zeroed at the LEADING amplitude rather than at the product,
  //: so one select per digit covers the whole run.
  //: `bs` is a BYTE BASE.  Every caller pins its amplitude plane in bytes so the
  //: one 64-bit multiply is the stride the displacements already carry; taking it
  //: as `void*` is what lets them.
  __device__ __forceinline__ void emit(const void* bs, char* dst, uint32_t tokOff, bool have,
                                       uint32_t gsp) const {
    [[maybe_unused]] const __nv_bfloat16* p0 =
        reinterpret_cast<const __nv_bfloat16*>(gather_ptr(bs, tokOff + colFast));
    const __nv_bfloat16* p1 =
        reinterpret_cast<const __nv_bfloat16*>(gather_ptr(bs, tokOff + colInner));
    __align__(8) unsigned short amp0[kLeadPerThr];
    if constexpr (kDepth == 1) {
      amp0[0] = 0x3F80;
    } else if constexpr (kLeadPerThr == 1) {
      amp0[0] = *reinterpret_cast<const unsigned short*>(p0);
    } else if constexpr (kLeadPerThr == 2) {
      *reinterpret_cast<uint32_t*>(amp0) = *reinterpret_cast<const uint32_t*>(p0);
    } else {
      *reinterpret_cast<uint2*>(amp0) = *reinterpret_cast<const uint2*>(p0);
    }
    if constexpr (kUppers >= 1) {
      uint32_t up =
          splat_bf16x2(*reinterpret_cast<const __nv_bfloat16*>(gather_ptr(bs, tokOff + colUpA)));
      if constexpr (kUppers >= 2)
        up = mul_bf16x2(
            up,
            splat_bf16x2(*reinterpret_cast<const __nv_bfloat16*>(gather_ptr(bs, tokOff + colUpB))));
#pragma unroll
      for (int j = 0; j < kLeadPerThr; ++j) {
        const uint32_t m =
            mul_bf16x2(splat_bf16x2(*reinterpret_cast<const __nv_bfloat16*>(&amp0[j])), up);
        amp0[j] = (unsigned short)(m & 0xffffu);
      }
    }
#pragma unroll
    for (int j = 0; j < kLeadPerThr; ++j) {
      const uint32_t m =
          mul_bf16x2(splat_bf16x2(*reinterpret_cast<const __nv_bfloat16*>(&amp0[j])), gsp);
      amp0[j] = (unsigned short)(m & 0xffffu);
    }
#pragma unroll
    for (int i = 0; i < kTailVec; ++i) {
      __align__(16) uint32_t a1[4];
      *reinterpret_cast<uint4*>(a1) = load_vec16(p1 + i * 8);
#pragma unroll
      for (int j = 0; j < kLeadPerThr; ++j) {
        __align__(16) uint32_t prod[4];
        const uint32_t a2 =
            have ? splat_bf16x2(*reinterpret_cast<const __nv_bfloat16*>(&amp0[j])) : 0u;
#pragma unroll
        for (int e = 0; e < 4; ++e) prod[e] = mul_bf16x2(a2, a1[e]);
        store_vec16(dst + pOff[j][i], *reinterpret_cast<const uint4*>(prod));
      }
    }
  }
};

// ------------------------------------------------------------------ geometry

//: THIS KERNEL'S LIVE REGISTER SET, per (chunk, block width). A measured
//: DECLARATION, not a derivation, and falsifiable: the build fails on any nonzero
//: `localSizeBytes`. ONE floor for both state variants -- docs/internals/chunk/chunk_kernel.md#reg-floor
constexpr int reg_floor(int C, int BC) {
  //: a two-group arm's PER-THREAD demand is one group's, i.e. the BC/2 row --
  //: each thread holds its group's slice, which is the same slice a BC/2
  //: single-group thread holds.
  return BC >= 256 ? reg_floor(C, BC / 2)
         : C == 16 ? (BC <= 64 ? 128 : 168)
         : C == 32 ? (BC <= 32   ? 136
                      : BC <= 64 ? 168
                                 : 236)
                   : (BC <= 32   ? 176
                      : BC <= 64 ? 232
                                 : 255);
}

//: THE SCANS' OWN FLOOR, and it is a SEPARATE declaration because the reverse
//: pass' live set does not fall with `BC` the way the forward's does: the state
//: halves, the chunk-shaped terms do not.  Measured at `C = 32`, the only chunk
//: the scans are built at; every row is falsified per arm by the census' spill
//: and budget gates -- docs/internals/chunk/backward.md#reg-floor-scan
constexpr int reg_floor_scan(int BC) {
  return BC >= 256 ? reg_floor_scan(BC / 2) : BC <= 32 ? 200 : BC <= 64 ? 224 : 236;
}

template <int C, int BC>
struct Geom {
  //: the swizzle AXIS is retired (receipt: noise).  The padded layout is THE layout;
  //: this constant folds every `SWZ ?` ternary to it, and the dead branches
  //: are excised at the e2e cleanup with the compiler as referee.
  static constexpr bool SWZ = false;
  //: the GROUP is the unit of panel/state/output geometry: at GROUPS = 1 the
  //: slice is the block and everything below is rung 5D's byte for byte.
  static constexpr int GRP = groups_for(BC);
  static constexpr int BCg = BC / GRP;
  static constexpr int LdV = SWZ ? kD : kD + kPadE;    // V ring row, bf16 elements
  static constexpr int LdS = SWZ ? BCg : BCg + kPadE;  // R/W row (one group's slice)
  static constexpr int LdG = C + kPadE;                // Gram row, bf16 elements
  //: the fp32 Gram partials, one per group, that the exchange sums -- they
  //: exist only when there is more than one group to exchange between.
  static constexpr int LdGf = C + kPadF;  // fp32 Gram row
  //: The output staging buffer is fp32 and ALWAYS PADDED.  The swizzle exists to
  //: serve `ldmatrix` on the operand buffers; nothing reads this one as an
  //: operand, and its two access patterns are conflict-free under the pad.
  static constexpr int LdO = kD + kPadF;  // output row, fp32 elements

  //: ONE SUPER-CHUNK'S INDEX LISTS: two 16-bit token indices and one causal
  //: threshold byte per slot, behind a header the whole CTA reads.  It rides the
  //: RING rather than the fixed buffers because a super-chunk's lists are needed
  //: from the moment it is staged to the moment it is drained, which is exactly
  //: the ring's own span -- one slot deeper, for the drain that trails by one.
  static constexpr int hdr = 16;
  static constexpr int idx_slot = hdr + ((5 * C + 15) / 16) * 16;
  //: The index ring rides the VALUE ring's accounting one slot per stage plus a
  //: fixed remainder, because its depth is the value depth plus a constant.
  static constexpr int slot = C * LdV * (int)sizeof(__nv_bfloat16) + idx_slot;
  //: The formed schedule and the walk's published state are RESIDENT rather than
  //: ringed with the stages: what they hold is a window on the token axis, whose
  //: size is a property of the walk and not of the staging depth.
  static constexpr int form_bytes =
      2 * kFormWords * (int)sizeof(uint32_t) + kCookSlots * (int)sizeof(int);
  static constexpr int gram_xchg = GRP > 1 ? GRP* C* LdGf*(int)sizeof(float) : 0;
  static constexpr int fixed = GRP * (2 * C * LdS) * (int)sizeof(__nv_bfloat16)
                               + C * LdG * (int)sizeof(__nv_bfloat16) + gram_xchg
                               + GRP * (C * LdO + kUnits * C) * (int)sizeof(float)
                               + (1 + kAheadExtra) * idx_slot + form_bytes;
  static constexpr int stages = design::stages_for(slot, fixed, threads_for(BC), reg_floor(C, BC));
  static constexpr size_t bytes = (size_t)stages * slot + fixed;
};

//: THE RING AND THE LAUNCH BOUND ARE SHARED BY BOTH STATE VARIANTS.  `Geom` is
//: the launch's shared-memory ABI and cannot depend on a template axis the
//: launcher picks per call, and the floor the bound is taken against is one
//: declaration for both kernels (`reg_floor`, above).
template <int C, int BC>
constexpr int kCtaTarget =
    design::cta_target((int)Geom<C, BC>::bytes, threads_for(BC), reg_floor(C, BC));

// -------------------------------------------------------------------- kernel

//: `STATE` compiles the carried state's I/O in or out. A template axis, not a
//: runtime null test. THE STATE POINTERS AND THE PAGE TABLE MUST STAY THE LAST
//: PARAMETERS, after `L`: a parameter inserted mid-list moves every later
//: parameter's constant-bank displacement -- docs/internals/chunk/chunk_kernel.md#state-axis
template <int C, int D, int B, int BC, bool STATE>
__global__ __launch_bounds__(threads_for(BC), (kCtaTarget<C, BC>)) void compacted_kernel(
    const __nv_bfloat16* __restrict__ pread, const __nv_bfloat16* __restrict__ pwrite,
    const __nv_bfloat16* __restrict__ gwrite, const __nv_bfloat16* __restrict__ v,
    const uint32_t* __restrict__ tbl, float* __restrict__ o, float* __restrict__ den, int L,
    int v_row_bytes, const float* __restrict__ s_in, float* __restrict__ s_out,
    const int32_t* __restrict__ page_tbl, const uint8_t* __restrict__ block_bits,
    uint32_t run_table) {
  using LP = LevelPlan<D, B, BC>;
  using G = Geom<C, BC>;
  //: THE LIVENESS EXIT, AND IT IS THE CTA'S FIRST STATEMENT.  CTA-uniform by
  //: construction (`blockIdx` alone), so it precedes every rendezvous and can
  //: leave none half-arrived -- docs/internals/chunk/chunk_kernel.md#liveness-exit
  if (block_bits != nullptr) {
    const uint8_t* const bitRow = pin_address(block_bits + (long)blockIdx.y * LP::blocks);
    if (!live_block(*reinterpret_cast<const uint8_t*>(gather_ptr(bitRow, blockIdx.x)), run_table))
      return;
  }
  static constexpr bool SWZ = false;  // the retired axis, folded (see Geom)
  //: the per-config carve, evaluated: how many slot groups, how many threads.
  constexpr int GROUPS = groups_for(BC);
  constexpr int kThreads = threads_for(BC);
  constexpr int kWarps = kThreads / 32;
  constexpr int kGroupWarps = kGroupThreads / 32;
  //: ONE GROUP'S SLOT SLICE.  Every per-warp compute constant below is the
  //: slice's, so at GROUPS = 1 the whole body is rung 5D's byte for byte.
  constexpr int BCg = BC / GROUPS;
  static_assert(BC % GROUPS == 0, "the groups must halve the block exactly");
  static_assert((C / kMmaM) % GROUPS == 0,
                "the intra term splits by token m-tiles across the groups, so "
                "the groups must divide them");
  static_assert(C % kMmaM == 0, "chunk must be a multiple of the mma m quantum");
  static_assert(BCg % kMmaK == 0, "a group's slice must be a whole number of mma k-tiles");
  //: THE STATE PLANE'S PAGE.  One atom is `kAtomLeaves` consecutive leaves by
  //: the leaf-order law, `cols` fp32 wide -- the value channels plus the mass
  //: column -- and it is staged through the output buffer, which is fp32 and
  //: padded and dead at both ends of the kernel.
  constexpr int kCols = kD + 1;
  constexpr int kAtoms = BCg / kAtomLeaves;
  static_assert(BCg % kAtomLeaves == 0,
                "a group's slice must be a whole number of page atoms, or an "
                "atom would straddle two CTAs");
  static_assert(kPadF >= 1 && C >= kAtomLeaves,
                "the atom's page is 16 rows of d_v+1 floats and it is staged in "
                "the output buffer, so that buffer needs 16 rows and one pad "
                "column");
  static_assert(C <= 255, "a causal threshold is one byte, chunk-relative");
  static_assert(design::smem_fits(G::bytes),
                "this arm does not fit this architecture's per-CTA shared-memory "
                "maximum");
  constexpr int STAGES = G::stages;
  static_assert(STAGES >= 2, "the wait/issue order needs at least two stages");
  //: The schedule stream runs deeper than the value stream, which is why the index
  //: ring is separate. The depth is DERIVED, not swept: at least as far ahead as
  //: the stage it feeds -- docs/internals/chunk/chunk_kernel.md#cook-distance
  constexpr int AHEAD = STAGES - 1 + kAheadExtra;
  //: THE RING HOLDS EVERY SUPER-CHUNK IN FLIGHT, which is what fences it: the slot
  //: the cook overwrites was drained an iteration earlier, across the rendezvous
  //: after the fold -- so that rendezvous is issued on EVERY super-chunk, not only
  //: on live ones.
  constexpr int IRING = AHEAD + 2;
  constexpr int LdV = G::LdV, LdS = G::LdS, LdG = G::LdG, LdO = G::LdO;
  constexpr int MTOK = C / kMmaM;   // token m-tiles in a chunk
  constexpr int NTOK = C / kMmaN;   // token n-tiles in a chunk
  constexpr int KD = BCg / kMmaK;   // slot k-tiles of THIS GROUP'S slice
  constexpr int NDV = BCg / kMmaN;  // state n-tiles (slots) of the slice
  constexpr uint32_t kRowV = (uint32_t)LdV * 2;
  constexpr uint32_t kRowS = (uint32_t)LdS * 2;
  constexpr int kChunkS = BCg / 8;  // 16-byte chunks in one R/W row
  constexpr int kChunkV = kD / 8;   // 16-byte chunks in one V row

  //: THE MASS CHANNEL'S CARVE.  One channel cannot be split along the channel
  //: axis, so it is split along the one the mma leaves free.  The readout's and
  //: the fold's free axis is the SLOTS; the intra term's is the TOKENS.
  constexpr int kMassUnits = KD < kUnits ? KD : kUnits;
  constexpr int kMassKD = KD / kMassUnits;  // slot k-tiles of the mass, per unit
  constexpr int kMassTiles = 2 * kMassKD;   // its state's n-tiles, per unit
  constexpr int kIntraUnits = MTOK < kUnits ? MTOK : kUnits;
  constexpr int kIntraKD = MTOK / kIntraUnits;
  static_assert(KD % kMassUnits == 0 && NDV == kMassUnits * kMassTiles,
                "the mass channel's slots must be carved across the units exactly");

  constexpr int MSPLIT = MTOK >= kGroupWarps ? kGroupWarps : MTOK;
  constexpr int NSPLIT_RAW = kGroupWarps / MSPLIT;
  constexpr int NSPLIT = NSPLIT_RAW < NTOK / 2 ? NSPLIT_RAW : NTOK / 2;
  constexpr int MPW = MTOK / MSPLIT;
  constexpr int NPW = NTOK / NSPLIT;
  constexpr int GWARPS = MSPLIT * NSPLIT;
  static_assert(NPW % 2 == 0, "n-tiles are consumed in ldmatrix pairs");
  constexpr int GPAIR = (NPW / 2) >= 2 ? 2 : 1;

  // ---- coefficient production geometry.  All compile-time. ----------------
  // A row's `kRowThr` threads split the block's BC coefficients.  There is one
  // rectangle to split and two ways to cut it, and which one applies is decided
  // by the rectangle's own shape.
  constexpr int kWtot = LP::col_base(LP::depth);
  //: THE PRODUCE'S TWO AXES, FROM THE PLAN (G3).  The INNERMOST level is the
  //: vectorized run; every level above it flattens into ONE composite OUTER
  //: axis, mixed-radix over the plan's spans with level D-2 fastest.  At
  //: D = 2 these are exactly the legacy lead/tail names.
  constexpr int kDepth = LP::depth;
  //: kDepth == 1 is the FLAT-ROUTING arm: no outer axis at
  //: all -- the produce's chain gets a literal 1.0 (rn(a*1) = a, bit-exact
  //: to a dedicated path), and the geometry law already yields outer = 1.
  //: Flat routing caps at N = B <= 256 BY THE MANIFEST (one entmax over B
  //: logits is the producer wall split-level exists to dodge), which makes
  //: D = 1 the benchmark's in-kernel flat BASELINE, not a capacity config.
  constexpr int kInnerSpan = LP::span_at(kDepth - 1);
  constexpr int kOuterSpan = BC / kInnerSpan;
  //: PRODUCTION IS A GROUP AFFAIR: a group's kGroupThreads produce ITS slice's
  //: coefficients -- `kLeadSpanG` outer digits by the whole innermost span --
  //: which at GROUPS = 1 is the block and exactly rung 5D's mapping.
  constexpr int kLeadSpanG = kOuterSpan / GROUPS;
  static_assert(kOuterSpan % GROUPS == 0, "the groups must split the outer axis exactly");
  constexpr int kRowThr = kGroupThreads / C;
  constexpr int kThrPerLead = kRowThr >= kLeadSpanG ? kRowThr / kLeadSpanG : 1;
  constexpr int kLeadPerThr = kLeadSpanG >= kRowThr ? kLeadSpanG / kRowThr : 1;
  constexpr int kTailPerThr = kInnerSpan / kThrPerLead;
  constexpr int kTailVec = (kTailPerThr * 2) / 16;
  static_assert(kGroupThreads % C == 0, "one production pass must cover the chunk exactly");
  static_assert(kRowThr >= kLeadSpanG ? kRowThr % kLeadSpanG == 0 : kLeadSpanG % kRowThr == 0,
                "the slice's outer digits and the row's threads must divide "
                "one another, or a thread owns a fractional digit");
  static_assert(kInnerSpan % kThrPerLead == 0,
                "the threads sharing an outer digit must divide the inner run");
  //: a thread's owned outer run must sit inside ONE row of the fastest outer
  //: level, or its amplitudes are not a contiguous vector load and the upper
  //: levels' digits stop being per-thread constants.
  constexpr int kFastSpan = kDepth >= 2 ? LP::span_at(kDepth - 2) : 1;
  static_assert(kLeadPerThr == 1 || kFastSpan % kLeadPerThr == 0,
                "a thread's outer run crosses a fast-level boundary");
  static_assert(kTailPerThr % 8 == 0,
                "a thread's trailing run is published in 16-byte vectors, so it "
                "must be a whole number of eight coefficients");
  static_assert(kLeadPerThr * kTailPerThr * kRowThr == BCg,
                "the row's threads must cover the slice exactly once");
  static_assert(kLeadPerThr <= 4, "leading-digit slice is 1, 2 or 4 amplitudes");

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  //: this thread's GROUP, its warp within the group, and its thread id within
  //: the group -- the coordinates every per-group buffer and role uses.  At
  //: GROUPS = 1 they collapse to (0, warp, tid).
  const int grp = warp / kGroupWarps;
  const int gwarp = warp % kGroupWarps;
  const int gtid = tid % kGroupThreads;

  extern __shared__ __align__(16) char smem_raw[];
  //: [ V ring (shared) | panels x GROUPS | summed Gram (shared) | fp32 Gram
  //:   partials x GROUPS (only when there is an exchange) | output staging x
  //:   GROUPS | denominators x GROUPS | index ring | formed schedule ] -- the
  //: order is Geom::fixed's, and the per-group regions are indexed by `grp`.
  extern __shared__ __align__(16) char smem_base[];

  //: THE (bh)-KEYED BASES, hoisted above the addressing: each is one widening
  //: multiply of `blockIdx.y`, and computed lower down ptxas rematerializes them
  //: INSIDE the super-chunk loop (docs/internals/chunk/chunk_kernel.md#state-addressing).
  const long base = (long)blockIdx.y * L * kD;
  const long pbase = (long)blockIdx.y * L * kWtot;
  const int tokWords = tok_words(L);
  //: V IS TOKEN-MAJOR: this (bh)'s rows start at its head's columns of token 0 and
  //: advance a whole token apart, which is the only place the two layouts differ.
  const int vRowElems = v_row_bytes / (int)sizeof(__nv_bfloat16);
  const int heads = vRowElems / kD;
  const __nv_bfloat16* const vPlane =
      pin_address(v + (long)(blockIdx.y / heads) * L * vRowElems + (long)(blockIdx.y % heads) * kD);
  float* const oPlane = pin_address(o + base);
  float* const denPlane = pin_address(den + (long)blockIdx.y * L);
  const __nv_bfloat16* const pRead = pin_address(pread + pbase);
  const __nv_bfloat16* const pWrite = pin_address(pwrite + pbase);
  //: THE WRITE GAIN, DECOUPLED (spec 4.2): a per-token positive scalar, its own
  //: plane -- the amplitudes stay pure simplexes, so the SUPPORT BITS ARE
  //: UNGAINED and a small gain can never underflow a live token out of the
  //: schedule.  The deposit is gained; a decay clock reads the UNGAINED
  //: coefficient, which is why the gain folds AFTER the level chain.
  const __nv_bfloat16* const gWrite = pin_address(gwrite + (long)blockIdx.y * L);
  //: ONE 64-BIT BASE PER (bh, side) and a 32-bit BYTE displacement per clause row
  //: after it -- the addressing discipline the table's own producer keeps.
  const uint32_t* const rowR =
      pin_address(tbl + ((long)(2 * blockIdx.y + 0) * LP::entries) * tokWords);
  const uint32_t* const rowW =
      pin_address(tbl + ((long)(2 * blockIdx.y + 1) * LP::entries) * tokWords);
  const uint32_t clauseStride = (uint32_t)tokWords * (uint32_t)sizeof(uint32_t);

  const int blk = (int)blockIdx.x;
  __nv_bfloat16* sv = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* srAll = sv + STAGES * C * LdV;
  __nv_bfloat16* sr = srAll + grp * (2 * C * LdS);
  __nv_bfloat16* sw = sr + C * LdS;
  __nv_bfloat16* sg = srAll + GROUPS * (2 * C * LdS);
  float* sgf = reinterpret_cast<float*>(sg + C * LdG);
  float* soAll = sgf + (GROUPS > 1 ? GROUPS * C * G::LdGf : 0);
  float* so = soAll + grp * (C * LdO + kUnits * C);
  float* sden = so + C * LdO;
  char* sidx = reinterpret_cast<char*>(soAll + GROUPS * (C * LdO + kUnits * C));
  //: THE FORMED SCHEDULE, one word per thirty-two tokens per side, and the walk's
  //: published state behind it.
  uint32_t* sfR = reinterpret_cast<uint32_t*>(sidx + IRING * G::idx_slot);
  uint32_t* sfW = sfR + kFormWords;
  int* scook = reinterpret_cast<int*>(sfW + kFormWords);
  static_assert((GROUPS * 2 * C * LdS + C * LdG) % 8 == 0,
                "the fp32 region must start 16-byte aligned");
  static_assert((C * LdO + kUnits * C) % 4 == 0, "the index ring must start 16-byte aligned");
  static_assert(G::idx_slot % 16 == 0,
                "the formed schedule follows the index ring and must stay aligned");

  //: THE INDEX RING'S ACCESSORS.  A slot is a header the whole CTA reads
  //: (`nr`, `nw`, whether the super-chunk exists at all) followed by the two
  //: lists and the thresholds.
  auto ihdr = [&](int d) { return reinterpret_cast<int*>(sidx + d * G::idx_slot); };
  auto iread = [&](int d) { return reinterpret_cast<uint16_t*>(sidx + d * G::idx_slot + G::hdr); };
  auto iwrite = [&](int d) { return iread(d) + C; };
  auto ithr = [&](int d) {
    return reinterpret_cast<uint8_t*>(sidx + d * G::idx_slot + G::hdr + 4 * C);
  };

  // The block's aligned digit ranges.  Compile-time divisors, formed once.
  //: (G3) the per-level digit offsets come from the plan's one map below, at
  //: the columns' construction -- the legacy lead_off/tail_off pair is gone.

  // ---- lane-invariant operand addressing -----------------------------------
  const int lr = lane & 7, tl = lane >> 3;
  const int rF = (tl & 1) * 8 + lr, cF = tl >> 1;
  const int rT = (tl >> 1) * 8 + lr, cT = tl & 1;
  const uint32_t rowF = SWZ ? (uint32_t)(rF * kRowS) : (uint32_t)((rF * LdS + cF * 8) * 2);
  const uint32_t rowT = SWZ ? (uint32_t)(rT * kRowS) : (uint32_t)((rT * LdS + cT * 8) * 2);
  const uint32_t rowTV = SWZ ? (uint32_t)(rT * kRowV) : (uint32_t)((rT * LdV + cT * 8) * 2);
  uint32_t tabF[kChunkS], tabT[kChunkS];
#pragma unroll
  for (int g = 0; g < kChunkS; ++g) {
    tabF[g] = (uint32_t)((((g + cF) & (kChunkS - 1)) ^ (lr & (kChunkS - 1))) << 4);
    tabT[g] = (uint32_t)((((g + cT) & (kChunkS - 1)) ^ (lr & (kChunkS - 1))) << 4);
  }
#define ADDRF(b_, R_, g_) \
  (SWZ ? ((b_) + (uint32_t)((R_) * (int)kRowS) + tabF[(g_)]) \
       : ((b_) + (uint32_t)(((R_) * LdS + (g_) * 8) * 2)))
#define ADDRT(b_, R_, g_) \
  (SWZ ? ((b_) + (uint32_t)((R_) * (int)kRowS) + tabT[(g_)]) \
       : ((b_) + (uint32_t)(((R_) * LdS + (g_) * 8) * 2)))

  //: THE MASS CHANNEL'S OWN OPERAND ADDRESSES.  This unit's tile `q` is the
  //: value path's tile `q * kMassUnits + warp`, so its address is the value
  //: address at a compile-time `q` plus a LOOP-INVARIANT per-unit displacement.
  const uint32_t massOff = (uint32_t)(gwarp * kMmaK * (int)sizeof(__nv_bfloat16));
  uint32_t tabMT[kMassKD], tabMF[kMassKD];
#pragma unroll
  for (int q = 0; q < kMassKD; ++q) {
    const int g = 2 * (q * kMassUnits + gwarp);
    tabMT[q] = (uint32_t)((((g + cT) & (kChunkS - 1)) ^ (lr & (kChunkS - 1))) << 4);
    tabMF[q] = (uint32_t)((((g + cF) & (kChunkS - 1)) ^ (lr & (kChunkS - 1))) << 4);
  }
#define ADDRT_M(b_, R_, q_) \
  (SWZ ? ((b_) + (uint32_t)((R_) * (int)kRowS) + tabMT[(q_)]) \
       : ((b_) + (uint32_t)(((R_) * LdS + (2 * (q_) * kMassUnits) * 8) * 2) + massOff))
#define ADDRF_M(b_, R_, q_) \
  (SWZ ? ((b_) + (uint32_t)((R_) * (int)kRowS) + tabMF[(q_)]) \
       : ((b_) + (uint32_t)(((R_) * LdS + (2 * (q_) * kMassUnits) * 8) * 2) + massOff))

  const int offBg = (rT * LdG + cT * 8) * 2;

  const int gm0 = (gwarp / NSPLIT) * MPW;
  const int gn0 = (gwarp % NSPLIT) * NPW;
  const int dv0 = gwarp * 16;
  constexpr int DVG_STEP = 2;
  const int dvg = gwarp * DVG_STEP;
  const uint32_t dvTabT = (uint32_t)((((dvg + cT) & (kChunkV - 1)) ^ lr) << 4);
#define ADDRT_DV(b_, R_) \
  (SWZ ? ((b_) + (uint32_t)((R_) * (int)kRowV) + dvTabT) \
       : ((b_) + (uint32_t)(((R_) * LdV + dv0) * 2)))

  // The state, transposed: m = d_v (this warp's 16 rows), n = this block's slots.
  float st[NDV][4];
#pragma unroll
  for (int j = 0; j < NDV; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) st[j][e] = 0.f;

  //: THE d_v+1-th CHANNEL'S STATE, in the same accumulator layout and on the
  //: same recurrence: m = the mass tile (row 0 is the channel, rows 1..15 are
  //: the m quantum's padding and are identically zero), n = this unit's share of
  //: the block's slots.
  float mass_st[kMassTiles][4];
#pragma unroll
  for (int j = 0; j < kMassTiles; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) mass_st[j][e] = 0.f;

  //: THE ONES OPERAND, WHICH IS AN IMMEDIATE.  The value of the mass channel is
  //: literally 1, so the A fragment of the fold and of the intra is the constant
  //: whose row 0 is all ones and whose padding rows are zero.
  constexpr uint32_t kOnesPair = 0x3F803F80u;  // two bf16 1.0
  uint32_t ones_a[4];
  ones_a[0] = (lane >> 2) == 0 ? kOnesPair : 0u;
  ones_a[1] = 0u;
  ones_a[2] = ones_a[0];
  ones_a[3] = 0u;

  // ---- global -> shared staging of V, GATHERED BY THE WRITE LIST -----------
  //: V IS A WRITE-SIDE QUANTITY.  Both terms that consume it -- the fold and the
  //: intra -- contract over the WRITE token axis, so one gather serves both and
  //: the read list never touches V.
  constexpr int kRowsPerPass = kThreads / kChunkV;
  constexpr int kPasses = C / kRowsPerPass;
  static_assert(kPasses >= 1, "a chunk must be at least kThreads/8 rows");
  const int ldRow = tid / kChunkV, ldCol = (tid % kChunkV) * 8;
  const uint32_t ldColB = (uint32_t)(ldCol * 2);
  const uint32_t kVRowB = (uint32_t)v_row_bytes;

  const uint32_t stgOff =
      SWZ ? (uint32_t)(ldRow * (int)kRowV + (((tid % kChunkV) ^ (ldRow & 7)) << 4))
          : (uint32_t)((ldRow * LdV + ldCol) * 2);

  constexpr uint32_t kSlot = (uint32_t)(C * LdV) * 2;
  constexpr uint32_t kRing = STAGES * kSlot;
  const uint32_t svL = smem_addr(sv) + stgOff;
  const uint32_t svB = smem_addr(sv);
  const uint32_t sgT = smem_addr(sg) + (uint32_t)offBg;
  uint32_t stgSlot = 0, cmpSlot = 0;

  const uint32_t arF = smem_addr(sr) + rowF;
  const uint32_t brT = smem_addr(sr) + rowT;
  const uint32_t bwT = smem_addr(sw) + rowT;
  const uint32_t bwF = smem_addr(sw) + rowF;

  // ---- the fan-in's own addressing -----------------------------------------
  // CONSECUTIVE CHANNELS SIT IN CONSECUTIVE LANES OF ONE REDUCTION.  That is a
  // property of the REDUCTION and not of the thread: this architecture's
  // reduction carries ONE channel, so what decides how much output one of them
  // covers is the span its 32 lanes reach.  Giving a THREAD the consecutive run
  // instead leaves a reduction's lanes strided by that run, and the coalescer
  // then touches four times the sectors for the same bytes.
  constexpr int kFanLanes = 32;
  constexpr int kFanPerThr = kD / kFanLanes;
  constexpr int kFanRows = kGroupThreads / kFanLanes;
  constexpr int kFanPasses = C / kFanRows;
  static_assert(kD % kFanLanes == 0 && kGroupThreads % kFanLanes == 0,
                "a reduction's lanes must tile both an output row and the CTA");
  static_assert(kFanPasses * kFanRows == C && kFanPasses >= 1,
                "the fan-in's passes must cover the chunk exactly");
  const int fanRow = gtid / kFanLanes, fanCol = gtid % kFanLanes;
  const uint32_t fanColB = (uint32_t)(fanCol * (int)sizeof(float));
  constexpr uint32_t kORowB = (uint32_t)(kD * (int)sizeof(float));
  const int fanE = fanRow * LdO + fanCol;

  // Output staging write index: this lane owns d_v = dv0 + (lane>>2) and that
  // + 8, at tokens 8j + 2*(lane&3) + {0,1}.
  const int oq = (lane & 3) * 2, om = lane >> 2;
  int oE[2][2];
#pragma unroll
  for (int e = 0; e < 2; ++e)
#pragma unroll
    for (int f = 0; f < 2; ++f) oE[e][f] = (oq + e) * LdO + dv0 + om + f * 8;

  //: ---- the state plane's addressing and its fragment ownership map --------
  //: WHICH THREAD OWNS WHICH ELEMENT of an atom's page, as an index into the
  //: output staging buffer.  Element `k` of an accumulator tile sits at row
  //: `(lane>>2) + 8*(k>>1)` and column `(lane&3)*2 + (k&1)`; for the value
  //: state the rows are channels and the columns are leaves, and for the mass
  //: state only row 0 is the channel, so only lanes 0..3 and `k in {0,1}` carry
  //: it -- the same restriction the denominator store already makes.
  auto page_value = [&](int p, int k) {
    return (p * 8 + (lane & 3) * 2 + (k & 1)) * LdO + dv0 + (lane >> 2) + 8 * (k >> 1);
  };
  auto page_mass = [&](int p, int k) { return (p * 8 + lane * 2 + k) * LdO + kD; };
  //: One pass over an atom's page; `body` takes the staging index and the global
  //: byte displacement. THE SWEEP CARRIES A CURSOR RATHER THAN DIVIDING, and the
  //: accesses are plain floats -- docs/internals/chunk/chunk_kernel.md#page-sweep
  constexpr int kPageElems = kAtomLeaves * kCols;
  constexpr int kPageCarry = kGroupThreads - kCols;
  static_assert(kGroupThreads > kCols && kGroupThreads < 2 * kCols,
                "the page sweep's step must cross one row boundary, so its row "
                "cursor advances by one or two and never by a division");
  auto page_sweep = [&](uint32_t pageOff, auto&& body) {
    int e = gtid;
    int row = gtid < kCols ? 0 : 1;
    int col = gtid < kCols ? gtid : gtid - kCols;
#pragma unroll 1
    while (e < kPageElems) {
      body(row * LdO + col, pageOff + (uint32_t)(e * (int)sizeof(float)));
      e += kGroupThreads;
      col += kPageCarry;
      const bool wrap = col >= kCols;
      row += wrap ? 2 : 1;
      col -= wrap ? kCols : 0;
    }
  };
  //: This group's first logical atom, `AtomKeying.atom_of` verbatim. DERIVED
  //: INSIDE each state block so no state I/O is live across the main loop, and
  //: ALL-INT because the 32-bit displacement already caps the page count
  //: -- docs/internals/chunk/chunk_kernel.md#atom-index
  constexpr int kAtomsPerBh = (int)(LP::slots / kAtomLeaves);
  auto atom_first = [&] {
    return (int)blockIdx.y * kAtomsPerBh + LP::block_base(blk) / kAtomLeaves + grp * kAtoms;
  };
  //: THE ONE BASE TRANSLATION: a null table is the dense plane, a table is the
  //: paged one, `-1` is an atom with no page. EVERYTHING PER ATOM MUST STAY A
  //: 32-BIT BYTE DISPLACEMENT from a PLAIN pinned origin -- a 64-bit pointer sum
  //: here reaches the main loop's span as a widening multiply
  //: -- docs/internals/chunk/chunk_kernel.md#base-translation, docs/internals/chunk/chunk_kernel.md#state-addressing
  auto slot_of = [](const int32_t* tbl, int atom) {
    return tbl != nullptr ? *reinterpret_cast<const int32_t*>(
                                gather_ptr(tbl, (uint32_t)atom * (uint32_t)sizeof(int32_t)))
                          : atom;
  };
  auto page_off = [](int slot) {
    return (uint32_t)slot * (uint32_t)(kPageElems * (int)sizeof(float));
  };

  //: THE STAGE IS WHAT IS SKIPPED; THE RING SLOT IS NOT.  A super-chunk with no
  //: write entry stages nothing, but the slot advances either way so that the
  //: wait/commit accounting stays rung 2's.
  auto stage_chunk = [&](int d, bool go) {
    if (go) {
      const uint16_t* lw = iwrite(d);
      //: the window, folded (6E): one uniform multiply-add per staged chunk.
      const uint32_t winCol = ldColB + (uint32_t)ihdr(d)[3] * (kWindow * kVRowB);
#pragma unroll
      for (int p = 0; p < kPasses; ++p) {
        const uint32_t off = (uint32_t)lw[ldRow + p * kRowsPerPass] * kVRowB + winCol;
        stage_tile(svL + stgSlot + (uint32_t)(p * kRowsPerPass * (int)kRowV),
                   gather_ptr(vPlane, off));
      }
    }
    stgSlot += kSlot;
    if (stgSlot == kRing) stgSlot = 0;
  };

  //: THE FAN-IN'S TARGET ROW IS GATHERED.  The partial staged for read entry `i`
  //: belongs to token `r[i]`; entries past `nr` are the buffer's padding and have
  //: no token, so they contribute nothing rather than contributing zero to
  //: somewhere.
  auto drain = [&](int d, int nr) {
    const uint16_t* lrd = iread(d);
    const uint32_t winFan =
        fanColB + (uint32_t)ihdr(d)[3] * (uint32_t)(kWindow * kD * (int)sizeof(float));
#pragma unroll
    for (int p = 0; p < kFanPasses; ++p) {
      const int row = fanRow + p * kFanRows;
      if (row < nr) {
        float* dp =
            reinterpret_cast<float*>(gather_ptr(oPlane, (uint32_t)lrd[row] * kORowB + winFan));
        const int e0 = fanE + p * kFanRows * LdO;
#pragma unroll
        for (int e = 0; e < kFanPerThr; ++e) fan_in_add(dp + e * kFanLanes, so[e0 + e * kFanLanes]);
      }
    }
  };

  //: THE UNITS' PARTIAL DENOMINATORS MEET HERE, at no cost in barriers: the
  //: rendezvous that already separates a chunk's staging from the next chunk's
  //: drain is the one that publishes them.
  auto drain_den = [&](int d, int nr) {
    if (gtid < C && gtid < nr) {
      float s = 0.f;
#pragma unroll
      for (int u = 0; u < kUnits; ++u) s += sden[u * C + gtid];
      fan_in_add(
          reinterpret_cast<float*>(gather_ptr(
              denPlane, (uint32_t)iread(d)[gtid] * 4u
                            + (uint32_t)ihdr(d)[3] * (uint32_t)(kWindow * (int)sizeof(float)))),
          s);
    }
  };

  // ---- routed Kronecker coefficient production, digit-major ---------------
  // Thread `tid` owns panel row `tid / kRowThr` and, within the block's
  // rectangle, the leading digits [leadLocal, leadLocal + kLeadPerThr) and the
  // trailing run [tailLocal, tailLocal + kTailPerThr).  THE DIGITS OF A SLOT ARE
  // NEVER COMPUTED: they are the loop indices.  What compaction changes is only
  // WHICH TOKEN a panel row reads -- the row's own slot addressing is untouched,
  // which is why the gather costs a displacement and not a recomputation.
  const int prow = gtid / kRowThr;
  const int rIn = gtid % kRowThr;
  const int leadLocal = (rIn / kThrPerLead) * kLeadPerThr;
  const int tailLocal = (rIn % kThrPerLead) * kTailPerThr;
  //: ONE BASE PER SIDE, AND THE TWO LEVELS ARE 32-BIT COLUMN DISPLACEMENTS.  A
  //: base is a live 64-bit value for the whole kernel; the level a thread reads
  //: is a byte offset inside its token's row and therefore adds before the
  //: widening, not after it.
  constexpr uint32_t kPRowB = (uint32_t)(kWtot * (int)sizeof(__nv_bfloat16));
  //: 1.0 as a bf16x2 splat: the read side's gain.  Folding it through the
  //: same multiply keeps ONE produce body, and rn(a*1) = a keeps the read
  //: panel BIT-EXACT -- the multiply is paid, the value is untouched.
  constexpr uint32_t kOneSplat = 0x3F803F80u;
  //: (G3) THE COLUMNS, ONE PER CONTRIBUTING LEVEL, all built from the plan's
  //: col_base/off_at map.  A thread's composite outer index `oBase` splits into
  //: a FAST digit (level D-2, varies across its owned run) and a fixed upper
  //: row; the upper levels' digits are per-thread CONSTANTS, so each upper
  //: level is one named column scalar -- at D = 2 there are none and the pair
  //: below is exactly the legacy colLead/colTail.
  const int oBase = grp * kLeadSpanG + leadLocal;
  [[maybe_unused]] uint32_t colFast = 0;
  if constexpr (kDepth >= 2)
    colFast =
        (uint32_t)((LP::col_base(kDepth - 2) + LP::off_at(kDepth - 2, blk) + oBase % kFastSpan)
                   * 2);
  const uint32_t colInner =
      (uint32_t)((LP::col_base(kDepth - 1) + LP::off_at(kDepth - 1, blk) + tailLocal) * 2);
  constexpr int kUppers = kDepth - 2;
  [[maybe_unused]] uint32_t colUpA = 0, colUpB = 0;  // named scalars, never an array
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

  const int slot0 = leadLocal * kInnerSpan + tailLocal;  // local slot of this thread
  char* dstR =
      reinterpret_cast<char*>(sr) + (SWZ ? prow * (int)kRowS : prow * (int)kRowS + slot0 * 2);
  char* dstW =
      reinterpret_cast<char*>(sw) + (SWZ ? prow * (int)kRowS : prow * (int)kRowS + slot0 * 2);
  const int pPh = prow & (kChunkS - 1);
  int pOff[kLeadPerThr][kTailVec];
#pragma unroll
  for (int j = 0; j < kLeadPerThr; ++j)
#pragma unroll
    for (int i = 0; i < kTailVec; ++i)
      pOff[j][i] = SWZ ? ((((slot0 + j * kInnerSpan) / 8 + i) & (kChunkS - 1)) ^ pPh) << 4
                       : ((j * kInnerSpan + i * 8) * 2);

  //: THE PRODUCTION IS A COMPONENT -- this file's `Producer`, which Scan B
  //: drives too.  The column offsets and slot displacements above are this
  //: carve's; the production itself is everyone's.
  Producer<kDepth, kLeadPerThr, kTailVec, kUppers> prod{colFast, colInner, colUpA, colUpB, {}};
#pragma unroll
  for (int j = 0; j < kLeadPerThr; ++j)
#pragma unroll
    for (int i = 0; i < kTailVec; ++i) prod.pOff[j][i] = pOff[j][i];
  auto produce = [&](const __nv_bfloat16* bs, char* dst, uint32_t tokOff, bool have, uint32_t gsp) {
    prod.emit(bs, dst, tokOff, have, gsp);
  };

  // ---- the membership bits, read out of the union table --------------------
  //: The block's entry rows, per side; the trailing row disappears when the block
  //: spans that level entirely -- docs/internals/chunk/chunk_kernel.md#clause-rows
  constexpr int kClauses = LP::clauses;
  //: NAMED SCALARS, never an array: the clause loop's callees carry inline asm,
  //: so a runtime-indexed array is demoted to a stack frame
  //: -- docs/internals/chunk/chunk_kernel.md#clause-rows
  uint32_t cA = 0, cB = 0, cC = 0, cD = 0;
#define ROLA_CLAUSE(CL_, C_)                                                   \
  if constexpr ((CL_) < kClauses) {                                            \
    constexpr int kDiv = LP::suffix_groups((CL_) + 1);                         \
    constexpr int kGrp = LP::groups_at(CL_);                                   \
    constexpr int kPre = LP::prefix_entries(CL_);                              \
    C_ = (uint32_t)(kPre + (blk / kDiv) % kGrp) * clauseStride;                \
  }
  ROLA_CLAUSE(0, cA)
  ROLA_CLAUSE(1, cB)
  ROLA_CLAUSE(2, cC)
  ROLA_CLAUSE(3, cD)
#undef ROLA_CLAUSE

  //: THE LAST WORD'S PADDING TOKENS DO NOT EXIST.  The table leaves them zero;
  //: the clause-free block has no table to leave them zero, so the mask is
  //: applied in the FORM, where it is one select on one word of a parallel map.
  const uint32_t tailMask =
      (L & (kTokBits - 1)) ? ((1u << (L & (kTokBits - 1))) - 1u) : 0xFFFFFFFFu;
  const int lastWord = tokWords - 1;

  //: THE FORM: the membership AND and nothing else. The ring's head is the walk's
  //: cursor plus its depth, so backpressure needs no test outside the loop bound
  //: -- docs/internals/chunk/chunk_kernel.md#the-form
  //: THE SCHEDULE IS A COMPONENT.  The closing rule has ONE implementation and
  //: Scan A drives the same object -- this file's `Ascent`.
  Ascent<C, kClauses, kThreads, G::idx_slot, G::hdr> aw{
      sidx, sfR, sfW, scook, rowR, rowW, cA, cB, cC, cD, tailMask, lastWord, tokWords};
  aw.begin_form();
  auto form = [&]() { aw.form(tid); };
  auto form_if_low = [&]() { aw.form_if_low(tid); };
  aw.begin_walk();
  auto walk = [&](int d) { aw.walk(d, lane); };
  auto walk_wait = [&](int d) { aw.walk_wait(d, tid, warp, lane); };

  // ---- the pipeline's prologue: cook and stage `STAGES - 1` super-chunks ----
  //: THE PROLOGUE FILLS THE SCHEDULE STREAM FIRST AND THE VALUE STREAM SECOND,
  //: which is the same ordering the steady state keeps: the walk runs `AHEAD`
  //: super-chunks out before the first value tile is issued.  It pays the two
  //: barriers that fill the formed ring ONCE, before the first walk; the steady
  //: state pays none.

  //: ---- the ENTRY STATE, the store read backwards through the same staging
  //: buffer. The fragments are SET, never accumulated: they are provably zero
  //: here. An absent atom is skipped and its fragments stay zero; what the skip
  //: omits is MEMORY TRAFFIC only, so every barrier stays CTA-uniform
  //: -- docs/internals/chunk/chunk_kernel.md#entry-state, docs/internals/chunk/chunk_kernel.md#base-translation
  if constexpr (STATE) {
    if (s_in != nullptr) {
      const int atomFirst = atom_first();
      const int32_t* const tbl = page_tbl != nullptr ? pin_address(page_tbl) : nullptr;
      const float* const plane = pin_address(s_in);
#pragma unroll
      for (int i = 0; i < kAtoms; ++i) {
        const int slot = slot_of(tbl, atomFirst + i);
        if (slot >= 0) {
          page_sweep(page_off(slot), [&](int s, uint32_t off) {
            so[s] = *reinterpret_cast<const float*>(gather_ptr(plane, off));
          });
        }
        rendezvous();
        if (slot >= 0) {
#pragma unroll
          for (int p = 0; p < 2; ++p)
#pragma unroll
            for (int k = 0; k < 4; ++k) st[2 * i + p][k] = so[page_value(p, k)];
          if (gwarp == i % kMassUnits && lane < 4) {
#pragma unroll
            for (int p = 0; p < 2; ++p)
#pragma unroll
              for (int k = 0; k < 2; ++k)
                mass_st[2 * (i / kMassUnits) + p][k] = so[page_mass(p, k)];
          }
        }
        rendezvous();
      }
    }
  }

  if (tid == 0) {
    scook[kCookWord] = 0;
    scook[kCookOpen] = 0;
  }
  rendezvous();
  form();
  rendezvous();
#pragma unroll 1
  for (int s = 0; s < AHEAD; ++s) {
    if (warp == 0) walk(s);
    rendezvous();
    walk_wait(s);
    form_if_low();
    rendezvous();
  }
#pragma unroll 1
  for (int s = 0; s < STAGES - 1; ++s) {
    stage_chunk(s, ihdr(s)[1] != 0);
    stage_commit();
  }

  //: THE RING CURSORS ARE COUNTERS, NOT REMAINDERS.  `IRING` is not a power of
  //: two, so a modulo would put a compile-time reciprocal in the body of the one
  //: loop the whole kernel runs in.  Three of them: what is computing, what is
  //: being staged, and what is being cooked.
  int dc = 0, ds = STAGES - 1, dn = AHEAD;
  int prev_d = 0, prev_nr = 0;
#pragma unroll 1
  for (;;) {
    const int* h = ihdr(dc);
    const int nr = h[0], nw = h[1];
    if (!h[2]) break;
    //: Per-side liveness gates per-side work. The intra term needs both sides
    //: PRESENT, not merely nonzero: a super-chunk with no write entry stages no
    //: value tile -- docs/internals/chunk/chunk_kernel.md#side-liveness
    const bool liveR = nr != 0, liveW = nw != 0;
    const bool live = liveR || liveW;

    //: THE COOK RUNS BEFORE THE PRODUCTION AND PUBLISHES AT THE SAME RENDEZVOUS
    //: THE PANELS DO.  It is one unit's work against three units' production, so
    //: what it costs the CTA is the difference and not the whole of it.
    if (warp == 0) walk(dn);

    if (live) {
      const uint32_t winP = (uint32_t)ihdr(dc)[3] * (uint32_t)(kWindow * kWtot * 2);
      const uint32_t rOff = (uint32_t)iread(dc)[prow] * kPRowB + winP;
      const uint32_t wOff = (uint32_t)iwrite(dc)[prow] * kPRowB + winP;
      //: the write token's gain: one 2-byte gather per row (padding rows read
      //: token zero's gain and are zeroed at the leading select regardless).
      const uint32_t winG = (uint32_t)ihdr(dc)[3] * (uint32_t)(kWindow * 2);
      const uint32_t gsp = splat_bf16x2(*reinterpret_cast<const __nv_bfloat16*>(
          gather_ptr(gWrite, (uint32_t)iwrite(dc)[prow] * 2u + winG)));
      produce(pRead, dstR, rOff, prow < nr, kOneSplat);
      produce(pWrite, dstW, wOff, prow < nw, gsp);
    }

    stage_wait<STAGES - 2>();
    rendezvous();  // (1) V visible / R,W visible / last chunk's reads done / O drained
                   //     and the walk's cursor and open flag are published

    walk_wait(dn);
    //: THE FORM RIDES THE BARRIERS THAT ARE ALREADY HERE.  It is issued where the
    //: value tile is, after the rendezvous that published the walk's cursor and
    //: before (2), which is what makes the formed words visible to the next
    //: super-chunk's walk.
    form_if_low();
    stage_chunk(ds, ihdr(ds)[1] != 0);
    stage_commit();

    const uint32_t atv = svB + cmpSlot + rowTV;
    cmpSlot += kSlot;
    if (cmpSlot == kRing) cmpSlot = 0;

    float acc[NTOK][4];
    float macc[NTOK][4];
    if (liveR) {
      // ---- 1. readout: O^T = S^T . R^T, A operand straight out of the state ---
      //      and, on the same B fragments, the d_v+1-th channel's row of it.
      {
        uint32_t sa[KD][4];
#pragma unroll
        for (int kk = 0; kk < KD; ++kk) {
          sa[kk][0] = pack_bf16x2(st[2 * kk][0], st[2 * kk][1]);
          sa[kk][1] = pack_bf16x2(st[2 * kk][2], st[2 * kk][3]);
          sa[kk][2] = pack_bf16x2(st[2 * kk + 1][0], st[2 * kk + 1][1]);
          sa[kk][3] = pack_bf16x2(st[2 * kk + 1][2], st[2 * kk + 1][3]);
        }
        uint32_t ma[kMassKD][4];
#pragma unroll
        for (int q = 0; q < kMassKD; ++q) {
          ma[q][0] = pack_bf16x2(mass_st[2 * q][0], mass_st[2 * q][1]);
          ma[q][1] = pack_bf16x2(mass_st[2 * q][2], mass_st[2 * q][3]);
          ma[q][2] = pack_bf16x2(mass_st[2 * q + 1][0], mass_st[2 * q + 1][1]);
          ma[q][3] = pack_bf16x2(mass_st[2 * q + 1][2], mass_st[2 * q + 1][3]);
        }
#pragma unroll
        for (int j = 0; j < NTOK; ++j)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            acc[j][e] = 0.f;
            macc[j][e] = 0.f;
          }
#pragma unroll
        for (int m = 0; m < MTOK; ++m) {
          uint32_t rb[KD][4];
#pragma unroll
          for (int kk = 0; kk < KD; ++kk) load_frag(rb[kk], ADDRT(brT, m * 16, 2 * kk));
#pragma unroll
          for (int kk = 0; kk < KD; ++kk)
#pragma unroll
            for (int p = 0; p < 2; ++p) mma(acc[2 * m + p], sa[kk], &rb[kk][2 * p]);
          if (gwarp < kMassUnits) {
            uint32_t mb[kMassKD][4];
#pragma unroll
            for (int q = 0; q < kMassKD; ++q) load_frag(mb[q], ADDRT_M(brT, m * 16, q));
#pragma unroll
            for (int q = 0; q < kMassKD; ++q)
#pragma unroll
              for (int p = 0; p < 2; ++p) mma(macc[2 * m + p], ma[q], &mb[q][2 * p]);
          }
        }
      }

      // ---- 2. Gram: G = staircase(R.W^T) over THIS BLOCK's slots ---------------
      //: THE MASK IS THE MONOTONE STAIRCASE, AND IT IS THE SAME SHAPE AS THE
      //: TRIANGLE IT GENERALISES.  Both lists are ascending, so read entry `tr`
      //: keeps exactly the write entries below its threshold `J[tr]` -- one byte
      //: the cook already observed.  At dense `J[tr] = tr + 1` and this reduces to
      //: `tc <= tr` term for term.
      if (gwarp < GWARPS) {
        int jv[MPW][2];
        if constexpr (GROUPS == 1) {
          const uint8_t* ljt = ithr(dc);
#pragma unroll
          for (int m = 0; m < MPW; ++m) {
            const int tr = (gm0 + m) * 16 + (lane >> 2);
            jv[m][0] = ljt[tr];
            jv[m][1] = ljt[tr + 8];
          }
        }
        uint32_t qa[MPW][KD][4];
#pragma unroll
        for (int m = 0; m < MPW; ++m)
#pragma unroll
          for (int kk = 0; kk < KD; ++kk) load_frag(qa[m][kk], ADDRF(arF, (gm0 + m) * 16, 2 * kk));

#pragma unroll
        for (int np = 0; np < NPW / 2; np += GPAIR) {
          uint32_t bf[GPAIR][KD][4];
#pragma unroll
          for (int u = 0; u < GPAIR; ++u)
#pragma unroll
            for (int kk = 0; kk < KD; ++kk)
              load_frag(bf[u][kk], ADDRT(bwT, (gn0 + (np + u) * 2) * 8, 2 * kk));
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
            for (int kk = 0; kk < KD; ++kk)
#pragma unroll
              for (int u = 0; u < GPAIR; ++u) {
                mma(g[u][0], qa[m][kk], &bf[u][kk][0]);
                mma(g[u][1], qa[m][kk], &bf[u][kk][2]);
              }
            const int tr = (gm0 + m) * 16 + (lane >> 2);
#pragma unroll
            for (int u = 0; u < GPAIR; ++u)
#pragma unroll
              for (int h2 = 0; h2 < 2; ++h2) {
                const int tc = (gn0 + (np + u) * 2 + h2) * 8 + (lane & 3) * 2;
                if constexpr (GROUPS == 1) {
                  //: one group: the block's whole Gram -- mask and publish, as
                  //: rung 5D did.
                  const int j0 = jv[m][0], j1 = jv[m][1];
                  const float m0 = (tc + 0 < j0) ? g[u][h2][0] : 0.f;
                  const float m1 = (tc + 1 < j0) ? g[u][h2][1] : 0.f;
                  const float m2 = (tc + 0 < j1) ? g[u][h2][2] : 0.f;
                  const float m3 = (tc + 1 < j1) ? g[u][h2][3] : 0.f;
                  store_bf16x2(&sg[tr * LdG + tc], pack_bf16x2(m0, m1));
                  store_bf16x2(&sg[(tr + 8) * LdG + tc], pack_bf16x2(m2, m3));
                } else {
                  //: two groups: this is HALF the Gram (this group's slots), so
                  //: it is published RAW in fp32; the exchange after (2) sums
                  //: the halves and applies the staircase ONCE.  The sum happens
                  //: in fp32 exactly because bf16 partials would round twice.
                  float* gp = sgf + grp * C * G::LdGf + tr * G::LdGf + tc;
                  store_f32(gp + 0, g[u][h2][0]);
                  store_f32(gp + 1, g[u][h2][1]);
                  store_f32(gp + G::LdGf * 8 + 0, g[u][h2][2]);
                  store_f32(gp + G::LdGf * 8 + 1, g[u][h2][3]);
                }
              }
          }
        }
      }
    }

    // ---- 3. fan the PREVIOUS super-chunk's partial output into the accumulator
    if (prev_nr) {
      drain(prev_d, prev_nr);
      drain_den(prev_d, prev_nr);
    }

    if (liveW) {
      // ---- 4. fold: S^T += V^T . W -------------------------------------------
#pragma unroll
      for (int kk = 0; kk < MTOK; ++kk) {
        uint32_t a[4];
        load_frag_t(a, ADDRT_DV(atv, kk * 16));
        uint32_t b[KD][4];
#pragma unroll
        for (int nj = 0; nj < KD; ++nj) load_frag_t(b[nj], ADDRF(bwF, kk * 16, 2 * nj));
#pragma unroll
        for (int nj = 0; nj < KD; ++nj) {
          mma(st[2 * nj], a, &b[nj][0]);
          mma(st[2 * nj + 1], a, &b[nj][2]);
        }
        //: The d_v+1-th channel deposits through the same W panel at this unit's
        //: own slots, with the literal 1 for its A operand.
        if (gwarp < kMassUnits) {
#pragma unroll
          for (int q = 0; q < kMassKD; ++q) {
            uint32_t mb[4];
            load_frag_t(mb, ADDRF_M(bwF, kk * 16, q));
            mma(mass_st[2 * q], ones_a, &mb[0]);
            mma(mass_st[2 * q + 1], ones_a, &mb[2]);
          }
        }
      }
    }

    //: (2) THE GRAM IS VISIBLE; R AND W ARE DEAD UNTIL THE NEXT PRODUCTION; AND
    //: THE PREVIOUS SUPER-CHUNK'S INDEX SLOT IS FREE FOR THE COOK TO REUSE.  It
    //: is issued on every super-chunk, live or dead, because that last clause is
    //: what separates the drain from a cook one iteration later.
    rendezvous();

    //: THE GRAM EXCHANGE -- the slot carve's ONE genuine cross-group sum, and
    //: also its prize: the summed Gram is multiplied into V ONCE per block
    //: instead of once per group.  fp32 partials in, one staircase, bf16 out;
    //: the extra rendezvous publishes the sum to every group's intra.  The
    //: conditions are CTA-uniform (they come off the shared header), so the
    //: barrier is uniform too.
    if constexpr (GROUPS > 1) {
      if (liveR && liveW) {
        const uint8_t* ljt = ithr(dc);
#pragma unroll 1
        for (int e = tid; e < C * (C / 2); e += kThreads) {
          const int tr = e / (C / 2);
          const int tc = (e - tr * (C / 2)) * 2;
          const int j = ljt[tr];
          float s0 = 0.f, s1 = 0.f;
#pragma unroll
          for (int g2 = 0; g2 < GROUPS; ++g2) {
            const float* gp = sgf + g2 * C * G::LdGf + tr * G::LdGf + tc;
            s0 += gp[0];
            s1 += gp[1];
          }
          store_bf16x2(&sg[tr * LdG + tc], pack_bf16x2(tc < j ? s0 : 0.f, tc + 1 < j ? s1 : 0.f));
        }
        rendezvous();
      }
    }

    if (liveR) {
      if (liveW) {
        // ---- 5. intra: O^T += V^T . G^T ----------------------------------------
        uint32_t va[MTOK][4];
#pragma unroll
        for (int kk = 0; kk < MTOK; ++kk) load_frag_t(va[kk], ADDRT_DV(atv, kk * 16));
        //: the contraction runs over EVERY write token (the k axis stays whole);
        //: what splits across the groups is the OUTPUT token axis -- each group
        //: computes its m-tiles into its own acc and the fan-in sums the groups
        //: exactly as it sums the blocks.  THE TILE BASE IS COMPILE-TIME: `acc`
        //: is a register array, and a runtime index into one is a spill, so the
        //: group takes its half through a uniform branch on its own id rather
        //: than through arithmetic the allocator cannot see through.
        constexpr int MI = MTOK / GROUPS;
        auto intra_half = [&](auto base_c) {
          constexpr int M0 = decltype(base_c)::value;
#pragma unroll
          for (int mi = 0; mi < MI; ++mi) {
            uint32_t gb[MTOK][4];
#pragma unroll
            for (int kk = 0; kk < MTOK; ++kk)
              load_frag(gb[kk], sgT + (uint32_t)(((M0 + mi) * 16 * LdG + kk * 16) * 2));
#pragma unroll
            for (int kk = 0; kk < MTOK; ++kk)
#pragma unroll
              for (int p = 0; p < 2; ++p) mma(acc[2 * (M0 + mi) + p], va[kk], &gb[kk][2 * p]);
          }
          //: THE INTRA TERM'S SHARE OF THE MASS, same compile-time base.
          if (gwarp < kIntraUnits) {
#pragma unroll
            for (int mi = 0; mi < MI; ++mi)
#pragma unroll
              for (int q = 0; q < kIntraKD; ++q) {
                uint32_t mg[4];
                load_frag(mg, sgT + (uint32_t)(((M0 + mi) * 16 * LdG + q * kIntraUnits * 16) * 2)
                                  + massOff);
#pragma unroll
                for (int p = 0; p < 2; ++p) mma(macc[2 * (M0 + mi) + p], ones_a, &mg[2 * p]);
              }
          }
        };
        if constexpr (GROUPS == 1) {
          intra_half(std::integral_constant<int, 0>{});
        } else {
          if (grp == 0)
            intra_half(std::integral_constant<int, 0>{});
          else
            intra_half(std::integral_constant<int, MI>{});
        }
      }

      // ---- 6. stage this chunk's partial output; the fan-in is next iteration --
#pragma unroll
      for (int j = 0; j < NTOK; ++j) {
        const int jo = j * 8 * LdO;
        store_f32(&so[oE[0][0] + jo], acc[j][0]);
        store_f32(&so[oE[1][0] + jo], acc[j][1]);
        store_f32(&so[oE[0][1] + jo], acc[j][2]);
        store_f32(&so[oE[1][1] + jo], acc[j][3]);
      }
      //: THE MASS ROW IS ROW 0 OF ITS TILE, which in this atom's accumulator is
      //: held by the four lanes of the first group, two tokens each.
      if (lane < 4) {
#pragma unroll
        for (int j = 0; j < NTOK; ++j) {
          store_f32(&sden[gwarp * C + j * 8 + lane * 2], macc[j][0]);
          store_f32(&sden[gwarp * C + j * 8 + lane * 2 + 1], macc[j][1]);
        }
      }
    }
    prev_d = dc;
    prev_nr = nr;
    if (++dc == IRING) dc = 0;
    if (++ds == IRING) ds = 0;
    if (++dn == IRING) dn = 0;
  }

  rendezvous();
  if (prev_nr) {
    drain(prev_d, prev_nr);
    drain_den(prev_d, prev_nr);
  }

  //: ---- the FINAL STATE.  The mirror of the entry load, after the last drain
  //: has consumed the staging buffer -- which is what the one extra rendezvous
  //: below is for.  Both groups iterate the same atom count against their own
  //: `so`, so every barrier here stays CTA-uniform.
  if constexpr (STATE) {
    if (s_out != nullptr) {
      const int atomFirst = atom_first();
      const int32_t* const tbl = page_tbl != nullptr ? pin_address(page_tbl) : nullptr;
      float* const plane = pin_address(s_out);
      rendezvous();
#pragma unroll
      for (int i = 0; i < kAtoms; ++i) {
        const int slot = slot_of(tbl, atomFirst + i);
        if (slot >= 0) {
#pragma unroll
          for (int p = 0; p < 2; ++p)
#pragma unroll
            for (int k = 0; k < 4; ++k) store_f32(&so[page_value(p, k)], st[2 * i + p][k]);
          if (gwarp == i % kMassUnits && lane < 4) {
#pragma unroll
            for (int p = 0; p < 2; ++p)
#pragma unroll
              for (int k = 0; k < 2; ++k)
                store_f32(&so[page_mass(p, k)], mass_st[2 * (i / kMassUnits) + p][k]);
          }
        }
        rendezvous();
        if (slot >= 0) {
          page_sweep(page_off(slot), [&](int s, uint32_t off) {
            store_f32(reinterpret_cast<float*>(gather_ptr(plane, off)), so[s]);
          });
        }
        rendezvous();
      }
    }
  }
#undef ADDRF
#undef ADDRT
#undef ADDRT_M
#undef ADDRF_M
#undef ADDRT_DV
}

// ------------------------------------------------------- the union table
//: THE TABLE IS BUILT ONCE FOR THE WHOLE GRID.  A lockstep unit takes thirty-two
//: consecutive tokens, one per lane, and votes once per entry; the vote IS the
//: word, so nothing is packed afterwards and nothing is zeroed first.
//:
//: Its cost does not depend on `BC`: a lane reads its token's amplitude row once
//: per level that has entries, whatever the spans partition it into.
template <int D, int B, int BC>
__global__ void union_table_kernel(const __nv_bfloat16* __restrict__ pread,
                                   const __nv_bfloat16* __restrict__ pwrite,
                                   uint32_t* __restrict__ tbl, int L) {
  using LP = LevelPlan<D, B, BC>;
  constexpr int kWtot = LP::col_base(LP::depth);
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
  uint32_t* const outBase[2] = {tbl + ((long)(2 * blockIdx.y + 0) * LP::entries) * entryStride + w,
                                tbl + ((long)(2 * blockIdx.y + 1) * LP::entries) * entryStride + w};
  const __nv_bfloat16* const ampBase[2] = {pread + off, pwrite + off};

  //: one stanza per level, spans from the plan -- a whole level emits
  //: nothing, exactly as its clause is not evaluated.  Amp row layout is
  //: [level 0 | level 1 | ...], each `width(l)` wide.
#pragma unroll
  for (int side = 0; side < 2; ++side) {
    uint32_t* out = outBase[side];
#define ROLA_TABLE_LEVEL(L_)                                                   \
    if constexpr (L_ < LP::depth) {                                            \
      const __nv_bfloat16 *p = ampBase[side] + LP::col_base(L_);               \
      constexpr int kSpan = LP::span_at(L_);                                   \
      constexpr int kEnt = LP::entries_at(L_);                                 \
      _Pragma("unroll") for (int e = 0; e < kEnt; ++e) {                       \
        const uint32_t v = or_run<kSpan>(p);                                   \
        const uint32_t vote =                                                  \
            lane_ballot(real && (v & kBf16MagnitudeMask) != 0u);               \
        if (lane == 0) *out = vote;                                            \
        p += kSpan;                                                            \
        out += entryStride;                                                    \
      }                                                                        \
    }
    ROLA_TABLE_LEVEL(0)
    ROLA_TABLE_LEVEL(1)
    ROLA_TABLE_LEVEL(2)
    ROLA_TABLE_LEVEL(3)
#undef ROLA_TABLE_LEVEL
  }
}

// ------------------------------------------------------- the atom-grain stats
//: THE STATS PASS.  Two kernels that answer ONE question: which atoms does this
//: call's routing WRITE?  The answer keys the state's residency, so it is taken
//: at the invariant quantum (`kAtomLeaves`) and never at `BC` -- the derivation,
//: the exactness argument and the rejected coarser reductions are in
//: docs/internals/paging/stats.md.
template <int D, int B>
__global__ void atom_table_kernel(const __nv_bfloat16* __restrict__ pwrite,
                                  uint32_t* __restrict__ tbl, int L) {
  using LP = LevelPlan<D, B, kAtomLeaves>;
  constexpr int kWtot = LP::col_base(LP::depth);
  const int tokWords = tok_words(L);
  const int w = blockIdx.x * (blockDim.x / kBallotLanes) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (w >= tokWords) return;  // a whole lockstep unit leaves together
  const int t = w * kTokBits + lane;
  const bool real = t < L;
  const long off = (long)blockIdx.y * L * kWtot + (long)(real ? t : 0) * kWtot;

  const long entryStride = tokWords;
  uint32_t* out = tbl + ((long)blockIdx.y * LP::entries) * entryStride + w;
  const __nv_bfloat16* const ampBase = pwrite + off;

#define ROLA_ATOM_TABLE_LEVEL(L_)                                              \
  if constexpr (L_ < LP::depth) {                                              \
    const __nv_bfloat16* p = ampBase + LP::col_base(L_);                       \
    constexpr int kSpan = LP::span_at(L_);                                     \
    constexpr int kEnt = LP::entries_at(L_);                                   \
    _Pragma("unroll") for (int e = 0; e < kEnt; ++e) {                          \
      const uint32_t v = or_run<kSpan>(p);                                     \
      const uint32_t vote = lane_ballot(real && (v & kBf16MagnitudeMask) != 0u); \
      if (lane == 0) *out = vote;                                              \
      p += kSpan;                                                              \
      out += entryStride;                                                      \
    }                                                                          \
  }
  ROLA_ATOM_TABLE_LEVEL(0)
  ROLA_ATOM_TABLE_LEVEL(1)
  ROLA_ATOM_TABLE_LEVEL(2)
  ROLA_ATOM_TABLE_LEVEL(3)
#undef ROLA_ATOM_TABLE_LEVEL
}

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
//: back (docs/internals/chunk/chunk_kernel.md#block-bitmap).
constexpr int block_bitmap_group(int inner) {
  return inner < (kAtomThreads / kBallotLanes) / 2 ? inner : (kAtomThreads / kBallotLanes) / 2;
}

//: THE LIVENESS EPILOGUE, at the LAUNCH's block grain: `atom_bitmap_kernel`'s
//: reduction re-instantiated at `BC`, over BOTH sides, with the page table's
//: residency condensed beside them.  The three facts are emitted SEPARATELY --
//: one byte per (bh, block), bits {READ, WRITE, STATE} -- because what a skipped
//: block would have contributed differs by call class and by backing, so the
//: predicate over them is the host's (docs/internals/chunk/chunk_kernel.md#block-bitmap).
template <int D, int B, int BC>
__global__ void block_bitmap_kernel(const uint32_t* __restrict__ tbl,
                                    const int32_t* __restrict__ page_tbl,
                                    uint8_t* __restrict__ block_bits, int L) {
  using LP = LevelPlan<D, B, BC>;
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
// is the whole of the kernel's knowledge of it -- docs/internals/chunk/chunk_kernel.md#arm-set
#include "../instantiations/chunk_arms.inc"

}  // namespace rola::chunk
