// csrc/rola/src/carry/box.cuh -- THE (k, m) BOX LATTICE, as constexpr arithmetic.
//
// The R0 spec's LATTICE-1 linearization (development/plans/K31_R0_ABI_RIPPLE_SPEC.md
// sections 2.1-2.5) made executable: a leaf's canonical index `ell` and its
// lattice index `lambda` relate by a fixed BIT PERMUTATION, so nothing here
// divides and nothing here tabulates. Full symbol table and derivation:
// see docs/internals/carry/box.md
#pragma once

#include <cstdint>

namespace rola::carry {

constexpr int ilog2c(int x) { return x <= 1 ? 0 : 1 + ilog2c(x >> 1); }
constexpr long ipowl_c(int b, int e) { return e == 0 ? 1 : b * ipowl_c(b, e - 1); }
//: the smallest power of two at least `x`, for a lane split whose natural count is -- see docs/internals/carry/box.md#ceilpow2c
constexpr int ceil_pow2c(int x) { return x <= 1 ? 1 : (1 << (ilog2c(2 * x - 1))); }
constexpr int align16c(int x) { return (x + 15) & ~15; }

//: the page granule: the leaves one atom holds, and the unit every DRAM gather -- see docs/internals/carry/box.md#katomleaves-3
static constexpr int kAtomLeaves = 16;

//: THE FAMILY'S ARM-INDEPENDENT CONSTANTS (moved here at K36). -- see docs/internals/carry/box.md#kcarrybuildmagic
static constexpr int kCarryBuildMagic = 31010026;

//: THE CHANNEL-SPLIT BODY'S EXECUTOR COUNT.  A CARVED arm's is its STREAM count -- see docs/internals/carry/box.md#kwarps-2
static constexpr int kWarps = 4;
static constexpr int kThreads = kWarps * 32;

//: THE OWNER'S SHAPE, DERIVED. Nothing in this struct is a table: every field -- see docs/internals/carry/box.md#boxplan
template <int D, int B, int K, int M>
struct BoxPlan {
  static constexpr int kLevels = D;
  static constexpr int kWidth = B;
  static constexpr int kSpan = K;  // the sub-box span, uniform BY LAW
  static constexpr int kCap = M;   // the capacity multiplier
  static constexpr int kKappa = ilog2c(K);
  static constexpr int kBeta = ilog2c(B);
  static constexpr int kJ = ilog2c(M);

  //: spec section 2.5. -- see docs/internals/carry/box.md#katombits
  static constexpr int kAtomBits = ilog2c(kAtomLeaves);
  static constexpr int kAFloor = kAtomBits > kKappa ? kAtomBits - kKappa : 0;
  static constexpr int kAShare = kJ / D + ((kJ % D) ? 1 : 0);
  static constexpr int kACeil = kBeta > kKappa ? kBeta - kKappa : 0;
  static constexpr int kAWant = kAShare > kAFloor ? kAShare : kAFloor;
  static constexpr int kAInner = kAWant > kACeil ? kACeil : kAWant;
  static constexpr int kJOuter = kJ - kAInner;

  static constexpr int a(int l) {
    return (l == D - 1 || D == 1)
               ? kAInner
               : kJOuter / (D - 1) + ((l >= (D - 1) - (kJOuter % (D - 1))) ? 1 : 0);
  }
  static constexpr int m_l(int l) { return 1 << a(l); }
  static constexpr int s(int l) { return K * m_l(l); }
  static constexpr int g(int l) { return B / s(l); }

  static constexpr int prod_s(int l) { return l == D ? 1 : s(l) * prod_s(l + 1); }
  static constexpr int prod_g(int l) { return l == D ? 1 : g(l) * prod_g(l + 1); }
  static constexpr int kBC = prod_s(0);
  static constexpr int kOwners = prod_g(0);
  static constexpr long kLeaves = ipowl_c(B, D);

  //: THE ATOM'S DIGIT RECTANGLE. An atom is `kAtomLeaves` consecutive `local` -- see docs/internals/carry/box.md#atom-base
  static constexpr int atom_base(int l) { return prod_s(l) <= kAtomLeaves ? l : atom_base(l + 1); }
  static constexpr int kAtomBase = atom_base(0);
  static constexpr int kAtomWhole = D - kAtomBase;
  static constexpr int kAtomPartial = kAtomLeaves / prod_s(kAtomBase);
  static constexpr int kAtomLevels = kAtomWhole + (kAtomPartial > 1 ? 1 : 0);

  //: THE WALK UNIT, reconciled with the atom. -- see docs/internals/carry/box.md#unit-from
  static constexpr int unit_from(int l) {
    return prod_s(l) % kAtomLeaves == 0 ? l : unit_from(l - 1);
  }
  static constexpr int kUnitBase = unit_from(D - 1);
  static constexpr int kUnitLeaves = prod_s(kUnitBase);
  static constexpr int kUnitAtoms = kUnitLeaves / kAtomLeaves;

  //: the divisor that extracts owner coordinate `o_l` from the owner index `O`: -- see docs/internals/carry/box.md#ownerdiv
  static constexpr int owner_div(int l) { return prod_g(l + 1); }

  //: THE LOCAL INDEX'S BIT FIELDS: `local = lambda - O*BC` is the owner's run -- see docs/internals/carry/box.md#span-bits
  static constexpr int span_bits(int l) { return kKappa + a(l); }  // log2 s_l
  static constexpr int sum_span(int l) { return l >= D ? 0 : span_bits(l) + sum_span(l + 1); }
  static constexpr int run_shift(int l) { return sum_span(l + 1); }
  static constexpr int kLocalBits = sum_span(0);

  //: the offset of `local`'s level-`l` digit INSIDE the owner's run.
  static constexpr int run_of(int l, int local) { return (local >> run_shift(l)) & (s(l) - 1); }
  static constexpr int u_of(int l, int local) { return run_of(l, local) >> kKappa; }
  static constexpr int c_of(int l, int local) { return run_of(l, local) & (K - 1); }

  //: THE OWNER'S BALLOT ROWS in the facts table (`SpanPlan<D, B, s_0..s_D-1>`'s -- see docs/internals/carry/box.md#row-g
  static constexpr int row_g(int l) { return s(l) < B ? g(l) : 0; }
  static constexpr int row_entries(int l) { return l == D ? 0 : row_g(l) + row_entries(l + 1); }
  static constexpr int kRows = row_entries(0);
  static constexpr int row_prefix(int l) { return l == 0 ? 0 : row_prefix(l - 1) + row_g(l - 1); }

  //: the level's column base in the packed `[BH, L, sum_l B_l]` amplitude plane.
  static constexpr int col_base(int l) { return l * B; }
  static constexpr int kWtot = D * B;

  //: THE STAGED FACTOR ROW (K35 S2). What a fetcher needs of one token on one -- see docs/internals/carry/box.md#kstageasync
  static constexpr int kStageAsync = 8;  // amplitudes in the widest asynchronous run
  static constexpr int stage_el(int l) { return s(l) < kStageAsync ? s(l) : kStageAsync; }
  static constexpr int stage_off(int l) {
    return l == 0 ? 0
                  : ((stage_off(l - 1) + s(l - 1) + stage_el(l) - 1) / stage_el(l)) * stage_el(l);
  }
  static constexpr int kStageW =
      ((stage_off(D - 1) + s(D - 1) + kStageAsync - 1) / kStageAsync) * kStageAsync;
  static constexpr int stage_chunks(int l) { return s(l) / stage_el(l); }
  static constexpr int stage_chunk_base(int l) {
    return l == 0 ? 0 : stage_chunk_base(l - 1) + stage_chunks(l - 1);
  }
  static constexpr int kStageChunks = stage_chunk_base(D - 1) + stage_chunks(D - 1);
  static_assert(kSpan * 2 >= 4, "a level's run is at least one asynchronous quantum wide");

  //: FORWARD, `pi: ell -> lambda`, and its inverse -- used by the host seam and -- see docs/internals/carry/box.md#digit-of
  static constexpr int digit_of(int l, long ell) {
    return (int)((ell / ipowl_c(B, D - 1 - l)) % B);
  }

  //: THE ATOM RE-KEY (P80). -- see docs/internals/carry/box.md#canon-leaf-acc
  template <int L>
  static __host__ __device__ __forceinline__ uint32_t canon_leaf_acc(int O, int local) {
    if constexpr (L >= D) {
      return 0u;
    } else {
      const int d = ((O / owner_div(L)) % g(L)) * s(L) + ((local >> run_shift(L)) & (s(L) - 1));
      return (uint32_t)d * (uint32_t)ipowl_c(B, D - 1 - L) + canon_leaf_acc<L + 1>(O, local);
    }
  }
  //: owner `O`'s atom `a`, as a CANONICAL atom id inside its own `bh`.
  static __host__ __device__ __forceinline__ uint32_t canon_atom(int O, int a) {
    return canon_leaf_acc<0>(O, a * kAtomLeaves) >> kAtomBits;
  }

  static_assert(kBC % kAtomLeaves == 0, "K2: an atom must not straddle two owners");

  //: K6: an owner's run at a level is an aligned run INSIDE that level. -- see docs/internals/carry/box.md#spans-fit
  static constexpr bool spans_fit(int l) {
    return l >= D || (s(l) <= B && B % s(l) == 0 && spans_fit(l + 1));
  }
  static constexpr int sum_a(int l) { return l >= D ? 0 : a(l) + sum_a(l + 1); }
  static_assert(kJOuter >= 0 && sum_a(0) == kJ && spans_fit(0),
                "K6: every owner span must be an aligned run inside its level; this (k, m) "
                "asks for more capacity than the topology can carry");
  //: K3/K4: an atom is 16 consecutive `local` values at a multiple of 16, so -- see docs/internals/carry/box.md#static-assert
  static_assert(kLocalBits >= 4, "K3: an atom must lie inside one owner");
  static_assert((1 << kKappa) == K && (1 << kBeta) == B && (1 << kJ) == M,
                "K1: k, m and B are powers of two");
};

//: THE PER-LEVEL MODE DECLARATION (the seam's own encoding): two bits per -- see docs/internals/carry/box.md#ksideread
constexpr uint32_t kSideRead = 1u, kSideWrite = 2u;
constexpr bool mode_declares(uint32_t modes, int l, uint32_t side) {
  return ((modes >> (2 * l)) & side) != 0u;
}

//: THE PRIVATE ORDER, BY MODE (P76, "ORDER BY MODE, THEN CARVE", ruled -- see docs/internals/carry/box.md#modeorder
template <int D, uint32_t MO, uint32_t SIDE>
struct ModeOrder {
  static constexpr int kDense = 0, kSparse = 1, kBothSparse = 2;
  static constexpr int group(int l) {
    return (mode_declares(MO, l, kSideRead) && mode_declares(MO, l, kSideWrite))
               ? kBothSparse
               : (mode_declares(MO, l, SIDE) ? kSparse : kDense);
  }
  //: the rule as a strict order, and `rank` as the position it induces -- `0` innermost. -- see docs/internals/carry/box.md#before-2
  static constexpr bool before(int a, int b) {
    return group(a) != group(b) ? group(a) < group(b) : a > b;
  }
  static constexpr int count_before(int l, int i) {
    return i >= D ? 0 : (before(i, l) ? 1 : 0) + count_before(l, i + 1);
  }
  static constexpr int rank(int l) { return count_before(l, 0); }
  //: the level at OUTER position `p`, which is the order the carve walks.
  static constexpr int at_rank(int r, int l) {
    return l >= D ? -1 : (rank(l) == r ? l : at_rank(r, l + 1));
  }
  static constexpr int outer(int p) { return at_rank(D - 1 - p, 0); }
};

//: THE STREAM BOX, DERIVED FROM THE ORDER AND A STREAM COUNT. A STREAM owns a -- see docs/internals/carry/box.md#streamspans
template <int D, int B, int K, int M, uint32_t MO, uint32_t SIDE, int NS>
struct StreamSpans {
  using BP = BoxPlan<D, B, K, M>;
  using Ord = ModeOrder<D, MO, SIDE>;
  //: the walk, as two mutually recursive counts over OUTER POSITIONS: what the count -- see docs/internals/carry/box.md#gat
  static constexpr int g_at(int p);
  static constexpr int need_at(int p) { return p <= 0 ? NS : need_at(p - 1) / g_at(p - 1); }
  static constexpr int s(int l) { return l >= D ? B : BP::s(l) / g_at(D - 1 - Ord::rank(l)); }
  static constexpr int g(int l) { return BP::s(l) / s(l); }
  static constexpr bool carves(int l) { return s(l) < BP::s(l); }
  static constexpr int prod_g(int l) { return l >= D ? 1 : g(l) * prod_g(l + 1); }
  static constexpr int prod_s(int l) { return l >= D ? 1 : s(l) * prod_s(l + 1); }
  static constexpr int kStreams = prod_g(0);
  static constexpr int kLeaf = prod_s(0);
  static constexpr int grid_div(int l) { return prod_g(l + 1); }

  static_assert(NS >= 1 && (NS & (NS - 1)) == 0, "a side's stream count is a power of two");
  static_assert(kStreams == NS,
                "the stream count must be reachable by carving this side's levels: it "
                "exceeds the leaves one owner holds");
  static_assert(kLeaf * kStreams == BP::kBC, "the stream grid partitions the owner's box");
};

template <int D, int B, int K, int M, uint32_t MO, uint32_t SIDE, int NS>
constexpr int StreamSpans<D, B, K, M, MO, SIDE, NS>::g_at(int p) {
  return BP::s(Ord::outer(p)) < need_at(p) ? BP::s(Ord::outer(p)) : need_at(p);
}

//: ONE SIDE'S BALLOT ROWS. -- see docs/internals/carry/box.md#carverows
template <class SV>
struct CarveRows {
  using BP = typename SV::BP;
  static constexpr int kLevels = BP::kLevels, kWidth = BP::kWidth;
  static constexpr int s(int l) { return SV::s(l); }
  static constexpr int row_g(int l) { return s(l) < kWidth ? kWidth / s(l) : 0; }
  static constexpr int row_entries(int l) {
    return l == kLevels ? 0 : row_g(l) + row_entries(l + 1);
  }
  static constexpr int kRows = row_entries(0);
  static constexpr int row_prefix(int l) { return l == 0 ? 0 : row_prefix(l - 1) + row_g(l - 1); }
  //: how many table rows a digit run of `span` covers at level `l`.
  static constexpr int rows_of(int span, int l) { return span / s(l); }
};

}  // namespace rola::carry
