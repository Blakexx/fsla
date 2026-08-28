// csrc/rola/src/carry/box.cuh -- THE (k, m) BOX LATTICE, as constexpr arithmetic.
//
// This is the R0 spec's LATTICE-1 linearization made executable
// (development/plans/K31_R0_ABI_RIPPLE_SPEC.md sections 2.1-2.5).  Symbols, each
// defined where it is introduced below: `D` levels, `B` digits per level,
// `K` = the uniform sub-box span per level, `M` = the capacity multiplier
// (`M = prod_l m_l`), `s_l = K * m_l` = an owner's span at level `l`,
// `g_l = B / s_l` = the owner-grid extent at level `l`, `BC = prod_l s_l` = the
// leaves one owner holds, `N = B^D`.
//
// A leaf's canonical (MSB-first mixed-radix) index is `ell`; its lattice index is
// `lambda`, and the map between them is a fixed BIT PERMUTATION -- which is why
// nothing here divides and nothing here tabulates.
#pragma once

#include <cstdint>

namespace rola::carry {

constexpr int ilog2c(int x) { return x <= 1 ? 0 : 1 + ilog2c(x >> 1); }
constexpr long ipowl_c(int b, int e) { return e == 0 ? 1 : b * ipowl_c(b, e - 1); }
//: the smallest power of two at least `x`, for a lane split whose natural count is
//: not one (a thread group must be a power of two to be a shift and a mask).
constexpr int ceil_pow2c(int x) { return x <= 1 ? 1 : (1 << (ilog2c(2 * x - 1))); }
constexpr int align16c(int x) { return (x + 15) & ~15; }

//: the page granule: the leaves one atom holds, and the unit every DRAM gather
//: moves.  It is an addressing constant of the lattice, which is why the box's
//: constraints (K2, K3, K6) are stated against it here rather than at a consumer.
static constexpr int kAtomLeaves = 16;

//: THE OWNER'S SHAPE, DERIVED.  Nothing in this struct is a table: every field is
//: a product or a bit width, so an owner's addressing is shifts and masks at every
//: use site.  The distribution of `M` across the levels is spec section 2.5 as the
//: atom-rectangle law generalizes it -- innermost first, capped by K6, and balanced
//: over what remains: docs/internals/carry/box.md.
template <int D, int B, int K, int M>
struct BoxPlan {
  static constexpr int kLevels = D;
  static constexpr int kWidth = B;
  static constexpr int kSpan = K;  // the sub-box span, uniform BY LAW
  static constexpr int kCap = M;   // the capacity multiplier
  static constexpr int kKappa = ilog2c(K);
  static constexpr int kBeta = ilog2c(B);
  static constexpr int kJ = ilog2c(M);

  //: spec section 2.5.  The innermost level takes as much capacity as it can carry
  //: -- never less than its balanced share, never less than an atom asks, and never
  //: more than its own width admits (K6) -- and the rest of `j` is spread balanced
  //: over the outer levels, innermost first, so the exponents differ by at most one
  //: there.  The K6 ceiling is what makes the convention total: a level narrower
  //: than an atom cannot absorb the atom floor, and the capacity spills outward
  //: instead of the plan being refused.
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

  //: THE ATOM'S DIGIT RECTANGLE.  An atom is `kAtomLeaves` consecutive `local`
  //: values at a multiple of `kAtomLeaves`, so under the mixed-radix order it is
  //: the TRAILING levels whose spans multiply into it, taken whole, times an
  //: aligned sub-run of the next level out.  `kAtomBase` names the outermost level
  //: lying wholly inside; `kAtomPartial` is how many of level `kAtomBase - 1`'s
  //: digits the atom takes.  A token fills an atom exactly when every level of the
  //: rectangle is DENSE for its side, which is why the convention pushes capacity
  //: innermost: a one-level rectangle asks that of one level only.
  static constexpr int atom_base(int l) { return prod_s(l) <= kAtomLeaves ? l : atom_base(l + 1); }
  static constexpr int kAtomBase = atom_base(0);
  static constexpr int kAtomWhole = D - kAtomBase;
  static constexpr int kAtomPartial = kAtomLeaves / prod_s(kAtomBase);
  static constexpr int kAtomLevels = kAtomWhole + (kAtomPartial > 1 ? 1 : 0);

  //: THE WALK UNIT, reconciled with the atom.  The decode walk enumerates
  //: `(owner, r_0 .. r_{kUnitBase-1})` and the unit's leaves are the trailing spans
  //: from `kUnitBase` inward -- the SHALLOWEST such suffix that holds a whole number
  //: of atoms, so the candidate enumeration is a partition at every admissible span
  //: vector.  K2 makes `l = 0` always qualify, so the search terminates.
  static constexpr int unit_from(int l) {
    return prod_s(l) % kAtomLeaves == 0 ? l : unit_from(l - 1);
  }
  static constexpr int kUnitBase = unit_from(D - 1);
  static constexpr int kUnitLeaves = prod_s(kUnitBase);
  static constexpr int kUnitAtoms = kUnitLeaves / kAtomLeaves;

  //: the divisor that extracts owner coordinate `o_l` from the owner index `O`:
  //: `o_l = (O / owner_div(l)) % g_l`, MSB-first exactly as the canonical order is.
  static constexpr int owner_div(int l) { return prod_g(l + 1); }

  //: THE LOCAL INDEX'S BIT FIELDS: `local = lambda - O*BC` is the owner's run
  //: offsets in MIXED RADIX, most significant level first --
  //: `local = sum_l r_l * prod_{l' > l} s_l'`, with `r_l = d_l - o_l*s_l` in
  //: `[0, s_l)` -- so each level's DIGITS ARE CONTIGUOUS IN LOCAL, level `D-1`
  //: fastest.  This is the K33 enabler (queue card, enabler 2): a bucket keyed on
  //: level `l`'s digit owns a run of `prod_{l' > l} s_l'` leaves at stride
  //: `s_l`-independent places, and for the OUTERMOST level that run is contiguous
  //: and `BC / s_0` wide, which at the flagship is two `kMmaN` tiles -- so a fold
  //: bucket accumulates DIRECTLY into whole state tiles with zero padding.  `pi`
  //: is a fixed BIT PERMUTATION of the canonical index: a level's whole field is
  //: one contiguous bit run.
  static constexpr int span_bits(int l) { return kKappa + a(l); }  // log2 s_l
  static constexpr int sum_span(int l) { return l >= D ? 0 : span_bits(l) + sum_span(l + 1); }
  static constexpr int run_shift(int l) { return sum_span(l + 1); }
  static constexpr int kLocalBits = sum_span(0);

  //: the offset of `local`'s level-`l` digit INSIDE the owner's run.
  static constexpr int run_of(int l, int local) { return (local >> run_shift(l)) & (s(l) - 1); }
  static constexpr int u_of(int l, int local) { return run_of(l, local) >> kKappa; }
  static constexpr int c_of(int l, int local) { return run_of(l, local) & (K - 1); }

  //: THE OWNER'S BALLOT ROWS in the facts table (`SpanPlan<D, B, s_0..s_D-1>`'s
  //: entry accounting, mirrored here for the same reason the seam derives its own
  //: strides: two derivations disagree exactly when one is wrong).  A level spanned
  //: WHOLE by the owner emits no rows -- its clause is true for every token and
  //: drops out of the conjunction on its own.
  static constexpr int row_g(int l) { return s(l) < B ? g(l) : 0; }
  static constexpr int row_entries(int l) { return l == D ? 0 : row_g(l) + row_entries(l + 1); }
  static constexpr int kRows = row_entries(0);
  static constexpr int row_prefix(int l) { return l == 0 ? 0 : row_prefix(l - 1) + row_g(l - 1); }

  //: the level's column base in the packed `[BH, L, sum_l B_l]` amplitude plane.
  static constexpr int col_base(int l) { return l * B; }
  static constexpr int kWtot = D * B;

  //: THE STAGED FACTOR ROW (K35 S2).  What a fetcher needs of one token on one side
  //: is the OWNER's digit run at each level -- `sum_l s_l` amplitudes -- and NOT the
  //: `kWtot`-wide packed row, which is the width of the whole topology.  The runs are
  //: laid out level by level, each ALIGNED to its own asynchronous quantum, so a
  //: level's run is one `stage_run` per `kStageAsync` amplitudes and both the global
  //: and the shared address are naturally aligned by construction (a level's column
  //: base is a multiple of `B` and an owner's run starts at a multiple of `s_l`).
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

  //: FORWARD, `pi: ell -> lambda`, and its inverse -- used by the host seam and by
  //: nothing on the device (the device never converts; it is born in lattice order).
  static constexpr int digit_of(int l, long ell) {
    return (int)((ell / ipowl_c(B, D - 1 - l)) % B);
  }

  static_assert(kBC % kAtomLeaves == 0, "K2: an atom must not straddle two owners");

  //: K6: an owner's run at a level is an aligned run INSIDE that level.  With the
  //: capacity ceiling above it is the constraint the convention can still miss --
  //: when `j` exceeds what `D` levels of width `B` can carry, there is no span
  //: vector and the arm is refused rather than served a narrower box.
  static constexpr bool spans_fit(int l) {
    return l >= D || (s(l) <= B && B % s(l) == 0 && spans_fit(l + 1));
  }
  static constexpr int sum_a(int l) { return l >= D ? 0 : a(l) + sum_a(l + 1); }
  static_assert(kJOuter >= 0 && sum_a(0) == kJ && spans_fit(0),
                "K6: every owner span must be an aligned run inside its level; this (k, m) "
                "asks for more capacity than the topology can carry");
  //: K3/K4: an atom is 16 consecutive `local` values at a multiple of 16, so under
  //: the mixed-radix order it is the set that agrees on every bit above bit 3 --
  //: the levels whose fields lie above are FIXED, the level straddling bit 4 keeps
  //: an aligned contiguous sub-run, and the levels below are free.  That is an
  //: aligned digit rectangle for EVERY admissible span vector, so the two clauses
  //: the interleaved order needed (`k^D >= 16` and `4 % kappa == 0`) are discharged
  //: by the order itself and only the owner's size survives as a condition.
  static_assert(kLocalBits >= 4, "K3: an atom must lie inside one owner");
  static_assert((1 << kKappa) == K && (1 << kBeta) == B && (1 << kJ) == M,
                "K1: k, m and B are powers of two");
};

//: THE PER-LEVEL MODE DECLARATION (the seam's own encoding): two bits per level, level 0
//: in the low pair, bit 0 = the READ side is value-sparse there, bit 1 = the WRITE side.
//: It PRUNES the liveness clause and the order and carve below are derived from it, so it
//: lives beside the lattice and not at a consumer.
constexpr uint32_t kSideRead = 1u, kSideWrite = 2u;
constexpr bool mode_declares(uint32_t modes, int l, uint32_t side) {
  return ((modes >> (2 * l)) & side) != 0u;
}

//: THE PRIVATE ORDER, BY MODE (P76, "ORDER BY MODE, THEN CARVE", ruled 2026-08-26):
//: DENSE innermost, then SPARSE (sparse on this side only), then BOTH_SPARSE (sparse on
//: both), lattice order inside a class.  It is a function of `(D, MO, SIDE)` ALONE -- no
//: span is consulted -- which is why the carve is derived from it rather than declared
//: beside it.  What each class buys: docs/internals/carry/box.md.
template <int D, uint32_t MO, uint32_t SIDE>
struct ModeOrder {
  static constexpr int kDense = 0, kSparse = 1, kBothSparse = 2;
  static constexpr int group(int l) {
    return (mode_declares(MO, l, kSideRead) && mode_declares(MO, l, kSideWrite))
               ? kBothSparse
               : (mode_declares(MO, l, SIDE) ? kSparse : kDense);
  }
  //: the rule as a strict order, and `rank` as the position it induces -- `0` innermost.
  //: Both are counts, so neither tabulates.
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

//: THE STREAM BOX, DERIVED FROM THE ORDER AND A STREAM COUNT.  A STREAM owns a box and
//: walks its data into tiles; an EXECUTOR runs streams (the terminology law).  The walk
//: takes digits from the OUTERMOST level of that order inward, carving each level whole
//: and then partially, until the stream grid's product IS the count -- so `S_l` is what
//: one stream keeps of level `l` and the box is an AFFINE RECTANGLE of digits, contiguity
//: in leaf order being a property of carving a prefix and never a requirement.  The grid
//: is `prod_l g_l` with `g_l = s_l / S_l`, a stream's coordinate at a level is a bit
//: field of its index MSB-first as the owner grid's is, and everything is a power of two,
//: so every use site is a shift and a mask.  A level past `D` carries `B`.
template <int D, int B, int K, int M, uint32_t MO, uint32_t SIDE, int NS>
struct StreamSpans {
  using BP = BoxPlan<D, B, K, M>;
  using Ord = ModeOrder<D, MO, SIDE>;
  //: the walk, as two mutually recursive counts over OUTER POSITIONS: what the count
  //: still asks for when position `p` is reached, and what that position hands back.
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

//: ONE SIDE'S BALLOT ROWS.  A box's schedule is the AND over levels of its runs'
//: bitlists, per side, per level, per run of THAT SIDE's partition; each side votes from
//: its own factors, so nothing was gained by holding one partition for both and a
//: stream's test reads EXACTLY ONE row per tested level (only the owner's whole-run OR
//: spans several).  A level the side spans WHOLE emits no rows -- its clause is true for
//: every token.  Sigma-shaped, `sum_l B/S_l`, so it does not grow with `N`.
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
