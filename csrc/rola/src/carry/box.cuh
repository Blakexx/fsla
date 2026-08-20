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

//: THE OWNER'S SHAPE, DERIVED.  Nothing in this struct is a table: every field is
//: a product or a bit width, so an owner's addressing is shifts and masks at every
//: use site.  The balanced round-robin distribution of `M` across the levels is
//: spec section 2.5 -- level 0 grows first, and the exponents differ by at most one.
template <int D, int B, int K, int M>
struct BoxPlan {
  static constexpr int kLevels = D;
  static constexpr int kWidth = B;
  static constexpr int kSpan = K;  // the sub-box span, uniform BY LAW
  static constexpr int kCap = M;   // the capacity multiplier
  static constexpr int kKappa = ilog2c(K);
  static constexpr int kBeta = ilog2c(B);
  static constexpr int kJ = ilog2c(M);

  //: spec section 2.5: `a_l = floor(j/D) + [l < j mod D]`.
  static constexpr int a(int l) { return kJ / D + ((l < (kJ % D)) ? 1 : 0); }
  static constexpr int m_l(int l) { return 1 << a(l); }
  static constexpr int s(int l) { return K * m_l(l); }
  static constexpr int g(int l) { return B / s(l); }

  static constexpr int prod_s(int l) { return l == D ? 1 : s(l) * prod_s(l + 1); }
  static constexpr int prod_g(int l) { return l == D ? 1 : g(l) * prod_g(l + 1); }
  static constexpr int kBC = prod_s(0);
  static constexpr int kOwners = prod_g(0);
  static constexpr long kLeaves = ipowl_c(B, D);

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
  //: and `BC / s_0` wide, which at the flagship is exactly one `kMmaN` tile -- so a
  //: fold bucket accumulates DIRECTLY into whole state tiles with zero padding.
  //: (It replaces the R0 spec's `[u_0|..|u_{D-1}|c_0|..|c_{D-1}]` interleave, which
  //: split each level's run across two fields and gave no level a contiguous run.
  //: `pi` is still a fixed BIT PERMUTATION of the canonical index -- more simply so,
  //: since a level's whole field is now one contiguous bit run.)
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

  //: FORWARD, `pi: ell -> lambda`, and its inverse -- used by the host seam and by
  //: nothing on the device (the device never converts; it is born in lattice order).
  static constexpr int digit_of(int l, long ell) {
    return (int)((ell / ipowl_c(B, D - 1 - l)) % B);
  }

  static_assert(kBC % 16 == 0, "K2: an atom must not straddle two owners");
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

}  // namespace rola::carry
