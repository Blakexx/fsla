// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// THE LATTICE'S LAW (docs/internals/common/plan/lattice.md): the `(k, m)` owner box of a hierarchy's level widths, the search
// that picks it and the atom's rectangle in it; pure and constexpr, the one statement of the law for every reader.
#pragma once

#include <cstdint>
#include <string>

namespace fsla::plan {

//: the leaves one page atom holds, and its log2 (the walk's page granule)
inline constexpr int kLatticeAtomLeaves = 16;
inline constexpr int kLatticeAtomShift = 4;
static_assert(1 << kLatticeAtomShift == kLatticeAtomLeaves,
              "the atom's shift is log2 of its leaves");

//: THE CEILING on the leaves one owner holds (`BC`), the one number of the law that is a choice rather than a consequence
inline constexpr int kLatticeCapBc = 128;

//: the widest owner span a level may take: the span the device reads as one 32-bit slice
inline constexpr int kLatticeMaxSpan = 32;

//: the largest uniform sub-box span `k` the search tries
inline constexpr int kLatticeMaxSpanK = 4;

enum class BoxRefusal {
  kNone,
  kSpanNotPowerOfTwo,
  kCapacityNotPowerOfTwo,
  kDoesNotFit,
  kInsideLevel,
};

//: the `(k, m)` box on `L` or fewer levels: the owner's span `s[l] = k * m_l` and grid `g[l] = width / s[l]` a level, the leaves
//: one owner holds `bc = prod s` and the owners `owners = prod g`; `refusal` names why no box exists, `level` the level an owner
//: sits inside, `carried` the exponent of `m` the levels can carry
template <int L>
struct LatticeBox {
  int levels;
  int k;
  std::int64_t m;
  BoxRefusal refusal;
  int level;
  int carried;
  std::int64_t s[L];
  std::int64_t g[L];
  std::int64_t bc;
  std::int64_t owners;
};

//: the trailing levels whose spans multiply into an atom, taken whole, and the digits of the next level out that complete it
struct AtomRectangle {
  int whole;
  int partial;
};

constexpr bool lattice_power_of_two(std::int64_t x) { return x >= 1 && (x & (x - 1)) == 0; }

constexpr int lattice_log2(std::int64_t x) {
  int e = 0;
  while ((std::int64_t{2} << e) <= x) ++e;
  return e;
}

//: THE BOX of `(k, m)` on `widths[0 .. levels)`: the innermost level takes as much of `m` as it can carry (never less than its
//: balanced share, never less than an atom asks, never more than its width admits) and the rest is spread over the outer levels,
//: innermost first, so their exponents differ by at most one
template <int L>
constexpr LatticeBox<L> lattice_box(const int* widths, int levels, int k, std::int64_t m) {
  LatticeBox<L> box{levels, k, m, BoxRefusal::kNone, 0, 0, {}, {}, 1, 1};
  if (!lattice_power_of_two(m)) {
    box.refusal = BoxRefusal::kCapacityNotPowerOfTwo;
    return box;
  }
  if (!lattice_power_of_two(k)) {
    box.refusal = BoxRefusal::kSpanNotPowerOfTwo;
    return box;
  }

  const int j = lattice_log2(m);
  const int kappa = lattice_log2(k);
  const int floor_inner = kLatticeAtomShift > kappa ? kLatticeAtomShift - kappa : 0;

  int ceil_inner = 0;
  while ((std::int64_t{k} << (ceil_inner + 1)) <= widths[levels - 1]
         && widths[levels - 1] % (std::int64_t{k} << (ceil_inner + 1)) == 0) {
    ++ceil_inner;
  }

  const int share = j / levels + ((j % levels) ? 1 : 0);
  int a_inner = share > floor_inner ? share : floor_inner;
  if (a_inner > ceil_inner) a_inner = ceil_inner;
  if (a_inner > j) a_inner = j;
  const int j_outer = j - a_inner;

  int a[L] = {};
  int carried = 0;
  for (int l = 0; l < levels; ++l) {
    a[l] = (l == levels - 1)
               ? a_inner
               : j_outer / (levels - 1) + ((l >= (levels - 1) - (j_outer % (levels - 1))) ? 1 : 0);
    carried += a[l];
  }

  if (carried != j) {
    box.refusal = BoxRefusal::kDoesNotFit;
    box.carried = carried;
    return box;
  }

  for (int l = 0; l < levels; ++l) {
    box.s[l] = std::int64_t{k} << a[l];
    if (widths[l] % box.s[l] != 0) {
      box.refusal = BoxRefusal::kInsideLevel;
      box.level = l;
      return box;
    }
    box.g[l] = widths[l] / box.s[l];
    box.bc *= box.s[l];
    box.owners *= box.g[l];
  }
  return box;
}

//: the atom's rectangle in a span vector
constexpr AtomRectangle lattice_atom_rectangle(const std::int64_t* spans, int levels) {
  int whole = 0;
  std::int64_t prod = 1;
  for (int l = levels - 1; l >= 0; --l) {
    if (prod * spans[l] > kLatticeAtomLeaves) break;
    prod *= spans[l];
    ++whole;
  }
  return {whole, static_cast<int>(kLatticeAtomLeaves / prod)};
}

//: whether a box sits inside the ceiling on `BC` and the span limit of every level
template <int L>
constexpr bool lattice_admissible(const LatticeBox<L>& box) {
  if (box.refusal != BoxRefusal::kNone || box.bc > kLatticeCapBc) return false;
  for (int l = 0; l < box.levels; ++l) {
    if (box.s[l] > kLatticeMaxSpan) return false;
  }
  return true;
}

//: THE SEARCH: over `k` in the powers of two up to `kLatticeMaxSpanK` and `m` in the powers of two up to the leaves, the admissible
//: box that MAXIMIZES `BC`, then has the SHALLOWEST atom rectangle (an atom spanning two levels is filled only when both are dense
//: for the side), then the LARGEST `k`; the `(1, 1)` box when none is admissible
template <int L>
constexpr LatticeBox<L> lattice_solve(const int* widths, int levels) {
  std::int64_t leaves = 1;
  for (int l = 0; l < levels; ++l) leaves *= widths[l];
  LatticeBox<L> best = lattice_box<L>(widths, levels, 1, 1);
  std::int64_t best_bc = 0;
  int best_atoms = 0;
  int best_k = 0;
  for (int k = 1; k <= kLatticeMaxSpanK; k <<= 1) {
    for (std::int64_t m = 1; m <= leaves; m <<= 1) {
      const LatticeBox<L> box = lattice_box<L>(widths, levels, k, m);
      if (!lattice_admissible(box)) continue;
      const AtomRectangle r = lattice_atom_rectangle(box.s, levels);
      const int atoms = r.whole + (r.partial > 1 ? 1 : 0);
      bool better = k > best_k;
      if (atoms != best_atoms) better = atoms < best_atoms;
      if (box.bc != best_bc) better = box.bc > best_bc;
      if (better) {
        best = box;
        best_bc = box.bc;
        best_atoms = atoms;
        best_k = k;
      }
    }
  }
  return best;
}

//: the words of a refused box
template <int L>
inline std::string lattice_refusal(const LatticeBox<L>& box, const int* widths) {
  std::string shape = "(";
  for (int l = 0; l < box.levels; ++l) shape += (l ? ", " : "") + std::to_string(widths[l]);
  shape += box.levels == 1 ? ",)" : ")";
  switch (box.refusal) {
    case BoxRefusal::kCapacityNotPowerOfTwo:
      return "the lattice capacity m must be a power of two, got " + std::to_string(box.m);
    case BoxRefusal::kSpanNotPowerOfTwo:
      return "the lattice span k must be a power of two, got " + std::to_string(box.k);
    case BoxRefusal::kDoesNotFit:
      return "the capacity m = " + std::to_string(box.m) + " does not fit widths " + shape
             + " at k = " + std::to_string(box.k) + ": the levels can carry 2^"
             + std::to_string(box.carried) + " (an owner sits inside a level)";
    case BoxRefusal::kInsideLevel:
      return "level " + std::to_string(box.level) + " of width " + std::to_string(widths[box.level])
             + " is not a whole number of owner spans k*m_l = " + std::to_string(box.s[box.level])
             + " (an owner sits inside a level)";
    case BoxRefusal::kNone:
      break;
  }
  return "";
}

}  // namespace fsla::plan
