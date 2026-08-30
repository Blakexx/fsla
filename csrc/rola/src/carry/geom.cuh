// csrc/rola/src/carry/geom.cuh -- THE ADDRESSING BLOCK: one derivation, computed
// once per launch on the host, consumed by every kernel of the family.
//
// K50's axis law: SHAPE stays compile-time because it sizes registers and shared
// memory; ADDRESSING becomes runtime because it does not, and every runtime value
// here is CTA-UNIFORM.  Symbols, the span rule and its atom invariant, the refusal
// set, and the transit's row map: see docs/internals/carry/geom.md
#pragma once

#include <cstdint>

namespace rola::carry {

//: the deepest topology the block is built for: every per-level field's array bound.
constexpr int kGeomMaxLevels = 4;

//: THE PAGE GRANULE, here because the span rule is CONSTRAINED by it.
static constexpr int kAtomLeaves = 16;

//: THE REGISTER LAW (K50): a warp owns two `m`-tiles (the `ldmatrix` granule).
constexpr int kLeavesPerWarp = 32;

//: THE BIT COUNT OF A POWER OF TWO, and -1 otherwise -- a REFUSAL, never a rounding.
constexpr int geom_ilog2(int x) {
  int b = 0;
  if (x <= 0) return -1;
  while ((1 << b) < x) ++b;
  return (1 << b) == x ? b : -1;
}

//: ONE SIDE'S CARVE, indexed BY LEVEL except `level_at`, which inverts the order.
struct SideGeom {
  int level_at[kGeomMaxLevels];    //: rank -> level (rank 0 is innermost)
  int rank[kGeomMaxLevels];        //: level -> rank
  int span[kGeomMaxLevels];        //: level -> sv(l)
  int grid[kGeomMaxLevels];        //: level -> s(l) / sv(l), the stream grid
  int grid_div[kGeomMaxLevels];    //: level -> prod_{l' > l} grid(l')
  int shift[kGeomMaxLevels];       //: level -> the private slot's bit offset
  int row_prefix[kGeomMaxLevels];  //: level -> its first ballot row
  int carves;                      //: bit `l` set when this side carves level `l`
  int row_g[kGeomMaxLevels];       //: level -> ballot rows at that level, 0 when spanned whole
  int rows;                        //: this side's ballot rows in total
  int leaf;                        //: prod_l sv(l) = BC / streams
  int streams;                     //: prod_l grid(l)
  //: EVERY DIVISOR AND MODULUS IN THIS FAMILY IS A POWER OF TWO, so the block carries
  //: the BIT COUNTS and no consumer ever emits an integer divide.
  int span_bits[kGeomMaxLevels];
  int grid_div_bits[kGeomMaxLevels];
};

//: THE BLOCK ITSELF -- what the host hands the kernel, by value, once per launch.
struct CarryGeomRT {
  int D;
  int bc;
  int width[kGeomMaxLevels];      //: B_l
  int span[kGeomMaxLevels];       //: s(l)
  int grid[kGeomMaxLevels];       //: g(l) = B_l / s(l)
  int col_base[kGeomMaxLevels];   //: the prefix sum of the widths
  int run_shift[kGeomMaxLevels];  //: sum_{l' > l} log2 s(l')
  int owner_div[kGeomMaxLevels];  //: prod_{l' > l} g(l')
  int weight[kGeomMaxLevels];     //: prod_{l' > l} B_l', the canonical mixed radix
  int wtot;                       //: sum_l B_l -- the packed amplitude row's stride
  int owners;                     //: prod_l g(l)
  int leaves;                     //: prod_l B_l
  int local_bits;                 //: log2 BC
  int min_width;                  //: min_l B_l -- a CENSUS FACT, never a refusal
  SideGeom r;
  SideGeom w;
  //: the former `kTied`, now a REPORTED FACT -- see docs/internals/carry/geom.md#row-map
  int identity_map;
  int span_bits[kGeomMaxLevels];       //: log2 s(l)
  int grid_bits[kGeomMaxLevels];       //: log2 g(l)
  int owner_div_bits[kGeomMaxLevels];  //: log2 owner_div(l)
};

//: THE MODE WORD'S ENCODING, as the seam spells it: two bits per level.
constexpr uint32_t kSideRead = 1u, kSideWrite = 2u;
constexpr bool mode_declares(uint32_t modes, int l, uint32_t side) {
  return ((modes >> (2 * l)) & side) != 0u;
}

namespace geom_detail {

//: THE ORDER BY MODE (P76), unchanged in law, on a runtime word.
inline int mode_group(uint32_t MO, int l, uint32_t side) {
  const bool rd = mode_declares(MO, l, kSideRead), wr = mode_declares(MO, l, kSideWrite);
  if (rd && wr) return 2;                     // both-sparse
  return mode_declares(MO, l, side) ? 1 : 0;  // sparse : dense
}

inline bool mode_before(uint32_t MO, uint32_t side, int a, int b) {
  const int ga = mode_group(MO, a, side), gb = mode_group(MO, b, side);
  return ga != gb ? ga < gb : a > b;
}

}  // namespace geom_detail

//: THE SPAN RULE (K50): balance `BC` over the levels, SUBJECT TO "a box is a union
//: of whole atoms".  Balance alone splits an atom across owners at `D = 3`/`D = 4`,
//: -- see docs/internals/carry/geom.md#span-rule
inline const char* derive_spans(int D, const int* width, int bc, int* span) {
  const int total = geom_ilog2(bc);
  if (total < 0) return "BC must be a power of two";
  if (D < 1 || D > kGeomMaxLevels) return "the addressing block is built to depth four";
  int cap[kGeomMaxLevels], bits[kGeomMaxLevels], floor_bits[kGeomMaxLevels];
  int room = 0;
  for (int l = 0; l < D; ++l) {
    cap[l] = geom_ilog2(width[l]);
    if (cap[l] < 0) return "every level width must be a power of two";
    floor_bits[l] = 0;
    room += cap[l];
  }
  if (room < total) return "the topology holds fewer leaves than one owner's box";
  //: the levels a page granule spans must be held WHOLE by the box.
  int need = geom_ilog2(kAtomLeaves);
  for (int l = D - 1; l >= 0 && need > 0; --l) {
    floor_bits[l] = cap[l] < need ? cap[l] : need;
    need -= floor_bits[l];
  }
  if (need > 0) return "the topology holds fewer leaves than one page atom";
  //: BALANCE, remainder INNERMOST-FIRST -- what reproduces `[8, 16]` at `D = 2`.
  const int base = total / D, rem = total % D;
  int sum = 0;
  for (int l = 0; l < D; ++l) {
    bits[l] = base + (l >= D - rem ? 1 : 0);
    if (bits[l] < floor_bits[l]) bits[l] = floor_bits[l];
    if (bits[l] > cap[l]) bits[l] = cap[l];
    sum += bits[l];
  }
  //: surplus back from the WIDEST level above its floor (ties OUTERMOST); shortfall in.
  while (sum > total) {
    int pick = -1;
    for (int l = 0; l < D; ++l)
      if (bits[l] > floor_bits[l] && (pick < 0 || bits[l] > bits[pick])) pick = l;
    if (pick < 0) return "the page atom and the level widths cannot both fit this box";
    --bits[pick];
    --sum;
  }
  while (sum < total) {
    int pick = -1;
    for (int l = D - 1; l >= 0; --l)
      if (bits[l] < cap[l]) {
        pick = l;
        break;
      }
    if (pick < 0) return "the topology holds fewer leaves than one owner's box";
    ++bits[pick];
    ++sum;
  }
  for (int l = 0; l < D; ++l) span[l] = 1 << bits[l];
  return nullptr;
}

//: ONE SIDE'S CARVE, derived from the order and a stream count: walk the levels
//: OUTERMOST-FIRST and take each one's whole span as stream grid until the count is
//: met (`StreamSpans::g_at`, unchanged in law).
inline const char* derive_side(int D, const int* width, const int* span, uint32_t MO, uint32_t side,
                               int NS, SideGeom& g) {
  if (NS < 1 || (NS & (NS - 1)) != 0) return "a side's stream count is a power of two";
  for (int l = 0; l < D; ++l) {
    int r = 0;
    for (int i = 0; i < D; ++i)
      if (geom_detail::mode_before(MO, side, i, l)) ++r;
    g.rank[l] = r;
    g.level_at[r] = l;
  }
  int need = NS;
  for (int p = 0; p < D; ++p) {
    const int l = g.level_at[D - 1 - p];  // the level at OUTER position `p`
    const int take = span[l] < need ? span[l] : need;
    g.grid[l] = take;
    g.span[l] = span[l] / take;
    need /= take;
  }
  if (need != 1) return "the stream count is not reachable by carving this side's levels";
  g.streams = 1;
  g.leaf = 1;
  g.carves = 0;
  for (int l = 0; l < D; ++l) {
    g.streams *= g.grid[l];
    g.leaf *= g.span[l];
    if (g.span[l] < span[l]) g.carves |= 1 << l;
  }
  if (g.streams != NS) return "the stream grid does not partition the owner's box";
  int gd = 1;
  for (int l = D - 1; l >= 0; --l) {
    g.grid_div[l] = gd;
    g.grid_div_bits[l] = geom_ilog2(gd);
    gd *= g.grid[l];
  }
  for (int l = 0; l < D; ++l) {
    g.span_bits[l] = geom_ilog2(g.span[l]);
    if (g.span_bits[l] < 0 || g.grid_div_bits[l] < 0)
      return "a side's spans and stream grid must be powers of two";
  }
  const int leaf_bits = geom_ilog2(g.leaf);
  for (int l = 0; l < D; ++l) {
    int sh = 0;
    for (int i = 0; i < D; ++i)
      if (g.rank[i] < g.rank[l]) sh += geom_ilog2(g.span[i]);
    g.shift[l] = sh;
    //: LAYER 3 of the structure branch's enforcement -- see
    //: docs/internals/carry/carry_kernel.md#expand-cases
    if (sh < 0 || geom_ilog2(g.span[l]) < 0 || sh + geom_ilog2(g.span[l]) > leaf_bits)
      return "a level's (span, shift) field is outside the body's admissible set";
  }
  //: THE BALLOT ROWS at this side's own grain (`CarveRows`): a level the stream
  //: spans whole votes no rows, and the rest carry one row per digit run.
  int pfx = 0;
  for (int l = 0; l < D; ++l) {
    g.row_g[l] = g.span[l] < width[l] ? width[l] / g.span[l] : 0;
    g.row_prefix[l] = pfx;
    pfx += g.row_g[l];
  }
  g.rows = pfx;
  for (int l = D; l < kGeomMaxLevels; ++l) {
    g.rank[l] = l;
    g.level_at[l] = l;
    g.span[l] = width[0];
    g.grid[l] = 1;
    g.grid_div[l] = 1;
    g.grid_div_bits[l] = 0;
    g.span_bits[l] = geom_ilog2(width[0]);
    g.shift[l] = 0;
    g.row_g[l] = 0;
    g.row_prefix[l] = pfx;
  }
  return nullptr;
}

//: THE ONE DERIVATION.  `bc` is the SHAPE the binary was compiled for; returns
//: nullptr, or the REFUSAL's own words, which the seam turns into its message.
inline const char* derive_carry_geom(int D, const int* width, int bc, uint32_t MO, int NSR, int NSW,
                                     CarryGeomRT& g) {
  if (D < 1 || D > kGeomMaxLevels) return "the addressing block is built to depth four";
  g.D = D;
  g.bc = bc;
  for (int l = 0; l < D; ++l) g.width[l] = width[l];
  if (const char* e = derive_spans(D, width, bc, g.span)) return e;
  long leaves = 1, owners = 1;
  int wtot = 0, local_bits = 0, min_width = width[0];
  for (int l = 0; l < D; ++l) {
    g.grid[l] = g.width[l] / g.span[l];
    g.col_base[l] = wtot;
    wtot += g.width[l];
    leaves *= g.width[l];
    owners *= g.grid[l];
    local_bits += geom_ilog2(g.span[l]);
    if (g.width[l] < min_width) min_width = g.width[l];
  }
  if (leaves > (long)0x40000000) return "the topology's leaf count does not fit the id space";
  g.wtot = wtot;
  g.owners = (int)owners;
  g.leaves = (int)leaves;
  g.local_bits = local_bits;
  g.min_width = min_width;
  int wgt = 1, odiv = 1, rsh = 0;
  for (int l = D - 1; l >= 0; --l) {
    g.weight[l] = wgt;
    g.owner_div[l] = odiv;
    g.owner_div_bits[l] = geom_ilog2(odiv);
    g.grid_bits[l] = geom_ilog2(g.grid[l]);
    g.span_bits[l] = geom_ilog2(g.span[l]);
    g.run_shift[l] = rsh;
    if (g.owner_div_bits[l] < 0 || g.grid_bits[l] < 0 || g.span_bits[l] < 0)
      return "the owner grid and span must be powers of two";
    wgt *= g.width[l];
    odiv *= g.grid[l];
    rsh += g.span_bits[l];
  }
  for (int l = D; l < kGeomMaxLevels; ++l) {
    g.width[l] = width[0];
    g.span[l] = width[0];
    g.grid[l] = 1;
    g.col_base[l] = wtot;
    g.weight[l] = 1;
    g.owner_div[l] = 1;
    g.owner_div_bits[l] = 0;
    g.grid_bits[l] = 0;
    g.span_bits[l] = geom_ilog2(width[0]);
    g.run_shift[l] = 0;
  }
  if (const char* e = derive_side(D, g.width, g.span, MO, kSideRead, NSR, g.r)) return e;
  if (const char* e = derive_side(D, g.width, g.span, MO, kSideWrite, NSW, g.w)) return e;
  int identical = 1;
  for (int l = 0; l < D; ++l)
    if (g.r.span[l] != g.w.span[l] || g.r.rank[l] != g.w.rank[l]) identical = 0;
  g.identity_map = identical;
  return nullptr;
}

//: THE TRANSIT'S ROW MAP (K35 F1), as arithmetic on the block.
//: -- see docs/internals/carry/geom.md#row-map
__host__ __device__ inline int geom_xch_row(const CarryGeomRT& g, int leaf_r, int w, int P) {
  int reader = 0, slot = 0;
#pragma unroll 1
  for (int l = 0; l < g.D; ++l) {
    const int digit = ((P >> g.w.shift[l]) & (g.w.span[l] - 1))
                      + (((w >> g.w.grid_div_bits[l]) & (g.w.grid[l] - 1)) << g.w.span_bits[l]);
    reader += (digit >> g.r.span_bits[l]) << g.r.grid_div_bits[l];
    slot += (digit & (g.r.span[l] - 1)) << g.r.shift[l];
  }
  return reader * leaf_r + slot;
}

//: the two swizzles, GF(2)-linear in the row so `swz(base ^ delta)` splits into a
//: per-lane term and a per-fragment one.  Unchanged from `XchMap`.
__host__ __device__ constexpr int geom_swz(int r) { return ((r >> 3) ^ (r >> 4) ^ (r >> 6)) & 7; }
__host__ __device__ constexpr int geom_msw(int r) { return ((r >> 3) ^ (r >> 5)) & 3; }

//: THE CANONICAL LEAF of owner `O`'s local slot, and the atom that holds it (P80).
__host__ __device__ inline int geom_canon_leaf(const CarryGeomRT& g, int O, int local) {
  int acc = 0;
#pragma unroll 1
  for (int l = 0; l < g.D; ++l) {
    const int d = (((O >> g.owner_div_bits[l]) & (g.grid[l] - 1)) << g.span_bits[l])
                  + ((local >> g.run_shift[l]) & (g.span[l] - 1));
    acc += d * g.weight[l];
  }
  return acc;
}

}  // namespace rola::carry
