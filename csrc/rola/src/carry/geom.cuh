// csrc/rola/src/carry/geom.cuh -- THE ADDRESSING BLOCK: one derivation, computed
// once per launch on the host, consumed by every kernel of the family.
//
// K50's axis law as G2b corrects it: the REGISTER SHAPE -- leaves per warp, `DV`, and
// the warp sub-box's spans and slot layout -- is compile-time `f(D, DV)`; the widths
// `B_l`, the per-side orders and the carve OFFSETS they fix are runtime and CTA-uniform.
// Symbols, the two span laws, the refusal set and the transit's row map:
// see docs/internals/carry/geom.md
#pragma once

#include <cstdint>

namespace rola::carry {

//: the deepest topology the block is built for: every per-level field's array bound.
constexpr int kGeomMaxLevels = 4;

//: THE PAGE GRANULE, here because the owner span law is CONSTRAINED by it.
static constexpr int kAtomLeaves = 16;

//: THE BIT COUNT OF A POWER OF TWO, and -1 otherwise -- a REFUSAL, never a rounding.
constexpr int geom_ilog2(int x) {
  int b = 0;
  if (x <= 0) return -1;
  while ((1 << b) < x) ++b;
  return (1 << b) == x ? b : -1;
}

//: THE REGISTER LAW (K50): what the register file admits at this value width.
constexpr int leaves_per_warp(int dv) { return 2048 / dv; }

//: A BALANCED BIT SPLIT over `D` levels, remainder INNERMOST-FIRST.
constexpr int balanced_bits(int bits, int D, int l) {
  return bits / D + (l >= D - bits % D ? 1 : 0);
}

//: THE WARP SUB-BOX -- REGISTER SHAPE, so compile-time `f(D, DV)`. -- see docs/internals/carry/geom.md#warp-box
constexpr int warp_span_bits(int D, int dv, int l) {
  return balanced_bits(geom_ilog2(leaves_per_warp(dv)), D, l);
}
constexpr int warp_span(int D, int dv, int l) { return 1 << warp_span_bits(D, dv, l); }

//: THE CANONICAL SLOT LAYOUT, THE SAME ON BOTH SIDES. -- see docs/internals/carry/geom.md#canonical-slots
constexpr int warp_shift(int D, int dv, int l) {
  int sh = 0;
  for (int i = l + 1; i < D; ++i) sh += warp_span_bits(D, dv, i);
  return sh;
}

//: THE OWNER BOX: `log2 BC` balanced, the ATOM held WHOLE innermost. -- see docs/internals/carry/geom.md#span-rule
constexpr int owner_span_bits(int D, int bc, int l) {
  const int total = geom_ilog2(bc), floor_in = geom_ilog2(kAtomLeaves);
  int b[kGeomMaxLevels] = {0, 0, 0, 0}, sum = 0;
  for (int i = 0; i < D; ++i) {
    b[i] = balanced_bits(total, D, i);
    if (i == D - 1 && b[i] < floor_in) b[i] = floor_in;
    sum += b[i];
  }
  while (sum > total) {
    int pick = -1;
    for (int i = 0; i < D; ++i) {
      const int fl = (i == D - 1) ? floor_in : 0;
      if (b[i] > fl && (pick < 0 || b[i] > b[pick])) pick = i;
    }
    if (pick < 0) break;
    --b[pick];
    --sum;
  }
  return b[l];
}
constexpr int owner_span(int D, int bc, int l) { return 1 << owner_span_bits(D, bc, l); }

//: the owner-local canonical index's bit offset for level `l`.
constexpr int owner_run_shift(int D, int bc, int l) {
  int sh = 0;
  for (int i = l + 1; i < D; ++i) sh += owner_span_bits(D, bc, i);
  return sh;
}

//: ONE SIDE'S CARVE, indexed BY LEVEL except `level_at`, which inverts the order.
struct SideGeom {
  int level_at[kGeomMaxLevels];    //: rank -> level (rank 0 is innermost)
  int rank[kGeomMaxLevels];        //: level -> rank
  int grid[kGeomMaxLevels];        //: level -> s(l) / sv(l), the stream grid
  int grid_div[kGeomMaxLevels];    //: level -> the stream index's weight at this level
  int row_prefix[kGeomMaxLevels];  //: level -> its first ballot row
  int carves;                      //: bit `l` set when this side carves level `l`
  int row_g[kGeomMaxLevels];       //: level -> ballot rows at that level, 0 when spanned whole
  int row_span[kGeomMaxLevels];    //: level -> the run one ballot row ORs, the rows pass's own
  int rows;                        //: this side's ballot rows in total
  int leaf;                        //: prod_l sv(l) = BC / streams
  int streams;                     //: prod_l grid(l)
  //: bit counts, never values: no consumer of this block emits an integer divide.
  int grid_div_bits[kGeomMaxLevels];
};

//: THE BLOCK ITSELF -- what the host hands the kernel, by value, once per launch.
struct CarryGeomRT {
  int D;
  int bc;
  int width[kGeomMaxLevels];      //: B_l
  int span[kGeomMaxLevels];       //: s(l), the compile-time owner law's own values
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

//: ONE SIDE'S CARVE; `NS == 1` is the S = 1 point of it. -- see docs/internals/carry/geom.md#carve
inline const char* derive_side(int D, const int* width, const int* span, const int* sub,
                               uint32_t MO, uint32_t side, int NS, SideGeom& g) {
  if (NS < 1 || (NS & (NS - 1)) != 0) return "a side's stream count is a power of two";
  for (int l = 0; l < D; ++l) {
    int r = 0;
    for (int i = 0; i < D; ++i)
      if (geom_detail::mode_before(MO, side, i, l)) ++r;
    g.rank[l] = r;
    g.level_at[r] = l;
  }
  g.streams = 1;
  g.leaf = 1;
  g.carves = 0;
  for (int l = 0; l < D; ++l) {
    g.grid[l] = NS == 1 ? 1 : span[l] / sub[l];
    g.row_span[l] = g.grid[l] == 0 ? 0 : span[l] / g.grid[l];
    if (g.grid[l] < 1 || g.grid[l] * g.row_span[l] != span[l])
      return "the warp sub-box does not divide the owner's box at some level";
    g.streams *= g.grid[l];
    g.leaf *= g.row_span[l];
    if (g.grid[l] > 1) g.carves |= 1 << l;
  }
  if (g.streams != NS) return "the stream grid does not partition the owner's box";
  //: THE STREAM INDEX IS READ IN THE SIDE'S OWN ORDER: rank 0 varies fastest.
  int gd = 1;
  for (int r = 0; r < D; ++r) {
    const int l = g.level_at[r];
    g.grid_div[l] = gd;
    g.grid_div_bits[l] = geom_ilog2(gd);
    if (g.grid_div_bits[l] < 0) return "a side's stream grid must be a power of two";
    gd *= g.grid[l];
  }
  //: THE BALLOT ROWS at this side's own grain: a level spanned whole votes none.
  int pfx = 0;
  for (int l = 0; l < D; ++l) {
    g.row_g[l] = g.row_span[l] < width[l] ? width[l] / g.row_span[l] : 0;
    g.row_prefix[l] = pfx;
    pfx += g.row_g[l];
  }
  g.rows = pfx;
  for (int l = D; l < kGeomMaxLevels; ++l) {
    g.rank[l] = l;
    g.level_at[l] = l;
    g.grid[l] = 1;
    g.grid_div[l] = 1;
    g.grid_div_bits[l] = 0;
    g.row_span[l] = width[0];
    g.row_g[l] = 0;
    g.row_prefix[l] = pfx;
  }
  return nullptr;
}

//: THE ONE DERIVATION; returns nullptr or the REFUSAL's own words. -- see docs/internals/carry/geom.md#refusals
inline const char* derive_carry_geom(int D, const int* width, int bc, int dv, uint32_t MO, int NSR,
                                     int NSW, CarryGeomRT& g) {
  if (D < 1 || D > kGeomMaxLevels) return "the addressing block is built to depth four";
  if (geom_ilog2(bc) < 0) return "BC must be a power of two";
  if (geom_ilog2(dv) < 0 || dv % 16 != 0) return "the value width is a power of two, sixteen up";
  g.D = D;
  g.bc = bc;
  int sub[kGeomMaxLevels] = {1, 1, 1, 1};
  for (int l = 0; l < D; ++l) {
    g.width[l] = width[l];
    g.span[l] = owner_span(D, bc, l);
    sub[l] = warp_span(D, dv, l);
    if (geom_ilog2(g.width[l]) < 0) return "every level width must be a power of two";
    //: THE ONE WIDTH LAW THIS SHAPE OWNS -- there is no narrower case to fall back to.
    if (g.width[l] < g.span[l]) return "a level is narrower than the owner box's span there";
  }
  long leaves = 1, owners = 1;
  int wtot = 0, local_bits = 0, min_width = width[0];
  for (int l = 0; l < D; ++l) {
    g.grid[l] = g.width[l] / g.span[l];
    g.col_base[l] = wtot;
    wtot += g.width[l];
    leaves *= g.width[l];
    owners *= g.grid[l];
    local_bits += owner_span_bits(D, bc, l);
    if (g.width[l] < min_width) min_width = g.width[l];
  }
  if (local_bits != geom_ilog2(bc)) return "the owner box's spans do not multiply to BC";
  if (leaves > (long)0x40000000) return "the topology's leaf count does not fit the id space";
  g.wtot = wtot;
  g.owners = (int)owners;
  g.leaves = (int)leaves;
  g.local_bits = local_bits;
  g.min_width = min_width;
  int wgt = 1, odiv = 1;
  for (int l = D - 1; l >= 0; --l) {
    g.weight[l] = wgt;
    g.owner_div[l] = odiv;
    g.owner_div_bits[l] = geom_ilog2(odiv);
    g.grid_bits[l] = geom_ilog2(g.grid[l]);
    g.span_bits[l] = owner_span_bits(D, bc, l);
    g.run_shift[l] = owner_run_shift(D, bc, l);
    if (g.owner_div_bits[l] < 0 || g.grid_bits[l] < 0)
      return "the owner grid must be a power of two";
    wgt *= g.width[l];
    odiv *= g.grid[l];
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
  if (const char* e = derive_side(D, g.width, g.span, sub, MO, kSideRead, NSR, g.r)) return e;
  if (const char* e = derive_side(D, g.width, g.span, sub, MO, kSideWrite, NSW, g.w)) return e;
  //: a level nobody carves fixes no digit, so its index weight is not part of the map.
  int identical = 1;
  for (int l = 0; l < D; ++l)
    if (g.r.grid[l] > 1 && g.r.grid_div_bits[l] != g.w.grid_div_bits[l]) identical = 0;
  g.identity_map = identical;
  return nullptr;
}

//: THE TRANSIT'S ROW MAP: one per-warp base word, the IDENTITY on the slot bits.
//: -- see docs/internals/carry/geom.md#row-map
__host__ __device__ inline int geom_xch_reader(const CarryGeomRT& g, int w) {
  int reader = 0;
#pragma unroll 1
  for (int l = 0; l < g.D; ++l)
    reader += (((w >> g.w.grid_div_bits[l]) & (g.w.grid[l] - 1)) << g.r.grid_div_bits[l]);
  return reader;
}

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
