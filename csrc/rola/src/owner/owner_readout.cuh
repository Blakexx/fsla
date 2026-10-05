// csrc/rola/src/owner/owner_readout.cuh -- THE READOUT, as the skeleton prices it: a warp's read digits over a pass of
// the read pool, per digit or merged, each n-tile 8 tokens' read-dense rows by plain `ldmatrix` (leaves on K, tokens
// on N) against the digit's resident state (channels on M, two state digits), den by DP4A, the products added into the
// shared panel. The state's formation (the transfer) and the per-token factor are not built; the counts are real.
// see docs/internals/owner/owner_readout.md
#pragma once

#include <cstdint>

#include "common/ops.cuh"
#include "common/static_for.cuh"
#include "owner/owner.cuh"
#include "owner/owner_fill.cuh"
#include "owner/owner_lists.cuh"

namespace rola::owner::detail {

//: A WARP'S READOUT STATE: per read digit, the box's state as A -- two channel m-tiles by RD / 32 k-steps, two digits.
template <class C>
struct ReadState {
  static constexpr int kSteps = C::RD / 32;
  uint32_t a[C::kReadDigits][2][kSteps][2][4];
  uint32_t mass[kSteps][2];
};

//: THE PANEL a pass's products land in: a read-pool slot's 32 channels and its den, s32.
struct ReadPanel {
  uint32_t num[kReadPool][kChannels];
  uint32_t den[kReadPool];
};

//: THE ZERO ROW (and the panel's spill row) an n-tile's idle lanes address: the last row, which a pass of `kReadPass`
//: slots never fills.
template <class C>
__device__ __forceinline__ const uint8_t* read_zero_row(const ReadPool<C>& pools) {
  return pools.read[kReadPass];
}

//: AN N-TILE'S B: 8 tokens' read-dense rows, a k-step's 32 bytes a token, plain `ldmatrix` (lane `l` addresses token
//: `l % 8` at chunk `l / 8`).
template <class C>
__device__ __forceinline__ void load_reads(const uint8_t* row, int lane,
                                           uint32_t (&b)[ReadState<C>::kSteps][2]) {
  if constexpr (ReadState<C>::kSteps == 2) {
    uint32_t v[4];
    ops::load_frag(v, ops::smem_addr(row + 16 * (lane >> 3)));
    b[0][0] = v[0];
    b[0][1] = v[1];
    b[1][0] = v[2];
    b[1][1] = v[3];
  } else {
    uint32_t v[2];
    ops::load_frag2(v, ops::smem_addr(row + 16 * ((lane >> 3) & 1)));
    b[0][0] = v[0];
    b[0][1] = v[1];
  }
}

//: A DIGIT'S PRODUCTS FOR AN N-TILE: two channel m-tiles x the k-steps x two state digits of IMMAs, and the token's
//: den share by DP4A over the same B bytes against the leaves' masses.
//: @burst
template <class C>
__device__ __forceinline__ void digit_products(const ReadState<C>& rs, int digit,
                                               const uint32_t (&b)[ReadState<C>::kSteps][2],
                                               int32_t (&acc)[2][4], int& den) {
  constexpr int kSteps = ReadState<C>::kSteps;
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[m][i] = 0;
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int k = 0; k < kSteps; ++k) {
      ops::imma_u8s8_k32(acc[m], rs.a[digit][m][k][0], b[k][0], b[k][1]);
      ops::imma_u8_k32(acc[m], rs.a[digit][m][k][1], b[k][0], b[k][1]);
    }
  den = 0;
#pragma unroll
  for (int k = 0; k < kSteps; ++k) {
    den = __dp4a(static_cast<int>(b[k][0]), static_cast<int>(rs.mass[k][0]), den);
    den = __dp4a(static_cast<int>(b[k][1]), static_cast<int>(rs.mass[k][1]), den);
  }
}

//: THE TOKEN FACTOR: the n-tile's two tokens of this lane, digit `digit`'s read-sparse byte of their rows.
__device__ __forceinline__ int token_factor(uint32_t word, int digit) {
  return (word >> (8 * digit)) & 0xff;
}

//: AN N-TILE INTO THE PANEL: the products, each column weighted by its token's factor, summed over the digits it carries
//: (`total`), its den over the token's four lanes, then one shared add an element.
__device__ __forceinline__ void panel_add(ReadPanel& panel, const int32_t (&total)[2][4], int den,
                                          const int (&slot)[2], int lane) {
  den += __shfl_xor_sync(0xffffffffu, den, 1);
  den += __shfl_xor_sync(0xffffffffu, den, 2);
  const int g = lane >> 2, tq = lane & 3;
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int i = 0; i < 4; ++i)
      ops::red_shared_add_u32(ops::smem_addr(&panel.num[slot[i & 1]][16 * m + g + 8 * (i >> 1)]),
                              static_cast<uint32_t>(total[m][i]));
  if (tq == 0)
    ops::red_shared_add_u32(ops::smem_addr(&panel.den[slot[0]]), static_cast<uint32_t>(den));
}

//: A WARP'S READOUT OF ONE PASS (slots `[first, first + n)`), per its window form: per digit, each digit's entries in
//: the pass, 8 a tile, each tile added; merged (or a dense side), the warp union's, the read rows loaded once a tile,
//: every digit's weighted products summed in registers and the tile added once. -- see
//: docs/internals/owner/owner_readout.md#forms
template <class C>
__device__ void readout_pass(ReadPanel& panel, const ReadPool<C>& pools, const ListTenants<C>& sm,
                             const OwnerParams& p, const ReadState<C>& rs,
                             int (&cursor)[C::kReadDigits], int first, int n, int warp, int lane) {
  constexpr int kFactorAt = C::RS == 16 ? ReadRows<C>::kRun : C::RD;
  const int row0 = C::RD + warp * C::kReadDigits;
  const int limit = first + n;
  const int factor_byte = kFactorAt + warp * C::kReadDigits;
  uint32_t b[ReadState<C>::kSteps][2];
  int32_t acc[2][4], total[2][4];
  int den, den_total;

  const auto entry_at = [&](int e, int end, bool listed) -> int {
    return e < end ? (listed ? static_cast<int>(sm.read_entries[warp][e]) : e) - first : -1;
  };
  const auto tile = [&](int at, int end, bool listed, int(&slot)[2],
                        uint32_t(&factors)[2]) -> const uint8_t* {
    const int mine = entry_at(at + (lane & 7), end, listed);
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int s = entry_at(at + 2 * (lane & 3) + c, end, listed);
      slot[c] = s < 0 ? kReadPass : s;
      factors[c] = *reinterpret_cast<const uint32_t*>(pools.read[slot[c]] + (factor_byte & ~3))
                   >> (8 * (factor_byte & 3));
    }
    return mine < 0 ? read_zero_row(pools) : pools.read[mine];
  };
  const auto weigh = [&](int digit, const uint32_t(&factors)[2]) {
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int i = 0; i < 4; ++i) total[m][i] += acc[m][i] * token_factor(factors[i & 1], digit);
    den_total += den * token_factor(factors[0], digit);
  };

  const auto run_end = [&](int at, int count) {
    int end = at;
#pragma unroll 1
    while (true) {
      const int e = end + lane;
      const bool in = e < count && sm.read_entries[warp][e] < limit;
      const uint32_t mask = __ballot_sync(0xffffffffu, in);
      end += __popc(mask);
      if (mask != 0xffffffffu) return end;
    }
  };
  const auto clear = [&]() {
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int i = 0; i < 4; ++i) total[m][i] = 0;
    den_total = 0;
  };

  if (p.sparse_read != 0 && sm.read_form[warp] == kStepPerList) {
    int off = 0;
    rola::static_for<C::kReadDigits>([&](auto I) {
      constexpr int kI = decltype(I)::value;
      const int count = off + sm.count[row0 + kI];
      const int end = run_end(cursor[kI], count);
#pragma unroll 1
      for (int at = cursor[kI]; at < end; at += 8) {
        int slot[2];
        uint32_t factors[2];
        load_reads<C>(tile(at, end, true, slot, factors), lane, b);
        digit_products<C>(rs, kI, b, acc, den);
        clear();
        weigh(kI, factors);
        panel_add(panel, total, den_total, slot, lane);
      }
      cursor[kI] = end;
      off = count;
    });
    return;
  }

  const bool listed = p.sparse_read != 0;
  const int at0 = listed ? cursor[0] : first;
  const int end = listed ? run_end(at0, sm.read_total[warp]) : limit;
#pragma unroll 1
  for (int at = at0; at < end; at += 8) {
    int slot[2];
    uint32_t factors[2];
    load_reads<C>(tile(at, end, listed, slot, factors), lane, b);
    clear();
    rola::static_for<C::kReadDigits>([&](auto I) {
      constexpr int kI = decltype(I)::value;
      digit_products<C>(rs, kI, b, acc, den);
      weigh(kI, factors);
    });
    panel_add(panel, total, den_total, slot, lane);
  }
  cursor[0] = end;
}

}  // namespace rola::owner::detail
