// csrc/rola/src/chunk/ladder_r0.cuh -- THE BACKWARD LADDER, RUNGS 0-4: the chunked
// linear-attention backward on ROUTED coefficients, written in RoLA's own
// quantities as a two-pass pair.  Pass 1 ascends, rebuilding the state and its mass
// column from zero and emitting the READ side's per-level digit gradient; pass 2
// descends, carrying both adjoints and emitting the value cotangent and the WRITE
// side's.  The slot space is carved into blocks; `dv` and both digit gradients fan
// in across them.  The operands are PER LEVEL and the Kronecker product is formed
// in kernel.  The token axis is COMPACTED: both passes REPLAY the saved union table
// and visit only the tokens a block's clause admits.  No write gain, no ragged
// tail -- those are later rungs.  docs/internals/chunk/ladder_r0.md
#pragma once

#include "backward.cuh"

namespace rola::chunk {

//: The ladder's launch arguments.  The coefficients are PER LEVEL -- `[BH, L, D*Bw]`
//: amplitude rows, one packed block of `Bw` columns per level -- and so are the two
//: gradients they carry.  The slot space `N = Bw^D` is a COMPILED quantity, not a
//: runtime width: the block lattice `N / BC` is the grid's x extent and the digit
//: structure a block sees is the geometry it is instantiated at.  `tbl` is the
//: SAVED union table, the forward's one retained byproduct: rung b4 REPLAYS it and
//: never rebuilds it -- docs/internals/chunk/ladder_r0.md#the-replay
struct Rung0Params {
  const __nv_bfloat16 *pr, *pw, *v, *dnum, *dden;
  const uint32_t* tbl;
  float *dpr, *dv, *dpw;
  int BH, L, v_row_bytes;
};

//: THE RUNG'S LIVE REGISTER SET, one measured DECLARATION for both passes and
//: falsified per arm by the census' spill and budget columns.  It is a single number
//: rather than a `BC` row because a GROUP's geometry does not move with the block
//: width; the pass that peaks is the descending one, which carries the adjoint state
//: through every block.  RE-MEASURED AT b4 AND MOVED, 176 -> 192, and the move is
//: DECLARED rather than absorbed: the replay adds the walk's cursors and the
//: schedule's clause columns to every visit, and the single-buffered panel adds the
//: expansion's carried run (`passes * 4` registers per panel, two of them in the
//: descending pass).  Peak 190 at `(32, 128, 2, 256)` pass 2, zero spill at all
//: fourteen.  The move costs NO residency: at 128 threads a floor of 192 still
//: admits two CTAs and at 256 threads one, which is eight warps per SM either way --
//: the same eight 176 bought, and the same `cta_target` this declaration derives.
//: The floor's LOWER bound is b2's: below 171 the `(16, 64)` arm derives a third
//: CTA, which is a residency decision and not the side effect of a re-measurement.
constexpr int reg_floor_r0() { return 192; }

//: THE LADDER'S OWN GROUP THRESHOLD, and it is LOWER than the forward's (which
//: splits at `BC = 256`, `groups_for`).  The adjoint carries two costs the forward
//: does not: a SECOND coefficient panel, and the PUBLICATION -- the `[d_v, BCg]`
//: transpose of the state that lets a unit finish a contraction over the carved
//: channel axis.  Both scale with the group's slot width, and at `BCg = 128` their
//: sum puts the carve past the two-CTA ceiling with no overlay available, because
//: all four operand buffers are live across the whole visit.  Splitting the BLOCK
//: into two 64-slot groups holds the carve at the width that fits and keeps the
//: warps per SM at the forward's eight -- one CTA of eight warps where `BC = 64`
//: runs two of four -- while the value ring and the channel Gram, which are
//: slot-independent, are shared rather than duplicated.
//: docs/internals/chunk/ladder_r0.md#the-group-threshold
constexpr int r0_groups_for(int BC) { return BC >= 128 ? 2 : 1; }
constexpr int r0_threads_for(int BC) { return kGroupThreads * r0_groups_for(BC); }

//: THE LADDER'S GEOMETRY.  What rides the RING is what arrives over time and
//: asynchronously: the two value-shaped panels, the readout's mass seed inside the
//: `dnum` panel's own row pad (which costs no bytes at all), and the SCHEDULE's own
//: index slot.  What does not ride it is the coefficient panel: rung b3 replaced
//: b2's `cp.async` staging with an in-kernel expansion, and an expansion's latency
//: is not the ring's to hide -- it is covered by splitting the expansion into a LOAD
//: issued before the visit's rendezvous and a STORE issued after it, which is what
//: gives rung b4 the bytes the index and form rings need
//: (docs/internals/chunk/ladder_r0.md#the-panel-leaves-the-ring).  The published
//: state's rows are UNPADDED -- it is read four times per unit per visit, where the
//: pad would buy nothing and cost the CTA target.  The GROUP is the unit of
//: coefficient, Gram and publication geometry; the value ring, the `d_v` Gram and
//: the whole schedule are CTA-wide.
template <int C, int BC, bool PASS2>
struct GeomR0 {
  static constexpr int GRP = r0_groups_for(BC);
  static constexpr int BCg = BC / GRP;
  static constexpr int LdV = kD + kPadE;
  static constexpr int LdS = BCg + kPadE;
  static constexpr int LdG = C + kPadE;
  static constexpr int LdDs = BCg;
  //: the SLOT Gram is one per group (it contracts over the group's own slots); the
  //: channel Gram is one per CTA (it has no slot axis at all).
  static constexpr int slot_grams = PASS2 ? 1 : 0;
  static constexpr int coef_panels = PASS2 ? 2 : 1;
  //: THE SCHEDULE'S SLOT, the shipped scans' own shape: a 16-byte header, the two
  //: `uint16` index lists and the `uint8` causal threshold, one set per cooked
  //: super-chunk -- docs/internals/chunk/ladder_r0.md#the-replay
  static constexpr int hdr = 16;
  static constexpr int idx_slot = hdr + ((5 * C + 15) / 16) * 16;
  static constexpr int form_bytes =
      2 * kFormWords * (int)sizeof(uint32_t) + kCookSlots * (int)sizeof(int);
  static constexpr int panel_bytes = GRP * coef_panels * C * LdS * (int)sizeof(__nv_bfloat16);
  static constexpr int slot = 2 * C * LdV * (int)sizeof(__nv_bfloat16) + idx_slot;
  static constexpr int gram_bytes = (GRP * slot_grams + 1) * C * LdG * (int)sizeof(__nv_bfloat16);
  static constexpr int ds_bytes = GRP * kD * LdDs * (int)sizeof(__nv_bfloat16);
  //: `stages * slot + fixed` carries `stages + 1 + kAheadExtra = IRING` index slots,
  //: which is exactly the ring depth the driver walks -- the accounting is the
  //: shipped `GeomB`'s, so the two carves are read the same way.
  static constexpr int fixed =
      panel_bytes + gram_bytes + ds_bytes + (1 + kAheadExtra) * idx_slot + form_bytes;
  static constexpr int stages = design::stages_for(slot, fixed, r0_threads_for(BC), reg_floor_r0());
  static constexpr size_t bytes = (size_t)stages * slot + fixed;
  static_assert((2 * C * LdV * (int)sizeof(__nv_bfloat16)) % 16 == 0 && panel_bytes % 16 == 0
                    && gram_bytes % 16 == 0 && ds_bytes % 16 == 0 && idx_slot % 16 == 0,
                "every region ahead of the schedule must leave it 16-byte aligned");
};

template <int C, int BC, bool PASS2>
constexpr int kR0CtaTarget =
    design::cta_target((int)GeomR0<C, BC, PASS2>::bytes, r0_threads_for(BC), reg_floor_r0());

//: THE DIGIT STRUCTURE A UNIT SEES, and it is the whole of rung b3's geometry.
//: Slots are canonical-leaf-ordered, `s = od * Bw^(D-1) + ...`, so at depth two
//: `od = s / Bw` and `i1 = s % Bw`.  A block's slots are a contiguous `BC`-aligned
//: run, a group's a contiguous `BCg`-aligned one, and a UNIT's a contiguous run of
//: `BCg / kUnits` -- so each of those is a rectangle in digit space and the level a
//: digit belongs to is a compile-time property of the offset, never a division.
//:
//: `spo` is how many of a unit's slots share one outer digit and `opu` how many
//: outer digits its run covers; exactly one of the two is 1 at every legal arm,
//: because `Bw >= 8` and a unit's run is a power of two.  `npo` converts that into
//: the accumulator's own axis: `n` counts eight-slot mma tiles, and `npo` of them
//: sit under one outer digit -- docs/internals/chunk/ladder_r0.md#the-digit-geometry
template <int D, int Bw, int BC, int BCg>
struct DigitR0 {
  static_assert(D >= 1 && D <= 2, "the ladder's digit gradients are built to depth two");
  static_assert((Bw & (Bw - 1)) == 0 && Bw >= 8, "one uniform power-of-two width, at least eight");
  static constexpr long slots = ipowl(Bw, D);
  static_assert(slots >= BC && slots % BC == 0, "the slot space must be a whole number of blocks");
  //: the packed amplitude row: level `l` owns columns `[l*Bw, (l+1)*Bw)`.
  static constexpr int cols = D * Bw;
  static constexpr int inner_base = (D - 1) * Bw;
  static constexpr int unit_run = BCg / kUnits;
  static constexpr int spo = unit_run < Bw ? unit_run : Bw;
  static constexpr int opu = unit_run / spo;
  static constexpr int npo = spo / kMmaN;
  static_assert(opu * spo == unit_run && npo * kMmaN == spo,
                "a unit's slot run must be a whole number of outer digits, each a whole "
                "number of mma tiles");
  //: THE OUTER ACCUMULATOR'S DEPTH.  It exists only at depth two: at depth one there
  //: is no other level to leave out, so the outer level is DELETED rather than
  //: weighted by one -- docs/internals/chunk/backward.md#digit-reduction
  static constexpr int u0_slots = D >= 2 ? opu : 1;
  static constexpr uint32_t row_bytes_p = (uint32_t)(cols * (int)sizeof(__nv_bfloat16));
  static constexpr uint32_t row_bytes_d = (uint32_t)(cols * (int)sizeof(float));
};

//: THE KRONECKER EXPANSION, IN KERNEL AND GATHERED.  A thread owns one ENTRY row of
//: its group's panel and an EIGHT-SLOT run of it.  The run is eight-aligned and
//: `Bw >= 8`, so it lies inside ONE outer digit and covers eight CONSECUTIVE inner
//: ones -- which is why the product is one splat times one vec8 and never a per-slot
//: gather, and why the expansion costs one `mul_bf16x2` per two slots.
//:
//: LOAD AND STORE ARE SEPARATE because the panel is single-buffered: the load reads
//: GLOBAL memory the visit does not own and races with nothing, so it is issued
//: BEFORE the rendezvous that retires the last visit's readers, and only the store
//: waits.  The carried run is `passes * 4` registers, which is what one panel copy
//: of shared memory costs in the other currency
//: -- docs/internals/chunk/ladder_r0.md#the-panel-leaves-the-ring
//:
//: A ROW PAST THE GATHERED COUNT IS EXACTLY ZERO, zeroed at the INNER amplitude so
//: one select covers the whole run: its list slot names token zero, so the gather
//: stays in range without a predicate and the coefficient it publishes is not.
template <int C, int BCg, int LdS, int D, int Bw>
struct ExpandR0 {
  static constexpr int gran = BCg / 8;
  static constexpr int rows = kGroupThreads / gran;
  static constexpr int passes = C / rows;
  static_assert(BCg % 8 == 0 && kGroupThreads % gran == 0 && C % rows == 0,
                "an expansion pass must cover the panel's rows exactly");
  static_assert((LdS * (int)sizeof(__nv_bfloat16)) % 16 == 0,
                "the panel's row pitch must keep an eight-slot run sixteen-byte aligned");
  int row;
  uint32_t dstOff;
  uint32_t colInnerB, colOuterB;
  uint32_t buf[passes][4];

  __device__ __forceinline__ void init(int gtid, int sbase) {
    row = gtid / gran;
    const int g0 = (gtid % gran) * 8;
    dstOff = (uint32_t)((row * LdS + g0) * (int)sizeof(__nv_bfloat16));
    const int s = sbase + g0;
    colInnerB = (uint32_t)(((D - 1) * Bw + (s % Bw)) * (int)sizeof(__nv_bfloat16));
    colOuterB = (uint32_t)((s / Bw) * (int)sizeof(__nv_bfloat16));
  }

  __device__ __forceinline__ void load(const void* plane, const uint16_t* lst, uint32_t winTok,
                                       int n) {
    constexpr uint32_t kRowB = (uint32_t)(D * Bw * (int)sizeof(__nv_bfloat16));
#pragma unroll
    for (int p = 0; p < passes; ++p) {
      const int e = row + p * rows;
      const uint32_t rowB = (winTok + (uint32_t)lst[e]) * kRowB;
      __align__(16) uint32_t a[4];
      *reinterpret_cast<uint4*>(a) = load_vec16(gather_ptr(plane, rowB + colInnerB));
      if (e >= n) {
#pragma unroll
        for (int k = 0; k < 4; ++k) a[k] = 0u;
      }
      if constexpr (D >= 2) {
        const uint32_t up = splat_bf16x2(
            *reinterpret_cast<const __nv_bfloat16*>(gather_ptr(plane, rowB + colOuterB)));
#pragma unroll
        for (int k = 0; k < 4; ++k) a[k] = mul_bf16x2(up, a[k]);
      }
#pragma unroll
      for (int k = 0; k < 4; ++k) buf[p][k] = a[k];
    }
  }

  __device__ __forceinline__ void store(char* panel) const {
    constexpr int kPassB = rows * LdS * (int)sizeof(__nv_bfloat16);
#pragma unroll
    for (int p = 0; p < passes; ++p)
      store_vec16(panel + dstOff + p * kPassB, *reinterpret_cast<const uint4*>(buf[p]));
  }
};

//: A [C, COLS] tile of a plane, GATHERED by one side's index list and staged
//: asynchronously.  A row past the gathered count reads the list's zero and lands on
//: token zero; every consumer of such a row is either masked by the causal threshold
//: (which is zero there) or multiplied by a coefficient row this rung zeroed.
template <int C, int COLS, int LD, int THREADS>
struct TileStage {
  static constexpr int gran = COLS / 8;
  static constexpr int rows = THREADS / gran;
  static constexpr int passes = C / rows;
  static_assert(COLS % 8 == 0 && THREADS % gran == 0 && C % rows == 0,
                "a staging pass must cover the tile's rows exactly");
  int row;
  uint32_t colBytes;
  uint32_t dstOff;

  __device__ __forceinline__ void init(int tid) {
    row = tid / gran;
    const int col = (tid % gran) * 8;
    colBytes = (uint32_t)(col * (int)sizeof(__nv_bfloat16));
    dstOff = (uint32_t)((row * LD + col) * (int)sizeof(__nv_bfloat16));
  }

  __device__ __forceinline__ void issue(uint32_t dst, const void* plane, const uint16_t* lst,
                                        uint32_t winTok, uint32_t rowBytes) const {
#pragma unroll
    for (int p = 0; p < passes; ++p) {
      const uint32_t off = (winTok + (uint32_t)lst[row + p * rows]) * rowBytes + colBytes;
      stage_tile(dst + dstOff + (uint32_t)(p * rows * LD * (int)sizeof(__nv_bfloat16)),
                 gather_ptr(plane, off));
    }
  }
};

//: THE DIGIT REDUCTION, IN LANE -- ONE component, driven by both passes, because
//: (A.9) and (A.10) differ only in which side's amplitudes they leave out:
//:
//:   d_p_{D-1}[t, i1] = sum_{s : i1(s) = i1} dX[t,s] * p_0[t, od(s)]      (INNER)
//:   d_p_0[t, od]     = sum_{s : od(s) = od} dX[t,s] * p_{D-1}[t, i1(s)]  (OUTER)
//:
//: THE PER-SLOT GRADIENT NEVER LEAVES A REGISTER, which is the whole of rung b3 on
//: the output side.  The leave-one-out factor is the OTHER level's amplitude and
//: never `X / p_l` -- under exact entmax both are zero off the support and the
//: division is `0/0` with no epsilon that does not invent a gradient
//: (docs/internals/chunk/backward.md#digit-reduction).
//:
//: The INNER digit is the lane's own slot pair, so its gradient needs NO reduction
//: at all: one fan-in per slot, the minimum the destination admits.  The OUTER one
//: sums over the slots of the unit's run that share its digit -- the lane's two,
//: then two shuffles over `lane & 3`, then the `npo` mma tiles the unroll already
//: visits -- and what remains, across units, groups and blocks, is the fan-in's,
//: exactly as `dv`'s was at rung b2.
//:
//: AT DEPTH ONE THE OUTER LEVEL IS DELETED, not weighted by one: level zero IS the
//: innermost, its leave-one-out product is empty, and `u0` does not exist.
//:
//: RUNG b4: THE ACCUMULATOR'S TOKEN AXIS IS THE ENTRY AXIS.  A row is emitted for
//: the token the side's index list names, and a row PAST the gathered count is not
//: emitted at all -- which is the executable statement that skipped work contributes
//: exactly zero -- docs/internals/chunk/ladder_r0.md#the-replay
template <int BC, int BCg, int D, int Bw, int MTOK, int NSLOT>
__device__ __forceinline__ void r0_emit_digits(const float (&acc)[MTOK][NSLOT][4],
                                               const void* pPlane, void* dPlane,
                                               const uint16_t* lst, uint32_t winTok, int n,
                                               int sbase, int lane) {
  using DG = DigitR0<D, Bw, BC, BCg>;
  //: THE SLOT TILE IS THE OUTER LOOP AND THE TOKEN THE INNER ONE, and the order was
  //: MEASURED rather than reasoned: the token-outer form shares one destination row
  //: base across every access a (token, half) makes and looks strictly cheaper in
  //: address arithmetic, but it holds the whole `acc` column live across the emit and
  //: cost 19 registers and 22 % of pass 1's wall at the pinned arm (162 regs / 2.548 ms
  //: against 143 / 2.087). The slot-outer form retires `acc[.][n]` a tile at a time.
  //: -- docs/internals/chunk/ladder_r0.md#the-emit-order
  //: THE LEAVE-ONE-OUT FACTORS ARE READ WHERE THEY ARE USED, and hoisting them to the
  //: top of the visit was BUILT AND REFUTED: it costs 38 registers and moves the wall
  //: by +1.2 % on pass 1 and -4.4 % on pass 2 at the pinned arm. The emit is not
  //: latency-bound -- docs/internals/chunk/ladder_r0.md#the-hoisted-leave-one-out
  const int e0 = lane >> 2;
  uint32_t tok[MTOK][2];
#pragma unroll
  for (int m = 0; m < MTOK; ++m)
#pragma unroll
    for (int h = 0; h < 2; ++h) tok[m][h] = winTok + (uint32_t)lst[e0 + m * kMmaM + h * 8];
  [[maybe_unused]] float fo[DG::u0_slots][MTOK][2];
  [[maybe_unused]] float u0[DG::u0_slots][MTOK][2];
  //: the OUTER amplitude is constant over a unit's `spo` slots, so it is read once per
  //: (outer digit, token) rather than once per slot tile.
  if constexpr (D >= 2) {
#pragma unroll
    for (int o = 0; o < DG::u0_slots; ++o) {
      const uint32_t cOut = (uint32_t)(((sbase + o * DG::spo) / Bw) * (int)sizeof(__nv_bfloat16));
#pragma unroll
      for (int m = 0; m < MTOK; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          fo[o][m][h] = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
              gather_ptr(pPlane, tok[m][h] * DG::row_bytes_p + cOut)));
          u0[o][m][h] = 0.f;
        }
    }
  }
#pragma unroll
  for (int n2 = 0; n2 < NSLOT; ++n2) {
    const int o = n2 / DG::npo;
    const int i1 = DG::inner_base + ((sbase + n2 * kMmaN) % Bw) + (lane & 3) * 2;
    const uint32_t cInD = (uint32_t)(i1 * (int)sizeof(float));
#pragma unroll
    for (int m = 0; m < MTOK; ++m)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float a0 = acc[m][n2][2 * h], a1 = acc[m][n2][2 * h + 1];
        float f = 1.f;
        if constexpr (D >= 2) f = fo[o][m][h];
        if (e0 + m * kMmaM + h * 8 < n) {
          float* p =
              reinterpret_cast<float*>(gather_ptr(dPlane, tok[m][h] * DG::row_bytes_d + cInD));
          fan_in_add(p, a0 * f);
          fan_in_add(p + 1, a1 * f);
        }
        if constexpr (D >= 2) {
          const uint32_t q = *reinterpret_cast<const uint32_t*>(gather_ptr(
              pPlane, tok[m][h] * DG::row_bytes_p + (uint32_t)(i1 * (int)sizeof(__nv_bfloat16))));
          u0[o][m][h] += a0 * __bfloat162float(__ushort_as_bfloat16((unsigned short)q))
                         + a1 * __bfloat162float(__ushort_as_bfloat16((unsigned short)(q >> 16)));
        }
      }
  }
  if constexpr (D >= 2) {
#pragma unroll
    for (int o = 0; o < DG::u0_slots; ++o) {
      const uint32_t cOut = (uint32_t)(((sbase + o * DG::spo) / Bw) * (int)sizeof(float));
#pragma unroll
      for (int m = 0; m < MTOK; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          float x = u0[o][m][h];
#pragma unroll
          for (int b2 = 1; b2 < 4; b2 <<= 1) x += __shfl_xor_sync(0xFFFFFFFFu, x, b2);
          if ((lane & 3) == 0 && e0 + m * kMmaM + h * 8 < n)
            fan_in_add(
                reinterpret_cast<float*>(gather_ptr(dPlane, tok[m][h] * DG::row_bytes_d + cOut)),
                x);
        }
    }
  }
}

#define R0_ADDR(b_, ld_, R_, g_) ((b_) + (uint32_t)(((R_) * (ld_) + (g_) * 8) * 2))

//: THE MASS SEED'S SEAT, in the `dnum` panel's row pad: entry `i` at row `i / 8`,
//: column `d_v + i % 8`.  Eight-wide groups are what makes `(i, i + 1)` adjacent for
//: every even `i`, which is the whole reason for the shape -- the fold reads the seed
//: as an mma A fragment, whose lanes want exactly those pairs.
#define R0_DDEN(ld_, i_) (((i_) >> 3) * (ld_) + kD + ((i_) & 7))

//: THE SEED'S SEATS, RESOLVED ONCE.  Every consumer's entry index is a fixed
//: function of the lane, so the seat arithmetic is loop-invariant and only the ring
//: slot's base moves; carrying the offsets is what keeps the visit's added work to
//: the arithmetic the mass terms actually are.
template <int LD, int ROWS>
__device__ __forceinline__ void r0_dden_seats(int (&off)[ROWS][2], int row0, int stride) {
#pragma unroll
  for (int i = 0; i < ROWS; ++i) {
    off[i][0] = R0_DDEN(LD, row0 + i * stride);
    off[i][1] = R0_DDEN(LD, row0 + i * stride + 8);
  }
}

//: The mass folds contract a token axis against a coefficient panel with an A
//: operand that is EXACTLY representable in bf16 -- ones on the write side, the
//: bf16 seed plane itself on the read side -- so the fold rides the tensor pipe
//: without a compensated split.  `kOnesFrag` is bf16 `1.0` in both halves.
static constexpr uint32_t kOnesFrag = 0x3F803F80u;

//: THE STAIRCASE, COMPACTED: rows are READ entries and columns WRITE entries, and a
//: write entry is kept when it lies at or below the read entry's token -- which the
//: walk resolved once, at close, as the per-read-entry threshold `J`.  At a fully
//: live cell `J[i] = i + 1` and this is rung b3's `tc <= tr` exactly; where the walk
//: gathered nothing for a read entry `J` is zero and the whole row is empty, which
//: is what makes a row past the gathered count contribute nothing without a second
//: predicate -- docs/internals/chunk/backward.md#pass-law
#define R0_STAIR(v_, tc_, jr_) ((tc_) < (jr_) ? (v_) : 0.f)

//: THE SCHEDULE'S CLAUSE COLUMNS.  A block's membership is the AND of one table row
//: per tested level, and which row is a compile-time function of the block index --
//: the shipped scans' own map, driven here through the same `LevelPlan`.
#define R0_CLAUSE(CL_, C_)                                     \
  if constexpr ((CL_) < LP::clauses) {                         \
    constexpr int kDiv = LP::suffix_groups((CL_) + 1);         \
    constexpr int kGrp = LP::groups_at(CL_);                   \
    constexpr int kPre = LP::prefix_entries(CL_);              \
    C_ = (uint32_t)(kPre + (blk / kDiv) % kGrp) * clauseStride; \
  }

//: PASS 1 -- the ascending rebuild.  It carries `S` and its mass column `z`, forms
//: `dR[i,s] = <dnum_i, S_a[s,:]> + dden_i z_a[s] + sum_{j<=i} W[j,s] Gd'[i,j]` with
//: `Gd'[i,j] = <dnum_i, v_j> + dden_i`, and emits its DIGIT REDUCTION (A.9).  The
//: write panel is the Kronecker product of `pw`'s levels, expanded in kernel; `pr`
//: is read only at the emit, where it supplies the leave-one-out factor.  The token
//: axis is the ASCENDING walk's: `Ascent`, the component the forward and Scan A
//: drive, replayed off the saved table.
template <int C, int BC, int D, int Bw>
__global__ __launch_bounds__(r0_threads_for(BC), (kR0CtaTarget<C, BC, false>)) void
rung0_pass1_kernel(const __nv_bfloat16* __restrict__ pread,
                   const __nv_bfloat16* __restrict__ pwrite, const __nv_bfloat16* __restrict__ v,
                   const __nv_bfloat16* __restrict__ dnum, const __nv_bfloat16* __restrict__ dden,
                   const uint32_t* __restrict__ tbl, float* __restrict__ dpr, int L,
                   int v_row_bytes) {
  using G = GeomR0<C, BC, false>;
  using DG = DigitR0<D, Bw, BC, G::BCg>;
  using LP = LevelPlan<D, Bw, BC>;
  constexpr int kThreads = r0_threads_for(BC);
  constexpr int kWarps = kThreads / 32;
  constexpr int GRP = G::GRP, BCg = G::BCg;
  constexpr int STAGES = G::stages;
  constexpr int AHEAD = STAGES - 1 + kAheadExtra;
  constexpr int IRING = AHEAD + 2;
  constexpr int LdV = G::LdV, LdS = G::LdS, LdG = G::LdG, LdDs = G::LdDs;
  constexpr int MTOK = C / kMmaM, NTOK = C / kMmaN;
  constexpr int KD = BCg / kMmaK, NDV = BCg / kMmaN, KV = kD / kMmaK;
  //: the per-slot output splits over the SLOT axis: every unit runs the whole
  //: channel contraction out of its group's published state, so the four outputs
  //: are disjoint and no cross-unit sum exists.
  constexpr int NSLOT = NDV / kUnits;
  constexpr int NPAIR = NSLOT / 2;
  static_assert(kWarps == GRP * kUnits, "a group is the carve: four units, one per d_v quarter");
  static_assert(NDV == kUnits * NSLOT && NSLOT == 2 * NPAIR,
                "a unit's slot run must be a whole number of ldmatrix pairs");
  static_assert(design::smem_fits(G::bytes),
                "this arm does not fit this architecture's per-CTA shared-memory maximum");
  static_assert(STAGES >= 2, "the wait/issue order needs at least two stages");
  static_assert(C % 8 == 0 && kPadE >= 8 && C <= 255,
                "the mass seed seats in the value panel's row pad, eight entries to a "
                "row, and a causal threshold is one byte");
  static_assert(LP::slots == DG::slots && LP::blocks == DG::slots / BC,
                "the level plan and the digit geometry must name ONE lattice");
  //: the `d_v` Gram has NO slot axis, so it is formed once for the whole CTA by the
  //: warps of group zero and read by every group.
  constexpr int MSPLIT = MTOK >= kUnits ? kUnits : MTOK;
  constexpr int NSPLIT_RAW = kUnits / MSPLIT;
  constexpr int NSPLIT = NSPLIT_RAW < NTOK / 2 ? NSPLIT_RAW : NTOK / 2;
  constexpr int MPW = MTOK / MSPLIT;
  constexpr int NPW = NTOK / NSPLIT;
  constexpr int GWARPS = MSPLIT * NSPLIT;
  constexpr int GPAIR = NPW / 2;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int grp = tid / kGroupThreads, gtid = tid % kGroupThreads;
  //: TWO WARP INDICES, and the distinction is load-bearing at `GRP = 2`: `warp` is
  //: the IN-GROUP one and names this thread's mma unit, while the SCHEDULE is
  //: CTA-wide and its cook must run in exactly one warp of the whole block.
  const int warp = gtid >> 5, ctawarp = tid >> 5;
  const int blk = (int)blockIdx.x;

  extern __shared__ __align__(16) char smem_raw[];
  __nv_bfloat16* sv = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* sdn = sv + STAGES * C * LdV;
  __nv_bfloat16* sw = sdn + STAGES * C * LdV;
  __nv_bfloat16* sg = sw + GRP * C * LdS;
  __nv_bfloat16* sds = sg + C * LdG;
  char* sidx = reinterpret_cast<char*>(sds + GRP * kD * LdDs);
  uint32_t* sfR = reinterpret_cast<uint32_t*>(sidx + IRING * G::idx_slot);
  uint32_t* sfW = sfR + kFormWords;
  int* scook = reinterpret_cast<int*>(sfW + kFormWords);
  sds += grp * kD * LdDs;

  const int bh = (int)blockIdx.y;
  //: THE BLOCK LATTICE.  A block owns `BC` of the `N` slots and a group owns `BCg`
  //: of the block's; the per-slot outputs are DISJOINT over that partition, so
  //: nothing about `dR` fans in -- what fans in is the DIGIT gradient, which is
  //: shared across the lattice where a slot was not.
  const long slot0 = (long)LP::block_base(blk) + (long)grp * BCg;
  const int vRowElems = v_row_bytes / (int)sizeof(__nv_bfloat16);
  const int heads = vRowElems / kD;
  const __nv_bfloat16* const vPlane =
      pin_address(v + (long)(bh / heads) * L * vRowElems + (long)(bh % heads) * kD);
  const __nv_bfloat16* const dnPlane = pin_address(dnum + (long)bh * L * kD);
  const __nv_bfloat16* const ddPlane = pin_address(dden + (long)bh * L);
  //: THE AMPLITUDE PLANES ARE NOT SLICED BY THE BLOCK.  A block's slots are a
  //: rectangle in digit space, so what it needs is a COLUMN RANGE per level, and the
  //: range is a compile-time function of the slot base -- carried in the expansion's
  //: and the emit's own offsets rather than folded into the base.
  const __nv_bfloat16* const pwPlane = pin_address(pwrite + (long)bh * L * DG::cols);
  const __nv_bfloat16* const prPlane = pin_address(pread + (long)bh * L * DG::cols);
  float* const dprPlane = pin_address(dpr + (long)bh * L * DG::cols);
  const int tokWords = tok_words(L);
  const uint32_t clauseStride = (uint32_t)tokWords * (uint32_t)sizeof(uint32_t);

  //: THE SCHEDULE IS A COMPONENT: the closing rule has ONE implementation, and the
  //: forward, Scan A and this rung all drive it -- docs/internals/chunk/ladder_r0.md#the-replay
  Ascent<C, LP::clauses, kThreads, G::idx_slot, G::hdr> aw{};
  aw.sidx = sidx;
  aw.sfR = sfR;
  aw.sfW = sfW;
  aw.scook = scook;
  aw.rowR = pin_address(tbl + ((long)(2 * bh + 0) * LP::entries) * tokWords);
  aw.rowW = pin_address(tbl + ((long)(2 * bh + 1) * LP::entries) * tokWords);
  aw.tokWords = tokWords;
  R0_CLAUSE(0, aw.cA)
  R0_CLAUSE(1, aw.cB)
  R0_CLAUSE(2, aw.cC)
  R0_CLAUSE(3, aw.cD)
  aw.tailMask = (L & (kTokBits - 1)) ? ((1u << (L & (kTokBits - 1))) - 1u) : 0xFFFFFFFFu;
  aw.lastWord = tokWords - 1;

  const int lr = lane & 7, tl = lane >> 3;
  const int rF = (tl & 1) * 8 + lr, cF = tl >> 1;
  const int rT = (tl >> 1) * 8 + lr, cT = tl & 1;
  const uint32_t offFv = (uint32_t)((rF * LdV + cF * 8) * 2);
  const uint32_t offTv = (uint32_t)((rT * LdV + cT * 8) * 2);
  const uint32_t offFs = (uint32_t)((rF * LdS + cF * 8) * 2);
  const uint32_t offFg = (uint32_t)((rF * LdG + cF * 8) * 2);
  const uint32_t offFd = (uint32_t)((rF * LdDs + cF * 8) * 2);
  const int gm0 = (warp / NSPLIT) * MPW, gn0 = (warp % NSPLIT) * NPW;
  int ddG[MPW][2], ddM[MTOK][2];
  r0_dden_seats<LdV, MPW>(ddG, gm0 * 16 + (lane >> 2), 16);
  r0_dden_seats<LdV, MTOK>(ddM, lane >> 2, 16);

  //: THE UNIT'S FIRST SLOT, hoisted: it is the digit geometry's own base and does not
  //: move with the super-chunk.
  const int sbaseU = (int)slot0 + warp * (BCg / kUnits);
  const uint32_t svB = smem_addr(sv), sdnB = smem_addr(sdn);
  const uint32_t swB =
      smem_addr(sw) + (uint32_t)grp * (uint32_t)(C * LdS * (int)sizeof(__nv_bfloat16));
  const uint32_t sgF = smem_addr(sg) + offFg;
  const uint32_t sdsF = smem_addr(sds) + offFd;
  char* const swThis = reinterpret_cast<char*>(sw) + (size_t)grp * (C * LdS * 2);

  //: the transposed state: m = d_v (this unit's 16 rows), n = this block's slots.
  float st[NDV][4];
#pragma unroll
  for (int j = 0; j < NDV; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) st[j][e] = 0.f;

  //: THE MASS COLUMN, in the accumulator layout its one consumer already reads.
  //: `A` is all ones, so every `m` row of the tile carries the same slot total and
  //: elements 0/1 are the pair the entry-mass term wants; no reduction exists.
  float zac[NSLOT][4];
#pragma unroll
  for (int n2 = 0; n2 < NSLOT; ++n2)
#pragma unroll
    for (int e = 0; e < 4; ++e) zac[n2][e] = 0.f;
  const uint32_t ones[4] = {kOnesFrag, kOnesFrag, kOnesFrag, kOnesFrag};

  //: the value ring is staged by the WHOLE CTA and the coefficient panel produced by
  //: each group over its own slot slice, which is why the two carry different thread
  //: counts and the expansion is indexed by the in-group thread.
  TileStage<C, kD, LdV, kThreads> stV;
  ExpandR0<C, BCg, LdS, D, Bw> exW;
  stV.init(tid);
  exW.init(gtid, (int)slot0);
  constexpr uint32_t kVSlotB = (uint32_t)(C * LdV * 2);
  constexpr uint32_t kVRingB = (uint32_t)STAGES * kVSlotB;
  const uint32_t vRowB = (uint32_t)v_row_bytes;
  constexpr uint32_t kDnRowB = (uint32_t)(kD * (int)sizeof(__nv_bfloat16));
  uint32_t stgV = 0, cmpV = 0;

  //: THE STAGE IS WHAT IS SKIPPED; THE RING SLOT IS NOT.  Both tiles ride one slot
  //: cursor, so the wait/commit accounting stays the forward's however sparse the
  //: super-chunk is.
  auto stage_chunk = [&](int d) {
    const int* h = aw.ihdr(d);
    const uint32_t win = (uint32_t)h[3] * (uint32_t)kWindow;
    if (h[1] != 0) stV.issue(svB + stgV, vPlane, aw.iwrite(d), win, vRowB);
    if (h[0] != 0) {
      stV.issue(sdnB + stgV, dnPlane, aw.iread(d), win, kDnRowB);
      if (tid < C)
        sdn[(stgV >> 1) + R0_DDEN(LdV, tid)] =
            tid < h[0] ? ddPlane[(int)win + (int)aw.iread(d)[tid]] : __float2bfloat16(0.f);
    }
    stgV += kVSlotB;
    if (stgV == kVRingB) stgV = 0;
  };

  aw.begin_form();
  aw.begin_walk();
  if (tid == 0) {
    scook[kCookWord] = 0;
    scook[kCookOpen] = 0;
  }
  rendezvous();
  aw.form(tid);
  rendezvous();
#pragma unroll 1
  for (int s = 0; s < AHEAD; ++s) {
    if (ctawarp == 0) aw.walk(s, lane);
    rendezvous();
    aw.walk_wait(s, tid, ctawarp, lane);
    aw.form_if_low(tid);
    rendezvous();
  }
#pragma unroll 1
  for (int s = 0; s < STAGES - 1; ++s) {
    stage_chunk(s);
    stage_commit();
  }

  //: THE RING CURSORS ARE COUNTERS, NOT REMAINDERS: `IRING` is not a power of two,
  //: so a modulo would put a compile-time reciprocal in the body of the one loop the
  //: whole kernel runs in.
  int dc = 0, ds = STAGES - 1, dn = AHEAD;
#pragma unroll 1
  for (;;) {
    const int* h = aw.ihdr(dc);
    const int nr = h[0], nw = h[1];
    if (!h[2]) break;
    const bool liveR = nr != 0, liveW = nw != 0;
    const uint32_t win = (uint32_t)h[3] * (uint32_t)kWindow;
    const uint16_t* const lrd = aw.iread(dc);
    const uint8_t* const ljt = aw.ithr(dc);

    //: THE COOK RUNS BEFORE THE PRODUCTION AND PUBLISHES AT THE SAME RENDEZVOUS the
    //: staged tiles do: one unit's work against four units' consumption.
    if (ctawarp == 0) aw.walk(dn, lane);
    //: THE EXPANSION'S LOAD IS READ-ONLY GLOBAL TRAFFIC and races with nothing, so
    //: it is issued HERE, where the last visit's tail covers its latency; only the
    //: store waits on the rendezvous.
    if (liveW) exW.load(pwPlane, aw.iwrite(dc), win, nw);

    stage_wait<STAGES - 2>();
    rendezvous();  // (1) this super-chunk's tiles are visible; the last visit's panel,
                   //     Gram and published state are dead; the walk has published

    aw.walk_wait(dn, tid, ctawarp, lane);
    aw.form_if_low(tid);
    stage_chunk(ds);
    stage_commit();

    if (liveW) exW.store(swThis);

    const uint32_t aVT = svB + cmpV + offTv;
    const uint32_t aDF = sdnB + cmpV + offFv;
    const uint32_t aWF = swB + offFs;
    const __nv_bfloat16* const dd = sdn + (cmpV >> 1);
    cmpV += kVSlotB;
    if (cmpV == kVRingB) cmpV = 0;

    //: THE ENTRY STATE, PUBLISHED ONCE PER VISIT.  A unit owns sixteen channels of
    //: every slot, so the four publications are disjoint and what comes back is the
    //: whole channel axis -- which is what lets the per-slot term finish in one unit.
#pragma unroll
    for (int nj = 0; nj < NDV; ++nj) {
      const int s0 = nj * kMmaN + (lane & 3) * 2;
      const int c0 = warp * kMmaM + (lane >> 2);
      store_bf16x2(&sds[c0 * LdDs + s0], pack_bf16x2(st[nj][0], st[nj][1]));
      store_bf16x2(&sds[(c0 + 8) * LdDs + s0], pack_bf16x2(st[nj][2], st[nj][3]));
    }

    // ---- the d_v Gram: Gd' = staircase(dnum . v^T + dden (x) 1) --------------
    //: ONE Gram for the whole CTA.  It contracts over channels and carries no slot
    //: axis, so a second group would form the same object twice.  It reads the two
    //: VALUE-shaped panels only, which is why pass 1 forms it in the phase that
    //: stores the coefficient panel and keeps its rendezvous count at two.
    if (liveR && liveW && grp == 0 && warp < GWARPS) {
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
        for (int kk = 0; kk < KV; ++kk) {
          uint32_t qa[4];
          load_frag(qa, R0_ADDR(aDF, LdV, (gm0 + m) * 16, 2 * kk));
#pragma unroll
          for (int u = 0; u < GPAIR; ++u) {
            uint32_t bf[4];
            load_frag(bf, R0_ADDR(aVT, LdV, (gn0 + u * 2) * 8, 2 * kk));
            mma(g[u][0], qa, &bf[0]);
            mma(g[u][1], qa, &bf[2]);
          }
        }
        const int tr = (gm0 + m) * 16 + (lane >> 2);
        const float m0 = __bfloat162float(dd[ddG[m][0]]), m1 = __bfloat162float(dd[ddG[m][1]]);
        const int j0 = (int)ljt[tr], j1 = (int)ljt[tr + 8];
#pragma unroll
        for (int u = 0; u < GPAIR; ++u)
#pragma unroll
          for (int h2 = 0; h2 < 2; ++h2) {
            const int tc = (gn0 + u * 2 + h2) * 8 + (lane & 3) * 2;
            store_bf16x2(&sg[tr * LdG + tc], pack_bf16x2(R0_STAIR(g[u][h2][0] + m0, tc, j0),
                                                         R0_STAIR(g[u][h2][1] + m0, tc + 1, j0)));
            store_bf16x2(&sg[(tr + 8) * LdG + tc],
                         pack_bf16x2(R0_STAIR(g[u][h2][2] + m1, tc, j1),
                                     R0_STAIR(g[u][h2][3] + m1, tc + 1, j1)));
          }
      }
    }

    rendezvous();  // (2) the coefficient panel, the published state and the Gram are visible

    // ---- dR, over this unit's slot run ---------------------------------------
    if (liveR) {
      float acc[MTOK][NSLOT][4];
#pragma unroll
      for (int m = 0; m < MTOK; ++m)
#pragma unroll
        for (int n2 = 0; n2 < NSLOT; ++n2)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[m][n2][e] = 0.f;
      const int n0 = warp * NSLOT;
      //: the INTER term: the whole channel axis, out of the published state.
#pragma unroll
      for (int kk = 0; kk < KV; ++kk)
#pragma unroll
        for (int np = 0; np < NPAIR; ++np) {
          uint32_t sb[4];
          load_frag_t(sb, R0_ADDR(sdsF, LdDs, kk * 16, n0 + np * 2));
#pragma unroll
          for (int m = 0; m < MTOK; ++m) {
            uint32_t da[4];
            load_frag(da, R0_ADDR(aDF, LdV, m * 16, 2 * kk));
            mma(acc[m][2 * np], da, &sb[0]);
            mma(acc[m][2 * np + 1], da, &sb[2]);
          }
        }
      //: the INTRA term: the staircase already lives in the Gram, so this is one
      //: contraction over the super-chunk's write entries with the Gram as its A
      //: operand.  A super-chunk with no write entry has no intra term at all.
      if (liveW) {
#pragma unroll
        for (int kk = 0; kk < MTOK; ++kk)
#pragma unroll
          for (int np = 0; np < NPAIR; ++np) {
            uint32_t wb[4];
            load_frag_t(wb, R0_ADDR(aWF, LdS, kk * 16, n0 + np * 2));
#pragma unroll
            for (int m = 0; m < MTOK; ++m) {
              uint32_t ga[4];
              load_frag(ga, R0_ADDR(sgF, LdG, m * 16, 2 * kk));
              mma(acc[m][2 * np], ga, &wb[0]);
              mma(acc[m][2 * np + 1], ga, &wb[2]);
            }
          }
      }
      //: the ENTRY MASS, `dden_i z_a[s]`: an OUTER product of a per-read-entry
      //: value and a per-slot one, so it is a rank-1 add and not a broadcast.
#pragma unroll
      for (int m = 0; m < MTOK; ++m) {
        const float d0 = __bfloat162float(dd[ddM[m][0]]);
        const float d1 = __bfloat162float(dd[ddM[m][1]]);
#pragma unroll
        for (int n2 = 0; n2 < NSLOT; ++n2) {
          acc[m][n2][0] += d0 * zac[n2][0];
          acc[m][n2][1] += d0 * zac[n2][1];
          acc[m][n2][2] += d1 * zac[n2][0];
          acc[m][n2][3] += d1 * zac[n2][1];
        }
      }
      //: (A.9), THE READ SIDE'S DIGIT REDUCTION.  `dR` is consumed here and never
      //: stored: what leaves the kernel is `D` columns per token, not `BCg`.
      r0_emit_digits<BC, BCg, D, Bw, MTOK, NSLOT>(acc, prPlane, dprPlane, lrd, win, nr, sbaseU,
                                                  lane);
    }

    // ---- the fold: S^T += V^T . W, and z += 1^T . W --------------------------
    if (liveW) {
#pragma unroll
      for (int kk = 0; kk < MTOK; ++kk) {
        uint32_t a[4];
        load_frag_t(a, R0_ADDR(aVT, LdV, kk * 16, 2 * warp));
#pragma unroll
        for (int nj = 0; nj < KD; ++nj) {
          uint32_t b[4];
          load_frag_t(b, R0_ADDR(aWF, LdS, kk * 16, 2 * nj));
          mma(st[2 * nj], a, &b[0]);
          mma(st[2 * nj + 1], a, &b[2]);
        }
      }
      //: the mass fold takes the SAME operand from the SAME tiles the per-slot term
      //: reads, so it re-reads rather than outliving the fold's own k-run.
      const int n0 = warp * NSLOT;
#pragma unroll
      for (int kk = 0; kk < MTOK; ++kk)
#pragma unroll
        for (int np = 0; np < NPAIR; ++np) {
          uint32_t wb[4];
          load_frag_t(wb, R0_ADDR(aWF, LdS, kk * 16, n0 + np * 2));
          mma(zac[2 * np], ones, &wb[0]);
          mma(zac[2 * np + 1], ones, &wb[2]);
        }
    }

    if (++dc == IRING) dc = 0;
    if (++ds == IRING) ds = 0;
    if (++dn == IRING) dn = 0;
  }
}

//: PASS 2 -- the descending adjoint.  It carries `dS` and its mass column `dz`, and
//: emits `dv[j,:] = sum_s W[j,s] dS_after[j,s,:]` and the DIGIT REDUCTION (A.10) of
//: `dW[j,s] = <dS_a[s,:], v_j> + dz_a[s] + sum_{i>=j} R[i,s] Gd'[i,j]`.  `dv` is a
//: value-channel quantity and the mass column does not enter it.  Both panels are
//: Kronecker products expanded in kernel; `pw` also supplies the write side's
//: leave-one-out factor at the emit.  The token axis is the DESCENDING walk's:
//: `Descent`, the component Scan B and the partition probe drive.
template <int C, int BC, int D, int Bw>
__global__ __launch_bounds__(r0_threads_for(BC), (kR0CtaTarget<C, BC, true>)) void
rung0_pass2_kernel(const __nv_bfloat16* __restrict__ pread,
                   const __nv_bfloat16* __restrict__ pwrite, const __nv_bfloat16* __restrict__ v,
                   const __nv_bfloat16* __restrict__ dnum, const __nv_bfloat16* __restrict__ dden,
                   const uint32_t* __restrict__ tbl, float* __restrict__ dv,
                   float* __restrict__ dpw, int L, int v_row_bytes) {
  using G = GeomR0<C, BC, true>;
  using DG = DigitR0<D, Bw, BC, G::BCg>;
  using LP = LevelPlan<D, Bw, BC>;
  constexpr int kThreads = r0_threads_for(BC);
  constexpr int kWarps = kThreads / 32;
  constexpr int GRP = G::GRP, BCg = G::BCg;
  constexpr int STAGES = G::stages;
  constexpr int AHEAD = STAGES - 1 + kAheadExtra;
  constexpr int IRING = AHEAD + 2;
  constexpr int LdV = G::LdV, LdS = G::LdS, LdG = G::LdG, LdDs = G::LdDs;
  constexpr int MTOK = C / kMmaM, NTOK = C / kMmaN;
  constexpr int KD = BCg / kMmaK, NDV = BCg / kMmaN, KV = kD / kMmaK;
  constexpr int NSLOT = NDV / kUnits;
  constexpr int NPAIR = NSLOT / 2;
  static_assert(kWarps == GRP * kUnits, "a group is the carve: four units, one per d_v quarter");
  static_assert(NDV == kUnits * NSLOT && NSLOT == 2 * NPAIR,
                "a unit's slot run must be a whole number of ldmatrix pairs");
  static_assert(design::smem_fits(G::bytes),
                "this arm does not fit this architecture's per-CTA shared-memory maximum");
  static_assert(STAGES >= 2, "the wait/issue order needs at least two stages");
  static_assert(C % 8 == 0 && kPadE >= 8 && C <= 255,
                "the mass seed seats in the value panel's row pad, eight entries to a "
                "row, and a causal threshold is one byte");
  static_assert(LP::slots == DG::slots && LP::blocks == DG::slots / BC,
                "the level plan and the digit geometry must name ONE lattice");
  constexpr int MSPLIT = MTOK >= kUnits ? kUnits : MTOK;
  constexpr int NSPLIT_RAW = kUnits / MSPLIT;
  constexpr int NSPLIT = NSPLIT_RAW < NTOK / 2 ? NSPLIT_RAW : NTOK / 2;
  constexpr int MPW = MTOK / MSPLIT;
  constexpr int NPW = NTOK / NSPLIT;
  constexpr int GWARPS = MSPLIT * NSPLIT;
  constexpr int GPAIR = NPW / 2;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int grp = tid / kGroupThreads, gtid = tid % kGroupThreads;
  //: TWO WARP INDICES, and the distinction is load-bearing at `GRP = 2`: `warp` is
  //: the IN-GROUP one and names this thread's mma unit, while the SCHEDULE is
  //: CTA-wide and its cook must run in exactly one warp of the whole block.
  const int warp = gtid >> 5, ctawarp = tid >> 5;
  const int blk = (int)blockIdx.x;

  extern __shared__ __align__(16) char smem_raw[];
  __nv_bfloat16* sv = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* sdn = sv + STAGES * C * LdV;
  __nv_bfloat16* sr = sdn + STAGES * C * LdV;
  __nv_bfloat16* sw = sr + GRP * C * LdS;
  //: the SLOT Gram is one per group; the channel Gram is one for the CTA.
  __nv_bfloat16* sguAll = sw + GRP * C * LdS;
  __nv_bfloat16* sgu = sguAll + grp * C * LdG;
  __nv_bfloat16* sgd = sguAll + GRP * C * LdG;
  __nv_bfloat16* sdsAll = sgd + C * LdG;
  char* sidx = reinterpret_cast<char*>(sdsAll + GRP * kD * LdDs);
  uint32_t* sfR = reinterpret_cast<uint32_t*>(sidx + IRING * G::idx_slot);
  uint32_t* sfW = sfR + kFormWords;
  int* scook = reinterpret_cast<int*>(sfW + kFormWords);
  __nv_bfloat16* sds = sdsAll + grp * kD * LdDs;

  const int bh = (int)blockIdx.y;
  const long slot0 = (long)LP::block_base(blk) + (long)grp * BCg;
  const int vRowElems = v_row_bytes / (int)sizeof(__nv_bfloat16);
  const int heads = vRowElems / kD;
  const __nv_bfloat16* const vPlane =
      pin_address(v + (long)(bh / heads) * L * vRowElems + (long)(bh % heads) * kD);
  const __nv_bfloat16* const dnPlane = pin_address(dnum + (long)bh * L * kD);
  const __nv_bfloat16* const ddPlane = pin_address(dden + (long)bh * L);
  const __nv_bfloat16* const prPlane = pin_address(pread + (long)bh * L * DG::cols);
  const __nv_bfloat16* const pwPlane = pin_address(pwrite + (long)bh * L * DG::cols);
  float* const dvPlane = pin_address(dv + (long)bh * L * kD);
  float* const dpwPlane = pin_address(dpw + (long)bh * L * DG::cols);
  const int tokWords = tok_words(L);
  const uint32_t clauseStride = (uint32_t)tokWords * (uint32_t)sizeof(uint32_t);

  Descent<C, LP::clauses, kThreads, G::idx_slot, G::hdr> dw{};
  dw.sidx = sidx;
  dw.sfR = sfR;
  dw.sfW = sfW;
  dw.scook = scook;
  dw.rowR = pin_address(tbl + ((long)(2 * bh + 0) * LP::entries) * tokWords);
  dw.rowW = pin_address(tbl + ((long)(2 * bh + 1) * LP::entries) * tokWords);
  dw.tokWords = tokWords;
  R0_CLAUSE(0, dw.cA)
  R0_CLAUSE(1, dw.cB)
  R0_CLAUSE(2, dw.cC)
  R0_CLAUSE(3, dw.cD)
  dw.tailMask = (L & (kTokBits - 1)) ? ((1u << (L & (kTokBits - 1))) - 1u) : 0xFFFFFFFFu;
  dw.lastWord = tokWords - 1;

  const int lr = lane & 7, tl = lane >> 3;
  const int rF = (tl & 1) * 8 + lr, cF = tl >> 1;
  const int rT = (tl >> 1) * 8 + lr, cT = tl & 1;
  const uint32_t offFv = (uint32_t)((rF * LdV + cF * 8) * 2);
  const uint32_t offTv = (uint32_t)((rT * LdV + cT * 8) * 2);
  const uint32_t offFs = (uint32_t)((rF * LdS + cF * 8) * 2);
  const uint32_t offTs = (uint32_t)((rT * LdS + cT * 8) * 2);
  const uint32_t offFg = (uint32_t)((rF * LdG + cF * 8) * 2);
  const uint32_t offTg = (uint32_t)((rT * LdG + cT * 8) * 2);
  const uint32_t offFd = (uint32_t)((rF * LdDs + cF * 8) * 2);
  const int gm0 = (warp / NSPLIT) * MPW, gn0 = (warp % NSPLIT) * NPW;
  int ddG[MPW][2], ddK[MTOK][2];
  r0_dden_seats<LdV, MPW>(ddG, gm0 * 16 + (lane >> 2), 16);
  r0_dden_seats<LdV, MTOK>(ddK, (lane & 3) * 2, 16);

  //: THE UNIT'S FIRST SLOT, hoisted: it is the digit geometry's own base and does not
  //: move with the super-chunk.
  const int sbaseU = (int)slot0 + warp * (BCg / kUnits);
  constexpr uint32_t kGrpS = (uint32_t)(C * LdS * (int)sizeof(__nv_bfloat16));
  const uint32_t svB = smem_addr(sv), sdnB = smem_addr(sdn);
  const uint32_t srB = smem_addr(sr) + (uint32_t)grp * kGrpS;
  const uint32_t swB = smem_addr(sw) + (uint32_t)grp * kGrpS;
  const uint32_t sguF = smem_addr(sgu) + offFg;
  const uint32_t sgdT = smem_addr(sgd) + offTg;
  const uint32_t sdsF = smem_addr(sds) + offFd;
  char* const srThis = reinterpret_cast<char*>(sr) + (size_t)grp * kGrpS;
  char* const swThis = reinterpret_cast<char*>(sw) + (size_t)grp * kGrpS;

  //: the adjoint state, transposed exactly as the forward's state is.
  float dst[NDV][4];
#pragma unroll
  for (int j = 0; j < NDV; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) dst[j][e] = 0.f;

  //: THE ADJOINT MASS COLUMN, in the accumulator layout the write-side term reads.
  //: Its `A` operand is the staged seed broadcast down `m`, which the seed plane's
  //: own bf16 makes exact.
  float dzc[NSLOT][4];
#pragma unroll
  for (int n2 = 0; n2 < NSLOT; ++n2)
#pragma unroll
    for (int e = 0; e < 4; ++e) dzc[n2][e] = 0.f;

  TileStage<C, kD, LdV, kThreads> stV;
  ExpandR0<C, BCg, LdS, D, Bw> exR, exW;
  stV.init(tid);
  exR.init(gtid, (int)slot0);
  exW.init(gtid, (int)slot0);
  constexpr uint32_t kVSlotB = (uint32_t)(C * LdV * 2);
  constexpr uint32_t kVRingB = (uint32_t)STAGES * kVSlotB;
  const uint32_t vRowB = (uint32_t)v_row_bytes;
  constexpr uint32_t kDnRowB = (uint32_t)(kD * (int)sizeof(__nv_bfloat16));
  constexpr uint32_t kDvRowB = (uint32_t)(kD * (int)sizeof(float));
  uint32_t stgV = 0, cmpV = 0;

  auto stage_chunk = [&](int d) {
    const int* h = dw.ihdr(d);
    const uint32_t win = (uint32_t)h[3] * (uint32_t)kWindow;
    if (h[1] != 0) stV.issue(svB + stgV, vPlane, dw.iwrite(d), win, vRowB);
    if (h[0] != 0) {
      stV.issue(sdnB + stgV, dnPlane, dw.iread(d), win, kDnRowB);
      if (tid < C)
        sdn[(stgV >> 1) + R0_DDEN(LdV, tid)] =
            tid < h[0] ? ddPlane[(int)win + (int)dw.iread(d)[tid]] : __float2bfloat16(0.f);
    }
    stgV += kVSlotB;
    if (stgV == kVRingB) stgV = 0;
  };

  dw.init(tid);
  rendezvous();
  dw.form(tid);
  rendezvous();
#pragma unroll 1
  for (int s = 0; s < AHEAD; ++s) {
    if (ctawarp == 0) dw.walk(s, lane);
    rendezvous();
    dw.walk_wait(s, tid, ctawarp, lane);
    dw.form_if_low(tid);
    rendezvous();
  }
#pragma unroll 1
  for (int s = 0; s < STAGES - 1; ++s) {
    stage_chunk(s);
    stage_commit();
  }

  int dc = 0, ds = STAGES - 1, dn = AHEAD;
#pragma unroll 1
  for (;;) {
    const int* h = dw.ihdr(dc);
    const int nr = h[0], nw = h[1];
    if (!h[2]) break;
    const bool liveR = nr != 0, liveW = nw != 0;
    const uint32_t win = (uint32_t)h[3] * (uint32_t)kWindow;
    const uint16_t* const lwr = dw.iwrite(dc);
    const uint8_t* const ljt = dw.ithr(dc);

    if (ctawarp == 0) dw.walk(dn, lane);
    if (liveR) exR.load(prPlane, dw.iread(dc), win, nr);
    if (liveW) exW.load(pwPlane, lwr, win, nw);

    stage_wait<STAGES - 2>();
    rendezvous();  // (1) this super-chunk's tiles are visible; the last visit's panels,
                   //     Grams and published adjoint are dead; the walk has published

    dw.walk_wait(dn, tid, ctawarp, lane);
    dw.form_if_low(tid);
    stage_chunk(ds);
    stage_commit();

    if (liveR) exR.store(srThis);
    if (liveW) exW.store(swThis);

    const uint32_t aVF = svB + cmpV + offFv, aVT = svB + cmpV + offTv;
    const uint32_t aDF = sdnB + cmpV + offFv, aDT = sdnB + cmpV + offTv;
    const uint32_t aRF = srB + offFs;
    const uint32_t aWT = swB + offTs;
    const __nv_bfloat16* const dd = sdn + (cmpV >> 1);
    cmpV += kVSlotB;
    if (cmpV == kVRingB) cmpV = 0;

    //: THE ENTRY ADJOINT, PUBLISHED ONCE PER VISIT -- the whole channel axis, which
    //: is what lets the per-slot write-side term finish inside one unit.
#pragma unroll
    for (int nj = 0; nj < NDV; ++nj) {
      const int s0 = nj * kMmaN + (lane & 3) * 2;
      const int c0 = warp * kMmaM + (lane >> 2);
      store_bf16x2(&sds[c0 * LdDs + s0], pack_bf16x2(dst[nj][0], dst[nj][1]));
      store_bf16x2(&sds[(c0 + 8) * LdDs + s0], pack_bf16x2(dst[nj][2], dst[nj][3]));
    }

    rendezvous();  // (2) both coefficient panels and the published adjoint are visible.
                   //     THE SLOT GRAM IS WHY PASS 2 TAKES A THIRD RENDEZVOUS AND PASS 1
                   //     DOES NOT: `Gu = R . W^T` reads the panels this phase stored.

    // ---- both Grams: Gu = staircase(R . W^T) over slots, Gd over channels -----
    if (liveR && liveW && warp < GWARPS) {
#pragma unroll
      for (int m = 0; m < MPW; ++m) {
        const int tr = (gm0 + m) * 16 + (lane >> 2);
        const int j0 = (int)ljt[tr], j1 = (int)ljt[tr + 8];
        //: THE SLOT GRAM, one per GROUP: it contracts over the group's own slots and
        //: is therefore a partial of the block's, which is what makes `dv` a fan-in.
        float gu[GPAIR][2][4];
#pragma unroll
        for (int u = 0; u < GPAIR; ++u)
#pragma unroll
          for (int h2 = 0; h2 < 2; ++h2)
#pragma unroll
            for (int e = 0; e < 4; ++e) gu[u][h2][e] = 0.f;
#pragma unroll
        for (int kk = 0; kk < KD; ++kk) {
          uint32_t qa[4];
          load_frag(qa, R0_ADDR(aRF, LdS, (gm0 + m) * 16, 2 * kk));
#pragma unroll
          for (int u = 0; u < GPAIR; ++u) {
            uint32_t bf[4];
            load_frag(bf, R0_ADDR(aWT, LdS, (gn0 + u * 2) * 8, 2 * kk));
            mma(gu[u][0], qa, &bf[0]);
            mma(gu[u][1], qa, &bf[2]);
          }
        }
#pragma unroll
        for (int u = 0; u < GPAIR; ++u)
#pragma unroll
          for (int h2 = 0; h2 < 2; ++h2) {
            const int tc = (gn0 + u * 2 + h2) * 8 + (lane & 3) * 2;
            store_bf16x2(&sgu[tr * LdG + tc], pack_bf16x2(R0_STAIR(gu[u][h2][0], tc, j0),
                                                          R0_STAIR(gu[u][h2][1], tc + 1, j0)));
            store_bf16x2(
                &sgu[(tr + 8) * LdG + tc],
                pack_bf16x2(R0_STAIR(gu[u][h2][2], tc, j1), R0_STAIR(gu[u][h2][3], tc + 1, j1)));
          }
        //: THE CHANNEL GRAM, one for the CTA: it has no slot axis, so a second group
        //: would form the same object twice.
        if (grp == 0) {
          float gd[GPAIR][2][4];
#pragma unroll
          for (int u = 0; u < GPAIR; ++u)
#pragma unroll
            for (int h2 = 0; h2 < 2; ++h2)
#pragma unroll
              for (int e = 0; e < 4; ++e) gd[u][h2][e] = 0.f;
#pragma unroll
          for (int kk = 0; kk < KV; ++kk) {
            uint32_t qa[4];
            load_frag(qa, R0_ADDR(aDF, LdV, (gm0 + m) * 16, 2 * kk));
#pragma unroll
            for (int u = 0; u < GPAIR; ++u) {
              uint32_t bf[4];
              load_frag(bf, R0_ADDR(aVT, LdV, (gn0 + u * 2) * 8, 2 * kk));
              mma(gd[u][0], qa, &bf[0]);
              mma(gd[u][1], qa, &bf[2]);
            }
          }
          const float m0 = __bfloat162float(dd[ddG[m][0]]), m1 = __bfloat162float(dd[ddG[m][1]]);
#pragma unroll
          for (int u = 0; u < GPAIR; ++u)
#pragma unroll
            for (int h2 = 0; h2 < 2; ++h2) {
              const int tc = (gn0 + u * 2 + h2) * 8 + (lane & 3) * 2;
              store_bf16x2(&sgd[tr * LdG + tc],
                           pack_bf16x2(R0_STAIR(gd[u][h2][0] + m0, tc, j0),
                                       R0_STAIR(gd[u][h2][1] + m0, tc + 1, j0)));
              store_bf16x2(&sgd[(tr + 8) * LdG + tc],
                           pack_bf16x2(R0_STAIR(gd[u][h2][2] + m1, tc, j1),
                                       R0_STAIR(gd[u][h2][3] + m1, tc + 1, j1)));
            }
        }
      }
    }

    rendezvous();  // (3) both Grams are visible

    // ---- dv: the readout with the write panel, plus the slot Gram's staircase --
    if (liveW) {
      float acc[NTOK][4];
#pragma unroll
      for (int j = 0; j < NTOK; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[j][e] = 0.f;
#pragma unroll
      for (int kk = 0; kk < KD; ++kk) {
        uint32_t da[4];
        da[0] = pack_bf16x2(dst[2 * kk][0], dst[2 * kk][1]);
        da[1] = pack_bf16x2(dst[2 * kk][2], dst[2 * kk][3]);
        da[2] = pack_bf16x2(dst[2 * kk + 1][0], dst[2 * kk + 1][1]);
        da[3] = pack_bf16x2(dst[2 * kk + 1][2], dst[2 * kk + 1][3]);
#pragma unroll
        for (int np = 0; np < NTOK / 2; ++np) {
          uint32_t wb[4];
          load_frag(wb, R0_ADDR(aWT, LdS, np * 16, 2 * kk));
          mma(acc[2 * np], da, &wb[0]);
          mma(acc[2 * np + 1], da, &wb[2]);
        }
      }
      if (liveR) {
#pragma unroll
        for (int kk = 0; kk < MTOK; ++kk) {
          uint32_t a[4];
          load_frag_t(a, R0_ADDR(aDT, LdV, kk * 16, 2 * warp));
#pragma unroll
          for (int np = 0; np < NTOK / 2; ++np) {
            uint32_t b[4];
            load_frag_t(b, R0_ADDR(sguF, LdG, kk * 16, 2 * np));
            mma(acc[2 * np], a, &b[0]);
            mma(acc[2 * np + 1], a, &b[2]);
          }
        }
      }
      //: THE SPLIT-K FAN-IN.  `dv` contracts over the WHOLE slot space, so every
      //: group of every block holds a partial of it and the arena is the only place
      //: they meet.  The identity is written ONCE, by the allocation, before any
      //: contribution -- a per-output fill would be `nblk` fills of the same plane
      //: and a race besides.  `fan_in_add` is the non-returning form, so this
      //: assembles as `RED` and nothing waits -- docs/internals/chunk/ladder_r0.md#fan-in
      //: A COLUMN PAST THE GATHERED COUNT IS NOT DRAINED AT ALL, which is where the
      //: compaction's "skipped work contributes exactly zero" is executed for `dv`.
      const int c0 = warp * kMmaM + (lane >> 2);
      const uint32_t cB = (uint32_t)(c0 * (int)sizeof(float));
#pragma unroll
      for (int j = 0; j < NTOK; ++j) {
        const int t0 = j * kMmaN + (lane & 3) * 2;
        if (t0 < nw) {
          float* p0 = reinterpret_cast<float*>(
              gather_ptr(dvPlane, (win + (uint32_t)lwr[t0]) * kDvRowB + cB));
          fan_in_add(p0, acc[j][0]);
          fan_in_add(p0 + 8, acc[j][2]);
        }
        if (t0 + 1 < nw) {
          float* p1 = reinterpret_cast<float*>(
              gather_ptr(dvPlane, (win + (uint32_t)lwr[t0 + 1]) * kDvRowB + cB));
          fan_in_add(p1, acc[j][1]);
          fan_in_add(p1 + 8, acc[j][3]);
        }
      }
    }

    // ---- dW, over this unit's slot run ---------------------------------------
    if (liveW) {
      float acc[MTOK][NSLOT][4];
#pragma unroll
      for (int m = 0; m < MTOK; ++m)
#pragma unroll
        for (int n2 = 0; n2 < NSLOT; ++n2)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[m][n2][e] = 0.f;
      const int n0 = warp * NSLOT;
      //: the INTER term: the whole channel axis, out of the published adjoint.
#pragma unroll
      for (int kk = 0; kk < KV; ++kk)
#pragma unroll
        for (int np = 0; np < NPAIR; ++np) {
          uint32_t sb[4];
          load_frag_t(sb, R0_ADDR(sdsF, LdDs, kk * 16, n0 + np * 2));
#pragma unroll
          for (int m = 0; m < MTOK; ++m) {
            uint32_t av[4];
            load_frag(av, R0_ADDR(aVF, LdV, m * 16, 2 * kk));
            mma(acc[m][2 * np], av, &sb[0]);
            mma(acc[m][2 * np + 1], av, &sb[2]);
          }
        }
      //: the INTRA term: the TRANSPOSED staircase against the read panel.
      if (liveR) {
#pragma unroll
        for (int kk = 0; kk < MTOK; ++kk)
#pragma unroll
          for (int np = 0; np < NPAIR; ++np) {
            uint32_t rb[4];
            load_frag_t(rb, R0_ADDR(aRF, LdS, kk * 16, n0 + np * 2));
#pragma unroll
            for (int m = 0; m < MTOK; ++m) {
              uint32_t ga[4];
              load_frag_t(ga, R0_ADDR(sgdT, LdG, kk * 16, 2 * m));
              mma(acc[m][2 * np], ga, &rb[0]);
              mma(acc[m][2 * np + 1], ga, &rb[2]);
            }
          }
      }
      //: the ENTRY ADJOINT MASS, `dz_a[s]`: per-slot and constant down the write
      //: entries, so it is a broadcast along the accumulator's `m` axis.
#pragma unroll
      for (int m = 0; m < MTOK; ++m)
#pragma unroll
        for (int n2 = 0; n2 < NSLOT; ++n2) {
          acc[m][n2][0] += dzc[n2][0];
          acc[m][n2][1] += dzc[n2][1];
          acc[m][n2][2] += dzc[n2][0];
          acc[m][n2][3] += dzc[n2][1];
        }
      //: (A.10), THE WRITE SIDE'S DIGIT REDUCTION.  Same component as pass 1's, with
      //: the write amplitudes supplying the leave-one-out factor.
      r0_emit_digits<BC, BCg, D, Bw, MTOK, NSLOT>(acc, pwPlane, dpwPlane, lwr, win, nw, sbaseU,
                                                  lane);
    }

    // ---- the fold: dS^T += dnum^T . R, and dz += dden^T . R ------------------
    if (liveR) {
#pragma unroll
      for (int kk = 0; kk < MTOK; ++kk) {
        uint32_t a[4];
        load_frag_t(a, R0_ADDR(aDT, LdV, kk * 16, 2 * warp));
#pragma unroll
        for (int nj = 0; nj < KD; ++nj) {
          uint32_t b[4];
          load_frag_t(b, R0_ADDR(aRF, LdS, kk * 16, 2 * nj));
          mma(dst[2 * nj], a, &b[0]);
          mma(dst[2 * nj + 1], a, &b[2]);
        }
      }
      const int n0 = warp * NSLOT;
#pragma unroll
      for (int kk = 0; kk < MTOK; ++kk) {
        //: the seed broadcast down `m`: a lane's A fragment is one k-pair repeated,
        //: and the pad's eight-wide groups are what make that pair adjacent.
        uint32_t da[4];
        da[0] = da[1] = *reinterpret_cast<const uint32_t*>(&dd[ddK[kk][0]]);
        da[2] = da[3] = *reinterpret_cast<const uint32_t*>(&dd[ddK[kk][1]]);
#pragma unroll
        for (int np = 0; np < NPAIR; ++np) {
          uint32_t rb[4];
          load_frag_t(rb, R0_ADDR(aRF, LdS, kk * 16, n0 + np * 2));
          mma(dzc[2 * np], da, &rb[0]);
          mma(dzc[2 * np + 1], da, &rb[2]);
        }
      }
    }

    if (++dc == IRING) dc = 0;
    if (++ds == IRING) ds = 0;
    if (++dn == IRING) dn = 0;
  }
}

#undef R0_ADDR
#undef R0_STAIR
#undef R0_DDEN
#undef R0_CLAUSE

// THE RUNG'S ARM SET, GENERATED: tools/gen_shards.py owns the one list.
#include "../instantiations/ladder_r0_arms.inc"

}  // namespace rola::chunk
