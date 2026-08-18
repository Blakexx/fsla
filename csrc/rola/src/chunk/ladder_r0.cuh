// csrc/rola/src/chunk/ladder_r0.cuh -- THE BACKWARD LADDER, RUNGS 0-3: the chunked
// linear-attention backward on ROUTED coefficients, written in RoLA's own
// quantities as a two-pass pair.  Pass 1 ascends, rebuilding the state and its mass
// column from zero and emitting the READ side's per-level digit gradient; pass 2
// descends, carrying both adjoints and emitting the value cotangent and the WRITE
// side's.  The slot space is carved into blocks; `dv` and both digit gradients fan
// in across them.  The operands are PER LEVEL and the Kronecker product is formed
// in kernel.  Contiguous chunks, no write gain, no compaction -- those are later
// rungs.  docs/internals/chunk/ladder_r0.md
#pragma once

#include "chunk_kernel.cuh"

namespace rola::chunk {

//: The ladder's launch arguments.  The coefficients are PER LEVEL -- `[BH, L, D*Bw]`
//: amplitude rows, one packed block of `Bw` columns per level -- and so are the two
//: gradients they carry.  The slot space `N = Bw^D` is a COMPILED quantity now, not a
//: runtime width: the block lattice `N / BC` is the grid's x extent and the digit
//: structure a block sees is the geometry it is instantiated at.
struct Rung0Params {
  const __nv_bfloat16 *pr, *pw, *v, *dnum, *dden;
  float *dpr, *dv, *dpw;
  int BH, L, v_row_bytes;
};

//: THE RUNG'S LIVE REGISTER SET, one measured DECLARATION for both passes and
//: falsified per arm by the census' spill and budget columns.  It is a single number
//: rather than a `BC` row because a GROUP's geometry does not move with the block
//: width; the pass that peaks is the descending one, which carries the adjoint state
//: through every block.  Re-measured at b2: peak 157, and 176 is the declaration --
//: the floor is also the budget `cta_target` is DERIVED from, and a declaration below
//: 171 derives a third CTA at the `(16, 64)` arm, which is a residency decision and
//: not the side effect of a re-measurement.
constexpr int reg_floor_r0() { return 176; }

//: THE LADDER'S OWN GROUP THRESHOLD, and it is LOWER than the forward's (which
//: splits at `BC = 256`, `groups_for`).  The adjoint carries two costs the forward
//: does not: a SECOND coefficient panel on the ring, and the PUBLICATION -- the
//: `[d_v, BCg]` transpose of the state that lets a unit finish a contraction over
//: the carved channel axis.  Both scale with the group's slot width, and at
//: `BCg = 128` their sum puts the ring 24 KB past the two-CTA ceiling with no
//: overlay available, because all four operand buffers are live across the whole
//: visit.  Splitting the BLOCK into two 64-slot groups holds the ring at the width
//: that fits and keeps the warps per SM at the forward's eight -- one CTA of eight
//: warps where `BC = 64` runs two of four -- while the value ring and the channel
//: Gram, which are slot-independent, are shared rather than duplicated.
//: docs/internals/chunk/ladder_r0.md#the-group-threshold
constexpr int r0_groups_for(int BC) { return BC >= 128 ? 2 : 1; }
constexpr int r0_threads_for(int BC) { return kGroupThreads * r0_groups_for(BC); }

//: THE LADDER'S GEOMETRY.  Every stream the kernel consumes over time rides the ring:
//: the two value-shaped panels always, the write coefficients always, the read
//: coefficients in the pass that carries the adjoint, and the readout's mass seed
//: inside the `dnum` panel's own row pad, which costs no bytes at all.  The
//: published state's rows are UNPADDED -- it is read four times per unit per visit,
//: where the pad would buy nothing and cost the CTA target.  The GROUP is the unit
//: of coefficient, Gram and publication geometry; the value ring and the `d_v` Gram
//: are CTA-wide because neither carries a slot axis.  At `GRP = 1` every line below
//: is b1's byte for byte -- docs/internals/chunk/ladder_r0.md
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
  static constexpr int slot =
      (2 * C * LdV + GRP * coef_panels * C * LdS) * (int)sizeof(__nv_bfloat16);
  static constexpr int gram_bytes = (GRP * slot_grams + 1) * C * LdG * (int)sizeof(__nv_bfloat16);
  static constexpr int ds_bytes = GRP * kD * LdDs * (int)sizeof(__nv_bfloat16);
  static constexpr int fixed = gram_bytes + ds_bytes;
  static constexpr int stages = design::stages_for(slot, fixed, r0_threads_for(BC), reg_floor_r0());
  static constexpr size_t bytes = (size_t)stages * slot + fixed;
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

//: THE KRONECKER EXPANSION, IN KERNEL.  A thread owns one token row and an
//: EIGHT-SLOT run of its group's panel.  The run is eight-aligned and `Bw >= 8`, so
//: it lies inside ONE outer digit and covers eight CONSECUTIVE inner ones -- which is
//: why the product is one splat times one vec8 and never a per-slot gather, and why
//: the expansion costs one `mul_bf16x2` per two slots.  The panel it fills is
//: byte-for-byte the panel rung b2 staged, so nothing downstream of it moved.
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

  __device__ __forceinline__ void init(int gtid, int sbase) {
    row = gtid / gran;
    const int g0 = (gtid % gran) * 8;
    dstOff = (uint32_t)((row * LdS + g0) * (int)sizeof(__nv_bfloat16));
    const int s = sbase + g0;
    colInnerB = (uint32_t)(((D - 1) * Bw + (s % Bw)) * (int)sizeof(__nv_bfloat16));
    colOuterB = (uint32_t)((s / Bw) * (int)sizeof(__nv_bfloat16));
  }

  __device__ __forceinline__ void emit(char* panel, const void* plane, uint32_t tok0) const {
    constexpr uint32_t kRowB = (uint32_t)(D * Bw * (int)sizeof(__nv_bfloat16));
    constexpr int kPassB = rows * LdS * (int)sizeof(__nv_bfloat16);
#pragma unroll
    for (int p = 0; p < passes; ++p) {
      const uint32_t rowB = (tok0 + (uint32_t)(row + p * rows)) * kRowB;
      __align__(16) uint32_t a[4];
      *reinterpret_cast<uint4*>(a) = load_vec16(gather_ptr(plane, rowB + colInnerB));
      if constexpr (D >= 2) {
        const uint32_t up = splat_bf16x2(
            *reinterpret_cast<const __nv_bfloat16*>(gather_ptr(plane, rowB + colOuterB)));
#pragma unroll
        for (int e = 0; e < 4; ++e) a[e] = mul_bf16x2(up, a[e]);
      }
      store_vec16(panel + dstOff + p * kPassB, *reinterpret_cast<const uint4*>(a));
    }
  }
};

//: A [C, COLS] tile of a contiguous plane, staged asynchronously.  The ladder's chunks
//: are contiguous token runs, so a tile's rows are the plane's own and the schedule
//: contributes no gather.
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

  __device__ __forceinline__ void issue(uint32_t dst, const void* plane, uint32_t tok0,
                                        uint32_t rowBytes) const {
#pragma unroll
    for (int p = 0; p < passes; ++p) {
      const uint32_t off = (tok0 + (uint32_t)(row + p * rows)) * rowBytes + colBytes;
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
template <int BC, int BCg, int D, int Bw, int MTOK, int NSLOT>
__device__ __forceinline__ void r0_emit_digits(const float (&acc)[MTOK][NSLOT][4],
                                               const void* pPlane, void* dPlane, uint32_t tokBase,
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
          const uint32_t tk = tokBase + (uint32_t)(m * kMmaM + (lane >> 2) + h * 8);
          fo[o][m][h] = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
              gather_ptr(pPlane, tk * DG::row_bytes_p + cOut)));
          u0[o][m][h] = 0.f;
        }
    }
  }
#pragma unroll
  for (int n = 0; n < NSLOT; ++n) {
    const int o = n / DG::npo;
    const int i1 = DG::inner_base + ((sbase + n * kMmaN) % Bw) + (lane & 3) * 2;
    const uint32_t cInD = (uint32_t)(i1 * (int)sizeof(float));
#pragma unroll
    for (int m = 0; m < MTOK; ++m)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const uint32_t tk = tokBase + (uint32_t)(m * kMmaM + (lane >> 2) + h * 8);
        const float a0 = acc[m][n][2 * h], a1 = acc[m][n][2 * h + 1];
        float f = 1.f;
        if constexpr (D >= 2) f = fo[o][m][h];
        float* p = reinterpret_cast<float*>(gather_ptr(dPlane, tk * DG::row_bytes_d + cInD));
        fan_in_add(p, a0 * f);
        fan_in_add(p + 1, a1 * f);
        if constexpr (D >= 2) {
          const uint32_t q = *reinterpret_cast<const uint32_t*>(gather_ptr(
              pPlane, tk * DG::row_bytes_p + (uint32_t)(i1 * (int)sizeof(__nv_bfloat16))));
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
          if ((lane & 3) == 0) {
            const uint32_t tk = tokBase + (uint32_t)(m * kMmaM + (lane >> 2) + h * 8);
            fan_in_add(reinterpret_cast<float*>(gather_ptr(dPlane, tk * DG::row_bytes_d + cOut)),
                       x);
          }
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

//: THE STAIRCASE, DENSE: chunk-local token indices, and a write entry is kept when
//: its token is at or before the read entry's.
#define R0_STAIR(v_, tc_, tr_) ((tc_) <= (tr_) ? (v_) : 0.f)

//: PASS 1 -- the ascending rebuild.  It carries `S` and its mass column `z`, forms
//: `dR[i,s] = <dnum_i, S_a[s,:]> + dden_i z_a[s] + sum_{j<=i} W[j,s] Gd'[i,j]` with
//: `Gd'[i,j] = <dnum_i, v_j> + dden_i`, and emits its DIGIT REDUCTION (A.9).  The
//: write panel is the Kronecker product of `pw`'s levels, expanded in kernel; `pr`
//: is read only at the emit, where it supplies the leave-one-out factor.
template <int C, int BC, int D, int Bw>
__global__ __launch_bounds__(r0_threads_for(BC), (kR0CtaTarget<C, BC, false>)) void
rung0_pass1_kernel(const __nv_bfloat16* __restrict__ pread,
                   const __nv_bfloat16* __restrict__ pwrite, const __nv_bfloat16* __restrict__ v,
                   const __nv_bfloat16* __restrict__ dnum, const __nv_bfloat16* __restrict__ dden,
                   float* __restrict__ dpr, int L, int v_row_bytes) {
  using G = GeomR0<C, BC, false>;
  using DG = DigitR0<D, Bw, BC, G::BCg>;
  constexpr int kThreads = r0_threads_for(BC);
  constexpr int kWarps = kThreads / 32;
  constexpr int GRP = G::GRP, BCg = G::BCg;
  constexpr int STAGES = G::stages;
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
  static_assert(C % 8 == 0 && kPadE >= 8 && C / 8 <= C,
                "the mass seed seats in the value panel's row pad, eight entries to a row");
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
  const int warp = gtid >> 5;

  extern __shared__ __align__(16) char smem_raw[];
  __nv_bfloat16* sv = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* sdn = sv + STAGES * C * LdV;
  __nv_bfloat16* sw = sdn + STAGES * C * LdV;
  __nv_bfloat16* sg = sw + STAGES * GRP * C * LdS;
  __nv_bfloat16* sds = sg + C * LdG + grp * kD * LdDs;

  const int bh = (int)blockIdx.y;
  //: THE BLOCK LATTICE.  A block owns `BC` of the `N` slots and a group owns `BCg`
  //: of the block's; the per-slot outputs are DISJOINT over that partition, so
  //: nothing about `dR` fans in -- what fans in is `dv`, in pass 2.
  const long slot0 = (long)blockIdx.x * BC + (long)grp * BCg;
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

  constexpr uint32_t kGrpS = (uint32_t)(C * LdS * (int)sizeof(__nv_bfloat16));
  //: THE UNIT'S FIRST SLOT, hoisted: it is the digit geometry's own base and does not
  //: move with the chunk.
  const int sbaseU = (int)slot0 + warp * (BCg / kUnits);
  const uint32_t svB = smem_addr(sv), sdnB = smem_addr(sdn);
  const uint32_t swB = smem_addr(sw) + (uint32_t)grp * kGrpS;
  const uint32_t sgF = smem_addr(sg) + offFg;
  const uint32_t sdsF = smem_addr(sds) + offFd;

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
  for (int n = 0; n < NSLOT; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) zac[n][e] = 0.f;
  const uint32_t ones[4] = {kOnesFrag, kOnesFrag, kOnesFrag, kOnesFrag};

  //: the value ring is staged by the WHOLE CTA and the coefficient ring by each
  //: group over its own slot slice, which is why the two carry different thread
  //: counts and the coefficient one is indexed by the in-group thread.
  TileStage<C, kD, LdV, kThreads> stV;
  //: THE COEFFICIENT PANEL IS PRODUCED, NOT STAGED.  Its bytes, its pitch and its
  //: place in the ring are byte-for-byte rung b2's; what changed is that the ring
  //: slot is filled by the expansion out of `D` amplitude columns instead of by
  //: `cp.async` out of `BCg` per-slot ones.
  ExpandR0<C, BCg, LdS, D, Bw> exW;
  stV.init(tid);
  exW.init(gtid, (int)slot0);
  constexpr uint32_t kVSlotB = (uint32_t)(C * LdV * 2);
  constexpr uint32_t kSSlotB = (uint32_t)(GRP * C * LdS * 2);
  constexpr uint32_t kVRingB = (uint32_t)STAGES * kVSlotB;
  constexpr uint32_t kSRingB = (uint32_t)STAGES * kSSlotB;
  const uint32_t vRowB = (uint32_t)v_row_bytes;
  constexpr uint32_t kDnRowB = (uint32_t)(kD * (int)sizeof(__nv_bfloat16));
  uint32_t stgV = 0, cmpV = 0, stgS = 0, cmpS = 0;
  char* const swThis = reinterpret_cast<char*>(sw) + (size_t)grp * kGrpS;

  const int NC = L / C;
  auto stage_chunk = [&](int c) {
    if (c < NC) {
      const uint32_t tok0 = (uint32_t)(c * C);
      stV.issue(svB + stgV, vPlane, tok0, vRowB);
      stV.issue(sdnB + stgV, dnPlane, tok0, kDnRowB);
      exW.emit(swThis + stgS, pwPlane, tok0);
      if (tid < C) sdn[(stgV >> 1) + R0_DDEN(LdV, tid)] = ddPlane[(int)tok0 + tid];
    }
    stgV += kVSlotB;
    if (stgV == kVRingB) stgV = 0;
    stgS += kSSlotB;
    if (stgS == kSRingB) stgS = 0;
  };

#pragma unroll 1
  for (int s = 0; s < STAGES - 1; ++s) {
    stage_chunk(s);
    stage_commit();
  }

#pragma unroll 1
  for (int c = 0; c < NC; ++c) {
    stage_wait<STAGES - 2>();
    rendezvous();  // (1) this chunk's tiles are visible; the last visit's Gram and
                   //     published state are dead
    stage_chunk(c + STAGES - 1);
    stage_commit();

    const uint32_t aVT = svB + cmpV + offTv;
    const uint32_t aDF = sdnB + cmpV + offFv;
    const uint32_t aWF = swB + cmpS + offFs;
    const __nv_bfloat16* const dd = sdn + (cmpV >> 1);
    const uint32_t tokBase = (uint32_t)(c * C);
    cmpV += kVSlotB;
    if (cmpV == kVRingB) cmpV = 0;
    cmpS += kSSlotB;
    if (cmpS == kSRingB) cmpS = 0;

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
    //: axis, so a second group would form the same object twice.
    if (grp == 0 && warp < GWARPS) {
#pragma unroll
      for (int m = 0; m < MPW; ++m) {
        float g[GPAIR][2][4];
#pragma unroll
        for (int u = 0; u < GPAIR; ++u)
#pragma unroll
          for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int e = 0; e < 4; ++e) g[u][h][e] = 0.f;
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
#pragma unroll
        for (int u = 0; u < GPAIR; ++u)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int tc = (gn0 + u * 2 + h) * 8 + (lane & 3) * 2;
            store_bf16x2(&sg[tr * LdG + tc], pack_bf16x2(R0_STAIR(g[u][h][0] + m0, tc, tr),
                                                         R0_STAIR(g[u][h][1] + m0, tc + 1, tr)));
            store_bf16x2(&sg[(tr + 8) * LdG + tc],
                         pack_bf16x2(R0_STAIR(g[u][h][2] + m1, tc, tr + 8),
                                     R0_STAIR(g[u][h][3] + m1, tc + 1, tr + 8)));
          }
      }
    }

    rendezvous();  // (2) the published state and the Gram are visible

    // ---- dR, over this unit's slot run ---------------------------------------
    {
      float acc[MTOK][NSLOT][4];
#pragma unroll
      for (int m = 0; m < MTOK; ++m)
#pragma unroll
        for (int n = 0; n < NSLOT; ++n)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[m][n][e] = 0.f;
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
      //: contraction over the chunk's write tokens with the Gram as its A operand.
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
      //: the ENTRY MASS, `dden_i z_a[s]`: an OUTER product of a per-read-token
      //: value and a per-slot one, so it is a rank-1 add and not a broadcast.
#pragma unroll
      for (int m = 0; m < MTOK; ++m) {
        const float d0 = __bfloat162float(dd[ddM[m][0]]);
        const float d1 = __bfloat162float(dd[ddM[m][1]]);
#pragma unroll
        for (int n = 0; n < NSLOT; ++n) {
          acc[m][n][0] += d0 * zac[n][0];
          acc[m][n][1] += d0 * zac[n][1];
          acc[m][n][2] += d1 * zac[n][0];
          acc[m][n][3] += d1 * zac[n][1];
        }
      }
      //: (A.9), THE READ SIDE'S DIGIT REDUCTION.  `dR` is consumed here and never
      //: stored: what leaves the kernel is `D` columns per token, not `BCg`.
      r0_emit_digits<BC, BCg, D, Bw, MTOK, NSLOT>(acc, prPlane, dprPlane, tokBase, sbaseU, lane);
    }

    // ---- the fold: S^T += V^T . W, and z += 1^T . W --------------------------
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
    {
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
  }
}

//: PASS 2 -- the descending adjoint.  It carries `dS` and its mass column `dz`, and
//: emits `dv[j,:] = sum_s W[j,s] dS_after[j,s,:]` and the DIGIT REDUCTION (A.10) of
//: `dW[j,s] = <dS_a[s,:], v_j> + dz_a[s] + sum_{i>=j} R[i,s] Gd'[i,j]`.  `dv` is a
//: value-channel quantity and the mass column does not enter it.  Both panels are
//: Kronecker products expanded in kernel; `pw` also supplies the write side's
//: leave-one-out factor at the emit.
template <int C, int BC, int D, int Bw>
__global__ __launch_bounds__(r0_threads_for(BC), (kR0CtaTarget<C, BC, true>)) void
rung0_pass2_kernel(const __nv_bfloat16* __restrict__ pread,
                   const __nv_bfloat16* __restrict__ pwrite, const __nv_bfloat16* __restrict__ v,
                   const __nv_bfloat16* __restrict__ dnum, const __nv_bfloat16* __restrict__ dden,
                   float* __restrict__ dv, float* __restrict__ dpw, int L, int v_row_bytes) {
  using G = GeomR0<C, BC, true>;
  using DG = DigitR0<D, Bw, BC, G::BCg>;
  constexpr int kThreads = r0_threads_for(BC);
  constexpr int kWarps = kThreads / 32;
  constexpr int GRP = G::GRP, BCg = G::BCg;
  constexpr int STAGES = G::stages;
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
  static_assert(C % 8 == 0 && kPadE >= 8 && C / 8 <= C,
                "the mass seed seats in the value panel's row pad, eight entries to a row");
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
  const int warp = gtid >> 5;

  extern __shared__ __align__(16) char smem_raw[];
  __nv_bfloat16* sv = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* sdn = sv + STAGES * C * LdV;
  __nv_bfloat16* sr = sdn + STAGES * C * LdV;
  __nv_bfloat16* sw = sr + STAGES * GRP * C * LdS;
  //: the SLOT Gram is one per group; the channel Gram is one for the CTA.
  __nv_bfloat16* sguAll = sw + STAGES * GRP * C * LdS;
  __nv_bfloat16* sgu = sguAll + grp * C * LdG;
  __nv_bfloat16* sgd = sguAll + GRP * C * LdG;
  __nv_bfloat16* sds = sgd + C * LdG + grp * kD * LdDs;

  const int bh = (int)blockIdx.y;
  const long slot0 = (long)blockIdx.x * BC + (long)grp * BCg;
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

  constexpr uint32_t kGrpS = (uint32_t)(C * LdS * (int)sizeof(__nv_bfloat16));
  //: THE UNIT'S FIRST SLOT, hoisted: it is the digit geometry's own base and does not
  //: move with the chunk.
  const int sbaseU = (int)slot0 + warp * (BCg / kUnits);
  const uint32_t svB = smem_addr(sv), sdnB = smem_addr(sdn);
  const uint32_t srB = smem_addr(sr) + (uint32_t)grp * kGrpS;
  const uint32_t swB = smem_addr(sw) + (uint32_t)grp * kGrpS;
  const uint32_t sguF = smem_addr(sgu) + offFg;
  const uint32_t sgdT = smem_addr(sgd) + offTg;
  const uint32_t sdsF = smem_addr(sds) + offFd;

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
  for (int n = 0; n < NSLOT; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) dzc[n][e] = 0.f;

  TileStage<C, kD, LdV, kThreads> stV;
  ExpandR0<C, BCg, LdS, D, Bw> ex;
  stV.init(tid);
  ex.init(gtid, (int)slot0);
  constexpr uint32_t kVSlotB = (uint32_t)(C * LdV * 2);
  constexpr uint32_t kSSlotB = (uint32_t)(GRP * C * LdS * 2);
  constexpr uint32_t kVRingB = (uint32_t)STAGES * kVSlotB;
  constexpr uint32_t kSRingB = (uint32_t)STAGES * kSSlotB;
  const uint32_t vRowB = (uint32_t)v_row_bytes;
  constexpr uint32_t kDnRowB = (uint32_t)(kD * (int)sizeof(__nv_bfloat16));
  constexpr uint32_t kDvRowB = (uint32_t)(kD * (int)sizeof(float));
  uint32_t stgV = 0, cmpV = 0, stgS = 0, cmpS = 0;
  char* const srThis = reinterpret_cast<char*>(sr) + (size_t)grp * kGrpS;
  char* const swThis = reinterpret_cast<char*>(sw) + (size_t)grp * kGrpS;

  const int NC = L / C;
  auto stage_chunk = [&](int c) {
    if (c >= 0) {
      const uint32_t tok0 = (uint32_t)(c * C);
      stV.issue(svB + stgV, vPlane, tok0, vRowB);
      stV.issue(sdnB + stgV, dnPlane, tok0, kDnRowB);
      ex.emit(srThis + stgS, prPlane, tok0);
      ex.emit(swThis + stgS, pwPlane, tok0);
      if (tid < C) sdn[(stgV >> 1) + R0_DDEN(LdV, tid)] = ddPlane[(int)tok0 + tid];
    }
    stgV += kVSlotB;
    if (stgV == kVRingB) stgV = 0;
    stgS += kSSlotB;
    if (stgS == kSRingB) stgS = 0;
  };

#pragma unroll 1
  for (int s = 0; s < STAGES - 1; ++s) {
    stage_chunk(NC - 1 - s);
    stage_commit();
  }

#pragma unroll 1
  for (int c = NC - 1; c >= 0; --c) {
    stage_wait<STAGES - 2>();
    rendezvous();  // (1) this chunk's tiles are visible; the last visit's Grams and
                   //     published state are dead
    stage_chunk(c - (STAGES - 1));
    stage_commit();

    const uint32_t aVF = svB + cmpV + offFv, aVT = svB + cmpV + offTv;
    const uint32_t aDF = sdnB + cmpV + offFv, aDT = sdnB + cmpV + offTv;
    const uint32_t aRF = srB + cmpS + offFs;
    const uint32_t aWT = swB + cmpS + offTs;
    const __nv_bfloat16* const dd = sdn + (cmpV >> 1);
    const uint32_t tokBase = (uint32_t)(c * C);
    cmpV += kVSlotB;
    if (cmpV == kVRingB) cmpV = 0;
    cmpS += kSSlotB;
    if (cmpS == kSRingB) cmpS = 0;

    //: THE ENTRY ADJOINT, PUBLISHED ONCE PER VISIT -- the whole channel axis, which
    //: is what lets the per-slot write-side term finish inside one unit.
#pragma unroll
    for (int nj = 0; nj < NDV; ++nj) {
      const int s0 = nj * kMmaN + (lane & 3) * 2;
      const int c0 = warp * kMmaM + (lane >> 2);
      store_bf16x2(&sds[c0 * LdDs + s0], pack_bf16x2(dst[nj][0], dst[nj][1]));
      store_bf16x2(&sds[(c0 + 8) * LdDs + s0], pack_bf16x2(dst[nj][2], dst[nj][3]));
    }

    // ---- both Grams: Gu = staircase(R . W^T) over slots, Gd over channels -----
    if (warp < GWARPS) {
#pragma unroll
      for (int m = 0; m < MPW; ++m) {
        const int tr = (gm0 + m) * 16 + (lane >> 2);
        //: THE SLOT GRAM, one per GROUP: it contracts over the group's own slots and
        //: is therefore a partial of the block's, which is what makes `dv` a fan-in.
        float gu[GPAIR][2][4];
#pragma unroll
        for (int u = 0; u < GPAIR; ++u)
#pragma unroll
          for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int e = 0; e < 4; ++e) gu[u][h][e] = 0.f;
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
          for (int h = 0; h < 2; ++h) {
            const int tc = (gn0 + u * 2 + h) * 8 + (lane & 3) * 2;
            store_bf16x2(&sgu[tr * LdG + tc], pack_bf16x2(R0_STAIR(gu[u][h][0], tc, tr),
                                                          R0_STAIR(gu[u][h][1], tc + 1, tr)));
            store_bf16x2(&sgu[(tr + 8) * LdG + tc],
                         pack_bf16x2(R0_STAIR(gu[u][h][2], tc, tr + 8),
                                     R0_STAIR(gu[u][h][3], tc + 1, tr + 8)));
          }
        //: THE CHANNEL GRAM, one for the CTA: it has no slot axis, so a second group
        //: would form the same object twice.
        if (grp == 0) {
          float gd[GPAIR][2][4];
#pragma unroll
          for (int u = 0; u < GPAIR; ++u)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
              for (int e = 0; e < 4; ++e) gd[u][h][e] = 0.f;
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
            for (int h = 0; h < 2; ++h) {
              const int tc = (gn0 + u * 2 + h) * 8 + (lane & 3) * 2;
              store_bf16x2(&sgd[tr * LdG + tc],
                           pack_bf16x2(R0_STAIR(gd[u][h][0] + m0, tc, tr),
                                       R0_STAIR(gd[u][h][1] + m0, tc + 1, tr)));
              store_bf16x2(&sgd[(tr + 8) * LdG + tc],
                           pack_bf16x2(R0_STAIR(gd[u][h][2] + m1, tc, tr + 8),
                                       R0_STAIR(gd[u][h][3] + m1, tc + 1, tr + 8)));
            }
        }
      }
    }

    rendezvous();  // (2) the published adjoint and both Grams are visible

    // ---- dv: the readout with the write panel, plus the slot Gram's staircase --
    {
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
      //: THE SPLIT-K FAN-IN.  `dv` contracts over the WHOLE slot space, so every
      //: group of every block holds a partial of it and the arena is the only place
      //: they meet.  The identity is written ONCE, by the allocation, before any
      //: contribution -- a per-output fill would be `nblk` fills of the same plane
      //: and a race besides.  `fan_in_add` is the non-returning form, so this
      //: assembles as `RED` and nothing waits -- docs/internals/chunk/ladder_r0.md#fan-in
      const int c0 = warp * kMmaM + (lane >> 2);
#pragma unroll
      for (int j = 0; j < NTOK; ++j) {
        const int t0 = j * kMmaN + (lane & 3) * 2;
        float* p0 = reinterpret_cast<float*>(gather_ptr(
            dvPlane, (tokBase + (uint32_t)t0) * kDvRowB + (uint32_t)(c0 * (int)sizeof(float))));
        float* p1 = reinterpret_cast<float*>(gather_ptr(
            dvPlane,
            (tokBase + (uint32_t)(t0 + 1)) * kDvRowB + (uint32_t)(c0 * (int)sizeof(float))));
        fan_in_add(p0, acc[j][0]);
        fan_in_add(p1, acc[j][1]);
        fan_in_add(p0 + 8, acc[j][2]);
        fan_in_add(p1 + 8, acc[j][3]);
      }
    }

    // ---- dW, over this unit's slot run ---------------------------------------
    {
      float acc[MTOK][NSLOT][4];
#pragma unroll
      for (int m = 0; m < MTOK; ++m)
#pragma unroll
        for (int n = 0; n < NSLOT; ++n)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[m][n][e] = 0.f;
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
      //: the ENTRY ADJOINT MASS, `dz_a[s]`: per-slot and constant down the write
      //: tokens, so it is a broadcast along the accumulator's `m` axis.
#pragma unroll
      for (int m = 0; m < MTOK; ++m)
#pragma unroll
        for (int n = 0; n < NSLOT; ++n) {
          acc[m][n][0] += dzc[n][0];
          acc[m][n][1] += dzc[n][1];
          acc[m][n][2] += dzc[n][0];
          acc[m][n][3] += dzc[n][1];
        }
      //: (A.10), THE WRITE SIDE'S DIGIT REDUCTION.  Same component as pass 1's, with
      //: the write amplitudes supplying the leave-one-out factor.
      r0_emit_digits<BC, BCg, D, Bw, MTOK, NSLOT>(acc, pwPlane, dpwPlane, tokBase, sbaseU, lane);
    }

    // ---- the fold: dS^T += dnum^T . R, and dz += dden^T . R ------------------
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
    {
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
  }
}

#undef R0_ADDR
#undef R0_STAIR
#undef R0_DDEN

// THE RUNG'S ARM SET, GENERATED: tools/gen_shards.py owns the one list.
#include "../instantiations/ladder_r0_arms.inc"

}  // namespace rola::chunk
