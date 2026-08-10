// LADDER RUNG 1 -- routed Kronecker coefficient production.
//
// Base: the hardened dense chunked linear attention of `denseref/` at commit
// d1cde36 -- two barriers per chunk, the deferred output store that doubles as
// the ring's WAR fence, whole-k-run fragment prefetch, the pad/XOR-swizzle arm
// as a template parameter, and the architecture capability layer.  This file
// therefore names NO ARCHITECTURE and issues NO INLINE PTX either; it is written
// against `ops.cuh`'s baseline operations exactly as the base is.
//
// EXACTLY ONE FEATURE is added and nothing else is touched: the dense per-token
// coefficient vectors Q[t][.] and K[t][.] are replaced by coefficients PRODUCED
// IN-KERNEL from a factored (Kronecker) form.
//
// SEMANTICS
// ---------
// A memory slot index n in [0, N) is a MIXED-RADIX number over D routing
// levels, most-significant digit first:
//     n = sum_l digit_l(n) * stride_l ,  stride_l = prod_{m>l} W_m
// and a token's coefficient for that slot is the product across levels of the
// per-level amplitude selected by that slot's digit:
//     R[t][n] = prod_l p_read [l][t][digit_l(n)]        (the Q role)
//     W[t][n] = prod_l p_write[l][t][digit_l(n)]        (the K role)
// Everything downstream is byte-for-byte the base algorithm with Q -> R and
// K -> W:
//     O_c  = R_c . S ;  O_c += tril(R_c . W_c^T) . V_c ;  S += W_c^T . V_c
// N is UNCHANGED at 64, so the state is the same N x d_v = 64 x 64 and it stays
// in registers, as the fold's accumulator, never in shared memory.  No
// schedule, no membership test, no atom mask, no decay, no normalisation, no
// capacity split.
//
// THE STATE IS THE READOUT'S A OPERAND
// ------------------------------------
// The fold accumulates S^T[d_v][slot] with m = d_v and n = slot.  For m16n8k16
// a lane's four fp32 of an accumulator tile are
//     (m = l/4, l/4+8) x (n = 2*(l%4), +1)
// and the four b32 of an A fragment are
//     (m = l/4, l/4+8) x (k = 2*(l%4), +1, +8, +9),
// so ONE A fragment spans TWO accumulator tiles along n, and its four halves
// are in order the (lo, hi) of tile 2k and of tile 2k+1:
//     a0 = pack(st[2k  ][0], st[2k  ][1])   a1 = pack(st[2k  ][2], st[2k  ][3])
//     a2 = pack(st[2k+1][0], st[2k+1][1])   a3 = pack(st[2k+1][2], st[2k+1][3])
// That is a RELABEL of the very same four `pack_bf16x2` results a B arrangement
// produces from the same tiles: no store, no reload, no shuffle, and the same
// conversion count.  The readout is then O^T = S^T . R^T -- state in A, the
// routing panel in B with n = tokens and k = slots, which is the panel's SAME
// [token][slot] buffer read through the other ldmatrix lane-offset form -- and
// the output accumulator transposes to (m = d_v) x (n = tokens).  The intra
// term follows it: O^T += V^T . G^T, whose A operand is V^T, the fragment the
// fold already loads.  The fold and the Gram are untouched.
//
// WHAT THE TRANSPOSE COSTS, AND IT IS THE ONLY THING IT COSTS
// -----------------------------------------------------------
// A transposed output tile holds two TOKEN-adjacent values of one d_v column.
// The output buffer is [token][d_v], because the drain out of it must be a
// fully coalesced 128-bit store, so those two are in different ROWS and the
// epilogue publishes four b16 per tile where the untransposed one published two
// b32: +4 instructions per accumulator tile, +C/2 per lane per chunk.  Every
// other count -- mma, ldmatrix, cp.async, barriers, and the packs of the state
// itself -- is unchanged.
//
// WHY THE SWAP IS WORTH THAT: REGISTER RESIDENCY UNDER A WARPGROUP MMA.
// `wgmma` sources B from a shared-memory descriptor unconditionally -- there is
// no `_SR` or `_RR` form -- so an accumulator that must land in B forces a
// shared-memory round trip.  Register-source A, the `_RS` atom family, has no
// such gap.  The arrangement here is the one that survives that move; building
// it is a separate piece of work and is not done here.
//
// DIGIT-MAJOR PRODUCTION -- the design decision of this rung, carried over
// ----------------------------------------------------------------------
// LEAF-MAJOR would walk slots: for each of the N slots extract its digits and
// gather the corresponding amplitudes.  That is N dependent shared-memory
// gathers per level per token, plus a digit extraction per slot (a divide and a
// modulo in the naive form).
//
// DIGIT-MAJOR walks digits instead: load the W_l amplitudes of a level ONCE
// into registers and broadcast each into the strided set of slots that carry
// it.  With the mixed-radix order above the LAST level's digit is the fastest
// axis, so a fixed choice of the leading digits names a CONTIGUOUS run of
// W_last slots.  The consequences are the whole point:
//   * sum_l W_l amplitudes are loaded per token instead of N (16 instead of 64
//     here) and each is loaded once into a register;
//   * digit_l(n) is NEVER COMPUTED.  The loop nest IS the digit decomposition,
//     so the divide and the modulo do not appear in the source at all, let
//     alone in the SASS -- there is no reliance on the compiler folding a
//     division by a compile-time width;
//   * the destination address is affine in the loop indices with compile-time
//     strides, so the whole production is one `store_vec16` per leading digit
//     per 8 slots, at an offset that is loop-invariant per thread;
//   * there is no pointer chase and no dependent shared access at all: the
//     amplitude registers feed `mul_bf16x2` directly and its packed result is
//     the operand buffer's own bf16 pair layout.
//
// THE PRODUCT IS ONE BASELINE OPERATION, AND IT IS THE SELECTING ONE
// ------------------------------------------------------------------
// A coefficient is `mul_bf16x2(splat_bf16x2(leading), trailing_pair)`.  That is
// `ops.cuh`'s archetypal SELECTING entry meeting its first real caller: on an
// architecture with bf16 SIMD arithmetic it is one instruction per two
// coefficients, and on one without it the row selects the fp32 route at 1
// multiply + 1/2 pack per coefficient.  The kernel states the product; the
// table decides what it costs.  Nothing here branches on an architecture, and
// the rung's cost is expected to fall by ~3x on a part whose row sets
// `bf16_simd_arith`.
//
// WHAT RIDES THE RING, AND WHY THE BARRIER COUNT DOES NOT MOVE
// ------------------------------------------------------------
// Only V is staged now: R and W are PRODUCED, not fetched, so two of the base's
// three ring buffers disappear and the ring shrinks by two thirds.  R and W
// live in single, unringed buffers, which makes their `load_frag` bases
// LOOP-INVARIANT -- address arithmetic the base had and this rung does not.
//
// Production is issued at the TOP of the chunk body, before the ring wait, so
// the amplitude loads overlap it.  Its WAR fence against the previous chunk's
// readers of R/W is rendezvous (2) of the previous iteration: the last read of
// R and W in a chunk is the fold, which the base already places before that
// barrier.  Production's own publication fence is rendezvous (1), which the
// base already has.  So the barrier count per chunk stays at TWO -- the base's,
// and FlashAttention-2's steady-state count -- and this rung adds none.
//
// TRAFFIC
// -------
// The amplitude rows are re-read by the `kThreads / C` threads that share a
// token row; those are the same bytes for all of them and L1 absorbs the
// redundancy.  Compulsory global traffic per token FALLS from 3*64*2 = 384 B
// (Q, K, V) to 64*2 + 2*(W0+W1)*2 = 192 B (V and the amplitudes).  A factored
// coefficient is a COMPRESSED coefficient.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <vector>

#include "arch_caps.cuh"
#include "ops.cuh"

namespace rung1 {

using namespace denseref::ops;
namespace arch = denseref::arch;

//: THIS KERNEL'S OWN OPERAND-RESIDENCY CLAUSE.  The state is the readout's A
//: operand, so A -- and not B -- is the side that must be readable from the
//: issuing unit's own registers.  This is the clause a warpgroup MMA satisfies.
static_assert(arch::kCaps.mma_a_from_regs,
              "UNSUPPORTED ARCHITECTURE FOR THIS KERNEL: the fold accumulator IS "
              "the readout's A operand, so A must be readable from the issuing "
              "unit's own registers or the state takes a shared-memory round "
              "trip and this is a different algorithm.");

// Bumped by hand whenever the source changes; read back device-side so a stale
// binary cannot masquerade as this source.  See build_stamp().
static constexpr int kBuildMagic = 590301;

static constexpr int kN = 64;   // slot count; also d_k of the state
static constexpr int kD = 64;   // d_v
static constexpr int kPad = 8;  // shared-memory row padding, in elements
static_assert(kN == kD, "this rung keeps the base's square state");

// THE CARVE, inherited unchanged: one MMA unit owns a private 16-row slice of
// the transposed state, so the unit count is d_v / the MMA's m quantum.
static constexpr int kUnits = kD / kMmaM;
static constexpr int kThreads = kUnits * kMmaUnitThreads;
static constexpr int kWarps = kThreads / 32;

// The measured radix.  Both widths are template parameters and the digit
// decomposition is compile-time; the 16-byte production vector additionally
// requires the trailing width to be a multiple of 8, which at N = 64 pins (8,8).
static constexpr int kW0 = 8, kW1 = 8;

// ------------------------------------------------------------------- geometry

//: Compile-time SMEM geometry, shared by the kernel and the host launcher so the
//: two can never disagree about a byte count.  Three ring slots became one: only
//: V is staged.
template <int C, int STAGES, bool SWZ> struct Geom {
  static constexpr int Ld = SWZ ? kN : kN + kPad;  // V/R/W/O row stride, elements
  static constexpr int CLd = C + kPad;             // Gram row stride, elements
  static constexpr size_t bytes =
      (size_t)((STAGES + 3) * C * Ld + C * CLd) * sizeof(__nv_bfloat16);
};

//: THE REGISTER BUDGET, AND THE ONE PLACE THE BASE'S RULE HAD TO GROW.
//:
//: The base sets the second launch bound to `ctas_by_smem` -- "the CTA count
//: this arm's shared-memory footprint already admits, precisely the point at
//: which registers would become the new binding constraint".  That rule has a
//: premise: that shared memory is the binding constraint.  This rung falsifies
//: it.  Dropping two of the three ring buffers cuts shared memory by ~45%, and
//: the SMEM ceiling then runs PAST what the register file can serve: the C = 16
//: arms reach 6 and 7 CTAs by shared memory, whose per-lane budgets are 84 and
//: 72, and the kernel's live set needs 94-96.  Asking for them SPILLS -- 16, 32
//: and 64 bytes of local on three arms.  A spill is a defect, not a footnote.
//:
//: So the rule is completed rather than replaced: the target is the SMEM ceiling
//: capped by the REGISTER ceiling, and the register ceiling is derived from the
//: table exactly as the SMEM one is -- the largest CTA count whose `reg_budget`
//: still holds this kernel's live set.  `kRegFloor` is that live set, stated
//: once and FALSIFIABLE: `build.py` fails the build on any nonzero
//: `localSizeBytes`, so a floor set too low cannot ship quietly.
constexpr int kRegFloor = 96;

constexpr int cta_by_regs(int n) {
  return n <= 1 ? 1
                : (arch::reg_budget(arch::kCaps, kThreads, n) >= kRegFloor
                       ? n
                       : cta_by_regs(n - 1));
}
constexpr int kCtaByRegs = cta_by_regs(arch::ctas_by_threads(arch::kCaps, kThreads));

template <int C, int STAGES, bool SWZ>
constexpr int kCtaTarget =
    arch::ctas_by_smem(arch::kCaps, (int)Geom<C, STAGES, SWZ>::bytes) < kCtaByRegs
        ? arch::ctas_by_smem(arch::kCaps, (int)Geom<C, STAGES, SWZ>::bytes)
        : kCtaByRegs;

// ------------------------------------------------------------------- kernel

template <int C, int STAGES, bool SWZ, bool PACKED, int W0, int W1>
__global__ __launch_bounds__(kThreads, (kCtaTarget<C, STAGES, SWZ>)) void routed_la_kernel(
    const __nv_bfloat16 *__restrict__ pread, const __nv_bfloat16 *__restrict__ pwrite,
    const __nv_bfloat16 *__restrict__ v, __nv_bfloat16 *__restrict__ o, int L) {
  static_assert(C % kMmaM == 0, "chunk must be a multiple of the mma m quantum");
  static_assert(STAGES >= 2, "the wait/issue order needs at least two stages");
  static_assert(W0 * W1 == kN, "level widths must multiply to the slot count");
  static_assert(Geom<C, STAGES, SWZ>::bytes <= (size_t)arch::kCaps.smem_per_cta_max,
                "this (C, STAGES, SWZ) arm does not fit this architecture's "
                "per-CTA shared-memory maximum");
  using G = Geom<C, STAGES, SWZ>;
  constexpr int Ld = G::Ld;
  constexpr int CLd = G::CLd;
  constexpr int MTOK = C / 16;  // token m-tiles in a chunk
  constexpr int NTOK = C / 8;   // token n-tiles in a chunk
  constexpr int KD = kN / 16;   // 4 -- slot k-tiles
  constexpr int NDV = kN / 8;   // 8 -- state n-tiles (slots)
  constexpr uint32_t kRowB = (uint32_t)Ld * 2;

  // Gram tile partition: split m first, spill the remainder onto n.
  constexpr int MSPLIT = MTOK >= kWarps ? kWarps : MTOK;
  constexpr int NSPLIT_RAW = kWarps / MSPLIT;
  constexpr int NSPLIT = NSPLIT_RAW < NTOK / 2 ? NSPLIT_RAW : NTOK / 2;
  constexpr int MPW = MTOK / MSPLIT;
  constexpr int NPW = NTOK / NSPLIT;
  constexpr int GWARPS = MSPLIT * NSPLIT;
  static_assert(NPW % 2 == 0, "n-tiles are consumed in ldmatrix pairs");
  constexpr int GPAIR = (NPW / 2) >= 2 ? 2 : 1;

  // ---- coefficient production geometry.  All compile-time. ----------------
  constexpr int kWtot = W0 + W1;         // amplitudes per token per operand
  constexpr int kRowThr = kThreads / C;  // threads cooperating on one token row
  constexpr int kD0PT = W0 / kRowThr;    // leading digits owned by one thread
  constexpr int kL1Vec = (W1 * 2) / 16;  // 16-byte vectors in a trailing run
  static_assert(kThreads % C == 0, "one production pass must cover the chunk exactly");
  static_assert(W0 % kRowThr == 0, "leading digits must divide over the row's threads");
  static_assert(W1 % 8 == 0, "the trailing level is published in 16-byte vectors");
  static_assert(kD0PT >= 1 && kD0PT <= 4, "leading-digit slice is 1, 2 or 4 amplitudes");

  extern __shared__ __align__(16) char smem_raw[];
  __nv_bfloat16 *sv = reinterpret_cast<__nv_bfloat16 *>(smem_raw);
  __nv_bfloat16 *sr = sv + STAGES * C * Ld;  // produced R, the Q role
  __nv_bfloat16 *sw = sr + C * Ld;           // produced W, the K role
  __nv_bfloat16 *sg = sw + C * Ld;
  __nv_bfloat16 *so = sg + C * CLd;

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const long base = (long)blockIdx.x * L * kD;
  const int NC = L / C;

  // ---- lane-invariant operand addressing, built ONCE (base's, unchanged) ---
  const int lr = lane & 7, tl = lane >> 3;
  const int rF = (tl & 1) * 8 + lr, cF = tl >> 1;
  const int rT = (tl >> 1) * 8 + lr, cT = tl & 1;
  const uint32_t rowF = SWZ ? (uint32_t)(rF * kRowB) : (uint32_t)((rF * Ld + cF * 8) * 2);
  const uint32_t rowT = SWZ ? (uint32_t)(rT * kRowB) : (uint32_t)((rT * Ld + cT * 8) * 2);
  uint32_t tabF[8], tabT[8];
#pragma unroll
  for (int g = 0; g < 8; ++g) {
    tabF[g] = (uint32_t)((((g + cF) & 7) ^ lr) << 4);
    tabT[g] = (uint32_t)((((g + cT) & 7) ^ lr) << 4);
  }
#define ADDRF(b_, R_, g_) \
  (SWZ ? ((b_) + (uint32_t)((R_) * (int)kRowB) + tabF[(g_)]) \
       : ((b_) + (uint32_t)(((R_) * Ld + (g_) * 8) * 2)))
#define ADDRT(b_, R_, g_) \
  (SWZ ? ((b_) + (uint32_t)((R_) * (int)kRowB) + tabT[(g_)]) \
       : ((b_) + (uint32_t)(((R_) * Ld + (g_) * 8) * 2)))

  const int offBg = (rT * CLd + cT * 8) * 2;  // Gram buffer: always padded

  const int gm0 = (warp / NSPLIT) * MPW;
  const int gn0 = (warp % NSPLIT) * NPW;
  const int dv0 = warp * 16;
  constexpr int DVG_STEP = 2;
  const int dvg = warp * DVG_STEP;
  const uint32_t dvTabT = (uint32_t)((((dvg + cT) & 7) ^ lr) << 4);
#define ADDRT_DV(b_, R_) \
  (SWZ ? ((b_) + (uint32_t)((R_) * (int)kRowB) + dvTabT) \
       : ((b_) + (uint32_t)(((R_) * Ld + dv0) * 2)))

  // The state, transposed: m = d_v (this warp's 16 rows), n = slots (8 tiles).
  float st[NDV][4];
#pragma unroll
  for (int j = 0; j < NDV; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) st[j][e] = 0.f;

  // ---- global -> shared staging of V; one row is 128 B = 8 lanes x 16 B ----
  // ADDRESSING DISCIPLINE, the base's: every 64-bit address is formed ONCE here
  // and then advanced by a constant stride, and the ring slot is an induction
  // variable with a compare-and-reset wrap, so there is no `% STAGES` and no
  // IMAD.WIDE inside the chunk loop.
  constexpr int kRowsPerPass = kThreads / 8;  // 16
  constexpr int kPasses = C / kRowsPerPass;
  static_assert(kPasses >= 1, "a chunk must be at least kThreads/8 rows");
  const int ldRow = tid >> 3, ldCol = (tid & 7) * 8;

  constexpr long kChunkElems = (long)C * kD;
  const __nv_bfloat16 *pv = v + base + (long)ldRow * kD + ldCol;
  __nv_bfloat16 *po = o + base + (long)ldRow * kD + ldCol;

  const uint32_t stgOff =
      SWZ ? (uint32_t)(ldRow * (int)kRowB + ((((tid & 7) ^ (ldRow & 7))) << 4))
          : (uint32_t)((ldRow * Ld + ldCol) * 2);

  constexpr uint32_t kSlot = (uint32_t)(C * Ld) * 2;  // bytes of one ring slot
  constexpr uint32_t kRing = STAGES * kSlot;
  const uint32_t svL = smem_addr(sv) + stgOff;
  const uint32_t svB = smem_addr(sv);
  const uint32_t sgT = smem_addr(sg) + (uint32_t)offBg;
  const int soDrainE = (int)(stgOff >> 1);
  uint32_t stgSlot = 0, cmpSlot = 0;

  // R and W are unringed, so their fragment bases are LOOP-INVARIANT: one
  // address each, formed here, and compile-time immediates below.
  const uint32_t arF = smem_addr(sr) + rowF;  // R as the Gram's A, m = tokens
  const uint32_t brT = smem_addr(sr) + rowT;  // R as the readout's B, n = tokens
  const uint32_t bwT = smem_addr(sw) + rowT;  // W as the Gram's B, n = tokens
  const uint32_t bwF = smem_addr(sw) + rowF;  // W as the fold's B, .trans

  // Output staging write index.  The accumulator is transposed, so this lane
  // owns d_v = dv0 + (lane>>2) and that + 8, at tokens 8j + 2*(lane&3) + {0,1}:
  // four elements in four distinct rows of the [token][d_v] buffer, whose only
  // per-chunk variation is the compile-time 8*j row step.
  const int oq = (lane & 3) * 2, om = lane >> 2;
  int oE[2][2];
#pragma unroll
  for (int e = 0; e < 2; ++e)
#pragma unroll
    for (int f = 0; f < 2; ++f)
      oE[e][f] = SWZ ? ((oq + e) * Ld + (((dvg + f) ^ (oq + e)) << 3) + om)
                     : ((oq + e) * Ld + dv0 + om + f * 8);

  auto stage_chunk = [&]() {
#pragma unroll
    for (int p = 0; p < kPasses; ++p) {
      stage_tile(svL + stgSlot + (uint32_t)(p * kRowsPerPass * (int)kRowB),
                 pv + (long)(p * kRowsPerPass * kD));
    }
    pv += kChunkElems;
    stgSlot += kSlot;
    if (stgSlot == kRing) stgSlot = 0;
  };

  // ---- routed Kronecker coefficient production, digit-major ---------------
  // Thread `tid` owns token row `tid / kRowThr` and the leading-digit range
  // [d0f, d0f + kD0PT).  Both divisors are compile-time powers of two, so the
  // two lines below are a shift and a mask; the DIGITS OF A SLOT ARE NEVER
  // COMPUTED, they are the loop indices.  A leading digit's slots are the
  // contiguous run [d * W1, d * W1 + W1), i.e. `kL1Vec` 16-byte vectors whose
  // in-row chunk index is `d * kL1Vec + i`.
  const int prow = tid / kRowThr;
  const int d0f = (tid % kRowThr) * kD0PT;
  const long pbase = (long)blockIdx.x * L * kWtot + (long)prow * kWtot;
  constexpr long kPChunk = (long)C * kWtot;
  const __nv_bfloat16 *pr0 = pread + pbase + d0f;
  const __nv_bfloat16 *pr1 = pread + pbase + W0;
  const __nv_bfloat16 *pw0 = pwrite + pbase + d0f;
  const __nv_bfloat16 *pw1 = pwrite + pbase + W0;

  // Destination: the row's byte base, plus a per-(digit, vector) offset that is
  // a compile-time immediate when padded and a loop-invariant register under the
  // swizzle.  The body computes no XOR and no address.
  char *dstR = reinterpret_cast<char *>(sr) + (SWZ ? prow * (int)kRowB
                                                   : prow * (int)kRowB + d0f * W1 * 2);
  char *dstW = reinterpret_cast<char *>(sw) + (SWZ ? prow * (int)kRowB
                                                   : prow * (int)kRowB + d0f * W1 * 2);
  const int pPh = prow & 7;
  int pOff[kD0PT][kL1Vec];
#pragma unroll
  for (int j = 0; j < kD0PT; ++j)
#pragma unroll
    for (int i = 0; i < kL1Vec; ++i)
      pOff[j][i] = SWZ ? (((((d0f + j) * kL1Vec + i) & 7) ^ pPh) << 4)
                       : ((j * W1 + i * 8) * 2);

  //: THE PRODUCTION, IN TWO ARMS, BOTH BUILT AND BOTH GATED.
  //:
  //: The `W1` trailing amplitudes are read once as packed pairs and reused
  //: across every leading digit this thread owns.  The two arms differ only in
  //: how the elementwise product is STATED, and the difference is which operand
  //: pays for the bf16 -> fp32 widening:
  //:
  //: `PACKED`  states the product as `mul_bf16x2` on two packed pairs -- the
  //:           table's SELECTING entry, one instruction on an architecture with
  //:           bf16 SIMD arithmetic.  On one without it, the packed CONTRACT
  //:           forces the widening of BOTH operands inside every product, so the
  //:           reused amplitude is re-widened once per coefficient.
  //: `WIDEN`   widens the reused operand ONCE per chunk with `widen_bf16x2`
  //:           (exact: bf16 is the top half of an fp32) and multiplies in fp32.
  //:           Per coefficient: 1 multiply + 1/2 `pack_bf16x2`, which is the
  //:           floor for an elementwise product anywhere.
  //:
  //: Both compute the same function bit for bit -- the fallback arm of
  //: `mul_bf16x2` is exactly this fp32 route -- so the choice is a pure
  //: instruction-count question and is decided by measurement, not by taste.
  auto produce = [&](const __nv_bfloat16 *p0, const __nv_bfloat16 *p1, char *dst) {
    __align__(16) uint32_t amp1[kL1Vec * 4];
#pragma unroll
    for (int i = 0; i < kL1Vec; ++i)
      *reinterpret_cast<uint4 *>(&amp1[4 * i]) = load_vec16(p1 + i * 8);

    float w1[W1];
    if constexpr (!PACKED) {
#pragma unroll
      for (int i = 0; i < W1 / 2; ++i) widen_bf16x2(amp1[i], w1[2 * i], w1[2 * i + 1]);
    }

    __align__(8) unsigned short amp0[kD0PT];
    if constexpr (kD0PT == 1) {
      amp0[0] = *reinterpret_cast<const unsigned short *>(p0);
    } else if constexpr (kD0PT == 2) {
      *reinterpret_cast<uint32_t *>(amp0) = *reinterpret_cast<const uint32_t *>(p0);
    } else {
      *reinterpret_cast<uint2 *>(amp0) = *reinterpret_cast<const uint2 *>(p0);
    }

#pragma unroll
    for (int j = 0; j < kD0PT; ++j) {
      __align__(16) uint32_t prod[kL1Vec * 4];
      if constexpr (PACKED) {
        const uint32_t a2 =
            splat_bf16x2(*reinterpret_cast<const __nv_bfloat16 *>(&amp0[j]));
#pragma unroll
        for (int i = 0; i < kL1Vec * 4; ++i) prod[i] = mul_bf16x2(a2, amp1[i]);
      } else {
        // The leading amplitude sits in the LOW half of a zero-extended word, so
        // the same exact widening delivers it in `alo` and a zero in `ahi`.
        float alo, ahi;
        widen_bf16x2((uint32_t)amp0[j], alo, ahi);
        (void)ahi;
#pragma unroll
        for (int i = 0; i < kL1Vec * 4; ++i)
          prod[i] = pack_bf16x2(alo * w1[2 * i], alo * w1[2 * i + 1]);
      }
#pragma unroll
      for (int i = 0; i < kL1Vec; ++i)
        store_vec16(dst + pOff[j][i], *reinterpret_cast<const uint4 *>(&prod[4 * i]));
    }
  };

#pragma unroll 1
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < NC) stage_chunk();
    stage_commit();
  }

#pragma unroll 1
  for (int c = 0; c < NC; ++c) {
    // ---- 0. produce this chunk's routed coefficients -----------------------
    // Issued before the ring wait so the amplitude loads overlap it.  Its WAR
    // fence is rendezvous (2) of the previous iteration -- the fold is the last
    // reader of R and W and the base already places it before that barrier.
    produce(pr0, pr1, dstR);
    produce(pw0, pw1, dstW);
    pr0 += kPChunk; pr1 += kPChunk; pw0 += kPChunk; pw1 += kPChunk;

    stage_wait<STAGES - 2>();
    rendezvous();  // (1) V visible / R,W visible / last chunk's reads done / O staged

    if (c + STAGES - 1 < NC) stage_chunk();
    stage_commit();

    const uint32_t atv = svB + cmpSlot + rowT;
    cmpSlot += kSlot;
    if (cmpSlot == kRing) cmpSlot = 0;

    // ---- 1. readout: O^T = S^T . R^T, A operand straight out of the state ---
    // `sa` is the relabel derived in this file's header: an A fragment's four
    // halves are the (lo, hi) of two adjacent state tiles along the slot axis.
    float acc[NTOK][4];
    {
      uint32_t sa[KD][4];
#pragma unroll
      for (int kk = 0; kk < KD; ++kk) {
        sa[kk][0] = pack_bf16x2(st[2 * kk][0], st[2 * kk][1]);
        sa[kk][1] = pack_bf16x2(st[2 * kk][2], st[2 * kk][3]);
        sa[kk][2] = pack_bf16x2(st[2 * kk + 1][0], st[2 * kk + 1][1]);
        sa[kk][3] = pack_bf16x2(st[2 * kk + 1][2], st[2 * kk + 1][3]);
      }
#pragma unroll
      for (int j = 0; j < NTOK; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[j][e] = 0.f;
#pragma unroll
      for (int m = 0; m < MTOK; ++m) {
        uint32_t rb[KD][4];
#pragma unroll
        for (int kk = 0; kk < KD; ++kk) load_frag(rb[kk], ADDRT(brT, m * 16, 2 * kk));
#pragma unroll
        for (int kk = 0; kk < KD; ++kk)
#pragma unroll
          for (int p = 0; p < 2; ++p) mma(acc[2 * m + p], sa[kk], &rb[kk][2 * p]);
      }
    }

    // ---- 2. Gram: G = tril(R.W^T), this warp's tiles, straight into shared --
    if (warp < GWARPS) {
      uint32_t qa[MPW][KD][4];
#pragma unroll
      for (int m = 0; m < MPW; ++m)
#pragma unroll
        for (int kk = 0; kk < KD; ++kk)
          load_frag(qa[m][kk], ADDRF(arF, (gm0 + m) * 16, 2 * kk));

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
            for (int h = 0; h < 2; ++h)
#pragma unroll
              for (int e = 0; e < 4; ++e) g[u][h][e] = 0.f;
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
            for (int h = 0; h < 2; ++h) {
              const int tc = (gn0 + (np + u) * 2 + h) * 8 + (lane & 3) * 2;
              const float m0 = (tc + 0 <= tr) ? g[u][h][0] : 0.f;
              const float m1 = (tc + 1 <= tr) ? g[u][h][1] : 0.f;
              const float m2 = (tc + 0 <= tr + 8) ? g[u][h][2] : 0.f;
              const float m3 = (tc + 1 <= tr + 8) ? g[u][h][3] : 0.f;
              store_bf16x2(&sg[tr * CLd + tc], pack_bf16x2(m0, m1));
              store_bf16x2(&sg[(tr + 8) * CLd + tc], pack_bf16x2(m2, m3));
            }
        }
      }
    }

    // ---- 3. drain the PREVIOUS chunk's output ------------------------------
    if (c) {
#pragma unroll
      for (int p = 0; p < kPasses; ++p)
        copy_shared_to_global_16b(po + (long)(p * kRowsPerPass * kD),
                                  &so[soDrainE + p * kRowsPerPass * Ld]);
      po += kChunkElems;
    }

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
    }

    rendezvous();  // (2) the Gram is visible; R and W are dead until produce()

    // ---- 5. intra: O^T += V^T . G^T ----------------------------------------
    // A is V^T -- the fold's own operand, at the same address form -- and the
    // Gram is read as B, n = query token, k = key token.
    {
      uint32_t va[MTOK][4];
#pragma unroll
      for (int kk = 0; kk < MTOK; ++kk) load_frag_t(va[kk], ADDRT_DV(atv, kk * 16));
#pragma unroll
      for (int m = 0; m < MTOK; ++m) {
        uint32_t gb[MTOK][4];
#pragma unroll
        for (int kk = 0; kk < MTOK; ++kk)
          load_frag(gb[kk], sgT + (uint32_t)((m * 16 * CLd + kk * 16) * 2));
#pragma unroll
        for (int kk = 0; kk < MTOK; ++kk)
#pragma unroll
          for (int p = 0; p < 2; ++p) mma(acc[2 * m + p], va[kk], &gb[kk][2 * p]);
      }
    }

    // ---- 6. stage this chunk's output; the store happens next iteration -----
#pragma unroll
    for (int j = 0; j < NTOK; ++j) {
      const int jo = j * 8 * Ld;
      store_bf16(&so[oE[0][0] + jo], acc[j][0]);
      store_bf16(&so[oE[1][0] + jo], acc[j][1]);
      store_bf16(&so[oE[0][1] + jo], acc[j][2]);
      store_bf16(&so[oE[1][1] + jo], acc[j][3]);
    }
  }

  rendezvous();
#pragma unroll
  for (int p = 0; p < kPasses; ++p)
    copy_shared_to_global_16b(po + (long)(p * kRowsPerPass * kD),
                              &so[soDrainE + p * kRowsPerPass * Ld]);
#undef ADDRF
#undef ADDRT
#undef ADDRT_DV
}

// --------------------------------------------------------------- host side

// The ratified (C, STAGES, SWZ) instantiation set -- the base's, arm for arm, so
// every number in this rung's table has a same-arm base to be a delta against.
#define RUNG1_SHAPES(X, P)                                                   \
  X(16, 3, false, P) X(16, 4, false, P) X(16, 5, false, P)                   \
  X(32, 2, false, P) X(32, 3, false, P) X(32, 4, false, P)                   \
  X(64, 2, false, P) X(64, 3, false, P)                                      \
  X(16, 3, true, P) X(16, 4, true, P) X(16, 6, true, P)                      \
  X(32, 3, true, P) X(32, 4, true, P) X(32, 5, true, P)                      \
  X(64, 3, true, P)
#define RUNG1_ARMS(X) RUNG1_SHAPES(X, false) RUNG1_SHAPES(X, true)

template <int C, int S, bool SWZ, bool P>
static void launch(const torch::Tensor &pr, const torch::Tensor &pw,
                   const torch::Tensor &v, torch::Tensor &o, int BH, int L) {
  constexpr size_t sm = Geom<C, S, SWZ>::bytes;
  static bool configured = false;
  if (!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(routed_la_kernel<C, S, SWZ, P, kW0, kW1>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int)sm));
    configured = true;
  }
  routed_la_kernel<C, S, SWZ, P, kW0, kW1>
      <<<BH, kThreads, sm, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const __nv_bfloat16 *>(pr.data_ptr()),
          reinterpret_cast<const __nv_bfloat16 *>(pw.data_ptr()),
          reinterpret_cast<const __nv_bfloat16 *>(v.data_ptr()),
          reinterpret_cast<__nv_bfloat16 *>(o.data_ptr()), L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor forward(torch::Tensor pr, torch::Tensor pw, torch::Tensor v,
                      int64_t chunk, int64_t stages, bool swizzle,
                      bool packed) {
  TORCH_CHECK(v.is_cuda() && v.scalar_type() == torch::kBFloat16 && v.is_contiguous());
  TORCH_CHECK(pr.is_contiguous() && pw.is_contiguous());
  TORCH_CHECK(pr.scalar_type() == torch::kBFloat16 && pw.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(v.dim() == 4 && v.size(3) == kD, "expect V [B,H,L,64]");
  TORCH_CHECK(pr.sizes() == pw.sizes() && pr.dim() == 4 && pr.size(3) == kW0 + kW1,
              "expect packed amplitudes [B,H,L,W0+W1]");
  const int B = v.size(0), H = v.size(1), L = v.size(2);
  TORCH_CHECK(pr.size(0) == B && pr.size(1) == H && pr.size(2) == L);
  TORCH_CHECK(L % chunk == 0, "L must be a multiple of the chunk");
  auto o = torch::empty_like(v);
  const int BH = B * H;
#define RUNG1_DISPATCH(C_, S_, W_, P_)                          \
  if (chunk == C_ && stages == S_ && swizzle == W_ && packed == P_) { \
    launch<C_, S_, W_, P_>(pr, pw, v, o, BH, L);                \
    return o;                                                   \
  }
  RUNG1_ARMS(RUNG1_DISPATCH)
#undef RUNG1_DISPATCH
  TORCH_CHECK(false, "unbuilt arm: chunk=", chunk, " stages=", stages,
              " swizzle=", swizzle, " packed=", packed);
}

// Device-side build identity: a magic constant WRITTEN BY A KERNEL that only
// this source compiles, the geometry stamp, the digest of the capability row THE
// DEVICE CODE WAS COMPILED AGAINST, and the ptxas census of the very entry the
// measurement runs.
__global__ void stamp_kernel(int *out) {
  out[0] = kBuildMagic;
  out[1] = kWarps * 1000 + kN;
  out[2] = arch::caps_digest(arch::kCaps);
  out[3] = kW0 * 1000 + kW1;
}

std::vector<int64_t> build_stamp(int64_t chunk, int64_t stages, bool swizzle,
                                 bool packed) {
  auto t = torch::zeros({4}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  stamp_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(t.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto h = t.cpu();
  std::vector<int64_t> r{h.data_ptr<int>()[0], h.data_ptr<int>()[1],
                         h.data_ptr<int>()[2], h.data_ptr<int>()[3]};
  cudaFuncAttributes a{};
#define RUNG1_CENSUS(C_, S_, W_, P_)                                                \
  if (chunk == C_ && stages == S_ && swizzle == W_ && packed == P_) {               \
    C10_CUDA_CHECK(cudaFuncGetAttributes(                                           \
        &a, routed_la_kernel<C_, S_, W_, P_, kW0, kW1>));                           \
    r.push_back(a.numRegs);                                                         \
    r.push_back((int64_t)a.sharedSizeBytes);                                        \
    r.push_back(a.binaryVersion);                                                   \
    r.push_back((int64_t)Geom<C_, S_, W_>::bytes);                                  \
    r.push_back((int64_t)a.maxThreadsPerBlock);                                     \
    r.push_back((int64_t)a.localSizeBytes);                                         \
    return r;                                                                       \
  }
  RUNG1_ARMS(RUNG1_CENSUS)
#undef RUNG1_CENSUS
  TORCH_CHECK(false, "unbuilt arm");
  return r;
}

std::vector<std::vector<int64_t>> arms() {
  std::vector<std::vector<int64_t>> out;
#define RUNG1_LIST(C_, S_, W_, P_) \
  out.push_back({C_, S_, (int64_t)(W_), (int64_t)(P_), (int64_t)Geom<C_, S_, W_>::bytes});
  RUNG1_ARMS(RUNG1_LIST)
#undef RUNG1_LIST
  return out;
}

}  // namespace rung1

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &rung1::forward, "routed-Kronecker chunked linear attention");
  m.def("build_stamp", &rung1::build_stamp, "build identity");
  m.def("arms", &rung1::arms, "the built (C, STAGES, SWZ, smem) set");
}
