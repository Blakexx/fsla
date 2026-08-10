// LADDER RUNG 3 -- THE TOKEN SCHEDULE AND THE MEMBERSHIP TEST.
//
// Base: rung 2 (`rung2/csrc/routed_capacity_la.cu` at `47fd58d`), which carves
// the slot axis into blocks of BC and fans the blocks' partial outputs into one
// fp32 accumulator.  EXACTLY ONE FEATURE is added here: a block may SKIP A GROUP
// OF TOKENS it has no work for, and a schedule says which groups those are.
// Everything else -- the carve, the block geometry law, the fan-in, the
// register-resident state, digit-major production -- is rung 2 unchanged, and
// the gate that says so is that a DENSE run is bit-identical to rung 2's.
//
// WHY THE SPLIT IS EXACT AND NEEDS NO NEW ALGEBRA
// -----------------------------------------------
// Every term of the recurrence is LINEAR in the slot axis:
//     O_c = R_c . S              S rows are slots
//     O_c += tril(R_c . W_c^T) . V_c
//     S   += W_c^T . V_c
// A block's state rows are disjoint from every other block's, `tril` is linear,
// and the Gram is a sum over slots.  So a CTA that restricts every slot sum to
// its own BC rows computes a COMPLETE PARTIAL OUTPUT, and the sum over blocks is
// the answer.  Nothing is approximated and no block needs another block's state.
//
// WHAT A BLOCK CAN SKIP, AND WHY SKIPPING IT IS EXACT
// ---------------------------------------------------
// A token's coefficient over the block's slots is the product of its per-level
// amplitudes at the digits the rectangle spans.  If every slot of the block gets
// a zero coefficient from every token of a group then, for that (block, group),
// `R_c` and `W_c` restricted to the block are identically zero, so the readout
// `R_c . S` is zero, the Gram is zero and with it the intra term, and the fold
// adds nothing to the state.  The block's whole contribution is the additive
// identity and its state is unchanged, so the group can be skipped OUTRIGHT.
// That is not an approximation and needs no new algebra: it is the observation
// that the carve above already made every term a SUM whose identity is zero.
//
// THE MEMBERSHIP TEST IS A PREFIX TEST, AND THAT IS WHAT MAKES IT CHEAP
// ---------------------------------------------------------------------
// A slot is live for a token when EVERY level's amplitude at that slot's digit
// is nonzero, so the block holds work for the token exactly when
//
//     AND over levels l of ( OR over the digits the block spans at level l )
//
// which is multiplicative over levels and enumerates no slots.  The block
// geometry law below does the rest: a block spans every level above the boundary
// at EXACTLY ONE digit, exactly one level PARTIALLY, and every level below
// ENTIRELY.  So a level spanned at one digit costs ONE amplitude test; the
// partially spanned level costs an OR over its aligned sub-range, which `or_run`
// answers in whole 16-byte accesses; and a level spanned ENTIRELY IS NOT TESTED,
// because its clause asks whether the token has any nonzero amplitude at that
// level at all -- a property of the TOKEN, not of the block, and one that holds
// for every simplex-valued routing, since an entmax or softmax row sums to one
// and some digit survives.
//
// That last clause is the ONE place this rung is CONSERVATIVE rather than exact,
// and the direction is the safe one: a level that really was all zero would make
// the test say LIVE for a group with no work, which costs time and changes no
// answer.  The test can never say DEAD for a group that has work, because every
// clause it does evaluate is exact.
//
// THE SCHEDULE: A BITMASK, BUILT ONCE, IN A PROLOGUE
// --------------------------------------------------
// One bit per (this block, token group), built before the chunk loop by a pass
// that reads only the amplitudes the test names.  A lockstep unit votes on the
// tokens its lanes hold (`lane_ballot`) and writes WHOLE mask words, so no two
// units write the same word: the prologue needs no atomic and no zeroing, and
// one rendezvous publishes the schedule to the chunk loop.
//
// THE GROUP IS THE CHUNK, AND THAT IS DERIVED RATHER THAN SWEPT
// -------------------------------------------------------------
// Membership is decided over the CHUNK, the unit the recurrence and the ring
// already step in.  A COARSER group would test no more cheaply -- the test is a
// prologue whose cost is per TOKEN, not per group -- and would skip strictly
// less, so it can only lose.  A FINER group is a different feature: deciding per
// token would compact tokens inside a chunk and destroy the affine addressing
// that keeps `digit_l(n)` uncomputed.  Skipping a whole group preserves it.
//
// WHAT THE SKIP DOES NOT TOUCH: EVERY ADDRESS STAYS AFFINE
// --------------------------------------------------------
// The loop still visits every chunk index, every pointer still advances by its
// compile-time constant on every iteration whether the chunk is live or not, and
// the ring slot still advances unconditionally -- so a staging group's index is
// still its chunk's index and the wait/commit accounting is rung 2's exactly.
// What the skip removes is WORK: the staging of V, the coefficient production,
// the four mma terms, the output staging and the fan-in.  It removes no address
// computation, and it adds none.  A COMPACTED chunk list was declined for the
// opposite reason: it replaces the induction with a gather.
//
// THE ONE BARRIER THAT IS NOT SKIPPED.  `rendezvous` (1) is the ring slot's WAR
// fence and runs on every chunk, live or dead, because the staging it protects
// runs on every chunk; (2) fences the Gram and lives inside the live branch.
// Two per LIVE chunk -- rung 2's count, and FA2's -- and one per dead chunk.
//
// THE BLOCK GEOMETRY LAW -- a block is a RECTANGLE IN DIGIT SPACE
// ---------------------------------------------------------------
// A slot is a mixed-radix number over the routing levels, most significant digit
// first.  A block of BC slots is admissible only when, reading the levels from
// the most significant down:
//
//     every level ABOVE the boundary is spanned at EXACTLY ONE digit;
//     EXACTLY ONE level is spanned PARTIALLY, by an ALIGNED sub-range whose
//         width divides that level's width;
//     every level BELOW the boundary is spanned ENTIRELY.
//
// That is not a convention, it is the necessary and sufficient condition for the
// block's slots to form a CONTIGUOUS RANGE -- equivalently for the block to be a
// rectangle in digit space.  A rectangle is what lets the coefficient loop nest
// BE the digit decomposition, so `digit_l(n)` is never computed here either.  A
// (widths, BC) pair that does not satisfy it FAILS THE BUILD at `BlockPlan`'s
// clauses; it is never approximated and never rounded to fit.
//
// For the two levels built here the law reads: `lead_span` digits of level 0 by
// `tail_span` digits of level 1, with `tail_span < W1` allowed only when
// `lead_span == 1`, and `lead_span > 1` allowed only when `tail_span == W1`.
//
// THE FAN-IN IS A PLAIN REDUCTION, AND IT IS FIRE-AND-FORGET
// ----------------------------------------------------------
// The N/BC contributors to an output element are added with `fan_in_add`, whose
// contract is that the issuing thread does not consume a result -- which is what
// lets the instruction be a REDUCTION rather than an atomic read-modify-write,
// so nothing waits on a return value.  A scratch-and-reduce arrangement was
// declined before this rung was written: its scratch is N/BC times the output,
// and bounding it by tiling the token axis costs a grid-wide synchronisation per
// tile.
//
// THE OUTPUT IS fp32, AND THE FAN-IN IS WHY
// ------------------------------------------
// Rung 1 published bf16 because one CTA owned an element outright.  With N/BC
// contributors a bf16 accumulator would round every partial, and the partials of
// a routed readout carry mixed signs, so the fan-in error would land near
// 2^-9 of the result -- above this design's own bf16 quantisation floor rather
// than inside it.  The staging buffer and the global accumulator are therefore
// fp32 and the reduction is one contribution per channel.  This architecture has
// no two-wide fp32 reduction; the two-wide form it does have is bf16x2, which is
// exactly the accumulator that was just refused.
//
// The staging buffer keeps the emission's layout property in the form that does
// survive: A LANE HOLDS A RUN OF CONSECUTIVE CHANNELS, so a warp's contributions
// cover whole aligned segments of the output row rather than a scatter.
//
// WHAT CAPACITY COSTS, STRUCTURALLY, BEYOND THE FAN-IN
// -----------------------------------------------------
// The intra term `tril(R_c . W_c^T) . V_c` multiplies a C x C Gram by V.  The
// GRAM is a slot sum and splits, but the V multiply does not: every block does
// its own, so it is performed N/BC times where one would do if the Gram were
// summed first.  Summing it first is a cross-CTA reduction per chunk, which is
// the grid-wide synchronisation the fan-in already declined.  This replication
// is the structural price of the carve and it is what makes BC an axis with a
// real optimum rather than a tuning knob.
//
// EVERYTHING ELSE IS RUNG 1, UNCHANGED IN DESIGN
// -----------------------------------------------
// Register-resident state as the fold's accumulator and the readout's A operand;
// digit-major production with the loop nest as the digit decomposition; one
// `mul_bf16x2` per coefficient; the ring staging only V; the deferred output
// store that doubles as the ring's WAR fence.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <vector>

#include "design.cuh"
#include "ops.cuh"

namespace rung3 {

using namespace denseref::ops;
namespace design = denseref::design;

static_assert(design::kStateResidency == design::StateResidency::REGISTERS &&
                  kMmaAcceptsRegisterA,
              "SELECTION UNSATISFIABLE GIVEN AVAILABLE OPERATIONS: this kernel's "
              "fold accumulator IS the readout's A operand, so it needs a state "
              "that lives in registers and a built `mma` that reads A out of "
              "them; otherwise the state takes a memory round trip and this is a "
              "different algorithm.");

static constexpr int kBuildMagic = 590603;

static constexpr int kD = 64;      // d_v
static constexpr int kPadE = 8;    // bf16 row padding, elements (16 B)
static constexpr int kPadF = 4;    // fp32 row padding, elements (16 B)

// THE CARVE, and it is DERIVED rather than chosen.  One MMA unit owns a private
// 16-row slice of the transposed state's m axis, which is d_v, so the unit count
// is d_v / the MMA's m quantum.  Carving the SLOT axis across units instead
// would give every unit a partial output over the whole of d_v and therefore a
// cross-unit reduction of the output accumulator -- the very thing the private
// accumulator exists to avoid.  The carve is forced by the requirement, not
// picked from a sweep.
static constexpr int kUnits = kD / kMmaM;
static constexpr int kThreads = kUnits * kMmaUnitThreads;
static constexpr int kWarps = kThreads / 32;

// --------------------------------------------------------------- block plan

//: THE GEOMETRY LAW, stated once and checked at build time.  `lead_span` digits
//: of level 0 by `tail_span` digits of level 1; `blocks` of them tile the slot
//: space.  Every clause below is the law's, and a (widths, BC) pair that breaks
//: one does not compile.
template <int W0, int W1, int BC> struct BlockPlan {
  static constexpr int slots = W0 * W1;
  static constexpr int lead_span = BC <= W1 ? 1 : BC / W1;
  static constexpr int tail_span = BC <= W1 ? BC : W1;
  static constexpr int tail_groups = W1 / tail_span;
  static constexpr int blocks = slots / BC;

  static_assert(lead_span * tail_span == BC,
                "BLOCK GEOMETRY: BC is not a rectangle in digit space -- it is "
                "neither a sub-range of the trailing level nor a whole number of "
                "trailing levels");
  static_assert(BC <= W1 || BC % W1 == 0,
                "BLOCK GEOMETRY: a block wider than the trailing level must span "
                "that level ENTIRELY, so its width must be a multiple of it");
  static_assert(W1 % tail_span == 0,
                "BLOCK GEOMETRY: the partially spanned level's sub-range must "
                "divide that level's width, or the sub-ranges do not tile it");
  static_assert(W0 % lead_span == 0,
                "BLOCK GEOMETRY: the leading level's spanned run must divide its "
                "width, or the runs do not tile it");
  static_assert(lead_span == 1 || tail_span == W1,
                "BLOCK GEOMETRY: EXACTLY ONE level may be partially spanned; "
                "every level below it must be spanned entirely");
  static_assert(blocks * BC == slots, "BLOCK GEOMETRY: the blocks do not tile N");
};

// ------------------------------------------------------------------ geometry

//: THIS KERNEL'S LIVE REGISTER SET, per (chunk, block width).  It is a
//: DECLARATION rather than a derivation, and deliberately so: ptxas's allocation
//: for this body does not follow from the operand counts -- the measured census
//: reads 95 / 137 / 161 registers per lane at BC = 64 and 136 / 168 / 209 at
//: BC = 128 for C = 16 / 32 / 64 -- and the number's only job is to stop the
//: shared-memory ceiling from asking for more CTAs than the register file holds.
//:
//: It is FALSIFIABLE, not asserted: `build.py` fails the build on any nonzero
//: `localSizeBytes`, so a floor set too low cannot ship quietly.  It HAS been
//: falsified: an additive model put (C = 16, BC = 128) at exactly the 128-register
//: budget of four CTAs and the arm spilled eight bytes, which is why what stands
//: here is the census and not the model that produced it.
constexpr int reg_floor(int C, int BC) {
  return C == 16 ? (BC <= 64 ? 96 : BC <= 128 ? 136 : 200)
       : C == 32 ? (BC <= 32 ? 120 : BC <= 64 ? 160 : BC <= 128 ? 176 : 208)
                 : (BC <= 32 ? 128 : BC <= 64 ? 192 : 224);
}

//: THE SCHEDULE'S CAPACITY, in mask words of 32 groups each.  It bounds the
//: number of token groups a launch may have, `L / C`, and the host refuses a
//: longer sequence rather than silently scheduling part of it.  128 bytes of
//: shared memory buys 1024 groups, which at the smallest chunk is L = 16384.
constexpr int kSchedWords = 32;
constexpr int kMaxGroups = kSchedWords * 32;

template <int C, bool SWZ, int BC> struct Geom {
  static constexpr int LdV = SWZ ? kD : kD + kPadE;   // V ring row, bf16 elements
  static constexpr int LdS = SWZ ? BC : BC + kPadE;   // R/W row, bf16 elements
  static constexpr int LdG = C + kPadE;               // Gram row, bf16 elements
  //: The output staging buffer is fp32 and ALWAYS PADDED.  The swizzle exists to
  //: serve `ldmatrix` on the operand buffers; nothing reads this one as an
  //: operand, and its two access patterns are conflict-free under the pad.
  static constexpr int LdO = kD + kPadF;              // output row, fp32 elements
  static constexpr int slot = C * LdV * (int)sizeof(__nv_bfloat16);
  static constexpr int fixed = (2 * C * LdS + C * LdG) * (int)sizeof(__nv_bfloat16) +
                               C * LdO * (int)sizeof(float) +
                               kSchedWords * (int)sizeof(uint32_t);
  static constexpr int stages =
      design::stages_for(slot, fixed, kThreads, reg_floor(C, BC));
  static constexpr size_t bytes = (size_t)stages * slot + fixed;
};

template <int C, bool SWZ, int BC>
constexpr int kCtaTarget =
    design::cta_target((int)Geom<C, SWZ, BC>::bytes, kThreads, reg_floor(C, BC));

// -------------------------------------------------------------------- kernel

template <int C, bool SWZ, int W0, int W1, int BC>
__global__ __launch_bounds__(kThreads, (kCtaTarget<C, SWZ, BC>)) void
routed_schedule_kernel(const __nv_bfloat16 *__restrict__ pread,
                       const __nv_bfloat16 *__restrict__ pwrite,
                       const __nv_bfloat16 *__restrict__ v, float *__restrict__ o,
                       int L) {
  using P = BlockPlan<W0, W1, BC>;
  using G = Geom<C, SWZ, BC>;
  static_assert(C % kMmaM == 0, "chunk must be a multiple of the mma m quantum");
  static_assert(BC % kMmaK == 0, "a block must be a whole number of mma k-tiles");
  static_assert(design::smem_fits(G::bytes),
                "this arm does not fit this architecture's per-CTA shared-memory "
                "maximum");
  constexpr int STAGES = G::stages;
  static_assert(STAGES >= 2, "the wait/issue order needs at least two stages");
  constexpr int LdV = G::LdV, LdS = G::LdS, LdG = G::LdG, LdO = G::LdO;
  constexpr int MTOK = C / kMmaM;   // token m-tiles in a chunk
  constexpr int NTOK = C / kMmaN;   // token n-tiles in a chunk
  constexpr int KD = BC / kMmaK;    // slot k-tiles of this block
  constexpr int NDV = BC / kMmaN;   // state n-tiles (slots) of this block
  constexpr uint32_t kRowV = (uint32_t)LdV * 2;
  constexpr uint32_t kRowS = (uint32_t)LdS * 2;
  constexpr int kChunkS = BC / 8;   // 16-byte chunks in one R/W row
  constexpr int kChunkV = kD / 8;   // 16-byte chunks in one V row

  constexpr int MSPLIT = MTOK >= kWarps ? kWarps : MTOK;
  constexpr int NSPLIT_RAW = kWarps / MSPLIT;
  constexpr int NSPLIT = NSPLIT_RAW < NTOK / 2 ? NSPLIT_RAW : NTOK / 2;
  constexpr int MPW = MTOK / MSPLIT;
  constexpr int NPW = NTOK / NSPLIT;
  constexpr int GWARPS = MSPLIT * NSPLIT;
  static_assert(NPW % 2 == 0, "n-tiles are consumed in ldmatrix pairs");
  constexpr int GPAIR = (NPW / 2) >= 2 ? 2 : 1;

  // ---- coefficient production geometry.  All compile-time. ----------------
  // A row's `kRowThr` threads split the block's BC coefficients.  There is one
  // rectangle to split and two ways to cut it, and which one applies is decided
  // by the rectangle's own shape: when the leading run is at least as wide as
  // the thread count the cut is along the leading digits, and otherwise several
  // threads share a leading digit and the cut is along the trailing run.
  constexpr int kWtot = W0 + W1;
  constexpr int kRowThr = kThreads / C;
  constexpr int kThrPerLead = kRowThr >= P::lead_span ? kRowThr / P::lead_span : 1;
  constexpr int kLeadPerThr = P::lead_span >= kRowThr ? P::lead_span / kRowThr : 1;
  constexpr int kTailPerThr = P::tail_span / kThrPerLead;
  constexpr int kTailVec = (kTailPerThr * 2) / 16;
  static_assert(kThreads % C == 0, "one production pass must cover the chunk exactly");
  static_assert(kRowThr >= P::lead_span ? kRowThr % P::lead_span == 0
                                        : P::lead_span % kRowThr == 0,
                "the block's leading digits and the row's threads must divide one "
                "another, or a thread owns a fractional digit");
  static_assert(P::tail_span % kThrPerLead == 0,
                "the threads sharing a leading digit must divide its trailing run");
  static_assert(kTailPerThr % 8 == 0,
                "a thread's trailing run is published in 16-byte vectors, so it "
                "must be a whole number of eight coefficients");
  static_assert(kLeadPerThr * kTailPerThr * kRowThr == BC,
                "the row's threads must cover the block exactly once");
  static_assert(kLeadPerThr <= 4, "leading-digit slice is 1, 2 or 4 amplitudes");

  extern __shared__ __align__(16) char smem_raw[];
  __nv_bfloat16 *sv = reinterpret_cast<__nv_bfloat16 *>(smem_raw);
  __nv_bfloat16 *sr = sv + STAGES * C * LdV;
  __nv_bfloat16 *sw = sr + C * LdS;
  __nv_bfloat16 *sg = sw + C * LdS;
  float *so = reinterpret_cast<float *>(sg + C * LdG);
  uint32_t *sched = reinterpret_cast<uint32_t *>(so + C * LdO);
  static_assert((2 * C * LdS + C * LdG) % 8 == 0,
                "the fp32 output buffer must start 16-byte aligned");

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int blk = blockIdx.x;   // which block of BC slots this CTA owns
  const long base = (long)blockIdx.y * L * kD;
  const int NC = L / C;

  // The block's aligned digit ranges.  Compile-time divisors, formed once.
  const int lead_off = (blk / P::tail_groups) * P::lead_span;
  const int tail_off = (blk % P::tail_groups) * P::tail_span;

  // ---- lane-invariant operand addressing -----------------------------------
  const int lr = lane & 7, tl = lane >> 3;
  const int rF = (tl & 1) * 8 + lr, cF = tl >> 1;
  const int rT = (tl >> 1) * 8 + lr, cT = tl & 1;
  // Two row strides, and they are DISTINCT: an operand buffer's row is the block
  // width and the value buffer's row is d_v.  One lane-row base per stride.
  const uint32_t rowF = SWZ ? (uint32_t)(rF * kRowS) : (uint32_t)((rF * LdS + cF * 8) * 2);
  const uint32_t rowT = SWZ ? (uint32_t)(rT * kRowS) : (uint32_t)((rT * LdS + cT * 8) * 2);
  const uint32_t rowTV = SWZ ? (uint32_t)(rT * kRowV) : (uint32_t)((rT * LdV + cT * 8) * 2);
  uint32_t tabF[kChunkS], tabT[kChunkS];
#pragma unroll
  for (int g = 0; g < kChunkS; ++g) {
    tabF[g] = (uint32_t)((((g + cF) & (kChunkS - 1)) ^ (lr & (kChunkS - 1))) << 4);
    tabT[g] = (uint32_t)((((g + cT) & (kChunkS - 1)) ^ (lr & (kChunkS - 1))) << 4);
  }
#define ADDRF(b_, R_, g_) \
  (SWZ ? ((b_) + (uint32_t)((R_) * (int)kRowS) + tabF[(g_)]) \
       : ((b_) + (uint32_t)(((R_) * LdS + (g_) * 8) * 2)))
#define ADDRT(b_, R_, g_) \
  (SWZ ? ((b_) + (uint32_t)((R_) * (int)kRowS) + tabT[(g_)]) \
       : ((b_) + (uint32_t)(((R_) * LdS + (g_) * 8) * 2)))

  const int offBg = (rT * LdG + cT * 8) * 2;

  const int gm0 = (warp / NSPLIT) * MPW;
  const int gn0 = (warp % NSPLIT) * NPW;
  const int dv0 = warp * 16;
  constexpr int DVG_STEP = 2;
  const int dvg = warp * DVG_STEP;
  const uint32_t dvTabT = (uint32_t)((((dvg + cT) & (kChunkV - 1)) ^ lr) << 4);
#define ADDRT_DV(b_, R_) \
  (SWZ ? ((b_) + (uint32_t)((R_) * (int)kRowV) + dvTabT) \
       : ((b_) + (uint32_t)(((R_) * LdV + dv0) * 2)))

  // The state, transposed: m = d_v (this warp's 16 rows), n = this block's slots.
  float st[NDV][4];
#pragma unroll
  for (int j = 0; j < NDV; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) st[j][e] = 0.f;

  // ---- global -> shared staging of V ---------------------------------------
  constexpr int kRowsPerPass = kThreads / kChunkV;
  constexpr int kPasses = C / kRowsPerPass;
  static_assert(kPasses >= 1, "a chunk must be at least kThreads/8 rows");
  const int ldRow = tid / kChunkV, ldCol = (tid % kChunkV) * 8;

  constexpr long kChunkElems = (long)C * kD;
  const __nv_bfloat16 *pv = v + base + (long)ldRow * kD + ldCol;

  const uint32_t stgOff =
      SWZ ? (uint32_t)(ldRow * (int)kRowV + (((tid % kChunkV) ^ (ldRow & 7)) << 4))
          : (uint32_t)((ldRow * LdV + ldCol) * 2);

  constexpr uint32_t kSlot = (uint32_t)(C * LdV) * 2;
  constexpr uint32_t kRing = STAGES * kSlot;
  const uint32_t svL = smem_addr(sv) + stgOff;
  const uint32_t svB = smem_addr(sv);
  const uint32_t sgT = smem_addr(sg) + (uint32_t)offBg;
  uint32_t stgSlot = 0, cmpSlot = 0;

  const uint32_t arF = smem_addr(sr) + rowF;
  const uint32_t brT = smem_addr(sr) + rowT;
  const uint32_t bwT = smem_addr(sw) + rowT;
  const uint32_t bwF = smem_addr(sw) + rowF;

  // ---- the fan-in's own addressing -----------------------------------------
  // CONSECUTIVE CHANNELS SIT IN CONSECUTIVE LANES OF ONE REDUCTION.  That is the
  // layout property that matters here, and it is a property of the REDUCTION and
  // not of the thread: this architecture's reduction carries ONE channel, so what
  // decides how much output one of them covers is the span its 32 lanes reach.
  // Giving a THREAD the consecutive run instead leaves a reduction's lanes
  // strided by that run, and the coalescer then touches four times the sectors
  // for the same bytes -- measured at exactly four times the ideal before this
  // was written as it is.  So a lane owns d_v/32 channels that are 32 apart, one
  // reduction covers 32 consecutive channels, and one pass covers kThreads/32
  // rows.  The shared read is one 32-bit access per channel across 32
  // consecutive banks, which is conflict-free by construction.
  constexpr int kFanLanes = 32;
  constexpr int kFanPerThr = kD / kFanLanes;
  constexpr int kFanRows = kThreads / kFanLanes;
  constexpr int kFanPasses = C / kFanRows;
  static_assert(kD % kFanLanes == 0 && kThreads % kFanLanes == 0,
                "a reduction's lanes must tile both an output row and the CTA");
  static_assert(kFanPasses * kFanRows == C && kFanPasses >= 1,
                "the fan-in's passes must cover the chunk exactly");
  const int fanRow = tid / kFanLanes, fanCol = tid % kFanLanes;
  float *po = o + base + (long)fanRow * kD + fanCol;
  const int fanE = fanRow * LdO + fanCol;

  // Output staging write index: this lane owns d_v = dv0 + (lane>>2) and that
  // + 8, at tokens 8j + 2*(lane&3) + {0,1} -- four elements in four rows of the
  // [token][d_v] buffer whose only per-chunk variation is the 8*j row step.
  const int oq = (lane & 3) * 2, om = lane >> 2;
  int oE[2][2];
#pragma unroll
  for (int e = 0; e < 2; ++e)
#pragma unroll
    for (int f = 0; f < 2; ++f) oE[e][f] = (oq + e) * LdO + dv0 + om + f * 8;

  //: THE STAGE IS WHAT IS SKIPPED; THE CURSOR IS NOT.  A dead group's V is never
  //: read, so its tiles are not issued -- but the ring slot and the source
  //: pointer advance either way, which is what keeps a staging group's index
  //: equal to its chunk's index and every address affine.
  auto stage_chunk = [&](bool go) {
    if (go) {
#pragma unroll
      for (int p = 0; p < kPasses; ++p) {
        stage_tile(svL + stgSlot + (uint32_t)(p * kRowsPerPass * (int)kRowV),
                   pv + (long)(p * kRowsPerPass * kD));
      }
    }
    pv += kChunkElems;
    stgSlot += kSlot;
    if (stgSlot == kRing) stgSlot = 0;
  };

  auto drain = [&](float *dst) {
#pragma unroll
    for (int p = 0; p < kFanPasses; ++p) {
      float *d = dst + (long)(p * kFanRows * kD);
      const int e0 = fanE + p * kFanRows * LdO;
#pragma unroll
      for (int e = 0; e < kFanPerThr; ++e)
        fan_in_add(d + e * kFanLanes, so[e0 + e * kFanLanes]);
    }
  };

  // ---- routed Kronecker coefficient production, digit-major ---------------
  // Thread `tid` owns token row `tid / kRowThr` and, within the block's
  // rectangle, the leading digits [leadLocal, leadLocal + kLeadPerThr) and the
  // trailing run [tailLocal, tailLocal + kTailPerThr).  Both divisors are
  // compile-time powers of two.  THE DIGITS OF A SLOT ARE NEVER COMPUTED: they
  // are the loop indices, and the block's rectangle is what keeps the local slot
  // index affine in them.
  const int prow = tid / kRowThr;
  const int rIn = tid % kRowThr;
  const int leadLocal = (rIn / kThrPerLead) * kLeadPerThr;
  const int tailLocal = (rIn % kThrPerLead) * kTailPerThr;
  const long pbase = (long)blockIdx.y * L * kWtot + (long)prow * kWtot;
  constexpr long kPChunk = (long)C * kWtot;
  const __nv_bfloat16 *pr0 = pread + pbase + lead_off + leadLocal;
  const __nv_bfloat16 *pr1 = pread + pbase + W0 + tail_off + tailLocal;
  const __nv_bfloat16 *pw0 = pwrite + pbase + lead_off + leadLocal;
  const __nv_bfloat16 *pw1 = pwrite + pbase + W0 + tail_off + tailLocal;

  const int slot0 = leadLocal * P::tail_span + tailLocal;  // local slot of this thread
  char *dstR = reinterpret_cast<char *>(sr) +
               (SWZ ? prow * (int)kRowS : prow * (int)kRowS + slot0 * 2);
  char *dstW = reinterpret_cast<char *>(sw) +
               (SWZ ? prow * (int)kRowS : prow * (int)kRowS + slot0 * 2);
  const int pPh = prow & (kChunkS - 1);
  int pOff[kLeadPerThr][kTailVec];
#pragma unroll
  for (int j = 0; j < kLeadPerThr; ++j)
#pragma unroll
    for (int i = 0; i < kTailVec; ++i)
      pOff[j][i] = SWZ ? ((((slot0 + j * P::tail_span) / 8 + i) & (kChunkS - 1)) ^ pPh) << 4
                       : ((j * P::tail_span + i * 8) * 2);

  //: THE PRODUCTION.  One trailing 16-byte vector is held at a time and reused
  //: across every leading digit this thread owns, so the register cost of the
  //: production does not grow with the block width; the instruction count is the
  //: same either way.  Each coefficient is one `mul_bf16x2` -- `ops.cuh`'s
  //: archetypal SELECTING entry, half an instruction per two coefficients on an
  //: architecture with bf16 SIMD arithmetic and 1 multiply + 1/2 pack per
  //: coefficient on one without it.
  auto produce = [&](const __nv_bfloat16 *p0, const __nv_bfloat16 *p1, char *dst) {
    __align__(8) unsigned short amp0[kLeadPerThr];
    if constexpr (kLeadPerThr == 1) {
      amp0[0] = *reinterpret_cast<const unsigned short *>(p0);
    } else if constexpr (kLeadPerThr == 2) {
      *reinterpret_cast<uint32_t *>(amp0) = *reinterpret_cast<const uint32_t *>(p0);
    } else {
      *reinterpret_cast<uint2 *>(amp0) = *reinterpret_cast<const uint2 *>(p0);
    }

#pragma unroll
    for (int i = 0; i < kTailVec; ++i) {
      __align__(16) uint32_t a1[4];
      *reinterpret_cast<uint4 *>(a1) = load_vec16(p1 + i * 8);
#pragma unroll
      for (int j = 0; j < kLeadPerThr; ++j) {
        __align__(16) uint32_t prod[4];
        const uint32_t a2 =
            splat_bf16x2(*reinterpret_cast<const __nv_bfloat16 *>(&amp0[j]));
#pragma unroll
        for (int e = 0; e < 4; ++e) prod[e] = mul_bf16x2(a2, a1[e]);
        store_vec16(dst + pOff[j][i], *reinterpret_cast<const uint4 *>(prod));
      }
    }
  };

  // ---- the membership test, and the schedule it builds --------------------
  // THE PREFIX TEST.  Level 0 is tested over the `lead_span` digits this block
  // spans and level 1 over its `tail_span`, and the trailing test DISAPPEARS
  // when the block spans that level entirely -- the clause is then a property of
  // the token rather than of the block and is true for any simplex-valued
  // routing.  Both spans are contiguous and aligned, so each is ONE access of
  // `or_run` whatever its length, and the read and write sides are tested
  // separately because a token needs its group only when one of them survives
  // ALL of the levels.
  constexpr bool kTailWhole = P::tail_span == W1;
  const long mem_base = (long)blockIdx.y * L * kWtot;
  const __nv_bfloat16 *mr_lead = pread + mem_base + lead_off;
  const __nv_bfloat16 *mw_lead = pwrite + mem_base + lead_off;
  const __nv_bfloat16 *mr_tail = pread + mem_base + W0 + tail_off;
  const __nv_bfloat16 *mw_tail = pwrite + mem_base + W0 + tail_off;

  auto member = [&](int t) {
    const long off = (long)t * kWtot;
    const uint32_t r0 = or_run<P::lead_span>(mr_lead + off);
    const uint32_t w0 = or_run<P::lead_span>(mw_lead + off);
    if constexpr (kTailWhole) {
      return ((r0 | w0) & kBf16MagnitudeMask) != 0u;
    } else {
      const uint32_t r1 = or_run<P::tail_span>(mr_tail + off);
      const uint32_t w1 = or_run<P::tail_span>(mw_tail + off);
      return ((r0 & kBf16MagnitudeMask) && (r1 & kBf16MagnitudeMask)) ||
             ((w0 & kBf16MagnitudeMask) && (w1 & kBf16MagnitudeMask));
    }
  };

  // ONE VOTE COVERS `kBallotLanes` TOKENS, which is `kGrpPerVote` whole groups
  // when a group is narrower than the lockstep unit and `kTokPerLane` passes of
  // one group when it is wider.  A unit owns WHOLE mask words, so the words are
  // written once each by one lane: no atomic, no zeroing, and one rendezvous
  // publishes the schedule.
  constexpr int kGrpPerVote = kBallotLanes / C >= 1 ? kBallotLanes / C : 1;
  constexpr int kTokPerLane = C / kBallotLanes >= 1 ? C / kBallotLanes : 1;
  constexpr int kVotesPerWord = 32 / kGrpPerVote;
  constexpr uint32_t kVoteMask =
      kGrpPerVote == 1 ? 0xFFFFFFFFu : (uint32_t)((1ull << C) - 1ull);
  static_assert(kGrpPerVote * C == kBallotLanes * kTokPerLane,
                "a vote must cover a whole number of groups exactly");

  const int words = (NC + 31) >> 5;
#pragma unroll 1
  for (int wd = warp; wd < words; wd += kWarps) {
    uint32_t bits = 0;
#pragma unroll 1
    for (int q = 0; q < kVotesPerWord; ++q) {
      const int t0 = ((wd << 5) + q * kGrpPerVote) * C;
      bool lv = false;
#pragma unroll
      for (int i = 0; i < kTokPerLane; ++i) {
        const int t = t0 + i * kBallotLanes + lane;
        if (t < L) lv |= member(t);
      }
      const uint32_t vote = lane_ballot(lv);
#pragma unroll
      for (int j = 0; j < kGrpPerVote; ++j)
        bits |= (uint32_t)(((vote >> (j * C)) & kVoteMask) != 0u) << (q * kGrpPerVote + j);
    }
    if (lane == 0) sched[wd] = bits;
  }
  rendezvous();  // the schedule is published

  //: THE SCHEDULE IS READ ONCE PER 32 GROUPS AND CARRIED IN A REGISTER WINDOW.
  //: The loop wants two bits of it -- the group it is COMPUTING and the group it
  //: is STAGING, `STAGES - 1` ahead -- and both walk forward by one per
  //: iteration.  So one word covers thirty-two iterations and a shift register
  //: covers the distance between the two readers; the naive form, two shared
  //: reads and two variable shifts per chunk, is per-chunk work that buys
  //: nothing.  `win` bit i is the liveness of group c + i.
  uint32_t ahead_word = sched[0];
  uint32_t win = 0;
#pragma unroll 1
  for (int s = 0; s < STAGES - 1; ++s) {
    const bool lv = s < NC && ((ahead_word >> s) & 1u);
    if (s < NC) stage_chunk(lv);
    stage_commit();
    win |= (uint32_t)lv << s;
  }

  bool prev_live = false;
#pragma unroll 1
  for (int c = 0; c < NC; ++c) {
    const bool live = win & 1u;
    if (live) {
      produce(pr0, pr1, dstR);
      produce(pw0, pw1, dstW);
    }
    pr0 += kPChunk; pr1 += kPChunk; pw0 += kPChunk; pw1 += kPChunk;

    stage_wait<STAGES - 2>();
    rendezvous();  // (1) V visible / R,W visible / last chunk's reads done / O drained

    const int nx = c + STAGES - 1;
    if ((nx & 31) == 0) ahead_word = sched[nx >> 5];
    const bool lv_nx = nx < NC && ((ahead_word >> (nx & 31)) & 1u);
    if (nx < NC) stage_chunk(lv_nx);
    stage_commit();
    win = (win >> 1) | ((uint32_t)lv_nx << (STAGES - 2));

    const uint32_t atv = svB + cmpSlot + rowTV;
    cmpSlot += kSlot;
    if (cmpSlot == kRing) cmpSlot = 0;

    // THE SKIP.  A dead group contributes the additive identity to every term,
    // so the whole of the arithmetic below is elided for it; the cursors above
    // and the ring's fence have already advanced without it.
    float acc[NTOK][4];
    if (live) {
    // ---- 1. readout: O^T = S^T . R^T, A operand straight out of the state ---
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

    // ---- 2. Gram: G = tril(R.W^T) over THIS BLOCK's slots -------------------
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
              store_bf16x2(&sg[tr * LdG + tc], pack_bf16x2(m0, m1));
              store_bf16x2(&sg[(tr + 8) * LdG + tc], pack_bf16x2(m2, m3));
            }
        }
      }
    }

    }

    // ---- 3. fan the PREVIOUS chunk's partial output into the accumulator ----
    if (c) {
      if (prev_live) drain(po);
      po += kChunkElems;
    }

    if (live) {
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
    {
      uint32_t va[MTOK][4];
#pragma unroll
      for (int kk = 0; kk < MTOK; ++kk) load_frag_t(va[kk], ADDRT_DV(atv, kk * 16));
#pragma unroll
      for (int m = 0; m < MTOK; ++m) {
        uint32_t gb[MTOK][4];
#pragma unroll
        for (int kk = 0; kk < MTOK; ++kk)
          load_frag(gb[kk], sgT + (uint32_t)((m * 16 * LdG + kk * 16) * 2));
#pragma unroll
        for (int kk = 0; kk < MTOK; ++kk)
#pragma unroll
          for (int p = 0; p < 2; ++p) mma(acc[2 * m + p], va[kk], &gb[kk][2 * p]);
      }
    }

    // ---- 6. stage this chunk's partial output; the fan-in is next iteration --
#pragma unroll
    for (int j = 0; j < NTOK; ++j) {
      const int jo = j * 8 * LdO;
      store_f32(&so[oE[0][0] + jo], acc[j][0]);
      store_f32(&so[oE[1][0] + jo], acc[j][1]);
      store_f32(&so[oE[0][1] + jo], acc[j][2]);
      store_f32(&so[oE[1][1] + jo], acc[j][3]);
    }
    }
    prev_live = live;
  }

  rendezvous();
  if (prev_live) drain(po);
#undef ADDRF
#undef ADDRT
#undef ADDRT_DV
}

// --------------------------------------------------------------- host side

// THE ARM SET.  Two measured axes (`C`, `SWZ`) and one this rung is on trial:
// `BC`, the block width, which trades the intra term's replication against
// register-resident state.  The three WIDTH CONFIGURATIONS are not axes: they
// are problem sizes.  (8,8) is N = 64 with ONE block, the same-work control
// against rung 1; (64,64) is the flagship N = 4096; (16,16) is N = 256, a real
// fan-in cheap enough for an fp64 reference to gate at full length.
#define RUNG3_ARMS(X)                                                        \
  X(16, false,  64,  8,  8) X(32, false,  64,  8,  8) X(64, false,  64,  8,  8) \
  X(16, true ,  64,  8,  8) X(32, true ,  64,  8,  8) X(64, true ,  64,  8,  8) \
  X(16, false,  64, 64, 64) X(32, false,  64, 64, 64) X(64, false,  64, 64, 64) \
  X(16, true ,  64, 64, 64) X(32, true ,  64, 64, 64) X(64, true ,  64, 64, 64) \
  X(32, false,  32, 64, 64) X(64, false,  32, 64, 64)                          \
  X(16, false, 128, 64, 64) X(32, false, 128, 64, 64) X(64, false, 128, 64, 64) \
  X(16, false, 256, 64, 64) X(32, false, 256, 64, 64)                          \
  X(16, false,  64, 16, 16) X(32, false,  64, 16, 16) X(64, false,  64, 16, 16)

template <int C, bool SWZ, int BC, int W0, int W1>
static void launch(const torch::Tensor &pr, const torch::Tensor &pw,
                   const torch::Tensor &v, torch::Tensor &o, int BH, int L) {
  constexpr size_t sm = Geom<C, SWZ, BC>::bytes;
  static bool configured = false;
  if (!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(routed_schedule_kernel<C, SWZ, W0, W1, BC>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int)sm));
    configured = true;
  }
  const dim3 grid(BlockPlan<W0, W1, BC>::blocks, BH);
  routed_schedule_kernel<C, SWZ, W0, W1, BC>
      <<<grid, kThreads, sm, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const __nv_bfloat16 *>(pr.data_ptr()),
          reinterpret_cast<const __nv_bfloat16 *>(pw.data_ptr()),
          reinterpret_cast<const __nv_bfloat16 *>(v.data_ptr()),
          o.data_ptr<float>(), L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor forward(torch::Tensor pr, torch::Tensor pw, torch::Tensor v,
                      int64_t chunk, bool swizzle, int64_t bc, int64_t w0, int64_t w1) {
  TORCH_CHECK(v.is_cuda() && v.scalar_type() == torch::kBFloat16 && v.is_contiguous());
  TORCH_CHECK(pr.is_contiguous() && pw.is_contiguous());
  TORCH_CHECK(pr.scalar_type() == torch::kBFloat16 && pw.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(v.dim() == 4 && v.size(3) == kD, "expect V [B,H,L,64]");
  TORCH_CHECK(pr.sizes() == pw.sizes() && pr.dim() == 4 && pr.size(3) == w0 + w1,
              "expect packed amplitudes [B,H,L,W0+W1]");
  const int B = v.size(0), H = v.size(1), L = v.size(2);
  TORCH_CHECK(pr.size(0) == B && pr.size(1) == H && pr.size(2) == L);
  TORCH_CHECK(L % chunk == 0, "L must be a multiple of the chunk");
  TORCH_CHECK(L / chunk <= kMaxGroups,
              "the schedule holds ", kMaxGroups, " token groups and this launch "
              "asks for ", L / chunk, "; a longer sequence is REFUSED rather "
              "than partly scheduled");
  //: The accumulator is zeroed, not merely allocated: the fan-in ADDS, so the
  //: identity of the sum has to be there before the first contribution.
  auto o = torch::zeros({B, H, L, (long)kD},
                        v.options().dtype(torch::kFloat32));
  const int BH = B * H;
#define RUNG3_DISPATCH(C_, W_, BC_, W0_, W1_)                            \
  if (chunk == C_ && swizzle == W_ && bc == BC_ && w0 == W0_ && w1 == W1_) { \
    launch<C_, W_, BC_, W0_, W1_>(pr, pw, v, o, BH, L);                  \
    return o;                                                            \
  }
  RUNG3_ARMS(RUNG3_DISPATCH)
#undef RUNG3_DISPATCH
  TORCH_CHECK(false, "unbuilt arm: chunk=", chunk, " swizzle=", swizzle, " BC=", bc,
              " widths=(", w0, ",", w1, ")");
}

__global__ void stamp_kernel(int *out) {
  out[0] = kBuildMagic;
  out[1] = kWarps * 1000 + kD;
  out[2] = design::kCapsDigest;
}

std::vector<int64_t> build_stamp(int64_t chunk, bool swizzle, int64_t bc, int64_t w0,
                                 int64_t w1) {
  auto t = torch::zeros({3}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  stamp_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(t.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto h = t.cpu();
  std::vector<int64_t> r{h.data_ptr<int>()[0], h.data_ptr<int>()[1], h.data_ptr<int>()[2]};
  cudaFuncAttributes a{};
#define RUNG3_CENSUS(C_, W_, BC_, W0_, W1_)                                        \
  if (chunk == C_ && swizzle == W_ && bc == BC_ && w0 == W0_ && w1 == W1_) {       \
    C10_CUDA_CHECK(cudaFuncGetAttributes(                                          \
        &a, routed_schedule_kernel<C_, W_, W0_, W1_, BC_>));                       \
    r.push_back(a.numRegs);                                                        \
    r.push_back((int64_t)a.sharedSizeBytes);                                       \
    r.push_back(a.binaryVersion);                                                  \
    r.push_back((int64_t)Geom<C_, W_, BC_>::bytes);                                \
    r.push_back((int64_t)a.maxThreadsPerBlock);                                    \
    r.push_back((int64_t)a.localSizeBytes);                                        \
    r.push_back((int64_t)Geom<C_, W_, BC_>::stages);                               \
    r.push_back((int64_t)BlockPlan<W0_, W1_, BC_>::blocks);                        \
    return r;                                                                      \
  }
  RUNG3_ARMS(RUNG3_CENSUS)
#undef RUNG3_CENSUS
  TORCH_CHECK(false, "unbuilt arm");
  return r;
}

//: The built set with every DERIVED quantity of an arm reported beside it: the
//: block plan's rectangle, the block count, the staging depth and the CTA target.
std::vector<std::vector<int64_t>> arms() {
  std::vector<std::vector<int64_t>> out;
#define RUNG3_LIST(C_, W_, BC_, W0_, W1_)                                     \
  out.push_back({C_, (int64_t)(W_), BC_, W0_, W1_,                            \
                 (int64_t)BlockPlan<W0_, W1_, BC_>::blocks,                   \
                 (int64_t)BlockPlan<W0_, W1_, BC_>::lead_span,                \
                 (int64_t)BlockPlan<W0_, W1_, BC_>::tail_span,                \
                 (int64_t)Geom<C_, W_, BC_>::stages,                          \
                 (int64_t)Geom<C_, W_, BC_>::bytes,                           \
                 (int64_t)kCtaTarget<C_, W_, BC_>});
  RUNG3_ARMS(RUNG3_LIST)
#undef RUNG3_LIST
  return out;
}

}  // namespace rung3

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &rung3::forward, "capacity-split routed chunked linear attention");
  m.def("build_stamp", &rung3::build_stamp, "build identity");
  m.def("arms", &rung3::arms, "the built arms with their derived block plan and geometry");
}
