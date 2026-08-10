// LADDER RUNG 5A -- THE SCHEDULE'S REPRESENTATION.
//
// Base: rung 4 (`rung4/csrc/routed_normalized_la.cu` at `72d1a5c`).  EXACTLY ONE
// THING CHANGES: where the membership test's clauses come from.  Rung 4
// evaluated them in the kernel, per (block, token), by OR-ing a run of raw
// amplitudes; this rung reads them out of a PRECOMPUTED UNION TABLE and ANDs
// them.  The predicate is bit-for-bit the same predicate -- the gate is that
// this rung's output is BIT-IDENTICAL to rung 4's at every one-block arm and at
// every support -- so nothing about the answer moves.  What moves is the cost of
// deciding, which stops scaling with the width of the sub-range a block spans.
//
// WHY THAT MATTERS: rung 4's test costs an `or_run` over the PARTIALLY SPANNED
// level's sub-range, so `BC = 32` -- the only built rectangle whose innermost
// level is the partial one, at a 32-digit sub-range against `BC = 128`'s two
// digits -- paid +34% / +37% of wall against +3.4% / +6.9% everywhere else.  A
// table entry is ONE WORD however wide the sub-range is.
//
// THE TABLE: ONE BIT PER (SIDE, ENTRY, TOKEN), AND THE ENTRIES ARE THE LAW'S
// ---------------------------------------------------------------------------
// The block geometry law says a block spans every level above the boundary at
// EXACTLY ONE digit, exactly one level PARTIALLY by an aligned sub-range, and
// every level below ENTIRELY.  The table carries exactly the clauses the test
// evaluates and no others:
//
//   * per (token, level, digit) rows for a level spanned at ONE digit;
//   * per (token, sub-range) unions for the PARTIALLY spanned level -- one entry
//     per SUB-RANGE, `width_j / c` of them, because every block that shares a
//     sub-range shares the bit;
//   * NOTHING for a level spanned ENTIRELY, whose clause is a property of the
//     token rather than of the block and holds for any simplex-valued routing.
//
// Both forms are one expression at two granularities -- entry `e` of level `l`
// is the OR over the digits of that level's `e`-th aligned run -- so the table
// is `sum_l width_l / span_l` entries per token per side, which for the two
// levels built here is `W0 / lead_span` plus, when the trailing level is not
// spanned entirely, `W1 / tail_span`.
//
// ITS SIZE IS `L x (sum_l width_l / span_l) x 2` BITS AND IS INDEPENDENT OF `N`.
// The Kronecker structure pays here exactly as it pays in the coefficients: the
// cost follows the SUM of the level widths, never their product.  The schedule
// gets RELATIVELY CHEAPER as the slot space grows.
//
// ONE STRUCTURE, TWO CONSUMERS.  The group schedule ORs a block's entry rows over
// a group's tokens; a per-token consumer reads the same rows per token.  They
// cannot disagree because there is nothing to disagree about: the bits are the
// same bits.
//
// WHAT IT COSTS TO BUILD.  One pass over the amplitudes, ONCE for the whole grid,
// where rung 4's prologue paid its test once per (block, token) -- `N/BC` times
// over.  It is a separate launch and its cost is reported on its own line, never
// folded into a kernel delta, exactly as `divide_pass` is.
//
// WHAT RUNG 4 CONTRIBUTED, AND IS UNCHANGED HERE: the self-normalising readout.
//
//     y[t] = num[t] / (den[t] + eps),   num[t] = sum_s R[t,s] S[t,s,0:d_v]
//                                       den[t] = sum_s R[t,s] S[t,s,d_v]
//
// `rola/ops/naive.py` is the spec and its only normalisation is `"global"`, so
// this is not a mode: it is what the readout IS.  Spec 4.2 fixes `g_read := 1`
// there, and that is why no read gain appears anywhere below -- under a ratio a
// per-token read scale cancels exactly, so carrying one would be carrying a
// quantity the answer does not depend on.
//
// WHICH AXIS THE EXTRA CHANNEL LANDS ON, AND WHAT IT COSTS
// --------------------------------------------------------
// The channel axis is the MMA's `m` axis here, and that is FORCED rather than
// chosen: only accumulator(m,n) -> A(m,k) is a free relabel, so if the mass
// state is to be an operand of the readout WITHOUT a memory round trip -- the
// property this whole design is built on -- its `n` axis must be the slots and
// its `m` axis must be the channel.  `m` has a quantum of `kMmaM`, so
// `d_v + 1 = 65` channels pad to 80, not to 72 (the `n` quantum, which this
// arrangement cannot use) and not to 128.  FIFTEEN OF THE ADDED TILE'S SIXTEEN
// ROWS ARE PADDING, in mma and in registers alike, and that is the price of the
// feature: one more `m` tile on each of the three terms that carry `d_v`.
//
// THE MASS TILE IS CARVED ACROSS THE UNITS ALONG THE SLOT AXIS
// ------------------------------------------------------------
// A fifth unit is not available: `kThreads = kUnits * kMmaUnitThreads` must
// divide the token axis and tile the block rectangle, and 160 threads does
// neither -- the production's `kThreads % C`, the staging's `kThreads / kChunkV`
// and the fan-in's `kThreads / 32` all stop covering the chunk exactly.  So the
// mass tile is carried by the four units that exist, and it is carved along the
// axis a single channel leaves free: the SLOTS.  Unit `u` owns the mass of the
// tiles `q * kMassUnits + u`, it holds a COMPLETE mass state for those slots and
// it produces a PARTIAL denominator over them -- exactly as each CTA holds a
// complete state for its own block and produces a partial output.  The intra
// term's free axis is tokens rather than slots, and it is carved the same way.
//
// THE CARVE IS IN THE ADDRESS, NOT IN A PREDICATE, AND THAT WAS MEASURED RATHER
// THAN ASSUMED.  The first build selected each unit's tiles with a per-mma test
// (`kk % kMassUnits == warp`) over the fragments the value channels had already
// loaded, which costs no data movement and reads like a free ride.  It is not
// one: the unit index is a runtime value, so the compiler emits one copy per
// unit AND the measured mma came back at FOUR TIMES the added tile --
// +75..90% instead of +18..22% -- because a predicated-off mma still occupies
// its issue slot on the tensor pipe.  So the unit's tiles are addressed rather
// than tested: its B operands are loaded at its own base, which is the value
// base plus `kMmaK * sizeof(bf16) * u` and therefore loop-invariant, and every
// index inside the loop is a compile-time constant again.  The ones column then
// costs exactly one unit's share of the mma, which is what it should cost.
//
// WHAT IT ADDS IN DATA MOVEMENT IS ONE FRAGMENT LOAD PER TILE IT OWNS -- the
// price of addressing rather than testing.  Its A operand is free either way:
// the literal 1 in the fold and the intra (an immediate; no memory at all) and
// its own state in the readout.
//
// WHY THE DIVISION IS NOT IN THIS KERNEL.  `den[t]` is a sum over the N/BC
// blocks and no block knows the total, so the ratio cannot be formed before the
// last contribution has landed -- a grid-wide dependency, which is the
// synchronisation the fan-in declined in rung 2.  The kernel therefore produces
// the readout's `d_v + 1` columns and `divide_pass` forms the ratio in one
// elementwise pass, whose cost is reported on its own line and never folded into
// a kernel delta.  `eps` is that pass's runtime argument, as it is production's.
//
// THE NUMERATOR AND THE DENOMINATOR ARE SEPARATE BUFFERS, and that is an
// alignment decision rather than a semantic one: a single `[.., d_v+1]` buffer
// gives every numerator row a 260-byte stride, which misaligns all 32 lanes of
// every fan-in reduction for the sake of packing one column.  The padding that
// matters is the MMA's, not the buffer's.
//
// WHAT RUNG 3 CONTRIBUTED, AND IS UNCHANGED HERE: a block may SKIP A GROUP OF
// TOKENS it has no work for, and a schedule says which groups those are.  The
// mass channel is skipped with them, exactly and for the same reason -- a group
// with no live slot deposits nothing into the mass either.
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
// THE MEMBERSHIP TEST IS A PREFIX TEST, AND THE TABLE IS ITS CLAUSES
// -------------------------------------------------------------------
// A slot is live for a token when EVERY level's amplitude at that slot's digit
// is nonzero, so the block holds work for the token exactly when
//
//     AND over levels l of ( OR over the digits the block spans at level l )
//
// which is multiplicative over levels and enumerates no slots.  Each surviving
// clause is one table entry, so the test is an AND of at most two words per
// thirty-two tokens, WHATEVER the spans are.  A level spanned ENTIRELY IS NOT
// TESTED and has no entry, because its clause asks whether the token has any
// nonzero amplitude at that level at all -- a property of the TOKEN, not of the
// block, and one that holds for every simplex-valued routing, since an entmax or
// softmax row sums to one and some digit survives.
//
// That last clause is the ONE place this test is CONSERVATIVE rather than exact,
// and the direction is the safe one: a level that really was all zero would make
// the test say LIVE for a group with no work, which costs time and changes no
// answer.  The test can never say DEAD for a group that has work, because every
// clause it does evaluate is exact.
//
// THE SCHEDULE: A BITMASK, BUILT ONCE, IN A PROLOGUE
// --------------------------------------------------
// One bit per (this block, token group).  The prologue is now a REDUCTION OVER
// THE TABLE rather than a pass over the amplitudes: a thread takes one group,
// ANDs the block's two entry rows over that group's token words on each side,
// ORs the sides, and a lockstep vote turns thirty-two of those into a whole mask
// word.  No two units write the same word, so there is no atomic and no zeroing,
// and one rendezvous publishes the schedule to the chunk loop.
//
// WHY THE PROLOGUE'S SHAPE CHANGED WITH IT.  Rung 4's prologue was a pass over
// TOKENS, because its clause was per token; this one is a pass over GROUPS,
// because a group's whole liveness now lives in `C/32` words.  A thread per
// group and a vote per thirty-two of them is the same publication with a
// thirty-two-fold shorter body.
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

namespace rung5 {

using namespace denseref::ops;
namespace design = denseref::design;

static_assert(design::kStateResidency == design::StateResidency::REGISTERS &&
                  kMmaAcceptsRegisterA,
              "SELECTION UNSATISFIABLE GIVEN AVAILABLE OPERATIONS: this kernel's "
              "fold accumulator IS the readout's A operand, so it needs a state "
              "that lives in registers and a built `mma` that reads A out of "
              "them; otherwise the state takes a memory round trip and this is a "
              "different algorithm.");

static constexpr int kBuildMagic = 5100107;

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

  //: THE UNION TABLE'S ENTRIES, WHICH ARE THIS LAW'S OWN CLAUSES.  Entry `e` of a
  //: level is the OR over the digits of that level's `e`-th aligned run, so a
  //: level spanned at one digit contributes its whole width and a partially
  //: spanned level contributes `width / span`.  A level spanned ENTIRELY
  //: contributes NOTHING: its clause is not evaluated, so there is nothing to
  //: tabulate.  The count is a SUM over levels of `width_l / span_l` and never a
  //: product, which is why the table does not grow with `N`.
  static constexpr int lead_entries = lead_span < W0 ? W0 / lead_span : 0;
  static constexpr int tail_entries = tail_span < W1 ? W1 / tail_span : 0;
  static constexpr int entries = lead_entries + tail_entries;
};

//: THE UNION TABLE'S ONE STRUCTURAL CONSTANT.  Tokens are the table's minor axis
//: and they are packed thirty-two to a word, which is both the lockstep unit's
//: width and the schedule mask's, so neither consumer needs a re-packing step.
//: A sequence that does not fill its last word leaves the tail bits ZERO, which
//: is the reading a dead token would have got anyway.
constexpr int kTokBits = 32;
__host__ __device__ constexpr int tok_words(int L) { return (L + kTokBits - 1) / kTokBits; }

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
//: The mass channel adds `4*NDV/kMassUnits` registers of state and `4*NTOK` of
//: readout accumulator per lane -- half of each being the padding rows of an m
//: tile that carries one useful channel, which is the register form of the same
//: 15-of-16 waste the mma pays.  Against rung 3's census that is +19 / +21 / +59
//: at BC = 64 and +12 / +24 / +0 at BC = 128, and it COSTS A RESIDENT CTA on two
//: arms.  It also prices the widest block out of the register file entirely: at
//: BC = 256 rung 3 already sat at 244 and 254 registers, and the mass tile spills
//: (72 B and 384 B of local, caught by the build gate), so those two arms are
//: NOT BUILT at this rung.  See the arm set below.
constexpr int reg_floor(int C, int BC) {
  return C == 16 ? (BC <= 64 ? 120 : 168)
       : C == 32 ? (BC <= 32 ? 128 : BC <= 64 ? 168 : 236)
                 : (BC <= 32 ? 168 : BC <= 64 ? 226 : 255);
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
  //: THE DENOMINATOR'S PARTIALS GET THEIR OWN BUFFER RATHER THAN A COLUMN OF THE
  //: ONE ABOVE.  `LdO` is 4 mod 16 fp32 elements, which is what makes the output
  //: staging's two access patterns conflict-free; widening it by the units' mass
  //: columns would break that for every value channel to serve one mass channel.
  //: One column per unit, `C` tokens long, summed by the drain.
  static constexpr int slot = C * LdV * (int)sizeof(__nv_bfloat16);
  static constexpr int fixed = (2 * C * LdS + C * LdG) * (int)sizeof(__nv_bfloat16) +
                               (C * LdO + kUnits * C) * (int)sizeof(float) +
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
routed_normalized_kernel(const __nv_bfloat16 *__restrict__ pread,
                         const __nv_bfloat16 *__restrict__ pwrite,
                         const __nv_bfloat16 *__restrict__ v,
                         const uint32_t *__restrict__ tbl,
                         float *__restrict__ o, float *__restrict__ den, int L) {
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

  //: THE MASS CHANNEL'S CARVE.  One channel cannot be split along the channel
  //: axis, so it is split along the one the mma leaves free.  The readout's and
  //: the fold's free axis is the SLOTS -- unit `u` owns the k-tiles with
  //: `kk % kMassUnits == u` and its local tile index `kk / kMassUnits` is a
  //: compile-time constant of the unrolled loop.  The intra term's is the
  //: TOKENS, carved the same way.  A unit that owns no tile of a term simply
  //: contributes nothing to it; the drain sums every unit's column either way.
  constexpr int kMassUnits = KD < kUnits ? KD : kUnits;
  constexpr int kMassKD = KD / kMassUnits;      // slot k-tiles of the mass, per unit
  constexpr int kMassTiles = 2 * kMassKD;       // its state's n-tiles, per unit
  constexpr int kIntraUnits = MTOK < kUnits ? MTOK : kUnits;
  constexpr int kIntraKD = MTOK / kIntraUnits;
  static_assert(KD % kMassUnits == 0 && NDV == kMassUnits * kMassTiles,
                "the mass channel's slots must be carved across the units exactly");

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
  float *sden = so + C * LdO;
  uint32_t *sched = reinterpret_cast<uint32_t *>(sden + kUnits * C);
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

  //: THE MASS CHANNEL'S OWN OPERAND ADDRESSES.  This unit's tile `q` is the
  //: value path's tile `q * kMassUnits + warp`, so its address is the value
  //: address at a compile-time `q` plus a LOOP-INVARIANT per-unit displacement
  //: of one mma k-tile per unit.  Under the swizzle that displacement is not a
  //: byte offset but a different table entry, so the unit's own table is built
  //: once here, by the same expression, at its own `g`.
  const uint32_t massOff = (uint32_t)(warp * kMmaK * (int)sizeof(__nv_bfloat16));
  uint32_t tabMT[kMassKD], tabMF[kMassKD];
#pragma unroll
  for (int q = 0; q < kMassKD; ++q) {
    const int g = 2 * (q * kMassUnits + warp);
    tabMT[q] = (uint32_t)((((g + cT) & (kChunkS - 1)) ^ (lr & (kChunkS - 1))) << 4);
    tabMF[q] = (uint32_t)((((g + cF) & (kChunkS - 1)) ^ (lr & (kChunkS - 1))) << 4);
  }
#define ADDRT_M(b_, R_, q_) \
  (SWZ ? ((b_) + (uint32_t)((R_) * (int)kRowS) + tabMT[(q_)]) \
       : ((b_) + (uint32_t)(((R_) * LdS + (2 * (q_) * kMassUnits) * 8) * 2) + massOff))
#define ADDRF_M(b_, R_, q_) \
  (SWZ ? ((b_) + (uint32_t)((R_) * (int)kRowS) + tabMF[(q_)]) \
       : ((b_) + (uint32_t)(((R_) * LdS + (2 * (q_) * kMassUnits) * 8) * 2) + massOff))

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

  //: THE d_v+1-th CHANNEL'S STATE, in the same accumulator layout and on the
  //: same recurrence: m = the mass tile (row 0 is the channel, rows 1..15 are
  //: the m quantum's padding and are identically zero), n = this unit's share of
  //: the block's slots.
  float mass_st[kMassTiles][4];
#pragma unroll
  for (int j = 0; j < kMassTiles; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) mass_st[j][e] = 0.f;

  //: THE ONES OPERAND, WHICH IS AN IMMEDIATE.  The value of the mass channel is
  //: literally 1, so the A fragment of the fold and of the intra is the constant
  //: whose row 0 is all ones and whose padding rows are zero.  In this atom a
  //: lane holds A rows `lane/4` and `lane/4 + 8` in registers {0,2} and {1,3},
  //: so exactly the four lanes of the first row carry the value and no lane
  //: touches memory to get it.
  constexpr uint32_t kOnesPair = 0x3F803F80u;   // two bf16 1.0
  uint32_t ones_a[4];
  ones_a[0] = (lane >> 2) == 0 ? kOnesPair : 0u;
  ones_a[1] = 0u;
  ones_a[2] = ones_a[0];
  ones_a[3] = 0u;

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
  //: The denominator's own accumulator is one value per token, so its fan-in is
  //: one reduction per token and its lanes are consecutive tokens.  The lane's
  //: own offset is folded in HERE, once, because a `float*` plus an `int` is a
  //: widening multiply and this pointer is read inside the chunk loop.
  float *pden = den + (long)blockIdx.y * L + tid;

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

  //: THE UNITS' PARTIAL DENOMINATORS MEET HERE, and they meet at no cost in
  //: barriers: the rendezvous that already separates a chunk's staging from the
  //: next chunk's drain is the one that publishes them.  One reduction per token
  //: goes out, so the denominator's contributor count is the numerator's.
  auto drain_den = [&](float *dst) {
    if (tid < C) {
      float s = 0.f;
#pragma unroll
      for (int u = 0; u < kUnits; ++u) s += sden[u * C + tid];
      fan_in_add(dst, s);
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

  // ---- the membership test, read out of the union table --------------------
  // THE BLOCK'S TWO ENTRY ROWS.  Level 0's entry is the aligned run of digits
  // this block spans there and level 1's is its sub-range; the trailing row
  // DISAPPEARS when the block spans that level entirely, exactly as the clause
  // does.  The two sides are separate rows because a token needs its group only
  // when ONE of them survives ALL of the levels, and because a per-side consumer
  // wants them apart.
  //: THE THREE SHAPES A BLOCK'S CLAUSE SET CAN TAKE, and they are the law's
  //: cases rather than a fallback chain: a level spanned ENTIRELY has no entry,
  //: so a block that spans BOTH levels entirely -- which is a block that IS the
  //: whole slot space, so there is exactly one of them and nothing to skip --
  //: has NO clause at all and is live for every token.
  constexpr bool kTailWhole = P::tail_span == W1;
  constexpr bool kNoClause = P::entries == 0;
  constexpr int kEntries = P::entries;
  const int tokWords = tok_words(L);
  //: The row base is formed ONCE, here, out of the loop: `blockIdx.y` is the
  //: (b,h) plane and the two side rows of one plane are `kEntries` apart.
  const long rowR = ((long)(2 * blockIdx.y + 0) * kEntries) * tokWords;
  const long rowW = ((long)(2 * blockIdx.y + 1) * kEntries) * tokWords;
  const int e_lead = kNoClause ? 0 : blk / P::tail_groups;
  const int e_tail = kTailWhole ? 0 : P::lead_entries + (blk % P::tail_groups);
  const uint32_t *tR0 = tbl + rowR + (long)e_lead * tokWords;
  const uint32_t *tW0 = tbl + rowW + (long)e_lead * tokWords;
  const uint32_t *tR1 = tbl + rowR + (long)e_tail * tokWords;
  const uint32_t *tW1 = tbl + rowW + (long)e_tail * tokWords;

  //: THE TEST, THIRTY-TWO TOKENS AT A TIME.  This is rung 4's `member` with the
  //: amplitude runs already reduced: the AND is over the levels, the OR is over
  //: the sides, and every operand is one word.
  auto tok_word = [&](int w) -> uint32_t {
    if constexpr (kNoClause) {
      return 0xFFFFFFFFu;
    } else if constexpr (kTailWhole) {
      return tR0[w] | tW0[w];
    } else {
      return (tR0[w] & tR1[w]) | (tW0[w] & tW1[w]);
    }
  };

  // A GROUP'S LIVENESS LIVES IN `C / 32` WORDS when a group is at least as wide
  // as the packing, and in one FIELD of one word when it is narrower.  A thread
  // takes a group and a lockstep vote turns thirty-two of them into a mask word,
  // so the words are written once each by one lane: no atomic and no zeroing.
  constexpr int kWordsPerGrp = C >= kTokBits ? C / kTokBits : 1;
  constexpr int kGrpPerWord = kTokBits / C >= 1 ? kTokBits / C : 1;
  constexpr uint32_t kGrpMask =
      C >= kTokBits ? 0xFFFFFFFFu : (uint32_t)((1ull << C) - 1ull);
  static_assert(C % kTokBits == 0 || kTokBits % C == 0,
                "a token group and the table's packing must divide one another");

  const int words = (NC + 31) >> 5;
#pragma unroll 1
  for (int g0 = 0; g0 < NC; g0 += kThreads) {
    const int g = g0 + tid;
    bool lv = false;
    if (g < NC) {
      if constexpr (C >= kTokBits) {
#pragma unroll
        for (int i = 0; i < kWordsPerGrp; ++i) lv |= tok_word(g * kWordsPerGrp + i) != 0u;
      } else {
        lv = (tok_word(g / kGrpPerWord) & (kGrpMask << (C * (g % kGrpPerWord)))) != 0u;
      }
    }
    const uint32_t vote = lane_ballot(lv);
    const int wd = (g0 >> 5) + warp;
    if (lane == 0 && wd < words) sched[wd] = vote;
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
    float macc[NTOK][4];
    if (live) {
    // ---- 1. readout: O^T = S^T . R^T, A operand straight out of the state ---
    //      and, on the same B fragments, the d_v+1-th channel's row of it.
    {
      uint32_t sa[KD][4];
#pragma unroll
      for (int kk = 0; kk < KD; ++kk) {
        sa[kk][0] = pack_bf16x2(st[2 * kk][0], st[2 * kk][1]);
        sa[kk][1] = pack_bf16x2(st[2 * kk][2], st[2 * kk][3]);
        sa[kk][2] = pack_bf16x2(st[2 * kk + 1][0], st[2 * kk + 1][1]);
        sa[kk][3] = pack_bf16x2(st[2 * kk + 1][2], st[2 * kk + 1][3]);
      }
      uint32_t ma[kMassKD][4];
#pragma unroll
      for (int q = 0; q < kMassKD; ++q) {
        ma[q][0] = pack_bf16x2(mass_st[2 * q][0], mass_st[2 * q][1]);
        ma[q][1] = pack_bf16x2(mass_st[2 * q][2], mass_st[2 * q][3]);
        ma[q][2] = pack_bf16x2(mass_st[2 * q + 1][0], mass_st[2 * q + 1][1]);
        ma[q][3] = pack_bf16x2(mass_st[2 * q + 1][2], mass_st[2 * q + 1][3]);
      }
#pragma unroll
      for (int j = 0; j < NTOK; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) { acc[j][e] = 0.f; macc[j][e] = 0.f; }
#pragma unroll
      for (int m = 0; m < MTOK; ++m) {
        uint32_t rb[KD][4];
#pragma unroll
        for (int kk = 0; kk < KD; ++kk) load_frag(rb[kk], ADDRT(brT, m * 16, 2 * kk));
#pragma unroll
        for (int kk = 0; kk < KD; ++kk)
#pragma unroll
          for (int p = 0; p < 2; ++p) mma(acc[2 * m + p], sa[kk], &rb[kk][2 * p]);
        if (warp < kMassUnits) {
          uint32_t mb[kMassKD][4];
#pragma unroll
          for (int q = 0; q < kMassKD; ++q)
            load_frag(mb[q], ADDRT_M(brT, m * 16, q));
#pragma unroll
          for (int q = 0; q < kMassKD; ++q)
#pragma unroll
            for (int p = 0; p < 2; ++p)
              mma(macc[2 * m + p], ma[q], &mb[q][2 * p]);
        }
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
      if (prev_live) { drain(po); drain_den(pden); }
      po += kChunkElems;
      pden += C;
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
      //: The d_v+1-th channel deposits through the same W panel at this unit's
      //: own slots, with the literal 1 for its A operand.
      if (warp < kMassUnits) {
#pragma unroll
        for (int q = 0; q < kMassKD; ++q) {
          uint32_t mb[4];
          load_frag_t(mb, ADDRF_M(bwF, kk * 16, q));
          mma(mass_st[2 * q], ones_a, &mb[0]);
          mma(mass_st[2 * q + 1], ones_a, &mb[2]);
        }
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
      //: THE INTRA TERM'S SHARE OF THE MASS, IN ONE GUARDED REGION.  Its free
      //: axis is the TOKENS, so it is carved over those at this unit's own base,
      //: and it needs none of the fragments above -- so it is hoisted out of the
      //: value loop entirely, which is both the clearer statement and the one
      //: that gives the guard something to guard.  When the chunk has fewer
      //: token k-tiles than there are units, the units past the last tile have
      //: no share and skip the whole region.
      if (warp < kIntraUnits) {
#pragma unroll
        for (int m = 0; m < MTOK; ++m)
#pragma unroll
          for (int q = 0; q < kIntraKD; ++q) {
            uint32_t mg[4];
            load_frag(mg, sgT + (uint32_t)((m * 16 * LdG + q * kIntraUnits * 16) * 2) +
                              massOff);
#pragma unroll
            for (int p = 0; p < 2; ++p) mma(macc[2 * m + p], ones_a, &mg[2 * p]);
          }
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
    //: THE MASS ROW IS ROW 0 OF ITS TILE, which in this atom's accumulator is
    //: held by the four lanes of the first group, two tokens each; the tile's
    //: other fifteen rows are the m quantum's padding and are staged by nobody.
    if (lane < 4) {
#pragma unroll
      for (int j = 0; j < NTOK; ++j) {
        store_f32(&sden[warp * C + j * 8 + lane * 2], macc[j][0]);
        store_f32(&sden[warp * C + j * 8 + lane * 2 + 1], macc[j][1]);
      }
    }
    }
    prev_live = live;
  }

  rendezvous();
  if (prev_live) { drain(po); drain_den(pden); }
#undef ADDRF
#undef ADDRT
#undef ADDRT_M
#undef ADDRF_M
#undef ADDRT_DV
}

// ------------------------------------------------------- the union table
//: THE TABLE IS BUILT ONCE FOR THE WHOLE GRID, which is the first thing this
//: rung buys: rung 4 evaluated the same clauses inside every one of the `N/BC`
//: blocks that share them.  A lockstep unit takes thirty-two consecutive tokens,
//: one per lane, and votes once per entry; the vote IS the word, so nothing is
//: packed afterwards and nothing is zeroed first.
//:
//: Its cost does not depend on `BC`: a lane reads its token's amplitude row once
//: per level that has entries, whatever the spans partition it into.
template <int W0, int W1, int BC>
__global__ void union_table_kernel(const __nv_bfloat16 *__restrict__ pread,
                                   const __nv_bfloat16 *__restrict__ pwrite,
                                   uint32_t *__restrict__ tbl, int L) {
  using P = BlockPlan<W0, W1, BC>;
  constexpr int kWtot = W0 + W1;
  const int tokWords = tok_words(L);
  const int w = blockIdx.x * (blockDim.x / kBallotLanes) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (w >= tokWords) return;  // a whole lockstep unit leaves together
  //: A TOKEN PAST THE END READS TOKEN ZERO AND VOTES DEAD.  Its bit has no
  //: consumer -- no group contains it -- and reading in range is what keeps the
  //: lockstep unit converged for the vote.
  const int t = w * kTokBits + lane;
  const bool real = t < L;
  const long off = (long)blockIdx.y * L * kWtot + (long)(real ? t : 0) * kWtot;

  //: ADDRESSING DISCIPLINE, THE SAME ONE THE CHUNK KERNEL KEEPS.  Every 64-bit
  //: address is formed ONCE and afterwards only advanced, by a stride that is
  //: itself 64-bit -- so no `pointer + int` widening multiply reaches a body.
  const long entryStride = tokWords;
  uint32_t *const outBase[2] = {
      tbl + ((long)(2 * blockIdx.y + 0) * P::entries) * entryStride + w,
      tbl + ((long)(2 * blockIdx.y + 1) * P::entries) * entryStride + w};
  const __nv_bfloat16 *const ampBase[2] = {pread + off, pwrite + off};

#pragma unroll
  for (int side = 0; side < 2; ++side) {
    const __nv_bfloat16 *p = ampBase[side];
    uint32_t *out = outBase[side];
#pragma unroll
    for (int e = 0; e < P::lead_entries; ++e) {
      const uint32_t v = or_run<P::lead_span>(p);
      const uint32_t vote = lane_ballot(real && (v & kBf16MagnitudeMask) != 0u);
      if (lane == 0) *out = vote;
      p += P::lead_span;
      out += entryStride;
    }
    p = ampBase[side] + W0;
#pragma unroll
    for (int e = 0; e < P::tail_entries; ++e) {
      const uint32_t v = or_run<P::tail_span>(p);
      const uint32_t vote = lane_ballot(real && (v & kBf16MagnitudeMask) != 0u);
      if (lane == 0) *out = vote;
      p += P::tail_span;
      out += entryStride;
    }
  }
}

// --------------------------------------------------------------- host side

// THE ARM SET.  Rung 3's, arm for arm, MINUS THE TWO BC = 256 ARMS: the mass
// tile's registers do not fit there and the build gate says so in local bytes
// rather than in a footnote (see `reg_floor`).  Two measured axes (`C`, `SWZ`)
// and `BC`, which carries rung 2's measured optimum and is not a tuned axis.
// The three WIDTH CONFIGURATIONS are not axes: they are problem sizes.  (8,8) is
// N = 64 with ONE block, the same-work control against rung 1; (64,64) is the
// flagship N = 4096; (16,16) is N = 256, a real fan-in cheap enough for an fp64
// reference to gate at full length.
#define RUNG5_ARMS(X)                                                        \
  X(16, false,  64,  8,  8) X(32, false,  64,  8,  8) X(64, false,  64,  8,  8) \
  X(16, true ,  64,  8,  8) X(32, true ,  64,  8,  8) X(64, true ,  64,  8,  8) \
  X(16, false,  64, 64, 64) X(32, false,  64, 64, 64) X(64, false,  64, 64, 64) \
  X(16, true ,  64, 64, 64) X(32, true ,  64, 64, 64) X(64, true ,  64, 64, 64) \
  X(32, false,  32, 64, 64) X(64, false,  32, 64, 64)                          \
  X(16, false, 128, 64, 64) X(32, false, 128, 64, 64) X(64, false, 128, 64, 64) \
  X(16, false,  64, 16, 16) X(32, false,  64, 16, 16) X(64, false,  64, 16, 16)

template <int C, bool SWZ, int BC, int W0, int W1>
static void launch(const torch::Tensor &pr, const torch::Tensor &pw,
                   const torch::Tensor &v, const torch::Tensor &tbl,
                   torch::Tensor &o, torch::Tensor &den, int BH, int L) {
  constexpr size_t sm = Geom<C, SWZ, BC>::bytes;
  static bool configured = false;
  if (!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(routed_normalized_kernel<C, SWZ, W0, W1, BC>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int)sm));
    configured = true;
  }
  const dim3 grid(BlockPlan<W0, W1, BC>::blocks, BH);
  routed_normalized_kernel<C, SWZ, W0, W1, BC>
      <<<grid, kThreads, sm, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const __nv_bfloat16 *>(pr.data_ptr()),
          reinterpret_cast<const __nv_bfloat16 *>(pw.data_ptr()),
          reinterpret_cast<const __nv_bfloat16 *>(v.data_ptr()),
          reinterpret_cast<const uint32_t *>(tbl.data_ptr()),
          o.data_ptr<float>(), den.data_ptr<float>(), L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

//: THE TABLE, AS A TENSOR: `[B, H, 2, entries, L/32]` of packed bits, sides
//: apart because a per-side consumer wants them apart and the group schedule
//: does not care.  It is a SEPARATE ENTRY POINT so that its cost is reported on
//: its own line, as `divide_pass` is, and never folded into a kernel delta.
template <int W0, int W1, int BC>
static torch::Tensor build_table(const torch::Tensor &pr, const torch::Tensor &pw,
                                 int BH, int L) {
  constexpr int E = BlockPlan<W0, W1, BC>::entries;
  const int tokWords = tok_words(L);
  auto tbl = torch::empty({BH, 2, E, tokWords},
                          pr.options().dtype(torch::kInt32));
  constexpr int kTblThreads = 128;
  constexpr int kTblWarps = kTblThreads / kBallotLanes;
  const dim3 grid((tokWords + kTblWarps - 1) / kTblWarps, BH);
  union_table_kernel<W0, W1, BC><<<grid, kTblThreads, 0,
                                   at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16 *>(pr.data_ptr()),
      reinterpret_cast<const __nv_bfloat16 *>(pw.data_ptr()),
      reinterpret_cast<uint32_t *>(tbl.data_ptr()), L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return tbl;
}

//: THE RATIO, IN ONE ELEMENTWISE PASS.  It cannot be in the kernel: `den[t]` is
//: a sum over the N/BC blocks and no block knows the total, so forming the
//: ratio there would need the grid-wide dependency the fan-in declined.  `eps`
//: is a runtime argument -- it is a constant of the DEFINITION and belongs to
//: the definition's file, not to a kernel's template parameters.  The divide is
//: round-to-nearest by name (`--use_fast_math` would otherwise take the
//: approximate reciprocal, and the whole point of this rung is the quotient).
__global__ void divide_kernel(const float *__restrict__ num,
                              const float *__restrict__ den,
                              float *__restrict__ y, long vecs, float eps) {
  constexpr int kVec = 4;
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= vecs) return;
  const float d = den[i / (kD / kVec)] + eps;
  const float4 x = reinterpret_cast<const float4 *>(num)[i];
  float4 r;
  r.x = __fdiv_rn(x.x, d);
  r.y = __fdiv_rn(x.y, d);
  r.z = __fdiv_rn(x.z, d);
  r.w = __fdiv_rn(x.w, d);
  reinterpret_cast<float4 *>(y)[i] = r;
}

torch::Tensor divide_pass(torch::Tensor num, torch::Tensor den, double eps) {
  TORCH_CHECK(num.is_contiguous() && den.is_contiguous());
  TORCH_CHECK(num.scalar_type() == torch::kFloat32 &&
              den.scalar_type() == torch::kFloat32);
  TORCH_CHECK(num.size(-1) == kD, "expect a numerator of d_v columns");
  auto y = torch::empty_like(num);
  const long vecs = num.numel() / 4;
  const int threads = 256;
  divide_kernel<<<(int)((vecs + threads - 1) / threads), threads, 0,
                  at::cuda::getCurrentCUDAStream()>>>(
      num.data_ptr<float>(), den.data_ptr<float>(), y.data_ptr<float>(), vecs,
      (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

//: THE READOUT'S d_v+1 COLUMNS, as the two buffers their shapes want: the
//: numerator `[B,H,L,d_v]` and the denominator `[B,H,L]`.  `forward` is this
//: followed by `divide_pass`, and the two are separable so that the ratio's cost
//: is reported on its own line and so that the NUMERATOR can be gated
//: bit-for-bit against the rung below, which has no denominator to compare.
//: THE ARGUMENT CHECKS OF A LAUNCH, SHARED BY THE TABLE AND THE KERNEL so that
//: the two can never be asked for different problems.
static void check_inputs(const torch::Tensor &pr, const torch::Tensor &pw,
                         const torch::Tensor &v, int64_t chunk, int64_t w0,
                         int64_t w1) {
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
}

torch::Tensor union_table(torch::Tensor pr, torch::Tensor pw, torch::Tensor v,
                          int64_t chunk, int64_t bc, int64_t w0, int64_t w1) {
  check_inputs(pr, pw, v, chunk, w0, w1);
  const int BH = v.size(0) * v.size(1), L = v.size(2);
#define RUNG5_TABLE(C_, W_, BC_, W0_, W1_)                                    \
  if (chunk == C_ && bc == BC_ && w0 == W0_ && w1 == W1_)                     \
    return build_table<W0_, W1_, BC_>(pr, pw, BH, L);
  RUNG5_ARMS(RUNG5_TABLE)
#undef RUNG5_TABLE
  TORCH_CHECK(false, "unbuilt arm: chunk=", chunk, " BC=", bc, " widths=(", w0,
              ",", w1, ")");
}

std::vector<torch::Tensor> forward_raw(torch::Tensor pr, torch::Tensor pw,
                                       torch::Tensor v, torch::Tensor tbl,
                                       int64_t chunk, bool swizzle,
                                       int64_t bc, int64_t w0, int64_t w1) {
  TORCH_CHECK(tbl.is_cuda() && tbl.is_contiguous() &&
                  tbl.scalar_type() == torch::kInt32,
              "the union table is a contiguous int32 tensor of packed bits");
  TORCH_CHECK(v.is_cuda() && v.scalar_type() == torch::kBFloat16 && v.is_contiguous());
  check_inputs(pr, pw, v, chunk, w0, w1);
  const int B = v.size(0), H = v.size(1), L = v.size(2);
  //: The accumulator is zeroed, not merely allocated: the fan-in ADDS, so the
  //: identity of the sum has to be there before the first contribution.
  auto opts = v.options().dtype(torch::kFloat32);
  auto o = torch::zeros({B, H, L, (long)kD}, opts);
  auto den = torch::zeros({B, H, L}, opts);
  const int BH = B * H;
#define RUNG5_DISPATCH(C_, W_, BC_, W0_, W1_)                                  \
  if (chunk == C_ && swizzle == W_ && bc == BC_ && w0 == W0_ && w1 == W1_) {   \
    const long want = (long)BH * 2 * BlockPlan<W0_, W1_, BC_>::entries *      \
                      tok_words(L);                                          \
    TORCH_CHECK(tbl.numel() == want,                                           \
                "the union table was built for a different arm");              \
    launch<C_, W_, BC_, W0_, W1_>(pr, pw, v, tbl, o, den, BH, L);              \
    return {o, den};                                                           \
  }
  RUNG5_ARMS(RUNG5_DISPATCH)
#undef RUNG5_DISPATCH
  TORCH_CHECK(false, "unbuilt arm: chunk=", chunk, " swizzle=", swizzle, " BC=", bc,
              " widths=(", w0, ",", w1, ")");
}

torch::Tensor forward(torch::Tensor pr, torch::Tensor pw, torch::Tensor v,
                      int64_t chunk, bool swizzle, int64_t bc, int64_t w0, int64_t w1,
                      double eps) {
  auto tbl = union_table(pr, pw, v, chunk, bc, w0, w1);
  auto nd = forward_raw(pr, pw, v, tbl, chunk, swizzle, bc, w0, w1);
  return divide_pass(nd[0], nd[1], eps);
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
#define RUNG5_CENSUS(C_, W_, BC_, W0_, W1_)                                        \
  if (chunk == C_ && swizzle == W_ && bc == BC_ && w0 == W0_ && w1 == W1_) {       \
    C10_CUDA_CHECK(cudaFuncGetAttributes(                                          \
        &a, routed_normalized_kernel<C_, W_, W0_, W1_, BC_>));                       \
    r.push_back(a.numRegs);                                                        \
    r.push_back((int64_t)a.sharedSizeBytes);                                       \
    r.push_back(a.binaryVersion);                                                  \
    r.push_back((int64_t)Geom<C_, W_, BC_>::bytes);                                \
    r.push_back((int64_t)a.maxThreadsPerBlock);                                    \
    r.push_back((int64_t)a.localSizeBytes);                                        \
    r.push_back((int64_t)Geom<C_, W_, BC_>::stages);                               \
    r.push_back((int64_t)BlockPlan<W0_, W1_, BC_>::blocks);                        \
    r.push_back((int64_t)BlockPlan<W0_, W1_, BC_>::entries);                       \
    return r;                                                                      \
  }
  RUNG5_ARMS(RUNG5_CENSUS)
#undef RUNG5_CENSUS
  TORCH_CHECK(false, "unbuilt arm");
  return r;
}

//: The built set with every DERIVED quantity of an arm reported beside it: the
//: block plan's rectangle, the block count, the staging depth and the CTA target.
std::vector<std::vector<int64_t>> arms() {
  std::vector<std::vector<int64_t>> out;
#define RUNG5_LIST(C_, W_, BC_, W0_, W1_)                                     \
  out.push_back({C_, (int64_t)(W_), BC_, W0_, W1_,                            \
                 (int64_t)BlockPlan<W0_, W1_, BC_>::blocks,                   \
                 (int64_t)BlockPlan<W0_, W1_, BC_>::lead_span,                \
                 (int64_t)BlockPlan<W0_, W1_, BC_>::tail_span,                \
                 (int64_t)Geom<C_, W_, BC_>::stages,                          \
                 (int64_t)Geom<C_, W_, BC_>::bytes,                           \
                 (int64_t)kCtaTarget<C_, W_, BC_>,                            \
                 (int64_t)BlockPlan<W0_, W1_, BC_>::entries});
  RUNG5_ARMS(RUNG5_LIST)
#undef RUNG5_LIST
  return out;
}

}  // namespace rung5

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &rung5::forward, "self-normalising routed chunked linear attention");
  m.def("forward_raw", &rung5::forward_raw, "the readout's d_v+1 columns, undivided");
  m.def("union_table", &rung5::union_table, "the precomputed membership union table");
  m.def("divide_pass", &rung5::divide_pass, "the ratio, one elementwise pass");
  m.def("build_stamp", &rung5::build_stamp, "build identity");
  m.def("arms", &rung5::arms, "the built arms with their derived block plan and geometry");
}
