// LADDER RUNG 5D -- THE SCHEDULE IS FORMED, THEN CONSUMED.
//
// Base: rung 5B (`rung5b/csrc/perside_compaction_la.cu` at `7cc0545`).  EXACTLY
// ONE THING CHANGES: WHERE THE MEMBERSHIP AND HAPPENS.  The partition, the
// staircase, the gather and every arithmetic term are rung 5B's, so this rung is
// BIT-IDENTICAL to it in every regime and the gate ASSERTS that rather than
// observing it.
//
// THE DEFECT THIS FIXES.  Rung 5B never FORMED a schedule.  The AND of the
// membership clauses lived INSIDE the walk that finds super-chunk boundaries:
//
//     rm = *tR0 & *tR1 & inrange
//
// sat in the same loop body that popcounts, ballots and emits indices.  Two
// consequences followed from that fusion, and both are structural rather than
// tuned:
//
//   * A SUPER-CHUNK THAT CLOSES MID-WORD LEFT THE WORD BEHIND.  The walk broke
//     with `cookBit = cut` and did NOT advance its four table cursors, so the
//     next call RE-LOADED the same words per side and RE-ANDED them, then
//     re-derived their prefix counts against a fresh `inrange` mask.  That is
//     exactly one redundant word per super-chunk: `L/32 + L/C` word decodes
//     against an ideal `L/32`, which is 3x at `C = 16`, 2x at `C = 32` and 1.5x
//     at `C = 64`.
//   * FUSING PUT THE WHOLE THING ON ONE WARP.  The AND is a pure map over words
//     with no dependencies and affine addresses; only the boundary search is
//     serial.  Fused, the serial half held the parallel half hostage on warp 0.
//
// `inrange` WAS AN ARTIFACT OF THE RE-ENTRY, not part of the schedule.  Its only
// job above the last word's padding mask was to skip the bits before `cookBit` --
// that is, to undo the fact that re-entry re-derived a word instead of reading
// one.  It is GONE here, and the padding mask moved into the form.
//
// THE TWO PHASES
// --------------
// FORM.  For each word of the token axis, AND the clause terms -- one per level
// spanned at a single digit, one for the partially spanned level -- and write ONE
// WORD PER SIDE into a resident ring in shared memory.  A level spanned entirely
// contributes nothing, exactly as its clause does.  This is a PURE MAP: no
// dependencies, no boundary logic, all `kThreads` threads, affine addresses, and
// about nine instructions covering thirty-two tokens.
//
// CONSUME.  Walk the FORMED bitmap: popcount, prefix, emit, find the boundary.
// Serial and on one lockstep unit, but READING a resident bitmap rather than
// re-deriving one.  Once formed, NOTHING ANDs AGAIN.
//
// THE CONSUMED BITS ARE CLEARED IN PLACE, which is what retires `inrange`.  When
// a super-chunk closes mid-word the walk writes the word back with the bits it
// took removed, so re-entry reads the remainder and the prefix counts it needs
// are the counts of what is left.  ONE SHARED STORE per super-chunk replaces a
// re-load, a re-AND and a mask on every access.
//
// IT IS A RING, AND THE TWO WAITS ARE ORDINARY
// ---------------------------------------------
// The former runs ahead of the walk in a ring of `kFormWords` words per side.
// Two things can happen and neither is an exception:
//
//   * THE FORMER'S HEAD MEETS ITS TAIL.  It stops forming until the walk frees
//     slots.  That is backpressure, and it is the ring working.
//   * THE WALK OUTRUNS THE FORMED REGION.  Its state -- the cursor and the two
//     running counts -- persists across calls, so it STOPS WHERE IT IS, the CTA
//     forms more words, and it CONTINUES.  The super-chunk is not closed early
//     and THE PARTITION IS THEREFORE UNCHANGED.
//
// The second is a WAIT, not a branch dodging an edge case: the condition is
// broadcast through shared memory so it is CTA-uniform, every warp takes the same
// path, and the loop's body simply does not run in the ordinary case.  Closing a
// super-chunk early instead would trade one iteration of latency for a
// PARTIALLY-FULL super-chunk -- a whole extra `C x BC x d_v` of mma -- and it
// would do so precisely in the ultra-sparse regime compaction exists for.
//
// The former is a parallel map: `kThreads` threads produce `kThreads` words, four
// thousand tokens, in one step.  Dense consumption is `C/32` words per
// super-chunk.  The producer outruns the consumer by three orders of magnitude,
// so the wait's frequency is COUNTED rather than assumed.
//
// NO NEW BARRIER IN THE STEADY STATE.  The form is issued where `stage_chunk()`
// is, after the rendezvous that publishes the walk's cursor and before the one
// that already makes the staged tiles visible -- so the formed words ride a
// barrier that was already there.  The per-super-chunk barrier count is TWO, as
// it was.  The PROLOGUE pays two extra one-time barriers to fill the ring before
// the first walk, and each wait round pays two more.
//
// -------------------------------------------------------------------------
// WHAT RUNG 5B IS, AND ALL OF IT STILL HOLDS BELOW
// -------------------------------------------------------------------------
//
// Base: rung 5A (`rung5/csrc/precomputed_schedule_la.cu` at `2c0130e`).  EXACTLY
// ONE THING CHANGES: which TOKENS a visit carries.  Rung 5A visited a contiguous
// GROUP of `C` tokens whenever any one of them was live on either side, and both
// panels were that group.  This rung walks the tokens in order and gathers the
// ones that are live ON EACH SIDE SEPARATELY into two lists, closing a
// SUPER-CHUNK when either list fills.  Every mma term's token axis is one of the
// two lists, so the visit count falls from live groups to super-chunks and every
// term falls with it.
//
// WHY THIS IS THE RUNG THAT MATTERS.  Routing sparsity reaches the hardware only
// through the token axis: a block that is live for two tokens of sixty-four
// still executes sixty-four tokens' worth of arithmetic without compaction, so
// the quantisation eats the whole sparsity win.  Compaction is not an
// optimisation on a working sparse kernel; it is the mechanism that converts
// routing sparsity into executed savings at all.
//
// IT CONSUMES RUNG 5A's TABLE, WHICH IS WHY THERE IS NOTHING TO DRIFT
// -------------------------------------------------------------------
// The union table is per SIDE and per TOKEN, one bit.  Rung 5A's group schedule
// was a REDUCTION over those bits; this rung reads the same bits per token.  One
// structure, two consumers, and the second one is the reason the first was built
// this way.  The group schedule itself is GONE: compaction subsumes it, because a
// stretch of tokens that is dead on both sides is absorbed into a super-chunk
// that gathers nothing rather than skipped as a group.
//
// THE COOK -- WHAT IT PRODUCES, AND WHAT IT COSTS
// -----------------------------------------------
// One lockstep unit walks the token axis a word at a time, `STAGES - 1`
// super-chunks ahead of consumption, and publishes into a ring:
//
//   * `r[0..nr)` the read-live tokens of this super-chunk, ASCENDING;
//   * `w[0..nw)` the write-live tokens, ASCENDING;
//   * `J[i]`     for read entry `i`, the number of write entries at positions
//                <= `r[i]`.  ONE BYTE, CHUNK-RELATIVE.
//
// `J` IS NOT COMPUTED, IT IS OBSERVED.  The walk already keeps a running write
// count; the value that count holds when a read token is recorded IS that
// token's causal threshold.  The causal relation therefore comes free with the
// gather, and it comes out SORTED, which is what makes the mask a MONOTONE
// STAIRCASE rather than a general predicate.
//
// THE STAIRCASE IS ONLY ON THE DIAGONAL TILE.  Both lists are ascending, so a
// pairing of a read tile with a write tile that ends before it begins is a fully
// valid rectangular product with no mask at all; only the tile a super-chunk
// shares with itself needs one.  At equal buffers the super-chunk IS the
// diagonal tile, so that is the only place the mask appears here.
//
// WHERE A SUPER-CHUNK CLOSES.  At the first token that would push EITHER list
// past its buffer -- the range is set by whichever side saturates first.  A
// stretch that is dead on both sides pushes neither, so it is swallowed whole.
//
// THE BUFFERS ARE HELD EQUAL AT `C`, AND THAT IS A STATED LIMIT.  A read entry
// is panel rows plus one `J` byte and a write entry carries a whole `d_v` value
// row, so read slots are two to three times cheaper and an ASYMMETRIC pair (a
// longer read buffer) would close strictly fewer super-chunks.  Equal buffers are
// the CONSERVATIVE reading and they are what makes the dense degeneration exact;
// the asymmetric form is NOT BUILT.
//
// THE RELEASE RULE IS THE FOLD-SAFETY CONDITION.  A read entry is finished once
// the write frontier has passed its token, which is exactly the condition under
// which that write tile may fold.  Closing a super-chunk at the first saturation
// and folding it whole satisfies both with one rule: every read token of a
// super-chunk sees the state of every write token before it (folded by earlier
// super-chunks) plus, through the staircase, every write token inside it that is
// not after it.
//
// AT DENSE THIS IS THE IDENTITY, AND THAT IS THE STOP GATE
// ---------------------------------------------------------
// With every token live on both sides the cook closes at exactly `C` tokens, both
// lists are the identity, `J[i] = i + 1` and the staircase reduces to `tc <= tr`
// -- rung 5A's triangle.  So the dense arm must be BIT-IDENTICAL to rung 5A's,
// and it is asserted rather than observed (`check.py` T5).  What remains at dense
// is the cook, the gathered addressing and the byte load behind the mask; pricing
// that residual is the whole purpose of the dense measurement.
//
// A GATHER IS AN INDIRECTION AND MUST NOT BECOME 64-BIT ARITHMETIC.  Every
// gathered address is `gather_ptr(base, off)`: a base formed once in the entry
// and a 32-bit BYTE displacement, zero-extended.  Writing `base + index` at
// element type lets the compiler re-derive a widening multiply per access, which
// the addressing gate forbids; the bound on the displacement is the caller's, and
// here it is `L <= 65536`, which the host refuses past rather than truncating.
//
// WHAT DOES NOT CHANGE.  The state stays register-resident and the fold's
// accumulator is still the readout's A operand; production is still digit-major
// with the loop nest as the digit decomposition, over a block that is still a
// rectangle in digit space; the ring still stages only V; the readout is still
// self-normalising and the ratio is still a separate pass.  What compaction
// changes is WHICH token row each of those touches, and nothing else.
//
// WHAT RUNG 4 CONTRIBUTED, AND IS UNCHANGED HERE: the self-normalising readout.
//
//     y[t] = num[t] / (den[t] + eps),   num[t] = sum_s R[t,s] S[t,s,0:d_v]
//                                       den[t] = sum_s R[t,s] S[t,s,d_v]
//
// `rola/ops/naive.py` is the spec and its only normalisation is `"global"`, so
// this is not a mode: it is what the readout IS.  Spec 4.2 fixes `g_read := 1`
// there, and that is why no read gain appears anywhere below -- under a ratio a
// per-token read scale cancels exactly.
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
// neither.  So the mass tile is carried by the four units that exist, carved
// along the axis a single channel leaves free: the SLOTS.  Unit `u` owns the mass
// of the tiles `q * kMassUnits + u`, it holds a COMPLETE mass state for those
// slots and produces a PARTIAL denominator over them.  The intra term's free axis
// is tokens rather than slots, and it is carved the same way.
//
// THE CARVE IS IN THE ADDRESS, NOT IN A PREDICATE, AND THAT WAS MEASURED RATHER
// THAN ASSUMED.  Selecting each unit's tiles with a per-mma test costs one copy
// of the body per unit AND the full mma -- a predicated-off mma still occupies
// its issue slot -- measured at four times the added tile.  So the unit's B
// operands are loaded at its own base, which is loop-invariant, and every index
// inside the loop is a compile-time constant again.
//
// WHY THE DIVISION IS NOT IN THIS KERNEL.  `den[t]` is a sum over the N/BC blocks
// and no block knows the total, so the ratio cannot be formed before the last
// contribution has landed -- a grid-wide dependency, which is the synchronisation
// the fan-in declined in rung 2.  The kernel produces the readout's `d_v + 1`
// columns and `divide_pass` forms the ratio in one elementwise pass.
//
// THE NUMERATOR AND THE DENOMINATOR ARE SEPARATE BUFFERS, and that is an
// alignment decision rather than a semantic one: a single `[.., d_v+1]` buffer
// gives every numerator row a 260-byte stride, which misaligns all 32 lanes of
// every fan-in reduction for the sake of packing one column.
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
// block, and one that holds for every simplex-valued routing.
//
// That last clause is the ONE place this test is CONSERVATIVE rather than exact,
// and the direction is the safe one: a level that really was all zero would make
// the test say LIVE for a token with no work, which costs a gathered slot and
// changes no answer.  The test can never say DEAD for a token that has work.
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
// That is the necessary and sufficient condition for the block's slots to form a
// CONTIGUOUS RANGE.  A rectangle is what lets the coefficient loop nest BE the
// digit decomposition, so `digit_l(n)` is never computed here either.  A
// (widths, BC) pair that does not satisfy it FAILS THE BUILD at `BlockPlan`'s
// clauses.
//
// THE FAN-IN IS A PLAIN REDUCTION, AND IT IS FIRE-AND-FORGET
// ----------------------------------------------------------
// The N/BC contributors to an output element are added with `fan_in_add`, whose
// contract is that the issuing thread does not consume a result -- which is what
// lets the instruction be a REDUCTION rather than an atomic read-modify-write.
// Under compaction its target row is GATHERED: the partial a super-chunk
// produced for its `i`-th read entry belongs to token `r[i]`.  Each token is a
// read entry of exactly one super-chunk, so the contributor count per output
// element is unchanged at N/BC.
//
// THE OUTPUT IS fp32, AND THE FAN-IN IS WHY.  With N/BC contributors a bf16
// accumulator would round every partial, and the partials of a routed readout
// carry mixed signs, so the fan-in error would land near 2^-9 of the result --
// above this design's own bf16 quantisation floor rather than inside it.
//
// WHAT CAPACITY COSTS, STRUCTURALLY, BEYOND THE FAN-IN
// -----------------------------------------------------
// The intra term `tril(R_c . W_c^T) . V_c` multiplies a Gram by V.  The GRAM is a
// slot sum and splits, but the V multiply does not: every block does its own, so
// it is performed N/BC times where one would do if the Gram were summed first.
// Summing it first is a cross-CTA reduction per chunk, which is the grid-wide
// synchronisation the fan-in already declined.  Combining the Gram is also
// EXCLUSIVE with compaction, because a combined Gram lives in global token
// space where it is dense over the whole span.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <type_traits>
#include <vector>

#include "design.cuh"
#include "ops.cuh"

namespace rung6e {

using namespace denseref::ops;
namespace design = denseref::design;

static_assert(design::kStateResidency == design::StateResidency::REGISTERS &&
                  kMmaAcceptsRegisterA,
              "SELECTION UNSATISFIABLE GIVEN AVAILABLE OPERATIONS: this kernel's "
              "fold accumulator IS the readout's A operand, so it needs a state "
              "that lives in registers and a built `mma` that reads A out of "
              "them; otherwise the state takes a memory round trip and this is a "
              "different algorithm.");

static constexpr int kBuildMagic = 6500001;

static constexpr int kD = 64;      // d_v
static constexpr int kPadE = 8;    // bf16 row padding, elements (16 B)
static constexpr int kPadF = 4;    // fp32 row padding, elements (16 B)

//: THE K-WINDOW (P63, standing-rule compliance: the kernel is designed for
//: L >= 1M).  Token indices are WINDOW-RELATIVE: a super-chunk lives inside
//: one 65536-token window (the walk forces a cut at window lines), its list
//: entries are 16-bit locals, and every consumer rebases its gather plane by
//: the super-chunk's window -- an AFFINE advance, never a per-chunk multiply.
//: Below one window no line is ever crossed, so the partition -- and the
//: output, bit for bit -- is exactly what it was when 65536 was a refusal.
static constexpr int kWindow = 65536;
static_assert((kWindow & (kWindow - 1)) == 0,
              "the token -> (window, local) split is a shift and a mask");

//: HOW MUCH FURTHER AHEAD THE SCHEDULE STREAM RUNS THAN THE VALUE STREAM, in
//: super-chunks, ABOVE the minimum the pipeline requires.  It costs one index
//: slot of shared memory per super-chunk of distance and NO registers, because
//: the cook's carried state is a cursor and four induction pointers whatever the
//: distance is.
//:
//: IT IS ZERO BECAUSE THE MEASUREMENT SAYS DISTANCE IS NOT WHAT THE COOK COSTS.
//: Swept at 0 and 2 on the flagship arms, the two agree to within two points and
//: the sign flips across arms.  What the cook actually costs is the latency of
//: its OWN dependent loads, which sits inside one call and immediately before the
//: rendezvous that follows it -- and no amount of distance to its CONSUMER moves
//: that.  So the depth is the pipeline's minimum, derived rather than swept: the
//: cook must be exactly as far ahead as the stage it feeds.
static constexpr int kAheadExtra = 0;

// THE CARVE, and it is DERIVED rather than chosen.  One MMA unit owns a private
// 16-row slice of the transposed state's m axis, which is d_v, so the unit count
// is d_v / the MMA's m quantum.  Carving the SLOT axis across units instead
// would give every unit a partial output over the whole of d_v and therefore a
// cross-unit reduction of the output accumulator -- the very thing the private
// accumulator exists to avoid.
static constexpr int kUnits = kD / kMmaM;
//: ONE GROUP'S carve: the value axis to its cap.  A group is exactly the
//: rung-5D CTA -- four units of sixteen d_v rows, each holding its slice of
//: the state for the GROUP'S slots.
static constexpr int kGroupThreads = kUnits * kMmaUnitThreads;

//: THE PER-CONFIG CARVE RULE (P64): warps = the stall floor inside the
//: register window, filled from the CHEAP axis first -- value units to their
//: cap (`d_v/16`; token-split is forbidden for state-holding warps, it would
//: duplicate the register-resident state), then SLOT GROUPS.  At d_v=64 the
//: value axis gives four warps; a second slot group is added exactly when the
//: block is wide enough that one group's k-run would exceed the measured
//: KD = 8 shape -- i.e. BC >= 256 -- which is also where the register window
//: narrows to the 8-warp point.
constexpr int groups_for(int BC) { return BC >= 256 ? 2 : 1; }
constexpr int threads_for(int BC) { return kGroupThreads * groups_for(BC); }

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
  //: tabulate.  The count is a SUM over levels and never a product, which is why
  //: the table does not grow with `N`.
  static constexpr int lead_entries = lead_span < W0 ? W0 / lead_span : 0;
  static constexpr int tail_entries = tail_span < W1 ? W1 / tail_span : 0;
  static constexpr int entries = lead_entries + tail_entries;
};

//: THE UNION TABLE'S ONE STRUCTURAL CONSTANT.  Tokens are the table's minor axis
//: and they are packed thirty-two to a word, which is the lockstep unit's width,
//: so the cook reads one word per unit per side and needs no re-packing step.  A
//: sequence that does not fill its last word leaves the tail bits ZERO.
constexpr int kTokBits = 32;
__host__ __device__ constexpr int tok_words(int L) { return (L + kTokBits - 1) / kTokBits; }

//: THE FORMED RING'S DEPTH, IN WORDS PER SIDE.  It is a RESIDENCY quantity and
//: not an axis: what it buys is distance between the parallel former and the
//: serial walk, and the walk consumes `C/32` words per super-chunk against a
//: former that produces `kThreads` of them in one step, so any depth at all puts
//: the producer orders of magnitude ahead.  What it must NOT do is cost a CTA,
//: which is what pins it: `2 * kFormWords * 4` bytes must leave every arm's
//: staging depth and CTA target where rung 5B had them, and `arms()` reports both
//: so that is CHECKED rather than asserted.
//:
//: IT IS A POWER OF TWO because the ring index is `word & (kFormWords - 1)`, and
//: a remainder by anything else would put a reciprocal in the walk's only loop.
//:
//: THE SHIPPED DEPTH IS `kThreads`, so ONE parallel step refills the ring whole.
//: It is overridable AT BUILD TIME because the wait is a ring's ordinary
//: behaviour and a gate must be able to make it fire: at depth one the walk waits
//: on almost every word, and the answer must still be bit for bit rung 5B's.
#ifndef RUNG6_FORM_WORDS
#define RUNG6_FORM_WORDS 128
#endif
constexpr int kFormWords = RUNG6_FORM_WORDS;
static_assert((kFormWords & (kFormWords - 1)) == 0,
              "the formed ring's index is a mask, so its depth is a power of two");

//: WHAT THE WALK PUBLISHES, AND IT IS ONLY WHAT ANOTHER WARP NEEDS: the cursor,
//: so the former knows how far it may run, and the "I ran out" flag, so the whole
//: CTA takes the same path.  The running counts stay in the walk's registers.
//: The slot count is four rather than two so the buffer behind it stays
//: sixteen-byte aligned.
constexpr int kCookWord = 0, kCookOpen = 1;
constexpr int kCookSlots = 4;

// ------------------------------------------------------------------ geometry

//: THIS KERNEL'S LIVE REGISTER SET, per (chunk, block width).  It is a
//: DECLARATION rather than a derivation, and deliberately so: ptxas's allocation
//: for this body does not follow from the operand counts, and the number's only
//: job is to stop the shared-memory ceiling from asking for more CTAs than the
//: register file holds.
//:
//: It is FALSIFIABLE, not asserted: `build.py` fails the build on any nonzero
//: `localSizeBytes`, so a floor set too low cannot ship quietly.  The values are
//: the measured census of THIS body, which carries the cook's cursor and the
//: staircase's thresholds on top of rung 5A's set.
constexpr int reg_floor(int C, int BC) {
  //: a two-group arm's PER-THREAD demand is one group's, i.e. the BC/2 row --
  //: each thread holds its group's slice, which is the same slice a BC/2
  //: single-group thread holds.
  return BC >= 256 ? reg_floor(C, BC / 2)
       : C == 16 ? (BC <= 64 ? 128 : 168)
       : C == 32 ? (BC <= 32 ? 136 : BC <= 64 ? 168 : 236)
                 : (BC <= 32 ? 176 : BC <= 64 ? 232 : 255);
}

template <int C, bool SWZ, int BC> struct Geom {
  //: the GROUP is the unit of panel/state/output geometry: at GROUPS = 1 the
  //: slice is the block and everything below is rung 5D's byte for byte.
  static constexpr int GRP = groups_for(BC);
  static constexpr int BCg = BC / GRP;
  static constexpr int LdV = SWZ ? kD : kD + kPadE;   // V ring row, bf16 elements
  static constexpr int LdS = SWZ ? BCg : BCg + kPadE; // R/W row (one group's slice)
  static constexpr int LdG = C + kPadE;               // Gram row, bf16 elements
  //: the fp32 Gram partials, one per group, that the exchange sums -- they
  //: exist only when there is more than one group to exchange between.
  static constexpr int LdGf = C + kPadF;              // fp32 Gram row
  //: The output staging buffer is fp32 and ALWAYS PADDED.  The swizzle exists to
  //: serve `ldmatrix` on the operand buffers; nothing reads this one as an
  //: operand, and its two access patterns are conflict-free under the pad.
  static constexpr int LdO = kD + kPadF;              // output row, fp32 elements

  //: ONE SUPER-CHUNK'S INDEX LISTS: two 16-bit token indices and one causal
  //: threshold byte per slot, behind a header the whole CTA reads.  It rides the
  //: RING rather than the fixed buffers because a super-chunk's lists are needed
  //: from the moment it is staged to the moment it is drained, which is exactly
  //: the ring's own span -- one slot deeper, for the drain that trails by one.
  static constexpr int hdr = 16;
  static constexpr int idx_slot = hdr + ((5 * C + 15) / 16) * 16;
  //: The index ring rides the VALUE ring's accounting one slot per stage plus a
  //: fixed remainder, because its depth is the value depth plus a constant.
  static constexpr int slot = C * LdV * (int)sizeof(__nv_bfloat16) + idx_slot;
  //: The formed schedule and the walk's published state are RESIDENT rather than
  //: ringed with the stages: what they hold is a window on the token axis, whose
  //: size is a property of the walk and not of the staging depth.
  static constexpr int form_bytes = 2 * kFormWords * (int)sizeof(uint32_t) +
                                    kCookSlots * (int)sizeof(int);
  static constexpr int gram_xchg =
      GRP > 1 ? GRP * C * LdGf * (int)sizeof(float) : 0;
  static constexpr int fixed =
      GRP * (2 * C * LdS) * (int)sizeof(__nv_bfloat16) +
      C * LdG * (int)sizeof(__nv_bfloat16) + gram_xchg +
      GRP * (C * LdO + kUnits * C) * (int)sizeof(float) +
      (1 + kAheadExtra) * idx_slot + form_bytes;
  static constexpr int stages =
      design::stages_for(slot, fixed, threads_for(BC), reg_floor(C, BC));
  static constexpr size_t bytes = (size_t)stages * slot + fixed;
};

template <int C, bool SWZ, int BC>
constexpr int kCtaTarget =
    design::cta_target((int)Geom<C, SWZ, BC>::bytes, threads_for(BC),
                       reg_floor(C, BC));

// -------------------------------------------------------------------- kernel

template <int C, bool SWZ, int W0, int W1, int BC>
__global__ __launch_bounds__(threads_for(BC), (kCtaTarget<C, SWZ, BC>)) void
compacted_kernel(const __nv_bfloat16 *__restrict__ pread,
                 const __nv_bfloat16 *__restrict__ pwrite,
                 const __nv_bfloat16 *__restrict__ v,
                 const uint32_t *__restrict__ tbl,
                 float *__restrict__ o, float *__restrict__ den, int L) {
  using P = BlockPlan<W0, W1, BC>;
  using G = Geom<C, SWZ, BC>;
  //: the per-config carve, evaluated: how many slot groups, how many threads.
  constexpr int GROUPS = groups_for(BC);
  constexpr int kThreads = threads_for(BC);
  constexpr int kWarps = kThreads / 32;
  constexpr int kGroupWarps = kGroupThreads / 32;
  //: ONE GROUP'S SLOT SLICE.  Every per-warp compute constant below is the
  //: slice's, so at GROUPS = 1 the whole body is rung 5D's byte for byte.
  constexpr int BCg = BC / GROUPS;
  static_assert(BC % GROUPS == 0, "the groups must halve the block exactly");
  static_assert((C / kMmaM) % GROUPS == 0,
                "the intra term splits by token m-tiles across the groups, so "
                "the groups must divide them");
  static_assert(C % kMmaM == 0, "chunk must be a multiple of the mma m quantum");
  static_assert(BCg % kMmaK == 0, "a group's slice must be a whole number of mma k-tiles");
  static_assert(C <= 255, "a causal threshold is one byte, chunk-relative");
  static_assert(design::smem_fits(G::bytes),
                "this arm does not fit this architecture's per-CTA shared-memory "
                "maximum");
  constexpr int STAGES = G::stages;
  static_assert(STAGES >= 2, "the wait/issue order needs at least two stages");
  //: THE SCHEDULE STREAM AND THE VALUE STREAM RUN AT DIFFERENT DEPTHS, and that
  //: is the whole reason the index ring is separate from the value ring.
  //:
  //: THE CHAIN COMPACTION CREATES.  Without it every address is affine and the
  //: only global round trip in flight is the value stage.  With it the chain is
  //: schedule word -> cook -> index -> gather issue -> gather lands -> mma, which
  //: is TWO round trips in series, and in a latency-bound arm that is the wrong
  //: shape.  Running the cook further ahead than the value ring collapses it back
  //: to one: by the time a super-chunk is staged its indices have been ready for
  //: `kAheadExtra` iterations, so only the gather is outstanding.
  //:
  //: WHY THE SCHEDULE IS NOT STAGED ASYNCHRONOUSLY.  A super-chunk's schedule is
  //: one word per side per thirty-two tokens -- eight bytes at `C = 64` -- against
  //: a value tile of `C x d_v`.  There is nothing to stream: what the cook needs
  //: is DISTANCE, not bandwidth, and distance is free here because the cook's
  //: carried state is a cursor and four induction pointers.  It also keeps the
  //: staging FIFO single-stream, so `stage_wait` still means what it means: one
  //: implicit group, retired in issue order, with nothing behind it that a later
  //: consumer needs first.
  //:
  //: THE DEPTH IS DERIVED, NOT SWEPT: the cook must be at least as far ahead as
  //: the stage it feeds, plus enough distance to cover its own dependent loads.
  constexpr int AHEAD = STAGES - 1 + kAheadExtra;
  //: THE INDEX RING HOLDS EVERY SUPER-CHUNK IN FLIGHT: the one being drained, the
  //: one computing, and the `AHEAD` cooked but not yet consumed.  That is also
  //: what fences it: the slot the cook overwrites belongs to the super-chunk
  //: drained a whole iteration earlier, across the rendezvous that follows the
  //: fold -- which is why that rendezvous is issued on every super-chunk rather
  //: than only on live ones.
  constexpr int IRING = AHEAD + 2;
  constexpr int LdV = G::LdV, LdS = G::LdS, LdG = G::LdG, LdO = G::LdO;
  constexpr int MTOK = C / kMmaM;   // token m-tiles in a chunk
  constexpr int NTOK = C / kMmaN;   // token n-tiles in a chunk
  constexpr int KD = BCg / kMmaK;   // slot k-tiles of THIS GROUP'S slice
  constexpr int NDV = BCg / kMmaN;  // state n-tiles (slots) of the slice
  constexpr uint32_t kRowV = (uint32_t)LdV * 2;
  constexpr uint32_t kRowS = (uint32_t)LdS * 2;
  constexpr int kChunkS = BCg / 8;  // 16-byte chunks in one R/W row
  constexpr int kChunkV = kD / 8;   // 16-byte chunks in one V row

  //: THE MASS CHANNEL'S CARVE.  One channel cannot be split along the channel
  //: axis, so it is split along the one the mma leaves free.  The readout's and
  //: the fold's free axis is the SLOTS; the intra term's is the TOKENS.
  constexpr int kMassUnits = KD < kUnits ? KD : kUnits;
  constexpr int kMassKD = KD / kMassUnits;      // slot k-tiles of the mass, per unit
  constexpr int kMassTiles = 2 * kMassKD;       // its state's n-tiles, per unit
  constexpr int kIntraUnits = MTOK < kUnits ? MTOK : kUnits;
  constexpr int kIntraKD = MTOK / kIntraUnits;
  static_assert(KD % kMassUnits == 0 && NDV == kMassUnits * kMassTiles,
                "the mass channel's slots must be carved across the units exactly");

  constexpr int MSPLIT = MTOK >= kGroupWarps ? kGroupWarps : MTOK;
  constexpr int NSPLIT_RAW = kGroupWarps / MSPLIT;
  constexpr int NSPLIT = NSPLIT_RAW < NTOK / 2 ? NSPLIT_RAW : NTOK / 2;
  constexpr int MPW = MTOK / MSPLIT;
  constexpr int NPW = NTOK / NSPLIT;
  constexpr int GWARPS = MSPLIT * NSPLIT;
  static_assert(NPW % 2 == 0, "n-tiles are consumed in ldmatrix pairs");
  constexpr int GPAIR = (NPW / 2) >= 2 ? 2 : 1;

  // ---- coefficient production geometry.  All compile-time. ----------------
  // A row's `kRowThr` threads split the block's BC coefficients.  There is one
  // rectangle to split and two ways to cut it, and which one applies is decided
  // by the rectangle's own shape.
  constexpr int kWtot = W0 + W1;
  //: PRODUCTION IS A GROUP AFFAIR: a group's kGroupThreads produce ITS slice's
  //: coefficients -- `kLeadSpanG` leading digits by the whole trailing span --
  //: which at GROUPS = 1 is the block and exactly rung 5D's mapping.
  constexpr int kLeadSpanG = P::lead_span / GROUPS;
  static_assert(P::lead_span % GROUPS == 0,
                "the groups must split the leading run exactly");
  constexpr int kRowThr = kGroupThreads / C;
  constexpr int kThrPerLead = kRowThr >= kLeadSpanG ? kRowThr / kLeadSpanG : 1;
  constexpr int kLeadPerThr = kLeadSpanG >= kRowThr ? kLeadSpanG / kRowThr : 1;
  constexpr int kTailPerThr = P::tail_span / kThrPerLead;
  constexpr int kTailVec = (kTailPerThr * 2) / 16;
  static_assert(kGroupThreads % C == 0,
                "one production pass must cover the chunk exactly");
  static_assert(kRowThr >= kLeadSpanG ? kRowThr % kLeadSpanG == 0
                                      : kLeadSpanG % kRowThr == 0,
                "the slice's leading digits and the row's threads must divide "
                "one another, or a thread owns a fractional digit");
  static_assert(P::tail_span % kThrPerLead == 0,
                "the threads sharing a leading digit must divide its trailing run");
  static_assert(kTailPerThr % 8 == 0,
                "a thread's trailing run is published in 16-byte vectors, so it "
                "must be a whole number of eight coefficients");
  static_assert(kLeadPerThr * kTailPerThr * kRowThr == BCg,
                "the row's threads must cover the slice exactly once");
  static_assert(kLeadPerThr <= 4, "leading-digit slice is 1, 2 or 4 amplitudes");

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  //: this thread's GROUP, its warp within the group, and its thread id within
  //: the group -- the coordinates every per-group buffer and role uses.  At
  //: GROUPS = 1 they collapse to (0, warp, tid).
  const int grp = warp / kGroupWarps;
  const int gwarp = warp % kGroupWarps;
  const int gtid = tid % kGroupThreads;

  extern __shared__ __align__(16) char smem_raw[];
  //: [ V ring (shared) | panels x GROUPS | summed Gram (shared) | fp32 Gram
  //:   partials x GROUPS (only when there is an exchange) | output staging x
  //:   GROUPS | denominators x GROUPS | index ring | formed schedule ] -- the
  //: order is Geom::fixed's, and the per-group regions are indexed by `grp`.
  extern __shared__ __align__(16) char smem_base[];
  __nv_bfloat16 *sv = reinterpret_cast<__nv_bfloat16 *>(smem_raw);
  __nv_bfloat16 *srAll = sv + STAGES * C * LdV;
  __nv_bfloat16 *sr = srAll + grp * (2 * C * LdS);
  __nv_bfloat16 *sw = sr + C * LdS;
  __nv_bfloat16 *sg = srAll + GROUPS * (2 * C * LdS);
  float *sgf = reinterpret_cast<float *>(sg + C * LdG);
  float *soAll = sgf + (GROUPS > 1 ? GROUPS * C * G::LdGf : 0);
  float *so = soAll + grp * (C * LdO + kUnits * C);
  float *sden = so + C * LdO;
  char *sidx = reinterpret_cast<char *>(soAll + GROUPS * (C * LdO + kUnits * C));
  //: THE FORMED SCHEDULE, one word per thirty-two tokens per side, and the walk's
  //: published state behind it.
  uint32_t *sfR = reinterpret_cast<uint32_t *>(sidx + IRING * G::idx_slot);
  uint32_t *sfW = sfR + kFormWords;
  int *scook = reinterpret_cast<int *>(sfW + kFormWords);
  static_assert((GROUPS * 2 * C * LdS + C * LdG) % 8 == 0,
                "the fp32 region must start 16-byte aligned");
  static_assert((C * LdO + kUnits * C) % 4 == 0,
                "the index ring must start 16-byte aligned");
  static_assert(G::idx_slot % 16 == 0,
                "the formed schedule follows the index ring and must stay aligned");

  const int blk = blockIdx.x;   // which block of BC slots this CTA owns
  const long base = (long)blockIdx.y * L * kD;

  //: THE INDEX RING'S ACCESSORS.  A slot is a header the whole CTA reads
  //: (`nr`, `nw`, whether the super-chunk exists at all) followed by the two
  //: lists and the thresholds.
  auto ihdr = [&](int d) { return reinterpret_cast<int *>(sidx + d * G::idx_slot); };
  auto iread = [&](int d) {
    return reinterpret_cast<uint16_t *>(sidx + d * G::idx_slot + G::hdr);
  };
  auto iwrite = [&](int d) { return iread(d) + C; };
  auto ithr = [&](int d) {
    return reinterpret_cast<uint8_t *>(sidx + d * G::idx_slot + G::hdr + 4 * C);
  };

  // The block's aligned digit ranges.  Compile-time divisors, formed once.
  const int lead_off = (blk / P::tail_groups) * P::lead_span + grp * kLeadSpanG;
  const int tail_off = (blk % P::tail_groups) * P::tail_span;

  // ---- lane-invariant operand addressing -----------------------------------
  const int lr = lane & 7, tl = lane >> 3;
  const int rF = (tl & 1) * 8 + lr, cF = tl >> 1;
  const int rT = (tl >> 1) * 8 + lr, cT = tl & 1;
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
  //: address at a compile-time `q` plus a LOOP-INVARIANT per-unit displacement.
  const uint32_t massOff = (uint32_t)(gwarp * kMmaK * (int)sizeof(__nv_bfloat16));
  uint32_t tabMT[kMassKD], tabMF[kMassKD];
#pragma unroll
  for (int q = 0; q < kMassKD; ++q) {
    const int g = 2 * (q * kMassUnits + gwarp);
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

  const int gm0 = (gwarp / NSPLIT) * MPW;
  const int gn0 = (gwarp % NSPLIT) * NPW;
  const int dv0 = gwarp * 16;
  constexpr int DVG_STEP = 2;
  const int dvg = gwarp * DVG_STEP;
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
  //: whose row 0 is all ones and whose padding rows are zero.
  constexpr uint32_t kOnesPair = 0x3F803F80u;   // two bf16 1.0
  uint32_t ones_a[4];
  ones_a[0] = (lane >> 2) == 0 ? kOnesPair : 0u;
  ones_a[1] = 0u;
  ones_a[2] = ones_a[0];
  ones_a[3] = 0u;

  // ---- global -> shared staging of V, GATHERED BY THE WRITE LIST -----------
  //: V IS A WRITE-SIDE QUANTITY.  Both terms that consume it -- the fold and the
  //: intra -- contract over the WRITE token axis, so one gather serves both and
  //: the read list never touches V.
  constexpr int kRowsPerPass = kThreads / kChunkV;
  constexpr int kPasses = C / kRowsPerPass;
  static_assert(kPasses >= 1, "a chunk must be at least kThreads/8 rows");
  const int ldRow = tid / kChunkV, ldCol = (tid % kChunkV) * 8;
  const __nv_bfloat16 *const vPlane = pin_address(v + base);
  const uint32_t ldColB = (uint32_t)(ldCol * 2);
  constexpr uint32_t kVRowB = (uint32_t)(kD * (int)sizeof(__nv_bfloat16));

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
  // CONSECUTIVE CHANNELS SIT IN CONSECUTIVE LANES OF ONE REDUCTION.  That is a
  // property of the REDUCTION and not of the thread: this architecture's
  // reduction carries ONE channel, so what decides how much output one of them
  // covers is the span its 32 lanes reach.  Giving a THREAD the consecutive run
  // instead leaves a reduction's lanes strided by that run, and the coalescer
  // then touches four times the sectors for the same bytes.
  constexpr int kFanLanes = 32;
  constexpr int kFanPerThr = kD / kFanLanes;
  constexpr int kFanRows = kGroupThreads / kFanLanes;
  constexpr int kFanPasses = C / kFanRows;
  static_assert(kD % kFanLanes == 0 && kGroupThreads % kFanLanes == 0,
                "a reduction's lanes must tile both an output row and the CTA");
  static_assert(kFanPasses * kFanRows == C && kFanPasses >= 1,
                "the fan-in's passes must cover the chunk exactly");
  const int fanRow = gtid / kFanLanes, fanCol = gtid % kFanLanes;
  float *const oPlane = pin_address(o + base);
  float *const denPlane = pin_address(den + (long)blockIdx.y * L);
  const uint32_t fanColB = (uint32_t)(fanCol * (int)sizeof(float));
  constexpr uint32_t kORowB = (uint32_t)(kD * (int)sizeof(float));
  const int fanE = fanRow * LdO + fanCol;

  // Output staging write index: this lane owns d_v = dv0 + (lane>>2) and that
  // + 8, at tokens 8j + 2*(lane&3) + {0,1}.
  const int oq = (lane & 3) * 2, om = lane >> 2;
  int oE[2][2];
#pragma unroll
  for (int e = 0; e < 2; ++e)
#pragma unroll
    for (int f = 0; f < 2; ++f) oE[e][f] = (oq + e) * LdO + dv0 + om + f * 8;

  //: THE STAGE IS WHAT IS SKIPPED; THE RING SLOT IS NOT.  A super-chunk with no
  //: write entry stages nothing, but the slot advances either way so that the
  //: wait/commit accounting stays rung 2's.
  //: WINDOW CURSORS.  Every consumer of a LOCAL index rebases its plane to
  //: the super-chunk's window first.  Windows are monotone along the ring, so
  //: each cursor ADVANCES by a compile-time stride -- an affine walk of at
  //: most L/K steps per launch, never a per-chunk multiply, and below one
  //: window none of these loops ever runs.
  const __nv_bfloat16 *vStageP = vPlane;
  int vStageWin = 0;
  auto stage_chunk = [&](int d, bool go) {
    if (go) {
      const uint16_t *lw = iwrite(d);
      const int wn = ihdr(d)[3];
      while (vStageWin < wn) {
        vStageP += (long)kWindow * kD;
        ++vStageWin;
      }
#pragma unroll
      for (int p = 0; p < kPasses; ++p) {
        const uint32_t off = (uint32_t)lw[ldRow + p * kRowsPerPass] * kVRowB + ldColB;
        stage_tile(svL + stgSlot + (uint32_t)(p * kRowsPerPass * (int)kRowV),
                   gather_ptr(vStageP, off));
      }
    }
    stgSlot += kSlot;
    if (stgSlot == kRing) stgSlot = 0;
  };

  //: THE FAN-IN'S TARGET ROW IS GATHERED.  The partial staged for read entry `i`
  //: belongs to token `r[i]`; entries past `nr` are the buffer's padding and have
  //: no token, so they contribute nothing rather than contributing zero to
  //: somewhere.
  float *oDrainP = oPlane;
  float *denDrainP = denPlane;
  int oDrainWin = 0;
  auto drain = [&](int d, int nr) {
    const uint16_t *lrd = iread(d);
    const int wn = ihdr(d)[3];
    while (oDrainWin < wn) {
      oDrainP += (long)kWindow * kD;
      denDrainP += (long)kWindow;
      ++oDrainWin;
    }
#pragma unroll
    for (int p = 0; p < kFanPasses; ++p) {
      const int row = fanRow + p * kFanRows;
      if (row < nr) {
        float *dp = reinterpret_cast<float *>(
            gather_ptr(oDrainP, (uint32_t)lrd[row] * kORowB + fanColB));
        const int e0 = fanE + p * kFanRows * LdO;
#pragma unroll
        for (int e = 0; e < kFanPerThr; ++e)
          fan_in_add(dp + e * kFanLanes, so[e0 + e * kFanLanes]);
      }
    }
  };

  //: THE UNITS' PARTIAL DENOMINATORS MEET HERE, at no cost in barriers: the
  //: rendezvous that already separates a chunk's staging from the next chunk's
  //: drain is the one that publishes them.
  auto drain_den = [&](int d, int nr) {
    if (gtid < C && gtid < nr) {
      float s = 0.f;
#pragma unroll
      for (int u = 0; u < kUnits; ++u) s += sden[u * C + gtid];
      fan_in_add(reinterpret_cast<float *>(
                     gather_ptr(denDrainP, (uint32_t)iread(d)[gtid] * 4u)),
                 s);
    }
  };

  // ---- routed Kronecker coefficient production, digit-major ---------------
  // Thread `tid` owns panel row `tid / kRowThr` and, within the block's
  // rectangle, the leading digits [leadLocal, leadLocal + kLeadPerThr) and the
  // trailing run [tailLocal, tailLocal + kTailPerThr).  THE DIGITS OF A SLOT ARE
  // NEVER COMPUTED: they are the loop indices.  What compaction changes is only
  // WHICH TOKEN a panel row reads -- the row's own slot addressing is untouched,
  // which is why the gather costs a displacement and not a recomputation.
  const int prow = gtid / kRowThr;
  const int rIn = gtid % kRowThr;
  const int leadLocal = (rIn / kThrPerLead) * kLeadPerThr;
  const int tailLocal = (rIn % kThrPerLead) * kTailPerThr;
  //: ONE BASE PER SIDE, AND THE TWO LEVELS ARE 32-BIT COLUMN DISPLACEMENTS.  A
  //: base is a live 64-bit value for the whole kernel; the level a thread reads
  //: is a byte offset inside its token's row and therefore adds before the
  //: widening, not after it.
  const long pbase = (long)blockIdx.y * L * kWtot;
  constexpr uint32_t kPRowB = (uint32_t)(kWtot * (int)sizeof(__nv_bfloat16));
  const __nv_bfloat16 *const pRead = pin_address(pread + pbase);
  const __nv_bfloat16 *const pWrite = pin_address(pwrite + pbase);
  const __nv_bfloat16 *pReadP = pRead;
  const __nv_bfloat16 *pWriteP = pWrite;
  int pWin = 0;
  const uint32_t colLead = (uint32_t)((lead_off + leadLocal) * 2);
  const uint32_t colTail = (uint32_t)((W0 + tail_off + tailLocal) * 2);

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
  //: across every leading digit this thread owns, so the register cost does not
  //: grow with the block width.  Each coefficient is one `mul_bf16x2`.
  //:
  //: A PANEL ROW PAST THE GATHERED COUNT IS EXACTLY ZERO, and it is zeroed at the
  //: LEADING amplitude rather than at the product: one select per leading digit
  //: instead of one per coefficient, and the zero propagates through a
  //: multiplication whose other operand is a finite simplex amplitude.  A zero
  //: row contributes nothing to the Gram, nothing to the fold and nothing the
  //: drain will publish, which is what makes a partly filled buffer exact.
  auto produce = [&](const __nv_bfloat16 *bs, char *dst, uint32_t tokOff, bool have) {
    const __nv_bfloat16 *p0 =
        reinterpret_cast<const __nv_bfloat16 *>(gather_ptr(bs, tokOff + colLead));
    const __nv_bfloat16 *p1 =
        reinterpret_cast<const __nv_bfloat16 *>(gather_ptr(bs, tokOff + colTail));
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
            have ? splat_bf16x2(*reinterpret_cast<const __nv_bfloat16 *>(&amp0[j])) : 0u;
#pragma unroll
        for (int e = 0; e < 4; ++e) prod[e] = mul_bf16x2(a2, a1[e]);
        store_vec16(dst + pOff[j][i], *reinterpret_cast<const uint4 *>(prod));
      }
    }
  };

  // ---- the membership bits, read out of the union table --------------------
  //: THE BLOCK'S TWO ENTRY ROWS, PER SIDE.  Level 0's entry is the aligned run of
  //: digits this block spans there and level 1's is its sub-range; the trailing
  //: row DISAPPEARS when the block spans that level entirely, exactly as the
  //: clause does, and a block that spans BOTH levels entirely -- which is the
  //: whole slot space, so there is exactly one of them -- has no clause at all.
  //: RUNG 5A OR-ED THE SIDES to get a group bit; the cook keeps them APART,
  //: which is the whole of per-side compaction at the schedule's end.
  constexpr bool kTailWhole = P::tail_span == W1;
  constexpr bool kNoClause = P::entries == 0;
  constexpr int kEntries = P::entries;
  const int tokWords = tok_words(L);
  const long rowR = ((long)(2 * blockIdx.y + 0) * kEntries) * tokWords;
  const long rowW = ((long)(2 * blockIdx.y + 1) * kEntries) * tokWords;
  const int e_lead = kNoClause ? 0 : blk / P::tail_groups;
  const int e_tail = kTailWhole ? 0 : P::lead_entries + (blk % P::tail_groups);
  //: THE FORM'S FOUR TABLE ROWS ARE PINNED BASES AND A 32-BIT DISPLACEMENT.  The
  //: form is a MAP over words, not a walk, so a thread's word is an index rather
  //: than the previous address plus one; `gather_ptr` is what keeps that index a
  //: zero-extended byte displacement instead of a widening multiply.  The bound is
  //: `tokWords` words, four bytes each.
  const uint32_t *const tR0 = pin_address(tbl + rowR + (long)e_lead * tokWords);
  const uint32_t *const tW0 = pin_address(tbl + rowW + (long)e_lead * tokWords);
  const uint32_t *const tR1 = pin_address(tbl + rowR + (long)e_tail * tokWords);
  const uint32_t *const tW1 = pin_address(tbl + rowW + (long)e_tail * tokWords);

  //: THE LAST WORD'S PADDING TOKENS DO NOT EXIST.  The table leaves them zero;
  //: the clause-free block has no table to leave them zero, so the mask is
  //: applied in the FORM, where it is one select on one word of a parallel map.
  const uint32_t tailMask =
      (L & (kTokBits - 1)) ? ((1u << (L & (kTokBits - 1))) - 1u) : 0xFFFFFFFFu;
  const int lastWord = tokWords - 1;

  //: THE FORM.  The membership AND, and nothing else: every thread takes whole
  //: words of the token axis, ANDs the block's surviving clauses on each side and
  //: publishes one word per side into the resident ring.  Affine, dependency-free
  //: and over the whole CTA.
  //:
  //: THE RING'S HEAD IS THE WALK'S CURSOR PLUS ITS DEPTH, so the former stops of
  //: its own accord when it catches its tail -- backpressure, with no test for it
  //: anywhere but the loop bound.
  int formEnd = 0;
  auto form = [&]() {
    const int head = scook[kCookWord] + kFormWords;
    const int limit = head < tokWords ? head : tokWords;
    for (int w = formEnd + tid; w < limit; w += kThreads) {
      const uint32_t off = (uint32_t)w * (uint32_t)sizeof(uint32_t);
      uint32_t rm, wm;
      if constexpr (kNoClause) {
        rm = 0xFFFFFFFFu;
        wm = 0xFFFFFFFFu;
      } else if constexpr (kTailWhole) {
        rm = *reinterpret_cast<const uint32_t *>(gather_ptr(tR0, off));
        wm = *reinterpret_cast<const uint32_t *>(gather_ptr(tW0, off));
      } else {
        rm = *reinterpret_cast<const uint32_t *>(gather_ptr(tR0, off)) &
             *reinterpret_cast<const uint32_t *>(gather_ptr(tR1, off));
        wm = *reinterpret_cast<const uint32_t *>(gather_ptr(tW0, off)) &
             *reinterpret_cast<const uint32_t *>(gather_ptr(tW1, off));
      }
      if (w == lastWord) {
        rm &= tailMask;
        wm &= tailMask;
      }
      const int slot = w & (kFormWords - 1);
      sfR[slot] = rm;
      sfW[slot] = wm;
    }
    formEnd = limit;
  };

  //: A PRODUCER REFILLS IN BATCHES; THAT IS WHAT THE RING IS FOR.  Forming the
  //: one or two words a dense super-chunk consumed, once per super-chunk, is the
  //: fused walk again with extra steps: the loop's own setup is a FIXED cost paid
  //: by EVERY warp on EVERY super-chunk, against a redundant decode that was one
  //: warp's.  So the former runs only once the ring has drained past half --
  //: every `2048 / C` super-chunks at dense, where the walk consumes `C / 32`
  //: words of each.
  auto form_if_low = [&]() {
    if (formEnd < tokWords && formEnd - scook[kCookWord] < kFormWords / 2) form();
  };

  //: THE WALK.  One lockstep unit consumes the formed bitmap a word at a time,
  //: keeps a running count on each side, and closes the super-chunk at the FIRST
  //: TOKEN that would push either count past its buffer.
  //:
  //: THE CLOSING RULE IS TOKEN-GRANULAR, NOT WORD-GRANULAR, and the difference is
  //: not cosmetic: a chunk whose read list is full keeps ABSORBING write-live
  //: tokens until a read-live one arrives, and a stretch dead on both sides is
  //: swallowed however many words it spans.  Stopping at the word boundary where
  //: a count happened to fill would put those write tokens in the next chunk and
  //: make the partition depend on the packing, which is an implementation detail
  //: the sizing model does not have.  So the loop runs on the CUT, and the counts
  //: appear only in the per-lane test that produces it.
  //:
  //: IT CLEARS WHAT IT TOOK.  On a mid-word close the remainder is written back,
  //: so re-entry reads what is left and its prefix counts ARE the counts of what
  //: is left -- which is why there is no `inrange` and no mask on any access.
  //:
  //: IT STOPS AT THE FORMED REGION'S EDGE, NOT AT THE TOKEN AXIS'S.  Running out
  //: of formed words leaves the super-chunk OPEN; the CTA forms more and calls
  //: again, and the partition is therefore unchanged.
  //:
  //: ITS STATE IS REGISTER-RESIDENT, exactly as rung 5B's cursor was.  Shared
  //: memory carries only what ANOTHER warp needs -- the cursor, so the former
  //: knows how far it may run, and the open flag, so the whole CTA takes the same
  //: path -- and it carries them by STORE ONLY.
  int cookWord = 0, cookNr = 0, cookNw = 0;
  bool cookOpen = false;
  //: the words of one K-window.  A super-chunk never crosses a window line,
  //: so its list entries share one window and sixteen LOCAL bits name any of
  //: them -- which is what unbinds L from the index width.
  constexpr int kWinWords = kWindow / kTokBits;
  int chunkWin = 0;
  auto walk = [&](int d) {
    uint16_t *lrd = iread(d);
    uint16_t *lwr = iwrite(d);
    uint8_t *ljt = ithr(d);
    const bool exists = cookOpen || cookWord < tokWords;
    int nr = cookOpen ? cookNr : 0;
    int nw = cookOpen ? cookNw : 0;
    if (!cookOpen) chunkWin = cookWord / kWinWords;
    bool closed = cookWord >= tokWords;
    const uint32_t lt = (1u << lane) - 1u;
#pragma unroll 1
    while (cookWord < formEnd) {
      //: THE WINDOW LINE.  A super-chunk holding entries closes at the line
      //: (its locals must share a window); an empty one slides its window
      //: forward -- a dead stretch crosses lines exactly as it crosses words.
      if (cookWord / kWinWords != chunkWin) {
        if (nr != 0 || nw != 0) {
          closed = true;
          break;
        }
        chunkWin = cookWord / kWinWords;
      }
      const int slot = cookWord & (kFormWords - 1);
      const uint32_t rm = sfR[slot];
      const uint32_t wm = sfW[slot];
      const int prx = bit_count(rm & lt);
      const int pwx = bit_count(wm & lt);
      const bool lvR = ((rm >> lane) & 1u) != 0u;
      const bool lvW = ((wm >> lane) & 1u) != 0u;
      const int cut = first_set(
          lane_ballot((lvR && nr + prx >= C) || (lvW && nw + pwx >= C)));
      const uint32_t keep = cut >= kTokBits ? 0xFFFFFFFFu : ((1u << cut) - 1u);
      const int t = (cookWord - chunkWin * kWinWords) * kTokBits + lane;
      if (lane < cut) {
        if (lvR) {
          lrd[nr + prx] = (uint16_t)t;
          ljt[nr + prx] = (uint8_t)(nw + pwx + (lvW ? 1 : 0));
        }
        if (lvW) lwr[nw + pwx] = (uint16_t)t;
      }
      nr += bit_count(rm & keep);
      nw += bit_count(wm & keep);
      if (cut < kTokBits) {
        if (lane == 0) {
          sfR[slot] = rm & ~keep;
          sfW[slot] = wm & ~keep;
        }
        closed = true;
        break;
      }
      ++cookWord;
      if (cookWord == tokWords) {
        closed = true;
        break;
      }
    }
    if (!closed) {
      cookNr = nr;
      cookNw = nw;
      cookOpen = true;
      if (lane == 0) {
        scook[kCookWord] = cookWord;
        scook[kCookOpen] = 1;
      }
      return;
    }
    //: THE BUFFER'S TAIL.  A slot past the gathered count names token zero, so
    //: every gather stays in range without a predicate, and carries a threshold
    //: of zero, so its Gram row is empty.  Its panel row is zeroed at production
    //: and its output row is never drained.
#pragma unroll
    for (int k = 0; k < (C + kTokBits - 1) / kTokBits; ++k) {
      const int i = lane + k * kTokBits;
      if (i < C) {
        if (i >= nr) {
          lrd[i] = 0;
          ljt[i] = 0;
        }
        if (i >= nw) lwr[i] = 0;
      }
    }
    const bool wasOpen = cookOpen;
    cookOpen = false;
    if (lane == 0) {
      int *h = ihdr(d);
      h[0] = nr;
      h[1] = nw;
      h[2] = exists ? 1 : 0;
      h[3] = chunkWin;  // whose window the LOCAL indices live in
      scook[kCookWord] = cookWord;
      if (wasOpen) scook[kCookOpen] = 0;
    }
  };

  //: THE WAIT, AND IT IS THE ORDINARY ONE.  `scook[kCookOpen]` is CTA-uniform by
  //: construction -- it is read out of shared memory after a rendezvous -- so
  //: every warp takes the same path and this loop's body does not run unless the
  //: walk genuinely outran the former.  The CALLER supplies the rendezvous that
  //: publishes the flag, because in the steady state that barrier already exists.
  auto walk_wait = [&](int d) {
#pragma unroll 1
    while (scook[kCookOpen]) {
      form();
      rendezvous();
      if (warp == 0) walk(d);
      rendezvous();
    }
  };

  // ---- the pipeline's prologue: cook and stage `STAGES - 1` super-chunks ----
  //: THE PROLOGUE FILLS THE SCHEDULE STREAM FIRST AND THE VALUE STREAM SECOND,
  //: which is the same ordering the steady state keeps: the walk runs `AHEAD`
  //: super-chunks out before the first value tile is issued.  It pays the two
  //: barriers that fill the formed ring ONCE, before the first walk; the steady
  //: state pays none.
  if (tid == 0) {
    scook[kCookWord] = 0;
    scook[kCookOpen] = 0;
  }
  rendezvous();
  form();
  rendezvous();
#pragma unroll 1
  for (int s = 0; s < AHEAD; ++s) {
    if (warp == 0) walk(s);
    rendezvous();
    walk_wait(s);
    form_if_low();
    rendezvous();
  }
#pragma unroll 1
  for (int s = 0; s < STAGES - 1; ++s) {
    stage_chunk(s, ihdr(s)[1] != 0);
    stage_commit();
  }

  //: THE RING CURSORS ARE COUNTERS, NOT REMAINDERS.  `IRING` is not a power of
  //: two, so a modulo would put a compile-time reciprocal in the body of the one
  //: loop the whole kernel runs in.  Three of them: what is computing, what is
  //: being staged, and what is being cooked.
  int dc = 0, ds = STAGES - 1, dn = AHEAD;
  int prev_d = 0, prev_nr = 0;
#pragma unroll 1
  for (;;) {
    const int *h = ihdr(dc);
    const int nr = h[0], nw = h[1];
    if (!h[2]) break;
    //: PER-SIDE LIVENESS GATES PER-SIDE WORK, and that is not an optimisation
    //: bolted on: it is what having two lists MEANS.  The readout, the Gram, the
    //: output and its drain are read-side work; the value staging and the fold
    //: are write-side work; the intra term is the one that needs both, and it
    //: needs both to be present rather than merely nonzero -- a super-chunk with
    //: no write entry stages no value tile, so a term that reads one must not
    //: run.  On a routing whose sides are separated (`split`) this is where the
    //: mma and the staging the union schedule could not skip actually go away.
    const bool liveR = nr != 0, liveW = nw != 0;
    const bool live = liveR || liveW;

    //: THE COOK RUNS BEFORE THE PRODUCTION AND PUBLISHES AT THE SAME RENDEZVOUS
    //: THE PANELS DO.  It is one unit's work against three units' production, so
    //: what it costs the CTA is the difference and not the whole of it.
    if (warp == 0) walk(dn);

    if (live) {
      const int wn = ihdr(dc)[3];
      while (pWin < wn) {
        pReadP += (long)kWindow * kWtot;
        pWriteP += (long)kWindow * kWtot;
        ++pWin;
      }
      const uint32_t rOff = (uint32_t)iread(dc)[prow] * kPRowB;
      const uint32_t wOff = (uint32_t)iwrite(dc)[prow] * kPRowB;
      produce(pReadP, dstR, rOff, prow < nr);
      produce(pWriteP, dstW, wOff, prow < nw);
    }

    stage_wait<STAGES - 2>();
    rendezvous();  // (1) V visible / R,W visible / last chunk's reads done / O drained
                   //     and the walk's cursor and open flag are published

    walk_wait(dn);
    //: THE FORM RIDES THE BARRIERS THAT ARE ALREADY HERE.  It is issued where the
    //: value tile is, after the rendezvous that published the walk's cursor and
    //: before (2), which is what makes the formed words visible to the next
    //: super-chunk's walk.
    form_if_low();
    stage_chunk(ds, ihdr(ds)[1] != 0);
    stage_commit();

    const uint32_t atv = svB + cmpSlot + rowTV;
    cmpSlot += kSlot;
    if (cmpSlot == kRing) cmpSlot = 0;

    float acc[NTOK][4];
    float macc[NTOK][4];
    if (liveR) {
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
        if (gwarp < kMassUnits) {
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

    // ---- 2. Gram: G = staircase(R.W^T) over THIS BLOCK's slots ---------------
    //: THE MASK IS THE MONOTONE STAIRCASE, AND IT IS THE SAME SHAPE AS THE
    //: TRIANGLE IT GENERALISES.  Both lists are ascending, so read entry `tr`
    //: keeps exactly the write entries below its threshold `J[tr]` -- one byte
    //: the cook already observed.  At dense `J[tr] = tr + 1` and this reduces to
    //: `tc <= tr` term for term.
    if (gwarp < GWARPS) {
      int jv[MPW][2];
      if constexpr (GROUPS == 1) {
        const uint8_t *ljt = ithr(dc);
#pragma unroll
        for (int m = 0; m < MPW; ++m) {
          const int tr = (gm0 + m) * 16 + (lane >> 2);
          jv[m][0] = ljt[tr];
          jv[m][1] = ljt[tr + 8];
        }
      }
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
            for (int h2 = 0; h2 < 2; ++h2)
#pragma unroll
              for (int e = 0; e < 4; ++e) g[u][h2][e] = 0.f;
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
            for (int h2 = 0; h2 < 2; ++h2) {
              const int tc = (gn0 + (np + u) * 2 + h2) * 8 + (lane & 3) * 2;
              if constexpr (GROUPS == 1) {
                //: one group: the block's whole Gram -- mask and publish, as
                //: rung 5D did.
                const int j0 = jv[m][0], j1 = jv[m][1];
                const float m0 = (tc + 0 < j0) ? g[u][h2][0] : 0.f;
                const float m1 = (tc + 1 < j0) ? g[u][h2][1] : 0.f;
                const float m2 = (tc + 0 < j1) ? g[u][h2][2] : 0.f;
                const float m3 = (tc + 1 < j1) ? g[u][h2][3] : 0.f;
                store_bf16x2(&sg[tr * LdG + tc], pack_bf16x2(m0, m1));
                store_bf16x2(&sg[(tr + 8) * LdG + tc], pack_bf16x2(m2, m3));
              } else {
                //: two groups: this is HALF the Gram (this group's slots), so
                //: it is published RAW in fp32; the exchange after (2) sums
                //: the halves and applies the staircase ONCE.  The sum happens
                //: in fp32 exactly because bf16 partials would round twice.
                float *gp = sgf + grp * C * G::LdGf + tr * G::LdGf + tc;
                store_f32(gp + 0, g[u][h2][0]);
                store_f32(gp + 1, g[u][h2][1]);
                store_f32(gp + G::LdGf * 8 + 0, g[u][h2][2]);
                store_f32(gp + G::LdGf * 8 + 1, g[u][h2][3]);
              }
            }
        }
      }
    }

    }

    // ---- 3. fan the PREVIOUS super-chunk's partial output into the accumulator
    if (prev_nr) { drain(prev_d, prev_nr); drain_den(prev_d, prev_nr); }

    if (liveW) {
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
      if (gwarp < kMassUnits) {
#pragma unroll
        for (int q = 0; q < kMassKD; ++q) {
          uint32_t mb[4];
          load_frag_t(mb, ADDRF_M(bwF, kk * 16, q));
          mma(mass_st[2 * q], ones_a, &mb[0]);
          mma(mass_st[2 * q + 1], ones_a, &mb[2]);
        }
      }
    }
    }

    //: (2) THE GRAM IS VISIBLE; R AND W ARE DEAD UNTIL THE NEXT PRODUCTION; AND
    //: THE PREVIOUS SUPER-CHUNK'S INDEX SLOT IS FREE FOR THE COOK TO REUSE.  It
    //: is issued on every super-chunk, live or dead, because that last clause is
    //: what separates the drain from a cook one iteration later.
    rendezvous();

    //: THE GRAM EXCHANGE -- the slot carve's ONE genuine cross-group sum, and
    //: also its prize: the summed Gram is multiplied into V ONCE per block
    //: instead of once per group.  fp32 partials in, one staircase, bf16 out;
    //: the extra rendezvous publishes the sum to every group's intra.  The
    //: conditions are CTA-uniform (they come off the shared header), so the
    //: barrier is uniform too.
    if constexpr (GROUPS > 1) {
      if (liveR && liveW) {
        const uint8_t *ljt = ithr(dc);
#pragma unroll 1
        for (int e = tid; e < C * (C / 2); e += kThreads) {
          const int tr = e / (C / 2);
          const int tc = (e - tr * (C / 2)) * 2;
          const int j = ljt[tr];
          float s0 = 0.f, s1 = 0.f;
#pragma unroll
          for (int g2 = 0; g2 < GROUPS; ++g2) {
            const float *gp = sgf + g2 * C * G::LdGf + tr * G::LdGf + tc;
            s0 += gp[0];
            s1 += gp[1];
          }
          store_bf16x2(&sg[tr * LdG + tc],
                       pack_bf16x2(tc < j ? s0 : 0.f, tc + 1 < j ? s1 : 0.f));
        }
        rendezvous();
      }
    }

    if (liveR) {
    if (liveW) {
    // ---- 5. intra: O^T += V^T . G^T ----------------------------------------
      uint32_t va[MTOK][4];
#pragma unroll
      for (int kk = 0; kk < MTOK; ++kk) load_frag_t(va[kk], ADDRT_DV(atv, kk * 16));
      //: the contraction runs over EVERY write token (the k axis stays whole);
      //: what splits across the groups is the OUTPUT token axis -- each group
      //: computes its m-tiles into its own acc and the fan-in sums the groups
      //: exactly as it sums the blocks.  THE TILE BASE IS COMPILE-TIME: `acc`
      //: is a register array, and a runtime index into one is a spill, so the
      //: group takes its half through a uniform branch on its own id rather
      //: than through arithmetic the allocator cannot see through.
      constexpr int MI = MTOK / GROUPS;
      auto intra_half = [&](auto base_c) {
        constexpr int M0 = decltype(base_c)::value;
#pragma unroll
        for (int mi = 0; mi < MI; ++mi) {
          uint32_t gb[MTOK][4];
#pragma unroll
          for (int kk = 0; kk < MTOK; ++kk)
            load_frag(gb[kk], sgT + (uint32_t)(((M0 + mi) * 16 * LdG + kk * 16) * 2));
#pragma unroll
          for (int kk = 0; kk < MTOK; ++kk)
#pragma unroll
            for (int p = 0; p < 2; ++p)
              mma(acc[2 * (M0 + mi) + p], va[kk], &gb[kk][2 * p]);
        }
        //: THE INTRA TERM'S SHARE OF THE MASS, same compile-time base.
        if (gwarp < kIntraUnits) {
#pragma unroll
          for (int mi = 0; mi < MI; ++mi)
#pragma unroll
            for (int q = 0; q < kIntraKD; ++q) {
              uint32_t mg[4];
              load_frag(mg, sgT +
                                (uint32_t)(((M0 + mi) * 16 * LdG +
                                            q * kIntraUnits * 16) *
                                           2) +
                                massOff);
#pragma unroll
              for (int p = 0; p < 2; ++p)
                mma(macc[2 * (M0 + mi) + p], ones_a, &mg[2 * p]);
            }
        }
      };
      if constexpr (GROUPS == 1) {
        intra_half(std::integral_constant<int, 0>{});
      } else {
        if (grp == 0)
          intra_half(std::integral_constant<int, 0>{});
        else
          intra_half(std::integral_constant<int, MI>{});
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
    //: held by the four lanes of the first group, two tokens each.
    if (lane < 4) {
#pragma unroll
      for (int j = 0; j < NTOK; ++j) {
        store_f32(&sden[gwarp * C + j * 8 + lane * 2], macc[j][0]);
        store_f32(&sden[gwarp * C + j * 8 + lane * 2 + 1], macc[j][1]);
      }
    }
    }
    prev_d = dc;
    prev_nr = nr;
    if (++dc == IRING) dc = 0;
    if (++ds == IRING) ds = 0;
    if (++dn == IRING) dn = 0;
  }

  rendezvous();
  if (prev_nr) { drain(prev_d, prev_nr); drain_den(prev_d, prev_nr); }
#undef ADDRF
#undef ADDRT
#undef ADDRT_M
#undef ADDRF_M
#undef ADDRT_DV
}

// ------------------------------------------------------- the union table
//: THE TABLE IS BUILT ONCE FOR THE WHOLE GRID.  A lockstep unit takes thirty-two
//: consecutive tokens, one per lane, and votes once per entry; the vote IS the
//: word, so nothing is packed afterwards and nothing is zeroed first.
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
  //: consumer and reading in range is what keeps the lockstep unit converged.
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

// THE ARM SET.  Rung 5A's, arm for arm, so every delta this rung reports is a
// same-arm delta.  Two measured axes (`C`, `SWZ`) and `BC`, which carries rung
// 2's measured optimum and is not a tuned axis.  The three WIDTH CONFIGURATIONS
// are not axes: they are problem sizes.  (8,8) is N = 64 with ONE block, the
// same-work control; (64,64) is the flagship N = 4096; (16,16) is N = 256, a real
// fan-in cheap enough for an fp64 reference to gate at full length.
#define RUNG5D_ARMS(X)                                                       \
  X(16, false,  64,  8,  8) X(32, false,  64,  8,  8) X(64, false,  64,  8,  8) \
  X(16, true ,  64,  8,  8) X(32, true ,  64,  8,  8) X(64, true ,  64,  8,  8) \
  X(16, false,  64, 64, 64) X(32, false,  64, 64, 64) X(64, false,  64, 64, 64) \
  X(16, true ,  64, 64, 64) X(32, true ,  64, 64, 64) X(64, true ,  64, 64, 64) \
  X(32, false,  32, 64, 64) X(64, false,  32, 64, 64)                          \
  X(16, false, 128, 64, 64) X(32, false, 128, 64, 64) X(64, false, 128, 64, 64) \
  X(32, false, 256, 64, 64)                                                    \
  X(16, false,  64, 16, 16) X(32, false,  64, 16, 16) X(64, false,  64, 16, 16)

template <int C, bool SWZ, int BC, int W0, int W1>
static void launch(const torch::Tensor &pr, const torch::Tensor &pw,
                   const torch::Tensor &v, const torch::Tensor &tbl,
                   torch::Tensor &o, torch::Tensor &den, int BH, int L) {
  constexpr size_t sm = Geom<C, SWZ, BC>::bytes;
  static bool configured = false;
  if (!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(compacted_kernel<C, SWZ, W0, W1, BC>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int)sm));
    configured = true;
  }
  const dim3 grid(BlockPlan<W0, W1, BC>::blocks, BH);
  compacted_kernel<C, SWZ, W0, W1, BC>
      <<<grid, threads_for(BC), sm, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const __nv_bfloat16 *>(pr.data_ptr()),
          reinterpret_cast<const __nv_bfloat16 *>(pw.data_ptr()),
          reinterpret_cast<const __nv_bfloat16 *>(v.data_ptr()),
          reinterpret_cast<const uint32_t *>(tbl.data_ptr()),
          o.data_ptr<float>(), den.data_ptr<float>(), L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

//: THE TABLE, AS A TENSOR: `[B*H, 2, entries, L/32]` of packed bits, sides apart
//: because the cook wants them apart.  It is a SEPARATE ENTRY POINT so that its
//: cost is reported on its own line and never folded into a kernel delta.
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
//: a sum over the N/BC blocks and no block knows the total.  `eps` is a runtime
//: argument -- it is a constant of the DEFINITION and belongs to the definition's
//: file.  The divide is round-to-nearest by name.
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
  TORCH_CHECK(L <= (1 << 27),
              "sequence length sanity bound (int arithmetic headroom)");
}

torch::Tensor union_table(torch::Tensor pr, torch::Tensor pw, torch::Tensor v,
                          int64_t chunk, int64_t bc, int64_t w0, int64_t w1) {
  check_inputs(pr, pw, v, chunk, w0, w1);
  const int BH = v.size(0) * v.size(1), L = v.size(2);
#define RUNG5D_TABLE(C_, W_, BC_, W0_, W1_)                                   \
  if (chunk == C_ && bc == BC_ && w0 == W0_ && w1 == W1_)                     \
    return build_table<W0_, W1_, BC_>(pr, pw, BH, L);
  RUNG5D_ARMS(RUNG5D_TABLE)
#undef RUNG5D_TABLE
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
#define RUNG5D_DISPATCH(C_, W_, BC_, W0_, W1_)                                 \
  if (chunk == C_ && swizzle == W_ && bc == BC_ && w0 == W0_ && w1 == W1_) {   \
    const long want = (long)BH * 2 * BlockPlan<W0_, W1_, BC_>::entries *      \
                      tok_words(L);                                          \
    TORCH_CHECK(tbl.numel() == want,                                           \
                "the union table was built for a different arm");              \
    launch<C_, W_, BC_, W0_, W1_>(pr, pw, v, tbl, o, den, BH, L);              \
    return {o, den};                                                           \
  }
  RUNG5D_ARMS(RUNG5D_DISPATCH)
#undef RUNG5D_DISPATCH
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

//: THE RING'S DEPTH IS PART OF THE BUILD'S IDENTITY, so a measurement can never
//: be attributed to a depth the running binary does not have.
__global__ void stamp_kernel(int *out) {
  out[0] = kBuildMagic + kFormWords;
  out[1] = (kGroupThreads / 32) * 1000 + kD;
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
#define RUNG5D_CENSUS(C_, W_, BC_, W0_, W1_)                                       \
  if (chunk == C_ && swizzle == W_ && bc == BC_ && w0 == W0_ && w1 == W1_) {       \
    C10_CUDA_CHECK(cudaFuncGetAttributes(                                          \
        &a, compacted_kernel<C_, W_, W0_, W1_, BC_>));                             \
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
  RUNG5D_ARMS(RUNG5D_CENSUS)
#undef RUNG5D_CENSUS
  TORCH_CHECK(false, "unbuilt arm");
  return r;
}

//: The built set with every DERIVED quantity of an arm reported beside it: the
//: block plan's rectangle, the block count, the staging depth and the CTA target.
std::vector<std::vector<int64_t>> arms() {
  std::vector<std::vector<int64_t>> out;
#define RUNG5D_LIST(C_, W_, BC_, W0_, W1_)                                    \
  out.push_back({C_, (int64_t)(W_), BC_, W0_, W1_,                            \
                 (int64_t)BlockPlan<W0_, W1_, BC_>::blocks,                   \
                 (int64_t)BlockPlan<W0_, W1_, BC_>::lead_span,                \
                 (int64_t)BlockPlan<W0_, W1_, BC_>::tail_span,                \
                 (int64_t)Geom<C_, W_, BC_>::stages,                          \
                 (int64_t)Geom<C_, W_, BC_>::bytes,                           \
                 (int64_t)kCtaTarget<C_, W_, BC_>,                            \
                 (int64_t)BlockPlan<W0_, W1_, BC_>::entries});
  RUNG5D_ARMS(RUNG5D_LIST)
#undef RUNG5D_LIST
  return out;
}

}  // namespace rung6e

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &rung6e::forward, "per-side compacted self-normalising routed CLA");
  m.def("forward_raw", &rung6e::forward_raw, "the readout's d_v+1 columns, undivided");
  m.def("union_table", &rung6e::union_table, "the precomputed membership union table");
  m.def("divide_pass", &rung6e::divide_pass, "the ratio, one elementwise pass");
  m.def("build_stamp", &rung6e::build_stamp, "build identity");
  m.def("arms", &rung6e::arms, "the built arms with their derived block plan and geometry");
}
