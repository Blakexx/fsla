// csrc/rola/src/carry/carry_kernel.cuh -- K31 RUNG R1: THE BRING-UP CARRY KERNEL.
//
// A NEW KERNEL FAMILY, BESIDE THE SHIPPED ONE.  Nothing here is reachable from
// `rola::chunk`'s dispatch and nothing here edits it; the shipped arms remain the
// baseline until this family beats them cell for cell (study section 6).
//
// WHAT IT COMPUTES.  The INTER (carried-state) term only:
//
//     O^T[:, t] += S^T[:, box] . R~^T[box, t]      readout, vs state-at-window-start
//     S^T[:, box] += V~^T[:, t] . W~[t, box]       fold, of the window's own writes
//
// where `R~`/`W~` are the read/write leaf coefficients restricted to the owner's
// box, `V~` is the value row with the ones column appended (the mass channel), and
// the window grid is GLOBAL and FIXED at `W` tokens -- identical for every owner,
// which is what removes the shipped kernel's boundary drift (study section 2.1).
// The within-window (INTRA) pairs are NOT computed here; rung R2 builds that kernel
// and R1's conformance adds the contribution on the host from the fp64 oracle.
//
// SYMBOLS, each at first use: `D` levels, `B` digits per level, `K` the uniform
// sub-box span, `M` the capacity multiplier, `BC` the leaves one owner holds,
// `W` the window length in tokens, `C` the private batch's token tile, `DV` (`d_v`)
// the value channel count -- AN ARM PARAMETER (K35 S3b), not a constant of this file:
// every derived quantity below (`kCols`, `LdV`, `LdO`, `kPlaneRows`, `MCH`, `kVW`,
// the segment/staging/plane byte counts and the straddle) follows from it, and no
// site anywhere tests its value.  `k_tok` (a token's nonzero digits per level) appears
// NOWHERE in this file, and that is the point: the schedule is computed, never
// branched on.
//
// THE ONE ORDERING LAW is window-grain read-before-write on the owner's own slots.
// A batch therefore NEVER crosses a window line: a batch that did would owe the
// cross-window pairs inside it to neither the readout (which is against
// state-at-batch-start) nor the intra (which is within-window).  See
// docs/internals/carry/carry_kernel.md.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <type_traits>

#include "box.cuh"
#include "facts/arch_caps.cuh"
#include "common/design.cuh"
#include "common/ops.cuh"

namespace rola::carry {

using namespace denseref::ops;
namespace design = denseref::design;

static constexpr int kPadE = 8;  // bf16 row padding, elements (16 B)
static constexpr int kPadF = 4;  // fp32 row padding, elements (16 B)
//: `d_v` IS AN ARM PARAMETER AND NOT A FILE CONSTANT (K35 S3b).  Both FORWARD bodies
//: take it as `DV` -- `CarryGeom`/`SplitGeom` name it `kDv` and derive `kCols = DV + 1`,
//: the mass column that rides the state -- so the two built widths are two arms of one
//: body and no site tests the width.  THE REVERSE FAMILY IS OUT OF SCOPE and keeps its
//: own `kDv = 64` (`backward_kernel.cuh`): that body is a separate two-scan design with
//: a standing seam gap of its own, and widening it is not this stage's work.
//: THE CHANNEL-SPLIT BODY'S WARP COUNT, and the reverse family's.  Those bodies split
//: the `kDv` CHANNELS over a fixed four warps and hold the whole box in each; the
//: CARVED body's executor count is the arm's STREAM count instead
//: (`CarryGeom::kWarps`) and is not a constant of this file; the channel-split
//: body's `kWarps`/`kThreads` are in `box.cuh` with the rest of the family's
//: arm-independent geometry (K36), so the DISPATCH TU can read them without
//: parsing this header.
static_assert(kAtomLeaves == kMmaK, "the page granule is one MMA k-step");

//: A COMPILE-TIME LOOP.  Every per-level quantity here is a template argument or a
//: register-array index, so the level index must be a constant expression; a
//: `for (int l = 0; l < D; ++l)` would make `load_run<n_l(l)>` ill-formed and would
//: turn every per-level array index dynamic, which is a stack frame.
//: a type carried through a generic lambda's argument, which is how a body that is
//: instantiated per SIDE names that side's span vector without a second template.
//: A SIDE OF THE FLOW MECHANISM, as a type: its stream count, its executors per
//: stream, its group's candidate-pair count and whether a placement's payload is the
//: TOKEN (the write side, whose value copy wants a global address) or the token's
//: UNION INDEX (the read side, whose fan-in ring is keyed on it).  It exists so that
//: the counters, the placement, the room test and the full-tile test are written ONCE
//: and instantiated twice -- see `carry_kernel.md`, "The flow counters".
template <int NS, int NW, int NC, bool TOKPAY>
struct FlowSide {
  static constexpr int kNS = NS, kNW = NW, kNC = NC;
  static constexpr bool kTokenPayload = TOKPAY;
};

template <class T>
struct type_tag {
  using type = T;
};

template <int N, class F>
__device__ __forceinline__ void static_for(F&& f) {
  if constexpr (N > 0) {
    static_for<N - 1>(f);
    f(std::integral_constant<int, N - 1>{});
  }
}

//: THE CARVE (K35 multi-level).  A STREAM owns an affine RECTANGLE of the owner's
//: digits -- a per-level span vector, `prod_l S_l` leaves and ALL `kCols` channels
//: -- and tiles those leaves in a private mixed-radix order.  That tiling is private
//: to the stream: the lattice's `local` order, the state plane, decode and the facts
//: geometry never see it, so the readout needs no state image, no `ldmatrix` for A
//: and no CTA barrier.  The two sides carve the SAME box along different axes, which
//: is why the state evacuates registers at each phase change (the exchange).
//:
//: THE EXECUTOR MAP (K35 S3a), under the terminology law: a STREAM owns a box, an
//: EXECUTOR (a warp on sm_80) runs streams, and the two counts are independent dials
//: -- streams are the in-CTA sparsity dial, warps the execution budget.  `WPS` is the
//: arm's declared WARPS PER WRITE STREAM: `kWarps = SVW::kStreams * WPS`, a warp's
//: write stream is `strm >> log2 WPS` and its COLUMN HALF is `strm & (WPS-1)`.  The
//: READ carve is declared at ONE STREAM PER WARP, so the readout is untouched by the
//: split and the accumulator's leaf count is the read box's at every arm.

template <int D, int B, int K, int M, int DV, int W, int C, int WPS, int NSW, int NSR, int PK,
          uint32_t MO>
struct CarryGeom {
  using BP = BoxPlan<D, B, K, M>;
  //: THE TWO CARVES, DERIVED FROM THE DECLARATION (K35 S1b): each side's private order
  //: comes from `MO` alone and its span vector from that order and its stream count.
  using SVW = StreamSpans<D, B, K, M, MO, kSideWrite, NSW>;
  using SVR = StreamSpans<D, B, K, M, MO, kSideRead, NSR>;
  static constexpr int BC = BP::kBC;
  //: THE VALUE WIDTH, AS A DECLARATION.  `kDv` channels plus the MASS column, which
  //: rides the state as one more row of every plane the state touches.  The clauses
  //: are STRUCTURAL and not a list of admitted widths: the channels tile into whole
  //: MMA m-tiles (`MCH`), and a value row is a whole number of WARP-WIDE 128-byte
  //: passes (`kVW = kDv / 64`), which is what lets the fetch read it in one coalesced
  //: transaction a pass and the send store it with no predicate (K35 R-C item 5).
  static constexpr int kDv = DV;
  static constexpr int kCols = DV + 1;
  static_assert(DV >= kMmaM && DV % kMmaM == 0, "the value channels tile into whole MMA m-tiles");
  static_assert((DV & (DV - 1)) == 0 && DV % 64 == 0,
                "a value row is a power-of-two number of channels and a whole number of "
                "warp-wide 128-byte passes, so the fetch and the send are unpredicated");
  //: THE TWO DIALS: `kStreams` is the WRITE side's ownership grain, `kWarps` the issue
  //: budget spent on it.
  static constexpr int kWPS = WPS;
  static constexpr int kStreams = SVW::kStreams;
  static constexpr int kWarps = kStreams * WPS;
  static constexpr int kThreads = kWarps * 32;
  static_assert(WPS >= 1 && (WPS & (WPS - 1)) == 0, "warps per write stream is a power of two");
  static_assert(SVR::kStreams == kWarps,
                "the READ carve declares one stream per WARP, so its stream count IS the "
                "executor count and the readout carries no column split");
  //: the WRITE STREAM's leaves -- the segment row's width -- and the EXECUTOR's own
  //: column half of them, which is the accumulator's width and the read box's size.
  static constexpr int kLeafW = SVW::kLeaf;
  static constexpr int kLeafE = kLeafW / WPS;
  static_assert(kLeafE * WPS == kLeafW && kLeafE == SVR::kLeaf,
                "a column half of the write stream's box is exactly one read stream's box");
  static constexpr int LdV = kDv + kPadE;
  static constexpr int LdS = kLeafW + kPadE;
  static constexpr int LdR = kLeafE + kPadE;
  static constexpr int LdO = kDv + kPadF;
  //: THE STRIP IS THE BATCH'S TOKEN TILE, and it is the MMA's whole `k` step: a batch
  //: of the union walk is `CP` tokens, a stream's rank inside it is below `CP` by
  //: construction, so ONE segment slot is exactly ONE contraction step.
  static constexpr int CP = kMmaK;
  static_assert(2 * CP == C, "a half holds exactly two strips");

  // ---- THE FOLD'S TENANTS (K35 S2, the cooperative fetch) ------------------
  //: THE PER-STREAM SEGMENT, SINGLE SLOT.  `SegValueRows` is `[stream][row][channel]`
  //: at row stride `LdV`, `SegCoeffRows` is `[stream][row][slot]` at row stride `LdS`;
  //: a ROW is one (token, stream) pair, placed by the FETCHER at the token's rank
  //: inside the batch.  The two sub-tiles are adjacent halves of one segment and are
  //: separate only because `ldmatrix` reads each at its own row stride.
  static constexpr int kSegRows = CP;
  static constexpr int seg_v_bytes = kStreams * kSegRows * LdV * (int)sizeof(__nv_bfloat16);
  static constexpr int seg_c_bytes = kStreams * kSegRows * LdS * (int)sizeof(__nv_bfloat16);
  //: THE WINDOW'S COEFFICIENT STAGING, as a ROLLING BLOCK: the fetcher expands from
  //: SMEM, and what it needs at a moment is the compressed factor rows of ONE batch's
  //: union tokens (`BP::kStageW` amplitudes each -- the owner's per-level runs, not
  //: the topology-wide packed row).  Two blocks, so the block for batch `b + 2` is in
  //: flight while `b + 1` is expanded: the same two-stages-in-flight law the segments
  //: obey, at a cost that does not grow with `W`.
  static constexpr int kStgSlots = 2;
  static constexpr int stg_bytes = kStgSlots * kSegRows * BP::kStageW * (int)sizeof(__nv_bfloat16);
  static constexpr int gain_bytes = align16c(kStgSlots * kSegRows * (int)sizeof(__nv_bfloat16));
  //: the state plane's page, at entry and exit only -- dead for the whole window loop,
  //: which is why the exchange and the read panels straddle it too.
  static constexpr int out_bytes = kAtomLeaves * LdO * (int)sizeof(float);

  //: ---- K35-E: THE EXCHANGE'S SEGMENTS -------------------------------------
  //: THE STATE IS RE-LAID AT EVERY PHASE CHANGE (P76's ORDER LAW), by WRITER-SIDE
  //: PLACEMENT: the publisher stores EACH LEAF'S ROW of channels at that leaf's slot in
  //: the CONSUMER's private order, into the consumer's own contiguous segment; a CTA
  //: barrier; the consumer pulls its segment straight into fragments with
  //: `ldmatrix.trans` and the segment is DEAD.  The two changes of a window -- write
  //: order to read order before the readout, read order back to write order after it --
  //: are consecutive moments and never co-live, so they are ONE region.
  //:
  //: THE LAYOUT, `XchgSegments`: `[consumer][leaf][channel]` over `BC` rows, because
  //: the consumers partition the owner's box on either side (`kWarps * kLeafE == BC`).
  //: A row is the `kCols` channels -- `kDv` value channels and the MASS -- padded to a
  //: sixteen-byte quantum, which is the same expression `LdV` carries because `kDv` is
  //: already a whole number of quanta: 72 and 136 at the two widths.
  static constexpr int LdX = kDv + kPadE;
  static constexpr int kXchG = kDv / 8;  // the row's sixteen-byte CHANNEL granules
  static constexpr int kXchRows = kWarps * kLeafE;
  static_assert(kXchRows == BC && LdX >= kCols && LdX % 8 == 0,
                "the consumers partition the box and a segment row is a whole number of "
                "sixteen-byte quanta");
  static constexpr int seg_xch_bytes = kXchRows * LdX * (int)sizeof(__nv_bfloat16);
  //: THE TIED ARM: THE EXCHANGE IS THE IDENTITY AND THE TRANSIT IS NOT ALLOCATED (P76,
  //: "THE EXCHANGE MOVES ONLY THE DISORDER", K35 S1b addendum 3).  Both sides class a
  //: level alike unless it is sparse on exactly one, so with no single-side-sparse level
  //: the ORDERS coincide, at equal stream counts the derivation yields the same VECTORS,
  //: and the leaf -> (reader, slot) permutation is the identity: nothing to publish,
  //: nothing to pull, `hi` stays in registers and this region has NO TENANT.  An arm is
  //: tied by its DECLARATION and never by a flag (docs/internals/carry/carry_kernel.md).
  static constexpr bool spans_agree(int l) {
    return l >= D || (SVW::s(l) == SVR::s(l) && spans_agree(l + 1));
  }
  static constexpr bool orders_agree(int l) {
    return l >= D || (SVW::Ord::rank(l) == SVR::Ord::rank(l) && orders_agree(l + 1));
  }
  static constexpr bool kTied = WPS == 1 && spans_agree(0) && orders_agree(0);
  //: THE EXCHANGE'S STATE, NAMED AND VISIBLE (Blake, 2026-08-27).  A shape the design
  //: has not yet optimized must still COMPILE and be CORRECT -- refusing it would trade a
  //: working arm for a missing one -- so the third shape takes the ordinary full transit
  //: and says so rather than paying it silently.  `kXchFullReorder` is addendum 3's case
  //: (c), PARKED: identical PARTITIONS with different ORDERS moves no leaf between warps,
  //: so its true cost is a cross-lane bit-permutation shuffle and its true transit is
  //: ZERO, but the shuffle network is not built and the full exchange is correct for it.
  //: The census reports this, which is how S6's table can show which rows OVERPAY and how
  //: the parked item finds its customers.
  static constexpr int kXchZero = 0, kXchFull = 1, kXchFullReorder = 2;
  static constexpr int kXchState =
      kTied ? kXchZero : ((WPS == 1 && spans_agree(0)) ? kXchFullReorder : kXchFull);
  static constexpr int xch_bytes = kTied ? 0 : seg_xch_bytes;
  //: THE SWIZZLE IS A DERIVATION FROM BOTH ACCESS PATTERNS, NOT A GUESS (findings,
  //: derivation 4).  `LdX / 2` is `4 (mod 32)` at both widths, so consecutive rows sit
  //: ONE four-bank block apart and a bank is `(4*row + 4*granule + word) mod 32`.
  //:   THE PULL is `ldmatrix.trans`, serviced eight lanes at a time: a phase is EIGHT
  //:   CONSECUTIVE rows at ONE granule, so it already walks eight distinct blocks and
  //:   asks only that the swizzle be CONSTANT over an aligned group of eight rows.
  //:   THE PUBLISH is `st.shared.b16`: one instruction covers FOUR rows (the lane's
  //:   quad, four leaves of one fragment) x EIGHT CONSECUTIVE channels, and two lanes
  //:   sharing a 32-bit word are a broadcast rather than a conflict -- so it is four
  //:   rows x four words in ONE granule, and it needs the four rows to land in four
  //:   DIFFERENT blocks.  Those four rows are the consumer-side images of four leaves
  //:   and are scattered by the two carves, so the row's own index must reach the
  //:   granule: `sw(row) = ((row>>3) ^ (row>>4) ^ (row>>6)) & 7` gives every row bit a
  //:   distinct nonzero image and is constant over an aligned eight.
  //: THE MASS rides the row's ONE PAD granule, whose four words the single mass element
  //: must spread over on its own: `n(row) = ((row>>3) ^ (row>>5)) & 3` at element
  //: `kDv + 2*n`.  `tests/k31/test_segment_layout.py` runs both legs of both changes
  //: over every built arm and both widths and pins max-way ONE.
  static_assert((LdX / 2) % kBallotLanes == kBallotLanes / 8,
                "a segment row must step one four-bank block, which is what makes the "
                "consumer's eight-row phase a bank permutation with no swizzle at all");
  static_assert(kXchG >= 8 && (kXchG & (kXchG - 1)) == 0,
                "the swizzle XORs a three-bit word into the granule index, so the CHANNEL "
                "granules must be a power-of-two count of at least eight -- which keeps the "
                "swizzled granule off the ONE pad granule the mass rides");

  // ---- THE READOUT'S TENANTS (K35 R-B) -------------------------------------
  //: THE READOUT IS WARP-PRIVATE END TO END (P76 "THE RESET", R-B).  A reader is a
  //: warp, it walks its OWN compacted list, stages and expands its OWN tokens and
  //: issues its own fan-in: no warp writes into another's tenant here, so the read
  //: side has no membership index, no row map, no flow counter and no ring.
  //:
  //: THE READER SEGMENT, `[reader][row][slot]` at row stride `LdR`.  On the readout the
  //: B operand is `R~^T [leaves x tokens]`, so TOKENS are the `n` axis and the state is
  //: the `k`-side A operand held in registers -- there is no value row to place and a
  //: reader's segment is coefficient rows alone.  A ROW is one token of this reader's
  //: own list at its position inside the group; one slot, because the second stage is
  //: registers exactly as it is in the fold (`ldmatrix` drains the segment).
  static constexpr int rseg_bytes = kWarps * kSegRows * LdR * (int)sizeof(__nv_bfloat16);
  //: THE READ SIDE'S FACTOR STAGING, the same rolling two-block window the fold has and
  //: the same `StagedFactorRow` layout -- but PER READER, because each warp's group is
  //: its own tokens and no two readers want the same rows.  What differs from the fold
  //: is the plane it is filled from (`pread`) and the list that names the tokens, so one
  //: body still serves both sides and the round trip that gates the layout gates both.
  static constexpr int rstg_bytes =
      kWarps * kStgSlots * kSegRows * BP::kStageW * (int)sizeof(__nv_bfloat16);
  //: THE FAN-IN'S PANEL, 32 CHANNELS WIDE AND WIDTH-INDEPENDENT (K35 R-C, deciding the
  //: open item R-B raised and Blake's 2026-08-28 ruling put to measurement).  The panel
  //: exists to TRANSPOSE the readout's accumulator so the reduction is issued
  //: LANE-PER-CHANNEL: out of the m16n8 C fragment a lane holds four TOKENS at eight
  //: channels, so a reduction straight from the fragment is four HALF-FILLED 32-byte
  //: sectors.  What R-C measured is that FULL sectors are not sufficient either -- a
  //: register SHUFFLE-PACK reaching four FULL sectors in four DIFFERENT TOKEN ROWS
  //: (four distinct 128-byte segments per instruction) cost +93% at the dense cell on
  //: +2.7% instructions, with the RED count identical to the staged form's to the
  //: instruction.  WHAT THE REDUCTION WANTS IS ONE SEGMENT, not four full sectors.
  //:
  //: SO THE PANEL STAYS AND ONLY ITS WIDTH CHANGES.  It never needs more than the 32
  //: channels ONE reduction covers live at once, so it is `kMmaN` token rows of 32
  //: channels plus the mass, per reader -- 9,216 B at EVERY value width, against
  //: 17,408 B at `d_v = 64` and 33,792 B at 128 for the full-width form.  That is what
  //: takes the `d_v = 128` container back off the two-CTA SMEM line R-B flagged, and it
  //: keeps the issue shape that measured best: 32 lanes over 32 CONSECUTIVE channels of
  //: ONE token, one aligned 128-byte transaction.  A channel block costs one extra
  //: `__syncwarp` per block and no extra store, load or reduction.
  //: sm_90 MAPPING: the same panel feeds `red.global.v4.f32` (a lane taking four
  //: contiguous channels), which cuts the reduction count 4x at one quarter the lanes.
  static constexpr int LdB = 32 + kPadF;
  static constexpr int kChBlk = kDv / 32;
  static_assert(kChBlk * 32 == kDv,
                "the reduction covers 32 channels, so a value row is "
                "a whole number of channel blocks");
  static constexpr int rout_bytes = kWarps * kMmaN * LdB * (int)sizeof(float);
  static_assert(kDv % kBallotLanes == 0,
                "the fan-in issues one reduction per lockstep unit over CONSECUTIVE "
                "channels, so a value row is a whole number of them");

  //: ---- THE PARKED `lo` (K35 S5), A PER-ARM DECLARATION, DEFAULT OFF ---------
  //: `PK == 1` spends the READOUT with the write-view `lo` in the warp's OWN exchange
  //: segment instead of in registers, so those registers are dead across the arm's
  //: register peak and become the output fragments' home (P76 F5: the payoff is
  //: registers -> LEAVES).  It costs the straddle: the readout's tenants sit ABOVE the
  //: segments.  THE FIT IS AN IDENTITY -- a warp keeps `32 * 4 * NL * (2*MCH + 1)`
  //: bytes = `kLeafE * LdX * 2` (carry_kernel.md, "The park").
  static constexpr int kParkWords = (kLeafE / kMmaN) * (2 * (kDv / kMmaM) + 1);
  static constexpr bool kPark = PK != 0;
  static_assert(PK == 0 || PK == 1, "the park is a declaration and carries no third value");
  static_assert(kParkWords * kBallotLanes * (int)sizeof(uint32_t) == kLeafE * LdX * 2,
                "a warp's parked `lo` is exactly its own exchange segment");
  //: the park's bytes are the SEGMENT's, not the transit's: a tied arm allocates no
  //: transit and can still park, because what a warp parks is its own `lo`.
  static constexpr int park_bytes = kPark ? seg_xch_bytes : 0;

  static constexpr int off_rseg = park_bytes;
  static constexpr int off_rstg = off_rseg + rseg_bytes;
  static constexpr int off_rout = off_rstg + rstg_bytes;
  static constexpr int read_bytes = off_rout + rout_bytes;
  static_assert(off_rstg % 16 == 0,
                "an asynchronously filled block begins at a sixteen-byte boundary");

  //: ---- THE STRADDLE.  Three sets of tenants live in the region below the window's
  //: index, and NO TWO OF THEM ARE EVER CO-LIVE: the FOLD's (the per-stream segments,
  //: the rolling staging, the gain and the page buffer), the CHANGE's (the exchange
  //: segments) and the READOUT's (the reader segments, the read staging, the row index
  //: and the fan-in ring).  So the allocation carries the MAXIMUM and every one of them
  //: begins at the region's base; nothing is shrunk to fit another.
  //: THE PARK IS THE ONE EXCEPTION (K35 S5): at `PK == 1` the exchange's segments stay
  //: live THROUGH the readout, so its tenants begin at `park_bytes` and `read_bytes`
  //: carries the park -- the maximum below is unchanged in form.
  static constexpr int fold_dead_bytes =
      seg_v_bytes + seg_c_bytes + stg_bytes + gain_bytes + out_bytes;
  static constexpr int straddle_bytes = align16c(
      fold_dead_bytes > xch_bytes ? (fold_dead_bytes > read_bytes ? fold_dead_bytes : read_bytes)
                                  : (xch_bytes > read_bytes ? xch_bytes : read_bytes));
  //: `kWarps` READ LISTS AND ONE WRITE LIST (K35 R-B).  The fold is cooperative -- its
  //: schedule is the CTA's union walk and a stream's membership is its published ballot
  //: word -- while the readout is warp-private, so each reader walks its OWN ballot into
  //: its OWN list and nothing about it is published.
  static constexpr int kLists = kWarps + 1;
  static constexpr int kListStride = ((W + 7) / 8) * 8;
  static constexpr int list_bytes = align16c(kLists * kListStride * (int)sizeof(uint16_t));
  //: THE WINDOW'S BALLOT WORDS, one per lane of the walk -- so a window may be at
  //: most a lockstep unit's worth of words wide.
  static constexpr int kLW = W / kBallotLanes;
  //: THE PUBLISHED MEMBERSHIP AND ITS PREFIX COUNTS.  A fetcher places rows into
  //: segments it does not consume, so it needs EVERY stream's write-side ballot and
  //: the per-word prefix popcount that turns a token into a RANK.  Each executor
  //: publishes its own once per window; `kStreams * kLW` words is the whole index.
  static constexpr int mem_bytes = align16c(kStreams * kLW * (int)sizeof(uint32_t));
  static constexpr int pfx_bytes = align16c(kStreams * kLW * (int)sizeof(uint16_t));
  //: THE PER-TILE LIVENESS MASK, ONE WORD PER EXECUTOR: bit `nj` says the batch wrote a
  //: nonzero coefficient somewhere in the write stream's n-tile pair `nj`, so skipping
  //: that pair's `ldmatrix` and MMAs is EXACT -- a tile with no live leaf contributes
  //: nothing.  The word is per EXECUTOR and not per stream so that a warp RESETS ONLY
  //: WHAT IT READS: the reset sits between a warp's own read of the mask and the
  //: barrier that lets the fetchers OR into it, and a word shared with the stream's
  //: other warp would be reset out from under that warp's read.  The `WPS` words of one
  //: stream carry the same bits; each warp tests its own column half's field of them.
  static constexpr int mask_bytes = align16c(kWarps * (int)sizeof(uint32_t));
  //: THE FLOW COUNTERS (K35 S2c), and they are the WHOLE of the fold's synchronisation:
  //: `landed[s]` (rows of write stream `s` that have ARRIVED this window), `drained[e]`
  //: (tiles executor `e` has PULLED, the fetcher taking the MINIMUM over a stream's
  //: `WPS` executors) and `done` (warps that have walked their share).  A stream fires
  //: when ITS OWN segment is full, which is exactly what the counts state; on sm_90+ the
  //: pair IS an mbarrier's arrive/wait, and this is the portable form of the same facts.
  static constexpr int flow_bytes = align16c((2 * kWarps + 1) * (int)sizeof(uint32_t));
  //: THE CTA'S SCHEDULE FACTS, TWO WORDS (K35 R-C item 1): the union list's LENGTH and
  //: the whole-window SKIP.  They exist because the union is DERIVED ONCE per CTA per
  //: window by one warp rather than re-derived by all of them -- the owner-grain ballots
  //: are functions of the OWNER alone, so every warp used to issue the same `sum_l g_l`
  //: table loads per side and OR them to the same words (`## STAGE-LEDGER` S3a's
  //: replicated derivation).  One warp now votes them, walks the list and publishes
  //: these two facts behind the barrier the walk ALREADY had, so the deduplication
  //: costs no barrier of its own.
  static constexpr int sched_bytes = align16c((kLW + 1) * (int)sizeof(uint32_t));

  //: `kRegFloor` is the DECLARED register floor for this family -- the measured census
  //: of the heaviest built arm.  The `d_v = 128` four-stream arm sits AT THE WALL: 255
  //: registers a lane on both ratified arches with zero spill and zero stack, which is
  //: the hardware's own maximum, so the floor is now the ceiling too and any growth
  //: there spills rather than widening.  The residency it derives is unchanged at both
  //: thread counts -- 64K/(128*255) still rounds to two CTAs and 64K/(256*255) to one --
  //: so raising the declaration moves no launch bound; it makes the census's own number
  //: the one the geometry is stated in.
  static constexpr int kRegFloor = 255;
  //: ONE slot per stream, because the SECOND STAGE IS REGISTERS (F4).
  static constexpr int kStages = 1;

  // ---- the allocation, as offsets -----------------------------------------
  static constexpr int off_seg_v = 0;
  static constexpr int off_seg_c = off_seg_v + seg_v_bytes;
  static constexpr int off_stg = off_seg_c + seg_c_bytes;
  static constexpr int off_gain = off_stg + stg_bytes;
  static constexpr int off_out = off_gain + gain_bytes;
  //: THE THREE MOMENTS ALL BEGIN AT THE REGION'S BASE, which is what "straddled" means
  //: here: the fold's tenants, the exchange's segments and the read panels are the same
  //: bytes read under three different layouts at three different moments.
  static constexpr int off_xch = 0;
  static constexpr int off_read = off_rseg;
  static constexpr int off_list = straddle_bytes;
  static constexpr int off_mem = off_list + list_bytes;
  static constexpr int off_pfx = off_mem + mem_bytes;
  static constexpr int off_mask = off_pfx + pfx_bytes;
  static constexpr int off_flow = off_mask + mask_bytes;
  static constexpr int off_sched = off_flow + flow_bytes;
  static constexpr size_t bytes = (size_t)(off_sched + sched_bytes);
  static constexpr int cta_target = design::cta_target((int)bytes, kThreads, kRegFloor);
  //: ---- THE LEDGER, PER MOMENT (P76 "THE SMEM LEDGER, PER MOMENT") ----------
  //: The window's INDEX -- the lists, the published ballots, the prefix counts and the
  //: liveness mask -- is live in every moment and is nobody's straddle.  The three
  //: moments over it are the FOLD (segments, staging, gain, the page buffer), the
  //: CHANGE (the exchange segments alone) and the READOUT (the reader segments and the
  //: per-reader factor staging and the fan-in's 32-channel panel).
  static constexpr int index_bytes =
      list_bytes + mem_bytes + pfx_bytes + mask_bytes + flow_bytes + sched_bytes;
  static constexpr int fold_bytes = fold_dead_bytes + index_bytes;
  static constexpr int change_bytes = xch_bytes + index_bytes;
  static constexpr int readout_bytes = read_bytes + index_bytes;
  static_assert((size_t)fold_bytes <= bytes && (size_t)change_bytes <= bytes
                    && (size_t)readout_bytes <= bytes,
                "the allocation covers every moment's tenants");
  static_assert(fold_dead_bytes <= straddle_bytes && xch_bytes <= straddle_bytes
                    && read_bytes <= straddle_bytes,
                "the straddled region covers each moment's tenants whole");
  static_assert(design::smem_fits(bytes), "this arm does not fit the SMEM maximum");
};

//: THE OWNER-GRAIN PRODUCE MAP, which the reverse family's panel is expanded
//: against (`backward_kernel.cuh`): a row's `kRowThr` threads split the box's `BC`
//: coefficients into `SPT` contiguous LATTICE slots each.  Because the lattice index
//: is a bit permutation (R0 spec section 2.4), a thread's slot run is a sub-lattice:
//: per level the digits it touches are a CONTIGUOUS run of the owner's own run at a
//: runtime base, and WHICH element of that run each of its slots wants is a
//: compile-time function of the unrolled slot index.
template <int D, int B, int K, int M, int C>
struct ProduceMap {
  using BP = BoxPlan<D, B, K, M>;
  static constexpr int kRowThr = kThreads / C;
  static constexpr int SPT = BP::kBC / kRowThr;
  static constexpr int kJBits = ilog2c(SPT);
  static constexpr int vary(int l) { return SPT >> BP::run_shift(l); }
  static constexpr int n_l(int l) {
    return vary(l) < 1 ? 1 : (vary(l) > BP::s(l) ? BP::s(l) : vary(l));
  }
  static constexpr int max_n(int l) {
    return l == D ? 1 : (n_l(l) > max_n(l + 1) ? n_l(l) : max_n(l + 1));
  }
  static constexpr int kMaxN = max_n(0);
  static constexpr int idx_of(int l, int j) { return (j >> BP::run_shift(l)) & (n_l(l) - 1); }
};

//: THE PRIVATE LEAF ORDER AND THE PRODUCE'S THREAD MAP.  A stream's leaves are
//: numbered by a MIXED RADIX over its own per-level spans -- `p = sum_l r_l <<
//: shift(l)`, with `r_l` the digit's offset inside the stream's range at level `l`
//: -- so `p` is a bit permutation of the levels' digit offsets exactly as `local`
//: is, and the expansion is `sum_l n_l` amplitude loads followed by products of
//: register values at CONSTANT indices: no dynamic indexing, no spill, no
//: materialized gate plane.  A span-1 level contributes a constant and drops out of
//: the order on its own.
//:
//: THE PRIVATE-ORDER RULE IS THE ORDER BY MODE (P76, ruled 2026-08-26; K35 S1b), and it
//: is fixed by the DECLARATION alone -- no span is consulted: DENSE innermost, then
//: SPARSE (sparse on this side only), then BOTH_SPARSE (sparse on both), lattice order
//: inside a class, and NO LEVEL IS PINNED.  Dense innermost makes an eight-slot n-tile
//: full; sparse next lets those levels index WHOLE tiles, where the digit skip is exact;
//: both-sparse outermost is where the two sides' carves agree, so the exchange is
//: block-diagonal there and vanishes when they agree everywhere.  `ModeOrder` (box.cuh)
//: states the rule; the CARVE is derived from it, which is why nothing here can drift
//: from what `StreamSpans` cut.
//:
//: THE PIN IS GONE BECAUSE THE EXCHANGE NO LONGER PAIRS LATTICE-ADJACENT LEAVES.  The
//: retained plane published `bf16x2` words of two adjacent leaves, which forced the
//: lattice's innermost level to lie lowest in BOTH private orders (`span(D-1) >= 2`);
//: writer-side placement stores each leaf on its own, so the carve vectors are fully
//: general on both sides (P76, THE ORDER LAW).
template <int D, int B, int K, int M, class SV>
struct SpanMap {
  using BP = BoxPlan<D, B, K, M>;
  static constexpr int kLeaf = SV::kLeaf;
  static constexpr int span(int l) { return SV::s(l); }
  static constexpr int bits(int l) { return ilog2c(span(l)); }
  //: the rule as a strict order on the levels, and `rank` as the position it induces --
  //: `0` innermost, and nothing pinned there.  Both are counts, so neither tabulates.
  static constexpr int rank(int l) { return SV::Ord::rank(l); }
  static constexpr int shift_from(int l, int i) {
    return i >= D ? 0 : (rank(i) < rank(l) ? bits(i) : 0) + shift_from(l, i + 1);
  }
  static constexpr int shift(int l) { return shift_from(l, 0); }
  //: THE PACKED PAIR IS SIMPLY THE INNERMOST LEVEL OF THE ORDER.  A lane mints two
  //: adjacent private slots at once, and they differ in the digit of whichever level
  //: lies lowest in `p` -- so that level's two amplitudes are adjacent in the staged
  //: run and one four-byte access covers the pair.  Levels of span one occupy no bits
  //: and their position is immaterial, so the level wanted is the lowest one with a
  //: span of at least two; a box holds at least two leaves, so exactly one level
  //: qualifies.  This is a statement about the ORDER only -- the lattice is not
  //: consulted, and no side of the exchange asks for adjacency any more.
  static constexpr int pair_from(int l) {
    return l >= D ? -1 : ((shift(l) == 0 && span(l) >= 2) ? l : pair_from(l + 1));
  }
  static constexpr int kPairLevel = pair_from(0);
  static_assert(kLeaf >= 2 && kPairLevel >= 0 && span(kPairLevel) >= 2,
                "a box has at least two leaves, so exactly one level lies lowest in the "
                "private order with a span of at least two and supplies the packed pair");
};

//: THE EXCHANGE'S ROW MAP, AS AN AFFINE MAP OVER THE SLOT'S BITS.  A publisher owns a
//: slot in ITS order and must name the CONSUMER's row -- the consumer's index times its
//: box plus the consumer's own private slot.  That map is AFFINE, and the proof is the
//: absence of carries: a slot's level-`l` digit is `offset_l | field_l` (the stream's
//: offset is a multiple of the span, so it occupies the bits above the field); the
//: consumer's group and private digit are a BIT SPLIT of that digit; the group terms sum
//: into the consumer index in mixed radix and the private terms into disjoint fields of
//: the consumer's slot.  Every sum is therefore an OR, so
//:     row(slot) = row(0) + sum over the slot's SET BITS of a per-bit STEP,
//: and every step is COMPILE-TIME -- it depends on the two orders and the two stream
//: grids and not on which stream is publishing.  The publisher pays the base ONCE per
//: warp; a fragment's row is then the base plus a constant.
//:
//: `PMP`/`SVP` are the PUBLISHER's order and spans, `PMC`/`SVC` the CONSUMER's, and
//: `LEAFC` the consumer's box, which is what its index scales by.
template <int D, class PMP, class PMC, class SVC, int LEAFC>
struct XchRow {
  static constexpr int level_of(int p, int l) {
    return l >= D ? -1
                  : ((PMP::bits(l) > 0 && p >= PMP::shift(l) && p < PMP::shift(l) + PMP::bits(l))
                         ? l
                         : level_of(p, l + 1));
  }
  static constexpr int step_at(int p, int l) {
    return l < 0
               ? 0
               : ((p - PMP::shift(l)) >= ilog2c(SVC::s(l))
                      ? (1 << ((p - PMP::shift(l)) - ilog2c(SVC::s(l)))) * SVC::grid_div(l) * LEAFC
                      : (1 << (PMC::shift(l) + (p - PMP::shift(l)))));
  }
  static constexpr int step(int p) { return step_at(p, level_of(p, 0)); }
  //: the map of a whole constant, and of a runtime value bit by bit -- both the same sum.
  static constexpr int kBits = 16;
  static constexpr int steps(int v, int p) {
    return p >= kBits ? 0 : (((v >> p) & 1) ? step(p) : 0) + steps(v, p + 1);
  }
  static constexpr int of(int v) { return steps(v, 0); }
};

//: THE EXCHANGE'S TWO SWIZZLE WORDS, AS CONSTEXPR FUNCTIONS -- and they are used at
//: COMPILE TIME as often as at run time.  Both are XOR-folds of the row index, so both
//: are GF(2)-LINEAR; and a publisher's row is `base | delta` with DISJOINT bits (that is
//: the same no-carry fact that makes `XchRow` affine), so
//:     swz(base | delta) == swz(base) ^ swz(delta)
//: with `swz(delta)` a constant.  A publish therefore carries ONE swizzle word per warp
//: and XORs a compile-time value into it per fragment: no per-row arithmetic at all.
constexpr int xch_swz_of(int row) { return ((row >> 3) ^ (row >> 4) ^ (row >> 6)) & 7; }
constexpr int xch_mw_of(int row) { return ((row >> 3) ^ (row >> 5)) & 3; }

//: a contiguous run of `n` bf16, at the widest access that covers it.  The runs are
//: naturally aligned by construction (a level's column base is a multiple of `B`, an
//: owner's run starts at a multiple of `s_l` and a group's sub-run at a multiple of
//: its own width), so this is one, two or four accesses.
template <int n>
__device__ __forceinline__ void load_run(unsigned short (&dst)[n], const void* p) {
  static_assert(n == 1 || n == 2 || n == 4 || n == 8 || n == 16,
                "an admissible (k, m) run is 1, 2, 4, 8 or 16 wide");
  if constexpr (n == 1) {
    dst[0] = *reinterpret_cast<const unsigned short*>(p);
  } else if constexpr (n == 2) {
    *reinterpret_cast<unsigned int*>(dst) = *reinterpret_cast<const unsigned int*>(p);
  } else if constexpr (n == 4) {
    *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(p);
  } else {
#pragma unroll
    for (int i = 0; i < n / 8; ++i)
      *reinterpret_cast<uint4*>(dst + i * 8) =
          load_vec16(reinterpret_cast<const char*>(p) + i * 16);
  }
}

//: THE FIRST LEVEL A SIDE SPANS WHOLE, which is where a GAINLESS expansion's product
//: begins: the read side has no gain, so seeding at this level is what keeps a multiply
//: by one out of the readout's coefficient rows.  `-1` where a side carves every level,
//: and the seed is then the identity the accumulator starts at.
template <class SV, int D>
constexpr int first_whole_level(int l = 0) {
  return l >= D ? -1 : (!SV::carves(l) ? l : first_whole_level<SV, D>(l + 1));
}

//: THE SECOND LAUNCH BOUND: the CTA count this arm can actually reach, taken from the
//: geometry that sized its tenants.  A residency the launch bound does not ask for is
//: SMEM spent for nothing.
template <int D, int B, int K, int M, int DV, int W, int C, int WPS, int NSW, int NSR, int PK,
          uint32_t MO>
constexpr int kCarryCtaTarget = CarryGeom<D, B, K, M, DV, W, C, WPS, NSW, NSR, PK, MO>::cta_target;

// ---------------------------------------------------------------- the kernel

template <int D, int B, int K, int M, int DV, int W, int C, int WPS, int NSW, int NSR, int PK,
          uint32_t MO, bool STATE>
__global__ __launch_bounds__((CarryGeom<D, B, K, M, DV, W, C, WPS, NSW, NSR, PK, MO>::kThreads),
                             (kCarryCtaTarget<D, B, K, M, DV, W, C, WPS, NSW, NSR, PK, MO>)) void
carry_kernel(const __nv_bfloat16* __restrict__ pread, const __nv_bfloat16* __restrict__ pwrite,
             const __nv_bfloat16* __restrict__ gwrite, const __nv_bfloat16* __restrict__ v,
             const uint32_t* __restrict__ rows, float* __restrict__ o, float* __restrict__ den,
             int L, int tok_words, int v_row_bytes, const float* __restrict__ s_in,
             float* __restrict__ s_out) {
  using BP = BoxPlan<D, B, K, M>;
  using G = CarryGeom<D, B, K, M, DV, W, C, WPS, NSW, NSR, PK, MO>;
  //: THE ARM'S VALUE WIDTH, named once so the body below reads a geometry and never a
  //: file constant: `kDv` channels and the mass column that rides them.
  constexpr int kDv = DV;
  constexpr int kCols = DV + 1;
  //: THE TWO SPAN VECTORS.  The READ side's carve and the WRITE side's carve of the
  //: same box, each with its own private order under the rule above.
  using SVR = typename G::SVR;
  using SVW = typename G::SVW;
  using PM = SpanMap<D, B, K, M, SVR>;
  using PMW = SpanMap<D, B, K, M, SVW>;
  //: THE ROW TABLES, ONE PER SIDE at that side's own partition (K35 S1b addendum): a
  //: bitlist is voted from ONE side's factors, so a stream's test reads exactly one row
  //: per tested level and only the OWNER's whole-run OR spans several.
  using CRR = CarveRows<SVR>;
  using CRW = CarveRows<SVW>;
  constexpr int kWarps = G::kWarps;
  constexpr int kThreads = G::kThreads;

  constexpr int BC = BP::kBC;
  constexpr int LdV = G::LdV, LdS = G::LdS, LdR = G::LdR, LdO = G::LdO;
  //: THE WRITE STREAM'S leaves (`kLeafW`, the segment row) and THE EXECUTOR'S OWN
  //: COLUMN HALF of them (`kLeafE`), which is what the accumulator holds: `NL` n-tiles
  //: and `KDw` readout k-steps.  `NLW` is the stream's whole n-tile count, which the
  //: fetcher's expansion and the liveness mask are stated in.
  constexpr int kLeafW = G::kLeafW, kLeafE = G::kLeafE;
  constexpr int NL = kLeafE / kMmaN;
  constexpr int NLW = kLeafW / kMmaN;
  constexpr int KDw = kLeafE / kMmaK;
  constexpr int MCH = kDv / kMmaM;  // the channel m-tiles, ALL of them per warp
  //: the fan-in's channel blocks -- one reduction's worth of channels each.
  constexpr int kChBlk = G::kChBlk;
  static_assert(C % kMmaM == 0 && kThreads % C == 0, "the batch tile must divide");
  static_assert(kLeafE % kMmaK == 0, "an executor's leaves are a whole number of k-tiles");
  static_assert(BC % kAtomLeaves == 0, "K2: an owner holds whole page atoms");
  static_assert(design::smem_fits(G::bytes), "this arm does not fit the SMEM maximum");
  static_assert(LdS % 8 == 0 && LdR % 8 == 0 && LdV % 8 == 0,
                "an ldmatrix row address must be 16-byte aligned");
  static_assert(kWarps <= kBallotLanes, "one lane of the walk slices each executor's share");
  static_assert(SVW::kStreams * WPS == SVR::kStreams,
                "the executor map: `WPS` warps per WRITE stream, one READ stream each");
  static_assert(W % kBallotLanes == 0 && kThreads % kBallotLanes == 0,
                "a warp's tokens must be exactly one ballot word of the owner's row");

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int owner = (int)blockIdx.x;
  const int bh = (int)blockIdx.y;

  extern __shared__ __align__(16) char carry_smem[];
  //: ONE SLOT PER STREAM, and the SECOND STAGE IS REGISTERS: `ldmatrix` drains the
  //: slot, the `__syncwarp` after it IS the drained fact, and the slot is refilled for
  //: the next batch while this one's MMAs run out of fragments (F4, form (b)).
  constexpr int STAGES = G::kStages;
  static_assert(STAGES == 1, "the cooperative fetch carries one slot per stream");
  constexpr int kStreams = G::kStreams;
  constexpr int kSegRows = G::kSegRows;
  //: THE FOLD'S TENANTS COME FIRST AND ADJACENT -- the per-stream segments, the
  //: rolling coefficient staging and the page buffer -- because the exchange's segments
  //: and the read panels straddle exactly those, and they are the set that is dead for the
  //: whole readout.
  __nv_bfloat16* const segV = reinterpret_cast<__nv_bfloat16*>(carry_smem + G::off_seg_v);
  __nv_bfloat16* const segC = reinterpret_cast<__nv_bfloat16*>(carry_smem + G::off_seg_c);
  __nv_bfloat16* const wstg = reinterpret_cast<__nv_bfloat16*>(carry_smem + G::off_stg);
  __nv_bfloat16* const gstg = reinterpret_cast<__nv_bfloat16*>(carry_smem + G::off_gain);
  float* const so = reinterpret_cast<float*>(carry_smem + G::off_out);
  //: THE READOUT'S TENANTS, straddling the fold's at the same base (K35 R-B): the
  //: per-reader segments, the per-reader rolling factor staging and the fan-in's
  //: per-reader panel.  Every one of them is warp-private -- the readout has no shared
  //: index and no meeting place.
  const SmemAddr rsegB = smem_addr(carry_smem + G::off_rseg);
  const SmemAddr rstgB = smem_addr(carry_smem + G::off_rstg);
  float* const rout = reinterpret_cast<float*>(carry_smem + G::off_rout);
  constexpr int kListStride = G::kListStride;
  uint16_t* const slist = reinterpret_cast<uint16_t*>(carry_smem + G::off_list);
  const SmemAddr memB = smem_addr(carry_smem + G::off_mem);
  const SmemAddr pfxB = smem_addr(carry_smem + G::off_pfx);
  const SmemAddr mskB = smem_addr(carry_smem + G::off_mask);
  //: THE FLOW COUNTERS as ONE base with compile-time displacements, so the three planes
  //: cost the register file one address and not three.
  const SmemAddr flowB = smem_addr(carry_smem + G::off_flow);
  //: the CTA's two published schedule facts -- the union length and the skip.
  const SmemAddr schB = smem_addr(carry_smem + G::off_sched);
  const SmemAddr segVB = smem_addr(segV), segCB = smem_addr(segC);
  const SmemAddr stgB = smem_addr(wstg), gnB = smem_addr(gstg);
  //: THE EXCHANGE'S SEGMENTS, straddled over the FOLD's tenants -- and over the read
  //: panels, which are the OTHER moment's tenant at the same base.  They are a CHANGE-
  //: moment tenant only: published, pulled into fragments, then DEAD.
  const SmemAddr xchB = smem_addr(carry_smem + G::off_xch);

  //: ONE 64-BIT BASE PER PLANE, and a 32-BIT BYTE DISPLACEMENT after it -- the
  //: shipped addressing discipline (docs/internals/carry/carry_kernel.md#state-addressing).
  const int vRowElems = v_row_bytes / (int)sizeof(__nv_bfloat16);
  const int heads = vRowElems / kDv;
  const __nv_bfloat16* const vPlane =
      pin_address(v + (long)(bh / heads) * L * vRowElems + (long)(bh % heads) * kDv);
  const __nv_bfloat16* const pR = pin_address(pread + (long)bh * L * BP::kWtot);
  const __nv_bfloat16* const pW = pin_address(pwrite + (long)bh * L * BP::kWtot);
  const __nv_bfloat16* const gW = pin_address(gwrite + (long)bh * L);
  float* const oPlane = pin_address(o + (long)bh * L * kDv);
  float* const denPlane = pin_address(den + (long)bh * L);

  //: THE STREAM ROTATION.  Two CTAs co-resident on an SM are one grid wave apart and
  //: the wave is a multiple of `kWarps`, so `owner % kWarps` would rotate both
  //: identically; the wave-stride form does not (K33 1.45).
  const int strm = (warp + ((owner + (owner >> 5)) & (kWarps - 1))) & (kWarps - 1);
  //: THE EXECUTOR MAP, read off the rotated executor index: `strm` IS this warp's read
  //: stream (one per warp by declaration), `wstrm` its write stream and `col` its
  //: COLUMN HALF of that stream's leaves.  Both are shifts and masks of compile-time
  //: powers of two, and at `WPS == 1` the mask is zero, so `col` folds to the constant
  //: and every column term below disappears.
  constexpr int kWpsBits = ilog2c(WPS);
  const int wstrm = strm >> kWpsBits;
  const int col = strm & (WPS - 1);
  const int colBase = col * kLeafE;  // the warp's first private slot of the stream's box

  // ---- the owner's digit runs, one per level -------------------------------
  //: `o_l` is a bit-field extract of the owner index: the lattice's owner
  //: coordinates are the top `sum_l log2 g_l` bits of `lambda` (R0 spec section 2.4).
  int runCol[D];
  //: AND THE STREAM'S OWN DIGIT OFFSET INSIDE THAT RUN, per level and per side: the
  //: stream index decomposes in MIXED RADIX over the per-level stream grids, so a
  //: multi-level carve is `grp_l * S_l` at EVERY carved level and nothing changes at
  //: the levels the side spans whole.  Computed ONCE per warp, before the window
  //: loop, out of shifts and masks over compile-time powers of two.
  int offR[D], offW[D];
  //: AND THE BALLOT ROW BASE, per level PER SIDE: the SAME `o_l`, addressing the FIRST
  //: row the facts pass voted for this owner's run at this level (K31-i-d), at THAT
  //: SIDE's own grain -- so the stream's own range is ONE row and the owner's word is the
  //: OR of its run's.
  uint32_t rowOffR[D], rowOffW[D];
  const uint32_t rowBytes = (uint32_t)tok_words * (uint32_t)sizeof(uint32_t);
  static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    const int o_l = (owner / BP::owner_div(l)) % BP::g(l);
    runCol[l] = BP::col_base(l) + o_l * BP::s(l);
    offR[l] = ((strm >> ilog2c(SVR::grid_div(l))) & (SVR::g(l) - 1)) * SVR::s(l);
    offW[l] = ((wstrm >> ilog2c(SVW::grid_div(l))) & (SVW::g(l) - 1)) * SVW::s(l);
    rowOffR[l] = (uint32_t)(CRR::row_prefix(l) + o_l * CRR::rows_of(BP::s(l), l)) * rowBytes;
    rowOffW[l] = (uint32_t)(CRW::row_prefix(l) + o_l * CRW::rows_of(BP::s(l), l)) * rowBytes;
  });
  //: ONE 64-BIT BASE PER SIDE, and 32-bit byte displacements after it: the READ block
  //: and then the WRITE block, each `kRows * tok_words * 4` bytes, which is 32-bit at
  //: every admitted length.
  const uint32_t* const rowsR =
      pin_address(rows + (long)bh * (long)(CRR::kRows + CRW::kRows) * (long)tok_words);
  const uint32_t* const rowsW = pin_address(
      rows + ((long)bh * (long)(CRR::kRows + CRW::kRows) + (long)CRR::kRows) * (long)tok_words);

  // ---- lane-invariant operand addressing (the shipped idiom) ---------------
  const int lr = lane & 7, tl = lane >> 3;
  const int rF = (tl & 1) * 8 + lr, cF = tl >> 1;
  const int rT = (tl >> 1) * 8 + lr, cT = tl & 1;
  //: THE CONSUMER'S TWO BASES INTO ITS STREAM'S SEGMENT.  The value sub-tile is pulled
  //: transposed (`k` = token, `m` = channel) and the coefficient sub-tile likewise
  //: (`k` = token, `n` = slot), so both are `ldmatrix.trans` at the segment's own row
  //: stride.  Both warps of a write stream pull the SAME value rows; the coefficient
  //: base carries the warp's COLUMN HALF, so the two read disjoint slot runs of one row.
  const uint32_t myV = segVB + (uint32_t)(((wstrm * kSegRows + rT) * LdV + cT * 8) * 2);
  const uint32_t myC = segCB + (uint32_t)(((wstrm * kSegRows + rF) * LdS + colBase + cF * 8) * 2);
#define ADDR_S(b_, R_, g_) ((b_) + (uint32_t)(((R_) * LdS + (g_) * 8) * 2))
#define ADDR_R(b_, R_, g_) ((b_) + (uint32_t)(((R_) * LdR + (g_) * 8) * 2))

  //: THE READER'S OWN TENANTS, by warp: its segment's rows, its two staging blocks and
  //: its list are all addressed off `strm`, and no other warp names them.
  const uint32_t myStg = rstgB + (uint32_t)(strm * G::kStgSlots * kSegRows * BP::kStageW * 2);
  //: the fan-in's own 32-channel panel, `kMmaN` token rows of it.
  float* const sow = rout + strm * kMmaN * G::LdB;

  // ---- the state, transposed: m = channels, n = this warp's leaves ---------
  float st[MCH][NL][4];
#pragma unroll
  for (int m = 0; m < MCH; ++m)
#pragma unroll
    for (int j = 0; j < NL; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) st[m][j][e] = 0.f;
  float mass_st[NL][4];
#pragma unroll
  for (int j = 0; j < NL; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) mass_st[j][e] = 0.f;

  //: ---- K35-E: THE EXCHANGE ------------------------------------------------
  //: `leafW_of` is the LATTICE leaf of an EXECUTOR-private slot `P` under the WRITE
  //: carve, which is the order the accumulator -- and therefore the state plane's page
  //: sweep -- is held in.  The private order's own `shift`/`span` give each level's
  //: digit inside `P`; the lattice's `run_shift` puts it back, and every carved level
  //: carries the stream's own digit offset.
  //: THE COLUMN HALF IS FOLDED IN HERE AND NOWHERE ELSE: `P` is the EXECUTOR's private
  //: slot in `[0, kLeafE)` and `colBase` lifts it to the STREAM's private slot in
  //: `[0, kLeafW)`, so `page_regs` and the fold's own accumulator address the right
  //: leaves with no extra nest and no per-slot test.
  auto leafW_of = [&](int P) {
    int lf = 0;
    const int Ps = P + colBase;
    static_for<D>([&](auto Lc) {
      constexpr int l = decltype(Lc)::value;
      const int dig = ((Ps >> PMW::shift(l)) & (PMW::span(l) - 1)) + offW[l];
      lf += dig << BP::run_shift(l);
    });
    return lf;
  };

  //: THE EXCHANGE'S ROW MAPS -- the ONLY place the two private orders meet.  A publisher
  //: owns a slot in ITS OWN order and must name the CONSUMER's row: the consumer's index
  //: times its box, plus the consumer's own private slot.  Both come out of the leaf's
  //: PER-LEVEL DIGITS -- the digit's group is the consumer's coordinate at that level and
  //: the offset inside the group is its private digit -- and `XchRow` shows that map is
  //: AFFINE, so each direction is a per-warp BASE plus a compile-time step per slot bit.
  //: `strm * kLeafE == wstrm * kLeafW + colBase`, which is why the WRITE side's row
  //: needs no column term of its own: the stream's row block already contains it.
  using XR = XchRow<D, PMW, PM, SVR, kLeafE>;  // WRITE slot -> READ consumer's row
  using XW = XchRow<D, PM, PMW, SVW, kLeafW>;  // READ slot  -> WRITE consumer's row
  //: THE PUBLISHER'S BASE, ONCE PER WARP.  The stream's own digit offsets and -- on the
  //: write side -- the executor's column half are the constant part of its slot, and the
  //: lane's slot pair `2 * (lane & 3)` is the rest; both are paid HERE, before the window
  //: loop, so a fragment's row inside a publish is the base plus a CONSTANT and its
  //: address is a per-warp pointer plus an IMMEDIATE.
  //: A TIED ARM PAYS NONE OF THIS: no consumer to name, no base through the window loop.
  constexpr int kSlotBits = ilog2c(kLeafW) + 1;
  int rowBaseR = 0, rowBaseW = 0;
  if constexpr (!G::kTied) {
    static_for<D>([&](auto Lc) {
      constexpr int l = decltype(Lc)::value;
      rowBaseR += (offW[l] >> ilog2c(SVR::s(l))) * (SVR::grid_div(l) * kLeafE)
                  + ((offW[l] & (SVR::s(l) - 1)) << PM::shift(l));
      rowBaseW += (offR[l] >> ilog2c(SVW::s(l))) * (SVW::grid_div(l) * kLeafW)
                  + ((offR[l] & (SVW::s(l) - 1)) << PMW::shift(l));
    });
    const int slotR = colBase + 2 * (lane & 3), slotW = 2 * (lane & 3);
    static_for<kSlotBits>([&](auto Pc) {
      constexpr int p = decltype(Pc)::value;
      rowBaseR += ((slotR >> p) & 1) ? XR::step(p) : 0;
      rowBaseW += ((slotW >> p) & 1) ? XW::step(p) : 0;
    });
  }
  //: THE SEGMENT'S ADDRESSES, at the swizzle `CarryGeom` derives: a CHANNEL granule is
  //: XORed by the row's swizzle word, and the MASS rides the row's one PAD granule at the
  //: four-word offset its own map gives.  A publisher's base carries the lane's channel
  //: INSIDE the granule -- `lane >> 2` on both sides of the exchange -- so the
  //: per-fragment term is the granule alone; the CONSUMER's rows are its own private
  //: slots, so its base is an offset and not a map at all.
  constexpr int LdX = G::LdX;
  //: AND THE BASES THEMSELVES ARE FORMED AT THE PHASE CHANGE, not before the window
  //: loop: they are a handful of shifts off `rowBaseR`/`rowBaseW`, and hoisting them
  //: would keep eight more registers live THROUGH THE FOLD, which is the arm's peak.
  //: Only the two row bases persist.
  auto xch_pub_base = [&](int rowBase) {
    return xchB + (uint32_t)((rowBase * LdX + (lane >> 2)) * 2);
  };
  auto xch_mass_base = [&](int rowBase) { return xchB + (uint32_t)((rowBase * LdX + kDv) * 2); };

  //: THE PUBLISH, WRITE ORDER -> READ ORDER, AND THE `lo` THE WARP KEEPS -- ONE NEST,
  //: because they consume the same fragment.  The accumulator's `(row, col)` is
  //: `(channel, private slot)`, so a lane's four elements of one (m-tile, n-tile) are two
  //: channels eight apart at two ADJACENT private slots -- and adjacent slots of one
  //: order belong to different consumers in the other, so each element is its own
  //: `st.shared.b16` at a compile-time granule off a per-slot base.  The `lo` halves are
  //: packed as each n-tile's four elements are consumed, so the fp32 accumulator DIES
  //: TILE BY TILE instead of living whole across the publish.  The mass moves as ONE row:
  //: its accumulator's rows 1..15 are identically zero by construction of `ones_a`.
  //: BODY COUNT: `NL * (2 + 2*(1 + 2*MCH) + MCH)` -- 84 at the 32-leaf `d_v = 64` arms,
  //: 42 at the 16-leaf ones and 74 at `d_v = 128`; O(fragments) in both modes.  STORE
  //: INSTRUCTIONS PER LANE: `4 * NL * MCH` channel halves (64, 32 and 64 at those three
  //: shapes) plus `2 * NL` mass halves (8, 4, 4), all of them 2-byte.
  auto publish_read = [&](auto& lo, auto& mlo) {
    const uint32_t xpubR = xch_pub_base(rowBaseR), xmasR = xch_mass_base(rowBaseR);
    const int swzR = xch_swz_of(rowBaseR), mwR = xch_mw_of(rowBaseR);
    //: THE CHANNEL m-TILE IS OUTERMOST so the fp32 accumulator DIES TILE BY TILE: an
    //: m-tile's `2 * NL` elements are published and immediately packed into `NL` `lo`
    //: words, and the slot's row is a compile-time offset off the warp's base, so
    //: repeating the slot loop per m-tile costs nothing.
    static_for<MCH>([&](auto Mc) {
      constexpr int m = decltype(Mc)::value;
      static_for<NL>([&](auto Jc) {
        constexpr int j = decltype(Jc)::value;
        static_for<2>([&](auto Ec) {
          constexpr int el = decltype(Ec)::value;
          constexpr int dr = XR::of(j * kMmaN + el);
          const uint32_t base = xpubR + (uint32_t)(dr * LdX * 2);
          const int swz = swzR ^ xch_swz_of(dr);
          static_for<2>([&](auto Hc) {
            constexpr int h = decltype(Hc)::value;
            store_shared_u16(base + ((uint32_t)((2 * m + h) ^ swz) << 4),
                             hi_bf16(st[m][j][el + 2 * h]));
          });
        });
        lo[m][j][0] = pack_lo(st[m][j][0], st[m][j][1]);
        lo[m][j][1] = pack_lo(st[m][j][2], st[m][j][3]);
      });
    });
    //: THE MASS IS ITS OWN PASS, one row of the state: its accumulator's rows 1..15 are
    //: identically zero by construction of `ones_a`, so only the lane quad that holds row
    //: zero carries it and only its first two elements are ever nonzero.
    static_for<NL>([&](auto Jc) {
      constexpr int j = decltype(Jc)::value;
      static_for<2>([&](auto Ec) {
        constexpr int el = decltype(Ec)::value;
        constexpr int dr = XR::of(j * kMmaN + el);
        if ((lane >> 2) == 0)
          store_shared_u16(
              xmasR + (uint32_t)(dr * LdX * 2) + ((uint32_t)(mwR ^ xch_mw_of(dr)) << 2),
              hi_bf16(mass_st[j][el]));
      });
      mlo[j] = pack_lo(mass_st[j][0], mass_st[j][1]);
    });
  };
  //: THE TIED CHANGE, FOLD -> READOUT, IN REGISTERS (K35 S1b addendum 3): what the
  //: publish and the pull COMPOSE TO under the identity permutation.  The readout's A
  //: fragments ARE the fold accumulator's top halves (KERNEL_STANDARDS section 2, K35
  //: M1a) and `lo` the bottom halves, at the SAME thread-value map -- an m16n8k16
  //: `C[m][n]` and an `A[m][k]` place `(row, col)` on a lane identically -- so a k-step's
  //: register `r` is the n-tile `2*kk + (r>>1)`'s element pair `2*(r&1)`.  Nothing is
  //: stored.  BODY COUNT: `MCH * (4*KDw + 2*NL) + 3*KDw`.
  auto split_hi = [&](auto& lo, auto& mlo, auto& sa, auto& ma) {
    static_for<MCH>([&](auto Mc) {
      constexpr int m = decltype(Mc)::value;
      static_for<KDw>([&](auto Kc) {
        constexpr int kk = decltype(Kc)::value;
        static_for<4>([&](auto Rc) {
          constexpr int r = decltype(Rc)::value;
          constexpr int j = 2 * kk + (r >> 1), e = 2 * (r & 1);
          sa[kk][m][r] = hi_bf16x2(st[m][j][e], st[m][j][e + 1]);
        });
      });
      static_for<NL>([&](auto Jc) {
        constexpr int j = decltype(Jc)::value;
        lo[m][j][0] = pack_lo(st[m][j][0], st[m][j][1]);
        lo[m][j][1] = pack_lo(st[m][j][2], st[m][j][3]);
      });
    });
    //: the mass rides row zero, so only the lane quad holding it carries a hi word; the
    //: other quads' rows are identically zero by construction of `ones_a`.
    static_for<KDw>([&](auto Kc) {
      constexpr int kk = decltype(Kc)::value;
      static_for<2>([&](auto Vc) {
        constexpr int v = decltype(Vc)::value;
        constexpr int j = 2 * kk + v;
        const uint32_t w = hi_bf16x2(mass_st[j][0], mass_st[j][1]);
        ma[kk][2 * v + 0] = ((lane >> 2) == 0) ? w : 0u;
        ma[kk][2 * v + 1] = 0u;
        mlo[j] = pack_lo(mass_st[j][0], mass_st[j][1]);
      });
    });
  };
  //: THE PULL, straight into the readout's A fragments, out of THIS warp's own contiguous
  //: segment: its rows ARE its private slots, so one `ldmatrix.trans` covers a whole
  //: (k-step, channel m-tile) tile.  The mass is one COLUMN and cannot be an `ldmatrix`
  //: tile: two `ld.shared.u16` at the pair's two rows build the one fragment word whose
  //: row 0 is the mass.  BODY COUNT: `KDw * (MCH + 2)`.
  //:
  //: THE FRAGMENT STACKS ARE TAKEN BY DEDUCED REFERENCE, NOT BY `uint32_t(&)[KDw]...`,
  //: AND THAT IS A TOOLCHAIN FACT, NOT A STYLE ONE.  `MCH` is dependent (`d_v` is a
  //: template parameter, K35 S3b) and the pinned front end (cudafe++, CUDA 12.4.99)
  //: SEGFAULTS at PARSE on a lambda parameter whose type is a reference to an array of
  //: dependent bound.  Do not spell the bound back in.
  auto pull_read = [&](auto& sa, auto& ma) {
    const int rowT = strm * kLeafE + (lane & 7) + 8 * (lane >> 4);
    const int rowM = strm * kLeafE + 2 * (lane & 3);
    const uint32_t xpullT = xchB + (uint32_t)(rowT * LdX * 2);
    const uint32_t xpullM = xchB + (uint32_t)((rowM * LdX + kDv) * 2);
    const int swzT = xch_swz_of(rowT), mwM = xch_mw_of(rowM);
    const int gT = (lane >> 3) & 1;
    static_for<KDw>([&](auto Kc) {
      constexpr int kk = decltype(Kc)::value;
      constexpr uint32_t off = (uint32_t)(kk * kMmaK * LdX * 2);
      constexpr int swz_d = xch_swz_of(kk * kMmaK);
      const int swz = swzT ^ swz_d;
      static_for<MCH>([&](auto Mc) {
        constexpr int m = decltype(Mc)::value;
        load_frag_t(sa[kk][m], xpullT + off + ((uint32_t)(((2 * m) | gT) ^ swz) << 4));
      });
      static_for<2>([&](auto Vc) {
        constexpr int v = decltype(Vc)::value;
        constexpr int dv0 = kk * kMmaK + 8 * v;
        const uint32_t a0 = xpullM + (uint32_t)(dv0 * LdX * 2);
        const int mw0 = mwM ^ xch_mw_of(dv0), mw1 = mwM ^ xch_mw_of(dv0 + 1);
        const uint32_t w =
            (uint32_t)load_shared_u16(a0 + ((uint32_t)mw0 << 2))
            | ((uint32_t)load_shared_u16(a0 + (uint32_t)(LdX * 2) + ((uint32_t)mw1 << 2)) << 16);
        ma[kk][2 * v + 0] = ((lane >> 2) == 0) ? w : 0u;
        ma[kk][2 * v + 1] = 0u;
      });
    });
  };
  //: THE PUBLISH BACK, READ ORDER -> WRITE ORDER, and it is the exact mirror: the
  //: readout's A fragments already hold the state as packed bf16 in READ order, so a
  //: register's two halves are two adjacent READ slots and each goes to its own writer's
  //: row.  Same store count as the forward publish.
  auto publish_write = [&](auto& sa, auto& ma) {
    const uint32_t xpubW = xch_pub_base(rowBaseW), xmasW = xch_mass_base(rowBaseW);
    const int swzW = xch_swz_of(rowBaseW), mwW = xch_mw_of(rowBaseW);
    //: the same m-outermost order, and for the same reason: the readout's A fragments die
    //: m-tile by m-tile instead of living whole beside the `lo` the fold still needs.
    static_for<MCH>([&](auto Mc) {
      constexpr int m = decltype(Mc)::value;
      static_for<KDw>([&](auto Kc) {
        constexpr int kk = decltype(Kc)::value;
        static_for<2>([&](auto Vc) {
          constexpr int v = decltype(Vc)::value;
          static_for<2>([&](auto Ec) {
            constexpr int el = decltype(Ec)::value;
            constexpr int dw = XW::of(kk * kMmaK + 8 * v + el);
            const uint32_t base = xpubW + (uint32_t)(dw * LdX * 2);
            const int swz = swzW ^ xch_swz_of(dw);
            static_for<2>([&](auto Uc) {
              constexpr int u = decltype(Uc)::value;
              store_shared_u16(base + ((uint32_t)((2 * m + u) ^ swz) << 4),
                               (uint16_t)(sa[kk][m][u + 2 * v] >> (16 * el)));
            });
          });
        });
      });
    });
    static_for<KDw>([&](auto Kc) {
      constexpr int kk = decltype(Kc)::value;
      static_for<2>([&](auto Vc) {
        constexpr int v = decltype(Vc)::value;
        static_for<2>([&](auto Ec) {
          constexpr int el = decltype(Ec)::value;
          constexpr int dw = XW::of(kk * kMmaK + 8 * v + el);
          if ((lane >> 2) == 0)
            store_shared_u16(
                xmasW + (uint32_t)(dw * LdX * 2) + ((uint32_t)(mwW ^ xch_mw_of(dw)) << 2),
                (uint16_t)(ma[kk][2 * v] >> (16 * el)));
        });
      });
    });
  };
  //: THE RECLAIM, out of this warp's OWN segment in WRITE order -- the same
  //: `ldmatrix.trans` shape as the read pull, rejoined with the `lo` the warp kept, so
  //: the round trip through the exchange is lossless by construction.
  auto reload_hi = [&](auto& lo, auto& mlo) {
    const int rowT = strm * kLeafE + (lane & 7) + 8 * (lane >> 4);
    const int rowM = strm * kLeafE + 2 * (lane & 3);
    const uint32_t xpullT = xchB + (uint32_t)(rowT * LdX * 2);
    const uint32_t xpullM = xchB + (uint32_t)((rowM * LdX + kDv) * 2);
    const int swzT = xch_swz_of(rowT), mwM = xch_mw_of(rowM);
    const int gT = (lane >> 3) & 1;
    static_for<KDw>([&](auto Kc) {
      constexpr int kk = decltype(Kc)::value;
      constexpr uint32_t off = (uint32_t)(kk * kMmaK * LdX * 2);
      constexpr int swz_d = xch_swz_of(kk * kMmaK);
      const int swz = swzT ^ swz_d;
      static_for<MCH>([&](auto Mc) {
        constexpr int m = decltype(Mc)::value;
        uint32_t a[4];
        load_frag_t(a, xpullT + off + ((uint32_t)(((2 * m) | gT) ^ swz) << 4));
        static_for<4>([&](auto Rc) {
          constexpr int r = decltype(Rc)::value;
          constexpr int j = 2 * kk + (r >> 1);
          st[m][j][2 * (r & 1) + 0] = join_hi_lo(a[r], lo[m][j][r & 1], 0);
          st[m][j][2 * (r & 1) + 1] = join_hi_lo(a[r], lo[m][j][r & 1], 1);
        });
      });
      //: the mass rides row 0, so only the lane quad that published it reads back; every
      //: other row is identically zero and rejoins as zero.
      static_for<2>([&](auto Vc) {
        constexpr int v = decltype(Vc)::value;
        constexpr int j = 2 * kk + v;
        constexpr int dv0 = kk * kMmaK + 8 * v;
        const uint32_t a0 = xpullM + (uint32_t)(dv0 * LdX * 2);
        const int mw0 = mwM ^ xch_mw_of(dv0), mw1 = mwM ^ xch_mw_of(dv0 + 1);
        const uint32_t mh =
            ((lane >> 2) == 0)
                ? ((uint32_t)load_shared_u16(a0 + ((uint32_t)mw0 << 2))
                   | ((uint32_t)load_shared_u16(a0 + (uint32_t)(LdX * 2) + ((uint32_t)mw1 << 2))
                      << 16))
                : 0u;
        mass_st[j][0] = join_hi_lo(mh, mlo[j], 0);
        mass_st[j][1] = join_hi_lo(mh, mlo[j], 1);
        mass_st[j][2] = 0.f;
        mass_st[j][3] = 0.f;
      });
    });
  };

  //: THE TIED CHANGE BACK, READOUT -> FOLD: `reload_hi`'s rejoin, with the A fragments
  //: read from registers rather than a segment.  BODY COUNT: `KDw * (8*MCH + 8)`.
  auto join_hi = [&](auto& sa, auto& ma, auto& lo, auto& mlo) {
    static_for<KDw>([&](auto Kc) {
      constexpr int kk = decltype(Kc)::value;
      static_for<MCH>([&](auto Mc) {
        constexpr int m = decltype(Mc)::value;
        static_for<4>([&](auto Rc) {
          constexpr int r = decltype(Rc)::value;
          constexpr int j = 2 * kk + (r >> 1);
          st[m][j][2 * (r & 1) + 0] = join_hi_lo(sa[kk][m][r], lo[m][j][r & 1], 0);
          st[m][j][2 * (r & 1) + 1] = join_hi_lo(sa[kk][m][r], lo[m][j][r & 1], 1);
        });
      });
      static_for<2>([&](auto Vc) {
        constexpr int v = decltype(Vc)::value;
        constexpr int j = 2 * kk + v;
        mass_st[j][0] = join_hi_lo(ma[kk][2 * v], mlo[j], 0);
        mass_st[j][1] = join_hi_lo(ma[kk][2 * v], mlo[j], 1);
        mass_st[j][2] = 0.f;
        mass_st[j][3] = 0.f;
      });
    });
  };

  //: THE PARK (K35 S5, P76 F5), WARP-PRIVATE: a warp's exchange segment is dead the
  //: moment IT has pulled, so the `lo` it must keep across the readout goes into those
  //: very bytes -- no barrier, no arrival count, no cross-warp sequencing.  ITS LAYOUT
  //: IS NOT THE SEGMENT'S and the difference is derived: parked `lo` is `kParkWords`
  //: OPAQUE words per lane with one writer and one reader, so word `w` of lane `i` sits
  //: at element `2*(32*w + i)`: a burst is bank `i` for lane `i`, max-way ONE, no
  //: swizzle (carry_kernel.md, "The park").  BODY COUNT `kParkWords = NL*(2*MCH + 1)`:
  //: 36 / 18 / 34 at the three shapes, O(fragments); that many `st.shared.u32` and as
  //: many `ld.shared.u32` per lane PER WINDOW, NOTHING per token.
  constexpr int kParkWords = G::kParkWords;
  auto park_visit = [&](auto& lo, auto& mlo, auto move) {
    const uint32_t p = xchB + (uint32_t)((strm * kLeafE * LdX + 2 * lane) * 2);
    static_for<MCH>([&](auto Mc) {
      constexpr int m = decltype(Mc)::value;
      static_for<NL>([&](auto Jc) {
        constexpr int j = decltype(Jc)::value;
        static_for<2>([&](auto Hc) {
          constexpr int h = decltype(Hc)::value;
          move(p + (uint32_t)((2 * (m * NL + j) + h) * kBallotLanes * 4), lo[m][j][h]);
        });
      });
    });
    static_for<NL>([&](auto Jc) {
      constexpr int j = decltype(Jc)::value;
      move(p + (uint32_t)((kParkWords - NL + j) * kBallotLanes * 4), mlo[j]);
    });
  };
  auto park_lo = [&](auto& lo, auto& mlo) {
    park_visit(lo, mlo, [](uint32_t a, uint32_t& x) { store_shared_u32(a, x); });
  };
  auto unpark_lo = [&](auto& lo, auto& mlo) {
    park_visit(lo, mlo, [](uint32_t a, uint32_t& x) { x = load_shared_u32(a); });
  };

  //: THE ONES OPERAND, AN IMMEDIATE: the mass channel's value is literally 1, so
  //: the fold's A fragment for it is the constant whose row 0 is ones.
  constexpr uint32_t kOnesPair = 0x3F803F80u;
  uint32_t ones_a[4];
  ones_a[0] = (lane >> 2) == 0 ? kOnesPair : 0u;
  ones_a[1] = 0u;
  ones_a[2] = ones_a[0];
  ones_a[3] = 0u;

  // ---- the state plane's page sweep, in LATTICE leaf order -----------------
  constexpr int kAtoms = BC / kAtomLeaves;
  constexpr int kPageElems = kAtomLeaves * kCols;
  //: THE SWEEP'S STEP, AS A CARRY.  A step of `kThreads` elements advances the row
  //: cursor by `kRowStep` or by one more, so the walk is a compile-time count of
  //: predicated subtractions and never a division -- at any executor count AND at any
  //: value width.  `kRowStep` is ZERO where a row is wider than the CTA (`d_v = 128`
  //: at four warps): the step then advances the cursor by one row or none, which the
  //: same two statements express because the carry is below `kCols` by construction.
  constexpr int kRowStep = kThreads / kCols;
  constexpr int kPageCarry = kThreads - kRowStep * kCols;
  static_assert(kPageCarry >= 0 && kPageCarry < kCols,
                "a sweep step's carry is the thread count modulo the row, so the cursor "
                "crosses at most one row boundary per step");
  auto page_sweep = [&](uint32_t pageOff, auto&& body) {
    int e = tid;
    int row = 0, col = tid;
    static_for<kRowStep>([&](auto) {
      const bool over = col >= kCols;
      row += over ? 1 : 0;
      col -= over ? kCols : 0;
    });
#pragma unroll 1
    while (e < kPageElems) {
      body(row * LdO + col, pageOff + (uint32_t)(e * (int)sizeof(float)));
      e += kThreads;
      col += kPageCarry;
      const bool wrap = col >= kCols;
      row += wrap ? kRowStep + 1 : kRowStep;
      col -= wrap ? kCols : 0;
    }
  };
  //: `lambda = owner*BC + local` (LATTICE-1), so the owner's atoms are
  //: `owner*(BC/16) + i`.  What the shipped kernel proves with `base_aligned` is a
  //: DEFINITION here (R0 spec section 2.6).
  const uint32_t atomBase = (uint32_t)(bh * (int)(BP::kLeaves / kAtomLeaves) + owner * kAtoms);
  auto page_off = [](uint32_t atom) { return atom * (uint32_t)(kPageElems * (int)sizeof(float)); };
  //: THE PAGE'S SLICE OF A STREAM, PER FRAGMENT.  Register-resident state exists only
  //: as fragments (KERNEL_STANDARDS section 1): the accumulator is `NL` n-tiles x
  //: `MCH` m-tiles of the m16n8 shape, and a lane's fragment holds TWO ADJACENT
  //: private slots at two channel rows.  So this nest iterates FRAGMENT MODES --
  //: n-tile, the pair's half, m-tile, the row half -- and the leaf map is folded into
  //: the ADDRESS: the private slot is a lane expression, its atom and its row inside
  //: the page come from the map, and every accumulator index is a constant.
  //: BODY COUNT: `NL * 2 * (2*MCH + 1)` per atom -- 72 at the `d_v = 64` arms (NL = 4
  //: or 2 against MCH = 4 or 8... precisely: 36 at 16 leaves, 72 at 32), 68 at the
  //: `d_v = 128` arms (NL = 2, MCH = 8) -- times `kAtoms` per call (4 at BC=64, 8 at
  //: BC=128, 16 at BC=256).  It is O(fragments) in BOTH modes, which is what keeps the
  //: width from reaching the nest.  The per-LEAF form this replaces
  //: was `kLeafW * (2*MCH + 1)` per atom with three quarters of the lanes predicated
  //: off -- 4,096 bodies at the eight-warp arm, and 50 minutes of cicc.
  auto page_regs = [&](auto Ac, auto Ld) {
    constexpr int a = decltype(Ac)::value;
    constexpr bool LOAD = decltype(Ld)::value;
    //: THE STATE PLANE IS WALKED UNDER THE WRITE CARVE, because that is the order the
    //: accumulator is held in.
    static_for<NL>([&](auto Jc) {
      constexpr int j = decltype(Jc)::value;
      static_for<2>([&](auto Lc) {
        constexpr int el = decltype(Lc)::value;
        const int lf = leafW_of(j * kMmaN + 2 * (lane & 3) + el);
        if ((lf >> ilog2c(kAtomLeaves)) != a) return;
        const int row = lf & (kAtomLeaves - 1);
        static_for<MCH>([&](auto Mc) {
          constexpr int m = decltype(Mc)::value;
          static_for<2>([&](auto Hc) {
            constexpr int h = decltype(Hc)::value;
            float* const cell = &so[row * LdO + m * kMmaM + (lane >> 2) + 8 * h];
            if constexpr (LOAD)
              st[m][j][el + 2 * h] = *cell;
            else
              store_f32(cell, st[m][j][el + 2 * h]);
          });
        });
        //: the mass rides row 0 of its own accumulator, so only the lane quad holding
        //: that row carries it.
        if ((lane >> 2) == 0) {
          float* const cell = &so[row * LdO + kDv];
          if constexpr (LOAD)
            mass_st[j][el] = *cell;
          else
            store_f32(cell, mass_st[j][el]);
        }
      });
    });
  };

  //: ---- THE ENTRY STATE.  The fragments are SET, never accumulated: they are
  //: provably zero here, and with no decay the recurrence is affine in the state
  //: with an inclusive-after-write readout, so this IS the oracle's seeding.
  if constexpr (STATE) {
    if (s_in != nullptr) {
      const float* const plane = pin_address(s_in);
      static_for<kAtoms>([&](auto Ac) {
        page_sweep(page_off(atomBase + (uint32_t)decltype(Ac)::value), [&](int sidx, uint32_t off) {
          so[sidx] = *reinterpret_cast<const float*>(gather_ptr(plane, off));
        });
        rendezvous();
        page_regs(Ac, std::true_type{});
        rendezvous();
      });
    }
  }

  // ---- THE COOPERATIVE FETCH'S SHARED PIECES (K35 S2 on the write side, S4b on the
  //      read side) --------------------------------------------------------
  //: THE TWO UNION LISTS, one per side, and they are the ONLY lists: neither side keeps
  //: a per-stream list any more.
  //: `kWarps` READ LISTS, ONE PER READER, then the WRITE side's union list: a reader
  //: walks its own ballot and nobody else reads what it wrote.
  uint16_t* const rlistW = slist + strm * kListStride;
  uint16_t* const wlistS = slist + kWarps * kListStride;

  //: ONE AMPLITUDE OUT OF A STAGED ROW, FOR EITHER SIDE.  The level lying LOWEST in the
  //: asking side's private order (`kPairLevel`) supplies BOTH slots of a packed pair
  //: with one four-byte access, because its two digits are adjacent there; every other
  //: level's digit is constant across the pair and is splatted.  The side is the type
  //: argument, so the two expansions share this and differ only in their orders.
  auto factor_at = [&](auto Map, uint32_t rowB, auto Lc, int p0, int off) {
    using PMx = typename decltype(Map)::type;
    constexpr int l = decltype(Lc)::value;
    const int d = ((p0 >> PMx::shift(l)) & (PMx::span(l) - 1)) + off;
    if constexpr (l == PMx::kPairLevel)
      return load_shared_u32(rowB + (uint32_t)((BP::stage_off(l) + d) * 2));
    else
      return splat_bf16x2_bits(load_shared_u16(rowB + (uint32_t)((BP::stage_off(l) + d) * 2)));
  };
  //: the digit offset of a stream this executor does not run: the stream index
  //: decomposes in mixed radix over the per-level stream grids, so it is a shift and a
  //: mask over compile-time powers of two.
  auto stream_off = [&](auto Vec, int s, auto Lc) {
    using SV = typename decltype(Vec)::type;
    constexpr int l = decltype(Lc)::value;
    return ((s >> ilog2c(SV::grid_div(l))) & (SV::g(l) - 1)) * SV::s(l);
  };
  //: THE ROLLING FACTOR STAGING, ONE BODY FOR BOTH SIDES AND PER WARP.  The compressed
  //: factor rows of a group's union tokens -- the OWNER's digit run at each level, which
  //: is what either side's expansion indexes with its own stream offsets -- `cp.async`ed
  //: one group AHEAD of the expansion that reads them, so no per-token global factor load
  //: ever appears in a fetch loop.
  //: THE SHARING GRAIN IS A COMPILE-TIME TAG, because the two sides fill a block
  //: differently and one body serves both.  `NW == kWarps` is the FOLD's: the block is
  //: the CTA's union group and each warp stages the rows of its own share `i % kWarps`,
  //: the shares being disjoint, which is what lets the fold's walk run with no barrier
  //: inside it at all.  `NW == 1` is the READOUT's (K35 R-B): the block is this reader's
  //: own group and the warp stages every row of it.  The LAYOUT is identical either way,
  //: so the round trip that gates it gates both.
  //: BODY COUNT: `ceil(kSegRows / (NW * kStgRows)) * D`.
  constexpr int kStgCh = ceil_pow2c(BP::kStageChunks);
  constexpr int kStgRows = kBallotLanes / kStgCh;
  static_assert(kStgCh <= kBallotLanes, "a staged row's chunks fit one lockstep unit");
  auto stage_factors = [&](auto Nw, uint32_t blk, const __nv_bfloat16* plane, const uint16_t* list,
                           uint32_t winP, int slot, int u0, int nb) {
    constexpr int NW = decltype(Nw)::value;
    constexpr int kStgOwn = kSegRows / NW;
    constexpr int kStgPass = (kStgOwn + kStgRows - 1) / kStgRows;
    static_assert(kStgOwn * NW == kSegRows, "a group's rows split evenly over its fillers");
    const int c = lane & (kStgCh - 1);
    const uint32_t blkB = blk + (uint32_t)(slot * kSegRows * BP::kStageW * 2);
    static_for<kStgPass>([&](auto Pc) {
      const int ic = (lane >> ilog2c(kStgCh)) + decltype(Pc)::value * kStgRows;
      const int r = NW == 1 ? ic : ic * NW + warp;
      if (ic < kStgOwn && r < nb) {
        const uint32_t tokB = winP + (uint32_t)list[u0 + r] * (uint32_t)(BP::kWtot * 2);
        static_for<D>([&](auto Lc) {
          constexpr int l = decltype(Lc)::value;
          if (c >= BP::stage_chunk_base(l) && c < BP::stage_chunk_base(l) + BP::stage_chunks(l)) {
            const int sub = (c - BP::stage_chunk_base(l)) * BP::stage_el(l);
            stage_run<BP::stage_el(l) * 2>(
                blkB + (uint32_t)((r * BP::kStageW + BP::stage_off(l) + sub) * 2),
                gather_ptr(plane, tokB + (uint32_t)((runCol[l] + sub) * 2)));
          }
        });
      }
    });
  };

  // ---- THE READ SIDE'S FETCH MAP AND THE FAN-IN RING (K35 S4b) -------------
  //: THE READ SIDE IS THE WRITE SIDE'S MIRROR WITH ONE DIFFERENCE OF SUBSTANCE.  On the
  //: readout the B operand is `R~^T [leaves x tokens]`: the leaves are the CONTRACTION
  //: axis and the TOKENS are `n`, in eights.  So a reader's segment carries coefficient
  //: rows and nothing else -- the A operand is the state, which is already in registers
  //: -- and the n-tile the fold skips by LEAF here is skipped by TOKEN.
  //: Each warp IS a reader stream (`SVR::kStreams == kWarps`) and OWNS its group whole
  //: (K35 R-B), so the read side resolves no candidate pair at all: a group's rows are
  //: this reader's own tokens, in this reader's own compacted order.
  constexpr int TPW = kSegRows / kWarps;  // union tokens one warp handles per group
  static_assert(TPW * kWarps == kSegRows, "a group's union tokens split evenly over the executors");
  //: AND THE WRITE SIDE'S: its `TPW` union tokens times the WRITE streams, one per
  //: lane.  At `WPS == 1` that is the whole group; under a column split the executors
  //: are more numerous and each holds fewer.
  constexpr int kCand = TPW * kStreams;
  static_assert(kCand <= kBallotLanes, "one lane resolves each candidate pair");
  constexpr int kSegBits = ilog2c(kSegRows);
  //: THE EXPANSION'S LANE MAP, the write side's at the READ carve's row width: a row is
  //: `kWordsRowR` packed pairs and the warp covers `kRowsPassR` of them at once.  A
  //: pass's rows are this reader's OWN TOKENS (K35 R-B), which is why no memo is shared
  //: across them.
  constexpr int kWordsRowR = kLeafE / 2;
  constexpr int kLanesRowR = kWordsRowR < kBallotLanes ? kWordsRowR : kBallotLanes;
  constexpr int kRowsPassR = kBallotLanes / kLanesRowR;
  constexpr int kWordsLaneR = kWordsRowR / kLanesRowR;
  //: THE READOUT'S TOKEN n-TILES PER BATCH, which is what the tile-live mask indexes.
  constexpr int kNTok = kSegRows / kMmaN;
  constexpr int kSeedR = first_whole_level<SVR, D>();
  const int exHR = lane >> ilog2c(kLanesRowR);
  const int exJR = lane & (kLanesRowR - 1);
  //: THE READER'S BASE INTO ITS OWN SEGMENT.  Its rows ARE its group's tokens in its own
  //: compacted order, so this is an offset and not a map: ONE `ldmatrix` covers the whole
  //: group -- `tl >> 1` selects the token n-tile and `cT` the leaf granule.
  //: THE S2e2 ROTATION IS GONE WITH THE RING IT EXISTED FOR: no two readers ever want
  //: one row now, so there is nothing left for a rotation to spread.
  const uint32_t myR =
      rsegB + (uint32_t)(((strm * kSegRows + (tl >> 1) * kMmaN + lr) * LdR + cT * 8) * 2);

  // ---- THE FLOW COUNTERS, ONE BODY FOR BOTH SIDES (P76 CORRECTION-2; S2c, S2d) ----
  //: STREAMS FIRE WHEN FULL, ON BOTH SIDES, AND THE MECHANISM IS WRITTEN ONCE.  Three
  //: monotone words per phase, at one shared base with compile-time displacements:
  //: `landed[s]` (rows of stream `s` that have ARRIVED), `drained[e]` at `+kWarps`
  //: (tiles executor `e` has PULLED) and `done` at `+2*kWarps` (warps that have walked
  //: their share).  A fetcher writes rank `r` of stream `s` only while
  //: `r >> log2 kSegRows <= min_e drained[e]`, and an executor pulls tile `k` when
  //: `landed[s] >= (k + 1) * kSegRows` -- so a stream issues exactly `ceil(n_s/16)`
  //: k-steps per window and NO BARRIER lives inside either walk.  The two phases are
  //: consecutive moments, so they are the SAME words; `kWarps >= kStreams`, so one
  //: zero serves both.  WHY A COUNT IS AS GOOD AS A PREFIX and the deadlock-freedom
  //: argument the code's order encodes: `carry_kernel.md`, "The flow counters".
  using WriteFlow = FlowSide<kStreams, WPS, kCand, true>;
  //: `rank_live(s, t)` -- the token's position in stream `s`'s compacted list, and
  //: whether the stream holds it at all, out of the side's published ballot and prefix.
  auto rank_live = [&](uint32_t memBase, uint32_t pfxBase, int s, int t) {
    constexpr int LW = G::kLW;
    const int w = t >> 5;
    const uint32_t word = load_shared_u32(memBase + (uint32_t)((s * LW + w) * 4));
    const int pre = (int)load_shared_u16(pfxBase + (uint32_t)((s * LW + w) * 2));
    const uint32_t below = (1u << (t & 31)) - 1u;
    return ((pre + bit_count(word & below)) << 1) | (int)((word >> (t & 31)) & 1u);
  };
  //: THE GROUP'S CANDIDATE PAIRS, RESOLVED ONCE.  Lane `q = (i, s)` owns the pair and
  //: every other lane reads it by shuffle, so a rank's two shared loads are paid once
  //: per pair.  `-1` is "this stream does not want this token"; otherwise the word is
  //: the stream's RUNNING RANK in the window over the side's payload.  THE RANK IS
  //: RUNNING and never group-local: the segment row is `rank mod kSegRows` and the TILE
  //: is `rank >> log2 kSegRows`, which is what makes a stream's rows accumulate.
  auto place_group = [&](auto Side, uint32_t memBase, uint32_t pfxBase, const uint16_t* list,
                         int u0, int nb) {
    using FS = typename decltype(Side)::type;
    int e = -1;
    if (lane < FS::kNC) {
      const int i = lane >> ilog2c(FS::kNS), s = lane & (FS::kNS - 1);
      const int ii = i * kWarps + warp;
      if (ii < nb) {
        const int t = (int)list[u0 + ii];
        const int rl = rank_live(memBase, pfxBase, s, t);
        if (rl & 1) e = ((rl >> 1) << 16) | (FS::kTokenPayload ? t : (u0 + ii));
      }
    }
    return e;
  };
  //: `drained[e]` has exactly ONE writer, its executor, so it is a plain store of a
  //: register that warp already holds; the room test takes the MINIMUM over the `kNW`
  //: executors of the stream being written into, which is ONE load wherever an executor
  //: is a whole stream (the read side always, the write side at `WPS == 1`).
  auto drained_min = [&](auto Side, int s) {
    using FS = typename decltype(Side)::type;
    uint32_t m = 0xFFFFFFFFu;
    static_for<FS::kNW>([&](auto Cc) {
      const uint32_t d =
          load_shared_u32(flowB + (uint32_t)((kWarps + s * FS::kNW + decltype(Cc)::value) * 4));
      m = d < m ? d : m;
    });
    return m;
  };
  auto publish_drained = [&](uint32_t d) {
    if (lane == 0) store_shared_u32(flowB + (uint32_t)((kWarps + strm) * 4), d);
  };
  //: THE ARRIVAL, ONE `red.shared.add.u32` PER STREAM THE GROUP TOUCHED.  A stream's
  //: candidate lanes are those congruent to `s` modulo `kNS`, so the count is a popcount
  //: of the placement's own ballot against a compile-time stride pattern: no lane
  //: iterates and nothing divides.
  auto publish_landed = [&](auto Side, int e) {
    using FS = typename decltype(Side)::type;
    constexpr uint32_t PAT = (uint32_t)(((1ull << FS::kNC) - 1ull) / ((1u << FS::kNS) - 1u));
    const uint32_t bal = lane_ballot(e >= 0);
    if (lane < FS::kNS) {
      const int c = bit_count(bal & (PAT << lane));
      if (c != 0) red_shared_add_u32(flowB + (uint32_t)(lane * 4), (uint32_t)c);
    }
  };
  auto flow_landed = [&](int s) { return load_shared_u32(flowB + (uint32_t)(s * 4)); };
  //: THE FULL-TILE TEST, an executor's whole schedule: tile `drained` is full when its
  //: stream's arrivals reach the tile's far edge, and once the phase's edge has SEALED
  //: the count the last, partial tile joins it on that fact instead of on fullness.
  auto tile_ready = [&](uint32_t l, uint32_t drained, bool sealed) {
    if (l >= ((drained + 1) << kSegBits)) return true;
    return sealed && drained < ((l + kSegRows - 1) >> kSegBits);
  };
  //: THE COUNTERS' ZERO -- they are monotone WITHIN a phase, and the barrier that opens
  //: the phase publishes them with the segment clear.
  auto flow_zero = [&]() {
    if (lane == 0) {
      store_shared_u32(flowB + (uint32_t)((kWarps + strm) * 4), 0u);
      store_shared_u32(flowB + (uint32_t)(strm * 4), 0u);
      if (strm == 0) store_shared_u32(flowB + (uint32_t)(2 * kWarps * 4), 0u);
    }
  };
  //: THE WALK'S EDGE.  A warp announces that its share is placed and keeps SERVICING
  //: until every warp has announced, because a warp parked at a barrier drains nothing
  //: and another warp may still wait on a tile this executor owns.
  auto announce_done = [&]() {
    if (lane == 0) red_shared_add_u32(flowB + (uint32_t)(2 * kWarps * 4), 1u);
  };
  auto all_done = [&]() {
    return load_shared_u32(flowB + (uint32_t)(2 * kWarps * 4)) >= (uint32_t)kWarps;
  };

  // ---- the GLOBAL window grid ---------------------------------------------
  const int nWindows = (L + W - 1) / W;

#pragma unroll 1
  for (int win = 0; win < nWindows; ++win) {
    const int tok0 = win * W;
    const int wLen = (L - tok0) < W ? (L - tok0) : W;
    const uint32_t winP = (uint32_t)tok0 * (uint32_t)(BP::kWtot * 2);
    const uint32_t winG = (uint32_t)tok0 * 2u;
    const uint32_t winV = (uint32_t)tok0 * (uint32_t)v_row_bytes;
    const uint32_t winO = (uint32_t)tok0 * (uint32_t)(kDv * (int)sizeof(float));
    const uint32_t winD = (uint32_t)tok0 * 4u;
    // ---- the liveness pass: one READ list per warp, ONE write list --------
    //: THE SCHEDULE IS TWO-SIDED (Blake ruling 2026-08-19).  Per level at most ONE
    //: side is sparse, so there is one stored row set per level -- but the two
    //: sides' CLAUSE SETS are different level subsets, and a dense side's clause is
    //: absent-means-full.  ANDing each side's OWN rows IS that conjunction over
    //: exactly the levels where that side is sparse, with no structure fact carried
    //: into the kernel.  At a level a side CARVES, that side's clause is its own
    //: STREAM's range -- the OR of the table rows the range covers -- and the owner's
    //: run is the OR over the whole run, which the shared derivations want.
    //:
    //: ONE BALLOT WORD PER LANE.  A warp's whole window is `LW` words, so a row is
    //: ONE `LDG.32` and every warp can afford to hold BOTH sides' owner-grain
    //: ballots.  That is what makes the schedule barrier-free: the lists, the
    //: counts and the whole-window skip are all warp-private computations of
    //: CTA-identical inputs, so the warps agree by construction and reach the
    //: window line without a barrier of the schedule's own.
    constexpr int LW = G::kLW;
    static_assert(W <= 0xFFFF, "a token's window-local index must fit the list's element");
    static_assert(LW <= kBallotLanes,
                  "the walk gives one ballot word to each lane, so a window is at most a "
                  "lockstep unit's worth of words");
    const int room = wLen - lane * kBallotLanes;
    const bool inW = lane < LW;
    const uint32_t valid =
        (!inW || room <= 0) ? 0u : (room >= kBallotLanes ? 0xFFFFFFFFu : ((1u << room) - 1u));
    //: a lane past the window's last word holds no word; it reads word zero, which
    //: is in range, and `valid` empties its ballot.
    const int wIdx = (tok0 >> 5) + (inW ? lane : 0);
    const uint32_t wordB = (uint32_t)(wIdx < tok_words ? wIdx : tok_words - 1) * 4u;

    //: The scatter is PER-LANE and divergent by choice, and its store is ADDRESSED
    //: rather than indexed; both are measured, and the word-major alternative is
    //: refused with numbers -- see docs/internals/carry/carry_kernel.md, "The
    //: schedule scatter".  `bits` is the writer's slice of each word and `below` the
    //: bits under it: one list with several writers, each deriving the same prefix.
    auto walk = [&](uint32_t m, uint16_t* dst, uint32_t bits, uint32_t below) {
      const int cnt = bit_count(m);
      const int pfx = lane_prefix_incl<LW>(cnt);
      const int k = (pfx - cnt) + bit_count(m & below);
      const uint16_t tb = (uint16_t)(lane * kBallotLanes);
      SmemAddr a = smem_addr(dst) + (SmemAddr)(k * (int)sizeof(uint16_t));
      uint32_t mm = m & bits;
      while (mm) {
        store_shared_u16(a, (uint16_t)(tb + (uint16_t)first_set(mm)));
        a += (SmemAddr)sizeof(uint16_t);
        mm &= mm - 1;
      }
      return lane_broadcast(pfx, LW - 1);
    };
    //: THE CONJUNCTION, ONE STATEMENT FOR EVERY LEVEL -- and EXACT at any carve,
    //: because a rectangle of digits holds a live leaf iff every level's range holds
    //: a live digit.  A digit RANGE's word is the OR of the table rows it covers
    //: (`RANGE`, in table rows), and the levels are ANDed.  Two ranges are asked for:
    //: the STREAM's own (`MINE`, its ownership test) and the OWNER's whole run (the
    //: shared derivation and the whole-window skip).  It is PRUNED BY THE
    //: DECLARATION: a side conjoins a level only where that side is declared sparse,
    //: because a dense side's row is all-ones and ANDing it is a load for nothing.
    auto ballot_of = [&](const uint32_t* rowsS, auto Side, auto Vec, auto Mine, const int* off,
                         const uint32_t* rowOff) {
      constexpr uint32_t SIDE = decltype(Side)::value;
      using SV = typename decltype(Vec)::type;
      using CRS = CarveRows<SV>;
      constexpr bool MINE = decltype(Mine)::value;
      uint32_t m = valid;
      static_for<D>([&](auto Lc) {
        constexpr int l = decltype(Lc)::value;
        if constexpr (CRS::row_g(l) > 0 && mode_declares(MO, l, SIDE)) {
          //: THE STREAM'S OWN RANGE IS EXACTLY ONE ROW, because the table is at this
          //: side's own grain; only the OWNER's whole run is an OR, over `g_l` rows.
          constexpr int RANGE = MINE ? 1 : SV::g(l);
          const uint32_t base =
              rowOff[l] + (MINE ? (uint32_t)(off[l] >> ilog2c(CRS::s(l))) * rowBytes : 0u);
          uint32_t r = 0u;
          static_for<RANGE>([&](auto Ic) {
            r |= *reinterpret_cast<const uint32_t*>(
                gather_ptr(rowsS, base + (uint32_t)decltype(Ic)::value * rowBytes + wordB));
          });
          m &= r;
        }
      });
      return m;
    };
    constexpr std::integral_constant<uint32_t, kSideRead> kRead{};
    constexpr std::integral_constant<uint32_t, kSideWrite> kWrite{};
    //: the OWNER-GRAIN ballots, which the cooperative walks are derived from, and the
    //: STREAM-GRAIN ballots, which each executor's pull tests its ids against.
    //: THE STREAM-GRAIN BALLOTS ARE GENUINELY PRIVATE and stay per warp: each is ONE
    //: table row per tested level, read at THIS stream's own digit offset, so no two
    //: warps compute the same word and there is nothing here to share.
    const uint32_t mineR =
        ballot_of(rowsR, kRead, type_tag<SVR>{}, std::true_type{}, offR, rowOffR);
    const uint32_t mineW =
        ballot_of(rowsW, kWrite, type_tag<SVW>{}, std::true_type{}, offW, rowOffW);

    //: THE READ SIDE WALKS PER READER AND NOT COOPERATIVELY (K35 R-B).  The shared walk
    //: was a per-cell TRADE and never a win -- where a token reads leaves in several
    //: boxes the owner's union is nearly the sum of the per-reader lists, so it saved
    //: nothing and the pull it forced was added work.  Its walk is PHASE 1's, below,
    //: against this warp's own `mineR`.
    //: THE OWNER'S WRITE IDS, DERIVED ONCE FOR THE CTA.  `walk` is already the cheap
    //: form -- one lane per ballot word, `ffs` per live bit -- and its `bits`/`below`
    //: parameters exist for exactly this: several writers deriving the SAME global
    //: prefix and filling disjoint slices of every word, so the list comes out in the
    //: one ascending order every consumer expects.  The CTA wall is here and nowhere
    //: else in the fold.
    //: ONE WARP DERIVES THE UNION; EVERY WARP STILL WALKS ITS SHARE (K35 R-C item 1).
    //: The OWNER-GRAIN ballots are functions of the OWNER and the window alone --
    //: `ballot_of` at `MINE = false` never reads a stream offset -- so all `kWarps` warps
    //: used to issue the same `sum_l g_l` table loads per side and OR them to the same
    //: words: at the eight-warp ALT arm, 96 redundant loads a lane a window against 12
    //: needed ones.  Warp zero votes both sides ONCE and publishes the write-side word
    //: per lane plus the whole-window SKIP; every warp then reads its own lane's word.
    //: DERIVATION COUNT PER (CTA, WINDOW) IS EXACTLY ONE.
    //:
    //: THE WALK STAYS SLICED, AND THAT IS A MEASURED CORRECTION.  R-C first put the walk
    //: on warp zero too, so that the dedup would need no barrier of its own; that cost
    //: +136% at the record cell, where one warp issuing every list store while three wait
    //: is far dearer than the loads it saved.  The walk is `nw` stores of REAL work and
    //: parallelizes perfectly; the derivation is redundant work and does not.  Only the
    //: latter is shared, and the extra barrier below is what that costs.
    if (warp == 0) {
      const uint32_t wideR =
          ballot_of(rowsR, kRead, type_tag<SVR>{}, std::false_type{}, offR, rowOffR);
      const uint32_t wideW =
          ballot_of(rowsW, kWrite, type_tag<SVW>{}, std::false_type{}, offW, rowOffW);
      if (inW) store_shared_u32(schB + (uint32_t)(lane * 4), wideW);
      //: THE WHOLE-WINDOW SKIP, at (owner, window) grain, per side -- the block-bitmap
      //: mechanism's harvest (study section 2.1).  It is a CTA fact, so it is voted and
      //: published as one: a dead window costs ONE warp's derivation, not every warp's.
      const uint32_t any = lane_ballot((wideR | wideW) != 0u);
      if (lane == 0) store_shared_u32(schB + (uint32_t)(LW * 4), any == 0u ? 1u : 0u);
    }
    rendezvous();
    //: CTA-UNIFORM BY CONSTRUCTION: every warp reads the one word warp zero wrote, so the
    //: `continue` is taken by all of them or by none and the window's remaining barriers
    //: are reached by the whole CTA or by no part of it.
    if (load_shared_u32(schB + (uint32_t)(LW * 4)) != 0u) continue;
    //: a lane past the window's last word holds no word; warp zero stored none there and
    //: an empty ballot contributes a zero count to the walk's prefix.
    const uint32_t wideW = inW ? load_shared_u32(schB + (uint32_t)(lane * 4)) : 0u;
    constexpr int kSliceBits = kBallotLanes / kWarps;
    const int nw =
        walk(wideW, wlistS, (uint32_t)(((1ull << kSliceBits) - 1ull) << (warp * kSliceBits)),
             (uint32_t)((1ull << (warp * kSliceBits)) - 1ull));
    rendezvous();
    //: THE WRITE SIDE'S SCHEDULE IS THE SHARED UNION LIST built above, and it is the
    //: ONLY write-side list there is: under the cooperative fetch a warp does not walk
    //: its own tokens, it takes a fixed share of the UNION's and places rows into the
    //: segments of whichever streams want them.  So each executor PUBLISHES its own
    //: membership -- one ballot word per lane -- together with the per-word PREFIX
    //: POPCOUNT, and those two turn a (token, stream) pair into the RANK the placement
    //: addresses.  `mineW` is already `valid`-masked, so a lane past the window's last
    //: word contributes a zero count and the prefix is exact at any `LW`.
    //: PUBLISHED ONCE PER WRITE STREAM, by its HALF-0 warp: the `WPS` warps of a stream
    //: hold the SAME `mineW` (they differ only in which leaf columns they fold), so the
    //: other halves would restate the same words.
    if (inW && col == 0) store_shared_u32(memB + (uint32_t)((wstrm * LW + lane) * 4), mineW);
    {
      const int cnt = bit_count(mineW);
      const int pfx = lane_prefix_incl<LW>(cnt) - cnt;
      if (inW && col == 0)
        store_shared_u16(pfxB + (uint32_t)((wstrm * LW + lane) * 2), (uint16_t)pfx);
    }

    //: THE PHASE CHANGE, FOLD -> READOUT.  Every warp publishes its write-view hi into
    //: the READ consumers' segments; a barrier; every warp pulls its OWN segment into
    //: its A fragments and the segments are DEAD.  `lo` is packed INSIDE the publish,
    //: n-tile by n-tile, so the fp32 accumulator never has to live whole beside it, and
    //: it spends the readout either in REGISTERS (`PK == 0`, half the fp32 footprint)
    //: or in the warp's own dead segment (`PK == 1`, the park).
    //: `mlo` IS ONE WORD PER n-TILE, not two.  The mass accumulator's elements 2 and 3
    //: are the C rows 8..15 of a tile whose A operand (`ones_a`) is nonzero on row 0
    //: alone, so they are IDENTICALLY ZERO -- there is no low half of them to keep.
    uint32_t lo[MCH][NL][2], mlo[NL];
    uint32_t sa[KDw][MCH][4], ma[KDw][4];
    if constexpr (G::kTied) {
      split_hi(lo, mlo, sa, ma);
    } else {
      publish_read(lo, mlo);
      rendezvous();
      pull_read(sa, ma);
    }
    //: AND THE SEGMENTS ARE DEAD HERE -- for THIS warp, the moment it has pulled.
    //: At `PK == 0` the readout's tenants are these same bytes under another layout, so
    //: no warp may produce into them until EVERY warp has pulled: that barrier is the
    //: price of the straddle.  At `PK == 1` the park holds the bytes, the readout's
    //: tenants sit above them, and the park is warp-private -- so the barrier is spent
    //: at the far end instead.
    //: A TIED ARM STILL PAYS ONE BARRIER HERE, the STRADDLE's and not the exchange's:
    //: the readout's tenants are the fold's very bytes.  At `PK == 1` the untied path
    //: spends that barrier inside the publish/pull pair instead.
    if constexpr (G::kTied) {
      rendezvous();
      if constexpr (G::kPark) park_lo(lo, mlo);
    } else if constexpr (G::kPark) {
      park_lo(lo, mlo);
    } else {
      rendezvous();
    }

    // ---- PHASE 1: THE READOUT, WARP-PRIVATE, against the window's ENTRY state.
    //      ALL of the window's readouts precede ANY of its folds, because a
    //      within-window pair is the INTRA term's and this kernel must not carry it.
    //      The window line is therefore the phase line on both sides.
    //: THE READ SIDE SHARES NOTHING (P76 "THE RESET", R-B; M1a `3dd4688` restored).  A
    //: READER IS A WARP: it walks its OWN ballot into its OWN list, stages and expands
    //: its OWN tokens into its OWN segment, and issues its partial `y` as a NATIVE
    //: GLOBAL REDUCTION straight out of the accumulator fragments.  No batch line, no
    //: rendezvous, no meeting place, and so no barrier of its own -- the only
    //: synchronisation in the whole phase is the `__syncwarp` between a lane's `STS` and
    //: the `ldmatrix` that reads another lane's row.
    //: THE SUMMATION ORDER IS THE ARRIVAL ORDER at the output plane, which is what the
    //: fan-in ring was built to control and what this design forfeits BY RULING: the
    //: oracle band is the gate on the readout's arithmetic, not bit-replay.
    const int nrM = walk(mineR, rlistW, 0xFFFFFFFFu, 0u);
    __syncwarp();
    const int nbrM = (nrM + kSegRows - 1) / kSegRows;
    //: THE GROUP IS A FETCH QUANTUM AND NOTHING MORE -- one staging block, one lane map
    //: -- so a reader issues exactly `ceil(n_reader / kSegRows)` k-steps per window and
    //: only its LAST group can be short.
    auto rgroup_n = [&](int b) {
      const int r = nrM - b * kSegRows;
      return r <= 0 ? 0 : (r < kSegRows ? r : kSegRows);
    };
    //: THE EXPANSION, WARP-PRIVATE.  DECLARED ARITHMETIC ORDER, per packed slot pair:
    //:     x f_l for every level the READ side SPANS WHOLE, ascending `l`,
    //:     then x f_l for every level it CARVES,            ascending `l`,
    //: each product rounded to nearest even in bf16.  THERE IS NO MEMO (K35 R-B): the
    //: memo existed because one token's rows were expanded for SEVERAL readers, and a
    //: reader now expands only its own.  THERE IS NO GAIN on this side -- its scalar is
    //: identically one -- so the first factor of the order IS the product's seed and no
    //: multiply by one is ever issued.
    //: A DEAD ROW IS ZEROED BY THE STORE THAT WRITES IT (the fold's law, S2b): the live
    //: rows of a reader's last group are the prefix `[0, n)` and the rest store the
    //: zero the MMA needs, so no clearing pass exists on this side at all.
    //: BODY COUNT: `kPassR * kWordsLaneR * (D + 1)`.
    constexpr int kPassR = kSegRows / kRowsPassR;
    static_assert(kPassR * kRowsPassR == kSegRows,
                  "a group's rows split evenly over the expansion's lane map");
    auto expand_read = [&](int slot, int n) {
      const uint32_t blkB = myStg + (uint32_t)(slot * kSegRows * BP::kStageW * 2);
      static_for<kPassR>([&](auto Pc) {
        const int row = decltype(Pc)::value * kRowsPassR + exHR;
        const uint32_t rowB = blkB + (uint32_t)(row * BP::kStageW * 2);
        const uint32_t rowR = rsegB + (uint32_t)(((strm * kSegRows + row) * LdR) * 2);
        static_for<kWordsLaneR>([&](auto Wc) {
          constexpr int wl = decltype(Wc)::value;
          const int jw = exJR + wl * kLanesRowR;
          uint32_t a = kOnesPair;
          static_for<D>([&](auto Lc) {
            constexpr int l = decltype(Lc)::value;
            if constexpr (!SVR::carves(l)) {
              const uint32_t f = factor_at(type_tag<PM>{}, rowB, Lc, 2 * jw, 0);
              if constexpr (l == kSeedR)
                a = f;
              else
                a = mul_bf16x2(a, f);
            }
          });
          static_for<D>([&](auto Lc) {
            constexpr int l = decltype(Lc)::value;
            if constexpr (SVR::carves(l))
              a = mul_bf16x2(a, factor_at(type_tag<PM>{}, rowB, Lc, 2 * jw,
                                          stream_off(type_tag<SVR>{}, strm, Lc)));
          });
          store_shared_u32(rowR + (uint32_t)(jw * 4), row < n ? a : 0u);
        });
      });
    };
    //: THE FAN-IN, THROUGH THE 32-CHANNEL PANEL (K35 R-C).  The accumulator is
    //: transposed a CHANNEL BLOCK at a time and the reduction is then issued
    //: LANE-PER-CHANNEL, straight into the output plane: no batch, no shared ring, no
    //: rendezvous -- `red.global.add.f32` is fire-and-forget, so a warp's contribution
    //: leaves the SM in the instruction that produces it and no protocol survives it.
    //:
    //: WHY A PANEL AND NOT A REGISTER PACK -- THE MEASUREMENT (R-C).  The transpose the
    //: reduction needs IS a lane permutation, so it looks like it belongs in registers,
    //: and one `__shfl_sync` per accumulator register does perform it: the C fragment's
    //: lane `4a + b` holds channel `m*kMmaM + a + 8h` at token `2b + eb`, and lane `d`
    //: wants what lane `4*(d & 7) + (d >> 3)` holds.  That form was BUILT AND MEASURED.
    //: It reaches four FULL 32-byte sectors per reduction -- RED count identical to this
    //: one's, 6,684,672 at the dense cell, sectors 3.82 per request -- and it is 93%
    //: SLOWER at that cell on 2.7% more instructions.  The reason is that its four
    //: sectors lie in four DIFFERENT TOKEN ROWS, i.e. four distinct 128-byte segments
    //: that the coalescer cannot merge, where the panel's four lie in ONE.  SECTOR
    //: FULLNESS IS NOT THE FIGURE OF MERIT; SEGMENT COUNT IS.  `e390764` failed the same
    //: test from the other side (half-filled sectors, +9.5% at k16 on -5.8%
    //: instructions), so both register-sourced forms are refuted by measurement and the
    //: staged form stands -- at a THIRD of its former bytes.
    //:
    //: THE MAPS ARE THE FRAGMENT'S OWN.  In channel block `blk` the panel holds channels
    //: `32*blk .. 32*blk + 31`; element `(m, e)` of a token n-tile carries token
    //: `2*(lane & 3) + (e & 1)` at channel `m*kMmaM + (lane >> 2) + 8*(e >> 1)`, so the
    //: two m-tiles of a block store at column `mm*kMmaM + 8*(e >> 1) + (lane >> 2)`.
    //: The load's is `[row][lane]`, and the two meet nowhere else.  It is MCH-generic by
    //: construction and no `d_v` is tested anywhere in it.
    //: THE MASS rides the panel's PAD column: its accumulator's A operand (`ones_a`) is
    //: nonzero on row 0 alone, so only the lanes holding channel 0 store it, and it goes
    //: out as one reduction per token n-tile against the `den` plane -- which is its own
    //: `[BH, L]` plane, so its addresses are scattered whatever the layout.
    //: BODY COUNT: `kChBlk * (2 * (2*2 + [blk == 0]) stores + kMmaN reductions)` per
    //: token n-tile -- 16 stores and 16 reductions at `d_v = 64`, and the reductions ARE
    //: the contribution count, `kDv + 1` values per (token, reader), with no other term.
    auto fan_in = [&](auto& acc, auto& macc, int u0, auto Pc, int n) {
      constexpr int p = decltype(Pc)::value;
      static_for<kChBlk>([&](auto Bc) {
        constexpr int blk = decltype(Bc)::value;
        //: THE BLOCK'S TRANSPOSE: the two channel m-tiles this block covers.
        static_for<2>([&](auto Mm) {
          constexpr int mm = decltype(Mm)::value;
          constexpr int m = 2 * blk + mm;
          static_for<2>([&](auto Ec) {
            constexpr int eb = decltype(Ec)::value;
            const int prow = ((lane & 3) * 2 + eb) * G::LdB;
            static_for<2>([&](auto Hc) {
              constexpr int h = decltype(Hc)::value;
              store_f32(&sow[prow + mm * kMmaM + 8 * h + (lane >> 2)], acc[m][2 * h + eb]);
            });
          });
        });
        if constexpr (blk == 0) {
          static_for<2>([&](auto Ec) {
            constexpr int eb = decltype(Ec)::value;
            if ((lane >> 2) == 0) store_f32(&sow[((lane & 3) * 2 + eb) * G::LdB + 32], macc[eb]);
          });
        }
        __syncwarp();
        //: ONE REDUCTION PER TOKEN: 32 lanes over the block's 32 CONSECUTIVE channels,
        //: one aligned 128-byte transaction into that token's row of the output plane.
        static_for<kMmaN>([&](auto Rc) {
          constexpr int r = decltype(Rc)::value;
          const int row = p * kMmaN + r;
          if (row < n)
            fan_in_add(reinterpret_cast<float*>(
                           gather_ptr(oPlane, (uint32_t)rlistW[u0 + row] * (uint32_t)(kDv * 4)
                                                  + winO + (uint32_t)((blk * 32 + lane) * 4))),
                       sow[r * G::LdB + lane]);
        });
        //: the panel is re-filled by the next block, so its rows are released here.
        __syncwarp();
      });
      if (lane < kMmaN && p * kMmaN + lane < n)
        fan_in_add(reinterpret_cast<float*>(
                       gather_ptr(denPlane, (uint32_t)rlistW[u0 + p * kMmaN + lane] * 4u + winD)),
                   sow[lane * G::LdB + 32]);
    };
    //: ONE GROUP'S READOUT: `ldmatrix` the group's coefficient rows into B fragments --
    //: ONE instruction covers the whole group, `tl >> 1` naming the token n-tile and
    //: `cT` the leaf granule -- and then one MMA block per n-tile against the A
    //: fragments the exchange left in registers, each ending in its own fan-in.
    //: THE `__syncwarp` AFTER THE PULL IS THE SEGMENT'S RELEASE: the rows are in
    //: registers, so the next group may expand into them (F4, the second stage in
    //: registers).  It is also what orders this warp's `LDSM` against its own next `STS`.
    //: BODY COUNT: `KDw` `ldmatrix` and `kNTok * KDw * (MCH + 1)` MMAs per group.
    auto readout_group = [&](int u0, int n) {
      uint32_t rb[KDw][4];
      static_for<KDw>([&](auto Kc) {
        constexpr int kk = decltype(Kc)::value;
        load_frag(rb[kk], ADDR_R(myR, 0, 2 * kk));
      });
      __syncwarp();
      static_for<kNTok>([&](auto Pc) {
        constexpr int p = decltype(Pc)::value;
        float acc[MCH][4], macc[4];
        static_for<MCH>([&](auto Mc) {
          constexpr int m = decltype(Mc)::value;
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[m][e] = 0.f;
        });
#pragma unroll
        for (int e = 0; e < 4; ++e) macc[e] = 0.f;
        static_for<KDw>([&](auto Kc) {
          constexpr int kk = decltype(Kc)::value;
          static_for<MCH>([&](auto Mc) {
            constexpr int m = decltype(Mc)::value;
            mma(acc[m], sa[kk][m], &rb[kk][2 * p]);
          });
          mma(macc, ma[kk], &rb[kk][2 * p]);
        });
        fan_in(acc, macc, u0, Pc, n);
      });
    };
    //: THE WALK, AND IT IS A STRAIGHT LOOP.  Stage the first group, then per group:
    //: wait for its factors, expand them, ISSUE THE NEXT GROUP'S `cp.async`, and read
    //: out.  The issue sits between the expansion and the readout so the transfer flies
    //: under this group's `ldmatrix` and MMA block -- two stages in flight, the second
    //: in registers, with exactly one asynchronous group outstanding at the wait.
#pragma unroll 1
    for (int b = 0; b < nbrM; ++b) {
      if (b == 0) {
        stage_factors(std::integral_constant<int, 1>{}, myStg, pR, rlistW, winP, 0, 0, rgroup_n(0));
        stage_commit();
      }
      const int n = rgroup_n(b);
      stage_wait<0>();
      expand_read(b & 1, n);
      __syncwarp();
      stage_factors(std::integral_constant<int, 1>{}, myStg, pR, rlistW, winP, (b + 1) & 1,
                    (b + 1) * kSegRows, rgroup_n(b + 1));
      stage_commit();
      readout_group(b * kSegRows, n);
    }

    //: THE WINDOW LINE, and it is there by the ordering law: no warp may fold this
    //: window's writes while another is still reading the window's entry state -- and
    //: the fold's segments are the readout's very bytes.
    rendezvous();

    //: THE PHASE CHANGE, READOUT -> FOLD: the mirror.  Each warp publishes its
    //: read-view hi -- which is exactly what its A fragments hold -- into the WRITE
    //: executors' segments; a barrier; each warp pulls its own back in WRITE order and
    //: rejoins it with the `lo` it kept, exactly.  The last barrier is the fold's: the
    //: segments it clears are these very bytes.
    //: THE UNPARK PRECEDES THE PUBLISH, and that order is the whole protocol at
    //: `PK == 1`: a publisher writes into the CONSUMER's segment, where that consumer's
    //: `lo` is parked, so every warp takes its own back before ANY warp publishes.
    if constexpr (G::kTied) {
      if constexpr (G::kPark) unpark_lo(lo, mlo);
      join_hi(sa, ma, lo, mlo);
    } else {
      if constexpr (G::kPark) {
        unpark_lo(lo, mlo);
        rendezvous();
      }
      publish_write(sa, ma);
      rendezvous();
      reload_hi(lo, mlo);
    }
    rendezvous();

    // ---- PHASE 2: THE FOLD -- ONE UNION WALK, WRITER-SIDE PLACEMENT ---------
    //: F2/F4, AND STREAMS FIRE WHEN FULL (P76 CORRECTION-2, K35 S2c).  The CTA walks its
    //: union list ONCE.  Warp `warp` takes the fixed share `i % kWarps` of the union, in
    //: ASCENDING union order -- no dynamic assignment -- and for each token it handles
    //: and each STREAM that wants it, it places into that stream's segment at the
    //: stream's RUNNING rank `rank(t, s) mod kSegRows`:
    //:   (a) the value row, by `cp.async`, and
    //:   (b) the FULLY EXPANDED coefficient row for that stream's box -- `kLeafW`
    //:       bf16 in the stream's private order, gain applied.
    //: The consumer does NO produce: tile skip -> `ldmatrix` -> HMMA.  The sub-products
    //: over the levels a stream does NOT carve are shared across the streams of one
    //: token and are minted once as `memo`; the memo is INTERNAL to the fetcher and its
    //: shape is the span vector's, at compile time.
    //:
    //: THERE IS NO BATCH AND NO BARRIER INSIDE THE WALK (P76 CORRECTION-2).  A stream's
    //: rows ACCUMULATE and the stream fires when its own tile is full, so it issues
    //: exactly `ceil(n_s / kSegRows)` k-steps per window.  The walk still advances in
    //: GROUPS of `kSegRows` union tokens, and that is a FETCH QUANTUM -- one staging
    //: block, one lane map over `kCand` pairs, PER WARP -- and no firing is tied to it.
    //: THE TWO COUNTERS ARE THE WHOLE PROTOCOL: a fetcher writes rank `r` of stream `s`
    //: only while `r >> log2 kSegRows <= min_e drained[e]`, and an executor pulls tile
    //: `k = drained[e]` when `landed[s] >= (k + 1) * kSegRows`.  WHY A COUNT IS AS GOOD
    //: AS A PREFIX, and the deadlock-freedom argument the code's order encodes:
    //: docs/internals/carry/carry_kernel.md, "The flow counters".
    //:
    //: COVERAGE IS BY CONSTRUCTION, NOT BY ACCOUNTING (the salvage's law: `0 * NaN` is
    //: `NaN`, so a slot row the MMA reads must never be arbitrary SMEM).  THE DEAD-ROW
    //: LAW: a dead row is a FINITE (stale) value row with a ZERO coefficient row, and
    //: coverage is the ONCE-PER-WINDOW clear plus ONE dead-coefficient zeroing, at the
    //: window's LAST, PARTIAL tile -- every other tile a stream fires is FULL by
    //: construction, so there is nothing dead in it.
    //: THE EXPANSION'S LANE MAP.  A coefficient row is `kWordsRow` packed pairs; the
    //: warp covers `kRowsPass` rows at once, one per lane group, and the rows of a pass
    //: are consecutive (token, stream) pairs -- SAME token, adjacent streams -- which
    //: is what lets one memo serve the whole pass.
    constexpr int kWordsRow = kLeafW / 2;
    constexpr int kLanesRow = kWordsRow < kBallotLanes ? kWordsRow : kBallotLanes;
    constexpr int kRowsPass = kBallotLanes / kLanesRow;
    constexpr int kWordsLane = kWordsRow / kLanesRow;
    constexpr int kStreamPass = kStreams / kRowsPass;
    static_assert(kStreams % kRowsPass == 0 && kStreamPass >= 1,
                  "a pass's rows are streams of ONE token, so the stream count carries them");
    static_assert(kLanesRow >= kMmaN, "an n-tile pair's liveness is a byte of the row's ballot");
    //: n-tile PAIRS -- one `ldmatrix` and one skip bit each -- of the WRITE STREAM's row
    //: (`kNJW`, which the expansion's ballot names) and of THIS EXECUTOR's column half
    //: (`kNJ`, which the fold issues).  The mask word carries the stream's bits; a
    //: warp's field of it starts at `col * kNJ`.
    constexpr int kNJW = NLW / 2;
    constexpr int kNJ = NL / 2;
    static_assert(kNJ * WPS == kNJW, "the column halves partition the stream's n-tile pairs");
    static_assert(kWordsLane * (kLanesRow / kMmaN) == kNJW,
                  "the mask carries exactly one bit per n-tile pair of the stream's row");
    static_assert(kLanesRow % WPS == 0,
                  "the `WPS` lanes that publish a row's mask are one per column half");
    //: THE VALUE ROW'S MAP, AND IT IS A WHOLE-WARP ROW (K35 R-C, item 5).  A row is
    //: `kDv` bf16 = `kDv * 2` bytes; the warp covers 128 of them per pass at one
    //: 32-bit word a lane, so a row is `kVW` words a lane and the lane's word `w`
    //: carries channels `2*lane + 64*w`.  The SAME expression addresses the global
    //: load and every shared store, which is what makes the fetch and the send one
    //: layout rather than two.
    constexpr int kVW = kDv / 64;
    static_assert(kVW * 64 == kDv && kVW >= 1,
                  "a value row is a whole number of warp-wide 128-byte passes");
    const int exH = lane >> ilog2c(kLanesRow);  // which row of the pass this lane serves
    const int exJ = lane & (kLanesRow - 1);     // its packed pair inside that row
    //: THE MASK'S TWO PER-WARP CONSTANTS: the field of the stream's mask word this
    //: executor's column half owns, and the bit its own n-tile pair `nj` sits at.  Both
    //: are computed once per warp, so the column split costs the fold NOTHING per batch
    //: beyond the tests it already ran.
    const uint32_t mBit = 1u << (col * kNJ);
    const uint32_t mFld = (uint32_t)((((1u << kNJ) - 1u) << (col * kNJ)));
    //: and the lanes that PUBLISH a row's mask: one per column half, so the `WPS` words
    //: of a stream are written by the same single instruction the one word was.
    constexpr int kMskLanes = kLanesRow / WPS;
    const int exM = exJ >> ilog2c(kMskLanes);
    const bool exPub = (exJ & (kMskLanes - 1)) == 0;

    //: THE WALKING WARP'S VALUE ROWS FOR THE GROUP IT HOLDS -- the staging copy the
    //: fill contract names, in registers.  `TPW` tokens of this warp's own share, `kVW`
    //: words of each: 2 registers a lane at the eight-warp `d_v = 64` arms, 4 at the
    //: four-warp and the wide ones.  It is written once per group by `load_values` and
    //: read by `send_values` in every placement round of that group.
    //: TWO BUFFERS AND A ROTATE, NOT AN INDEXED PAIR: `vnxt` is filled a WHOLE GROUP
    //: ahead (beside that group's `stage_factors`) and rotated into `vcur` when the
    //: group is picked up, so the global latency runs under an entire group's placement,
    //: expansion and folding -- the two-stages-in-flight law with the second stage in
    //: registers (F4), which is exactly what the `cp.async` this replaces provided.
    //: THE ROTATE IS REGISTER MOVES AND THE INDEX IS NEVER DYNAMIC: `vrow[g & 1]` would
    //: put the whole array in local memory whatever the parity's provenance, so the two
    //: buffers are named and copied instead (`TPW * kVW` moves a group -- 4 at the
    //: four-warp arms, 2 at the eight-warp ones).
    uint32_t vcur[TPW][kVW], vnxt[TPW][kVW];
    //: THE VALUE LEG, AS A FETCH AND A SEND (K35 R-C item 5; P76 R-C's fill contract).
    //: THE DEFECT THIS REPLACES: the leg used to run over the CANDIDATE PAIRS and issue
    //: one `cp.async` per (token, stream), so a token live in `k` streams was pulled
    //: from GLOBAL `k` times -- R-A measured that as DRAM +56% and L2 +58% at `07c4409`,
    //: and it is the one place the built kernel collapsed the schedule's part 1 (fetch,
    //: over TOKENS) into its part 2 (send, over PAIRS).
    //:
    //: THE TWO HALVES ARE NOW SEPARATE AND EACH RUNS AT ITS OWN GRAIN.
    //:   `load_values` is PART 1: one coalesced 128-byte-per-pass global read per UNION
    //:   TOKEN, over the walking warp's own share `ic * kWarps + warp` -- the same share
    //:   `stage_factors` uses for the factor rows, so a token's value row and its factor
    //:   rows are fetched by one warp, once, together.  FETCH COUNT PER (TOKEN, WINDOW)
    //:   IS EXACTLY ONE and does not depend on how many streams want the token.
    //:   `send_values` is PART 2: the row is already in registers, so a live pair costs
    //:   `kVW` shared stores and no memory traffic at all.  SEND COUNT IS THE LIVE
    //:   STREAM COUNT, which is the irreducible number -- the segments are private and
    //:   each consumer's `ldmatrix` reads its own rows.
    //:
    //: WHY REGISTERS AND NOT A SHARED STAGING COPY: at `d_v = 64` a whole value row is
    //: ONE 32-bit register a lane (`kVW = 1`, two at the wide plane), so the "staging
    //: copy" the fill contract names costs `kVW` registers and the send reads it at
    //: register latency; a shared staging plane would cost the same stores PLUS a
    //: load and `kSegRows * LdV * 2` more bytes of an already-tight budget.
    //: THE LOAD IS ISSUED A WHOLE GROUP BEFORE THE SEND CONSUMES IT (at the point the
    //: group is picked up, before `place_group`), so its latency runs under the
    //: placement's own shared loads -- the two-stages-in-flight law (F4) with the
    //: second stage in registers, exactly as the segments themselves obey it.
    //: BODY COUNT: `load_values` `TPW * kVW` predicated loads; `send_values`
    //: `TPW * kStreams * kVW` predicated stores, both compile-time nests.
    auto load_values = [&](uint32_t(&dst)[TPW][kVW], int u0, int nb) {
      static_for<TPW>([&](auto Ic) {
        constexpr int ic = decltype(Ic)::value;
        const int ii = ic * kWarps + warp;
        const bool live = ii < nb;
        //: a token past the group's end reads token zero, which is in range, and its
        //: row is never SENT -- no candidate names it.
        const uint32_t tokV = (uint32_t)(live ? wlistS[u0 + ii] : 0) * (uint32_t)v_row_bytes;
        static_for<kVW>([&](auto Wc) {
          constexpr int w = decltype(Wc)::value;
          dst[ic][w] = *reinterpret_cast<const uint32_t*>(
              gather_ptr(vPlane, tokV + winV + (uint32_t)((w * kBallotLanes + lane) * 4)));
        });
      });
    };
    //: THE SEND.  The candidate word for the pair `(ic, s)` lives on lane
    //: `ic * kStreams + s` (`WriteFlow::kNS == kStreams`), so one broadcast names both
    //: the liveness and the destination RANK, and every lane then stores its own
    //: channels of the row into that stream's segment row.
    //: the group cursor's advance for the value staging: what was prefetched becomes
    //: what is sent.  Compile-time indices throughout, so it is register moves.
    auto rotate_values = [&]() {
      static_for<TPW>([&](auto Ic) {
        constexpr int ic = decltype(Ic)::value;
        static_for<kVW>([&](auto Wc) {
          constexpr int w = decltype(Wc)::value;
          vcur[ic][w] = vnxt[ic][w];
        });
      });
    };
    auto send_values = [&](int e0) {
      static_for<TPW>([&](auto Ic) {
        constexpr int ic = decltype(Ic)::value;
        static_for<kStreams>([&](auto Sc) {
          constexpr int st_ = decltype(Sc)::value;
          const int e = lane_broadcast(e0, ic * kStreams + st_);
          if (e >= 0) {
            const uint32_t rowB =
                segVB + (uint32_t)(((st_ * kSegRows + ((e >> 16) & (kSegRows - 1))) * LdV) * 2);
            static_for<kVW>([&](auto Wc) {
              constexpr int w = decltype(Wc)::value;
              store_shared_u32(rowB + (uint32_t)((w * kBallotLanes + lane) * 4), vcur[ic][w]);
            });
          }
        });
      });
    };
    //: ONE AMPLITUDE OUT OF THE STAGED ROW.  The level lying LOWEST in this side's
    //: private order (`PMW::kPairLevel`) supplies BOTH slots of a packed pair with one
    //: four-byte access, because its two digits are adjacent there; every other level's
    //: digit is constant across the pair and is splatted.
    //: THE EXPANSION.  DECLARED ARITHMETIC ORDER, per packed slot pair:
    //:     w = gain, then x f_l for every level the side SPANS WHOLE in ascending `l`,
    //:                then x f_l for every level the side CARVES in ascending `l`,
    //: each product rounded to nearest even in bf16 (`mul_bf16x2`).  The prefix through
    //: the whole-spanned levels is the MEMO -- it does not depend on the stream, so it
    //: is minted once per (token, lane) and reused by every stream of that token.  This
    //: order is NOT the deleted produce's (which multiplied the pair level first and
    //: the gain last), so the fold's coefficients are not bit-identical to it; the
    //: total multiply count is unchanged and the oracle is the gate.
    //: BODY COUNT per call: `TPW * (1 + kWordsLane * U)` for the memo and
    //: `TPW * kStreamPass * kWordsLane * (Cv + 2)` for the rows, with `U`/`Cv` the
    //: whole-spanned and carved level counts -- 4*(1+1) + 4*2*1*3 = 32 at the D=2 ALT
    //: arms, 2*(1+3) + 2*4*1*4 = 40 at the D=4 multi-level carve.
    auto expand = [&](int slot, int e0) {
      const uint32_t blkB = stgB + (uint32_t)(slot * kSegRows * BP::kStageW * 2);
      const uint32_t gblkB = gnB + (uint32_t)(slot * kSegRows * 2);
      static_for<TPW>([&](auto Ic) {
        constexpr int ic = decltype(Ic)::value;
        const int ii = ic * kWarps + warp;
        const uint32_t rowB = blkB + (uint32_t)(ii * BP::kStageW * 2);
        const uint32_t gain = splat_bf16x2_bits(load_shared_u16(gblkB + (uint32_t)(ii * 2)));
        uint32_t memo[kWordsLane];
        static_for<kWordsLane>([&](auto Wc) {
          constexpr int wl = decltype(Wc)::value;
          const int p0 = 2 * (exJ + wl * kLanesRow);
          uint32_t a = gain;
          static_for<D>([&](auto Lc) {
            constexpr int l = decltype(Lc)::value;
            if constexpr (!SVW::carves(l))
              a = mul_bf16x2(a, factor_at(type_tag<PMW>{}, rowB, Lc, p0, 0));
          });
          memo[wl] = a;
        });
        static_for<kStreamPass>([&](auto Pc) {
          constexpr int pp = decltype(Pc)::value;
          const int s = pp * kRowsPass + exH;
          const int e = lane_broadcast(e0, ic * kStreams + s);
          const uint32_t rowW =
              segCB + (uint32_t)(((s * kSegRows + ((e >> 16) & (kSegRows - 1))) * LdS) * 2);
          uint32_t msk = 0u;
          static_for<kWordsLane>([&](auto Wc) {
            constexpr int wl = decltype(Wc)::value;
            const int jw = exJ + wl * kLanesRow;
            uint32_t a = memo[wl];
            static_for<D>([&](auto Lc) {
              constexpr int l = decltype(Lc)::value;
              if constexpr (SVW::carves(l))
                a = mul_bf16x2(a, factor_at(type_tag<PMW>{}, rowB, Lc, 2 * jw,
                                            stream_off(type_tag<SVW>{}, s, Lc)));
            });
            if (e >= 0) store_shared_u32(rowW + (uint32_t)(jw * 4), a);
            //: THE PER-ROW TILE-LIVE MASK, produced BY the expansion because only the
            //: expansion knows which slots it wrote nonzero.  One ballot names every
            //: live packed pair of the pass's rows; a byte of it is an n-tile pair.
            const uint32_t bal = lane_ballot(e >= 0 && (a & kBf16MagnitudeMask) != 0u);
            static_for<kLanesRow / kMmaN>([&](auto Gc) {
              constexpr int g = decltype(Gc)::value;
              if (((bal >> (exH * kLanesRow + kMmaN * g)) & 0xFFu) != 0u)
                msk |= 1u << (wl * (kLanesRow / kMmaN) + g);
            });
          });
          //: ONE LANE PER COLUMN HALF publishes the row's mask into that half's word.
          //: Every word of the stream carries the WHOLE row's bits and each consumer
          //: reads only its own field, so this is the single instruction the one-word
          //: form was, with more lanes live rather than more issues.
          if (exPub && msk != 0u) or_shared_u32(mskB + (uint32_t)((s * WPS + exM) * 4), msk);
        });
      });
    };
    //: THE GAIN rides the same block, PER WARP like the staging: `TPW` lanes carry the
    //: warp's own share of the group's tokens and nothing crosses a warp boundary.
    auto gain_load = [&](int u0, int nb) {
      unsigned short g = 0;
      if (lane < TPW && lane * kWarps + warp < nb)
        g = *reinterpret_cast<const unsigned short*>(
            gather_ptr(gW, winG + (uint32_t)wlistS[u0 + lane * kWarps + warp] * 2u));
      return g;
    };
    auto gain_store = [&](int slot, unsigned short g) {
      if (lane < TPW)
        store_shared_u16(gnB + (uint32_t)((slot * kSegRows + lane * kWarps + warp) * 2), g);
    };
    //: THE WINDOW'S CLEAR, paid ONCE: it makes every row of the stream's segment finite,
    //: which is what lets every later tile leave its stale value rows in place.  IT IS
    //: SPLIT BY ROWS over the stream's `WPS` warps -- the whole segment is covered before
    //: the rendezvous that follows, so any partition is legal and rows keep both legs
    //: contiguous.
    constexpr int kClrRows = kSegRows / WPS;  // segment rows one executor clears
    static_assert(kClrRows * WPS == kSegRows, "the segment's rows split over the stream's warps");
    constexpr int kClrV = kClrRows * LdV * 2 / 16, kClrC = kClrRows * LdS * 2 / 16;
    static_assert(kClrV * 16 == kClrRows * LdV * 2 && kClrC * 16 == kClrRows * LdS * 2,
                  "a segment row block is a whole number of sixteen-byte quanta");
    //: BODY COUNT: `ceil(kClrRows*LdV*2/512) + ceil(kClrRows*LdS*2/512)` = 5 + 3 at the
    //: uncolumned `d_v = 64` arms (`LdV = 72`, `LdS = 40`, `kSegRows = 16`), 3 + 2 at
    //: `WPS = 2`, and 5 + 2 at `d_v = 128` `WPS = 2` (`LdV = 136`, `LdS = 40`).
    auto clear_segment = [&]() {
      const uint4 z = make_uint4(0u, 0u, 0u, 0u);
      const int r0 = wstrm * kSegRows + col * kClrRows;
      const uint32_t vB = segVB + (uint32_t)(r0 * LdV * 2 + lane * 16);
      const uint32_t cB = segCB + (uint32_t)(r0 * LdS * 2 + lane * 16);
      static_for<(kClrV + kBallotLanes - 1) / kBallotLanes>([&](auto Pc) {
        constexpr int pp = decltype(Pc)::value;
        if (pp * kBallotLanes + 32 <= kClrV || pp * kBallotLanes + lane < kClrV)
          store_shared_vec16(vB + (uint32_t)(pp * kBallotLanes * 16), z);
      });
      static_for<(kClrC + kBallotLanes - 1) / kBallotLanes>([&](auto Pc) {
        constexpr int pp = decltype(Pc)::value;
        if (pp * kBallotLanes + 32 <= kClrC || pp * kBallotLanes + lane < kClrC)
          store_shared_vec16(cB + (uint32_t)(pp * kBallotLanes * 16), z);
      });
      if (lane == 0) store_shared_u32(mskB + (uint32_t)(strm * 4), 0u);
    };
    //: THE WINDOW'S DEAD ROWS, and there is exactly ONE set of them: the stream's LAST
    //: tile is the only one that is not full.  Its live rows are the PREFIX `[0, n_s mod
    //: kSegRows)` and the rest are zeroed here.  THE CLEAR IS SPLIT BY COLUMNS, not rows:
    //: a warp may only zero bytes IT ITSELF reads, the stream's other executor holding
    //: the other column half, which is `kLeafE` contiguous sixteen-byte-aligned slots.
    constexpr int kClrHalf = kLeafE * 2 / 16;  // quanta in ONE row's column half
    static_assert(kClrHalf * 16 == kLeafE * 2 && (kClrHalf & (kClrHalf - 1)) == 0,
                  "a column half of a coefficient row is a power-of-two number of quanta");
    constexpr int kDeadQ = kSegRows * kClrHalf;  // the whole segment's worth, the bound
    //: BODY COUNT: `ceil(kSegRows*kLeafE*2/512)` -- 2 at 32 leaves an executor, 1 at 16
    //: -- each one predicated store, and the stores actually issued are
    //: `(kSegRows - n) * kClrHalf`, once per window.
    auto clear_dead = [&](int n) {
      const uint4 z = make_uint4(0u, 0u, 0u, 0u);
      const int q = (kSegRows - n) * kClrHalf;
      const uint32_t cB = segCB + (uint32_t)(((wstrm * kSegRows + n) * LdS + colBase) * 2);
      static_for<(kDeadQ + kBallotLanes - 1) / kBallotLanes>([&](auto Pc) {
        constexpr int pp = decltype(Pc)::value;
        const int qi = pp * kBallotLanes + lane;
        if (qi < q)
          store_shared_vec16(
              cB + (uint32_t)((qi >> ilog2c(kClrHalf)) * LdS * 2 + (qi & (kClrHalf - 1)) * 16), z);
      });
    };

    //: THE WALK'S GROUPS -- a FETCH quantum of `kSegRows` union tokens: one staging
    //: block, one lane map, no firing boundary, no cross-warp synchronisation.
    const int nGroup = (nw + kSegRows - 1) >> kSegBits;
    auto group_n = [&](int g) {
      const int r = nw - (g << kSegBits);
      return r <= 0 ? 0 : (r < kSegRows ? r : kSegRows);
    };
    if (nGroup > 0) {
      //: `drained` is this executor's own count and lives in a REGISTER: it has exactly
      //: one writer, and the SMEM word beside it is that register PUBLISHED.
      uint32_t drained = 0u;
      //: THE PULL, AND IT IS THE WHOLE OF THE CONSUMER.  Read the mask, pull the tile
      //: into fragments, RESET the mask, publish `drained`, fold from the fragments.
      //: THE ORDER IS THE PROTOCOL: the count is what lets a fetcher write the NEXT tile
      //: -- its rows AND its mask bits -- so the `ldmatrix` and the reset precede it, and
      //: the MMAs run after it under the copies in flight (F4).
      //: BODY COUNT: `MCH + kNJ` `ldmatrix` (4 + 2 at the four-warp `d_v = 64` arms,
      //: 8 + 1 at the `d_v = 128` arms) and `kNJ * (2*MCH + 2)` MMAs, both O(fragments)
      //: and both gated by the tile-live mask.
      auto pull_and_fold = [&]() {
        uint32_t av[MCH][4], bc[kNJ][4];
        const uint32_t tm = load_shared_u32(mskB + (uint32_t)(strm * 4));
        if (tm & mFld) {
          static_for<MCH>([&](auto Mc) {
            constexpr int m = decltype(Mc)::value;
            load_frag_t(av[m], myV + (uint32_t)(m * kMmaM * 2));
          });
          static_for<kNJ>([&](auto Nc) {
            constexpr int nj = decltype(Nc)::value;
            if (tm & (mBit << nj)) load_frag_t(bc[nj], ADDR_S(myC, 0, 2 * nj));
          });
        }
        if (lane == 0) store_shared_u32(mskB + (uint32_t)(strm * 4), 0u);
        __syncwarp();
        publish_fence();
        ++drained;
        publish_drained(drained);
        static_for<kNJ>([&](auto Nc) {
          constexpr int nj = decltype(Nc)::value;
          if (tm & (mBit << nj)) {
            static_for<MCH>([&](auto Mc) {
              constexpr int m = decltype(Mc)::value;
              mma(st[m][2 * nj], av[m], &bc[nj][0]);
              mma(st[m][2 * nj + 1], av[m], &bc[nj][2]);
            });
            mma(mass_st[2 * nj], ones_a, &bc[nj][0]);
            mma(mass_st[2 * nj + 1], ones_a, &bc[nj][2]);
          }
        });
      };
      //: SEALED: the edge barrier has passed, so `landed[s]` is final and the LAST tile
      //: -- the only one that can be short -- is folded on that fact, not on fullness.
      bool sealed = false;
      clear_segment();
      flow_zero();
      stage_factors(std::integral_constant<int, kWarps>{}, stgB, pW, wlistS, winP, 0, 0,
                    group_n(0));
      stage_commit();
      gain_store(0, gain_load(0, group_n(0)));
      //: THE FIRST GROUP'S VALUE ROWS, issued with its factor rows: one fetch per union
      //: token, and the only one this window pays for them.
      load_values(vnxt, 0, group_n(0));
      //: THE FOLD'S OPENING BARRIER; it has no other until the window's edge.
      rendezvous();
      //: THE WALK, AS ONE LOOP WITH THE FOLD AT ITS HEAD.  Every wait in the body is a
      //: `continue` back to that head, so a warp that waits keeps DRAINING AND FOLDING
      //: its own stream -- which makes every wait deadlock-free, and is also why there is
      //: exactly ONE `pull_and_fold` and ONE `expand` in the generated code (a second
      //: inlining of the fold costs 16 registers and spills at the D=4 arms).  The state
      //: is the group cursor `g`, which doubles as the announcement flag past `nGroup`,
      //: and the per-lane word `e`, whose `-2` says NO GROUP IS HELD.
      int g = 0, e = -2;
#pragma unroll 1
      for (;;) {
        if (tile_ready(flow_landed(wstrm), drained, sealed)) {
          pull_and_fold();
          continue;
        }
        if (sealed) break;
        if (e == -2) {
          if (g >= nGroup) {
            //: THE WALK'S EDGE.  This warp's share is placed; announce it and keep
            //: servicing until every warp has announced.
            if (g == nGroup) {
              stage_wait<0>();
              announce_done();
              g = nGroup + 1;
            }
            if (!all_done()) continue;
            //: THE WINDOW'S EDGE.  One barrier, reached exactly once by every thread --
            //: after it `landed[s] == n_s` is final.  AT MOST ONE TILE IS LEFT UNDRAINED
            //: and it is the window's last, because a fetcher cannot write tile `k+1`
            //: before every executor has drained tile `k`.  Its dead coefficient rows
            //: are zeroed HERE and nowhere else.
            rendezvous();
            const int nMine = (int)flow_landed(wstrm);
            const int part = nMine & (kSegRows - 1);
            if (part != 0 && drained < (uint32_t)((nMine + kSegRows - 1) >> kSegBits))
              clear_dead(part);
            sealed = true;
            continue;
          }
          //: this warp's factor rows for the group have landed (issued one group ago).
          stage_wait<0>();
          //: AND ITS VALUE ROWS ARE ALREADY IN REGISTERS -- prefetched a whole group
          //: ago, beside this group's factor rows -- so picking the group up is a
          //: rotate and never a load.
          rotate_values();
          e = place_group(type_tag<WriteFlow>{}, memB, pfxB, wlistS, g << kSegBits, group_n(g));
        }
        //: THE GROUP'S PLACEMENTS, IN ROUNDS OF WHAT THERE IS ROOM FOR.  A candidate
        //: whose tile is full and not yet drained by every executor of its stream waits;
        //: the rest are placed and PUBLISHED first.  The round IS the deadlock-freedom
        //: invariant: a group can straddle a stream's tile boundary, so an
        //: all-or-nothing group would have a warp wait on rows it is itself holding
        //: back.  At most two rounds, one in the ordinary case.
        //: THE ROOM TEST, ONE EVALUATION PER STREAM PER TURN (K35 R-C item 6).  It used
        //: to run `drained_min` on EVERY LANE -- `WPS` shared loads a lane, for a word
        //: that takes only `kStreams` distinct values -- which is `## STAGE-LEDGER` S2c's
        //: gate at lane-and-turn grain against a floor of one load per stream.  The
        //: `kStreams` low lanes now read it once each and a single variable-index
        //: shuffle hands each candidate lane its own stream's word, so the loads fall
        //: from `32 * WPS` to `kStreams * WPS` per turn and the ballot below is the only
        //: per-warp fact the round needs.
        const uint32_t dmin =
            lane < kStreams ? drained_min(type_tag<WriteFlow>{}, lane) : 0xFFFFFFFFu;
        const uint32_t myRoom = __shfl_sync(0xFFFFFFFFu, dmin, lane & (kStreams - 1));
        const int er = (e >= 0 && (uint32_t)(e >> (16 + kSegBits)) <= myRoom) ? e : -1;
        if (lane_ballot(er >= 0) != 0u) {
          send_values(er);
          expand(g & 1, er);
          //: LANDED.  The value rows and the expansion's rows and mask bits are all
          //: STORED by this warp now -- the value leg is a register-sourced shared store
          //: since R-C, so there is no copy in flight to wait on here -- and the fence
          //: orders every one of them before the arrival that publishes them: NOTHING
          //: THIS WARP ISSUED IS UNPUBLISHED WHEN IT WAITS.
          publish_fence();
          publish_landed(type_tag<WriteFlow>{}, er);
          if (er >= 0) e = -1;
        }
        if (lane_ballot(e >= 0) != 0u) continue;
        //: THE NEXT GROUP'S FACTORS AND GAINS, one group ahead, so the folds at the
        //: loop's head run under them.
        const int n1 = group_n(g + 1);
        const unsigned short gn = gain_load((g + 1) << kSegBits, n1);
        stage_factors(std::integral_constant<int, kWarps>{}, stgB, pW, wlistS, winP, (g + 1) & 1,
                      (g + 1) << kSegBits, n1);
        stage_commit();
        gain_store((g + 1) & 1, gn);
        //: and the next group's value rows, one fetch per union token, a whole group
        //: before the send that consumes them.
        load_values(vnxt, (g + 1) << kSegBits, n1);
        ++g;
        e = -2;
      }
    }
    rendezvous();
  }

  // ---- the FINAL STATE, in LATTICE leaf order ------------------------------
  //: the mirror of the entry load, after the last fan-in has consumed the staging
  //: buffer -- which is what the one extra rendezvous below is for.
  if constexpr (STATE) {
    if (s_out != nullptr) {
      float* const plane = pin_address(s_out);
      rendezvous();
      static_for<kAtoms>([&](auto Ac) {
        page_regs(Ac, std::false_type{});
        rendezvous();
        page_sweep(page_off(atomBase + (uint32_t)decltype(Ac)::value),
                   [&](int sidx, uint32_t offb) {
                     store_f32(reinterpret_cast<float*>(gather_ptr(plane, offb)), so[sidx]);
                   });
        rendezvous();
      });
    }
  }
#undef ADDR_S
#undef ADDR_R
}

}  // namespace rola::carry
