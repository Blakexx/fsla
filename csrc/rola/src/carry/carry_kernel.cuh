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
// every derived quantity below (`kCols`, `LdV`, `LdO`, `kPlaneRows`, `MCH`, `kChunkV`,
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

//: THE BUILD'S IDENTITY, device-side.  A green battery against a stale binary is
//: this project's recurring failure; a stamp read OFF THE DEVICE is the only check
//: that catches it, so every gate in `tests/k31/` asserts this first.
static constexpr int kCarryBuildMagic = 31010025;

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
//: (`CarryGeom::kWarps`) and is not a constant of this file.
static constexpr int kWarps = 4;
static constexpr int kThreads = kWarps * 32;
static_assert(kAtomLeaves == kMmaK, "the page granule is one MMA k-step");

//: A COMPILE-TIME LOOP.  Every per-level quantity here is a template argument or a
//: register-array index, so the level index must be a constant expression; a
//: `for (int l = 0; l < D; ++l)` would make `load_run<n_l(l)>` ill-formed and would
//: turn every per-level array index dynamic, which is a stack frame.
//: a type carried through a generic lambda's argument, which is how a body that is
//: instantiated per SIDE names that side's span vector without a second template.
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

template <int D, int B, int K, int M, int DV, int W, int C, int WPS, int W0, int W1, int W2, int W3,
          int R0, int R1, int R2, int R3, int RC, int PK>
struct CarryGeom {
  using BP = BoxPlan<D, B, K, M>;
  using SVW = StreamSpans<D, B, K, M, W0, W1, W2, W3>;
  using SVR = StreamSpans<D, B, K, M, R0, R1, R2, R3>;
  static constexpr int BC = BP::kBC;
  //: THE VALUE WIDTH, AS A DECLARATION.  `kDv` channels plus the MASS column, which
  //: rides the state as one more row of every plane the state touches.  The clauses
  //: are STRUCTURAL and not a list of admitted widths: the channels tile into whole
  //: MMA m-tiles (`MCH`), a value row is a whole number of sixteen-byte asynchronous
  //: quanta (`kChunkV = kDv / 8`), and those quanta divide a lockstep unit so the
  //: fetcher's value pass is a lane split rather than a predicate.
  static constexpr int kDv = DV;
  static constexpr int kCols = DV + 1;
  static_assert(DV >= kMmaM && DV % kMmaM == 0, "the value channels tile into whole MMA m-tiles");
  static_assert((DV & (DV - 1)) == 0 && DV / 8 <= kBallotLanes,
                "a value row is a power-of-two number of sixteen-byte quanta and a "
                "lockstep unit covers at least one whole row");
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
  static constexpr int xch_bytes = kXchRows * LdX * (int)sizeof(__nv_bfloat16);
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

  // ---- THE READOUT'S TENANTS (K35 S4b) -------------------------------------
  //: THE READER SEGMENTS.  The read side's cooperative fetch is the mirror of the
  //: write side's with ONE difference of substance: on the readout the B operand is
  //: `R~^T [leaves x tokens]`, so TOKENS are the `n` axis and the state is the `k`-side
  //: A operand held in registers -- there is no value row to place and a reader's
  //: segment is coefficient rows alone.  `ReaderSegRows` is `[reader][row][slot]` at
  //: row stride `LdR`, a ROW being one (union token, reader) pair placed at the token's
  //: rank inside that reader's own compacted list.  Each warp IS a reader stream
  //: (`SVR::kStreams == kWarps`, the executor map's declaration), so the segment
  //: replaces the deleted two-slot read panel at the same stride and the same
  //: `ldmatrix` addressing, with one slot instead of two: the second stage is
  //: registers here exactly as it is in the fold.
  static constexpr int rseg_bytes = kWarps * kSegRows * LdR * (int)sizeof(__nv_bfloat16);
  //: THE READ SIDE'S FACTOR STAGING, the same rolling two-block window the fold has:
  //: the compressed owner runs of one batch's union tokens, `cp.async`ed a batch ahead
  //: of the expansion that reads them.  It is the SAME `StagedFactorRow` layout at the
  //: same stride -- what differs is the plane it is filled from (`pread`) and the list
  //: that names the tokens -- so one body serves both sides and the round trip that
  //: gates the layout gates both.
  static constexpr int rstg_bytes = kStgSlots * kSegRows * BP::kStageW * (int)sizeof(__nv_bfloat16);
  //: THE ROW INDEX OF A BATCH: for each (reader, segment row) the token's UNION RANK
  //: inside the batch and the number of readers EXPECTED to arrive at it, packed into
  //: one halfword by the same fetcher lane that resolved the placement.  It is what
  //: turns a reader's private row back into the CTA-wide rank the ring is keyed on,
  //: and it is read into registers during the drain, so it has the segment's own
  //: lifetime and needs no second slot.
  static constexpr int rmap_bytes = align16c(kWarps * kSegRows * (int)sizeof(uint16_t));
  //: THE FAN-IN RING (P76 section 2.6-C).  `kRingC` rows, rank `r` at slot `r % C`,
  //: each row `kDv` output channels and the MASS in fp32.  The row stride is padded to
  //: `kDv + 8` WORDS and that number is derived rather than chosen: a scatter
  //: instruction covers FOUR ring rows (the accumulator's four lane quads are four
  //: tokens) at eight consecutive channels each, so what the layout must give is eight
  //: whole banks per row -- `LdRing % 32 == 8` puts consecutive rows exactly one
  //: eight-bank block apart at both widths.
  static constexpr int kRingC = RC;
  static constexpr int LdRing = kDv + 8;
  static_assert(RC >= kSegRows && (RC & (RC - 1)) == 0,
                "the ring depth is a power of two of at least one batch: a scatter "
                "instruction holds four ranks of ONE batch at once, so they must land in "
                "four distinct slots");
  static_assert(LdRing % 32 == 8 && kDv % 32 == 0,
                "a ring row must step one eight-bank block, which is what makes the "
                "four-row scatter a bank permutation");
  static constexpr int ring_bytes = kRingC * LdRing * (int)sizeof(float);
  //: the ring's control words -- `{tag, arrived, expected}` per slot, as three planes
  //: rather than a struct, because the tag is the only one read on the fast path.
  static constexpr int rctl_bytes = align16c(3 * kRingC * (int)sizeof(uint32_t));

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
  static constexpr int park_bytes = kPark ? xch_bytes : 0;

  static constexpr int off_rseg = park_bytes;
  static constexpr int off_rstg = off_rseg + rseg_bytes;
  static constexpr int off_rmap = off_rstg + rstg_bytes;
  static constexpr int off_ring = align16c(off_rmap + rmap_bytes);
  static constexpr int off_rctl = off_ring + ring_bytes;
  static constexpr int read_bytes = off_rctl + rctl_bytes;
  static_assert(off_rstg % 16 == 0 && off_rmap % 16 == 0,
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
  //: TWO LISTS, AND THEY ARE BOTH THE CTA's (K35 S4b).  Neither side keeps a
  //: per-stream list any more: on both sides the UNION list is the whole schedule and a
  //: stream's membership is its published ballot word, so what survives is one union
  //: walk per side.
  static constexpr int kLists = 2;
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
  //: AND THE SAME INDEX FOR THE READ SIDE, whose streams are the EXECUTORS themselves
  //: -- one reader per warp, so this table is `kWarps` rows where the write side's is
  //: `kStreams`, and at `WPS == 1` the two counts coincide without the tables ever
  //: being the same bytes.
  static constexpr int memr_bytes = align16c(kWarps * kLW * (int)sizeof(uint32_t));
  static constexpr int pfxr_bytes = align16c(kWarps * kLW * (int)sizeof(uint16_t));
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
  static constexpr int off_memr = off_pfx + pfx_bytes;
  static constexpr int off_pfxr = off_memr + memr_bytes;
  static constexpr int off_mask = off_pfxr + pfxr_bytes;
  static constexpr int off_flow = off_mask + mask_bytes;
  static constexpr size_t bytes = (size_t)(off_flow + flow_bytes);
  static constexpr int cta_target = design::cta_target((int)bytes, kThreads, kRegFloor);
  //: ---- THE LEDGER, PER MOMENT (P76 "THE SMEM LEDGER, PER MOMENT") ----------
  //: The window's INDEX -- the lists, the published ballots, the prefix counts and the
  //: liveness mask -- is live in every moment and is nobody's straddle.  The three
  //: moments over it are the FOLD (segments, staging, gain, the page buffer), the
  //: CHANGE (the exchange segments alone) and the READOUT (the reader segments, the
  //: read staging, the batch's row index and the fan-in ring).
  static constexpr int index_bytes =
      list_bytes + mem_bytes + pfx_bytes + memr_bytes + pfxr_bytes + mask_bytes + flow_bytes;
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
//: THE PRIVATE-ORDER RULE, which fixes the order from the span vector alone AND FROM
//: NOTHING ELSE (K35 S4a): the levels this side CARVES go OUTERMOST keeping their
//: lattice order among themselves, the levels it spans whole go inside them in lattice
//: order, and NO LEVEL IS PINNED.  A side carves where it is declared sparse, so the
//: rule is "sparse outermost, dense innermost": an eight-slot n-tile spans dense digits
//: and each sparse level indexes WHOLE tiles, where the digit skip is exact.
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
  static constexpr bool before(int a, int b) {
    return SV::carves(a) != SV::carves(b) ? !SV::carves(a) : a > b;
  }
  static constexpr int count_before(int l, int i) {
    return i >= D ? 0 : (before(i, l) ? 1 : 0) + count_before(l, i + 1);
  }
  static constexpr int rank(int l) { return count_before(l, 0); }
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

//: THE PER-LEVEL MODE DECLARATION, as the kernel reads it (the seam's own encoding):
//: two bits per level, level 0 in the low pair, bit 0 = the READ side is value-sparse
//: there, bit 1 = the WRITE side.  The kernel consumes it to PRUNE its clause: a side
//: reads a level's row only where that side is declared sparse.
//: THE FIRST LEVEL A SIDE SPANS WHOLE, which is where a GAINLESS expansion's product
//: begins: the read side has no gain, so seeding at this level is what keeps a multiply
//: by one out of the readout's coefficient rows.  `-1` where a side carves every level,
//: and the seed is then the identity the accumulator starts at.
template <class SV, int D>
constexpr int first_whole_level(int l = 0) {
  return l >= D ? -1 : (!SV::carves(l) ? l : first_whole_level<SV, D>(l + 1));
}

constexpr uint32_t kSideRead = 1u, kSideWrite = 2u;
constexpr bool mode_declares(uint32_t modes, int l, uint32_t side) {
  return ((modes >> (2 * l)) & side) != 0u;
}

//: THE SECOND LAUNCH BOUND: the CTA count this arm can actually reach, taken from
//: the same geometry the ring depth is.  A ring sized for a residency the launch
//: bound does not ask for is a ring sized for nothing.
template <int D, int B, int K, int M, int DV, int W, int C, int WPS, int W0, int W1, int W2, int W3,
          int R0, int R1, int R2, int R3, int RC, int PK>
constexpr int kCarryCtaTarget =
    CarryGeom<D, B, K, M, DV, W, C, WPS, W0, W1, W2, W3, R0, R1, R2, R3, RC, PK>::cta_target;

// ---------------------------------------------------------------- the kernel

template <int D, int B, int K, int M, int DV, int W, int C, int WPS, int W0, int W1, int W2, int W3,
          int R0, int R1, int R2, int R3, int RC, int PK, uint32_t MO, bool STATE>
__global__ __launch_bounds__((CarryGeom<D, B, K, M, DV, W, C, WPS, W0, W1, W2, W3, R0, R1, R2, R3,
                                        RC, PK>::kThreads),
                             (kCarryCtaTarget<D, B, K, M, DV, W, C, WPS, W0, W1, W2, W3, R0, R1, R2,
                                              R3, RC, PK>)) void
carry_kernel(const __nv_bfloat16* __restrict__ pread, const __nv_bfloat16* __restrict__ pwrite,
             const __nv_bfloat16* __restrict__ gwrite, const __nv_bfloat16* __restrict__ v,
             const uint32_t* __restrict__ rows, float* __restrict__ o, float* __restrict__ den,
             int L, int tok_words, int v_row_bytes, const float* __restrict__ s_in,
             float* __restrict__ s_out) {
  using BP = BoxPlan<D, B, K, M>;
  using G = CarryGeom<D, B, K, M, DV, W, C, WPS, W0, W1, W2, W3, R0, R1, R2, R3, RC, PK>;
  //: THE ARM'S VALUE WIDTH, named once so the body below reads a geometry and never a
  //: file constant: `kDv` channels and the mass column that rides them.
  constexpr int kDv = DV;
  constexpr int kCols = DV + 1;
  //: THE TWO SPAN VECTORS.  The READ side's carve and the WRITE side's carve of the
  //: same box, each with its own private order under the rule above.
  using SVR = StreamSpans<D, B, K, M, R0, R1, R2, R3>;
  using SVW = StreamSpans<D, B, K, M, W0, W1, W2, W3>;
  using PM = SpanMap<D, B, K, M, SVR>;
  using PMW = SpanMap<D, B, K, M, SVW>;
  //: THE ROW TABLE, refined at every level EITHER side carves, so each stream reads
  //: its OWN range's rows on either side and a side wanting a coarser range ORs the
  //: finer rows it already holds.
  using CR = CarveRows<SVW, SVR>;
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
  //: THE READOUT'S TENANTS, straddling the fold's at the same base (K35 S4b): the
  //: per-reader segments, the read side's rolling factor staging, the batch's row index
  //: and the fan-in ring with its three control planes.
  const SmemAddr rsegB = smem_addr(carry_smem + G::off_rseg);
  const SmemAddr rstgB = smem_addr(carry_smem + G::off_rstg);
  const SmemAddr rmapB = smem_addr(carry_smem + G::off_rmap);
  const SmemAddr ringB = smem_addr(carry_smem + G::off_ring);
  const SmemAddr tagB = smem_addr(carry_smem + G::off_rctl);
  const SmemAddr arrB = tagB + (uint32_t)(G::kRingC * 4);
  const SmemAddr expB = arrB + (uint32_t)(G::kRingC * 4);
  constexpr int kListStride = G::kListStride;
  uint16_t* const slist = reinterpret_cast<uint16_t*>(carry_smem + G::off_list);
  const SmemAddr memB = smem_addr(carry_smem + G::off_mem);
  const SmemAddr pfxB = smem_addr(carry_smem + G::off_pfx);
  const SmemAddr memrB = smem_addr(carry_smem + G::off_memr);
  const SmemAddr pfxrB = smem_addr(carry_smem + G::off_pfxr);
  const SmemAddr mskB = smem_addr(carry_smem + G::off_mask);
  //: THE FLOW COUNTERS as ONE base with compile-time displacements, so the three planes
  //: cost the register file one address and not three.
  const SmemAddr flowB = smem_addr(carry_smem + G::off_flow);
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
  //: AND THE BALLOT ROW BASE, per level: the SAME `o_l`, addressing the FIRST row the
  //: facts pass voted for this owner's run at this level (K31-i-d) -- at the table's
  //: grain, which is the finer of the two sides' spans, so a side reads its own
  //: range's rows and the owner's word is the OR of the run's.  The liveness pass
  //: reads these words instead of re-reading the amplitude runs itself.
  uint32_t rowOff[D];
  const uint32_t rowBytes = (uint32_t)tok_words * (uint32_t)sizeof(uint32_t);
  static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    const int o_l = (owner / BP::owner_div(l)) % BP::g(l);
    runCol[l] = BP::col_base(l) + o_l * BP::s(l);
    offR[l] = ((strm >> ilog2c(SVR::grid_div(l))) & (SVR::g(l) - 1)) * SVR::s(l);
    offW[l] = ((wstrm >> ilog2c(SVW::grid_div(l))) & (SVW::g(l) - 1)) * SVW::s(l);
    rowOff[l] = (uint32_t)(CR::row_prefix(l) + o_l * CR::rows_of(BP::s(l), l)) * rowBytes;
  });
  //: ONE 64-BIT BASE PER SIDE, and 32-bit byte displacements after it: the side's
  //: whole row block is `kRows * tok_words * 4` bytes, which is 32-bit at every
  //: admitted length.
  const uint32_t* const rowsR =
      pin_address(rows + (long)(2 * bh + 0) * (long)CR::kRows * (long)tok_words);
  const uint32_t* const rowsW =
      pin_address(rows + (long)(2 * bh + 1) * (long)CR::kRows * (long)tok_words);

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

  //: the warp's own staging region of the readout block.

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
  constexpr int kSlotBits = ilog2c(kLeafW) + 1;
  int rowBaseR = 0, rowBaseW = 0;
  static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    rowBaseR += (offW[l] >> ilog2c(SVR::s(l))) * (SVR::grid_div(l) * kLeafE)
                + ((offW[l] & (SVR::s(l) - 1)) << PM::shift(l));
    rowBaseW += (offR[l] >> ilog2c(SVW::s(l))) * (SVW::grid_div(l) * kLeafW)
                + ((offR[l] & (SVW::s(l) - 1)) << PMW::shift(l));
  });
  {
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
  uint16_t* const wlistS = slist;
  uint16_t* const rlistS = slist + kListStride;

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
  //: IT IS PER WARP AND NOT PER CTA (K35 S2c): a warp expands exactly the rows of its own
  //: share `i % kWarps` of the union, so it stages exactly those, and the shares are
  //: disjoint -- which is what lets the fold's walk run with no barrier inside it at all.
  //: The LAYOUT is unchanged, so the two sides still share one body and one round trip.
  //: BODY COUNT: `kStgPass * D` = `D` at every built arm.
  constexpr int kStgCh = ceil_pow2c(BP::kStageChunks);
  constexpr int kStgRows = kBallotLanes / kStgCh;
  constexpr int kStgOwn = kSegRows / kWarps;
  constexpr int kStgPass = (kStgOwn + kStgRows - 1) / kStgRows;
  static_assert(kStgCh <= kBallotLanes, "a staged row's chunks fit one lockstep unit");
  auto stage_factors = [&](uint32_t blk, const __nv_bfloat16* plane, const uint16_t* list,
                           uint32_t winP, int slot, int u0, int nb) {
    const int c = lane & (kStgCh - 1);
    const uint32_t blkB = blk + (uint32_t)(slot * kSegRows * BP::kStageW * 2);
    static_for<kStgPass>([&](auto Pc) {
      const int ic = (lane >> ilog2c(kStgCh)) + decltype(Pc)::value * kStgRows;
      const int r = ic * kWarps + warp;
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
  //: Each warp IS a reader stream (`SVR::kStreams == kWarps`), so a batch's candidate
  //: pairs are `TPWR` union tokens times the readers, one per lane, and `kCandR` is the
  //: batch itself.
  constexpr int TPWR = kSegRows / kWarps;
  constexpr int kCandR = TPWR * kWarps;
  constexpr int kRdBits = ilog2c(kWarps);
  static_assert(kCandR == kSegRows && kCandR <= kBallotLanes,
                "one lane resolves each (union token, reader) pair of a batch");
  //: THE EXPANSION'S LANE MAP, the write side's at the READ carve's row width: a row is
  //: `kWordsRowR` packed pairs, the warp covers `kRowsPassR` rows at once and the rows of
  //: a pass are READERS of ONE token, which is what lets one memo serve the whole pass.
  constexpr int kWordsRowR = kLeafE / 2;
  constexpr int kLanesRowR = kWordsRowR < kBallotLanes ? kWordsRowR : kBallotLanes;
  constexpr int kRowsPassR = kBallotLanes / kLanesRowR;
  constexpr int kWordsLaneR = kWordsRowR / kLanesRowR;
  constexpr int kReadPass = kWarps / kRowsPassR;
  static_assert(kWarps % kRowsPassR == 0 && kReadPass >= 1,
                "a pass's rows are readers of ONE token, so the reader count carries them");
  //: THE READOUT'S TOKEN n-TILES PER BATCH, which is what the tile-live mask indexes.
  constexpr int kNTok = kSegRows / kMmaN;
  constexpr int kSeedR = first_whole_level<SVR, D>();
  const int exHR = lane >> ilog2c(kLanesRowR);
  const int exJR = lane & (kLanesRowR - 1);
  //: THE CONSUMER'S BASE INTO ITS OWN READER SEGMENT.  Its rows ARE the batch's ranks in
  //: its own compacted order, so this is an offset and not a map -- the same
  //: `[row][slot]` stride the deleted read panel had, one slot instead of two.
  const uint32_t myR = rsegB + (uint32_t)(((strm * kSegRows + rT) * LdR + cT * 8) * 2);

  //: ---- THE RING (P76 section 2.6-C), ITS ADDRESSES AND ITS PROTOCOL --------
  //: THE RING IS KEYED ON THE UNION RANK: rank `r` -- a token's position in the CTA's
  //: union read walk -- lives at slot `r % C`, and every reader of that token meets
  //: there instead of meeting in the output plane.  What that buys is stated as a count:
  //: the RED-direct form issued one global reduction per (owner warp, token, element),
  //: uncoalesced by construction; the ring issues ONE per (token, element) per CTA, from
  //: 32 lanes covering 32 consecutive channels, which is a 128-byte transaction.
  //:
  //: THE MEMORY ORDER, and it is the whole correctness argument.  A slot carries
  //: `{tag, arrived, expected}` beside its `kCols` floats.  `tag == r` is the ONLY
  //: statement that the row belongs to rank `r`, and it is published LAST: the
  //: reclaimer flushes and clears the row, writes `arrived = 0` and `expected`, fences
  //: (`publish_fence`, block scope -- the participants are warps of one CTA talking
  //: through shared memory and nothing wider is observable), and only then stores the
  //: tag.  A contributor that reads `tag == r` has therefore seen a cleared row.  In the
  //: other direction a contributor issues its `red.shared` first, fences, and only then
  //: increments `arrived`; so a reclaimer that reads `arrived == expected` has seen
  //: every contribution, and its CAS on the tag makes exactly one warp the flusher.
  //: TAGS ADVANCE MONOTONICALLY, so the first contributor to a rank is by construction
  //: its reclaimer and there is no separate zero pass -- the clear rides the flush,
  //: which is reading the row anyway.
  //:
  //: WHY THE SPIN IS PRESENT AND WHY IT IS NOT TAKEN HERE.  The protocol is correct for
  //: any `C >= kSegRows` with no help from any barrier: a warp that finds an INCOMPLETE
  //: stale row waits rather than evicting it, which is backpressure on the laggard's
  //: real work.  Under form (b) the readout's per-batch CTA barrier bounds the skew at
  //: one batch, so at `C = 32 = 2 * kSegRows` a slot's stale rank is two batches old and
  //: is always complete; the wait is the protocol's, not this schedule's.
  constexpr int RCC = G::kRingC;
  constexpr int LdRing = G::LdRing;
  constexpr uint32_t kReclaiming = 0x7FFFFFFFu;
  //: THE ROW'S GRANULE SWIZZLE, AND IT IS A DERIVATION.  A scatter instruction covers
  //: the accumulator's FOUR lane quads = four tokens = four ring rows, at ONE
  //: eight-channel granule; with `LdRing % 32 == 8` a row's granule occupies a whole
  //: eight-bank block and that block is `(row + granule) % 4`, so the four rows collide
  //: exactly when two of their ranks agree mod four.  Four-way is therefore the FLOOR
  //: for any row permutation -- there are four blocks and the rows are DATA (the union
  //: rank is the datum, and no compile-time pattern names it).  What a swizzle can buy
  //: is the DENSE case, which is the common one and the only one with a shape: a reader
  //: holding the whole batch scatters ranks `r, r+2, r+4, r+6`, whose blocks are
  //: `(r + 2q + swz)` -- two-way ALWAYS with no swizzle.  `swz(row) = (row >> 1) & 3`
  //: makes that sum `const + 3q (mod 4)`, a permutation of the four blocks at every
  //: alignment and both widths, and it stays a permutation INSIDE the row (it XORs the
  //: granule's low two bits), so the flush's own leg is untouched.
  auto ring_cell = [&](int slot, int g, int sub) {
    return ringB + (uint32_t)((slot * LdRing + (((g ^ ((slot >> 1) & 3))) << 3) + sub) * 4);
  };
  //: THE MASS rides the row's ONE PAD granule and must spread over its eight words on
  //: its own, the granule swizzle being one word wide per row there.
  auto ring_mass = [&](int slot) {
    return ringB + (uint32_t)((slot * LdRing + kDv + (slot & 7)) * 4);
  };
  //: THE FLUSH, AND IT IS THE ONLY GLOBAL REDUCTION THE READOUT ISSUES.  Thirty-two
  //: lanes cover thirty-two CONSECUTIVE output channels, so one issue is one 128-byte
  //: transaction against the RED-direct form's four 32-byte segments per issue, and the
  //: count is one per (token, element) per CTA rather than one per owner WARP.  The row
  //: is CLEARED in the same pass -- it is already being read, so the clear is a store on
  //: a value in a register and no zero pass exists -- and the whole thing is skipped for
  //: a generation that never held a rank (`old < 0`, the window's first `C` slots).
  //: BODY COUNT: `kDv / 32 + 1`.
  auto ring_flush = [&](int slot, int old, uint32_t winO_, uint32_t winD_) {
    const uint32_t tok = (uint32_t)rlistS[old >= 0 ? old : 0];
    const uint32_t rowO = tok * (uint32_t)(kDv * 4) + winO_;
    static_for<kDv / kBallotLanes>([&](auto Cc) {
      constexpr int c = decltype(Cc)::value;
      const uint32_t a = ring_cell(slot, c * 4 + (lane >> 3), lane & 7);
      const float v = load_shared_f32(a);
      if (old >= 0)
        fan_in_add(reinterpret_cast<float*>(
                       gather_ptr(oPlane, rowO + (uint32_t)((c * kBallotLanes + lane) * 4))),
                   v);
      store_shared_u32(a, 0u);
    });
    if (lane == 0) {
      const uint32_t a = ring_mass(slot);
      const float v = load_shared_f32(a);
      if (old >= 0) fan_in_add(reinterpret_cast<float*>(gather_ptr(denPlane, tok * 4u + winD_)), v);
      store_shared_u32(a, 0u);
    }
  };
  //: THE ACQUIRE.  A scatter instruction needs FOUR slots at once -- the four lane
  //: quads are four ranks -- so this makes all four say `tag == r` before any datum
  //: moves.  On the fast path that is ONE tag load and one vote; a quad that is not
  //: ready is served one at a time, because reclaiming means FLUSHING a row and the
  //: flush is the whole warp's.  A lane's rank is a function of `lane & 3` alone, so the
  //: leaders (`lane >> 2 == 0`) are the four ranks and the ballot is over them.
  auto ring_acquire = [&](int r, int exp, bool live, uint32_t winO_, uint32_t winD_) {
#pragma unroll 1
    for (;;) {
      const int t = live ? (int)load_shared_u32(tagB + (uint32_t)((r & (RCC - 1)) * 4)) : r;
      const uint32_t bad = lane_ballot(t != r && (lane >> 2) == 0);
      if (bad == 0u) break;
      const int q = first_set(bad);
      const int rq = lane_broadcast(r, q), eq = lane_broadcast(exp, q);
      const int sq = rq & (RCC - 1);
      int won = 0;
      if (lane == q) {
        const int told = (int)load_shared_u32(tagB + (uint32_t)(sq * 4));
        //: NEVER EVICT AN INCOMPLETE ROW: a laggard's contribution is still coming and
        //: the race-free form of waiting for it is to wait.  The CAS is what makes
        //: exactly one of the warps that find the row stale the flusher.
        if (told == rq - RCC
            && load_shared_u32(arrB + (uint32_t)(sq * 4))
                   == load_shared_u32(expB + (uint32_t)(sq * 4)))
          won = cas_shared_u32(tagB + (uint32_t)(sq * 4), (uint32_t)told, kReclaiming)
                        == (uint32_t)told
                    ? 1
                    : 0;
      }
      if (lane_broadcast(won, q) != 0) {
        publish_fence();
        ring_flush(sq, rq - RCC, winO_, winD_);
        store_shared_u32(arrB + (uint32_t)(sq * 4), 0u);
        store_shared_u32(expB + (uint32_t)(sq * 4), (uint32_t)eq);
        //: the row and the counters are published BEFORE the tag that claims them: a
        //: contributor reading `tag == r` has therefore seen a cleared row.
        publish_fence();
        store_shared_u32(tagB + (uint32_t)(sq * 4), (uint32_t)rq);
      }
    }
  };
  //: THE CONTRIBUTION.  One `red.shared` per element and ONE arrival per warp-row, in
  //: that order with a fence between: a reclaimer that reads `arrived == expected` has
  //: therefore seen every element.  The accumulator element `(m, e, lane)` carries
  //:     token   = p*kMmaN + 2*(lane & 3) + (e & 1)
  //:     channel = m*kMmaM + (lane >> 2) + 8*(e >> 1)
  //: so ONE issue covers four ring rows at eight consecutive channels, which is the
  //: access the row stride and the granule swizzle are derived from.
  //: BODY COUNT: `2 * MCH + 1` reductions and one arrival per (n-tile, token half).
  auto ring_scatter = [&](auto& acc, auto& macc, auto Ec, int r, int exp, bool live, uint32_t winO_,
                          uint32_t winD_) {
    constexpr int eb = decltype(Ec)::value;
    ring_acquire(r, exp, live, winO_, winD_);
    const int slot = r & (RCC - 1);
    if (live) {
      static_for<MCH>([&](auto Mc) {
        constexpr int m = decltype(Mc)::value;
        static_for<2>([&](auto Hc) {
          constexpr int h = decltype(Hc)::value;
          red_shared_add_f32(ring_cell(slot, 2 * m + h, lane >> 2), acc[m][2 * h + eb]);
        });
      });
      //: the mass rides row 0 of its own accumulator, so only the quad at `lane>>2 == 0`
      //: carries it -- and that is the same quad that counts the warp's arrival.
      if ((lane >> 2) == 0) red_shared_add_f32(ring_mass(slot), macc[eb]);
      publish_fence();
    }
    //: THE ARRIVAL IS LAST, AND IT IS ONE PER WARP-ROW.  A row's eight channel lanes
    //: fence their own reductions and the `__syncwarp` carries that fact to the lane
    //: that counts, so a reclaimer reading `arrived == expected` has seen all of them.
    __syncwarp();
    if (live && (lane >> 2) == 0) red_shared_add_u32(arrB + (uint32_t)(slot * 4), 1u);
  };

  // ---- the GLOBAL window grid ---------------------------------------------
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
    auto ballot_of = [&](const uint32_t* rowsS, auto Side, auto Vec, auto Mine, const int* off) {
      constexpr uint32_t SIDE = decltype(Side)::value;
      using SV = typename decltype(Vec)::type;
      constexpr bool MINE = decltype(Mine)::value;
      uint32_t m = valid;
      static_for<D>([&](auto Lc) {
        constexpr int l = decltype(Lc)::value;
        if constexpr (CR::row_g(l) > 0 && mode_declares(MO, l, SIDE)) {
          constexpr int RANGE = CR::rows_of(MINE ? SV::s(l) : BP::s(l), l);
          const uint32_t base =
              rowOff[l] + (MINE ? (uint32_t)(off[l] >> ilog2c(CR::s(l))) * rowBytes : 0u);
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
    const uint32_t wideR = ballot_of(rowsR, kRead, type_tag<SVR>{}, std::false_type{}, offR);
    const uint32_t wideW = ballot_of(rowsW, kWrite, type_tag<SVW>{}, std::false_type{}, offW);
    const uint32_t mineR = ballot_of(rowsR, kRead, type_tag<SVR>{}, std::true_type{}, offR);
    const uint32_t mineW = ballot_of(rowsW, kWrite, type_tag<SVW>{}, std::true_type{}, offW);
    //: THE WHOLE-WINDOW SKIP, at (owner, window) grain, per side -- the block-bitmap
    //: mechanism's harvest (study section 2.1).  It is CTA-uniform WITHOUT a
    //: barrier: the two owner-grain ballots are the same words in every warp.
    if (lane_ballot((wideR | wideW) != 0u) == 0u) continue;

    //: THE READ SIDE GETS THE SAME SHAPE AS THE WRITE SIDE.  Every stream walking the
    //: whole bitmap under its own ballot is one pass per stream over one window;
    //: instead ONE cooperative walk derives the OWNER's read ids and each stream PULLS
    //: its own subsequence.  It is a per-cell TRADE, not a win -- where a token reads
    //: leaves in several boxes the owner's union is nearly the sum of the per-stream
    //: lists, so the shared walk saves nothing and the pull is added work.
    constexpr int kSliceR = kBallotLanes / kWarps;
    const int nrO = walk(wideR, rlistS, (uint32_t)(((1ull << kSliceR) - 1ull) << (warp * kSliceR)),
                         (uint32_t)((1ull << (warp * kSliceR)) - 1ull));
    //: THE OWNER'S WRITE IDS, DERIVED ONCE FOR THE CTA.  `walk` is already the cheap
    //: form -- one lane per ballot word, `ffs` per live bit -- and its `bits`/`below`
    //: parameters exist for exactly this: several writers deriving the SAME global
    //: prefix and filling disjoint slices of every word, so the list comes out in the
    //: one ascending order every consumer expects.  The CTA wall is here and nowhere
    //: else in the fold.
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
    //: AND THE READ SIDE PUBLISHES THE SAME TWO WORDS (K35 S4b), per READER -- which is
    //: per WARP, the read carve being declared at one stream per executor -- because the
    //: readout's fetch places rows into segments the placing warp does not consume, in
    //: exactly the write side's shape.
    if (inW && col == 0) store_shared_u32(memB + (uint32_t)((wstrm * LW + lane) * 4), mineW);
    if (inW) store_shared_u32(memrB + (uint32_t)((strm * LW + lane) * 4), mineR);
    {
      const int cnt = bit_count(mineW);
      const int pfx = lane_prefix_incl<LW>(cnt) - cnt;
      if (inW && col == 0)
        store_shared_u16(pfxB + (uint32_t)((wstrm * LW + lane) * 2), (uint16_t)pfx);
    }
    {
      const int cnt = bit_count(mineR);
      const int pfx = lane_prefix_incl<LW>(cnt) - cnt;
      if (inW) store_shared_u16(pfxrB + (uint32_t)((strm * LW + lane) * 2), (uint16_t)pfx);
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
    publish_read(lo, mlo);
    rendezvous();
    uint32_t sa[KDw][MCH][4], ma[KDw][4];
    pull_read(sa, ma);
    //: AND THE SEGMENTS ARE DEAD HERE -- for THIS warp, the moment it has pulled.
    //: At `PK == 0` the readout's tenants are these same bytes under another layout, so
    //: no warp may produce into them until EVERY warp has pulled: that barrier is the
    //: price of the straddle.  At `PK == 1` the park holds the bytes, the readout's
    //: tenants sit above them, and the park is warp-private -- so the barrier is spent
    //: at the far end instead.
    if constexpr (G::kPark) {
      park_lo(lo, mlo);
    } else {
      rendezvous();
    }

    // ---- PHASE 1: THE READOUT of every UNION read batch, against the window's ENTRY
    //      state.  ALL of the window's readouts precede ANY of its folds, because a
    //      within-window pair is the INTRA term's and this kernel must not carry it.
    //      The batch line is therefore the window line on both sides.
    //: THE READ SIDE IS NOW THE WRITE SIDE'S MIRROR (K35 S4b).  A BATCH is `kSegRows`
    //: consecutive tokens of the CTA's UNION read list; executor `warp` takes the fixed
    //: share `i % kWarps` of them and, for each READER that reads a token it handles,
    //: expands that reader's `R` row -- `kLeafE` bf16 in the reader's private order, no
    //: gain -- into that reader's segment at `rank(t, reader)`.  The consumer does NO
    //: produce: tile skip -> `ldmatrix` -> HMMA -> ring.  Two stages in flight with the
    //: registers as stage two, one CTA barrier per batch each way, exactly as the fold.
    const int nbrO = (nrO + kSegRows - 1) / kSegRows;
    auto rbatch_n = [&](int b) {
      const int r = nrO - b * kSegRows;
      return r <= 0 ? 0 : (r < kSegRows ? r : kSegRows);
    };
    //: `rank_live(reader, t)` -- the token's position in that reader's compacted list,
    //: and whether the reader holds it at all, out of the published ballot and prefix.
    auto rank_live_r = [&](int s, int t) {
      const int w = t >> 5;
      const uint32_t word = load_shared_u32(memrB + (uint32_t)((s * LW + w) * 4));
      const int pre = (int)load_shared_u16(pfxrB + (uint32_t)((s * LW + w) * 2));
      const uint32_t below = (1u << (t & 31)) - 1u;
      return ((pre + bit_count(word & below)) << 1) | (int)((word >> (t & 31)) & 1u);
    };
    //: THE BATCH'S CANDIDATE PAIRS, RESOLVED ONCE.  Lane `q` owns candidate
    //: `(i, reader)`; every lane that fills that row reads the placement by shuffle, so
    //: the four shared loads of a rank are paid once per pair.  `-1` is "this reader does
    //: not read this token"; otherwise the word is `expected | rank | union index`, and
    //: it READS shared memory and writes none -- which is why it may run before the
    //: barrier that releases the fetchers, exactly as the fold's placement does.
    auto rbatch_place = [&](int u0, int nb) {
      const int i = lane >> kRdBits, s = lane & (kWarps - 1);
      const int ii = i * kWarps + warp;
      int rank = 0;
      bool has = false;
      if (lane < kCandR && ii < nb) {
        const int rl = rank_live_r(s, (int)rlistS[u0 + ii]);
        if (rl & 1) {
          has = true;
          rank = (rl >> 1) - (rank_live_r(s, (int)rlistS[u0]) >> 1);
        }
      }
      //: THE EXPECTED COUNT IS ONE POPCOUNT OVER THE VOTE THE PLACEMENT ALREADY HELD.
      //: A token's `kWarps` candidate lanes are contiguous, so the number of readers
      //: that will arrive at that token's ring row is a field of the ballot -- and every
      //: reader's row gets the SAME number, because they all read the one vote.
      const uint32_t bal = lane_ballot(has);
      const int exp = bit_count((bal >> (i * kWarps)) & ((1u << kWarps) - 1u));
      return has ? ((exp << 24) | (rank << 16) | ii) : -1;
    };
    //: THE ROW INDEX, PUBLISHED WITH THE EXPANSION AND NOT BEFORE IT.  The reader's
    //: private row keeps its token's UNION INDEX inside the batch and the expected
    //: arrival count, in one halfword, so the consumer turns a segment row back into the
    //: CTA-wide rank the ring is keyed on with ONE shared load.  It is a WRITE into
    //: another warp's tenant, so it belongs on the fetch side of the drained barrier
    //: beside the rows it indexes -- the consumer is still reading the previous batch's
    //: index until then.
    auto rmap_publish = [&](int e) {
      if (e >= 0)
        store_shared_u16(
            rmapB + (uint32_t)(((lane & (kWarps - 1)) * kSegRows + ((e >> 16) & 0xFF)) * 2),
            (uint16_t)((((e >> 24) & 0xFF) << 8) | (e & 0xFF)));
    };
    //: THE BATCH'S ROW COUNT FOR THIS EXECUTOR'S OWN READER, out of the same published
    //: ballot: the union list contains every reader's tokens in ascending order, so
    //: every token of reader `strm` between the batch's first and last union token IS in
    //: the batch and the count is the difference of the two ranks.
    auto rbatch_rows = [&](int u0, int nb) {
      int n = 0;
      if (lane == 0 && nb > 0) {
        const int rl = rank_live_r(strm, (int)rlistS[u0 + nb - 1]);
        n = (rl >> 1) + (rl & 1) - (rank_live_r(strm, (int)rlistS[u0]) >> 1);
      }
      return lane_broadcast(n, 0);
    };
    //: THE READ SIDE'S EXPANSION.  DECLARED ARITHMETIC ORDER, per packed slot pair:
    //:     x f_l for every level the READ side SPANS WHOLE, ascending `l`,
    //:     then x f_l for every level it CARVES,            ascending `l`,
    //: each product rounded to nearest even in bf16.  The prefix through the
    //: whole-spanned levels is the MEMO: it does not depend on the reader, so it is
    //: minted once per (token, lane) and reused by every reader of that token.  THERE IS
    //: NO GAIN ON THIS SIDE -- its scalar is identically one -- so the first factor of
    //: the order IS the product's seed and no multiply by one is ever issued.
    //: BODY COUNT: `TPWR * (1 + kWordsLaneR * U)` for the memo and
    //: `TPWR * kReadPass * kWordsLaneR * (Cv + 2)` for the rows, `U`/`Cv` being the
    //: whole-spanned and carved level counts of the READ span vector.
    auto expand_read = [&](int slot, int e0) {
      const uint32_t blkB = rstgB + (uint32_t)(slot * kSegRows * BP::kStageW * 2);
      static_for<TPWR>([&](auto Ic) {
        constexpr int ic = decltype(Ic)::value;
        const int ii = ic * kWarps + warp;
        const uint32_t rowB = blkB + (uint32_t)(ii * BP::kStageW * 2);
        uint32_t memo[kWordsLaneR];
        static_for<kWordsLaneR>([&](auto Wc) {
          constexpr int wl = decltype(Wc)::value;
          const int p0 = 2 * (exJR + wl * kLanesRowR);
          uint32_t a = kOnesPair;
          static_for<D>([&](auto Lc) {
            constexpr int l = decltype(Lc)::value;
            if constexpr (!SVR::carves(l)) {
              const uint32_t f = factor_at(type_tag<PM>{}, rowB, Lc, p0, 0);
              if constexpr (l == kSeedR)
                a = f;
              else
                a = mul_bf16x2(a, f);
            }
          });
          memo[wl] = a;
        });
        static_for<kReadPass>([&](auto Pc) {
          constexpr int pp = decltype(Pc)::value;
          const int s = pp * kRowsPassR + exHR;
          const int e = lane_broadcast(e0, ic * kWarps + s);
          const int rank = (e >> 16) & 0xFF;
          const uint32_t rowR = rsegB + (uint32_t)(((s * kSegRows + rank) * LdR) * 2);
          uint32_t any = 0u;
          static_for<kWordsLaneR>([&](auto Wc) {
            constexpr int wl = decltype(Wc)::value;
            const int jw = exJR + wl * kLanesRowR;
            uint32_t a = memo[wl];
            static_for<D>([&](auto Lc) {
              constexpr int l = decltype(Lc)::value;
              if constexpr (SVR::carves(l))
                a = mul_bf16x2(a, factor_at(type_tag<PM>{}, rowB, Lc, 2 * jw,
                                            stream_off(type_tag<SVR>{}, s, Lc)));
            });
            if (e >= 0) store_shared_u32(rowR + (uint32_t)(jw * 4), a);
            //: THE PER-ROW TILE-LIVE MASK, produced BY the expansion because only the
            //: expansion knows which slots it wrote nonzero.  On the READOUT the
            //: n-tile is EIGHT TOKENS, so the bit a row sets is `rank >> 3` and the OR
            //: over the row's own lanes is what decides it.
            const uint32_t bal = lane_ballot(e >= 0 && (a & kBf16MagnitudeMask) != 0u);
            any |= (bal >> (exHR * kLanesRowR)) & (uint32_t)((1ull << kLanesRowR) - 1ull);
          });
          if (exJR == 0 && any != 0u)
            or_shared_u32(mskB + (uint32_t)(s * 4), 1u << (rank >> ilog2c(kMmaN)));
        });
      });
    };
    //: THE BATCH'S DEAD ROWS, and the law is the fold's (S2b): a row the MMA reads is
    //: never arbitrary shared memory.  The union list ascends, so a reader's own tokens
    //: inside a batch are a PREFIX of its segment and the dead rows are `[n, kSegRows)`;
    //: a reader owns its whole segment here (one stream per warp), so the clear is a row
    //: split and needs no column reasoning.  It runs between the drain and the barrier
    //: that releases the fetchers, and it resets THIS reader's mask word with it.
    //: BODY COUNT: `ceil(kSegRows * kLeafE * 2 / 512)` predicated stores.
    //: THE MASK'S RESET IS NOT PART OF IT, and that is deliberate: the clear runs on the
    //: FETCH side of the drained barrier in the prologue and on the CONSUMER side inside
    //: the loop, so a reset carried here would race the fetchers' `red.shared.or` in the
    //: prologue.  A warp resets its own word where it has just READ it instead.
    constexpr int kClrR = kLeafE * 2 / 16;
    static_assert(kClrR * 16 == kLeafE * 2 && (kClrR & (kClrR - 1)) == 0,
                  "a reader's coefficient row is a power-of-two number of quanta");
    constexpr int kDeadR = kSegRows * kClrR;
    auto clear_dead_read = [&](int n) {
      const uint4 z = make_uint4(0u, 0u, 0u, 0u);
      const int q = (kSegRows - n) * kClrR;
      const uint32_t cB = rsegB + (uint32_t)(((strm * kSegRows + n) * LdR) * 2);
      static_for<(kDeadR + kBallotLanes - 1) / kBallotLanes>([&](auto Pc) {
        constexpr int pp = decltype(Pc)::value;
        const int qi = pp * kBallotLanes + lane;
        if (qi < q)
          store_shared_vec16(
              cB + (uint32_t)((qi >> ilog2c(kClrR)) * LdR * 2 + (qi & (kClrR - 1)) * 16), z);
      });
    };
    //: THE RING'S GENERATION ZERO, AND ALL THREE WORDS OF IT.  Slot `i`'s tag starts at
    //: `i - C`, which is exactly what the first rank to land there will find stale, and
    //: it is NEGATIVE -- so the reclaim that claims it clears the row and issues no
    //: global reduction.  That is the whole of "no zero pass": the first generation's
    //: flush IS the zero pass.  `arrived` and `expected` start EQUAL (at zero) because
    //: the reclaim's completeness test reads them: generation zero has no contributors,
    //: so it is complete, and leaving them as whatever the previous moment's tenant left
    //: in those bytes is a spin that never ends.
    auto ring_init = [&]() {
#pragma unroll 1
      for (int i = tid; i < RCC; i += kThreads) {
        store_shared_u32(arrB + (uint32_t)(i * 4), 0u);
        store_shared_u32(expB + (uint32_t)(i * 4), 0u);
        store_shared_u32(tagB + (uint32_t)(i * 4), (uint32_t)(i - RCC));
      }
    };
    //: THE DRAIN, at the window's phase barrier: the last `C` rows have no next writer,
    //: so each executor flushes the complete slots of its share.  A claimed row short of
    //: its expected arrivals is a LOST CONTRIBUTION and there is nothing to do with one
    //: but stop -- that is an assertion, not a path.
    constexpr int kDrainSlots = RCC / kWarps;
    static_assert(kDrainSlots * kWarps == RCC, "the ring's rows split over the executors");
    auto ring_drain = [&](uint32_t winO_, uint32_t winD_) {
      static_for<kDrainSlots>([&](auto Sc) {
        const int slot = warp * kDrainSlots + decltype(Sc)::value;
        const int told = (int)load_shared_u32(tagB + (uint32_t)(slot * 4));
        if (told >= 0 && lane == 0
            && load_shared_u32(arrB + (uint32_t)(slot * 4))
                   != load_shared_u32(expB + (uint32_t)(slot * 4)))
          __trap();
        ring_flush(slot, told, winO_, winD_);
      });
    };

    if (nbrO > 0) {
      ring_init();
      if (lane == 0) store_shared_u32(mskB + (uint32_t)(strm * 4), 0u);
      stage_factors(rstgB, pR, rlistS, winP, 0, 0, rbatch_n(0));
      stage_commit();
      stage_wait<0>();
      rendezvous();
      {
        const int e0 = rbatch_place(0, rbatch_n(0));
        clear_dead_read(rbatch_rows(0, rbatch_n(0)));
        stage_factors(rstgB, pR, rlistS, winP, 1, kSegRows, rbatch_n(1));
        stage_commit();
        rmap_publish(e0);
        expand_read(0, e0);
      }
#pragma unroll 1
      for (int b = 0; b < nbrO; ++b) {
        //: LANDED.  Every fetcher's expansion for batch `b` has been written and the
        //: barrier makes every reader's rows visible to the reader that consumes them.
        stage_wait<0>();
        rendezvous();
        const uint32_t tm = load_shared_u32(mskB + (uint32_t)(strm * 4));
        if (lane == 0) store_shared_u32(mskB + (uint32_t)(strm * 4), 0u);
        const int ns = rbatch_rows(b * kSegRows, rbatch_n(b));
        //: THE ROW INDEX INTO REGISTERS, DURING THE DRAIN.  A lane's four ranks are the
        //: rows `p*8 + 2*(lane&3) + e` and the two halfwords of a row pair are one word,
        //: so the whole index a lane needs is TWO shared loads -- and taking them here
        //: is what lets the fetchers overwrite the index for batch `b+1` while this
        //: batch's scatter is still running.
        const uint32_t rm0 =
            load_shared_u32(rmapB + (uint32_t)((strm * kSegRows + 2 * (lane & 3)) * 2));
        const uint32_t rm1 =
            load_shared_u32(rmapB + (uint32_t)((strm * kSegRows + kMmaN + 2 * (lane & 3)) * 2));
        uint32_t rb[KDw][4];
        if (tm != 0u)
          static_for<KDw>([&](auto Kc) {
            constexpr int kk = decltype(Kc)::value;
            load_frag(rb[kk], ADDR_R(myR, 0, 2 * kk));
          });
        const int n1 = rbatch_n(b + 1);
        const int e1 = rbatch_place((b + 1) * kSegRows, n1);
        clear_dead_read(rbatch_rows((b + 1) * kSegRows, n1));
        //: DRAINED.  The `ldmatrix` above has read the slot and the next batch's dead
        //: rows are zero, so the fetchers may refill.
        __syncwarp();
        rendezvous();
        stage_factors(rstgB, pR, rlistS, winP, b & 1, (b + 2) * kSegRows, rbatch_n(b + 2));
        stage_commit();
        rmap_publish(e1);
        expand_read((b + 1) & 1, e1);
        //: AND THE MMAs RUN FROM FRAGMENTS, underneath the copies just issued.  THE
        //: TILE-LIVE MASK GATES THE MMAs AND THE MASS, AND NOTHING ELSE: the scatter is
        //: gated by `rank < n`, because `expected` counts the readers that HOLD the
        //: token and a reader whose tile is dead by VALUE still has to arrive -- it adds
        //: the zero its accumulator holds.  Gating the scatter on the mask would leave
        //: `arrived < expected` standing and the next reclaimer would wait forever.
        //: BODY COUNT: `KDw * (MCH + 1)` MMAs per token n-tile.
        const int bBase = b * kSegRows;
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
          if (tm & (1u << p))
            static_for<KDw>([&](auto Kc) {
              constexpr int kk = decltype(Kc)::value;
              static_for<MCH>([&](auto Mc) {
                constexpr int m = decltype(Mc)::value;
                mma(acc[m], sa[kk][m], &rb[kk][2 * p]);
              });
              mma(macc, ma[kk], &rb[kk][2 * p]);
            });
          const uint32_t rmw = p ? rm1 : rm0;
          static_for<2>([&](auto Ec) {
            constexpr int eb = decltype(Ec)::value;
            const int h = (int)((rmw >> (16 * eb)) & 0xFFFFu);
            ring_scatter(acc, macc, Ec, bBase + (h & 0xFF), h >> 8,
                         (p * kMmaN + 2 * (lane & 3) + eb) < ns, winO, winD);
          });
        });
      }
      //: THE WINDOW'S EDGE ON THIS SIDE.  Drain the copies first, then a barrier that
      //: says every contribution has landed, and only then flush the ring's last rows.
      stage_wait<0>();
      rendezvous();
      ring_drain(winO, winD);
    }

    //: THE WINDOW LINE, and it is there by the ordering law: no warp may fold this
    //: window's writes while another is still reading the window's entry state -- and
    //: the fold's segments are the ring's very bytes.
    rendezvous();

    //: THE PHASE CHANGE, READOUT -> FOLD: the mirror.  Each warp publishes its
    //: read-view hi -- which is exactly what its A fragments hold -- into the WRITE
    //: executors' segments; a barrier; each warp pulls its own back in WRITE order and
    //: rejoins it with the `lo` it kept, exactly.  The last barrier is the fold's: the
    //: segments it clears are these very bytes.
    //: THE UNPARK PRECEDES THE PUBLISH, and that order is the whole protocol at
    //: `PK == 1`: a publisher writes into the CONSUMER's segment, where that consumer's
    //: `lo` is parked, so every warp takes its own back before ANY warp publishes.
    if constexpr (G::kPark) {
      unpark_lo(lo, mlo);
      rendezvous();
    }
    publish_write(sa, ma);
    rendezvous();
    reload_hi(lo, mlo);
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
    constexpr int TPW = kSegRows / kWarps;  // union tokens one warp handles per group
    static_assert(TPW * kWarps == kSegRows,
                  "the group's union tokens split evenly over the executors");
    constexpr int kSegBits = ilog2c(kSegRows);
    constexpr int kStrBits = ilog2c(kStreams);
    //: THE EXECUTOR'S CANDIDATE PAIRS: its `TPW` union tokens times the WRITE streams,
    //: one per lane.  At `WPS == 1` this is the whole batch (`kStreams == kWarps`);
    //: under a column split the executors are more numerous and each holds fewer.
    constexpr int kCand = TPW * kStreams;
    static_assert(kCand <= kBallotLanes, "one lane resolves each candidate pair");
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
    //: the value row's copy map: `kChunkV` sixteen-byte chunks, `kValRows` rows a pass.
    constexpr int kChunkV = kDv / 8;
    constexpr int kValRows = kBallotLanes / kChunkV;
    constexpr int kValPass = kCand / kValRows;
    static_assert(kValPass * kValRows == kCand, "the candidate pairs split evenly");
    const int exH = lane >> ilog2c(kLanesRow);  // which row of the pass this lane serves
    const int exJ = lane & (kLanesRow - 1);     // its packed pair inside that row
    const int vlC = lane & (kChunkV - 1);       // its sixteen-byte chunk of a value row
    const int vlR = lane >> ilog2c(kChunkV);    // which row of the value pass
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

    //: `rank_live(s, t)` -- the token's position in stream `s`'s compacted list, and
    //: whether the stream holds it at all, out of the published ballot and prefix.
    auto rank_live = [&](int s, int t) {
      const int w = t >> 5;
      const uint32_t word = load_shared_u32(memB + (uint32_t)((s * LW + w) * 4));
      const int pre = (int)load_shared_u16(pfxB + (uint32_t)((s * LW + w) * 2));
      const uint32_t below = (1u << (t & 31)) - 1u;
      return ((pre + bit_count(word & below)) << 1) | (int)((word >> (t & 31)) & 1u);
    };
    //: THE GROUP'S CANDIDATE PAIRS, RESOLVED ONCE.  Lane `q < kCand` owns candidate
    //: `q = (i, s)` and every other lane reads it by shuffle, so a rank's two shared
    //: loads are paid once per pair.  `-1` is "this stream does not want this token".
    //: THE RANK IS THE STREAM'S RUNNING RANK IN THE WINDOW, not a group-local one: the
    //: segment row is `rank mod kSegRows` and the TILE is `rank >> log2 kSegRows`, which
    //: is what makes a stream's rows accumulate across the walk.
    auto place_group = [&](int u0, int nb) {
      int e = -1;
      if (lane < kCand) {
        const int i = lane >> kStrBits, s = lane & (kStreams - 1);
        const int ii = i * kWarps + warp;
        if (ii < nb) {
          const int t = (int)wlistS[u0 + ii];
          const int rl = rank_live(s, t);
          if (rl & 1) e = ((rl >> 1) << 16) | t;
        }
      }
      return e;
    };
    //: `drained[strm]` has exactly ONE writer, this warp, so it is a plain store of a
    //: register the warp already holds; the room test takes the MINIMUM over the `WPS`
    //: executors of the stream being written into -- at `WPS == 1` one load.
    auto drained_min = [&](int s) {
      uint32_t m = 0xFFFFFFFFu;
      static_for<WPS>([&](auto Cc) {
        const uint32_t d =
            load_shared_u32(flowB + (uint32_t)((kWarps + s * WPS + decltype(Cc)::value) * 4));
        m = d < m ? d : m;
      });
      return m;
    };
    //: THE ARRIVAL, ONE `red.shared.add.u32` PER STREAM THE GROUP TOUCHED.  A stream's
    //: candidate lanes are those congruent to `s` modulo `kStreams`, so the count is a
    //: popcount of the placement's own ballot against a compile-time stride pattern and
    //: no lane iterates.
    constexpr uint32_t kStrPat = (uint32_t)(((1ull << kCand) - 1ull) / ((1u << kStreams) - 1u));
    auto publish_landed = [&](int e) {
      const uint32_t bal = lane_ballot(e >= 0);
      if (lane < kStreams) {
        const int c = bit_count(bal & (kStrPat << lane));
        if (c != 0) red_shared_add_u32(flowB + (uint32_t)(lane * 4), (uint32_t)c);
      }
    };
    //: THE VALUE LEG.  `cp.async` never touches registers, and the fetcher issues it
    //: into a segment it does not consume -- the fan-out pipeline's first owner.
    //: BODY COUNT: `kValPass` = `kCand * kChunkV / 32` -- 4 where the executors are the
    //: streams, 2 at the four-stream column-split arm, DOUBLE either at `d_v = 128` --
    //: each body one predicated issue.  This is the one nest the width reaches.
    auto fetch_values = [&](int e0) {
      static_for<kValPass>([&](auto Pc) {
        constexpr int pp = decltype(Pc)::value;
        const int q = pp * kValRows + vlR;
        const int e = lane_broadcast(e0, q);
        if (e >= 0) {
          const int s = q & (kStreams - 1);
          stage_tile(
              segVB
                  + (uint32_t)((((s * kSegRows + ((e >> 16) & (kSegRows - 1))) * LdV) + vlC * 8)
                               * 2),
              gather_ptr(vPlane, (uint32_t)(e & 0xFFFF) * (uint32_t)v_row_bytes + winV
                                     + (uint32_t)(vlC * 8 * 2)));
        }
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
        if (lane == 0) store_shared_u32(flowB + (uint32_t)((kWarps + strm) * 4), drained);
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
      //: THE FULL-TILE TEST, the executor's whole schedule: tile `drained` is full when
      //: the stream's arrivals reach its far edge, and once sealed the last one joins it.
      auto ready = [&]() {
        const uint32_t l = load_shared_u32(flowB + (uint32_t)(wstrm * 4));
        if (l >= ((drained + 1) << kSegBits)) return true;
        return sealed && drained < ((l + kSegRows - 1) >> kSegBits);
      };
      clear_segment();
      //: THE COUNTERS' ZERO -- they are monotone WITHIN a window, and the barrier that
      //: publishes the clear publishes these with it.
      if (lane == 0) {
        store_shared_u32(flowB + (uint32_t)((kWarps + strm) * 4), 0u);
        if (strm < kStreams) store_shared_u32(flowB + (uint32_t)(strm * 4), 0u);
        if (strm == 0) store_shared_u32(flowB + (uint32_t)(2 * kWarps * 4), 0u);
      }
      stage_factors(stgB, pW, wlistS, winP, 0, 0, group_n(0));
      stage_commit();
      gain_store(0, gain_load(0, group_n(0)));
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
        if (ready()) {
          pull_and_fold();
          continue;
        }
        if (sealed) break;
        if (e == -2) {
          if (g >= nGroup) {
            //: THE WALK'S EDGE.  This warp's share is placed; announce it and keep
            //: servicing until every warp has announced, because a warp parked at a
            //: barrier drains nothing and another warp may still wait on a tile this
            //: executor owns.
            if (g == nGroup) {
              stage_wait<0>();
              if (lane == 0) red_shared_add_u32(flowB + (uint32_t)(2 * kWarps * 4), 1u);
              g = nGroup + 1;
            }
            if (load_shared_u32(flowB + (uint32_t)(2 * kWarps * 4)) < (uint32_t)kWarps) continue;
            //: THE WINDOW'S EDGE.  One barrier, reached exactly once by every thread --
            //: after it `landed[s] == n_s` is final.  AT MOST ONE TILE IS LEFT UNDRAINED
            //: and it is the window's last, because a fetcher cannot write tile `k+1`
            //: before every executor has drained tile `k`.  Its dead coefficient rows
            //: are zeroed HERE and nowhere else.
            rendezvous();
            const int nMine = (int)load_shared_u32(flowB + (uint32_t)(wstrm * 4));
            const int part = nMine & (kSegRows - 1);
            if (part != 0 && drained < (uint32_t)((nMine + kSegRows - 1) >> kSegBits))
              clear_dead(part);
            sealed = true;
            continue;
          }
          //: this warp's factor rows for the group have landed (issued one group ago).
          stage_wait<0>();
          e = place_group(g << kSegBits, group_n(g));
        }
        //: THE GROUP'S PLACEMENTS, IN ROUNDS OF WHAT THERE IS ROOM FOR.  A candidate
        //: whose tile is full and not yet drained by every executor of its stream waits;
        //: the rest are placed and PUBLISHED first.  The round IS the deadlock-freedom
        //: invariant: a group can straddle a stream's tile boundary, so an
        //: all-or-nothing group would have a warp wait on rows it is itself holding
        //: back.  At most two rounds, one in the ordinary case.
        const int er =
            (e >= 0 && (uint32_t)(e >> (16 + kSegBits)) <= drained_min(lane & (kStreams - 1))) ? e
                                                                                               : -1;
        if (lane_ballot(er >= 0) != 0u) {
          fetch_values(er);
          stage_commit();
          expand(g & 1, er);
          //: LANDED.  The value copies have arrived, the expansion's rows and mask bits
          //: are stored, and the fence orders both before the arrival that publishes
          //: them: NOTHING THIS WARP ISSUED IS UNPUBLISHED WHEN IT WAITS.
          stage_wait<0>();
          publish_fence();
          publish_landed(er);
          if (er >= 0) e = -1;
        }
        if (lane_ballot(e >= 0) != 0u) continue;
        //: THE NEXT GROUP'S FACTORS AND GAINS, one group ahead, so the folds at the
        //: loop's head run under them.
        const int n1 = group_n(g + 1);
        const unsigned short gn = gain_load((g + 1) << kSegBits, n1);
        stage_factors(stgB, pW, wlistS, winP, (g + 1) & 1, (g + 1) << kSegBits, n1);
        stage_commit();
        gain_store((g + 1) & 1, gn);
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
