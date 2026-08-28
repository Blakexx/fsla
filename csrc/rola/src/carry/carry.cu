// csrc/rola/src/carry/carry.cu -- K31 R1: the carry family's ONE translation unit.
//
// Dispatch, torch entries and the built arm list.  Each arm DECLARES its two carves
// -- the per-level span vector one STREAM owns on the write side and on the read
// side -- so the carve is a property of the arm and never a constant of a kernel,
// and the seam's `carve` scalar survives only to refuse a carve nothing builds.
//
// The arms (`s` = the owner box, `SW`/`SR` = the two stream boxes):
//   flagship   (256,256)  D=2  k=4 m=8   s=(8,16)     BC=128  W=384  C=32
//   flagship+  (256,256)  D=2  k=4 m=8   s=(8,16)     BC=128  W=512  C=32
//   small      ( 64, 64)  D=2  k=4 m=8   s=(8,16)     BC=128  W=64   C=32
//   deep-3     (16^3)     D=3  k=2 m=16  s=(2,4,16)   BC=128  W=64   C=32
//   deep-4     ( 8^4)     D=4  k=2 m=8   s=(2,2,4,8)  BC=128  W=64   C=32
//   carve-4    ( 4^4)     D=4  k=4 m=1   s=(4,4,4,4)  BC=256  W=64   C=32
//   flagship 1x8 (256,256) D=2  k=4 m=8   s=(8,16)     BC=128  W=384  C=32, twice
//   flagship 1x8 d_v=128   the SAME two, at the wide value plane
//   flagship 2x4 d_v=128 (256,256) D=2 k=4 m=4 s=(4,16) BC=64  W=384  C=32
//   + the PARK's twins (K35 S5): those three and the flagship d_v=64 ALT row again
//   flagship tied (256,256) D=2 k=4 m=8   s=(8,16)     BC=128  W=384  C=32, MO=3
// The first five hold ONE box of `BC = 128` -- the register budget's number, which
// does not move with depth -- so they differ only in how the same 128 leaves cut
// into digits, and `deep-4` is the topology whose atom rectangle cannot be a single
// level (`B = 8` is narrower than an atom).  `carve-4` is the MULTI-LEVEL CARVE arm
// and the one whose owner is the whole topology: its 256 leaves are cut into EIGHT
// streams of 32 by `SW = [1,4,2,4]` and `SR = [4,1,4,2]`, two carved levels per
// side, one warp per stream -- so it is also the arm that runs eight executors.
// `k = 4, m = 1` is the only capacity reaching the uniform box `(4,4,4,4)`.
// The LAST row is the TIED arm: the flagship cell declared BOTH-SPARSE at level zero,
// where the order-by-mode rule gives the two sides the SAME order and the derivation the
// SAME vector, so the exchange is the identity and is elided at compile time.
//
// `d_v` IS THE THIRD DIAL AND A KEY OF THE ARM (K35 S3b).  It is a template parameter
// of both bodies, so the `d_v = 128` rows are not a wider path but the same body at a
// wider declaration, and the seam tells them apart by the width the caller's value
// plane carries.  The card's CONTAINERS entry rules what the width costs: sixteen
// leaves per warp is REQUIRED at 128, and eight resident warps per SM is the target, so
// the two ways to spend them are 1x8 at `BC = 128` (rows 13/14, the two container
// declarations at the wide plane) and 2x4 at `BC = 64` (row 15, the CTA's box halved
// instead of its warps doubled -- four streams, one warp each, the ruled leaf split).
// The `d_v = 64` arms are UNTOUCHED by all of it, which is the point of making the
// width a parameter rather than a branch.
//
// THE EXECUTOR MAP IS THE SECOND DIAL (K35 S3a).  An arm declares `WPS`, the warps per
// WRITE stream, so the CTA's warp count is `streams x WPS` and the READ carve is
// declared at one stream per warp.  The two `flagship 1x8` rows are the SAME cell as
// `flagship ALT` reached with EIGHT executors instead of four, by the two available
// routes -- four write streams column-split two ways (`SW = [2,16]`, `WPS = 2`) and
// eight write streams one warp each (`SW = [1,16]`, `WPS = 1`) -- so the container
// question the card leaves to measurement has both of its answers built.  The existing
// four-warp `ALT` arms ARE the 2x4 container: two CTAs of four warps per SM at half the
// SM's box each, which is the same eight resident warps, so no arm is added for it.
// The window is the architecture doc's TWO-CONSTRAINT quantity (integration rung
// I1(b)): the C* cost basin is shallow, and the intra kernel -- which runs on the
// SAME window grid -- has ideal arithmetic intensity `W/4` flop/byte and needs
// `W >= ~380` to clear sm_86's compute/bandwidth balance.  The doc's admitted band
// is 384-512 and BOTH ends are built, because a third constraint decides between
// them: the intra refuses a window that does not divide `L`, and 384 does not
// divide the flagship length `L = N = 65536`.  512 does, at every power-of-two
// length, while clearing the intensity bar with room (128 flop/byte).  The
// `(64, 64)` arm is the conformance cell: the fp64 oracle materializes
// `[B,T,H,N]`, so `N = 4096` is where a whole cell is checkable, and its `W = 64`
// is held to keep that cell's window count meaningful at the lengths a reference
// can hold.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <array>

#include "carry_api.cuh"
#include "carry_probe_kernel.cuh"
#include "carry_split_kernel.cuh"
#include "facts/facts_kernel.cuh"

//: THE OWNER BALLOT ROWS, DECLARED (K31-i-d).  The facts family owns the pass and
//: its instantiation shard owns the arms; this TU sees a DECLARATION, exactly as the
//: facts dispatch TU does, so nothing here can satisfy an arm locally.  The SPANS the
//: carry asks for are `BoxPlan::s(l)` -- the lattice's balanced owner runs, which is
//: the vocabulary the lex plan could not state.
namespace rola::chunk {
template <int D, int B, int R0, int R1, int R2, int R3, int W0, int W1, int W2, int W3, int MASK>
cudaError_t launch_span_facts_arm(const __nv_bfloat16* pr, const __nv_bfloat16* pw, uint32_t* tbl,
                                  int BH, int L, cudaStream_t stream);
}  // namespace rola::chunk

namespace rola::carry {

//: the plan the carry's rows are voted under, named ONCE: a `BoxPlan`'s spans as a
//: `SpanPlan`.  `D` is at most four and a level past `D` carries `B`.
template <int D, int B, int K, int M>
using OwnerRows =
    rola::chunk::SpanPlan<D, B, BoxPlan<D, B, K, M>::s(0), (D > 1 ? BoxPlan<D, B, K, M>::s(1) : B),
                          (D > 2 ? BoxPlan<D, B, K, M>::s(2) : B),
                          (D > 3 ? BoxPlan<D, B, K, M>::s(3) : B)>;

//: the plan ONE SIDE's rows are voted under (K35 S1b addendum): that side's own
//: partition, at its own run size, with a level it spans whole emitting nothing.  The
//: two sides are independent -- each votes its bitlists from its own factors -- so a
//: stream's test reads exactly one row per tested level and only the owner's whole-run
//: OR spans several (`CarveRows`).
template <class SV>
using SideRowPlan =
    rola::chunk::SpanPlan<CarveRows<SV>::kLevels, CarveRows<SV>::kWidth, CarveRows<SV>::s(0),
                          CarveRows<SV>::s(1), CarveRows<SV>::s(2), CarveRows<SV>::s(3)>;

//: THE PER-LEVEL MODE DECLARATION (K33 law 3) is two bits per level with level 0 in
//: the low pair: bit 0 = the READ side is value-sparse there, bit 1 = the WRITE side.
//: `DENSE_BOTH` everywhere is always correct and only ever slower.  The encoding, the
//: private order it induces and the carve derived from that order are all in `box.cuh`
//: (`mode_declares`, `ModeOrder`, `StreamSpans`); this TU only passes the word through.

//: THE BUILT SET, as one macro so the dispatch, the census and the arm list cannot
//: drift apart -- the shipped family's `CHUNK_ARMS_X` discipline.
//: MO_ is the per-level declaration this arm is BUILT for.  `ALT` at depth `D` is
//: write-sparse on the even levels and read-sparse on the odd ones, i.e.
//: `sum_l mode_l << 2l` = 6 / 38 / 102 at D = 2 / 3 / 4.
//:
//: `RC_` is the FAN-IN RING's depth in rows (P76 section 2.6-C and F7's dial): the ring
//: is `C` rows deep, rank `r` sits at `r % C`, and `C` is correctness-independent -- it
//: buys tolerance for the skew between readers.  32 is the declared first value.
//: `NSW_`/`NSR_` ARE THE ARM'S TWO CARVES, AS STREAM COUNTS (K35 S1b): the write side's
//: streams and the read side's, the latter ONE PER WARP.  The span vectors are DERIVED
//: from `(MO_, side, count)` by the order-by-mode rule, never declared.  A count of one on
//: both sides carves nothing and takes the channel-split body; the executor map must
//: agree, `NSR_ == NSW_ * WPS_`.
#define CARRY_ARM_0(X_) X_(2, 256, 4, 8, 64, 384, 32, 1, 0, 1, 1, 1, 32, 0)
#define CARRY_ARM_1(X_) X_(2, 256, 4, 8, 64, 384, 32, 1, 6, 1, 4, 4, 32, 0)
#define CARRY_ARM_2(X_) X_(2, 256, 4, 8, 64, 512, 32, 1, 0, 1, 1, 1, 32, 0)
#define CARRY_ARM_3(X_) X_(2, 256, 4, 8, 64, 512, 32, 1, 6, 1, 4, 4, 32, 0)
#define CARRY_ARM_4(X_) X_(2, 64, 4, 8, 64, 64, 32, 1, 0, 1, 1, 1, 32, 0)
#define CARRY_ARM_5(X_) X_(2, 64, 4, 8, 64, 64, 32, 1, 6, 1, 4, 4, 32, 0)
#define CARRY_ARM_6(X_) X_(3, 16, 2, 16, 64, 64, 32, 1, 0, 1, 1, 1, 32, 0)
#define CARRY_ARM_7(X_) X_(3, 16, 2, 16, 64, 64, 32, 1, 38, 1, 1, 1, 32, 0)
#define CARRY_ARM_8(X_) X_(4, 8, 2, 8, 64, 64, 32, 1, 0, 1, 1, 1, 32, 0)
#define CARRY_ARM_9(X_) X_(4, 8, 2, 8, 64, 64, 32, 1, 102, 1, 4, 4, 32, 0)
#define CARRY_ARM_10(X_) X_(4, 4, 4, 1, 64, 64, 32, 1, 102, 1, 8, 8, 32, 0)
#define CARRY_ARM_11(X_) X_(2, 256, 4, 8, 64, 384, 32, 2, 6, 2, 4, 8, 32, 0)
#define CARRY_ARM_12(X_) X_(2, 256, 4, 8, 64, 384, 32, 3, 6, 1, 8, 8, 32, 0)
//: THE `d_v = 128` ARMS (K35 S3b).  The card's CONTAINERS entry: at this width sixteen
//: leaves per warp is REQUIRED, and eight resident warps per SM is the target, so the
//: two containers are 1x8 at `BC = 128` -- rows 13 and 14, the SAME two declarations as
//: rows 11 and 12 at the other width -- and 2x4 at `BC = 64`, row 15, which halves the
//: CTA's box instead of doubling its warps and reaches sixteen leaves per warp with the
//: ruled one-warp-per-stream map.  `BC = 64` is `(k, m) = (4, 4)` on the flagship
//: topology: spans `[4, 16]`, 1024 owners, the same window and declaration as the
//: flagship cell so the three containers are comparable at one cell.
#define CARRY_ARM_13(X_) X_(2, 256, 4, 8, 128, 384, 32, 2, 6, 2, 4, 8, 32, 0)
#define CARRY_ARM_14(X_) X_(2, 256, 4, 8, 128, 384, 32, 3, 6, 1, 8, 8, 32, 0)
#define CARRY_ARM_15(X_) X_(2, 256, 4, 4, 128, 384, 32, 1, 6, 1, 4, 4, 32, 0)
//: THE PARK'S TWINS.  `PK_`, the row's LAST field, is the PARK (K35 S5, P76 F5): the
//: readout spends with `lo` in SMEM, not registers.  Each row here is the row above it
//: at `PK_ = 1`, so the pair IS a MEASUREMENT.
#define CARRY_ARM_16(X_) X_(2, 256, 4, 8, 128, 384, 32, 6, 6, 2, 4, 8, 32, 1)
#define CARRY_ARM_17(X_) X_(2, 256, 4, 8, 128, 384, 32, 7, 6, 1, 8, 8, 32, 1)
#define CARRY_ARM_18(X_) X_(2, 256, 4, 4, 128, 384, 32, 5, 6, 1, 4, 4, 32, 1)
#define CARRY_ARM_19(X_) X_(2, 256, 4, 8, 64, 384, 32, 5, 6, 1, 4, 4, 32, 1)
//: THE TIED ARM (K35 S1b, addendum 3).  The flagship cell declared BOTH-SPARSE at level
//: zero, at equal stream counts: both sides then order and carve identically, the
//: exchange's permutation is the IDENTITY and the transit is ELIDED AT COMPILE TIME.  It
//: is row 1's cell again so that the zero is a MEASUREMENT.
#define CARRY_ARM_20(X_) X_(2, 256, 4, 8, 64, 384, 32, 1, 3, 1, 4, 4, 32, 0)

#define CARRY_ARMS_ALL(X_)                                                \
  CARRY_ARM_0(X_) CARRY_ARM_1(X_) CARRY_ARM_2(X_) CARRY_ARM_3(X_)         \
  CARRY_ARM_4(X_) CARRY_ARM_5(X_) CARRY_ARM_6(X_) CARRY_ARM_7(X_)         \
  CARRY_ARM_8(X_) CARRY_ARM_9(X_) CARRY_ARM_10(X_) CARRY_ARM_11(X_)       \
  CARRY_ARM_12(X_) CARRY_ARM_13(X_) CARRY_ARM_14(X_) CARRY_ARM_15(X_)     \
  CARRY_ARM_16(X_) CARRY_ARM_17(X_) CARRY_ARM_18(X_) CARRY_ARM_19(X_)     \
  CARRY_ARM_20(X_)

//: THE ITERATION SUBSET, A BUILD KNOB AND NOTHING MORE.  Compiling sixteen arms to
//: test one is the difference between a minute and an hour, so a build may define
//: `ROLA_CARRY_ARMS(X_)` as any list of the rows above and get exactly those.  The
//: DEFAULT IS EVERY DECLARED ARM, so a plain build is the closed-world one; a subset
//: binary is SELF-IDENTIFYING, because `arms()` lists what was built and every entry
//: point refuses an arm it does not carry.  A gate build never sets it.
//: `ROLA_CARRY_ARMS=11 python -m pip install -e .` (setup.py, `docs/build.md`).
#ifndef ROLA_CARRY_ARMS
#define ROLA_CARRY_ARMS(X_) CARRY_ARMS_ALL(X_)
#endif
#define CARRY_ARMS_X(X_) ROLA_CARRY_ARMS(X_)

//: READING ONE ROW OF THE LIST.  The row is a topology, a window shape, a declaration
//: and two stream counts, so the pieces a consumer wants are taken from it HERE and
//: nowhere else, and no consumer re-spells the row's shape.
#define CARRY_ARM(D_, B_, K_, M_, DV_, W_, C_, CV_, MO_, WPS_, NSW_, NSR_, RC_, PK_) \
  Arm<D_, B_, K_, M_, DV_, W_, C_, WPS_, NSW_, NSR_, RC_, PK_, (uint32_t)(MO_)>
#define CARRY_MO(D_, B_, K_, M_, DV_, W_, C_, CV_, MO_, ...) (MO_)
#define CARRY_ROW_HEAD(D_, B_, K_, M_, DV_, W_, C_, CV_, MO_, ...) \
  D_, B_, K_, M_, DV_, W_, C_, CV_, MO_
//: the seam's key, matched against the call's `(topology, k, m, d_v, W, C, carve)`.
//: `d_v` IS PART OF THE KEY (K35 S3b): the two widths of one declaration are two arms,
//: so the container rows 11/12 and 13/14 share their declaration scalar and are told
//: apart by the width the caller's value plane actually carries.
#define CARRY_MATCH(D_, B_, K_, M_, DV_, W_, C_, CV_, ...)                                \
  (is_arm(widths, D_, B_) && k == K_ && m == M_ && d_v == DV_ && window == W_              \
   && chunk == C_ && carve == CV_)

//: `CV` is the SEAM's DECLARATION scalar, not a kernel axis: it names which declared
//: (carve, executor map, PARK) an arm is, so a caller that asks for the deleted one --
//: or for an executor map nothing builds -- is refused rather than served something
//: else.  Its low two bits are the (carve, executor map): `1` the ruled leaf split, ONE
//: warp per write stream (`rola.ops.carry.LEAF_SPLIT`); `2` the 1x8 container as FOUR
//: write streams x TWO warps each (the column split); `3` as EIGHT streams x ONE warp;
//: and BIT 2 IS THE PARK (`+4`, `PARKED`), which changes nothing the caller can see;
//: the row's `PK_` must agree with it, asserted where the list is.
#define CARRY_CV_GATE(D_, B_, K_, M_, DV_, W_, C_, CV_, MO_, WPS_, NSW_, NSR_, RC_, PK_) \
  static_assert(((CV_) & 3) >= 1 && ((CV_) & 3) <= 3 && ((CV_) & ~7) == 0                    \
                    && (((CV_) >> 2) & 1) == (PK_) && (NSR_) == (NSW_) * (WPS_),             \
                "the seam's declaration scalar names a built (carve, executor map, park), "  \
                "its park bit IS the row's park declaration, and the executor map is the "   \
                "row's own arithmetic: one READ stream per warp over `WPS` warps a stream");
CARRY_ARMS_X(CARRY_CV_GATE)
#undef CARRY_CV_GATE

//: THE SINGLE-LEVEL LEAF SPLIT the seam's `owner_rows(carve=True)` diagnostic names --
//: the innermost level's owner run over `kSeamCarveStreams` streams, which is the grain
//: that entry point documents.  It is a DIAGNOSTIC's declared grain and not an arm's
//: carve, so it is spelled as a span vector here and derived from nothing.
static constexpr int kSeamCarveStreams = 4;
template <int D, int B, int K, int M>
constexpr int owner_span(int l) {
  return l < D ? BoxPlan<D, B, K, M>::s(l) : B;
}
template <int D, int B, int K, int M>
constexpr int split_span(int l) {
  return owner_span<D, B, K, M>(l) / (l == D - 1 ? kSeamCarveStreams : 1);
}
template <int D, int B, int K, int M>
using SeamSplitPlan =
    rola::chunk::SpanPlan<D, B, split_span<D, B, K, M>(0), split_span<D, B, K, M>(1),
                          split_span<D, B, K, M>(2), split_span<D, B, K, M>(3)>;

//: ONE ARM, AS ONE TYPE: the topology, the window shape, the declaration and the two
//: carves.  The body, the geometry, the row plan and the census all read it, so an
//: arm cannot be described two ways.
template <int D, int B, int K, int M, int DV, int W, int C, int WPS, int NSW, int NSR, int RC,
          int PK, uint32_t MO>
struct Arm {
  using BP = BoxPlan<D, B, K, M>;
  //: THE TWO CARVES, DERIVED (K35 S1b): the side's order comes from `MO` and its span
  //: vector from that order and its stream count.  A carve is never declared.
  using SW = StreamSpans<D, B, K, M, MO, kSideWrite, NSW>;
  using SR = StreamSpans<D, B, K, M, MO, kSideRead, NSR>;
  //: A CARVED ARM is one whose streams are finer than the owner's box on either side;
  //: an arm that carves nothing takes the channel-split body, which holds the whole box
  //: in every warp and splits the CHANNELS instead.
  static constexpr bool carved = (NSW > 1) || (NSR > 1);
  //: the arm's VALUE WIDTH, which both bodies take as their `DV` and the census
  //: reports, so the width is readable off the device like the two executor dials.
  static constexpr int kDv = DV;
  //: the READ side's innermost carved level, which the seam reports as the arm's
  //: carve level, and `-1` for a body that carves nothing.
  static constexpr int read_carve(int l) {
    return l < 0 ? -1 : (SR::carves(l) ? l : read_carve(l - 1));
  }
  static constexpr int kReadCarve = read_carve(D - 1);

  //: THE CARVED BODY'S GEOMETRY, named once so the segment probe and the launch read
  //: the same one.
  using GC = CarryGeom<D, B, K, M, DV, W, C, WPS, NSW, NSR, RC, PK, MO>;
  static constexpr auto probe_fn() {
    return &carry_layout_probe_kernel<D, B, K, M, DV, W, C, WPS, NSW, NSR, RC, PK, MO>;
  }
  //: the exchange's transit in bytes -- ZERO where the two carves are identical, which
  //: the geometry recognizes at compile time (K35 S1b addendum 3), and zero again on an
  //: arm that carves nothing and therefore has no exchange to size.  `if constexpr` and
  //: not a ternary: naming `GC::xch_bytes` in a discarded ARM of a conditional still
  //: COMPLETES `CarryGeom`, and an uncarved arm's geometry is not a thing this family
  //: asks to be instantiated (it takes the channel-split body).
  static constexpr int64_t xch_bytes_of() {
    if constexpr (carved)
      return (int64_t)GC::xch_bytes;
    else
      return 0;
  }
  static constexpr int64_t kXchBytes = xch_bytes_of();
  //: the exchange's STATE beside its size: `0` elided, `1` full, `2` full but only
  //: because case (c)'s register reorder is not built (K35 S1b, PARKED), `-1` on an arm
  //: that carves nothing and has no exchange at all.
  static constexpr int64_t xch_state_of() {
    if constexpr (carved)
      return (int64_t)GC::kXchState;
    else
      return -1;
  }
  static constexpr int64_t kXchState = xch_state_of();

  template <bool STATE>
  static constexpr auto fn() {
    if constexpr (carved)
      return &carry_kernel<D, B, K, M, DV, W, C, WPS, NSW, NSR, RC, PK, MO, STATE>;
    else
      return &carry_split_kernel<D, B, K, M, DV, W, C, STATE>;
  }
  //: `{smem bytes, ring depth, CTA target, THREADS, WRITE STREAMS}` -- the launch reads
  //: the fourth, and the last two ARE THE TWO DIALS: `THREADS/32` is the executor
  //: budget, `WRITE STREAMS` the in-CTA ownership grain, and their ratio is `WPS`.
  static constexpr std::array<int64_t, 5> geom() {
    if constexpr (carved) {
      using G = GC;
      return {(int64_t)G::bytes, (int64_t)G::kStages, (int64_t)G::cta_target, (int64_t)G::kThreads,
              (int64_t)G::kStreams};
    } else {
      using G = SplitGeom<D, B, K, M, DV, W, C>;
      return {(int64_t)G::bytes, (int64_t)G::kStages, (int64_t)G::cta_target, (int64_t)kThreads,
              (int64_t)1};
    }
  }
};

template <class A>
static cudaError_t launch(const __nv_bfloat16* pr, const __nv_bfloat16* pw, const __nv_bfloat16* gw,
                          const __nv_bfloat16* v, const uint32_t* rows, float* o, float* den,
                          int BH, int L, int tok_words, int v_row_bytes, const float* s_in,
                          float* s_out, cudaStream_t stream) {
  const auto fn = A::template fn<true>();
  constexpr size_t sm = (size_t)A::geom()[0];
  static bool configured = false;
  if (!configured) {
    const cudaError_t e =
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    if (e != cudaSuccess) return e;
    configured = true;
  }
  const dim3 grid((unsigned)A::BP::kOwners, (unsigned)BH);
  fn<<<grid, (unsigned)A::geom()[3], sm, stream>>>(pr, pw, gw, v, rows, o, den, L, tok_words,
                                                   v_row_bytes, s_in, s_out);
  return cudaGetLastError();
}

//: THE CENSUS, whose last ten fields are THE DERIVATION READ BACK OFF THE DEVICE (K35
//: S1b): the two span vectors the order-by-mode rule produced, the exchange's transit in
//: bytes -- zero where the two carves came out identical -- and the exchange's STATE, so
//: an arm that OVERPAYS is visible rather than silent.
template <class A>
static std::array<int64_t, 23> census() {
  cudaFuncAttributes a{};
  if (cudaFuncGetAttributes(&a, A::template fn<true>()) != cudaSuccess)
    return {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
  constexpr auto g = A::geom();
  return {a.numRegs,
          (int64_t)a.sharedSizeBytes,
          a.binaryVersion,
          g[0],
          (int64_t)a.maxThreadsPerBlock,
          (int64_t)a.localSizeBytes,
          (int64_t)A::BP::kBC,
          (int64_t)A::BP::kOwners,
          g[1],
          g[2],
          (int64_t)A::kReadCarve,
          g[4],
          (int64_t)A::kDv,
          (int64_t)A::SW::s(0),
          (int64_t)A::SW::s(1),
          (int64_t)A::SW::s(2),
          (int64_t)A::SW::s(3),
          (int64_t)A::SR::s(0),
          (int64_t)A::SR::s(1),
          (int64_t)A::SR::s(2),
          (int64_t)A::SR::s(3),
          A::kXchBytes,
          A::kXchState};
}

static bool is_arm(const std::vector<int64_t>& widths, int D, int B) {
  if ((int)widths.size() != D) return false;
  for (auto w : widths)
    if (w != B) return false;
  return true;
}

//: THE SEGMENT LAYOUT'S ROUND TRIP, as its own entry point so the layout can be
//: gated without running the kernel that uses it (KERNEL_STANDARDS section 3).
template <class A>
static std::vector<torch::Tensor> run_probe() {
  if constexpr (A::carved) {
    using G = typename A::GC;
    using BPA = typename A::BP;
    auto opt = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto vv = torch::zeros({(long)G::kStreams * G::kSegRows, (long)G::kDv}, opt);
    auto cc = torch::zeros({(long)G::kStreams * G::kSegRows, (long)G::kLeafW}, opt);
    auto ss = torch::zeros({(long)G::kSegRows, (long)BPA::kStageW}, opt);
    //: the READOUT's two layouts (K35 S4b): a reader's segment and one ring row per slot.
    auto rr = torch::zeros({(long)G::kWarps * G::kSegRows, (long)G::kLeafE}, opt);
    auto gg = torch::zeros({(long)G::kRingC, (long)G::kDv + 1}, opt);
    //: the staged leg's SOURCE plane: cell `(r, c)` names itself, so a run that lands
    //: at the wrong offset names the column it actually came from.
    auto pw = (torch::arange(0, (long)G::kSegRows, opt).reshape({-1, 1}) * 1024
               + torch::arange(0, (long)BPA::kWtot, opt).reshape({1, -1}))
                  .to(torch::kInt16)
                  .view(torch::kBFloat16)
                  .contiguous();
    constexpr size_t sm =
        (size_t)(G::seg_v_bytes + G::seg_c_bytes
                 + G::kSegRows * BPA::kStageW * (int)sizeof(__nv_bfloat16)
                 + align16c(G::kWarps * G::kSegRows * G::LdR * (int)sizeof(__nv_bfloat16))
                 + G::kRingC * G::LdRing * (int)sizeof(float));
    const auto fn = A::probe_fn();
    C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
    fn<<<1, (unsigned)G::kThreads, sm, at::cuda::getCurrentCUDAStream()>>>(
        (const __nv_bfloat16*)pw.data_ptr(), vv.template data_ptr<int>(),
        cc.template data_ptr<int>(), ss.template data_ptr<int>(), rr.template data_ptr<int>(),
        gg.template data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {vv.reshape({(long)G::kStreams, (long)G::kSegRows, (long)G::kDv}),
            cc.reshape({(long)G::kStreams, (long)G::kSegRows, (long)G::kLeafW}), ss,
            rr.reshape({(long)G::kWarps, (long)G::kSegRows, (long)G::kLeafE}), gg};
  } else {
    TORCH_CHECK(false, "the per-stream segment belongs to the carved body");
    return {};
  }
}

std::vector<torch::Tensor> layout_probe(std::vector<int64_t> widths, int64_t k, int64_t m,
                                        int64_t window, int64_t chunk, int64_t carve,
                                        int64_t level_modes, int64_t d_v) {
#define CARRY_PROBE(...)                                                     \
  if (CARRY_MATCH(__VA_ARGS__) && level_modes == CARRY_MO(__VA_ARGS__)) {    \
    return run_probe<CARRY_ARM(__VA_ARGS__)>();                              \
  }
  CARRY_ARMS_X(CARRY_PROBE)
#undef CARRY_PROBE
  TORCH_CHECK(false, "unbuilt carry arm");
  return {};
}

//: THE TABLE, AS A TENSOR: `[BH, entries_read + entries_write, L/32]` packed bits, the
//: READ plan's rows and then the WRITE plan's, one row per (level, run of THAT side's
//: partition).  The two plans are independent (K35 S1b addendum): a side's bitlists are
//: voted from its own factors, so nothing is gained by refining one side's table by the
//: other's carve.  Where the two plans coincide the block IS the old `[BH, 2, E, L/32]`.
//: Also a SEPARATE ENTRY POINT, so the grain the kernel consumes and the grain the seam
//: names can be compared bit for bit (`tests/k31/test_owner_rows.py`).
template <class RPR, class RPW>
static torch::Tensor build_rows(const torch::Tensor& pr, const torch::Tensor& pw, int BH, int L) {
  auto tbl = torch::empty({BH, (long)RPR::entries + (long)RPW::entries, rola::chunk::tok_words(L)},
                          pr.options().dtype(torch::kInt32));
  C10_CUDA_CHECK(
      (rola::chunk::launch_span_facts_arm<RPR::depth, RPR::kWidth, RPR::span_at(0), RPR::span_at(1),
                                          RPR::span_at(2), RPR::span_at(3), RPW::span_at(0),
                                          RPW::span_at(1), RPW::span_at(2), RPW::span_at(3),
                                          rola::chunk::kPrefillStateless>(
          (const __nv_bfloat16*)pr.data_ptr(), (const __nv_bfloat16*)pw.data_ptr(),
          reinterpret_cast<uint32_t*>(tbl.data_ptr()), BH, L, at::cuda::getCurrentCUDAStream())));
  return tbl;
}

//: the ROW PLANS the arm's body consumes -- one per side, at that side's own derived
//: partition.  An uncarved arm's tables ARE the owner's, because the derived vectors are
//: then the owner's spans; one statement, not two paths.
template <class A>
static torch::Tensor build_arm_rows(const torch::Tensor& pr, const torch::Tensor& pw, int BH,
                                    int L) {
  return build_rows<SideRowPlan<typename A::SR>, SideRowPlan<typename A::SW>>(pr, pw, BH, L);
}

std::vector<torch::Tensor> forward_inter(torch::Tensor pr, torch::Tensor pw, torch::Tensor gw,
                                         torch::Tensor v, std::vector<int64_t> widths, int64_t k,
                                         int64_t m, int64_t window, int64_t chunk, int64_t carve,
                                         int64_t level_modes, c10::optional<torch::Tensor> s_in) {
  TORCH_CHECK(pr.is_cuda() && pw.is_cuda() && gw.is_cuda() && v.is_cuda(), "cuda tensors only");
  TORCH_CHECK(pr.scalar_type() == torch::kBFloat16 && pw.scalar_type() == torch::kBFloat16
                  && gw.scalar_type() == torch::kBFloat16 && v.scalar_type() == torch::kBFloat16,
              "the operand planes are bf16");
  TORCH_CHECK(pr.is_contiguous() && pw.is_contiguous() && v.is_contiguous(), "contiguous planes");
  //: `pr`/`pw` are `[BH, L, sum_l B_l]`; `v` is `[B, T, H, d_v]`; `gw` is `[BH, L]`.
  const int BH = (int)pr.size(0);
  const int L = (int)pr.size(1);
  //: THE VALUE WIDTH IS READ OFF THE CALLER'S PLANE and is part of the arm key, so an
  //: unbuilt width is refused by the dispatch exactly as an unbuilt topology is.
  const int d_v = (int)v.size(-1);
  const int v_row_bytes = (int)(v.size(2) * d_v * v.element_size());

  int64_t N = 1;
  for (auto w : widths) N *= w;
  auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(v.device());
  auto o = torch::zeros({BH, L, (int64_t)d_v}, opt);
  auto den = torch::zeros({BH, L}, opt);
  auto s_out = torch::zeros({BH, N / kAtomLeaves, (int64_t)kAtomLeaves, (int64_t)(d_v + 1)}, opt);

  const float* sin_p = s_in.has_value() ? s_in->data_ptr<float>() : nullptr;
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaError_t err = cudaErrorInvalidValue;
  //: THE BALLOT ROWS COME FIRST, ON THE SAME STREAM (K31-i-d).  One pass over the
  //: amplitude planes votes every owner's rectangle test once; the kernel below then
  //: reads WORDS instead of re-reading those planes per (owner, window, token).  The
  //: table is a TRANSIENT of this call -- nothing else asks for it -- and it is held
  //: here rather than inside the dispatch so its lifetime outlives the launch.
  torch::Tensor rows;
  const int tok_words = rola::chunk::tok_words(L);
  auto run = [&](int64_t mo) {
    bool matched = false;
#define CARRY_DISPATCH(...)                                                                 \
  if (CARRY_MATCH(__VA_ARGS__) && mo == CARRY_MO(__VA_ARGS__)) {                            \
    matched = true;                                                                         \
    rows = build_arm_rows<CARRY_ARM(__VA_ARGS__)>(pr, pw, BH, L);                           \
    err = launch<CARRY_ARM(__VA_ARGS__)>(                                                   \
        (const __nv_bfloat16*)pr.data_ptr(), (const __nv_bfloat16*)pw.data_ptr(),           \
        (const __nv_bfloat16*)gw.data_ptr(), (const __nv_bfloat16*)v.data_ptr(),            \
        reinterpret_cast<const uint32_t*>(rows.data_ptr()), o.data_ptr<float>(),            \
        den.data_ptr<float>(), BH, L, tok_words, v_row_bytes, sin_p,                        \
        s_out.data_ptr<float>(), stream);                                                   \
  } else
    CARRY_ARMS_X(CARRY_DISPATCH) {}
#undef CARRY_DISPATCH
    return matched;
  };
  //: THE DECLARATION IS RESOLVED AT THIS SEAM (law 3).  A declaration this family
  //: has no body built for is served DENSE_BOTH, which is always correct and only
  //: ever slower; the TOPOLOGY is never degraded -- an unbuilt (D, B, k, m, W, C)
  //: is refused outright.
  if (!run(level_modes)) TORCH_CHECK(run(0), "unbuilt carry arm");
  C10_CUDA_CHECK(err);
  return {o, den, s_out};
}

//: THE FILE'S OWN CONSTANTS, device-side, beside the magic: the channel-split body's
//: executor count and the page granule.  `d_v` LEFT THIS SET at K35 S3b -- it is an arm
//: parameter now, so it is reported by the per-arm CENSUS below and not by the stamp.
__global__ void stamp_kernel_entry(int* out) {
  out[0] = kCarryBuildMagic;
  out[1] = kWarps * 1000 + kAtomLeaves;
  out[2] = design::kCapsDigest;
}

std::vector<int64_t> build_stamp(std::vector<int64_t> widths, int64_t k, int64_t m, int64_t window,
                                 int64_t chunk, int64_t carve, int64_t level_modes, int64_t d_v) {
  auto t = torch::zeros({3}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  stamp_kernel_entry<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(t.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto h = t.cpu();
  std::vector<int64_t> r{h.data_ptr<int>()[0], h.data_ptr<int>()[1], h.data_ptr<int>()[2]};
#define CARRY_CENSUS(...)                                                                   \
  if (CARRY_MATCH(__VA_ARGS__) && level_modes == CARRY_MO(__VA_ARGS__)) {                   \
    for (auto x : census<CARRY_ARM(__VA_ARGS__)>()) r.push_back(x);                         \
    return r;                                                                               \
  }
  CARRY_ARMS_X(CARRY_CENSUS)
#undef CARRY_CENSUS
  TORCH_CHECK(false, "unbuilt carry arm");
  return r;
}

torch::Tensor owner_rows(torch::Tensor pr, torch::Tensor pw, std::vector<int64_t> widths, int64_t k,
                         int64_t m, bool carve) {
  TORCH_CHECK(pr.is_cuda() && pw.is_cuda() && pr.is_contiguous() && pw.is_contiguous(),
              "contiguous cuda planes");
  TORCH_CHECK(pr.scalar_type() == torch::kBFloat16 && pw.scalar_type() == torch::kBFloat16,
              "the operand planes are bf16");
  TORCH_CHECK(pr.sizes() == pw.sizes() && pr.dim() == 3, "expect packed amplitudes [BH, L, sumW]");
  const int BH = (int)pr.size(0), L = (int)pr.size(1);
#define CARRY_ROWS(D_, B_, K_, M_, DV_, W_, C_, CV_, MO_, ...)                                \
  if constexpr ((MO_) == 0) {                                                                 \
    if (is_arm(widths, D_, B_) && k == K_ && m == M_)                                         \
      return carve                                                                            \
                 ? build_rows<SeamSplitPlan<D_, B_, K_, M_>, SeamSplitPlan<D_, B_, K_, M_>>(   \
                       pr, pw, BH, L)                                                         \
                 : build_rows<OwnerRows<D_, B_, K_, M_>, OwnerRows<D_, B_, K_, M_>>(pr, pw, BH,\
                                                                                    L);       \
  }
  CARRY_ARMS_X(CARRY_ROWS)
#undef CARRY_ROWS
  TORCH_CHECK(false, "unbuilt carry arm");
}

std::vector<std::vector<int64_t>> arms() {
  std::vector<std::vector<int64_t>> out;
#define CARRY_LIST(...)                                                                     \
  out.push_back({CARRY_ROW_HEAD(__VA_ARGS__), CARRY_ARM(__VA_ARGS__)::BP::kBC,              \
                 CARRY_ARM(__VA_ARGS__)::BP::kOwners, CARRY_ARM(__VA_ARGS__)::geom()[0],    \
                 (int64_t)CARRY_ARM(__VA_ARGS__)::kReadCarve});
  CARRY_ARMS_X(CARRY_LIST)
#undef CARRY_LIST
  return out;
}

}  // namespace rola::carry
