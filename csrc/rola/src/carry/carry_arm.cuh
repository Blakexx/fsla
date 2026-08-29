// csrc/rola/src/carry/carry_arm.cuh -- an arm, as a type, and its four entry points.
//
// THE HEAVY HEADER, included by the per-arm TUs and by NOTHING else.  `carry.cu`
// must not include it: that is what stops the dispatch from instantiating an arm
// locally and leaving the split doing nothing (the failure mode
// `tools/fatbin_entries.py` exists to catch on the facts family).
//
// `CARRY_ARM_DEFINE` is the whole per-arm translation unit: the generated file
// under `../instantiations/` is an include, a guard and one expansion of this.
// docs/internals/carry/carry_kernel.md.
#pragma once

#include <array>

#include "carry_arm_abi.cuh"
#include "carry_probe_kernel.cuh"
#include "carry_rows.cuh"
#include "carry_split_kernel.cuh"

namespace rola::carry {

//: ONE ARM, AS ONE TYPE: the topology, the window shape, the declaration and the two
//: carves.  The body, the geometry, the row plan and the census all read it, so an
//: arm cannot be described two ways.
template <int D, int B, int K, int M, int DV, int W, int C, int WPS, int NSW, int NSR, int PK,
          uint32_t MO>
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
  using GC = CarryGeom<D, B, K, M, DV, W, C, WPS, NSW, NSR, PK, MO>;
  static constexpr auto probe_fn() {
    return &carry_layout_probe_kernel<D, B, K, M, DV, W, C, WPS, NSW, NSR, PK, MO>;
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
      return &carry_kernel<D, B, K, M, DV, W, C, WPS, NSW, NSR, PK, MO, STATE>;
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

//: THE SEGMENT LAYOUT'S ROUND TRIP, as its own entry point so the layout can be
//: gated without running the kernel that uses it (KERNEL_STANDARDS section 3).  The
//: four planes are the DISPATCH's allocation (`carry_arm_abi.cuh`); this states their
//: shapes and this launches into them.
template <class A>
static constexpr size_t probe_smem() {
  using G = typename A::GC;
  using BPA = typename A::BP;
  return (size_t)(G::seg_v_bytes + G::seg_c_bytes
                  + G::kSegRows * BPA::kStageW * (int)sizeof(__nv_bfloat16)
                  + align16c(G::kWarps * G::kSegRows * G::LdR * (int)sizeof(__nv_bfloat16)));
}

template <class A>
static CarryProbeShape probe_shape() {
  if constexpr (A::carved) {
    using G = typename A::GC;
    using BPA = typename A::BP;
    return {1,
            G::kStreams,
            G::kWarps,
            G::kSegRows,
            G::kDv,
            G::kLeafW,
            G::kLeafE,
            BPA::kStageW,
            BPA::kWtot,
            G::kThreads,
            (int64_t)probe_smem<A>()};
  } else {
    return {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  }
}

template <class A>
static cudaError_t run_probe(const __nv_bfloat16* pw, int* vv, int* cc, int* ss, int* rr,
                             cudaStream_t stream) {
  if constexpr (A::carved) {
    using G = typename A::GC;
    constexpr size_t sm = probe_smem<A>();
    const auto fn = A::probe_fn();
    const cudaError_t e =
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    if (e != cudaSuccess) return e;
    fn<<<1, (unsigned)G::kThreads, sm, stream>>>(pw, vv, cc, ss, rr);
    return cudaGetLastError();
  } else {
    //: the per-stream segment belongs to the carved body; the dispatch refuses this
    //: before it gets here, on the shape's `carved` flag.
    return cudaErrorInvalidValue;
  }
}

//: the ROW PLANS the arm's body consumes -- one per side, at that side's own derived
//: partition.  An uncarved arm's tables ARE the owner's, because the derived vectors are
//: then the owner's spans; one statement, not two paths.
template <class A>
using ArmRowsR = SideRowPlan<typename A::SR>;
template <class A>
using ArmRowsW = SideRowPlan<typename A::SW>;

//: ONE ROW OF THE LIST, AS THE ARM TYPE.  The row's field order is spelled HERE and
//: in `tools/gen_shards.py`'s emitter and nowhere else.
#define CARRY_ARM_TYPE(I_, D_, B_, K_, M_, DV_, W_, C_, CV_, MO_, WPS_, NSW_, NSR_, PK_) \
  Arm<D_, B_, K_, M_, DV_, W_, C_, WPS_, NSW_, NSR_, PK_, (uint32_t)(MO_)>

//: THE PER-ARM TRANSLATION UNIT'S ENTIRE CONTENT.  One expansion instantiates this
//: arm's kernel (the carved body or the channel-split one, whichever the arm takes)
//: and the probe kernel keyed to it, and nothing else in the build does.
#define CARRY_ARM_DEFINE(I_, D_, B_, K_, M_, DV_, W_, C_, CV_, MO_, WPS_, NSW_, NSR_, PK_)      \
  using CarryArm##I_ = CARRY_ARM_TYPE(I_, D_, B_, K_, M_, DV_, W_, C_, CV_, MO_, WPS_, NSW_,    \
                                      NSR_, PK_);                                               \
  cudaError_t carry_arm_forward_##I_(                                                            \
      const __nv_bfloat16* pr, const __nv_bfloat16* pw, const __nv_bfloat16* gw,                 \
      const __nv_bfloat16* v, const uint32_t* rows, float* o, float* den, int BH, int L,         \
      int tok_words, int v_row_bytes, const float* s_in, float* s_out, cudaStream_t stream) {    \
    return launch<CarryArm##I_>(pr, pw, gw, v, rows, o, den, BH, L, tok_words, v_row_bytes,      \
                                s_in, s_out, stream);                                            \
  }                                                                                              \
  cudaError_t carry_arm_rows_##I_(const __nv_bfloat16* pr, const __nv_bfloat16* pw,              \
                                  uint32_t* tbl, int BH, int L, cudaStream_t stream) {           \
    return launch_rows<ArmRowsR<CarryArm##I_>, ArmRowsW<CarryArm##I_>>(pr, pw, tbl, BH, L,       \
                                                                      stream);                   \
  }                                                                                              \
  int carry_arm_rows_entries_##I_() {                                                            \
    return rows_entries<ArmRowsR<CarryArm##I_>, ArmRowsW<CarryArm##I_>>();                       \
  }                                                                                              \
  CarryProbeShape carry_arm_probe_shape_##I_() { return probe_shape<CarryArm##I_>(); }           \
  cudaError_t carry_arm_probe_##I_(const __nv_bfloat16* pw, int* vv, int* cc, int* ss, int* rr,  \
                                   cudaStream_t stream) {                                        \
    return run_probe<CarryArm##I_>(pw, vv, cc, ss, rr, stream);                                   \
  }                                                                                              \
  void carry_arm_census_##I_(int64_t* out) {                                                     \
    const auto c = census<CarryArm##I_>();                                                        \
    for (int i = 0; i < kCarryCensusFields; ++i) out[i] = c[i];                                   \
  }                                                                                              \
  void carry_arm_row_##I_(int64_t* out) {                                                        \
    using A = CarryArm##I_;                                                                       \
    const int64_t row[kCarryRowFields] = {D_,                                                     \
                                          B_,                                                     \
                                          K_,                                                     \
                                          M_,                                                     \
                                          DV_,                                                    \
                                          W_,                                                     \
                                          C_,                                                     \
                                          CV_,                                                    \
                                          MO_,                                                    \
                                          (int64_t)A::BP::kBC,                                    \
                                          (int64_t)A::BP::kOwners,                                \
                                          A::geom()[0],                                           \
                                          (int64_t)A::kReadCarve};                                \
    for (int i = 0; i < kCarryRowFields; ++i) out[i] = row[i];                                    \
  }

}  // namespace rola::carry
