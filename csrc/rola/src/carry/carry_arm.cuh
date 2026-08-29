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
#include "carry_rows.cuh"
#include "carry_split_kernel.cuh"

namespace rola::carry {

//: ONE ARM, AS ONE TYPE: the topology, the window shape, the declaration and the two -- see docs/internals/carry/carry_arm.md#near-line-21
template <int D, int B, int K, int M, int DV, int W, int C, int WPS, int NSW, int NSR, int PK,
          uint32_t MO>
struct Arm {
  using BP = BoxPlan<D, B, K, M>;
  //: THE CARVE, DERIVED (K35 S1b): the side's order comes from `MO` and its span -- see docs/internals/carry/carry_arm.md#sr
  using SR = StreamSpans<D, B, K, M, MO, kSideRead, NSR>;
  using SW = StreamSpans<D, B, K, M, MO, kSideWrite, NSW>;
  //: A CARVED ARM is one whose streams are finer than the owner's box on either side; -- see docs/internals/carry/carry_arm.md#carved
  static constexpr bool carved = (NSW > 1) || (NSR > 1);
  //: the arm's VALUE WIDTH, which both bodies take as their `DV` and the census -- see docs/internals/carry/carry_arm.md#kdv
  static constexpr int kDv = DV;
  //: the READ side's innermost carved level, which the seam reports as the arm's -- see docs/internals/carry/carry_arm.md#readcarve
  static constexpr int read_carve(int l) {
    return l < 0 ? -1 : (SR::carves(l) ? l : read_carve(l - 1));
  }
  static constexpr int kReadCarve = read_carve(D - 1);

  //: THE CARVED BODY'S GEOMETRY, named once so the segment probe and the launch read -- see docs/internals/carry/carry_arm.md#gc
  using GC = CarryGeom<D, B, K, M, DV, W, C, WPS, NSW, NSR, PK, MO>;
  //: THE EXCHANGE, READ BACK OFF THE DEVICE (K35 F1): its bytes, and WHICH OF THE -- see docs/internals/carry/carry_arm.md#kxchbytes
  //: `if constexpr`, not a conditional expression: naming `GC::` in an initializer
  //: INSTANTIATES the carved geometry, whose asserts an uncarved arm cannot satisfy.
  static constexpr int64_t xch_bytes_of() {
    if constexpr (carved)
      return (int64_t)GC::kXchBytes;
    else
      return 0;
  }
  static constexpr int64_t xch_state_of() {
    if constexpr (carved)
      return (int64_t)GC::kXchState;
    else
      return -1;
  }
  static constexpr int64_t kXchBytes = xch_bytes_of();
  static constexpr int64_t kXchState = xch_state_of();

  static constexpr auto fn() {
    if constexpr (carved)
      return &carry_kernel<D, B, K, M, DV, W, C, WPS, NSW, NSR, PK, MO>;
    else
      return &carry_split_kernel<D, B, K, M, DV, W, C>;
  }
  //: `{smem bytes, ring depth, CTA target, THREADS, WRITE STREAMS}` -- the launch reads -- see docs/internals/carry/carry_arm.md#geom
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

//: ONE VARIANT'S LAUNCH. -- see docs/internals/carry/carry_arm.md#launch
template <class A>
static cudaError_t launch(const __nv_bfloat16* pr, const __nv_bfloat16* pw, const __nv_bfloat16* gw,
                          const __nv_bfloat16* v, const uint32_t* rows, float* o, float* den,
                          int BH, int L, int tok_words, int v_row_bytes, const float* s_in,
                          float* s_out, const int32_t* page_tbl, const uint8_t* atom_bits,
                          cudaStream_t stream) {
  const auto fn = A::fn();
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
                                                   v_row_bytes, s_in, s_out, page_tbl, atom_bits);
  return cudaGetLastError();
}

//: THE CENSUS, whose last ten fields are THE DERIVATION READ BACK OFF THE -- see docs/internals/carry/carry_arm.md#census
template <class A>
static std::array<int64_t, 23> census() {
  cudaFuncAttributes a{};
  if (cudaFuncGetAttributes(&a, A::fn()) != cudaSuccess)
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

//: the ROW PLANS the arm's body consumes -- one per side, at that side's own derived -- see docs/internals/carry/carry_arm.md#a
template <class A>
using ArmRowsR = SideRowPlan<typename A::SR>;
template <class A>
using ArmRowsW = SideRowPlan<typename A::SW>;

//: ONE ROW OF THE LIST, AS THE ARM TYPE.  The row's field order is spelled HERE and -- see docs/internals/carry/carry_arm.md#carryarmtype
#define CARRY_ARM_TYPE(I_, D_, B_, K_, M_, DV_, W_, C_, CV_, MO_, WPS_, NSW_, NSR_, PK_) \
  Arm<D_, B_, K_, M_, DV_, W_, C_, WPS_, NSW_, NSR_, PK_, (uint32_t)(MO_)>

//: THE PER-ARM TRANSLATION UNIT'S ENTIRE CONTENT.  One expansion instantiates this -- see docs/internals/carry/carry_arm.md#carryarmdefine
#define CARRY_ARM_DEFINE(I_, D_, B_, K_, M_, DV_, W_, C_, CV_, MO_, WPS_, NSW_, NSR_, PK_)      \
  using CarryArm##I_ = CARRY_ARM_TYPE(I_, D_, B_, K_, M_, DV_, W_, C_, CV_, MO_, WPS_, NSW_,    \
                                      NSR_, PK_);                                               \
  cudaError_t carry_arm_forward_##I_(                                                            \
      const __nv_bfloat16* pr, const __nv_bfloat16* pw, const __nv_bfloat16* gw,                 \
      const __nv_bfloat16* v, const uint32_t* rows, float* o, float* den, int BH, int L,         \
      int tok_words, int v_row_bytes, const float* s_in, float* s_out, \
      const int32_t* page_tbl, const uint8_t* atom_bits, cudaStream_t stream) { \
    return launch<CarryArm##I_>(pr, pw, gw, v, rows, o, den, BH, L, tok_words, v_row_bytes,      \
                                s_in, s_out, page_tbl, atom_bits, stream); \
  }                                                                                              \
  cudaError_t carry_arm_rows_##I_(const __nv_bfloat16* pr, const __nv_bfloat16* pw,              \
                                  uint32_t* tbl, int BH, int L, cudaStream_t stream) {           \
    return launch_rows<ArmRowsR<CarryArm##I_>, ArmRowsW<CarryArm##I_>>(pr, pw, tbl, BH, L,       \
                                                                      stream);                   \
  }                                                                                              \
  int carry_arm_rows_entries_##I_() {                                                            \
    return rows_entries<ArmRowsR<CarryArm##I_>, ArmRowsW<CarryArm##I_>>();                       \
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
