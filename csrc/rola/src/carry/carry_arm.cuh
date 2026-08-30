// csrc/rola/src/carry/carry_arm.cuh -- an arm, as a type, and its entry points.
//
// THE HEAVY HEADER, included by the per-arm TUs and by NOTHING else (`carry.cu` must
// not include it: that is what stops the dispatch from instantiating an arm locally).
// AN ARM IS A SHAPE (K50) -- `<D, DV, W, C, kWarps, WPS>` plus which BODY it carries;
// the widths, the mode word and the stream counts arrive as a `CarryGeomRT`.
// See docs/internals/carry/carry_arm.md
#pragma once

#include <array>

#include "carry_arm_abi.cuh"
#include "carry_rows.cuh"
#include "carry_split_kernel.cuh"

namespace rola::carry {

//: ONE ARM, AS ONE TYPE.  `SPLIT` names a BODY, not a geometry. -- see docs/internals/carry/carry_arm.md#near-line-21
template <int D, int DV, int W, int C, int KWARPS, int WPS, bool SPLIT>
struct Arm {
  static constexpr bool carved = !SPLIT;
  //: the arm's VALUE WIDTH, which both bodies take as their `DV`. -- see docs/internals/carry/carry_arm.md#kdv
  static constexpr int kDv = DV;
  static constexpr int kWarpsA = KWARPS;
  //: THE CARVED BODY'S GEOMETRY, named once so the launch reads it. -- see docs/internals/carry/carry_arm.md#gc
  using GC = CarryGeom<D, DV, W, C, KWARPS, WPS>;
  using GS = SplitGeom<D, DV, W, C>;
  //: the stream counts are DERIVED from the warp count (K50); the split body carves none.
  static constexpr int kStreamsR = SPLIT ? 1 : KWARPS;
  static constexpr int kStreamsW = SPLIT ? 1 : KWARPS / WPS;
  static constexpr int BC = SPLIT ? GS::BC : GC::BC;
  //: THE TRANSIT'S BYTES: UNCONDITIONAL on the carved body (K50 deletes the elided form).
  static constexpr int64_t kXchBytes = SPLIT ? 0 : (int64_t)GC::kXchBytes;

  static constexpr auto fn() {
    if constexpr (SPLIT)
      return &carry_split_kernel<D, DV, W, C>;
    else
      return &carry_kernel<D, DV, W, C, KWARPS, WPS>;
  }
  //: `{smem bytes, ring depth, CTA target, THREADS, STREAMS}` -- the launch reads it. -- see docs/internals/carry/carry_arm.md#geom
  static constexpr std::array<int64_t, 5> geom() {
    if constexpr (SPLIT)
      return {(int64_t)GS::bytes, (int64_t)GS::kStages, (int64_t)GS::cta_target, (int64_t)kThreads,
              (int64_t)1};
    else
      return {(int64_t)GC::bytes, (int64_t)GC::kStages, (int64_t)GC::cta_target,
              (int64_t)GC::kThreads, (int64_t)GC::kStreams};
  }
};

//: ONE VARIANT'S LAUNCH.  The grid's owner extent is the BLOCK's. -- see docs/internals/carry/carry_arm.md#launch
template <class A>
static cudaError_t launch(const __nv_bfloat16* pr, const __nv_bfloat16* pw, const __nv_bfloat16* gw,
                          const __nv_bfloat16* v, const uint32_t* rows, float* o, float* den,
                          int BH, int L, int tok_words, int v_row_bytes, const float* s_in,
                          float* s_out, const int32_t* page_tbl, const uint8_t* atom_bits,
                          const CarryGeomRT& geo, cudaStream_t stream) {
  const auto fn = A::fn();
  constexpr size_t sm = (size_t)A::geom()[0];
  //: THE LAYOUT IS LAUNCH-SIZED: the dynamic allocation and the >48 KB attribute, once.
  static bool configured = false;
  if (!configured) {
    const cudaError_t e =
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    if (e != cudaSuccess) return e;
    configured = true;
  }
  const dim3 grid((unsigned)geo.owners, (unsigned)BH);
  fn<<<grid, (unsigned)A::geom()[3], sm, stream>>>(pr, pw, gw, v, rows, o, den, L, tok_words,
                                                   v_row_bytes, s_in, s_out, page_tbl, atom_bits,
                                                   geo);
  return cudaGetLastError();
}

//: THE CENSUS: THE DERIVATION READ BACK, incl. the IDENTITY-map fact. -- see docs/internals/carry/carry_arm.md#census
template <class A>
static std::array<int64_t, kCarryCensusFields> census(const CarryGeomRT& geo) {
  cudaFuncAttributes a{};
  if (cudaFuncGetAttributes(&a, A::fn()) != cudaSuccess) {
    std::array<int64_t, kCarryCensusFields> bad{};
    for (auto& x : bad) x = -1;
    return bad;
  }
  constexpr auto g = A::geom();
  const int rc = [&] {
    int c = -1;
    for (int l = 0; l < geo.D; ++l)
      if (((geo.r.carves >> l) & 1) != 0) c = l;
    return c;
  }();
  return {a.numRegs,
          (int64_t)a.sharedSizeBytes,
          a.binaryVersion,
          g[0],
          (int64_t)a.maxThreadsPerBlock,
          (int64_t)a.localSizeBytes,
          (int64_t)geo.bc,
          (int64_t)geo.owners,
          g[1],
          g[2],
          (int64_t)rc,
          g[4],
          (int64_t)A::kDv,
          (int64_t)geo.w.row_span[0],
          (int64_t)geo.w.row_span[1],
          (int64_t)geo.w.row_span[2],
          (int64_t)geo.w.row_span[3],
          (int64_t)geo.r.row_span[0],
          (int64_t)geo.r.row_span[1],
          (int64_t)geo.r.row_span[2],
          (int64_t)geo.r.row_span[3],
          A::kXchBytes,
          (int64_t)geo.identity_map};
}

//: ONE ROW OF THE LIST, AS THE ARM TYPE.  The row's field order is spelled HERE and -- see docs/internals/carry/carry_arm.md#carryarmtype
#define CARRY_ARM_TYPE(I_, D_, DV_, W_, C_, KW_, WPS_, SPLIT_) \
  Arm<D_, DV_, W_, C_, KW_, WPS_, (bool)(SPLIT_)>

//: THE PER-ARM TRANSLATION UNIT'S ENTIRE CONTENT. -- see docs/internals/carry/carry_arm.md#carryarmdefine
#define CARRY_ARM_DEFINE(I_, D_, DV_, W_, C_, KW_, WPS_, SPLIT_)                                \
  using CarryArm##I_ = CARRY_ARM_TYPE(I_, D_, DV_, W_, C_, KW_, WPS_, SPLIT_);                  \
  cudaError_t carry_arm_forward_##I_(                                                           \
      const __nv_bfloat16* pr, const __nv_bfloat16* pw, const __nv_bfloat16* gw,                \
      const __nv_bfloat16* v, const uint32_t* rows, float* o, float* den, int BH, int L,        \
      int tok_words, int v_row_bytes, const float* s_in, float* s_out,                          \
      const int32_t* page_tbl, const uint8_t* atom_bits, const CarryGeomRT& geo,                \
      cudaStream_t stream) {                                                                    \
    return launch<CarryArm##I_>(pr, pw, gw, v, rows, o, den, BH, L, tok_words, v_row_bytes,      \
                                s_in, s_out, page_tbl, atom_bits, geo, stream);                 \
  }                                                                                             \
  cudaError_t carry_arm_rows_##I_(const __nv_bfloat16* pr, const __nv_bfloat16* pw,             \
                                  uint32_t* tbl, int BH, int L, const CarryGeomRT& geo,         \
                                  cudaStream_t stream) {                                        \
    return launch_rows<D_>(pr, pw, tbl, BH, L, geo, stream);                                    \
  }                                                                                             \
  void carry_arm_census_##I_(int64_t* out, const CarryGeomRT& geo) {                            \
    const auto c = census<CarryArm##I_>(geo);                                                   \
    for (int i = 0; i < kCarryCensusFields; ++i) out[i] = c[i];                                 \
  }                                                                                             \
  void carry_arm_row_##I_(int64_t* out) {                                                       \
    using A = CarryArm##I_;                                                                     \
    const int64_t row[kCarryRowFields] = {D_,                                                   \
                                          DV_,                                                  \
                                          W_,                                                   \
                                          C_,                                                   \
                                          KW_,                                                  \
                                          WPS_,                                                 \
                                          (int64_t)(SPLIT_),                                    \
                                          (int64_t)A::BC,                                       \
                                          A::geom()[0],                                         \
                                          A::geom()[2],                                         \
                                          (int64_t)kCarryRowSchema};                                        \
    for (int i = 0; i < kCarryRowFields; ++i) out[i] = row[i];                                  \
  }

}  // namespace rola::carry
