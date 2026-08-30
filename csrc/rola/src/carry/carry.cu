// csrc/rola/src/carry/carry.cu -- the carry family's DISPATCH translation unit:
// dispatch and torch entries. THE ARM LIST LEFT THIS FILE AT K36 (tools/gen_shards.py
// + one TU per arm, KERNEL_STANDARDS section 7); this file instantiates no kernel,
// it matches a call against the generated rows. Full arm catalogue (12 arms, their
// (D,k,m,s,BC,W,C) and why each exists): see docs/internals/carry/carry.md#file-header-arms

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <array>

#include "carry_api.cuh"
#include "carry_arm_abi.cuh"
#include "carry_rows.cuh"
#include "geom.cuh"
//: FOR THE BUILD STAMP ONLY -- the caps digest.  The kernel header is DELIBERATELY -- see docs/internals/carry/carry.md#near-line-81
#include "common/design.cuh"

namespace rola::carry {

//: the caps digest's namespace, aliased as the kernel header aliases it, so the -- see docs/internals/carry/carry.md#design
namespace design = denseref::design;

//: THE PER-LEVEL MODE DECLARATION (K33 law 3) is two bits per level with level -- see docs/internals/carry/carry.md#note-l92

//: THE BUILT SET, GENERATED (K36): the rows, and which of them this build -- see docs/internals/carry/carry.md#build-rows
#include "../instantiations/carry_arms.inc"

//: THE ADDRESSING BLOCK, DERIVED AND REFUSED AT THE SEAM.  Every launch-time law K50
//: names lives here; `min_l B_l < 16` is a CENSUS FACT and not one of them.
//: -- see docs/internals/carry/carry.md#block
static CarryGeomRT derive_block(const std::vector<int64_t>& widths, int64_t level_modes, int bc,
                                int dv, int nsr, int nsw) {
  int w[kGeomMaxLevels] = {1, 1, 1, 1};
  const int D = (int)widths.size();
  TORCH_CHECK(D >= 1 && D <= kGeomMaxLevels, "the routing depth is one to four levels, got ", D);
  for (int l = 0; l < D; ++l) w[l] = (int)widths[l];
  CarryGeomRT geo{};
  const char* err = derive_carry_geom(D, w, bc, dv, (uint32_t)level_modes, nsr, nsw, geo);
  TORCH_CHECK(err == nullptr, "the addressing block refuses this call: ", err == nullptr ? "" : err,
              " (widths=", widths, ", BC=", bc, ", level_modes=", level_modes, ")");
  TORCH_CHECK(geo.wtot % 8 == 0,
              "the packed amplitude row must be a whole number of sixteen-byte granules: "
              "sum_l B_l = ",
              geo.wtot);
  for (int l = 0; l < D; ++l)
    TORCH_CHECK(
        geo.col_base[l] % geo.r.row_span[l] == 0 && geo.col_base[l] % geo.w.row_span[l] == 0,
        "level ", l, "'s column base ", geo.col_base[l],
        " is not a whole number of the runs read out of it (read ", geo.r.row_span[l], ", write ",
        geo.w.row_span[l], "): the level runs would not be aligned");
  return geo;
}

//: THE TABLE, AS A TENSOR, at the block's own grain. -- see docs/internals/carry/carry.md#rpr
template <int D>
static torch::Tensor build_rows(const torch::Tensor& pr, const torch::Tensor& pw, int BH, int L,
                                const CarryGeomRT& geo) {
  auto tbl =
      torch::empty({BH, (long)rows_entries(geo), tok_words(L)}, pr.options().dtype(torch::kInt32));
  C10_CUDA_CHECK((launch_rows<D>(
      (const __nv_bfloat16*)pr.data_ptr(), (const __nv_bfloat16*)pw.data_ptr(),
      reinterpret_cast<uint32_t*>(tbl.data_ptr()), BH, L, geo, at::cuda::getCurrentCUDAStream())));
  return tbl;
}

//: THE SEAM'S KEY IS THE ARM'S SHAPE (K50).  The digit widths and the mode word are
//: NOT matched -- they are derived into the block and refused there if the shape cannot
//: hold them, which is what lets one binary serve a topology nobody compiled. -- see docs/internals/carry/carry.md#carry-match
#define CARRY_MATCH(D_, DV_, W_, C_, KW_, WPS_, SPLIT_)                                    \
  (is_arm(widths, D_) && d_v == DV_ && window == W_ && chunk == C_ && split == (SPLIT_))
#define CARRY_BC(D_, DV_, W_, C_, KW_, WPS_, SPLIT_) (rola::carry::leaves_per_warp(DV_) * (KW_))
#define CARRY_DV(D_, DV_, W_, C_, KW_, WPS_, SPLIT_) (DV_)
#define CARRY_NSR(D_, DV_, W_, C_, KW_, WPS_, SPLIT_) ((SPLIT_) ? 1 : (KW_))
#define CARRY_NSW(D_, DV_, W_, C_, KW_, WPS_, SPLIT_) ((SPLIT_) ? 1 : (KW_) / (WPS_))
//: THE ARM'S OWN SHAPE-SET SIZE, from the SAME generator the bodies compile against --
//: R9's enforcement layer 3, the launch-time refusal that names the uncovered member.
#define CARRY_NA(D_, DV_, W_, C_, KW_, WPS_, SPLIT_)                                       \
  (rola::carry::sub_boxes((D_), rola::carry::leaves_per_warp(DV_) * (KW_),                \
                           (SPLIT_) ? 1 : (KW_))                                           \
       .count)

//: THE ROW'S OWN LAW, checked where it is written. -- see docs/internals/carry/carry.md#carry-row-gate
#define CARRY_ROW_GATE(I_, D_, DV_, W_, C_, KW_, WPS_, SPLIT_)                             \
  static_assert((KW_) == rola::carry::kWarps && (WPS_) >= 1 && (KW_) % (WPS_) == 0          \
                    && ((SPLIT_) == 0 || (SPLIT_) == 1),                                    \
                "an arm's executor map is its own arithmetic: one READ stream per warp "    \
                "over `WPS` warps a stream, and the body flag is a body, not a geometry");
CARRY_ARMS_X(CARRY_ROW_GATE)
#undef CARRY_ROW_GATE

//: WHAT EACH BUILT ARM'S TRANSLATION UNIT EXPORTS. -- see docs/internals/carry/carry.md#carry-arms-x
CARRY_ARMS_X(CARRY_ARM_DECLARE)

static bool is_arm(const std::vector<int64_t>& widths, int D) { return (int)widths.size() == D; }

std::vector<torch::Tensor> forward_inter(torch::Tensor pr, torch::Tensor pw, torch::Tensor gw,
                                         torch::Tensor v, std::vector<int64_t> widths,
                                         int64_t window, int64_t chunk, int64_t split,
                                         int64_t level_modes, c10::optional<torch::Tensor> s_in,
                                         c10::optional<torch::Tensor> s_out,
                                         c10::optional<torch::Tensor> page_tbl,
                                         c10::optional<torch::Tensor> atom_bits) {
  TORCH_CHECK(pr.is_cuda() && pw.is_cuda() && gw.is_cuda() && v.is_cuda(), "cuda tensors only");
  TORCH_CHECK(pr.scalar_type() == torch::kBFloat16 && pw.scalar_type() == torch::kBFloat16
                  && gw.scalar_type() == torch::kBFloat16 && v.scalar_type() == torch::kBFloat16,
              "the operand planes are bf16");
  TORCH_CHECK(pr.is_contiguous() && pw.is_contiguous() && v.is_contiguous(), "contiguous planes");
  //: `pr`/`pw` are `[BH, L, sum_l B_l]`; `v` is `[B, T, H, d_v]`; `gw` is `[BH, L]`.
  const int BH = (int)pr.size(0);
  const int L = (int)pr.size(1);
  //: THE VALUE WIDTH IS READ OFF THE CALLER'S PLANE and is part of the arm key, so an -- see docs/internals/carry/carry.md#dv
  const int d_v = (int)v.size(-1);
  const int v_row_bytes = (int)(v.size(2) * d_v * v.element_size());

  auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(v.device());
  auto o = torch::zeros({BH, L, (int64_t)d_v}, opt);
  auto den = torch::zeros({BH, L}, opt);

  //: THE STATE PLANES ARE THE CALLER'S (docs/internals/state.md section 4). -- see docs/internals/carry/carry.md#torch-check
  for (const auto& t : {s_in, s_out})
    if (t.has_value())
      TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->scalar_type() == torch::kFloat32,
                  "a state plane is a contiguous fp32 cuda tensor");
  if (page_tbl.has_value()) {
    TORCH_CHECK(page_tbl->is_cuda() && page_tbl->is_contiguous()
                    && page_tbl->scalar_type() == torch::kInt32,
                "the page table is a contiguous int32 cuda tensor");
    TORCH_CHECK(s_in.has_value() || s_out.has_value(),
                "a page table addresses a state plane, and this call carries neither");
  }
  const float* sin_p = s_in.has_value() ? s_in->data_ptr<float>() : nullptr;
  float* sout_p = s_out.has_value() ? s_out->data_ptr<float>() : nullptr;
  const int32_t* ptbl_p =
      page_tbl.has_value() ? reinterpret_cast<const int32_t*>(page_tbl->data_ptr()) : nullptr;
  //: THE ACTIVITY FACT IS THE FACTS PASS' (P80b), and it is OPTIONAL at this seam -- see docs/internals/carry/carry.md#atom-bits
  if (atom_bits.has_value()) {
    long N = 1;
    for (auto w : widths) N *= w;
    TORCH_CHECK(atom_bits->is_cuda() && atom_bits->is_contiguous()
                    && atom_bits->scalar_type() == torch::kUInt8,
                "the atom activity bitmap is a contiguous uint8 cuda tensor");
    TORCH_CHECK(atom_bits->numel() == (long)BH * (N / kAtomLeaves),
                "the activity bitmap holds one byte per atom: BH * N / ", kAtomLeaves,
                " entries, got ", atom_bits->numel());
    TORCH_CHECK(s_in.has_value() || s_out.has_value(),
                "an activity bitmap gates a state sweep, and this call carries neither plane");
    TORCH_CHECK(!(s_in.has_value() && s_out.has_value()) || s_in->data_ptr() == s_out->data_ptr(),
                "a gated continuation ADVANCES IN PLACE (docs/internals/state.md): "
                "the exit sweep stores the atoms this call writes and no "
                "others, so an atom it skips must already hold what it carried in. Two "
                "distinct planes would silently drop every unwritten atom. Pass the same "
                "plane as both, or pass no activity bitmap.");
  }
  const uint8_t* abits_p =
      atom_bits.has_value() ? reinterpret_cast<const uint8_t*>(atom_bits->data_ptr()) : nullptr;
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaError_t err = cudaErrorInvalidValue;
  //: THE BALLOT ROWS COME FIRST, ON THE SAME STREAM (K31-i-d). One pass over the -- see docs/internals/carry/carry.md#tok-words
  torch::Tensor rows;
  const int tokWords = tok_words(L);
  auto run = [&]() {
    bool matched = false;
#define CARRY_DISPATCH(I_, ...)                                                             \
  if (CARRY_MATCH(__VA_ARGS__)) {                                                            \
    matched = true;                                                                         \
    const CarryGeomRT geo = derive_block(widths, level_modes, CARRY_BC(__VA_ARGS__),         \
                                         CARRY_DV(__VA_ARGS__), CARRY_NSR(__VA_ARGS__),       \
                                         CARRY_NSW(__VA_ARGS__));                            \
    TORCH_CHECK(geo.r.assign >= 0 && geo.r.assign < CARRY_NA(__VA_ARGS__)                    \
                    && geo.w.assign >= 0 && geo.w.assign < CARRY_NA(__VA_ARGS__),             \
                "the mode word selects a warp sub-box this arm does not carry: read ",       \
                (int64_t)geo.r.assign, " write ", (int64_t)geo.w.assign, " of ",             \
                (int64_t)CARRY_NA(__VA_ARGS__));                                             \
    rows = torch::empty({BH, (long)rows_entries(geo), tokWords},                            \
                        pr.options().dtype(torch::kInt32));                                 \
    err = carry_arm_rows_##I_((const __nv_bfloat16*)pr.data_ptr(),                          \
                              (const __nv_bfloat16*)pw.data_ptr(),                          \
                              reinterpret_cast<uint32_t*>(rows.data_ptr()), BH, L, geo,      \
                              stream);                                                       \
    if (err == cudaSuccess)                                                                 \
      err = carry_arm_forward_##I_(                                                          \
          (const __nv_bfloat16*)pr.data_ptr(), (const __nv_bfloat16*)pw.data_ptr(),          \
          (const __nv_bfloat16*)gw.data_ptr(), (const __nv_bfloat16*)v.data_ptr(),           \
          reinterpret_cast<const uint32_t*>(rows.data_ptr()), o.data_ptr<float>(),           \
          den.data_ptr<float>(), BH, L, tokWords, v_row_bytes, sin_p, sout_p, ptbl_p,      \
          abits_p, geo, stream);                                                             \
  } else
    CARRY_ARMS_X(CARRY_DISPATCH) {}
#undef CARRY_DISPATCH
    return matched;
  };
  //: THE SHAPE IS RESOLVED AT THIS SEAM, AND AN UNBUILT ONE IS REFUSED (P76 -- see docs/internals/carry/carry.md#torch-check-2
  TORCH_CHECK(run(), "unbuilt carry arm: depth=", (int64_t)widths.size(), " d_v=", d_v,
              " window=", window, " chunk=", chunk, " split=", split);
  C10_CUDA_CHECK(err);
  return {o, den};
}

//: THE FILE'S OWN CONSTANTS, device-side, beside the magic: the channel-split body's -- see docs/internals/carry/carry.md#stampkernelentry
__global__ void stamp_kernel_entry(int* out) {
  out[0] = kCarryBuildMagic;
  out[1] = kWarps * 1000 + kAtomLeaves;
  out[2] = design::kCapsDigest;
}

std::vector<int64_t> build_stamp(std::vector<int64_t> widths, int64_t window, int64_t chunk,
                                 int64_t split, int64_t level_modes, int64_t d_v) {
  auto t = torch::zeros({3}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  stamp_kernel_entry<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(t.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto h = t.cpu();
  std::vector<int64_t> r{h.data_ptr<int>()[0], h.data_ptr<int>()[1], h.data_ptr<int>()[2]};
#define CARRY_CENSUS(I_, ...)                                                               \
  if (CARRY_MATCH(__VA_ARGS__)) {                                                            \
    const CarryGeomRT geo = derive_block(widths, level_modes, CARRY_BC(__VA_ARGS__),         \
                                         CARRY_DV(__VA_ARGS__), CARRY_NSR(__VA_ARGS__),       \
                                         CARRY_NSW(__VA_ARGS__));                            \
    int64_t c[kCarryCensusFields];                                                          \
    carry_arm_census_##I_(c, geo);                                                           \
    for (int i = 0; i < kCarryCensusFields; ++i) r.push_back(c[i]);                           \
    return r;                                                                               \
  }
  CARRY_ARMS_X(CARRY_CENSUS)
#undef CARRY_CENSUS
  TORCH_CHECK(false, "unbuilt carry arm");
  return r;
}

torch::Tensor owner_rows(torch::Tensor pr, torch::Tensor pw, std::vector<int64_t> widths,
                         int64_t level_modes, bool carve) {
  TORCH_CHECK(pr.is_cuda() && pw.is_cuda() && pr.is_contiguous() && pw.is_contiguous(),
              "contiguous cuda planes");
  TORCH_CHECK(pr.scalar_type() == torch::kBFloat16 && pw.scalar_type() == torch::kBFloat16,
              "the operand planes are bf16");
  TORCH_CHECK(pr.sizes() == pw.sizes() && pr.dim() == 3, "expect packed amplitudes [BH, L, sumW]");
  const int BH = (int)pr.size(0), L = (int)pr.size(1);
  //: THE DIAGNOSTIC IS THE SAME DERIVATION the body runs: `carve = false` is the
  //: OWNER's own box (one stream a side), `carve = true` is the carved body's grain
  //: (`kSeamCarveStreams` streams a side, ordered by the mode word).
  const int ns = carve ? kSeamCarveStreams : 1;
#define CARRY_ROWS(I_, D_, DV_, W_, C_, KW_, WPS_, SPLIT_)                                 \
  if (is_arm(widths, D_)) {                                                                 \
    const CarryGeomRT geo =                                                                 \
        derive_block(widths, level_modes, rola::carry::leaves_per_warp(DV_) * (KW_), DV_, ns,  \
                     ns);                                                                    \
    return build_rows<D_>(pr, pw, BH, L, geo);                                               \
  }
  CARRY_ARMS_X(CARRY_ROWS)
#undef CARRY_ROWS
  TORCH_CHECK(false, "no built carry arm carries this routing depth");
}

//: THE DERIVATION, EXPOSED.  The model test golden-checks THIS function -- the one the
//: dispatch runs -- rather than a Python mirror of it, because a mirror proves only
//: that two derivations agree with each other. -- see docs/internals/carry/carry.md#geometry
std::vector<int64_t> geometry(std::vector<int64_t> widths, int64_t level_modes, int64_t bc,
                              int64_t nsr, int64_t nsw, int64_t dv) {
  const CarryGeomRT g = derive_block(widths, level_modes, (int)bc, (int)dv, (int)nsr, (int)nsw);
  std::vector<int64_t> o;
  auto push_side = [&](const SideGeom& s) {
    for (int l = 0; l < kGeomMaxLevels; ++l) o.push_back(s.level_at[l]);
    for (int l = 0; l < kGeomMaxLevels; ++l) o.push_back(s.rank[l]);
    for (int l = 0; l < kGeomMaxLevels; ++l) o.push_back(s.row_span[l]);
    for (int l = 0; l < kGeomMaxLevels; ++l) o.push_back(s.grid[l]);
    for (int l = 0; l < kGeomMaxLevels; ++l) o.push_back(s.grid_div[l]);
    for (int l = 0; l < kGeomMaxLevels; ++l) o.push_back(s.row_g[l]);
    for (int l = 0; l < kGeomMaxLevels; ++l) o.push_back(s.row_prefix[l]);
    for (int l = 0; l < kGeomMaxLevels; ++l) o.push_back(s.sub_bits[l]);
    for (int l = 0; l < kGeomMaxLevels; ++l) o.push_back(s.sub_shift[l]);
    o.push_back(s.carves);
    o.push_back(s.rows);
    o.push_back(s.leaf);
    o.push_back(s.streams);
    o.push_back(s.assign);
  };
  o.push_back(g.D);
  o.push_back(g.bc);
  for (int l = 0; l < kGeomMaxLevels; ++l) o.push_back(g.width[l]);
  for (int l = 0; l < kGeomMaxLevels; ++l) o.push_back(g.span[l]);
  for (int l = 0; l < kGeomMaxLevels; ++l) o.push_back(g.grid[l]);
  for (int l = 0; l < kGeomMaxLevels; ++l) o.push_back(g.col_base[l]);
  for (int l = 0; l < kGeomMaxLevels; ++l) o.push_back(g.run_shift[l]);
  for (int l = 0; l < kGeomMaxLevels; ++l) o.push_back(g.owner_div[l]);
  for (int l = 0; l < kGeomMaxLevels; ++l) o.push_back(g.weight[l]);
  o.push_back(g.wtot);
  o.push_back(g.owners);
  o.push_back(g.leaves);
  o.push_back(g.local_bits);
  o.push_back(g.min_width);
  o.push_back(g.identity_map);
  push_side(g.r);
  push_side(g.w);
  //: THE TRANSIT'S ROW MAP, BOTH SIDES, evaluated over the whole box: the model test
  //: asserts each side's map is a bijection of the region, that the two together are a
  //: round trip, and that they coincide exactly when `identity_map` says so.
  for (int w = 0; w < g.w.streams; ++w)
    for (int P = 0; P < g.w.leaf; ++P) o.push_back(geom_xch_row(g, g.w, w, P));
  for (int r = 0; r < g.r.streams; ++r)
    for (int P = 0; P < g.r.leaf; ++P) o.push_back(geom_xch_row(g, g.r, r, P));
  return o;
}

std::vector<std::vector<int64_t>> arms() {
  std::vector<std::vector<int64_t>> out;
#define CARRY_LIST(I_, ...)                                     \
  {                                                             \
    int64_t row[kCarryRowFields];                               \
    carry_arm_row_##I_(row);                                    \
    out.push_back(std::vector<int64_t>(row, row + kCarryRowFields)); \
  }
  CARRY_ARMS_X(CARRY_LIST)
#undef CARRY_LIST
  return out;
}

}  // namespace rola::carry
