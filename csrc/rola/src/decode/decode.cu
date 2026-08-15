// THE T=1 DECODE GEMV -- token-stationary, state-streaming, bitmap-walked. Owns the
// GEMV kernel, its 32-way instantiation matrix and the decode path's host entry.
//
// The ratified shape, literally: the token's routing weights and value are RESIDENT
// (SMEM/registers), the STATE streams past, and the grid partitions the support.
//
// Three facts a reader must not miss:
//   1. There is one token, so the contraction index is the LEAF and the output index is
//      the COLUMN. Lanes map to OUTPUT COLUMNS: no MMA, and no warp reduction either --
//      the dot never crosses a lane.
//   2. DETERMINISM IS A SHIPPED PROPERTY HERE and it diverges from the prefill arm
//      ON PURPOSE: `y` is combined in SLOT ORDER, never by an `atomicAdd` race, because
//      decode's `y` picks the next token and a last-bit flip compounds
//      autoregressively. The state needs no argument at all -- the row partition is a
//      PARTITION and no atomic touches `state`.
//   3. NO BARRIER PER ANYTHING: two `__syncthreads` per CTA per step (three when
//      `n_split > 1`) against a tiled prefill's six per candidate tile.
//
// The GEMV/GEMM argument, the measured reproducibility divergence, the barrier ledger,
// and the derivation that collapses the causal intra Gram into UPDATE-THEN-READ:
// docs/internals/decode/decode.md

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include "decode.cuh"
#include "decode_api.cuh"
#include "dispatch_switch.cuh"

namespace rola {
namespace decode {
namespace detail {

//: `expf(88) = 1.65e38 < FLT_MAX`. THE SAME VALUE AND THE SAME CONVENTION as
//: `intra_panel.cuh`'s `kPanelMaxExponent`, restated because decode shares none of that
//: header's panel algebra. On this path the clamp is STRUCTURALLY INERT -- the exponent
//: cannot go positive -- and is written anyway so ONE convention governs every survival
//: exponential in the repository: docs/internals/decode/decode.md#decode-max-exponent
constexpr float kDecodeMaxExponent = 88.0f;

//: Locate the bitmap word holding global set-bit index `row`, and how many set bits of
//: that word precede it. Two steps, NEITHER a scan of the bitmap: a binary search over
//: the coarse index, then at most 32 popcounts. docs/internals/decode/decode.md#locate-row
__device__ __forceinline__ void locate_row(const int32_t* pc, const uint32_t* bits_r,
                                           const uint32_t* bits_w, int n_words, int n_groups,
                                           int row, int& out_word, int& out_skip) {
  int lo = 0;
  int hi = n_groups;
  while (lo < hi - 1) {
    const int mid = (lo + hi) >> 1;
    if (pc[mid] <= row)
      lo = mid;
    else
      hi = mid;
  }
  int acc = pc[lo];
  int w = lo * kCoarseGroupWords;
  while (w < n_words) {
    const int c = __popc(bits_r[w] | bits_w[w]);
    if (acc + c > row) break;
    acc += c;
    ++w;
  }
  out_word = w;
  out_skip = row - acc;
}

template <int DV, int D, bool DECAY>
__global__ __launch_bounds__(kDecodeThreads) void decode_gemv_kernel(DecodeParams p) {
  //: HAZARD gate-before-anything -- docs/internals/decode/decode.md#gate-before-anything
  if (p.gate != nullptr && *p.gate != 0) return;
  //: The mass column is UNCONDITIONAL (raw is dead), which is why the shape is a
  //: template axis and not a runtime test: docs/internals/decode/decode.md#template-axes
  constexpr int cols = DV + 1;
  //: Columns per lane: `NC` fully coalesced 128 B transactions, and the lane's
  //: accumulators are its own output columns.
  constexpr int NC = DV / 32;

  extern __shared__ char smem_raw[];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int seg = blockIdx.x;
  const int bh = blockIdx.y;
  const int nw = p.n_words;
  const int ng = p.n_groups;
  const int total_rows = p.total_rows;

  // ---- SMEM ledger ----------------------------------------------------------
  //   sa_r, sa_w   [total_rows] fp32   per-level amplitude rows, UNCOMPACTED and indexed
  //                                    by DIGIT: the bitmap carries membership, so the
  //                                    digit IS the index.
  //   sa_c, sd     [total_rows] fp32   clock amplitudes and this head's dials; DECAY only
  //   v_tok        [DV]         fp32   the token's value, resident
  //   fold  [warps][cols+3]     fp32   the cross-warp partials, bank-padded (see below)
  float* sa_r = reinterpret_cast<float*>(smem_raw);
  float* sa_w = sa_r + total_rows;
  float* sa_c = sa_w + total_rows;
  float* sd = sa_c + (DECAY ? total_rows : 0);
  float* v_tok = sd + (DECAY ? total_rows : 0);
  float* fold = v_tok + DV;

  __shared__ float part[cols + 3];
  __shared__ int s_w_start, s_w_last, s_drop, s_keep;
  __shared__ int s_last_cta;

  // ---- stage the token: routing rows, dials, value, row scales ---------------
  for (int l = 0; l < D; ++l) {
    const int width = p.level_width[l];
    const int off = p.level_amp_offset[l];
    const float* rr = p.amp_read[l] + static_cast<int64_t>(bh) * width;
    const float* rw = p.amp_write[l] + static_cast<int64_t>(bh) * width;
    for (int g = tid; g < width; g += kDecodeThreads) {
      sa_r[off + g] = rr[g];
      sa_w[off + g] = rw[g];
    }
    if (DECAY) {
      const float* rc = p.amp_clock[l] + static_cast<int64_t>(bh) * width;
      const int h = bh % p.H;
      const float* rd = p.dials + static_cast<int64_t>(h) * total_rows + p.level_row_offset[l];
      for (int g = tid; g < width; g += kDecodeThreads) {
        sa_c[off + g] = rc[g];
        sd[off + g] = rd[g];
      }
    }
  }
  for (int c = tid; c < DV; c += kDecodeThreads) {
    v_tok[c] = p.v[static_cast<int64_t>(bh) * DV + c];
  }

  //: The two per-token row scales. At `T = 1` there is no owner rectangle, so what
  //: remains here is exactly the per-TOKEN part -- the side gains; the per-LEVEL factors
  //: the tiled kernel hoists stay in the per-LEAF product.
  //: docs/internals/decode/decode.md#row-scales
  const float sr = 1.0f;  // global fixes g_read := 1 (raw is dead)
  const float sw = p.g_write[bh];

  const uint32_t* bits_r = p.bits_r + static_cast<int64_t>(bh) * nw;
  const uint32_t* bits_w = p.bits_w + static_cast<int64_t>(bh) * nw;
  const int32_t* pc = p.pc_coarse + static_cast<int64_t>(bh) * (ng + 1);

  // ---- resolve this CTA's ROW range into a WORD range -------------------------
  // Defined in ROWS (popcount-balanced by construction), handed out in WORDS; the two
  // boundary words are cut by dropping / keeping set bits, exact rather than by an index
  // comparison. docs/internals/decode/decode.md#row-range-to-words
  if (tid == 0) {
    const int M = pc[ng];
    const int row_lo = static_cast<int>(static_cast<int64_t>(seg) * M / p.n_split);
    const int row_hi = static_cast<int>(static_cast<int64_t>(seg + 1) * M / p.n_split);
    if (row_hi <= row_lo) {
      s_w_start = 1;
      s_w_last = 0;  // empty range: the walk below runs zero iterations
      s_drop = 0;
      s_keep = 0;
    } else {
      int w0, skip0, w1, skip1;
      locate_row(pc, bits_r, bits_w, nw, ng, row_lo, w0, skip0);
      locate_row(pc, bits_r, bits_w, nw, ng, row_hi - 1, w1, skip1);
      s_w_start = w0;
      s_drop = skip0;
      s_w_last = w1;
      s_keep = skip1 + 1;
    }
  }
  __syncthreads();

  const int w_start = s_w_start;
  const int w_last = s_w_last;
  const int drop_first = s_drop;
  const int keep_last = s_keep;

  float acc[NC];
#pragma unroll
  for (int k = 0; k < NC; ++k) acc[k] = 0.0f;
  float acc_m = 0.0f;

  // ---- THE WALK --------------------------------------------------------------
  int dig[D];
  float o_r = 1.0f, o_w = 1.0f, o_c = 1.0f, o_rate = 1.0f;
  auto recompute_outer = [&]() {
    o_r = 1.0f;
    o_w = 1.0f;
    o_c = 1.0f;
    o_rate = 1.0f;
#pragma unroll
    for (int l = 0; l < D - 1; ++l) {
      const int off = p.level_amp_offset[l] + dig[l];
      o_r *= sa_r[off];
      o_w *= sa_w[off];
      if (DECAY) {
        o_c *= sa_c[off];
        o_rate *= sd[off];
      }
    }
  };

  const int b_last = p.level_width[D - 1];
  const int off_last_base = p.level_amp_offset[D - 1];

  //: THE PAGED BASE TRANSLATION, resolved ONCE PER TOUCHED ATOM. The table row is pinned
  //: here (a plain origin, no per-atom widening multiply); the cache is keyed on the atom
  //: VALUE, so it is correct whatever order the walk reaches leaves in and costs at most
  //: two int32 loads per 32-leaf bitmap word. `atom_base` is the atom's first row as an
  //: ELEMENT offset from `p.state`: dense is `bh * N + (atom << 4)` (which is why a
  //: topology whose `N` is not a whole number of atoms still addresses correctly -- the
  //: two halves recombine to `bh * N + leaf`), paged is `slot << 4`.
  //: docs/internals/decode/decode.md#paged-address
  const int32_t* page_row =
      (p.page_tbl == nullptr) ? nullptr : p.page_tbl + static_cast<int64_t>(bh) * p.atoms_per_bh;
  const int64_t dense_bh_base = static_cast<int64_t>(bh) * p.N;
  int cached_atom = -1;
  int64_t atom_base = 0;
  bool atom_present = false;

  int w = w_start + warp;
  int g_cached = -1;
  bool g_empty = false;
  while (w <= w_last) {
    //: THE COARSE-INDEX CONSULT AT WALK TIME: a group whose prefix does not advance
    //: holds no touched row, so its 32 words are never read. Once per GROUP, not once
    //: per word: docs/internals/decode/decode.md#coarse-consult
    const int grp = w >> 5;
    if (grp != g_cached) {
      g_cached = grp;
      g_empty = (pc[grp + 1] == pc[grp]);
    }
    if (g_empty) {
      const int next = (grp + 1) << 5;
      w += ceil_div_int(next - w, kDecodeWarps) * kDecodeWarps;
      continue;
    }

    const uint32_t rw_word = bits_r[w];
    const uint32_t ww_word = bits_w[w];
    //: THE UNION, formed in a REGISTER and never stored.
    uint32_t u = rw_word | ww_word;
    //: HAZARD boundary-cut-order -- docs/internals/decode/decode.md#boundary-cut-order
    if (w == w_last) u = keep_low_set_bits(u, keep_last);
    if (w == w_start) u = drop_low_set_bits(u, drop_first);
    if (u == 0u) {
      w += kDecodeWarps;
      continue;
    }

    const int leaf_base = w * 32;
    //: The `D` integer divisions are paid ONCE PER WORD, not once per row -- a per-row
    //: decomposition would make the A100 dense arm instruction-bound:
    //: docs/internals/decode/decode.md#divisions-per-word
    {
      int rem = leaf_base;
#pragma unroll
      for (int l = 0; l < D; ++l) {
        const int s = p.level_stride[l];
        dig[l] = rem / s;
        rem -= dig[l] * s;
      }
    }
    recompute_outer();

    int q = 0;
    int cur = dig[D - 1];
    //: The word is walked in ALIGNED SUB-RANGES, each inside ONE inner block, so the
    //: innermost digit needs no division and the outer digits are constant. ONE uniform
    //: loop that degenerates, not a fast path with a fallback:
    //: docs/internals/decode/decode.md#aligned-subranges
    while (q < 32) {
      int take = b_last - cur;
      if (take > 32 - q) take = 32 - q;
      uint32_t sub = (take >= 32) ? u : ((u >> q) & ((1u << take) - 1u));
      while (sub != 0u) {
        const int t = __ffs(static_cast<int>(sub)) - 1;
        sub &= sub - 1u;
        const int bit = q + t;
        //: Bits at or beyond `N` are never set (the expansion's loop stops at
        //: `span_out`), so the walk needs no leaf bound test.
        const int leaf = leaf_base + bit;
        const int off_last = off_last_base + cur + t;
        const bool in_r = ((rw_word >> bit) & 1u) != 0u;
        const bool in_w = ((ww_word >> bit) & 1u) != 0u;

        const int atom = leaf >> kAtomShift;
        if (atom != cached_atom) {
          cached_atom = atom;
          if (page_row != nullptr) {
            const int slot = page_row[atom];
            atom_present = slot >= 0;
            atom_base = static_cast<int64_t>(slot << kAtomShift) * cols;
          } else {
            atom_present = true;
            atom_base = (dense_bh_base + (atom << kAtomShift)) * cols;
          }
        }

        //: AN ABSENT ATOM READS AS ZEROS AND IS NOT WRITTEN. Exact, not defensive: the
        //: plan admits the step's write set exactly, so an atom with no slot is outside
        //: it, and a never-deposited leaf holds zeros in the dense backing too. The
        //: self-normalizing readout then contributes exactly nothing through it.
        //: docs/internals/decode/decode.md#absent-atom
        float* row = atom_present ? (p.state + atom_base + (leaf & kAtomMask) * cols) : nullptr;
        float sv[NC];
#pragma unroll
        for (int k = 0; k < NC; ++k) sv[k] = 0.0f;
        float smass = 0.0f;
        if (atom_present) {
#pragma unroll
          for (int k = 0; k < NC; ++k) sv[k] = row[lane + 32 * k];
          if (lane == 0) smass = row[DV];
        }

        //: UPDATE FIRST, THEN READ THE UPDATED ROW -- AND THAT ORDER IS THE SEMANTICS,
        //: not an optimization: survival is INCLUSIVE at `j = t`, so the whole
        //: causal intra Gram collapses into reading the row after it is written. A row
        //: in `supp(R) \ supp(W)` has `keep == 1` and no deposit, so the same expression
        //: is its readout too -- one path, no arm.
        //: HAZARD update-then-read -- docs/internals/decode/decode.md#update-then-read
        if (in_w && atom_present) {
          //: DECAY AT READ, VIA THE DETACHED CLOCK, WITH NO SWEEP AND NO EXPONENT
          //: SPLIT: at `T = 1` the tiled kernel's rule-1 half-mass shift is exactly
          //: the identity, so ONE `expf` replaces its three. `log1pf` IS the composition
          //: (never a rounded `1 - rate`) and there is no epsilon floor -- both of that rule's
          //: rule 2's prohibitions: docs/internals/decode/decode.md#decay-at-read
          float keep = 1.0f;
          if (DECAY) {
            const float rate = o_rate * sd[off_last];
            const float clock = o_c * sa_c[off_last];
            keep = expf(fminf(clock * log1pf(-rate), kDecodeMaxExponent));
          }
          const float wv = (o_w * sa_w[off_last]) * sw;
#pragma unroll
          for (int k = 0; k < NC; ++k) {
            sv[k] = keep * sv[k] + wv * v_tok[lane + 32 * k];
            row[lane + 32 * k] = sv[k];
          }
          //: The mass column is the `v := 1` column of the identical recurrence.
          if (lane == 0) {
            smass = keep * smass + wv;
            row[DV] = smass;
          }
        }
        if (in_r) {
          const float a = o_r * sa_r[off_last];
#pragma unroll
          for (int k = 0; k < NC; ++k) acc[k] += a * sv[k];
          if (lane == 0) acc_m += a * smass;
        }
      }
      q += take;
      cur += take;
      if (cur >= b_last) {
        cur = 0;
#pragma unroll
        for (int l = D - 2; l >= 0; --l) {
          if (++dig[l] < p.level_width[l]) break;
          dig[l] = 0;
        }
        recompute_outer();
      }
    }
    w += kDecodeWarps;
  }

  // ---- the 8-way cross-warp fold, in FIXED warp order ------------------------
  //: HAZARD fold-bank-padding -- docs/internals/decode/decode.md#fold-bank-padding
  const int fold_pitch = cols + 3;
#pragma unroll
  for (int k = 0; k < NC; ++k) fold[warp * fold_pitch + lane + 32 * k] = acc[k];
  if (lane == 0) fold[warp * fold_pitch + DV] = acc_m;
  __syncthreads();
  if (tid < cols) {
    float t = 0.0f;
#pragma unroll
    for (int wq = 0; wq < kDecodeWarps; ++wq) t += fold[wq * fold_pitch + tid];
    part[tid] = sr * t;  // sr == 1 under global; kept as the one scale site
  }
  __syncthreads();

  // ---- the deterministic split-K combine, and the fused divide ---------------
  if (p.n_split == 1) {
    if (tid < DV) {
      const float den = part[DV];
      p.y[static_cast<int64_t>(bh) * DV + tid] = part[tid] / (den + p.eps);
    }
    return;
  }

  float* slot = p.ws + (static_cast<int64_t>(bh) * p.n_split + seg) * cols;
  if (tid < cols) slot[tid] = part[tid];
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    s_last_cta = (atomicAdd(&p.ctr[bh], 1) == p.n_split - 1) ? 1 : 0;
  }
  __syncthreads();
  if (s_last_cta == 0) return;

  //: SLOT ORDER, always -- the whole determinism argument. `__ldcg` bypasses L1 so the
  //: reader sees the stores the `__threadfence` above made visible.
  //: HAZARD slot-order -- docs/internals/decode/decode.md#slot-order
  if (tid < cols) {
    const float* base = p.ws + static_cast<int64_t>(bh) * p.n_split * cols;
    float t = 0.0f;
    for (int s = 0; s < p.n_split; ++s) t += __ldcg(base + static_cast<int64_t>(s) * cols + tid);
    part[tid] = t;
  }
  __syncthreads();
  if (tid < DV) {
    const float den = part[DV];
    p.y[static_cast<int64_t>(bh) * DV + tid] = part[tid] / (den + p.eps);
  }
  //: Reset for the next step: no `memset` launch between two decode steps, and the
  //: counter is safe to capture in a CUDA graph.
  if (tid == 0) p.ctr[bh] = 0;
}

}  // namespace detail

size_t decode_gemv_smem_bytes(int total_rows, int d_v, int cols, bool decay) {
  const size_t amps = static_cast<size_t>(total_rows) * (decay ? 4 : 2);
  const size_t fold = static_cast<size_t>(kDecodeWarps) * (cols + 3);
  return (amps + static_cast<size_t>(d_v) + fold) * sizeof(float);
}

void launch_decode_gemv(const DecodeParams& p, int d_v, bool decay) {
  //: The instantiation matrix is `DV(2) x D(4) x DECAY(2) = 16` (the GLOBAL axis
  //: died with raw). What is NOT an axis -- `BT`, `BC` -- is the interesting part:
  //: docs/internals/decode/decode.md#instantiation-matrix
  TORCH_CHECK(d_v == 32 || d_v == 64, "decode instantiates d_v in {32, 64}, got ", d_v);
  const int cols = d_v + 1;
  const size_t smem = decode_gemv_smem_bytes(p.total_rows, d_v, cols, decay);
  auto stream = at::cuda::getCurrentCUDAStream();
  //: THE LAUNCH IS WRITTEN ONCE. Each axis walker hands its value back as an
  //: `integral_constant`, so the three `::value`s below are template arguments and
  //: the `<<<...>>>` configuration cannot drift between arms:
  //: docs/internals/dispatch_switch.md#one-launch
  int_switch<32, 64>(d_v, "decode d_v", [&](auto DV) {
    bool_switch(decay, [&](auto DECAY) {
      int_switch<1, 2, 3, 4>(p.D, "D", [&](auto LEVELS) {
        detail::decode_gemv_kernel<decltype(DV)::value, decltype(LEVELS)::value,
                                   decltype(DECAY)::value>
            <<<dim3(p.n_split, p.BH), kDecodeThreads, smem, stream>>>(p);
      });
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Host entry
// ---------------------------------------------------------------------------

namespace detail {

void check_f32(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32, got ", t.scalar_type());
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_i32(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == at::kInt, name, " must be int32, got ", t.scalar_type());
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

}  // namespace detail

int64_t rola_decode_capacity() { return kDecodeMaxLevelWidth; }

//: The MIRRORED copy of the producer's `MAX_BRANCH_WIDTH`, bound separately from the
//: capacity because the two are DIFFERENT CLAIMS -- the second is what stops the
//: `static_assert` being satisfied by a stale mirror:
//: docs/internals/decode/decode.md#producer-width-mirror
int64_t rola_decode_producer_width_mirror() { return kProducerMaxBranchWidth; }

at::Tensor rola_decode_forward(std::vector<at::Tensor> amp_read, std::vector<at::Tensor> amp_write,
                               c10::optional<std::vector<at::Tensor>> amp_clock,
                               c10::optional<at::Tensor> dials, at::Tensor g_write, at::Tensor v,
                               at::Tensor state, at::Tensor bits, at::Tensor pc_coarse,
                               at::Tensor ws, at::Tensor ctr, std::vector<int64_t> level_widths,
                               std::vector<int64_t> level_strides, std::vector<int64_t> level_spans,
                               std::vector<int64_t> level_row_offsets, int64_t heads,
                               int64_t n_split, double eps, c10::optional<at::Tensor> page_table,
                               c10::optional<at::Tensor> gate) {
  const int D = static_cast<int>(amp_read.size());
  TORCH_CHECK(D >= 1 && D <= kMaxLevels, "D must be in [1, 4], got ", D);
  TORCH_CHECK(static_cast<int>(amp_write.size()) == D, "amp_write must have D entries");
  TORCH_CHECK(static_cast<int>(level_widths.size()) == D
                  && static_cast<int>(level_strides.size()) == D
                  && static_cast<int>(level_spans.size()) == D
                  && static_cast<int>(level_row_offsets.size()) == D,
              "the four per-level integer vectors must each have D entries");

  detail::check_f32(v, "v");
  detail::check_f32(g_write, "g_write");
  detail::check_f32(state, "state");
  detail::check_i32(bits, "bits");
  detail::check_i32(pc_coarse, "pc_coarse");
  detail::check_f32(ws, "ws");
  detail::check_i32(ctr, "ctr");

  TORCH_CHECK(v.dim() == 2, "v must be [BH, d_v]");
  const int BH = static_cast<int>(v.size(0));
  const int d_v = static_cast<int>(v.size(1));
  const int cols = d_v + 1;  // the mass column is unconditional (raw is dead)
  //: THE TWO BACKINGS, validated where they differ and nowhere else. A page table names
  //: the leaf space (`[BH, N / kAtomLeaves]`) and the plane holds only the slots a plan
  //: committed; without one the plane IS the leaf space. Slot VALUES are not
  //: bounds-checked -- that would be a device-side read of the table, and the plan owns
  //: the bound. docs/internals/decode/decode_api.md#paged-state
  int64_t N = 0;
  int atoms_per_bh = 0;
  const int32_t* page_ptr = nullptr;
  if (page_table.has_value()) {
    const at::Tensor& tbl = *page_table;
    detail::check_i32(tbl, "page_table");
    TORCH_CHECK(tbl.dim() == 2 && tbl.size(0) == BH, "page_table must be [BH, N / ", kAtomLeaves,
                "] int32");
    atoms_per_bh = static_cast<int>(tbl.size(1));
    N = static_cast<int64_t>(atoms_per_bh) * kAtomLeaves;
    TORCH_CHECK(state.dim() == 3 && state.size(1) == kAtomLeaves && state.size(2) == cols,
                "under a page table the state is the arena's plane, [slots, ", kAtomLeaves,
                ", d_v + 1]");
    page_ptr = tbl.data_ptr<int32_t>();
  } else {
    TORCH_CHECK(state.dim() == 3 && state.size(0) == BH && state.size(2) == cols,
                "state must be [BH, N, d_v + 1]");
    N = state.size(1);
  }
  TORCH_CHECK(g_write.dim() == 1 && g_write.size(0) == BH, "g_write must be [BH]");
  TORCH_CHECK(heads >= 1 && BH % heads == 0, "heads must divide BH");

  const int n_words = static_cast<int>((N + 31) / 32);
  const int n_groups = (n_words + kCoarseGroupWords - 1) / kCoarseGroupWords;
  //: ONE FLAT allocation, carved into `bits_r | bits_w | scratch`. Flat rather than
  //: shaped, because a shape that lies is worse than no shape:
  //: docs/internals/decode/decode.md#flat-bits
  TORCH_CHECK(bits.dim() == 1 && bits.size(0) == 4LL * BH * n_words,
              "bits must be a flat [4 * BH * ceil(N/32)] int32 workspace");
  TORCH_CHECK(pc_coarse.dim() == 2 && pc_coarse.size(0) == BH && pc_coarse.size(1) == n_groups + 1,
              "pc_coarse must be [BH, ceil(n_words/32) + 1] int32");
  TORCH_CHECK(n_split >= 1, "n_split must be >= 1, got ", n_split);
  TORCH_CHECK(ws.dim() == 3 && ws.size(0) == BH && ws.size(1) == n_split && ws.size(2) == cols,
              "ws must be [BH, n_split, cols]");
  TORCH_CHECK(ctr.dim() == 1 && ctr.size(0) == BH, "ctr must be [BH] int32");

  const bool decay = amp_clock.has_value();
  TORCH_CHECK(decay == dials.has_value(),
              "the clock stream and the decay dials are one arm: supply both or neither");

  DecodeParams p{};
  int total_rows = 0;
  int64_t expect_n = 1;
  for (int l = 0; l < D; ++l) {
    const int width = static_cast<int>(level_widths[l]);
    TORCH_CHECK(width >= 1 && width <= kDecodeMaxLevelWidth, "level ", l, " width ", width,
                " exceeds the decode path's capacity ", kDecodeMaxLevelWidth,
                " (see kProducerMaxBranchWidth)");
    detail::check_f32(amp_read[l], "amp_read[l]");
    detail::check_f32(amp_write[l], "amp_write[l]");
    TORCH_CHECK(amp_read[l].dim() == 2 && amp_read[l].size(0) == BH && amp_read[l].size(1) == width,
                "amp_read[", l, "] must be [BH, width] contiguous");
    TORCH_CHECK(amp_write[l].sizes() == amp_read[l].sizes(), "amp_write[", l,
                "] must match amp_read[", l, "]");
    p.amp_read[l] = amp_read[l].data_ptr<float>();
    p.amp_write[l] = amp_write[l].data_ptr<float>();
    if (decay) {
      detail::check_f32((*amp_clock)[l], "amp_clock[l]");
      TORCH_CHECK((*amp_clock)[l].sizes() == amp_read[l].sizes(), "amp_clock[", l,
                  "] must match amp_read[", l, "]");
      p.amp_clock[l] = (*amp_clock)[l].data_ptr<float>();
    } else {
      p.amp_clock[l] = nullptr;
    }
    p.level_width[l] = width;
    p.level_stride[l] = static_cast<int>(level_strides[l]);
    p.level_span[l] = static_cast<int>(level_spans[l]);
    p.level_amp_offset[l] = total_rows;
    p.level_row_offset[l] = static_cast<int>(level_row_offsets[l]);
    total_rows += width;
    expect_n *= width;
  }
  TORCH_CHECK(expect_n == N, "prod(level_widths) = ", expect_n,
              " disagrees with the state's leaf axis ", N);
  TORCH_CHECK(p.level_span[0] == static_cast<int>(N),
              "level_spans[0] must be N (the whole leaf space)");
  TORCH_CHECK(p.level_stride[D - 1] == 1,
              "level_strides[D-1] must be 1 (the innermost level is contiguous in leaf order)");

  if (decay) {
    detail::check_f32(*dials, "dials");
    TORCH_CHECK(dials->dim() == 2 && dials->size(0) == heads && dials->size(1) == total_rows,
                "dials must be [H, sum_l width_l]");
    p.dials = dials->data_ptr<float>();
  }

  //: `empty`, not `zeros`: every element is written, so a zero-fill is a launch the
  //: launch-latency-bound small-batch regime pays for nothing:
  //: docs/internals/decode/decode.md#empty-not-zeros
  auto y = at::empty({BH, d_v}, v.options());
  auto* bits_base = reinterpret_cast<uint32_t*>(bits.data_ptr<int32_t>());
  p.g_write = g_write.data_ptr<float>();
  p.v = v.data_ptr<float>();
  p.state = state.data_ptr<float>();
  p.page_tbl = page_ptr;
  p.atoms_per_bh = atoms_per_bh;
  p.y = y.data_ptr<float>();
  p.bits_r = bits_base;
  p.bits_w = bits_base + static_cast<int64_t>(BH) * n_words;
  p.scratch = bits_base + 2 * static_cast<int64_t>(BH) * n_words;
  p.pc_coarse = pc_coarse.data_ptr<int32_t>();
  p.ws = ws.data_ptr<float>();
  p.ctr = ctr.data_ptr<int32_t>();
  if (gate.has_value()) {
    detail::check_i32(*gate, "gate");
    TORCH_CHECK(gate->dim() == 1 && gate->size(0) == 1, "gate must be [1] int32");
    p.gate = gate->data_ptr<int32_t>();
  }
  p.N = N;
  p.BH = BH;
  p.H = static_cast<int>(heads);
  p.D = D;
  p.n_words = n_words;
  p.n_groups = n_groups;
  p.n_split = static_cast<int>(n_split);
  p.total_rows = total_rows;
  p.eps = static_cast<float>(eps);

  int smem_per_block = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&smem_per_block, cudaDevAttrMaxSharedMemoryPerBlock,
                                        v.device().index()));
  launch_decode_expand(p, smem_per_block);
  const size_t gemv_smem = decode_gemv_smem_bytes(total_rows, d_v, cols, decay);
  TORCH_CHECK(static_cast<int64_t>(gemv_smem) <= smem_per_block,
              "the decode GEMV's staged amplitude rows need ", gemv_smem,
              " B of shared memory against a per-block budget of ", smem_per_block,
              " B; sum_l width_l = ", total_rows,
              ". This is a declared capacity, refused loudly rather than taken as a "
              "slower path.");
  launch_decode_gemv(p, d_v, decay);
  return y;
}

void rola_decode_growth_probe(std::vector<at::Tensor> amp_write, at::Tensor page_table,
                              at::Tensor bits, at::Tensor growth, at::Tensor growth_any,
                              at::Tensor probe_ctr, at::Tensor atom_bits,
                              std::vector<int64_t> level_widths, std::vector<int64_t> level_strides,
                              std::vector<int64_t> level_spans) {
  const int D = static_cast<int>(amp_write.size());
  TORCH_CHECK(D >= 1 && D <= kMaxLevels, "D must be in [1, 4], got ", D);
  TORCH_CHECK(static_cast<int>(level_widths.size()) == D
                  && static_cast<int>(level_strides.size()) == D
                  && static_cast<int>(level_spans.size()) == D,
              "the three per-level integer vectors must each have D entries");
  detail::check_i32(page_table, "page_table");
  detail::check_i32(bits, "bits");
  detail::check_i32(growth, "growth");
  TORCH_CHECK(page_table.dim() == 2, "page_table must be [BH, N / ", kAtomLeaves, "] int32");
  const int BH = static_cast<int>(page_table.size(0));
  const int atoms_per_bh = static_cast<int>(page_table.size(1));
  const int64_t N = static_cast<int64_t>(atoms_per_bh) * kAtomLeaves;
  detail::check_i32(growth_any, "growth_any");
  detail::check_i32(probe_ctr, "probe_ctr");
  TORCH_CHECK(growth.dim() == 1 && growth.size(0) == BH, "growth must be [BH] int32");
  TORCH_CHECK(growth_any.dim() == 1 && growth_any.size(0) == 1, "growth_any must be [1] int32");
  TORCH_CHECK(probe_ctr.dim() == 1 && probe_ctr.size(0) == 1, "probe_ctr must be [1] int32");
  TORCH_CHECK(
      atom_bits.is_cuda() && atom_bits.scalar_type() == at::kBool && atom_bits.is_contiguous(),
      "atom_bits must be a contiguous CUDA bool tensor, got ", atom_bits.scalar_type());
  TORCH_CHECK(atom_bits.dim() == 2 && atom_bits.size(0) == BH && atom_bits.size(1) == atoms_per_bh,
              "atom_bits must be the page table's shape [BH, N / ", kAtomLeaves, "] bool, got ",
              atom_bits.sizes());

  const int n_words = static_cast<int>((N + 31) / 32);
  TORCH_CHECK(bits.dim() == 1 && bits.size(0) == 4LL * BH * n_words,
              "bits must be the step's flat [4 * BH * ceil(N/32)] int32 workspace");

  DecodeParams p{};
  int64_t expect_n = 1;
  for (int l = 0; l < D; ++l) {
    const int width = static_cast<int>(level_widths[l]);
    TORCH_CHECK(width >= 1 && width <= kDecodeMaxLevelWidth, "level ", l, " width ", width,
                " exceeds the decode path's capacity ", kDecodeMaxLevelWidth);
    detail::check_f32(amp_write[l], "amp_write[l]");
    TORCH_CHECK(
        amp_write[l].dim() == 2 && amp_write[l].size(0) == BH && amp_write[l].size(1) == width,
        "amp_write[", l, "] must be [BH, width] contiguous");
    p.amp_write[l] = amp_write[l].data_ptr<float>();
    p.level_width[l] = width;
    p.level_stride[l] = static_cast<int>(level_strides[l]);
    p.level_span[l] = static_cast<int>(level_spans[l]);
    expect_n *= width;
  }
  TORCH_CHECK(expect_n == N, "prod(level_widths) = ", expect_n,
              " disagrees with the page table's leaf space ", N);
  TORCH_CHECK(p.level_span[0] == static_cast<int>(N),
              "level_spans[0] must be N (the whole leaf space)");

  auto* bits_base = reinterpret_cast<uint32_t*>(bits.data_ptr<int32_t>());
  p.page_tbl = page_table.data_ptr<int32_t>();
  p.growth = growth.data_ptr<int32_t>();
  p.growth_any = growth_any.data_ptr<int32_t>();
  p.probe_ctr = probe_ctr.data_ptr<int32_t>();
  p.atom_bits = atom_bits.data_ptr<bool>();
  p.bits_w = bits_base + static_cast<int64_t>(BH) * n_words;
  p.scratch = bits_base + 2 * static_cast<int64_t>(BH) * n_words;
  p.atoms_per_bh = atoms_per_bh;
  p.N = N;
  p.BH = BH;
  p.D = D;
  p.n_words = n_words;

  int smem_per_block = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&smem_per_block, cudaDevAttrMaxSharedMemoryPerBlock,
                                        page_table.device().index()));
  launch_decode_growth_probe(p, smem_per_block);
}

}  // namespace decode
}  // namespace rola
