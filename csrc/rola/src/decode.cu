// THE T=1 DECODE GEMV -- token-stationary, state-streaming, bitmap-walked.
//
// §6b's ratified shape, literally: the token's routing weights and value are
// RESIDENT (SMEM/registers), the STATE streams past, and the grid partitions the
// support. What makes it a GEMV rather than a GEMM is that there is one token, so the
// contraction index is the LEAF and the output index is the COLUMN -- and that is why
// there is no MMA here and no warp reduction either: lanes are mapped to OUTPUT
// COLUMNS, so each lane accumulates its own partials in registers for the whole row
// stream and the dot never crosses a lane. The only reductions in this file are the
// 8-way cross-warp fold and the `n_split`-way cross-CTA combine, both over a small
// FIXED index set, and both are fixed-order sums rather than tree primitives because
// each is `cols` INDEPENDENT reductions over 8 (or `n_split`) values -- a shape no
// block-reduce primitive has.
//
// **DETERMINISM IS A SHIPPED PROPERTY HERE, AND IT DIVERGES FROM THE TILED CONSUMER
// ON PURPOSE.** The tiled consumer emits each token's output as a split-K partial
// through `atomicAdd` into `num`/`den`, so its `y` is not reproducible against its own
// binary wherever a `bh` holds more than one owner (measured: 384 of 976 fixture cases
// differ between two runs of one binary). Prefill's `y` feeds a loss or an eval
// aggregate and that is tolerable; decode's `y` feeds an argmax or a sample that picks
// the NEXT TOKEN, so a last-bit difference in `den` flips a near-tie logit and the
// divergence compounds autoregressively for the rest of the generation. At `T = 1` the
// partials are `n_split * cols` floats per `bh`, so the fixed-order combine costs one
// workspace slot per CTA, one `__threadfence()` and one `atomicAdd` on an INT counter
// -- an atomic that defines the schedule, not the arithmetic. The state is
// unconditionally reproducible for a stronger reason: the row partition is a
// PARTITION, so every written row has exactly one owning CTA, warp and lane, and no
// atomic touches `state` at all.
//
// **NO BARRIER PER ANYTHING.** Two `__syncthreads` per CTA per step (three when
// `n_split > 1`), against the tiled consumer's six per candidate tile. The tiled
// kernel spends one of its six ordering the tile's readout against the tile's fold; at
// `T = 1` the two are a single lane's two statements on one row in program order, and
// the ordering is UPDATE-THEN-READ because §2's survival is inclusive at `j = t` -- see
// the row body, where the collapse of the causal intra Gram into that order is
// derived.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include "decode.cuh"
#include "decode_api.cuh"

namespace rola {
namespace decode {
namespace {

//: `expf(88) = 1.65e38 < FLT_MAX`. THE SAME VALUE AND THE SAME CONVENTION as
//: `intra_panel.cuh`'s `kPanelMaxExponent`, restated rather than included because
//: that header is the tiled consumer's panel algebra and decode shares none of it.
//: On this path the clamp is STRUCTURALLY INERT and that is worth saying rather than
//: leaving a reader to wonder: the exponent is `clock * log1pf(-rate)` with
//: `clock >= 0` and `rate in (0, 1)`, so `log1pf(-rate) < 0` and the exponent is
//: non-positive -- `expf` can underflow to zero (the correct limit) but cannot
//: overflow. It is written anyway so that ONE convention governs every survival
//: exponential in the repository, and a future decay form that can go positive finds
//: the clamp already in place. `tests/unit/test_decode_capacity.py` pins the two
//: constants equal.
constexpr float kDecodeMaxExponent = 88.0f;

//: Locate the bitmap word holding global set-bit index `row`, and how many set bits of
//: that word precede it. Two steps and neither is a scan of the bitmap: a binary
//: search over the coarse index lands in a 32-word group, then at most 32 popcounts
//: walk to the word. This is what the coarse index is FOR, and it is the same array
//: the walk consults to skip empty groups.
__device__ __forceinline__ void locate_row(const int32_t* pc, const uint32_t* bits_r,
                                           const uint32_t* bits_w, int n_words,
                                           int n_groups, int row, int& out_word,
                                           int& out_skip) {
  int lo = 0;
  int hi = n_groups;
  while (lo < hi - 1) {
    const int mid = (lo + hi) >> 1;
    if (pc[mid] <= row) lo = mid; else hi = mid;
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

template <int DV, int D, bool DECAY, bool GLOBAL>
__global__ __launch_bounds__(kDecodeThreads) void decode_gemv_kernel(DecodeParams p) {
  //: `GLOBAL` changes the STATE'S SHAPE (the `+1` mass column), so it is a template
  //: axis for the same reason it is one in the tiled consumer: a runtime test would
  //: make `cols` a runtime value in every state address and would keep the mass
  //: accumulator live on an arm that has no mass.
  constexpr int cols = GLOBAL ? DV + 1 : DV;
  //: Columns per lane. The row load is `NC` fully coalesced 128 B transactions and the
  //: lane's accumulators are its own output columns -- the whole reason no reduction
  //: is needed on the dot.
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
  //   sa_r, sa_w        [total_rows] fp32   the per-level amplitude rows, UNCOMPACTED
  //                                         and indexed by DIGIT -- the bitmap carries
  //                                         membership, so the digit IS the index and
  //                                         no compaction, id list or flag packing
  //                                         exists.
  //   sa_c, sd          [total_rows] fp32   clock amplitudes and this head's dial row;
  //                                         DECAY only.
  //   v_tok             [DV]         fp32   the token's value, resident.
  //   fold [warps][cols+3]           fp32   the cross-warp partials. `+3` pads the row
  //                                         stride to `cols + 3` so the eight warps'
  //                                         stores land on distinct banks: the fold
  //                                         reads DOWN the warp axis, so an unpadded
  //                                         stride sharing a factor with 32 would
  //                                         collide. `gcd(cols+3, 32)` is 4 at
  //                                         `cols = 65` and 1 at `cols = 32`/`64+3`.
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

  //: The two per-token row scales. At `T = 1` there is no owner rectangle, so the
  //: per-LEVEL factors the tiled kernel hoists into `sr`/`sw` for the fixed levels
  //: `l < j` stay in the per-LEAF product instead -- the split between `R[0,s]` and
  //: `sr` is an owner-rectangle artefact and the product is the same product either
  //: way. What remains here is exactly the per-TOKEN part: the side gains.
  const float sr = GLOBAL ? 1.0f
                          : ((p.g_read != nullptr) ? p.g_read[bh] : 1.0f);
  const float sw = p.g_write[bh];

  const uint32_t* bits_r = p.bits_r + static_cast<int64_t>(bh) * nw;
  const uint32_t* bits_w = p.bits_w + static_cast<int64_t>(bh) * nw;
  const int32_t* pc = p.pc_coarse + static_cast<int64_t>(bh) * (ng + 1);

  // ---- resolve this CTA's ROW range into a WORD range -------------------------
  // The partition is defined in ROWS (popcount-balanced by construction) and handed
  // out in WORDS; the two boundary words are cut by dropping / keeping set bits, which
  // is exact rather than an index comparison.
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

  int w = w_start + warp;
  int g_cached = -1;
  bool g_empty = false;
  while (w <= w_last) {
    //: THE COARSE-INDEX CONSULT AT WALK TIME. A group whose prefix does not advance
    //: holds no touched row, so its 32 words need not be read at all. Consulted once
    //: per GROUP rather than once per word (the two `pc` loads would otherwise cost as
    //: much as the bitmap words they save), which is what keeps the bitmap read from
    //: dominating a high-batch, high-sparsity step. The boundary words `w_start` and
    //: `w_last` are non-empty by construction, so a skip can never cross them.
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
    //: THE UNION, formed in a REGISTER and never stored. One instruction, one third of
    //: the storage a materialized third bitmap would cost.
    uint32_t u = rw_word | ww_word;
    if (w == w_last) u = keep_low_set_bits(u, keep_last);
    if (w == w_start) u = drop_low_set_bits(u, drop_first);
    if (u == 0u) {
      w += kDecodeWarps;
      continue;
    }

    const int leaf_base = w * 32;
    //: The `D` integer divisions are paid ONCE PER WORD, not once per row. NVIDIA
    //: hardware has no integer divide (~20 instructions each), so a per-row
    //: decomposition would put ~80 instructions on a row whose payload is `cols * 4`
    //: bytes -- inside the sm_86 budget but not the sm_80 one, i.e. it would make the
    //: A100 dense arm instruction-bound. Amortized over a word it is ~2.5 per row on
    //: the dense arm.
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
    //: The word is walked in ALIGNED SUB-RANGES, each lying inside ONE inner block, so
    //: the innermost digit is `cur + (bit - q)` with no division and the outer digits
    //: are constant. The trip count is `ceil(32 / b_{D-1})`, which is 1 for every
    //: topology whose innermost width reaches a word (`widths = (256,256)`) and 2 at
    //: the flagship `(8,16)`. ONE uniform loop that degenerates -- not a fast path
    //: with a fallback.
    while (q < 32) {
      int take = b_last - cur;
      if (take > 32 - q) take = 32 - q;
      uint32_t sub = (take >= 32) ? u : ((u >> q) & ((1u << take) - 1u));
      while (sub != 0u) {
        const int t = __ffs(static_cast<int>(sub)) - 1;
        sub &= sub - 1u;
        const int bit = q + t;
        //: Bits at or beyond `N` are never set: the expansion's destination-word loop
        //: stops at `span_out`, which is `S_0 = N` at level 0, leaving the tail zero.
        //: So the walk needs no leaf bound test and never fabricates a row.
        const int leaf = leaf_base + bit;
        const int off_last = off_last_base + cur + t;
        const bool in_r = ((rw_word >> bit) & 1u) != 0u;
        const bool in_w = ((ww_word >> bit) & 1u) != 0u;
        float* row = p.state + (static_cast<int64_t>(bh) * p.N + leaf) * cols;

        float sv[NC];
#pragma unroll
        for (int k = 0; k < NC; ++k) sv[k] = row[lane + 32 * k];
        float smass = 0.0f;
        if (GLOBAL && lane == 0) smass = row[DV];

        //: UPDATE FIRST, THEN READ THE UPDATED ROW -- AND THAT ORDER IS THE SEMANTICS,
        //: not an optimization. §2's survival is INCLUSIVE at `j = t`: the token's own
        //: deposit is read by the token itself with survival exactly 1. The tiled
        //: consumer expresses that as TWO terms, because at `BT > 1` the deposit and the
        //: readout live in different GEMMs: the readout against the decayed entry state,
        //: plus the causal intra Gram's self term `G[0,0] * Vw`. At `T = 1` those two
        //: are
        //:      a_r * keep^c * M_entry  +  a_r * a_w * sw * v   ==   a_r * S_exit,
        //: so THE WHOLE INTRA GRAM COLLAPSES INTO READING THE ROW AFTER IT IS WRITTEN.
        //: One pass over the row, one register, and the causal structure is exact rather
        //: than reconstructed. A row in `supp(R) \ supp(W)` has `keep == 1` and no
        //: deposit, so `S_exit == M_entry` and the same expression is its readout too --
        //: one path, no arm. Both off-by-ones (decaying after the deposit, reading
        //: before it) are O(1) errors that G4/G6 discriminate at the fp32 tolerance.
        if (in_w) {
          //: DECAY AT READ, VIA THE DETACHED CLOCK, WITH NO SWEEP AND NO EXPONENT
          //: SPLIT. At `T = 1` the tiled kernel's §12 rule-1 half-mass shift is
          //: exactly the identity: with `P_ref = P[0]/2` the readout's `fr * s0`
          //: collapses to `keep^c` and the fold's `s0 * (s0 M + W fw Vw)` collapses to
          //: `keep^c M + W Vw`. So there is no clock prefix buffer, no `p_ref`, no
          //: `s0`, no division by a cumulative survival, and no `fw` underflow class
          //: -- ONE `expf` where the tiled kernel evaluates three. `log1pf` IS the
          //: composition (never a rounded linear-domain `1 - rate`) and there is no
          //: epsilon floor: both of §12 rule 2's prohibitions, inherited verbatim.
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
          if (GLOBAL && lane == 0) {
            smass = keep * smass + wv;
            row[DV] = smass;
          }
        }
        if (in_r) {
          const float a = o_r * sa_r[off_last];
#pragma unroll
          for (int k = 0; k < NC; ++k) acc[k] += a * sv[k];
          if (GLOBAL && lane == 0) acc_m += a * smass;
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
  const int fold_pitch = cols + 3;
#pragma unroll
  for (int k = 0; k < NC; ++k) fold[warp * fold_pitch + lane + 32 * k] = acc[k];
  if (GLOBAL && lane == 0) fold[warp * fold_pitch + DV] = acc_m;
  __syncthreads();
  if (tid < cols) {
    float t = 0.0f;
#pragma unroll
    for (int wq = 0; wq < kDecodeWarps; ++wq) t += fold[wq * fold_pitch + tid];
    //: `sr` is applied HERE, before the split-K store, exactly as the tiled kernel
    //: emits `sr * y` into `num` and `sr * den_acc` into `den`. That makes the combine
    //: below a plain sum and keeps the two consumers' dividend and divisor the same
    //: objects.
    part[tid] = sr * t;
  }
  __syncthreads();

  // ---- the deterministic split-K combine, and the fused divide ---------------
  if (p.n_split == 1) {
    if (tid < DV) {
      const float den = GLOBAL ? part[DV] : 1.0f;
      p.y[static_cast<int64_t>(bh) * DV + tid] =
          GLOBAL ? (part[tid] / (den + p.eps)) : part[tid];
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

  //: SLOT ORDER, always. This is the whole determinism argument: `n_split` fp32
  //: addends summed in a FIXED order give one answer per binary, where an `atomicAdd`
  //: race gives one answer per arrival order. `__ldcg` bypasses L1 so the reader sees
  //: the other CTAs' stores that the `__threadfence` above made visible.
  if (tid < cols) {
    const float* base = p.ws + static_cast<int64_t>(bh) * p.n_split * cols;
    float t = 0.0f;
    for (int s = 0; s < p.n_split; ++s) t += __ldcg(base + static_cast<int64_t>(s) * cols + tid);
    part[tid] = t;
  }
  __syncthreads();
  if (tid < DV) {
    const float den = GLOBAL ? part[DV] : 1.0f;
    p.y[static_cast<int64_t>(bh) * DV + tid] =
        GLOBAL ? (part[tid] / (den + p.eps)) : part[tid];
  }
  //: Reset for the next step, so no `memset` launch stands between two decode steps
  //: and the counter is safe to capture in a CUDA graph.
  if (tid == 0) p.ctr[bh] = 0;
}

#define ROLA_DECODE_LAUNCH(DV_, D_, DECAY_, GLOBAL_)                                  \
  decode_gemv_kernel<DV_, D_, DECAY_, GLOBAL_>                                        \
      <<<dim3(p.n_split, p.BH), kDecodeThreads, smem, stream>>>(p)

#define ROLA_DECODE_DISPATCH_D(DV_, DECAY_, GLOBAL_)                                  \
  switch (p.D) {                                                                      \
    case 1: ROLA_DECODE_LAUNCH(DV_, 1, DECAY_, GLOBAL_); break;                        \
    case 2: ROLA_DECODE_LAUNCH(DV_, 2, DECAY_, GLOBAL_); break;                        \
    case 3: ROLA_DECODE_LAUNCH(DV_, 3, DECAY_, GLOBAL_); break;                        \
    case 4: ROLA_DECODE_LAUNCH(DV_, 4, DECAY_, GLOBAL_); break;                        \
    default: TORCH_CHECK(false, "D must be in [1, 4] (spec §1), got ", p.D);           \
  }

#define ROLA_DECODE_DISPATCH_DECAY(DV_, GLOBAL_)                                      \
  if (decay) { ROLA_DECODE_DISPATCH_D(DV_, true, GLOBAL_) }                            \
  else { ROLA_DECODE_DISPATCH_D(DV_, false, GLOBAL_) }

#define ROLA_DECODE_DISPATCH_GLOBAL(DV_)                                              \
  if (global_norm) { ROLA_DECODE_DISPATCH_DECAY(DV_, true) }                           \
  else { ROLA_DECODE_DISPATCH_DECAY(DV_, false) }

}  // namespace

size_t decode_gemv_smem_bytes(int total_rows, int d_v, int cols, bool decay) {
  const size_t amps = static_cast<size_t>(total_rows) * (decay ? 4 : 2);
  const size_t fold = static_cast<size_t>(kDecodeWarps) * (cols + 3);
  return (amps + static_cast<size_t>(d_v) + fold) * sizeof(float);
}

void launch_decode_gemv(const DecodeParams& p, int d_v, bool global_norm, bool decay) {
  //: The instantiation matrix is `DV(2) x D(4) x DECAY(2) x GLOBAL(2) = 32`. Note what
  //: is NOT an axis: no `BT` (one token), no `BC` (no owner blocking), no `COMPACT`
  //: (the bitmap IS the compaction), no `MmaPrec` (there is no MMA). Every axis is a
  //: structural plan fact with few legal values, which is the standing "branch on
  //: structure, not thresholds" rule.
  TORCH_CHECK(d_v == 32 || d_v == 64, "decode instantiates d_v in {32, 64}, got ", d_v);
  const int cols = d_v + (global_norm ? 1 : 0);
  const size_t smem = decode_gemv_smem_bytes(p.total_rows, d_v, cols, decay);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (d_v == 64) { ROLA_DECODE_DISPATCH_GLOBAL(64) } else { ROLA_DECODE_DISPATCH_GLOBAL(32) }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#undef ROLA_DECODE_LAUNCH
#undef ROLA_DECODE_DISPATCH_D
#undef ROLA_DECODE_DISPATCH_DECAY
#undef ROLA_DECODE_DISPATCH_GLOBAL

// ---------------------------------------------------------------------------
// Host entry
// ---------------------------------------------------------------------------

namespace {

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

}  // namespace

int64_t rola_decode_capacity() { return kDecodeMaxLevelWidth; }

//: The MIRRORED copy of the producer's `MAX_BRANCH_WIDTH`, exposed separately from the
//: capacity because the two are different claims. `capacity >= producer` is the
//: safety property and the `static_assert` covers it; `mirror == the real Python
//: constant` is what stops the assertion from being satisfied by a stale mirror after
//: the producer's limit moves. Only a test that can see both languages can check the
//: second, which is why this is bound.
int64_t rola_decode_producer_width_mirror() { return kProducerMaxBranchWidth; }

at::Tensor rola_decode_forward(
    std::vector<at::Tensor> amp_read,
    std::vector<at::Tensor> amp_write,
    c10::optional<std::vector<at::Tensor>> amp_clock,
    c10::optional<at::Tensor> dials,
    c10::optional<at::Tensor> g_read,
    at::Tensor g_write,
    at::Tensor v,
    at::Tensor state,
    at::Tensor bits,
    at::Tensor pc_coarse,
    at::Tensor ws,
    at::Tensor ctr,
    std::vector<int64_t> level_widths,
    std::vector<int64_t> level_strides,
    std::vector<int64_t> level_spans,
    std::vector<int64_t> level_row_offsets,
    int64_t heads,
    int64_t n_split,
    double eps,
    bool global_norm) {
  const int D = static_cast<int>(amp_read.size());
  TORCH_CHECK(D >= 1 && D <= kMaxLevels, "D must be in [1, 4] (spec §1), got ", D);
  TORCH_CHECK(static_cast<int>(amp_write.size()) == D, "amp_write must have D entries");
  TORCH_CHECK(static_cast<int>(level_widths.size()) == D &&
                  static_cast<int>(level_strides.size()) == D &&
                  static_cast<int>(level_spans.size()) == D &&
                  static_cast<int>(level_row_offsets.size()) == D,
              "the four per-level integer vectors must each have D entries");

  check_f32(v, "v");
  check_f32(g_write, "g_write");
  check_f32(state, "state");
  check_i32(bits, "bits");
  check_i32(pc_coarse, "pc_coarse");
  check_f32(ws, "ws");
  check_i32(ctr, "ctr");

  TORCH_CHECK(v.dim() == 2, "v must be [BH, d_v]");
  const int BH = static_cast<int>(v.size(0));
  const int d_v = static_cast<int>(v.size(1));
  const int cols = d_v + (global_norm ? 1 : 0);
  TORCH_CHECK(state.dim() == 3 && state.size(0) == BH && state.size(2) == cols,
              "state must be [BH, N, d_v + is_global]");
  const int64_t N = state.size(1);
  TORCH_CHECK(g_write.dim() == 1 && g_write.size(0) == BH, "g_write must be [BH]");
  TORCH_CHECK(heads >= 1 && BH % heads == 0, "heads must divide BH");

  const int n_words = static_cast<int>((N + 31) / 32);
  const int n_groups = (n_words + kCoarseGroupWords - 1) / kCoarseGroupWords;
  //: ONE FLAT allocation, carved into `bits_r | bits_w | scratch` blocks of
  //: `[BH, n_words]`, `[BH, n_words]` and `[BH, 2 * n_words]`. Flat rather than shaped,
  //: because a `[BH, 4 * n_words]` shape would imply a per-`bh` interleaving the carve
  //: below does not use, and a shape that lies is worse than no shape.
  TORCH_CHECK(bits.dim() == 1 && bits.size(0) == 4LL * BH * n_words,
              "bits must be a flat [4 * BH * ceil(N/32)] int32 workspace");
  TORCH_CHECK(pc_coarse.dim() == 2 && pc_coarse.size(0) == BH &&
                  pc_coarse.size(1) == n_groups + 1,
              "pc_coarse must be [BH, ceil(n_words/32) + 1] int32");
  TORCH_CHECK(n_split >= 1, "n_split must be >= 1, got ", n_split);
  TORCH_CHECK(ws.dim() == 3 && ws.size(0) == BH && ws.size(1) == n_split &&
                  ws.size(2) == cols,
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
    TORCH_CHECK(width >= 1 && width <= kDecodeMaxLevelWidth,
                "level ", l, " width ", width, " exceeds the decode path's capacity ",
                kDecodeMaxLevelWidth, " (see kProducerMaxBranchWidth)");
    check_f32(amp_read[l], "amp_read[l]");
    check_f32(amp_write[l], "amp_write[l]");
    TORCH_CHECK(amp_read[l].dim() == 2 && amp_read[l].size(0) == BH &&
                    amp_read[l].size(1) == width,
                "amp_read[", l, "] must be [BH, width] contiguous");
    TORCH_CHECK(amp_write[l].sizes() == amp_read[l].sizes(),
                "amp_write[", l, "] must match amp_read[", l, "]");
    p.amp_read[l] = amp_read[l].data_ptr<float>();
    p.amp_write[l] = amp_write[l].data_ptr<float>();
    if (decay) {
      check_f32((*amp_clock)[l], "amp_clock[l]");
      TORCH_CHECK((*amp_clock)[l].sizes() == amp_read[l].sizes(),
                  "amp_clock[", l, "] must match amp_read[", l, "]");
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
    check_f32(*dials, "dials");
    TORCH_CHECK(dials->dim() == 2 && dials->size(0) == heads && dials->size(1) == total_rows,
                "dials must be [H, sum_l width_l]");
    p.dials = dials->data_ptr<float>();
  }
  if (g_read.has_value()) {
    check_f32(*g_read, "g_read");
    TORCH_CHECK(g_read->dim() == 1 && g_read->size(0) == BH, "g_read must be [BH]");
    TORCH_CHECK(!global_norm,
                "the global arm fixes g_read := 1 and must not be handed one");
    p.g_read = g_read->data_ptr<float>();
  }

  //: `empty`, not `zeros`: every element is written by the `n_split == 1` CTA or by the
  //: last-arriving CTA of the combine, so a zero-fill would be a launch the
  //: launch-latency-bound small-batch regime pays for nothing -- and an unwritten
  //: element showing as garbage rather than as a plausible zero is what a gate wants.
  auto y = at::empty({BH, d_v}, v.options());
  auto* bits_base = reinterpret_cast<uint32_t*>(bits.data_ptr<int32_t>());
  p.g_write = g_write.data_ptr<float>();
  p.v = v.data_ptr<float>();
  p.state = state.data_ptr<float>();
  p.y = y.data_ptr<float>();
  p.bits_r = bits_base;
  p.bits_w = bits_base + static_cast<int64_t>(BH) * n_words;
  p.scratch = bits_base + 2 * static_cast<int64_t>(BH) * n_words;
  p.pc_coarse = pc_coarse.data_ptr<int32_t>();
  p.ws = ws.data_ptr<float>();
  p.ctr = ctr.data_ptr<int32_t>();
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
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&smem_per_block,
                                        cudaDevAttrMaxSharedMemoryPerBlock,
                                        v.device().index()));
  launch_decode_expand(p, smem_per_block);
  const size_t gemv_smem = decode_gemv_smem_bytes(total_rows, d_v, cols, decay);
  TORCH_CHECK(static_cast<int64_t>(gemv_smem) <= smem_per_block,
              "the decode GEMV's staged amplitude rows need ", gemv_smem,
              " B of shared memory against a per-block budget of ", smem_per_block,
              " B; sum_l width_l = ", total_rows,
              ". This is a declared capacity, refused loudly rather than taken as a "
              "slower path.");
  launch_decode_gemv(p, d_v, global_norm, decay);
  return y;
}

}  // namespace decode
}  // namespace rola
