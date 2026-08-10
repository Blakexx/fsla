// Clean-room dense chunked linear attention, bf16 in / fp32 accumulate.
//
// This file names NO ARCHITECTURE and issues NO INLINE PTX.  It is written
// against the baseline operations of `ops.cuh`; which implementation of each it
// receives is decided by the capability table in `arch_caps.cuh`, and an
// architecture that cannot supply one of the REQUIRED capabilities fails the
// build there rather than being served a slower route here.
//
// SEMANTICS (the whole of it):
//     O[t] = sum_{s <= t} (Q[t] . K[s]) * V[s]
// realised chunk-wise with a carried state S = sum_s K[s] (x) V[s]:
//     O_c  = Q_c . S                    readout against the ENTRY state
//     O_c += tril(Q_c . K_c^T) . V_c    within-chunk causal correction
//     S   += K_c^T . V_c                fold, at the chunk boundary
// No routing, no decay, no normalisation, no masks beyond causality.
//
// THE ONE STRUCTURAL DECISION EVERYTHING FOLLOWS FROM
// ---------------------------------------------------
// The state is held TRANSPOSED, S^T[d_v][d_k], as the fold's accumulator with
// m = d_v and n = d_k.  For the MMA's m16n8k16 shape the fp32 accumulator gives
// lane l the elements (m = l/4, l/4+8) x (n = 2*(l%4), +1); the readout needs S
// as a B operand, i.e. (n = d_v = l/4) x (k = d_k = 2*(l%4), +1, +8, +9).  With
// the transposed state those are the SAME registers.  The state therefore never
// leaves registers -- no shared-memory round trip, no shuffle, no transpose
// pass, one `pack_bf16x2` per b32.  `mma_b_from_regs` is therefore a clause of
// THIS kernel, asserted at the kernel below rather than in the shared table: an
// MMA that can only source B from a shared-memory descriptor computes this by a
// different algorithm.
//
// PARTITION (one MMA unit per 16 d_v rows)
// ----------------------------------------
// Unit u owns d_v in [16u, 16u+16).  That slice of the state is PRIVATE to the
// unit for the whole sequence.  There is no cross-unit reduction anywhere and
// no state in shared memory.  The one thing every unit needs in full is the
// Gram block G = tril(Q.K^T), whose m axis is tokens; it is computed once
// (token-split across units) and exchanged through shared memory in bf16.
//
// SHARED-MEMORY LAYOUT: TWO ARMS, BOTH BUILT
// ------------------------------------------
// `SWZ = false`  row stride 72 elements (pad 8).  Conflict-free because row r
//                starts 16 B further along the bank cycle than row r-1.  Costs
//                12.5% SMEM, costs ZERO address arithmetic (every in-body
//                ldmatrix address is a lane-invariant base plus a compile-time
//                immediate).
// `SWZ = true`   row stride 64 elements, XOR swizzle: the 16-B chunk index
//                within a row is XORed with (row & 7).  This is CUTLASS's
//                Swizzle<3,3,3>, the atom FlashAttention-2 uses for every
//                64-wide bf16 tile (kernel_traits.h:79-83).  Costs ZERO SMEM,
//                costs one IADD3 per ldmatrix -- the per-lane XOR table `tabF`
//                / `tabT` is built once at entry, so the body never issues a
//                LOP3, only an add of a loop-invariant register.
// The Gram buffer is ALWAYS padded: its rows are C elements wide, and for
// C < 64 a row is under 128 B, where an 8-phase XOR does not separate the
// eight rows of an ldmatrix tile.  Padding is the correct tool there.
//
// PIPELINE ORDER (dependency depth, not source convenience)
// ---------------------------------------------------------
//   stage_wait<STAGES-2>()          this chunk's tiles have landed
//   rendezvous()                    (1) tiles visible; last chunk's reads done;
//                                       last chunk's output staged
//   stage chunk c+STAGES-1          issued AFTER the barrier, which is what
//                                       makes the barrier the WAR fence too
//   readout   O = Q.S               depends on nothing but Q and registers
//   Gram      G = tril(Q.K^T) -> sg only GWARPS units have work here
//   drain     previous chunk's O    the other units' work during the Gram
//   fold      S += V^T.K            independent of G
//   rendezvous()                    (2) sg visible
//   intra     O += G.V
//   stage O into sg-adjacent buffer
// TWO barriers per chunk, which is FlashAttention-2's steady-state count
// (flash_fwd_kernel.h:388, 401).  The previous revision had three: the output
// staging buffer was written and re-read inside the same iteration.  Deferring
// the global store by one chunk folds that barrier into barrier (1).
//
// OPERAND STAGING
// ---------------
// Every mma's A/B fragments for a whole k-run are fetched BEFORE the first mma
// of that run, not one-ahead-of-one.  This is the register-prefetch discipline
// of FA2's `utils.h:150-159` taken one step further: FA2 issues load_frag(i+1)
// before mma(i); here the entire fragment array is live, so load_frag(i+1..n)
// all precede mma(i).  Verified in SASS, not assumed.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <vector>

#include "arch_caps.cuh"
#include "ops.cuh"

namespace denseref {

using namespace ops;

//: THIS KERNEL'S OWN OPERAND-RESIDENCY CLAUSE.  The baseline requires only that
//: an accumulator can be an operand from registers at all (`mma_a_from_regs`);
//: which SIDE it lands on is a property of the kernel, and this one lands in B.
static_assert(arch::kCaps.mma_b_from_regs,
              "UNSUPPORTED ARCHITECTURE FOR THIS KERNEL: the fold accumulator IS "
              "the readout's B operand here, so B must be readable from the "
              "issuing unit's own registers. A warpgroup MMA sources B only from "
              "a shared-memory descriptor, which is a different algorithm.");

// Bumped by hand whenever the source changes; read back device-side so a stale
// binary cannot masquerade as this source.  See build_stamp().
static constexpr int kBuildMagic = 580102;

static constexpr int kD = 64;   // d_k = d_v
static constexpr int kPad = 8;  // shared-memory row padding, in elements

// THE CARVE, DERIVED RATHER THAN CHOSEN.  One MMA unit owns a private 16-row
// slice of the transposed state, so the number of units is d_v / the MMA's m
// quantum, and the CTA is that many units wide.  Both terms come from the
// operation, not from a literal: on an architecture whose MMA unit is a
// warpgroup this would be one unit of 128 threads owning all 64 rows, which is
// the shape `arch_caps.cuh` records as unwritten rather than served here.
static constexpr int kUnits = kD / kMmaM;
static constexpr int kThreads = kUnits * kMmaUnitThreads;
static constexpr int kWarps = kThreads / 32;

// ------------------------------------------------------------------- kernel

// Compile-time SMEM geometry, shared by the kernel and the host launcher so the
// two can never disagree about a byte count.
template <int C, int STAGES, bool SWZ> struct Geom {
  static constexpr int Ld = SWZ ? kD : kD + kPad;  // Q/K/V/O row stride, elements
  static constexpr int CLd = C + kPad;             // Gram row stride, elements
  static constexpr size_t bytes =
      (size_t)(3 * STAGES * C * Ld + C * CLd + C * Ld) * sizeof(__nv_bfloat16);
};

//: THE REGISTER BUDGET, DERIVED PER ARCHITECTURE RATHER THAN WRITTEN DOWN.
//: The second launch bound is the CTA count this arm's shared-memory footprint
//: already admits on THIS architecture -- precisely the point at which
//: registers would become the new binding constraint, so it asks ptxas for the
//: largest register allocation that costs no residency.  Both terms come from
//: the table; a global register literal would be tuned for whichever machine
//: was last measured.
template <int C, int STAGES, bool SWZ>
constexpr int kCtaTarget =
    arch::ctas_by_smem(arch::kCaps, (int)Geom<C, STAGES, SWZ>::bytes);

template <int C, int STAGES, bool SWZ>
__global__ __launch_bounds__(kThreads, (kCtaTarget<C, STAGES, SWZ>)) void dense_la_kernel(
    const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v, __nv_bfloat16 *__restrict__ o, int L) {
  static_assert(C % kMmaM == 0, "chunk must be a multiple of the mma m quantum");
  static_assert(STAGES >= 2, "the wait/issue order needs at least two stages");
  // The one hard resource refusal, and it comes from the table rather than a
  // literal: C = 128 is designed and buildable everywhere the shared-memory
  // budget admits it, and nowhere else.  No arm is silently shrunk to fit.
  static_assert(Geom<C, STAGES, SWZ>::bytes <= (size_t)arch::kCaps.smem_per_cta_max,
                "this (C, STAGES, SWZ) arm does not fit this architecture's "
                "per-CTA shared-memory maximum");
  using G = Geom<C, STAGES, SWZ>;
  constexpr int Ld = G::Ld;
  constexpr int CLd = G::CLd;
  constexpr int MTOK = C / 16;  // token m-tiles in a chunk
  constexpr int NTOK = C / 8;   // token n-tiles in a chunk
  constexpr int KD = kD / 16;   // 4 -- d_k k-tiles
  constexpr int NDV = kD / 8;   // 8 -- state n-tiles (d_k)
  constexpr uint32_t kRowB = (uint32_t)Ld * 2;

  // Gram tile partition: split m first, spill the remainder onto n.
  constexpr int MSPLIT = MTOK >= kWarps ? kWarps : MTOK;
  constexpr int NSPLIT_RAW = kWarps / MSPLIT;
  constexpr int NSPLIT = NSPLIT_RAW < NTOK / 2 ? NSPLIT_RAW : NTOK / 2;
  constexpr int MPW = MTOK / MSPLIT;
  constexpr int NPW = NTOK / NSPLIT;
  constexpr int GWARPS = MSPLIT * NSPLIT;
  static_assert(NPW % 2 == 0, "n-tiles are consumed in ldmatrix pairs");
  // How many independent Gram accumulator chains a warp runs at once.  Each
  // n-tile pair is one ldmatrix and two chains of depth KD; pairing two of them
  // doubles the ILP at the cost of one extra B fragment set.
  constexpr int GPAIR = (NPW / 2) >= 2 ? 2 : 1;

  extern __shared__ __align__(16) char smem_raw[];
  __nv_bfloat16 *sq = reinterpret_cast<__nv_bfloat16 *>(smem_raw);
  __nv_bfloat16 *sk = sq + STAGES * C * Ld;
  __nv_bfloat16 *sv = sk + STAGES * C * Ld;
  __nv_bfloat16 *sg = sv + STAGES * C * Ld;
  __nv_bfloat16 *so = sg + C * CLd;

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const long base = (long)blockIdx.x * L * kD;
  const int NC = L / C;

  // ---- lane-invariant operand addressing, built ONCE ----------------------
  // ldmatrix.x4 addresses tile t (= lane>>3) with row lane&7.  Two lane-offset
  // forms suffice; `F` is the (tile&1)-major form (A from [m][k], B pair from
  // [k][n] via .trans), `T` is the (tile>>1)-major form (B pair from [n][k],
  // A^T from [k][m] via .trans).
  const int lr = lane & 7, tl = lane >> 3;
  const int rF = (tl & 1) * 8 + lr, cF = tl >> 1;  // cF/cT are 16-B chunk indices
  const int rT = (tl >> 1) * 8 + lr, cT = tl & 1;
  // Row component of the lane's address.  Under the swizzle the column is a
  // pure XOR of the chunk index, so the row term carries no column at all.
  const uint32_t rowF = SWZ ? (uint32_t)(rF * kRowB) : (uint32_t)((rF * Ld + cF * 8) * 2);
  const uint32_t rowT = SWZ ? (uint32_t)(rT * kRowB) : (uint32_t)((rT * Ld + cT * 8) * 2);
  // The XOR tables.  tab?[g] is the byte offset of logical 16-B chunk g of this
  // lane's row.  Eight registers per form, filled once; the body only ever adds
  // one of them, never computes an XOR.
  uint32_t tabF[8], tabT[8];
#pragma unroll
  for (int g = 0; g < 8; ++g) {
    tabF[g] = (uint32_t)((((g + cF) & 7) ^ lr) << 4);
    tabT[g] = (uint32_t)((((g + cT) & 7) ^ lr) << 4);
  }
  // ADDR(base, R, g): tile whose top-left is logical row R (a multiple of 8,
  // so the swizzle phase is the lane's own) and 16-B chunk g.
#define ADDRF(b_, R_, g_) \
  (SWZ ? ((b_) + (uint32_t)((R_) * (int)kRowB) + tabF[(g_)]) \
       : ((b_) + (uint32_t)(((R_) * Ld + (g_) * 8) * 2)))
#define ADDRT(b_, R_, g_) \
  (SWZ ? ((b_) + (uint32_t)((R_) * (int)kRowB) + tabT[(g_)]) \
       : ((b_) + (uint32_t)(((R_) * Ld + (g_) * 8) * 2)))

  const int offAg = (rF * CLd + cF * 8) * 2;  // Gram buffer: always padded

  const int gm0 = (warp / NSPLIT) * MPW;  // this warp's Gram m-tile base
  const int gn0 = (warp % NSPLIT) * NPW;  // ... and n-tile base
  const int dv0 = warp * 16;              // this warp's private d_v slice
  constexpr int DVG_STEP = 2;             // 16 d_v elements = 2 chunks of 8
  const int dvg = warp * DVG_STEP;        // ... as a 16-B chunk index
  // `dvg` is the ONE chunk index that is not a compile-time constant (it is the
  // warp's own d_v slice).  Indexing the XOR tables with it would force them
  // into local memory, so its two entries are computed directly instead; the
  // tables themselves are then only ever indexed by literals.
  const uint32_t dvTabF = (uint32_t)((((dvg + cF) & 7) ^ lr) << 4);
  const uint32_t dvTabT = (uint32_t)((((dvg + cT) & 7) ^ lr) << 4);
#define ADDRF_DV(b_, R_) \
  (SWZ ? ((b_) + (uint32_t)((R_) * (int)kRowB) + dvTabF) \
       : ((b_) + (uint32_t)(((R_) * Ld + dv0) * 2)))
#define ADDRT_DV(b_, R_) \
  (SWZ ? ((b_) + (uint32_t)((R_) * (int)kRowB) + dvTabT) \
       : ((b_) + (uint32_t)(((R_) * Ld + dv0) * 2)))

  // The state, transposed: m = d_v (this warp's 16 rows), n = d_k (8 tiles).
  float st[NDV][4];
#pragma unroll
  for (int j = 0; j < NDV; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) st[j][e] = 0.f;

  // ---- global -> shared staging: one row is 128 B = 8 lanes x 16 B ---------
  // ADDRESSING DISCIPLINE.  Every 64-bit address is formed ONCE, here, and then
  // only advanced by a constant stride; the ring slots are induction variables
  // in 32 bits with a compare-and-reset wrap, so there is no `% STAGES` and no
  // IMAD.WIDE anywhere inside the chunk loop.
  constexpr int kRowsPerPass = kThreads / 8;  // 16
  constexpr int kPasses = C / kRowsPerPass;
  static_assert(kPasses >= 1, "a chunk must be at least kThreads/8 rows");
  const int ldRow = tid >> 3, ldCol = (tid & 7) * 8;

  constexpr long kChunkElems = (long)C * kD;
  const __nv_bfloat16 *pq = q + base + (long)ldRow * kD + ldCol;
  const __nv_bfloat16 *pk = k + base + (long)ldRow * kD + ldCol;
  const __nv_bfloat16 *pv = v + base + (long)ldRow * kD + ldCol;
  __nv_bfloat16 *po = o + base + (long)ldRow * kD + ldCol;

  // Row-major 128-bit lane offset, swizzled or padded.  Used by BOTH the
  // cp.async destination and the output buffer's LDS.128 read, which is why
  // those two agree on the layout by construction.
  const uint32_t stgOff =
      SWZ ? (uint32_t)(ldRow * (int)kRowB + ((((tid & 7) ^ (ldRow & 7))) << 4))
          : (uint32_t)((ldRow * Ld + ldCol) * 2);

  constexpr uint32_t kSlot = (uint32_t)(C * Ld) * 2;  // bytes of one ring slot
  constexpr uint32_t kRing = STAGES * kSlot;
  const uint32_t sqL = smem_addr(sq) + stgOff;
  const uint32_t skL = smem_addr(sk) + stgOff;
  const uint32_t svL = smem_addr(sv) + stgOff;
  const uint32_t sqB = smem_addr(sq), skB = smem_addr(sk), svB = smem_addr(sv);
  const uint32_t sgA = smem_addr(sg) + (uint32_t)offAg;
  const int soDrainE = (int)(stgOff >> 1);  // output drain: same layout as staging
  uint32_t stgSlot = 0, cmpSlot = 0;

  // Output staging write index: the mma D fragment gives lane l the rows
  // l/4 and l/4+8 and the column pair 2*(l%4).  Under the swizzle the d_v
  // column block is the 16-B chunk index `dvg + p`, XORed with the row phase.
  const int oPh = (lane >> 2) & 7;
  const int oBaseE = (lane >> 2) * Ld + (SWZ ? ((lane & 3) * 2) : (dv0 + (lane & 3) * 2));
  int oXe[2];
#pragma unroll
  for (int p = 0; p < 2; ++p) oXe[p] = SWZ ? (((dvg + p) ^ oPh) << 3) : (p * 8);

  auto stage_chunk = [&]() {
#pragma unroll
    for (int p = 0; p < kPasses; ++p) {
      const uint32_t so_ = stgSlot + (uint32_t)(p * kRowsPerPass * (int)kRowB);
      const long go = (long)(p * kRowsPerPass * kD);
      stage_tile(sqL + so_, pq + go);
      stage_tile(skL + so_, pk + go);
      stage_tile(svL + so_, pv + go);
    }
    pq += kChunkElems; pk += kChunkElems; pv += kChunkElems;
    stgSlot += kSlot;
    if (stgSlot == kRing) stgSlot = 0;
  };

#pragma unroll 1
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < NC) stage_chunk();
    stage_commit();
  }

#pragma unroll 1
  for (int c = 0; c < NC; ++c) {
    stage_wait<STAGES - 2>();
    rendezvous();  // (1) tiles visible / last chunk's reads done / O staged

    // Issued AFTER the barrier: the barrier is this write's WAR fence against
    // the previous chunk's readers of the same ring slot.
    if (c + STAGES - 1 < NC) stage_chunk();
    stage_commit();

    const uint32_t aq = sqB + cmpSlot + rowF;
    const uint32_t bk = skB + cmpSlot + rowT;
    const uint32_t btk = skB + cmpSlot + rowF;
    const uint32_t atv = svB + cmpSlot + rowT;
    const uint32_t btv = svB + cmpSlot + rowF;
    cmpSlot += kSlot;
    if (cmpSlot == kRing) cmpSlot = 0;

    // ---- 1. readout: O = Q . S, B operand straight out of the state regs ----
    // First because it depends on nothing that this chunk has to compute.
    float acc[MTOK][2][4];
    {
      uint32_t sb[2][KD][2];
#pragma unroll
      for (int p = 0; p < 2; ++p)
#pragma unroll
        for (int kk = 0; kk < KD; ++kk) {
          sb[p][kk][0] = p ? pack_bf16x2(st[2 * kk][2], st[2 * kk][3])
                           : pack_bf16x2(st[2 * kk][0], st[2 * kk][1]);
          sb[p][kk][1] = p ? pack_bf16x2(st[2 * kk + 1][2], st[2 * kk + 1][3])
                           : pack_bf16x2(st[2 * kk + 1][0], st[2 * kk + 1][1]);
        }
#pragma unroll
      for (int m = 0; m < MTOK; ++m) {
#pragma unroll
        for (int p = 0; p < 2; ++p)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[m][p][e] = 0.f;
        // WHOLE k-run fetched before the first mma of the run.
        uint32_t af[KD][4];
#pragma unroll
        for (int kk = 0; kk < KD; ++kk)
          load_frag(af[kk], ADDRF(aq, m * 16, 2 * kk));
#pragma unroll
        for (int kk = 0; kk < KD; ++kk)
#pragma unroll
          for (int p = 0; p < 2; ++p) mma(acc[m][p], af[kk], sb[p][kk]);
      }
    }

    // ---- 2. Gram: G = tril(Q.K^T), this warp's tiles, straight into shared --
    if (warp < GWARPS) {
      uint32_t qa[MPW][KD][4];
#pragma unroll
      for (int m = 0; m < MPW; ++m)
#pragma unroll
        for (int kk = 0; kk < KD; ++kk)
          load_frag(qa[m][kk], ADDRF(aq, (gm0 + m) * 16, 2 * kk));

#pragma unroll
      for (int np = 0; np < NPW / 2; np += GPAIR) {
        uint32_t bf[GPAIR][KD][4];
#pragma unroll
        for (int u = 0; u < GPAIR; ++u)
#pragma unroll
          for (int kk = 0; kk < KD; ++kk)
            load_frag(bf[u][kk], ADDRT(bk, (gn0 + (np + u) * 2) * 8, 2 * kk));
#pragma unroll
        for (int m = 0; m < MPW; ++m) {
          // 2*GPAIR independent accumulator chains, each of depth KD.
          float g[GPAIR][2][4];
#pragma unroll
          for (int u = 0; u < GPAIR; ++u)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
              for (int e = 0; e < 4; ++e) g[u][h][e] = 0.f;
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
            for (int h = 0; h < 2; ++h) {
              const int tc = (gn0 + (np + u) * 2 + h) * 8 + (lane & 3) * 2;
              const float m0 = (tc + 0 <= tr) ? g[u][h][0] : 0.f;
              const float m1 = (tc + 1 <= tr) ? g[u][h][1] : 0.f;
              const float m2 = (tc + 0 <= tr + 8) ? g[u][h][2] : 0.f;
              const float m3 = (tc + 1 <= tr + 8) ? g[u][h][3] : 0.f;
              store_bf16x2(&sg[tr * CLd + tc], pack_bf16x2(m0, m1));
              store_bf16x2(&sg[(tr + 8) * CLd + tc], pack_bf16x2(m2, m3));
            }
        }
      }
    }

    // ---- 3. drain the PREVIOUS chunk's output ------------------------------
    // Placed here on purpose: at C = 16 only one warp has Gram work, and this
    // is the other three warps' independent instruction stream.  The store also
    // gets a full chunk of latency to retire before anything waits on it.
    if (c) {
#pragma unroll
      for (int p = 0; p < kPasses; ++p)
        copy_shared_to_global_16b(po + (long)(p * kRowsPerPass * kD),
                                  &so[soDrainE + p * kRowsPerPass * Ld]);
      po += kChunkElems;
    }

    // ---- 4. fold: S^T += V^T . K -------------------------------------------
    // Independent of the Gram, so it belongs before the second barrier.
#pragma unroll
    for (int kk = 0; kk < MTOK; ++kk) {
      uint32_t a[4];
      load_frag_t(a, ADDRT_DV(atv, kk * 16));
      uint32_t b[KD][4];
#pragma unroll
      for (int nj = 0; nj < KD; ++nj) load_frag_t(b[nj], ADDRF(btk, kk * 16, 2 * nj));
#pragma unroll
      for (int nj = 0; nj < KD; ++nj) {
        mma(st[2 * nj], a, &b[nj][0]);
        mma(st[2 * nj + 1], a, &b[nj][2]);
      }
    }

    rendezvous();  // (2) the Gram is visible

    // ---- 5. intra: O += G . V ----------------------------------------------
    {
      uint32_t vb[MTOK][4];
#pragma unroll
      for (int kk = 0; kk < MTOK; ++kk) load_frag_t(vb[kk], ADDRF_DV(btv, kk * 16));
#pragma unroll
      for (int m = 0; m < MTOK; ++m) {
        uint32_t ga[MTOK][4];
#pragma unroll
        for (int kk = 0; kk < MTOK; ++kk)
          load_frag(ga[kk], sgA + (uint32_t)((m * 16 * CLd + kk * 16) * 2));
#pragma unroll
        for (int kk = 0; kk < MTOK; ++kk)
#pragma unroll
          for (int p = 0; p < 2; ++p) mma(acc[m][p], ga[kk], &vb[kk][2 * p]);
      }
    }

    // ---- 6. stage this chunk's output; the store happens next iteration -----
#pragma unroll
    for (int m = 0; m < MTOK; ++m)
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        const int e0 = oBaseE + oXe[p] + m * 16 * Ld;
        store_bf16x2(&so[e0], pack_bf16x2(acc[m][p][0], acc[m][p][1]));
        store_bf16x2(&so[e0 + 8 * Ld], pack_bf16x2(acc[m][p][2], acc[m][p][3]));
      }
  }

  rendezvous();
#pragma unroll
  for (int p = 0; p < kPasses; ++p)
    copy_shared_to_global_16b(po + (long)(p * kRowsPerPass * kD),
                              &so[soDrainE + p * kRowsPerPass * Ld]);
#undef ADDRF
#undef ADDRT
}

// ------------------------------------------------- fragment convention probe
// One m16n8k16 pair per mode, so the ldmatrix/mma conventions asserted in the
// header are TESTED rather than assumed.  16x16 output = two n-tiles.
//   mode 0/2: A from a row-major [m][k] source, B from a row-major [n][k]
//             source -> D = A . Bsrc^T
//   mode 1/3: A from a row-major [k][m] source (.trans), B from a row-major
//             [k][n] source (.trans) -> D = Asrc^T . Bsrc
// Modes 2/3 repeat 0/1 through the XOR-swizzled layout, so the swizzle's
// address algebra is tested against torch exactly as the padded one is.
template <bool SWZ> __device__ __forceinline__ int probe_off(int row, int col) {
  constexpr int Ld = SWZ ? 16 : 16 + kPad;
  // A 16-wide probe tile is one 16-B chunk per row under the swizzle, so the
  // meaningful test is the 64-wide production geometry; the probe therefore
  // stores its 16x16 tile inside a 64-wide swizzled row block.
  if (SWZ) return row * kD + (((col >> 3) ^ (row & 7)) << 3) + (col & 7);
  return row * Ld + col;
}

template <bool SWZ>
__global__ void probe_kernel(const __nv_bfloat16 *a_src, const __nv_bfloat16 *b_src,
                             float *d_out, int mode) {
  constexpr int Ld = SWZ ? kD : 16 + kPad;
  __shared__ __nv_bfloat16 sa[16 * (SWZ ? kD : 16 + kPad)], sb[16 * (SWZ ? kD : 16 + kPad)];
  const int tid = threadIdx.x, lane = tid & 31;
  for (int i = tid; i < 16 * 16; i += 32) {
    sa[probe_off<SWZ>(i / 16, i % 16)] = a_src[i];
    sb[probe_off<SWZ>(i / 16, i % 16)] = b_src[i];
  }
  __syncwarp();
  const int lr = lane & 7, tl = lane >> 3;
  const int rF = (tl & 1) * 8 + lr, cF = tl >> 1;
  const int rT = (tl >> 1) * 8 + lr, cT = tl & 1;
  const int offF = SWZ ? (rF * kD * 2 + (((cF & 7) ^ lr) << 4)) : (rF * Ld + cF * 8) * 2;
  const int offT = SWZ ? (rT * kD * 2 + (((cT & 7) ^ lr) << 4)) : (rT * Ld + cT * 8) * 2;
  uint32_t af[4], bf[4];
  if (mode == 0) {
    load_frag(af, smem_addr(sa) + offF);
    load_frag(bf, smem_addr(sb) + offT);
  } else {
    load_frag_t(af, smem_addr(sa) + offT);
    load_frag_t(bf, smem_addr(sb) + offF);
  }
  float d0[4] = {0, 0, 0, 0}, d1[4] = {0, 0, 0, 0};
  mma(d0, af, &bf[0]);
  mma(d1, af, &bf[2]);
  const int r = lane >> 2, c = (lane & 3) * 2;
  d_out[r * 16 + c] = d0[0];
  d_out[r * 16 + c + 1] = d0[1];
  d_out[(r + 8) * 16 + c] = d0[2];
  d_out[(r + 8) * 16 + c + 1] = d0[3];
  d_out[r * 16 + c + 8] = d1[0];
  d_out[r * 16 + c + 9] = d1[1];
  d_out[(r + 8) * 16 + c + 8] = d1[2];
  d_out[(r + 8) * 16 + c + 9] = d1[3];
}

// ------------------------------------------------ capability conformance probe
// Every baseline operation with more than one possible implementation is
// exercised HERE, in whichever implementation this architecture's row selects,
// against a torch reference.  That is what keeps the capability table honest: a
// flag that no test ever exercises is a claim, not a capability.  `mul_bf16x2`
// has no caller in the dense baseline -- it is the entry that measures the
// sm_86 fp32 detour -- so this probe is its only consumer until a rung that
// produces elementwise coefficients arrives.
__global__ void caps_probe_kernel(const __nv_bfloat16 *a, const __nv_bfloat16 *b,
                                  __nv_bfloat16 *out, int pairs) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pairs) return;
  const uint32_t x = reinterpret_cast<const uint32_t *>(a)[i];
  const uint32_t y = reinterpret_cast<const uint32_t *>(b)[i];
  store_bf16x2(out + 2 * i, mul_bf16x2(x, y));
}

torch::Tensor caps_probe(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == torch::kBFloat16 && a.is_contiguous());
  TORCH_CHECK(b.sizes() == a.sizes() && a.numel() % 2 == 0);
  auto out = torch::empty_like(a);
  const int pairs = (int)(a.numel() / 2);
  caps_probe_kernel<<<(pairs + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16 *>(a.data_ptr()),
      reinterpret_cast<const __nv_bfloat16 *>(b.data_ptr()),
      reinterpret_cast<__nv_bfloat16 *>(out.data_ptr()), pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor probe(torch::Tensor a, torch::Tensor b, int64_t mode) {
  auto d = torch::zeros({16, 16}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  const auto *ap = reinterpret_cast<const __nv_bfloat16 *>(a.data_ptr());
  const auto *bp = reinterpret_cast<const __nv_bfloat16 *>(b.data_ptr());
  if (mode < 2)
    probe_kernel<false><<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        ap, bp, d.data_ptr<float>(), (int)mode);
  else
    probe_kernel<true><<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        ap, bp, d.data_ptr<float>(), (int)mode - 2);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return d;
}

// --------------------------------------------------------------- host side

// The ratified (C, STAGES, SWZ) instantiation set.  Everything measurable is
// built; nothing is chosen at run time by a heuristic.
#define DENSEREF_ARMS(X)                                                     \
  X(16, 3, false) X(16, 4, false) X(16, 5, false)                            \
  X(32, 2, false) X(32, 3, false) X(32, 4, false)                            \
  X(64, 2, false) X(64, 3, false)                                            \
  X(16, 3, true) X(16, 4, true) X(16, 6, true)                               \
  X(32, 3, true) X(32, 4, true) X(32, 5, true)                               \
  X(64, 3, true)

template <int C, int S, bool SWZ>
static void launch(const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &v,
                   torch::Tensor &o, int BH, int L) {
  constexpr size_t sm = Geom<C, S, SWZ>::bytes;
  static bool configured = false;
  if (!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(dense_la_kernel<C, S, SWZ>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int)sm));
    configured = true;
  }
  dense_la_kernel<C, S, SWZ><<<BH, kThreads, sm, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16 *>(q.data_ptr()),
      reinterpret_cast<const __nv_bfloat16 *>(k.data_ptr()),
      reinterpret_cast<const __nv_bfloat16 *>(v.data_ptr()),
      reinterpret_cast<__nv_bfloat16 *>(o.data_ptr()), L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, int64_t chunk,
                      int64_t stages, bool swizzle) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16 && q.is_contiguous());
  TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes());
  TORCH_CHECK(q.dim() == 4 && q.size(3) == kD, "expect [B,H,L,64]");
  const int B = q.size(0), H = q.size(1), L = q.size(2);
  TORCH_CHECK(L % chunk == 0, "L must be a multiple of the chunk");
  auto o = torch::empty_like(q);
  const int BH = B * H;
#define DENSEREF_DISPATCH(C_, S_, W_)                            \
  if (chunk == C_ && stages == S_ && swizzle == W_) {            \
    launch<C_, S_, W_>(q, k, v, o, BH, L);                       \
    return o;                                                    \
  }
  DENSEREF_ARMS(DENSEREF_DISPATCH)
#undef DENSEREF_DISPATCH
  TORCH_CHECK(false, "unbuilt arm: chunk=", chunk, " stages=", stages, " swizzle=", swizzle);
}

// Device-side build identity.  Two independent facts, neither of which a host
// path or a file hash can supply: (a) a magic constant WRITTEN BY A KERNEL that
// only this source compiles, and (b) the register/shared-memory census that
// ptxas produced for the very kernel entry the measurement will run.
// (c) the digest of the capability row THE DEVICE CODE WAS COMPILED AGAINST.  A
// binary built against a different table cannot masquerade as this one, and no
// host-side path or file hash can establish that -- only a device-side fact can.
__global__ void stamp_kernel(int *out) {
  out[0] = kBuildMagic;
  out[1] = kWarps * 1000 + kD;
  out[2] = arch::caps_digest(arch::kCaps);
}

std::vector<int64_t> build_stamp(int64_t chunk, int64_t stages, bool swizzle) {
  auto t = torch::zeros({3}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  stamp_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(t.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto h = t.cpu();
  std::vector<int64_t> r{h.data_ptr<int>()[0], h.data_ptr<int>()[1],
                         h.data_ptr<int>()[2]};
  cudaFuncAttributes a{};
#define DENSEREF_CENSUS(C_, S_, W_)                                             \
  if (chunk == C_ && stages == S_ && swizzle == W_) {                           \
    C10_CUDA_CHECK(cudaFuncGetAttributes(&a, dense_la_kernel<C_, S_, W_>));     \
    r.push_back(a.numRegs);                                                     \
    r.push_back((int64_t)a.sharedSizeBytes);                                    \
    r.push_back(a.binaryVersion);                                               \
    r.push_back((int64_t)Geom<C_, S_, W_>::bytes);                              \
    r.push_back((int64_t)a.maxThreadsPerBlock);                                 \
    r.push_back((int64_t)a.localSizeBytes);                                     \
    return r;                                                                   \
  }
  DENSEREF_ARMS(DENSEREF_CENSUS)
#undef DENSEREF_CENSUS
  TORCH_CHECK(false, "unbuilt arm");
  return r;
}

// ------------------------------------------------------ the table, queryable
// THE QUANTITATIVE HALF IS USELESS IF ONLY THE COMPILING TARGET CAN SEE IT.
// The whole point of tabulating per-SM resources is to derive a plan for an
// architecture that is not in the machine: these two entry points evaluate the
// same constexpr derivations for ANY tabulated compute capability, so the A100
// residency of an arm is a fact this box can state without owning an A100.

std::vector<int64_t> arch_row(int64_t cc) {
  TORCH_CHECK(arch::tabulated((int)cc), "untabulated compute capability ", cc);
  const arch::Caps c = arch::caps_of((int)cc);
  return {c.cc,
          c.async_copy, c.tile_load, c.mma_bf16_f32, c.mma_a_from_regs,
          c.mma_b_from_regs, c.bf16_cvt_packed,
          c.bf16_simd_arith, c.async_bulk_copy, c.warpgroup_mma,
          arch::supported(c),
          c.mma_unit_threads, c.smem_per_sm, c.smem_per_cta_max, c.regs_per_sm,
          c.reg_alloc_unit, c.max_threads_per_sm, c.max_ctas_per_sm,
          c.driver_smem_reserve, c.tensor_f32_accum_pct,
          arch::caps_digest(c)};
}

//: For one arm on one architecture: the three residency ceilings, the binding
//: one, and the per-lane register budget that would sustain 1..4 CTAs.  The
//: register budget is DERIVED here and never written as a literal anywhere --
//: a global register number tunes for whichever machine was measured last.
std::vector<int64_t> residency_plan(int64_t smem_bytes, int64_t regs_per_lane,
                                    int64_t cc, int64_t threads) {
  TORCH_CHECK(arch::tabulated((int)cc), "untabulated compute capability ", cc);
  TORCH_CHECK(threads > 0 && threads % 32 == 0, "threads must be whole warps");
  const arch::Caps c = arch::caps_of((int)cc);
  const int t = (int)threads;
  std::vector<int64_t> r{arch::ctas_by_smem(c, (int)smem_bytes),
                         arch::ctas_by_regs(c, t, (int)regs_per_lane),
                         arch::ctas_by_threads(c, t),
                         c.max_ctas_per_sm,
                         arch::residency(c, (int)smem_bytes, t, (int)regs_per_lane)};
  for (int n = 1; n <= 4; ++n) r.push_back(arch::reg_budget(c, t, n));
  return r;
}

std::vector<std::vector<int64_t>> arms() {
  std::vector<std::vector<int64_t>> out;
#define DENSEREF_LIST(C_, S_, W_) \
  out.push_back({C_, S_, (int64_t)(W_), (int64_t)Geom<C_, S_, W_>::bytes});
  DENSEREF_ARMS(DENSEREF_LIST)
#undef DENSEREF_LIST
  return out;
}

}  // namespace denseref

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &denseref::forward, "dense chunked linear attention");
  m.def("probe", &denseref::probe, "fragment convention probe");
  m.def("caps_probe", &denseref::caps_probe, "capability conformance probe");
  m.def("build_stamp", &denseref::build_stamp, "build identity");
  m.def("arms", &denseref::arms, "the built (C, STAGES, SWZ, smem) set");
  m.def("arch_row", &denseref::arch_row, "the capability table row for a cc");
  m.def("residency_plan", &denseref::residency_plan, "derived residency for a cc",
        pybind11::arg("smem_bytes"), pybind11::arg("regs_per_lane"), pybind11::arg("cc"),
        pybind11::arg("threads") = denseref::kThreads);
}
