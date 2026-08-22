// csrc/rola/src/carry/carry_write_kernel.cuh -- K34 FORM B: THE WRITE-CARVE BODY.
//
// THE THIRD BODY OF THE CARRY FAMILY, beside `carry_split_kernel` (the dense
// declaration's channel split) and `carry_kernel` (the read declaration's leaf
// carve).  It is selected by the SAME per-level declaration those two are, and it
// is what a declaration that marks the OUTERMOST level write-sparse and the
// innermost level read-sparse asks for: the two sides want two different carves of
// the same box, and this body gives them both.
//
// WHAT IS DIFFERENT, in one sentence each.
//
//  * THE STATE IS CARVED BY THE WRITE AXIS.  A warp owns one level-0 digit's
//    `kLeafW = BC / kFbWarps` CONTIGUOUS leaves across ALL `kCols` channels, in a
//    permanent fp32 register slice.  Under the lattice's `local` order the
//    outermost level's digit owns a contiguous run, so the slice is whole
//    accumulator n-tiles with no padding.
//  * THE FOLD SKIPS WHOLLY-DEAD k-TILES, AND THAT IS BIT-IDENTICAL.  It walks the
//    OWNER's write list in its existing `C`-token batches and its existing k-tile
//    order -- so the k-sequence is byte-identical to the other two bodies' and to
//    the reverse family's replay -- and uses the warp's own level-0 DIGIT ballot
//    row as a per-k-tile SKIP PREDICATE.  A k-tile in which the digit is wholly
//    dead contributes an exact zero at every one of the warp's leaves, so dropping
//    it is an identity rather than an approximation (measured on eight arms,
//    prototypes/k31_formb2_build).  Compacting the list instead would move the
//    tile boundaries and REASSOCIATE the fp32 accumulate, which is the one thing
//    the reverse family's `torch.equal` replay gate cannot absorb.
//  * THE STATE IS ADDRESSABLE FOR THE READOUT.  Once per window the fp32 slice is
//    published as a bf16 IMAGE -- the SAME `pack_bf16x2` rounding the readout
//    formed anyway, at the same point in the recurrence -- in the LAY 3 slot order
//    below, and the readout's A operand is `ldmatrix.x4` out of it.  Because the
//    slice is an address, read batch `bi` belongs to warp `bi % kFbWarps` whatever
//    group it is drawn from, so a coherent window's one live read group is drained
//    by every warp.
//
// THE IMAGE'S SLOT ORDER, LAY 3 (prototypes/k31_listprobe, part B).  A leaf is
// `local = q*16 + 2g + j`: the WRITE carve wants `q` fixed and its 16 leaves
// contiguous, the READ carve wants `g` fixed and ITS 16 contiguous.  Those two
// partitions of the same 128 leaves intersect in two elements, so NO linear order
// makes both contiguous and `ldmatrix` can be given to exactly one side.  It is
// given to the READ side, which is the side that needs a 16-wide A operand; the
// write side never wanted `ldmatrix` because it STOREs.  The XOR is what makes the
// swap free -- keyed on the OTHER carve's digit, which is what is constant within
// an access:
//
//     slot(local) = g*16 + 2*(q ^ (g & 3)) + j
//
// Measured on that layout: publish 0 store-bank conflicts, read slice 0 load-bank
// conflicts AND `ldmatrix`-able.  The k permutation the XOR induces is absorbed by
// the readout's B operand, which reads level-0 amplitude `q ^ (g & 3)` -- a
// warp-constant XOR folded into a runtime ADDRESS, never a register-array index.
//
// SYMBOLS ARE `carry_kernel.cuh`'s, plus: `q` the level-0 (WRITE) digit, `g` the
// level-1 (READ) group at span `kGSR`, `j` the pair bit, `Ld` the image's leaf
// stride.  See docs/internals/carry/carry_kernel.md.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <type_traits>

#include "box.cuh"
#include "carry_kernel.cuh"

namespace rola::carry {

//: EIGHT WARPS IS THE CARVE, NOT A TUNING.  The write carve gives a warp ONE of
//: the outermost level's digits, and the flagship owner run is eight of them; the
//: register slice that follows (`kCols` channels x `BC/8` leaves) is what the
//: 128-register occupancy gate was measured against.
static constexpr int kFbWarps = 8;
static constexpr int kFbThreads = kFbWarps * 32;

template <int D, int B, int K, int M, int W, int C, int CR, int RS>
struct WriteGeom {
  using BP = BoxPlan<D, B, K, M>;
  static constexpr int BC = BP::kBC;
  static constexpr int kLeafW = BC / kFbWarps;
  static constexpr int LdV = kDv + kPadE;
  //: the FOLD's panel is CTA-SHARED and holds the WHOLE box, because the eight
  //: warps' leaf slices are DISJOINT and their union is the owner: one cooperative
  //: produce serves all eight where the read carve needed one panel per warp.
  static constexpr int LdS = BC + kPadE;
  static constexpr int LdO = kDv + kPadF;

  static constexpr int panel_bytes = 2 * C * LdS * (int)sizeof(__nv_bfloat16);
  //: THERE IS NO IMAGE (round 8 M2).  The readout reads the master itself, in the
  //: READ carve, out of registers; the migration streams through the FOLD's panel
  //: bytes, which are dead for the whole read phase.  One chunk is
  //: `8 channels x BC` fp32 = 4 KB, so the panel region holds any `K <= 4`.
  static constexpr int chunk_bytes = 8 * BC * (int)sizeof(float);
  static constexpr int face_bytes = panel_bytes;
  static_assert(face_bytes >= chunk_bytes, "the migration chunk lives in the panel bytes");
  //: THE READOUT'S TRANSPOSE BLOCK, one `kMmaN`-token region per warp -- and, at
  //: entry and exit, the state plane's page.  ROUND 4 ITEM 2 TRIED TO DELETE IT and
  //: the measurement refused: see the findings doc.  It is what turns the fan-in's
  //: FOUR cache lines per instruction into one, and at the entry-densest cell that
  //: is worth its 17,408 B.
  static constexpr int out_bytes = kFbWarps * kMmaN * LdO * (int)sizeof(float);
  static_assert(out_bytes >= kAtomLeaves * LdO * (int)sizeof(float),
                "the staging block must hold a whole state page");
  static constexpr int kListStride = ((W + 7) / 8) * 8;
  static constexpr int wlist_bytes = kListStride * (int)sizeof(uint16_t);
  //: ONE READ LIST PER GROUP.  A register-resident read carve gives a warp only
  //: its own group, so the list is the warp's own and the read phase has no CTA
  //: barrier at all -- round 1's property, recovered.
  static constexpr int rlist_bytes = kFbWarps * kListStride * (int)sizeof(uint16_t);
  static constexpr int kLW = W / kBallotLanes;
  static constexpr int kRegFloor = 128;
  static constexpr int slot_bytes = C * LdV * (int)sizeof(__nv_bfloat16);
  //: ROUND 8 (c) AND (d) WERE BUILT ON THIS BODY AND ARE REFUTED, and the receipt
  //: belongs here because both look free on paper.  (c) PER-WARP PANEL SLICES: each
  //: warp expanding only its own digit's sixteen leaves turns the panel's RAW and
  //: WAR into `__syncwarp` and leaves the CTA barrier guarding only the value ring.
  //: It does not move the barrier -- `barStall` 1.79 -> 1.66 at ALT k16 scattered and
  //: 2.12 -> 2.11 at k16 cohort=64 -- BECAUSE THE BARRIER WAS NEVER THE PANEL'S: the
  //: ring's slot reuse is a 256-thread-to-256-thread edge on its own and the panel's
  //: edges were riding along inside it for free.  What it does move is the fold's
  //: `ldmatrix` traffic, `LDS` 1.26x, because a transposed read of a 16-wide slice at
  //: a 24-element stride is worse than of the 128-wide panel at 136.  (d) ALIASING
  //: the fan-in staging onto the value ring is correct and saves 1,296 B, and round
  //: 5 already measured that slack under the two-CTA bar buys nothing.  Measured
  //: together: 0.864 -> 0.909, 0.794 -> 0.841, 1.087 -> 1.148, 0.999 -> 1.025.
  //:
  //: THE READ LISTS AND THE VALUE RING ARE PHASE-EXCLUSIVE and share their bytes:
  //: the lists are built at the top of the window and dead at the window line, the
  //: ring is filled after it, and the handover line the IMAGE already needs
  //: separates them.  Un-aliasing them was measured in round 4 and it LOSES -- not
  //: on `cta_target`, which stays 2 either way, but on the sm_86 L1/SMEM carveout:
  //: 49,424 B per CTA leaves less L1 than 45,328, and the cohort cells feel it.
  //: THE FAN-IN STAGING AND THE VALUE RING SHARE THEIR BYTES (round 8d): the
  //: staging is a READOUT tenant, dead at the window line, and the ring is a FOLD
  //: tenant, filled after it.  Correct and free -- and here it is what pays for the
  //: eight per-group read lists the register read carve needs.
  static constexpr int fixed_bytes = face_bytes + rlist_bytes + wlist_bytes;
  static constexpr int kStages =
      design::stages_for(slot_bytes, fixed_bytes + out_bytes, kFbThreads, kRegFloor);
  static constexpr int mid_bytes =
      out_bytes > kStages* slot_bytes ? out_bytes : kStages* slot_bytes;
  static constexpr size_t bytes = (size_t)fixed_bytes + (size_t)mid_bytes;
  static constexpr int cta_target = design::cta_target((int)bytes, kFbThreads, kRegFloor);
};

template <int D, int B, int K, int M, int W, int C, int CR, int RS>
constexpr int kWriteCtaTarget = WriteGeom<D, B, K, M, W, C, CR, RS>::cta_target;

// ---------------------------------------------------------------- the kernel

template <int D, int B, int K, int M, int W, int C, int CR, int RS, bool WSC, bool RS0, bool STATE>
__global__ __launch_bounds__(kFbThreads, (kWriteCtaTarget<D, B, K, M, W, C, CR, RS>)) void
carry_write_kernel(const __nv_bfloat16* __restrict__ pread,
                   const __nv_bfloat16* __restrict__ pwrite,
                   const __nv_bfloat16* __restrict__ gwrite, const __nv_bfloat16* __restrict__ v,
                   const uint32_t* __restrict__ rows, float* __restrict__ o,
                   float* __restrict__ den, int L, int tok_words, int v_row_bytes,
                   const float* __restrict__ s_in, float* __restrict__ s_out) {
  using BP = BoxPlan<D, B, K, M>;
  using G = WriteGeom<D, B, K, M, W, C, CR, RS>;
  using PM = ProduceMap<D, B, K, M, C, kFbThreads>;
  using CR2 = TwoCarveRows<D, B, K, M, kFbWarps, 0, CR>;

  constexpr int BC = BP::kBC;
  constexpr int LdV = G::LdV, LdS = G::LdS, LdO = G::LdO;
  constexpr int MTOK = C / kMmaM;
  constexpr int kLeafW = G::kLeafW;
  constexpr int NL = kLeafW / kMmaN;  // the warp's accumulator n-tiles
  constexpr int MCH = kDv / kMmaM;    // the channel m-tiles, ALL of them per warp
  //: THE READ BATCH IS TOKEN-CARVED, NOT GROUP-CARVED (round 6).  Round 5's ledger
  //: named the last term: a token live in `m` read groups was listed `m` times and
  //: FANNED IN `m` times, and that fan-in plus its staging was 2,146 warp-instructions
  //: per (owner, window) against 512 HMMA.  So a batch is now `C` of the OWNER's read
  //: tokens -- ONE list, multiplicity ONE -- and a warp drains the whole batch against
  //: the WHOLE image, accumulating across the token's live groups in registers and
  //: fanning in ONCE.  The read carve's group axis goes with it: the only grain left
  //: is the IMAGE's own k-tile, which under LAY 3 is exactly one pair-grain (sigma 2)
  //: group of sixteen slots.
  constexpr int kOutHi = BC / BP::s(CR);  // the leaves ABOVE the read level
  constexpr int kSigma = BC / kMmaK;      // the image's k-tiles = its sigma-2 groups

  //: THE SHAPE THIS BODY IS BUILT FOR, STATED RATHER THAN ASSUMED.  LAY 3's slot
  //: algebra and the write carve's contiguity are properties of a two-level box
  //: whose outer span is the warp count and whose read group is an ADJACENT PAIR;
  //: a declaration outside it is served one of the other two bodies at the seam,
  //: which is always correct and only ever slower.
  static_assert(D == 2 && CR == 1, "Form B carves level 0 by the writer and level 1 by the reader");
  static_assert(BP::s(0) == kFbWarps, "a warp owns exactly one of the writer's digits");
  static_assert(kOutHi == kMmaN && kSigma == kFbWarps,
                "LAY 3 places a pair-grain group's eight write digits in one k-tile, and the "
                "sorted list gives each warp one of those groups' segments");
  static_assert(kLeafW == kAtomLeaves, "a write digit's leaves are exactly one state page");
  static_assert(C % kMmaM == 0 && kFbThreads % C == 0, "the batch tile must divide");
  static_assert(kLeafW % kMmaK == 0, "the warp's leaves are a whole number of k-tiles");
  static_assert(BC % kAtomLeaves == 0, "K2: an owner holds whole page atoms");
  static_assert(design::smem_fits(G::bytes), "this arm does not fit the SMEM maximum");
  static_assert(LdS % 8 == 0 && LdV % 8 == 0, "an ldmatrix row address must be 16-byte aligned");
  static_assert(W % kBallotLanes == 0 && kFbThreads % kBallotLanes == 0,
                "a warp's tokens must be exactly one ballot word of the owner's row");
  static_assert(W / kMmaK <= 32, "the tile-skip predicate is one word of k-tiles per window");
  static_assert(PM::kRowThr * PM::SPT == BC, "the row's threads cover the box exactly");
  static_assert(PM::SPT % 2 == 0, "a thread publishes whole bf16 pairs");

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int owner = (int)blockIdx.x;
  const int bh = (int)blockIdx.y;

  extern __shared__ __align__(16) char carry_smem[];
  constexpr int STAGES = G::kStages;
  static_assert(STAGES >= 2, "the wait/issue order needs at least two stages");
  //: THE MAP.  `sr`/`sw` are the fold's two panel slots and the MIGRATION CHUNK
  //: shares their bytes (the panels are dead for the whole read phase, and the
  //: migration's own barriers plus the window line separate them).  The fan-in
  //: staging shares the value ring, which is the round-8(d) alias.  The lists own
  //: their bytes: the write list is live from the schedule to the fold's end.
  __nv_bfloat16* const sr = reinterpret_cast<__nv_bfloat16*>(carry_smem);
  __nv_bfloat16* const sw = sr + C * LdS;
  float* const schunk = reinterpret_cast<float*>(carry_smem);
  float* const spage = reinterpret_cast<float*>(carry_smem + G::face_bytes);
  __nv_bfloat16* const sv = reinterpret_cast<__nv_bfloat16*>(carry_smem + G::face_bytes);
  uint16_t* const srlist = reinterpret_cast<uint16_t*>(carry_smem + G::face_bytes + G::mid_bytes);
  uint16_t* const wlist = srlist + kFbWarps * G::kListStride;
  float* const sow = spage + warp * kMmaN * LdO;
  constexpr int kListStride = G::kListStride;

  const int vRowElems = v_row_bytes / (int)sizeof(__nv_bfloat16);
  const int heads = vRowElems / kDv;
  //: ONE PINNED GLOBAL BASE PER PLANE, and INDEX arithmetic after it.  The storage
  //: class is a property of the base, so every access below stays `LDG`/`RED` while
  //: ptxas keeps the right to fold a compile-time displacement into the access's own
  //: immediate and to share one `IMAD.WIDE` between neighbouring accesses.
  const __nv_bfloat16* const vPlane =
      global_base(v + (long)(bh / heads) * L * vRowElems + (long)(bh % heads) * kDv);
  const __nv_bfloat16* const pR = global_base(pread + (long)bh * L * BP::kWtot);
  const __nv_bfloat16* const pW = global_base(pwrite + (long)bh * L * BP::kWtot);
  const __nv_bfloat16* const gW = global_base(gwrite + (long)bh * L);
  float* const oPlane = global_base(o + (long)bh * L * kDv);
  float* const denPlane = global_base(den + (long)bh * L);

  //: THE GROUP ROTATION (K33 1.45), on the WRITE digit: two CTAs co-resident on an
  //: SM are one grid wave apart and the wave is a multiple of the warp count, so
  //: `owner % kFbWarps` would rotate both identically.
  const int grp = (warp + ((owner + (owner >> 5)) & (kFbWarps - 1))) & (kFbWarps - 1);

  // ---- the owner's digit runs, and the two carves' ballot rows -------------
  int runCol[D], rowIdx[D];
  static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    const int o_l = (owner / BP::owner_div(l)) % BP::g(l);
    runCol[l] = BP::col_base(l) + o_l * BP::s(l);
    //: both carved levels store `kFbWarps` rows per owner run, so the owner's
    //: block starts at `o_l * kFbWarps` in each.  This is a ROW INDEX, not a byte
    //: displacement: the row stride is `tok_words` words and a group step is
    //: therefore one `IMAD.WIDE` off a base the window forms once.
    rowIdx[l] = CR2::row_prefix(l) + o_l * kFbWarps;
  });
  const uint32_t* const rowsR =
      global_base(rows + (long)(2 * bh + 0) * (long)CR2::kRows * (long)tok_words);
  const uint32_t* const rowsW =
      global_base(rows + (long)(2 * bh + 1) * (long)CR2::kRows * (long)tok_words);

  // ---- lane-invariant operand addressing ----------------------------------
  const int lr = lane & 7, tl = lane >> 3;
  const int rF = (tl & 1) * 8 + lr, cF = tl >> 1;
  const int rT = (tl >> 1) * 8 + lr, cT = tl & 1;
  const uint32_t bwF0 = smem_addr(sw) + (uint32_t)((rF * LdS + cF * 8) * 2);
  const uint32_t bwF1 = smem_addr(sr) + (uint32_t)((rF * LdS + cF * 8) * 2);
  const uint32_t svB = smem_addr(sv) + (uint32_t)((rT * LdV + cT * 8) * 2);
  constexpr uint32_t kVSlot = (uint32_t)(C * LdV) * 2;
#define ADDR_S(b_, R_, g_) ((b_) + (uint32_t)(((R_) * LdS + (g_) * 8) * 2))
#define ADDR_V(b_, R_, dv_) ((b_) + (uint32_t)(((R_) * LdV + (dv_)) * 2))

  // ---- the state, transposed: m = channels, n = this warp's own leaves -----
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

  //: THE PUBLISH, IN LAY 3.  Accumulator element `e` of n-tile `jt` carries the
  //: warp's leaf `p = 8*jt + 2*(lane&3) + (e&1)`, and `p` IS the level-CR digit
  //: `2g + j` because the warp's leaves are the outer digit's contiguous run.  So
  //: `g = 4*jt + (lane&3)`, whose low two bits are `lane&3` -- the axis that varies
  //: WITHIN the access -- and `(e=2h, e=2h+1)` are `j = 0, 1`, i.e. two ADJACENT
  //: slots.  One `STS.32` per (m-tile, n-tile, half), sixteen in all, and the
  //: sixteen lanes of a phase cover banks 0-31 exactly once.
  //: THE READOUT'S A OPERAND IS THE ACCUMULATOR ITSELF.  After the migration the
  //: master is in the READ carve -- warp `grp` owns level-CR group `grp` across
  //: every level-0 digit, `p = (r1 - 2*grp)*8 + q` -- and its own `NL` n-tiles ARE
  //: the sixteen leaves of one `mma` k-step, so the fragment is `pack_bf16x2` in
  //: place: no image, no `ldmatrix`, no SMEM read at all on the state's side.
  auto state_operand = [&](uint32_t(&sa)[MCH][4], uint32_t(&ma)[4]) {
    static_for<MCH>([&](auto Mc) {
      constexpr int m = decltype(Mc)::value;
      sa[m][0] = pack_bf16x2(st[m][0][0], st[m][0][1]);
      sa[m][1] = pack_bf16x2(st[m][0][2], st[m][0][3]);
      sa[m][2] = pack_bf16x2(st[m][1][0], st[m][1][1]);
      sa[m][3] = pack_bf16x2(st[m][1][2], st[m][1][3]);
    });
    ma[0] = pack_bf16x2(mass_st[0][0], mass_st[0][1]);
    ma[1] = pack_bf16x2(mass_st[0][2], mass_st[0][3]);
    ma[2] = pack_bf16x2(mass_st[1][0], mass_st[1][1]);
    ma[3] = pack_bf16x2(mass_st[1][2], mass_st[1][3]);
  };

  constexpr uint32_t kOnesPair = 0x3F803F80u;
  uint32_t ones_a[4];
  ones_a[0] = (lane >> 2) == 0 ? kOnesPair : 0u;
  ones_a[1] = 0u;
  ones_a[2] = ones_a[0];
  ones_a[3] = 0u;

  // ---- the state plane's page sweep, in LATTICE leaf order -----------------
  //: A PAGE IS EXACTLY ONE WARP'S SLICE.  `kAtomLeaves` consecutive `local` values
  //: at a multiple of `kAtomLeaves` is precisely the outermost level's digit run
  //: under this carve, so atom `a` belongs whole to warp `a` and no page is split.
  constexpr int kAtoms = BC / kAtomLeaves;
  constexpr int kPageElems = kAtomLeaves * kCols;
  //: the cursor is a division by a compile-time `kCols`, which lowers to a
  //: multiply-shift: at `kFbThreads` the step crosses several row boundaries and a
  //: carry chain would need a loop where this needs three instructions.
  auto page_sweep = [&](uint32_t pageOff, auto&& body) {
#pragma unroll 1
    for (int e = tid; e < kPageElems; e += kFbThreads)
      body((e / kCols) * LdO + (e % kCols), pageOff + (uint32_t)(e * (int)sizeof(float)));
  };
  const uint32_t atomBase = (uint32_t)(bh * (int)(BP::kLeaves / kAtomLeaves) + owner * kAtoms);
  auto page_off = [](uint32_t atom) { return atom * (uint32_t)(kPageElems * (int)sizeof(float)); };
  auto page_regs = [&](int a, auto Ld_) {
    constexpr bool LOAD = decltype(Ld_)::value;
    if (grp != a) return;
    static_for<kLeafW>([&](auto Pc) {
      constexpr int p = decltype(Pc)::value;
      constexpr int tile = p / kMmaN, nix = p % kMmaN;
      if ((lane & 3) == nix / 2) {
        static_for<MCH>([&](auto Mc) {
          constexpr int m = decltype(Mc)::value;
          static_for<2>([&](auto Hc) {
            constexpr int k = (nix & 1) + 2 * decltype(Hc)::value;
            float* const cell = &spage[p * LdO + m * kMmaM + (lane >> 2) + 8 * (k >> 1)];
            if constexpr (LOAD)
              st[m][tile][k] = *cell;
            else
              store_f32(cell, st[m][tile][k]);
          });
        });
        if (lane == nix / 2) {
          float* const cell = &spage[p * LdO + kDv];
          if constexpr (LOAD)
            mass_st[tile][nix & 1] = *cell;
          else
            store_f32(cell, mass_st[tile][nix & 1]);
        }
      }
    });
  };

  if constexpr (STATE) {
    if (s_in != nullptr) {
      const float* const plane = pin_address(s_in);
#pragma unroll
      for (int a = 0; a < kAtoms; ++a) {
        page_sweep(page_off(atomBase + (uint32_t)a), [&](int sidx, uint32_t off) {
          spage[sidx] = *reinterpret_cast<const float*>(gather_ptr(plane, off));
        });
        rendezvous();
        page_regs(a, std::true_type{});
        rendezvous();
      }
    }
  }

  // ---- the fold panel's per-thread produce map -----------------------------
  constexpr int kRowThr = PM::kRowThr, SPT = PM::SPT;
  const int prow = tid / kRowThr;
  const int slot0 = (tid % kRowThr) * SPT;
  int lvCol[D];
  static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    lvCol[l] = runCol[l] + BP::run_of(l, slot0);
  });

  //: THE FOLD'S PANEL, expanded from factors, CTA-COOPERATIVELY.  A padding row of
  //: a partial batch has no token and is zeroed through the GAIN, which every write
  //: coefficient carries.  The slot accumulators are the only array this holds: a
  //: level's run is loaded, folded into every slot and dropped inside one
  //: `static_for` body.
  auto produce = [&](__nv_bfloat16* dst, const __nv_bfloat16* tokp, uint32_t gain) {
    uint32_t acc[SPT / 2];
    {
      unsigned short run[PM::n_l(D - 1)];
      load_run<PM::n_l(D - 1)>(run, tokp + lvCol[D - 1]);
#pragma unroll
      for (int j = 0; j < SPT; j += 2)
        acc[j / 2] =
            ((uint32_t)run[PM::idx_of(D - 1, j + 1)] << 16) | (uint32_t)run[PM::idx_of(D - 1, j)];
    }
    static_for<D - 1>([&](auto Lc) {
      constexpr int l = decltype(Lc)::value;
      unsigned short run[PM::n_l(l)];
      load_run<PM::n_l(l)>(run, tokp + lvCol[l]);
#pragma unroll
      for (int j = 0; j < SPT; j += 2)
        acc[j / 2] = mul_bf16x2(acc[j / 2], splat_bf16x2_bits(run[PM::idx_of(l, j)]));
    });
    __nv_bfloat16* const row = dst + prow * LdS + slot0;
#pragma unroll
    for (int j = 0; j < SPT; j += 2) store_bf16x2(row + j, mul_bf16x2(acc[j / 2], gain));
  };

  //: THE READOUT'S B FRAGMENT, SYNTHESIZED DIRECTLY IN MMA LAYOUT.  Lane `l` serves
  //: token `n = l >> 2`; register `r`'s two halves are k-indices
  //: `kx = 8r + 2*(l&3) + h`, which under LAY 3 name leaf `(q, j)` with
  //: `q = 4r + ((l&3) ^ (g&3))` and `j = h`.  The two halves therefore share `q` and
  //: differ only in the level-CR pair, so ONE 4-byte load carries both and the
  //: level-0 factor is a scalar load at a RUNTIME ADDRESS -- no panel, no `STS`, no
  //: `ldmatrix` for coefficients, and no register-array index.
  //: TWO ROW BASES, THEN INDICES.  The token's amplitude row is one `IMAD.WIDE` off
  //: the window's base, and the level-CR and level-0 column origins are folded into
  //: it ONCE per token tile; what is left per k-step is the XOR'd level-0 digit
  //: (`qb`, one more `IMAD.WIDE`) and, off that, COMPILE-TIME displacements that
  //: ptxas folds into the load's own immediate.  Round 4's ledger measured the old
  //: per-access `gather_ptr` form at three opaque 64-bit adds per k-step per token
  //: tile, none of them foldable and none of them shared.
  //: THE READOUT'S B FRAGMENT, SYNTHESIZED DIRECTLY IN MMA LAYOUT.  Under the read
  //: carve's private order `p = (r1 - 2*grp)*8 + q`, register `r`'s two halves are
  //: k-indices `kx = 8r + 2*(l&3) + h`, so `r1 = 2*grp + r` (compile-time per
  //: register) and `q = 2*(l&3) + h` -- the PAIR is the LEVEL-0 run, one 4-byte load
  //: carries both halves, and the level-CR factor is a scalar at a runtime address.
  //: LAY 3's XOR is gone with the image that needed it.
  auto read_pair = [&](const __nv_bfloat16* tokp, bool have, uint32_t(&rb)[2]) {
    const uint32_t pair =
        have ? *reinterpret_cast<const uint32_t*>(tokp + runCol[0] + 2 * (lane & 3)) : 0u;
#pragma unroll
    for (int r = 0; r < 2; ++r)
      rb[r] = mul_bf16x2(pair, splat_bf16x2_bits(*reinterpret_cast<const unsigned short*>(
                                   tokp + runCol[CR] + 2 * grp + r)));
  };

  // ---- V staging -----------------------------------------------------------
  constexpr int kChunkV = kDv / 8;
  constexpr int kRowsPerPass = kFbThreads / kChunkV;
  constexpr int kPasses = C / kRowsPerPass;
  static_assert(kPasses >= 1, "a batch must be at least kFbThreads/8 rows");
  const int ldRow = tid / kChunkV, ldCol = (tid % kChunkV) * 8;
  const uint32_t svL = smem_addr(sv) + (uint32_t)((ldRow * LdV + ldCol) * 2);

  const int nWindows = (L + W - 1) / W;
  //: THE WINDOW BASES ARE WALKED, NOT RE-DERIVED.  Forming them from the plane each
  //: window keeps BOTH the plane base and the window base live -- ten 64-bit
  //: registers where five will do, and at a 128-register bound a long-lived 64-bit
  //: value is the scarcest thing there is (measured: +6 registers and a wall
  //: regression on every ALT cell).  Advancing them in the loop's own increment
  //: retires the plane bases after the first use and survives the whole-window
  //: `continue`.
  const __nv_bfloat16* pRw = pR;
  const __nv_bfloat16* pWw = pW;
  const __nv_bfloat16* gWw = gW;
  float* oWin = oPlane + lane;
  float* dWin = denPlane;

#pragma unroll 1
  for (int win = 0; win < nWindows; ++win, pRw += (long)W * BP::kWtot, pWw += (long)W * BP::kWtot,
           gWw += W, oWin += (long)W * kDv, dWin += W) {
    const int tok0 = win * W;
    const int wLen = (L - tok0) < W ? (L - tok0) : W;
    const uint32_t winV = (uint32_t)tok0 * (uint32_t)v_row_bytes;
    //: THE ROW TABLES KEEP THEIR BYTE-OFFSET GATHER, and that is a REGISTER ruling
    //: rather than an instruction one.  Hoisting a base per (side, level) saves
    //: sixteen `IMAD.WIDE` per window and costs four LIVE 64-bit registers, which is
    //: a four-byte spill on sm_80 at the 128-register bound -- measured, and no
    //: spill is admitted.  The row's stride is a runtime word count, so this is a
    //: genuine byte offset and `gather_ptr` is the right primitive for it.
    const int wIdx0 = (tok0 >> 5) + ((lane < G::kLW) ? lane : 0);
    const uint32_t wordB = (uint32_t)(wIdx0 < tok_words ? wIdx0 : tok_words - 1) * 4u;
    const uint32_t rowBytes = (uint32_t)tok_words * 4u;
    auto produce_batch = [&](__nv_bfloat16* dst, int o0, int ntot) {
      const int nb = (ntot - o0) < C ? (ntot - o0) : C;
      const bool have = prow < nb;
      const int tok = wlist[o0 + (have ? prow : nb - 1)];
      const uint32_t gain = have ? splat_bf16x2(gWw[tok]) : 0u;
      produce(dst, pWw + (int)tok * BP::kWtot, gain);
    };

    // ---- the liveness pass ------------------------------------------------
    constexpr int LW = G::kLW;
    static_assert(W <= 0xFFFF, "a token's window-local index must fit the list's element");
    static_assert(LW <= kBallotLanes, "the walk gives one ballot word to each lane");
    const int room = wLen - lane * kBallotLanes;
    const bool inW = lane < LW;
    const uint32_t valid =
        (!inW || room <= 0) ? 0u : (room >= kBallotLanes ? 0xFFFFFFFFu : ((1u << room) - 1u));
    auto walk = [&](uint32_t m, uint16_t* dst, uint32_t bits, uint32_t below) {
      const int cnt = bit_count(m);
      const int pfx = lane_prefix_incl<LW>(cnt);
      int k = (pfx - cnt) + bit_count(m & below);
      const uint16_t tb = (uint16_t)(lane * kBallotLanes);
      uint32_t mm = m & bits;
      while (mm) {
        dst[k++] = tb + (uint16_t)first_set(mm);
        mm &= mm - 1;
      }
      return lane_broadcast(pfx, LW - 1);
    };
    auto row_word = [&](const uint32_t* rowsS, int l, int g) {
      return *reinterpret_cast<const uint32_t*>(
          gather_ptr(rowsS, (uint32_t)(rowIdx[l] + g) * rowBytes + wordB));
    };

    //: BOTH SIDES READ BOTH CARVES.  Each level's rows are stored at the grain ONE
    //: side wanted, so the other side ORs the `kFbWarps` of them to recover its own
    //: owner-grain clause -- which is exactly what a refinement is for, and it is
    //: the price of one table serving two carves.
    //: EVERY WARP DERIVES EVERY GROUP'S COUNT in the same pass that loads the rows,
    //: from ballot words it holds anyway -- which is what makes the read phase's
    //: work a CTA-WIDE SUPPLY rather than `kFbWarps` private ones, with no
    //: publication and no barrier of its own.  Only the WARP'S OWN read row
    //: survives the pass: a `[kFbWarps]` array of them is thirty-two bytes of live
    //: state at a 128-register bound, which is a stack frame.
    //: A DENSE SIDE HAS NO CLAUSE AT ITS DENSE LEVEL, and the DECLARATION is what
    //: says so (K33 law 3, absent-means-full).  The table stores level 0 at the
    //: WRITE carve's grain and level CR at the READ carve's, so the OTHER side
    //: would have to OR `kFbWarps` rows to recover an owner-grain word it does not
    //: need -- sixteen of the thirty-two row loads per warp per window.  `WSC` and
    //: `RS0` are those two declarations, template arguments named at the seam, so
    //: the loads are absent from the body rather than predicated in it.
    uint32_t orW0 = 0u, orW1 = 0u, orR0 = 0u, orR1 = 0u, myW0 = 0u, mineR = 0u;
    static_for<kFbWarps>([&](auto Gc) {
      constexpr int g = decltype(Gc)::value;
      const uint32_t w0 = row_word(rowsW, 0, g);
      orW0 |= w0;
      myW0 = (g == grp) ? w0 : myW0;
      const uint32_t rC = row_word(rowsR, CR, g);
      orR1 |= rC;
      mineR = (g == grp) ? rC : mineR;
      if constexpr (WSC) orW1 |= row_word(rowsW, CR, g);
      if constexpr (RS0) orR0 |= row_word(rowsR, 0, g);
    });
    const uint32_t clauseW = WSC ? (valid & orW1) : valid;
    const uint32_t clauseR = RS0 ? (valid & orR0) : valid;
    const uint32_t liveW = clauseW & orW0;

    //: THE READ LIST IS THE WARP'S OWN GROUP'S (round 8 M2).  With the master
    //: READ-CARVED INTO REGISTERS a warp can serve only its own group, so the list,
    //: the count, the batches and the fan-in are all warp-private and the read phase
    //: has NO CTA BARRIER of its own -- round 1's property, recovered.  The price is
    //: the fan-in MULTIPLICITY the carve forces: a token whose level-CR support
    //: spans `m` groups is drained by `m` warps.  It is the design's bet against
    //: round 7's addressable image, and the five cells decide it.
    const int nr = lane_broadcast(lane_prefix_incl<LW>(bit_count(clauseR & mineR)), LW - 1);
    const int nbr = (nr + C - 1) / C;
    //: THE WHOLE-WINDOW SKIP, at (owner, window) grain, per side.  CTA-uniform
    //: without a barrier: the two owner-grain ballots are the same words in every
    //: warp, and `orR1` is the OR the row pass already produced.
    if (lane_ballot(((clauseR & orR1) | liveW) != 0u) == 0u) continue;

    constexpr uint32_t kBitsW = (uint32_t)(kBallotLanes / kFbWarps);
    const int nw =
        walk(liveW, wlist, ((1u << kBitsW) - 1u) << (warp * kBitsW), (1u << (warp * kBitsW)) - 1u);
    uint16_t* const rlist = srlist + grp * kListStride;
    if (nbr > 0) walk(clauseR & mineR, rlist, 0xFFFFFFFFu, 0u);
    __syncwarp();

    //: THE TILE-SKIP PREDICATE.  Bit `T` says the warp's own write digit is live in
    //: k-tile `T` of the owner's write list.  A token's list slot is its rank in
    //: `liveW`, so the tile is that rank's top bits.  NOTHING about the list, its
    //: order or its tile boundaries moves -- only which tiles this warp executes.
    uint32_t tiles;
    {
      const int cntL = bit_count(liveW);
      const int baseL = lane_prefix_incl<LW>(cntL) - cntL;
      uint32_t t = 0u;
      uint32_t mm = clauseW & myW0;
      while (mm) {
        const int b = first_set(mm);
        mm &= mm - 1;
        t |= 1u << ((baseL + bit_count(liveW & ((1u << b) - 1u))) >> ilog2c(kMmaK));
      }
      tiles = lane_reduce_or(t);
    }

    //: ---- THE CARVE MIGRATION (round 8 base, law 12) -----------------------
    //: ONE MASTER THAT CHANGES CARVE, streamed through the fold's panel bytes as a
    //: RING OF `KR` CHANNEL-BLOCK CHUNKS.  Chunk `c` is channels `[8c, 8c+8)`,
    //: which in the accumulator's lane map are EXACTLY the four registers
    //: `st[c>>1][j][2*(c&1)+b]` -- so the publish is the last use of those four
    //: scalars and the pull is their next def, and the peak is one master plus one
    //: chunk.  The WRITE carve's two floats per n-tile are adjacent `local` (one
    //: `STS.64`); the READ carve's are stride `s(CR)` (two `LDS.32`).  The mass
    //: column is the ninth chunk and rides the four lanes holding accumulator row 0.
    //:
    //: THE RING IS WHAT COLLAPSES THE BARRIERS.  Publishing `KR` chunks before the
    //: handoff costs ONE rendezvous per round instead of one per chunk: 36 CTA
    //: barriers per window at M1/M2 become 10.
    constexpr int NCH = MCH * 2;  // the value chunks
    constexpr int NCM = NCH + 1;  // and the mass
    constexpr int KR = 4;         // ring slots; a chunk is 8 * BC fp32 = 4 KB
    static_assert(KR * 8 * BC * (int)sizeof(float) <= G::face_bytes,
                  "the chunk ring lives in the fold's panel bytes");
    //: the last `KR` published chunks land in `KR` DISTINCT slots (`slot = c % KR`
    //: and publishing stops at `NCM-1`), which is what lets them be retained.
    static_assert(NCM >= KR, "the retained tail is one ring's worth of chunks");
    const int chl = lane >> 2;
    auto wi = [&](int j, int b) { return chl * BC + grp * kLeafW + 8 * j + 2 * (lane & 3) + b; };
    auto ri = [&](int j, int b) {
      const int pp = 8 * j + 2 * (lane & 3) + b;
      return chl * BC + (pp & 7) * BP::s(CR) + 2 * grp + (pp >> 3);
    };
    auto mwi = [&](int j, int b) { return grp * kLeafW + 8 * j + 2 * lane + b; };
    auto mri = [&](int j, int b) {
      const int pp = 8 * j + 2 * lane + b;
      return (pp & 7) * BP::s(CR) + 2 * grp + (pp >> 3);
    };
    //: `FROMW` names the carve the registers currently hold.
    auto pub = [&](auto Cc, auto Fw, int slot) {
      constexpr int c = decltype(Cc)::value;
      constexpr bool FROMW = decltype(Fw)::value;
      float* const sp = schunk + slot * (8 * BC);
      if constexpr (c < NCH) {
        constexpr int m = c >> 1, eh = c & 1;
        static_for<NL>([&](auto Jc) {
          constexpr int j = decltype(Jc)::value;
          sp[FROMW ? wi(j, 0) : ri(j, 0)] = st[m][j][2 * eh];
          sp[FROMW ? wi(j, 1) : ri(j, 1)] = st[m][j][2 * eh + 1];
        });
      } else if (lane < 4) {
        static_for<NL>([&](auto Jc) {
          constexpr int j = decltype(Jc)::value;
          static_for<2>([&](auto Bc) {
            constexpr int b = decltype(Bc)::value;
            sp[FROMW ? mwi(j, b) : mri(j, b)] = mass_st[j][b];
          });
        });
      }
    };
    auto pull = [&](auto Cc, auto Tw, int slot) {
      constexpr int c = decltype(Cc)::value;
      constexpr bool TOW = decltype(Tw)::value;
      const float* const sp = schunk + slot * (8 * BC);
      if constexpr (c < NCH) {
        constexpr int m = c >> 1, eh = c & 1;
        static_for<NL>([&](auto Jc) {
          constexpr int j = decltype(Jc)::value;
          st[m][j][2 * eh] = sp[TOW ? wi(j, 0) : ri(j, 0)];
          st[m][j][2 * eh + 1] = sp[TOW ? wi(j, 1) : ri(j, 1)];
        });
      } else if (lane < 4) {
        static_for<NL>([&](auto Jc) {
          constexpr int j = decltype(Jc)::value;
          static_for<2>([&](auto Bc) {
            constexpr int b = decltype(Bc)::value;
            mass_st[j][b] = sp[TOW ? mwi(j, b) : mri(j, b)];
          });
        });
      }
    };
    constexpr int NR = (NCM + KR - 1) / KR;
    //: FORWARD, write -> read: every chunk streams.
    auto migrate_fwd = [&]() {
      static_for<NR>([&](auto Rc) {
        constexpr int r = decltype(Rc)::value;
        static_for<KR>([&](auto Sc) {
          constexpr int sidx = decltype(Sc)::value;
          if constexpr (r * KR + sidx < NCM)
            pub(std::integral_constant<int, r * KR + sidx>{}, std::true_type{}, sidx);
        });
        rendezvous();
        static_for<KR>([&](auto Sc) {
          constexpr int sidx = decltype(Sc)::value;
          if constexpr (r * KR + sidx < NCM)
            pull(std::integral_constant<int, r * KR + sidx>{}, std::false_type{}, sidx);
        });
        if constexpr (r + 1 < NR) rendezvous();
      });
    };
    //: REVERSE, read -> write: THE LAST `KR` CHUNKS ARE STILL IN THE RING AND
    //: UNMODIFIED -- the readout never writes the state and never touches these
    //: bytes -- so their writers reload them DIRECTLY, with no publish and NO
    //: SYNC.  Only `NCM - KR` chunks stream.
    auto migrate_rev = [&]() {
      static_for<KR>([&](auto Sc) {
        constexpr int sidx = decltype(Sc)::value;
        constexpr int c = NCM - KR + sidx;
        pull(std::integral_constant<int, c>{}, std::true_type{}, c % KR);
      });
      rendezvous();
      constexpr int NRB = (NCM - KR + KR - 1) / KR;
      static_for<NRB>([&](auto Rc) {
        constexpr int r = decltype(Rc)::value;
        static_for<KR>([&](auto Sc) {
          constexpr int sidx = decltype(Sc)::value;
          if constexpr (r * KR + sidx < NCM - KR)
            pub(std::integral_constant<int, r * KR + sidx>{}, std::false_type{}, sidx);
        });
        rendezvous();
        static_for<KR>([&](auto Sc) {
          constexpr int sidx = decltype(Sc)::value;
          if constexpr (r * KR + sidx < NCM - KR)
            pull(std::integral_constant<int, r * KR + sidx>{}, std::true_type{}, sidx);
        });
        if constexpr (r + 1 < NRB) rendezvous();
      });
    };

    //: THE HANDOVER LINE, and it is the migration's alone: the chunk ring lives in
    //: the fold's panel bytes, so the previous window's fold must be done reading
    //: them before this window's first publish overwrites them.  One barrier.
    rendezvous();
    migrate_fwd();

    // ---- PHASE 1: the READOUT, against the window's ENTRY state ------------
    //: THE WARP DRAINS ITS OWN GROUP'S TOKENS against its own read-carved slice.
    //: The A operand is `pack_bf16x2` of the accumulator in place -- hoisted per
    //: BATCH, since it does not move with the token.
#pragma unroll 1
    for (int i = 0; i < nbr; ++i) {
      const int off = i * C;
      const int n = (nr - off) < C ? (nr - off) : C;
      uint32_t sa[MCH][4], ma[4];
      state_operand(sa, ma);
#pragma unroll
      for (int j = 0; j < C / kMmaN; ++j) {
        const int idx = j * kMmaN + (lane >> 2);
        const bool have = idx < n;
        const __nv_bfloat16* const tokp = pRw + (int)rlist[off + (have ? idx : 0)] * BP::kWtot;
        uint32_t rb[2];
        read_pair(tokp, have, rb);
        float acc[MCH][4], macc[4];
#pragma unroll
        for (int m = 0; m < MCH; ++m)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[m][e] = 0.f;
#pragma unroll
        for (int e = 0; e < 4; ++e) macc[e] = 0.f;
#pragma unroll
        for (int m = 0; m < MCH; ++m) mma(acc[m], sa[m], rb);
        mma(macc, ma, rb);
        //: THE FAN-IN IS WARP-PRIVATE, AND IT IS STAGED (round 4 item 2's ruling).
#pragma unroll
        for (int m = 0; m < MCH; ++m)
#pragma unroll
          for (int e = 0; e < 4; ++e)
            store_f32(
                &sow[((lane & 3) * 2 + (e & 1)) * LdO + m * kMmaM + (lane >> 2) + 8 * (e >> 1)],
                acc[m][e]);
        if (lane < 4) {
          store_f32(&sow[(lane * 2 + 0) * LdO + kDv], macc[0]);
          store_f32(&sow[(lane * 2 + 1) * LdO + kDv], macc[1]);
        }
        __syncwarp();
#pragma unroll
        for (int r = 0; r < kMmaN; ++r) {
          if (j * kMmaN + r < n) {
            float* const dp = oWin + (int)rlist[off + j * kMmaN + r] * kDv;
            fan_in_add(dp, sow[r * LdO + lane]);
            fan_in_add(dp + kBallotLanes, sow[r * LdO + lane + kBallotLanes]);
          }
        }
        if (lane < kMmaN && j * kMmaN + lane < n)
          fan_in_add(dWin + (int)rlist[off + j * kMmaN + lane], sow[lane * LdO + kDv]);
        __syncwarp();
      }
    }

    //: back to the WRITE carve for the fold.
    migrate_rev();

    //: THE WINDOW LINE, by the ordering law: no warp may fold this window's writes
    //: while another is still reading the window's entry state.
    rendezvous();

    // ---- PHASE 2: the FOLD of every write batch ---------------------------
    const int nbw = (nw + C - 1) / C;
    auto stage_batch = [&](int b, int slot) {
      const int o0 = b * C;
      const int nb = (nw - o0) < C ? (nw - o0) : C;
#pragma unroll
      for (int p = 0; p < kPasses; ++p) {
        const int r = ldRow + p * kRowsPerPass;
        const int t = wlist[o0 + (r < nb ? r : nb - 1)];
        stage_tile(
            svL + (uint32_t)slot * kVSlot + (uint32_t)(p * kRowsPerPass * LdV * 2),
            gather_ptr(vPlane, (uint32_t)t * (uint32_t)v_row_bytes + winV + (uint32_t)(ldCol * 2)));
      }
    };
#pragma unroll 1
    for (int sst = 0; sst < STAGES - 1; ++sst) {
      if (sst < nbw) stage_batch(sst, sst % STAGES);
      stage_commit();
    }
    if (nbw > 0) produce_batch(sw, 0, nw);
#pragma unroll 1
    for (int i = 0; i < nbw; ++i) {
      const uint32_t bwF = (i & 1) ? bwF1 : bwF0;
      const uint32_t vsl = svB + (uint32_t)(i % STAGES) * kVSlot;
      stage_wait<STAGES - 2>();
      rendezvous();
      const int f = i + STAGES - 1;
      if (f < nbw) stage_batch(f, f % STAGES);
      stage_commit();

#pragma unroll
      for (int kk = 0; kk < MTOK; ++kk) {
        //: THE SKIP, AND IT IS AN IDENTITY.  Every leaf of this warp carries this
        //: digit, so a k-tile the digit is dead in contributes an exact zero to all
        //: `NL x MCH` accumulators; the tile boundaries, the list and its order are
        //: untouched, which is why the reverse family's replay stays bit-identical.
        if (((tiles >> (i * MTOK + kk)) & 1u) == 0u) continue;
        uint32_t a[MCH][4];
#pragma unroll
        for (int m = 0; m < MCH; ++m) load_frag_t(a[m], ADDR_V(vsl, kk * 16, m * kMmaM));
#pragma unroll
        for (int nj = 0; nj < NL / 2; ++nj) {
          uint32_t b[4];
          load_frag_t(b, ADDR_S(bwF, kk * 16, grp * (kLeafW / 8) + 2 * nj));
#pragma unroll
          for (int m = 0; m < MCH; ++m) {
            mma(st[m][2 * nj], a[m], &b[0]);
            mma(st[m][2 * nj + 1], a[m], &b[2]);
          }
          mma(mass_st[2 * nj], ones_a, &b[0]);
          mma(mass_st[2 * nj + 1], ones_a, &b[2]);
        }
      }
      produce_batch(((i + 1) & 1) ? sr : sw, (i + 1) * C, nw);
    }
    stage_wait<0>();
  }

  // ---- the FINAL STATE, in LATTICE leaf order ------------------------------
  if constexpr (STATE) {
    if (s_out != nullptr) {
      float* const plane = pin_address(s_out);
      rendezvous();
#pragma unroll
      for (int a = 0; a < kAtoms; ++a) {
        page_regs(a, std::false_type{});
        rendezvous();
        page_sweep(page_off(atomBase + (uint32_t)a), [&](int sidx, uint32_t offb) {
          store_f32(reinterpret_cast<float*>(gather_ptr(plane, offb)), spage[sidx]);
        });
        rendezvous();
      }
    }
  }
#undef ADDR_S
#undef ADDR_V
}

}  // namespace rola::carry
