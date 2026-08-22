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
  static constexpr int Ld = BC + kPadE;

  static constexpr int panel_bytes = 2 * C * LdS * (int)sizeof(__nv_bfloat16);
  static constexpr int img_bytes = kCols * Ld * (int)sizeof(__nv_bfloat16);
  //: THE IMAGE AND THE FOLD'S PANELS ARE PHASE-EXCLUSIVE WITHIN A WINDOW and share
  //: their bytes: the readout has no panel at all (its coefficients are synthesized
  //: straight into MMA-layout registers), so the panel region is dead through phase
  //: 1, and the window line separates the last image read from the first panel
  //: write exactly as the ordering law already requires.
  static constexpr int face_bytes = img_bytes > panel_bytes ? img_bytes : panel_bytes;
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
  //: ONE LIST PER READ GROUP, and the group count is `RS`'s, not the warp count's:
  //: dynamic slice assignment decoupled them, so a coarser read grain allocates
  //: fewer lists as well as reaching fewer warps.
  static constexpr int kGR = BoxPlan<D, B, K, M>::s(CR) / RS;
  static constexpr int rlist_bytes = kGR * kListStride * (int)sizeof(uint16_t);
  static constexpr int kLW = W / kBallotLanes;
  static constexpr int kRegFloor = 128;
  static constexpr int slot_bytes = C * LdV * (int)sizeof(__nv_bfloat16);
  //: THE READ LISTS AND THE VALUE RING ARE PHASE-EXCLUSIVE and share their bytes:
  //: the lists are built at the top of the window and dead at the window line, the
  //: ring is filled after it, and the handover line the IMAGE already needs
  //: separates them.  Un-aliasing them was measured in round 4 and it LOSES -- not
  //: on `cta_target`, which stays 2 either way, but on the sm_86 L1/SMEM carveout:
  //: 49,424 B per CTA leaves less L1 than 45,328, and the cohort cells feel it.
  static constexpr int fixed_bytes = face_bytes + out_bytes + wlist_bytes;
  static constexpr int kStages = design::stages_for(slot_bytes, fixed_bytes, kFbThreads, kRegFloor);
  static constexpr int ring_bytes =
      kStages * slot_bytes > rlist_bytes ? kStages* slot_bytes : rlist_bytes;
  static constexpr size_t bytes = (size_t)fixed_bytes + (size_t)ring_bytes;
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
  constexpr int LdV = G::LdV, LdS = G::LdS, LdO = G::LdO, Ld = G::Ld;
  constexpr int MTOK = C / kMmaM;
  constexpr int kLeafW = G::kLeafW;
  constexpr int NL = kLeafW / kMmaN;  // the warp's accumulator n-tiles
  constexpr int MCH = kDv / kMmaM;    // the channel m-tiles, ALL of them per warp
  //: THE READ CARVE'S GRAIN IS ITS OWN AXIS, and it is NOT the warp count.  Once
  //: the state is addressable the warp -> slice map is dynamic, so the number of
  //: read GROUPS and the number of WARPS are independent: `RS` names the group's
  //: level-CR digits and `kGR` how many groups an owner run holds.  The finest
  //: grain minimises read k-TILES and the coarsest minimises read-entry
  //: MULTIPLICITY, and multiplicity is what the per-entry fan-in and staging cost
  //: is proportional to -- so the grain is swept, not assumed.
  constexpr int kGSR = RS;                // a read group's level-CR digits
  constexpr int kGR = BP::s(CR) / kGSR;   // read groups per owner run
  constexpr int kSub = kGSR / 2;          // its adjacent pairs, i.e. its k-steps
  constexpr int kOutHi = BC / BP::s(CR);  // the leaves ABOVE the read level

  //: THE SHAPE THIS BODY IS BUILT FOR, STATED RATHER THAN ASSUMED.  LAY 3's slot
  //: algebra and the write carve's contiguity are properties of a two-level box
  //: whose outer span is the warp count and whose read group is an ADJACENT PAIR;
  //: a declaration outside it is served one of the other two bodies at the seam,
  //: which is always correct and only ever slower.
  static_assert(D == 2 && CR == 1, "Form B carves level 0 by the writer and level 1 by the reader");
  static_assert(BP::s(0) == kFbWarps, "a warp owns exactly one of the writer's digits");
  static_assert(kGSR >= 2 && kGSR % 2 == 0 && kGR >= 1 && kOutHi == kMmaN,
                "LAY 3 places a read group's adjacent pairs beside the eight write digits, so "
                "a group is a whole number of them");
  static_assert(kLeafW == kAtomLeaves, "a write digit's leaves are exactly one state page");
  static_assert(C % kMmaM == 0 && kFbThreads % C == 0, "the batch tile must divide");
  static_assert(kLeafW % kMmaK == 0, "the warp's leaves are a whole number of k-tiles");
  static_assert(BC % kAtomLeaves == 0, "K2: an owner holds whole page atoms");
  static_assert(design::smem_fits(G::bytes), "this arm does not fit the SMEM maximum");
  static_assert(LdS % 8 == 0 && LdV % 8 == 0 && Ld % 8 == 0,
                "an ldmatrix row address must be 16-byte aligned");
  static_assert(W % kBallotLanes == 0 && kFbThreads % kBallotLanes == 0,
                "a warp's tokens must be exactly one ballot word of the owner's row");
  static_assert(W / kMmaK <= 32, "the tile-skip predicate is one word of k-tiles per window");
  static_assert(PM::kRowThr * PM::SPT == BC, "the row's threads cover the box exactly");
  static_assert(PM::SPT % 2 == 0, "a thread publishes whole bf16 pairs");

  constexpr int KDw = kSub;  // the read group's `ldmatrix` k-steps

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int owner = (int)blockIdx.x;
  const int bh = (int)blockIdx.y;

  extern __shared__ __align__(16) char carry_smem[];
  constexpr int STAGES = G::kStages;
  static_assert(STAGES >= 2, "the wait/issue order needs at least two stages");
  //: THE MAP, IN ITS TWO ALIASED PAIRS.  `sr`/`sw` are the fold's two panel slots
  //: and `simg` is the image over the same bytes; `sv` is the value ring and
  //: `srlist` the read lists over the same bytes.  Both pairs are phase-exclusive
  //: within a window and the handover line at the top of it is what separates them.
  __nv_bfloat16* const sr = reinterpret_cast<__nv_bfloat16*>(carry_smem);
  __nv_bfloat16* const sw = sr + C * LdS;
  __nv_bfloat16* const simg = sr;
  float* const spage = reinterpret_cast<float*>(carry_smem + G::face_bytes);
  __nv_bfloat16* const sv =
      reinterpret_cast<__nv_bfloat16*>(carry_smem + G::face_bytes + G::out_bytes);
  uint16_t* const srlist = reinterpret_cast<uint16_t*>(sv);
  uint16_t* const wlist =
      reinterpret_cast<uint16_t*>(carry_smem + G::face_bytes + G::out_bytes + G::ring_bytes);
  float* const sow = spage + warp * kMmaN * LdO;
  constexpr int kListStride = G::kListStride;

  const int vRowElems = v_row_bytes / (int)sizeof(__nv_bfloat16);
  const int heads = vRowElems / kDv;
  const __nv_bfloat16* const vPlane =
      pin_address(v + (long)(bh / heads) * L * vRowElems + (long)(bh % heads) * kDv);
  const __nv_bfloat16* const pR = pin_address(pread + (long)bh * L * BP::kWtot);
  const __nv_bfloat16* const pW = pin_address(pwrite + (long)bh * L * BP::kWtot);
  const __nv_bfloat16* const gW = pin_address(gwrite + (long)bh * L);
  float* const oPlane = pin_address(o + (long)bh * L * kDv);
  float* const denPlane = pin_address(den + (long)bh * L);

  //: THE GROUP ROTATION (K33 1.45), on the WRITE digit: two CTAs co-resident on an
  //: SM are one grid wave apart and the wave is a multiple of the warp count, so
  //: `owner % kFbWarps` would rotate both identically.
  const int grp = (warp + ((owner + (owner >> 5)) & (kFbWarps - 1))) & (kFbWarps - 1);

  // ---- the owner's digit runs, and the two carves' ballot rows -------------
  int runCol[D];
  uint32_t rowOff[D];
  const uint32_t rowBytes = (uint32_t)tok_words * (uint32_t)sizeof(uint32_t);
  static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    const int o_l = (owner / BP::owner_div(l)) % BP::g(l);
    runCol[l] = BP::col_base(l) + o_l * BP::s(l);
    //: both carved levels store `kFbWarps` rows per owner run, so the owner's
    //: block starts at `o_l * kFbWarps` in each.
    rowOff[l] = (uint32_t)(CR2::row_prefix(l) + o_l * kFbWarps) * rowBytes;
  });
  const uint32_t* const rowsR =
      pin_address(rows + (long)(2 * bh + 0) * (long)CR2::kRows * (long)tok_words);
  const uint32_t* const rowsW =
      pin_address(rows + (long)(2 * bh + 1) * (long)CR2::kRows * (long)tok_words);

  // ---- lane-invariant operand addressing ----------------------------------
  const int lr = lane & 7, tl = lane >> 3;
  const int rF = (tl & 1) * 8 + lr, cF = tl >> 1;
  const int rT = (tl >> 1) * 8 + lr, cT = tl & 1;
  const uint32_t bwF0 = smem_addr(sw) + (uint32_t)((rF * LdS + cF * 8) * 2);
  const uint32_t bwF1 = smem_addr(sr) + (uint32_t)((rF * LdS + cF * 8) * 2);
  const uint32_t imgA = smem_addr(simg) + (uint32_t)((rF * Ld + cF * 8) * 2);
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
  auto pub_slot = [&](int jt) {
    const int g = 4 * jt + (lane & 3);
    return g * (2 * kOutHi) + 2 * (grp ^ (g & 3));
  };
  auto publish = [&]() {
    static_for<NL>([&](auto Jc) {
      constexpr int jt = decltype(Jc)::value;
      const int slot = pub_slot(jt);
      static_for<MCH>([&](auto Mc) {
        constexpr int m = decltype(Mc)::value;
        static_for<2>([&](auto Hc) {
          constexpr int h = decltype(Hc)::value;
          store_bf16x2(simg + (m * kMmaM + (lane >> 2) + 8 * h) * Ld + slot,
                       pack_bf16x2(st[m][jt][2 * h], st[m][jt][2 * h + 1]));
        });
      });
      //: the mass rides channel row `kDv` and lives in accumulator row 0, lanes 0-3.
      if (lane < 4)
        store_bf16x2(simg + kDv * Ld + slot, pack_bf16x2(mass_st[jt][0], mass_st[jt][1]));
    });
  };

  //: THE READOUT'S A OPERAND: a read group's `kLeafW` slots are CONTIGUOUS under
  //: LAY 3, so it is `MCH` `ldmatrix.x4` out of the image plus two scalar reads for
  //: the mass -- which is ONE row of a sixteen-row tile.
  auto state_operand = [&](int gslot, auto Kc, uint32_t(&sa)[MCH][4], uint32_t(&ma)[4]) {
    constexpr int kk = decltype(Kc)::value;
    const uint32_t base = imgA + (uint32_t)((gslot + kk * kMmaK) * 2);
    static_for<MCH>([&](auto Mc) {
      constexpr int m = decltype(Mc)::value;
      load_frag(sa[m], base + (uint32_t)(m * kMmaM * Ld * 2));
    });
    const __nv_bfloat16* const mrow = simg + kDv * Ld + gslot + kk * kMmaK + (lane & 3) * 2;
    ma[0] = (lane >> 2) == 0 ? *reinterpret_cast<const uint32_t*>(mrow) : 0u;
    ma[1] = 0u;
    ma[2] = (lane >> 2) == 0 ? *reinterpret_cast<const uint32_t*>(mrow + 8) : 0u;
    ma[3] = 0u;
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
  auto produce = [&](__nv_bfloat16* dst, uint32_t tokOff, uint32_t gain) {
    uint32_t acc[SPT / 2];
    {
      unsigned short run[PM::n_l(D - 1)];
      load_run<PM::n_l(D - 1)>(run, gather_ptr(pW, tokOff + (uint32_t)(lvCol[D - 1] * 2)));
#pragma unroll
      for (int j = 0; j < SPT; j += 2)
        acc[j / 2] =
            ((uint32_t)run[PM::idx_of(D - 1, j + 1)] << 16) | (uint32_t)run[PM::idx_of(D - 1, j)];
    }
    static_for<D - 1>([&](auto Lc) {
      constexpr int l = decltype(Lc)::value;
      unsigned short run[PM::n_l(l)];
      load_run<PM::n_l(l)>(run, gather_ptr(pW, tokOff + (uint32_t)(lvCol[l] * 2)));
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
  auto read_pair = [&](int gs, uint32_t tokOff, bool have, uint32_t(&rb)[2]) {
    const uint32_t pair = have ? *reinterpret_cast<const uint32_t*>(
                                     gather_ptr(pR, tokOff + (uint32_t)((runCol[CR] + 2 * gs) * 2)))
                               : 0u;
    const int qb = (lane & 3) ^ (gs & 3);
#pragma unroll
    for (int r = 0; r < 2; ++r)
      rb[r] =
          mul_bf16x2(pair, splat_bf16x2_bits(*reinterpret_cast<const unsigned short*>(
                               gather_ptr(pR, tokOff + (uint32_t)((runCol[0] + 4 * r + qb) * 2)))));
  };

  // ---- V staging -----------------------------------------------------------
  constexpr int kChunkV = kDv / 8;
  constexpr int kRowsPerPass = kFbThreads / kChunkV;
  constexpr int kPasses = C / kRowsPerPass;
  static_assert(kPasses >= 1, "a batch must be at least kFbThreads/8 rows");
  const int ldRow = tid / kChunkV, ldCol = (tid % kChunkV) * 8;
  const uint32_t svL = smem_addr(sv) + (uint32_t)((ldRow * LdV + ldCol) * 2);

  const int nWindows = (L + W - 1) / W;
  //: the image is republished only where the fold moved the state, so a window that
  //: writes nothing costs the readout nothing.  The entry state is a write.
  bool dirty = true;

#pragma unroll 1
  for (int win = 0; win < nWindows; ++win) {
    const int tok0 = win * W;
    const int wLen = (L - tok0) < W ? (L - tok0) : W;
    const uint32_t winP = (uint32_t)tok0 * (uint32_t)(BP::kWtot * 2);
    const uint32_t winG = (uint32_t)tok0 * 2u;
    const uint32_t winV = (uint32_t)tok0 * (uint32_t)v_row_bytes;
    const uint32_t winO = (uint32_t)tok0 * (uint32_t)(kDv * (int)sizeof(float));
    const uint32_t winD = (uint32_t)tok0 * 4u;
    auto produce_batch = [&](__nv_bfloat16* dst, int o0, int ntot) {
      const int nb = (ntot - o0) < C ? (ntot - o0) : C;
      const bool have = prow < nb;
      const int tok = wlist[o0 + (have ? prow : nb - 1)];
      const uint32_t gain = have ? splat_bf16x2(*reinterpret_cast<const __nv_bfloat16*>(
                                       gather_ptr(gW, winG + (uint32_t)tok * 2u)))
                                 : 0u;
      produce(dst, winP + (uint32_t)tok * (uint32_t)(BP::kWtot * 2), gain);
    };

    // ---- the liveness pass ------------------------------------------------
    constexpr int LW = G::kLW;
    static_assert(W <= 0xFFFF, "a token's window-local index must fit the list's element");
    static_assert(LW <= kBallotLanes, "the walk gives one ballot word to each lane");
    const int room = wLen - lane * kBallotLanes;
    const bool inW = lane < LW;
    const uint32_t valid =
        (!inW || room <= 0) ? 0u : (room >= kBallotLanes ? 0xFFFFFFFFu : ((1u << room) - 1u));
    const int wIdx = (tok0 >> 5) + (inW ? lane : 0);
    const uint32_t wordB = (uint32_t)(wIdx < tok_words ? wIdx : tok_words - 1) * 4u;

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
          gather_ptr(rowsS, rowOff[l] + (uint32_t)g * rowBytes + wordB));
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
    uint32_t orW0 = 0u, orW1 = 0u, orR0 = 0u, myW0 = 0u;
    static_for<kFbWarps>([&](auto Gc) {
      constexpr int g = decltype(Gc)::value;
      const uint32_t w0 = row_word(rowsW, 0, g);
      orW0 |= w0;
      myW0 = (g == grp) ? w0 : myW0;
      if constexpr (WSC) orW1 |= row_word(rowsW, CR, g);
      if constexpr (RS0) orR0 |= row_word(rowsR, 0, g);
    });
    //: a READ GROUP's word is the OR of the `kSub` pair-grain rows it spans -- the
    //: table is stored at the FINEST grain either carve asks for, and a coarser
    //: consumer ORs what it already holds.
    auto read_group_row = [&](int G) {
      uint32_t r = 0u;
      for (int t = 0; t < kSub; ++t) r |= row_word(rowsR, CR, G * kSub + t);
      return r;
    };
    const uint32_t clauseW = WSC ? (valid & orW1) : valid;
    const uint32_t clauseR = RS0 ? (valid & orR0) : valid;
    const uint32_t liveW = clauseW & orW0;
    int nrg[kGR], nbrAll = 0;
    static_for<kGR>([&](auto Gc) {
      constexpr int G = decltype(Gc)::value;
      const uint32_t rg = read_group_row(G);
      nrg[G] = lane_broadcast(lane_prefix_incl<LW>(bit_count(clauseR & rg)), LW - 1);
      nbrAll += (nrg[G] + C - 1) / C;
    });
    //: THE WHOLE-WINDOW SKIP, at (owner, window) grain, per side.  The read side's
    //: half IS `nbrAll`, which the counts already produced: a group's count is the
    //: population of its own clause, so their sum is zero exactly when the union is.
    //: It is CTA-uniform without a barrier -- the ballots are the same words in
    //: every warp.
    if (nbrAll == 0 && lane_ballot(liveW != 0u) == 0u) continue;

    //: THE HANDOVER LINE, and it is now the image's ALONE: the read lists and the
    //: value ring no longer share bytes, so this separates the previous window's
    //: FOLD (writing the panels) from this window's PUBLISH (writing the image over
    //: them) and nothing else.
    rendezvous();
    if (dirty) {
      publish();
      dirty = false;
    }
    constexpr uint32_t kBitsW = (uint32_t)(kBallotLanes / kFbWarps);
    const int nw =
        walk(liveW, wlist, ((1u << kBitsW) - 1u) << (warp * kBitsW), (1u << (warp * kBitsW)) - 1u);
    //: the warp's OWN read group's word is recomputed here rather than carried out
    //: of the count pass: `grp` is a runtime index, so a carried copy is a select
    //: chain and one more live register at a bound that has none to spare, while
    //: the row is a runtime ADDRESS and costs nothing to re-read.
    if (nbrAll > 0 && grp < kGR)
      walk(clauseR & read_group_row(grp), srlist + grp * kListStride, 0xFFFFFFFFu, 0u);
    rendezvous();

    //: THE TILE-SKIP PREDICATE.  Bit `T` says the warp's own write digit is live in
    //: k-tile `T` of the owner's write list.  A token's list slot is its rank in
    //: `liveW`, so the tile is that rank's top bits: the walk is over the digit's
    //: OWN live bits, which is `popc` iterations of a word, and the whole-unit
    //: disjunction is one `redux.sync`.  NOTHING about the list, its order or its
    //: tile boundaries moves -- only which tiles this warp executes.
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

    // ---- PHASE 1: the READOUT, against the window's ENTRY state ------------
    //: THE BATCH IS THE UNIT OF WORK AND THE WARP IS ONLY ITS WORKER: batch `bi`
    //: belongs to warp `bi % kFbWarps` whatever group it is drawn from, so a window
    //: whose live support falls in ONE group is drained by every warp.  The carve's
    //: coherence exposure is a property of REGISTER-resident state and it does not
    //: survive an addressable one.
#pragma unroll 1
    for (int bi = warp; bi < nbrAll; bi += kFbWarps) {
      int rem = bi, gsel = 0, osel = 0, nsel = 0;
      static_for<kGR>([&](auto Gc) {
        constexpr int g = decltype(Gc)::value;
        const int nb = (nrg[g] + C - 1) / C;
        const bool mine = rem >= 0 && rem < nb;
        gsel = mine ? g : gsel;
        osel = mine ? rem * C : osel;
        nsel = mine ? nrg[g] : nsel;
        rem -= nb;
      });
      const uint16_t* const rlist = srlist + gsel * kListStride;
      const int gslot = gsel * (kGSR * kOutHi);
      const int n = (nsel - osel) < C ? (nsel - osel) : C;

#pragma unroll
      for (int j = 0; j < C / kMmaN; ++j) {
        const int idx = j * kMmaN + (lane >> 2);
        const bool have = idx < n;
        const uint32_t tokOff =
            winP + (uint32_t)rlist[osel + (have ? idx : 0)] * (uint32_t)(BP::kWtot * 2);
        float acc[MCH][4], macc[4];
#pragma unroll
        for (int m = 0; m < MCH; ++m)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[m][e] = 0.f;
#pragma unroll
        for (int e = 0; e < 4; ++e) macc[e] = 0.f;
        //: THE k-STEP IS A PAIR-GRAIN SUB-GROUP, and the XOR key is ITS index --
        //: which is what LAY 3's slot order permutes the A operand by, absorbed
        //: here at zero cost as a warp-constant XOR on a runtime address.
        //: THE A OPERAND IS LOADED INSIDE THE TOKEN TILE, not hoisted across the
        //: batch's four of them: `KDw x MCH` A fragments is `16 * KDw` live
        //: registers on top of the permanent state slice, and past `KDw = 1` that
        //: is a spill at the 128-register bound the occupancy target sets.  An
        //: `ldmatrix` out of SMEM is the cheap side of that trade.
        static_for<KDw>([&](auto Kc) {
          uint32_t sa[MCH][4], ma[4], rb[2];
          state_operand(gslot, Kc, sa, ma);
          read_pair(gsel * kSub + decltype(Kc)::value, tokOff, have, rb);
#pragma unroll
          for (int m = 0; m < MCH; ++m) mma(acc[m], sa[m], rb);
          mma(macc, ma, rb);
        });
        //: THE FAN-IN IS WARP-PRIVATE, AND IT IS STAGED.  ROUND 4 ITEM 2 BUILT AND
        //: MEASURED THE DIRECT FORM -- `red.global.add.f32` straight out of
        //: `acc[m][e]`, no `STS`, no `LDS`, no `__syncwarp`, and the 17,408 B block
        //: deleted -- AND IT IS REFUTED.  The sector arithmetic is right: the
        //: accumulator already puts eight consecutive channels of one token in the
        //: eight lanes sharing `lane & 3`, so the direct RED touches the same 64
        //: sectors the staged one does.  But it touches them through FOUR cache
        //: lines per instruction instead of one, and that is what binds: measured,
        //: local/global L1 wavefronts 3.01x, MIO throttle 8.7x, l1tex 42.9% -> 63.6%,
        //: and the ALT k16 scattered cell 0.820 -> 0.941.  The staging pass is what
        //: makes the RED lane-per-CHANNEL and therefore one coalesced segment per
        //: issue, and at the entry-densest cell that is worth its bytes.
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
        //: the row loop stays FULLY UNROLLED with a per-row guard.  Two cheaper
        //: shapes were measured in round 4 and both LOSE: rolling it to a liveness
        //: trip count (0.845 -> 0.853 at ALT k4 scattered, because sixteen unrolled
        //: REDs issue back to back and a rolled loop serializes them behind its own
        //: branch), and skipping a wholly dead token tile with a warp-uniform test
        //: before the MMA (1.248 -> 1.266 at ALT k16 cohort=64, because the branch
        //: costs the four tiles their straight-line schedule and the cohort cells
        //: are where dead tiles are common).
#pragma unroll
        for (int r = 0; r < kMmaN; ++r) {
          if (j * kMmaN + r < n) {
            float* const dp = reinterpret_cast<float*>(
                gather_ptr(oPlane, (uint32_t)rlist[osel + j * kMmaN + r] * (uint32_t)(kDv * 4)
                                       + winO + (uint32_t)(lane * (int)sizeof(float))));
            fan_in_add(dp, sow[r * LdO + lane]);
            fan_in_add(dp + kBallotLanes, sow[r * LdO + lane + kBallotLanes]);
          }
        }
        if (lane < kMmaN && j * kMmaN + lane < n)
          fan_in_add(reinterpret_cast<float*>(gather_ptr(
                         denPlane, (uint32_t)rlist[osel + j * kMmaN + lane] * 4u + winD)),
                     sow[lane * LdO + kDv]);
        __syncwarp();
      }
    }

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
    dirty = dirty || nbw > 0;
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
