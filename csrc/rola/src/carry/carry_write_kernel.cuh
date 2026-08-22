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
  //: ONE READ LIST.  The batch is token-carved, so the owner has a single
  //: group-major list instead of one per read group -- 1,024 B where round 5 held
  //: 4,096, and the multiplicity that paid for those lists is gone with them.
  static constexpr int rlist_bytes = kListStride * (int)sizeof(uint16_t);
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
  static_assert(LdS % 8 == 0 && LdV % 8 == 0 && Ld % 8 == 0,
                "an ldmatrix row address must be 16-byte aligned");
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
  //: THE A OPERAND IS LOADED AND CONSUMED ONE m-TILE AT A TIME.  Holding all `MCH`
  //: of them is sixteen live registers on top of the permanent slice and the token
  //: tile's accumulators, which is a spill at the 128-register bound; an `ldmatrix`
  //: out of SMEM is the cheap side of that trade and the loads pipeline anyway.
  auto state_mma = [&](uint32_t imgG, const __nv_bfloat16* mrow, const uint32_t(&rb)[2],
                       float(&acc)[MCH][4], float(&macc)[4]) {
    static_for<MCH>([&](auto Mc) {
      constexpr int m = decltype(Mc)::value;
      uint32_t sa[4];
      load_frag(sa, imgG + (uint32_t)(m * kMmaM * Ld * 2));
      mma(acc[m], sa, rb);
    });
    uint32_t ma[4];
    ma[0] = (lane >> 2) == 0 ? *reinterpret_cast<const uint32_t*>(mrow) : 0u;
    ma[1] = 0u;
    ma[2] = (lane >> 2) == 0 ? *reinterpret_cast<const uint32_t*>(mrow + 8) : 0u;
    ma[3] = 0u;
    mma(macc, ma, rb);
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
  auto read_pair = [&](const __nv_bfloat16* tokp, int colC, int col0, int qb, bool have,
                       uint32_t(&rb)[2]) {
    //: ONE token row pointer, TWO column indices.  Carrying the two displaced
    //: pointers instead costs two more live 64-bit registers across the group loop,
    //: which is a spill on sm_80 at the 128-register bound -- measured.
    const uint32_t pair = have ? *reinterpret_cast<const uint32_t*>(tokp + colC) : 0u;
    const __nv_bfloat16* const a0 = tokp + (col0 + qb);
#pragma unroll
    for (int r = 0; r < 2; ++r)
      rb[r] =
          mul_bf16x2(pair, splat_bf16x2_bits(*reinterpret_cast<const unsigned short*>(a0 + 4 * r)));
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
  //: the image is republished only where the fold moved the state, so a window that
  //: writes nothing costs the readout nothing.  The entry state is a write.
  bool dirty = true;

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
    uint32_t orW0 = 0u, orW1 = 0u, orR0 = 0u, myW0 = 0u;
    static_for<kFbWarps>([&](auto Gc) {
      constexpr int g = decltype(Gc)::value;
      const uint32_t w0 = row_word(rowsW, 0, g);
      orW0 |= w0;
      myW0 = (g == grp) ? w0 : myW0;
      if constexpr (WSC) orW1 |= row_word(rowsW, CR, g);
      if constexpr (RS0) orR0 |= row_word(rowsR, 0, g);
    });
    const uint32_t clauseW = WSC ? (valid & orW1) : valid;
    const uint32_t clauseR = RS0 ? (valid & orR0) : valid;
    const uint32_t liveW = clauseW & orW0;

    //: THE READ LIST IS ONE LIST, BUILT GROUP-MAJOR, AND THAT ORDER IS WHAT KEEPS
    //: THE TILE UNIONS TIGHT.  A token is written ONCE, into the segment of its
    //: FIRST live pair-grain group -- `own = clause & row_G & ~seen`, a running mask
    //: over the `kSigma` rows every warp already holds -- so consecutive list
    //: entries share a leading group and an eight-token tile's union stays near ONE
    //: group where the support is scattered and near the cohort's own run where it
    //: is coherent.  It is an ORDER, not a density branch: the skip below is per
    //: tile and reads the certificate.
    //:
    //: `kSigma == kFbWarps`, so warp `grp` owns segment `grp` outright: it needs
    //: only its own `own` word and the segment's base, and the base is the running
    //: total the same pass produces.  No `[kGR]` count array survives the pass.
    uint32_t mineOwn = 0u, seen = 0u;
    int nr = 0, myBase = 0;
    static_for<kSigma>([&](auto Gc) {
      constexpr int Gi = decltype(Gc)::value;
      const uint32_t rg = row_word(rowsR, CR, Gi);
      const uint32_t own = clauseR & rg & ~seen;
      seen |= rg;
      if (Gi == grp) {
        mineOwn = own;
        myBase = nr;
      }
      nr += lane_broadcast(lane_prefix_incl<LW>(bit_count(own)), LW - 1);
    });
    const int nTiles = (nr + kMmaN - 1) / kMmaN;
    //: THE WHOLE-WINDOW SKIP, at (owner, window) grain, per side.  The read side's
    //: half IS `nr`: every live read token appears in exactly one segment, so the
    //: running total is zero exactly when the owner reads nothing.  CTA-uniform
    //: without a barrier -- the ballots are the same words in every warp.
    if (nr == 0 && lane_ballot(liveW != 0u) == 0u) continue;

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
    //: the warp scatters its own segment at the base the count pass gave it, so the
    //: `kSigma` segments concatenate into ONE group-major list with no CTA prefix.
    if (nr > 0) walk(mineOwn, srlist + myBase, 0xFFFFFFFFu, 0u);
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
    //: THE UNIT OF ASSIGNMENT IS THE mma n-TILE, NOT THE BATCH.  The readout reads a
    //: CTA-SHARED image, so eight tokens is a legal unit: no token is split, its
    //: partial sums across the image's k-tiles stay in ONE warp and it is fanned in
    //: ONCE, so multiplicity stays one and the per-tile union skip is unchanged --
    //: and the unit count is FOUR TIMES the batch count.  That is the whole of round
    //: 6's defect: a `C`-entry batch gave the eight warps 1.95 units of work at ALT
    //: k16 cohort=64 and 1.22 at k4 cohort=64, so six of eight warps waited at the
    //: window line.  At n-tile grain the same window offers 7.8 and 4.9.
#pragma unroll 1
    for (int tile = warp; tile < nTiles; tile += kFbWarps) {
      {
        const int idx = tile * kMmaN + (lane >> 2);
        const bool have = idx < nr;
        const int tok = srlist[have ? idx : 0];
        const __nv_bfloat16* const tokp = pRw + tok * BP::kWtot;
        //: THE SKIP, AND IT IS THE WRITE SIDE'S MIRROR.  A token tile folds the
        //: UNION of its eight tokens' live pair-grain groups, read from the
        //: CERTIFICATE: a lane tests its own token's bit in each of the `kSigma`
        //: rows and one `redux.sync.or` makes the union warp-wide.  A group in which
        //: all eight tokens are dead contributes an exact zero at every channel, so
        //: dropping its k-tile is an IDENTITY, not an approximation -- the same
        //: argument the fold's tile skip rests on.  THE READOUT HAS NO REPLAY
        //: CONTRACT (the reverse family replays the FOLD, and `o`/`den` are
        //: global-atomic fan-ins that are not bit-reproducible against themselves),
        //: so the identity here is a correctness statement rather than a gate --
        //: and it is still exact.
        //: THE UNION COSTS TWO ROW LOADS PER LANE, NOT EIGHT.  A token is served by
        //: FOUR lanes, so lane `l` tests only pair-grain groups `2*(l&3)` and `+1` of
        //: ITS token; the whole-unit `redux.sync.or` then combines the four lanes'
        //: halves AND the tile's eight tokens in one instruction, because an OR does
        //: not care which lane a bit came from.  Loading all eight rows in every lane
        //: is the same union at four times the cost.
        uint32_t mine = 0u;
        {
          const int gpair = (lane & 3) * 2;
          const uint32_t roff =
              (uint32_t)(rowIdx[CR] + gpair) * rowBytes + (uint32_t)((tok0 >> 5) + (tok >> 5)) * 4u;
          const int bit = tok & 31;
          const uint32_t r0 = *reinterpret_cast<const uint32_t*>(gather_ptr(rowsR, roff));
          const uint32_t r1 =
              *reinterpret_cast<const uint32_t*>(gather_ptr(rowsR, roff + rowBytes));
          mine = have ? (((((r1 >> bit) & 1u) << 1) | ((r0 >> bit) & 1u)) << gpair) : 0u;
        }
        const uint32_t uni = lane_reduce_or(mine);

        float acc[MCH][4], macc[4];
#pragma unroll
        for (int m = 0; m < MCH; ++m)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[m][e] = 0.f;
#pragma unroll
        for (int e = 0; e < 4; ++e) macc[e] = 0.f;
        //: THE WHOLE IMAGE, ONE k-TILE PER PAIR-GRAIN GROUP -- WALKED OVER THE
        //: UNION'S SET BITS, not unrolled over all `kSigma` with a predicate.  The
        //: trip count is then the tile's LIVENESS and a dead group costs nothing at
        //: all, where eight unrolled bodies cost eight basic blocks and the live
        //: ranges that come with them (measured: an eight-byte spill at the
        //: 128-register bound, and no spill is admitted).  The price is four ALU
        //: operations of runtime addressing per live group -- the image offset, the
        //: mass row, the level-CR column and LAY 3's XOR key are all shifts of `Gi`
        //: -- against a `gather_ptr`-free token row that is already formed.
#pragma unroll 1
        for (uint32_t u = uni; u != 0u; u &= u - 1u) {
          const int Gi = first_set(u);
          uint32_t rb[2];
          read_pair(tokp, runCol[CR] + 2 * Gi, runCol[0], (lane & 3) ^ (Gi & 3), have, rb);
          state_mma(imgA + (uint32_t)(Gi * (kMmaK * 2)),
                    simg + kDv * Ld + Gi * kMmaK + (lane & 3) * 2, rb, acc, macc);
        }
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
          if (tile * kMmaN + r < nr) {
            //: `oWin` already carries the window's origin AND the lane's channel, so
            //: the token is one `IMAD.WIDE` and the second RED is `+128` immediate.
            float* const dp = oWin + (int)srlist[tile * kMmaN + r] * kDv;
            fan_in_add(dp, sow[r * LdO + lane]);
            fan_in_add(dp + kBallotLanes, sow[r * LdO + lane + kBallotLanes]);
          }
        }
        if (lane < kMmaN && tile * kMmaN + lane < nr)
          fan_in_add(dWin + (int)srlist[tile * kMmaN + lane], sow[lane * LdO + kDv]);
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
