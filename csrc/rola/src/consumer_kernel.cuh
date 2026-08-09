#pragma once

// THE RoLA V3 CONSUMER KERNEL -- THE ENDGAME EXECUTOR. It is the ONLY consumer, and
// this file is its BODY: the membership scan, the group assembler, the
// compact-into-registers atom walk, the four GEMMs in their phase order and the
// split-K epilogue. `ConsumerParams` and the host dispatch are `consumer.{cuh,cu}`;
// the supported shape surface is declared at `launch_consumer` and is deliberately
// NOT restated here.
//
// THE EXECUTOR'S SHAPE (P14 endgame form -- the schedule IS the tiling):
//
//   * MEMBERSHIP = STREAMED FINE SCAN. For each 32-token word of the segment's
//     range: AND the owner's fixed-level token-bit rows, AND the token-granular
//     slice-union bit row. There is no tile-candidacy question and no coarse
//     pre-filter (intersections do not survive coarsening -- AND-consumers read
//     fine); the scan streams `(j + 1)` words per 32 tokens, static addresses.
//   * GROUPS ARE ALWAYS FULL: members ballot into groups of `kGroupTokens == 16`
//     (the MMA quantum), stable ascending -- ORDER IS THE CAUSALITY REQUIREMENT.
//     The only ragged group is each segment's final flush. Visit counts are exact:
//     `ceil(gathered / 16)`. Dense inputs self-heal to IDENTITY GROUPING (all-ones
//     membership -> groups are consecutive 16-token blocks).
//   * THE ATOM WALK: the per-token atom-granule union bytes (schedule form (b))
//     are OR'd over the group's <= 16 members at ballot time into per-side ATOM
//     MASKS; every leaf-axis consumer (read-panel build, readout, Gram, mass
//     column, write-panel build, fold) walks only LIVE atoms. `mask == FULL`
//     selects the statically-unrolled dense arm -- dense pays no mask
//     instructions. Skipping is EXACT, not approximate: a dead atom's
//     coefficients are exact zeros (self-masking), and under mass decay a dead
//     atom's leaves have a zero clock prefix, hence survival factor s0 == 1 --
//     TICKS RIDE WRITES, which is the architecture fact that makes the endgame
//     legal under decay.
//   * WHOLE-COVERED-SPAN FETCH: a member token's covered amplitude span is read
//     by position (no support inspection anywhere below the ballot); zeros ride
//     the self-masking theorem.
//   * THE AMPLITUDE STREAMS ARE PREFETCHED. Both panels' coefficients come out of a
//     SHARED-MEMORY SLAB filled by `cp.async` from inside the PREVIOUS window's
//     visit, so the gather's DRAM latency leaves the visit's critical path entirely
//     and is not hidden by co-residency. A chunk no live atom reads is issued with
//     src-size zero -- the hardware zero-fills it, which is exactly the value a dead
//     atom's coefficient must take. Zero new barriers in the steady state: the issue
//     rides the staging barrier and the drain rides the end-of-window barrier.
//
// THE ONE PRECISION STORY: bf16 storage, fp32 accumulation, fp64 reference as
// oracle. The staged operands, amplitude streams and value stream are bf16; the
// state, accumulators, clock, row scales AND the decay panel application are
// fp32. (U07-d's fp16 application was built and CARVED on measurement: an
// admitted hot routing put the survival factor past fp16's 65504 range --
// measured NaN, not rounding. bf16 amplitude STORAGE on the decay arms
// survives the carve; the register cost is the ratchet's to adjudicate.)
//
// Three facts a reader must not miss:
//   1. THE STATE LIVES IN REGISTERS, TRANSPOSED, AS THE FOLD'S OWN D FRAGMENTS.
//      `mma.sync`'s D map and its A map are the same map, so holding the state
//      `[channel][leaf]` makes the fold's accumulator the readout's A operand
//      for free: the fold's `mma.sync` accumulates IN PLACE (`D == C == S^T`)
//      and the readout consumes the result with no load at all. THE CARVE `GL x GC`
//      gives each warp a contiguous ATOM RANGE of leaves and a contiguous
//      CHANNEL-TILE RANGE, which is what makes the writer and the reader of an
//      element the same warp at the same lane; `GC` sets how many independent
//      accumulators a warp issues into and `GL` sets its dependent chain length.
//      Only the MASS COLUMN keeps a shared-memory home -- fp32, `[BC]` -- because
//      it is genuinely shared between warps. Accumulation is fp32 throughout; the
//      one narrowing is the state's bf16 A-operand fragment at the readout.
//   2. THE ONE-COPY INVARIANT IS A PROPERTY OF THE LAUNCH SHAPE: one CTA per
//      owner, run to completion, one HBM read and one HBM write of the owner's
//      state block. There is no wave machinery anywhere.
//   3. EVERY STRUCTURAL AXIS IS A TEMPLATE PARAMETER and no axis has a runtime
//      fallback -- one comprehensive path ships. The endgame walk IS the path;
//      there is no aligned walk (dense is its degeneracy, by data).
//
// The order of operations, the symbol glossary, the bank-conflict proofs, the
// deadlock argument and the hazard index: docs/internals/consumer_kernel.md

#include <cuda_bf16.h>

#include "arch_table.cuh"
#include "cp_async.cuh"
#include "intra_panel.cuh"
#include "consumer.cuh"
//: The two layers, in one include: `mma_tile.cuh` -> `smem_layout.cuh`.
//: docs/internals/mma_tile.md#cute-pin
#include "mma_tile.cuh"
#include "readout_mma.cuh"

namespace rola {
namespace consumer {

//: §12 rules 1 and 3's survival factors, taken from `intra_panel.cuh` rather than
//: reimplemented: it is the one clamp point and the one exponent sign convention.
using rola::panel_read_factor;
using rola::panel_write_factor;

//: §4.5's PAGED ADDRESS, RE-KEYED TO THE ATOM. A page is ONE `MMA_K_QUANTUM`-leaf
//: ATOM, not one owner, so an owner spans `QPO` INDEPENDENTLY PAGED atoms whose slots
//: are not contiguous in general and a CTA resolves `QPO` bases at startup. Returns
//: the first STATE ROW of atom `q` of `owner`, or `-1` when the atom has no committed
//: slot -- which under total admission means no write candidate at all, hence a dense
//: state that is identically zero. ONE EXPRESSION SERVES BOTH LAYOUTS: unpaged is the
//: identity slot map. docs/internals/consumer_kernel.md#atom-row-base
template <int BC>
__device__ __forceinline__ int64_t atom_row_base(const ConsumerParams& p, int bh, int owner,
                                                 int q) {
  constexpr int kAtomsPerOwner = BC / kMmaKQuantum;
  if (p.page_table != nullptr) {
    //: DERIVED rather than carried: the host sets `p.N = owners_total * BC` under
    //: paging, so there is no second place for the table's stride to be wrong.
    const int atoms_per_bh = static_cast<int>(p.N / kMmaKQuantum);
    const int64_t entry =
        static_cast<int64_t>(bh) * atoms_per_bh + static_cast<int64_t>(owner) * kAtomsPerOwner + q;
    const int physical = p.page_table[entry];
    return physical >= 0 ? static_cast<int64_t>(physical) * kMmaKQuantum : static_cast<int64_t>(-1);
  }
  return static_cast<int64_t>(bh) * p.N + static_cast<int64_t>(owner) * BC
         + static_cast<int64_t>(q) * kMmaKQuantum;
}

#define CONSUMER_D_ROWS(lane) const int _g = (lane) >> 2, _r = (lane) & 3

//: **THE FORCING-CENSUS INSTRUMENT, NEVER A SHIPPING KNOB.** When defined it
//: replaces the DERIVED launch bound with a fixed CTA count on every instantiation,
//: so a candidate register budget can be censused over the whole dispatch matrix in
//: one probe compile. `tools/ratify.py --census` is its only legitimate definer and
//: a manifest is only ever ratified off the UNFORCED compile:
//: docs/internals/consumer_kernel.md#force-ctas
#ifdef ROLA_FORCE_CTAS
#define ROLA_CONSUMER_CTAS_EXPR (ROLA_FORCE_CTAS)
#else
#define ROLA_CONSUMER_CTAS_EXPR                               \
  ctas_for_residency(consumer_smem_bytes<BC, DV, D, DECAY>(),  \
                     consumer_reg_budget(BC, DECAY, DV, D), consumer_threads(GL, GC))
#endif
//: `GL` x `GC` IS THE CARVE -- leaf groups by channel groups, the grid the CTA's warps
//: are laid out on and the kernel's third structural axis. It is a TEMPLATE PARAMETER
//: like every other axis: nothing below assumes a corner of the space (a warp owning
//: every channel, or every leaf). `launch_consumer_arm` instantiates the policy's
//: point; a probe compile instantiates any other. docs/internals/consumer_kernel.md#carve
template <int BC, int DV, int D, bool DECAY, int GL, int GC>
__global__ __launch_bounds__(consumer_threads(GL, GC), ROLA_CONSUMER_CTAS_EXPR) void
consumer_forward_kernel(ConsumerParams p) {
  // ===========================================================================
  // GUTTED FOR STUDY (learn/consumer-skeleton). This is the body of the ONE
  // consumer kernel, as a staged specification instead of an implementation.
  // Every helper above this function, every template/function parameter, and
  // the surrounding machinery (host dispatch, oracle, launch code) are all
  // present and untouched -- only the executable body below is gone. Real
  // implementation: `git diff a7cb0a7 -- csrc/rola/src/consumer_kernel.cuh`.
  // ===========================================================================

  // ---------------------------------------------------------------------------
  // ORIENTATION
  // ---------------------------------------------------------------------------
  //
  // THE RECURRENCE. RoLA's per-leaf state update is a gated linear-attention
  // recurrence: each leaf carries a state that is written by tokens routed to
  // it and read back by later tokens through the same routing. Because the
  // recurrence is INPUT-AFFINE (the update at step u is S <- gate(S) + (rank-1
  // deposit from token u), never S-dependent in a nonlinear way), the readout at
  // step t telescopes into a closed form:
  //
  //     y_t = R_t . S_entry + sum_{u <= t} (R_t . W_u) . Vw_u
  //
  //   R_t   : token t's read-routing row over the owner's BC leaves.
  //   W_u   : token u's write-routing row over the same BC leaves.
  //   Vw_u  : token u's value vector, already carrying its write-gate scale.
  //   S_entry: the state as of the start of the current chunk -- see the
  //            invariant below.
  //
  // Telescoping is what turns an O(T) sequential scan into O(1) state reads
  // plus a small dense correction: everything before the current chunk is
  // already folded into S_entry, and only within-chunk causality needs an
  // explicit u <= t sum.
  //
  // WHAT A CTA OWNS. One CTA is permanently bound to one (owner, segment) work
  // item for its whole life: `owner` names a contiguous block of BC leaves (a
  // node of the routing tree at a fixed depth), and the CTA's job is to produce
  // every token's readout against that block and to fold every token's write
  // into that block's running state. The CTA's warps are laid out on a
  // `GL x GC` grid -- GL leaf groups by GC channel groups -- so warp (lgrp,
  // cgrp) owns a contiguous ATOM RANGE of the owner's leaves (an "atom" is one
  // MMA-k-quantum-sized bundle of leaves) crossed with a contiguous
  // CHANNEL-TILE RANGE of the value/output dimension. Every stage below that
  // touches "the state" or "the leaf axis" is implicitly scoped to a warp's own
  // (leaf range, channel range) tuple; nothing here assumes a corner of that
  // space (one warp owning every leaf, or every channel).
  //
  // THE THREE GEMMS.
  //   readout  (m = channels, n = tokens,        k = leaves) : y = S^T . R
  //   fold     (m = leaves,   n = channels,       k = tokens) : S += W^T . Vw
  //   intra    (m = channels, n = tokens (a),     k = tokens (b), via a small
  //             leaf-free Gram G[b][a] = W_b . R_a) : y += Vw^T . G
  // The Gram itself is a fourth, cheap GEMM (leaves contracted out): G[b][a] =
  // sum_leaves W_b[leaf] * R_a[leaf], masked to token pairs with b <= a.
  //
  // THE INVARIANT THAT MAKES IT CORRECT. Tokens are processed in ordered
  // groups ("chunks" below; the code's unit is `kGroupTokens` tokens per visit
  // of the main loop). WITHIN one chunk, S is treated as FROZEN at its value
  // from chunk entry (S_entry) for the whole chunk's readout GEMM -- none of
  // the chunk's own writes are visible to that GEMM. Causality among the
  // chunk's own tokens (a write at u <= a read at t, both inside the same
  // chunk) is instead carried by the intra/Gram term. Only AFTER the whole
  // chunk's writes are folded does S advance to become the next chunk's
  // S_entry. Get this backwards -- let a chunk's own writes leak into its own
  // readout GEMM, or let the Gram cover pairs outside b <= a -- and every
  // downstream token sees a token that comes after it.
  //
  // The stages below are grouped into SETUP (once per CTA), the WALK (once per
  // chunk, looped until the owner's token range is exhausted), and EXIT (once
  // per CTA). Each stage names its job, its inputs, what it must leave behind,
  // and the ordering/visibility contract another stage relies on. Performance
  // machinery that is not part of the mathematical contract -- how far ahead
  // work is staged, what lives in registers vs. shared memory, bit-tricks for
  // finding live lanes -- is deliberately not specified; build the simplest
  // thing that satisfies each contract first.

  // ---------------------------------------------------------------------------
  // SETUP -- once per CTA, before any token is processed.
  // ---------------------------------------------------------------------------

  // STAGE S1: WORK-ITEM RESOLUTION.
  //   WHAT: decide which owner (leaf block) and which token range of that
  //     owner this CTA is responsible for. A single owner's whole token range
  //     may be split across several CTAs ("segments") when it is long; each
  //     segment is an independent CTA that reads/writes a shared intermediate
  //     stash instead of the owner's real state block.
  //   INPUTS: `p.owners`, `p.work_seg`/`p.work_nseg`/`p.work_slot` (if the
  //     launch used segmentation), `blockIdx.x`.
  //   OUTPUTS: `bh` (which batch*head), `owner` (which leaf block), the
  //     token-word range `[w_lo, w_hi)` this CTA scans, and whether this
  //     segment is the owner's last one.
  //   CONTRACT: every token in the owner's full range is claimed by exactly
  //     one CTA/segment, segments partition the range without gaps or overlap,
  //     and "last segment" is knowable without communicating with any other
  //     CTA (it is a property of this CTA's own segment index).
  //   CANDIDATE PRIMITIVES:
  //     - `blockIdx.x` / grid-stride index math: the raw work-item index this
  //       stage decodes.
  //     - integer div/mod (or a precomputed reciprocal, `cutlass::FastDivmod`
  //       (verify) if pulled in) to turn a flat CTA index into (bh, owner,
  //       segment) coordinates cheaply.
  //     - `__ldg` / `const __restrict__` pointers for the read-only
  //       `p.owners`/`p.work_seg` lookup tables.

  // STAGE S2: OWNER GEOMETRY.
  //   WHAT: decode each of the owner's BC leaves into its address at every
  //     routing level (which digit it occupies at level 0, level 1, ...). This
  //     is the address rectangle the routing tree assigns to `owner`.
  //   INPUTS: `owner`, the topology's level widths/strides.
  //   OUTPUTS: a per-(level, leaf) digit table, usable by every later stage
  //     that needs to index a level's amplitude/decay tensors by leaf.
  //   CONTRACT: computed once per CTA (an owner-level fact, identical for
  //     every token and every chunk), not recomputed per token or per chunk.
  //   CANDIDATE PRIMITIVES:
  //     - plain integer div/mod against the topology's per-level widths
  //       (constants known at compile time for a fixed template
  //       instantiation, so the compiler can strength-reduce them).
  //     - a small register/shared array indexed by level, filled once by a
  //       handful of threads or one thread and broadcast with `__shfl_sync`.
  //     - `constexpr`/`static_assert` topology checks (as `arch_table.cuh`
  //       already does) to keep the address decode a compile-time fact where
  //       possible.

  // STAGE S3: DECAY-RATE LOOKUP (only when `DECAY` is set).
  //   WHAT: for each of the owner's BC leaves, look up its per-step retention
  //     rate from the routing dials and convert it to a log-space quantity
  //     (log(1 - rate)) suitable for accumulating across many steps without
  //     repeated multiplication underflow.
  //   INPUTS: the per-leaf digit table (S2), the dial tensor.
  //   OUTPUTS: one log-retention scalar per leaf, live for the whole CTA.
  //   CONTRACT: an owner-level fact (leaves determine the rate, tokens do
  //     not), computed once and reused by every chunk's decay math (V1 below).
  //   CANDIDATE PRIMITIVES:
  //     - `__logf` / `log1pf` (fast, approximate transcendentals -- the
  //       accumulation is fp32 per the file header, so a fast intrinsic is in
  //       scope where precision allows) for the `log(1 - rate)` conversion.
  //     - `__ldg` for the read-only dial-tensor fetch.
  //     - vectorized loads (`float2`/`float4`) if several leaves' dials are
  //       laid out contiguously.

  // STAGE S4: ENTRY STATE LOAD.
  //   WHAT: bring the owner's current state S_entry -- and its accompanying
  //     normalizer ("mass") column -- onto the CTA, from wherever it lives:
  //     the owner's persistent HBM state block on a first/only segment, an
  //     intermediate per-segment stash on a later segment, or all-zeros if
  //     this pass is building that stash for the first time.
  //   INPUTS: `p.state` (or the segment stash), the owner/atom addressing
  //     from S1/S2.
  //   OUTPUTS: S_entry resident on-chip for the CTA's whole life; the entry
  //     mass column likewise.
  //   CONTRACT: exactly ONE read of the owner's state for the whole CTA (the
  //     one-copy invariant) -- this is the only place the state is fetched
  //     from off-chip. Every later stage mutates the on-chip copy in place;
  //     nothing re-reads from HBM until the exit stages.
  //   CANDIDATE PRIMITIVES:
  //     - `cp.async` (`cuda_pipeline_primitives.h`'s `__pipeline_memcpy_async`,
  //       or this file's own `cp_async_16`/`cp_async_commit`/
  //       `cp_async_wait_all` wrappers in `cp_async.cuh`) for the
  //       global-to-shared bulk copy of the state block, so the fetch latency
  //       can overlap with other setup work.
  //     - `ldmatrix` (via `mma_tile.cuh`'s existing fetchers, or `nvcuda::wmma
  //       ::load_matrix_sync` (verify availability/fit)) once the state is
  //       shared-memory resident and needs to reach registers as an MMA
  //       operand.
  //     - plain vectorized `float4`/`int4` loads for the mass column, which
  //       the file header says stays a small fp32 `[BC]` shared array rather
  //       than going through the tensor-core path.
  //     - `cuda::pipeline` (`<cuda/pipeline>`, cooperative-groups-based; a
  //       heavier-weight alternative to the raw `cp.async` wrappers -- verify
  //       it is worth the abstraction over the hand-rolled primitives already
  //       in `cp_async.cuh`).

  // STAGE S5: MEMBERSHIP / SCHEDULING.
  //   WHAT: walk the CTA's token-word range and decide, for each token,
  //     whether it reads this owner, writes this owner, or both -- then
  //     collect the resulting member tokens, IN ASCENDING ORDER, into
  //     fixed-size chunks of `kGroupTokens` tokens each (the last chunk of the
  //     range may be short). For each chunk, also determine which of the
  //     owner's atoms (leaf bundles) are actually touched on the read side and
  //     on the write side -- a chunk need not touch every atom.
  //   INPUTS: the per-token, per-level read/write membership bits and the
  //     per-atom bitlists that make membership testable at atom granularity.
  //   OUTPUTS: an ordered sequence of chunk descriptors -- each one a token
  //     list (size <= kGroupTokens), a read-atom mask, and a write-atom mask
  //     -- that the walk below consumes one at a time.
  //   CONTRACT: chunk order must be globally ascending in token index --
  //     ORDER IS THE CAUSALITY REQUIREMENT the intra term (V7) and the fold's
  //     accumulation order (V8) both depend on. A chunk's atom masks must be
  //     exact: an atom absent from a chunk's mask must genuinely contribute
  //     nothing to that chunk's coefficients (so later stages may skip it
  //     without changing the result). This is a pure bookkeeping computation
  //     with no dependency on any other stage below it; it may legally run
  //     entirely up front, or be interleaved/overlapped with the walk for
  //     performance -- that overlap is a performance choice, not something the
  //     result depends on.
  //   CANDIDATE PRIMITIVES:
  //     - `__ballot_sync` to turn a warp's 32 per-token membership predicates
  //       into one bitmask in a single instruction.
  //     - `__popc` / `__popcll` to count set bits (chunk sizes, live-atom
  //       counts) and `__ffs` / `__ffsll` to walk a mask's set bits one at a
  //       time in ascending order.
  //     - `cub::WarpScan`/`cub::BlockScan` (`<cub/warp/warp_scan.cuh>`,
  //       `<cub/block/block_scan.cuh>` -- both already used elsewhere in this
  //       project, `entmax.cu` and `decode_expand.cu` respectively) for a
  //       prefix sum turning a membership mask into compacted destination
  //       slots.
  //     - bitwise AND/OR/XOR on the per-level bit rows the WHAT already
  //       describes (`&`, `|`, `^` on `unsigned`/`unsigned long long`), which
  //       is the actual membership test before any of the above.

  // STAGE S6: GATHER.
  //   WHAT: for each chunk's member tokens, fetch the raw per-level amplitude
  //     coefficients (both read-side and write-side), the decay ingredients
  //     needed to build this chunk's survival factors, and the value vectors
  //     of the write-member tokens.
  //   INPUTS: the chunk descriptors from S5, the per-level amplitude tensors,
  //     the value tensor.
  //   OUTPUTS: the raw per-token, per-level data each chunk's panel-build
  //     stages (V3/V4/V5) need, staged somewhere on-chip.
  //   CONTRACT: a chunk's gathered data must be visible before that chunk's
  //     panel-build stages run. Whether this gather happens synchronously
  //     right before it's needed, or is issued ahead of time and overlapped
  //     with earlier chunks' compute, is a PERFORMANCE CHOICE ONLY -- start
  //     with the synchronous form; only the "visible before use" contract is
  //     required for correctness. A leaf/level not touched by a chunk (dead in
  //     both S5 masks) contributes an exact zero coefficient, never garbage --
  //     whatever is not fetched must be treated as zero downstream, not
  //     skipped in a way that leaves it undefined.
  //   CANDIDATE PRIMITIVES:
  //     - `cp_async_16`/`cp_async_4` (this file's own `cp_async.cuh`
  //       wrappers around `cp.async.cg`/`cp.async.ca`) or the raw
  //       `__pipeline_memcpy_async` from `cuda_pipeline_primitives.h`, for
  //       the async global-to-shared fetch of amplitude/value data, with the
  //       predicated src-size-zero trick `cp_async.cuh`'s header comment
  //       already documents for a dead lane.
  //     - `cp_async_commit`/`cp_async_wait_all`/`cp_async_wait_group` (or
  //       `__pipeline_commit`/`__pipeline_wait_prior`) to manage when a
  //       staged batch becomes visible.
  //     - vectorized `float4`/`__nv_bfloat162` loads for the value vectors if
  //       the synchronous/non-`cp.async` path is taken first.
  //     - `__ldg` for scalar read-only amplitude fetches.

  // ---------------------------------------------------------------------------
  // THE WALK -- one iteration per chunk, in the order S5 produced them, until
  // the owner's token range is exhausted (a chunk descriptor with zero members
  // is the end-of-stream signal).
  // ---------------------------------------------------------------------------

  // STAGE V1: DECAY CLOCK AND SURVIVAL FACTORS (only when `DECAY` and this
  //   chunk has any write-member tokens).
  //   WHAT: for each leaf, walk this chunk's member tokens IN ORDER and build
  //     a running "clock" -- the cumulative decay exposure up to and including
  //     each token. From the clock, derive: a per-leaf reference point for the
  //     chunk (so the read-side and write-side factors below can be expressed
  //     relative to a common frame rather than an absolute step count), and a
  //     per-leaf "entry scale" s0 that brings S_entry into that chunk-local
  //     reference frame.
  //   INPUTS: the per-leaf log-retention (S3), the per-token decay ingredients
  //     gathered in S6.
  //   OUTPUTS: per-leaf clock values (used per-token by V3/V4 to build each
  //     token's read/write decay factor), the reference point, and s0.
  //   CONTRACT: must complete, and its results (s0 in particular) must be
  //     applied to S_entry and the entry mass column, BEFORE any GEMM in this
  //     chunk reads the state (V6, V8) -- s0 rescales the entry values into
  //     the frame this chunk's factors are expressed in. An atom with no live
  //     leaf this chunk must end up with s0 == 1 (identity), so that leaving
  //     it untouched is exactly the correct answer, not an approximation of
  //     it. This stage does not run for a chunk that writes nothing (an
  //     entirely-read chunk needs no fresh entry scale).
  //   CANDIDATE PRIMITIVES:
  //     - `__expf` (fast, approximate exponential) to turn a log-space clock
  //       back into a survival factor, and `__logf`/`log1pf` if any further
  //       log-domain accumulation is needed within the chunk.
  //     - `__shfl_sync`/`__shfl_up_sync`/`__shfl_xor_sync` for a warp-local
  //       running sum/scan across the (small, <= 16-token) chunk when the
  //       accumulation is done lane-per-token within one warp.
  //     - `cub::WarpScan` (`<cub/warp/warp_scan.cuh>`, already used in
  //       `entmax.cu`) as a packaged inclusive/exclusive-scan alternative to
  //       hand-rolled shuffle butterfly steps.
  //     - `fmaf` for the per-step clock update if it is a plain
  //       multiply-accumulate rather than sum-then-exponentiate.

  // STAGE V2: ROW-SCALE REDUCTION.
  //   WHAT: for each member token, multiply together whatever per-token
  //     amplitude factors are the SAME across every leaf of this owner (levels
  //     "above" the owner's own address rectangle contribute one scalar per
  //     token, not a per-leaf value) into one read-gate scalar and one
  //     write-gate scalar per token.
  //   INPUTS: the gathered per-level scalars (S6) for the levels that don't
  //     vary across this owner's leaves.
  //   OUTPUTS: per-token read-scale and write-scale scalars for this chunk.
  //   CONTRACT: computed once per chunk per token, not once per (leaf, token)
  //     pair -- these values are leaf-invariant by construction, so folding
  //     them into the per-leaf panels below would be redundant work, not
  //     merely slow. A token absent from a chunk's live range takes an exact
  //     zero scale, which is what makes its row inert through every GEMM that
  //     follows without a separate branch.
  //   CANDIDATE PRIMITIVES:
  //     - `fmaf`/plain `*` across a short (compile-time-bounded) unrolled
  //       loop of per-level scalars -- this is a small per-token product, not
  //       a reduction across lanes, so it is register-local arithmetic.
  //     - `__shfl_sync` if a token's per-level factors are scattered across
  //       lanes and need to be gathered onto the lane that owns that token
  //       before multiplying.
  //     - `#pragma unroll` on the fixed-depth level loop (the level count is
  //       a compile-time topology constant per S2).

  // STAGE V3: READ PANEL BUILD.
  //   WHAT: for every (leaf, member token) pair, form that token's read-routing
  //     coefficient at that leaf -- the product of the token's per-level
  //     routed amplitudes at the leaf's digits, times the read-side decay
  //     factor from V1 if applicable. This materializes R_t (restricted to
  //     this chunk's tokens and this warp's leaf range).
  //   INPUTS: the gathered amplitude coefficients (S6), the owner's digit
  //     table (S2), the decay clock/reference (V1).
  //   OUTPUTS: the chunk's read panel -- one coefficient per (leaf, token) --
  //     consumed by the readout GEMM (V6) and as one operand of the Gram
  //     (V7).
  //   CONTRACT: a leaf with no live read atom, or a token past the chunk's
  //     real member count (a short trailing chunk), must produce an exact
  //     zero, not merely a small value -- downstream stages rely on exact
  //     zero to skip work safely. Skipped only if this chunk has no read
  //     members at all.
  //   CANDIDATE PRIMITIVES:
  //     - `__float2bfloat16`/`__floats2bfloat162_rn` (this file's
  //       `mma.cuh::pack_bf16x2` already wraps the latter) to narrow the
  //       fp32-built coefficient into the bf16 storage the file header says
  //       the staged operands live in.
  //     - `ldmatrix` (through `mma_tile.cuh`'s existing fetchers) once the
  //       panel is shared-memory resident and an `mma.sync` needs it as an
  //       operand fragment.
  //     - `__stcs`/`__stwt` (streaming/write-through store hints, `<cuda_
  //       runtime.h>`'s cache-hint intrinsics) if the panel is written to
  //       shared memory in a pattern where the normal cache policy is wrong
  //       (verify whether shared-memory writes even take these hints, vs.
  //       global only).

  // STAGE V4: WRITE PANEL BUILD.
  //   WHAT: the write-side mirror of V3: for every (leaf, member token) pair,
  //     form the token's write-routing coefficient at that leaf, times the
  //     write-side decay factor from V1 if applicable. This materializes W_u.
  //   INPUTS: same as V3, write side.
  //   OUTPUTS: the chunk's write panel, consumed by the fold GEMM (V8) and as
  //     the other operand of the Gram (V7).
  //   CONTRACT: same exact-zero requirement as V3. Skipped if this chunk has
  //     no write members.
  //   CANDIDATE PRIMITIVES: the write-side mirror of V3's list --
  //     `__float2bfloat16`/`__floats2bfloat162_rn` for the narrowing,
  //     `ldmatrix` for the eventual operand fetch, `#pragma unroll` on the
  //     level loop.

  // STAGE V5: VALUE OPERAND PREP.
  //   WHAT: take the write-member tokens' value vectors (gathered in S6) and
  //     apply the per-token write-gate scale from V2, producing Vw_u.
  //   INPUTS: gathered value vectors (S6), write-scale (V2).
  //   OUTPUTS: the chunk's scaled value operand, used identically by the fold
  //     GEMM (V8) and the intra GEMM (V9) -- it is the same Vw_u in both
  //     formulas.
  //   CONTRACT: the scale must be applied before either consuming GEMM runs;
  //     it may be applied to the raw fetched value at any point up to their
  //     use, on any operand representation -- the requirement is only that
  //     the value each GEMM contracts against already carries the write gate.
  //   CANDIDATE PRIMITIVES:
  //     - `fmaf`/`__fmaf_rn` for the scale-and-narrow in one step, or plain
  //       `*` if the compiler already fuses it.
  //     - `__float2bfloat16`/`__floats2bfloat162_rn` to land the scaled
  //       value in the bf16 representation the fold/intra GEMMs consume.
  //     - `ldmatrix`/`mma_tile.cuh`'s existing operand fetchers once this
  //       value is staged and needs to become an `mma.sync` B or A fragment.

  // STAGE V6: GEMM "READOUT" -- y_partial = R_t . S_entry, and its mass
  //   companion dm_partial = R_t . mass_entry.
  //   WHAT: multiply this chunk's read panel (V3) against S_entry -- the
  //     state as of the START of this chunk, per the frozen-S invariant in
  //     ORIENTATION -- to get each read-member token's partial output. Do the
  //     same against the entry mass column for the ratio's denominator.
  //   INPUTS: read panel (V3), S_entry (S4, scaled by V1's s0 if decay is on),
  //     entry mass column.
  //   OUTPUTS: y_partial, dm_partial per read-member token -- NOT yet the
  //     final answer; the intra term (V9) still has to add this chunk's own
  //     within-chunk causal contribution.
  //   CONTRACT: must read S_entry/mass_entry BEFORE V8 (the fold) overwrites
  //     them with this chunk's own writes -- this is the frozen-S invariant
  //     made concrete: this GEMM and V8 must not observe each other's chunk-
  //     local effects out of order. Restricted to atoms live on the read side
  //     this chunk (S5); a leaf outside that mask contributes zero by V3's
  //     exact-zero guarantee, so it may be skipped rather than multiplied.
  //   CANDIDATE PRIMITIVES:
  //     - `mma.sync.aligned.m16n8k16` (this file's `mma.cuh::mma_m16n8k16`
  //       wrapper already exists for exactly this bf16-in/fp32-accumulate
  //       shape) as the tensor-core instruction the GEMM issues.
  //     - `ldmatrix`/`ldmatrix.x4`/`ldmatrix.x4.trans` (via `mma_tile.cuh`'s
  //       fetchers, which already document the fragment/operand maps for
  //       this exact GEMM's A and B operands) to bring the read panel and
  //       the state into the register layout `mma.sync` expects.
  //     - `cute::TiledMma`/`cute::Tensor`/`cute::gemm` (available: this repo
  //       vendors `cutlass`/`cute` and `mma_tile.cuh` already `#include`s
  //       `<cute/tensor.hpp>`, `<cute/atom/mma_traits_sm80.hpp>` -- currently
  //       used there as a codegen/static-assert cross-check on the hand-
  //       rolled PTX, not to issue MMAs at runtime; using it directly is an
  //       alternative to the hand-rolled `mma.sync` path, not something this
  //       codebase currently does in the kernel body).
  //     - `nvcuda::wmma::mma_sync`/`load_matrix_sync` (`<mma.h>`, the CUDA
  //       C++ warp-matrix API) as a higher-level alternative to raw
  //       `mma.sync` PTX -- available on this toolchain but not used
  //       elsewhere in this project's kernels, which stay on hand-rolled PTX.

  // STAGE V7: GEMM "GRAM" -- G[u][t] = W_u . R_t, masked to u <= t.
  //   WHAT: form the small (chunk-tokens x chunk-tokens) Gram matrix pairing
  //     every write-member token u against every read-member token t in THIS
  //     SAME chunk, contracting over the leaf axis, kept only where u <= t
  //     (inclusive causal: a token may read a write that happened at or
  //     before it, never after).
  //   INPUTS: write panel (V4), read panel (V3).
  //   OUTPUTS: the chunk's Gram matrix, feeding the intra GEMM (V9) and its
  //     mass companion.
  //   CONTRACT: only runs when the chunk has BOTH read and write members
  //     (otherwise there is no within-chunk pair to form). The u <= t mask is
  //     exact -- it is what makes V9 the correct within-chunk causal
  //     correction that S_entry (frozen at chunk entry) structurally cannot
  //     supply. An entry masked out must be an exact zero, since V9 sums over
  //     the whole matrix without re-checking causality.
  //   CANDIDATE PRIMITIVES:
  //     - `mma.sync.aligned.m16n8k16` again (this file's `mma.cuh` wrapper):
  //       the Gram is described as a fourth GEMM contracting over the leaf
  //       axis, so it is the same instruction on a different operand pair.
  //     - `__shfl_sync`/warp-local exchange, or a small shared-memory tile,
  //       if the (<=16 x <=16) Gram result is more naturally produced/
  //       consumed lane-per-(u,t)-pair than through the tensor core for a
  //       matrix this small (verify which is faster; not something this
  //       skeleton prescribes).
  //     - a compile-time-unrolled masked accumulate (`#pragma unroll` +
  //       ternary/`min`) for the `u <= t` triangular mask if built by hand
  //       rather than by zeroing masked GEMM inputs.

  // STAGE V8: GEMM "FOLD" -- S_entry_next = S_entry (rescaled) + sum_u W_u . Vw_u,
  //   and its mass companion (TERM 3) mass_next = s0 * (mass_entry + sum_u
  //   sw_u . W_u).
  //   WHAT: accumulate this chunk's write contributions into the running
  //     state, advancing it from this chunk's S_entry to the NEXT chunk's
  //     S_entry. This is the step that makes the recurrence a recurrence: every
  //     later chunk's V6 sees this chunk's writes only through this folded
  //     state, never directly.
  //   INPUTS: write panel (V4), scaled value operand (V5), the state storage
  //     (S4, already read by V6), decay exit scale (V1's s0, applied once per
  //     chunk here).
  //   OUTPUTS: the state and mass column advanced in place, ready to be this
  //     chunk's V6... no -- ready to be the NEXT chunk's V6 entry state.
  //   CONTRACT: must not be visible to THIS chunk's own V6/V7 (those already
  //     ran, or logically must be treated as having already run, against the
  //     pre-fold state) -- this is the frozen-S invariant's write side.
  //     Restricted to the union of atoms live on the write side this chunk;
  //     an atom outside that mask is left bit-for-bit unchanged (decay's s0
  //     must evaluate to exactly 1 there, per V1's contract, so "untouched"
  //     and "multiplied by an identity decay" agree).
  //   CANDIDATE PRIMITIVES:
  //     - `mma.sync.aligned.m16n8k16` with `D == C` (in-place accumulation
  //       into the same fragment the operand map is read from -- the file
  //       header names this as the reason the state's `[channel][leaf]`
  //       transpose was chosen) as the fold's core instruction.
  //     - `ldmatrix` for bringing the write panel/value operand into
  //       register operand form.
  //     - `fmaf` for the s0 rescale (V1's decay exit scale) applied to the
  //       pre-fold state fragment before/alongside the accumulate.

  // STAGE V9: GEMM "INTRA" -- y_partial += Vw_u . G[u][t] (accumulating into
  //   the SAME y_partial V6 started), and mass TERM 2: dm_partial += sw_u .
  //   G[u][t].
  //   WHAT: add the within-chunk causal contribution that S_entry could not
  //     carry, because those writes happen inside this very chunk: for each
  //     read-member token t, sum over write-member tokens u with u <= t (via
  //     the Gram, V7) the corresponding Vw_u contribution.
  //   INPUTS: the Gram (V7), scaled value operand (V5), the running
  //     y_partial/dm_partial from V6.
  //   OUTPUTS: the chunk's final y/dm for its read-member tokens -- the full
  //     R_t . S_entry + sum_{u<=t}(...) formula, now complete for this chunk.
  //   CONTRACT: only runs when the chunk has both read and write members (same
  //     condition as V7, whose output it consumes). Must accumulate into, not
  //     overwrite, V6's partial -- the two terms of the telescoped formula are
  //     additive.
  //   CANDIDATE PRIMITIVES:
  //     - `mma.sync.aligned.m16n8k16` with `C` seeded from V6's partial
  //       result (accumulate-in-place, same trick V8 uses) as the intra
  //       GEMM's instruction.
  //     - `ldmatrix` to fetch the Gram (V7) and value operand (V5) into
  //       register form for the multiply.
  //     - `fmaf` if any part of this accumulation is more natural as scalar
  //       lane arithmetic than a tensor-core pass (e.g. the mass companion
  //       term, which is a much smaller reduction than the channel-wide `y`
  //       term).

  // STAGE V10: EMISSION.
  //   WHAT: scatter this chunk's finished y (numerator) and dm (denominator)
  //     contributions out to the token-indexed global accumulators, for this
  //     chunk's read-member tokens.
  //   INPUTS: finished y/dm (V9, or V6 alone if the chunk had no write
  //     members and V9 was skipped), the read-scale from V2.
  //   OUTPUTS: global `num`/`den` accumulators updated for this chunk's
  //     read-member tokens.
  //   CONTRACT: must use addition (not overwrite) into the global
  //     accumulators -- a token may be a read-member of more than one owner
  //     across the whole kernel launch (it belongs to multiple leaf blocks
  //     across levels/segments), and its final value is the SUM of every
  //     owner's contribution. A token with an exact-zero read-scale (a padded
  //     slot in a short trailing chunk) contributes nothing and may be
  //     skipped.
  //   CANDIDATE PRIMITIVES:
  //     - `atomicAdd` (this project already uses it for fan-in in
  //       `summaries.cu`/`decode.cu`) for the multi-owner-per-token
  //       accumulation into `num`/`den`, since the CTA writing a given
  //       token's contribution is not the only writer of that token's slot.
  //     - `red.global.add` (the PTX reduction instruction `atomicAdd`
  //       lowers to; naming it as the underlying instruction, not a separate
  //       API) and `__stwt`/`__stcg` write-hint intrinsics if a non-atomic
  //       store is legal on some path.
  //     - `fmaf`/`*` to apply V2's read-scale immediately before the
  //       addition if it was not already folded in earlier.

  // STAGE V11 (auxiliary; only in the "summary" pass mode used to precompute
  //   per-leaf write-inclusion ahead of a later, separate scan pass over a
  //   long segmented owner -- NOT part of ordinary single-segment forward
  //   math):
  //   WHAT: track, per leaf, whether it received any nonzero write this
  //     chunk, and accumulate a running survival product for later
  //     publication.
  //   INPUTS: this chunk's write panel (V4) and decay entry scale (V1).
  //   OUTPUTS: per-leaf inclusion flags and a running survival product, used
  //     only by the EXIT stage's summary publication (E2 below).
  //   CONTRACT: only meaningful under the summary pass; ordinary forward
  //     passes skip this entirely. Not required to understand the core
  //     recurrence -- it exists to support splitting one very long owner's
  //     token range across multiple CTAs/passes.
  //   CANDIDATE PRIMITIVES:
  //     - `__ballot_sync` to turn "did leaf L receive a nonzero write this
  //       chunk" (one predicate per lane/leaf) into a packed word in one
  //       instruction (`summaries.cu` already does exactly this: `__ballot_
  //       sync` then `atomicOr` the resulting word).
  //     - `atomicOr` to fold a newly-observed inclusion word into a running
  //       per-leaf inclusion bitmap shared across warps.
  //     - `__popc` to count newly-set bits if a running total is kept
  //       alongside the bitmap (`summaries.cu`'s comment on `atomicOr`
  //       returning the old word plus `__popc` of the delta is the same
  //       pattern).
  //     - `fmaf`/`*` for the running survival-product accumulation.

  // STAGE V12: CARRY.
  //   WHAT: the state and mass column V8 just produced become the next
  //     chunk's S_entry and mass_entry.
  //   CONTRACT: this carry must be visible to the next loop iteration's V6
  //     before that iteration's V6 runs; loop back to V1 for the next chunk,
  //     or exit the walk if S5 signalled end-of-stream (a chunk descriptor
  //     with zero member tokens).
  //   CANDIDATE PRIMITIVES:
  //     - none needed if the state truly stays in the same registers/shared
  //       location across iterations (the file header says the state lives
  //       in registers as the fold's own accumulator fragments, so V8's
  //       output may already BE V6's next input with no separate move).
  //     - `__syncthreads()`/`__syncwarp()` only if the carry crosses a
  //       thread/warp boundary that the fold and the next readout don't
  //       naturally share -- a barrier is a correctness requirement here,
  //       never a default to reach for.

  // ---------------------------------------------------------------------------
  // EXIT -- once per CTA, after the walk terminates.
  // ---------------------------------------------------------------------------

  // STAGE E1 (performance-only, no semantic content): if any gathers (S6) were
  //   issued ahead of their consuming chunk and are still outstanding when the
  //   walk ends, wait for them to land before the CTA exits -- a kernel must
  //   not release memory it told the copy engine to keep writing into. Not
  //   needed at all in a synchronous, no-lookahead implementation.
  //   CANDIDATE PRIMITIVES:
  //     - `cp_async_wait_all`/`cp_async_wait_group` (this file's own
  //       `cp_async.cuh` wrappers around `cp.async.wait_all`/`cp.async.
  //       wait_group`) or the raw `__pipeline_wait_prior`/`__pipeline_
  //       wait_all` from `cuda_pipeline_primitives.h`.
  //     - `__syncthreads()` after the drain if other threads in the CTA
  //       depend on the drained data being visible CTA-wide, not just to the
  //       issuing thread.

  // STAGE E2 (summary pass only): publish the per-leaf write-inclusion
  //   summary and the accumulated survival product (V11) to the intermediate
  //   segment stash, along with the current state, so a later pass over this
  //   owner's next segment can pick them up as its entry state (S4).
  //   CONTRACT: only this CTA's own segment's data is published, at a slot
  //     keyed by this segment's identity; a later segment's S4 depends on
  //     finding it there.
  //   CANDIDATE PRIMITIVES:
  //     - `cub::BlockReduce` (`<cub/cub.cuh>`, already used in `summaries.cu`
  //       for a CTA-wide reduction) if the survival product/inclusion
  //       summary needs a block-wide combine before the single publishing
  //       write.
  //     - plain non-atomic stores (this segment owns its own stash slot
  //       exclusively, per the CONTRACT) rather than `atomicAdd` -- no
  //       cross-CTA race to guard against here, unlike V10's emission.
  //     - `__syncthreads()` before the publish if the summary was assembled
  //       cooperatively across the CTA's warps and needs a CTA-wide barrier
  //       first.

  // STAGE E3 (ordinary forward pass, not the summary pass): if -- and only if
  //   -- this CTA covers the owner's LAST segment (S1), write the final state
  //   and mass column back to the owner's persistent HBM state block.
  //   CONTRACT: this is the second and last half of the one-copy invariant
  //     (S4 was the first): exactly one HBM write per owner, and only from the
  //     segment that has seen the owner's entire token range, never from an
  //     intermediate segment (whose result belongs in the stash from E2, not
  //     in the owner's real state).
  //   CANDIDATE PRIMITIVES:
  //     - plain (non-atomic) global stores, or vectorized `float4`/`int4`
  //       stores, for the bulk state write -- the one-copy invariant means
  //       this CTA is the state block's sole writer, so no atomic is needed.
  //     - `__stcg`/`__stwt` (cache-hint store intrinsics) if a non-default
  //       cache policy on this final, non-reread write is worth stating
  //       explicitly (verify measured benefit before reaching for it).
  //     - `cp_async_16`/`__pipeline_memcpy_async` are NOT applicable here --
  //       `cp.async` is a global<-shared (load-direction) primitive; this is
  //       a store back to HBM.

  // ---------------------------------------------------------------------------
  // Kept only for the translation unit to parse -- delete these two lines once
  // you have a real body. They silence "unused parameter" on every axis that a
  // real implementation will, in fact, use.
  // ---------------------------------------------------------------------------
  (void)p;
}

#undef CONSUMER_D_ROWS

}  // namespace consumer
}  // namespace rola
