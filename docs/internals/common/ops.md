# `csrc/rola/src/common/ops.cuh` — the device op helpers

The instruction-level helpers every kernel executes through: the MMA atoms (bf16, f16 and int8) and their fragment
loads, the asynchronous staging and its barriers, the pinned-base addressing discipline the SASS gate patrols, the
lane votes, the reductions and the scheduling controls.

## Layer 2b -- the baseline operations

HOW EACH STEP IS PERFORMED.  Layer 2a (`design.cuh`) says what the design
LOOKS LIKE on this architecture; this file says how each of its steps is
EXECUTED, as operations with contracts.  Every entry is defined by WHAT THE
ALGORITHM NEEDS, never by what an instruction does: "stage this tile and tell
me when it is visible", not "issue a cp.async".  Get that level wrong and the
abstraction leaks on the first new architecture -- TMA cannot implement "issue
a cp.async", but it can implement `stage_tile`.

A kernel above this header names no architecture. It reads layer 2a for its design and calls this file to execute
it.

This file is also where THE CONSISTENCY BETWEEN 2a AND 2b is checked (see the
section at the foot): a selection is invalid if no available operation can
serve it, and that is what the assertion says -- "selection unsatisfiable
given available operations", never "this architecture lacks instruction X".
Where the capability row offers a better implementation than the one built,
the gap is ANNOUNCED rather than taken silently.

### Operations and contracts

Every operation is defined by WHAT THE ALGORITHM NEEDS. A helper with no caller in a shipped kernel is marked
(no shipped caller); it is kept or removed with the code.

#### Staging

- `smem_addr(p)`: the CTA-local address of a shared object, the address type every shared operation consumes.
- `stage_tile(dst, src)`: copy <a id="kstagebytes"></a>`kStageBytes` (16) bytes global -> shared without occupying
  the issuing thread. `kStageBytes` is a fact of the operation, the widest single asynchronous copy the staging
  issues, and staging loops are shaped from it. A group is every `stage_tile` since the last `stage_commit()`;
  `stage_wait<n>()` means all but the newest `n` groups have landed FOR THE ISSUING THREAD, and CTA-wide visibility
  needs a following `rendezvous()` -- a property of the contract that TMA/mbarrier satisfies the same way.
  <a id="kstagingisbulk"></a>`kStagingIsBulk` states that the built form moves `kStageBytes` an issue; a bulk (TMA)
  form would take one descriptor issue a tile and is not built (announced, below).
- <a id="stage-tile-if"></a>`stage_tile_if`: the copy with its source predicated -- a dead quantum ZERO-FILLS its
  destination and reads nothing, so a partial tile is the same code as a full one and never reads past an operand's
  end.
- <a id="stage-tile-n"></a>`stage_tile_n`: the copy of a run's first `bytes` (0 to 16), the rest zero-filled and
  nothing past them read, so a row that ends inside a quantum (a level whose width is not a multiple of sixteen) never
  reads the next level's digits (the facts pass's rows, `docs/internals/facts/facts.md#load`).
- <a id="stage-run"></a>`stage_run`: the same copy at a narrower quantum. A run shorter than 16 bytes is still ONE
  asynchronous access, never a padded 16-byte one that would read a neighbour's bytes. The 4- and 8-byte forms are
  the cache-all variant, because the 16-byte bypass has no narrower member; source and destination are naturally
  aligned, the caller's contract, so no size is ever split here.
- <a id="stage-run-if"></a>`stage_run_if`: `stage_run`'s predicated form -- a dead run reads nothing and zero-fills,
  which makes a partial window arithmetic rather than a tail case (a token past the window lands zero amplitudes).
- <a id="stage-run-lanes"></a>`stage_run_lanes`: `stage_run`'s LANE-predicated form; a dead lane issues nothing and
  its destination keeps what it holds. Measured: dead lanes landing zero-size cost the memory pipe nothing on
  16-byte copies and ~60 cycles a copy on the 4-byte `.ca` ones (no shipped caller).
- <a id="load-vec16"></a>`load_vec16` / `store_vec16` and <a id="storesharedvec16"></a>`store_shared_vec16`: the
  16-byte quantum moved synchronously, the same width as every staged copy, so one bank analysis covers a buffer
  whether its rows were staged or computed. `copy_shared_to_global_16b` composes the two into one coalesced 128-bit
  store.
- `rendezvous()`: every thread of the CTA arrives and all prior shared accesses are ordered before all subsequent
  ones; also the WAR fence that lets a ring slot be overwritten. <a id="rendezvous-group"></a>`rendezvous_group(id, n)`
  is the same restricted to `n` warp-aligned threads at named barrier `id` (id 0 is the CTA's and is not offered;
  sm_80/86/90 provide sixteen).
- <a id="mbarrier"></a>`mbar_init`, `mbar_arrive_when_landed`, `mbar_arrive`, `mbar_wait`: an `mbarrier` (sm_80+)
  completes a phase at `count` arrivals and flips its parity. A thread's `cp.async` copies arrive when they LAND
  (`cp.async.mbarrier.arrive.noinc`: the arrival is one of the counted ones); a thread's own arrive releases its
  prior shared stores; a wait acquires. `mbar_wait` is a `test_wait` spin, cheap when the phase is already complete,
  which is the case a design arranges. `mbar_test` asks once, for a loop that does other work while copies have not
  landed (its answer voted over the warp before it steers a branch). `mbar_wait_sleep` polls with a 64 ns
  `__nanosleep`: on sm_80/86 a wait is a polled shared word, and a tight poll takes the issue slots of the warps doing
  the work it waits on.

#### Addressing

- <a id="gather-ptr"></a>`gather_ptr(base, off)`: a gathered row's address, a 64-bit base formed once plus a 32-bit
  BYTE displacement ZERO-EXTENDED rather than multiplied out. The arithmetic is the contract: at element type the
  compiler re-derives `base + index` as a widening multiply per access (measured 30x `IMAD.WIDE` in a loop this form
  emits none for). The caller owns the 32-bit bound. The base is asserted GLOBAL, because a generic pointer turns a
  reduction into an atomic read-modify-write and every gathered load into a generic one. (no shipped caller)
- <a id="t"></a>`pin_address(p)`: a base the compiler must keep rather than re-derive. Under register pressure a
  compiler drops a live base and recomputes it in the loop -- the widening multiply moved rather than removed; the
  value becomes the result of an operation ptxas cannot see through, at the cost of the two registers it occupies.
- <a id="page-slot"></a>`page_slot`: the state plane's one base translation, `slot = page_tbl ? page_tbl[atom] :
  atom` ([`paging/paging.md`](../paging/paging.md)). A negative slot is skipped at both ends of the state I/O, and
  the predicate is CTA-uniform because an atom belongs to exactly one box, which is what licenses skipping the
  barriers inside a sweep with the traffic. (no shipped caller)
- <a id="atom-activity"></a>`atom_activity`: the facts' two orthogonal per-atom bits, `kAtomWritten` and
  `kAtomRead`, never pre-ORed by the producer (the read-only distinction is what the touched-atom census is about, and
  ORing saves nothing). LOAD an atom iff it is resident and `(read | written)`; STORE it iff `written` -- identical
  under both backings, which is what makes the paging gate's `torch.equal` a certificate of the skip logic. A null
  `atom_bits` is the universal fact: every resident atom is read and written. (no shipped caller)
- <a id="store-shared-u16"></a>`store_shared_u16`, <a id="storesharedu32"></a>`store_shared_u32`,
  <a id="loadsharedu16"></a>`load_shared_u16`, `load_shared_f32`, <a id="store-shared-f32x2"></a>`store_shared_f32x2`,
  <a id="load-shared-v4"></a>`load_shared_v4`: shared accesses ADDRESSED rather than indexed. A swizzled or
  dynamically indexed address is not an affine function of an index, and `dst[k++]` through a typed pointer
  re-derives a widening multiply every iteration (measured: a loop of one `STS.U16` a live bit ran as `IMAD:114,
  IADD3:69` with no `STS` in its top opcodes); a 32-bit shared address advanced by the element size costs one add.
  The pair and quad forms are aligned single accesses (`st.shared.v2.f32`, `ld.shared.v4.u32`); a quad every lane
  reads at one address is a broadcast. (`store_shared_u16`, `store_shared_u32`, `load_shared_u16`,
  `store_shared_f32x2`, `load_shared_v4` and `load_shared_f32`: no shipped caller)

#### Votes and lanes

- <a id="kballotlanes"></a>`kBallotLanes`: the lockstep unit's width, deliberately not `kMmaUnitThreads`: a warpgroup
  issues one MMA across 128 threads and still votes in lanes of 32, and conflating them reads a quarter of a vote.
- <a id="lane-ballot"></a>`lane_ballot(pred)`: one bit per lane, in lane order -- the unit's own agreement about its
  lanes' data, with no memory round trip and nothing ordered. Written as the intrinsic: the promise is the bit vector.
- <a id="bit-count"></a>`bit_count`, <a id="firstset"></a>`first_set`, <a id="lastset"></a>`last_set`: the readings
  of a vote that turn agreement into POSITIONS -- how many, and where first or last. An empty vector gives
  `kBallotLanes` for `first_set` and `-1` for `last_set`, so "nowhere" compares past every lane with no separate case.
- <a id="lane-prefix-incl"></a>`lane_prefix_incl<N>`: the unit's running total; `N` a power of two no wider than the
  unit, lanes at or above it the caller's to mask. Every lane executes every step, so the unit is converged at each
  `__shfl_up_sync`.
- <a id="nth-set-bit"></a>`nth_set_bit`: the n-th set bit of a word by a branch-free binary search on popcounts,
  about thirty instructions, against the intrinsic `__fns`, a software loop (no shipped caller).
- <a id="uniform-warp"></a>`uniform_warp`: the warp's index as a value ptxas knows to be warp-uniform (three ballots
  on the thread index's bits); `threadIdx.x >> 5` is per-thread to its analysis, and a loop bounded by it is divergent,
  after which every shuffle is fenced and outlined.
- <a id="warp-sync"></a>`warp_sync()`: `bar.warp.sync 0xffffffff`. `__syncwarp()` at a site the compiler cannot prove
  convergent lowers to a CALL into `__cuda_sm70_warpsync`; the instruction keeps the edge a single WARPSYNC (no
  shipped caller).

#### The MMA atoms and their operands

- <a id="kmmaunitthreads"></a>`kMmaUnitThreads`: the threads that jointly issue one MMA and own its accumulator --
  a property of the built `mma`, not of an architecture; the consistency section is where it and the capability row
  must agree.
- <a id="load-frag"></a>`load_frag` / `load_frag_t`: four 8x8 b16 tiles from shared memory in the register order the
  MMA consumes; `_t` transposes each 8x8 so one shared tile serves either handedness. THE `memory` CLOBBER IS PART OF
  THE CONTRACT: `ldmatrix` reads shared memory, and without the clobber the compiler may move it across a barrier, so
  a warp could pull rows another warp has not published.
- <a id="mma"></a>`mma(acc, a, b)`: acc += A(16x16) . B(16x8), bf16 operands, fp32 accumulator private to the issuing
  unit, both operands from its registers. The atom is chosen by SHAPE FAMILY: `kMmaM/N/K` is the shape every fragment
  index is written in, and an architecture whose register-source family is shaped differently (Hopper's `_RS` atoms,
  64 x {8..256} x 16, K-major A) substitutes its own member. <a id="kmmaacceptsregistera"></a>`kMmaAcceptsRegisterA`
  states the operand sources the built `mma` accepts; a kernel whose accumulator must land on a particular side
  asserts against them at its own site.
- <a id="hmma"></a>`hmma`, `hmma_fresh`, `hmul2`, `hfma2`, `pack_half2`: the fp16 family -- `m16n8k16` f16 into fp32
  (`hmma_fresh` into a fresh fragment, C = 0), two f16 products or fused products each rounded once to nearest, two
  fp32 to an f16 pair rounded to nearest (`cvt.rn.f16x2.f32`, `lo` in the low half). Every rounding is explicit so
  `tests/oracle/fold_reference.py` can mirror it.
- <a id="imma-u8"></a>`imma_u8_k32`, `imma_u8_k32_fresh`, `imma_u8s8_k32`: `mma.sync.m16n8k32` over u8 and s8 into
  s32, 4096 multiply-adds. A is four registers a lane (rows `lane / 4` and `+ 8`, four consecutive k a register at
  `4 (lane % 4)` and `+ 16`), B two (column `lane / 4`), the accumulator four s32 -- exactly the register image
  `ldmatrix` gives of two 16-byte rows a lane, so a token-major row of digits is an operand with no shuffle. `u8.u8`
  is digits against digits, `u8.s8` unsigned weights against signed values; `imma_u8_k32_fresh` takes C from the
  zero register so the accumulator is not live before a sum's first step. On sm_86 one issues every 16 cycles a
  scheduler (8 on sm_80),
  four times the bf16 fp32-accumulate rate (probe of 2026-09-28).
- <a id="prmt"></a>`prmt`, `pack_byte4`: `prmt.b32` picks four of two words' eight bytes by a selector nibble each;
  `pack_byte4` takes the same byte of four words with three of them (pair selectors `0x0040`, `0x0051`, `0x0062`, then
  `0x5410`), which is how an int8 operand row is formed from four gathered words without shifts or masks.
- <a id="transpose-frag-b16"></a>`transpose_frag_b16`: an 8x8 b16 tile transposed where it lies
  ([below](#reg-transpose)).

#### The bf16 pair helpers (no shipped caller)

- <a id="hi-bf16"></a>`hi_bf16`, `hi_bf16x2`: an fp32's top half IS its bf16 truncation, so a register pair becomes an
  operand with one `prmt.b32` (selector `0x7632`) against `pack_bf16x2`'s conversion. It TRUNCATES where
  `pack_bf16x2` rounds to nearest even: one bf16 ulp, biased toward zero, a declared rounding an oracle mirror must
  carry. <a id="lo-bf16x2"></a>`lo_bf16x2` packs the two LOW halves (`prmt` 0x5410), and
  <a id="pack-lo"></a>`pack_lo` / `join_hi_lo` split a word into hi and lo and rejoin it bit for bit.
- <a id="splat-bf16x2"></a>`splat_bf16x2` puts one bf16 in both halves so a scalar is an operand of `mul_bf16x2`;
  <a id="splat-bf16x2-bits"></a>`splat_bf16x2_bits` does it from a raw bit pattern, because a register array must
  never be address-taken (one `&run[i]` puts the whole run in local memory).
- <a id="mul-bf16x2"></a>`mul_bf16x2`: the elementwise product of two packed bf16 pairs, rounded to nearest even. The
  PTX ISA gates `mul.rn.bf16x2` at sm_90 and `fma.rn.bf16x2` at sm_80, so on Ampere the product is one HFMA2 with a
  <a id="knegzeropair"></a>`kNegZeroPair` addend: `a*b + (-0.0)` rounds the exact product once and keeps a signed
  zero, where a `+0.0` addend would turn every `-0` into `+0`. There is no fp32 route: every tabulated row carries
  `bf16_simd_fma`, and an untabulated architecture fails the build.
- <a id="store-bf16"></a>`store_bf16`: one fp32 to one bf16, rounded as `pack_bf16x2` rounds each half, for a
  transposed accumulator whose adjacent pair is adjacent in TOKEN and so in different rows of a [token][d_v] buffer.
- <a id="or-run"></a>`or_run<N>` with <a id="kbf16magnitudemask"></a>`kBf16MagnitudeMask` and
  <a id="koctets"></a>`kOctets`: the OR of a contiguous, naturally aligned run at the widest access that covers it;
  the run is entirely zero exactly when the OR has no magnitude bit set, since bf16's only zeros are +0 and -0 and OR
  cannot clear a set bit. `kOctets` names `N / 8` for the unroll-discipline lint.

#### Reductions

- <a id="fan-in-add"></a>`fan_in_add(p, x)`: ONE contribution into an accumulator other units also contribute to,
  fire and forget. Nothing is returned, and that is the substance: a returning atomic keeps the issuing thread
  waiting, and the point of a reduction is that nothing waits. No order across contributors is defined, so a caller
  that needs a reproducible summation order needs another operation. Stated as `red.global.add.f32`, because an
  `atomicAdd` with an unused result is only sometimes lowered to a reduction; the SASS gate checks the non-returning
  form. <a id="storef32"></a>`store_f32` stores a summand at fp32, unrounded, for a buffer whose consumer is a fan-in.
- <a id="red-global"></a>`red_global_add_f32`: the same with the address converted to the global window first. A
  generic `atomicAdd` on a pinned base lowers to an address-space probe, a shared-memory CAS spin and a local fallback
  beside a RETURNING atomic, whose round trip the next branch waits on (measured: 30% of a dense window's stall
  samples on those branches).
- <a id="or-shared-u32"></a>`or_shared_u32`: one contribution into a shared bitset, fire and forget -- the fan-in's
  contract over the bitwise monoid (no shipped caller). <a id="redsharedaddu32"></a>`red_shared_add_u32`: the same
  over the counting monoid, an arrival count whose incrementer never learns the total; `red_shared_add_f32` over fp32.
- <a id="atom-shared-add"></a>`atom_shared_add_u32`: `atom.shared.add.u32` with the old value returned -- a shared
  counter's take. Stated shared, so ptxas emits no address-space probe.
- <a id="publish-fence"></a>`publish_fence()`: the order between a unit's shared writes and the flag that publishes
  them, at BLOCK scope: producers and consumers that are warps of one CTA need no device-scope fence, which would
  order traffic no participant can observe (no shipped caller).

#### Scheduling controls

- <a id="fresh"></a>`fresh(a, b)`: two values made opaque where it stands (an empty `asm volatile` reading and
  writing them), so nothing derived from them afterwards is computed before it. A step run once a window rebuilds its
  addressing where it is used instead of ptxas hoisting it out of the window loop and holding it live across the
  MMA bursts, where registers are scarce.
- <a id="once"></a>`once(cond)`: `for (int n = ops::once(cond); n > 0; --n)` is a real branch where `if (cond)` on a
  warp-uniform value would be if-converted and every warp would issue the body predicated off (KERNEL_STANDARDS §20).
  The bound is hidden from ptxas by an empty `asm volatile`, so the loop cannot be flattened as 0-or-1 trips.
- <a id="cycles"></a>`cycles()`: `%clock64` by `mov`, for the phase ledgers; the `clock64` intrinsic is not declared
  in every host-pass compilation of a header both a launch and a harness include.
  <a id="cycles-after"></a>`cycles_after(after)` takes `after` as an input it does not use, so the read cannot issue
  before `after` arrives: ptxas deletes a clock read whose value is unused, so a dependency is the only way to order
  one.

### <a id="reg-transpose"></a>The register transpose

`transpose_frag_b16(x)` transposes an 8x8 b16 tile where it already lies, across
the issuing unit's own registers, for a contraction whose `k` axis is the axis an
accumulator carries on its `m` side (no shipped caller). The contract is about MAPS,
not about data.

For `m16n8k16`, the accumulator spreads its two axes over the lanes as
`(m = t/4 [+8], n = (t%4)*2 + {0,1})` and a `B` operand spreads its two as
`(k = (t%4)*2 + {0,1} [+8], n = t/4)`. The two are the same matrix through a
transposed lane map, so an accumulator can be an operand of a contraction over its
own `m` axis only through this operation.

**MEASURED, not read off the ISA.** A standalone probe loaded a known 16x8 matrix
into a thread's registers using the accumulator map, fed it as `B` against
`A = I16`, and read the mma's own output back:

| candidate | verdict |
|---|---|
| the accumulator map already IS the `B` map | REFUTED: 112 of 128 elements differ, and the difference is exactly the transpose |
| `movmatrix.sync.aligned.m8n8.trans.b16` | 0 of 128 differ. Two instructions per (16 `d_v` x 8 slot) tile |
| a shared-memory round trip through the padded buffers | not needed, not built |

The capability is a REQUIRED clause of the baseline (`arch_caps.cuh`) rather than a
selecting one, because the alternative is not a slower implementation of the same
operation — it is a different algorithm, with a `BCg x d_v` staging buffer that
does not fit beside the ring.

The SASS gate needs no carve-out for it: `MOVM` is not one of the signatures
`tools/sass_gate.py` reds on (local memory, convergence subroutines, predicated
HMMA, instructions-per-HMMA).

### Operations that are not here

There is no `warp_reduce`, no `compact` and no publish/acquire pair weaker than a full rendezvous: an operation
invented before it has a caller cannot be defined by what the algorithm needs, because no algorithm needs it yet.

### What is deliberately not a capability

The shared-memory layout (pad-8 against an XOR swizzle) is legal on every
tabulated architecture and is chosen by MEASUREMENT, not by the table.  A
layout knob in the capability table would make every future measurement
unfalsifiable.  It is a free axis, and `design.cuh` says why it stays one.

## <a id="consistency"></a>The consistency between layers 2a and 2b

<a id="static-assert"></a>`design.cuh` claims REGISTERS is a residency this baseline builds: it is built by an `mma`
that reads an operand from the issuing unit's own registers and by `pack_bf16x2`, the accumulator-to-operand
conversion that keeps the state out of shared memory. If either went away the claim would be false, and the first
assertion is where that is caught.

<a id="static-assert-2"></a>The second asserts THE CARVE'S UNIT: `mma`, `load_frag` and the carve are written for a
32-thread unit owning a 16-row slice of the state, and a 128-thread warpgroup unit owns 64 and issues a differently
shaped atom. It is a consistency clause between the design's carve and the operations that execute it, and its failure
reads NOT BUILT rather than UNSUPPORTED.

<a id="note-l562"></a>THE ANNOUNCEMENT SITES are where the capability row records a better implementation of a
baseline operation than the one built: `stage_tile` on a part with bulk asynchronous copy, where TMA satisfies the same
contract with one descriptor issue instead of one a 16 bytes. None is an error -- the built path is correct everywhere
it compiles -- and what is forbidden is taking the slower path silently.
