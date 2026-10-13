# `csrc/rola/src/common/arch_caps.cuh` — the capability rows

The ladder's per-architecture capability table. Its rows are data in
[`hardware.json`](hardware.md), generated into `common/hardware.cuh` (`Caps`, `tabulated`, `caps_of`), which
this file includes and derives from: one `Caps` row per SM generation (async copy, tile loads,
MMA operand sourcing, TMEM, SMEM/register budgets), consumed by `design.cuh`'s
derivations. The sm_90 row builds and announces capabilities the 32-thread
kernels do not yet use (wgmma-class residency, TMA staging) — an
announcement, not a refusal, by the ladder's sm_90 ruling.

## Layer 1 -- the architecture capability facts

LAYER 1 -- THE ARCHITECTURE CAPABILITY FACTS.

THE THREE LAYERS.  This file is the first of them and it is the only one that
knows an architecture exists:

  1  CAPABILITY FACTS      this file.  Objective, per architecture, no
                           judgment.  Does the MMA accept an operand from the
                           issuing unit's registers?  Does the accumulator
                           live in TMEM?  How many bytes of shared memory does
                           an SM have?  Nothing here decides anything.
  2a DERIVED SELECTIONS    `design.cuh`.  What the design LOOKS LIKE on this
                           architecture -- state residency, staging depth --
                           each COMPUTED from layer 1 rather than looked up
                           per architecture.  A per-arch table of selections
                           would be an architecture branch in a costume.
  2b BASELINE OPERATIONS   `ops.cuh`.  HOW each step is performed, as
                           operations with contracts, plus the statement of
                           which implementations are actually built.
  3  THE KERNEL            reads 2a for its design and calls 2b to execute.
                           It touches this file never and names an
                           architecture never.

THIS COSTS NO SPECIALISATION.  Every fact below is a constant of the compile
target, so layers 1 and 2b collapse to a single arm per compilation; only the
FREE axes (the shared-memory layout) multiply the binary, and those are measured
rather than derived.  Nothing here is ever a template
parameter.

THE KINDS OF ENTRY.  A fact is one of:

  REQUIRED   a capability whose absence would change the STRUCTURE, not the
             implementation.  Never a branch: `supported()` goes false and the
             build stops with the clause that failed.
  SELECTING  a capability that implements a baseline operation more
             efficiently.  A compile-time bool.  Code branches on the FLAG,
             never on an architecture number.
  RESIDENCY  where an accumulator can live and where an operand can be
             sourced from.  These are INSTRUCTION facts under their honest
             names, and they are NOT read by any kernel: layer 2a turns them
             into `StateResidency`, which is what the algorithm actually
             requires.  Named as instructions at the level a kernel sees, a
             part whose accumulator lives in TMEM would be REFUSED for having
             the wrong storage class while meeting the real requirement.
  FACT       a quantitative resource (bytes, registers, threads).  NOT a
             branch under any circumstances.  It is an INPUT TO DERIVATION:
             residency, staging depth, the register budget.

THERE IS NO OTHER KIND, and in particular there are no fallback chains.  A
capability must never silently select a slower route that then ships
unnoticed; that is the difference between this table and a fallback chain.
Where a better implementation exists and is not built, layer 2a ANNOUNCES it
at compile time -- a documented gap a reader sees is a stated engineering
decision, and a silent slower path is the forbidden thing.

PER-SM FACTS LIVE HERE; PER-DEVICE FACTS DO NOT.  SM count is not an
architectural constant -- GA102 ships at 80 and 82 SMs, GA100 at 108 -- so it
is a runtime `cudaDeviceProp` query, never a table row.  Everything in
`Caps` is a property of the compute capability itself.

## <a id="note-l125"></a>The row's fields

The generated rows initialize the fields by name, in this order, and an untabulated compute capability gets an
empty row (`tabulated` refuses it first).

| field | kind | what it states |
|---|---|---|
| `async_copy`, `tile_load`, `mma_bf16_f32`, `bf16_cvt_packed` | REQUIRED | `cp.async`, `ldmatrix`, the bf16 MMA with an fp32 accumulator, the packed two-float conversion |
| <a id="reg-transpose-b16"></a>`reg_transpose_b16` | REQUIRED | an 8x8 b16 tile spread across the MMA unit's own registers is transposed IN PLACE. A contraction over the accumulator's own m axis puts the accumulator on the k side, and the accumulator map and the k-side map differ by exactly this transpose. |
| <a id="mma-a-from-regs"></a>`mma_a_from_regs` | RESIDENCY | the A operand may come from the issuing unit's own registers; a warpgroup MMA keeps a register-source A (the `_RS` atom family), so this holds on every tabulated architecture |
| <a id="mma-b-from-regs"></a>`mma_b_from_regs` | RESIDENCY | the same for B. A warpgroup MMA sources B ONLY from a shared-memory descriptor; which SIDE a kernel's accumulator lands on is a clause of that kernel, asserted at its own site |
| <a id="accum-in-tmem"></a>`accum_in_tmem` | RESIDENCY | the accumulator lives in a dedicated tensor memory (`tcgen05`). False on every row; a Blackwell row would set it, and the residency DERIVATION, not a new refusal, is what would then have to hold |
| <a id="mma-a-from-tmem"></a>`mma_a_from_tmem` | RESIDENCY | A may be sourced from that tensor memory, which lets the invariant survive with no register-resident state |
| <a id="bf16-simd-arith"></a>`bf16_simd_arith` | SELECTING | `mul.rn.bf16x2`. ptxas rejects it below sm_90 (measured 2026-08-09): Ampere has bf16 conversion and bf16 MMA but no bf16 SIMD multiply, so the product costs 1 FMUL + 1/2 pack an element there against ~1/2 instruction on sm_90 -- the archetypal flag |
| <a id="bf16-simd-fma"></a>`bf16_simd_fma` | SELECTING | `fma.rn.bf16x2`, available at sm_80 (measured 2026-08-28): a product is one instruction through a `-0.0` addend, bit-identical to `mul.rn.bf16x2` (one rounding of the exact product; the addend's sign keeps a signed zero) |
| <a id="async-bulk-copy"></a>`async_bulk_copy` | SELECTING | TMA: `stage_tile`'s contract in one descriptor issue instead of one a 16 B |
| <a id="warpgroup-mma"></a>`warpgroup_mma` | SELECTING | `wgmma`, present on sm_90 and the reason `mma_b_from_regs` is false there |
| <a id="dyn-reg-realloc"></a>`dyn_reg_realloc` | SELECTING | `setmaxnreg`: a warpgroup may give up or claim registers at run time. Without it allocation is uniform across the kernel and sized by the maximum live set over a thread's paths, so a thread that produces then consumes pays `max(producer, consumer)` and dedicating a unit to production cannot be paid for by shrinking it |
| <a id="mma-unit-threads"></a>`mma_unit_threads` | FACT | the threads that jointly issue one MMA and own its accumulator: 32 (warp) or 128 (warpgroup). The carve is expressed in these units, not warps |
| `smem_per_sm`, `smem_per_cta_max`, `smem_granularity`, `regs_per_sm`, `regs_per_lane_max`, `reg_alloc_unit`, `max_threads_per_sm`, `max_ctas_per_sm` | FACT | the SM's budgets |
| <a id="driver-smem-reserve"></a>`driver_smem_reserve` | FACT | the per-CTA reserve the driver takes off the top; the residency condition is on `smem_per_cta + driver_smem_reserve` |

## The queries over a row

<a id="note-l163"></a>**`supported()` is the conjunction of the REQUIRED clauses**, with no partial support.
`mma_a_from_regs` is NOT among them and must not be: whether the state can be an operand without a memory round trip
is a RESIDENCY question, answered by `design.cuh` from the residency facts, and an architecture that answers it
differently is a derivation rather than a refusal.

<a id="caps-of-note-l147"></a>**sm_90 is TABULATED AND NOT BUILT** (`caps_of`). Every REQUIRED clause holds and the
residency derivation succeeds -- `wgmma` keeps a register-source A, so the state can be an operand without a round
trip there too. What is missing is an IMPLEMENTATION: the built MMA, fragment loads and carve are written for a
32-thread unit, and a warpgroup unit is 128. A capability gap, not a structural refusal.

<a id="tabulated"></a>**`tabulated` is the RATIFIED SET.** A compute capability that is not a row fails the build; it
is never approximated by a neighbouring row.

<a id="regspercta"></a>**`regs_per_cta`** is the registers a CTA actually reserves: allocation is per warp in
`reg_alloc_unit` quanta, so a lane asking for 129 costs the same as one asking for 136.
<a id="min4"></a>**`min4`** is the resident CTAs an SM: the binding constraint among shared memory, registers,
threads and the hard CTA cap.

<a id="note-l203"></a>**THE PER-ARCH REGISTER BUDGET** is the largest per-lane register count that still admits
`target_ctas` CTAs of `threads` threads, honouring the allocation quantum. It must NEVER be a global literal: at 256
threads and 3 CTAs it is 80 on every part with 65536 registers an SM, and the arithmetic that produced it is the
table's, not a measurement session's.

<a id="capsdigest"></a>**`caps_digest`** is a digest of the whole active row, written by a kernel and checked on the
host. A binary compiled against a different table cannot masquerade as this one -- no path or file hash can establish
that, only a device-side fact.
