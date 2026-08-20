# `csrc/rola/src/chunk/arch_caps.cuh` — the capability rows

The ladder's per-architecture capability table (P58 lineage), imported with
the chunk family: one `Caps` row per SM generation (async copy, tile loads,
MMA operand sourcing, TMEM, SMEM/register budgets), consumed by `design.cuh`'s
derivations. The sm_90 row builds and announces capabilities the 32-thread
consumer does not yet use (wgmma-class residency, TMA staging) — an
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
FREE axes (the chunk size, the shared-memory layout) multiply the binary, and
those are measured rather than derived.  Nothing here is ever a template
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
