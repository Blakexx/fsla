# `csrc/rola/src/chunk/arch_caps.cuh` — the capability rows

The ladder's per-architecture capability table (P58 lineage), imported with
the chunk family: one `Caps` row per SM generation (async copy, tile loads,
MMA operand sourcing, TMEM, SMEM/register budgets), consumed by `design.cuh`'s
derivations. The sm_90 row builds and announces capabilities the 32-thread
consumer does not yet use (wgmma-class residency, TMA staging) — an
announcement, not a refusal, by the ladder's sm_90 ruling.
