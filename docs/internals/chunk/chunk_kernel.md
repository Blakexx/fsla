# `csrc/rola/src/chunk/chunk_kernel.cuh` — the chunk consumer's device world

The ladder rung-6G kernel, adopted whole: `LevelPlan<D,B,BC>` (the
single geometry authority — spans, entries, clause count, the mixed-radix
block→entry map, all-int), `Geom` (SMEM layout and the derived staging depth),
the compacted kernel (form/consume schedule, per-side compaction, K-windows as
folded displacements, level-generic produce with the decoupled write gain) and
the union-table kernel.

Where its reasons live: the ladder repo's rung ladder (5A→6G) is the
derivation record — seven refuted alternative designs, the placement/mutual-
cover/wall-coupling laws, and the wall prices — and the development journal
carries the rulings. This tree's own gates are the operative discipline:
`tools/chunk_sass_gate.py` (zero spill, zero banned body opcodes over the
shipped fatbin) and the conformance battery against `naive_rola`.

Capability layer: `arch_caps.cuh`/`design.cuh`/`ops.cuh` ride with the family
(same directory); dedup against this tree's `mma.cuh`/`cp_async.cuh` lineage
is deliberately deferred to the e2e cleanup step, so the adoption stayed a
verbatim behavioral port (ladder parity gate: tables bit-equal, one-block
arms exact).
