# `csrc/rola/src/chunk/chunk.cu` — the chunk consumer's dispatch and torch entries

Adopted from the ladder's rung 6G; the ladder repo (rola-wt-ladder) holds the
rung-by-rung derivation and every measurement receipt behind this kernel's
shape.

The TU's one structural rule, inherited from the consumer family: **it never
touches a kernel symbol.** Arms — the launch, the table launch and the census —
arrive as extern template declarations satisfied by the instantiation shards
(`chunk_s*.cu`); even `chunk_build_stamp`'s register/spill census goes through
`query_chunk_census_arm` so this TU cannot re-instantiate a kernel locally and
leave the build green with nothing sharded.

Entries (all bound in `rola_api.cpp` with the `chunk_` prefix, private seam
only): `union_table` (the schedule producer — one packed bit-row per (head,
side, table entry)), `forward_raw` (the readout's `d_v+1` columns, undivided),
`forward` (table + kernel + divide), `divide_pass`, `build_stamp` (device-side
build identity: magic + form words, then the census row), `arms` (the built
manifest with derived geometry).

The arm set is the generator-owned `CHUNK_ARMS_X` macro
(`../instantiations/chunk_arms.inc`, emitted by `tools/gen_shards.py` — one
list, one owner): 21 `(C, BC, D, B)` arms — 15 two-level, three deep
((16,16,16) and (8,8,8,8) at matched N=4096, and the BC=32 depth-4 arm that
exercises the three-clause form), and three flat D=1 arms (the benchmark's
in-kernel flat-routing baseline, N=B≤256).  The swizzle axis is RETIRED
(measured −2.1..+2.2%, noise, no winner — one layout ships).

## Decay is not a feature of this kernel (retired 2026-08-13)

The mass-clock decay phase was designed, built and measured on the ladder: the
decay-off specialization was free (−0.7..0.0%), but the mass clock cost
+63–93% wall dense/sparse — the C-deep serial per-column scan is a bare
dependency chain plus one barrier per live chunk.  That price fired the
stop-gate: decay ships NOWHERE in the operator, and a decay-configured call is
routed to the fp64 reference arm at the `rola_op` seam (direct consumer
entries refuse loudly).  Reopen candidates, in order of promise: a parallel
prefix across the scan rows, scan/produce overlap, dressing fused into the
produce rendezvous, coarser clock granularity (a semantic change) — any of
them justified only by a quality measurement showing decay matters at scale.
