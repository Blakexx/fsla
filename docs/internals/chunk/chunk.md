# `csrc/rola/src/chunk/chunk.cu` — the chunk consumer's dispatch and torch entries

Adopted from the ladder's rung 6G at P69; the ladder repo holds the rung-by-rung
derivation and every measurement receipt behind this kernel's shape.

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

The arm set is the `RUNG5D_ARMS` macro in `chunk_kernel.cuh`: 24 arms — 21
two-level plus three deep ((16,16,16) and (8,8,8,8) at matched N=4096, and the
BC=32 depth-4 arm that exercises the three-clause form.
