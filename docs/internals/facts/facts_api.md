# `csrc/rola/src/facts/facts_api.cuh` — the facts layer's binding declarations

The `host_api.cuh` convention: declarations only, so the pybind TU
(`csrc/rola/rola_api.cpp`) never sees a device body. The three entries are
`facts.md`'s; the binding names (`chunk_facts`, `chunk_block_bits`,
`chunk_arms`) keep their pre-K31 spellings so the python facts layer
(`rola/engine/facts/`) did not move with the deletion.
