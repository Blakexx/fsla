# `csrc/rola/src/chunk/chunk_api.cuh` — host entry declarations

The `host_api.cuh` convention applied to the chunk family: declaration-only,
so the pybind TU (`rola_api.cpp`) never sees a device body. Nothing here is a
public API — `rola/ops/chunk.py` is the only intended caller, behind
`rola_op`'s facade.
