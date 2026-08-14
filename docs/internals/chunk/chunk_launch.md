# `csrc/rola/src/chunk/chunk_launch.cuh` — the arm surface

Included ONLY by the instantiation shards. Defines `launch_chunk_arm`
(configures the arm's dynamic-SMEM ceiling once, launches), `launch_chunk_
table_arm`, and `query_chunk_census_arm` (the attributes + derived geometry
`build_stamp` reports). Torch-free on the consumer-launch rule: shards carry
no host framework; every CUDA error is RETURNED and the dispatch TU raises
through C10. `chunk.cu` must never include this header — an arm satisfied
locally by the dispatch TU would leave the build green and nothing sharded.
