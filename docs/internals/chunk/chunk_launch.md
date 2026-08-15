# `csrc/rola/src/chunk/chunk_launch.cuh` — the arm surface

Included ONLY by the instantiation shards. Defines `launch_chunk_arm`
(configures the arm's dynamic-SMEM ceiling once, launches), `launch_chunk_
table_arm`, and `query_chunk_census_arm` (the attributes + derived geometry
`build_stamp` reports). Torch-free on the consumer-launch rule: shards carry
no host framework; every CUDA error is RETURNED and the dispatch TU raises
through C10. `chunk.cu` must never include this header — an arm satisfied
locally by the dispatch TU would leave the build green and nothing sharded.

## <a id="persistent-grid"></a>The two launch shapes

`launch_chunk_arm` is the only place the grid is chosen, and it is chosen from the
`ChunkParams` pointers rather than from any property of the problem. Without a liveness
bitmap the grid is `(blocks, BH)` — one CTA per block, the launch this kernel has always
had. With one it is `(min(ceil(kCtaTarget * sm_count / BH), blocks), BH)`: a FIXED grid of
about this arm's resident CTAs, over which the kernel's work loop strides
(`chunk_kernel.md#work-loop`).

`sm_count` arrives through `ChunkParams` because it is a RUNTIME DEVICE FACT and never an
architectural constant — GA102 ships at 80 and 82 SMs, GA100 at 108 — so it must not reach
`arch_caps.cuh`, and a receipt taken on an 82-SM part must not be attributed to an 80-SM
one. It is a host struct field, so it does not move any kernel parameter's constant-bank
displacement.

The launcher asserts the pointers and the compiled variant agree, in both directions: a
state pointer handed to the variant with no state code, or a bitmap handed to the variant
with no work loop, would be read by nothing. `PERSIST != (block_bits != nullptr)` is
`cudaErrorInvalidValue`, exactly as the state mismatch is.

`per_bh` is DEVICE-DEPENDENT and must be reported as such: it is not a derived constant of
the arm, and a census that printed it beside `kCtaTarget` and `blocks` would imply
otherwise.
