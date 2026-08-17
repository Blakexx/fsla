# `csrc/rola/src/chunk/chunk_launch.cuh` — the arm surface

Included ONLY by the instantiation shards. Defines `launch_chunk_arm`
(configures the arm's dynamic-SMEM ceiling once, launches),
`launch_chunk_facts_arm`, the two bitmap arms, and `query_chunk_census_arm`
(the attributes + derived geometry `build_stamp` reports). Torch-free on the consumer-launch rule: shards carry
no host framework; every CUDA error is RETURNED and the dispatch TU raises
through C10. `chunk.cu` must never include this header — an arm satisfied
locally by the dispatch TU would leave the build green and nothing sharded.

## <a id="the-grid"></a>The grid

`launch_chunk_arm` is the only place the grid is chosen, and there is one shape:
`(blocks, BH)`, one CTA per block. LIVENESS IS NOT A LAUNCH SHAPE — the bitmap is a
pointer the CTA exits on (`chunk_kernel.md#liveness-exit`), so a call that hands one
over launches exactly the grid a call that does not hands over.

The launcher asserts the state pointers and the compiled variant agree: a state pointer
handed to the variant with no state code would be read by nothing, and that is
`cudaErrorInvalidValue`. `launch_chunk_facts_arm` asserts the same agreement for its
own compiled-out half: an atom table handed to the stateless call class would come
back unwritten (`chunk_kernel.md#the-facts-pass`).
