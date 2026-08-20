# `csrc/rola/src/facts/facts_launch.cuh` — the facts kernels' arm surface

One launcher template per facts kernel, included only by the instantiation
shards (the dispatch TU sees declarations alone, so an arm cannot be satisfied
locally and un-sharded). Torch-free: every CUDA error is returned and the
dispatch TU raises through C10.

<a id="the-grids"></a>**The three grids.** The union-table pass launches one
warp per token word (`kFactsThreads / 32` words per CTA) by `BH`; its `MASK`
axis is the prefill call class, refused — `cudaErrorInvalidValue` — when the
stateless class is handed an atom table it compiled no emission for. The stats
reduction launches one CTA per `(bh, upper prefix)` at the invariant
`kAtomLeaves` grain, so its arm list is `(D, B)` alone. The liveness epilogue
re-instantiates the same reduction at the launch's `BC` grain, one CTA per
`block_bitmap_group` of blocks per `bh`.
