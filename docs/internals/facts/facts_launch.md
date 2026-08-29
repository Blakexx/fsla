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

## Comments moved from source (S5b)

Verbatim `//:` prose blocks from `csrc/rola/src/facts/facts_launch.cuh`, in source order, one heading per declaration. Source keeps only a short decl block pointing here.

<a id="ktblwarps"></a>
### `kTblWarps`

ONE PASS, BOTH ROW SETS.  `MASK` is the CALL CLASS and never a bool per fact:
the stateless class compiles no atom-grain emission at all, and within the
stateful one a call with no arena to admit into hands a null atom table -- the
runtime-null OUTPUT skip, not a second arm.

<a id="note-l34"></a>
### near line 34

THE LEX ARM, unchanged as a SURFACE: the dispatch vocabulary is still
`(D, B, BC)` x call class, and the plan it names is the lex policy's.  The
SPAN arm below is the same launcher under a plan whose spans are given, which
is the whole of what K31-i-d needed.

<a id="note-l45"></a>
### near line 45

THE GIVEN-SPAN ARM -- the box lattice's owner rows, AT ONE PARTITION PER SIDE
(K35 S1b).  `R0..R3` are the READ side's per-level runs and `W0..W3` the WRITE
side's; a level past `D` passes `B` (the identity), and a level a side spans whole
emits no rows there.  The carry consumes these rows, and there is no atom grain
here: residency is the paging layer's question and the carry family does not ask it.
