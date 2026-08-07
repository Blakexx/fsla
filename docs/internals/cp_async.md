# `csrc/rola/src/cp_async.cuh` — the asynchronous copy primitive

Three operations over one PTX instruction family: a predicated 16-byte global-to-shared
copy, a commit, and a drain. **No RoLA semantics live here** — the only caller is the
consumer kernel's amplitude prefetch, and every decision about WHAT to copy, WHEN and
under what predicate is that caller's.

Symbols: a *commit group* is the set of `cp.async` instructions a warp issued between
two `cp.async.commit_group`s; *src-size* is the instruction's third operand, the number
of bytes actually read from global memory, which may be smaller than the copy size.

---

## <a id="zfill"></a>1. The predicate is src-size, not a branch

`cp_async_16(dst, src, live)` passes `live ? 16 : 0` as src-size. The hardware
zero-fills whatever the source did not supply, so a copy with `live == false` writes
sixteen zero bytes to `dst`.

This is CUTLASS's `cp_async_zfill` (`cutlass/arch/memory_sm80.h`) and it is taken for
its correctness property rather than to save a branch. A predicate expressed as
control flow would leave `dst` holding the PREVIOUS window's bytes, and every consumer
of the slab would then need the mask that decided the skip in order to know whether
what it read was real. With src-size the destination is always written, and "the atom
was dead" and "the coefficient is zero" become the same fact — which is exactly the
self-masking property the whole executor is built on.

The address is still FORMED when the predicate is false. Callers therefore CLAMP an
out-of-range index rather than relying on the copy not to dereference it; the
consumer's ragged-group tail does this with the token index.

## <a id="width"></a>2. Sixteen bytes is the only width

It is the widest `cp.async` and it is the width the tree's
[sector invariant](smem_layout.md#prefetch-slab) is stated in — one atom of bf16
digits is 32 bytes, the fetch floor and the MMA k-quantum at once. Both the shared
destination and the global source must be 16-byte aligned, and both callers discharge
that STRUCTURALLY rather than at runtime: the slab is the last region of the carve and
is aligned by `align16`, and the amplitude rows are padded to the same quantum by
`Plan.amplitude_pitch`, checked once in the binding.

`.ca` (cache in L1) rather than `.cg`: neighbouring tokens' amplitude sectors share
cache lines under any routing that is not fully incoherent, and the streams are re-read
across owners.

## <a id="completion"></a>3. The completion model, and why the wait sits where it does

`cp.async.wait_group 0` drains every outstanding group **of the calling warp**, and the
data becomes visible **to the executing thread**. Neither is a CTA-wide statement:

- commit/wait state is per warp on every architecture that has `cp.async`, which is why
  the consumer issues ONE COMMIT GROUP PER WARP PER VISIT — there is one drain point, so
  a partial-fill ordering would have no consumer;
- cross-thread visibility takes a barrier. The consumer never adds one: it places the
  drain immediately before the end-of-window `__syncthreads()` that already exists,
  which is the same idiom CUTLASS's multistage mainloop uses.

There is no `cp.async.wait_group N` wrapper here, and its absence is deliberate: a
partial wait is only meaningful with more than one group in flight, and the consumer's
slab is consumed at a single point.
