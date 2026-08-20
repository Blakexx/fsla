# `backward.cu` — the reverse carry family's dispatch and census

The mirror of `csrc/rola/src/carry/backward.cu`. One translation unit: the torch entry,
the arm dispatch, the two-stream launch and the per-kernel census.

## The arm list is the forward's, without the carve

`BWD_ARMS_X` carries the same `(topology, k, m, W, C)` rows `CARRY_ARMS_X` does, because
the two families walk one window grid and rebuild one schedule. The carve scalar is
absent: the backward's carve is LEAF-split by derivation (its readouts contract over
channels), so it is a consequence of the arithmetic rather than a value a caller can ask
for.

## The two scans are launched concurrently

Scan A takes a stream from the pool and Scan B keeps the caller's; an event forks and
another joins. That is the whole content of the two-kernels resolution — the constraint
asks for two resident slabs, and two concurrent launches put both in the machine while
neither thread holds two. Every tensor Scan A touches is recorded against the side
stream, so the caching allocator cannot reuse a buffer under a launch it does not know
about.

The concurrency is a RESIDENCY claim, not a wall-clock one: both scans saturate the card
at the flagship, so the measured pair wall is close to their serial sum.

## Every gradient plane starts at the identity

There is no output any owner writes WHOLE — `d_pr`, `d_pw`, `d_gw` and `d_v` are all
per-token dense rows written by every owner that touches the token, exactly as the
forward's `o` and `den` are. They are therefore allocated zeroed. Only `d_state_in` is
exclusive: one owner owns its leaves, and it is written by the page sweep rather than
reduced.

`replay` is Scan A's slab, and it exists for one gate: the same state the forward's own
walk ends with, bit for bit. It is allocated only when asked for, because a training
call has no use for it.

## The census reports both kernels

`backward_build_stamp` returns the device-side magic, the thread geometry and the
capability digest, then twelve numbers: registers, LOCAL bytes, SMEM, ring depth, CTA
target and the DECLARED floor, for Scan A and then for Scan B. LOCAL bytes must be zero
for both; the pair of numbers a review reads is the measured register count beside the
floor the build declared.
