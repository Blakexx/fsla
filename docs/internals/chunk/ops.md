# `csrc/rola/src/chunk/ops.cuh` — the device op helpers

The ladder's instruction-level helpers: `mma` (m16n8k16 bf16→fp32),
`load_frag`/ldmatrix wrappers, `cp_async` staging, `gather_ptr`/`pin_address`
(the pinned-base + zero-extended-32-bit-displacement addressing discipline the
SASS gate patrols), ballot/scan primitives, packed bf16 arithmetic. Overlaps
in role with this tree's `mma.cuh`/`cp_async.cuh`; the dedup is the e2e
cleanup step's, after which one helper family serves both consumers.
