# `carry/box.cuh` — the CTA box's compile-time plan

Mirrors `csrc/rola/src/carry/box.cuh`.

`BoxPlan<D, DV, warps_per_cta>` holds every DERIVED quantity of one arm as a `constexpr`:
the box's leaves and pages, the accumulator's tile counts and register cost, the segment
grain and its slot count, and the shared-memory ledger's offsets and total. Nothing in it
is a literal, and every relation that could silently round instead refuses as a
`static_assert`.

Which class each quantity belongs to is the primitive ledger's table, in
[`carry_kernel.md#ledger`](carry_kernel.md#ledger); the accumulator's orientation and its
padding are in [`carry_kernel.md#accumulator`](carry_kernel.md#accumulator); the byte
offsets and the per-moment aliasing are in [`smem_ledger.md`](smem_ledger.md).

`Edge` names the frame's three split edges and lives here rather than in the body because
the ledger sizes their barriers; what each edge separates is in
[`carry_kernel.md#edges`](carry_kernel.md#edges).
