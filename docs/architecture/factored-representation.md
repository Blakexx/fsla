# Factored representation

> **Store the factors. Expand transiently, where they are consumed. Never
> materialize the expansion.**

Every large object in this engine is a Kronecker-style product of small
per-level objects. The rule is that it stays that way in memory: what is written
down is the `Σ_l b_l` factors, and the `Π_l b_l` expansion exists only inside the
consumer that needs it, in registers or shared memory, for as long as one tile
takes.

This was recorded as a **unification** — one principle that turns out to be the
memory rule, the schedule rule and the schedulability rule at once (campaign
journal, 2026-08-05, the affine-schedulability entry's closing amendment):
*"routing matrix AND its support are both Kronecker-factored objects — store the
factors, expand transiently where consumed (panel in registers, candidacy in
SMEM), never in HBM."*

## Why it is one principle and not four coincidences

The generative fact is that the leaf index is a **mixed-radix digit string**:
leaf `s` decomposes into per-level digits `d_l(s)` via the topology's radix
strides, and every per-leaf quantity in the model is a product over levels of a
per-digit quantity:

```
W[t,s] = g_write[t] · Π_l  p_write[l][t, d_l(s)]
R[t,s] = g_read[t]  · Π_l  p_read [l][t, d_l(s)]
rate[s] =             Π_l  delta_l[d_l(s)]
```

(`rola/ops/naive.py`, `_leaf_product`.) A product over levels is a rank-1 object
per level; storing the expansion stores `Π b_l` numbers to represent `Σ b_l`
numbers. At the design scale — `N ≳ 65K` states, the standing target — the
difference is the difference between a plan that fits and a plan that does not.

The **address side factors too**: `leaf_digit(p, l, base, i) = base +
(i / level_stride[l]) % level_modulus[l]` (`csrc/rola/src/consumer.cuh:219-221`)
is a pure function of the leaf index that reads no routing data. So the support's
*bitmap* is likewise a Kronecker product of the per-level indicators
([`decode.md`](../internals/decode.md), which expands two `N`-bit maps by
Kronecker broadcast rather than storing one).

## The instances it unifies

| Object | Factored form stored | Expansion, and where it lives |
|---|---|---|
| **Routing gates** | per-level amplitude arrays `[BH, T, b_l]` | the `[BT × BC]` coefficient panel, synthesized in registers inside the consumer; the gates are **never materialized** as `[L, N]`. The fused per-head routing path exists precisely so that they cannot be. |
| **The Kronecker ledger** — the leaf-index arithmetic | `radix_strides`, `level_stride`, `level_modulus` | `leaf_digit()` evaluated per element; the digit table is a per-owner hoist, not a stored map (`consumer.cuh:219-221`). |
| **Bitmap / support words** | per-level digit bitmaps and slice unions, sized by `Σ_l b_l` | candidacy is **evaluated** by a warp-local AND chain in shared memory (~0.2% of the instruction stream), never stored per owner. |
| **Panel factors** | per-level factors + the clock | the survival and readout panels, built per tile in registers, consumed by the MMA, discarded ([`intra_panel.md`](../internals/intra_panel.md)). |
| **The schedule** | the same per-level bitmaps | the per-owner work set, re-evaluated in kernel. Storing it would be the `O(L·N)` wall. |
| **The token-gather ballots** | the same factor words | live-token masks computed from the support words already loaded. |

The negative cases are as instructive, and both were priced and refused:

- **Precomputing the `W` panel to VRAM** — `O(candidates × BT × BC)` ≈ **8.6 GB**
  at the parity cell. Refused (journal, 2026-08-04 user Q&A, item 2). The
  implication of "the operand is computed" is *make it cheap* (affine addressing)
  and *hide it* (pipeline), never *materialize it*. This is FlashAttention's
  founding lesson applied to the operand instead of to the score matrix.
- **Per-candidate compaction** — the same `O(L·N)` wall in a different coordinate
  (8.59 GB at parity). Per-**token** compaction, `O(BH · T · support)`, is
  `N`-independent (13.5 MB against 320 MiB dense) and is therefore the admissible
  form (journal, 2026-08-04, adaptive-specialization §1B).

## When it does NOT apply

The rule is about **representation**, and it earns its keep only when the
expansion is (a) larger than the factors by a factor that grows with a design
axis, and (b) consumed in a bounded window. Where either fails, factoring is
waste or worse:

1. **When the expansion is consumed repeatedly at a distance.** The rule says
   expand *where consumed*. If the same expansion is needed by many consumers
   spread over time, re-expansion is recomputation, and the trade becomes a real
   measurement rather than an obvious win. The engine's own case where the
   staging transform *lost*: an amplitude staging buffer was designed to reshape
   the per-element gather, and refuted by a theorem plus a probe — each element is
   loaded **exactly once per visit**, so the buffer has no reuse to exploit; the
   measured load-reduction factor was `1.0030` work-weighted and exactly `1.0000`
   at the flagship (journal, 2026-08-04, `AMP_STAGING_DESIGN.md` declined). *There
   was nothing to amortize.*
2. **When the factored form is smaller than the hardware's quantum.** Below a
   32-byte sector or a 16-row MMA quantum, a finer representation cannot save a
   byte or an instruction. Measured: with fp32 amplitudes, 8 digits are one 32 B
   sector, so support granularities `G ∈ {1,2,4,8}` have *identical* sector
   counts; and `kMmaKQuantum = 16` makes `G ∈ {1..16}` identical in MMA rows
   (journal, 2026-08-04, support-granularity design). **The continuum collapses
   to the hardware's grid.**
3. **When the expansion is dense.** Factoring is a statement that most of the
   product is structurally absent. Where occupancy approaches 1, the dense form
   *is* the compact form, and the factored path pays index arithmetic for nothing.
   This is why the engine treats representation as a **planner-selected axis**
   with measured cost curves rather than a universal default — see
   [planner.md](planner.md).
4. **When it would make a declaration execution-affecting.** Expansion may be
   skipped only where the realized masks still decide membership; see the scope
   boundary of the [self-masking theorem](self-masking-theorem.md#7-scope-boundary).

## Provenance

- Unification statement: campaign journal, 2026-08-05, affine-schedulability
  entry, precision amendment.
- Leaf-digit purity: `csrc/rola/src/consumer.cuh:219-221`; bitmap Kronecker
  structure: [`docs/internals/decode.md`](../internals/decode.md),
  [`decode_expand.md`](../internals/decode_expand.md); reference semantics:
  `rola/ops/naive.py::_leaf_product`.
- Materialization refusals and the staging decline: campaign journal,
  2026-08-04.
