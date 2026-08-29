# `carry/carry_rows.cuh` — the owner ballot rows

The row plans the carry family votes under, and the one call that materializes
them. This header exists because of a translation-unit boundary, not because of a
new idea: since K36 the dispatch (`carry/carry.cu`) and the per-arm translation
units (`instantiations/carry_arm_<i>.cu`) are different objects, and both need the
row vocabulary. Putting it here is what lets the dispatch keep the `owner_rows`
entry point without including the kernel header — which is the whole point of the
split, since including that header is what used to cost the dispatch its 23 arms.

## What is here

* `OwnerRows<D, B, K, M>` — the lattice's balanced owner runs as a `SpanPlan`, the
  vocabulary the lex plan could not state.
* `SideRowPlan<SV>` — ONE SIDE's rows, at that side's own partition and run size.
  The two sides are independent: a side votes its bitlists from its own factors, so
  a stream's test reads exactly one row per tested level (K35 S1b addendum).
* `SeamSplitPlan<D, B, K, M>` — the single-level leaf split that
  `owner_rows(carve=True)` documents. It is a DIAGNOSTIC's declared grain, not an
  arm's carve, so it is spelled as a span vector and derived from nothing.
* `rows_entries<RPR, RPW>()` / `launch_rows<RPR, RPW>(...)` — how many rows the
  table needs, and the launch that fills it. The table is `[BH, entries_read +
  entries_write, L/32]` packed bits, read rows then write rows. **The allocation is
  the caller's**: this header is included by the per-arm translation units, which
  must not parse the tensor library
  ([`carry_arm_abi.md`](carry_arm_abi.md#pointers-not-tensors)), so `carry.cu`
  wraps these two in the `build_rows` that returns a tensor.

## The facts launcher is a DECLARATION here

`rola::chunk::launch_span_facts_arm` is declared, never defined. The facts family
owns that pass and its instantiation shard owns its arms, exactly as the facts
dispatch TU sees it — so nothing on the carry side can satisfy a ballot arm
locally, which would leave the build green and nothing sharded. Two gates close
that hole (`tools/gen_shards.py --check`, `tools/fatbin_entries.py --check`).

Nothing in this header instantiates a carry kernel, so it costs the dispatch a
parse and no codegen. That is the property the split depends on and the reason the
row plans are here rather than in `carry_arm.cuh`.
