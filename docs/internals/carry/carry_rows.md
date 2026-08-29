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

## Comments moved from source (S5b)

Verbatim `//:` prose blocks from `csrc/rola/src/carry/carry_rows.cuh`, in source order, one heading per declaration. Source keeps only a short decl block pointing here.

<a id="note-l24"></a>
### near line 24

THE OWNER BALLOT ROWS, DECLARED (K31-i-d).  The facts family owns the pass and
its instantiation shard owns the arms; this TU sees a DECLARATION, exactly as the
facts dispatch TU does, so nothing here can satisfy an arm locally.  The SPANS the
carry asks for are `BoxPlan::s(l)` -- the lattice's balanced owner runs, which is
the vocabulary the lex plan could not state.

<a id="siderowplan"></a>
### `SideRowPlan`

the plan ONE SIDE's rows are voted under (K35 S1b addendum): that side's own
partition, at its own run size, with a level it spans whole emitting nothing.  The
two sides are independent -- each votes its bitlists from its own factors -- so a
stream's test reads exactly one row per tested level and only the owner's whole-run
OR spans several (`CarveRows`).

<a id="kseamcarvestreams"></a>
### `kSeamCarveStreams`

THE SINGLE-LEVEL LEAF SPLIT the seam's `owner_rows(carve=True)` diagnostic names --
the innermost level's owner run over `kSeamCarveStreams` streams, which is the grain
that entry point documents.  It is a DIAGNOSTIC's declared grain and not an arm's
carve, so it is spelled as a span vector here and derived from nothing.

<a id="rows-entries"></a>
### `rows_entries`

THE TABLE: `[BH, entries_read + entries_write, L/32]` packed bits, the READ plan's
rows and then the WRITE plan's, one row per (level, run of THAT side's partition).
The two plans are independent (K35 S1b addendum): a side's bitlists are voted from
its own factors, so nothing is gained by refining one side's table by the other's
carve.  Where the two plans coincide the block IS the old `[BH, 2, E, L/32]`.
Also a SEPARATE ENTRY POINT, so the grain the kernel consumes and the grain the
seam names can be compared bit for bit (`tests/k31/test_owner_rows.py`).

THE ALLOCATION IS THE CALLER'S (K36): this launches into a table the caller sized
with `rows_entries` and owns.

<a id="near-line-36"></a>
### near line 36

the plan the carry's rows are voted under, named ONCE: a `BoxPlan`'s spans as a
`SpanPlan`.  `D` is at most four and a level past `D` carries `B`.
