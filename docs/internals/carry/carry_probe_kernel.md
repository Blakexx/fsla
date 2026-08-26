# `carry_probe_kernel.cuh` — the segment layout's round trip

A named shared-memory layout is derived from BOTH of its access patterns and gated by a
round trip (`KERNEL_STANDARDS` §3). This header is that gate's device half for the five
layouts the K35 cooperative fetch introduces, and it exists as its own entry point so
the layout can be gated WITHOUT running the kernel that uses it — a conformance failure
in the fold then has one fewer thing it could be.

## What it does

One CTA, the arm's own thread count, five legs:

* **the fetcher's stores** — the same base-plus-displacement arithmetic `fetch_values`
  hands `cp.async` and `expand` hands `st.shared.u32`, writing a pattern whose sixteen
  bits NAME the `(row, column)` it was stored at;
* **the consumer's pulls** — `ldmatrix.trans` at the fold's own `myV`/`myC` bases,
  unpacked at the fragments' own thread-value maps and scattered to global;
* **the staged factor row** — `stage_factors`' arithmetic verbatim at owner zero,
  against a `[kSegRows][kWtot]` plane whose cells name their own column.  ONE body serves
  both sides (the fold stages from `pwrite`, the readout from `pread`), so this leg
  covers the read staging too;
* **the reader segment** (K35 S4b) — the READ expansion's `st.shared.u32` against the
  readout's `load_frag`, whose fragment map puts the segment's ROWS on the MMA's `n`
  axis, because on the readout the rows are TOKENS and the columns are the contraction;
* **the fan-in ring** (K35 S4b) — written at the SCATTER's cell address (four rows at one
  eight-channel granule) and read back at the FLUSH's (one row, thirty-two lanes over
  thirty-two consecutive channels), so a granule swizzle that disagreed between the two
  would come back as a permuted row.

The probe is instantiated PER ARM and therefore per `d_v` (K35 S3b), so the value
sub-tile's leg runs at the arm's own `LdV` and `kChunkV` — 72 and 8 quanta at the narrow
plane, 136 and 16 at the wide one — and the round trip is what says the fetcher's stores
and the consumer's `ldmatrix.trans` agree at BOTH widths rather than at one.

The pattern is a RAW bf16 BIT PATTERN and never a number: what is being exercised is
movement, so the only property asked of a value is that it be distinct per cell. That is
also why the staged leg has to be EXECUTED rather than modelled — a misaligned
asynchronous copy is a fault, not a wrong answer.

## What it cannot see, and where that is covered

A round trip sees the ANSWER, not the cost: it cannot observe a bank conflict. The bank
behaviour of each leg is therefore derived from the same address functions in
`tests/k31/test_segment_layout.py`'s model leg, and the table of results is in
`carry_kernel.md` under "The segments, and the two facts of the fold's sync".

## Surface

`rola::carry::layout_probe(widths, k, m, window, chunk, carve, level_modes)` →
`(values, coefficients, staged, reader_rows, ring)`, each `int32` of the recovered
pattern -- sixteen-bit for the four bf16 layouts, the fp32 cell's own integer value for
the ring.
CARVED arms only: the per-stream segment belongs to the carved body, and the
channel-split body is refused rather than served something else. `carve` is the seam's
DECLARATION scalar, so the same cell under a different EXECUTOR MAP is a different probe:
the consumer's leg reads the fold's own `wstrm`/`col`, so at `WPS = 2` the two warps of a
stream recover DISJOINT halves of one coefficient row and the round trip is what says the
column split partitions rather than overlaps.
