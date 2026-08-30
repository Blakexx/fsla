# `carry_rows_kernel.cuh` — the owner ballot rows

The mirror of `csrc/rola/src/carry/carry_rows_kernel.cuh`.

One bit per (digit run, token): set when any amplitude in that run is nonzero. The
carry's walk ANDs these words to get a stream's live tokens, so the runs are the STREAM
spans of each side's carve — which K50 makes runtime.

<a id="why-the-carry-owns-this-pass"></a>
## Why the carry owns this pass now

Until G2 the table was voted by `rola::chunk::launch_span_facts_arm<D, B, s_r..., s_w...>`
— one instantiation per (topology, mode word), emitted into the facts family's shard. That
is exactly the arm-registration problem K50 removes: the spans are DATA now, so no template
can carry them, and the pass is one kernel per DEPTH.

It also closes a drift class by construction. `## F3`'s first wall was a carry-row
partition change leaving a carved arm asking for a `launch_span_facts_arm` instantiation
nothing emitted — a clean link and an undefined symbol at IMPORT. With the spans runtime
there is no instantiation to miss.

The reduction is the same law `facts_kernel.cuh` performs for the P80b bit emission, and
this file does not touch that pass: the two share a law (a run is live when its magnitude
bits are), not a translation unit. Folding them back together is G4's, once both sides of
the G milestone have landed.

<a id="tok-words"></a>
## `tok_words`

The table's minor axis: tokens are packed 32 to a word, which is the ballot's own width.
The carry family owns this constant now that it votes its own rows; the reverse pass
(`backward.cu`), whose arms still carry a compile-time `BoxPlan`, keeps reading the facts
family's copy.

<a id="or-run-rt"></a>
## `or_run_rt`

The OR of a contiguous run of RUNTIME width. The width is a power of two and the address is
a multiple of it — the seam refuses otherwise (`geom.md#refusals`) — so the wide path is
aligned and the narrow widths are one access each.

<a id="carry-rows-kernel"></a>
## The table's layout

`[BH, rows_read + rows_write, ceil(L/32)]` packed bits, the READ side's rows first. A level
a side spans WHOLE votes no rows at all — its clause is vacuous — which is what `row_g == 0`
means in the block.
