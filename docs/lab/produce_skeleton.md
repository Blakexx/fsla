# `produce_skeleton`

Mirrors `measure/lab/experiments/produce_skeleton.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

Does the library's producer skeleton ([`produce.md`](../internals/produce/produce.md)) -- its grid, its two math warpgroups and loader warp, its ring's two edges, its job table and the density's ticket -- write every element of every output for every routing of the test matrix, and what does the empty pipeline cost at the design's cell?

THE FORM. The experiment compiles the library's `produce/produce.cuh` into one kernel a lo-plane variant, a CTA an SM (the ring's three stages reserved). THE CHECK: every output, and a guard of 256 bytes past each, filled with a sentinel; the launch, twice, so the ticket and its counts must leave themselves zero; then every output byte zero, every guard byte the sentinel, each density byte 0 where its level-side is masked and 255 where not. The plans: the five routings of `tests/integration/test_production_levels.py` (`tied`, `mixed_per_level`, `dense_only`, `both_sparse`, `grouped`) at alpha 1.5 and 2, at two shapes (3 sequences of 40 tokens, 2 write heads, hidden 96, 16 value channels; 2 of 300, 3 write heads, hidden 64, 24 channels: a tile past a sequence's end, a ragged mask word, a K tail), without and with the lo plane, and the timed shape. THE ROWS: the timed shape (B 1, T 65,536, H_w 4, G 1, hidden 256, widths 256 and 256, d_v 64, every level-side entmax 1.5 and untied: the design's cell), without and with the lo plane, by events, and over its outputs' bytes at HBM's write rate.

PREDICTION, the run to come (written 2026-11-28, before it ran; its basis: the outputs at the cell, rows 4 x 65,536 x 4 x 256 = 268.4 MB, scales 4 x 1.05 MB, mask words 4 x 8.39 MB, values 33.6 MB: 339.7 MB at HBM's write rate (stream_rate, 3,170 GB/s), 107.2 us; the lo plane adds the rows' 268.4 MB, 608.2 MB and 191.9 us. A CTA's 165,888 bytes take 12,800 cycles at an SM's share of that rate, its 20 ring stages ~2,000 and its ~18K warp instructions ~4,500 of issue (tools/budgets/produce.json): an SM's stores outrun its share, so the launch is write-bound; up to ~10% more where the scales' 4-byte entries, interleaved by write head and written by CTAs ~27 us apart, leave L2 as partial sectors (33.5 MB if each is a read-modify-write)):

- the empty pipeline at the cell: 107.00-124.00 us.
- the empty pipeline at the cell: against the write floor: 1.00-1.16 times.
- the empty pipeline at the cell, the lo plane: 192.00-215.00 us.
- the empty pipeline at the cell, the lo plane: against the write floor: 1.00-1.12 times.
