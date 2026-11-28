# `produce_skeleton`

Mirrors `measure/lab/experiments/produce_skeleton.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

Does the library's producer skeleton ([`produce.md`](../internals/produce/produce.md)) -- its grid, its two math warpgroups and loader warp, its ring's two edges, its job table and its tickets -- write every element of every output for every routing of the test matrix, and where does the empty pipeline's time go at the design's cell?

THE FORM. The rows launch the library's CTA (`produce_cta`; the library holds the CTA and the lab its launch until the op launches it; a CTA an SM, the ring four stages, or two CTAs an SM and two stages); each traced twin launches the same CTA through warpline's stamps -- every phase (entry, setup, a job's window and epilogue, the tickets, exit) and each ring barrier's wait and arrival -- and records two CTAs' timelines (`lab trace produce_skeleton`). THE CHECK, every row: every output (the tile-major rows, scales, masks a (row, level), values, growth, density) and a guard of 256 bytes past each filled with a sentinel; the launch, twice, so the tickets and their sums must leave themselves zero; then every output byte zero, every guard byte the sentinel, each density byte 0 where its level row is masked and 255 where not -- over the five routings of `tests/integration/test_production_levels.py` (`tied`, `mixed_per_level`, `dense_only`, `both_sparse`, `grouped`) at alpha 1.5 and 2, at two shapes (3 sequences of 40 tokens, 2 write heads, hidden 96, 16 value channels; 2 of 300, 3 write heads, hidden 64, 24 channels: a tile past a sequence's end, a ragged mask word, a K tail), and at the cell. THE ROWS, at the cell `layer-alt-entmax1.5-256x256-T65536-h4` (B 1, T 65,536, H_w 4, G 1, hidden 256, widths 256 and 256, d_v 64): the launch; at hidden 0 (no ring stage); at two CTAs an SM; with the lo plane. The floor is warpline's over the kernel's declared work ([`produce.md#floor`](../internals/produce/produce.md#floor)).

PREDICTION, run 20261128T081918Z-081208-723dc457 (written 2026-11-28, before it ran; its basis: warpline's floor at the cell is 101.9 us, its 323.0 MB at HBM's write rate (measure/floors/produce.work.json). THE HYPOTHESIS, NOT A FINDING until these rows name the price: at one CTA an SM a CTA's non-store time (launch, setup, its 20 ring stages' edges, the tickets, its exit) runs in series with its own writes, ~10K cycles a CTA (the CTA's ~25K warp instructions of tools/budgets/produce.json at ~2.5 a cycle) over 2,048 CTAs on 132 SMs, 15.5 waves at ~1.8 GHz: ~85 us past the floor, ~187 us; at hidden 0 the 20 stages' edges go (~100 cycles each, ~17 us a launch); at two CTAs an SM one CTA's serial time runs under the other's writes, back near the floor. The rival (the stub stores' instruction issue, their runtime division 29 of a trip's ~46 instructions) would leave hidden 0 and two CTAs an SM where the plain row is, its time inside the traced rows' epilogue phases; the traced twins' timelines (lab trace: each role's phases, the path) say which. The lo plane adds its 268.4 MB at the write rate):

- the empty pipeline at the cell: 170.00-205.00 us.
- the empty pipeline at the cell: over its floor: 1.67-2.01 times.
- the empty pipeline at the cell, hidden 0: 152.00-190.00 us.
- the empty pipeline at the cell, two CTAs an SM: 104.00-128.00 us.
- the empty pipeline at the cell, the lo plane: 255.00-292.00 us.
- the empty pipeline at the cell, the lo plane: over its floor: 1.37-1.57 times.

PREDICTION, run 20261128T095100Z-086748-f002e06d (written 2026-11-28, before it ran; its basis: the first run held at one CTA an SM (202.8 us, 1.99x the floor), at hidden 0 (186.2) and with the lo plane (267.1, 1.43x) and MISSED at two CTAs an SM (164.0 against 104-128): its traced twins name why. A lone CTA's 25-27K cycles are five job windows of ~4,550: the ring's empty edges ~1,350 (a K-step's round trip, the loader waiting on the release, the math warps on the arrival) and the EPILOGUE's stores ~2,900, 64% of the CTA, at ~11 bytes a cycle an SM; with no ring (hidden 0) a CTA still takes ~24K cycles and the launch writes at 1,736 GB/s, 55% of HBM's write rate; at two CTAs an SM each CTA takes ~40K, twice a lone one. So the price was not a serial non-store latency (the hypothesis, refuted as the main price) but the stub stores' instruction stream on the SM: ~46 instructions a 16-byte chunk, 29 of them the runtime division's dependent chain, with 8 math warps to hide it. THE STORE COMPONENT has no division (power-of-two-shaped runs: a row tile in a thread's 128-chunk stride, a share's 64 tokens two threads each): ~6 instructions a 16-byte store, ~14.2K warp instructions a CTA against ~25K (tools/budgets/produce.json). Its stores then run at the SM's write path, so a launch lies between the floor (101.9 us, every SM's writes overlapping the others' ring edges) and each CTA's serial sum (its five ring windows' ~6.75K cycles plus its 157.7 KB at HBM's write share, 13.2 bytes a cycle an SM: ~11.9K, and ~2K of setup and tickets: ~20.7K over 15.5 waves at 1.97 GHz, 163 us); at hidden 0 the sum is ~13.9K a CTA, 110 us; at two CTAs an SM one CTA's ring runs under the other's stores; the lo plane adds its 268.4 MB at the write rate):

- the empty pipeline at the cell: 110.00-165.00 us.
- the empty pipeline at the cell: over its floor: 1.08-1.62 times.
- the empty pipeline at the cell, hidden 0: 100.00-120.00 us.
- the empty pipeline at the cell, two CTAs an SM: 100.00-125.00 us.
- the empty pipeline at the cell, the lo plane: 190.00-250.00 us.

PREDICTION, the run to come (written 2026-11-28, before it ran; its basis: the store component's run held its coverage and MISSED every band (plain 204.4 us, hidden 0 195.7, two CTAs an SM 185.5, the lo plane 290.6): its stores' instructions fell ~25K to ~14.1K a CTA (ncu) and the launch did not move, so the instruction stream was not the price. ncu on the traced launch: DRAM 49% of its peak (321.5 MB written, the declared bytes), 28.8M warp instructions (~28 us of issue), no drain, store-queue or MIO stall, LONG SCOREBOARD 5 a issue, warps active 12% of peak: latency with ~2 warps a scheduler to hide it -- each 16-byte store's address reused by the next trip of a rolled loop, each ring edge a polled wait. THIS FORM: four 16-byte stores in flight a thread (each its own address), a token's run by consecutive threads (coalesced), the stub's ring one stage a job. A CTA's ~25K cycles were ~2,900 a job's epilogue (~10 latency-bound trips of ~290) and ~1,350 its ring edges; now ~4 trips a job (~1,200) and one round trip (~340): ~9.7K a CTA, under its 157.7 KB at HBM's write share (13.2 bytes a cycle an SM: 11.9K), so the launch meets the floor's write rate: 101.9 us and the tail's wave; hidden 0 and two CTAs an SM the same; the lo plane at its 186.6 us floor):

- the empty pipeline at the cell: 104.00-130.00 us.
- the empty pipeline at the cell: over its floor: 1.02-1.28 times.
- the empty pipeline at the cell, hidden 0: 102.00-128.00 us.
- the empty pipeline at the cell, two CTAs an SM: 102.00-128.00 us.
- the empty pipeline at the cell, the lo plane: 188.00-225.00 us.

MEASURED (NVIDIA H100 80GB HBM3, Nebius preemptible VM e00hhhhq5yervt2aev, driver 580.196.07; 2026-11-28, the experiment at
bcd48e72, run 20261128T095100Z-086748-f002e06d; 7 repetitions, the 4 checked variants exact; the median, the widest
spread of a row 6.0%; the clock as the card left it):

| the figure | measured | the prediction | held |
|---|---:|---|---|
| the empty pipeline at the cell | 204.38 us | 110.00-165.00 us | MISSED |
| the empty pipeline at the cell: over its floor | 2.01 times | 1.08-1.62 times | MISSED |
| the empty pipeline at the cell, hidden 0 | 195.71 us | 100.00-120.00 us | MISSED |
| the empty pipeline at the cell, two CTAs an SM | 185.50 us | 100.00-125.00 us | MISSED |
| the empty pipeline at the cell, the lo plane | 290.62 us | 190.00-250.00 us | MISSED |

The record's rows:
- the empty pipeline at the cell: 204.38 microseconds a launch.
- the empty pipeline at the cell, every warp's phases stamped: 223.58 microseconds a launch.
- the empty pipeline at the cell, hidden 0: 195.71 microseconds a launch.
- the empty pipeline at the cell, hidden 0, every warp's phases stamped: 205.70 microseconds a launch.
- the empty pipeline at the cell, two CTAs an SM: 185.50 microseconds a launch.
- the empty pipeline at the cell, two CTAs an SM, every warp's phases stamped: 203.94 microseconds a launch.
- the empty pipeline at the cell, the lo plane: 290.62 microseconds a launch.

Record: `data:records/fsla/measure/lab/produce_skeleton.run/20261128T095100Z-086748-f002e06d/output.json`.
