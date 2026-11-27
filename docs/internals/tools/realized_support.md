# `tools/realized_support.py` — what a layer cell routes

A carry cell DECLARES its support (its routing's `k<n>` entries, its draw); a layer cell's weights REALIZE one, through
the producer's solve on the cell's hidden states (`measure.cells.layer.build`). A layer reading's regime is that
realized support, so the tool reads it from the bundle the kernels take, after quantization: per level and side, the
nonzero digits a token over every batch row, token and head -- the mean, the median, the 99th percentile and the
fraction of the level's width -- and whether the side is declared sparse (a dense side's zeros are values that rounded
to zero).

    bazel run //tools:realized_support -- [layer-alt-entmax1.5-256x256-T65536-h4] [--json support.json]   # one layer cell, or every one

No target records it: the producer it reads is not built on this line, so it would record only a refusal.

The cell's logit gain is the knob that moves it (`tests/integration/test_producer_sweep.py` holds the direction); the
central layer cells hold the long layer at gains 1, 4 and 16 (`layer-alt-entmax1.5-256x256-T65536-h4`, `-h4-g4`,
`-h4-g16`), so a layer-level comparison spans the support instead of reading one point of it.
