# `mask_pack`

Mirrors `measure/lab/experiments/mask_pack.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The producer writes a level-side's mask as 32-token words, a bit the u8 digit is nonzero, from the epilogue's registers. What does the packer cost a 64-token block of 256 digits in a `wgmma` m64n256 accumulator's layout?

THE FORM. A warpgroup holds its block's quantized values in that layout, loaded before the clock; a digit is nonzero where its value rounds above 0. A ballot a register index (8 tokens x 4 columns); lane q keeps its column group's four ballots; each of its 8 columns' 16 token bits are every fourth bit of two ballots, gathered by three shift-and-mask rounds; a warp's halves go to shared memory as one 16-byte store a lane; a named barrier joins the warp pair of a 32-token half; each thread writes 16 bytes of the half's 256 words. Every word is checked; densities 10%, 50%, 100%.

PREDICTION (written 2026-11-27, before the first run). A warp's 16 tokens x 256 columns: 128 compares, 128 ballots, ~160 keeps, 8 columns x 2 gathers x 8, ~20 more: ~560 instructions; a 64-token block, one warpgroup: 500-900 cycles, flat in density. Against the solve: 10-15% of [`newton_row`](newton_row.md)'s alpha 1.5 random block (560 instructions against a solve of 4K-6K cycles).
