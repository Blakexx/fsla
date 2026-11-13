# `intra_window`

Mirrors `measure/lab/experiments/intra_window.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

[`intra_chain`](intra_chain.md) measured the intra's chain a tile pair on constant tiles. What does the intra cost a
WINDOW on hopper as a kernel of its own, on real rows -- every pair of a window's 1,024 tokens whose writer is not
after its reader, exact against the host -- and so what is its share of a call beside the carry's
([`carry_window`](carry_window.md))? The hopper intra is a kernel run beside the carry, not a role in its CTA
([`ARCHITECTURE.md`](../ARCHITECTURE.md#64-the-intra--the-within-window-term)); this is its first form.

THE FORM. A CTA a tile of 64 READERS, 16 CTAs a window, a CTA an SM: one compute warpgroup and one loader warp.
- THE ROWS ARE STAGED BY TENSOR COPIES. A level's rows are token-major in global memory, 256 bytes a token; a step
  reads its operand 16 bytes a token a line, 8 tokens a brick. So a copy is a BOX of 64 tokens by one 16-digit
  block, a kilobyte landed as 64 consecutive lines, and a step's descriptor finds a token's second 16 digits one
  block on. The values likewise: a box of 64 tokens by 8 channels, a line a writer's 8 channels. The readers' two
  levels land once a CTA (32 copies); a tile of 64 WRITERS -- its two levels and its values, 40 copies issued in
  two instructions, a lane a copy -- lands a trip ahead of its use, two slots deep.
- A TRIP is `intra_chain`'s tile pair on those tiles: each level's pair products as 8 integer steps, the levels
  multiplied in fp32, the pairs whose writer is after its reader dropped on the diagonal tile, the weight rounded to
  bf16 in place as the A of 4 value steps. The value steps are 72 columns wide: the 65th column is the den's, a
  constant brick of ones behind the values' 8.
- A CTA's last act is its 64 tokens' lines: f16 pairs added to the output and the den's sums. Reader tile `i` runs
  `i + 1` trips, so a window is 136.
The cell: a level's row is 8 live digits of byte 1 (a pair's weight at most 64, exact in bf16 and in every sum), a
value 1 or 2; every output value and every den sum is checked against the host's. Shared memory 114 KB.

MEASURED: not yet run.
