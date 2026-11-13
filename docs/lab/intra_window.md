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
TWO FORMS OF THE ROWS in global memory: TOKEN-MAJOR, the bundle's (the copies above), and TILE-MAJOR -- a tile of 64
tokens its blocks one after another, each block a token's 16 bytes a line, which is the layout a step reads -- where
a tile's level is one contiguous copy of 16 KB and its values one of 8. A third form is MIXED: the levels tile-major and the values
token-major, as a reader of a token's values at one run would have them.

The cell: a level's row is 8 live digits of byte 1 (a pair's weight at most 64, exact in bf16 and in every sum), a
value 1 or 2; every output value and every den sum is checked against the host's. Shared memory 114 KB.

MEASURED (H100 80GB HBM3, 132 SMs, vast.ai instance 60544158, driver 595.91.07; 2026-11-13, the experiment at
3e13195f; 7 repetitions, 8 windows; each form's 524,288 output values and 8,192 den sums exact; the median):

| the rows in global memory | SM-cycles a window | a trip | SM-cycles a token |
|---|---|---|---|
| token-major, a tile landed as 40 tensor copies of a kilobyte | 429,099 | 3,155 | 419 |
| tile-major, a tile landed as 3 contiguous copies | 177,498 | 1,305 | 173 |
| the levels tile-major, the values token-major: 2 contiguous copies and 8 tensor copies | 183,538 | 1,350 | 179 |

WHAT IT SHOWS.
- THE INTRA COMPOSES, EXACTLY, as a kernel of its own: the integer steps' operands as tensor copies land them, the
  weights in place as the value steps' A, the den as the 65th column, a token's line from registers.
- THE LEVELS' LAYOUT IS THE COST: token-major, a trip is 3,155 cycles where its chain alone is 985
  ([`intra_chain`](intra_chain.md)) -- 40 copies of a kilobyte a writer tile hold it -- and with the levels
  tile-major it is 1,350, the values' order then worth 44 cycles a trip (1,305 with the values tile-major too).
- THE VALUES STAY TOKEN-MAJOR: their order is 3% of the intra's window, and the carry's producers, who read a
  token's values as one run, lose most of a window to them tile-major ([`carry_window`](carry_window.md)). The
  levels tile-major and the values token-major is the layout both readers' measurements give.
- BESIDE THE CARRY (179 against its 817 SM-cycles a token with no clock inside a role,
  [`carry_window`](carry_window.md)) the intra in that layout is about a fifth of the carry's window. Adjacent
  measurements of two labs at their own cells, not a call: the tables here are 2 MB a level, cached, and no pair is
  skipped.
- NOT MEASURED: a window whose tile pairs are skipped by the facts pass's unions; rows the cache does not hold; the
  two kernels launched together.

Record: `measure/lab/results/intra_window/sm_90a.json`.
