# `byte_expand`

Mirrors `measure/lab/experiments/byte_expand.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

What does a warp take to expand a tile of u8 bytes into bf16 times each token's fp32 scale, with nothing else in its
loop -- no ring, no barrier?

THE FORM. 1, 4 or 8 warps a CTA, a CTA an SM. Each warp expands its own 1 KB of bytes (64 tokens × 16 K, in shared
memory) into its own 2 KB operand tile, over and over: [`row_ring`](row_ring.md)'s expansion -- a lane's two tokens, a
byte to `2^23 + byte` by one `PRMT`, the product by one `FFMA`, a pair to a bf16 word by one conversion. A lane stores
its bytes from registers every round and loads them back (ptxas forwards one of the two loads). The checked forms'
last tile is the host's. Four forms each drop a class: the pair packed by truncation (a `PRMT` in place of the
conversion), no product, no store and reload of the bytes, no store of the tile (ptxas then forwards the remaining
load too, so that form drops the tile's stores AND the load). One more form changes only which tokens a lane takes --
`lane` and `lane + 32` in place of two neighbours -- so a warp's 16-byte accesses fall 16 bytes apart, on all eight
bank groups: UNMEASURED, owed to its first run.

MEASURED (H100 SXM, 132 SMs, a rented box: vast.ai instance 60170194, driver 595.91.07; 2026-11-11, one session, 7
repetitions, the checked forms exact against the host's; cycles a 1,024-element tile a warp, the median, no spread):

| the form | its 128-bit shared accesses a lane a tile | 1 warp | 4 warps | 8 warps |
|---|---|---|---|---|
| the expansion | 6 stores, 1 load | 202.1 | 224.1 | 448.1 |
| the pair packed by truncation | the same | | 224.1 | 448.1 |
| no product | the same | | 224.1 | 448.1 |
| no store and reload of the bytes | 4 stores | | 128.1 | 256.1 |
| no store of the tile | 2 stores | | 64.1 | 128.1 |

WHAT IT SHOWS. The arithmetic is not what the expansion costs: dropping the conversion or the product moves no row by
a tenth of a cycle. The rows follow the shared-memory accesses alone -- at four warps 32 cycles a warp for each
128-bit access a lane makes (7: 224; 4: 128; 2: 64), twice that at eight -- so an SM completes one warp-wide 128-bit
access every 8 cycles in this form, and its tile every 56. In this form a lane's two tokens are neighbours, which puts
a warp's 32 accesses 32 bytes apart: eight lanes on each of four bank groups, where a 512-byte access can take four
lanes on each of eight. THAT THE 8 CYCLES ARE THOSE 8 CONFLICTING LANES IS AN INFERENCE from the addresses; the form
with the lanes spread is the measurement, and until it runs 56 cycles a tile is this layout's reading, not the
expansion's floor.

Record: `measure/lab/results/byte_expand/sm_90a.json`; its counters are the written exception (no profiler on that
box).
