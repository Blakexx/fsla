# `byte_expand`

Mirrors `measure/lab/experiments/byte_expand.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

What does a warp take to expand a tile of u8 bytes into bf16 times each token's fp32 scale, with nothing else in its
loop -- no ring, no barrier?

THE FORM. 1, 4 or 8 warps a CTA, a CTA an SM. Each warp expands its own 1 KB of bytes (64 tokens × 16 K, in shared
memory) into its own 2 KB operand tile, over and over: [`row_ring`](row_ring.md)'s expansion -- a lane's two tokens, a
byte to `2^23 + byte` by one `PRMT`, the product by one `FFMA`, a pair to a bf16 word by one conversion. A lane stores
its bytes from registers and loads them back every round, as a landed copy's would be read. The last tile is checked
against the host's.

MEASURED (H100 SXM, 132 SMs, a rented box: vast.ai instance 60165936, driver 595.91.07; 2026-11-11, one session, 7
repetitions, the last tile exact against the host's; cycles a 1,024-element tile a warp, the median, no spread):

| warps a CTA | cycles a tile a warp | cycles a tile an SM |
|---|---|---|
| 1 | 202.1 | 202 |
| 4 | 224.1 | 56 |
| 8 | 448.1 | 56 |

WHAT IT SHOWS. A warp alone expands a tile in 202 cycles; four warps an SM each take 224 and eight each take twice
that, so an SM expands a tile every 56 cycles however many warps do it past four. The tensor pipe takes a tile in 32
([`wgmma_rate`](wgmma_rate.md)): AN SM SPENDING ALL ITS ISSUE ON THIS EXPANSION FEEDS ITS OWN TENSOR PIPE AT 57% of the
pipe's rate, where every B element is expanded for one wgmma -- and the warps that issue the wgmmas issue too. Of
[`row_ring`](row_ring.md)'s 773 cycles a tile a producer warp, 202 are the expansion and the rest its hand-over.

Record: `measure/lab/results/byte_expand/sm_90a.json`; its counters are the written exception (no profiler on that
box).
