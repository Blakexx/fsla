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

UNMEASURED: no hopper card has run it (`bazel run //tools:lab -- pack byte_expand --arch sm_90a`, then `carried` on
the card, under its kill timeout). It separates the expansion from the barrier waits in `row_ring`'s producer row.
