# `csrc/rola/src/chunk/chunk_s0.cu` — instantiation shard 0

8 of the chunk consumer's 24 arms, plus the union-table arms (one per
topology × BC). Hand-carried at P69 stage B; integrating the chunk matrix
into `tools/gen_shards.py` (membership hash, regenerate-and-diff gate) is the
stage-D item, and until then an edit here must update `chunk_kernel.cuh`'s
arms macro in step.
