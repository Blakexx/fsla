# `carry.cu` — the carry family's dispatch, census and arm list

The mirror of `csrc/rola/src/carry/carry.cu`. One translation unit: the torch entries,
the arm dispatch, the per-arm register/SMEM census and the built-set listing.

## The arm list is small on purpose

R1 answers the carve question at ONE topology, and an arm nobody measured has no
business in a closed-world binary. `CARRY_ARMS_X` is the single X-macro the dispatch,
the census and the listing all walk, so the three cannot drift apart — the shipped
family's `CHUNK_ARMS_X` discipline.

| arm | topology | `k` | `m` | `s_l` | `BC` | `W` | `C` | carve |
|---|---|---|---|---|---|---|---|---|
| A | `(256,256)` | 4 | 8 | (16,8) | 128 | 256 | 32 | state-split |
| B | `(256,256)` | 4 | 8 | (16,8) | 128 | 256 | 32 | channel-split |
| C | `(256,256)` | 4 | 4 | (8,8)  | 64  | 256 | 32 | state-split |
| A64 / B64 / C64 | `(64,64)` | as above | | | | 64 | 32 | |

A, B and C are the R0 solver's flagship row and its occupancy-leaning sensitivity row
(`K31_R0_ABI_RIPPLE_SPEC.md` §7.4). The `(64,64)` triple exists because the fp64 oracle
materializes `[B,T,H,N]`: `N = 4096` is where a whole cell — output, denominator AND
final state — is checkable, and `N = 65536` bounds what a reference can hold at all.

## The build stamp is read off the device

A path check or a hash check cannot catch an extension trap; only a device-side fact
can. `build_stamp` launches a kernel that returns the source's magic, the carve's thread
geometry and the architecture capability digest, and appends the arm's
`cudaFuncGetAttributes` census beside them: registers, shared bytes, binary version,
the geometry's own byte count, the launch bound, LOCAL bytes (which must be zero),
`BC`, the owner count, the derived ring depth and the CTA target. Every gate in
`tests/k31/` asserts the magic before it asserts anything else.

## What this file deliberately does not do

It does not reach `rola::chunk` and `rola::chunk` does not reach it. It is registered in
`rola_api.cpp` as its own entries and is absent from every DAG, plan and rule in
`rola/engine/`. The seam is `rola/ops/carry.py`, which is a bring-up surface and not a
shipped path.
