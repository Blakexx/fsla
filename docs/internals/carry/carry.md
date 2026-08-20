# `carry.cu` — the carry family's dispatch, census and arm list

The mirror of `csrc/rola/src/carry/carry.cu`. One translation unit: the torch entries,
the arm dispatch, the per-arm register/SMEM census and the built-set listing.

## The arm list is small on purpose

The carve question is answered, so an arm is a `(topology, W)` pair and nothing more.
`CARRY_ARMS_X` is the single X-macro the dispatch, the census and the listing all walk,
so the three cannot drift apart — the shipped family's `CHUNK_ARMS_X` discipline.

| arm | topology | `k` | `m` | `s_l` | `BC` | owners | `W` | `C` | SMEM | regs |
|---|---|---|---|---|---|---|---|---|---|---|
| flagship | `(256,256)` | 4 | 8 | (8,16) | 128 | 512 | 384 | 32 | 37 440 B | 176 |
| flagship+ | `(256,256)` | 4 | 8 | (8,16) | 128 | 512 | 512 | 32 | 37 952 B | 176 |
| small | `(64,64)` | 4 | 8 | (8,16) | 128 | 32 | 64 | 32 | 36 160 B | 176 |

`W` is the architecture doc's TWO-CONSTRAINT quantity. The C\* cost law
(`sqrt(v/kappa p)`) has a shallow basin around it, so moving off the C\*-optimal 256
costs the carry a few percent; the binding constraint is the INTRA kernel, which runs on
the same global window grid and whose ideal arithmetic intensity is `W/4` flop/byte —
`W >= ~380` is what clears sm_86's compute/bandwidth balance. The two families are
instantiated at the same windows for that reason.

The doc's admitted band is 384–512 and BOTH ends are built, because a THIRD constraint
decides between them and the doc does not state it: **the window must divide `L`.** The
intra kernel refuses a partial window per arm (its tile grid is a compile-time count),
and `384` does not divide the flagship length `L = N = 65536` — 65536 / 384 = 170.67.
`512` divides every power-of-two length and still clears the intensity bar with room
(128 flop/byte against the 94 balance). So `W = 512` is what the capacity-fair
`N = L = 65536` cell is measured at, and `W = 384` is the doc's nominal sizing, measured
at `L = 65280 = 384 · 170`.

The `(64,64)` arm exists because the fp64 oracle materializes `[B,T,H,N]`: `N = 4096` is
where a whole cell — output, denominator AND final state — is checkable, and
`N = 65536` bounds what a reference can hold at all. Its `W = 64` is held so that cell
has a meaningful window count at the lengths a reference fits in.

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
