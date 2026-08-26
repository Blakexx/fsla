# `carry.cu` — the carry family's dispatch, census and arm list

The mirror of `csrc/rola/src/carry/carry.cu`. One translation unit: the torch entries,
the arm dispatch, the per-arm register/SMEM census and the built-set listing.

## An arm is a (topology, `d_v`, DECLARATION, EXECUTOR MAP) row

The key the seam matches is `(widths, k, m, d_v, W, C, carve)`, and `carve` is the
DECLARATION SCALAR naming a built (carve, executor map): `1` the ruled leaf split, `2`
and `3` the two 1x8 containers. `d_v` JOINED THE KEY at K35 S3b and is read off the
caller's value plane, so a declaration at the two built widths is TWO ARMS of one body
rather than one arm with a width switch. `arms()` lists thirteen fields per row —
`D, B, k, m, d_v, W, C, carve, modes, BC, owners, smem, carve-level` — and the census
carries `d_v` as its own field, so the THREE dials are readable off the device:
executors `stamp[7] // 32`, write streams `stamp[14]`, width `stamp[15]`.

## The old framing: an arm is a (topology, DECLARATION) pair

`CARRY_ARMS_X` is the single X-macro the dispatch, the census and the listing all walk,
so the three cannot drift apart — the shipped family's `CHUNK_ARMS_X` discipline. Its
last column is the per-level MODE DECLARATION, because the declaration CHOOSES THE BODY
(`carry_kernel.md`): the same topology therefore appears once per declaration built for
it, and DENSE_BOTH appears for every topology because it is the seam's answer to any
declaration this family has no carved body for.

`carve_level<D, MODES>` is what the seam resolves: the innermost level the declaration
marks read-sparse, or `-1` for the channel-split body. `ALT` at depth `D` — write-sparse
on the even levels, read-sparse on the odd — packs to 6 / 38 / 102 at `D = 2 / 3 / 4`,
and at `D = 3` it marks the INNERMOST level write-sparse, so that arm's carve level is
`-1` and it takes the channel-split body.

| topology | `k` | `m` | `s_l` | `BC` | owners | `W` | declarations built | carve level |
|---|---|---|---|---|---|---|---|---|
| `(256,256)` | 4 | 8 | (8,16) | 128 | 512 | 384 | DENSE_BOTH, ALT | −1, 1 |
| `(256,256)` | 4 | 8 | (8,16) | 128 | 512 | 512 | DENSE_BOTH, ALT | −1, 1 |
| `(64,64)` | 4 | 8 | (8,16) | 128 | 32 | 64 | DENSE_BOTH, ALT | −1, 1 |
| `(16,16,16)` | 2 | 16 | (2,4,16) | 128 | 32 | 64 | DENSE_BOTH, ALT | −1, −1 |
| `(8,8,8,8)` | 2 | 8 | (2,2,4,8) | 128 | 32 | 64 | DENSE_BOTH, ALT | −1, 3 |

Beside those sit the CARVED-ONLY rows, which exist for their executor map or their width
and carry no DENSE_BOTH twin: the `(4,4,4,4)` multi-level carve (`k=4, m=1`, `BC=256`,
eight streams), the two 1x8 containers at the flagship cell (`carve = 2` and `3`, both
`d_v = 64`), and the THREE `d_v = 128` containers of K35 S3b —

| container | `k` | `m` | `s_l` | `BC` | owners | warps | streams | `WPS` | leaves/warp | SMEM | CTAs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1x8 column split (`carve = 2`) | 4 | 8 | (8,16) | 128 | 512 | 8 | 4 | 2 | 16 | 57,440 | 1 |
| 1x8 eight streams (`carve = 3`) | 4 | 8 | (8,16) | 128 | 512 | 8 | 8 | 1 | 16 | 71,584 | 1 |
| 2x4 halved box (`carve = 1`) | 4 | 4 | (4,16) | 64 | 1024 | 4 | 4 | 1 | 16 | 41,584 | 2 |

— all three at `W = 384`, `C = 32` and the ALT declaration, so the container question the
card leaves to measurement has its three answers built at ONE cell. Sixteen leaves per
warp is the card's requirement at the wide plane, and each row reaches it a different
way: by splitting the columns, by cutting more streams, or by halving the CTA's box.

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
`BC`, the owner count, the derived ring depth, the CTA target, the CARVE LEVEL the
declaration named, the WRITE STREAM count and the arm's `d_v`. The stamp is per
`(topology, d_v, declaration)`, so it also answers which body an arm resolved to. The
kernel-entry half of it carries the FILE's constants only — the channel-split body's
warp count and the page granule; `d_v` left that set when it became an arm parameter and
is reported by the census instead. Every gate in
`tests/k31/` asserts the magic before it asserts anything else.

## What this file deliberately does not do

It does not reach `rola::chunk` and `rola::chunk` does not reach it. It is registered in
`rola_api.cpp` as its own entries and is absent from every DAG, plan and rule in
`rola/engine/`. The seam is `rola/ops/carry.py`, which is a bring-up surface and not a
shipped path.
