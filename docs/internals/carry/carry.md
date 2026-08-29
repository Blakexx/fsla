# `carry.cu` — the carry family's dispatch, census and arm list

The mirror of `csrc/rola/src/carry/carry.cu`. One translation unit: the torch entries,
the arm dispatch, the per-arm register/SMEM census and the built-set listing.

## An arm is a (topology, `d_v`, DECLARATION, EXECUTOR MAP) row

The key the seam matches is `(widths, k, m, d_v, W, C, carve)`, and `carve` is the
DECLARATION SCALAR naming a built (carve, executor map, PARK): its low two bits are
`1` the ruled leaf split, `2` and `3` the two 1x8 containers, and BIT 2 (`+4`,
`carry_ops.PARKED`) is the park of K35 S5 — the same declaration with the write-view
`lo` spending the readout in SMEM (`carry_kernel.md`, "The park"). The park changes no
arithmetic and nothing else the caller can see, which is why it is a bit of the
declaration rather than an axis of its own; the arm row's `PK_` field must agree with
it, and a `static_assert` over the whole list says so. `d_v` JOINED THE KEY at K35 S3b and is read off the
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

Beside those sit the CARVED rows, which exist for their executor map or their width:
the `(4,4,4,4)` multi-level carve (`k=4, m=1`, `BC=256`, eight streams), the two 1x8
containers at the flagship cell (`carve = 2` and `3`, both `d_v = 64`), and the THREE
`d_v = 128` containers of K35 S3b —

| container | `k` | `m` | `s_l` | `BC` | owners | warps | streams | `WPS` | leaves/warp | SMEM | CTAs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1x8 column split (`carve = 2`) | 4 | 8 | (8,16) | 128 | 512 | 8 | 4 | 2 | 16 | 57,440 | 1 |
| 1x8 eight streams (`carve = 3`) | 4 | 8 | (8,16) | 128 | 512 | 8 | 8 | 1 | 16 | 71,584 | 1 |
| 2x4 halved box (`carve = 1`) | 4 | 4 | (4,16) | 64 | 1024 | 4 | 4 | 1 | 16 | 41,584 | 2 |

— all three at `W = 384`, `C = 32` and the ALT declaration, so the container question the
card leaves to measurement has its three answers built at ONE cell.

### The DENSE declarations on the carved body (K35 S1c)

Two of those carved rows DO carry a DENSE_BOTH twin, and they are the only ones that do:
the 1x8 column split (`carve = 2`) at both widths, declared `DENSE_BOTH` at every level.
They exist so the carved body and the channel-split body can be MEASURED at one dense
cell — a compare this line's history had never run, the recorded DENSE_BOTH compares
being split-vs-ancestor SASS identity rather than carved-vs-split.

The declaration reaches the carved body at ONE place, `ballot_of`'s per-level
conjunction, which a side does not enter where it is dense. At `MO = 0` no level enters
on either side, and the conjunction's identity is `valid` — the window's own in-range
mask — so the ballot comes out VACUOUSLY FULL rather than empty. Nothing else in the
body reads the declaration; every other quantity it holds is derived from the CARVE.

The rows the facts pass votes are NOT pruned with it. `CarveRows::row_g` keys on the
carve, so these arms' tables are built (144 write rows, 272 read) and then never read.
That is a cost the facts pass pays for a dense declaration on a carved body, and it is
the one asymmetry between the two prunings: the declaration prunes the CONJUNCTION, the
carve sizes the TABLE.

Both sides class every level DENSE, so the two private ORDERS coincide — the condition
the tied arm (`MO = 3`) needs. It is not sufficient. The tie also needs the two derived
VECTORS to agree, and under the column split the read side carves at `NSR = NSW · WPS`
streams against the write side's `NSW`: the walk cuts the outer level of a shared order
by two different counts, so the vectors part (write `[2,16]`, read `[1,16]`), the
leaf-to-reader permutation is not the identity, and the exchange is FULL. `kTied` also
requires `WPS == 1`, which these rows are not.

The seam still serves a production dense declaration from the CHANNEL-SPLIT body: these
rows are reached only by asking for `carve = 2` with `level_modes = 0`, which nothing in
the fall-back path does. They add measurable rows and change no dispatch.

### The PARK's twins (K35 S5)

Four more rows, each the SAME cell, box and executor map as one above it with `PK = 1`:
the three wide containers (which sit at or near the register wall) and the flagship
`d_v = 64` ALT arm as a control. A twin is a MEASUREMENT, so what the table carries is
the census on both sides of it — sm_86, zero spill and zero in-loop locals everywhere:

| twin | `carve` | regs `PK=0` → `PK=1` | SMEM `PK=0` → `PK=1` | CTA target |
|---|---|---|---|---|
| 1x8 column split, `d_v = 128` | 2 → 6 | 255 → 225 | 37,328 → 63,056 | 1 |
| 1x8 eight streams, `d_v = 128` | 3 → 7 | 248 → 227 | 53,808 → 63,344 | 1 |
| 2x4 halved box, `d_v = 128` | 1 → 5 | 242 → 221 | 32,704 → 42,112 | 2 |
| flagship ALT, `d_v = 64` (control) | 1 → 5 | 231 → 215 | 22,464 → 36,992 | 2 |

The register fall is 16–30 a lane against the `MCH*NL*2 + NL` (34 wide, 36 narrow) the
park frees, so the arms do NOT hand back the whole of it: with `lo` gone the peak moves
off the readout and onto another moment, and what the arm keeps is the difference. The
column-split row is the one the declaration exists for — it leaves the hardware's own
255-register ceiling with thirty registers of headroom, which is the budget P76 F5 wants
spent on LEAVES. The SMEM cost is the readout moment's, the exchange's segments no longer
being straddled by it. Sixteen leaves per
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
`tests/oracle/` and `tests/integration/`'s carry families asserts the magic before it
asserts anything else.

## What this file deliberately does not do

It does not reach `rola::chunk` and `rola::chunk` does not reach it. It is registered in
`rola_api.cpp` as its own entries and is absent from every DAG, plan and rule in
`rola/engine/`. The seam is `rola/ops/carry.py`, which is a bring-up surface and not a
shipped path.

## Comments moved from source (S5b)

Verbatim `//:` prose blocks from `csrc/rola/src/carry/carry.cu`, in source order, one heading per declaration. Source keeps only a short decl block pointing here.

<a id="note-l92"></a>
### near line 92

THE PER-LEVEL MODE DECLARATION (K33 law 3) is two bits per level with level 0 in
the low pair: bit 0 = the READ side is value-sparse there, bit 1 = the WRITE side.
`DENSE_BOTH` everywhere is always correct and only ever slower.  The encoding, the
private order it induces and the carve derived from that order are all in `box.cuh`
(`mode_declares`, `ModeOrder`, `StreamSpans`); this TU only passes the word through.

<a id="build-rows"></a>
### `build_rows`

THE BUILT SET, GENERATED (K36): the rows, and which of them this build carries.
`tools/gen_shards.py` owns the list and emits one translation unit per arm
beside this include; `setup.py` selects the subset by compiling those files and
defining `ROLA_CARRY_BUILD_<i>` for each, so `CARRY_ARMS_X` expands to exactly
the arms the binary carries.  docs/build.md.

<a id="carry-mo"></a>
### `CARRY_MO`

READING ONE ROW OF THE LIST.  A row arrives with its INDEX first -- the index is
what names the arm's translation unit and its four entry points -- and the pieces a
consumer wants are taken from it HERE and nowhere else.  `CARRY_ARM_TYPE`, the arm
AS A TYPE, is not here: it needs the geometry, so it lives with the bodies in
`carry_arm.cuh` and this TU never names it.

<a id="carry-match"></a>
### `CARRY_MATCH`

the seam's key, matched against the call's `(topology, k, m, d_v, W, C, carve)`.
`d_v` IS PART OF THE KEY (K35 S3b): the two widths of one declaration are two arms,
so the container rows 11/12 and 13/14 share their declaration scalar and are told
apart by the width the caller's value plane actually carries.

<a id="carry-cv-gate"></a>
### `CARRY_CV_GATE`

`CV` is the SEAM's DECLARATION scalar, not a kernel axis: it names which declared
(carve, executor map, PARK) an arm is, so a caller that asks for the deleted one --
or for an executor map nothing builds -- is refused rather than served something
else.  Its low two bits are the (carve, executor map): `1` the ruled leaf split, ONE
warp per write stream (`rola.ops.carry.LEAF_SPLIT`); `2` the 1x8 container as FOUR
write streams x TWO warps each (the column split); `3` as EIGHT streams x ONE warp;
and BIT 2 IS THE PARK (`+4`, `PARKED`), which changes nothing the caller can see;
the row's `PK_` must agree with it, asserted over the arms THIS BUILD carries.

<a id="carry-arms-x"></a>
### `CARRY_ARMS_X`

WHAT EACH BUILT ARM'S TRANSLATION UNIT EXPORTS.  Declared over `CARRY_ARMS_X` and
therefore over exactly the arms this build compiled: an arm outside the subset has
no declaration, no call site and no entry in `arms()`, so the seam's refusal below
is the only thing a caller asking for it can reach.

<a id="torch-check"></a>
### `TORCH_CHECK`

THE STATE PLANES ARE THE CALLER'S (docs/internals/state.md section 4).  A
continuation hands the SAME tensor as both, and advances it in place; a call that
hands NEITHER is the NULL-STATE call: a runtime null check inside the one
compiled body (K41; no STATE template axis) skips the sweep at that end, so
nothing is read, written, or allocated there.  `page_tbl` null is the dense
backing -- the slot is then the atom -- so the two backings are one kernel, one
ABI and one accumulation order.

<a id="tok-words"></a>
### `tok_words`

THE BALLOT ROWS COME FIRST, ON THE SAME STREAM (K31-i-d).  One pass over the
amplitude planes votes every owner's rectangle test once; the kernel below then
reads WORDS instead of re-reading those planes per (owner, window, token).  The
table is a TRANSIENT of this call -- nothing else asks for it -- and it is held
here rather than inside the dispatch so its lifetime outlives the launch.

<a id="torch-check-2"></a>
### `TORCH_CHECK`

THE DECLARATION IS RESOLVED AT THIS SEAM, AND AN UNBUILT ONE IS REFUSED (P76 R-B,
overruling law 3).  The retry that forced `mo` to zero served the DENSE_BOTH body
at the requested carve with no diagnostic -- a no-fallbacks violation that hid a
5x loss from a harness for nine stages (`## BISECT (corrected)`).  DENSE_BOTH is
served when it is DECLARED and never as a substitute for something else, so the
refusal names the whole key the caller asked for.

<a id="near-line-81"></a>
### near line 81

FOR THE BUILD STAMP ONLY -- the caps digest.  The kernel header is DELIBERATELY
absent: this TU instantiates no arm, and the two constants the stamp needs beside
the digest are in `box.cuh` with the family's other arm-independent geometry.

<a id="design"></a>
### `design`

the caps digest's namespace, aliased as the kernel header aliases it, so the
stamp names the same constant by the same name it always did.

<a id="rpr"></a>
### `RPR`

THE TABLE, AS A TENSOR.  `carry_rows.cuh` speaks pointers so that an arm TU never
parses the tensor library (K36); the allocation it leaves to the caller is done
here, in the one TU that already pays for torch.

<a id="dv"></a>
### `d_v`

THE VALUE WIDTH IS READ OFF THE CALLER'S PLANE and is part of the arm key, so an
unbuilt width is refused by the dispatch exactly as an unbuilt topology is.

<a id="stampkernelentry"></a>
### `stamp_kernel_entry`

THE FILE'S OWN CONSTANTS, device-side, beside the magic: the channel-split body's
executor count and the page granule.  `d_v` LEFT THIS SET at K35 S3b -- it is an arm
parameter now, so it is reported by the per-arm CENSUS below and not by the stamp.

<a id="file-header-arms"></a>
### The file header's arm catalogue (moved verbatim, S5c)

Dispatch and torch entries.  THE ARM LIST LEFT THIS FILE AT K36: it lives in
`tools/gen_shards.py`, which emits `../instantiations/carry_arms.inc` (the rows,
and which of them this build carries) and ONE TRANSLATION UNIT PER ARM beside it
(KERNEL_STANDARDS section 7).  This file therefore instantiates no kernel: it
matches a call against the rows and calls the arm's four entry points, which is
what makes an arm-subset build a FILE subset instead of a macro that still costs
a parse of every row.  Each arm DECLARES its two carves
-- the per-level span vector one STREAM owns on the write side and on the read
side -- so the carve is a property of the arm and never a constant of a kernel,
and the seam's `carve` scalar survives only to refuse a carve nothing builds.

The arms (`s` = the owner box, `SW`/`SR` = the two stream boxes):
```
  flagship   (256,256)  D=2  k=4 m=8   s=(8,16)     BC=128  W=384  C=32
  flagship+  (256,256)  D=2  k=4 m=8   s=(8,16)     BC=128  W=512  C=32
  small      ( 64, 64)  D=2  k=4 m=8   s=(8,16)     BC=128  W=64   C=32
  deep-3     (16^3)     D=3  k=2 m=16  s=(2,4,16)   BC=128  W=64   C=32
  deep-4     ( 8^4)     D=4  k=2 m=8   s=(2,2,4,8)  BC=128  W=64   C=32
  carve-4    ( 4^4)     D=4  k=4 m=1   s=(4,4,4,4)  BC=256  W=64   C=32
  flagship 1x8 (256,256) D=2  k=4 m=8   s=(8,16)     BC=128  W=384  C=32, twice
  flagship 1x8 d_v=128   the SAME two, at the wide value plane
  flagship 2x4 d_v=128 (256,256) D=2 k=4 m=4 s=(4,16) BC=64  W=384  C=32
  + the PARK's twins (K35 S5): those three and the flagship d_v=64 ALT row again
  flagship tied (256,256) D=2 k=4 m=8   s=(8,16)     BC=128  W=384  C=32, MO=3
```
The first five hold ONE box of `BC = 128` -- the register budget's number, which
does not move with depth -- so they differ only in how the same 128 leaves cut
into digits, and `deep-4` is the topology whose atom rectangle cannot be a single
level (`B = 8` is narrower than an atom).  `carve-4` is the MULTI-LEVEL CARVE arm
and the one whose owner is the whole topology: its 256 leaves are cut into EIGHT
streams of 32 by `SW = [1,4,2,4]` and `SR = [4,1,4,2]`, two carved levels per
side, one warp per stream -- so it is also the arm that runs eight executors.
`k = 4, m = 1` is the only capacity reaching the uniform box `(4,4,4,4)`.
The LAST row is the TIED arm: the flagship cell declared BOTH-SPARSE at level zero,
where the order-by-mode rule gives the two sides the SAME order and the derivation the
SAME vector, so the exchange is the identity and is elided at compile time.

`d_v` IS THE THIRD DIAL AND A KEY OF THE ARM (K35 S3b).  It is a template parameter
of both bodies, so the `d_v = 128` rows are not a wider path but the same body at a
wider declaration, and the seam tells them apart by the width the caller's value
plane carries.  The card's CONTAINERS entry rules what the width costs: sixteen
leaves per warp is REQUIRED at 128, and eight resident warps per SM is the target, so
the two ways to spend them are 1x8 at `BC = 128` (rows 13/14, the two container
declarations at the wide plane) and 2x4 at `BC = 64` (row 15, the CTA's box halved
instead of its warps doubled -- four streams, one warp each, the ruled leaf split).
The `d_v = 64` arms are UNTOUCHED by all of it, which is the point of making the
width a parameter rather than a branch.

THE EXECUTOR MAP IS THE SECOND DIAL (K35 S3a).  An arm declares `WPS`, the warps per
WRITE stream, so the CTA's warp count is `streams x WPS` and the READ carve is
declared at one stream per warp.  The two `flagship 1x8` rows are the SAME cell as
`flagship ALT` reached with EIGHT executors instead of four, by the two available
routes -- four write streams column-split two ways (`SW = [2,16]`, `WPS = 2`) and
eight write streams one warp each (`SW = [1,16]`, `WPS = 1`) -- so the container
question the card leaves to measurement has both of its answers built.  The existing
four-warp `ALT` arms ARE the 2x4 container: two CTAs of four warps per SM at half the
SM's box each, which is the same eight resident warps, so no arm is added for it.
The window is the architecture doc's TWO-CONSTRAINT quantity (integration rung
I1(b)): the C* cost basin is shallow, and the intra kernel -- which runs on the
SAME window grid -- has ideal arithmetic intensity `W/4` flop/byte and needs
`W >= ~380` to clear sm_86's compute/bandwidth balance.  The doc's admitted band
is 384-512 and BOTH ends are built, because a third constraint decides between
them: the intra refuses a window that does not divide `L`, and 384 does not
divide the flagship length `L = N = 65536`.  512 does, at every power-of-two
length, while clearing the intensity bar with room (128 flop/byte).  The
`(64, 64)` arm is the conformance cell: the fp64 oracle materializes
`[B,T,H,N]`, so `N = 4096` is where a whole cell is checkable, and its `W = 64`
is held to keep that cell's window count meaningful at the lengths a reference
can hold.
