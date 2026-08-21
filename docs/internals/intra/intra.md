# `intra.cu` / `intra.cuh` — the intra family's dispatch and vocabulary

The mirror of `csrc/rola/src/intra/intra.cu` and `csrc/rola/src/intra/intra.cuh`. One
translation unit: the support-mode vocabulary, the arm dispatch, the arm listing and the
device build stamp.

## The support modes are a DECLARATION, not a switch

Two bits per level, level 0 in the low pair: bit 0 says the READ side's rows are
value-sparse, bit 1 the WRITE side's. Neither bit is `DENSE_BOTH`, both is
`SHARED_SPARSE`. The kernel is pattern-generic — the slab mask machinery answers
all-ones for an undeclared side and therefore cannot gate — so `DENSE_BOTH` everywhere
is always CORRECT and only ever slower. That is what makes the mode word a statement
about the routing rather than a code path, and it is why a caller who declares wrongly
loses a harvest instead of an answer.

## The window is an instantiation axis

`W` pairs with the carry family's `W` template argument, because the two kernels run ONE
global window grid: an arm is a `(D, B, W)` triple on both sides. `kWindowFlagship = 384` is
the architecture doc's two-constraint window — the intra's ideal arithmetic intensity is
`W/4` flop/byte, which must clear sm_86's compute/bandwidth balance — and
`kWindowWide = 512` is the end of the admitted band that also DIVIDES a power-of-two
length, which is the third constraint and the reason both ends are built.
`kWindowSmall = 64` is the conformance cell's.

The LEVEL WIDTH is the second instantiation axis, mirroring the carry family's
topology arms: `kWidthFlat = 256`, `kWidthDeep3 = 16` and `kWidthDeep4 = 8`. The other
shape constants are the tile grid the causal staircase is expressed on: `kTile = 64`
output tokens, `kWarpRows = 16` rows to a warp, `kSlabDigits = 16` digits to an MMA
k-step, and `kValueWidth = 64`.

## It reduces into `o` and `den`

The kernel's write-back is a RED, not a store, so the caller owns those buffers and their
initial contents. The carry allocates and fills them; the intra then folds its
within-window term into the same buffers on the same stream. Nothing is summed on the
host, and there is no second output pair to reconcile — which is the whole reason the
intra was built as a reducer rather than as a returner.

## The stamp is a footprint

`intra_build_stamp` returns the heaviest built arm's shared-memory footprint together
with its thread count, so the value moves whenever any arm's shape does. A path check or
a hash check cannot catch an extension trap; a fact only the loaded binary can produce
can.

## The kernel names are not anonymous

`nvcc` mangles an anonymous-namespace kernel with a hash that depends on the BUILD PATH,
so a manifest ratified in one worktree certifies a name no other checkout's build
produces and the fatbin gate then refuses a byte-identical kernel. A ratifiable kernel
needs a named symbol, which is why the stamp lives in a named `stampdet` namespace.
