# `intra_backward.cu` — the reverse intra's dispatch

The mirror of `csrc/rola/src/intra/intra_backward.cu`. One translation unit: the torch
entry, the arm dispatch, the arm listing and the device build stamp.

## It reduces, it does not return

The kernel REDs its within-window half into the four gradient planes rather than
returning them, exactly as the forward intra REDs into `o` and `den`. The carry's two
reverse scans own the cross-window half of the same planes, and the stream order is the
only synchronization the pair needs — nothing is summed on the host and there is no
second output set to reconcile.

A standalone conformance run therefore allocates its own zeroed planes, and a run beside
the scans must not. Off the declared support of a sparse duty this kernel writes nothing
at all, so in a standalone run those cells hold the zeros the allocation gave them and
beside the scans they hold the scans' contribution alone.

## The arms mirror the forward's

`INTRA_BWD_ARMS_X` carries the same `(levels, window)` rows the forward's
`INTRA_ARMS_X` does, because the two kernels run one window grid. The block footprint is
the forward's plus the ROW side's seed rows, the per-warp fan-in relay plane and the
`dG_l` image, against operand slabs half the forward's digit width — the reverse's stage
cap is its own, and the trade is stated in
[`intra_backward_kernel.md`](intra_backward_kernel.md). The seed rows are staged once per
output block rather than once per column tile: their tokens are the block's readers and
do not change with the column tile.

## The stamp is a footprint

`intra_backward_stamp` returns the heaviest built arm's shared-memory footprint, its
thread count and the family's magic, so the value moves whenever any arm's shape does —
a fact only the loaded binary can produce, which is what makes reading it back an
extension-trap check rather than a path check.
