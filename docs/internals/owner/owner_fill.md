# `csrc/rola/src/owner/owner_fill.cuh` — the fill

Mirror doc for the owner carry's fill: a pass of the window's live tokens copied from the bundle, where it lies, into
the CTA's write pool (what the fold reads) or read pool (what the readout reads). Vocabulary: [`owner.md`](owner.md);
the slots and liveness come from the lists ([`owner_lists.md`](owner_lists.md)).

<a id="pools"></a>

## The pools

A write row, one a write-live token, 160 bytes: a 64-byte coefficient run (at `(16, 64)` the write-sparse slice, level
R's write coefficients for the box's 64 digits; at `(32, 32)` the write-dense slice, level S's for the box's 32 digits,
then the write-sparse slice), the two value digits for the CTA's 32 channels (`v_hat`, then `v_lo` or zeros at one
value digit), then at `(16, 64)` the 16-byte dense slice, then the four factors (write gain, value scale, the two write
scales). A read row, one a read-live token: a 64-byte run (the read-dense slice, level R's read coefficients, or at
`(32, 32)` the read-dense and read-sparse slices), then at `(16, 64)` the read-sparse chunk: 96 or 64 bytes. Each pool
holds 128 rows; a window whose union is larger is filled, and consumed, in passes.

EVERY RUN SITS AT ITS GLOBAL SECTOR'S 32-BYTE PHASE: the pools are 128-byte aligned, the runs at 32-byte offsets, the
pitches multiples of 32. A `cp.async` lane pair coalesces into one L2 sector request only when its shared destination
has the source's 32-byte phase (measured by a standalone copy probe: 2 sectors a token aligned, 4 at a
16-byte offset, 3 at a 144-byte pitch). The first layout put the pools at 16 modulo 32 and read 41.1M sectors a head at
`(32, 32)` against 27.1M aligned.

<a id="quads"></a>

## A quad a token

Four lanes fill one token, eight tokens a warp instruction, each lane taking the same part of every token it fills —
the measured copy ranking: a quad on a 64-byte run costs a third of one lane a row. So `FillLanes` computes each lane's
plane bases once a CTA (the batch and head are the CTA's), and a token costs one 64-bit multiply-add a plane and its
copies. The first form, a (token, chunk) task a thread with its address recomputed every task, issued 3.3x the
instructions (16 x 64 at `nl64k-alt-k4`: 516M against 159M a head, 5.0 against 2.4 ms under ncu).

<a id="write-pass"></a>

## The write pass

At `(16, 64)` the quad copies the sparse slice's 64-byte run, a 32-byte sector it is not live in landing as zeros and
reading nothing (the lists' sector unions), then the value planes' two 32-byte runs, then the four factors (4 bytes
each, through L1), and its first lane the 16-byte dense slice. At `(32, 32)` the dense and sparse slices are two 32-byte
runs of one copy. The read pass is the read-dense run and the read-sparse chunk (or the two 32-byte runs).

<a id="measured"></a>

## Measured

`--check fill` holds every pool row against the model (`flagship-alt-k4`, `flagship-dense`). Under ncu at
`nl64k-alt-k4` (64K tokens, one head), a CTA-window reads 2,245 L2 sectors at `(16, 64)` and 1,657 at `(32, 32)`: the
byte model's 1,989 and 1,404 plus the factor columns' ~256 (four 4-byte reads a token through L1, which keeps a
window's 64 sectors a column). The fill part (lists included, each pass waited on) runs 2.35 and 1.93 ms.
