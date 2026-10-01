# `csrc/rola/src/owner/owner_fill.cuh` — the fill

Mirror doc for the owner carry's fill: a pass of the window's live tokens copied from the bundle, where it lies, into
the CTA's write pool (what the fold reads) or read pool (what the readout reads). Vocabulary: [`owner.md`](owner.md);
the slots and liveness come from the lists ([`owner_lists.md`](owner_lists.md)).

<a id="pools"></a>

## The pools

A write row, one a write-live token: its write-dense slice (level S's write coefficients for the box's `RS` digits),
its write-sparse slice (level R's for the box's `RD` digits), its two value digits for the CTA's 32 channels
(`v_hat`, then `v_lo` or zeros at one value digit) and its four factors (write gain, value scale, the two write
scales): 160 bytes at `(16, 64)`, 144 at `(32, 32)`. A read row, one a read-live token: its read-dense slice (level R's
read coefficients, `RD` bytes) then its read-sparse slice (level S's, `RS` bytes). Each pool holds 128 rows; a window
whose union is larger is filled, and consumed, in passes.

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
`nl64k-alt-k4` (64K tokens, one head), a CTA-window reads 2,363 L2 sectors at `(16, 64)` and 2,508 at `(32, 32)`. The
model of the bytes predicts the opposite order (1,989 and 1,404 without the factors): per instruction, `(16, 64)`
meets its ideal sectors except the 16-byte slices' half-sector waste, but at `(32, 32)` every two-lane 32-byte run
costs 1.5 sectors and a read run 2, about 850 excess sectors a CTA-window from the same copy code — open, to be probed
before the carve is priced on fill bytes.
