# `csrc/rola/src/owner/owner_fill.cuh` — the fill

Mirror doc for the owner carry's fill: a pass of the window's live tokens copied from the bundle, where it lies, into
the CTA's write pool (what the fold reads) or read pool (what the readout reads). Vocabulary: [`owner.md`](owner.md);
the slots and liveness come from the lists ([`owner_lists.md`](owner_lists.md)).

<a id="pools"></a>

## The pools

A write row, one a write-live token, takes each level's write digits in their stored type — f16 on a dense level, u8
on a sparse one ([`liveness_contract.md`](../facts/liveness_contract.md#floor)) — so its layout is a WRITE FORMAT,
`WriteRows<C, kHalfS, kHalfR>`: at `(16, 64)` level R's run (the box's 64 digits, 64 or 128 bytes), the CTA's 32
channels of the token's f16 values (64 bytes), level S's slice (the box's 16 digits, 16 or 32 bytes) and the token's
`(F, m)` from the facts pass ([`facts.md`](../facts/facts.md#factor-products), 8 bytes); at `(32, 32)` level S's run
then level R's (32 or 64 bytes each), the values and `(F, m)`. The row rounds up to 32 bytes: 160, 192, 224 or 256 at
`(16, 64)`. A read row, one a read-live token, is u8 at every level (`ReadRows`): a 64-byte run (the read-dense slice,
level R's read coefficients, or at `(32, 32)` the read-dense and read-sparse slices), then at `(16, 64)` the
read-sparse chunk: 96 or 64 bytes. Each pool holds 128 rows; a window whose union is larger is filled, and consumed,
in passes. The write pool (`WritePool`) is the fold's, the read pool (`ReadPool`) the readout's. They are separate
tenants so a kernel places each with its phase.

EVERY RUN SITS AT ITS GLOBAL SECTOR'S 32-BYTE PHASE: the pools are 128-byte aligned, the runs at 32-byte offsets, the
pitches multiples of 32. A `cp.async` lane pair coalesces into one L2 sector request only when its shared destination
has the source's 32-byte phase (measured by a standalone copy probe: 2 sectors a token aligned, 4 at a
16-byte offset, 3 at a 144-byte pitch). The first layout put the pools at 16 modulo 32 and read 41.1M sectors a head at
`(32, 32)` against 27.1M aligned.

<a id="quads"></a>

## A quad a token

Four lanes fill one token, eight tokens a warp instruction, each lane taking the same part of every token it fills —
the measured copy ranking: a quad on a 64-byte run costs a third of one lane a row. Lane q takes the run's 16-byte
quanta `q, q + 4, ...`, so each copy instruction covers a contiguous 64 bytes and a lane pair one sector. So
`WriteLanes` and `ReadLanes` compute each lane's plane bases once a CTA (the batch and head are the CTA's), and a token
costs one 64-bit multiply-add a plane and its copies. The first form, a (token, chunk) task a thread with its address recomputed every task, issued 3.3x the
instructions (16 x 64 at `nl64k-alt-k4`: 516M against 159M a head, 5.0 against 2.4 ms under ncu).

<a id="write-pass"></a>

## The write pass

At `(16, 64)` the quad copies level R's run — a sparse level's 64 bytes by its live sectors, a 32-byte sector the
token is not live in landing as zeros and reading nothing (the lists' sector unions), or a dense level's 128 bytes in
two copies — then the values' two 32-byte runs, level S's slice on its first one or two lanes, and the token's `(F, m)`
on its last (8 bytes, through L1). At `(32, 32)` the two levels' runs are one run of four to eight quanta. The read pass
is the read-dense run and the read-sparse chunk (or the two 32-byte runs).

<a id="measured"></a>

## Measured

`--check fill` holds every pool row against the model: `flagship-dense` (both write levels f16), `flagship-alt-k4`
and `-cohort-k16` (S f16, R u8), `flagship-both-k4` and its L4096 (S u8, R f16). No cell declares both write levels
sparse, so the u8-u8 format is built and gated by no cell. Under ncu at `nl64k-alt-k4` (64K tokens, one head), a
CTA-window reads 2,116 L2 sectors at `(16, 64)` and 1,730 at `(32, 32)`: the byte model's 1,989 at `(16, 64)` plus the
`(F, m)` plane's 128 (a window's 512 tokens x 8 bytes through L1), where the four factor columns read ~256; at `(32,
32)` level S's f16 run is a second sector a token. The fill part (lists included, each pass waited on) runs 1.88 and
1.64 ms (2.35 and 1.93 with the four factor columns).
