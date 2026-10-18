# `row_stream`

Mirrors `measure/lab/experiments/row_stream.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

When the fold's rows are streamed a list at a time into each warp's ring (merge width 1) instead of staged once a
window, what does the stream cost beyond the fold, at what ring depth, and in what form?

The fold step is the carry's: a warp folds its 4 lists `w + 8 j` into 128 fp32 registers, 16-token steps while more
than 8 entries remain, then an 8-token tail. The inputs (`row_stream.inputs.py`) are three cells' write lists:
nl64k-alt-k4 over 64 windows, and flagship-alt-k16-L16384 and flagship-cohort-k16-L16384 over 32. The 80 CTAs are
boxes 0 .. 79 of one head, channel tile fastest as the carry orders them. A box's row for a token is its level-1 slice
group's 32 bytes (the carry's u8 slice bytes, zero-filled to a 64-byte slot) and its channel tile's 64 bytes of values,
so L2 serves what one box fetched to the others, as in the carry. Every forming variant's state is checked exactly.

| a CTA-window | alt-k4 | alt-k16 | cohort-k16 |
|---|---|---|---|
| entries (union) | 255.6 (216.2) | 1,022.0 (459.1) | 1,022.3 (444.0) |
| stages of 8 rows: mean; the busiest warp's x 8 | 45.1; 46.5 | 141.8; 144.9 | 128.0; 153.0 |

- RESIDENT: each window's union rows copied once into a pool and its entries staged as slots (their lists and tokens
  read from global memory at their use), a CTA barrier, the fold, a barrier. This is not today's carry, whose fills of
  the next chunk run under the fold of this one.
- STREAMED: each warp's ring of `D` stages of 8 rows, each slot refilled as its step ends, across lists and windows.
  The producer walks its window's STAGE RECORD -- the warp's stages and each stage's 8 tokens (a pad row the stage's
  first: the same line, its c zero), a plan a pre-pass emits -- copied into shared memory two windows ahead with the
  stage group it is issued in; its record, slot and per-window row bases are running pointers, a stage's two copies
  one `IMAD.WIDE` each. The consumer reads its lists' entries a window ahead into registers.

sm_86 (RTX 3080 Ti), SM clock locked at 1.665 GHz, cycles a CTA-window:

| form | alt-k4 | alt-k16 | cohort-k16 |
|---|---|---|---|
| resident (copies waited, lookups at their use) | 9,676 | | |
| streamed, ring 2 (a 16-token step's own two stages) | 3,614 | | |
| streamed, ring 3 | 3,240 | | |
| streamed, ring 4 | 3,150 | 7,770 | 7,950 |
| streamed, ring 6 | 3,143 | | |
| streamed, ring 4, records on mbarriers | 3,402 | | 8,197 |
| streamed, ring 4, lookups from global memory at their use | 4,576 | | |
| bound: the fold with the producer, no row copies | 2,827 | 6,613 | 6,603 |
| bound: the same, records on mbarriers | 3,069 | | |
| bound: the fold alone, no producer | 2,018 | 5,458 | 5,360 |
| bound: the copies alone | 2,344 | 7,024 | 7,030 |
| bound: the copies alone, 40 boxes | 2,039 | | |
| bound: the copies alone, lookups at their use | 3,751 | | |
| bound: resident, the fold alone | 5,986 | | |
| bound: resident, the copies alone | 7,898 | | |
| resident, copies allocated in L1 (`.ca`) | 9,849 | | |
| streamed, ring 4, copies allocated in L1 (`.ca`) | 3,164 | | |

Instructions (ncu, a stage): the producer 20.1 at alt-k4, 14.1 at alt-k16, 17.6 at cohort-k16 (~400 a CTA-window plus
~11 a stage); the copies 15.4, 13.5, 13.5.

What it settles:

- The stream costs the fold 1.56x at alt-k4, 1.42x at alt-k16 and 1.48x at cohort-k16 (streamed against the fold with
  no producer): 1,132, 2,312 and 2,590 cycles a CTA-window. As a factor of the bare fold it predicts its third cell from
  the other two within 1% (1.49 against 1.48); as a fixed part and a part a busiest-warp stage, within 7%.
- The producer is most of the cost: its records, cursor and waits add 809 cycles at alt-k4 and 1,155 at alt-k16. The
  copies' address arithmetic is the part a form decides: per-window bases and pads pointing at a live row halve the
  copies' instructions (28.3 a stage to 13.5-15.4) and cut the stream's cost 18% at alt-k4.
- Records land with the stage groups: on an mbarrier (`cp.async.mbarrier.arrive`, `ARRIVES.LDGSTSBAR`) the stream runs
  8% slower at alt-k4 and 3% at cohort-k16, though a window's entry then never drains the ring.
- A ring of 4 stages is enough: 6 gains 0.2%, 3 costs 3%, 2 (a 16-token step's own stages, nothing ahead) 15%.
- Lookups ahead are a requirement of the form: read from global memory at their use (each stage's tokens and each
  list's entries a dependent load), the stream runs 1.45x.
- Allocating the copies in L1 (`.ca`) gains nothing on the re-fetched rows (+0.4% streamed, +1.8% resident): the
  carry's `.cg` stays.

Record: `measure/lab/results/row_stream/sm_86.json`.
