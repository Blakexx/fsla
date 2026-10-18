# `row_stream`

Mirrors `measure/lab/experiments/row_stream.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

When the fold's rows are streamed a list at a time into each warp's ring (merge width 1) instead of staged once a
window, what does the stream cost beyond the fold, how much of its copies hides, and at what ring depth?

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
  The producer walks its window's STAGE RECORD -- the warp's stages and each stage's 8 tokens, a plan a pre-pass emits
  -- copied into shared memory two windows ahead, its record, slot and row addresses running pointers; the consumer
  reads its lists' entries a window ahead into registers.

sm_86 (RTX 3080 Ti), SM clock locked at 1.665 GHz, cycles a CTA-window:

| form | alt-k4 | alt-k16 | cohort-k16 |
|---|---|---|---|
| resident (copies waited, lookups at their use) | 9,636 | | |
| streamed, ring 2 (a 16-token step's own two stages) | 3,656 | | |
| streamed, ring 3 | 3,391 | | |
| streamed, ring 4 | 3,406 | 8,301 | 8,466 |
| streamed, ring 6 | 3,365 | | |
| streamed, ring 4, lookups from global memory at their use | 4,652 | | |
| bound: the fold alone (the producer, no row copies) | 2,833 | 6,616 | 6,604 |
| bound: the fold alone, no producer | 2,022 | 5,456 | 5,361 |
| bound: the copies alone | 2,269 | 6,284 | 7,159 |
| bound: the copies alone, 40 boxes | 2,219 | | |
| bound: the copies alone, a stage's pad rows fetched too | 2,762 | | |
| bound: the copies alone, lookups at their use | 3,710 | | |
| bound: resident, the fold alone | 5,955 | | |
| bound: resident, the copies alone | 7,864 | | |
| resident, copies allocated in L1 (`.ca`) | 9,827 | | |
| streamed, ring 4, copies allocated in L1 (`.ca`) | 3,419 | | |

What it settles:

- The stream costs the fold 1.68x at alt-k4, 1.52x at alt-k16 and 1.58x at cohort-k16 (streamed against the fold with
  no producer): 1,384, 2,845 and 3,105 cycles a CTA-window. A fixed part and a part a stage of the busiest warp fit
  it: 693 + 14.85 a stage, fit at alt-k4 and alt-k16, predicts cohort-k16 at 2,965 (-4.5%). The mean stage count
  mispredicts it by -15%: at cohort-k16 13% of warp-windows hold no stage, and with no barrier between windows a CTA
  runs as long as its busiest warp.
- The producer is most of the cost: its records, cursor and waits add 811 cycles at alt-k4 and 1,160 at alt-k16 --
  ~400 instructions a CTA-window plus ~11 a stage (20.1 a stage at alt-k4, 14.1 at alt-k16, ncu); the copies add 28.3
  instructions a stage at every cell, of which 573 cycles stay exposed at alt-k4.
- A ring of 3 or 4 stages is enough: 6 gains 1%; 2 (a 16-token step's own stages, nothing ahead) costs 7%.
- Lookups ahead are a requirement of the form: read from global memory at their use (each stage's tokens and each
  list's entries a dependent load), the stream runs 1.37x.
- The copies are the SM's rate, not the chip's L2: half the SMs busy (40 boxes) leaves the copies alone within 2.2%.
- Allocating the copies in L1 (`.ca`) gains nothing on the re-fetched rows (+0.4% streamed, +2.0% resident): the
  carry's `.cg` stays.

Record: `measure/lab/results/row_stream/sm_86.json`.
