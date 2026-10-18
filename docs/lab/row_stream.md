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
- STREAMED: each warp's ring of `D` stages of 8 rows and their 8 coefficients (f16, in a 128-byte pad so every stage
  keeps the rows' 32-byte phase), each slot refilled as its step ends, across lists and windows. The producer walks its
  window's STAGE RECORD -- the warp's stages, each stage's 8 tokens and 8 coefficients (a pad row the stage's first:
  the same line, its coefficient zero), a plan a pre-pass emits -- copied into shared memory two windows ahead with the
  stage group it is issued in; a stage's coefficients ride in the slice copy's spare lane, its record, slot and
  per-window row bases are running pointers. The consumer's step reads its coefficients from the stage, its lane's row
  and swizzle offsets held in registers (opaque, so ptxas cannot re-form them each step); a list's entries arrive a
  window ahead in registers.

sm_86 (RTX 3080 Ti), SM clock locked at 1.665 GHz, cycles a CTA-window:

| form | alt-k4 | alt-k16 | cohort-k16 |
|---|---|---|---|
| resident (copies waited, lookups at their use) | 9,632 | | |
| streamed, ring 2 (a 16-token step's own two stages) | 3,564 | | |
| streamed, ring 3 | 3,205 | | |
| streamed, ring 4 | 3,057 | 8,016 | 8,707 |
| streamed, ring 6 | 3,075 | | |
| streamed, ring 4, records on mbarriers | 3,298 | | 8,785 |
| streamed, ring 4, lookups from global memory at their use | 5,461 | | |
| bound: the fold with the producer, no row copies | 3,055 | 7,162 | 7,322 |
| bound: the same, records on mbarriers | 3,331 | | |
| bound: the fold alone, no producer | 1,894 | 5,130 | 5,141 |
| bound: the copies alone | 2,486 | 7,595 | 8,031 |
| bound: the copies alone, 40 boxes | 2,101 | | |
| bound: the copies alone, lookups at their use | 4,620 | | |
| bound: resident, the fold alone | 5,944 | | |
| bound: resident, the copies alone | 7,862 | | |
| resident, copies allocated in L1 (`.ca`) | 9,826 | | |
| streamed, ring 4, copies allocated in L1 (`.ca`) | 3,099 | | |

Instructions a MMA (ncu): the fold alone 6.3 at alt-k4, 6.2 at alt-k16, 7.0 at cohort-k16 (9.5 before its lane
offsets were held and its coefficients read from the stage: ptxas re-formed ~20 lane constants each step); streamed
14.5, 14.4, 16.6 -- the producer ~27 instructions a stage of 8 rows (a token load, two 64-bit sources, two
destinations, the two copies and ptxas's three fillers, the record pointers, the window check, the slot, the commit),
its structural floor ~22, against ~5.7 MMAs a stage at alt-k4.

What it settles:

- The stream costs the fold 1.61x at alt-k4, 1.56x at alt-k16 and 1.69x at cohort-k16 (streamed against the fold with
  no producer); in instructions it is the producer's ~8 a MMA, its bookkeeping a stage of 8 rows -- halved only by
  larger stages.
- A copy's shared destination keeps its source's 32-byte phase: stages of 1,040 bytes (the coefficients appended
  unpadded, every other stage 16 bytes off) ran the stream 15-35% slower (unlocked runs, not recorded).
- Records land with the stage groups: on an mbarrier (`cp.async.mbarrier.arrive`, `ARRIVES.LDGSTSBAR`) the stream
  runs 8% slower at alt-k4, 1% at cohort-k16.
- A ring of 4 stages is enough: 6 gains nothing, 3 costs 5%, 2 (a 16-token step's own stages, nothing ahead) 17%.
- Lookups ahead are a requirement of the form: read from global memory at their use, the stream runs 1.79x.
- Allocating the copies in L1 (`.ca`) gains nothing on the re-fetched rows (+1% streamed, +2% resident): `.cg`.

Record: `measure/lab/results/row_stream/sm_86.json`.
