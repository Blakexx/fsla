# `row_stream`

Mirrors `measure/lab/experiments/row_stream.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

When the fold's rows are streamed a list at a time into each warp's ring (merge width 1) instead of staged once a
window, how much of their copies hides under the fold, at what ring depth, and what do the copies cost?

The fold step is the carry's: a warp folds its 4 lists `w + 8 j` into 128 fp32 registers, 16-token steps while more
than 8 entries remain, then an 8-token tail. The inputs (`row_stream.inputs.py`) are nl64k-alt-k4's write lists over
64 windows. Its 80 CTAs are boxes 0 .. 79 of one head, channel tile fastest as the carry orders them. A box's row for a
token is its level-1 slice group's 32 bytes (the carry's u8 slice bytes, zero-filled to a 64-byte slot) and its
channel tile's 64 bytes of values, so a token's slice is fetched by every box of its slice group and its values by
every box of its channel tile, and L2 serves what one box fetched to the others, as in the carry. A CTA-window holds
255.6 entries (union 216.2, refetch 1.18x): 45.1 stages of 8 rows, 90.1 warp `cp.async` instructions, 511 global
lines. Every forming variant's state is checked exactly against the mirror's.

- RESIDENT: each window's union rows copied once into a pool and its entries staged as slots (their lists and
  tokens read from global memory at their use), a CTA barrier, the fold, a barrier: the copies waited, not
  overlapped. This is not today's carry, whose fills of the next chunk run under the fold of this one.
- STREAMED: each warp's ring of `D` stages of 8 rows, a stage copied as soon as a slot no step holds is free, across
  lists and windows. The producer walks its window's STAGE RECORD -- the warp's stages and each stage's 8 tokens, a
  plan a pre-pass emits -- copied into shared memory a window ahead; the consumer reads its lists' entries a window
  ahead into registers.

sm_86 (RTX 3080 Ti), SM clock locked at 1.665 GHz, cycles a CTA-window:

| form | cycles |
|---|---|
| resident (copies waited, lookups at their use) | 9,641 |
| streamed, ring 2 (a 16-token step's own two stages) | 4,324 |
| streamed, ring 3 | 3,920 |
| streamed, ring 4 | 3,862 |
| streamed, ring 6 | 3,888 |
| streamed, ring 4, lookups from global memory at their use | 5,280 |
| bound: streamed, the fold alone | 3,159 |
| bound: streamed, the copies alone | 2,737 |
| bound: streamed, the copies alone, 40 boxes | 2,705 |
| bound: streamed, the copies alone, a stage's pad rows fetched too | 2,911 |
| bound: streamed, the copies alone, lookups at their use | 4,294 |
| bound: resident, the fold alone | 5,954 |
| bound: resident, the copies alone | 7,864 |
| resident, copies allocated in L1 (`.ca`) | 9,827 |
| streamed, ring 4, copies allocated in L1 (`.ca`) | 3,875 |

What it settles:

- The streamed fold runs at 1.22x its fold-alone bound; of the copies' 2,737 cycles, 2,034 (74%) hide under the fold.
  The fold binds at this cell: the copies alone take 0.87x the fold alone.
- A ring of 4 stages is enough: ring 3 is 1.5% over it, ring 6 0.7%. Ring 2 (only a 16-token step's own stages,
  nothing ahead) is 12% over.
- Lookups ahead are a requirement of the form, not a refinement: read from global memory at their use (each stage's
  tokens and each list's entries a dependent load), the copies alone take 1.57x and the streamed fold 1.37x. The producer's
  cursor reads shared memory only, staged a window ahead.
- A copy costs per instruction, with a small charge per line: fetching a stage's pad rows too (722 lines in place of
  511, the same 90.1 instructions) adds 174 cycles. From the two points: 25.7 cycles a warp `cp.async` instruction
  plus 0.82 a global line, an SM's. Half the SMs busy (40 boxes) leave it within 1.2% (2,705), so it is the SM's
  rate, not the chip's L2.
- Allocating the copies in L1 (`.ca`) gains nothing on the re-fetched rows (+0.3% streamed, +1.9% resident): the
  carry's `.cg` stays.

Record: `measure/lab/results/row_stream/sm_86.json`.
