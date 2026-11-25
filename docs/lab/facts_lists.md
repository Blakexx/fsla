# `facts_lists`

Mirrors `measure/lab/experiments/facts_lists.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

A sparse level's liveness is RULED to be a mask the op is handed, a bit a digit a token, used blindly; the design's
gather reads the words and counts the mask-preparation pass prepares and tests no row's value
([`ARCHITECTURE.md`](../ARCHITECTURE.md#62-the-mask-preparation-pass)). This
lab prices the ruled form beside the access that tests the rows' values. The mask says, for a token, which digits are live; a
carry CTA needs the opposite each window -- for each of its digits, WHICH TOKENS are live on it, the data it pulls.
What does turning one into the other cost a token on hopper, and what does the value test it replaces cost beside
it?

THE FORM. A CTA a window of 1,024 tokens, a thread a token, one side's sparse level of 256 digits. The pass is a
counting sort of the window's live (token, digit) pairs by digit, in token order:
- a thread ORs its lane's bit into a word a (warp, digit) for each of its live digits, a shared-memory atomic each;
- a digit's thread sums the 32 warps' counts before each warp; the digits' offsets are a scan by the 8 warps of
  digits and their totals -- no thread works while the CTA waits;
- a thread stores each of its live digits' ENTRY at its digit's offset, the count before its warp and the lanes
  before it in its warp's word. AN ENTRY IS THE TOKEN, with no byte: this
  lab's pass turns the mask around and reads no row, and writes its lists to global memory; it is not the design's
  pass. In the design the gather hands the carry no list: it places a sparse block's live tokens with their sparse
  bytes from the prepared write-side words and counts
  (KERNEL_STANDARDS 29, item 5; [`hopper-gather.md`](../architecture/hopper-gather.md#no-gather-kernel)).
Two forms find a token's live digits: the mask's 32 bytes as handed, or a value test of the token's 256 bytes read
from the tile-major rows the bundle holds. A call is the window's COUNTS alone -- what sizes its lists before they
are written -- or the LISTS too. Each is timed with every table in cache (a call's 16 rounds over the same
windows) and ONCE-READ (a round a fresh window: 2.2 million tokens, 69 MB of mask and 540 MB of rows), at a cell of 4 to 5 live
digits a token and at one of 32. Every count and entry of the first round is checked against the host's.

THE GATHER PASS IN ITS PLAINEST FORM (a third call). With the carry landing whole entries by bulk copy
([`carry_window`](carry_window.md)), the pass that turns the mask around can move the DATA: for each live (token,
digit) pair, the token's 8 value lines, its byte on the digit and its dense row's 16 blocks, to its slot of its
digit's steps of 16 entries -- a step 393 lines, a layout of this call's own, not the tile the carry lands. A load
and a store of global memory a line, each load used in its own trip, a thread a token. Every arranged line is
checked, each byte of the tables its own position's. Its tables and what it writes are past the cache in either
mode, so its rows are named by their windows, not by a cache. The form that moves the same lines in shared memory
between bulk copies, and the flat stream that is its floor, are [`gather_pass`](gather_pass.md).

MEASURED (H100 80GB HBM3, 132 SMs, vast.ai instance 60600636, driver 595.91.07; 2026-11-14, the experiment at
e34189c2; 7 repetitions; every count, entry and arranged line of each form's first round exact; the median,
SM-cycles a token for one side's sparse level):

| the cell | a token's live digits from | counts, cached | counts, once-read | lists, cached | lists, once-read |
|---|---|---|---|---|---|
| 4 to 5 live digits a token | the mask as handed | 5.0 | 6.1 | 8.3 | 9.9 |
| 4 to 5 live digits a token | a value test of the rows | 11.0 | 19.3 | 14.0 | 20.6 |
| 32 live digits a token | the mask as handed | 9.5 | 10.5 | 43.6 | 45.4 |
| 32 live digits a token | a value test of the rows | 15.1 | 18.8 | 47.0 | 46.8 |

| the cell | the gather call, its rounds the same windows | a fresh window a round |
|---|---|---|
| 4 to 5 live digits a token | 1,579 | 1,557 |
| 32 live digits a token | 5,622 | 4,548 |

WHAT IT SHOWS.
- THE LISTS ARE PRICED AT 9.9 SM-CYCLES A TOKEN a sparse level-side at 4 to 5 live digits, its tables read once
  (R22.2), and 8.3 when the cache holds them; the counts alone 6.1 and 5.0. A value test of the rows is
  twice that (20.6, 14.0); at 32 live digits the two forms meet (45.4, 46.8).
- THE GATHER CALL'S FORM LOSES: 1,557 SM-cycles a token a side at 4 to 5 live digits, twice the carry's whole
  window ([`carry_window`](carry_window.md)), and 4,548 to 5,622 at 32. It is exact, every arranged line. A pair's
  25 lines are 50 loads and stores of global memory: 7.2 cycles a line operation a multiprocessor at the
  sparse cell and 2.8 to 3.5 at 32 -- the two cells do not share a rate, and this lab does not say why.
  The verdict is this form's only. [`gather_pass`](gather_pass.md) measures the same re-arrangement with bulk
  copies at both ends, and its floor.
- ONE CELL WIDTH, two densities, live digits spread evenly: a dense level and a clustered one are unmeasured.

Record: `measure/lab/results/facts_lists/sm_90a.json`.
