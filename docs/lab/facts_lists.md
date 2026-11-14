# `facts_lists`

Mirrors `measure/lab/experiments/facts_lists.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

A sparse level's liveness is RULED to be a mask the op is handed, a bit a digit a token, used blindly; the pass
this tree builds still tests the rows' values ([`ARCHITECTURE.md`](../ARCHITECTURE.md#62-the-facts-pass)). This
lab prices the ruled form beside the built one's access. The mask says, for a token, which digits are live; a
carry CTA needs the opposite each window -- for each of its digits, WHICH TOKENS are live on it, the data it pulls.
What does turning one into the other cost a token on hopper, and what does the value test it replaces cost beside
it?

THE FORM. A CTA a window of 1,024 tokens, a thread a token, one side's sparse level of 256 digits. The pass is a
counting sort of the window's live (token, digit) pairs by digit, in token order:
- a thread ORs its lane's bit into a word a (warp, digit) for each of its live digits, a shared-memory atomic each;
- a digit's thread sums the 32 warps' counts before each warp; the digits' offsets are a scan by the 8 warps of
  digits and their totals -- no thread works while the CTA waits;
- a thread stores each of its live digits' ENTRY at its digit's offset, the count before its warp and the lanes
  before it in its warp's word. AN ENTRY IS THE TOKEN, with no byte (KERNEL_STANDARDS 29, item 5): the pass emits
  the mask turned around and reads no row.
Two forms find a token's live digits: the mask's 32 bytes as handed, or a value test of the token's 256 bytes read
from the tile-major rows the bundle holds. A call is the window's COUNTS alone -- what sizes its lists before they
are written -- or the LISTS too. Each is timed with every table in cache (a call's 16 rounds over the same
windows) and ONCE-READ (a round a fresh window: 2.2 million tokens, 540 MB of rows), at a cell of 4 to 5 live
digits a token and at one of 32. Every count and entry of the first round is checked against the host's.

THE GATHER PASS (a third call). With the carry landing whole entries by bulk copy
([`carry_window`](carry_window.md)), no list is written to global memory: the pass that makes a window's lists is
the pass that gathers. For each live (token, digit) pair it moves the token's 8 value lines, its byte on the digit
and its dense row's 16 blocks to its slot of its digit's steps of 16 entries, in the layout the carry lands -- 393
lines a step; nothing is computed -- and writes each digit's count. A load and a store a line, unstaged, a thread a
token. Every arranged line is checked, each byte of the tables its own position's.

MEASURED (H100 80GB HBM3, 132 SMs, vast.ai instance 60599125, driver 595.91.07; 2026-11-14, the experiment at
1b43da74; 7 repetitions; every count, entry and arranged line of each form's first round exact; the median,
SM-cycles a token for one side's sparse level):

| the cell | a token's live digits from | counts, cached | counts, once-read | lists, cached | lists, once-read |
|---|---|---|---|---|---|
| 4 to 5 live digits a token | the mask as handed | 5.1 | 5.5 | 8.4 | 9.7 |
| 4 to 5 live digits a token | a value test of the rows | 11.4 | 20.8 | 14.5 | 23.0 |
| 32 live digits a token | the mask as handed | 9.5 | 9.9 | 44.6 | 47.1 |
| 32 live digits a token | a value test of the rows | 15.5 | 21.6 | 49.6 | 50.9 |

| the cell | the gather pass, cached | once-read |
|---|---|---|
| 4 to 5 live digits a token | 1,593 | 1,577 |
| 32 live digits a token | 5,746 | 5,690 |

WHAT IT SHOWS.
- THE LISTS ARE PRICED AT 9.7 SM-CYCLES A TOKEN a sparse level-side at 4 to 5 live digits, its tables read once
  (R22.2), and 8.4 when the cache holds them; the counts alone 5.5 and 5.1. A value test of the rows is
  twice that (23.0, 14.5); at 32 live digits the two forms meet (47.1, 50.9).
- THE GATHER PASS IN THIS FORM IS NOT A FORM: 1,577 SM-cycles a token a side at 4 to 5 live digits, twice the carry's
  whole window ([`carry_window`](carry_window.md)), and 5,690 at 32. It is exact -- every arranged line -- and it
  moves a pair's 25 lines as 25 loads and 25 stores of global memory by the pair's own thread: some
  7 cycles a line of 16 bytes to or from global memory a multiprocessor, the rate [`slot_sum`](slot_sum.md)
  read for a line. The cache makes no difference to it. A gather pass moves its bytes through shared memory with
  bulk copies at both ends, or it costs more than what it feeds; that form is not built.
- ONE CELL WIDTH, two densities, live digits spread evenly: a dense level and a clustered one are unmeasured, and so
  is any of it against a flat stream's rate over the same bytes.

Record: `measure/lab/results/facts_lists/sm_90a.json`.
