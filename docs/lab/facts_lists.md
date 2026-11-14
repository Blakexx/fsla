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

MEASURED (H100 80GB HBM3, 132 SMs, vast.ai instance 60558804, driver 595.91.07; 2026-11-13, the experiment at
f93be08e; 7 repetitions; every count and entry of each form's first round exact; the median, SM-cycles a token
for one side's sparse level):

| the cell | a token's live digits from | counts, cached | counts, once-read | lists, cached | lists, once-read |
|---|---|---|---|---|---|
| 4 to 5 live digits a token | the mask as handed | 5.0 | 6.1 | 8.2 | 9.9 |
| 4 to 5 live digits a token | a value test of the rows | 11.0 | 19.4 | 14.1 | 20.5 |
| 32 live digits a token | the mask as handed | 9.5 | 10.5 | 43.5 | 45.7 |
| 32 live digits a token | a value test of the rows | 15.1 | 18.6 | 47.1 | 46.4 |

WHAT IT SHOWS.
- THE PASS IS PRICED AT 9.9 SM-CYCLES A TOKEN a sparse level-side at 4 to 5 live digits, its tables read once
  (R22.2), and 8.2 when the cache holds them; the counts alone 6.1 and 5.0. It reads the mask's 32 bytes a
  token and no row.
- A VALUE TEST OF THE ROWS IS TWICE THAT, 20.5 once-read and 14.1 cached: 256 bytes a token where the mask is 32.
- AT 32 LIVE DIGITS A TOKEN THE PASS IS ITS ENTRIES: 45.7 from the mask and 46.4 by the value test, once-read --
  some 1.1 cycles an entry past the counts (the lists' call adds the offsets and a store a live digit to the counts'; which
  of them it is is not isolated). The mask's gain is the sparse cell's; at this density the two forms meet.
- WITH ITS BYTE AN ENTRY COST MORE THAN THE REST OF THE PASS: the form before this one, whose entry held the token's
  byte read from its row, was 23.2 cycles a token cached at the sparse cell where this one is 8.2 (the experiment at 5d489776).
- BESIDE THE CARRY's window ([`carry_window`](carry_window.md)), two sparse level-sides are some 20 SM-cycles a
  token at the sparse cell. Two cells of one level width, their live digits spread evenly: a DENSE level (every
  digit live, which a level walked by its lists may turn out to be) and a clustered one, where lanes' atomics meet
  on one word, are unmeasured; so is the pass against a flat stream's rate over the same bytes.

Record: `measure/lab/results/facts_lists/sm_90a.json`.
